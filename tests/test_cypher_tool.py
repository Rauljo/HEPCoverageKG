"""Cypher as a second query language for the free-SQL control."""
from types import SimpleNamespace
from hepcoveragekg.eval import free_sql as F


class _Rec:
    def __init__(self, d): self._d = d
    def values(self): return list(self._d.values())


class _Res:
    def __init__(self, keys, rows): self._k = keys; self._rows = rows
    def keys(self): return self._k
    def __iter__(self): return iter(_Rec(dict(zip(self._k, r))) for r in self._rows)


class _Sess:
    def __init__(self, rows): self.rows = rows; self.last = None
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def run(self, q): self.last = q; return _Res(["p.arxiv_id", "c.id"], self.rows)


class _Driver:
    def __init__(self, rows): self.sess = _Sess(rows); self.kw = None
    def session(self, **kw): self.kw = kw; return self.sess


def test_run_cypher_returns_rows_and_stays_read_only():
    d = _Driver([("2106.01676", "hepkg:method:abcd-method"), ("2001.06899", "hepkg:method:x")])
    r = F.run_cypher(d, "neo4j", "MATCH (p:Paper)-[:MENTIONS]->(c) RETURN p.arxiv_id, c.id")
    assert not r.error and len(r.rows) == 2 and r.columns == ["p.arxiv_id", "c.id"]
    assert d.kw["default_access_mode"] == "READ"
    assert F.papers_in(r) == ["2001.06899", "2106.01676"]
    assert "hepkg:method:abcd-method" in F.entities_in(r)


def test_write_clauses_are_refused_before_reaching_the_driver():
    d = _Driver([])
    for q in ("MATCH (n) DETACH DELETE n", "CREATE (n:X)", "MATCH (n) SET n.a=1", "MERGE (n:X)"):
        assert F.run_cypher(d, "neo4j", q).error.startswith("read-only")
    assert d.sess.last is None


def test_no_driver_is_a_clear_error_not_a_crash():
    r = F.run_cypher(None, "neo4j", "MATCH (n) RETURN n LIMIT 1")
    assert "not available" in r.error


def test_row_cap_marks_truncation():
    d = _Driver([(f"20{i:02d}.0000{i%10}", "x") for i in range(F.MAX_ROWS + 5)])
    r = F.run_cypher(d, "neo4j", "MATCH (p:Paper) RETURN p.arxiv_id, 'x'")
    assert r.truncated and len(r.rows) == F.MAX_ROWS


def test_languages_select_tools_and_prompt(monkeypatch, tmp_path):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
      CREATE TABLE paper(arxiv_id TEXT, category TEXT); INSERT INTO paper VALUES('2001.00001','search');
      CREATE TABLE entity(entity_id TEXT, kind TEXT, label TEXT);
      CREATE TABLE entity_occurrence(paper_id TEXT, entity_id TEXT, kind TEXT, label TEXT, bundle_id TEXT);
      CREATE TABLE assertion(paper_id TEXT, predicate TEXT, subject_id TEXT, object_id TEXT);
      INSERT INTO assertion VALUES('2001.00001','result_uses_statistical_method','a','b');
    """)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    both = F.FreeSQLSystem(conn, None, languages=("sql", "cypher"), name="free-both")
    assert both.config["languages"] == ["sql", "cypher"] and both.config["neo4j"] is False
    assert "GRAPH (Neo4j" in both._schema and "TABLES (SQLite)" in both._schema
    only = F.FreeSQLSystem(conn, None, languages=("cypher",), name="free-cypher")
    assert "TABLES (SQLite)" not in only._schema and "result_uses_statistical_method" in only._schema
