"""Enumeration expansion (D-131): search the concepts the question names."""
from hepcoveragekg.query import planner, retrieve


def test_gf02_yields_its_three_methods():
    q = ("Which analyses estimate a background using an ABCD method, or an "
         "ABCD-style sideband or matrix method over independent regions?")
    got = planner.enumerated_concepts(q)
    joined = " | ".join(got).lower()
    assert "abcd" in joined and "sideband" in joined and "matrix method" in joined


def test_parenthetical_and_rather_than_are_dropped():
    q = ("Which analyses are searches (rather than measurements) whose event "
         "selection uses both b-tagged jets and missing transverse momentum?")
    got = planner.enumerated_concepts(q)
    assert not any("measurement" in g.lower() for g in got)
    assert any("b-tagged" in g for g in got) and any("missing transverse" in g for g in got)


def test_covers_uses_token_overlap():
    assert planner._covers("ABCD method", "estimate background using ABCD method")
    assert not planner._covers("ABCD", "matrix method over independent regions")


def _hit(eid): return retrieve.Hit(entity_id=eid, label=eid, kind="k", score=1.0)


def test_expansion_searches_only_uncovered_concepts_once(monkeypatch):
    import sqlite3
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.executescript("CREATE TABLE entity_canonical(entity_id TEXT, canonical_id TEXT);")
    seen = []
    def fake(index, text, conn=None, kind=None, limit=60, pool=200):
        seen.append(text); return [_hit("e_" + text.split()[0].lower())]
    monkeypatch.setattr(retrieve, "search", fake)
    session = planner.Session(question="q"); sets = {}
    q = "Which analyses use an ABCD method, or a sideband or matrix method?"
    ex = planner.build_executor(conn, index=None, sets=sets, critic=None, enum_expand=True,
                                question_text=q, on_enum=planner._enum_counter(session))
    res = ex("search", {"text": "ABCD method"})
    assert seen[0] == "ABCD method"
    assert any("sideband" in t.lower() for t in seen[1:]) and any("matrix" in t.lower() for t in seen[1:])
    assert not any("abcd" in t.lower() for t in seen[1:]), "the covered concept is not re-searched"
    assert session.enum_concepts >= 2 and session.enum_added >= 2
    assert "also names" in res.note
    ex("search", {"text": "sideband"})
    assert session.enum_concepts >= 2, "only the first search of a run expands"
    assert len([t for t in seen if "matrix" in t.lower()]) == 1


def test_off_by_default_is_a_single_search(monkeypatch):
    import sqlite3
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.executescript("CREATE TABLE entity_canonical(entity_id TEXT, canonical_id TEXT);")
    seen = []
    monkeypatch.setattr(retrieve, "search", lambda index, text, conn=None, kind=None, limit=60, pool=200: (seen.append(text) or [_hit("e")]))
    ex = planner.build_executor(conn, index=None, sets={}, critic=None, question_text="A or B or C")
    ex("search", {"text": "A"})
    assert seen == ["A"]
