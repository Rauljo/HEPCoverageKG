"""Constrained selection on the free-SQL side (CONSTRAINED_IDS=1)."""
import json
from types import SimpleNamespace
from hepcoveragekg.eval import free_sql as F


class _Client:
    def __init__(self, reply): self.reply = reply; self.calls = []
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kw):
        self.calls.append(kw); return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def test_constrained_papers_selects_only_touched_ids():
    cl = _Client(json.dumps({"papers": ["2001.00001", "2099.09999"]}))
    picked, mode = F.constrained_papers(cl, "m", "q", "draft", {"2001.00001", "2002.00002"}, {"2001.00001": "SELECT ..."})
    assert picked == ["2001.00001"] and mode == "json_schema"
    assert cl.calls[0]["response_format"]["json_schema"]["schema"]["properties"]["papers"]["items"]["enum"] == ["2001.00001", "2002.00002"]


def test_no_candidates_no_call():
    cl = _Client("{}")
    assert F.constrained_papers(cl, "m", "q", "d", set()) == ([], "") and not cl.calls


def test_critic_selects_papers_judges_free_sql_ids_on_paper_wide_evidence():
    """D-166: the judge decides free-SQL's list from the graph's evidence for each paper."""
    import sqlite3
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.executescript("""
      CREATE TABLE paper(arxiv_id TEXT);
      CREATE TABLE entity_occurrence(entity_id TEXT, paper_id TEXT, label TEXT, aliases TEXT);
      CREATE TABLE assertion(assertion_id TEXT, paper_id TEXT, subject_id TEXT, object_id TEXT);
      CREATE TABLE assertion_evidence(assertion_id TEXT, evidence_id TEXT);
      CREATE TABLE evidence(evidence_id TEXT, quote TEXT);
      INSERT INTO paper VALUES('2001.00001'),('2002.00002');
      INSERT INTO entity_occurrence VALUES('E1','2001.00001','b-tagged jet','["b-jet"]'),('E2','2001.00001','Muon','[]'),('E3','2002.00002','Photon','[]');
      INSERT INTO assertion VALUES('a1','2001.00001','E1',NULL),('a2','2002.00002','E3',NULL);
      INSERT INTO assertion_evidence VALUES('a1','ev1'),('a2','ev2');
      INSERT INTO evidence VALUES('ev1','Events are required to have at least one b-jet.'),('ev2','Photons are calibrated.');
    """)
    seen = {}
    def chat(messages):
        seen["prompt"] = messages[1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"verdicts": [{"paper": "2001.00001", "keep": true, "why": "requires b-jet"}, {"paper": "2002.00002", "keep": false, "why": "no b-jets"}]}'))])
    picked, rv = F.critic_selects_papers(conn, "Which analyses use b-tagged jets in their selection?", {"2001.00001", "2002.00002", "2099.09999"}, chat=chat)
    assert picked == ["2001.00001"] and rv["kept"] == 1
    assert "b-tagged jet" in seen["prompt"] and "Events are required to have at least one b-jet." in seen["prompt"]
    assert "2099.09999" not in seen["prompt"]        # not in the graph: never a candidate


def test_module_level_os_alias_exists():
    """The exit path uses `_os.environ` at module scope; a local import inside
    __init__ hid its absence from the test suite once (24 of 27 records errored)."""
    assert hasattr(F, "_os")
