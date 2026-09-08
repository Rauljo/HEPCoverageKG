"""The kind fallback (D-119): keep the kinded hits first, append the rest."""
import json
from types import SimpleNamespace
from hepcoveragekg.query import planner, retrieve


def _hit(eid, kind="detector_object"):
    return retrieve.Hit(entity_id=eid, label=eid, kind=kind, score=1.0)


def _executor(monkeypatch, tmp_path, kind_fallback):
    import sqlite3
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.executescript("""CREATE TABLE entity_canonical(entity_id TEXT, canonical_id TEXT);
                          CREATE TABLE entity_facet(paper_id, entity_id, field, value, version, vocabulary);""")
    calls = []
    def fake_search(index, text, conn=None, kind=None, limit=60, pool=200):
        calls.append(kind)
        return [_hit("e_det", "detector_object")] if kind else \
               [_hit("e_det", "detector_object"), _hit("e_reg", "event_region"), _hit("e_proc", "physics_process")]
    monkeypatch.setattr(retrieve, "search", fake_search)
    sets = {}
    ex = planner.build_executor(conn, index=None, sets=sets, critic=None, kind_fallback=kind_fallback)
    return ex, sets, calls


def test_off_by_default_leaves_search_unchanged(monkeypatch, tmp_path):
    ex, sets, calls = _executor(monkeypatch, tmp_path, kind_fallback=False)
    ex("search", {"text": "Higgs", "kind": "detector_object"})
    assert calls == ["detector_object"], "one kinded search, nothing else"
    assert sets["set_1"] == ["e_det"]


def test_fallback_appends_other_kinds_after_the_kinded_hits(monkeypatch, tmp_path):
    ex, sets, calls = _executor(monkeypatch, tmp_path, kind_fallback=True)
    res = ex("search", {"text": "Higgs", "kind": "detector_object"})
    assert calls == ["detector_object", None], "kinded first, then unfiltered"
    assert sets["set_1"] == ["e_det", "e_proc", "e_reg"] or set(sets["set_1"]) == {"e_det", "e_reg", "e_proc"}
    assert "OTHER kinds" in res.note and "2 more entities" in res.note


def test_fallback_is_a_no_op_without_a_kind(monkeypatch, tmp_path):
    ex, sets, calls = _executor(monkeypatch, tmp_path, kind_fallback=True)
    res = ex("search", {"text": "Higgs"})
    assert calls == [None], "no kind given: exactly one search, no second call"
    assert "OTHER kinds" not in res.note


def test_fallback_adds_nothing_when_unfiltered_finds_nothing_new(monkeypatch, tmp_path):
    def same(index, text, conn=None, kind=None, limit=60, pool=200):
        return [_hit("e_det")]
    monkeypatch.setattr(retrieve, "search", same)
    import sqlite3
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.executescript("CREATE TABLE entity_canonical(entity_id TEXT, canonical_id TEXT);")
    ex = planner.build_executor(conn, index=None, sets={}, critic=None, kind_fallback=True)
    res = ex("search", {"text": "x", "kind": "detector_object"})
    assert "OTHER kinds" not in res.note


def test_the_fallback_counts_itself_so_armcheck_can_see_it(monkeypatch, tmp_path):
    """Switched on but silent is the D-105 shape; the record must carry both
    how many kinded searches happened and how many entities were appended."""
    from hepcoveragekg.query import planner
    import sqlite3
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.executescript("CREATE TABLE entity_canonical(entity_id TEXT, canonical_id TEXT);")
    def fake_search(index, text, conn=None, kind=None, limit=60, pool=200):
        return [_hit("a")] if kind else [_hit("a"), _hit("b", "event_region"), _hit("c", "observable")]
    monkeypatch.setattr(retrieve, "search", fake_search)
    session = planner.Session(question="q")
    ex = planner.build_executor(conn, index=None, sets={}, critic=None, kind_fallback=True,
                                on_fallback=planner._fallback_counter(session))
    ex("search", {"text": "Higgs", "kind": "detector_object"})
    ex("search", {"text": "Higgs"})                       # unkinded: not a chance
    assert session.kinded_searches == 1
    assert session.kind_fallback_added == 2
