"""Anchored candidates (D-197): only finding-calls on the question's own terms nominate."""
from types import SimpleNamespace as NS

from hepcoveragekg.query import anchor as A


def _s():
    return NS(nominated_entity_ids=set(), nominated_papers=set(), known_entity_ids=set(), anchor_values=None, anchor_groups_terms=[],
              anchor_mode="all", anchor_terms=set(), anchor_round=None,
              anchor_nominating=0, anchor_drift=0, anchor_relax=0, anchor_fallback=False)


def _collect(rows, note):
    return {r["entity_id"] for r in rows if isinstance(r, dict) and r.get("entity_id")}


def test_switch(monkeypatch):
    monkeypatch.delenv("ANCHOR", raising=False); assert not A.enabled()
    monkeypatch.setenv("ANCHOR", "1"); assert A.enabled()


def test_the_first_facets_call_sets_the_anchor_and_nominates():
    s = _s()
    tag = A.classify(s, 2, "facets", {"values": ["BJet", "MET"], "mode": "all"})
    assert tag == A.ANCHOR
    assert s.anchor_values == {"bjet", "met"} and s.anchor_mode == "all" and s.anchor_round == 2


def test_gf01_trace_is_classified_as_the_diagnosis_said():
    """The exact calls from 54468 on gf-01: intersection, then relaxations and drift."""
    s = _s()
    assert A.classify(s, 1, "search", {"text": "b-tagged jets"}) == A.ANCHOR
    assert A.classify(s, 1, "search", {"text": "missing transverse momentum"}) == A.ANCHOR
    assert A.classify(s, 2, "facets", {"values": ["BJet", "MET"], "mode": "all"}) == A.ANCHOR
    assert A.classify(s, 3, "subjects_of", {"object_set": "set_1"}) == A.READ
    assert A.classify(s, 4, "facets", {"values": ["MET"]}) == A.RELAX          # AND -> OR
    assert A.classify(s, 4, "facets", {"values": ["BJet"]}) == A.RELAX
    assert A.classify(s, 6, "search", {"kind": "paper", "text": "ATLAS CMS search for new physics"}) == A.DRIFT
    assert A.classify(s, 7, "facets", {"values": ["Electron"]}) == A.DRIFT      # never asked
    assert A.classify(s, 8, "contents_of", {"paper_ids": ["2107.12553"]}) == A.READ
    assert A.classify(s, 10, "search", {"text": "signal region with b jets and missing transverse momentum"}) == A.NOMINATE
    assert A.classify(s, 11, "facets", {"values": ["MET", "BJet"], "mode": "all", "category": "search"}) == A.NOMINATE
    # v4: under a conjunctive anchor, one-sided finding calls are relaxations whatever the tool
    assert A.classify(s, 11, "search", {"kind": "detector_object", "text": "DeepJet ParticleNet b-tagged jets"}) == A.RELAX
    assert A.classify(s, 11, "papers_of", {"entity_ids": ["hepkg:object:bb-tagged-ak8-jet"]}) == A.RELAX


def test_same_values_but_any_instead_of_all_is_a_relaxation():
    s = _s()
    A.classify(s, 1, "facets", {"values": ["BJet", "MET"], "mode": "all"})
    assert A.classify(s, 2, "facets", {"values": ["BJet", "MET"], "mode": "any"}) == A.RELAX


def test_a_single_condition_anchor_never_relaxes():
    s = _s()
    A.classify(s, 1, "facets", {"values": ["Pythia"]})
    assert A.classify(s, 3, "facets", {"values": ["Pythia"], "mode": "any"}) == A.NOMINATE
    assert A.classify(s, 3, "facets", {"values": ["Herwig"]}) == A.DRIFT


def test_papers_of_is_on_anchor_only_for_nominated_entities():
    """Single-condition anchor: papers_of on a found entity nominates."""
    s = _s(); s.nominated_entity_ids = {"hepkg:object:bjet"}; s.anchor_values = {"bjet"}
    assert A.classify(s, 3, "papers_of", {"entity_ids": ["hepkg:object:bjet"]}) == A.NOMINATE
    assert A.classify(s, 3, "papers_of", {"entity_ids": ["hepkg:object:tau"]}) == A.DRIFT
    assert A.classify(s, 3, "papers_of", {}) == A.READ


def test_observe_folds_only_nominating_rows_into_the_candidate_set():
    s = _s()
    rows = [{"paper_id": "P1", "entity_ids": ["e1"]}, {"paper_id": "P2", "entity_ids": ["e2"]}]
    assert A.observe(s, 1, "facets", {"values": ["BJet", "MET"]}, rows, "", _collect) == A.ANCHOR
    assert s.nominated_papers == {"P1", "P2"} and s.anchor_nominating == 1
    assert s.nominated_entity_ids == set(), "a paper-returning call nominates papers"
    drift = [{"paper_id": "P9", "entity_ids": ["e9"]}]
    assert A.observe(s, 6, "facets", {"values": ["Tau"]}, drift, "", _collect) == A.DRIFT
    assert "P9" not in s.nominated_papers and s.anchor_drift == 1
    assert A.observe(s, 4, "facets", {"values": ["MET"]}, drift, "", _collect) == A.RELAX
    assert s.anchor_relax == 1
    read = [{"entity_id": "e7"}]
    assert A.observe(s, 3, "subjects_of", {}, read, "", _collect) == A.READ
    assert "e7" not in s.nominated_entity_ids
    ent = [{"entity_id": "e5"}]
    assert A.observe(s, 1, "search", {"text": "b-tagged jets"}, ent, "", _collect) == A.ANCHOR
    assert s.nominated_entity_ids == {"e5"}, "an entity-returning call nominates entities"


def test_feedback_names_the_anchor_and_the_override():
    s = _s(); s.anchor_values = {"bjet", "met"}
    txt = A.feedback(A.DRIFT, s, "facets", {"values": ["Electron"]}, 22)
    assert "electron" in txt and "bjet, met" in txt and "not candidates" in txt and "22 rows" in txt
    txt = A.feedback(A.RELAX, s, "facets", {"values": ["MET"]}, 23)
    assert "fewer conditions" in txt and "answer()" not in txt, "v6: no override by writing ids"
    assert A.feedback(A.NOMINATE, s, "facets", {}, 1) == ""
    assert A.feedback(A.READ, s, "describe", {}, 1) == ""


def test_candidates_fall_back_to_the_footprint_only_when_nothing_was_nominated():
    s = _s(); s.known_entity_ids = {"a", "b", "c"}
    ids, fell = A.candidate_ids(s)
    assert ids == ["a", "b", "c"] and fell is True
    s.nominated_entity_ids = {"b"}
    ids, fell = A.candidate_ids(s)
    assert ids == ["b"] and fell is False


def test_expansion_cap_cuts_rows_before_ids_are_collected(monkeypatch):
    monkeypatch.setenv("EXPANSION_CAP", "3")
    r = NS(rows=[{"entity_id": "e%d" % i} for i in range(10)], note="")
    assert A.cap_expansion("subjects_of", r) is True
    assert len(r.rows) == 3 and "showing 3 of 10" in r.note
    r2 = NS(rows=[{"entity_id": "x"}] * 10, note="prior note")
    assert A.cap_expansion("facets", r2) is False, "only expansion tools are capped"
    monkeypatch.setenv("EXPANSION_CAP", "0")
    r3 = NS(rows=[{"entity_id": "x"}] * 10, note="")
    assert A.cap_expansion("contents_of", r3) is False and len(r3.rows) == 10


def test_terms_drop_stopwords_but_keep_physics():
    assert A.terms("Which analyses use b-tagged jets and E_T^miss?") == {"b-tagged", "jets", "e_t^miss"}


def test_paper_level_nomination_keeps_an_intersection_an_intersection():
    """gf-01: `facets [BJet, MET] mode=all` returns 18 papers whose rows carry
    entities occurring across 45. Nominating the rows' PAPERS, not their
    entities, is what keeps the judge's pool at 18."""
    import sqlite3
    s = _s(); s.nominated_papers = set()
    rows = [{"paper_id": "P%d" % i, "entity_ids": ["hepkg:object:bjet", "hepkg:object:met"]} for i in range(18)]
    tag = A.observe(s, 1, "facets", {"values": ["BJet", "MET"], "mode": "all"}, rows, "", _collect)
    assert tag == A.ANCHOR
    assert s.nominated_papers == {"P%d" % i for i in range(18)}
    assert s.nominated_entity_ids == set(), "entities on paper rows are NOT nominated"
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE entity_occurrence (entity_id TEXT, paper_id TEXT)")
    c.executemany("INSERT INTO entity_occurrence VALUES (?,?)",
                  [("hepkg:object:bjet", "P%d" % i) for i in range(45)])
    pool, fell = A.candidate_pool(s, c)
    assert pool == {"P%d" % i for i in range(18)} and fell is False


def test_search_nominates_entities_and_their_papers_join_the_pool():
    import sqlite3
    s = _s(); s.nominated_papers = set()
    A.observe(s, 1, "search", {"text": "b-tagged jets"}, [{"entity_id": "e1"}], "", _collect)
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE entity_occurrence (entity_id TEXT, paper_id TEXT)")
    c.executemany("INSERT INTO entity_occurrence VALUES (?,?)", [("e1", "P1"), ("e1", "P2"), ("e9", "P9")])
    pool, fell = A.candidate_pool(s, c)
    assert pool == {"P1", "P2"} and fell is False, "entity tier used only when no paper-level call was made"
    # Once a paper-returning call has nominated, the entity tier no longer widens the pool.
    A.observe(s, 2, "facets", {"values": ["BJet"]}, [{"paper_id": "P7"}], "", _collect)
    pool, fell = A.candidate_pool(s, c)
    assert pool == {"P7"} and fell is False


def test_search_never_defines_candidates_even_when_it_returns_papers():
    """v5: a semantic search returns its top-k whatever the match. Its rows go
    to the entity tier, used only when no structural call was ever made."""
    s = _s(); s.nominated_papers = set()
    A.classify(s, 1, "search", {"text": "b-tagged jets"})           # sets anchor terms
    rows = [{"paper_id": "P3", "entity_id": "hepkg:paper:P3"}]
    tag = A.observe(s, 5, "search", {"kind": "paper", "text": "b-tagged jets and MET"}, rows, "", _collect)
    assert tag == A.NOMINATE and s.nominated_papers == set()
    assert s.nominated_entity_ids == {"hepkg:paper:P3"}
    # A structural call then owns the pool outright.
    A.observe(s, 6, "facets", {"values": ["BJet"]}, [{"paper_id": "P9"}], "", _collect)
    pool, fell = A.candidate_pool(s, None)
    assert pool == {"P9"} and fell is False


def test_candidate_pool_falls_back_when_nothing_was_nominated():
    s = _s(); s.nominated_papers = set()
    pool, fell = A.candidate_pool(s, None)
    assert pool == set() and fell is True


def test_sql_fetch_by_id_is_on_anchor_when_the_ids_came_from_an_anchor_query():
    """The two-step shape free-SQL writes: find entities by label, then fetch
    papers by entity id. The second query's literals are ids, not terms."""
    h = A.new_state()
    assert A.classify_sql(h, 1, "SELECT entity_id FROM entity WHERE label LIKE '%ABCD%'") == A.ANCHOR
    h.nominated_entity_ids |= {"hepkg:method:abcd"}
    q2 = "SELECT DISTINCT paper_id FROM entity_occurrence WHERE entity_id IN ('hepkg:method:abcd')"
    assert A.classify_sql(h, 2, q2) == A.NOMINATE
    q3 = "SELECT DISTINCT paper_id FROM entity_occurrence WHERE entity_id IN ('hepkg:object:tau')"
    assert A.classify_sql(h, 3, q3) == A.DRIFT, "ids from nowhere on-anchor"
    assert A.classify_sql(A.new_state(), 1, q2) == A.READ, "ids before any anchor hold to nothing"


def test_free_sql_two_step_shape_from_a_real_32b_trace():
    """54507 gf-07: search 'ttZ background' -> hits, then
    SELECT ... WHERE entity_id IN ('hepkg:background:ttz'). The search must not
    open the anchor round, and the id fetch must anchor."""
    h = A.new_state()
    A.note_search(h, "ttZ background", ["hepkg:background:ttz"])
    assert h.anchor_round is None and "ttz" in h.anchor_terms
    q = "SELECT DISTINCT eo.paper_id FROM entity_occurrence eo WHERE eo.entity_id IN ('hepkg:background:ttz')"
    assert A.classify_sql(h, 2, q) == A.ANCHOR
    q2 = "SELECT p.arxiv_id FROM assertion a JOIN paper p ON a.paper_id=p.arxiv_id WHERE a.predicate='result_uses_statistical_method' AND a.object_id='hepkg:method:histfitter'"
    h2 = A.new_state(); A.note_search(h2, "HistFitter", ["hepkg:method:histfitter"])
    assert A.sql_literals(q2) == [{"hepkg:method:histfitter"}], "the predicate literal is schema, not a condition"
    assert A.classify_sql(h2, 2, q2) == A.ANCHOR


def test_select_at_ceiling_runs_the_selection_when_no_answer_was_called(monkeypatch):
    """A run that spends every round without answer() must still hand its pool
    to the judge when the switch is on -- and must not when it is off."""
    from types import SimpleNamespace as NS
    from hepcoveragekg.query import graph as G
    calls = []
    monkeypatch.setattr(G, "_answer_exit", lambda runtime, session: calls.append(session))
    monkeypatch.setattr(G, "_runtime", lambda config: {"conn": None, "answer_critic": True})
    import hepcoveragekg.query.verify as V
    monkeypatch.setattr(V, "verify_session", lambda s: None)
    monkeypatch.setenv("CONSTRAINED_IDS", "1"); monkeypatch.setenv("CRITIC_SELECTS", "1")

    def _state():
        s = NS(answer="", answer_before_gate="", stopped_because="hit max_rounds (12)",
               known_entity_ids={"e1"}, evidence_ids=[], selected_at_ceiling=False,
               answer_syntax="", verification=None)
        return {"session": s, "_prose_answer": False}
    monkeypatch.delenv("SELECT_AT_CEILING", raising=False)
    G.finish(_state(), None)
    assert calls == [], "off: silence stays silence"
    monkeypatch.setenv("SELECT_AT_CEILING", "1")
    st = _state(); G.finish(st, None)
    assert len(calls) == 1 and st["session"].selected_at_ceiling is True
    assert "selected at the ceiling" in st["session"].stopped_because


def test_a_superset_lookup_tightens_the_anchor_and_rebuilds_the_pool():
    """v7: facets [BJet] first, then after reading facets [BJet, MET] mode=all.
    The second call states the conjunction and must become the anchor, not drift."""
    s = _s(); s.nominated_papers = set(); s.anchor_tightened = 0
    A.observe(s, 1, "facets", {"values": ["BJet"]}, [{"paper_id": "P%d" % i} for i in range(18)], "", _collect)
    assert len(s.nominated_papers) == 18
    tag = A.observe(s, 3, "facets", {"values": ["BJet", "MET"], "mode": "all"},
                    [{"paper_id": "P1"}, {"paper_id": "P2"}], "", _collect)
    assert tag == A.TIGHTEN and s.anchor_values == {"bjet", "met"} and s.anchor_tightened == 1
    assert s.nominated_papers == {"P1", "P2"}, "pool rebuilt from the tighter claim"
    assert A.classify(s, 4, "facets", {"values": ["MET"]}) == A.RELAX, "and the old anchor is now a relaxation"
    assert A.classify(s, 4, "facets", {"values": ["Tau"]}) == A.DRIFT, "disjoint is still drift"


def test_sql_adding_a_condition_tightens():
    h = A.new_state()
    A.classify_sql(h, 1, "... WHERE label LIKE '%ttZ%'")
    assert A.classify_sql(h, 3, "... WHERE label LIKE '%ttZ%' AND role LIKE '%control region%'") == A.TIGHTEN
    assert h.anchor_groups == 2 and "control" in h.anchor_terms


def test_chain_judges_the_nominated_union_not_the_footprint(monkeypatch):
    """D-207: unioning the legs' footprints would make the anchor a no-op inside
    a chain -- the pool the chain-level judge sees must be what the legs
    nominated."""
    from hepcoveragekg.eval.systems import Answer
    a1 = Answer(papers=["P1", "P2", "P3"], nominated_papers=["P1"])
    a2 = Answer(papers=["P4", "P5"], nominated_papers=["P4"])
    papers, nominated = set(), set()
    for a in (a1, a2):
        papers |= set(a.papers or [])
        nominated |= set(getattr(a, "nominated_papers", None) or [])
    assert papers == {"P1", "P2", "P3", "P4", "P5"}
    assert nominated == {"P1", "P4"}
    monkeypatch.setenv("ANCHOR", "1")
    assert (nominated if (A.enabled() and nominated) else papers) == {"P1", "P4"}
    monkeypatch.setenv("ANCHOR", "0")
    assert (nominated if (A.enabled() and nominated) else papers) == papers, "anchor off: footprint union"
