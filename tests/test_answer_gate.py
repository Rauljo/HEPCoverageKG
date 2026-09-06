"""The answer gate, and the two citation bugs it was built beside (D-107).

Every case below is a verbatim shape from runs 54201-06, where 23 of 60
Gabriel answers scored a hard 0 while the run had already found the papers.
"""
import pytest

from hepcoveragekg.query import answer_gate as G
from hepcoveragekg.query import planner


# --------------------------------------------------------------------------
# the gate itself
# --------------------------------------------------------------------------

def test_ids_in_the_text_pass():
    r = G.check("The analyses are 2004.14060, 2006.05880 and 2012.03799.")
    assert r.ok and r.named == 3 and r.kind == ""


def test_placeholder_is_caught():
    # gf-08, control-b, scored 0.00 -- this was the whole final sentence.
    r = G.check("The analyses that require exactly two electrons or exactly "
                "two muons are: [list of papers from the intersection].")
    assert not r.ok and r.kind == "placeholder"
    assert "list of papers" in r.problem


def test_titles_without_ids_are_caught():
    # gf-04, subgoal-status: seven CORRECT papers, named by title, scored 0.
    r = G.check("1. **Measurement of the production cross section for a W "
                "boson in association with a charm quark** (uses unfolding to "
                "measure fiducial cross sections at particle level).")
    assert not r.ok and r.kind == "silent"


def test_deferred_work_is_caught():
    # gf-07, reviewer: it described the call instead of making it.
    r = G.check("The analyses are the papers linked to the intersection of "
                "set_3 and set_4. To finalize, run `papers_of` on the "
                "intersection of these two result sets.")
    assert not r.ok and r.kind == "deferred"


def test_a_resolved_citation_is_an_answer():
    """The gate must not fire when the papers already reached the scorer."""
    r = G.check("The analyses are the ones in the kept set.", cited="papers=set_1")
    assert r.ok


def test_an_abstention_is_exempt():
    """Gating "not in the graph" would push the system to fabricate coverage."""
    assert G.check("The graph does not record this.", reason="not_in_graph").ok
    assert G.check("The graph does not record this.", answerable=False).ok


def test_empty_text_is_caught():
    assert G.check("").kind == "silent"


def test_ordinary_brackets_are_not_placeholders():
    """"[1]" and "[see fig. 3]" are prose, not a missing list."""
    r = G.check("Both analyses [1] use unfolding: 2004.14060 and 2006.05880.")
    assert r.ok


def test_the_gate_reads_ids_exactly_as_the_scorer_does():
    """A gate that passed what the scorer then zeroed would be worse than none."""
    from hepcoveragekg.eval import scoring
    text = "See 2004.14060 and also 2211.08028."
    assert set(G.ARXIV.findall(text)) == set(scoring._arxiv_ids_in(text))


def test_the_message_names_the_problem_and_asks_for_ids():
    msg = G.check("...are: [list of papers].").message
    assert "[list of papers]" in msg and "arXiv ids" in msg


# --------------------------------------------------------------------------
# `papers_from` -- the two ways a citation was silently lost
# --------------------------------------------------------------------------

class _Session:
    def __init__(self, sets):
        self.sets = sets
        self.answer_papers = []
        self.answer_value = None
        self.answer_cited = ""
        self.citation_refused = ""
        self.citation_disagrees = ()
        self.answer = ""
        self.steps = []


def _resolve(sets, args):
    s = _Session(sets)
    planner.resolve_citations(s, args, lambda ids: [f"paper:{i}" for i in ids])
    return s


def test_a_compound_citation_resolves_to_the_union():
    """"set_1_kept and set_3_kept" was the commonest unresolved value."""
    s = _resolve({"set_1_kept": ["a"], "set_3_kept": ["b"]},
                 {"papers_from": "set_1_kept and set_3_kept"})
    assert s.answer_papers == ["paper:a", "paper:b"]
    assert s.answer_cited == "papers=set_1_kept+set_3_kept"
    assert not s.citation_refused


def test_a_compound_citation_does_not_duplicate_shared_entities():
    s = _resolve({"set_1": ["a", "b"], "set_2": ["b", "c"]},
                 {"papers_from": "set_1, set_2"})
    assert s.answer_papers == ["paper:a", "paper:b", "paper:c"]


def test_an_unknown_set_is_refused_out_loud():
    """It used to drop in silence, which threw the whole answer away."""
    s = _resolve({"set_1": ["a"]}, {"papers_from": "set_9"})
    assert not s.answer_papers
    assert "does not name a set" in s.citation_refused
    assert "set_1" in s.citation_refused          # says what CAN be cited


def test_a_single_name_still_works():
    s = _resolve({"set_1": ["a"]}, {"papers_from": "set_1"})
    assert s.answer_papers == ["paper:a"] and s.answer_cited == "papers=set_1"


def test_no_citation_is_not_a_refusal():
    s = _resolve({"set_1": ["a"]}, {"text": "prose"})
    assert not s.citation_refused and not s.answer_papers


# --------------------------------------------------------------------------
# the scorer's side: the print rate must be reported, not multiplied in
# --------------------------------------------------------------------------

def test_answer_form_separates_printing_from_knowing():
    """The two columns runs 54201-06 could not tell apart."""
    from hepcoveragekg.eval import scoring
    from hepcoveragekg.eval.systems import Answer
    from hepcoveragekg.eval.questions import Question, Truth

    q = Question(qid="x", text="which analyses?", shape="set",
                 truth=Truth(kind="set", papers=["2004.14060"]))
    printed = scoring.answer_form(q, Answer(text="2004.14060", named_ids=1))
    silent = scoring.answer_form(q, Answer(text="the kept set", named_ids=0))
    assert printed["answer_names_papers"] == 1.0
    assert silent["answer_names_papers"] == 0.0


def test_answer_form_counts_a_citation_as_naming():
    from hepcoveragekg.eval import scoring
    from hepcoveragekg.eval.systems import Answer
    from hepcoveragekg.eval.questions import Question, Truth

    q = Question(qid="x", text="which?", shape="set",
                 truth=Truth(kind="set", papers=["2004.14060"]))
    a = Answer(text="the kept set", named_ids=0, cited="papers=set_1")
    assert scoring.answer_form(q, a)["answer_names_papers"] == 1.0


def test_answer_form_abstains_on_a_count():
    """A count that names no papers is correct, not a formatting failure."""
    from hepcoveragekg.eval import scoring
    from hepcoveragekg.eval.systems import Answer
    from hepcoveragekg.eval.questions import Question, Truth

    q = Question(qid="x", text="how many?", shape="count",
                 truth=Truth(kind="count", value=7))
    assert scoring.answer_form(q, Answer(text="7", value=7)) == {}


# --------------------------------------------------------------------------
# the gate inside the loop
# --------------------------------------------------------------------------

class _Rows:
    def __init__(self, papers):
        self.rows = [{"paper_id": p} for p in papers]


def _run(args, gate=True, sets=None, papers=()):
    """One `answer` call through the graph, with the gate on.

    `papers=None` makes `papers_of` return something malformed, which is how a
    citation fails in the wild.
    """
    import json
    from hepcoveragekg.query import graph as G

    def _execute(name, a):
        return None if papers is None else _Rows(papers)

    session = planner.Session(question="q")
    session.sets.update(sets or {})
    state = {"session": session, "messages": [], "round": 1, "max_rounds": 6,
             "max_places": 8, "max_rows": 25,
             "pending_calls": [{"id": "1", "name": "answer",
                                "arguments": json.dumps(args)}],
             "last_content": ""}
    cfg = {"configurable": {"execute": _execute, "tools": [],
                            "chat": None, "contract": "v3",
                            "answer_gate": gate}}
    session.steps.append(planner.Step(1, "search", {}, rows=5))
    G.execute(state, cfg)
    return session, state


def test_the_gate_sends_a_placeholder_back():
    session, state = _run({"text": "They are: [list of papers from the "
                                   "intersection].", "reason": "answered"})
    assert session.answer_gate_retried
    assert not session.answer, "it must not stand as an answer"
    assert "does not name any papers" in state["messages"][-1]["content"]


def test_the_gate_accepts_the_second_try_and_flags_it():
    """Asked once. A second rejection would be a loop, so it is recorded."""
    import json
    from hepcoveragekg.query import graph as G

    session, state = _run({"text": "They are: [list of papers].",
                           "reason": "answered"})
    state["pending_calls"] = [{"id": "2", "name": "answer", "arguments":
                               json.dumps({"text": "Still no ids, sorry.",
                                           "reason": "answered"})}]
    G.execute(state, {"configurable": {"execute": lambda n, a: None,
                                       "tools": [], "chat": None,
                                       "contract": "v3", "answer_gate": True}})
    assert session.answer_gate_failed
    assert session.answer == "Still no ids, sorry.", "the second one goes through"


def test_the_gate_is_off_by_default():
    """An arm, not a default, until it is measured."""
    session, _ = _run({"text": "They are: [list of papers].",
                       "reason": "answered"}, gate=False)
    assert not session.answer_gate_retried
    assert session.answer.startswith("They are")


def test_the_gate_does_not_fire_on_a_resolved_citation():
    session, _ = _run({"text": "The kept set.", "papers_from": "set_1",
                       "reason": "answered"}, sets={"set_1": ["e1"]},
                      papers=["2004.14060"])
    assert not session.answer_gate_retried
    assert session.answer_papers == ["2004.14060"]


def test_a_broken_papers_of_leaves_the_answer_uncited_rather_than_dead():
    """The guard used to stop at the tool call, so a malformed result raised
    out of `resolve_citations` and killed the run."""
    session, _ = _run({"text": "See 2004.14060.", "papers_from": "set_1",
                       "reason": "answered"}, sets={"set_1": ["e1"]},
                      papers=None)
    assert session.answer == "See 2004.14060."
    assert not session.answer_cited


def test_named_none_means_named_none_not_named_unjudged():
    """Two different failures. Both score 0; only one is a formatting failure."""
    from hepcoveragekg.eval import scoring
    from hepcoveragekg.eval.systems import Answer
    from hepcoveragekg.eval.questions import Question, Truth

    q = Question(qid="x", text="which?", shape="set",
                 truth=Truth(kind="set", papers=["2004.14060"],
                             universe=["2004.14060", "2006.05880"]))
    # named twenty papers, none of them ever judged
    outside = scoring.judged_set_f1(q, Answer(text="9901.11111 and 9902.22222"))
    assert outside["judged_f1"] == 0.0
    assert outside["judged_named_none"] == 0.0, "it DID name papers"
    assert outside["judged_named_unjudged"] == 1.0

    silent = scoring.judged_set_f1(q, Answer(text="the papers in the kept set"))
    assert silent["judged_named_none"] == 1.0
    assert silent["judged_named_unjudged"] == 0.0


def test_a_cited_answer_is_not_named_none():
    """The `cited` path puts papers where the scorer reads them, so an answer
    with no ids in the prose has still named something."""
    from hepcoveragekg.eval import scoring
    from hepcoveragekg.eval.systems import Answer
    from hepcoveragekg.eval.questions import Question, Truth

    q = Question(qid="x", text="which?", shape="set",
                 truth=Truth(kind="set", papers=["2004.14060"],
                             universe=["2004.14060", "2006.05880"]))
    a = Answer(text="the kept set", papers=["2004.14060"], cited="papers=set_1")
    got = scoring.judged_set_f1(q, a)
    assert got["judged_f1"] > 0 and got["judged_named_none"] == 0.0


# --------------------------------------------------------------------------
# the path the gate could not see
# --------------------------------------------------------------------------

def test_the_gate_also_runs_when_the_model_never_called_answer():
    """`after_plan` turns `last_content` into the answer when the model writes
    prose and makes no tool call -- and for a reasoning model `last_content` is
    its chain of thought. 23 of 125 set answers in the D-108 gate arm reached
    the scorer that way and the gate never saw one of them."""
    from hepcoveragekg.query import graph as G

    session = planner.Session(question="q")
    session.steps.append(planner.Step(1, "search", {}, rows=5))
    state = {"session": session, "pending_calls": [],
             "last_content": "Okay, let's tackle this question step by step. "
                             "First I need to find the analyses that ..."}
    assert G.after_plan(state) == "finish"
    assert session.answer_gate_kind == "silent"
    assert session.answer_gate_failed, "it must be recorded, not silently scored"


def test_that_path_is_clean_when_the_prose_does_name_papers():
    from hepcoveragekg.query import graph as G

    session = planner.Session(question="q")
    session.steps.append(planner.Step(1, "search", {}, rows=5))
    state = {"session": session, "pending_calls": [],
             "last_content": "The analyses are 2004.14060 and 2006.05880."}
    assert G.after_plan(state) == "finish"
    assert not session.answer_gate_failed
    assert session.answer_gate_kind == ""


def test_an_abstention_on_that_path_is_still_exempt():
    from hepcoveragekg.query import graph as G

    session = planner.Session(question="q")
    session.reason = "not_in_graph"
    session.steps.append(planner.Step(1, "search", {}, rows=5))
    state = {"session": session, "pending_calls": [],
             "last_content": "The graph does not record this."}
    G.after_plan(state)
    assert not session.answer_gate_failed
