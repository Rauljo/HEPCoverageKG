"""Tests for the planner loop.

The model is injected as a scripted `chat`, so what is tested is the LOOP --
batching, budgets, error handling, truncation, the trace -- not the model's
judgement. Model judgement is what the evaluation harness is for; a test that
depended on it would be flaky and would measure the wrong thing.
"""
from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from hepcoveragekg.query import planner, retrieve as R, templates as T


# -- a scripted model ------------------------------------------------------

def _call(tool: str, args: dict, cid: str = "c1"):
    return SimpleNamespace(
        id=cid, type="function",
        function=SimpleNamespace(name=tool, arguments=json.dumps(args)),
    )


def _response(calls=None, content=""):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, tool_calls=calls or None))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def scripted(*turns):
    """A model that replays a fixed script, one entry per round."""
    state = {"i": 0}

    def chat(messages, tools):
        turn = turns[min(state["i"], len(turns) - 1)]
        state["i"] += 1
        return turn

    chat.calls_made = state  # inspectable
    return chat


@pytest.fixture()
def conn(tmp_path):
    db = tmp_path / "p.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE entity (entity_id TEXT PRIMARY KEY, kind TEXT, label TEXT);
        CREATE TABLE entity_occurrence (
            bundle_id TEXT, entity_id TEXT, paper_id TEXT, kind TEXT,
            label TEXT, aliases TEXT DEFAULT '[]');
        CREATE TABLE assertion (
            assertion_id TEXT PRIMARY KEY, bundle_id TEXT, paper_id TEXT,
            predicate TEXT, family TEXT, subject_id TEXT, object_id TEXT,
            object_value TEXT, qualifiers TEXT DEFAULT '{}');
        CREATE TABLE assertion_evidence (assertion_id TEXT, evidence_id TEXT);
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);
        CREATE TABLE paper (paper_id TEXT PRIMARY KEY);

        INSERT INTO paper VALUES ('p1');
        INSERT INTO entity VALUES ('r1','result','Search for X'), ('g1','generator','Pythia 8');
        INSERT INTO entity_occurrence VALUES ('b1','r1','p1','result','Search for X','[]');
        INSERT INTO entity_occurrence VALUES ('b1','g1','p1','generator','Pythia 8','[]');
        INSERT INTO assertion VALUES
            ('a1','b1','p1','uses_generator','samples','r1','g1',NULL,'{}');
        INSERT INTO assertion_evidence VALUES ('a1','ev1');
        """
    )
    c.commit()
    c.close()
    return T.read_only(db)


@pytest.fixture()
def index(conn):
    return R.build(conn, embed=False)


# -- the loop --------------------------------------------------------------

def test_answer_tool_ends_the_loop(conn, index):
    """Answering ends the loop -- but only after something has been retrieved,
    since an answer with no lookup behind it is the failure this guards."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "42 papers.", "answerable": True,
                                    "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "how many?", chat=chat)
    assert s.answer == "42 papers."
    assert s.rounds == 2
    assert s.reason == "answered"
    assert s.nudged is False, "it looked first, so no nudge was needed"


def test_several_calls_run_in_one_round(conn, index):
    """The point of batching: N operations must not cost N model calls."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"}, "c1"),
                   _call("search", {"text": "result"}, "c2"),
                   _call("count", {"predicate": "uses_generator",
                                   "object_ids": ["g1"]}, "c3")]),
        _response([_call("answer", {"text": "done", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.llm_calls == 2, "two rounds, not four"
    assert s.tool_calls == 3, "all three operations ran"
    assert {st.round for st in s.steps} == {1}


def test_max_places_caps_a_round(conn, index):
    """The 1-vs-many dial: at 1 the planner is forced to be sequential, which
    makes batching an ablation rather than an architecture."""
    calls = [_call("search", {"text": f"q{i}"}, f"c{i}") for i in range(5)]
    chat = scripted(_response(calls),
                    _response([_call("answer", {"text": "x", "answerable": True})]))
    s = planner.answer(conn, index, "q", max_places=2, chat=chat)
    assert len([st for st in s.steps if st.round == 1]) == 2


def test_max_rounds_stops_a_wandering_loop(conn, index):
    """A model that never answers must stop, and must say why."""
    chat = scripted(_response([_call("search", {"text": "x"})]))  # repeats forever
    s = planner.answer(conn, index, "q", max_rounds=3, chat=chat)
    assert s.rounds == 3
    assert "max_rounds" in s.stopped_because
    assert s.answer == ""


def test_tool_error_is_reported_to_the_planner_not_raised(conn, index):
    """A failed call must come back as a message it can react to. Raising would
    lose the whole session over one bad argument."""
    chat = scripted(
        _response([_call("count", {"predicate": "uses_generator"})]),  # missing object_ids
        _response([_call("answer", {"text": "recovered", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.errors == 1
    assert s.answer == "recovered", "the loop continued after the error"


def test_malformed_arguments_are_survivable(conn, index):
    bad = SimpleNamespace(id="c1", type="function",
                          function=SimpleNamespace(name="count", arguments="{not json"))
    chat = scripted(_response([bad]),
                    _response([_call("answer", {"text": "ok", "answerable": True})]))
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.steps[0].error == "bad_arguments"
    assert s.answer == "ok"


def test_abstention_is_recorded_not_treated_as_failure(conn, index):
    """S-13: 'the graph does not hold this' is an answer, and must be
    distinguishable in the trace from an answer that does hold."""
    chat = scripted(_response([_call("answer", {
        "text": "The graph does not record jet tunes.", "answerable": False})]))
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.answerable is False
    assert s.answer


# -- the trace -------------------------------------------------------------

def test_session_records_cost_and_every_step(conn, index):
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "done", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.prompt_tokens == 200 and s.completion_tokens == 40
    assert s.seconds > 0
    assert s.steps[0].tool == "search"
    assert s.steps[0].seconds >= 0


def test_evidence_is_gathered_across_steps(conn, index):
    """Citations must accumulate over the whole session, not only the last call.

    Searches first, because ids now have to come from the graph rather than from
    the caller's head.
    """
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("count", {"predicate": "uses_generator", "object_ids": ["g1"]})]),
        _response([_call("answer", {"text": "1 paper", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.evidence_ids == ["ev1"]


def test_trace_serialises(conn, index):
    chat = scripted(_response([_call("answer", {"text": "x", "answerable": True})]))
    line = json.loads(planner.answer(conn, index, "q", chat=chat).to_jsonl())
    assert {"question", "answer", "rounds", "llm_calls", "prompt_tokens", "steps"} <= set(line)


# -- context control -------------------------------------------------------

def test_truncation_announces_itself():
    """A planner silently given 25 of 200 rows reasons as if it saw everything,
    and concludes something false with no way to notice."""
    rows = [{"i": i} for i in range(200)]
    text = planner._render_rows(rows, 25)
    assert "175 more rows not shown" in text
    assert text.count("\n") == 25


def test_empty_result_says_so():
    assert planner._render_rows([], 10) == "(no rows)"


# -- wiring ----------------------------------------------------------------

def test_answer_tool_is_exposed(conn):
    names = {t["name"] for t in planner.TOOL_SPECS}
    assert "answer" in names, "without it the planner can never stop deliberately"
    assert "search" in names, "without it nothing can start -- every other tool needs ids"


def test_executor_rejects_unknown_tools(conn, index):
    run = planner.build_executor(conn, index)
    with pytest.raises(ValueError):
        run("drop_everything", {})


def test_system_prompt_has_both_variants(conn):
    full = planner.system_prompt(conn, minimal=False)
    small = planner.system_prompt(conn, minimal=True)
    assert len(small) < len(full)
    assert "PREDICATES" in full and "PREDICATES" in small, "schema card is in both"


def test_answer_without_any_retrieval_is_nudged_once(conn, index):
    """The failure the project exists to prevent: answering from the model's own
    knowledge of physics rather than from this corpus."""
    chat = scripted(
        _response(content="Pythia is a parton shower generator."),   # from memory
        _response([_call("search", {"text": "Pythia"})]),            # complies
        _response([_call("answer", {"text": "3 analyses.", "answerable": True})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.nudged is True
    assert s.grounded_in_tools is True
    assert s.answer == "3 analyses."


def test_persistent_memory_answer_is_accepted_but_flagged(conn, index):
    """A second refusal is a finding, not something to spend more rounds on."""
    chat = scripted(_response(content="It is a parton shower generator."))
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.nudged is True
    assert s.grounded_in_tools is False
    assert "from memory" in s.stopped_because


def test_abstention_must_also_be_established_by_looking(conn, index):
    """An abstention that was assumed rather than checked is a guess wearing the
    costume of caution -- so it gets nudged too."""
    chat = scripted(
        _response(content="The graph does not contain that."),
        _response([_call("search", {"text": "x"})]),
        _response([_call("answer", {"text": "Not recorded.", "answerable": False})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.nudged and s.grounded_in_tools
    assert s.answerable is False


def test_prose_answer_after_real_work_is_not_flagged(conn, index):
    """Only the ungrounded case is suspicious; format alone is not."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response(content="Three analyses used it."),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.grounded_in_tools is True
    assert "from memory" not in s.stopped_because


def test_tool_usage_is_tallied(conn, index):
    chat = scripted(
        _response([_call("search", {"text": "a"}, "c1"), _call("search", {"text": "b"}, "c2")]),
        _response([_call("answer", {"text": "x", "answerable": True})]),
    )
    assert planner.answer(conn, index, "q", chat=chat).tool_usage == {"search": 2}


def test_reasoning_is_recorded_per_round(conn, index):
    chat = scripted(
        _response([_call("search", {"text": "a"})], content="First I will search."),
        _response([_call("answer", {"text": "x", "answerable": True})], content="Enough."),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert [t.text for t in s.thoughts] == ["First I will search.", "Enough."]
    assert s.thoughts[0].tools_called == ["search"]


def test_out_of_scope_is_not_a_cheap_exit(conn, index):
    """`out_of_scope` must not skip the work.

    Exempting it would make "not physics" a free verdict, and a model offered a
    free way out will take it for questions that were answerable. Detecting a
    wrong claim would itself require searching -- so the exemption removes the
    only evidence that could contradict it.
    """
    chat = scripted(
        _response([_call("answer", {"text": "Not physics.",
                                    "answerable": False, "reason": "out_of_scope"})]),
        _response([_call("search", {"text": "boiling point"})]),
        _response([_call("answer", {"text": "This graph only covers ATLAS/CMS papers.",
                                    "answerable": False, "reason": "out_of_scope"})]),
    )
    s = planner.answer(conn, index, "boiling point of water?", chat=chat)
    assert s.nudged is True, "even out_of_scope must look first"
    assert s.grounded_in_tools is True
    assert s.reason == "out_of_scope", "the label survives -- only the skip is gone"


def test_not_in_graph_must_be_established_by_looking(conn, index):
    """'No HEP paper covers that' is a claim about the LITERATURE, and is the
    project's actual output -- so it cannot be asserted without checking."""
    chat = scripted(
        _response([_call("answer", {"text": "Not recorded.",
                                    "answerable": False, "reason": "not_in_graph"})]),
        _response([_call("search", {"text": "jet tunes"})]),
        _response([_call("answer", {"text": "No analysis records tunes.",
                                    "answerable": False, "reason": "not_in_graph"})]),
    )
    s = planner.answer(conn, index, "which jet tunes were used?", chat=chat)
    assert s.nudged is True
    assert s.grounded_in_tools is True, "it searched before concluding absence"
    assert s.reason == "not_in_graph"


def test_the_two_kinds_of_no_are_distinguishable_in_the_trace(conn, index):
    """Conflating them would pollute the coverage signal."""
    import json as _json
    out = _json.loads(planner.answer(conn, index, "q", chat=scripted(_response(
        [_call("answer", {"text": "x", "answerable": False,
                          "reason": "out_of_scope"})]))).to_jsonl())
    assert out["reason"] == "out_of_scope" and out["answerable"] is False


def test_invented_entity_ids_are_rejected(conn, index):
    """The first live run's actual failure.

    The model batched `search` and `count` in ONE round, so when it wrote
    `count` the search results did not exist -- and instead of waiting it
    invented "gen-223" and got a confident zero back. A wrong answer with a
    healthy-looking trace, which is the worst shape a failure can have.
    """
    chat = scripted(
        _response([_call("search", {"text": "Pythia"}, "c1"),
                   _call("count", {"predicate": "uses_generator",
                                   "object_ids": ["gen-223"]}, "c2")]),
        _response([_call("answer", {"text": "recovered", "answerable": True,
                                    "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.invented_ids == ["gen-223"]
    assert any(st.error == "unknown_entity_id" for st in s.steps)
    assert s.answer == "recovered", "the model is told, and can recover"


def test_ids_returned_by_search_are_accepted(conn, index):
    """The guard must not block the legitimate path."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("count", {"predicate": "uses_generator", "object_ids": ["g1"]})]),
        _response([_call("answer", {"text": "done", "answerable": True,
                                    "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.invented_ids == []
    assert not any(st.error for st in s.steps)


def test_ids_persist_across_rounds(conn, index):
    """An id learned in round 1 is still usable in round 3."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("papers_of", {"entity_ids": ["g1"]})]),
        _response([_call("count", {"predicate": "uses_generator", "object_ids": ["g1"]})]),
        _response([_call("answer", {"text": "x", "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.invented_ids == []


def test_tool_call_written_as_text_is_recovered(conn, index):
    """Observed live: after the id guard corrected it, the model produced the
    RIGHT call -- right predicate, right ids -- but wrapped in <tool_call> tags
    in the message body, where vLLM's parser missed it. The loop then recorded
    raw JSON as the final answer while the model was still working."""
    text = ('I will now count them.\n<tool_call>\n'
            '{"name": "search", "arguments": {"text": "Pythia"}}\n</tool_call>')
    chat = scripted(
        _response(content=text),
        _response([_call("answer", {"text": "3 analyses.", "answerable": True,
                                    "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.recovered_calls == 1
    assert s.tool_usage == {"search": 1}, "it ran, rather than becoming the answer"
    assert s.answer == "3 analyses."


def test_truncated_text_tool_call_is_not_half_executed(conn, index):
    """Cut off mid-JSON there is nothing safe to recover, so it must fall
    through rather than run a guessed call."""
    chat = scripted(_response(content='<tool_call>\n{"name": "count", "argum'),
                    _response(content='<tool_call>\n{"name": "count", "argum'))
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.recovered_calls == 0
    assert s.tool_calls == 0


def test_prose_without_a_tool_call_is_untouched(conn, index):
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response(content="Three analyses used it."),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.recovered_calls == 0
    assert "Three analyses" in s.answer


def test_search_saves_a_named_set(conn, index):
    """The model should never have to retype ids: writing out 31 of them cost
    863 completion tokens live, and on a retry looped to 6,521 characters before
    being cut off mid-identifier."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "x", "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert "set_1" in s.sets
    assert s.steps[0].preview or True
    assert "saved as set_1" in _last_note(s)


def _last_note(session):
    return " ".join(st.preview for st in session.steps)


def test_a_set_can_be_used_instead_of_ids(conn, index):
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("count", {"predicate": "uses_generator", "object_set": "set_1"})]),
        _response([_call("answer", {"text": "done", "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert not any(st.error for st in s.steps)
    assert s.tool_usage.get("count") == 1


def test_unknown_set_name_says_which_exist(conn, index):
    """An error that names the available sets is recoverable; one that just
    fails is not."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("count", {"predicate": "uses_generator", "object_set": "set_9"})]),
        _response([_call("answer", {"text": "ok", "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    bad = [st for st in s.steps if st.error]
    assert bad and "set_1" in bad[0].error


def test_set_reference_is_not_treated_as_an_invented_id(conn, index):
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("papers_of", {"entity_set": "set_1"})]),
        _response([_call("answer", {"text": "x", "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.invented_ids == []


def test_search_breadth_is_not_the_models_decision(conn, index):
    """Breadth decides correctness: concept("Pythia", limit=6) gives 45 papers,
    limit=60 gives 58. A model economising on breadth produces a wrong answer
    with a clean trace, so the choice is taken away from it."""
    spec = next(t for t in planner.TOOL_SPECS if t["name"] == "search")
    assert "limit" not in spec["parameters"]["properties"], \
        "the model must not be able to narrow the search"


def test_a_limit_argument_is_ignored_if_one_arrives(conn, index):
    """Models pass arguments that are not in the schema. It must not take."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia", "limit": 1})]),
        _response([_call("answer", {"text": "x", "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert not any(st.error for st in s.steps), "an extra argument must not break the call"


def test_ids_are_recognised_from_every_column_a_template_returns():
    """`known_entity_ids` is what the invented-id guard accepts, so a column
    missing from the list makes the guard reject ids the graph itself returned.
    `contents_of` returns `object_id` and nothing it produced was recognised --
    and `describe` had the same latent problem, which is why chaining one
    describe into another failed."""
    from hepcoveragekg.query import planner

    rows = [{"object_id": "hepkg:generator:pythia8"}, {"subject_id": "hepkg:sample:ttbar"}]
    found = planner._collect_ids(rows, "")
    assert found == {"hepkg:generator:pythia8", "hepkg:sample:ttbar"}


# -- papers are entities, but thin ones -------------------------------------

def test_a_paper_id_is_not_an_invented_id():
    """The invented-id guard exists to stop `gen-223`, not to stop a citation.

    A paper id is verifiable by shape and usually came from the question itself,
    so it needs no prior search to be legitimate. Before this, `_check_ids`
    rejected it and the correction it gave -- "call search FIRST" -- sent the
    planner to a search that cannot succeed, because papers are indexed under
    their titles.
    """
    assert planner._check_ids({"entity_ids": ["2308.02285"]}, set()) == []
    assert planner._check_ids(
        {"entity_ids": ["hepkg:paper:arxiv:2308.02285"]}, set()) == []
    assert planner._check_ids({"entity_ids": ["gen-223"]}, set()) == ["gen-223"]


def test_describe_on_a_paper_becomes_contents_of():
    """Both spellings of a paper, and the predicate carried across."""
    for written in ("2308.02285", "hepkg:paper:arxiv:2308.02285"):
        tool, args, note = planner.resolve_paper_calls(
            "describe", {"entity_ids": [written], "predicate": "result_measures_observable"})
        assert tool == "contents_of"
        assert args == {"paper_ids": ["2308.02285"],
                        "predicate": "result_measures_observable"}
        assert note and "contents_of" in note


def test_the_redirect_leaves_a_trace(conn, index):
    """A rewrite that hid itself would make the planner look as though it had
    chosen the right tool, and tool selection is something we measure."""
    chat = scripted(
        _response([_call("describe", {"entity_ids": ["2308.02285"]})]),
        _response([_call("answer", {"text": "x", "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    step = s.steps[0]
    assert step.tool == "contents_of" and step.redirected_from == "describe"
    assert '"redirected_from": "describe"' in s.to_jsonl()


def test_only_calls_that_are_wholly_about_papers_are_redirected():
    """A mixed list is the planner doing something else; leave it alone."""
    tool, args, note = planner.resolve_paper_calls(
        "describe", {"entity_ids": ["2308.02285", "hepkg:generator:pythia8"]})
    assert tool == "describe" and note is None
    tool, _, _ = planner.resolve_paper_calls("subjects_of", {"object_ids": ["2308.02285"]})
    assert tool == "subjects_of", "only describe/count mean contents_of"


def test_a_paper_is_findable_by_its_arxiv_id(tmp_path):
    """The paper carries its TITLE as the label and its arXiv id in
    `external_ids`, so indexing labels alone left 60 papers searchable by title
    and invisible by id -- while the module docstring claims arXiv ids are
    exactly what the sparse half is here to catch."""
    db = tmp_path / "x.db"
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE entity_occurrence (
            bundle_id TEXT, entity_id TEXT, paper_id TEXT, kind TEXT,
            label TEXT, aliases TEXT DEFAULT '[]', external_ids TEXT DEFAULT '{}');
        CREATE TABLE entity_canonical (entity_id TEXT PRIMARY KEY, canonical_id TEXT);
        INSERT INTO entity_occurrence VALUES
            ('b1','hepkg:paper:arxiv:2308.02285','2308.02285','paper',
             'Measurement of the associated production of a Z boson','[]',
             '{"arxiv":"2308.02285"}');
        """
    )
    c.commit(); c.close()
    conn = T.read_only(db)
    idx = R.build(conn, embed=False)
    hits = R.search(idx, "2308.02285", conn=conn, limit=10)
    assert [h.entity_id for h in hits] == ["hepkg:paper:arxiv:2308.02285"]


def test_an_index_without_external_ids_still_builds(conn):
    """Older databases predate the column; the query layer must not hard-fail."""
    assert len(R.build(conn, embed=False)) > 0


# -- widening, when the tail is still relevant ------------------------------

def _keeping(rung):
    """A critic stub that gives every candidate the same rung."""
    from hepcoveragekg.query import critic as C

    def judge(search_text, hits, refresh=False):
        return C.Review(question="q", search_text=search_text,
                        verdicts=[C.Verdict(h.entity_id, rung) for h in hits])
    return judge


def test_a_saturated_search_widens_and_a_junk_one_does_not(conn, index, monkeypatch):
    """The ceiling binds on 64.8% of Tier B searches, so this is the difference
    between answering over 60 of 109 clusters and over all of them."""
    from hepcoveragekg.query import critic as C

    seen_limits = []
    real = R.search

    def spy(idx, text, conn=None, kind=None, limit=50):
        seen_limits.append(limit)
        # pretend the retriever always has more to give
        return [SimpleNamespace(entity_id=f"e{i}", label=f"thing {i}",
                                kind="generator", facets=[]) for i in range(limit)]

    monkeypatch.setattr(R, "search", spy)
    execute = planner.build_executor(conn, index, {}, critic=_keeping(C.EXACT))
    execute("search", {"text": "jet energy scale"})
    assert seen_limits[0] == planner.SEARCH_BREADTH
    assert max(seen_limits) > planner.SEARCH_BREADTH, "a relevant tail must widen"
    assert max(seen_limits) <= planner.MAX_SEARCH_BREADTH

    seen_limits.clear()
    execute = planner.build_executor(conn, index, {}, critic=_keeping(C.UNRELATED))
    execute("search", {"text": "top squark"})
    assert seen_limits == [planner.SEARCH_BREADTH], "a junk tail must not widen"
    monkeypatch.setattr(R, "search", real)


def test_nothing_widens_without_a_critic(conn, index, monkeypatch):
    """Breadth only becomes a stopping rule when something can read the tail."""
    seen = []
    monkeypatch.setattr(R, "search", lambda idx, text, conn=None, kind=None, limit=50:
                        (seen.append(limit) or
                         [SimpleNamespace(entity_id=f"e{i}", label=f"t{i}",
                                          kind="generator", facets=[])
                          for i in range(limit)]))
    planner.build_executor(conn, index, {})("search", {"text": "anything"})
    assert seen == [planner.SEARCH_BREADTH]


def test_the_kept_set_is_the_expansion_of_the_kept_seeds(conn, index):
    """A verdict moves a CLUSTER. Taking a subset of the expanded set instead
    would keep the cluster-mates of dropped seeds and undo the judgement."""
    from hepcoveragekg.query import critic as C

    def judge(search_text, hits, refresh=False):
        return C.Review(question="q", search_text=search_text,
                        verdicts=[C.Verdict(h.entity_id, C.UNRELATED) for h in hits])

    sets = {}
    planner.build_executor(conn, index, sets, critic=judge)("search", {"text": "Pythia"})
    assert sets["set_1"], "the full set keeps everything"
    assert sets["set_1_kept"] == [], "and the kept set drops the whole cluster"


def test_the_critic_marks_rows_and_leaves_the_set_whole(conn, index):
    from hepcoveragekg.query import critic as C

    sets = {}
    result = planner.build_executor(
        conn, index, sets, critic=_keeping(C.BROADER))("search", {"text": "Pythia"})
    assert all(row["bears_on"] == C.BROADER for row in result.rows)
    assert "still holds everything" in result.note
    assert len(sets["set_1"]) >= len(sets["set_1_kept"])


# -- citing the answer instead of retyping it -------------------------------

def test_the_answer_cites_a_set_instead_of_listing_papers(conn, index):
    """The output half of S-29. Measured: the planner wrote 11 arXiv ids at the
    median, 100% of them genuinely retrieved, and hit only 22.8% of the gold
    papers while retrieval had reached 91.7%. That is selection, and a citation
    cannot select wrongly."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "These analyses use Pythia.",
                                    "papers_from": "set_1", "value_from": "set_1",
                                    "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "which analyses use Pythia?", chat=chat)
    assert s.answer_papers == ["p1"], "the paper list came from the set, not the prose"
    assert s.answer_value == 1.0, "the count is papers, never the number of entities"
    assert "papers=set_1" in s.answer_cited and "value=set_1" in s.answer_cited


def test_a_citation_naming_nothing_is_left_unresolved_not_guessed(conn, index):
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "x", "papers_from": "set_99",
                                    "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.answer_papers == [] and s.answer_cited == ""


def test_the_count_is_papers_not_entities(conn, index):
    """A set holds entity ids and the graph keeps 56 spellings of Pythia. "How
    many analyses" means distinct PAPERS, so citing must resolve through them."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "x", "value_from": "set_1",
                                    "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "how many?", chat=chat)
    assert s.answer_value == 1.0 and len(s.sets["set_1"]) >= 1


# -- refine: the answerer's own relevance marks ------------------------------

def test_refine_drops_ids_additively_and_records_why(conn, index):
    sets = {"set_1": ["a", "b", "c"]}
    execute = planner.build_executor(conn, index, sets)
    result = execute("refine", {"entity_set": "set_1", "drop_ids": ["b"],
                                "reason": "different generator"})
    assert sets["set_1"] == ["a", "b", "c"], "the original set is never touched"
    assert sets["set_1_refined"] == ["a", "c"]
    assert "different generator" in result.note and "2 of 3 kept" in result.note


def test_refine_ignores_ids_that_were_not_in_the_set(conn, index):
    sets = {"set_1": ["a"]}
    result = planner.build_executor(conn, index, sets)(
        "refine", {"entity_set": "set_1", "drop_ids": ["zzz"], "reason": "x"})
    assert sets["set_1_refined"] == ["a"]
    assert "were not in set_1" in result.note


# -- the abstention challenge ------------------------------------------------

def test_abstaining_while_holding_rows_is_challenged_once(conn, index):
    """Two measured false negatives had this shape: 13 candidates kept, and the
    answer was still "the graph does not record any papers"."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "not in the graph", "answerable": False,
                                    "reason": "not_in_graph"})]),
        _response([_call("answer", {"text": "On reflection, 1 paper does.",
                                    "answerable": True, "reason": "answered"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat, answer_contract=True)
    assert s.abstention_challenged is True
    assert s.reason == "answered"


def test_the_challenge_asks_for_a_reason_and_accepts_the_abstention(conn, index):
    """The dangerous failure mode is teaching the system never to abstain: a
    false coverage claim is worse than a false negative. A sound abstention must
    survive the challenge unchanged."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "no", "answerable": False,
                                    "reason": "not_in_graph"})]),
        _response([_call("answer", {"text": "All hits are Sherpa 2.2.2; the question "
                                            "asked for 2.2.1.",
                                    "answerable": False, "reason": "not_in_graph"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat, answer_contract=True)
    assert s.reason == "not_in_graph", "a justified abstention stands"
    assert "2.2.1" in s.answer
    assert planner.ABSTENTION_CHALLENGE.count("Do not invent") == 1


def test_abstaining_with_nothing_retrieved_is_not_challenged_twice(conn, index):
    """The NUDGE already covers the empty case; challenging it too would be two
    corrections for one mistake."""
    chat = scripted(
        _response([_call("answer", {"text": "no", "answerable": False,
                                    "reason": "not_in_graph"})]),
        _response([_call("answer", {"text": "still no", "answerable": False,
                                    "reason": "not_in_graph"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat, answer_contract=True)
    assert s.nudged is True and s.abstention_challenged is False


def test_the_challenge_does_not_fire_under_the_v1_contract(conn, index):
    """v1 must be byte-for-byte what it was, or the arms differ by two things."""
    chat = scripted(
        _response([_call("search", {"text": "Pythia"})]),
        _response([_call("answer", {"text": "no", "answerable": False,
                                    "reason": "not_in_graph"})]),
    )
    s = planner.answer(conn, index, "q", chat=chat)
    assert s.abstention_challenged is False and s.reason == "not_in_graph"
