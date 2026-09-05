"""Tests for the evaluation harness.

Most of these guard against failures that are *silent* -- a harness that
produces a plausible number for the wrong reason is worse than one that
crashes, because the number ends up in a dissertation.
"""
from __future__ import annotations

import json
import types

import pytest

from hepcoveragekg.eval import questions as Q
from hepcoveragekg.eval import report as R
from hepcoveragekg.eval import runner, scoring, systems


# --------------------------------------------------------------------------
# question set
# --------------------------------------------------------------------------

def _record(**over) -> dict:
    base = {"qid": "q1", "text": "How many analyses used Pythia?", "shape": "count",
            "needs": ["sql"], "truth": {"kind": "count", "value": 58, "papers": ["1"]},
            "truth_source": "sql"}
    base.update(over)
    return base


def write(tmp_path, records, name="dev.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_a_truth_without_a_source_is_rejected():
    """An answer nobody can trace is not ground truth, it is a number."""
    with pytest.raises(Q.QuestionError, match="truth_source"):
        Q.parse(_record(truth_source="none"))


def test_a_source_without_a_truth_is_rejected():
    with pytest.raises(Q.QuestionError, match="no truth value"):
        Q.parse(_record(truth={"kind": "none"}, truth_source="sql"))


def test_unknown_needs_tag_is_rejected():
    """`needs` is what makes a low score attributable (S-10). A typo silently
    creates a category of one that nothing ever reports on."""
    with pytest.raises(Q.QuestionError, match="unknown needs"):
        Q.parse(_record(needs=["signature"]))  # the real tag is "signatures"


def test_duplicate_qids_are_rejected(tmp_path):
    """They do not error downstream -- they silently overwrite each other in
    every join, and the record count still looks right."""
    path = write(tmp_path, [_record(qid="a"), _record(qid="a")])
    with pytest.raises(Q.QuestionError, match="duplicate"):
        Q.load(path)


def test_test_split_is_locked_by_default(tmp_path):
    """S-10. Discipline that depends on remembering fails by week three."""
    path = write(tmp_path, [_record(split="test", source="gabriel")])
    with pytest.raises(Q.QuestionError, match="unlock"):
        Q.load(path)


def test_unlocking_the_test_split_is_recorded(tmp_path, monkeypatch):
    """The value of the lock is the trace it leaves, not the obstacle it is."""
    log = tmp_path / "TEST_OPENED.log"
    monkeypatch.setattr(Q, "UNLOCK_LOG", log)
    path = write(tmp_path, [_record(split="test", source="gabriel")])
    Q.load(path, allow_test=True, reason="final run")
    assert "final run" in log.read_text()


def test_m3_blocked_questions_are_flagged(tmp_path):
    """S-57: a question needing signatures is blocked upstream, and no amount of
    query-layer work fixes it. It must be visible before it drags a score down."""
    path = write(tmp_path, [_record(qid="a", needs=["signatures"], truth={"kind": "none"},
                                    truth_source="none")])
    qset = Q.load(path)
    assert qset.summary()["blocked_on_m3"] == 1


def test_saving_is_stable_across_regeneration(tmp_path):
    """Ids must survive a re-run of the generator or cross-run comparison breaks."""
    qs = [Q.parse(_record(qid="b")), Q.parse(_record(qid="a"))]
    first = Q.save(qs, tmp_path / "x.jsonl").read_text()
    second = Q.save(list(reversed(qs)), tmp_path / "y.jsonl").read_text()
    assert first == second


# --------------------------------------------------------------------------
# scoring -- the abstain-not-zero rule
# --------------------------------------------------------------------------

def test_scorers_abstain_rather_than_score_zero():
    """A baseline with no tool calls has no plan validity. Scoring it 0 invents
    a difference that does not exist and hides why the average moved."""
    q = Q.parse(_record())
    flat = systems.Answer(text="58 analyses.", answered=True)
    assert scoring.tool_use(q, flat) is None
    assert scoring.faithfulness(q, flat) is None


def test_faithfulness_applies_when_the_trace_carries_it():
    q = Q.parse(_record())
    a = systems.Answer(text="58", verification_score=0.5, unsupported_claims=["12"])
    assert scoring.faithfulness(q, a) == {"faithfulness": 0.5, "unsupported_claims": 1.0}


def test_an_unlabelled_question_is_not_a_failed_question():
    q = Q.parse(_record(truth={"kind": "none"}, truth_source="none"))
    assert scoring.exact_count(q, systems.Answer(text="anything")) is None


def test_the_count_must_be_the_right_one():
    """'about 60' is not 58. The generosity is about where the number appears,
    never about what it is."""
    q = Q.parse(_record())
    assert scoring.exact_count(q, systems.Answer(text="58 analyses used it"))["count_correct"] == 1.0
    assert scoring.exact_count(q, systems.Answer(text="about 60"))["count_correct"] == 0.0


def test_abstaining_is_correct_for_not_in_graph_questions():
    """S-13: declining is the right answer here, and must not be scored as a miss."""
    q = Q.parse(_record(needs=["not_in_graph"], truth={"kind": "none"}, truth_source="none"))
    declined = systems.Answer(text="Not in the graph.", answered=False)
    assert scoring.responded(q, declined)["abstention_correct"] == 1.0


def test_a_broken_scorer_does_not_kill_the_run():
    def boom(q, a):
        raise RuntimeError("nope")

    out = scoring.score_all(Q.parse(_record()), systems.Answer(), [boom, scoring.cost])
    assert "seconds" in out and any(k.startswith("_error_") for k in out)


# --------------------------------------------------------------------------
# runner -- identity and durability
# --------------------------------------------------------------------------

def test_every_record_carries_what_produced_it(tmp_path):
    """Without git_sha + config_hash an ablation study is unattributable, and it
    is the most expensive thing to retrofit."""
    path = write(tmp_path, [_record()])
    out = runner.run(Q.load(path), systems.StubSystem(), out_dir=tmp_path / "runs")
    meta, records = runner.load_records(out)
    assert meta["git_sha"] and meta["config_hash"] and meta["questions_hash"]
    assert records[0].git_sha == meta["git_sha"]
    assert records[0].config_hash == meta["config_hash"]


def test_config_hash_changes_with_the_config():
    """Ablations differ only by config, so the hash is what tells two runs apart."""
    assert systems.config_hash({"a": 1}) != systems.config_hash({"a": 2})
    assert systems.config_hash({"a": 1, "b": 2}) == systems.config_hash({"b": 2, "a": 1})


def test_records_survive_a_killed_run(tmp_path):
    """Flushed per record: 180 of 200 questions is 180 usable records, not zero."""
    class Explodes:
        name, config = "boom", {}

        def __init__(self):
            self.n = 0

        def answer(self, q):
            self.n += 1
            if self.n > 2:
                raise KeyboardInterrupt
            return systems.Answer(text="ok")

    path = write(tmp_path, [_record(qid=f"q{i}") for i in range(5)])
    with pytest.raises(KeyboardInterrupt):
        runner.run(Q.load(path), Explodes(), out_dir=tmp_path / "runs")
    written = (tmp_path / "runs").glob("*.jsonl")
    lines = [l for l in next(written).read_text().splitlines() if l.strip()]
    assert len(lines) == 3, "the header plus the two answered questions"


def test_the_planner_adapter_calls_a_function_that_exists():
    """An adapter naming a function that does not exist fails only at run time,
    and the harness dutifully records 30 identical errors rather than crashing --
    which is correct behaviour and a wasted run. Caught for real on 2026-08-02:
    the adapter called `planner.run`; the entry point is `planner.answer`.
    """
    import inspect

    from hepcoveragekg.query import planner

    source = inspect.getsource(systems.PlannerSystem.answer)
    called = {n for n in ("answer", "stream", "run") if f"planner.{n}(" in source}
    assert called, "the adapter calls no planner entry point at all"
    for name in called:
        assert hasattr(planner, name), f"planner has no attribute {name!r}"


def test_a_system_that_raises_is_recorded_not_propagated(tmp_path):
    """One dead question must not end a two-hour run."""
    class Broken:
        name, config = "broken", {}

        def answer(self, q):
            return systems.Answer(error="ConnectionError: no model")

    path = write(tmp_path, [_record()])
    out = runner.run(Q.load(path), Broken(), out_dir=tmp_path / "runs")
    _, records = runner.load_records(out)
    assert records[0].scores["errored"] == 1.0


# --------------------------------------------------------------------------
# report -- variance and coverage
# --------------------------------------------------------------------------

def _run_with(tmp_path, answers, repeats=1, name="runs"):
    class Fixed:
        name_, config = "fixed", {"k": 1}
        name = "fixed"

        def __init__(self):
            self.i = -1

        def answer(self, q):
            self.i += 1
            return answers[self.i % len(answers)]

    path = write(tmp_path, [_record()], name=f"{name}.jsonl")
    return runner.run(Q.load(path), Fixed(), repeats=repeats, out_dir=tmp_path / name)


def test_spread_is_reported_when_repeats_disagree(tmp_path):
    """The whole reason repeats exist: a 3% gain inside a 5% band is noise."""
    out = _run_with(tmp_path, [systems.Answer(text="58"), systems.Answer(text="12")], repeats=2)
    _, records = runner.load_records(out)
    metrics = R.summarise(records)
    assert metrics["count_correct"].spread > 0


def test_coverage_is_shown_when_a_metric_does_not_apply_everywhere(tmp_path):
    m = R.Metric("faithfulness", 0.9, 0.0, n=3, total=10)
    assert "n=3/10" in m.format()


def test_a_single_repeat_warns_that_no_noise_floor_is_known(tmp_path):
    out = _run_with(tmp_path, [systems.Answer(text="58")], repeats=1, name="r1")
    assert "no spread is known" in R.render_path(out)


def test_compare_refuses_to_call_a_difference_inside_the_noise_readable(tmp_path):
    a = _run_with(tmp_path, [systems.Answer(text="58"), systems.Answer(text="12")],
                  repeats=2, name="A")
    b = _run_with(tmp_path, [systems.Answer(text="58"), systems.Answer(text="12")],
                  repeats=2, name="B")
    text = R.compare(a, b)
    assert "inside noise" in text


def test_compare_warns_when_the_question_sets_differ(tmp_path):
    a = _run_with(tmp_path, [systems.Answer(text="58")], name="A2")
    p = write(tmp_path, [_record(qid="different")], name="other.jsonl")
    b = runner.run(Q.load(p), systems.StubSystem(), out_dir=tmp_path / "B2")
    assert "not comparable" in R.compare(a, b)


# --------------------------------------------------------------------------
# the loop itself (S-22)
# --------------------------------------------------------------------------

def test_the_stub_scores_zero_and_a_real_answer_does_not(tmp_path):
    """S-22's point, as a test: point the harness at something that answers
    nothing, and the number must move when something real is plugged in.
    Until this passes, no measurement downstream means anything."""
    path = write(tmp_path, [_record()])
    qset = Q.load(path)

    class Correct:
        name, config = "correct", {}

        def answer(self, q):
            return systems.Answer(text="58 analyses used Pythia.", papers=["1"])

    stub = runner.run(qset, systems.StubSystem(), out_dir=tmp_path / "s")
    real = runner.run(qset, Correct(), out_dir=tmp_path / "r")

    s = R.summarise(runner.load_records(stub)[1])
    r = R.summarise(runner.load_records(real)[1])
    assert s["count_correct"].mean == 0.0
    assert r["count_correct"].mean == 1.0
    assert s["answered"].mean == 0.0 and r["answered"].mean == 1.0


def test_one_slow_question_cannot_dominate_a_run(tmp_path):
    """2026-08-02: a single question consumed an hour of a 53-question run and
    produced nothing. A run is a long unattended job -- one pathological question
    must cost a minute, not an evening."""
    import time as _time

    class Slow:
        name, config = "slow", {}

        def answer(self, q):
            _time.sleep(30)
            return systems.Answer(text="eventually")

    path = write(tmp_path, [_record()])
    started = _time.time()
    out = runner.run(Q.load(path), Slow(), out_dir=tmp_path / "runs", timeout=0.5)
    assert _time.time() - started < 10, "it waited for the slow answer"
    _, records = runner.load_records(out)
    assert "Timeout" in records[0].answer["error"]
    assert records[0].scores["errored"] == 1.0


def test_the_planner_caps_its_completion_length():
    """Without a cap one call runs to the context limit; at ~26 tok/s that
    exceeds the client's 120s timeout, retries, and regenerates -- the hang looks
    like a stall rather than an error.

    Checks that a cap is APPLIED, not which constant supplies it: the cap became
    model-dependent in D-084, and a test pinned to the old spelling would have
    to be edited every time the policy changes, which is how a guard stops
    guarding.
    """
    import inspect

    from hepcoveragekg.query import planner

    assert planner.MAX_COMPLETION_TOKENS > 0
    src = inspect.getsource(planner._prepare)
    assert "max_tokens=cap" in src and "completion_cap(model)" in src


def test_reasoning_models_get_room_to_think():
    """800 tokens is the 72B's budget and truncates a reasoning model mid-thought.

    D-084: the truncated model returns an empty message with no tool call, the
    harness records answered=True with empty text, and the scorer gives it zero
    -- so a plumbing limit reads as a model that cannot answer. Measured on
    qwen3-32b, same everything else: free-SQL 0.199 -> 0.592.
    """
    from hepcoveragekg.query import planner

    assert planner.completion_cap("Qwen/Qwen2.5-72B-Instruct-AWQ") == 800
    for m in ("Qwen/QwQ-32B-AWQ", "qwen/qwen3-32b", "openai/gpt-5.6-sol",
              "deepseek/deepseek-v4-flash-0731"):
        assert planner.completion_cap(m) > 800, m


def test_an_explicit_cap_still_wins(monkeypatch):
    """The cluster's timeout arithmetic is real: an operator serving a slow local
    model must be able to pull the allowance back down."""
    from hepcoveragekg.query import planner

    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "900")
    assert planner.completion_cap("Qwen/QwQ-32B-AWQ") == 900


def test_the_read_only_connection_survives_the_timeout_worker(tmp_path):
    """The timeout guard runs `answer` in a worker thread, and SQLite refuses
    cross-thread use by default. Caught the hard way on 2026-08-02: the guard
    turned an hour-long stall into 53 instant ProgrammingErrors, which is a
    different kind of useless."""
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor

    from hepcoveragekg.query import templates as T

    db = tmp_path / "t.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE entity (entity_id TEXT, kind TEXT, label TEXT)")
    c.commit()
    c.close()

    conn = T.read_only(db)  # built here, in the main thread
    with ThreadPoolExecutor(max_workers=1) as pool:
        rows = pool.submit(lambda: conn.execute("SELECT * FROM entity").fetchall()).result()
    assert rows == []


def test_known_positive_recall_ignores_extra_papers(tmp_path):
    """Returning more than the known positive is not an error -- the concept
    boundary is genuinely uncertain, which is the reason this shape exists."""
    q = Q.parse(_record(shape="set", truth={"kind": "subset", "value": 1, "papers": ["p1"]},
                        truth_source="sql"))
    a = systems.Answer(text="", papers=["p1", "p2", "p3"])
    out = scoring.known_positive_recall(q, a)
    assert out["retrieved_paper_recall"] == 1.0
    assert out["returned_papers"] == 3.0


def test_known_positive_recall_reports_set_size_so_it_cannot_be_gamed(tmp_path):
    """Recall of a subset is trivially gamed by returning everything. The
    companion number is what makes that visible."""
    q = Q.parse(_record(shape="set", truth={"kind": "subset", "value": 1, "papers": ["p1"]},
                        truth_source="sql"))
    dumped = systems.Answer(text="", papers=[f"p{i}" for i in range(60)])
    out = scoring.known_positive_recall(q, dumped)
    assert out["retrieved_paper_recall"] == 1.0, "recall alone looks perfect"
    assert out["returned_papers"] == 60.0, "and the companion metric exposes it"


def test_set_f1_does_not_also_score_subset_questions(tmp_path):
    """Two scorers on one question would double-count it and, worse, penalise
    the extras that `subset` deliberately allows."""
    q = Q.parse(_record(shape="set", truth={"kind": "subset", "value": 1, "papers": ["p1"]},
                        truth_source="sql"))
    assert scoring.set_f1(q, systems.Answer(papers=["p1", "p2"])) is None


def test_retrieval_recall_and_mention_recall_are_separate(tmp_path):
    """They diverge badly: asked which papers used Pythia the planner retrieved
    most of the corpus and wrote down 13, because no English answer lists 58.
    Scoring only the prose measured its writing, not its searching."""
    q = Q.parse(_record(shape="set", truth={"kind": "subset", "value": 1, "papers": ["p9"]},
                        truth_source="sql", provenance={"concept_papers": 4}))
    found_but_unsaid = systems.Answer(text="Several analyses did.", papers=["p9", "p1"])
    out = scoring.known_positive_recall(q, found_but_unsaid)
    assert out["retrieved_paper_recall"] == 1.0
    assert out["mentioned_paper_recall"] == 0.0
    assert out["truncation_loss"] == 1.0


def test_the_mention_scorer_abstains_on_common_concepts(tmp_path):
    """Asking a model to list 58 papers and scoring whether one survived the
    summary measures nothing about the system."""
    common = Q.parse(_record(shape="set", truth={"kind": "subset", "value": 1, "papers": ["p9"]},
                             truth_source="sql", provenance={"concept_papers": 58}))
    out = scoring.known_positive_recall(common, systems.Answer(papers=["p9"]))
    assert "retrieved_paper_recall" in out
    assert "mentioned_paper_recall" not in out, "unfair by construction, so not scored"


def test_retrieved_entities_include_what_tools_returned_not_only_search_sets():
    """`contents_of` returns entities without going through `search`, so reading
    `session.sets` alone reported zero retrieval while the answer named 83% of
    them correctly -- an impossible pair, and the sign that the metric was
    following one code path rather than the system."""
    session = types.SimpleNamespace(
        answer="x", answerable=True, sets={"set_1": ["a"]},
        known_entity_ids={"a", "b", "c"}, steps=[], evidence_ids=[], seen_values=set(),
        llm_calls=1, prompt_tokens=1, completion_tokens=1, rounds=1, seconds=1.0,
        verification=None,
    )
    got = systems.from_session(session)
    assert set(got.entity_ids) == {"a", "b", "c"}, "everything the graph returned"


def test_a_dead_endpoint_stops_the_run_instead_of_hanging(tmp_path):
    """An overnight job can outlive its server. Without this it then spends the
    full timeout on every remaining question -- 1,352 questions at 180 s is three
    days of a process hanging on a socket."""
    class Dead:
        name, config = "dead", {}

        def answer(self, q):
            return systems.Answer(error="ConnectionError: endpoint gone")

    path = write(tmp_path, [_record(qid=f"q{i}") for i in range(50)])
    out = runner.run(Q.load(path), Dead(), out_dir=tmp_path / "runs",
                     abort_after_consecutive_errors=5)
    _, records = runner.load_records(out)
    assert len(records) == 5, "stopped early, and kept what it had"


def test_scattered_errors_do_not_abort_a_healthy_run(tmp_path):
    """Consecutive, not total: a handful of scattered failures is normal."""
    class Flaky:
        name, config = "flaky", {}

        def __init__(self):
            self.n = 0

        def answer(self, q):
            self.n += 1
            return (systems.Answer(error="blip") if self.n % 3 == 0
                    else systems.Answer(text="58"))

    path = write(tmp_path, [_record(qid=f"q{i}") for i in range(30)])
    out = runner.run(Q.load(path), Flaky(), out_dir=tmp_path / "runs",
                     abort_after_consecutive_errors=5)
    _, records = runner.load_records(out)
    assert len(records) == 30, "ran to completion despite regular failures"


def test_an_aborted_run_leaves_a_READABLE_file(tmp_path):
    """The abort used to finalise the file from inside the still-open `with`
    block, so `_rewrite_meta` interleaved with the buffer and produced lines like
    `{"r{"run_id":...`. The run stopped correctly and every reader then died on a
    JSONDecodeError -- a good guard made useless by where it was placed."""
    class Dead:
        name, config = "dead", {}

        def answer(self, q):
            return systems.Answer(error="ModuleNotFoundError: no langgraph")

    path = write(tmp_path, [_record(qid=f"q{i}") for i in range(40)])
    out = runner.run(Q.load(path), Dead(), out_dir=tmp_path / "runs",
                     abort_after_consecutive_errors=5)
    meta, records = runner.load_records(out)   # must not raise
    assert meta["n_records"] == 5 and len(records) == 5


def test_run_ids_do_not_collide_between_simultaneous_jobs(monkeypatch):
    """Four jobs submitted together start in the same second. With a timestamp
    alone they generate identical ids, open the same file with mode "w", and
    truncate each other -- leaving one interleaved file that still parses, so
    nothing looks wrong until the results make no sense."""
    monkeypatch.setenv("SLURM_JOB_ID", "111")
    a = runner.new_run_id("hepkg")
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    b = runner.new_run_id("hepkg")
    assert a != b, "same second, different jobs, must differ"


def test_the_critic_arm_hashes_differently_from_the_control():
    """Two runs differing only by the critic must not look like repeats of one
    another -- the config hash is what the harness uses to tell arms apart, and
    a collision would silently pool the treatment with its own control."""
    from hepcoveragekg.eval import systems

    off = systems.PlannerSystem(None, None, max_rounds=6, use_critic=False)
    on = systems.PlannerSystem(None, None, max_rounds=6, use_critic=True)
    assert off.config["use_critic"] is False and on.config["use_critic"] is True
    assert systems.config_hash(off.config) != systems.config_hash(on.config)


def test_the_run_record_carries_what_the_critic_did():
    """An ablation that reports a score difference with no evidence of what
    produced it is unreadable. This is the D-059 shape: the CLI stripped the
    provenance and the unit test bypassed the CLI, so the fix was invisible.
    Assert on what `from_session` really emits, not on the Session."""
    from hepcoveragekg.eval import systems
    from hepcoveragekg.query import critic as C, planner

    session = planner.Session(question="q")
    session.steps.append(planner.Step(1, "contents_of", {}, rows=9,
                                      redirected_from="describe"))
    session.reviews.append(C.Review("q", "Pythia", [
        C.Verdict("e1", C.EXACT, "the version asked for"),
        C.Verdict("e2", C.UNRELATED, "Herwig, a different generator"),
    ]))
    session.recovered_calls = 3
    answer = systems.from_session(session)

    assert answer.recovered_calls == 3, "the server's parser failures must survive"
    assert answer.steps[0]["redirected_from"] == "describe"
    review = answer.reviews[0]
    assert review["tally"][C.UNRELATED] == 1 and review["kept"] == 1
    assert review["dropped"] == [{"id": "e2", "why": "Herwig, a different generator"}]


def test_a_run_without_a_critic_carries_an_empty_record_not_a_missing_one():
    from hepcoveragekg.eval import systems
    from hepcoveragekg.query import planner

    answer = systems.from_session(planner.Session(question="q"))
    assert answer.reviews == [] and answer.recovered_calls == 0


# -- scoring the answer, not the retrieval footprint -------------------------

def _q(shape="set", papers=None, value=None, kind=None):
    from hepcoveragekg.eval.questions import Question, Truth
    return Question(qid="q1", text="which analyses?", shape=shape, split="dev",
                    truth=Truth(kind=kind or ("set" if shape == "set" else "count"),
                                value=value, papers=papers or [], items=[]))


def _a(text="", papers=None, value=None):
    from hepcoveragekg.eval.systems import Answer
    return Answer(text=text, answered=True, papers=papers or [], value=value)


def test_set_scoring_reads_the_answer_not_the_search_footprint():
    """`a.papers` is every paper any retrieved entity appears in -- a by-product
    of looking things up. Grading it compared a 39-paper footprint against a
    2-paper gold and scored a perfect answer at 0.05."""
    from hepcoveragekg.eval import scoring

    footprint = [f"20{i:02d}.0000{i%10}" for i in range(39)]
    out = scoring.set_f1(_q(papers=["2308.02285", "2312.04450"]),
                         _a(text="Two analyses: 2308.02285 and 2312.04450.",
                            papers=footprint))
    assert out["set_precision"] == 1.0 and out["set_recall"] == 1.0


def test_the_footprint_is_still_measured_under_its_own_name():
    """It measures something real -- whether retrieval REACHED the right papers
    -- just not answer quality. Separate name so the two cannot be confused."""
    from hepcoveragekg.eval import scoring

    out = scoring.retrieval_reach(_q(papers=["2308.02285", "2312.04450"]),
                                  _a(text="no ids here", papers=["2308.02285"]))
    assert out == {"retrieval_reach": 0.5}


def test_a_gold_too_long_to_list_abstains_rather_than_punishing_brevity():
    """No prose answer lists forty papers, so scoring one would measure
    truncation. Abstaining shows up honestly as reduced coverage."""
    from hepcoveragekg.eval import scoring

    big = [f"2{i:03d}.00001" for i in range(scoring.MAX_LISTABLE + 5)]
    assert scoring.set_f1(_q(papers=big), _a(text="2000.00001")) is None
    small = big[:scoring.MAX_LISTABLE]
    assert scoring.set_f1(_q(papers=small), _a(text="2000.00001")) is not None


def test_naming_nothing_falls_back_and_is_flagged():
    from hepcoveragekg.eval import scoring

    out = scoring.set_f1(_q(papers=["2308.02285"]),
                         _a(text="I found some analyses.", papers=["2308.02285"]))
    assert out["set_recall"] == 1.0
    assert out["set_named_none"] == 1.0, "scored on the fallback -- say so"


def test_the_claimed_number_is_extracted_so_direction_is_visible():
    """`exact_count` says hit or miss. Over- and under-counting are different
    diagnoses: too high means junk in the set, too low means over-filtering."""
    from hepcoveragekg.eval import scoring

    q = _q(shape="count", value=38, kind="count")
    assert scoring.claimed_count(q, _a(text="...is recorded in 41 papers."))["count_error"] == 3.0
    assert scoring.claimed_count(q, _a(text="12 analyses use it."))["count_error"] == -26.0
    assert scoring.claimed_count(q, _a(text="1,286 papers"))["claimed_count"] == 1286.0
    assert scoring.claimed_count(q, _a(text="no number at all")) is None


def test_the_answer_contract_is_its_own_arm():
    """It adds a tool and changes `answer`'s schema, so a run with it is a
    different system -- the config hash has to say so."""
    from hepcoveragekg.eval import systems

    v1 = systems.PlannerSystem(None, None, max_rounds=6, answer_contract=False)
    v2 = systems.PlannerSystem(None, None, max_rounds=6, answer_contract=True)
    assert systems.config_hash(v1.config) != systems.config_hash(v2.config)


def test_the_v1_tool_list_is_unchanged_by_the_new_contract():
    """The arms differ by one flag only if v1 is byte-for-byte what it was."""
    from hepcoveragekg.query import planner

    v1 = {t["name"] for t in planner.tools_for(False)}
    assert "refine" not in v1
    answer = [t for t in planner.tools_for(False) if t["name"] == "answer"][0]
    assert set(answer["parameters"]["properties"]) == {"text", "answerable", "reason"}
    # and the module constant must not have been mutated by building v1
    spec = [t for t in planner.TOOL_SPECS if t["name"] == "answer"][0]
    assert "papers_from" in spec["parameters"]["properties"]


# --------------------------------------------------------------------------
# Concurrency (added 2026-09-02). The runner answered one question at a time;
# the bottleneck is a ~90s network call, so a 300-question arm took 22 hours.
# --------------------------------------------------------------------------

def _concurrency_questions(n=12):
    from hepcoveragekg.eval.questions import Question, QuestionSet
    return QuestionSet([
        Question(qid=f"q{i}", text=f"question {i}", source="test", split="dev",
                 shape="count", truth={"kind": "count", "value": i})
        for i in range(n)
    ])


class _CountingSystem:
    """Answers with the question's own index, and records which instance ran it."""
    name = "stub"
    config: dict = {}

    def __init__(self, tag):
        self.tag = tag
        self.seen = []

    def answer(self, q):
        from hepcoveragekg.eval.systems import Answer
        import time as _t
        idx = int(q.qid[1:])
        _t.sleep(0.05)              # long enough that workers genuinely overlap
        self.seen.append(q.qid)
        return Answer(text=str(idx), answered=True, seconds=0.05,
                      value=idx, papers=[])


def test_concurrent_run_produces_every_record_exactly_once(tmp_path):
    """Concurrency must not drop, duplicate, or mis-pair (qid, repeat)."""
    from hepcoveragekg.eval import runner

    qs = _concurrency_questions(12)
    made = []
    def factory():
        s = _CountingSystem(f"w{len(made)}"); made.append(s); return s

    path = runner.run(qs, _CountingSystem("main"), repeats=3, out_dir=tmp_path,
                      timeout=None, max_workers=4, make_system=factory)
    _, records = runner.load_records(path)

    assert len(records) == 36, f"expected 12x3, got {len(records)}"
    pairs = sorted((r.qid, r.repeat) for r in records)
    assert pairs == sorted((f"q{i}", rep) for i in range(12) for rep in range(3))
    # each record must carry ITS OWN question's answer, not a neighbour's
    for r in records:
        assert r.answer["value"] == int(r.qid[1:]), (
            f"{r.qid} got value {r.answer['value']} -- answers crossed between questions")
    assert len(made) > 1, "concurrency should have built more than one system"


def test_concurrency_refuses_without_a_factory(tmp_path):
    """The silent-corruption path must be closed off loudly.

    FreeSQLSystem names its search sets `search_N` in temp tables on one
    connection. Shared across threads the counter races and one question drops
    another's table mid-join -- plausible, wrong answers. Refusing is the only
    safe default.
    """
    import pytest
    from hepcoveragekg.eval import runner

    with pytest.raises(ValueError, match="not.*thread-safe|make_system"):
        runner.run(_concurrency_questions(4), _CountingSystem("main"),
                   repeats=1, out_dir=tmp_path, timeout=None,
                   max_workers=8, make_system=None)


def test_serial_and_concurrent_agree(tmp_path):
    """Same questions, same answers, regardless of worker count."""
    from hepcoveragekg.eval import runner

    qs = _concurrency_questions(8)
    p1 = runner.run(qs, _CountingSystem("serial"), repeats=2,
                    out_dir=tmp_path / "a", timeout=None, max_workers=1)
    p2 = runner.run(qs, _CountingSystem("main"), repeats=2,
                    out_dir=tmp_path / "b", timeout=None, max_workers=5,
                    make_system=lambda: _CountingSystem("w"))
    _, r1 = runner.load_records(p1)
    _, r2 = runner.load_records(p2)
    got = lambda rs: sorted((r.qid, r.repeat, r.answer["value"]) for r in rs)
    assert got(r1) == got(r2)


def test_count_closeness_distinguishes_near_from_far():
    """D-096: exact_count scored a claim of 11 and a claim of 500 identically
    against a truth of 12 -- both zero. Across 207 real count questions on
    QwQ, two identical runs never disagreed on a single one for that reason:
    every question was right every time or wrong every time. count_closeness
    exists to recover the ones in between."""
    q = Q.parse(_record())  # truth value = 58
    assert scoring.count_closeness(q, systems.Answer(value=58))["count_closeness"] == 1.0
    close = scoring.count_closeness(q, systems.Answer(value=57))["count_closeness"]
    far = scoring.count_closeness(q, systems.Answer(value=6))["count_closeness"]
    wild = scoring.count_closeness(q, systems.Answer(value=5000))["count_closeness"]
    assert close > far > wild
    assert wild == 0.0  # floored, never negative
    assert 0.9 < close < 1.0


def test_count_closeness_is_relative_not_absolute():
    """Being off by one paper against a truth of 12 is a good answer; the same
    absolute gap against a truth of 2 is a bad one. The metric must know that."""
    small = Q.parse(_record(truth={"kind": "count", "value": 2, "papers": ["1"]}))
    big = Q.parse(_record(truth={"kind": "count", "value": 12, "papers": ["1"]}))
    off_by_one_small = scoring.count_closeness(small, systems.Answer(value=1))["count_closeness"]
    off_by_one_big = scoring.count_closeness(big, systems.Answer(value=11))["count_closeness"]
    assert off_by_one_big > off_by_one_small


def test_count_closeness_zero_truth_has_no_room_to_be_close():
    q = Q.parse(_record(truth={"kind": "count", "value": 0, "papers": []}))
    assert scoring.count_closeness(q, systems.Answer(value=0))["count_closeness"] == 1.0
    assert scoring.count_closeness(q, systems.Answer(value=1))["count_closeness"] == 0.0


def test_count_closeness_falls_back_to_prose_number_like_exact_count():
    q = Q.parse(_record())  # truth value = 58
    a = systems.Answer(text="I count 55 analyses.")
    assert scoring.count_closeness(q, a)["count_closeness"] > 0.9


def test_count_closeness_no_number_scores_zero_not_none():
    """Unlike exact_count's implicit False, an unparseable answer must still
    produce a number -- a scorer that abstains here would hide non-answers
    from the discrimination screen rather than correctly marking them worst."""
    q = Q.parse(_record())
    a = systems.Answer(text="I don't know.")
    assert scoring.count_closeness(q, a) == {"count_closeness": 0.0}


def test_count_closeness_abstains_for_non_count_questions():
    q = Q.parse(dict(_record(), shape="set",
                     truth={"kind": "labels", "items": ["hepkg:x"], "papers": []}))
    assert scoring.count_closeness(q, systems.Answer(value=3)) is None


def _judged(gold_n, universe_n=40):
    """A judged-gold set question with `gold_n` correct papers."""
    papers = [f"20{i:02d}.0000{i%10}" for i in range(gold_n)]
    universe = papers + [f"21{i:02d}.0000{i%10}" for i in range(universe_n - gold_n)]
    return Q.parse({"qid": "g1", "text": "Which analyses...?", "shape": "set",
                    "needs": ["sql"], "truth_source": "gabriel",
                    "truth": {"kind": "set", "papers": papers, "universe": universe}})


def test_judged_gold_up_to_25_is_scored_not_silently_dropped():
    """Gabriel's full batch 2 grew three golds past the old cap of 15 --
    gf-04 to 18, gf-05 to 16, gf-08 to 24 -- and judged_set_f1 returned None for
    all three. They ran, were answered, and scored NOTHING with no error, so
    improving the ground truth silently deleted the three most valuable
    questions. Every real Gabriel gold must score."""
    for n in (16, 18, 24):
        q = _judged(n)
        a = systems.Answer(text=f"See {q.truth.papers[0]} and {q.truth.papers[1]}.")
        out = scoring.judged_set_f1(q, a)
        assert out is not None, f"gold of {n} was silently dropped"
        assert out["judged_f1"] > 0


def test_judged_gold_beyond_any_listable_answer_still_abstains():
    """The guard still exists -- 26+ is past the largest answer ever observed."""
    q = _judged(26, universe_n=50)
    a = systems.Answer(text=f"See {q.truth.papers[0]}.")
    assert scoring.judged_set_f1(q, a) is None


def test_oversized_gold_is_flagged_so_capped_recall_is_readable():
    """A 24-paper gold caps recall by construction: only 1 answer in 562 named
    that many. The flag lets a low recall be read as the ceiling it partly is."""
    big = scoring.judged_set_f1(_judged(24), systems.Answer(text="See 2000.00000."))
    small = scoring.judged_set_f1(_judged(8), systems.Answer(text="See 2000.00000."))
    assert big["judged_gold_exceeds_typical"] == 1.0
    assert small["judged_gold_exceeds_typical"] == 0.0
