"""Tests for LLM rewording of generated questions.

The risk here is silent: a rewrite that drifts from its meaning keeps the OLD
truth attached, so the question looks fine and its answer is wrong. Every guard
below exists to make that loud instead.
"""
from __future__ import annotations

from hepcoveragekg.eval import questions as Q
from hepcoveragekg.eval import reword as R


def _q(text="How many analyses used the generator Pythia 8?", **over):
    base = dict(qid="gen-count-1", text=text, shape="count", needs=["sql"],
                truth={"kind": "count", "value": 58}, truth_source="sql",
                provenance={"label": "Pythia 8", "kind": "generator",
                            "predicate": "sample_uses_generator"})
    base.update(over)
    return Q.parse(base)


def test_faithful_mode_requires_the_name_to_survive():
    """The truth is inherited unchanged, so a rewrite that drops or shortens the
    entity silently relabels the question."""
    assert R.keeps_the_name("In how many papers was Pythia 8 used?", "Pythia 8")
    assert not R.keeps_the_name("How many analyses used Pythia?", "Pythia 8")


def test_faithful_guard_ignores_punctuation_and_case():
    assert R.keeps_the_name("how many used PYTHIA 8?", "Pythia 8")


def test_vague_mode_rejects_a_leaked_name():
    """If the entity appears, retrieval succeeds by string matching and the test
    measures BM25 rather than retrieval."""
    assert not R.hides_the_name("Which papers used Pythia for showering?", "Pythia 8")
    assert R.hides_the_name(
        "What did they use to model the parton shower in the ttbar samples?", "Pythia 8")


def test_vague_guard_ignores_short_words():
    """'of', 'b', 'jet' appear in ordinary English; rejecting on them would throw
    away almost every rewrite."""
    assert R.hides_the_name("Which analyses tag heavy-flavour jets?", "b jet")


def test_a_rewrite_keeps_the_original_truth_and_links_back():
    calls = []

    def chat(prompt):
        calls.append(prompt)
        return "In how many papers was Pythia 8 used to shower events?"

    out, stats = R.reword([_q()], mode="faithful", chat=chat)
    assert stats == {"asked": 1, "kept": 1, "failed_guard": 0, "errored": 0}
    assert out[0].truth.value == 58, "the label is inherited, never regenerated"
    assert out[0].relation == "faithful_rewrite_of:gen-count-1"
    assert out[0].provenance["original_text"] == _q().text


def test_a_drifted_rewrite_is_discarded_not_repaired():
    """Repairing would mean guessing what the model meant; a clumsy question is
    better than one whose answer quietly no longer applies."""
    out, stats = R.reword([_q()], mode="faithful",
                          chat=lambda p: "How many analyses used a parton shower?")
    assert out == [] and stats["failed_guard"] == 1


def test_a_leaked_name_is_discarded_in_vague_mode():
    out, stats = R.reword([_q()], mode="vague",
                          chat=lambda p: "Which analyses used Pythia 8 for showering?")
    assert out == [] and stats["failed_guard"] == 1


def test_a_failed_call_is_recorded_not_raised():
    def boom(prompt):
        raise RuntimeError("endpoint gone")

    out, stats = R.reword([_q()], mode="faithful", chat=boom)
    assert out == [] and stats["errored"] == 1


def test_output_must_actually_be_a_question():
    out, stats = R.reword([_q()], mode="faithful",
                          chat=lambda p: "Pythia 8 was used in 58 analyses.")
    assert out == [] and stats["failed_guard"] == 1


def test_the_rewrite_shares_a_group_with_its_original():
    """The report compares them directly -- the retrieval-breadth difference
    between a template question and its rewrite IS the ambiguity signal."""
    original = _q(group="mm-pythia")
    out, _ = R.reword([original], mode="faithful",
                      chat=lambda p: "In how many papers was Pythia 8 used?")
    assert out[0].group == "mm-pythia"


def test_a_failing_rewrite_is_retried_before_giving_up():
    """The vague guard rejects often -- 5 of 6 on the first real pass, because
    'describe it without naming it' is genuinely hard. Retrying is far cheaper
    than lowering the bar, and lowering it would defeat the purpose."""
    tries = []

    def chat(prompt):
        tries.append(1)
        return ("Which analyses used Pythia 8 for showering?" if len(tries) < 3
                else "What modelled the parton shower in those samples?")

    out, stats = R.reword([_q()], mode="vague", chat=chat, attempts=3)
    assert len(tries) == 3 and stats["kept"] == 1


def test_retries_are_bounded():
    out, stats = R.reword([_q()], mode="vague", attempts=2,
                          chat=lambda p: "Which analyses used Pythia 8?")
    assert out == [] and stats["failed_guard"] == 1
