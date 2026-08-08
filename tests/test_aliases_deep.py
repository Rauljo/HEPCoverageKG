"""Tests for the deep aliases pipeline: proposals, review sampling,
stability, and whole-cluster checking.
"""
from __future__ import annotations

import pytest




def test_load_proposals_reads_both_formats(tmp_path):
    """Runs before 2026-08-02 wrote one JSON array at the end; runs after stream
    JSONL so a killed overnight job keeps what it finished. Both exist on disk,
    and a reader that understood only one would silently return nothing."""
    import json as _json

    from hepcoveragekg.aliases import run as R

    old = tmp_path / "old.json"
    old.write_text(_json.dumps([{"term_a": "a", "term_b": "b", "is_match": True}]))
    new = tmp_path / "new.jsonl"
    new.write_text(_json.dumps({"term_a": "a", "term_b": "b", "is_match": True}) + "\n")
    assert len(R.load_proposals(old)) == 1
    assert len(R.load_proposals(new)) == 1


def test_review_sample_is_stratified_and_prefers_suspect_pairs(tmp_path):
    """Precision was judged by reading fourteen rows (D-044). Without a tool the
    next run gets judged the same way.

    `in_tainted_cluster` is the flag that matters here, not `transitivity_paradox`:
    the latter marks the REJECTED side of a contradiction and this file samples
    matches, so selecting on it prioritised nothing at all."""
    import json as _json

    from hepcoveragekg.aliases import run as R

    rows = []
    for i in range(20):
        rows.append({"term_a": f"a{i}", "term_b": f"b{i}", "kind": "systematic_uncertainty",
                     "is_match": True, "status": "ok", "confidence": 0.9,
                     "in_tainted_cluster": i < 3, "explanation": "x"})
    for i in range(5):
        rows.append({"term_a": f"c{i}", "term_b": f"d{i}", "kind": "generator",
                     "is_match": True, "status": "ok", "confidence": 0.9,
                     "in_tainted_cluster": False, "explanation": "x"})
    rows.append({"term_a": "no", "term_b": "match", "kind": "generator",
                 "is_match": False, "status": "ok", "confidence": 0.1, "explanation": "x"})

    src = tmp_path / "p.jsonl"
    src.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "review.tsv"
    stats = R.sample_for_review(src, out, per_kind=4, seed=1)

    assert stats["matches"] == 25, "rejections are not up for review"
    assert stats["kinds"] == 2, "both kinds represented, not just the big one"
    data = [l.split("\t") for l in out.read_text().splitlines() if not l.startswith("#")]
    assert sum(1 for r in data if len(r) > 3 and r[3] == "SUSPECT") == 3, \
        "every suspect pair included"
    assert "verdict" in out.read_text(), "a blank column to fill in"


def test_the_review_file_and_the_stability_check_use_the_SAME_pairs():
    """Deliberate: if a person labels one sample and stability is measured on a
    different one, a human/model disagreement cannot be read -- it is either
    'the model is wrong' or 'the model is unstable' and you cannot tell which."""
    import inspect

    from hepcoveragekg.aliases import run as R

    assert "stratified_matches" in inspect.getsource(R.sample_for_review)
    assert "stratified_matches" in inspect.getsource(R._stability_check)


def test_stratified_sampling_is_deterministic_for_a_seed():
    from hepcoveragekg.aliases import run as R

    rows = [{"term_a": f"a{i}", "term_b": f"b{i}", "kind": "k", "is_match": True,
             "status": "ok"} for i in range(50)]
    a = R.stratified_matches(rows, per_kind=5, seed=3)
    b = R.stratified_matches(rows, per_kind=5, seed=3)
    assert [r["term_a"] for r in a] == [r["term_a"] for r in b]


def test_a_failed_cluster_call_is_not_an_approval():
    """Recording an error as "one" would silently approve a chain nobody judged
    -- the exact failure the cluster check exists to catch."""
    import asyncio

    from hepcoveragekg.aliases import clusters as C

    out = asyncio.run(C.check_one({}, ["a", "b", "c"], "generator"))
    assert out["status"] == "error"
    assert out["verdict"] is None


def test_a_split_verdict_with_unusable_groups_still_blocks_the_merge():
    """The model said 'do not merge this' but could not say how to divide it.
    That is still a useful answer and must not be downgraded to 'one'."""
    from hepcoveragekg.aliases import clusters as C

    parsed = C._parse({"verdict": "split", "groups": [[0, 1]], "confidence": 0.9,
                       "explanation": "x"}, size=4)
    assert parsed["verdict"] == "split"
    assert parsed["groups"] == [], "an incomplete partition is dropped, the verdict is not"


def test_an_unreadable_verdict_raises_rather_than_defaulting():
    from hepcoveragekg.aliases import clusters as C

    with pytest.raises(ValueError):
        C._parse({"verdict": "maybe"}, size=3)


def test_the_prompt_shows_every_member_with_an_index():
    """The model must be able to name which members split off, so each one needs
    a stable number in the prompt."""
    from hepcoveragekg.aliases import clusters as C

    class Ctx:
        entity_id = "e"
        labels = ["Profile likelihood method"]
        aliases: list = []
        attributes: dict = {}
        quotes: list = []

    members = [C.render_member(Ctx(), i) for i in range(3)]
    prompt = C.build_prompt("statistical_method", members)
    assert "[0]" in prompt and "[1]" in prompt and "[2]" in prompt
    assert "groups" in prompt


def test_agreement_ignores_failed_calls():
    """A timeout is not the model changing its mind. Counting it as disagreement
    would make a flaky network look like an uncertain model."""
    from hepcoveragekg.aliases import confidence as C

    assert C.agreement([True, True, None, True]) == 1.0
    assert C.agreement([True, False, True, True]) == 0.75
    assert C.agreement([None, None]) == 0.0


def test_greedy_and_sampled_verdicts_are_kept_apart():
    """The temperature-0 verdict is what the merge was approved on; the sampled
    ones measure how sure the model is. Overwriting one with the other would lose
    the thing being measured."""
    import inspect

    from hepcoveragekg.aliases import confidence as C

    src = inspect.getsource(C.measure)
    assert '"greedy_is_match"' in src and '"sampled_verdicts"' in src


def test_the_confidence_sample_matches_the_review_sample():
    """If a person labels one sample and agreement is measured on another, a
    human/model disagreement cannot be read -- 'the model is wrong' and 'the
    model is unstable' need different fixes."""
    import inspect

    from hepcoveragekg.aliases import confidence as C

    assert "stratified_matches" in inspect.getsource(C.measure)


def test_the_streamed_file_is_brought_up_to_date_with_the_paradox_flags():
    """The stream is written during adjudication so a crash keeps completed work,
    which means it necessarily predates the transitivity check. Leaving it stale
    is not harmless: every downstream tool reads this file, and on 2026-08-02 that
    silently disabled paradox-first ordering in the review sample -- the file
    looked complete and the flag was absent from all 10,839 records."""
    import inspect

    from hepcoveragekg.aliases import run as R

    src = inspect.getsource(R.propose_deep_semantics)
    rewrite = src.index("Bring the streamed file up to date")
    paradox = src.index("Phase D: Transitivity")
    assert rewrite > paradox, "the rewrite must come after the flags exist"
