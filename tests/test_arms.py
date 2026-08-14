"""Comparing two sharded ablation arms.

The tests that matter here are the ones about the comparison being a comparison
at all: same questions, one system, and the flag under test having reached it.
Every one of them is a failure this project has actually had.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hepcoveragekg.eval import arms


def shard(tmp_path: Path, name: str, qids, *, config="cfg-off", sha="abc123",
          reviews=None, scores=None):
    path = tmp_path / f"{name}.jsonl"
    lines = [json.dumps({"_meta": {"run_id": name, "system": "hepkg",
                                   "config_hash": config, "git_sha": sha,
                                   "questions_path": f"{name}.jsonl",
                                   "questions_hash": name}})]
    for q in qids:
        lines.append(json.dumps({
            "run_id": name, "qid": q, "repeat": 0, "system": "hepkg",
            "config_hash": config, "git_sha": sha, "question": q,
            "shape": "count", "split": "dev", "needs": ["sql"], "difficulty": "easy",
            "answer": {"text": "x", "answered": True, "steps": [],
                       "reviews": reviews or [], "recovered_calls": 2},
            "scores": scores or {"count_correct": 1.0},
        }))
    path.write_text("\n".join(lines))
    return path


def test_shards_merge_into_one_arm(tmp_path):
    a = shard(tmp_path, "a0", ["q1", "q2"])
    b = shard(tmp_path, "a1", ["q3"])
    meta, recs = arms.merge([a, b])
    assert meta["shards"] == 2 and len(recs) == 3
    assert meta["shard_run_ids"] == ["a0", "a1"]


def test_arms_are_aligned_on_question_ids_not_counts(tmp_path):
    """A shard that died leaves its arm short. Comparing the survivors against a
    complete arm compares two different question sets and reports one clean
    delta -- which is the whole failure this guards."""
    a = shard(tmp_path, "ctrl", ["q1", "q2", "q3"])
    b = shard(tmp_path, "crit", ["q1", "q2"], config="cfg-on")
    _, rec_a = arms.merge([a])
    _, rec_b = arms.merge([b])
    left, right, overlap = arms.align(rec_a, rec_b)
    assert len(left) == len(right) == 2
    assert overlap["a_only"] == 1 and overlap["shared"] == 2
    assert [r.qid for r in left] == [r.qid for r in right]


def test_a_dropped_shard_is_reported_not_hidden(tmp_path):
    a = shard(tmp_path, "ctrl", ["q1", "q2", "q3"])
    b = shard(tmp_path, "crit", ["q1"], config="cfg-on")
    text = arms.compare_arms([a], [b])
    assert "aligned on 1 shared" in text
    assert "2 answered only by control" in text


def test_arms_differing_by_more_than_the_flag_are_flagged(tmp_path):
    """Code moved between the arms -- the confound caught twice on 2026-08-14."""
    a = shard(tmp_path, "ctrl", ["q1"], sha="aaa")
    b = shard(tmp_path, "crit", ["q1"], sha="bbb", config="cfg-on")
    text = arms.compare_arms([a], [b])
    assert "git_sha differs between arms" in text
    assert "more than the flag under test" in text


def test_identical_configs_mean_the_flag_never_took(tmp_path):
    a = shard(tmp_path, "ctrl", ["q1"])
    b = shard(tmp_path, "crit", ["q1"])          # same config_hash
    text = arms.compare_arms([a], [b])
    assert "share a config_hash" in text


def test_a_dirty_sha_is_called_out(tmp_path):
    a = shard(tmp_path, "ctrl", ["q1"], sha="abc-dirty")
    b = shard(tmp_path, "crit", ["q1"], sha="abc-dirty", config="cfg-on")
    assert "-dirty" in arms.compare_arms([a], [b])


def test_a_critic_arm_that_judged_nothing_says_so(tmp_path):
    """The quietest possible failure: the flag is set, the config hashes differ,
    and the critic never ran."""
    a = shard(tmp_path, "ctrl", ["q1"])
    b = shard(tmp_path, "crit", ["q1"], config="cfg-on")
    assert "judged nothing" in arms.compare_arms([a], [b])


def test_critic_activity_is_aggregated(tmp_path):
    review = {"search_text": "Pythia", "candidates": 60, "kept": 31,
              "defaulted": 0, "calls": 4, "errors": 0,
              "tally": {"exact": 20, "broader": 11, "unrelated": 29}}
    b = shard(tmp_path, "crit", ["q1", "q2"], config="cfg-on", reviews=[review])
    _, recs = arms.merge([b])
    act = arms.critic_activity(recs)
    assert act["searches_judged"] == 2
    assert act["candidates"] == 120 and act["kept"] == 62
    assert act["rungs"]["unrelated"] == 58
    assert act["recovered_calls"] == 4, "the server's parser failures aggregate too"


def test_a_widened_search_is_counted(tmp_path):
    wide = {"search_text": "jet energy scale", "candidates": 120, "kept": 100,
            "defaulted": 0, "calls": 8, "errors": 0, "tally": {"exact": 100}}
    b = shard(tmp_path, "crit", ["q1"], config="cfg-on", reviews=[wide])
    _, recs = arms.merge([b])
    assert arms.critic_activity(recs)["widened"] == 1
