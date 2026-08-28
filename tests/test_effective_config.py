"""A run must be able to name its own arm.

25 of 44 stored runs record no configuration at all, because `config` was
`{**planner_kwargs}` -- only what the caller overrode. Everything left at its
default was invisible, and the three knobs the ablation turns on are exactly the
defaulted ones: `critic_seed` (None ranked, int shuffled), `contract`, and the
`CRITIC_BASE_URL` that picks the 8B judge over the 72B. Which arm produced a
result had to be reconstructed from job scripts.
"""
from __future__ import annotations

import inspect

import pytest

from hepcoveragekg.eval.systems import (
    _NOT_CONFIG,
    config_hash,
    effective_config,
)
from hepcoveragekg.query import planner


def test_every_knob_on_the_planner_is_recorded():
    """The drift guard. Add a parameter to `planner.answer` without deciding
    whether it is configuration, and this fails -- which is the point: the cost
    of forgetting is a week of unattributable runs, and it is silent."""
    config = effective_config(planner.answer, {})
    for name, param in inspect.signature(planner.answer).parameters.items():
        if name in _NOT_CONFIG:
            continue
        assert name in config, (
            f"planner.answer has a knob '{name}' that no run will record. "
            f"Add it to the config or list it in _NOT_CONFIG as plumbing."
        )


def test_defaults_are_recorded_not_just_overrides():
    config = effective_config(planner.answer, {})
    assert config["critic_seed"] is None
    assert config["contract"] == ""
    assert config["force_critic_set"] is False
    assert config["critic_order"] == "ranked"


@pytest.mark.parametrize("override,knob", [
    ({"critic_seed": 7}, "the ranked/shuffled arm"),
    ({"contract": "v3"}, "the answer contract"),
    ({"use_critic": True}, "the critic itself"),
    ({"force_critic_set": True}, "set substitution"),
])
def test_changing_an_arm_changes_the_hash(override, knob):
    """Two arms that hash the same are two results that cannot be told apart."""
    base = config_hash(effective_config(planner.answer, {}))
    assert config_hash(effective_config(planner.answer, override)) != base, \
        f"{knob} does not change the config hash"


def test_the_environment_is_part_of_the_configuration(monkeypatch):
    """The judge model and the search breadth never passed through
    `planner.answer`, so a run recording only its arguments still could not say
    which judge answered it."""
    base = config_hash(effective_config(planner.answer, {}))
    monkeypatch.setenv("CRITIC_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ")
    assert config_hash(effective_config(planner.answer, {})) != base
    monkeypatch.setenv("CRITIC_MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
    monkeypatch.setenv("SEARCH_BREADTH_MAX", "60")
    assert config_hash(effective_config(planner.answer, {})) != base


def test_live_objects_never_reach_the_hash():
    """`chat` and `checkpointer` are per-process objects; hashing their repr
    would give every run a different config hash and silently destroy the
    comparison the hash exists for."""
    with_chat = effective_config(planner.answer, {"chat": lambda *a, **k: None})
    assert config_hash(with_chat) == config_hash(effective_config(planner.answer, {}))
