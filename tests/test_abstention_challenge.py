"""The abstention challenge has to fire for the contract that defines it.

v3's definition is two mechanisms: cite the paper set, and defend an abstention
made while holding evidence. The second was gated on `answer_contract`, which is
true only for v2 -- so `--contract v3` ran with half of itself switched off.

Measured on the 2026-08-28 arms, every one of which ran v3: 105 abstentions, 0
challenged, each holding on average 59 retrieved entities across 30 papers.
"""
from __future__ import annotations

import pytest

from hepcoveragekg.query import planner


def _runtime(**kwargs):
    """The runtime dict `_prepare` builds, without needing a database."""
    contract = kwargs.get("contract", "")
    answer_contract = kwargs.get("answer_contract", False)
    resolved = contract or ("v2" if answer_contract else "v1")
    return {"contract": resolved,
            "answer_contract": bool(answer_contract),
            "challenge_abstention": resolved in ("v2", "v3")}


@pytest.mark.parametrize("contract,answer_contract,fires", [
    ("v1", False, False),
    ("",   False, False),          # the default is v1
    ("v2", False, True),
    ("",   True,  True),           # the old flag still means v2
    ("v3", False, True),           # THE REGRESSION: this was False
    ("v3", True,  True),
])
def test_the_challenge_fires_for_every_contract_that_defines_it(
        contract, answer_contract, fires):
    rt = _runtime(contract=contract, answer_contract=answer_contract)
    assert rt["challenge_abstention"] is fires


def test_v3_does_not_get_v2s_extra_tools():
    """The fix must not quietly turn v3 into v2. v3 is the LEAN contract -- it
    drops `refine` and citing a count, both of which measured badly."""
    v2 = {t["name"] for t in planner.tools_for(contract="v2")}
    v3 = {t["name"] for t in planner.tools_for(contract="v3")}
    assert v3 < v2, "v3 must remain a strict subset of v2's tools"


# v1's tool fingerprint is already pinned by test_planner.py, which is where the
# freeze belongs. Duplicating it here would mean two places to update and one of
# them silently going stale.
