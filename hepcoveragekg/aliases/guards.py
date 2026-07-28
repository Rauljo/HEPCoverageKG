"""
HEPCoverageKG aliases: Phase B - Semantic Guards

Deterministic vetoes for pairs the embedding scores as similar but which are
physically distinct: different energies, luminosities, generator versions,
working points. Pure string comparison, no model.

Asymmetric cost. A veto is FINAL -- the pair never reaches the LLM -- whereas a
pass only costs one adjudication call, and the LLM is still there to reject it.
So these rules are written to be conservative: veto only on evidence that is
unambiguous, and let anything doubtful through.

Two rules:
  1. numbers differ  -> different quantity (13 vs 14 TeV, Pythia 8.212 vs 8.230)
  2. units differ    -> different scale   (20 GeV vs 20 TeV, pb vs fb)
Both only fire when BOTH strings carry the feature; "MadGraph" vs "MG5" is not
a number mismatch, because one side simply says nothing about a version.
"""
from __future__ import annotations

import re

# Numbers, including dotted versions (13, 2.2.1, 5.02).
_NUM_PATTERN = re.compile(r"\d+(?:\.\d+)*")

_UNITS = ("tev", "gev", "mev", "pb", "fb", "nb", "ab")

# A unit counts when it either follows a digit ("13TeV", "139 fb") or stands as
# its own word ("cross section in pb").
#
# Two subtleties, both learned from real labels:
#   - the digit-adjacent branch is what catches the GLUED form. Splitting on
#     separators alone leaves "13tev" as a single token that matches no unit, so
#     "13TeV" vs "13GeV" slipped through the guard entirely.
#   - "ab" is excluded from the standalone branch: "ab initio" is ordinary
#     physics prose, and treating it as attobarns made "ab initio method" vs
#     "pb-based method" a spurious unit mismatch. As attobarns it effectively
#     always follows a number, so the digit-adjacent branch still covers it.
_UNIT_RE = re.compile(
    r"\d\s*(" + "|".join(_UNITS) + r")\b"
    r"|\b(" + "|".join(u for u in _UNITS if u != "ab") + r")\b"
)


def _extract_numbers(s: str) -> set[str]:
    """All numeric sequences in the string."""
    return set(_NUM_PATTERN.findall(s))


def _digit_signature(s: str) -> str:
    """Every digit, in order, with separators discarded.

    This is what actually gets compared, because the set of number *tokens* is
    an artefact of punctuation: "Pythia 8.212" yields {"8.212"} while
    "Pythia8 212" yields {"8", "212"}. Those are the same release, and vetoing
    them would destroy a true synonym irrecoverably. Concatenating gives "8212"
    for both, while genuinely different versions ("8.212" -> 8212 vs "8.230" ->
    8230) still differ. Same trick as the Tier 1 normalizer.
    """
    return "".join(re.findall(r"\d", s))


def _extract_units(s: str) -> set[str]:
    """Physics units mentioned in the string."""
    return {a or b for a, b in _UNIT_RE.findall(s.lower())}


def passes_semantic_guards(str_a: str, str_b: str) -> tuple[bool, str]:
    """Apply the deterministic vetoes.

    Returns (True, "passed") if the pair may proceed to LLM adjudication, or
    (False, reason) if it is rejected outright.
    """
    # 1. Number guard. Two independent views of "the numbers differ", and the
    #    veto needs BOTH to agree -- each one alone produces false vetoes on real
    #    labels, and a false veto silently destroys a true synonym:
    #      - token sets alone: {"8.212"} != {"8", "212"} vetoes one version
    #        written two ways ("Pythia 8.212" vs "Pythia8 212").
    #      - digit signature alone: "3L channel (exactly 3 light leptons)" gives
    #        "33" while "(3ℓF)" gives "3", vetoing a genuine synonym because one
    #        label happens to repeat the multiplicity in prose.
    #    Requiring agreement keeps the real catches (13 vs 14 TeV, 8.212 vs
    #    8.230, 139 vs 140 fb) while dropping both false-veto modes.
    nums_a, nums_b = _extract_numbers(str_a), _extract_numbers(str_b)
    if nums_a and nums_b and nums_a != nums_b:
        if _digit_signature(str_a) != _digit_signature(str_b):
            return False, f"number_mismatch ({sorted(nums_a)} != {sorted(nums_b)})"

    # 2. Unit guard. Catches equal numbers at different scales (20 GeV / 20 TeV).
    units_a, units_b = _extract_units(str_a), _extract_units(str_b)
    if units_a and units_b and units_a != units_b:
        return False, f"unit_mismatch ({sorted(units_a)} != {sorted(units_b)})"

    return True, "passed"
