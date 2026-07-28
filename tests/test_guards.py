# =============================================================================
# Tier 2.5 semantic guards -- the deterministic safety layer.
#
# These vetoes are the only thing standing between a high-similarity embedding
# score and an LLM call on a pair that is physically distinct. The cost is
# asymmetric: a veto is FINAL (the pair never reaches adjudication), a pass just
# costs one call. So the tests come in two halves -- vetoes that MUST fire, and
# true synonyms that must NOT be vetoed.
# =============================================================================
from __future__ import annotations

import pytest

from hepcoveragekg.aliases.guards import (
    _digit_signature,
    _extract_numbers,
    _extract_units,
    passes_semantic_guards,
)


def vetoed(a: str, b: str) -> bool:
    return not passes_semantic_guards(a, b)[0]


# --- must veto: physically distinct ------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        ("13 TeV", "14 TeV"),                  # collision energy
        ("Pythia8.212", "Pythia8.230"),        # generator version
        ("sherpa 2.2.1", "sherpa 2.2.2"),
        ("139 fb-1", "140 fb-1"),              # integrated luminosity
        ("MadGraph5 2.6", "MadGraph5 2.6.2"),
        ("pT > 20 GeV", "pT > 25 GeV"),        # working point
    ],
)
def test_different_numbers_are_vetoed(a, b):
    assert vetoed(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("20 GeV", "20 TeV"),                  # same number, different scale
        ("cross section in pb", "cross section in fb"),
        ("13TeV", "13GeV"),                    # glued: no separator to split on
    ],
)
def test_different_units_are_vetoed(a, b):
    assert vetoed(a, b)


def test_glued_units_are_caught():
    """Regression: splitting on separators alone leaves '13tev' as one token
    matching no unit, so '13TeV' vs '13GeV' passed the guard entirely."""
    assert _extract_units("13tev") == {"tev"}
    assert _extract_units("139fb") == {"fb"}
    assert vetoed("13TeV", "13GeV")


# --- must NOT veto: same thing, or nothing to compare ------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        ("Pythia 8.212", "Pythia8 212"),       # separator variants of ONE version
        ("sherpa 2.2.1", "sherpa 2 2 1"),
        ("pythia8.230", "pythia-8-230"),
    ],
)
def test_separator_variants_of_one_version_survive(a, b):
    """Regression: comparing the SET of number tokens made {'8.212'} != {'8','212'},
    vetoing a true synonym. A veto is unrecoverable, so this had to be fixed --
    the digit signature is what gets compared now."""
    assert not vetoed(a, b)


def test_digit_signature_ignores_punctuation_but_not_value():
    assert _digit_signature("Pythia 8.212") == _digit_signature("Pythia8 212") == "8212"
    assert _digit_signature("sherpa 2.2.1") == _digit_signature("sherpa 2 2 1") == "221"
    assert _digit_signature("pythia8.212") != _digit_signature("pythia8.230")


def test_ab_initio_is_prose_not_attobarns():
    """Regression: treating the word 'ab' as a unit made 'ab initio method' vs
    'pb-based method' a spurious unit mismatch."""
    assert _extract_units("ab initio method") == set()
    assert not vetoed("ab initio method", "pb-based method")
    # still a unit when it follows a number, which is how attobarns appear
    assert _extract_units("300 ab-1") == {"ab"}


@pytest.mark.parametrize(
    "a, b",
    [
        ("MadGraph", "MG5"),                   # only one side states a version
        ("Pythia", "Pythia8"),
        ("signal region", "control region"),   # no numbers, no units -> LLM's call
        ("colour reconnection", "color reconnection"),
        ("139 fb-1", "139 fb^-1"),             # same values, different formatting
    ],
)
def test_pairs_with_nothing_to_compare_pass_through(a, b):
    assert not vetoed(a, b)


def test_guards_do_not_judge_semantics():
    """The guards are deliberately narrow: look-alikes that carry no number or
    unit are NOT their job. Vetoing these would be guessing, and the LLM is the
    stage equipped to decide."""
    for a, b in [("b-jet", "c-jet"), ("s-channel", "t-channel"),
                 ("W boson polarization", "Z boson polarization")]:
        assert not vetoed(a, b)


# --- unit / number extraction ------------------------------------------------


def test_unit_extraction_requires_a_word_boundary():
    """Substrings must not register as units."""
    assert _extract_units("Tevatron") == set()
    assert _extract_units("pbar-p collisions") == set()
    assert _extract_units("nbody simulation") == set()


def test_number_extraction_keeps_dotted_versions_whole():
    assert _extract_numbers("sherpa 2.2.1") == {"2.2.1"}
    assert _extract_numbers("13 TeV, 139 fb-1") == {"13", "139", "1"}
    assert _extract_numbers("no digits here") == set()


# --- general properties ------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [("13 TeV", "14 TeV"), ("Pythia 8.212", "Pythia8 212"), ("20 GeV", "20 TeV"), ("", "")],
)
def test_verdict_is_symmetric(a, b):
    assert passes_semantic_guards(a, b)[0] == passes_semantic_guards(b, a)[0]


def test_identical_strings_always_pass():
    for s in ["13 TeV", "Pythia8.212", "", "signal region"]:
        assert not vetoed(s, s)


def test_reason_is_empty_only_when_passing():
    assert passes_semantic_guards("13 TeV", "13 TeV") == (True, "passed")
    ok, reason = passes_semantic_guards("13 TeV", "14 TeV")
    assert not ok and "number_mismatch" in reason
    ok, reason = passes_semantic_guards("20 GeV", "20 TeV")
    assert not ok and "unit_mismatch" in reason
