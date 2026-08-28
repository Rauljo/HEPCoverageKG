"""Region role: the distinction Gabriel's review needed and the schema lacked.

The graph already held the answer for 74% of regions -- under three different
keys with 33 spellings. These lock the normalisation, and more importantly lock
what it refuses to guess.
"""
from __future__ import annotations

import json

import pytest

from hepcoveragekg.facets.region_role import (
    CONTROL,
    FIDUCIAL,
    PRESELECTION,
    SIGNAL,
    VALIDATION,
    region_role,
    role_from_attributes,
    role_from_label,
)


@pytest.mark.parametrize("value,expected", [
    # The spellings actually present in the graph, all 33 of them collapsing
    # to five roles.
    ("SR", SIGNAL), ("signal_region", SIGNAL), ("signal region", SIGNAL),
    ("signal", SIGNAL), ("signal-region-component", SIGNAL),
    ("discovery signal region", SIGNAL), ("signal_enriched", SIGNAL),
    ("CR", CONTROL), ("control_region", CONTROL), ("control region", CONTROL),
    ("control", CONTROL), ("control-region", CONTROL), ("SB", CONTROL),
    ("VR", VALIDATION), ("validation region", VALIDATION),
    ("validation", VALIDATION), ("validation_region", VALIDATION),
    ("cross-check", VALIDATION),
    ("fiducial", FIDUCIAL), ("fiducial_region", FIDUCIAL),
    ("fiducial-definition", FIDUCIAL), ("fiducial_phase_space", FIDUCIAL),
    ("preselection", PRESELECTION), ("baseline", PRESELECTION),
])
def test_every_spelling_in_the_graph_normalises(value, expected):
    assert role_from_attributes({"role": value}) == expected


def test_a_qualified_signal_region_is_still_a_signal_region():
    """`SR (counting)` and `SR (unbinned fit)` say how the region is fitted, not
    that it is a different kind of region. An `^sr$` anchor dropped all three."""
    for value in ["SR (counting)", "SR (unbinned fit)"]:
        assert role_from_attributes({"role": value}) == SIGNAL


def test_the_three_keys_are_all_read():
    assert role_from_attributes({"region_role": "VR"}) == VALIDATION
    assert role_from_attributes({"is_signal_region": True}) == SIGNAL
    assert role_from_attributes(json.dumps({"role": "CR"})) == CONTROL


def test_is_signal_region_false_is_not_a_role():
    """False says what the region is not. Reading it as "control" would invent
    the one fact the supervisor is checking."""
    assert role_from_attributes({"is_signal_region": False}) is None
    assert role_from_attributes({"is_signal_region": "False"}) is None


@pytest.mark.parametrize("value", [
    "model-independent superbin", "aggregate of superbins", "excluded region",
    "extra_jet_definition", "full_phase_space",
])
def test_values_that_are_not_roles_stay_untagged(value):
    assert role_from_attributes({"role": value}) is None


def test_a_label_states_the_role_when_no_attribute_does():
    """193 regions carry no role attribute at all, and most name it in prose."""
    assert role_from_label("Orthogonal validation region, m_ll in [70,105] GeV") == VALIDATION
    assert role_from_label("Z(ee) control region") == CONTROL
    assert role_from_label("Final signal region (Region H)") == SIGNAL
    assert role_from_label("Particle-level fiducial phase space") == FIDUCIAL
    assert role_from_label("Z+jets baseline selection region") == PRESELECTION
    assert role_from_label("ttH leptonic category, p_T^H < 60 GeV") is None


def test_a_validation_region_that_names_its_signal_region_is_not_a_signal_region():
    """The reason the label rung tries qualifying roles first: a VR or CR label
    routinely names the SR it supports, and first-match-wins would file it as
    signal."""
    assert role_from_label("validation region for the 2-lepton signal region") == VALIDATION
    assert role_from_label("control region for SR-2j") == CONTROL


def test_an_asserted_role_beats_the_label():
    """Both rungs fire on 365 regions and disagree on 2. The attribute is what
    the extractor asserted about the region; the label is a guess from its name."""
    role, source = region_role("Baseline H->4l ZZ-candidate selection",
                               {"role": "signal_region"})
    assert (role, source) == (SIGNAL, "attribute")
    assert region_role("Z(ee) control region", {}) == (CONTROL, "label")
    assert region_role("ttH leptonic category", {}) == (None, None)


def test_abbreviations_do_not_fire_inside_words():
    assert role_from_label("secret cranial vr-free region") != FIDUCIAL
    assert role_from_label("microscrew assembly") is None
    assert role_from_label("SR-2j") == SIGNAL
    assert role_from_label("CR_ttbar") == CONTROL
