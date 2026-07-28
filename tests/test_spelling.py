# =============================================================================
# HEPCoverageKG: aliases Tier 1.5 (spelling + plural) tests
#
# Proves the SAFE cases merge (US/UK, plural-when-singular-exists) and the
# dangerous look-alikes do NOT (w/z, s/t, version numbers, higgs/mass).
# =============================================================================
from __future__ import annotations

from hepcoveragekg.aliases import spelling


def _key(entity_id, vocab=frozenset()):
    return spelling.spelling_key(entity_id, set(vocab))


# --- US/UK spelling (safe, no corpus needed) ---------------------------------


def test_uk_us_spelling_unifies():
    assert _key("hepkg:syst:parton-shower-ue-modelling") == _key("hepkg:syst:parton_shower_ue_modeling")
    assert _key("hepkg:syst:colour_reconnection") == _key("hepkg:syst:color-reconnection")
    assert _key("hepkg:syst:hadronisation") == _key("hepkg:syst:hadronization")


def test_spelling_does_not_touch_lookalike_words():
    # 'precise' ends in 'ise' but is NOT in the map -> unchanged (no precize)
    assert _key("hepkg:x:precise_measurement") == "precisemeasurement"


# --- plural (data-driven: only when the singular exists in the corpus) --------


def test_plural_merges_only_when_singular_exists():
    vocab = {"warped", "extra", "dimension", "dimensions"}
    assert _key("hepkg:model:warped-extra-dimensions", vocab) == _key("hepkg:model:warped_extra_dimension", vocab)


def test_plural_left_alone_when_singular_absent():
    # 'higgs' must never become 'higg'; 'mass'/'analysis' protected by suffix rule
    assert _key("hepkg:process:higgs") == "higgs"
    assert _key("hepkg:x:invariant-mass") == "invariantmass"
    assert _key("hepkg:method:analysis") == "analysis"


# --- the dangerous look-alikes must STAY apart --------------------------------


def test_boson_flavour_not_merged():
    # w vs z polarization: different bosons, must not merge
    assert _key("hepkg:obs:w-longitudinal-polarization") != _key("hepkg:obs:z-longitudinal-polarization")


def test_channels_not_merged():
    assert _key("hepkg:sample:single_top_schannel") != _key("hepkg:sample:single_top_tchannel")


def test_versions_not_merged():
    assert _key("hepkg:generator:madgraph5-amcatnlo-2.6") != _key("hepkg:generator:madgraph5-amcatnlo-2.6.2")


# --- propose() only bridges NEW pairs (not Tier-1's job) ----------------------


def test_propose_bridges_spelling_not_pure_separators():
    ents = [
        ("hepkg:syst:modelling", "systematic_uncertainty"),
        ("hepkg:syst:modeling", "systematic_uncertainty"),
        ("hepkg:object:b_jet", "detector_object"),   # pure separator -> Tier 1's job, not here
        ("hepkg:object:b-jet", "detector_object"),
    ]
    props = spelling.propose(ents)
    pairs = {frozenset((p.entity_id_a, p.entity_id_b)) for p in props}
    assert frozenset({"hepkg:syst:modelling", "hepkg:syst:modeling"}) in pairs
    # the b_jet pair is NOT proposed by spelling (same Tier-1 slug already)
    assert frozenset({"hepkg:object:b_jet", "hepkg:object:b-jet"}) not in pairs
    assert all(p.method == "spelling" for p in props)
