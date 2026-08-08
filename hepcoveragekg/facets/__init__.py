# =============================================================================
# HEPCoverageKG facets: the closed-vocabulary rung of the abstraction ladder
#
# Two derived layers, both ported from the acquisition repo (read-only
# reference), both deterministic and model-free:
#
#   vocabulary.py  -- VENDORED VERBATIM. Regex tables that map a free-text
#       entity label onto a closed enum: multi-label `facet_tags` for the
#       descriptive kinds, 1:1 `canonicalize_entity` for detector objects and
#       generators. Pure stdlib, so it copies across without adaptation.
#   signatures.py  -- PORTED, not vendored. Upstream operates on pydantic
#       models; ours operates on our SQLite rows. Same two rules, and a parity
#       test asserts the two implementations agree on the pilot bundles.
#
# Why derive rather than import the frozen `pilot/analysis_facets.jsonl`:
# recomputing it from our own graph reproduces all 60 of its cards exactly, so
# the file is a snapshot of a function we can run — and a snapshot goes stale
# the moment paper 61 arrives. The file is kept as a TEST FIXTURE (the oracle),
# never as production data. See S-68..S-70 and D-052.
#
# On drift: `vocabulary.py` is upstream's file and must never be edited here.
# test_facets_parity pins its hash, so an accidental local edit fails loudly;
# refreshing means re-copying and updating the constant below.
# =============================================================================
from __future__ import annotations

# Provenance of the vendored file. `hepkg_promopt_tests` @ these commits.
VOCABULARY_SOURCE = {
    "repo": "gfacini/HEPKG_promopt_tests",
    "branch": "facet-canonical-layer-mr",
    "commit": "abd26bf735a6973bfe61ba6977fdb1efef7dcf31",
    "path": "hepkg_acquisition/vocabulary.py",
    "sha256": "18f681062980253b8527e2fe6ea1b16e67d9deea22501a41a6d43b6fa66d7381",
}

SIGNATURES_SOURCE = {
    "repo": "gfacini/HEPKG_promopt_tests",
    "branch": "signature-synthesis",
    "commit": "5dfcde74cbf8419dee53b80f6c5fccea796a96bd",
    "path": "hepkg_acquisition/signatures.py",
    "sha256": "db047f8f28a17ebfe30975e3e8273091d83c9a37eff092e124d8375d47caf9dc",
    "note": "ported, not vendored — hash pins what the port was written against",
}

# Entity kind -> analysis-card field. Objects and generators contribute their
# canonical name; the descriptive kinds contribute facet tags. Mirrors
# upstream's compiler._CARD_FIELDS, which is what makes card parity possible.
CARD_FIELDS: dict[str, str] = {
    "detector_object": "objects",
    "generator": "generators",
    "background": "process_families",
    "physics_process": "process_families",
    "bsm_model": "process_families",
    "background_method": "background_methods",
    "statistical_method": "statistical_methods",
    "systematic_uncertainty": "systematic_sources",
    "sample": "process_families",
    "observable": "observable_types",
}

# The kinds whose value is a rename rather than a tag: exactly zero or one.
RENAMED_KINDS = ("detector_object", "generator")
