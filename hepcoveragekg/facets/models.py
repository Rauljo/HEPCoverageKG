# =============================================================================
# HEPCoverageKG facets: the one upstream model the vendored vocabulary needs.
#
# `vocabulary.py` is vendored VERBATIM (its hash is pinned by a test), and it
# does `from .models import DetectorObjectName`. Upstream's models.py is a large
# pydantic module we have no use for, so this file provides only that enum,
# copied unchanged from
#   hepkg_acquisition/models.py @ facet-canonical-layer-mr abd26bf
# so the vendored import resolves without pulling pydantic in or editing a file
# we do not own. Nothing else from upstream's models belongs here.
# =============================================================================
from __future__ import annotations

from enum import StrEnum


class DetectorObjectName(StrEnum):
    """Closed detector-object vocabulary (lab finding: closed enums measure
    better than free text). `LEPTON` is the flavour-summed e/mu umbrella.

    objects-v2 (2026-07-27, fleet-driven): members added so the canonical
    map covers what the 60-paper pilot actually produced as detector_object
    entities. MET and the vertices are canonical *identities* here; they
    stay non-countable (never valid as object_count expectations) — the lab
    decision survives as vocabulary.NON_COUNTABLE, not as absence from the
    enum."""

    ELECTRON = "Electron"
    MUON = "Muon"
    TAU = "Tau"
    PHOTON = "Photon"
    JET = "Jet"
    BJET = "BJet"
    CJET = "CJet"
    TRACK_JET = "TrackJet"
    LARGE_R_JET = "LargeRJet"
    TOP_CANDIDATE = "TopCandidate"
    W_CANDIDATE = "WCandidate"
    Z_CANDIDATE = "ZCandidate"
    HIGGS_CANDIDATE = "HiggsCandidate"
    DISPLACED_VERTEX = "DisplacedVertex"
    PRIMARY_VERTEX = "PrimaryVertex"
    TRACK = "Track"
    MET = "MET"
    LEPTON = "Lepton"

