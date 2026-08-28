"""What KIND of region a region is: signal, control, validation, fiducial, preselection.

Gabriel's review turned on this and the schema could not express it. His row 1:
the answer is *yes for a validation region and no for a signal region*, and "this
distinction is lost in the current schema" (D-066). A question like "which
analyses use a same-sign control region" is unanswerable when every region is
just an `event_region`.

The information was never missing -- it was written down inconsistently. Across
771 region entities the extraction put a role under three different keys with 33
different spellings: `role`, `region_role`, and a boolean `is_signal_region`,
holding `SR`, `signal_region`, `signal region`, `signal` and
`signal-region-component` for one concept. That is not a modelling gap, it is an
un-normalised field, and normalising it is cheaper and less lossy than
re-extracting.

TWO RUNGS, ATTRIBUTE FIRST.

An attribute is what the extractor asserted about the region and is authoritative.
A label is a fallback for the 193 regions carrying no role attribute at all,
where "Orthogonal validation region, m_ll in [70,105] GeV" plainly states the
role in prose. `source` travels with the value so a downstream claim can be
restricted to asserted roles if the label rung ever proves unreliable.

WHY VALIDATION AND CONTROL ARE TESTED BEFORE SIGNAL.

Only in the label rung, and only because of an asymmetry in how physicists name
things: a CR or VR label routinely names the SR it supports ("validation region
for the 2-lepton signal region"), while an SR label rarely mentions a CR. First
match wins, so the qualifying role has to be tried first or every VR whose name
mentions its SR would be filed as a signal region.

WHAT IS DELIBERATELY LEFT UNTAGGED.

`model-independent superbin` (7), `aggregate of superbins`, `excluded region`,
`extra_jet_definition`, `full_phase_space`, and `is_signal_region: False`. Each
is either not a role or -- in the False case -- says only what the region is not.
Guessing here would put invented structure into the one field the supervisor is
going to check.
"""
from __future__ import annotations

import json
import re
from typing import Optional

REGION_ROLE_VOCABULARY_VERSION = "region-roles-v1"
REGION_ROLE_FIELD = "region_roles"

#: The keys the extraction actually used, in the order they are trusted.
ROLE_KEYS = ("role", "region_role", "is_signal_region")

SIGNAL = "signal"
CONTROL = "control"
VALIDATION = "validation"
FIDUCIAL = "fiducial"
PRESELECTION = "preselection"

REGION_ROLES = (SIGNAL, CONTROL, VALIDATION, FIDUCIAL, PRESELECTION)

# Attribute values -> canonical role. Matched against the value lowercased with
# separators folded to a single space, so `signal_region`, `signal-region` and
# `signal region` collapse to one entry.
#
# Two judgement calls, both recorded rather than hidden:
#   `sideband`/`SB` -> control. A sideband exists to estimate a background from
#       data, which is what a control region is; the word describes its shape,
#       not a different function.
#   `baseline` -> preselection. Both name the common selection applied before
#       the analysis splits into regions.
# Anchored with \b rather than $ so a qualified value keeps its role:
# `SR (counting)` and `SR (unbinned fit)` are signal regions that happen to say
# how they are fitted.
_ATTRIBUTE_ROLES: list[tuple[str, str]] = [
    (r"^sr\b|^signal$|^signal region", SIGNAL),
    (r"^discovery signal region$|^signal enriched$", SIGNAL),
    (r"^cr\b|^control$|^control region$|^sb$|^sideband$", CONTROL),
    (r"^vr\b|^validation$|^validation region$|^cross check$", VALIDATION),
    (r"^fiducial|^fiducial region$|^fiducial phase space$|^fiducial definition$", FIDUCIAL),
    (r"^preselection$|^baseline$", PRESELECTION),
]

# Label fallback. Qualifying roles first -- see the module docstring. Anchored on
# word boundaries so `CR` does not fire inside a word, and the abbreviations
# require either a boundary or the usual `SR-2j` / `CR_ttbar` separators.
_LABEL_ROLES: list[tuple[str, str]] = [
    (r"\bvalidation regions?\b|\bvrs?\b|\bvr[-_]", VALIDATION),
    (r"\bcontrol regions?\b|\bcrs?\b|\bcr[-_]|\bcontrol/|\bcontrol sample\b", CONTROL),
    (r"\bsignal regions?\b|\bsrs?\b|\bsr[-_]|\bsignal enriched\b", SIGNAL),
    (r"\bfiducial\b", FIDUCIAL),
    (r"\bpreselect\w*\b|\bbaseline\b", PRESELECTION),
]

_SEPARATORS = re.compile(r"[_\-/]+")
_PUNCT = re.compile(r"[(),.;:]+")


def _fold(text: str) -> str:
    """Lowercase, separators to spaces, punctuation dropped, whitespace squeezed."""
    text = _SEPARATORS.sub(" ", str(text).lower())
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def role_from_attributes(attributes) -> Optional[str]:
    """The canonical role asserted by the extraction, or None.

    `is_signal_region` is read only when true: false says the region is not a
    signal region, which is not a role.
    """
    attrs = _as_dict(attributes)
    for key in ROLE_KEYS:
        if key not in attrs:
            continue
        raw = attrs[key]
        if key == "is_signal_region":
            if _is_true(raw):
                return SIGNAL
            continue
        folded = _fold(raw)
        for pattern, role in _ATTRIBUTE_ROLES:
            if re.search(pattern, folded):
                return role
    return None


def role_from_label(label: str) -> Optional[str]:
    """The role stated in the region's name, or None."""
    folded = _fold(label or "")
    for pattern, role in _LABEL_ROLES:
        if re.search(pattern, folded):
            return role
    return None


def region_role(label: str, attributes=None) -> tuple[Optional[str], Optional[str]]:
    """(canonical role, where it came from). Source is "attribute" or "label"."""
    role = role_from_attributes(attributes)
    if role:
        return role, "attribute"
    role = role_from_label(label)
    if role:
        return role, "label"
    return None, None


def _as_dict(attributes) -> dict:
    if isinstance(attributes, dict):
        return attributes
    if isinstance(attributes, (str, bytes)) and attributes:
        try:
            loaded = json.loads(attributes)
        except (ValueError, TypeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _is_true(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}
