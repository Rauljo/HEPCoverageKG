# =============================================================================
# HEPCoverageKG facets: signature synthesis (signatures-v1)
#
# A PORT of hepkg_acquisition/signatures.py, not a vendored copy: upstream takes
# pydantic Assertion models, we take SQLite rows. The rules are upstream's and
# must stay upstream's — tests/test_facets_parity.py runs both implementations
# over the pilot bundles and asserts identical output, which is the only reason
# this port can be trusted.
#
# What it repairs. A *signature* is the structured form of a selection cut:
# "this region requires >=3 leptons" as a tree rather than as prose. The
# extraction produced ZERO of them (0 of 14,188 assertions) — models never
# emitted the grammar, and malformed attempts were demoted by the coercion
# layer. The numbers survived only as free text in `qualifiers`:
#     {"count": ">=3", "pT": ">25 GeV"}
# This module parses them back. It is a repair, not a fix: the extraction is
# unchanged, and a native signature always wins when one finally appears.
#
# Its reach is bounded by what the extractor wrote down: 734 leaves out of 2,692
# candidate assertions, and the OR rule's `subchannel` flag appears on 35 of
# them. Recall is a property of the extraction, not of the rules.
#
# NEVER writes to `assertion`. Assertion ids are content hashes and re-import
# compares `identity_hash`; mutating them would break the corruption alarm. The
# derived tree lives beside the assertion, marked as derived.
# =============================================================================
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Iterable, NamedTuple, Optional

SIGNATURE_SYNTHESIS_VERSION = "signatures-v1"

# Predicates whose object is countable, so a `count` qualifier means something.
COUNT_PREDICATES = ("region_requires_object", "region_vetoes_object", "object_has_selection")
_VETO_PREDICATE = "region_vetoes_object"

_COMPARATOR_RE = re.compile(r"^\s*(>=|==|>|<=|<|=)\s*(\d+)\s*$")
_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

# The extractor writes the channel flag under any of these. Its presence is the
# ONLY signal that sibling cuts are alternatives rather than simultaneous.
_SUBCHANNEL_KEYS = ("subchannel", "flavour", "flavor")


class Cut(NamedTuple):
    """One assertion, reduced to what signature synthesis needs."""

    assertion_id: str
    paper_id: Optional[str]
    subject_id: str
    predicate: str
    object_id: Optional[str]
    qualifiers: dict
    has_native_signature: bool = False


def _parse_count(text: str) -> Optional[tuple[str, int, Optional[int]]]:
    """'>=3' -> ('>=', 3, None); '2-4' -> ('range', 2, 4); junk -> None."""
    match = _COMPARATOR_RE.match(text)
    if match:
        comparator = "==" if match.group(1) == "=" else match.group(1)
        return comparator, int(match.group(2)), None
    match = _RANGE_RE.match(text)
    if match:
        low, high = int(match.group(1)), int(match.group(2))
        if low <= high:
            return "range", low, high
    return None


def synthesize_signature(cut: Cut) -> Optional[dict]:
    """An object_count leaf derived from the count qualifier, or None.

    Only fires when there is no native signature, the predicate is
    object-flavoured with a real object entity, and the count parses. A veto
    predicate carrying no count defaults to ==0 — "vetoes b-jets" and "requires
    0 b-jets" are the same claim, and the fleet writes it both ways.
    """
    if cut.has_native_signature or cut.object_id is None:
        return None
    if cut.predicate not in COUNT_PREDICATES:
        return None

    raw = cut.qualifiers.get("count")
    parsed = _parse_count(raw) if isinstance(raw, str) else None
    if parsed is None:
        if cut.predicate == _VETO_PREDICATE and raw is None:
            parsed = ("==", 0, None)
        else:
            return None

    comparator, value, range_max = parsed
    return {
        "operator": "object_count",
        "object_ref": cut.object_id,
        "comparator": comparator,
        "value": value,
        "range_max": range_max,
    }


def _subchannel(cut: Cut) -> Optional[str]:
    for key in _SUBCHANNEL_KEYS:
        value = cut.qualifiers.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def synthesize_or_groups(cuts: Iterable[Cut]) -> dict[str, dict]:
    """assertion_id -> the shared any_of tree, for the explicit flavour-OR shape.

    Read literally, "region R requires 2 electrons" plus "region R requires 2
    muons" means four leptons. The paper means two OR two — the analysis runs in
    parallel channels. The tell is the `subchannel` qualifier: the extractor
    saying these are separate channels.

    Group condition, all deterministic: same subject, same predicate, IDENTICAL
    count qualifier, every member carrying a DISTINCT subchannel, at least two
    members. Every member maps to the same any_of of the members' leaves.

    Deliberately narrow. Turning a real AND into an OR makes a region look looser
    than it is and nothing downstream can detect it, so looser inference is a
    non-goal (upstream's words, kept).
    """
    groups: dict[tuple, list[tuple[Cut, dict]]] = defaultdict(list)
    for cut in cuts:
        leaf = synthesize_signature(cut)
        if leaf is None or _subchannel(cut) is None:
            continue
        raw = cut.qualifiers.get("count")
        if not isinstance(raw, str):
            continue  # the veto default has no explicit count to match members on
        groups[(cut.subject_id, cut.predicate, raw.strip())].append((cut, leaf))

    result: dict[str, dict] = {}
    for members in groups.values():
        channels = {_subchannel(cut) for cut, _ in members}
        if len(members) < 2 or len(channels) != len(members):
            continue
        members.sort(key=lambda pair: pair[0].assertion_id)
        tree = {"operator": "any_of", "children": [leaf for _, leaf in members]}
        for cut, _ in members:
            result[cut.assertion_id] = tree
    return result


# --------------------------------------------------------------------------
# Reading our rows
# --------------------------------------------------------------------------

def cuts_from_db(conn) -> list[Cut]:
    """Every assertion signature synthesis could possibly fire on.

    Scoped to COUNT_PREDICATES in SQL rather than filtered in Python, because
    the rest of the assertion table is an order of magnitude larger and none of
    it can ever produce a leaf.
    """
    marks = ",".join("?" * len(COUNT_PREDICATES))
    rows = conn.execute(
        "SELECT assertion_id, paper_id, subject_id, predicate, object_id,"
        "       qualifiers, signature"
        f"  FROM assertion WHERE predicate IN ({marks})",
        COUNT_PREDICATES,
    ).fetchall()

    cuts: list[Cut] = []
    for row in rows:
        try:
            qualifiers = json.loads(row["qualifiers"]) if row["qualifiers"] else {}
        except (TypeError, ValueError):
            qualifiers = {}
        if not isinstance(qualifiers, dict):
            qualifiers = {}
        cuts.append(Cut(
            assertion_id=row["assertion_id"],
            paper_id=row["paper_id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            qualifiers=qualifiers,
            has_native_signature=row["signature"] is not None,
        ))
    return cuts


def derive(cuts: Iterable[Cut]) -> list[tuple[str, dict, bool]]:
    """(assertion_id, signature tree, is_or_group) for everything derivable.

    OR-group members carry the shared any_of rather than their own leaf — that
    shared identity is how a reader tells alternatives from separate cuts.
    """
    cuts = list(cuts)
    or_groups = synthesize_or_groups(cuts)

    derived: list[tuple[str, dict, bool]] = []
    for cut in cuts:
        tree = or_groups.get(cut.assertion_id)
        if tree is not None:
            derived.append((cut.assertion_id, tree, True))
            continue
        leaf = synthesize_signature(cut)
        if leaf is not None:
            derived.append((cut.assertion_id, leaf, False))
    return derived
