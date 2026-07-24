# =============================================================================
# HEPCoverageKG: canonical serialization and content fingerprints
#
# Exact port of the supervisor's hepkg_acquisition/ids.py. This has to be
# byte-identical to his: bundle_id and assertion_id ARE hashes computed this
# way, so a single differing space would make them irreproducible and we could
# not verify anything against them.
#
# Pure functions, no decisions -- this module only computes fingerprints. The
# skip / update / abort logic that consumes them lives in the importer (step 6).
# See vault/ideas/bundle-importer-design.md.
# =============================================================================
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

# The only assertion field that may legitimately differ under an existing
# assertion_id. The claim-defining fields (subject / predicate / object / value
# / signature / qualifiers / evidence) are baked into the assertion_id hash
# itself, so they cannot change without producing a different id; and the
# review pipeline only ever sets status on an existing assertion (corrections
# are appended as brand-new assertions). Anything else differing under a known
# id is therefore something the pipeline cannot produce -> conflict, abort.
MUTABLE_ASSERTION_FIELDS = frozenset({"status"})


def stable_json(value: Any) -> str:
    """Canonical JSON: sorted keys, no whitespace, unicode preserved."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: Any) -> str:
    """Full content fingerprint: 'sha256:<64 hex>' of the canonical JSON."""
    return "sha256:" + hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def content_id(namespace: str, value: Any) -> str:
    """Recompute an id in the supervisor's format: 'hepkg:<ns>:<24 hex>'.

    We never mint ids -- bundles carry their own and we trust them. This exists
    only to VERIFY a declared id against the content it claims to describe.
    """
    digest = hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()
    return f"hepkg:{namespace}:{digest[:24]}"


def bundle_fingerprint(bundle: Mapping[str, Any]) -> str:
    """Idempotency key: does this bundle differ *at all* from one already imported?

    Hashes the whole bundle, telemetry included. It answers only "is this
    byte-identical?" -- meaning-level comparison happens per assertion via
    assertion_identity_hash(). Stored as bundle_import.bundle_content_hash.
    """
    return content_hash(bundle)


def expected_bundle_id(bundle: Mapping[str, Any]) -> str:
    """The bundle_id this bundle's identity fields imply.

    The supervisor derives bundle_id from schema + paper + source + normalization
    ONLY (hepkg_acquisition/pipeline.py). It therefore stays the same across
    re-extractions of the same paper: assertions and statuses can change without
    changing it. That is why conflict detection cannot live at the bundle level.
    """
    source = bundle.get("source") or {}
    paper = bundle.get("paper") or {}
    return content_id(
        "bundle",
        {
            "schema": bundle.get("schema_version"),
            "paper": paper.get("arxiv_id"),
            "source": source.get("source_hash"),
            "normalization": source.get("normalization_version"),
        },
    )


def bundle_id_matches_content(bundle: Mapping[str, Any]) -> bool:
    """Authenticity check: is the declared bundle_id the one its content implies?

    Warning-level only (see the validator). A mismatch means a mislabelled,
    corrupted or tampered bundle -- but it could also mean the supervisor
    changed his id algorithm, which must not hard-fail our importer.
    """
    return bundle.get("bundle_id") == expected_bundle_id(bundle)


def assertion_identity_hash(assertion: Mapping[str, Any]) -> str:
    """Fingerprint of an assertion EXCLUDING its mutable lifecycle fields.

    Same hash  -> only the status moved (or nothing did)  -> allowed update.
    Differs    -> a non-status field changed under an existing assertion_id,
                  which the pipeline cannot legitimately produce -> conflict.
    """
    identity = {k: v for k, v in assertion.items() if k not in MUTABLE_ASSERTION_FIELDS}
    return content_hash(identity)
