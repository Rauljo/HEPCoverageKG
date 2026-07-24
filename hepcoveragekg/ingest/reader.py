# =============================================================================
# HEPCoverageKG: bundle reader + shape gate (pipeline stages 0-1)
#
# Reads a bundle off disk (.json.gz or .json), parses it, and validates its
# SHAPE against the supervisor's JSON Schema. Nothing here touches the database.
#
# The schema is VENDORED (schemas/) rather than read from the acquisition repo,
# so the importer and its tests are self-contained and reproducible even if that
# repo moves or this one is merged into it. Two guards keep the copy honest:
# VENDORED_SCHEMA_SHA256 below, and a drift test against upstream in
# tests/test_reader.py (skipped when the acquisition repo is absent).
#
# This gate is SHAPE ONLY. Business rules are the semantic layer's job (step 4):
# the invalid_accepted_without_evidence fixture passes here by design, because
# the schema itself permits an accepted assertion with no evidence.
# See vault/ideas/bundle-importer-design.md (D-023).
# =============================================================================
from __future__ import annotations

import gzip
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from jsonschema import Draft202012Validator

from hepcoveragekg.ingest.errors import BundleReadError, BundleShapeError

SCHEMA_PATH = Path(__file__).parent / "schemas" / "hepkg-acquisition-v0.2.schema.json"

# sha256 of the vendored schema exactly as copied from the acquisition repo.
# Pinned so an accidental local edit fails loudly instead of silently changing
# what we accept.
VENDORED_SCHEMA_SHA256 = "90ec3ad16b99e1452fa3734ca40c453bca827c0b00143fb6215cebfe04c68062"

SUPPORTED_SCHEMA_VERSIONS = frozenset({"hepkg-acquisition-v0.2"})

# A malformed bundle can produce hundreds of violations; report the first few
# with their JSON paths, which is enough to diagnose without flooding the log.
MAX_REPORTED_ERRORS = 5


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    """Compile the schema once and reuse it (~59 ms per bundle, 3.5 s for 60).

    The schema declares no $schema, so the draft is chosen explicitly; it uses
    $defs and validates cleanly as Draft 2020-12.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def read_bundle(path: Union[str, Path]) -> dict[str, Any]:
    """Stage 0: decompress (.json.gz) or open (.json), parse. No validation."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            bundle = json.load(handle)
    except OSError as exc:  # missing file, permissions, corrupt gzip
        raise BundleReadError(f"cannot read file ({exc})", path=path) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BundleReadError(f"not valid JSON/UTF-8 ({exc})", path=path) from exc

    if not isinstance(bundle, dict):
        raise BundleReadError(
            f"expected a JSON object, got {type(bundle).__name__}", path=path
        )
    return bundle


def validate_shape(bundle: Any, *, path: Optional[Path] = None) -> None:
    """Stage 1: check the bundle against the schema and the version we support.

    The explicit schema_version check is NOT redundant with the schema: the
    schema declares schema_version as `const`, but does not list it as required
    (root requires only bundle_id, paper, source), so `const` never fires for a
    bundle that omits the field entirely. Checking it here also turns an
    unsupported version into a clear message rather than a confusing cascade of
    "additional properties are not allowed" errors.
    """
    if not isinstance(bundle, dict):
        raise BundleShapeError(
            f"expected a JSON object, got {type(bundle).__name__}", path=path
        )

    version = bundle.get("schema_version")
    if version is None:
        raise BundleShapeError("missing schema_version", path=path)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise BundleShapeError(
            f"unsupported schema_version {version!r} (supported: {supported})", path=path
        )

    errors = sorted(_validator().iter_errors(bundle), key=lambda e: list(e.path))
    if errors:
        shown = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:MAX_REPORTED_ERRORS]
        )
        omitted = len(errors) - MAX_REPORTED_ERRORS
        suffix = f" (+{omitted} more)" if omitted > 0 else ""
        raise BundleShapeError(
            f"{len(errors)} schema violation(s) -- {shown}{suffix}", path=path
        )


def load_bundle(path: Union[str, Path]) -> dict[str, Any]:
    """Stages 0-1 together: read, parse, shape-validate. The importer entry point."""
    path = Path(path)
    bundle = read_bundle(path)
    validate_shape(bundle, path=path)
    return bundle
