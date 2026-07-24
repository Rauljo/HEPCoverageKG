# =============================================================================
# HEPCoverageKG: bundle-import errors
#
# One hierarchy, defined once, so callers can catch BundleError for "this bundle
# did not make it in" or a specific subclass for a specific gate. Each subclass
# maps to one pipeline stage, so a rejection always says which gate refused it.
# See vault/ideas/bundle-importer-design.md.
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import Optional


class BundleError(Exception):
    """Base: a bundle could not be imported. Carries the offending file path."""

    def __init__(self, message: str, *, path: Optional[Path] = None) -> None:
        self.path = path
        self.reason = message
        super().__init__(f"{path}: {message}" if path else message)


class BundleReadError(BundleError):
    """Stage 0: the file could not be read, decompressed, or parsed as JSON."""


class BundleShapeError(BundleError):
    """Stage 1: does not match the supervisor's JSON Schema, or declares a
    schema_version we do not support."""


class BundleSemanticError(BundleError):
    """Stage 2: shape is valid but a business rule is broken -- e.g. an accepted
    assertion with no evidence, which the schema itself permits. (Step 4.)"""


class BundleConflictError(BundleError):
    """Stage 4: an existing assertion_id came back with a non-status change,
    which the acquisition pipeline cannot legitimately produce. Aborts the
    import transactionally. (Step 6.)"""
