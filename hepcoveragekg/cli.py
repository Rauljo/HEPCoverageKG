# =============================================================================
# HEPCoverageKG: command-line importer
#
#   python -m hepcoveragekg.cli import <path>      # a bundles dir or one file
#   python -m hepcoveragekg.cli verify-counts      # check the milestone-1 target
#
# One bad bundle is reported and the run continues; the command exits non-zero
# if anything failed. verify-counts checks the pilot target and sets its exit
# code accordingly. See vault/ideas/bundle-importer-design.md (build step 8).
# =============================================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from hepcoveragekg.ingest import importer
from hepcoveragekg.ingest.errors import BundleError
from hepcoveragekg.kg import queries, store

# The milestone-1 acceptance target: the 60 pilot bundles by status.
PILOT_TARGET = {"machine_verified": 11309, "quarantined": 2555, "rejected": 324}


def _find_bundles(path: Path) -> list[Path]:
    """Bundles ship as .json.gz, so a directory is scanned for those only --
    directories also hold metadata (manifest.json, README.md) that must not be
    imported. A single-file path is taken as-is (.json or .json.gz), so an
    individual fixture can still be imported directly.
    """
    if path.is_dir():
        return sorted(path.glob("*.json.gz"))
    return [path]


def _cmd_import(args) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)

    bundles = _find_bundles(Path(args.path))
    if not bundles:
        print(f"no bundles found at {args.path}", file=sys.stderr)
        return 1

    print(f"importing {len(bundles)} bundle(s) into {args.db}")
    imported = skipped = failed = 0
    for path in bundles:
        try:
            result = importer.import_bundle_file(conn, path)
        except BundleError as exc:
            failed += 1
            print(f"  ✗ {path.name} — {exc.reason}")
            continue

        if result.skipped:
            skipped += 1
            print(f"  ~ {result.paper_id} — already imported (skipped)")
        else:
            imported += 1
            print(
                f"  ✓ {result.paper_id} — "
                f"{result.inserted.get('assertion', 0)} assertions, "
                f"{result.inserted.get('entity', 0)} entities"
            )
        for warning in result.warnings:
            print(f"      ⚠ {warning}")

    print(f"\ndone: {imported} imported, {skipped} skipped, {failed} failed")
    _print_status(conn)
    return 1 if failed else 0


def _cmd_verify_counts(args) -> int:
    conn = store.connect(args.db)
    store.init_schema(conn)
    counts = _print_status(conn)

    missing = {k: (counts.get(k, 0), v) for k, v in PILOT_TARGET.items() if counts.get(k, 0) != v}
    total_ok = sum(counts.values()) == sum(PILOT_TARGET.values())
    if not missing and total_ok:
        print("✓ milestone-1 target met (14,188 / 11,309 / 2,555 / 324)")
        return 0
    print("✗ milestone-1 target NOT met:")
    for status, (got, want) in missing.items():
        print(f"    {status}: got {got}, want {want}")
    if not total_ok:
        print(f"    total: got {sum(counts.values())}, want {sum(PILOT_TARGET.values())}")
    return 1


def _print_status(conn) -> dict[str, int]:
    counts = queries.status_counts(conn)
    total = sum(counts.values())
    parts = ", ".join(f"{status} {n}" for status, n in sorted(counts.items()))
    print(f"status: {parts or '(empty)'}  (total {total})")
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hepcoveragekg", description="HEP coverage KG importer")
    parser.add_argument(
        "--db", default=str(store.DEFAULT_DB_PATH), help="SQLite database path"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="import a bundle directory or a single file")
    p_import.add_argument("path", help="a directory of bundles, or one .json/.json.gz file")
    p_import.set_defaults(func=_cmd_import)

    p_verify = sub.add_parser("verify-counts", help="check the milestone-1 count target")
    p_verify.set_defaults(func=_cmd_verify_counts)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
