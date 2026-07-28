# Entity Identity Fix and Neo4j Graph Projection: Implementation Walkthrough

This document summarizes the successful implementation of the two-phase handoff plan: correcting the SQLite entity identity model (Phase 1) and projecting that corrected model into a Neo4j Property Graph (Phase 2).

## Phase 1: Entity Identity Fix

### Changes Made
- Modified the SQLite schema in `schema.sql` to redefine the foreign key constraints in the `assertion` table. `subject_id` and `object_id` now refer directly to `entity_occurrence` using the composite keys `(bundle_id, subject_id)` and `(bundle_id, object_id)`. This guarantees that assertions cannot improperly conflate different concepts that happen to share an ID across different bundles.
- Updated tests in `test_kg.py` to seed data conforming to the new constraints.
- Created a regression test in `test_importer.py` explicitly encoding the bug to ensure it remains fixed.

### Data Migration & Verification
- Moved the existing database to `data/processed/hepkg.db.bak`.
- Performed a full fresh import of the 60 reference bundles.
- Re-built and confirmed the aliases.
- Verified that all acceptance test requirements were met and baseline counts perfectly match: 60 bundles, 11,309 `machine_verified`, 2,555 `quarantined`, 324 `rejected` (Total: 14,188 assertions).

## Phase 2: Neo4j Graph Projection

### Changes Made
- Added `graph export` and `graph import` commands to the `hepcoveragekg` CLI via `cli.py`.
- Implemented CSV export logic in `export.py`. This reads directly from SQLite and writes highly optimized, bulk-import formatted CSVs for:
  - **Nodes**: `:Paper`, `:Occurrence`, `:Canonical`, and synthetic `:LiteralValue` nodes.
  - **Edges**: `HAS_OCCURRENCE`, `RESOLVES_TO`, and `ASSERTIONS`.
- Wrote `import_neo4j.sh`, an executable bash script that leverages the native `neo4j-admin database import full` tool to hydrate a pristine local database instantly from those CSVs.
- Logged the architectural graph structure and projection pipeline decision as `D-031` in `vault/decisions.md`.

### Verification
- Ran the `export` command locally, confirming it correctly generated 7 properly-formatted CSV files.
- Ran the `import` command via the CLI (`hepcoveragekg graph import`). It imported the full graph in 2.18 seconds.
- Confirmed the final dataset stats (from the 60 bundle sample):
  - 14,645 nodes
  - 26,282 relationships
  - 126,969 properties
