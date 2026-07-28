# Handoff: fix entity identity, then build the Neo4j projection

**Written for**: an implementing agent (Gemini, in Antigravity) with full repo access but no
memory of the conversation that produced this brief. Everything needed to act is below or
directly discoverable in the repo — do not guess at anything not stated here; grep/read first.

**Repo**: `HEPCoverageKG` (this repo). Reference-only sibling repo (read-only, port patterns,
never edit): `../HEPKG_promopt_tests` (the supervisor's acquisition pipeline — referred to below
as "the contract").

**Scope of this brief**: two sequential pieces of work.
1. **Phase 1 (required first)**: fix a real correctness bug in how entities are keyed, confirmed
   by the supervisor's own data contract.
2. **Phase 2 (depends on Phase 1)**: build a Neo4j graph projection from the corrected model.

Everything else discussed elsewhere in the vault (consuming the supervisor's `canonical`/`facets`
vocabulary, Tier 2/3 fuzzy aliases, final-state signature compilation) is **out of scope** — do
not touch it. Keep this change small and reversible.

---

## Why (grounded — read this before changing anything)

### The bug

`hepcoveragekg/kg/schema.sql` currently has:

```sql
CREATE TABLE IF NOT EXISTS entity (
    entity_id     TEXT PRIMARY KEY,   -- no bundle_id: this is the merged, bundle-agnostic node
    ...
);

CREATE TABLE IF NOT EXISTS entity_occurrence (
    bundle_id  TEXT NOT NULL,
    entity_id  TEXT NOT NULL,
    ...
    PRIMARY KEY (bundle_id, entity_id)
);

CREATE TABLE IF NOT EXISTS assertion (
    assertion_id  TEXT PRIMARY KEY,
    bundle_id     TEXT NOT NULL REFERENCES bundle_import(bundle_id),
    subject_id    TEXT NOT NULL REFERENCES entity(entity_id),   -- <-- points at the MERGED table
    object_id     TEXT REFERENCES entity(entity_id),            -- <-- same
    ...
);
```

Every edge (`assertion`, 14,188 rows in the current pilot DB) points at the bare-`entity_id`
merged table, **not** at `entity_occurrence` (which is per-paper and correct).

The merge itself happens in `hepcoveragekg/ingest/importer.py`, function `_upsert_entities`
(around line 178): for every entity in a bundle it does
`INSERT OR IGNORE INTO entity (entity_id, ...)` — i.e. if an `entity_id` string has been seen
before (in *any* prior paper), the new paper's occurrence is folded into the *same* node via
`merge_entity()` (mode/consensus logic, lines 90–98).

### Why that's wrong (confirmed by the supervisor's contract, not just our opinion)

The supervisor's data contract (`../HEPKG_promopt_tests/docs/data-contract.md`) states explicitly:

> "Entity labels and aliases never establish automatic cross-paper identity."

and his PR #1 (`gfacini/HEPKG_promopt_tests#1`) reports measured numbers for how often this goes
wrong: **347 entity_ids shared across papers, 752 of those shared ids have conflicting
kind/label under the same id** — i.e. two different papers reused the same id string for two
different things. Our importer currently treats id-equality as identity and silently merges
those 752 conflicting cases into one node. That is an *accidental collision* being encoded as a
real fact.

`entity_occurrence` already has the correct grain (`bundle_id, entity_id`) — it is a faithful,
per-paper record. It was built correctly from the start; only `assertion`'s foreign keys (and by
extension anything downstream, like a future Neo4j export) point at the wrong table.

### What NOT to conclude from this

This does **not** mean cross-paper identity is impossible or that the `entity` table should be
deleted. It means: cross-paper "this is the same real thing" must be an **explicit, separate
resolution step** — which this codebase already has, as the `aliases/` package
(`hepcoveragekg/aliases/`: `same_as` table + `entity_canonical` table, non-destructive,
human-confirmable). That layer is architecturally correct and should be reused, not rebuilt — see
Phase 2 below.

---

## Phase 1 — fix entity identity

### 1. Schema change (`hepcoveragekg/kg/schema.sql`)

Change `assertion`'s foreign keys from single-column (against `entity.entity_id`) to composite
(against `entity_occurrence(bundle_id, entity_id)`):

```sql
CREATE TABLE IF NOT EXISTS assertion (
    assertion_id  TEXT PRIMARY KEY,
    bundle_id     TEXT NOT NULL REFERENCES bundle_import(bundle_id),
    paper_id      TEXT,
    subject_id    TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    family        TEXT NOT NULL,
    object_id     TEXT,
    ...
    FOREIGN KEY (bundle_id, subject_id) REFERENCES entity_occurrence(bundle_id, entity_id),
    FOREIGN KEY (bundle_id, object_id)  REFERENCES entity_occurrence(bundle_id, entity_id),
    ...
);
```

Note `assertion` already carries its own `bundle_id` column (every assertion belongs to exactly
one bundle; subject/object entities are always local to that same bundle in the source data) —
so this is a **narrowing of an existing column pairing**, not a new column. Confirm this
assumption holds by checking that no assertion's `subject_id`/`object_id` ever needs to resolve
to a *different* bundle's occurrence before implementing (grep the bundle JSON schema /
`hepkg_acquisition/models.py` `Assertion` class in the reference repo if unsure — but the
contract's per-bundle design strongly implies this is always true).

SQLite does not support `ALTER TABLE` to change a foreign key, so this requires a full schema
rebuild (see "Data migration" below), not an in-place `ALTER`.

Also fix the now-misleading comment on `entity`:

```sql
CREATE TABLE IF NOT EXISTS entity (
    entity_id     TEXT PRIMARY KEY,   -- convenience rollup ONLY, not identity — see entity_occurrence
    ...
);
```

Keep the `entity` table itself (do not delete it). It remains useful as: (a) a fast per-`entity_id`
lookup/rollup for tooling, (b) the source catalogue the `aliases/` layer already reads from
(`hepcoveragekg/aliases/store.py::entities_with_kind()` — reads `SELECT entity_id, kind FROM
entity`, this does **not** need to change, since a catalogue of "every entity_id ever seen, with
its most-common kind" is a legitimate and different use from "this IS the node identity"). Only
its role as an `assertion` foreign-key target is being removed.

### 2. Importer code — no logic change needed, only what it feeds

`_upsert_entities` and `merge_entity` in `hepcoveragekg/ingest/importer.py` can stay exactly as
they are — they still correctly populate both `entity` (rollup) and `entity_occurrence`
(per-paper truth). `_insert_assertions` (around line 258) already writes `bundle_id` on every
assertion row alongside `subject_id`/`object_id`; no column changes needed there either. This
confirms the fix is schema-metadata-only, not a rewrite of the write path.

### 3. Data migration

The existing `data/processed/hepkg.db` was built under the old (incorrect) FK. Since this is a
schema change SQLite can't do in-place, and the import pipeline is documented as idempotent and
order-independent (see `merge_entity` docstring), the clean path is:

1. Back up or move aside `data/processed/hepkg.db` (do not delete without confirming with the
   user first — check `git status`/ask if unsure whether it's reproducible-only or has any
   manually-curated state riding along, e.g. `same_as` rows with `status='confirmed'` set by a
   human that didn't come from a deterministic tier).
2. Re-run the import CLI (`hepcoveragekg/cli.py`, `_cmd_import`, wired as `hepcoveragekg import
   <bundle-dir-or-file>`) over the full pilot bundle set (60 bundles; locate via the acquisition
   pipeline's `pilot/bundles/` directory in the reference repo, or wherever this repo's own
   `data/raw/` currently sources bundles from — check `tests/` fixtures / existing import scripts
   for the exact invocation already in use).
3. Re-run `hepcoveragekg aliases build` then `hepcoveragekg aliases confirm` to repopulate
   `same_as`/`entity_canonical` (this package's own tables are untouched by the schema change but
   depend on the `entity` table being repopulated first).
4. Verify against known-good counts before this change: `entity` 5,114 rows, `entity_occurrence`
   6,047 rows, `assertion` 14,188 rows, `entity_canonical` 575 rows → 247 distinct canonicals
   (247 may shift slightly once the 19 pending `status='proposed'` spelling-tier rows are
   reviewed and confirmed — that's an independent, already-known pending task, not part of this
   fix). Row counts for `entity`/`entity_occurrence`/`assertion` should be **unchanged** by this
   fix (it changes what a foreign key points at, not what data exists).

### 4. Tests

- Existing tests in `tests/test_importer.py` and `tests/test_aliases.py` should still pass
  unmodified in spirit; update any that assert against the old FK shape or that construct
  in-memory DBs by hand and rely on the old single-column FK.
- Add a regression test that specifically encodes the bug this fixes: two bundles that both use
  the same `entity_id` for **different** real things (differing `kind` or clearly differing
  `label`) should NOT cause their assertions to be silently treated as sharing one identity for
  traversal purposes — i.e. an assertion's subject must resolve via `(bundle_id, subject_id)`,
  and two same-`entity_id`-different-bundle occurrences must remain distinguishable rows in
  `entity_occurrence`, each correctly linked to their own bundle's assertions.

---

## Phase 2 — Neo4j projection (built on the corrected model)

### Design (already agreed with the project owner — implement as specified, this is not open)

The graph must give **one real canonical node per real-world concept** (e.g. one `BJet` node),
with each paper's version of that concept connected to it by an **explicit edge**, not by
sharing a raw id. Concretely, three node types:

1. **`:Paper {arxiv_id, title, ...}`** — one per row in `paper`.
2. **`:Occurrence {bundle_id, entity_id, kind, label}`** — one per row in `entity_occurrence`
   (the faithful, per-paper node — this is what assertions attach to, post-Phase-1).
3. **`:Canonical {id, kind, label}`** — one per distinct resolved identity. Derive as: every
   `entity_canonical.canonical_id` (deduplicated), **plus** every `entity_id` that appears in
   `entity_occurrence` but has **no** row in `entity_canonical` (singleton — resolves to itself).
   This guarantees every occurrence has exactly one canonical to connect to.

Edges:

- `(:Paper)-[:HAS_OCCURRENCE]->(:Occurrence)` — from `entity_occurrence.paper_id`.
- `(:Occurrence)-[:RESOLVES_TO]->(:Canonical)` — from `entity_canonical` (or self-resolve for
  singletons). **This is the edge that answers "is this really the same BJet?" — deliberately,
  not by accident.**
- One relationship per `assertion` row, `(:Occurrence)-[:<PREDICATE>]->(:Occurrence)` for
  `object_id` assertions (subject/object both resolved via the Phase-1-corrected
  `(bundle_id, subject_id)` / `(bundle_id, object_id)` lookup into `entity_occurrence`), with
  assertion's `family`, `qualifiers` (JSON, store as a stringified property or flatten known
  keys), `status`, `support`, `assertion_id` as relationship properties. For assertions with
  `object_value` (literal) or `signature` (AST) instead of `object_id`, store as properties on
  the relationship's source node or a synthetic value node — pick one approach and apply it
  consistently; note the choice in the resulting code's docstring since it's a real design call
  a reviewer will ask about.

This gives exactly the property the project owner asked for: queries like "which papers use
BJet?" walk `Canonical <-[:RESOLVES_TO]- Occurrence <-[:HAS_OCCURRENCE]- Paper` — one real node,
reached through real edges, with per-paper detail (tagger, working point, etc., stored as
`Occurrence`/relationship properties) preserved rather than lost.

### Mechanics

- New package `hepcoveragekg/graph/` (does not exist yet — confirmed via repo search).
- Target: **Neo4j Community, local** (already decided in `vault/ideas/` — a projection for
  ergonomics/visualization, not for scale; do not build for a multi-node/cloud deployment).
- Recommended export mechanism: **CSV + Cypher `LOAD CSV`**, not a live driver dependency — it's
  reproducible, diffable, and needs no running Neo4j instance to generate. Write one CSV per node
  type and one per relationship type (standard Neo4j bulk-import CSV shape: `:ID`, `:LABEL`,
  property columns; relationship CSVs need `:START_ID`, `:END_ID`, `:TYPE`). Provide a
  `.cypher` script (or `neo4j-admin import` invocation, since this is a fresh local DB each time)
  that loads them. A thin Python driver-based writer is an acceptable alternative if the CSV
  approach proves awkward for the qualifiers/signature JSON — pick one, don't build both.
- Wire a CLI command, `hepcoveragekg graph export <out-dir>`, following the existing pattern in
  `hepcoveragekg/cli.py` (see `_cmd_import`, `_cmd_aliases` for the established style: a
  `_cmd_graph(args)` function + `sub.add_parser("graph", ...)`).

### Acceptance criteria

- Running the export against the Phase-1-corrected `data/processed/hepkg.db` produces node/edge
  files with no dangling references (every `:START_ID`/`:END_ID` has a matching node row).
- Count sanity: `:Occurrence` nodes = 6,047 (matches `entity_occurrence` row count);
  `:Canonical` nodes ≈ 247 + (count of `entity_id`s with zero `entity_canonical` row); every
  `:Occurrence` has exactly one outgoing `:RESOLVES_TO`.
- Spot-check the motivating example: pick an entity_id that participates in a known alias cluster
  (e.g. a `pythia8` generator-version cluster from `data/processed/draft-aliases.md`) and confirm
  in the loaded Neo4j instance that all its paper-occurrences point to one shared `:Canonical`
  node, reachable from each paper via `HAS_OCCURRENCE → RESOLVES_TO`.
- Loads cleanly into a local Neo4j Community instance with no import errors.

### Explicitly out of scope for this pass (do not build)

- Consuming the supervisor's `canonicalize_entity`/`facet_tags` vocabulary into our importer
  (separate, already-scoped backlog item — vault/backlog.md, "Graph & canonicalization").
- Building aliases Tiers 2/2.5/3 (fuzzy/embedding-based matching) — deferred, deemed unsafe
  without much more guarding (measured: 0.97-similarity fuzzy matches conflate real synonyms with
  dangerous look-alikes like w/z polarization or differing generator versions).
- Compiling `signature` from `qualifiers` (`count`/`subchannel`/`column`) for final-state
  representation — separate, blocks a different milestone (M3), not needed for this graph
  projection to work correctly at the entity/assertion level.
- Any change to the `aliases/` package's own tables/logic — it is already correctly designed
  (confirmed by the supervisor's contract independently arriving at the same "explicit resolution
  layer" architecture) and only needs to be *fed by* the corrected `entity_occurrence`-based model,
  not modified.

---

## Reporting back

When this is done, the project owner will want, in plain terms: (1) confirmation the row counts
above still hold post-migration, (2) one concrete before/after example showing an entity that
used to collide silently now resolving correctly (or, if none exists in the 60-paper pilot
corpus, confirmation of that and the closest near-miss), (3) how to load the exported CSVs into a
local Neo4j Community instance and run one example Cypher query. This is a real design decision
(node/edge shape for the graph) — log it in `vault/decisions.md` as a new `D-nnn` entry (see the
file for the existing numbering and one-line format) rather than leaving it undocumented.
