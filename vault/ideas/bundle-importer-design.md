# Bundle importer design (milestone 1)

*Status: adopted / in-build. Decisions D-021..D-026. This is the durable design reference —
build from it. Grounded in the 60 pilot bundles + the v0.2 schema, verified against both
(not taken from docs alone).*

The importer consumes Gabriel's pilot bundles (`HEPKG_promopt_tests/pilot/bundles/*.json.gz`)
and builds one queryable knowledge graph, preserving status/evidence/history. Facts are
**given** — correctness is not my job; the output is the graph. Graded by the
`examples/integration/` fixtures + the count target.

## Data facts that drive the design (verified against all 60 bundles)

- **Counts (the M1 target):** 14,188 assertions = 11,309 `machine_verified` + 2,555
  `quarantined` + 324 `rejected`. 6,047 entities, 8,482 evidence. Zero `expert_accepted`,
  zero `expert_decisions` in the pilot → the **accepted view is empty for the pilot**; only
  the fixtures exercise it.
- **Signatures: 0/14,188 populated.** Object shapes are only `object_id` (9,831) or
  `object_value` (4,357). Store `signature` losslessly, build no logic around it yet.
- **assertion_ids never collide across bundles** (0) → assertions are cleanly one-bundle-owned.
- **entity_ids DO collide across papers by design** (slug IDs): 347 shared, and **334 of those
  carry divergent content** (label 312, aliases 237, attributes 203, kind 5). Sharing is the
  feature (it stitches papers into one graph); merge, never reject.
- **Redundant IDs exist** (the inverse problem): 487 entity IDs collapse to 223 concepts on
  hyphen/underscore/spacing alone (`b_jet`/`bjet`/`b-jet`; `pp_13tev`/`pp-13tev`/`pp13tev`).
  The slug mechanism leaks. → the aliases layer (D-025).
- **Evidence is shared:** one quote backs up to 37 assertions; 3,362 quotes cited by >1
  assertion → many-to-many is mandatory, not optional.
- **Qualifiers:** 55.3% of assertions carry them; 835 distinct keys, values messy free-text
  (`role` 200+ values, `level` a long tail) → store lossless as JSON, promote NOTHING in M1.
- **All referential integrity holds in-bundle** (subject/object/evidence/activity/revision_of
  resolve; exactly-one object shape for all 14,188; block_ids unique per snapshot;
  one bundle per paper in the pilot).

## Technology (D-022, D-023)

| Job | Tool | Note |
|---|---|---|
| Read bundles | `gzip` + `json` | stdlib |
| Shape gate | **`jsonschema`** vs Gabriel's shipped schema | the one new dep; authoritative, no drift |
| Meaning gate | hand-written checks | rules the schema can't express |
| Typed access | **deferred** | validated dicts for M1; generate from schema later if needed |
| Fingerprint | `hashlib` + ported `stable_json` (their `ids.py`) | stdlib |
| Store | **SQLite** (`sqlite3`) | system of record |
| Aliases Tier 1 | `re` | stdlib |
| Graph (later) | NetworkX now / **Neo4j Community local** later | a *projection* from SQLite, never the foundation |
| Tests | `pytest` | mirrors their fixture contract |

Store rationale: SQLite gives transactions + idempotency (PK/UNIQUE) + CHECK + the accepted
view + trace query with zero ops and one portable file. Neo4j/Postgres deferred — their wins
(traversal, concurrency, scale) are not what M1 needs; migration is cheap because the model is
fixed by the schema and it's a projection. Postgres+pgvector kept as a later option if the
aliases layer wants in-DB vectors.

## Pipeline (per bundle — all-or-nothing)

```
read (gunzip+parse)
 → [1] jsonschema shape gate        ── reject bundle
 → [2] semantic meaning gate        ── reject bundle  (accepted⇒evidence, one-object-shape, refs resolve)
 → [3] fingerprint (content hash)
 → [4] ledger check:  absent→proceed | same hash→NO-OP | same bundle_id+diff hash→ABORT
 → [5] transactional upsert (one txn; rollback on any error)
```

Loop over all 60; each bundle is its own transaction; order-independent; re-running is a no-op.
Entities **merge** in stage 5 (never abort). Completeness findings go to their own table.
The aliases layer is NOT in this pipeline — it runs later over the finished DB. A missing
aliases layer can never cause a rejection (validation is intra-bundle; every bundle re-declares
every entity it uses).

## Identity & idempotency (D-024, **corrected by D-027**)

- **paper = `arxiv_id`** (durable, one row per paper). Scientific records point at the bundle;
  the bundle points at the paper → reach the paper *through* the bundle, preserving version
  history.
- **`bundle_id` is IDENTITY-ONLY** — verified: `content_id("bundle", {schema, paper, source_hash,
  normalization})`. It does **not** cover assertions or statuses, so it is *stable across
  re-extractions of the same paper*. (The 4 fixtures legitimately share one bundle_id.)
- **`assertion_id` is claim-derived** — `content_id("assertion", {paper, family, subject,
  predicate, object, value, signature, qualifiers, evidence})`. Change any claim field and the id
  changes, i.e. it is a *new* assertion. Corrections therefore always arrive as new assertions
  (new id + `revision_of`), with the original flipped to `superseded`.
- **Idempotency**: `bundle_content_hash` (whole-bundle fingerprint) in the ledger is the fast
  **"byte-identical → no-op"** path only. It cannot decide conflicts — all 3 fixtures share a
  bundle_id yet have 3 different fingerprints.
- **Conflict detection is PER ASSERTION** (D-027):
  - id not seen → **insert** (corrections land here);
  - seen, only `status` differs → **update status** + append to `assertion_status_history`;
  - seen, any other field differs → **CONFLICT, abort the whole import**.
  `MUTABLE_ASSERTION_FIELDS = {"status"}` — the only field that legitimately moves.
- **`bundle_id` authenticity** (`bundle_id_matches_content`) is **warning-level, never a hard
  reject** — a mismatch may just mean the supervisor changed his id algorithm. 0/60 fail it.

## Table layout (D-024)

Groups + FK spine:
```
assertion.bundle_id ─► bundle_import.bundle_id ─► paper.arxiv_id
bundle_import.source_hash ─► source_snapshot          (many bundles → one snapshot)
evidence.(source_hash,block_id) ─► source_block
assertion ─ assertion_evidence ─ evidence             (many-to-many)
entity_occurrence.bundle_id ─► bundle_import           (entity node sits above, bundle-agnostic)
```

- **bundle_import** — ledger; PK `bundle_id`; FKs `paper_arxiv_id`, `source_hash`;
  `bundle_content_hash`, `schema_version`, `imported_at`; `usage`/`warnings` as JSON.
- **paper** — PK `arxiv_id`; the 7 PaperIdentity fields; denormalized `latest_bundle_id`,
  `source_hash`, `paper_map`(JSON).
- **source_snapshot** — PK `source_hash`; `paper_id`, `normalization_version`, `title`,
  `experiments`, `problems`, `source_path`, **`conversion_qa`(JSON)** (was missed first pass).
- **source_block** — PK (`source_hash`[injected], `block_id`); kind/order/section/dom_id/text/
  text_hash/table_rows/attributes.
- **entity** — PK `entity_id`; kind/label/aliases/attributes/external_ids; **no bundle_id**
  (merged canonical node = summary of all occurrences).
- **entity_occurrence** — PK (`bundle_id`[inj], `entity_id`); + `paper_id`[inj]; the raw
  per-paper version, kept verbatim. Merge = union onto the `entity` node; provenance lives here.
- **entity_divergence** — a **VIEW** over entity_occurrence (disagreements), not a stored table.
- **assertion** — PK `assertion_id`; `bundle_id`[inj], `paper_id`[inj]; subject_id, predicate,
  family, object_id | object_value(JSON) | signature(JSON) with **CHECK exactly-one-non-null**;
  status, support, extraction_method, activity_id, revision_of, notes, **qualifiers(JSON,
  lossless, NO promoted columns in M1)**. Indexes: status, predicate, subject_id, object_id,
  revision_of.
- **assertion_evidence** — junction PK (assertion_id, evidence_id).
- **evidence** — PK `evidence_id`; `bundle_id`[inj]; source_hash, block_id, dom_id, quote,
  quote_hash, char_start/end, section_title, normalization_version.
- **activity / artifact / qa_finding / expert_decision** — metadata tables (+ `bundle_id`[inj]).
- **completeness_finding** — its own table, **never joined as a scientific edge** (contract #7);
  rule_id/rule_version/trigger_fired/verdict/note/evidence-ptr arrays.
- **assertion_status_history** (D-028) — append-only `assertion_id` · `old_status` · `new_status`
  · `bundle_id` · `observed_at`. Records **only observed changes between our imports** (not
  initial inserts, not the supervisor's internal lifecycle). Empty after a first import; needed so
  milestone 4 can *show* a promotion rather than silently overwrite it.
- **accepted_view** — VIEW `WHERE status='expert_accepted'` (superseded drop out, corrections
  stay). Filter, never a destructive rebuild.

`[inj]` = injected provenance key (not a native schema field; verified none exists natively).
All columns verified 1:1 against the v0.2 `$defs` — no hallucinated fields; only omission
fixed was `conversion_qa`.

## Contract → mechanism (the 7 pass conditions)

1 counts → full load + assert · 2 reimport no-op → ledger+hash · 3 conflict aborts → stage-4 +
rollback · 4 accepted view → `accepted_view` (expert_accepted only) · 5 evidence trace → the
join · 6 correction links → `revision_of` + `expert_decision.corrected_assertion_id` ·
7 completeness isolated → separate table. Fixtures reuse one `bundle_id` as a test handle → each
runs against a fresh DB in isolation. The six-way "no accepted result" (absent / excluded
category / failed extraction / quarantined / awaiting review / no match) is already derivable
from `status` + `category` + `qa_finding`/`warnings` + presence — the honesty backbone for gap
reporting, built in now, no schema change later.

## Module layout (D-026)

```
hepcoveragekg/
  kg/        store.py  schema.sql  queries.py  export.py  graph.py     ── store + query layer
  ingest/    reader.py  validate.py  canonical.py  importer.py         ── the importer (M1)
  aliases/   normalize.py (+ later fuzzy/embed/adjudicate)             ── aliases layer (dedicated)
  cli.py                                                                ── import / verify-counts / export
tests/       test_import_counts.py   test_integration_fixtures.py
```
`ingest/` maps 1:1 to pipeline stages; depends on `kg/store` (importer knows bundles, store
knows only tables). Three different "schema" words kept distinct: `config/schema.py` (old RAG
ontology) ≠ Gabriel's bundle JSON schema (input) ≠ `kg/schema.sql` (our tables).

## Build order (each step independently checkable)

1. ✅ **DONE** `kg/store.py` + `schema.sql` — 15 tables + `accepted_view` + 6 indexes, STRICT,
   `json_valid` guards, exactly-one-object CHECK; 12 tests green. Two **soft pointers** (no FK) by
   necessity: `paper.latest_bundle_id`/`paper.source_hash` (would be a paper↔bundle cycle) and
   `assertion.revision_of` (self-ref → insert-order fragility; semantic layer validates instead).
   21 FKs enforced; insert order must be parent-first.
2. ✅ **DONE** `ingest/canonical.py` — `stable_json`, `content_hash`, `content_id`,
   `bundle_fingerprint`, `expected_bundle_id`, `bundle_id_matches_content`,
   `assertion_identity_hash`; 16 tests green. Pinned byte-for-byte to the supervisor by
   recomputing a real declared `bundle_id`. Verified on the real fixtures: status-only → same
   identity hash (allow), notes-change → different (abort); 0/60 bundles fail the authenticity check.
3. ✅ **DONE** `ingest/reader.py` + `ingest/errors.py` + **vendored schema** — 19 tests green.
   Schema **vendored** into `ingest/schemas/` (self-contained/reproducible; survives the planned
   repo merge), guarded by a pinned sha256 **and** a skippable upstream-drift test.
   **Two schema facts found by checking**: (a) no `$schema` declared → Draft 2020-12 chosen
   explicitly; (b) **`schema_version` is `const` but NOT required**, so a bundle omitting it would
   pass — hence an explicit version check on top of jsonschema. `additionalProperties:false` at
   root and in 14/26 defs. Validator compiled once (~59 ms/bundle, 3.5 s for 60).
   **Verified: 60/60 pilot bundles and all 4 fixtures pass the shape gate — including
   `invalid_accepted_without_evidence`**, which is exactly why step 4 must exist; a test asserts
   that fixture passes here, marking the gate boundary.
   `errors.py` defines the whole hierarchy up front: `BundleError` → `BundleReadError`,
   `BundleShapeError`, `BundleSemanticError` (step 4), `BundleConflictError` (step 6).
4. ✅ **DONE** `ingest/validate.py` — 26 tests green. **Errors** (reject): accepted⇒evidence;
   exactly-one-object-shape; in-bundle referential integrity (subject/object/evidence/
   `revision_of`/`activity_id`/decision refs/evidence→block); ids unique **within** a bundle (all
   8 record types — across bundles entity_ids repeat by design and merge); `paper.arxiv_id ==
   source.paper_id`; `evidence.source_hash == source.source_hash`. **Warnings** (never block):
   inauthentic `bundle_id`; non-quarantined assertion with no evidence; dangling qa/completeness
   refs. Principle: *errors = contract + anything the store would refuse (pre-checked for a clear
   message); warnings = metadata + authenticity*.
   `revision_of` is the one the DB cannot guard (soft pointer) — this gate is its only check.
   **Verified: 0 violations across all 60 bundles for every rule, and 0 warnings raised** (the
   real data is fully consistent). Fixtures land exactly right: `invalid_accepted_without_evidence`
   **rejected here** (it passed the shape gate), `conflicting_bundle` **passes** — its defect is a
   conflict, which is step 6's job; a test asserts that boundary.
5. ✅ **DONE** `ingest/importer.py` — 14 tests green; **86 total**. Single-bundle transactional
   upsert, parent-first (satisfies 21 FKs), `executemany` for bulk lists. Entities **merge**
   order-independently (D-029: recompute from all occurrences; consensus-only attributes; verified
   byte-identical forward vs reversed over all 60). Two entry points: `import_bundle(dict)` /
   `import_bundle_file(path)`; returns an `ImportResult` receipt (counts + warnings + `skipped` for
   step 6). Warnings surfaced in the receipt, **not persisted** (CLI prints them; 0 across the pilot).
   **Schema fix D-030**: `qa_finding`/`completeness_finding`/`artifact` → `(bundle_id, id)`
   composite PK (content-derived finding ids recur across bundles — 18 qa collisions).
   **★ Verified end-to-end: importing all 60 through the real importer reproduces the milestone-1
   target exactly — 14,188 / 11,309 / 2,555 / 324**, plus entity 5,114 · entity_occurrence 6,047 ·
   evidence 8,482 · activity 617 · qa_finding 3,372. 60 bundles in ~1.3 s. Merged-entity attributes
   after consensus: 52% populated; only 5 of 347 shared entities fully contested.
6. ✅ **DONE** re-import guard in `importer.py` — 5 tests; **91 total**. A read-only check runs
   BEFORE any write: new bundle_id → import (step 5); same bundle_id + identical content-hash →
   **no-op** (`skipped=True`, condition 2); same bundle_id + changed content → classify assertions
   by `identity_hash`: any non-status change under an existing id → **`BundleConflictError`, abort
   before any write** (condition 3); only status changes / new claims → **skip + warn** ("applying
   re-import updates is milestone 4"). New `identity_hash` column on `assertion` (importer-maintained,
   nullable = claim hash minus status). Conflict aborts cleanly *without* a transaction (nothing was
   written). **Verified**: import all 60 twice → 60/60 skipped, counts unchanged and still
   14,188/11,309/2,555/324; `conflicting_bundle` aborts leaving the DB untouched; a status-only
   re-import warns rather than conflicts (proves the guard keys on identity, not equality).
7. ✅ **DONE** `tests/test_integration_fixtures.py` + minimal `kg/queries.py` — **all 7 contract
   pass conditions green**; 10 acceptance tests, **101 total**. `queries.import_counts` (keys match
   `expected-import-counts.json`), `queries.status_counts`, `queries.trace_assertion` (condition 5 —
   the M2 trace seed). C1 counts are compared to **Gabriel's `expected-import-counts.json`**, never
   hardcoded. C4 exclusion proven two ways: `corrected_bundle` (excludes superseded) + a real pilot
   bundle (accepted_view empty, excludes machine_verified/quarantined/rejected). C5 traces
   `accepted_bundle` → exact quote "Search for X in two-lepton events", block `title`, paper
   9999.00001. **Out of scope (separate deliverables)**: v0.1 migration input, `expected-export/`.
8. ✅ **DONE** `cli.py` + `tests/test_import_counts.py` — **← MILESTONE 1 COMPLETE. 108 tests green.**
   CLI: `python -m hepcoveragekg.cli [--db PATH] import <dir|file>` and `verify-counts` (argparse,
   no new dep). Dir mode globs `*.json.gz` only (ignores `manifest.json`/`README.md`); a bad bundle
   is reported and the run continues (exit non-zero if any failed); `verify-counts` checks the pilot
   target and sets its exit code. **Real CLI run over the 60 bundles → `✓ milestone-1 target met
   (14,188 / 11,309 / 2,555 / 324)`, reimport → 60/60 skipped.** Test gate: exact status counts +
   record totals (entity 5,114 · occurrence 6,047 · evidence 8,482 · activity 617) + no-op reimport
   + order-independence + CLI end-to-end.
9+ (beyond M1) `aliases/normalize.py` (487→223), `kg/export.py` (byte-match expected-export/),
   `kg/queries.py` trace polish.

## Open

- **Known limitation — `init_schema` does not MIGRATE.** Every table is `CREATE TABLE IF NOT
  EXISTS`, so a schema change does not reach an existing `.db` file (the new column/table simply
  never appears). Tests never hit this — they use fresh `:memory:` DBs — but the persistent
  `data/processed/hepkg.db` did (created before the `identity_hash` column, then failed to import
  until deleted). **For M1 this is fine: the graph is rebuildable in ~2 s, so the fix is always
  "delete the file and re-import."** It becomes real at **milestone 4**, once the DB holds data that
  is NOT rebuildable (human review decisions, aliases-layer confirmed merges) — then we need a
  proper migration step, not delete-and-recreate.
- **TODO (user asked to be reminded): regenerate the ER diagrams** in
  `hepcoveragekg/kg/Pictures/` — made in a parallel Claude session on 2026-07-23 23:04–23:10,
  **now stale** (they predate `assertion_status_history`, so they show 14 tables not 15).
  Deliberately deferred until the schema stops changing. Also decide their home — they currently
  sit inside the Python package, and ~1.5 MB of PNG/SVG is untracked and not gitignored.

- **RAG pipeline repositioning** (D-021 note) — importer is the M1–4 backbone; the existing
  `harvesting/`+`extraction/` stack (working, verified) moves to the payoff/quality arms +
  baseline comparison. Not formally decided; possibly a supervisor conversation. See
  [[multi-agent-extension]], [[held-out-gap-validation]], [[grounding-and-evaluation]].
- Promoted qualifier columns — deferred; normalize-then-index via SQLite generated columns when
  a query needs it. Same messiness as [[open-vocab-reconciliation]].
