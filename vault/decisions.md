# Decisions — append-only

*Never rewrite entries. New decisions get the next number. Supersede by adding a new entry that references the old one.*

---

## D-001 (2026-07-05) — Reference DeepCollector, don't fork it
**Decision**: Build HEPCoverageKG fresh, porting DeepCollector's patterns (schema-as-config, cellular RAG, confidence-beats, hybrid retrieval, oracle dedup) individually rather than branching its codebase.
**Context**: Kickoff constraint from the user. DeepCollector solves a different problem (time-series dataset cataloging via Google Sheets) with heavy machinery we don't need.
**Rejected**: Forking — would drag in Sheets/Colab/agent infrastructure irrelevant here.

## D-002 (2026-07-05) — Three-layer sequential pipeline, no orchestration agent
**Decision**: harvest → extract → store as a plain sequential pipeline over a fixed paper list. DeepCollector's CatalogAgent and ResearchTools dropped; ResearchTools replaced by a plain `generate()` function.
**Context**: Kickoff architecture. The set of questions per paper is fixed by the schema, so there is nothing for an agent to decide.

## D-003 (2026-07-05) — Initial stack
**Decision**: Groq `llama-3.1-8b-instant` for generation (hosting later superseded by D-012; model family kept), BAAI/bge-small-en-v1.5 embeddings local on CPU, SQLite + NetworkX for the KG (Neo4j later), Python 3.11.8 via pyenv.
**Context**: Kickoff givens from the user.

## D-004 (2026-07-05) — One generic HTML harvester over a fixed page list
**Decision**: Harvesting = a fixed, hand-picked list of HTML pages (no dynamic crawl), processed by a single generic `html_harvester.py` with source-aware parsing, instead of three per-source harvester classes.
**Context**: All three sources (InspireHEP/arXiv/HEPData) reduce to "fetch a known URL, parse its HTML" — no per-source dynamic-query logic to justify separate classes.
**Rejected**: Per-source harvester files (original scaffold); dynamic discovery (possible future expansion).

## D-005 (2026-07-05) — Private GitHub repository
**Decision**: Repo published private via VSCode. HPC clones use an SSH key added to the GitHub account.
**Rejected**: Public — unpublished dissertation work.

## D-006 (2026-07-06) — Schema composed from three sources; 6 of 15 predicates implemented
**Decision**: `config/schema.py` combines (a) DeepCollector's CATALOG_SCHEMA *pattern* (query templates + plausibility thresholds + missing-value placeholders), (b) the supervisor's `graph_schema.py` *ontology* (entity kinds, predicates, statuses, dataclass contracts), (c) `graph_extractor.py`'s regex term lists as *controlled-vocabulary seeds*. Query templates written for 6 predicates central to coverage mapping; the other 8 listed explicitly in `DEFERRED_PREDICATES`.
**Context**: The supervisor's files define a graph/triple contract, not a flat table — so the DeepCollector pattern is keyed by predicate, not by column.
**Consequences**: `paper_reports_result` is structural (always added, confidence 1.0), never RAG-queried.

## D-007 (2026-07-07) — Harvester interface: fetch/parse, no shared state
**Decision**: `BaseHarvester` = `fetch(url) -> html` + `parse(html, url) -> HarvestedPaper`, replacing DeepCollector's `execute_harvest(state: CatalogState)`.
**Context**: DeepCollector mutates one shared catalog across a whole project's discovery run. We have no shared state at harvest time — each paper is independent; its retrieval index is built later, per paper, at extraction time. The fetch/parse split also future-proofs for supervisor-provided local files (parse without fetch).

## D-008 (2026-07-07) — Table extraction deferred
**Decision**: `html_harvester.py` extracts prose only; structured numeric results (yields, limits, cut-flows) are out of scope for now.
**Context**: At the time, HEPData (machine-readable result tables) was framed as the eventual numeric-results source, with arXiv `<table>` scraping only as a fallback if its coverage proved incomplete.
**Clarified 2026-07-10 (user, during vault review)**: HEPData harvesting itself is **not part of the current scope** — it's a possible future expansion (→ ideas index), not a planned pipeline stage. Revisit both HEPData and table scraping if/when structured numerics become needed.

## D-009 (2026-07-07) — CatalogState rescoped per-paper; arxiv_id is the primary key
**Decision**: Port CatalogState + HybridRetriever but scope one instance per paper: retrieval index over that paper's sections + accumulating `list[CoverageAssertion]` (name `catalog` kept for continuity). `PaperExtraction.arxiv_id` required, `cds_id` optional (flipped from supervisor's CDS-first original — our chain is arXiv-first). Dropped: find_item_by_name / multi-dataset management.

## D-010 (2026-07-07) — Predicates are single-value or multi-value, declared in schema
**Decision**: `multi_value` flag per predicate in PREDICATE_SCHEMA. Single-value (collision energy): two different answers are a conflict → confidence-beats arbitration keyed by predicate. Multi-value (targeted processes): different answers are coexisting facts → keyed by (predicate, object label).
**Context**: Without this, "supersymmetry" and "dark_matter" would wrongly compete for one slot.

## D-011 (2026-07-07) — Confidence: normalize in engine, validate strictly in dataclasses
**Decision**: `rag_engine.py` normalizes malformed LLM confidences (85 → 0.85) before constructing assertions; `CoverageAssertion.validate()` keeps the supervisor's strict [0,1] raise as a backstop.
**Context**: Reconciles DeepCollector's forgiving normalization with graph_schema.py's hard raise — each at the right layer.

## D-012 (2026-07-07) — Zero-marginal-cost LLM: self-hosted vLLM on DIAS, backend swappable via env only
**Decision**: `generate.py` targets any OpenAI-compatible endpoint via `LLM_BASE_URL`/`LLM_MODEL_NAME`/`LLM_API_KEY`. Production backend: vLLM on the DIAS cluster (SLURM GPU partition), Apptainer image `vllm/vllm-openai:v0.8.5`, model `NousResearch/Meta-Llama-3.1-8B-Instruct`, `--mem 128G`, SSH-tunnel access. Groq kept as commented fallback in `.env`.
**Context**: User requires zero cost at any scale but wants painless return to hosted APIs.
**Rejected**: Ollama on the user's Mac (user has HPC, didn't want the laptop tied up); conda/pip vLLM install (CUDA/toolchain fragility vs the cluster's GCCcore 11.2.0 era — container sidesteps it); `vllm-openai:latest` (bundles CUDA 12.9; cluster driver 550.163.01 caps at CUDA 12.4 → v0.8.5 is the newest CUDA-12.4.1 image); gated `meta-llama` repo (HF token lacked approval; Nous mirror has identical weights).
**Hard-won facts**: `/tmp` is node-local on DIAS (write job output under home); mmap of model shards counts against the SLURM mem cgroup (32G failed, 128G works); GPUs are A100 **80GB** (docs say 40).

## D-013 (2026-07-09) — Dedup at entity level via canonical IDs; never merge Result/Paper nodes
**Decision**: Two papers reporting the same final state remain two `Result` nodes pointing at one shared entity node — that multiplicity IS the coverage signal. Canonicalization tiers: (1) closed-vocab kinds → ID = `kind:vocab_term`, dedup free; (2) numeric kinds (energy, luminosity) → normalization function, no LLM; (3) Result/Paper → deterministic ID from arxiv_id, idempotent re-ingestion only; (4) open-vocab machinery deferred until something needs it. `merger.py` = pure function PaperExtraction → canonical graph ops, only `accepted` assertions reach the graph.
**Context**: Most entity kinds are closed-vocabulary — extraction already snapped labels to the vocab, so DeepCollector's fuzzy UniversalOracle matching isn't needed for them. Cross-paper confidence-beats is a non-problem (different subjects, no competing edges).

## D-014 (2026-07-09) — Dataset granularity: one node per distinct luminosity value
**Decision** (user's call): each distinct luminosity gets its own `dataset` node — no bucketing into run-period datasets.
**Rejected**: coarse (experiment, energy, run-period) buckets.

## D-015 (2026-07-09) — Identical final states across papers share one node
**Decision** (user asked directly; resolved): same composition → same canonical ID → same node, regardless of how the final-state representation question (composite vs fan-out) is eventually settled. Coverage counting must be a direct "how many Results point here" query, not query-time re-matching.
**Caveat noted**: flavor-inclusive vs flavor-specific labels ("2 leptons" ≠ "2 electrons") must not be conflated by canonicalization.

## D-016 (2026-07-10) — Reconciliation: offline batch + alias table + human confirm
**Decision**: Handling labels beyond exact match = (a) ingestion stays exact (vocab/alias hit or new node); (b) every N papers, a batch script nominates candidate pairs (bge embeddings + string similarity), an LLM adjudicates each pair (grounded with the labels' stored EvidenceSpans), the **user confirms**, and confirmed matches append rows to an **alias table**; (c) merges are non-destructive — raw labels stay on assertions forever, canonicalization resolves through aliases at build time, undo = delete a row; (d) confirmed-new labels get **promoted into the vocabulary** as data changes with provenance. Ports UniversalOracle's tiered shape (hard blocks → cheap signals → LLM only for the ambiguous middle → cache/budget/reject-on-error).
**Context**: User proposed an "agent system" doing this + tuning the schema; agreed the mechanism but repackaged: it's a ~150-line batch script, and feedback writes alias/vocab *data*, never prompt text (auditable method for the dissertation).
**Rejected**: agent framing (no planning/tool-choice exists in the task); LLM auto-applied merges (drift risk — NELL's lesson); LLM-edited prompts (unauditable behavior drift); physics-specific embedding models and external theory-RAG for the judge (see `ideas/open-vocab-reconciliation.md` for the upgrade ladder).

## D-017 (2026-07-10) — No memory between the six primary extraction calls
**Decision**: Primary predicate extractions stay mutually independent. Accumulated facts may be the *subject* of downstream checks, never *input* to primary measurements.
**Context**: Chaining anchors later calls to earlier errors (correlated failures defeat the cross-checking safety net — the deliberate "cellular" isolation in cellular RAG), muddies EvidenceSpan provenance (an assertion grounded in another assertion breaks the evidence-grounded-method claim in the dissertation), and introduces predicate-order dependence.

## D-018 (2026-07-10) — Consistency-check pass (flag-only)
**Decision** (user confirmed): after primary extraction, one call sees all extracted facts and judges cross-field consistency (e.g. energy↔luminosity run pairing). Its output can only flag — demote to `needs_review`, lower confidence — **never rewrite** an answer. Trivial cases handled by a deterministic run-configuration lookup, no LLM.

## D-019 (2026-07-10) — Leftovers pass: log-only predicate suggestions; no automated schema changes
**Decision** (user confirmed the idea): per informative section, ask "given the facts already extracted (serialized from state.catalog), what claims does this section make that none capture?" → append to a suggestions log. No graph writes, no schema edits. Recurrence across papers (found by clustering the log — same embed-group-adjudicate tooling as D-016 — or by reading) signals which DEFERRED_PREDICATES to implement or what to raise with the supervisor.
**Rejected**: automated predicate discovery/application (OpenIE's uncanonicalized-predicate explosion; a discovered predicate arrives without object_kind/template/validation/query-role anyway, so humans design regardless; predicate set is the supervisor contract); whole-paper prompts (8B long-context degradation; retrieval can't target unknown unknowns — the section is the right unit).

## D-020 (2026-07-10) — Vault created; single source of truth for project facts
**Decision**: This vault (`vault/`: overview, decisions, ideas, literature, logs, reports, inbox + root CLAUDE.md conventions) replaces `PROGRESS.md` (absorbed, deleted) and supersedes the assistant's private project memory (which now keeps only non-repo user preferences + a pointer here). Vault is self-contained in one folder to move atomically if this repo merges into the supervisor's. Vault commits are separate, prefixed `vault:`. Supervisor-facing reports generated on demand into `reports/`, not continuously maintained.

## D-021 (2026-07-23) — Milestone 1 = faithful bundle importer; facts are given
**Decision**: Build the importer that loads Gabriel's 60 pilot bundles into one queryable KG,
preserving status/evidence/history. Correctness of the facts is **not my job** — output is the
graph. Acceptance = reproduce the count target (14,188 / 11,309 / 2,555 / 324 by status) AND
pass the `examples/integration/` fixtures (the 7 pass conditions). Full design in
`ideas/bundle-importer-design.md`.
**Context**: The onboarding report (`reports/2026-07-23-hepkg-repo-onboarding.md`) reframed the
KG-construction thread — the bundles carry pre-extracted facts; I import, not re-extract.
**Consequences**: the existing `harvesting/`+`extraction/` RAG stack is repositioned, not the
KG backbone anymore (see D-021 open note in the design doc; not formally re-scoped yet).

## D-022 (2026-07-23) — SQLite is the system of record; the graph is a projection
**Decision**: The durable, validated store is **SQLite** (one file, stdlib). Any graph engine —
NetworkX now, **Neo4j Community Edition (local)** later — is a *projection built from* SQLite,
never the foundation. Move to Neo4j only when query ergonomics / visualization / a live service
demand it (not scale — the full corpus is ~100k nodes, trivial for either).
**Context**: M1 is graded on transactional idempotency, conflict, accepted view, trace — all
relational bookkeeping SQLite enforces declaratively (PK/UNIQUE/CHECK/FK/txn). The graph model
is fixed by the schema, so deferring the engine carries no model risk; migration is an additive
projector. Extends D-003's "SQLite + NetworkX now, Neo4j later".
**Rejected**: Neo4j-first (adds ops + weaker bookkeeping guarantees for the part M1 is graded
on; closes the cheap-optionality door); Postgres-first (its wins — concurrency, multi-user,
scale — are problems a single-writer ~250k-row solo project doesn't have; portability/repro
favor one SQLite file; `pgvector` kept as a later option for the aliases layer if wanted).

## D-023 (2026-07-23) — Validation: jsonschema gate + semantic layer; Pydantic deferred
**Decision**: Two validation responsibilities, run before any DB write: (1) **`jsonschema`**
against Gabriel's shipped v0.2 schema = the authoritative *shape* gate (honors "read enums from
the schema, not hardcoded"); (2) a small **hand-written semantic layer** = the *meaning* rules
the schema can't express (accepted⇒evidence, exactly-one-object-shape, in-bundle ref
resolution). **Pydantic is NOT a third validator** — deferred for M1 (work with validated
dicts); if the later query/domain layer wants typed access, **generate** models from Gabriel's
schema (no hand-maintained second copy → no drift).
**Context**: `invalid_accepted_without_evidence` proves the semantic layer is needed (the schema
allows accepted-without-evidence). One new dependency total: `jsonschema`.
**Rejected**: hand-porting Gabriel's Pydantic models as source of truth (drift on schema bumps);
Pydantic as the gate (belt-and-suspenders + drift). Revisits `ideas/pydantic-validation.md`.

## D-024 (2026-07-23) — Table layout, identity model, idempotency/conflict
**Decision**: Store schema per `ideas/bundle-importer-design.md`. Load-bearing choices:
(a) **paper = `arxiv_id`** (durable, one row) vs **bundle = `bundle_id`** (content-derived,
one per version) — records point at the bundle, bundle points at the paper, so version history
is preserved; (b) **entity split** — one merged bundle-agnostic `entity` node + faithful
per-paper `entity_occurrence` rows (merge, never abort; 334 shared entities diverge legitimately);
`entity_divergence` is a **view**, not a stored table; (c) **qualifiers stored lossless as JSON,
NO promoted columns in M1** (reverses the earlier role/level plan — values are messy free-text;
promote later via generated columns after normalization); (d) **exactly-one-object-shape** as a
DB `CHECK`; (e) **completeness_finding isolated**, never a scientific edge; (f) **accepted_view**
= a filter (`status='expert_accepted'`), never a destructive rebuild; (g) **idempotency** keyed
by `bundle_id`+content-hash, **conflict** = same bundle_id + changed content → abort; a new paper
version = new bundle_id (allowed).
**Context**: Every column verified 1:1 against the v0.2 `$defs` and every invariant against all
60 bundles (no hallucinated fields; only `conversion_qa` was missing, now added).

## D-025 (2026-07-23) — Aliases layer: dedicated reusable package, non-destructive, tiered
**Decision**: Entity de-duplication (`b_jet`/`bjet`/`b-jet`; `pp_13tev` split 5 ways) lives in a
**dedicated `aliases/` package**, run as a **separate pass over the finished DB** (never inside
import — import stays deterministic/offline). Non-destructive: it writes a `same_as` link table;
originals are never deleted; the canonical node is a derived cluster. Tiers: (0) block by kind →
(1) deterministic slug-normalize [M1 only] → (2) n-grams + **exact number/version guard** →
(2.5) embeddings (`bge-small`, candidate-generator only) → (3) LLM adjudication + human confirm
for genuine synonyms. This is **D-016's mechanism applied to Gabriel's bundle entity_ids**, and
it is **reusable project-wide** (coverage axes, query resolution, gap finder) → build it
first-class, versioned/auditable (coverage claims depend on the alias map).
**Context**: 487 IDs collapse to 223 concepts on spelling alone; the collision-energy axis is
split 5 ways — canonicalization is load-bearing for the gap map, not cleanup.
**Rejected**: merging inside import (breaks determinism); auto-merge on embedding/n-gram
similarity (over-merges b-jet/c-jet, sherpa 2.2.1/2.2.2 — similarity ≠ identity, worse in HEP).
See [[open-vocab-reconciliation]].

## D-026 (2026-07-23) — Module layout & build order
**Decision**: `kg/` = store+query+export+projection (fills existing stubs: store.py, schema.sql,
queries.py, export.py, graph.py); **`ingest/`** = the importer (reader/validate/canonical/
importer), mapping 1:1 to pipeline stages, depending on `kg/store`; **`aliases/`** = dedicated
package (D-025); `cli.py` + two test files. Build bottom-up, each step independently checkable;
milestone-1 gate is step 8 (60-bundle count + fixtures). Order in `ideas/bundle-importer-design.md`.
**Context**: Fits the existing package layout; keeps the three "schema" meanings distinct
(`config/schema.py` old ontology ≠ bundle JSON schema ≠ `kg/schema.sql`).
**Confirmed**: package name `ingest/`; `aliases/` as its own package (both user calls).

## D-027 (2026-07-24) — `bundle_id` is identity-only; conflict detection moves to the assertion level
**Decision**: **Supersedes the bundle-level conflict rule in D-024.** Verified in
`hepkg_acquisition/pipeline.py` that
`bundle_id = content_id("bundle", {schema, paper, source_hash, normalization})` — it does **not**
include assertions, statuses, or anything else, and reproduces exactly on 7/7 real bundles and
all 4 fixtures. So `bundle_id` is *stable across re-extractions of the same paper*. Conflict
detection is therefore **per assertion**:
- `assertion_id` not seen → **insert** (this is how corrections land — they carry a new id);
- seen, only `status` differs → **update the status** (+ a row in `assertion_status_history`);
- seen, any other field differs → **CONFLICT, abort the whole import**.
The whole-bundle fingerprint (`bundle_content_hash`) is kept but demoted: it is only the fast
"byte-identical → no-op" path, and cannot decide conflicts.
**Why exactly this rule**: `assertion_id = content_id("assertion", {paper, family, subject,
predicate, object, value, signature, qualifiers, evidence})` — the claim-defining fields are baked
into the id, so they *cannot* change without producing a different id. And
`review.py::apply_decisions` only ever sets `status` on an existing assertion; corrections are
appended as **new** assertions with a new id + `revision_of`. Hence `status` is the only field
that legitimately moves → `MUTABLE_ASSERTION_FIELDS = {"status"}`.
**Evidence**: `accepted` vs `corrected` fixture (assertion c898) differs only in status → allowed;
`accepted` vs `conflicting` (c898) differs in `notes` → abort. All three fixtures share one
`bundle_id` yet produce three different whole-bundle fingerprints — demonstrating that the bundle
level cannot distinguish a legitimate re-import from corruption.
**Also corrected**: the earlier claim that the fixtures "reuse a bundle_id as an artificial test
handle" was **wrong** — they legitimately share it (same paper/source/schema/normalization).
**Consequence**: a `bundle_id` authenticity check is available (`bundle_id_matches_content`) and
is **warning-level, never a hard reject** — a mismatch may just mean the supervisor changed his id
algorithm, which must not break our importer. 0/60 pilot bundles fail it.

## D-028 (2026-07-24) — status transitions are logged, not overwritten
**Decision**: Add `assertion_status_history` (`assertion_id`, `old_status`, `new_status`,
`bundle_id`, `observed_at`), append-only, recording **only observed changes** — not initial
inserts. The status-only branch of D-027 writes one row, then updates `assertion.status`.
**Context**: overwriting `status` in place loses the prior value. `expert_decision` records that a
decision *happened* but not the status it moved *from*, and machine-driven transitions leave no
trace at all. **Milestone 4 requires *showing* promotions land**, which is impossible without a
before/after.
**Scope caveat (do not over-claim this table)**: we only ever see per-bundle snapshots, so it
records what changed **between our imports**, never the supervisor's internal lifecycle
(`proposed → machine_verified → quarantined` all happen inside his pipeline before we see it).
**Rejected**: logging initial statuses too (14,188 redundant rows on first import; any past state
is reconstructable by walking changes backwards from the current status).
**Note**: not required by the 7 pass conditions — a deliberate judgement call, taken because it is
~6 lines now and a lost-history problem later.

## D-029 (2026-07-24) — entity merge is order-independent (recompute + consensus attributes)
**Decision**: The single canonical `entity` row is **recomputed from the full set of occurrences**
every time a new one lands, never first-write-wins. Rules (all pure functions of the occurrence
*set*): `aliases` = sorted union; `kind`/`label` = most frequent, ties lexicographic;
`attributes`/`external_ids` = **consensus-only** (keep a key iff every occurrence that states it
agrees; drop contested keys). Contested detail stays per-paper in `entity_occurrence`.
**Why order-independent matters**: first-write-wins would make the merged row depend on import
order, breaking the step-8 order-independence check *and* the contract's "export deterministically"
requirement. **Verified**: forward vs reversed import of all 60 bundles → byte-identical `entity`
table (5,114 rows).
**Why consensus-only over deterministic-pick**: never presents a contested value as settled;
nothing lost (occurrences keep everything); the aliases layer (D-025) does the real
value-reconciliation later. **Measured**: of 347 shared entities only **5** end up
fully-contested (empty merged attributes despite occurrences having them); 201 keep ≥1 consensus
attribute. So the merge is gentle and the reconciliation backlog is tiny.
**Rejected**: first-write-wins (order-dependent); deterministic most-frequent pick (can present a
contested attribute as authoritative); omitting attributes entirely (loses the free consensus ones).

## D-030 (2026-07-24) — content-derived metadata ids get a (bundle_id, id) composite key
**Decision**: `qa_finding`, `completeness_finding`, and `artifact` are keyed by
**(bundle_id, <id>)**, not the id alone.
**Context**: surfaced by importing all 60 into one DB — `qa_finding.finding_id` is content-derived,
so a finding about a *shared* entity recurs across papers with the **same id** (e.g. "object `met`
is outside the vocabulary" fires in every paper with a MET object → **18 colliding ids** in the
pilot). This is the same phenomenon as shared `entity_id`s (347) — content-derived ids legitimately
recur across bundles. One row per bundle preserves which papers each finding fired on.
**Not affected** (verified 0 cross-bundle collisions): `assertion_id`, `evidence_id`,
`activity_id`, `decision_id` — each is either claim-derived (includes the paper) or bundle-scoped by
construction, so a single-column PK is safe.
**Method note**: this is the *fifth* time checking one level deeper changed something — the earlier
within-bundle uniqueness check passed, but cross-bundle uniqueness for metadata did not; caught by
running the full 60 rather than trusting the per-bundle check.

## D-031 (2026-07-27) — Neo4j Graph Projection Architecture
**Decision**: The graph projection is materialized using offline CSV export and `neo4j-admin import` rather than live Cypher inserts. Nodes are split into `:Paper`, `:Occurrence`, and `:Canonical` to preserve the faithful per-paper entity context. Edges map `:Paper -[:HAS_OCCURRENCE]-> :Occurrence -[:RESOLVES_TO]-> :Canonical`. Assertions strictly connect `:Occurrence` nodes (or synthetic `:LiteralValue` nodes for scalar properties like signature or numeric values).
**Context**: Matches the project requirement (D-026/Phase 2) to maintain SQLite as the offline system of record. Generating CSVs is reproducible, fast, and testable without a live Neo4j service. `neo4j-admin import` with `--multiline-fields=true` allows rapid full-database hydration on demand.

## D-032 (2026-07-28) — entity node identity is (bundle_id, entity_id); the merged `entity` table is a rollup, not identity
**Decision**: `assertion.subject_id` / `object_id` now carry **composite** foreign keys into
`entity_occurrence(bundle_id, entity_id)`. The bare-`entity_id` `entity` table stays, demoted to a
convenience rollup (fast per-id lookup, the catalogue the aliases layer reads) — it is no longer a
foreign-key target and no longer claims to be identity.
**Context**: revisits D-024/D-029. Gabriel's contract states that entity-id equality across bundles
is **not** entity resolution — measured there as **347 shared ids, 752 with conflicting kind/label
under the same id**. Our importer merged on bare id, so two papers reusing an id string for
different things silently became one node. `entity_occurrence` already had the right grain.
**Cost**: near zero — assertion rows already carried `bundle_id`, so no write-path change was
needed. Composite FKs with a NULL column are unenforced under SQLite's default MATCH SIMPLE, which
is exactly right for `object_value` / `signature` assertions.
**Consequence**: cross-paper "same thing" is now *only* ever an explicit claim — which is precisely
what the `aliases/` layer emits. The architecture Gabriel's contract calls `resolves_to` is the one
we already had.

## D-033 (2026-07-28) — Tiers 1/1.5 and Tiers 2/3 are separate commands, and the deep pass takes an explicit output path
**Decision**: `build()` runs Tiers 1+1.5 only (deterministic, offline, 0.09s, no model, no network).
Tiers 2/2.5/3 live behind `aliases deep`, with `--deep-out` **required** at the library boundary and
`--dry-run` to report candidate counts without contacting an LLM.
**Context**: `build()` had come to call the deep pass unconditionally, so the cheap tier needed a GPU
and a live endpoint, and the test suite loaded an embedding model and hit the network. The deep pass
also wrote to a path hardcoded relative to the working directory — running `pytest` from the repo
root destroyed a real 68MB result file (recovered from the cluster). Output paths in library code
must never depend on the caller's CWD.
**Also**: the SLURM script stamps the output with `$SLURM_JOB_ID`, so two runs cannot overwrite each
other either — the class of bug is closed, not just the instance.

## D-034 (2026-07-28) — a failed LLM call is an `error`, never a `False`
**Decision**: adjudication results carry `status` ("ok"/"error") and `error_type`; on failure
`is_match` is **None**, never `False`. `propose_deep_semantics` routes on `status`, so errors reach
neither `matched` nor `rejected`, and a run above 2% failures is logged at ERROR and flagged by the CLI.
Auth/permission/not-found **raise** rather than return — they would fail identically on every pair.
**Context**: the previous code returned `is_match=False` on any exception, making a rate-limit or a
dropped connection indistinguishable from a genuine negative verdict — and the negatives are exactly
what the contradiction analysis reads.
**Measured after the fact**: the 126,335-pair run contained **75 errors (0.059%)** — 74 timeouts and
one parse failure. So the contamination was negligible *on a dedicated vLLM*, and the earlier
"742 contradictions" finding stands. The fix still matters: on a shared, rate-limited API 429s are
routine, and that is now a supported deployment (concurrency, timeout and retries are configurable,
default concurrency 8 rather than the hardcoded 100 that suits a private server).

## D-035 (2026-07-28) — Tier 2 encoder is BAAI/bge-base-en-v1.5; Jaccard is kept over Levenshtein
**Decision**: replace SciBERT with **bge-base-en-v1.5**, semantic threshold **0.85**. Keep the
Jaccard character-trigram lexical pass. Both configurable (`ALIASES_EMBED_MODEL`,
`ALIASES_SEM_THRESHOLD`, `ALIASES_LEX_THRESHOLD`).
**Context (measured on HEP probe pairs, scoring only what the embedding is responsible for — guards
already veto number-bearing pairs)**:

| model | worst synonym | best look-alike | margin |
|---|---|---|---|
| allenai/scibert (previous) | 0.877 | 0.975 | −0.098 |
| all-MiniLM-L6-v2 | 0.748 | 0.873 | −0.125 |
| **BAAI/bge-base-en-v1.5** | 0.892 | 0.861 | **+0.031** |
| BAAI/bge-large-en-v1.5 | 0.850 | 0.882 | −0.032 |
| intfloat/e5-base-v2 | 0.932 | 0.929 | +0.003 |

SciBERT is a masked-LM checkpoint with **no trained similarity head** — sentence-transformers wraps
it as Transformer+Pooling, crushing scores into a narrow band (unrelated text floors at 0.67, so
`signal region` vs `control region` scored 0.855 against a 0.85 threshold). **bge-large is worse
than bge-base** despite 3× the size. The threshold sits deliberately *below* the measured 0.876
boundary: this stage blocks rather than decides, an extra candidate costs one LLM call, a missed
synonym is unrecoverable.
**Jaccard over Levenshtein**: every dangerous HEP look-alike is a single-character swap (b/c jet,
s/t channel, W/Z, 13/14 TeV) — and so is every true spelling variant (colour/color). Edit distance
therefore scores both classes alike and rates the dangerous ones *higher* than Jaccard does (W/Z
polarization: Jaccard 0.882, Levenshtein 0.944). Trigram Jaccard penalises a changed character much
harder on short strings, which is the behaviour this corpus needs.
**Effect**: candidates fell **126,335 → 14,656** (9,570 after guards) at the same threshold.
**Known gap, unsolved**: abbreviations. `MET` vs `missing transverse momentum` scores 0.496
semantically and 0.091 lexically — invisible to *both* signals. Needs a dictionary, not a threshold.

## D-036 (2026-07-28) — guard vetoes require two independent views of "the numbers differ"
**Decision**: the number guard vetoes only when the **set of number tokens** *and* the **digit
signature** both differ. Units are matched by regex (digit-adjacent or standalone), with `ab`
recognised only when it follows a digit.
**Context**: writing the first tests for `guards.py` exposed three real defects — glued units
defeated the unit guard entirely (`13TeV` vs `13GeV` **passed**); `"ab initio"` was read as
attobarns; and comparing number-token *sets* vetoed `Pythia 8.212` vs `Pythia8 212` as different
versions. The first fix for the third bug (compare concatenated digit signatures) passed all 29 unit
tests and was **wrong** — real data showed `"3L channel (exactly 3 light leptons)"` has signature
`"33"` while `"(3ℓF)"` has `"3"`, turning **86 LLM-confirmed synonyms** into false vetoes. Neither
view is correct alone.
**Principle behind it**: a veto is **final** — the pair never reaches the LLM — whereas a pass costs
one call and the LLM can still reject it. So guards veto only on unambiguous evidence.
**Verified**: over **1,075,821** same-kind label pairs from the pilot, 0 newly blocked, 52 no longer
vetoed on spurious numeric grounds.

## D-037 (2026-07-28) — STRICT tables degrade at runtime instead of forking the schema
**Decision**: `kg.store.read_schema()` strips `STRICT` when `sqlite3.sqlite_version_info < (3,37,0)`.
It is the shared reader for *all* DDL in the project (import store and aliases store both use it).
**Context**: the cluster ships SQLite **3.36.0** (an earlier note recording 3.7.17 was wrong);
STRICT arrived in 3.37 and is a parse error even for `CREATE TABLE IF NOT EXISTS` on an existing
table. The workaround had been a hand-edited `schema.sql` living on the cluster — a fork that
silently drifts from the real one every time the schema changes. Only per-column type enforcement is
lost; every CHECK, foreign key and index still applies.

## D-038 (2026-07-29) — final states are **parsed prose**, not compiled qualifiers; LLM reads, code names
**Decision**: M3's signature step is (1) an **LLM parse** of the final-state label + evidence quote
into structured `{object, count, comparator}` triples, then (2) a **deterministic serialization** of
that structure into the canonical id (D-015). The LLM never writes the id string itself. Grouping
related signatures (flavour hierarchy, `2e+MET` / `2μ+MET` → "2 same-flavour leptons + MET") is a
**separate, later** LLM layer, not part of the parse.
**Context**: the standing plan — compile signatures from `count` / `subchannel` qualifiers — rested
on a description of the data that is **wrong**, and had been driving M3 planning since 2026-07-06.
Measured against `hepkg.db`: **161** `result_has_final_state` assertions across all 60 papers;
**zero** carry a `count` qualifier; 34% carry no qualifier at all; the ~15 that do encode
multiplicity use ~6 invented key names. There is nothing to compile. Nor is there the assumed
fan-out into object edges: the whole signature sits on **one** node as English prose
("Exactly one lepton plus ≥4 b-tagged jets"). Prose is therefore the only available source, so
reading it is the only possible mechanism.
**Why the split**: **126 distinct labels out of 138** — coverage counting only works if two papers
describing the same search reach the same id. A model asked to name the signature emits `1L+4b` once
and `1lep_4bjet` the next, splitting the counts that the project's central claim rests on. Reading
is what the model is good at; naming has to be consistent, so it is code's job. Same principle as
D-034 and the faithful-base/LLM-layers architecture.
**Scale**: 161 items — cheap enough to run several times and take the agreement rate as a free
reliability number.
**Consequence**: M3 is **on the critical path**, not deferred. Aggregate coverage questions are the
backbone of the evaluation (the one place the KG structurally beats document retrieval), and none of
them are answerable until signatures exist. `vault/ideas/final-state-representation.md` rewritten.

### D-038 addendum (2026-07-30) — superseded in part: signatures are coming from upstream
Gabriel is working on populating **signatures in the acquisition pipeline**. So the LLM parse
described above is **no longer the plan — it is the fallback.** Do not build it yet.
**What stays true**: the measurement (161 assertions, 0 count qualifiers, prose-only labels, 126
distinct labels of 138), and the principle that a model must never write the canonical id itself.
**What changes**: we wait, and we ask for two things — the **JSON shape** (which unblocks the query
layer immediately, without any data), and a **date** plus whether the existing 60 bundles are
re-extracted or only new papers carry signatures. Re-extraction ⇒ re-import ⇒ M4.
**Fallback trigger**: no signature data by the end of query-system build ⇒ run the parse on the 138
labels as a stopgap rather than let the evaluation stall.
**New risk to watch**: the backbone of the evaluation now depends on a deliverable we do not
control. See system.md §5 Phase 1.

## D-039 (2026-07-31) — LIGHTGPU/MIG is unusable for vLLM 0.8.5, at any model size
**Finding**: a single 20GB MIG slice fails in `get_device_capability()` →
`nvmlDeviceGetHandleByIndex()` with `NVMLError_InvalidArgument`, **before any weights load**.
Slurm places a MIG *UUID* in `CUDA_VISIBLE_DEVICES`; vLLM resolves it as a plain device index.
`VLLM_USE_V1=0` does not help — the call is reached on other paths too (jobs 48123, 48124).
**Corrects an earlier note** in `serve_vllm_70b.sh` claiming one slice was "fine for the old 8B".
That was an assumption, never an observation: `hpc/serve_vllm.sh` always used `-p GPU` with a real
A100. Two jobs have now disproved it.
**Consequence**: the *only* route to a model is **compute-gpu-0-1** (3× real A100 80GB), which has
been `IDLE+DRAIN` for a reboot since 2026-07-29 16:58. While it is down there is **no LLM available
at all** — not a small one for smoke tests, not the 72B. Job 48122 sits queued for it.
**What this does not block**: everything deterministic. The query layer's machinery, retrieval, the
planner loop (tested against an injected `chat`), and the faithfulness check all run without a GPU.
What is blocked is verifying the **wire format** — that a real vLLM server emits tool calls our
schemas accept — which scripted tests cannot cover because they use a fake response object.

## D-040 (2026-07-31) — one A100 on compute-gpu-0-1 is faulty; serve on a single card and refuse it
**Finding**: GPU `00000000:CA:00.0` fails with `uncorrectable ECC error encountered` on the **first
inference request**, killing the vLLM engine. Reproduced twice — jobs **48125** (5:45) and **48128**
(5:23), both at `rank=1`, both allocations including that card.
| | |
|---|---|
| uncorrectable ECC since the 15:30 reboot | **27** |
| uncorrectable ECC, aggregate | **805** |
| remapped rows, uncorrectable | 8 |
| **remapped rows, failure** | **1** |

The other two cards (`17:00.0`, `65:00.0`) show **zero** errors and zero remapped rows.
`remapped_rows.failure=1` with `pending=0` is the decisive reading: the card attempted to retire a
bad memory row and **could not**, and nothing is queued awaiting a reset — so a reboot cannot fix it
(and did not; the node rebooted at 15:30 and the card failed again at 17:00 and 17:13).
**Decision**: serve with **TP=1** (AWQ weights are ~39GB, comfortably inside one 80GB card, leaving
~35GB of KV cache), and **refuse to start on the bad card** — checked by **bus id**, since Slurm
renumbers visible devices per job so an index means nothing inside an allocation. Exit 75
(EX_TEMPFAIL) so a bad draw costs ten seconds rather than a five-minute load followed by a crash.
**Why not TP=2**: it takes two of three cards, so it is *likely* to include the faulty one — it did,
both times. TP=1 makes a healthy draw the common case. Revert to TP=2 for throughput once the card
is out of service.
**Not caused by us**: software cannot produce uncorrectable ECC errors, and eight already-retired
rows are a long-term degradation. Heavy sustained use (72B at 0.90 utilisation for 24h) *exposed* a
latent fault rather than creating one. Reported to the cluster admin.
**No silent corruption risk**: uncorrectable means *detected*, and CUDA aborts rather than returning
wrong values. The 2026-07-29/30 Qwen trial results stand.

## D-041 (2026-08-01) — Gabriel's extraction pipeline runs on our own vLLM; extraction is Sonnet, querying is Qwen
**Finding**, from reading `HEPKG_promopt_tests` directly:
```
cli.py:  run.add_argument("--provider", choices=["anthropic","openai","claude-cli"], default="anthropic")
run_pilot.sh:  hepkg-acquire run "$pid" --provider claude-cli --model sonnet ...
               export HEPKG_CORPUS_ROOT=/Users/gfacini/.../hep-papers-html-since-2020
```
Two consequences.

**1. Extraction is not blocked on Gabriel.** `--provider openai` accepts any OpenAI-compatible
endpoint — i.e. our vLLM server. `HEPKG_CORPUS_ROOT` is an env var, repointable at the DIAS corpus
(`/share/data1/xucapswo/hep-papers-html`, 2,969 papers). The pilot used `claude-cli` against
Gabriel's subscription and needed a probe/sleep loop for usage limits, so running it ourselves is
*better*, not merely possible. **The coverage-growth curve (S-47 version A) is bounded by cluster
time, which we control** — an earlier assumption that it waited on Gabriel was wrong.

**2. The graph was extracted by Claude Sonnet, and is queried by Qwen2.5-72B — already two different
models.** This matters for evaluation independence (S-37): correlated blind spots are the risk, and
they are already partly avoided. It also sets the constraint on the reference reader — it must be
**neither Sonnet nor Qwen** (so Gemini or GPT), or a disagreement is a model agreeing with itself.
**Note on same-model retrieval** (the question that prompted this): the query model never sees the
extraction prompt. What passes between the two halves is *data* — entity labels, predicate names,
kind distributions, via the generated schema card — not weights. A different model reading the same
card gets the same information. The residual mismatch is at the string level (`PYTHIA 8.212` vs
`Pythia8`) and is absorbed by hybrid retrieval and the aliases layer, which is what they are for.

## D-042 (2026-08-01) — a hand-written gold set already exists (`GROUND_TRUTH.md`), and it is stale
`HEPKG_promopt_tests/GROUND_TRUTH.md` — **~17 papers** with selection cuts, object definitions,
multiplicities and expected event class, written by reading the papers' selection sections. It even
carries a self-correction where the author re-checked against the paper and found their own entry
wrong (2103.01290, AK8 vs AK4 for the ISR jet).
**Why it matters**: it is human-made, it is about **final states and multiplicities** — i.e. M3, the
thing the whole evaluation backbone waits on — and it means *"we have no ground truth"* was never
quite true. It is also an independent check on the reference reader (S-37): does the big model agree
with a human on those 17?
**The caveat**: last committed **2026-07-05** ("M0+M1: measurement paper set, ground truth (11
papers)"), while the repo moved on to **2026-07-27** (`canonical entity map v2 + facet layer`). Those
are schema changes. The *physics* does not go stale — it was written from the papers, not the schema
— but the field names may no longer line up with what the pipeline emits.
**Action before scoring anything against it**: check the field names against current output (~30
min), and ask Gabriel whether he considers it current. It is his, and it is the only human-made gold
set either of us has.

## D-043 (2026-08-02) — `list_papers` and `count` disagree; `count` is right
**Finding**, while computing question truth: for the same predicate and the same entity set,
```
count()        -> 58 papers          list_papers() -> 59 papers      (the Pythia concept)
```
`count` ties the subject's occurrence to the assertion's bundle (`eo.bundle_id = a.bundle_id`).
`list_papers` composes `subjects_of` + `papers_of`, and **`papers_of` is a generic entity → papers
lookup that cannot know which assertion it came from**, so it returns every paper the subject appears
in. A sample entity shared across ten papers contributes all ten even when only nine assert the link.
**Consequence today**: asking *"which analyses used Pythia?"* returns 59 while *"how many?"* returns
58. Set questions and counting questions about the same thing disagree with each other.
**Decision**: `count` is correct; the evaluation generator computes truth with `count`'s join
(`eval/generate.py::_papers_for`) rather than calling `list_papers`. **The query-layer fix is NOT
applied yet** — changing query semantics while building the measurement would move the thing being
measured. Fix `list_papers` to do a single joined query, then re-run any affected numbers.
**Same family as the trap the vault already records** (join to papers via `entity_occurrence`): the
join is right, the *bundle tie* is what was missing.
**And it bit twice**: the hand-written seed truth for `hw-0004` (Herwig) made exactly this mistake and
claimed 22 papers where the correct answer is 17. Written by hand, in a file about being careful.
That is the argument for computing truth from `templates` rather than from bespoke SQL.

## D-044 (2026-08-02) — the deep alias proposals must NOT be confirmed; use them as a filter
`data/processed/aliases_proposed.json` (2026-07-28, 68 MB) holds the Tier 2/2.5/3 output that was
never confirmed into `same_as`:
```
126,335 pairs adjudicated · 4,238 marked match (3.4%) · 4,144 of them at confidence exactly 0.9
```
**Precision is roughly half**, judged by reading a random sample of 14, and the errors are physics
errors rather than near misses:
```
X  Dilepton ttbar final state     == Four-lepton final state (4e,4mu,2e2mu)
X  |eta| of LEADING DNN jet       == |eta| of SUBLEADING DNN jet
X  Muon IDENTIFICATION efficiency == Muon SELECTION AND TRIGGER efficiency
V  Pile-up reweighting modelling  == Pileup modeling uncertainty
V  ATLAS Run 2 13 TeV (139 fb)    == ATLAS Run 2 pp dataset, 139 fb
```
**Decision**: do not confirm. Applying ~4,238 merges at that precision would put ~2,000 wrong merges
into the graph, and every count downstream would be wrong **invisibly**.
**Confidence cannot rescue it**: 4,144 of 4,238 sit at *exactly* 0.9 — the "confidence measures
fluency, not correctness" failure the vault predicted, now observed on real output.
**But the file is immediately useful as an ambiguity FILTER (S-60)**, folded into the *broad* reading
of the invariance test. A false positive then only drops a question and can never produce a wrong
answer — the precision that makes it unusable for merging is harmless for filtering.
**Current dedup state, for the record**: `same_as` holds `normalize` 328 + `spelling` 19 only;
**342 of 5,114 entities merged = 6.7%.** Tier 1/1.5 only. So every ambiguity number measured today is
a *before* figure.
**Consistent with D-016 / the NELL lesson**: LLM-proposed merges are advisory, never auto-applied.

### AMENDMENT (2026-08-02, same day) — the file is STALE, and two claims above are wrong
Checked the dates instead of assuming (user's challenge, and correct):
```
2026-07-28 23:27   aliases_proposed.json written
------------------------------------------------ everything below is AFTER it
2026-07-29 13:04   Tier 3 REDESIGN: trial set, graph context, identity prompt, evaluator
2026-07-29 13:13   Keep confidence raw and measure calibration instead of bucketing it
2026-07-29 13:20   Send nearly all quotes, ordered by section
2026-07-29 13:27   Learn section priority per kind
2026-07-29 15:21   Accept the verdict under whichever key the model used
```
**1. The ~50% precision is a verdict on superseded code**, not on the current design. The redesign
added exactly the machinery aimed at the failures observed (graph context and an identity prompt are
what stop `dilepton == four-lepton` and `leading == subleading`).
**2. The "4,144 of 4,238 at confidence exactly 0.9" was OUR bucketing, not model fluency.** The old
`_confidence()` mapped values onto an enum (`if num >= 0.9: ...`); b28fa3b replaced it with a raw
float precisely so calibration could be *measured* rather than destroyed. Citing it as "the
confidence-measures-fluency failure, now observed" was wrong. (The underlying phenomenon is real and
recorded in that commit message — the 8B run put all 2,712 matches at >= 0.9 — but that is a
different run from the one inspected here.)
**Therefore**: **re-run Tiers 2/3 with the current code.** The DB has not changed since
2026-07-28 02:08, so the graph is the same; only the adjudicator improved. Do NOT confirm the output
either way (the standing decision above), and re-measure precision on a labelled random sample rather
than by eyeballing 14.
**What stands unchanged**: the *use* of the output as an ambiguity **filter** (S-60) rather than as
merges. That argument never depended on precision — a false positive only drops a question.

## D-045 (2026-08-02) — same `entity_id` across papers is mostly the same concept
Correcting an overstatement made earlier the same day (and used to justify rejecting `same_id` as a
deduplication label source).
```
entity_ids in >1 paper: 347   |   same label everywhere: 35   |   different labels: 312
```
The 312 look alarming and are mostly benign — the id is the stable anchor and the label is
**paper-local prose for the same thing**:
```
hepkg:generator:pythia8      21 labels  "Pythia 8" / "PYTHIA v8.2.2 (parton shower)" / "pythia8"
hepkg:object:jet             20 labels  "Jet (anti-$k_t$, R=0.4)" / "Reconstructed jet (anti-kT...)"
hepkg:systematic:luminosity  11 labels  "Luminosity uncertainty" / "Integrated luminosity unc. (1.7%)"
```
So the vault's "334 divergent" counts **label** divergence, not **concept** divergence, and
`same_id` looks like a usable positive label source after all (revises the S-51 framing).
**Caveat, stated rather than hidden**: this is the top cases eyeballed, not a random sample checked.
Confirming it means reading ~30 pairs and asking "same thing?" — a good ten-minute task for the
supervisor, since it needs physics judgement rather than effort.
**The two failure directions are not symmetric here**: *same concept, different ids* (under-merging)
is heavy and real — 6.7% merged, 48% of concepts ambiguous. *Same id, different concepts*
(over-merging) has little evidence. The problem points in one direction, which simplifies the fix.

## D-046 (2026-08-02) — intra-paper duplicate entities: 22 cases, a free dedup benchmark
Same paper, same kind, same normalised label, **different entity ids**:
```
22 cases across 7 of 60 papers
2207.00348  event_region  "qq h p t v 150 gev"  -> 2 ids
2106.01676  bsm_model     "wino bino simplified susy" -> 2 ids
```
Within one paper the extractor should have reused the id, so each of these is an unambiguous
extraction defect — no judgement call, no threshold.
**Two uses**: (1) they corrupt Tier A truth (S-59) — *"which regions does 2207.00348 define?"* would
list one region twice, so generation must pre-check for them; (2) **a free recall benchmark for the
aliases layer** — 22 pairs we *know* must merge, that nobody had to label. If deduplication misses
these it is broken.
Small (7 of 60 papers), so not the dominant problem — but free signal.

## D-047 (2026-08-03) — `contents_of`: the graph had no paper -> contents operation
**Finding**: every template started from a CONCEPT and found papers. `papers_of` went
entities -> papers; **nothing was its inverse**. So *"which generators does analysis 2001.06899
use?"* could not be answered at all — the planner searched for the arXiv id, found nothing usable,
and passed the id into `describe` as though it were an entity id:
```
60 `unknown_entity_id` errors across 45 questions · tool_error_rate 0.386 · count_correct 0.06
answers stating a paper "is not found in the current graph" when it plainly is
```
**Why it went unnoticed**: every question shape built before Tier A starts from a concept, so the
reverse direction was never exercised. It is also the most natural question a physicist has about one
paper — *"what does this analysis cover?"* — so a **coverage map** could not answer its own core
question.
**Decision**: add `templates.contents_of(paper_ids, predicate=None)`, exposed as a planner tool whose
description says explicitly to pass the arXiv id directly and NOT to search for the paper or hand it
to `describe`. Joins `entity_occurrence` on **bundle_id AND entity_id**, like `count`, so it does not
inherit the D-043 over-count.
**Measured after**: `count_correct` 0.06 -> 0.50, `tool_error_rate` 0.386 -> 0.083, `answered`
0.56 -> 0.85, set questions at 1.00 retrieval / 0.895 naming.
**Two rows, one query each**: display rows are DISTINCT per fact so the model sees one line per fact;
the evidence lookup needs `assertion_id`, and adding it to the display query would undo that grouping.

## D-048 (2026-08-03) — `_evidence_for` silently matched every row; now it refuses
**Finding**: the helper wraps a caller's SQL as `SELECT assertion_id FROM (<sql>)`. When the caller's
query does not select `assertion_id`, SQLite resolves the bare name against the **enclosing** scope
(`ae.assertion_id`), so the condition becomes always-true. `contents_of` hit it on first use:
**11 rows of output, 8,369 evidence ids, no error, no warning.**
**Decision**: `_evidence_for` raises if `assertion_id` is absent from the query. Not defensive
padding — the failure is invisible, produces a plausible result, and every template that ever calls
this helper is exposed to it.
**Family**: the third silent-SQL trap in this project, after "join via `entity_occurrence`, not
`paper_reports_result`" and "count facts, not rows". All three return a wrong answer with no error.

## D-049 (2026-08-03) — compute-0-1 accepts jobs and runs nothing; excluded
**Finding**: two evaluation jobs ran **8 hours and produced no output file at all**. Slurm always
creates the output file if the script runs, so nothing ran.
```
compute-0-0   srun hostname  -> returns instantly
compute-0-1   srun hostname  -> hangs indefinitely; Slurm later reports ReqNodeNotAvail
```
**Decision**: `#SBATCH --exclude=compute-0-1` in `hpc/eval_job.sh`, with the evidence written into the
script so it is not silently reverted. Remove once the node is fixed or drained.
**Same shape as D-040** (the faulty A100): a resource that accepts work and does none, where the
failure looks like slowness rather than an error.
**Two other things that night, worth keeping together**: `langgraph` was not installed in the DIAS
venv, so every question died with `ModuleNotFoundError` — that alone would have cost the night. And
the new circuit breaker (20 consecutive errors -> stop) **worked**, cutting a doomed run to 20
seconds — then corrupted its own output file by calling `_rewrite_meta` from inside the still-open
`with` block, producing `{"r{"run_id":...`. A correct guard ruined by where it was placed.

## D-050 (2026-08-07) — Tier 1 keys on the label as well as the id; under-merging is not the safe direction
**Finding**: every merge in the graph came from `normalize` (328) and `spelling` (19), all scoring
1.0 — no judgement call had ever been applied. And Tier 1 folds the **entity id slug**, never the
label, so two entities agree only when their ids look alike:
```
madgraph5_amcnlo  -> madgraph5amcnlo  ]  merged
madgraph5-amcnlo  -> madgraph5amcnlo  ]
mg5-amcnlo        -> mg5amcnlo           not merged
```
All three carry the **byte-identical label** `MadGraph5_aMC@NLO`. Seven such entities sat in four
clusters. Measured across the graph: **41 groups of byte-identical, same-kind labels split across
clusters** — wrong by definition, no physics involved.
**Decision**: add two rules beside the id rule. `label_exact` (same kind, identical label) and
`label_norm` (same kind, labels equal after folding case and separators), the latter emitted only
for pairs `label_exact` missed so the two counts do not double-count one edge.
**The dot is kept when folding labels, unlike slugs.** Slugs are identifiers where `pythia8.210` and
`pythia8210` are one thing; labels carry cut values, and stripping the dot folds `pT > 2.0 GeV` onto
`pT > 20 GeV` — a false merge that changes a number.
**Result**: +138 `label_exact`, +20 `label_norm`; 41 split groups -> **0**. All 20 folded merges
inspected by hand and correct (hyphenation, capitalisation, spacing around `=`); no version boundary
crossed. Node counts fell — generators 223->150, systematic uncertainties 803->716,
detector objects 227->191.
**Why this was the right direction**: conservatism is not accuracy here, it is bias. Under-merging
inflates every "how many distinct X" answer, which is the coverage map's headline number. Backup of
the pre-change DB kept outside the repo; `same_as` is additive and `entity_canonical` is rebuilt from
it, so the change is reversible by deleting the `label_*` edges and re-materialising.

## D-051 (2026-08-07) — merging moves *inventory* counts, not *paper* counts
**Finding**, measured while designing the dedup ablation. Canonical resolution inside `_FACT_KEY`
changes almost nothing: **66 of 13,839 distinct facts, 0.48%**. For the b-jet cluster the answer was
identical either way — 31 papers, 162 facts. The reason is that spelling is consistent *within* a
paper, so two papers using different spellings already have different subjects; there is nothing for
the fact key to collapse. `expand_canonical` plus search breadth already deliver the papers.
Counting *things* is a different story: detector objects 227->191, generators 223->150 (33%).
**Decision**: the alias ablation must be scored on **inventory questions** ("how many distinct X does
the corpus define", "list the distinct X"), not on the paper-counting questions the generator
currently emits. Run as-is and every arm returns the same number, which would read as
"deduplication does not matter" — the wrong conclusion drawn from the wrong questions.
**Correction to the comment on `_FACT_KEY`**: it claims the canonical resolution is what makes
counting dedup-sensitive. It is doing that for 0.48% of facts. The sensitivity comes from entity
inventory queries instead. Not a bug — an overstated rationale.

## D-052 (2026-08-07) — derive the facet + signature layers, don't import them
**The choice**: the supervisor's `pilot/analysis_facets.jsonl` is 60 per-paper analysis cards. It
could be imported as data. Instead we recompute it from our own graph using his vocabulary, and keep
his file as a **test fixture** — the oracle we check against, never production data.
**Why**: recomputing reproduces all 60 cards exactly (verified). So the file is a snapshot of a
function, and a snapshot is stale the moment paper 61 arrives. Deriving keeps the layer live.
**Cost, stated plainly**: we now own a copy of his regex tables and they drift when he edits them.
`test_vendored_vocabulary_is_unmodified` pins the hash, so drift is a failing test with a diff rather
than a silent disagreement.
**What is vendored vs ported**: `vocabulary.py` is copied **verbatim** (hash-pinned) plus a 30-line
`models.py` carrying only the `DetectorObjectName` enum it imports — the alternative was editing a
file we do not own. `signatures.py` is **ported**, because upstream takes pydantic models and we take
SQLite rows; two tests run both implementations over every candidate and assert identical verdicts.
Upstream modules are loaded **from git at the pinned revision**, not from the working tree, so a
branch checkout cannot silently skip the tests that validate our copies.

### The bug parity caught: facets are per-occurrence, not per-entity
First implementation derived from `entity.label` and got extra and missing values on 8 papers.
Cause: `entity.label` is a rollup — the schema literally calls entity_id a "convenience rollup ONLY,
not identity — see entity_occurrence". **312 entity_ids carry different labels in different papers**,
and the difference decides the tag:
```
2107.12553 wrote "b-tagged small-R jet"  -> no match (the object patterns are anchored)
the rollup label is "b-tagged jet"       -> BJet
```
Deriving from the rollup credits a paper with an object it never named. `entity_facet` is therefore
keyed `(paper_id, entity_id, field, value, vocabulary)`. This is the fourth silent-SQL-shape trap in
the project (D-048's family): a plausible answer, no error.

### Shape
Two tables and a view, not four tables. `entity_facet` (tags and renames together — the difference is
a cardinality constraint, not a shape) and `assertion_signature_derived`. `analysis_card` is a **VIEW**
because it is fully computable from `entity_facet`; storing it would add a second source of truth that
can drift, which this project has hit twice. Every table carries `vocabulary`, so re-deriving is
delete-then-insert scoped to one version and a future `facets-v2` can be held alongside v1 and diffed.
Neither table touches `entity` / `entity_occurrence` / `assertion`; a test asserts that.
**Naming**: `entity_facet` vs the aliases layer's `entity_canonical` are different rungs (S-68) —
one says a label belongs in an enum, the other says two entities are the same entity.

### Measured
Facets, per occurrence: **82.0% overall** (4,202 occurrences, 757 unmatched). Worst kinds
detector_object **74.1%** and systematic_uncertainty **70.8%**. Misses have two causes that need
separating by hand: genuine vocabulary gaps ("OS-SS subtraction", "Displaced electron") and
mis-kinded entities (2006.05880 files its signal regions as `detector_object`, which is why its card
shows 9/24 objects matched).
Signatures: **734 leaves from 2,692 candidates, 0 native**, and the OR rule's `subchannel` flag
appears on **46**. Recall is a property of the extraction, not of the rules — the AND/OR distinction
survives extraction ~2% of the time.
**Over-tagging is real too** and not yet quantified: 68 entities carry 3+ tags, and
`"Data-driven control sample ... replacing simulation"` is tagged `SimulationBased` — the regex
cannot see negation. Precision needs a hand-labelled sample; recall is measured, precision is not.

## D-053 (2026-08-07) — the `facets` tool returns the entity trail, never bare paper ids
**The tool**: `facets(field, values, mode='all'|'any', category=, experiment=)` over `entity_facet`.
Zero model calls, zero retrieval, no spelling to get right. All of the supervisor's Tier 1 gold
answers reproduce through the planner's executor: Q1 18 papers, Q2/Q3/Q5 exact sets.
**Three deliberate choices, each against an obvious cheaper option.**

**1. It returns the labels that caused each match, not just paper ids.** The card is lossy by design.
Six papers carry `ABCD` and all six do a different ABCD — `Modified ABCD estimate`,
`Two-dimensional ABCD sideband method`, `Multidimensional ABCD reweighting`. Shown six ids, a reader
concludes they share a method; shown the labels, they see six variants, which for a coverage map is
the more interesting answer. The tag says where to look, the label says what is there (S-69).

**2. The whole vocabulary is inline in the schema card, not behind a lookup tool.** Measured at
**~590 tokens** for every value in every field, against a 1,762-token card. Cheap enough that a round
trip would be the wrong trade — and inline, the model never has to guess that the key is `BJet`
rather than "b-jet", which is exactly the Tier 1 failure the layer exists to prevent. Values listed
are those the vocabulary actually MATCHED in this corpus, not the full enum: listing keys that match
nothing invites calls that correctly return zero, and an empty result cannot distinguish "no paper
does this" from "wrong key". Per-field coverage is printed beside each list.

**3. An empty result says WHY it is empty.** `facets('objects', ['b-jet'])` now answers
`NOT IN THE VOCABULARY: ['b-jet'] -- these match nothing by definition`, while a real key that
happens to match no paper is not flagged. Without this the two cases are byte-identical to a reader,
and the misspelling silently becomes a confident "no papers do this".

**Every result carries coverage and a candidate-set caveat** in `note`, because a facet miss is
invisible: the entity keeps its label and all its facts and simply never appears. An uncaveated set
of ids reads as complete when 26% of objects and 29% of systematics carry no tag at all.
**Coverage denominators come from the declared `CARD_FIELDS` mapping, not from the kinds present in
`entity_facet`.** Reading them back from data joins on entity_id alone, and an entity_id that is
`detector_object` in one paper and `object_definition` in another inflates the denominator (515 vs
486 for `objects`) and understates coverage — the same rollup-vs-occurrence confusion as D-052.

## D-054 (2026-08-07) — search hits carry facet keys; the prompt never lists them
**Reversal of a choice made the same day.** D-053 put the whole closed vocabulary inline in the
schema card (~590 tokens), arguing the model must never have to guess that the key is `BJet` rather
than "b-jet". Two objections from the user killed it, and the second is disqualifying.

**1. It does not scale.** The vocabulary is curated and will grow; a design whose prompt cost tracks
it is a design with a deadline.

**2. It leaks the answers.** The values that read as most natural to list — BJet, MET, ABCD,
Unfolding, SUSY — are **five of the seven** distinct keys in the supervisor's Tier 1 gold answers.
Any Tier 1 score would have been partly measuring that the answer key was pasted into the prompt.
This is not hypothetical: writing the section that *warns* about leakage, I used `ABCD` as the
worked example and the regression guard caught it.

**The measurement that made the reversal safe.** The keys do not need shipping, because search
already supplies them. BM25 only, no embeddings, on the phrasings his Tier 1 questions use:
```
b-jets -> BJet  |  missing transverse momentum -> MET  |  ABCD background estimate -> ABCD
HistFitter -> HistFitter  |  unfold their distributions -> Unfolding  |  supersymmetry search -> SUSY
```
**rank 1, 8 times out of 8.** A hit IS the answer to "which key covers this concept", and it is
grounded in an entity that demonstrably exists.

**The shape now**: `Hit.facets` carries the tags; the planner includes them on every search row and
says so in the note; the card block is **~137 tokens** of field names plus per-field coverage, no
values; `PURPOSE` gains a descriptive `FACET TAGS` section.
**The line, stated once and applied at three levels**: *the prompt describes structure, the data
supplies content.* The rest of the card already worked this way — it lists entity kinds and
predicate names, never example labels.

**Known residual**: `MET` appears in the NOTATION block, which predates facets and explains a
standard physics abbreviation. The facet key happens to be spelled the same. Removing useful notation
to avoid the coincidence would cost more than it buys; the leakage test excludes short keys and says
why.

**Weakness accepted, not hidden.** On *described* phrasings search is much weaker — 6/8, and both
misses were total (`"estimating background from data in sidebands"` finds no ABCD-tagged entity;
`"correcting detector effects to particle level"` finds no Unfolding). Under the inline vocabulary
the model could have reached those keys anyway. That advantage is exactly the contaminated kind. The
hole is now a *measurable retrieval failure* rather than a hidden prompt advantage.

### The same trap one level up: describing vs prescribing
`FACET TAGS` describes what the layer is, what it covers and where keys come from. It contains no
sentence telling the model which route to prefer — because his scoring says an LLM call on a Tier 1
question is a soft fail **even when the answer is right**, so *which rung the agent picks is the
thing being graded*. "Prefer the cheapest route" in the prompt would not build an agent that chooses
well; it would hard-code the exam answer and delete the result. That line is kept as
`prompts.PREFER_CHEAPEST_ROUTE`, unused by default, as ablation arm C:
```
A  no FACET TAGS section        B  descriptive (default)        C  B + prefer-cheapest
```
B ~ C means tier-appropriate choice emerges. C >> B is the more interesting finding, and either way
it is a result rather than an assumption. Three regression tests pin all of this.

## D-055 (2026-08-07) — `facet_entities`: the inventory shape, and why it is a tool not an ablation
**The question that had no tool.** `facets` answers "which papers use an ABCD-family estimate" — 6.
It could not answer "and what are the six of them", which for a coverage map is usually the question
actually being asked: *how many different ways does this literature do X, and what are they.*
```
ABCD data-driven background estimation method
ABCD (matrix) ... using control regions in data with parametric fits
Data-driven ABCD-style ratio method using eight non-overlapping regions (A-H)
Modified ABCD estimate
Multidimensional ABCD reweighting technique (CR-to-SR, data-driven)
Two-dimensional ABCD sideband method using control regions B, C, D
```
One tag, six genuinely different methods. A tag is a family; this is what the family contains.

**Three counts, reported separately**, for the same reason `count` reports three — they answer
different questions and conflating them is how "how many" goes wrong:
```
tag        papers   entity records   distinct after merging
BJet          38          10                   4
JES           45          21                  13
TTbar         51         159                 150
ABCD           6           6                   6
```
**`distinct` is the column that moves with deduplication.** Paper counts do not — canonical expansion
and search breadth already reach every spelling (D-051, 0.48%). So this is not a nice-to-have: it is
the **question shape the dedup ablation requires**, without which every arm returns the same number
and the result reads as "deduplication does not matter". Hence a tool, built now, rather than an
ablation arm — it is what *enables* the ablation.

**Bug caught in the first run**: `GROUP_CONCAT(DISTINCT x)` cannot take a separator in SQLite, so it
joins on commas — and these labels contain commas. `b-tagged jet (MV2c10, 77% efficiency)` came back
as two labels, one of them ` 77% efficiency)`. Labels are now a second query grouped in Python, the
same two-query pattern `contents_of` and `facets` already use for the same class of reason. A
regression test puts a comma inside a fixture label.

**What the BJet inventory actually shows**, and it is a good coverage-map answer in itself: 38 papers,
4 distinct objects — one dominant cluster of 18 wordings across 34 papers, plus separately-kept
tight- and relaxed-working-point variants. Whether those variants *should* stay separate is a physics
call, and it is exactly the kind of pair sitting in the 111-pair review queue.

## D-056 (2026-08-09) — the reference reader: what it measures, and the six ways it lied first
**What it is**: an independent second opinion, read from the PAPERS. The supervisor's gold is his
own words "graph-agreement gold, not physics truth" — computed by filtering his pilot export — so
scoring our graph against it measures whether two pipelines built from one extraction agree with each
other. Where the extraction dropped something, both drop it and both score 100%. The reader touches
`source_block` and nothing else.
**Ran**: 420 paper-reads, 19,251 calls, 0.2% errors, 97.3% usable, 1h43m on Mistral-Small-24B.

### Six failures, every one silent, every one reporting success
1. **Windows sized from a guess.** `CHUNK_CHARS` assumed 4 chars/token; measured with the served
   tokenizer, this corpus is **3.46**. Half the calls 400'd on context overflow — and systematically,
   because overflow kills the biggest windows, which are the content-rich ones holding the answers.
   Reported confident negatives on answers already located by hand.
2. **`failed: 0` while 55% of calls errored**, because `failed` counted only reads where EVERY sample
   died. API error, unparseable reply and a real "no" were three things flattened into one.
3. **Consensus pooled across windows.** A 23-window paper has the answer in one window; the other 22
   correctly say "not here". Pooling made unanimity impossible EXACTLY when the answer was found, so
   successful reads became "split" and were counted as misses, while papers where nothing was found
   agreed trivially and reported a confident False. **Finding the answer was what made the result
   unusable.** gf-02 reported 0 of 6 gold papers; five sat in the split bucket with verified quotes.
4. **A verified quote is not a correct answer.** `verify_quote` proves a sentence is in the paper, not
   that it answers the question. Measured: **43–45% precision** — "exactly two muons, no identified
   electrons" cited as evidence for an electron-OR-muon requirement.
5. **The support judge was asked a corpus question about one sentence** and refused nearly everything
   (3% precision), including quotes that plainly establish the fact. No sentence can name which of 60
   papers do something.
6. **The prompts asked for the verdict BEFORE the reasoning.** JSON generates left to right, so the
   model committed to yes/no before emitting a token of justification — chain-of-thought backwards.

### The finding underneath: a sweep question must be asked one paper at a time
gf-03 found **0 of 4** HistFitter papers. Same window, same model, temperature 0 — only the phrasing:
```
"Which SUSY searches did their statistics in HistFitter?"
   -> no,  "The text does not mention SUSY searches in HistFitter."
"Does this analysis use the HistFitter framework?"
   -> yes, "implemented in the HistFitter [ 168 ] framework"
```
The prompt already INSTRUCTED "does THIS analysis do it". The model anchored on the question's own
wording anyway. **Instructing around a corpus-wide question does not work; decomposing it does.**
Seven hand-written per-paper forms, recorded beside the originals. gf-03 went **0/4 -> 3/4**.
This same confusion cost three separate runs (the reader, the judge, and gf-05).

### After every fix
```
gf-02 ABCD          4/6 gold      gf-03 HistFitter  3/4 gold     gf-08 e-OR-mu  1/1 gold
gf-05 Higgs object  0/2 gold, and 10 false positives — his trap question, and the reader falls in
```
**The reader is a LITERAL reader**, measured three ways: it will not invert "masses up to 875 GeV are
excluded" into "lower limit" (gf-12, confirmed against three phrasings); it will not answer a
corpus-wide question about one paper; it matches surface topic over structural role. Good
corroborator, poor contradictor — it can support the gold and cannot yet overturn it.

### What it corrected in US
gf-11: I reported "CSVv2 appears in 0 evidence quotes and 0 source blocks". True of the literal
string; the paper spells it **"the combined secondary vertex (Version 2)"**, which the reader found.
A string-matching artefact — the exact failure his Tier 1 is designed to catch — in my own analysis.

### Also established
- gf-05's 0/2 splits in two: 2006.05880 says "A jet pair is tagged as a Higgs boson candidate if the
  neural network score..." — findable, missed, a **model** failure. 2504.13081 has **zero**
  candidate-ish mentions in the readable text — no model can find what is not there.
- gf-14 is answerable and his premise is wrong twice over: the reader cites "extending beyond the
  previous limits ... by up to 160 GeV", a DIFFERENT sentence from the "approximately 300 GeV" one in
  the evidence table.
- Q9 remains unanswerable by any reader: provenance metadata is in no paper.

## D-057 (2026-08-09) — reason-first ordering is free; a thinking model is the open question
**Measured, gf-05, same windows and temperature, only the JSON field order:**
```
paper        truth   ANSWER-FIRST  REASON-FIRST
2302.05225     no        True        False   <- fixed
2602.18611     no        True        False   <- fixed
2605.14245     no        True        False   <- fixed
2006.05880    YES       False        False   <- recall unchanged
```
**3 of 5 false positives fixed for nothing.** Precision is a prompt-ordering property; recall is not.
Kept honest: reasoning that precedes the verdict in TOKEN order is not thereby a faithful account of
the computation (Turpin et al.). This buys accuracy, not interpretability.
**No thinking mode was ever enabled and none exists to enable** — Mistral-Small-24B-Instruct-2501 is
not a reasoning model, and the server ran `enable_reasoning=False, reasoning_parser=None`. Field
order is the version of "thinking" available to a non-reasoning model.
**QwQ-32B-AWQ downloaded (19GB, under 3 min) and served on port 8001 beside the 24B** so both arms
compare without giving up an allocation. Needed three code changes: a per-call token budget (vLLM
counts prompt+completion, so QwQ's 4000 makes an 8k server reject outright), `<think>` stripping, and
taking the LAST balanced JSON object because a chain of thought contains abandoned drafts.
**The comparison did not complete**: the QwQ server died 7 minutes in with
`CUDA error: uncorrectable ECC error encountered` — failing GPU memory, on the third A100 of
compute-gpu-0-1. Slurm still reports the node `mixed` with no drain reason, so a resubmit can land on
the same card. Third hardware fault of this kind after D-040 and D-049: **a resource that looks
healthy until you use it.** Retry submitted; the head-to-head is still open.

### D-057 continued (2026-08-09, late) — QwQ-32B measured: recall recovered, precision lost
Reached a healthy card only after the allocation fight below. Result on gf-05, where the 24B scored
0/2 with 10 false positives:
```
paper        truth   24B(reason-first)   QwQ-32B
2006.05880    YES         False           True    <- the genuine win: evidence WAS present
2504.13081    YES         False           True    <- but the text has ZERO candidate mentions
2302.05225     no         False           True    <- regression
2008.05928     no          True           True
2510.07527     no          True           True
```
**QwQ answers True to everything.** It recovers the one case a stronger reader should recover
(2006.05880 states "A jet pair is tagged as a Higgs boson candidate if the neural network score...",
which the 24B read and rejected) — and then says yes to a paper whose readable text contains **no
Higgs-candidate mention at all**, with a quote that passes verification. That is the
misreading-not-invention failure amplified, not fixed: reasoning lets it argue from weaker evidence.

**So the two models fail in opposite directions**: the 24B is literal and under-finds; QwQ reasons
and over-finds. Neither is usable alone, and the support judge becomes more necessary with QwQ, not
less. The sensible configuration to test next is QwQ for RECALL feeding the judge for PRECISION —
which is the retrieve-then-verify shape the whole harness already uses, one level up.

**gf-12 refuses on both.** Neither model converts "masses up to 875 GeV are excluded" into a "lower
limit". Not a capacity limit — a defensible reading, and the finding to put to the supervisor.

**The allocation fight, worth recording because it will recur.** GPU 2 of compute-gpu-0-1
(`GPU-bd028739`) carries 569 volatile / 1413 aggregate uncorrected ECC errors, loads a model happily,
and dies on the first inference. Slurm reports the node `mixed` with no drain reason and assigned that
card on **six consecutive** single-GPU requests — the allocator is deterministic, so retrying never
escapes it. Fixed by requesting TWO GPUs and pinning to the clean one. Two further bugs surfaced in
the process, both fatal before any model loaded: `nvidia-smi | awk '{print; exit}'` killed the pipe
and, under `set -o pipefail`, the script (exit 13, 00:00:00 elapsed); and `nvidia-smi` reports **all
three** GPUs regardless of the allocation, so "pick the first healthy index" would have pointed
CUDA_VISIBLE_DEVICES at a card another user's job owned. The choice is now made only from Slurm's own
CUDA_VISIBLE_DEVICES — narrow the allocation, never widen it. **Report the card to the sysadmins**;
it will keep eating jobs and the failure always presents as the user's bug.

## D-058
### 2026-08-11/12 — The conjunction problem: one fault wearing five costumes

Most of Gabriel's questions ask for **several things at once**. gf-01 wants a *search* AND
*b-tagged jets* AND *missing transverse momentum*. gf-11 wants a b-tagging *algorithm* AND its
*working point* AND its *performance*. The harness was built around single-fact existence questions
and mishandled conjunctions at **five separate places**, two of which presented as *passes* rather
than failures — which is why it took this long to see that they were the same fault.

**Form 1 — the reader sees one window at a time.** No single 12,000-char window states that a paper
is a search *and* uses b-jets *and* uses MET. Asked the whole conjunction per window, the reader says
no everywhere. gf-01 scored **2/18**.
*Fix:* ask one condition at a time, compute the AND ourselves over the whole paper.

**Form 2 — the judge is handed one sentence and the whole question.** With the conditions confirmed
separately, `Consensus.best_quote` flattened them to the FIRST quote on the way to the support judge,
which then rejected papers for not proving in one sentence what three sentences had established. It
narrated itself doing it: *"The sentence mentions a search (not a measurement) and explicitly includes
missing transverse momentum..."* → downgraded. **Four papers with all three conditions confirmed were
thrown away this way.**
*Fix:* the judge sees every distinct verified quote, condition-labelled. Precision **38% → 62%**,
gf-01 **4/18 → 6/18**. Measured on the sweep, 16 downgraded reads had other verified quotes the judge
never saw — so this was silently costing us everywhere, not just on gf-01.

**Form 3 — the extractor stops at the first hit.** `sweep_windows` returned as soon as one sample was
supported. Sound for existence ("does this paper do X" — one confirmation settles it), **unsound for
extraction**: the parts of an answer sit in different sections by construction. gf-11 returned the
b-tagging algorithm and never looked for the working point or the performance — both of which were in
the **same window it had just read**.
*Fix:* `mode == EXTRACTION` keeps reading every window.

**Form 4 — a third of an answer scores as a pass.** gf-11 was recorded True on the algorithm alone.
This is the dangerous form: it inflates the score *and* hides the defect, and it is why the
single-paper questions looked healthier than they were. gf-06 is probably the same shape.
*Fix:* still open — scoring a partial extraction needs a per-part gold, not a boolean.

**Form 5 — telling the recall stage to care about completeness makes it hand back nothing.** The
extraction prompt said *"a partial answer that looks complete is worse than one that says which parts
are missing"*. Reasonable-sounding, and catastrophic: gf-11 and gf-14 went from a verified quote each
to **zero found across 32 and 54 calls**. A model asked to judge sufficiency withholds the fragments
it judges insufficient.
*Fix:* the gather prompt now says the opposite — *"Your job here is to COLLECT EVIDENCE, not to decide
whether the question is fully answered... A later step decides whether the parts add up, and it can
only do that with what you hand it."*

**The generalisation.** Every one of these is the same mistake: **deciding sufficiency at a stage that
cannot see all the evidence.** The window can't, the single-quote judge can't, the early-stopping
sweep can't, the recall model shouldn't. The architecture that follows is the one the harness already
uses one level up — **gather wide, decide once, at the only point where everything is visible.**

**Consequence for the single-paper questions.** gf-06 and gf-10..gf-15 had been running one model,
one pass, no judge, while every sweep question got 24B → QwQ → judge. That asymmetry was never a
decision — it is an accident of the order things were built in, and it is the best available
explanation for why those seven have been the flakiest results in the set. They now run the same
cascade: both models gather every window with `--no-cascade`, the results are **unioned** (recall is a
union, not a vote — the two models fail in opposite directions per D-057, so requiring agreement
discards exactly what the second model was added to find), and QwQ judges the whole union at once.
`MAX_MERGED_JUDGE_QUOTES = 16` against 4 for the sweep, because a union assembled by two models over
every window is a different object from a handful of verified quotes.

### D-058 addendum — both servers from one three-GPU allocation
Two one-GPU jobs **cannot** both draw a healthy card on compute-gpu-0-1. Confirmed today: the bad
card of D-040 and the ECC card of D-057 are **the same GPU** — bus `00000000:CA:00.0`, index 2,
**1413 aggregate uncorrected ECC errors**. With three cards and one dead, any job holding a spare
forces the next job onto it; job 48365 was submitted alongside a two-GPU QwQ job, drew CA:00.0, and
refused to start — correctly and uselessly. `hpc/serve_both.sh` takes all three and serves the 24B and
QwQ on the two clean ones, holding the dead card unused. That costs nobody anything: no job can
compute on it anyway. The health gate now reads the **aggregate** ECC counter rather than a bus-id
blocklist — volatile counters reset on driver reload, so the card that killed two jobs yesterday reads
clean today on the volatile column, and a blocklist only knows about failures that already happened.

## D-059
### 2026-08-12 — Two parser faults, three bad comparisons, and what the numbers actually say

**The faults.** `parse_reply` discarded 20% of the single-paper run's calls and 2.5% of the
sweep's. 124 of 149, and 475 of 483, were **complete, well-formed answers**.

1. The JSON extractor took the last **brace-free** object, `\{[^{}]*\}`, on the theory that an
   object with no nesting is the whole object. True of the object, false of its contents: a quote
   carrying `${\approx}4.8$` contains `{\approx}`, which is brace-free, comes last, and is not JSON.
   The greedy fallback beneath it never ran — it was guarded on finding *no* match rather than on
   failing to parse the one it found.
2. The 24B copies LaTeX into JSON strings verbatim, `"$\mathup{{{t}}}$"`, and `\m` is not a JSON
   escape. Repaired by doubling only the backslashes that do not begin a real escape, valid pairs
   consumed first so the pass is idempotent, and applied only after strict parsing fails.

What remains is genuine truncation: 20 of 366 on the 24B, where 500 completion tokens no longer fit
a reply asked for *every* relevant sentence. A truncated reply is still rejected — recovery must not
become invention.

**The loss was one-directional, and that is the mechanism, not a coincidence.**
```
sweep, recovered from storage:   YES: 475     NO: 0
```
A "no" reply carries no quote, so there is no LaTeX to choke on and it always parsed. Only a "yes"
quotes the paper. The bug could therefore only ever delete evidence, never invent it — pure recall
loss, reported downstream as "the paper does not say".

**Three comparisons I got wrong before getting one right.** Worth recording in full, because the
error was mine each time and the same shape each time: changing more than one thing and then
measuring.

- Compared the reparsed sweep against the **rechecked** baseline. That baseline has an extra QwQ
  escalation stage the reparsed run never had. Different pipelines, not different parsers.
- Compared against the right stage, but the two files were **judged by different models** — the 24B
  on 08-09, QwQ now. On 2004.04545 the evidence string was byte-identical and the verdicts differ:
  the 24B said *"mentions the use of b-tagged jets, which is part of the event selection"* and
  passed it; QwQ said *"explicitly mentions b-tagged jets ... but it does not mention missing
  transverse momentum"* and failed it. **QwQ is right.** The apparent regression was a better judge.
- Re-ran gf-01 at `repeats=3` where its baseline used `2`, then read the drop as a result.

**`repeats` is not a free knob.** Consensus requires the repeats *within a window* to agree, so a
third sample can only ever make a yes harder to reach. Raising it silently tightens the threshold.
That belongs in the writeup as a property of the measurement, not a default buried in a job script.

**The controlled numbers.** Same stage, same judge, same settings, parser the only difference:

| | sweep (60 papers) | gf-01 conditions | single-paper |
|---|---|---|---|
| before | 15 gold | 6/18 | gf-10, gf-11: **0 evidence in 60 calls each** |
| after | **16 gold**, +12 false positives | **8/18, 0 false positives** | gf-11 fully correct |

So the fix is close to score-neutral on the corpus sweep and *worsens precision there* — gf-08 went
from 7 found to 16, every extra one wrong. It is decisive on the two places where the answer is
LaTeX-dense: gf-01's conjunction, where a lost YES on any single condition fails the whole paper,
and the single-paper extraction questions.

**The conclusion that matters: recall is no longer the bottleneck. The judge is.** Every question
except gf-02 gained candidates and converted none of them. gf-05 stands at 1 hit against 18 false
positives. The next measurement worth making is not "can the reader find it" but "can anything
separate a probative sentence from a merely relevant one" — which is exactly what Gabriel's hand
labels would give a ground truth for.

### D-059 addendum — a fix that never ran, and a test that could not see it
The value judge (`check_extraction`, judging an extraction question on its VALUE rather than its
topic) was written, tested, deployed, and **never executed on a single real row**. The CLI trimmed
each question to `{qid, text}` before handing it to the judge; `verify_supports` routed on
`provenance.paper_scope`; with provenance stripped that set was always empty. Its unit test passed
throughout, because the test called `verify_supports` directly with full supervisor records and
never went through the shape the pipeline actually uses. **A test that builds its own input cannot
catch a caller that builds a different one.** The trimming now lives in `reader.judge_records`,
beside the code that depends on it, and the regression test asserts on the list the CLI really
passes.

### D-059 addendum — a server job that did not hold its GPUs
`wait -n` is unsupported by the bash on these nodes; under `set -e` that took `serve_both.sh` down
**31 seconds after launch** while both vLLM processes carried on serving as orphans. Slurm marked
the job FAILED and reported compute-gpu-0-1 **idle with all three A100s free** — so it would have
handed those cards to the next user, whose job would have met 75GB of our model on each. Nothing
downstream noticed, because the servers kept answering; that is exactly why it ran unseen for two
hours. Found only because the *replacement* server job vanished from the queue. Fixed with a
portable supervisor loop plus a pre-flight reap of our own stale processes. Related: `chmod +x` is
itself a tracked change and was rejecting every `git pull` on the cluster; `core.fileMode` is now
off there.

---

### D-060 (2026-08-14) — the query critic: flag the candidates, and let it widen the search
The relevance step S-69 asked for, now specified. It sits between retrieval and use: `search`
returns candidates, the critic marks which bear on **this** question, and the planner proceeds. It
is the query critic, not Sunny's extraction panel ([[multi-agent-extension]]) — a different object
at a different stage.

**Why it is needed, in one path.** `search("Pythia")` takes the top `SEARCH_BREADTH` hits and saves
*all* of them as `set_1`; `count(set_1)` then treats every one as part of the concept. Nothing
between retrieval and use asks whether a hit belongs. That is the Tier B 0.058 failure — 60
near-neighbours counted as one thing — and it cannot be fixed with a score cutoff, because the
cutoff that saves `Pythia` (rank 20 at 50% of the best hit, still a real Pythia) destroys
`top squark` (rank 20 at 71%, already "single top"). Only something that reads the question can
separate those.

**Decided:**

1. **Flag, never filter, in stage 1.** A wrong discard is silent — nothing in the trace shows what
   was removed (D-016, D-018). `set_N` keeps every candidate; a second handle `set_N_kept` holds the
   survivors. Filtering is a later switch, turned on only once flags agree with what answers use,
   and it doubles as the ablation arm.
2. **Three rungs, not a binary.** `exact` / `broader` / `unrelated`. Binary cannot tell the two
   failures apart that S-69 requires both of: Tier B errs **too coarse**, the supervisor's Q5 errs
   **too fine**. Collapsing to binary for scoring stays available; recovering the direction later
   does not.
3. **Same model as the planner** for now. Different-family (S-37) is an ablation, not a
   prerequisite; a second endpoint costs GPUs we do not have spare.
4. **Ranked order, chunked — not shuffled.** Shuffling makes position bias measurable but
   *relocates* the lost-in-the-middle risk onto the best candidate, which is a real cost paid for
   measurement convenience. Chunks of ~15 are the actual mitigation: lost-in-the-middle is a
   long-context effect, and each chunk is judged in its own call, so a rank-50 candidate is assessed
   against the question rather than against rank 1 at the top of the same prompt.
5. **Bias is measured, not assumed away** — three arms on a sample: shuffle A vs shuffle B (pure
   position bias), ranked vs shuffled (how far the critic is merely restating the retriever), chunk
   15 vs 60 (whether chunking earns its calls). The middle arm is the one that matters: if verdicts
   barely move, we built an expensive restatement of BM25.
6. **Both controls, in both directions.** *All-keep*: a search narrow enough that every hit belongs
   — if the critic can never say "keep them all" it is hedging, not judging (the AgentRivet lesson,
   [[multi-agent-extension]]). *All-drop*: a chunk where nothing belongs — ranked order puts the
   weakest candidates together in the last chunk, so a model that feels obliged to keep something
   per chunk inflates every count downstream by roughly one item per chunk. Both gate everything
   after them.
7. **A verdict moves a cluster, not an entity.** `concept` expands the seed hits through
   `expand_canonical`, so the set handed to `count` is larger than the hit list and contains ids the
   critic never saw. `set_N_kept` is therefore the *expansion of the kept seeds*. Dropping one seed
   drops its cluster-mates with it — correct, since deduplication already ruled them the same thing,
   but it makes each verdict weigh more than one row and the trace must say so.
8. **Breadth becomes a stopping rule, not a constant.** `SEARCH_BREADTH = 60` is admitted in its own
   comment to be a guess, and it exists only because everything retrieved gets used. Once a
   relevance step exists, over-retrieval costs one line of prompt instead of a wrong count. So:
   retrieve 60; if the list came back **exactly full** *and* the keep-rate in the **last** chunk is
   still high, fetch the next 60. The trigger is relevance **at the tail**, not the overall keep
   count — a search where all 60 are relevant is the most truncated case there is, and a rule that
   widens "when not all were relevant" would stop there and widen on `top squark` instead, which is
   junk all the way down.
9. **Flagging may be strict; widening must be generous.** A wrongly-flagged candidate is still in
   `set_N` and still in the trace. A candidate never retrieved is gone, and nothing downstream can
   recover it. Different steps, different failure costs. The stop decision is logged with its tail
   keep-rate so a run can be audited for "stopped because the tail was junk" versus "stopped because
   the critic was harsh".

**How it is scored** — the three tiers map onto the experiment cleanly. **Tier B is the treatment**
(concept counting runs `search` → set → count; if the critic does not move Tier B it does not work).
**Tier A is the control** — per-paper questions go through `contents_of`, so the critic has no
business changing them, and a Tier A move is a bug signal. **Retrieval is the mechanism**:
precision@k on the same hit lists, which is also where the named-vs-described gap lives (0.893 vs
0.335). The supervisor's Tier 1 is a *direction* check only, not a scoreboard — its gold is
graph-agreement gold, so a critic that correctly drops what his pipeline kept loses points for being
right.

**Sufficiency is a second critic, and it lives beside `verify.py`, not inside it.** Different
question: `verify` asks "did every number come from a retrieved row", which is mechanical because
97% of assertions carry their verbatim sentence, and that mechanical-ness is its entire claim.
"Is this evidence *enough*" admits no such rule. Two checks at the `finish` seam reporting
separately keeps both claims; folding an LLM into `verify` would forfeit the strong one. It catches
what `verify` structurally cannot — answering from 3 rows when 60 existed, reasoning over the 25
rows `_render_rows` showed as though they were all 200, and cheap `not_in_graph` abstention after a
single phrasing. Flag-only, and measured on the abstention tier rather than Tier B. Built **after**
the candidate critic, which has a measured failure attached to it.

### D-060 addendum — the ceiling already binds, measured before any code
Scanned every `search` step in `eval/runs/` for hit lists that came back exactly full, which is an
exact truncation signal needing no LLM:

| question set | searches | hit the 60 ceiling |
|---|---|---|
| conceptB (Tier B) | 458 | **297 — 64.8%** |
| conceptB reworded | 470 | 288 — 61.3% |
| retrieval | 788 | 429 — 54.4% |
| retrieval reworded | 524 | 106 — 20.2% |
| paperA (Tier A) | 218 | 1 — 0.5% |

**Two thirds of Tier B searches are truncated**, so the ceiling is not a footnote and the stopping
rule in decision 8 is load-bearing rather than a refinement. `jet energy scale` has 109 clusters
against a limit of 60, and the trace of that loss looks like a clean successful search.

**Tier A confirms itself as the control** at 0.5% — and surfaces an unrelated bug: **206 of its 218
searches returned zero rows, and the search text was an arXiv id** (`search("2604.27044")`). The
planner is searching for papers, which `contents_of` explicitly tells it not to do, and getting
nothing 206 times. Independent of the critic, and cheap to fix.

**The baselines cannot be reused.** Every run above is `6945b02-dirty`, and the one commit to touch
`query/` since is `7ad9cd6` — the facet layer and both facet tools. So the numbers predate the
rungs the critic is supposed to pick between, and `-dirty` means the sha does not identify the code
that produced them anyway. Re-baseline on today's code before the critic exists, or this repeats
D-059's pattern of changing several things and then measuring.

### D-060 addendum — two sites, and the second one never filters
Judging `search` alone covers half of what S-69 asks for. The two failures are
symmetric: **too fine** is `search` drowning in string matches (the supervisor's Q5),
**too coarse** is a facet key standing in for a question that needed the variant. Only the
first is guarded by a critic on `search`.

*The facets site is a LABEL READER, not a filter.* Six papers carry
`background_methods = ABCD` and describe six different methods -- Modified; ABCD data-driven;
ABCD-style ratio over eight regions (A-H); two-dimensional sideband over CRs B, C, D; ABCD
(matrix); multidimensional reweighting. *"How many use a data-driven estimate?"* is **6**.
*"How many use the standard four-region ABCD?"* is **not 6**. Both answers come from those same
six rows, and the difference is entirely in reading the labels against the question -- which the
tool's own note asks the planner to do and then leaves to hope.

**Nothing is removed there, unlike at `search`.** At `search`, `unrelated` means the candidate is
not the thing (Herwig in a Pythia set) and dropping it loses nothing. At `facets` every match is
genuine: the tag is right and only the variant differs -- and *for a coverage map the variants are
the finding*. "There are at least five distinct ways this literature does ABCD" is the kind of
answer this project exists to produce, and narrowing to the two that match as asked would destroy
it. So a `broader` verdict is a **lead, not a demotion**: it names how the paper differs, which is
directly usable as the next query, and the summary says to follow it with `facet_entities` or
`contents_of`. (The user's correction; the original design had it filtering.)

*A retraction*: the claim that this catches the Tier B 0.058 is **not established**. That was
measured on the `search` path before facets existed. What S-69 warns about is a critic that
*chooses* the facet rung when the question needed a finer one -- a rung-**selection** failure. The
remedy is the same either way: notice the facet level does not answer the question, and say
*go finer*, with the labels as the map.

### D-060 addendum — relevance is not only about matching, and mostly it is not the judge's job
Prompted by the question "why only judge `search`?". **40% of all tool results are truncated at 25
rows** -- `subjects_of` 87% (139,878 rows hidden), `crosstab` 95%, `search` 71% -- and the 25 shown
are the first 25 in SQL order, which has nothing to do with the question.

But an LLM judge is the right instrument for only one of three situations:

| situation | example | instrument |
|---|---|---|
| the question is "how many" | 160 analyses that all use Pythia | `count`, in SQL. Nothing to judge |
| more relevant rows than fit | 187 rows from `contents_of` | rank the truncation, or filter by predicate |
| membership is genuinely ambiguous | is this Pythia? is this ABCD? | **the critic** |

*Traced case, which decided it.* **"How many analyses estimate the t̄t+γ background?"** -- true
answer **1**. `search` returned 60 backgrounds (γ+jets, Multiboson, Wt, Z+X, Z+jets, W+jets, tWZ,
ttW, tZq...), expansion gave 78 ids, `subjects_of` returned **142 rows**, 25 were shown. **141 were
wrong**, and they were wrong because the SET was wrong -- the hop faithfully returned the analyses
that estimate those other backgrounds. Filter the set and the hop returns one row; truncation stops
existing.

**And judging those rows was impossible anyway**: not one of the 142 named the background it was
about. `subjects_of` returned subject, kind and predicate, never the matched object. So a judge --
or a BM25 re-ranking, which was the plan -- would have been reading evidence that did not contain
the answer. Fixed: the hop now returns `matched`, as `facets` already did. The tally reads
`10 $t\bar{t}$ · 10 Diboson · 8 Z/γ*+jets`, which makes the failure legible instead of invisible.

*Consequence*: relevance-ordered truncation drops down the list -- it cannot rank rows that carry no
discriminating text -- and **filtering the set is confirmed as the root fix**, with a worked case
that turns 142 rows into 1.

### D-061 (2026-08-14) — the corpus does not saturate, so the search cap does not survive
Measured on the pilot, subsampling 5/10/20/30/40/50/60 papers (five random subsets each) -- the
growth-curve item S-47 B, partly answered:

| papers | entities | new per paper |
|---|---|---|
| 5 | 459 | |
| 10 | 983 | 104.7 |
| 20 | 1,791 | 80.8 |
| 30 | 2,639 | 84.8 |
| 40 | 3,529 | 88.9 |
| 60 | 5,114 | 87.1 |

**No saturation.** The marginal rate at 60 papers is the rate at 10. The reason is that the
vocabulary is paper-local: **only 1-22% of entities appear in more than one paper** -- `result` 0%,
`event_region` 1%, `observable` 2%, `systematic_uncertainty` 8%, `generator` 22%. Extrapolated to
the 2,969-paper corpus: **~255,000 entities**, fifty times today.

What grows is not new physics but **new spellings of the same physics**. 78% of generator entities
are paper-local wordings of a handful of generators.

Three consequences, only one of which is a real problem:

1. *The widening RULE survives* -- "widen while the tail is still relevant" is a stopping condition,
   not a number, and does not care about corpus size.
2. *The CAP does not.* `MAX_SEARCH_BREADTH = 240` is a rounding error against a concept with
   thousands of spellings. It is env-overridable and must never be read as a considered value at
   scale.
3. *The cost model breaks, and that is the real issue.* Judging 5,000 candidates at chunk 15 is
   **333 model calls for one search**. But scanning is the wrong algorithm: the goal is not to label
   every candidate, it is to find **where relevance dies**, which is a boundary. Bisect it -- probe
   at 240, 1,000, 4,000 and narrow -- for `log(n)` calls instead of `n/15`. Exhaustive scanning is
   only right while the whole list fits in a handful of calls, which at 60 papers it does.
   *Caveat*: bisection assumes relevance falls off monotonically with rank. It roughly does -- that
   is what ranking means -- so a boundary should be confirmed with a chunk either side rather than
   trusted from one probe.

**Deliberately not built now.** The corpus is 60 papers, the cap does not bind at that size, and
building for a corpus we do not have is guessing dressed as rigour. The ceiling counter (64.8% of
Tier B searches came back exactly full) is what will say *when* it starts to hurt.

**The conclusion that matters is not about search at all.** If 1,500 entities are mostly rewrites of
the same few generators, they should have been merged before search ever saw them. Merging today is
**7%** (5,114 -> 4,772). At 60 papers that is a quality issue; at 3,000 it decides whether a concept
search is 30 clusters or 1,500. *Deduplication quality becomes the binding constraint on everything
above it* -- a far stronger argument for the aliases work than tidiness, and it raises the stakes on
the discovered-grouping rung ([[discovered-grouping-layer]]) too.

### D-060 addendum — a confound caught in the queue, not in the results
The critic arm was queued behind a baseline that had already started, and the code moved between
them. `subjects_of` gained its `matched` column in the interval -- and that is not a logging change:
the extra field goes into the rows the planner READS, lengthens them against the 400-character
per-row truncation, and hands the model information the control never had.

So control and treatment would have differed by **two** things, and the write-up would have
attributed all of it to the critic. That is D-059's pattern for the fifth time, and the only reason
it was caught is that the arm was still `PENDING`.

*Fixed by freezing the code and re-running the control on it*: `control-v2` (48433-5) then the critic
arm (48436-8), both from the same commit, differing by one flag. The morning's baseline (48426-8)
keeps its own value as the measurement of the paper-id fix, and is **not** the control for the
critic.

*The rule, since this keeps recurring*: a run that has already started has frozen its code; anything
committed afterwards makes it a different system. Queue depth is not the same as comparability.

### D-060 addendum — the serving-side tool parser fails constantly, and we were silently absorbing it
The live server logged **1,327 `hermes_tool_parser` errors in 90 minutes** while returning 200 OK
throughout. vLLM fails to extract the tool call, the model's call arrives in the message body
instead, and `planner._recover_tool_calls` picks it out of the text. Nothing downstream ever noticed,
which is the point: the recovery was written for exactly this and then never measured.

`recovered_calls` existed on the Session and was **dropped by `from_session`**, so no run file has
ever carried it. Now it does. The planner's own docstring called this "a count worth watching: if it
is high, the serving-side tool parser is underperforming" -- and it is high.

Worth reading once the arms land, because it bears on a claim the project makes: if a meaningful
share of tool calls only survive because of a regex in our loop, then "the model uses the tools
correctly" and "vLLM's `hermes` parser handles Qwen2.5's format" are two different statements, and
only the first is ours to make.

### D-060 addendum — the paper-id fix, measured on the same 540 Tier A questions
Like-for-like: the same question ids in the 2026-08-03 run and today's, so no sampling difference.

| | 2026-08-03 | today | delta |
|---|---|---|---|
| dead-end sessions | **79** | **0** | -79 |
| abstained | 14.1% | 2.0% | **-12.0** |
| count correct | 53.1% | 71.1% | **+18.0** |
| faithfulness | 77.7% | 88.8% | +11.1 |
| seconds/question | 13.1 | 12.5 | -0.6 |

**The 79 that dead-ended before**: abstention **93.7% -> 2.5%**, count correct **5.1% -> 70.9%**. They
now perform like the general population (71.1%), which is the strongest form the result could take --
not "somewhat better", but *indistinguishable from questions that never had the bug*.

*What is attributable and what is not.* The 79 -> 0 and the 12-point abstention drop are the fix, by
mechanism: those sessions failed at a specific step that no longer exists. Of the +18 on count
correct, the 79 recovering from 5.1% to 70.9% accounts for **about +9.6**; the remaining ~8 points
are **unattributed**. Saying "the paper fix is worth 18 points" would be the same overclaim this
project keeps having to retract. *(Corrected later the same day: the first version of this entry
blamed the facet layer for those 8 points. The facet layer was not running on the cluster at all --
see the D-061 addendum. Both runs used the same pre-facets graph.)*

*Wall-clock did not move* (13.1 -> 12.5 s/question) even though far more work is now done per
question -- because a dead-end session was CHEAP. It gave up after two calls. The bug was fast and
wrong, which is exactly why nothing flagged it.

### D-061 addendum (2026-08-14) — the cluster had been running a different graph, and could not open ours
The gate's all-keep control drew 3 candidates from a canonical cluster that has 12 members locally.
Chasing that found the cause: **`data/processed/hepkg.db` is gitignored (`*.db`), so it has never been
synced**, and the two machines had drifted:

| | laptop | cluster (before today) |
|---|---|---|
| `entity` | 5,114 | 5,114 |
| `assertion` | 14,188 | 14,188 |
| `entity_canonical` | 663 | **597** |
| `entity_facet` | 4,594 | **table does not exist** |
| `assertion_signature_derived` | 734 | **table does not exist** |

**Every evaluation run this project has ever done -- August's and today's -- ran without the facet
layer.** The code for it has been on the cluster since `7ad9cd6`; the data never was. `facets` and
`facet_entities` would have raised `no such table` and been reported to the planner as tool errors,
which is exactly the kind of failure that looks like a model choosing not to use a tool.

*And copying the database did not work either*: the cluster's SQLite is **3.36.0**, `STRICT` tables
need **3.37.0**, and the two facet-era tables are the only ones that use `STRICT`. So the newer graph
was unopenable there -- `malformed database schema (assertion_signature_derived)`. Fixed by
rebuilding those two tables without the modifier into a compatibility copy (schema otherwise
byte-identical, indexes recreated, row counts verified) and syncing that. `facets(ABCD)` now returns
its 6 papers with labels on the cluster, for the first time.

**A correction to the paper-id measurement.** That entry said the unattributed ~8 points "also
contains the facet layer". It does not -- the facet layer was not running on either side. Both runs
used the same pre-facets graph, so that comparison is *cleaner* than claimed, and the ~8 points are
attributable to the interval's other changes (the removed double retrieval per search) or to
run-to-run variation. The original wording overclaimed a confound that did not exist, which is the
same failure as claiming a fix that did not run -- just in the opposite direction.

*Standing consequence*: the database is an input to every measurement and is not under version
control. Its identity has to be checked, not assumed -- row counts for `entity`, `entity_canonical`
and `entity_facet` are enough to tell two builds apart, and belong in the run metadata.

---

### D-062 (2026-08-15) — the query critic, measured: it does not pay for itself yet
2,508 questions per arm, nine shards each, same commit, same graph, one flag apart, aligned on all
2,508 shared `(qid, repeat)` pairs. The critic ran properly: **1,074 searches judged, 50,496
candidates, 3,599 calls, zero critic errors**.

| metric | control | critic | delta |
|---|---|---|---|
| count correct | 0.486 | 0.511 | **+0.025** |
| set recall | 0.917 | 0.734 | **-0.183** |
| set F1 | 0.137 | 0.111 | -0.026 |
| abstained | 0.018 | 0.062 | +0.044 |
| errored | **0** | **0.046** | +0.046 |
| seconds/question | 27.0 | 60.1 | **x2.2** |
| entities retrieved | 58 | 41.8 | -16.2 |
| evidence quotes | 58.3 | 25.8 | -32.5 |

**The honest reading: a 2.5-point gain on the headline metric, bought with 2.2x the wall-clock, 18
points of set recall, and a 4.6% error rate that was zero.** And the 2.5 points have **no error
bar** -- `repeats=1`, so `compare` cannot print a noise floor, and S-52 says three repeats before
comparing anything. It is not yet distinguishable from run-to-run variation.

*The 116 errors are all one thing*: `TimeoutError: no answer within 180s`. Not a logic fault --
the critic simply makes sessions slow enough that 4.6% hit the runner's wall.

*It is very aggressive*: **38,045 of 50,496 candidates marked `unrelated` (75%)**, against 6,305
`exact` and 6,146 `broader`.

#### Three mechanism findings, each fixable

**1. The widen loop never fired -- 0 widened searches -- and I caused it.** `should_widen` requires
a tail keep-rate >= 0.5, and a critic marking 75% unrelated almost never has one. D-060 decision 9
says in as many words that *flagging may be strict but widening must be generous*; I then fed both
from the same verdicts. So the ceiling measurement that motivated the whole stopping rule (64.8% of
Tier B searches truncated) went untested. The widen signal needs its own, looser threshold -- or to
count `exact` at the tail rather than `kept`.

**2. The `chunk60` arm is not a measurement of chunk size.** 2,087 of its 2,099 verdicts
**defaulted**: sixty verdicts do not fit in `MAX_COMPLETION_TOKENS = 800`, so the reply truncated
and the parser found almost nothing. "Chunking earns its calls" is therefore **unmeasured**.
*But it is an excellent live demonstration of the defaulting asymmetry*: 2,087 unparseable verdicts
became **keeps, not silent drops**, and the arm reports `defaulted=2087` rather than a suspiciously
clean 100% keep rate. The guard did exactly its job on a case I did not anticipate.

**3. Position bias is real and large.** Two shuffles of the same candidates, same model,
temperature 0: **21.7% of rungs and 16.6% of keep/drop decisions flip**. One decision in six depends
on where the candidate sat in the prompt.

#### And a result that contradicts the design decision behind it
`ranked vs shuffled` moves 19.5% of keep decisions against a shuffle-vs-shuffle floor of 16.6% --
so only about **3 points** is the ranking signal. The critic is mostly *not* restating BM25; it is
mostly noisy.

Worse for the chosen default, keep-rate by TRUE retrieval rank:

```
ranked    [0.422, 0.393, 0.350, 0.273]     gentle slope
shuffleA  [0.529, 0.357, 0.276, 0.243]     steeper
shuffleB  [0.511, 0.319, 0.243, 0.210]     steeper
```

**The shuffled arms discriminate by true rank BETTER than the ranked arm.** Plausible mechanism: in
rank order each chunk is homogeneous -- chunk 1 all plausible, chunk 4 all marginal -- so the model
calibrates within the chunk; a shuffled chunk contains a mix and can be compared against itself.

That is evidence *against* D-060 decision 4 (ranked by default), which was chosen to protect the
best candidate from the lost-in-the-middle effect. The protection appears to cost more than it
saves. **Not yet acted on** -- it is one measurement, on 37 searches, and it should be repeated with
`repeats>=3` before a default changes on it.

#### The gate is marginal, not passed
In-job it read **4/30 (13.3%)** on the `Pythia question / jet-energy-scale search` control, against
0/30 and 2/30 in two standalone runs an hour earlier. Same prompt, same model, temperature 0 -- the
spread is vLLM's batching non-determinism. A 10% threshold is inside the run-to-run noise, so the
gate needs either a wider band or repeats of its own. The failing verdicts are the old shape:
*"This is a systematic uncertainty related to jet energy scale"* -- correct reasoning, wrong label --
now at 13% instead of 100%.

#### What this means for the arm
Not "the critic does not work". It works, it is measurably aggressive, and every cost above has a
named mechanism. What it does not yet have is a reason to be switched on: **before running it again,
fix the widen signal, re-measure with repeats>=3, and settle ranked-vs-shuffled on evidence rather
than on the argument that lost the moment it was tested.**

### D-062 addendum — the gate's run-to-run variance was the PROMPT, not just the server
Gate v2, three repeats per control, on the sharpened prompt:

```
PASS  all-keep      100.0% +/- 0.0%  of 12   (pileup cluster)
PASS  all-keep      100.0% +/- 0.0%  of  8   (pp-13TeV cluster)
PASS  all-drop        0.0% +/- 0.0%  of 30
PASS  all-drop        6.7% +/- 0.0%  of 30
PASS  mixed-family   70.0% +/- 0.0%  of 30
```

**Zero spread on every control.** The same all-drop case read 0/30, 2/30 and 4/30 within one hour the
day before.

Earlier this was written off as "vLLM batching non-determinism, so the gate was reading noise". That
is the *mechanism* but not the *cause*, and the distinction matters. Batching non-determinism is
always present at temperature 0; it only shows up in the output when the model is **near a decision
boundary**. The readings that wobbled came from the prompt whose rungs were defined against the
search rather than the question -- exactly the version that wrote *"Pythia is a generator, not an
analysis applying jet energy scale uncertainty"* and then labelled it `broader`. A model on the fence
flips under reordering; a model with an unambiguous rule does not.

So **prompt ambiguity presents as run-to-run variance**, and the variance is a usable signal about
the prompt rather than a fact of the infrastructure to be tolerated. Worth carrying into the reader
work too: the per-window splits there ([[ground-truth]]) may be measuring prompt boundaries as much
as genuine ambiguity in the papers.

*Repeats stay*, and the straddle check with them -- the point is not that the noise was fake, it is
that it was diagnostic. Removing the measurement would remove the diagnosis.

### D-062 addendum — pooling hid the result: Tier B splits into two shapes that move OPPOSITE ways
The critic's effect is not "+2.5 points overall". Broken out:

| tier / shape | n | metric | control | critic | delta |
|---|---|---|---|---|---|
| **B / count** | 327 | count correct | **0.058** | **0.144** | **+0.086** |
| **B / set** | 109 | set recall | 0.917 | 0.734 | -0.183 |
| | | set precision | 0.077 | 0.062 | -0.014 |
| | | set F1 | 0.137 | 0.111 | -0.026 |
| **A / per-paper** | 757 | count correct | -- | -- | **-0.001** |

**0.058 is *the* number** -- the figure recorded in S-68 as the Tier B failure, "answering too coarse,
60 near-neighbours counted as one thing". The critic was built for exactly that and **more than
doubles it**. Pooling with 757 Tier A questions that correctly did not move diluted a +8.6-point
effect into +2.5.

**Tier A moved by -0.001 of 757 questions.** The control tier behaved as a control, which is the
result that makes the Tier B movement readable at all.

*Why the shapes diverge, and it is mechanical rather than empirical*: a **count** is a PROPERTY of the
set, so removing junk makes the number right. A **"which analyses"** answer IS the set, so removing
candidates removes correct members. Filtering trades recall for precision -- a gain where precision
is the problem, a loss where recall is the answer. **Consequence for the design: the filtered handle
belongs on counting questions and the full set on listing questions**, which the planner can choose
per question rather than per run.

*One thing that does not fit, and is not being smoothed over*: set **precision** also fell, and
filtering should raise it. The likely cause is in the same table -- set-question abstention jumped
**0.018 -> 0.193**, and an abstained question scores zero on everything. Since 87% of the extra
abstention was the 180 s wall, much of the set damage may be an artefact rather than judgement. Not
separable until the wall is raised.

*Paired, not pooled, is also the right STATISTIC.* Per question: 41 better, 14 worse, 55 changed of
1,084. McNemar p = **0.00036**. D-062 called +2.5 "not distinguishable from run-to-run variation" by
reaching for S-52's repeats rule -- correct for comparing MEANS across arms, wrong here, because a
paired within-question comparison makes each question its own control and needs no repeats for the
sign. **Retracted.**

### D-062 addendum — the 180 s wall was not neutral across arms
It cut **33 of 436 Tier B questions out of the critic arm and 1 out of the control**. Not incidental:
concept questions go through `search`, which the critic makes slow; per-paper questions go through
`contents_of`, which it does not touch. So the wall landed on the arm under test, in the tier under
test, and scored those questions as abstentions.

Raised to **600 s** (`EVAL_QUESTION_TIMEOUT`), with the reasoning recorded at the constant: the bound
exists to stop a hung socket, not to cap honest work, so it belongs above the SLOWEST arm's tail
rather than near the fastest arm's mean.

### D-062 addendum — rescored: the critic HELPS set questions, and the direction test confirms why
Prompted by the supervisor-facing worry that Tier B's gold is graph-derived and our system might be
righter than it. Chasing that found the instrument was wrong, not just the gold. Rescored from
stored answers (872 records, no GPU):

| metric | control | critic | |
|---|---|---|---|
| set precision | 0.127 | **0.252** | x2.0 |
| set recall | 0.439 | **0.608** | +0.169 |
| set F1 | 0.145 | **0.314** | x2.2 |
| retrieval reach | 0.917 | 0.889 | -0.028 |

**The earlier "set recall 0.917 -> 0.734, the critic hurts set questions" was an artefact and is
withdrawn.** Those numbers graded the *retrieval footprint* -- every paper containing any entity
touched while searching, median 39 against a median gold of 2 -- and shrinking that footprint is the
critic's whole job. Scored on the papers the answer NAMES, the critic roughly doubles every set
metric.

**The decisive pair**: retrieval reach falls only 2.8 points while set F1 doubles. The critic is
removing junk *without* losing the papers that matter.

#### The direction test answers the "maybe our gold is wrong" worry
`claimed_count` extracts the number the answer asserts, so over- and under-counting are separable:

| arm | n | median claimed minus gold | over | under | exact |
|---|---|---|---|---|---|
| control | 291 | **+34** | 94% | 3% | 3% |
| critic | 269 | **+8** | 82% | 3% | **15%** |

The system **massively over-counts** -- the median answer is 34 papers above a gold whose median is
small -- and the critic cuts that to +8, with exact answers rising 3% -> 15%. Crucially
**under-counting stays at 3% in both arms**: the critic is not cutting too hard, it is removing
things that were never in the answer. That is the shape a working relevance step should have, and it
is evidence the gold is not simply being disagreed with.

*Caveats kept*: 25-30% of set answers name no papers at all and are still scored on the footprint
fallback (`set_named_none`), so those numbers are a blend. The critic arm has fewer scored records
(n=90 vs 109 on reach) because the 180 s wall removed some -- being fixed in the run now in flight.

### D-062 addendum — the gold-free check disagrees with the gold-scored one, and both are right
Paraphrase invariance (S-34): two wordings of one question have the same true answer whatever it is,
so comparing the system to ITSELF needs no gold and nothing about how Tier B's gold was built can
contest it. 218 pairs, from `relation: paraphrase_of:<qid>`.

| arm | usable pairs | count agreement | median gap | one side abstained |
|---|---|---|---|---|
| control | 204 | **0.714** | 0 | 10 |
| critic | 178 | **0.439** | 1 | 21 |

**The critic makes the system markedly less self-consistent.** Against the gold it looks good --
set F1 x2.2, count error +34 -> +8, exact counts 3% -> 15%. Against itself it looks worse.

*Both are true, and the resolution is not a contradiction.* The critic moves answers toward the truth
**on average** while inserting a **noisy step**. The noise is already measured: 16.6% of keep/drop
decisions flip between two shuffles of the same candidate list. Two paraphrases produce two different
search texts, hence two different candidate lists, hence two different sets of verdicts -- so a
noisy filter converts a stable-but-wrong system into a less stable, more-often-right one.

**This is the finding to take to the write-up**, because accuracy alone would have hidden it: a
coverage map that answers the same question two ways and gives two numbers is not usable by a
physicist, however good its mean. Consistency is a requirement here, not a secondary metric.

*Consequences.* The fix for both is the same -- make the critic itself less noisy. That is exactly
what the ranked-vs-shuffled arm and `repeats=3` are for, and it raises the value of a cheaper
mechanism (self-consistency over repeated verdicts, or caching a concept's verdicts across
paraphrases) over a better prompt.

*Caveat kept*: the critic arm has fewer usable pairs (178 vs 204) because the 180 s wall removed
some, and one-sided abstentions doubled (10 -> 21). Some of the measured instability is therefore the
timeout, not the judgement. The run in flight, at 600 s, separates them.

*Also*: `set_jaccard` reported nothing -- the paraphrase pairs in this set are all count-shaped, so
the set half of the measure is untested and awaits a question set that pairs them.

### D-062 addendum — adjudicating the 11 losses: none of them is the critic over-filtering
Read by hand, with the critic's own recorded drop reasons beside each.

| verdict | n | reading |
|---|---|---|
| timeout | 3 | infrastructure; the 180 s wall, since raised |
| **over-counted** | **6** | claimed 3, 10, 36, 6, 14, 45 against golds of 2, 2, 3, 5, 2, 2 -- filtered too LITTLE |
| said nothing found | 1 | the answering step failed *while holding kept candidates* |
| under-counted | 1 | the only shape consistent with over-filtering |

**Not one of the eleven is the critic dropping a correct answer**, which is what "the critic lost us
11 questions" naturally reads as. Six are the opposite failure -- the critic did not cut enough, and
the system still answered 45 papers where the gold says 2.

*The one under-count, examined*: "V+jets (QCD/EW) simulation sample (Sherpa 2.2.1)", gold 4, answered
0. Its drops are **correct physics** -- `vjets_powheg_symmetry` rejected as *"a V+jets sample but a
different generator (Powheg)"*, `wz-zz-jets-nominal` as *"Sherpa 2.2.2 instead of 2.2.1"*. The
question names 2.2.1 specifically. And the critic **kept 13 candidates** and the system still said
"the graph does not record any papers". So even this one is the answering step, not the filter.

*The same pattern in the other "nothing found"*: f_CP^Htt, where 41 of 42 candidates were dropped
with reasons that are each individually right -- f_SM, mu_f, pT^H, f_LR^Z and f_LR^W are genuinely
different observables -- and the answer was still a false negative.

**So the recurring failure is not the filter, it is what happens after it.** A harsh verdict does
push the system toward giving up, but only mildly: abstention runs 1.3% / 1.9% / 3.5% as the dropped
share rises through 0-50% / 50-80% / 80-100%, against 1.8% in the control. Real, and far too small to
explain the false negatives above.

*Consequence*: the next thing worth fixing is not the critic's judgement but the step that turns
kept rows into an answer -- consistent with the separate finding that gold papers sit in the
retrieval footprint 91.7% of the time and are named in the answer 22.8% of the time.

### D-063 (2026-08-16) — a replication control caught a one-line prompt regression, and gave us a real noise floor
Two runs were compared, both nominally `critic=off, answer=v1`:

| comparison | count_correct | reading |
|---|---|---|
| phase 1 (b6ce7b2) vs bridge (c209974) | 0.220 -> 0.204, **-0.0159** | different commits |
| bridge vs replica, **same commit** | 0.204 -> 0.204, **-0.0008** | different runs only |

**Run-to-run variation is 0.0008. The cross-commit difference is nineteen times that.** So the
commit changed v1's behaviour, and the "byte-identical v1" claim was false.

*The cause, one line*:

```
- "text": {"description": "the answer, citing what was retrieved"}
+ "text": {"description": "the answer, in prose"}
```

Adding the v2 contract rewrote the `answer` tool's `text` description, and `tools_for(False)`
stripped the new FIELDS without restoring the old WORDING. So v1 quietly stopped telling the model
to cite what it retrieved. The downstream numbers agree: `unsupported_claims` 0.284 -> 0.244,
`answered` 0.955 -> 0.974, `abstained` 0.042 -> 0.023.

**The unit test passed throughout**, because it asserted the property NAMES were
`{text, answerable, reason}`. Descriptions *are* the prompt. A name check cannot see a prompt change,
and this is the second time this week a guard has been shown to test a proxy rather than the thing
(the first: `from_session` dropping the critic's record while a unit test asserted on the Session).

*Fixed two ways.* The polarity is inverted -- **v1's wording is now the canonical text and v2
replaces it**, so the control cannot drift when an arm is added. And the v1 schema is frozen by
fingerprint: `aa028ae68fd37d83`, taken from b6ce7b2, the code that actually produced the phase-1
arms. Verified identical after the fix. A test fails if it ever moves again.

#### Two things this changes beyond the bug
**We now have a measured noise floor, and it is small.** 0.0008 on `count_correct` between runs of
the same code. Every result this week clears it by a wide margin -- the critic's Tier B counting
effect (+0.086) by 100x, the paper-id fix (+0.18) by 200x, the set F1 doubling (+0.169) by 200x. The
one number that would NOT have cleared it is the original pooled +0.025, which is why pooling was
the wrong reading rather than merely a weak one.

*And it means the within-run repeat spread was not the problem I feared.* The `±` printed across 3
repeats is of the same order as the between-run figure, so `compare`'s noise floor is honest. The
bridge's apparent 3x-noise deltas were a real regression, not an underestimated bar.

**Phase 2 must run on the fixed commit.** Because the fix restores v1 exactly, a v1 arm on the fixed
code is comparable with phase 1's arms again -- so the 2x2 can use phase 1's `ranked/v1` as its
control rather than needing it re-run. Had the regression gone unnoticed, every v1-vs-v2 comparison
would have carried a one-line prompt change inside it and been read as the answer contract's effect.

### D-064 (2026-08-16) — phase 1: the critic works, shuffled beats ranked, and the widen fix fires
Three arms, `repeats=3`, aligned on 1,627 (qid, repeat) triples all three answered. Scored with the
current scorers (the run files needed rescoring first -- see the caution below).

| metric | off | ranked | shuffled |
|---|---|---|---|
| count correct | 0.233 | **0.288** | **0.304** |
| set F1 | 0.146 | **0.331** | **0.330** |
| set precision | 0.127 | 0.268 | 0.263 |
| set recall | 0.442 | 0.673 | 0.676 |
| retrieval reach | 0.908 | 0.887 | 0.887 |
| abstained | 0.036 | 0.049 | 0.034 |
| **widened searches** | -- | **224** | **180** |

Against the measured between-run noise floor of **0.0008** (D-063), the counting gains are 70-90x the
bar and the set F1 gain is 200x. Paired on set questions: **106 better / 43 worse** for ranked,
93 / 39 for shuffled.

**The signature holds**: retrieval reach falls 2 points while set F1 more than doubles. The critic
removes junk without losing the papers that matter.

**The widen fix works.** 224 and 180 widened searches against **zero** across 1,074 searches before
D-063's decay ratio. A critic marking ~76% unrelated can now widen when relevance has not decayed by
the tail, which an absolute threshold made impossible.

**Shuffled beats ranked on counting** (0.304 vs 0.288) and abstains less (0.034 vs 0.049); on set
questions they tie. That is the second independent measurement pointing the same way -- D-062 saw it
on 37 searches with one repeat, this is 1,627 triples with three -- so **the default should change to
shuffled**. Ranked was chosen to protect the best candidate from the lost-in-the-middle effect; the
protection costs more than it saves, and the mechanism proposed then still fits: in rank order each
chunk is homogeneous and the model calibrates within it, while a shuffled chunk holds a mix that can
be judged against the question.

#### A reporting error worth recording, because it was avoidable
Phase 1's numbers were first read straight from the run files, which still carried the **old
footprint-based `set_f1`** -- the metric retired the day before. That produced "set F1 slightly worse
with the critic", reported as contradicting the previous day's rescore, with the honest-sounding
caveat that I did not know which was right.

There was no contradiction. One dataset had been rescored and the other had not. **The check that was
skipped is trivial**: confirm both sides were scored by the same code before comparing them.
`rescore` exists precisely so stored answers can be re-read under a new metric, and it costs no GPU.

*A second, quieter fault in the same analysis*: `scores.get("set_named_none", 0)` reported **0%** for
runs where the metric did not exist, which reads as a measurement rather than an absence. The real
figure is 20-27%. **A missing metric must never default to a value that looks like data** -- the same
failure shape as a silently dropped critic record (D-060) and a name-only schema check (D-063).

*Consequence for the harness*: run records should carry the scorer version alongside `git_sha`,
`config_hash` and the database identity already noted as missing. Four things now identify a
measurement, and only two of them are recorded.

### D-065 (2026-08-17) — a small judge is viable, and the failure was the prompt asking the wrong thing
The critic is **48% of all LLM calls**, so it is the obvious place for a small model -- and at scale
it is the only place that matters, since D-061's growth curve makes the critic's call volume the
thing that grows. Llama-3.1-8B judging, Qwen2.5-72B answering, on separate cards.

The prior was discouraging and specific: on the aliases task the 8B answered *"are these related?"*
rather than *"are these the same?"*, merging 8 distinct SMEFT Wilson coefficients and accepting WH/ZH
at confidence >= 0.9 ([[logs/2026-07-28]]).

**Three prompt iterations, gated at 45 calls each:**

| control | v1 definitions | v2 structural | v3 balanced | 72B |
|---|---|---|---|---|
| all-drop, Pythia question | 38.9% | 6.7% | **3.3%** | 0.0% |
| all-drop, JES question | 86.7% | 3.3% | **3.3%** | 6.7% |
| all-keep, pileup cluster | 100% | 58.3% | **100%** | 100% |
| all-keep, pp-13TeV cluster | 100% | 100% | **100%** | 100% |
| mixed-family | 63.3% | 40.0% | **44.4%** | 70.0% |

#### The failure was never that the model could not compare
v1 reproduced the aliases-task fault exactly: every reason it gave was a **label for the candidate**
and none mentioned the question. Asked whether `jet-energy-scale` bore on a **Pythia** question it
answered *"jet energy scale systematic"* and kept it. Asking "why" invites a description, and a small
model gives you one.

**v2 made the comparison structural rather than requested**: the model fills separate `is` and `asks`
fields before choosing a rung, so the question has a slot that must be filled. Two controls moved by
**30 and 83 points**. That is not prompt polish -- it is removing the option to skip the comparison.

#### And then it learned the wrong direction from the examples
Both v2 examples were DROPS, chosen to correct over-keeping. The 8B promptly began splitting
`pileup reweighting` from `pileup modelling` -- two names the aliases layer has already adjudicated
as one uncertainty -- and an all-keep control fell to 58.3%. **Examples teach a direction, so they
have to point both ways.** v3 adds one KEEP example on a wording variant and states the rule in
words: different WORDING for one thing is `exact`, a different THING is `unrelated`.

**The asymmetry against the 72B is the transferable finding.** The large model needed its
*definitions* corrected (D-060: rungs defined against the search rather than the question, then
record type mistaken for relevance). The small model needed its *demonstrations* balanced and was far
more sensitive to them. Prompt work does not transfer between sizes; the failure modes are different
in kind.

#### Held back from calling this a win
Three iterations against five controls is a small target, and fitting the prompt TO the controls is
the obvious risk. The full arm on real questions is the test of whether it generalises, and it is
running. Read against the 72B-judge arm on the same shard, which took counting 0.235 -> 0.298:
landing near 0.235 would mean the gate was fitted rather than passed.

*The gate itself is now 2 for 2* -- it caught the reason-vs-label prompt bug on the 72B, and here it
turned "spend twenty hours finding out" into three cheap iterations that fixed the thing.

### D-065 addendum — the 8B judge measured on real questions: most of the benefit, a third of the cost
Full arm, `repeats=3`, 72B answering throughout. Aligned across all three arms.

**Counting** (conceptB-00/01 + paperA-200, 1,342 triples):

| arm | count correct | seconds | kept | critic calls |
|---|---|---|---|---|
| critic off | 0.2352 | 14.6 | -- | 0 |
| critic **72B** | **0.2982** | 93.7 | 24.2% | 3,091 |
| critic **8B** | **0.2709** | 44.0 | 36.9% | 3,572 |

**Set questions** (conceptB-02, which holds all 109, 315 triples):

| arm | set F1 | set recall | set precision | reach | seconds |
|---|---|---|---|---|---|
| critic off | 0.1459 | 0.4424 | 0.1269 | 0.9083 | 35.1 |
| critic **72B** | **0.3311** | 0.6733 | 0.2675 | 0.8868 | 182.9 |
| critic **8B** | **0.2896** | 0.5929 | 0.2491 | 0.8899 | 86.9 |

**The trade, both ways:**

| | share of the 72B's gain | share of its added cost | efficiency |
|---|---|---|---|
| counting | 57% | 37% | **1.5x** |
| set questions | **78%** | **35%** | **2.2x** |

*It is more permissive* -- 36.9% kept against the 72B's 24.2% -- which is exactly why it captures less
of the gain: it removes less junk. The direction is right and the aggression is lower.

*And it is faster per question even with the 72B at TP=1*: 44.0s against 93.7s on counting, 86.9s
against 182.9s on sets. The earlier hedge that the topology change would be "roughly neutral" was
wrong; shedding 48% of calls to a small model on its own card beats halving the answerer's KV cache,
and by a wide margin. **The user's instinct was right and my arithmetic was one-sided.**

*Consequence for D-061's scaling problem.* At 2,969 papers the critic's call volume is the quantity
that grows, and bisecting a 5,000-candidate list is only affordable with a cheap judge. This makes
that architecture viable rather than hypothetical -- and it is the same conclusion from the other
direction: the expensive model should answer, the cheap one should filter.

### D-065 addendum — the shard split accidentally stratified by question shape
`split -n l/3` on the Tier B file produced:

```
conceptB-00: 146 count,   0 set
conceptB-01: 145 count,   0 set
conceptB-02:  36 count, 109 set     <- every set question
```

The generator emits counting questions first and set questions last, so splitting by LINE split by
SHAPE. Consequences, all of which bit:

- every comparison run on shards 00/01 had **zero** set questions, and `set_f1` came back `nan`
  correctly while I read it as a missing metric
- shard 02 was also the **slowest** (it was at 72% when phase 1's 16-hour wall hit), so the set
  questions were the ones disproportionately lost -- which is why set F1 there had small n and odd
  values
- any per-shard timing comparison conflates shard identity with question shape

*Fix*: shards must be built by interleaving (`split -n r/3`, round-robin) rather than by contiguous
lines, so each carries the same mix. **A split that is not random is a stratification**, and this one
was invisible because the file happened to be ordered.

---

### D-066 (2026-08-17) — the supervisor's first 50 verdicts: precision 0.12, and the misses were found by cheap retrieval
The first externally-authored gold this project has had. 50 rows, all **gf-01** -- the hardest
question in the set, a three-way conjunction (SEARCH **and** b-tagged jets **and** missing transverse
momentum) and the one D-058 already documented failing in five distinct forms.

|  | he says yes | no | unsure |
|---|---|---|---|
| **we said YES** | 3 | **22** | 4 |
| **we said NO** | **5** | 12 | 4 |

**Strict precision 0.120, recall 0.375.** Of 25 confident positives, 3 were upheld.

#### The finding that matters most is not the precision
All five false negatives carry `found_by=none` -- the reader found nothing, so the judge never had a
chance to be wrong. But the sheet showed him, for every miss, the sentences our **hybrid retrieval**
surfaced (BM25 + dense over the paper's sentences, the same retrieval the query layer uses), because
a miss cannot be reviewed without something to look at.

**In at least three of the five, he answered YES from those sentences alone.** Row 10 is the clearest:

> "The experimental signature of this search, for all signal topologies, consists of multiple jets,
> one or two of which are $b$-tagged, no electrons and muons, and large missing transverse momentum."

Search, b-tagged, MET -- all three conditions, one clause -- and we reported *WE FOUND NO EVIDENCE*.
His comment is "Huge failure here. How is this not identified as a paper that matches?" and it is
entirely fair.

**So a single-shot cheap retrieval found what a two-model LLM cascade reading the whole paper in
windows did not.** That is the same shape as three other findings this week and they now form one
thesis: *retrieval is not this system's bottleneck, judgement is.*

- the query layer reaches 91.7% of gold papers and names 22.8% in the answer
- the critic wrote correct reasons and attached wrong labels (D-060)
- the 8B judge described candidates instead of comparing them (D-065)
- and here, the reader misses what a BM25+dense pass finds in one shot

#### What his notes add beyond the verdicts
**A distinction we did not encode.** Rows 12, 17, 22 and 26 all say a version of: *"in the strict
interpretation of this evidence alone the answer is no; in the sense of whether I should keep looking
at this paper, it is yes."* That is **evidence-sufficient** versus **paper-true**, and it is a
different axis from yes/no. Four of the eight `unsure` verdicts are this, not genuine uncertainty.

**A schema gap he is already fixing.** Row 1: the answer is *yes for a validation region and no for a
signal region*, and "this distinction is lost in the current schema". He has opened an MR
(gfacini/HEPKG_promopt_tests PR #6). Region ROLE -- signal / control / validation -- is not
represented, and several of his verdicts turn on it.

**A defect to chase.** Row 21: *"'What our model said' does not match what is in the quote."*

**A caveat on his own method, which he volunteered**: he judged some early rows against the full
paper before deciding he should judge only the extracted text. So the first rows are a mix, and the
precision above is a floor rather than a clean number.

#### What follows
1. *Do not "fix" precision by tightening the judge.* The 22 false positives and the 5 misses have
   different causes, and the misses are the expensive ones -- a false negative is a false claim about
   coverage, which is the output this project exists to produce.
2. *Test the cheap retrieval as a reader.* If BM25+dense over sentences finds what the cascade misses,
   it belongs in the reader, not only in the review sheet. That is measurable on the 5 misses today
   and on all 202 rows when the rest of the review lands.
3. *Encode `evidence_sufficient` separately from `true_in_paper`.* His four `unsure` verdicts are
   answers to a question we never asked.
4. *Region role belongs in the schema*, and his MR is the specification.

### D-067 (2026-08-25) — the LaTeX corruption Gabriel found: the D-059 repair exempted exactly the wrong letters

Gabriel's review sheet showed `$b\bar{b}$` stored as `$b<backspace>ar{b}$`, in 12 of 202 rows. The
database was never wrong -- 0 backspaces in `source_block.text` (31,051), `evidence.quote` (8,482) or
`assertion.object_value` (14,188). The corruption was introduced when the reader parsed the model's
reply, and it survived into every file derived from that run.

**Cause.** D-059 added `_repair_latex_escapes` to rescue LaTeX from a backslash JSON cannot read. Its
regex exempted `b f n r t u` because those *are* legitimate JSON escapes. They are also the letters
HEP LaTeX collides with most. Measured on one sweep's raw output:

| escape | as LaTeX | example | as real whitespace |
|---|---|---|---|
| `\t` | 3,030 | `\tilde` 919, `\to` 892, `\text` 491, `\tau` 204 | 0 |
| `\r` | 906 | `\raise` 732, `\rightarrow` 129, `\rho` 24 | 0 |
| `\b` | 471 | `\bar` 469 | 0 |
| `\n` | 106 | `\nu` 106 | **38,640** |
| `\f` | 1 | `\frac` | 0 |

So `\bar` decoded to a backspace and three characters vanished.

**Fix.** The discriminator is what FOLLOWS: `\n{` is a newline, `\nu$` is a Greek letter. An escape
letter counts as valid only when a letter does not follow it; `\u` stays exempt outright. Plus one
refinement -- an escape directly after another whitespace escape is part of a whitespace run, so
`a\n\nb` keeps both newlines.

**What stays unresolvable.** A lone real newline before a word: `\ntwo` and `\nu` are the same three
characters and no local rule separates them. The repair leans toward LaTeX, because the two mistakes
are not equal -- escaping a real newline prints a visible `\n` and loses nothing, while missing LaTeX
destroys characters. In the corpus the ambiguous shape never occurred: all 106 `\n`+letter were `\nu`.

**Back-repair, and the mistake inside it.** 1,297 escapes were restored across 18 existing files. The
first attempt applied the letter rule to every field and ate 34 `\nThis`, 32 `\nThese`, 22 `\nIf you
need m...` -- genuine paragraph breaks in free-form answers. Restored from backup and rescoped: the
letter rule applies only to fields holding verbatim paper text (`quote`, `quotes`, `evidence`,
`source_block`, `span`), where a paragraph break cannot occur and 100% of the hits were LaTeX.
Backspace and formfeed are healed everywhere, since neither is ever legitimate.

**The general lesson.** A repair that has to guess needs its scope narrowed to where the guess is
safe, not its heuristic sharpened. The same rule was right in quotes and wrong in prose.

**Reach.** Only the sweep-derived files carried it; `gf01-full` has 0, because D-059's partial fix
landed between the two runs. Batch 1 is already sent and is not being regenerated -- the 12 affected
rows are a display defect in a sheet Gabriel has already judged, and his verdicts on them stand.

### D-068 (2026-08-25) — region role: the distinction was already in the graph, spelled 33 ways

Gabriel's row 1: the answer is *yes for a validation region and no for a signal region*, and "this
distinction is lost in the current schema" (D-066). It was not lost. It was un-normalised.

Across 781 region occurrences the extraction wrote a role under **three keys** -- `role` (488),
`region_role` (82), `is_signal_region` (10) -- in **33 spellings**: `SR`, `signal_region`, `signal
region`, `signal`, `signal-region-component`, `discovery signal region`, `SR (counting)` all mean one
thing. So this is a normalisation, not a re-extraction: cheaper, and it invents nothing.

**Five canonical roles**: signal, control, validation, fiducial, preselection.

**Two rungs, attribute first.** The attribute is what the extractor asserted and wins; the label is a
fallback for the 193 regions with no role attribute, where the name states it in prose ("Orthogonal
validation region, m_ll in [70,105] GeV"). The rung travels with the value in `version`, so a claim
can be restricted to asserted roles later.

*Both rungs fire on 365 regions and agree on 363 (99%).* That is the validation: two independent
readings of the same fact, derived differently, converging. The 2 disagreements both go to the
attribute correctly -- "Baseline H->4l ZZ-candidate selection" is labelled like a preselection but
asserted `signal_region`.

**Coverage 640/781 (82%)** -- 572 from attributes, 68 from labels. signal 295, control 199, fiducial
72, validation 51, preselection 23. 51 papers have a signal region, 41 a control region, 15 a
validation region, and **13 have all three**.

**Two judgement calls, recorded not hidden**: `sideband`/`SB` -> control (a sideband exists to
estimate a background from data, which is what a control region does); `baseline` -> preselection.

**What it refuses to guess.** `model-independent superbin` (7), `aggregate of superbins`, `excluded
region`, `extra_jet_definition`, `full_phase_space`, and `is_signal_region: False` -- which says only
what the region is *not*. This is the field the supervisor is going to check; inventing structure in
it would be the worst possible place to be clever.

**Why it derives on its own vocabulary version.** `CARD_FIELDS` mirrors upstream's `_CARD_FIELDS` and
the parity test compares our derivation against their frozen snapshot card for card. Adding
`event_region` there would fail the check that exists to catch drift. So region roles derive on
`region-roles-v1` with their own delete scope, and the query layer resolves field -> kinds and field
-> vocabulary through a new `FIELD_KINDS` / `FIELD_VOCABULARY` map, so a caller naming a field never
has to know which vocabulary holds it.

`facets("region_roles", ["signal","control","validation"], mode="all")` now answers the question that
was unanswerable. CLI: `facets region-roles`.

### D-069 (2026-08-25) — decomposition, measured: it kills the false yes and the true yes together

The conjunction problem (D-058) says precision falls as conditions are conjoined. Gabriel's 199
verdicts measure it, counting only rows where the system actually cited a sentence:

| conditions | questions | claimed | right | precision |
|---|---|---|---|---|
| 1 | gf-02, gf-03, gf-04 | 21 | 19 | **0.90** |
| 2 | gf-05, gf-07, gf-08 | 78 | 36 | **0.46** |
| 3 | gf-01 | 31 | 5 | **0.16** |

Roughly halving per added condition, and falling faster than independent errors would predict
(0.90^2 = 0.81, 0.90^3 = 0.73) -- the conditions are not independent, later ones are harder.

**The mechanism already exists and was not wired to where it mattered.** `reader.read_conditions`
asks one condition at a time, ANDs them in code, and records `missing` -- which condition failed, not
just "no". But the sheet Gabriel judged was built from the SWEEP, and the sweep asks the whole
conjunction as one question. Verified in the files: sweep gf-01 rows carry no `conditions` key; the
later `gf01-full` run carries it on all 60.

**So the direct A/B was run, and here is what it says.** On the 24 judged papers where the compound
question cited a sentence -- 3 right, 21 wrong -- the decomposed run flags a missing condition on:

- **17 of the 21 Gabriel says NO** -- it would correctly refuse 81% of the false claims
- **3 of the 3 Gabriel says YES** -- it would also refuse every true one

That is not the clean win the trend table implies. Decomposition converts *wrong yes* into *wrong
no*, because the reader cannot find the evidence for an individual condition either. It fixes the
compounding without fixing the retrieval underneath it, and a false negative is the more expensive
error for a coverage map -- it is a false claim that nothing is there.

*n = 3 on the yes side. This is a direction, not a rate.*

**What follows.** Decomposition is not adoptable on this evidence alone. The gap-fill sheet is the
test that settles it: put the DECOMPOSED answers in front of Gabriel for the same gf-01 papers, with
the per-condition quotes, and the trade becomes measurable rather than inferred.

**A correction to how the sheet has been read.** `gabriel-review.tsv` mixes two row types -- rows
where the system cited a sentence and rows saying "WE FOUND NO EVIDENCE. The closest sentences
were...". Pooling them and calling yes/(yes+no) "precision" mixes a precision with a base rate. Split:
precision **0.46** on 130 claimed rows; on the 55 abstention rows the system missed **18 real yes
(33%)**. Overall accuracy 97/185 = **0.52**. The per-question precision table already reported was the
claimed-only cut and stands.

### D-070 (2026-08-25) — a run must be able to name its own arm

`PlannerSystem.config` was `{**planner_kwargs}` -- only what the caller **overrode**. Every knob left
at its default recorded nothing, so 25 of 44 stored runs name no arm, and the knobs the ablation turns
on are precisely the defaulted ones: `critic_seed` (None = ranked, int = shuffled), `contract`, and
the `CRITIC_BASE_URL` that decides whether the 8B judge runs or the 72B one. Which arm produced a
result had to be reconstructed from job scripts and notes. The whole ablation table is currently
attributable to memory rather than to artefacts.

**Fixed by recording the EFFECTIVE configuration** -- every parameter's actual value, defaults
included -- derived from `inspect.signature(planner.answer)` rather than a hand-kept list, so a knob
added next month is recorded without anyone remembering to. Plus the five environment variables that
change behaviour and never passed through the function at all (`CRITIC_MODEL`, `SEARCH_BREADTH_MAX`,
`LLM_MODEL_NAME`, `LLM_MAX_COMPLETION_TOKENS`, `CRITIC_BASE_URL`), and a derived `critic_order`
because "ranked"/"shuffled" is the name a human uses and re-deriving it from `critic_seed is None` at
analysis time is how it gets misread.

Excluded as plumbing: `conn`, `index`, `question`, `thread_id`, and -- importantly -- `chat` and
`checkpointer`, which are live per-process objects whose repr would give every run a different hash
and silently destroy the comparison the hash exists for.

**The guard is a test, not a convention.** `test_every_knob_on_the_planner_is_recorded` fails if a
parameter is added to `planner.answer` without a decision about whether it is configuration. The cost
of forgetting is a week of unattributable runs, and it is silent.

The arm is also logged at the top of the job log: a 20-hour job that ran the wrong arm should be
caught on submission, not when the results disagree.

*Nothing before today can be retrofitted. Runs prior to this commit keep whatever they recorded, and
the arm labels in D-062..D-065 rest on the job scripts.*

### D-071 (2026-08-25) — what to test Gabriel's questions on

Proposal was: run his questions across critic on/off, shuffled on/off, big/small judge, rather than
only on a frozen configuration.

**The knobs do not reach the runs as they stand.** Critic, ordering and judge size are QUERY-LAYER
knobs. Gabriel's sheet was produced by the READER, per paper. Running critic on/off over those rows
gives byte-identical output.

**The version that works**: convert his 199 per-paper verdicts into **8 set-valued query-layer
questions with human gold** -- "which papers are searches using b-tagged jets and MET?", gold = the
papers he marked yes. Then all three knobs bite, and the query layer is measured against human ground
truth instead of our synthetic gold, which is a strictly better test than the freeze-only plan.

**Two limits, both scoreable around.** He only judged papers our system surfaced, so recall against
this gold is inflated -- restrict scoring to the judged subset and precision is clean. And 8 questions
is thin, so the unit is the (question, paper) pair, ~185 of them, paired across arms.

Full 2x2x2 = 8 arms, 8 questions, 3 repeats ~ 5 GPU-hours. Cheap enough to run all eight rather than
pick.

### D-072 (2026-08-25) — Gabriel's verdicts as a query-layer question set, and what partial gold costs

Built `eval/questions/gabriel-gold-2026-08-25.jsonl`: 8 set-valued questions, gold written by a
physicist. His per-paper reader questions reworded from "does THIS paper" to "which analyses", and
nothing else -- the physics must not drift or the verdicts stop applying.

| question | gold | judged | unsure | conflicted | conditions |
|---|---|---|---|---|---|
| gf-01 | 3 | 28 | 4 | **7** | 3 |
| gf-01-condition | 8 | 10 | 0 | 0 | 1 |
| gf-02 | 9 | 14 | 0 | 0 | 1 |
| gf-03 | 5 | 9 | 0 | 0 | 1 |
| gf-04 | 11 | 17 | 0 | 0 | 1 |
| gf-05 | 15 | 29 | 2 | 0 | 2 |
| gf-07 | 9 | 39 | 0 | 0 | 2 |
| gf-08 | 13 | 23 | 0 | 0 | 2 |

**Three things get excluded from the gold, each for a different reason.**

*Conflicts (7, all gf-01).* 199 rows cover only 186 distinct (question, paper) pairs -- the sheet
asked about 13 papers twice, and he answered differently on 7. Three are a flat **yes against a no**:
2006.05880, 2202.08676, 2508.13900. Keying a dict by paper keeps whichever row came last and puts a
coin-flip into the ground truth. Disagreement is treated as UNRESOLVED and leaves both the gold and
the universe. That every conflict is on the three-part conjunction is itself evidence for D-069: the
question is genuinely ambiguous, not carelessly answered.

*`unsure` (14).* Four are "the evidence is thin but the paper is probably true" (D-066) -- a different
axis from yes/no. Scoring them either way invents a verdict he declined to give.

*Open questions (gf-06, gf-10..gf-15).* One row each, and they ask for a value, not a set.

**A new scorer, because neither existing one is honest here.** He was only shown papers our system
surfaced, so the corpus splits three ways: yes, no, and **never judged**. `set_f1` would treat all
~40 unjudged papers as negatives and charge for every one named -- measuring how the sheet was sampled
rather than how the system answered. `Truth.universe` carries the judged papers and
`scoring.judged_set_f1` restricts both sides to them. Metric names deliberately distinct from
`set_f1`: two measures under one name is exactly what made a month of numbers uninterpretable before
(see `retrieval_reach`).

**Precision under this restriction is clean. Recall is not, and cannot be.** A paper the system never
surfaced was never put in front of him, so it could not become a gold positive -- the misses that
would hurt most are invisible by construction. `judged_coverage` travels with every result: the
judged universe is 15-65% of the corpus depending on the question, 23% for gf-02.

**The split is `dev`, and that is a real cost.** These are the best labels in the project and the
instinct is to lock them as `test` (S-10). But the immediate use is choosing between eight ablation
arms, and choosing on a set is developing against it; marking them `test` and running the ablation
anyway would launder that. **Batch 1 is spent on selection. The held-out human evaluation must be
batch 2 -- the questions Gabriel has not returned yet -- and it must not be looked at until the arm is
frozen.**

### D-072 addendum — "he contradicted himself" was my misreading; the strongest verdict wins

Raul pushed back: if one quote says yes and another says no, the paper is yes, because a quote said
yes. He is right, and checking the sheet settles it -- **all 13 duplicated (question, paper) pairs
carry DIFFERENT cited sentences**, none is a repeat.

A sheet row is (question, paper, ONE cited sentence). So `no` is a statement about the SENTENCE --
this quote does not establish the claim -- not about the paper. If another quote establishes it, the
paper is yes. Treating the pair as a contradiction and dropping the paper cost gf-01 **five of its
eight gold papers**.

`resolve` now takes the strongest verdict, yes > unsure > no. `unsure` still leaves the gold and the
universe: no row established the claim and he declined to reject it.

    gf-01  2006.05880   ['yes','no']      -> yes
    gf-01  2202.08676   ['no','yes']      -> yes
    gf-01  2508.13900   ['no','yes']      -> yes
    gf-01  2004.14060   ['yes','unsure']  -> yes
    gf-01  2012.03799   ['unsure','yes']  -> yes
    gf-01  2012.08600   ['unsure','no']   -> unsure  (dropped)
    gf-01  2106.01676   ['unsure','no']   -> unsure  (dropped)

gf-01's gold: 3 -> **8**. The duplication itself is the review builder's missing dedupe by
(qid, paper), already on the list to fix -- the sheet should show one row per paper carrying all its
candidate sentences.

**D-069's conjunction table, recomputed per paper rather than per row:**

| conditions | claimed | right | precision |
|---|---|---|---|
| 1 | 21 | 19 | 0.90 |
| 2 | 78 | 36 | 0.46 |
| 3 | 27 | 8 | **0.30** (was 0.16 per row) |

The finding survives -- precision still roughly halves per added condition -- but the three-condition
point was overstated at row level, because a paper with a supporting second quote was counted as a
miss. **0.16 was wrong; 0.30 is the number.**

### D-073 (2026-08-29) — gf-08: the one place where retrieval IS the bottleneck, and it is a type error

Every one of the six arms scored **0.000** on gf-08, and every one gave the same answer:

> "The graph does not record any analyses that require exactly two electrons or exactly two muons as
> alternative selections."

Thirteen papers do. `retrieval_reach` was 0.0 on all six -- not one gold paper was touched.

**The evidence was in the graph the whole time**, in `channel` entities:

    2009.14537   "ee+p (electron pair plus tagged forward proton)"
    2009.14537   "μμ+p (muon pair plus tagged forward proton)"
    2001.06899   "Z(→e+e-/μ+μ-) + jet(s) final state"
    2011.07812   "Two displaced leptons (ee, mumu, or emu)"

A regex over channel labels reaches 7 of the 13.

**The plan, from the trace:**

    search("exactly two electrons", kind="selection_requirement")  -> 8
    search("exactly two muons",     kind="selection_requirement")  -> 6
    subjects_of("region_requires_object", set_1)                   -> 0
    subjects_of("region_requires_object", set_2)                   -> 0

Search worked. The HOP was impossible: `region_requires_object` relates objects of kind
`detector_object` (1,022) and `object_definition` (33), and never a `selection_requirement`. The join
was guaranteed empty before it ran. The model read 0 as "nothing exists" and asserted it.

**So this is a planning failure wearing a retrieval failure's clothes** -- and the only case so far
where the project's thesis ("retrieval is not the bottleneck, judgement is") does not hold, because
here neither retrieval nor judgement failed. A type error did.

**Sized across the stored arms**: 518 `subjects_of`/`objects_of` calls, **75 returned 0 (14%)**, and
**30 of those 75 (40%) were a predicate paired with a kind it can never accept**:

| n | predicate | given | actually accepts |
|---|---|---|---|
| 18 | `result_defines_region` | background | detector_object, event_region |
| 9 | `region_requires_object` | selection_requirement | detector_object, object_definition |
| 3 | `region_requires_object` | event_region | detector_object, object_definition |

Two in five empty hops were answerable before the query ran.

**Fixed**: an empty hop now asks the typed graph the question only a typed graph can answer -- which
kinds does this predicate relate, and which predicates would accept the kind in hand:

> 0 rows, and it could not have been otherwise: 'region_requires_object' relates objects of kind
> ['detector_object', 'object_definition'], but this set holds ['selection_requirement']. Predicates
> that DO accept ['selection_requirement']: ['object_has_selection', 'region_has_selection'].

A note, never an exception: a genuinely empty result is still a legitimate answer and must not become
a crash. A well-typed hop that simply matches nothing stays silent.

**Why this matters beyond one question.** It is the typed layer earning its keep in a way no ablation
had shown: the schema already knew the query was impossible, and nothing was asking it. That is a
better argument for typing than any of the critic measurements, and it arrived from a failure rather
than from a success.

*Not yet measured: whether the note changes behaviour. The model has to read it and re-plan, and
that is an arm, not an assumption.*

### D-074 (2026-08-29) — v3 has been running with half of itself switched off

Raul asked whether the agent re-plans when a route fails. Measured across the stored arms, it mostly
does not:

| after a step returned 0 rows | n | share |
|---|---|---|
| a new search | 97 | 35% |
| **stopped there** | 81 | **29%** |
| the SAME tool and predicate again | 66 | 24% |
| a different tool | 34 | 12% |

And it quits with resources in hand. Of 105 abstentions on conceptB-200:

- **98 (93%) still had rounds left** -- the budget is 6, the mode is 3
- **105 (100%) had already retrieved entities**, on average **59 entities across 30 papers**

gf-08 is the shape of it: 3 rounds of 6, two searches, two impossible hops, then "the graph does not
record any analyses that require exactly two electrons or exactly two muons". Thirteen papers do.

**The mechanism built to stop exactly this never ran.** Contract v3 is defined as two things -- cite
the paper set, and defend an abstention made while holding evidence. v3 was on for every arm. Result:
**105 abstentions, 0 challenged.**

**Cause.** `tools_for` resolves v1/v2/v3 correctly, but the challenge in `graph.py` was gated on
`runtime["answer_contract"]`, which is `bool(answer_contract)` -- the **v2 flag**. `--contract v3`
leaves it False. One value read as if it were two different questions: *which contract is this* and
*was the old flag passed*.

**Consequence for what has already been reported.** Every v3 measurement was made with one of its two
mechanisms dead -- including the **+0.059 set F1** that made v3 the proposed default. That number is
real, but it is the value of citing the paper set ALONE. The challenge has never been measured at
all, in any run, ever. D-062's "challenged abstention: kept, still unmeasured" was more literally
true than intended.

**Fixed** by gating on the resolved contract (`v2` or `v3`), not on the legacy flag. v1 stays frozen
at fingerprint aa028ae68fd37d83, and v3 remains a strict subset of v2's tools -- the fix must not
quietly promote v3 into v2.

**What is still not fixed, and matters more.** The challenge only fires when the model calls `answer`
with `not_in_graph`. It does nothing about the 29% that stop dead after an empty hop, or the 24% that
retry the identical failing call. Persistence is a separate defect: the loop should not permit an
abstention while rounds AND retrieved material remain, and after an empty hop it should widen --
drop a condition, or fall back to `papers_of` on what it already holds -- before giving up.

*This is why the type-error note (D-073) is necessary but not sufficient: telling the model what went
wrong does not help if it is not going to try again.*

### D-075 (2026-08-29) — persistence, and the objection that it would just loop

The proposal: stop the agent abstaining while it still holds rounds and retrieved entities. Raul's
objection: that would make it "go in loops infinitely without actually finding anything".

**The objection is right about the mechanism and wrong about the size, and the numbers say which.**

*What a forced retry would cost.* The runs that give up are worth nothing already:

| | count_correct | set_f1 | rounds | seconds |
|---|---|---|---|---|
| abstained | **0.000** | 0.061 | 3.5 | 44 |
| answered | 0.077 | 0.323 | 3.4 | 57 |

They spend 44 seconds to produce a zero. Persistence spends budget that is already allocated (6
rounds, mode 3) on the runs currently contributing nothing, so the downside is bounded at *time
wasted on questions we are already failing*.

*A correction to D-074.* "24% retry the identical failing call" was too strong -- that figure was
same tool and same predicate, not identical arguments. **Exact duplicates are 68 of 4,479 steps
(1.5%).** The looping today is milder than stated.

**The real risk is not looping, it is fabrication.** "No paper here covers that" is this project's
actual output, and a system taught never to abstain invents coverage instead. `graph.py` already says
this. So abstention stays available and becomes *earned* rather than *impossible* -- the challenge
asks for the reason, and a sound abstention passes and is recorded with its justification.

**Four bounds, none of them open-ended:**

1. `max_rounds = 6`, already allocated and currently unspent. Persistence adds no budget.
2. **No exact call runs twice.** Answered from the record, with what it returned -- "you already ran
   this and it gave 0 rows" is an argument for doing something else, where "you already ran this"
   invites a third attempt. Errors are recorded too: repeating a failing call is the same waste.
3. The widening ladder is finite: drop a `kind` filter, take the predicate the schema suggested
   (D-073), fall back to `papers_of` on what is already held. Roughly four rungs, each usable once.
4. It only triggers where the run would otherwise score zero.

**Built so far**: the duplicate guard (2) and the contract fix (D-074). `answer` is deliberately
exempt from deduplication -- blocking a repeat of the exit would trap the run in the very loop this
prevents.

**Not built**: the refusal to abstain with budget in hand, and the widening ladder. Those ship as an
ARM, not a default. If they burn rounds without moving `count_correct`, that shows up against
`seconds` immediately and they are not adopted.

### D-076 (2026-08-29) — the widening ladder: persistence that provably terminates

Built behind `--persist`, an arm and not a default. Before a run abstains while holding retrieved
rows with rounds to spare, it is offered ONE concrete untried route.

**Four rungs, each offered at most once per run**, built from the run's own trace:

| rung | fires when | says |
|---|---|---|
| `drop_kind` | a search carried `kind=X` | try the same text without the filter |
| `compatible_predicate` | a hop hit a type mismatch | the predicates D-073 already named |
| `shorter_text` | search text has qualifiers | `'exactly two electrons'` -> `'electrons'` |
| `papers_of` | anything was retrieved | you hold N entities; this returns something |

**Why it terminates, demonstrated rather than asserted.** Against a stub model that abstains on every
single turn: search, then challenge, then `drop_kind`, then `papers_of`, then stop. Three pushbacks
and the abstention goes through. `shorter_text` and `compatible_predicate` were correctly skipped as
inapplicable -- only rungs with something real to say are offered.

Three independent bounds, none of them a promise about the model:

1. the ladder is four items and strictly consumed (`session.widenings_used`)
2. `rounds_left >= 2`, so it cannot fire with no room to act
3. an exact repeat is never executed (D-075), so a suggested call that was already made costs nothing

**Every suggestion is concrete.** Never "try harder" -- always a specific call built from this run's
trace. Generic exhortation is what produces flailing; a named alternative produces a different query.

**What it will not do.** It fires only on `not_in_graph`, only with entities in hand, and never on a
real answer. An abstention with nothing retrieved is left alone: there is nothing to widen from and
the refusal is very likely correct. Pushing there would be pushing toward invention.

**How it gets judged, including the failure that would matter most.** Paired against the identical
config without it. Adopt only if `count_correct` and `set_f1` rise, `seconds` does not blow up, and
**the abstention rate does not collapse toward zero**. If abstentions vanish while precision falls,
the system has learned to fabricate coverage rather than find it -- worse than the problem being
fixed, because it is invisible. Faithfulness on the previously-abstaining runs is the specific check.

`widenings_offered` and `widenings_taken` are recorded separately, because a mechanism the model
ignores is a mechanism that does not work -- the same gap flagged for D-073's note.

### D-077 (2026-08-29) — force-critic-set does nothing, and the 8B critic halves counting

Both from the 72B pair on conceptB-200, stopped at 430/600 to free the server for iteration. Scored
on the **426 records every arm completed**, so all four are paired on the same questions.

| arm (planner is always the 72B) | count_correct | set F1 | answered | seconds |
|---|---|---|---|---|
| critic off | 0.034 | 0.182 | 0.930 | 20 |
| 8B critic | 0.080 | 0.394 | 0.920 | 73 |
| 72B critic | **0.144** | **0.433** | 0.880 | 150 |
| 72B critic + force-set | 0.147 | 0.433 | 0.887 | 151 |

**1. `--force-critic-set` is settled: it changes the plan and not the outcome.**

    count questions  +0.0033   148 of 150 tied
    set questions    +0.0002    46 of  49 tied

It is not inert -- **417 of 426 traces differ** -- so the substitution really happens. It simply does
not matter. That closes the question D-060's mechanism analysis opened (the kept set reaches the same
gold papers 96.7% of the time while being 4-7x smaller), and it closes it against the mechanism:
being handed a smaller, better-judged set does not change what the model then does with it. Leave it
off, which is where the recorded default already was.

**2. Moving the critic onto the 8B nearly halves counting accuracy: 0.144 -> 0.080.**

That gap (-0.064) is almost the size of the entire critic-on effect (+0.071 on count). The 8B is
2x faster, so the trade is real: **2x the wall clock for ~1.8x the counting accuracy.**

**And Gabriel's gold said the opposite** -- -0.0035, "no difference". All eight of his questions are
SET questions, where the gap is small (0.394 -> 0.433). The 8B critic's weakness is on COUNTING, and
his set contains none. It is structurally incapable of seeing the difference.

*Third time today a shape split reversed a conclusion*: ranked-vs-shuffled goes opposite ways between
count and set questions (D-073's neighbourhood), the 8B critic looks fine on set and fails on count,
and gf-08 failed on a path no other question exercises. **A single-shape evaluation set is not a
smaller version of a mixed one -- it is a blind one**, and that is now the argument for the mixed
fast set rather than a convenience.

**Decision (Raul's, 2026-08-29): the 8B critic is the standard for iteration anyway**, because 90
minutes per arm against 9 hours is what makes a day of implementation possible at all. The cost is
recorded here rather than absorbed: anything that wins on the 8B critic has to be confirmed on the
72B before it is believed, since a mechanism that only helps because the critic is weak will look
good and then evaporate.

### D-078 (2026-08-29) — the free-SQL control's first transcript, and a sharper claim than the one it was built to test

Smoke-tested against the live 72B before the full run. One question --
*"which analyses use b-tagged jets in their event selection?"*, gold 8 papers -- and three queries:

    1  ... JOIN entity e ON p.arxiv_id = e.paper_id ...  ERROR: no such column: e.paper_id
    2  ... entity_occurrence ... WHERE e.kind = 'selection_requirement'
                                  AND e.label LIKE '%b-tagged jet%'   -> 0 rows
    3  ... WHERE e.kind = 'selection_requirement'
           AND (e.label LIKE '%b-jet%' OR e.label LIKE '%b-tagging%') -> 0 rows

Then: *"there are no analyses in the graph that explicitly mention these terms in their event
selection requirements."*

It recovered from the schema error on its own, which is the mark of a fair opponent rather than a
straw man. What it could not recover from is the **kind**: b-tagging lives under `detector_object`
(19 entities), not `selection_requirement` (3). The equivalent query on the right kind returns 39
papers.

**This is gf-08 again, in a different system.** Same wrong-kind mistake, same `0 rows`, same
confident false negative -- and gf-08 is where all six planner arms scored 0.000. **Both
architectures make this error.**

**What differs is what happens next, and that is the real claim.** After D-073 the typed hop answers:

> 0 rows, and it could not have been otherwise: 'region_requires_object' relates objects of kind
> ['detector_object', 'object_definition'], but this set holds ['selection_requirement'].

SQL returns `0 rows` and nothing else. **There is no way, in SQL, to be told you asked about the
wrong kind** -- the empty set is the same empty set whether the question was wrong or the answer is
genuinely nothing.

So the thesis narrows and gets better:

- *not* "typed tools retrieve more" -- both systems failed this question
- *but* **"a typed layer can explain an empty result; raw SQL cannot"**

That is falsifiable, it is measurable (30 of 75 empty hops were type errors, D-073), and it survives
the Pythia embarrassment where a one-line `LIKE` matched the whole typed pipeline.

**An honest caveat that belongs beside it.** The "correct" query returns **39 papers against a gold
of 8**. Getting the kind right is necessary and nowhere near sufficient; label matching over-returns
by 5x. Neither system is close, and the typed layer's advantage here is diagnostic, not accuracy.

**Harness note.** `FreeSQLSystem` now records every query and its row count. Without the transcripts
"free SQL lost" is unfalsifiable -- a fair defeat and a broken prompt look identical from the score,
and the first question anyone asks about a control is whether it was rigged. The evidence has to
exist before the run rather than after the argument.

### D-078 addendum — the control was handicapped, and fixing it made it beat us

Three defects in the free-SQL control, all found by reading its first transcript rather than its
score. Each is a way the experiment could have been rigged without anyone intending it.

**1. A keyword blacklist rejected ordinary reads.** `WHERE label LIKE '%update%'` and `'%drop%'`
came back "only read-only SELECT is allowed" -- over a corpus whose labels contain English words.
Removed entirely. The connection is opened read-only and a statement that must begin with SELECT
cannot write in SQLite, so the barrier was never the regex; it was pure handicap.

**2. The schema never showed a LABEL.** It listed kinds and predicates, so the agent had to guess
that a b-tagged jet is filed under `detector_object` and written "b-tagged jet (MV2c10, 77%)".
Guessing our filing conventions is not the skill under test. Three real labels per kind now travel
with the schema, which is the analogue of the closed vocabularies the typed agent gets free.

**3. Nothing warned it about the trap it fell into.** It filtered `kind='selection_requirement'`
three times and concluded the data was absent. The brief now says one concept lives under several
kinds, and the prompt says to suspect the filters before concluding absence -- with the single query
that settles it (`SELECT kind, COUNT(*) ... GROUP BY kind`).

**Re-run on the same question, same model.** *"Which analyses use b-tagged jets in their event
selection?"*

| | queries | result | judged F1 |
|---|---|---|---|
| before | 3, all 0 rows | "there are no analyses" | **0.000** |
| after | 2 | 45 papers, 10 named | **0.889** (P 0.80, R 1.00) |

**The typed planner scores 0.364 on this question.** So on this one, the control does not merely
survive -- it beats us by a factor of two and a half.

That is consistent with prediction (1) in the design doc: free SQL wins on literal lookups. "Does the
selection use b-tagged jets" is exactly that. The prediction that matters is still (2) -- synonymy
with no shared substring -- and it is untested.

**The methodological point, which is the durable one.** This was found by reading the transcript, not
the score. A control that scores 0.000 looks like a strong result for the thesis and was in fact a
bug in the control. Anyone who reports a baseline without reading what it actually did is reporting
their own defaults. The full run was relaunched on the fixed code; the handicapped numbers are
discarded rather than kept as a "before".

### D-079 (2026-08-29) — the control matches us on average and fails somewhere else

The free-SQL control, upgraded to hold the SAME raw materials as the planner -- the corpus, the
BM25+dense index, the 72B -- and run against Gabriel's human gold. What it still lacks is the typed
layer itself: predicate-aware hops, the critic, empty-hop diagnostics, automatic canonical expansion,
the answer contract.

    free-SQL + search    judged_f1 0.455    12.1 s    30 search calls, 27 sql calls
    typed planner        judged_f1 0.477    45.1 s

**Within noise on the mean, at 3.7x the cost for the typed layer.** But the means hide the finding:

| question | conditions | free-SQL | planner |
|---|---|---|---|
| gf-02 | 1 | **0.889** | 0.275 |
| gf-07 | 2 | **0.577** | 0.414 |
| gf-01-condition | 1 | 0.364 | 0.364 |
| gf-03 | 1 | 0.889 | 0.889 |
| gf-04 | 1 | 0.800 | 0.800 |
| gf-05 | 2 | 0.118 | **0.435** |
| **gf-01** | **3** | **0.000** | **0.545** |
| gf-08 | 2 | 0.000 | 0.000 |

Three ties, two clear wins each way, one mutual failure. **They do not fail on the same questions.**

The sharpest cell is gf-01, the three-part conjunction and the hardest question in the set: the
control scores **zero**, the planner **0.545**. That is the best evidence for the typed layer in the
project, and it is on labels a physicist wrote.

**So the thesis changes shape, and improves.** Not "typed tools beat a model turned loose on the
data" -- on average they do not, and they cost four times as much. Instead:

> Given the same corpus, index and model, a flat SQL+search agent matches the typed planner on
> average at a quarter of the cost, and fails completely on the multi-condition question the typed
> planner half-answers. **The typed layer's value is concentrated in compositional questions, not in
> retrieval and not in single-fact lookup.**

That is narrower, falsifiable, and consistent with everything else measured: the critic's biggest win
was gf-01 (0.000 -> 0.545, D-073's neighbourhood), and precision falls with conjunct count (D-069,
0.90 -> 0.46 -> 0.30).

**Caveats that travel with it.** n = 8 questions, one observation each over 3 repeats -- the
per-question differences are suggestive, not established. Gabriel's set is all SET questions, so this
says nothing about counting; the fast-set run is where that lands. And the control has now been
upgraded four times in one afternoon, each time from reading its transcripts, so its number is a
moving target in a way the planner's is not.

**What would settle it**: both systems on conceptB-200, stratified by conjunct count. If the gap is
concentrated in the multi-condition questions there too, the claim holds at power.

### D-080 (2026-08-29) — judged_set_f1 was scoring the retrieval footprint, and it inverts D-079

A second planner model found a bug in the scorer, not in itself.

`judged_set_f1` fell back to `a.papers` when an answer named no papers, then restricted to the judged
universe. **The restriction makes the fallback nearly free**: intersecting a 58-paper retrieval
footprint with a 10-paper judged universe recovers every gold paper automatically. gpt-5.6-luna
scored **judged_f1 0.889 on gf-01-condition having written no answer at all.**

This is the mistake D-062 found in `set_f1` -- "score the essay, not the library shelf" -- reproduced
in `judged_set_f1` a week later by the same hand, and made worse rather than better by the universe
restriction that was supposed to make the scorer honest.

**Fixed** by asking where the papers came from. Ids named in prose count. A CITED set counts, because
citing a set instead of retyping ids is v3's entire design and `resolve_citations` puts that set into
`a.papers`. A raw footprint counts for nothing.

**Re-scored, on Gabriel's human gold:**

| system | old | corrected |
|---|---|---|
| typed planner, Qwen-72B | 0.477 | **0.423** |
| free-SQL + search, Qwen-72B | 0.455 | **0.455** |
| typed planner, gpt-5.6-luna | 0.677 | **0.382** |

**free-SQL is unchanged, because it always named its papers.** The typed planner lost 0.054 of
footprint credit and luna lost 0.295.

**So D-079 inverts.** It reported the planner slightly ahead of the control, 0.477 to 0.455. Corrected,
**the control is ahead: 0.455 to 0.423** -- at a quarter of the cost. The per-question finding in
D-079 survives untouched, because it was computed the same way for both systems: they still fail on
different questions, and the planner still wins gf-01, the three-part conjunction, where the control
scores zero.

**On gpt-5.6-luna.** Its real score is 0.382, the lowest of the three, and it is not a weak model --
it found gf-01's answer in round 2 with the correct `facets` call. It explores instead of concluding,
burns 5.9 of 6 rounds, and half the time never writes an answer. Our PURPOSE prompt has had weeks of
tuning against Qwen's habits and none against anything else, which is the most likely explanation and
is itself worth reporting.

**What this run bought for under a dollar.** Four latent harness bugs, none of which Qwen could have
revealed, and every one of which made results look BETTER:

1. a run hitting `max_rounds` discarded everything and scored the footprint
2. the critic ran on the wrong API key, 401 on every chunk, defaulting all candidates to kept -- a
   run labelled critic-on that ran critic-off
3. the `answer` tool had no `required` array, so a model could call it with nothing
4. `judged_set_f1` scored the footprint

Single-model evaluation does not merely limit generality. **It lets harness bugs hide behind one
model's habits**, and every one of these flattered us.

## D-081 — a value row shows the sentence containing the value, not the retrieval footprint

2026-08-30.

The seven value rows carried `all_quotes` from the run: the sentences the agent
READ on its way to an answer. That is the retrieval footprint, and it is not
support. Three of the seven stated a number appearing in none of the sentences
shown beside it — gf-12 asked "is 875 GeV right?" beside three quotes about
figure numbering and SR binning; gf-14 and gf-15 the same. The supporting
sentence existed in the paper in every case ("Masses of the t̃2 up to 875 GeV
are excluded at 95% CL…"). We were showing the wrong sentences, not missing
ones.

This is the SAME error as D-062 and D-080 in a third place: treating what the
system touched as what the system proved. It is worth naming as a pattern —
footprint is not evidence — because it has now been made three times
independently.

`eval/value_evidence.py` searches the paper for sentences containing the values
the answer commits to. A whole number must carry a unit to count, or citation
brackets and author affiliations match ("[ 14 , 15 ]" answered a question about
15 GeV; "Phys. Rev. D 60 (1999)" answered one about 60% b-tagging). Decimals are
exempt — the yields table writes 5.7 ± 1.0 bare. Sentences rank by how much of
the answer they account for, not one per anchor. A prose answer with no value
to anchor to returns nothing and keeps the run's quotes.

Also fixed, all found only by reading the seven rows end to end:
`\mathchoice{a}{b}{c}{d}` printed its own name and all four arguments, so
χ̃⁰₂ appeared four times in gf-12; our answer was appended to `question` as
HTML and the app escaped it, so the sheet read "875 GeV&lt;/div&gt;"; and
`\prime` rendered as the word "prime" in "bffprimeχ̃01".

Batch 2 stands at 104 rows, sha256 892867990155dd54. Still not sent.

## D-082 — the idle GPUs are MIG slices, and the queue is not the bottleneck we thought

2026-08-30.

QwQ has never run. Job 53993 sat `PENDING (Priority)` with a start estimate of
2026-09-02: another user's array `TF_exp9_scale` holds all three cards of
compute-gpu-0-1 with a **3-day limit per element** and five more elements queued
behind. The `GPU` partition is that one node. Nothing was wrong with our job.

`LIGHTGPU` looked like the answer — compute-gpu-0-0, `gpu:a100:6`, fully idle,
`AllowAccounts=ALL`, PriorityTier 1000 — and a job submitted there started
instantly. It is not the answer. The node is **MIG-partitioned**: three
A100-40GB cards each split into two `3g.20gb` instances. Six allocatable GPUs,
20GB each, and MIG instances cannot be pooled, so vLLM cannot tensor-parallel
across them. Against ~18GB usable:

    QwQ-32B-AWQ            19G   does not fit
    Qwen3.6-27B            52G   not quantised, needs two whole cards
    Qwen3-Coder-Next-FP8   75G   no
    Llama-3.1-8B           15G   fits -- the critic, and nothing else

So LIGHTGPU can host our critic and none of our planners. Recorded because the
partition will keep looking free.

The failure was unreadable, which is the part worth fixing. Slurm hands out
slice ids (`CUDA_VISIBLE_DEVICES=12,21`) that `nvidia-smi -i` rejects, so the
ECC probe returned "No devices were found" for every card and the job stopped
with "need two clean cards, found 0". A sizing problem wearing the costume of
D-040's dying card. Both server scripts now refuse a MIG node by name.

Second bug, found in the same job and older than it: the scripts source `.env`
with `set -a` BEFORE reading `LLM_MODEL_NAME`, so `sbatch --export` was silently
overwritten by the 72B pinned in `.env`. Today's QwQ attempt asked a server for
QwQ that had been told to load the 72B — one record, empty answer, and no line
in the log naming the weights. Precedence is now submitted env > .env >
default, and both scripts echo the model they are serving.

53993 was cancelled while chasing LIGHTGPU; 53997 restored the place with the
same 2026-09-02 estimate, so no priority was lost. QwQ on-prem is 3 days out.

## D-083 — the Aug 29-30 runs, re-scored: luna was never strong

2026-08-30.

130 run files sat on the cluster, never synced, carrying run-time scores only.
`rescored` was null on every record, so the D-080 footprint correction had never
been applied to any of them. Pulled and re-scored against the merged question
set; originals kept alongside as `.preD080`.

**6 of 32 Gabriel-question runs changed, and they are all one model.**

    luna 20:32   0.666 -> 0.000    24 of 24 answers empty
    luna 20:49   0.635 -> 0.214    12 of 24 empty
    luna 21:04   0.677 -> 0.382    10 of 24 empty
    luna 20:45   0.708 -> 0.388     2 of 4  empty
    Qwen 15:04   0.477 -> 0.423   (x2, same run repeated)

gpt-5.6-luna had looked like the best typed model at 0.666-0.708. It was
scoring the retrieval footprint: no answer text, no citation, and the same
three arXiv ids returned for every question. Corrected, it lands at 0.382 --
which is exactly the luna/typed 0.382 already in the corrected sweep, so the
two independent paths now agree.

Everything else was already right: free-SQL (0.669, 0.675 on sol), deepseek
typed 0.512, sol typed 0.287, and all eight Aug 30 Qwen arms unchanged. The
headline ordering survives -- free-SQL beats typed on every model tested.

THE TELL IS THE EMPTY COUNT. Every run that moved had empty answers, and the
size of the drop tracks how many. `named_none` was already the best predictor
of score; this says the same thing from the other side. Worth a guard: a run
whose answers are mostly empty should not report a score at all without saying
so, because the number it reports is about retrieval and reads as competence.

## D-084 — the completion cap was throttling every reasoning model; hosted re-runs dropped

2026-08-31.

`MAX_COMPLETION_TOKENS = 800` (planner.py:100) was sized for a 72B AWQ on one
A100 at ~26 tok/s, where any completion past ~3,100 tokens times out, retries and
regenerates. It is the wrong constant for a hosted reasoning model, which spends
that budget on `<think>` BEFORE it can emit a tool call. Truncated mid-thought,
the model returns an empty message with no tool call; the harness records
`answered=True` with empty text; `papers` still holds the retrieval footprint;
and the scorer correctly gives it zero. A plumbing limit therefore arrived
looking like a model that could not answer.

Same model, same questions, same prompt, cap 800 -> 4000:

    qwen3-32b free-SQL   0.199 -> 0.592     empty answers 13/24 -> 0/24
                                            named_none    0.667 -> 0.125

`budget.py` had already written this down in its REASONING_MODELS comment --
"max_tokens set for Qwen can leave no room for an answer after the reasoning is
spent". The knowledge was in one module and the constant in another. The fix
belongs in code, not in an env var passed by hand.

**Consequence for the record**: every hosted number measured before this is a
FLOOR, not a score -- sol 0.669, deepseek 0.512, luna 0.382. Qwen2.5-72B is
unaffected (not a reasoning model, 800 was always enough), so the local baseline
stands.

**Decision: the hosted re-runs are dropped**, paid models to be revisited later.
The deepseek re-run timed out on 9 of 24 records with `rounds=0, calls=0` while a
direct call to the same model answered in 4.0s, so the client timeout/retry path
is at fault and is undiagnosed. Its file is in `eval/runs/quarantine/` with the
reasoning; neither its 0.218 (nine zeros) nor its 0.655 (eight survivors,
selected by having completed) may be quoted.

**Process note, worth more than the finding.** The re-runs were launched as one
five-arm sequential chain and left unwatched for ~19 hours; deepseek's failure
blocked both luna arms behind it. That is exactly the shape the nightly-batch
convention exists to avoid -- separate jobs, so one failure does not take the
batch down. Cost of the whole episode: ~$0.28, because timeouts bill nothing.

## D-085 — 31% of the graph was unsearchable; the fix helps the typed planner and hurts the control

2026-09-01.

An assertion's object is either an ENTITY or a LITERAL VALUE. 4,357 of 14,188
(31%, ~279,000 characters) are literal -- the selections, the reported
quantities, the region definitions. The retrieval index was built from
`entity_occurrence` alone, so it held 5,434 labels saying WHAT THINGS ARE CALLED
and nothing saying WHAT IS TRUE OF THEM.

    search("exactly two electrons")   0 label matches
    the requirement, in the graph:    "HLT: two electrons with pT > 33 (25) GeV"

gf-08 has 13 gold papers and the typed planner returned 0. It could not have
done otherwise: there was no path from the question to the fact. The critic
could not help either -- `_render_candidates` shows it `label [kind] facets`
and no quotes, so the same text was invisible to it. Three layers, one blind
spot; only free-SQL reached it, with `LIKE`.

`--index-values` indexes each value as another surface form OF ITS SUBJECT
ENTITY (8,509 -> 12,758 forms). Because `label` becomes the matched form, the
critic starts seeing "exactly two isolated oppositely charged electrons/muons"
where it saw "Electron".

**It is not a global default, because it moves the two systems in opposite
directions:**

    typed      0.352 -> 0.438   +0.086   precision AND recall both up
    free-SQL   0.592 -> 0.509   -0.083   precision up, recall down
    the gap    0.240 -> 0.071   -70%

free-SQL already reached that text with `LIKE`; adding value forms changes what
`search` RETURNS, so its vocabulary discovery degrades -- it reads value strings
where it used to read entity names, and builds its patterns from the wrong
material.

**The consequence for the write-up is larger than the arm.** Most of the
typed-vs-free-SQL gap was this index defect, not the query paradigm. "Declarative
beats procedural" was the wrong reading of a number that was mostly one missing
third of the index.

Not yet adopted: it needs `paperA-200` before it becomes a per-system default,
and adopting it re-baselines every arm measured against the old index.

## D-086 — these mechanisms are compensations, not improvements

2026-09-01.

Nine arms across four mechanisms, one model (qwen3-32b), one question set, n=24
each. Every arm's SIGN is predicted by the strength of the baseline it was added
to, and nothing else:

    baseline 0.352 (typed)       4 arms   ALL POSITIVE   mean +0.056
    baseline 0.438 (typed+idx)   1 arm    negative       -0.044
    baseline 0.592 (free-SQL)    4 arms   ALL NEGATIVE   mean -0.108

The cleanest pair is one flag against two baselines: `--subgoals` is +0.024 on
0.352 and -0.044 on 0.438. Same flag, same model, same questions.

9 of 9 in the predicted direction is roughly p=0.004 under random signs. The
Pearson r of -0.939 OVERSTATES it -- there are only three distinct baseline
levels, so it is three group means, not a continuous relationship. Report the
sign consistency, not the r.

**Why it happens.** Every mechanism tried -- an objective block, a plan
reviewer, question decomposition -- makes the agent commit earlier and narrower.
Measured on free-SQL: precision rises, recall falls, F1 falls. That is the right
trade for a system that answers with one entity and the wrong one for a system
scored on set coverage, where `judged_set_f1` restricts to the judged universe
and breadth is nearly free (gf-01-condition named 44 papers for 8 gold and
scored precision 0.80).

**This cuts against PoG (NeurIPS 2024)**, whose Guidance/Memory/Reflection are
presented as broadly beneficial. Their baseline, ToG at 57.1, explores with a
fixed breadth and cannot self-correct -- it is the weak configuration. We are
seeing what those mechanisms do when the baseline is already good.

CAVEAT, and it is not small: `--subgoal-status` -- PoG's HIGHEST-value mechanism
(-4.3 when removed, against -3.1 for Guidance) -- was still running when this was
written. What has been tested is `--subgoals`, which is their `w/o Memory`
variant. The pattern above may not survive it.

## D-087 — the encoder loses "Higgs" at the tokenizer, and no amount of size fixes it

**The question**, raised 2026-09-01: `Higgs candidate` retrieves every other
candidate, so is the encoder too small? Try a bigger one.

**The answer: size is irrelevant, and the failure is upstream of the vector.**
`bge-base` has no `Higgs` token. It splits the word into `hi` + `##ggs` — the
greeting, then a fragment. The model is matching roughly *"hi ggs candidate"*,
where the only surviving signal is `candidate`, which is exactly why every
candidate ranks alike. The information is destroyed by the tokenizer, before
any of the 768 dimensions get to see it. Adding dimensions cannot recover a
word that never reached the model.

**The measurement that settles it.** Separation = mean cosine to correct
surface forms minus mean cosine to confusable ones, over three discriminations
this corpus actually needs (Higgs candidate / b-tagged jet / ttZ control region):

    kipark chATLAS mpnet      768   Higgs +0.076   ttZ +0.248   mean +0.146
    all-mpnet-base-v2         768         +0.056                mean +0.132
    physbert_cased            768         +0.116   ttZ +0.097   mean +0.111
    physbert_uncased          768         +0.022                mean +0.104
    all-roberta-large-v1     1024         -0.009   ttZ +0.205   mean +0.094
    mxbai-embed-large-v1     1024         -0.019                mean +0.088
    bge-large-en-v1.5        1024                               mean +0.083
    bge-small-en-v1.5         384                               mean +0.077
    bge-base-en-v1.5 (ours)   768         +0.006                mean +0.071
    Qwen3-Embedding-0.6B     1024         -0.040                mean +0.057
    e5-large-v2              1024         -0.041                mean +0.023
    scibert (mean-pool)       768         -0.092                mean -0.010

**Every 1024-dim model is NEGATIVE on the Higgs case** — they rank
`electron candidate` above `H->bb candidate jet`. The decisive control is
`all-roberta-large-v1`: same sentence-transformers training recipe as
`all-mpnet-base-v2`, 1024 dims against 768, and it scores WORSE (+0.094 vs
+0.132). Same lineage, scaled up, degraded. Dimension-vs-separation across all
tested models is r = +0.065, i.e. nothing.

**What does work is domain pre-training, and the mechanism is legible.**
PhysBERT (BERT pre-trained from scratch on arXiv physics) keeps `Higgs`,
`boson`, `quark`, `luminosity`, `pseudorapidity`, `calorimeter` each as ONE
token. Its vocabulary is 30,522 — *identical* to bge-base. It did not buy a
bigger vocabulary; it spent the same budget on physics instead of general
English. That trade alone takes the Higgs case from +0.006 to +0.116, the best
of anything tested. Compare SciBERT, same architecture and same mean-pooling,
trained on general science: dead last at -0.010. Science is not the domain;
physics is.

**The two leaders fail on different axes, and the split is explanatory.**
PhysBERT wins on particle names (Higgs +0.116 vs +0.076); the chATLAS encoder
wins hard on analysis jargon (ttZ control region +0.248 vs +0.097). PhysBERT
read papers; kipark's model was fine-tuned on ATLAS twiki, chat and git. `CR-ttZ`
is twiki vocabulary, not paper vocabulary. Neither dominates, so the encoder
choice is an empirical arm, not an argument — both are running on Gabriel gold.

**`b-tagged` splits in every model tested**, PhysBERT included: the hyphen
forces it. That predicts, correctly, that b-tagging is the one discrimination
where PhysBERT has no edge, and it means hyphenated detector jargon stays a
lexical-retrieval problem no encoder swap will solve.

**PhysBERT is RETRIEVAL-ONLY.** It is a raw `transformers` checkpoint with no
trained pooling layer, so sentence vectors are mean-pooled, and mean-pooled BERT
is anisotropic — everything lands in a narrow 0.62-0.68 cosine band. Ranking is
unaffected, but the alias-merge boundary near 0.861 is a *bge-base constant*
(`semantics.py`). Swapping PhysBERT in there without re-tuning would merge
nearly everything, silently. Retrieval ranks; aliasing thresholds.

**Credit where due:** the user proposed the rare-word hypothesis unprompted, and
I had earlier dismissed changing the encoder on the grounds that the signal was
not there. The signal was there. bge-base was not trained to encode it.

## D-088 — six QwQ arms ran with `--critic` set and no critic alive

**What happened.** Jobs 54044-54049 (2026-09-01, ~1h each) **and 54042** each
requested a judge model that the judge server did not have. Port 8001 was
serving `Qwen/Qwen3.5-9B`; the client asked for
`NousResearch/Meta-Llama-3.1-8B-Instruct`. Every judge call returned 404. In
54046: **118 calls to 8001, 0 successful** — 97 "critic chunk failed", 21
"facet reader chunk failed". All seven jobs show the same pattern.

**THE SCRIPT WAS NOT ALWAYS WRONG, AND THAT IS THE LESSON.** It hardcodes the
Llama name, and port 8001 *served exactly that* for weeks — 54009 made 68
successful judge calls, 53980 made 635, both with 0 failures. The hardcoding
became wrong the moment I repointed 8001 at Qwen3.5-9B without changing what
the script asks for. A hardcoded name is a latent break waiting for the
environment to move under it.

**A REPORTED FINDING WAS AN ARTEFACT.** 54042 (`--critic`, 91 404s, 0 OK) was
compared against 54043 (genuinely `--critic` off) and reported as: *the critic
makes no F1 difference on QwQ (0.511 vs 0.519) but faithfulness rises
0.728 -> 0.908 with Qwen3.5-9B*. Both halves are void. "No F1 difference" is
two critic-OFF runs scoring alike — the expected result, not a discovery. The
faithfulness gap cannot be attributed to the critic that never ran; 54042 also
carried `--critic-seed` (row shuffling) and 54043 had no `--critic` at all, so
they differ in more than one way, and 0.728 carries a +/-0.13 bar.

**What is unaffected:** QwQ typed **0.519** (54009) and **0.477** (53980) had
real, working Llama-8B critics. Those stand. Nothing attributed to
Qwen3.5-9B before 54050 ever happened.

**Why it got through.** Two independent guards each had a hole, and the holes
lined up.

1. `hpc/gabriel_arm_job.sh` **hardcodes** `CRITIC_MODEL` in the `*-8b` arm
   table, overwriting whatever `sbatch --export` passes. This is D-082 exactly
   — submitted value silently discarded — and the fix written for D-082 was
   applied to `LLM_MODEL_NAME` and **never to `CRITIC_MODEL`**, ten lines away
   in the same file.
2. The `wait_for` gate proves *a server answers at the URL*. It says nothing
   about the model NAME in the request body. A healthy server serving the wrong
   model passes the probe.

So the run started, completed, scored, and reported — labelled critic-on.

**Why it matters more than a normal bug.** The critic ablation is the largest
single effect measured on this project: **+0.231** judged F1 (0.411 with,
0.180 without). These are not slightly-off numbers; they are critic-off runs
wearing a critic-on label. It also explains the QwQ baseline reading 0.388 here
against 0.519 measured earlier with a working critic — a 0.131 gap, the right
order for a missing critic.

**What is salvageable.** All six failed identically, so they remain internally
comparable *as a critic-off series*. The +0.084 that `--path-tool` shows over
baseline (0.472 ± 0.009 vs 0.388 ± 0.027) is a real critic-off effect and the
tightest error bar of the night. It does NOT establish what the path tool does
with a working critic, and none of the six may be compared against any
critic-on number.

**Fixed** in `hpc/gabriel_arm_job.sh`: `_OVERRIDE_CRITIC` restores the submitted
judge after the arm table, mirroring `_OVERRIDE_MODEL`; and a new
`model_served()` asserts the exact model id we will send appears in the
server's `/v1/models`, refusing to start otherwise (exit 4). Verified on 54050:
`answerer serves 'Qwen/QwQ-32B-AWQ'` / `judge serves 'Qwen/Qwen3.5-9B'`.
All six arms resubmitted as **54050-54055**.

**The rule this earns.** A liveness probe must assert the CONFIGURATION, not
the connection. Any check that would pass against a healthy server running the
wrong thing is not a check. Third time an env-var override has silently changed
what a labelled run actually ran (D-082, D-084, here) — the pattern is that the
label is written from the submission and the behaviour from the environment,
and nothing compares the two.

## D-089 — a night of arms, mostly uninterpretable, and the protocol that follows

**What happened.** 2026-09-01: ~40 arms across 14 scripts, 11:00 to 23:57,
while shared code changed four times (schema card 17:39, retrieve + free_sql
18:34, templates 18:47, planner 18:48). Each arm is a fresh Python process that
re-imports, so **arms in the same script ran different software**. The free-SQL
baseline was measured four times: **0.592, 0.526, 0.473, 0.465**. A 0.127
spread in the reference line, against same-arm repeat variability of 0.004-0.073
and effects being chased of ~0.04-0.06. Most of the night cannot be read.

**The root cause is one habit:** putting a change in the frozen path instead of
behind a flag. The kind-semantics schema card went in unflagged at 17:39 and
silently moved every run after it, in both directions (typed up ~0.10, free-SQL
down ~0.07). The project's own convention -- *default OFF, one flag, one
guarded branch* -- exists in two idea docs and was not followed.

**Two provenance gaps made it unrecoverable after the fact.**
`ALIASES_EMBED_MODEL` was never recorded in `_ENV_CONFIG`, so a run file cannot
say which encoder produced it -- the encoder arms could not be reconstructed
from their outputs. Same class as the `index_values` gap that corrupted two
analyses and produced a spurious r = -0.945.

**THE BIGGER ERROR, and it is a design error not a bookkeeping one.** Every
encoder arm ran as plain `--system free-sql`. But `--search-sets` sets
`SET_CAP = 400`: with it, retrieval goes 400 deep and the whole match set is
materialised as a temp table the model filters in SQL; without it, `limit=20`
and the model pastes ~3 ids. **Encoder quality below rank ~3 is invisible
without search-sets.** So the encoder was tested in the only configuration
where it cannot matter -- and then reported as "no effect".

Direct evidence it does matter, measured 2026-09-02 on the `Higgs candidate`
query at rank 5-9:

    bge-base   lepton candidate, $W$ boson candidate, proton candidate,
               Photon candidate, leptonic particle candidate
    chATLAS    (none -- zero cross-particle contamination in the top 15)
    PhysBERT   finds `h candidate` TWICE, which neither other model surfaces
               at all, and which no LIKE '%Higgs%' can reach

All three put the right answer at ranks 1-3, which is why end-to-end F1 could
not see the difference. The gain lives in the tail, and only search-sets reads
the tail. **Retrieval improved; the metric was blind to it.** That is also
consistent with the trace finding that 100% of named papers were retrieved but
only 22.8% of gold were named -- a SELECTION failure downstream of retrieval.

**THE PROTOCOL, five rules, each earned by a named failure.** Implemented in
`eval/batches/2026-09-02-frozen-baseline.sh`:

    1. One script, nothing else running, no edits under hepcoveragekg/ during.
    2. The control runs FIRST AND LAST. If they disagree by more than the noise
       floor, the batch drifted and nothing in it is believable. Drift is
       measured, not assumed away.
    3. One seed everywhere (mixing seeded/unseeded broke sets_v2).
    4. Every experimental change behind a flag, default OFF. KIND_SEMANTICS is
       now one.
    5. Anything that changes behaviour is recorded in `_ENV_CONFIG`.

**Standing prediction, recorded before the run so it cannot be adjusted after:**
chATLAS beats bge-base by MORE with `--search-sets` than without, and the gain
appears as **precision** rather than recall -- search-sets v1 cost precision
0.704 -> 0.602 because the model joined the ~400-row table with no WHERE and
"could not see that the set was heterogeneous". A cleaner 400-row set is
exactly the fix for that, and a cleaner set is what the encoder buys.

## D-090 — with a live critic, the QwQ arms reorder, and one flips sign

The 2026-09-01 QwQ arms were rerun as 54050-54055 after D-088, this time with
the judge verified alive (67-118 successful calls each, zero 404s):

    arm                        dead critic    LIVE critic    change
    baseline                       0.388         0.398        +0.010
    + index-values                 0.367         0.366        -0.001
    + path-tool                    0.472         0.441        -0.031
    + path-tool + index-values     0.356         0.428        +0.072
    + index-quotes                 0.405         0.372        -0.033
    + subgoal-status               0.430         0.333        -0.097

**`--path-tool` survives**, and remains the best QwQ arm: +0.043 over baseline
(0.441 vs 0.398). Smaller than the +0.084 the broken run showed, and now inside
its own error bar (+/-0.073), so it is suggestive rather than established.

**`--subgoal-status` REVERSES.** It read +0.042 over baseline with a dead
critic and reads **-0.065** with a live one -- a 0.097 swing, the largest
critic-dependence of any arm. It looked like PoG's best mechanism working; it
was PoG's best mechanism filling a hole left by a missing critic. On this
evidence the two do the same job and the critic does it better.

That is the third distinct reading of `--subgoal-status`: +0.104 on typed
against a since-deleted baseline (2026-09-01), then 0.456/0.403 on repeat, now
-0.065 on QwQ with a working judge. **No arm in this project has been more
sensitive to what else was true at the time.** It is in the frozen batch
(B3) for a fourth, controlled reading; until that lands nothing about it should
be claimed in either direction.

**`--path-tool + index-values` was genuinely corrupted** (0.356 -> 0.428): the
one cell whose repeats collapsed to a single value under the broken critic, and
the reason the missing error bar was worth chasing.

**Method note.** Every number in the left column was reported to the supervisor
track as a result on 2026-09-01. The rule that follows is in D-088: assert the
configuration, not the connection.

## D-091 — the drift check fired: the typed half of the clean batch is void

Job 54057 ran the D-089 protocol: 15 arms, one script, one code state, one
seed, control FIRST and LAST. The bracketing controls:

    A1  free-SQL control   0.471      A1'  0.486     drift 0.015   PASS
    B1  typed    control   0.336      B1'  0.464     drift 0.128   FAIL

**The typed control moved 0.128 between the first and last arm of the same
script, on identical code, with the same seed.** By the rule written into the
protocol before the run -- *if the pair differs by more than the noise floor,
no arm in it should be believed* -- every typed arm in this batch is
unreadable. That includes B5 `--index-values` at 0.522, which I had already
reported as "the strongest result we have". It is not established.

**The free-SQL half PASSES** (drift 0.015) and its arms are readable, against a
control mean of ~0.478:

    + concept-prompt          0.599    +0.121   <- survives, largest effect
    + index-quotes            0.518    +0.040
    + chATLAS + search-sets   0.513    +0.035
    + kind-semantics          0.503    +0.025
    + chATLAS alone           0.480    +0.002
    + index-values            0.467    -0.011
    + search-sets alone       0.404    -0.074

`--concept-prompt` is the only free-SQL arm clearing the drift band, and it is
a PURE PROMPT CHANGE -- no retrieval difference at all.

**THE PATTERN WORTH CHASING: the ARM is stable and the CONTROL is not.**

    typed control      0.378 (09-01)   0.336 (B1)   0.464 (B1')   range 0.128
    typed + idxvalues  0.512 (09-01)   0.522 (B5)                 range 0.010

Two independent readings of the arm land within 0.010 of each other, and BOTH
sit above the highest of three control readings. That is suggestive that
`--index-values` helps AND that it damps variance -- but the effect SIZE cannot
be quoted while the baseline it is measured against spans 0.128.

**What this changes about method.** The protocol's four other rules were about
keeping conditions identical; they all held here and it was not enough. The
typed planner is simply high-variance at n=24 (8 questions), and one repeat of
a control is not a control. **Bracketing controls are now mandatory, and the
typed system needs repeats of the CONTROL, not just of the arms.**

**Cross-check from dev-200** (54066-69, 201 records each): those runs cannot
compute `judged_f1` at all -- it needs the judged universe -- so they did not
confirm the headline metric. What they show is a shape split, since
gabriel-gold is 100% `set` while dev-200 is 93 set / 107 count:

    --index-values, count questions:  count_correct 0.794 -> 0.692  (-0.102)

Helping set questions while hurting counts is coherent -- 4,249 extra value
surface forms make papers easier to FIND and harder to COUNT -- and it is the
one durable thing to come out of the dev-200 run. Both halves need repeats.

## D-092 — batch 2 is merged into the gold set, not held out

**Decision (user, 2026-09-02): do not hold batch 2 out.** Ground truth is the
scarce resource; a hold-out we cannot afford is worth less than an eval set we
can use. Train/test separation will be reconstructed later from a fresh batch.
This overrides the promise in `eval/review/batch2/MESSAGE.md` ("I would rather
not look at it until the configuration is frozen"), knowingly and on the record.

**What came back:** 51 of 104 rows, stopping at a CLEAN FAMILY BOUNDARY --
gf-01-cond 35/35, gf-02 6/6, gf-04 10/10 complete; gf-05, gf-07, gf-08 and the
value questions entirely untouched. So the merged gold sets are complete for
what they cover, not truncated mid-question. The unjudged 53 rows are the HARD
families, so anything measured here is an optimistic bound.

**The mistake worth recording.** I first built batch 2 as a SEPARATE eval set.
Three of its four questions duplicated `gf-01-condition`, `gf-02` and `gf-04`
with SMALLER gold and universes than the file we already had -- strictly worse,
and I ran 35 minutes of arms against it before the user asked whether those
questions were already in batch 1. They were. The right operation was a MERGE.

**Merged** into `eval/questions/gabriel-gold-2026-09-02-merged.jsonl`:

    gold rows      78 -> 93   (+15)
    judged papers 174 -> 209  (+35)

    gf-01-condition   gold  8 -> 11   universe 10 -> 18
    gf-02             gold  9 -> 11   universe 14 -> 19
    gf-04             gold 11 -> 18   universe 17 -> 26   (+50% universe)
    gf-01-met         NEW   gold  3   universe 13

`gf-01-met` is the one genuinely new question: batch 2 split gf-01 into one
condition per row, the b-tag rows merged into `gf-01-condition`, the
"is it a search" rows came back ALL NEGATIVE (no positives, set F1 undefined,
dropped), and MET alone had never been asked. Only 3 of 13 positive, so it is
the most fragile of the nine -- one paper moves it materially.

**A GROUND-TRUTH CONFLICT, and it is not noise.** `2012.01581` on gf-02:
**batch 1 "no", batch 2 "yes"**, same judge, same question, different evidence
snippet. His batch-2 note reasons it out -- *"matrix method with f = fail and
p = pass and there being 3 leptons"*. Resolved BATCH 2 WINS (later, and
reasoned); confirmed by the user. The general lesson is larger than the row:
**our ground truth is snippet-dependent at roughly 1 in 20.** Gabriel said as
much himself on row 1011 -- *"If this is ground truth for the paper, the correct
answer is yes. If this is ground truth for this snippet alone, then no"* -- and
our own brief specified per-snippet ("one condition each, with its own quote").

**Error structure, for later, NOT acted on as a fix** (acting on it is what
would have burned the set as a measure): errors split perfectly by question --
gf-01-cond 8 misses and 0 over-claims; gf-02 and gf-04 6 over-claims and 0
misses. Named causes from his notes: `veto = use` (2 rows), symbols not read
(`pTmiss` in notation, 1 row), a plain reading miss (1 row), and domain
inference no quote supports (2 rows). **Our critic agreed with him on 22 of 23
rows where it ruled (0.957)** -- the judge is not the weak part.

## D-093 — two identical QwQ runs differ by 0.113; the arm spread is 0.048

Four QwQ arms on the merged gold set (2026-09-02), submitted together, running
in PARALLEL on the same servers with the same code and the same seed:

    54074  CONTROL          0.407 +/- 0.096    critic calls 103
    54075  + path-tool      0.424 +/- 0.03     critic calls 144
    54076  + index-values   0.376 +/- 0.064    critic calls 147
    54077  CONTROL REPEAT   0.294 +/- 0.1      critic calls  49

**The two identical controls differ by 0.113.** The whole spread between the
three distinct arms is 0.048 -- less than half the gap between one arm and
itself. The critic-call counts differ by 2x for the same configuration (103 vs
49), so the runs did materially different amounts of work.

**This is stronger than the sequential drift check in D-089.** Those two
controls were separated by ~10 hours of wall clock, so drift could be blamed on
time or on the machine. These were parallel: same instant, same node, same
weights, same questions. Nothing is left to blame but **run-to-run sampling
variance in the model itself**.

**Consequence: no QwQ arm measured at n=27 x 3 repeats is interpretable**, and
that retroactively covers `--path-tool`, whose +0.084 (dead critic, D-088) and
+0.043 (live critic, D-090) and +0.017 (here) have now been three different
numbers across three runs. It was never established.

**What this actually demands.** The fix is not more arms, it is more SAMPLES per
arm -- repeats well above 3, or a question set large enough that per-question
variance averages out. 9 questions is not that set even at 209 judged papers.
The honest statement of the project's measurement floor: **on gabriel-scale
question sets we can detect effects of roughly 0.15 and nothing smaller.**
`--index-values` on typed (+0.134 and +0.186, twice, same direction, precision
AND recall both up) is the only arm that has ever cleared a bar like that.

**Method rule, promoted:** run the control TWICE IN PARALLEL in every batch, not
just first-and-last. It costs one arm and it prices the noise floor directly
instead of assuming it.

## D-094 — the runner answers questions concurrently; and index-values does not replicate

**The harness was strictly serial.** `runner.run` looped `for repeat: for q in
questions:`; the only `ThreadPoolExecutor` in the file was a timeout wrapper
with `max_workers=1`. Since a question is ~100s of network wait, every arm cost
`n x repeats x 100s`: a 300-question x 3-repeat arm was 26 hours.

    workers=1   941s      (9 questions, 1 repeat)
    workers=9   198s      4.75x

198s against a ~104s mean means the wall is now bounded by the SLOWEST QUESTION
rather than the sum, i.e. it is already near-optimal at this size, and the win
GROWS with the set: 300 x 3 at 12 workers is ~5h instead of ~26h.

**THE PART THAT WOULD HAVE CORRUPTED RESULTS SILENTLY.** The systems are not
thread-safe and do not fail loudly. `FreeSQLSystem` materialises each search as
a temp table `search_N` on its connection, bumps `self._set_n`, and calls
`_drop_sets()` between questions -- its own docstring already names the serial
hazard ("question 12 could join against question 3's search set and score on
it"). Threaded on one connection: the counter races, names collide, and one
question's `_drop_sets()` drops a table another is mid-join on. The answers come
back PLAUSIBLE AND WRONG.

So `--workers N` requires a FACTORY: each worker builds its own system and its
own sqlite connection. Asking for concurrency without one raises `ValueError`
rather than corrupting. The retrieval index IS shared, deliberately -- 44MB, and
`build()` calls `_prepare_sparse()` eagerly so it is immutable when queried.
Records are written in COMPLETION order; every consumer groups by (qid, repeat),
so this costs nothing but the files are no longer question-ordered.

## The measurement that matters more than the speed-up

Seven arms, qwen3-32b, merged gold set (9 questions, 209 judged papers),
5 repeats, all seven in parallel, BOTH CONTROLS DUPLICATED per D-093:

    freesql CONTROL-a   0.446 +/- 0.10     typed CONTROL-a   0.358 +/- 0.03
    freesql CONTROL-b   0.396 +/- 0.068    typed CONTROL-b   0.408 +/- 0.084
            delta 0.050                            delta 0.050

**Both duplicate pairs disagree by exactly 0.050, independently.** That is the
noise floor on this set -- much better than QwQ's 0.113 (D-093), so repeats=5
plus the larger universes bought real precision.

    concept-prompt   0.493   +0.072 vs control mean   ABOVE noise
    index-quotes     0.374   -0.047                   inside noise
    index-values     0.361   -0.022                   INSIDE NOISE

**`--index-values` DOES NOT REPLICATE.** It read +0.134 (2026-09-01) and +0.186
(frozen batch) on the OLD 8-question set and reads **-0.022** on the merged set.
It was reported twice as the one arm that had cleared the bar; on a better
instrument it has not. The difference is the instrument, not the code: better
ground truth (209 judged papers vs 174), a ninth question, 5 repeats instead of
3, and a control measured twice instead of assumed.

**`--concept-prompt` is now the only mechanism standing** (+0.072 against a
0.050 floor -- marginal, not established), and it is a pure PROMPT change with
no retrieval component. Every retrieval-side mechanism tried so far sits inside
the noise.

## D-095 — 86% of our questions cannot tell two arms apart, and Gabriel's are the exception

Paired control on QwQ, 200 stratified questions, two identical runs per system
(2026-09-02). 104 questions produced a comparable score in both runs.

**NOISE FLOOR vs QUESTION COUNT** -- bootstrapped from the two runs, so it cost
nothing beyond the runs themselves. 95th percentile of |mean(a) - mean(b)| over
random subsets; an effect must EXCEED this to be real at that n:

        n      typed    free-SQL
       25      0.080       0.151
       50      0.049       0.105
       75      0.035       0.062
      100      0.024       0.029

This is the number the whole session was missing. At n=9 the floor was 0.050;
at n=100 it is ~0.03. **Roughly 100 questions buys +/-0.03**, which is what
turns a 0.07 effect from "suggestive" into "measured".

**THE FINDING THAT MATTERS MORE.** Sort each question by whether the two runs
agree and where it scores:

    typed      always-0 48%   always-1 33%   DISCRIMINATING 14%   unstable  5%
    free-SQL   always-0 51%   always-1 15%   DISCRIMINATING 10%   unstable 24%

**Only 10-14% of questions can distinguish one arm from another.** A question
everything gets right, or everything gets wrong, contributes variance and no
signal. We have been paying for ~90% dead weight in every arm ever run.

**By source, and this settles an argument:**

    pool        n    discriminating (typed)
    gabriel     8         75%     <-- human truth
    tierB      59         14%     38 of 59 always-0 (too hard)
    type2       8         12%      6 of  8 always-0 (too hard)
    tierA      29          0%     24 of 29 always-1 (too easy)

**Gabriel's questions discriminate 5x better than anything we generate**, and
`tierA` discriminates NOT AT ALL -- 24 of 29 are answered correctly every time.
The user's instruction to keep Gabriel's questions as the primary signal and
ours as confirmation (2026-09-02) is now measured rather than assumed. Our
SQL-truth generators produce questions clustered at the extremes: Tier A too
easy, Tier B and the mined type-2 too hard.

**free-SQL IS FOUR TIMES NOISIER PER QUESTION.** Mean |a-b| per question is
0.229 for free-SQL against 0.056 for typed, and 24% of its questions are
outright unstable against typed's 5%. Its aggregate looks calm (both runs mean
within 0.016) only because the per-question swings cancel. Any per-question
analysis of free-SQL needs far more repeats than typed.

**A GAP TO FIX:** all 62 `retrieval` questions produced no comparable score in
either run and dropped out of the analysis. Cause not yet established.

**Consequences.**
1. Build the evaluation set from DISCRIMINATING questions, not from whatever
   exists. Screen first, then run arms -- the screen pays for itself in one arm.
2. Tier A as currently generated is unusable for arm comparison. It measures
   that the system works, not which version works better.
3. Cost follows from n, and n now has a number: ~100 discriminating questions.
   Every earlier cost estimate assumed 150 arbitrary ones.

## D-096 — every count question is useless for comparing arms, and we own 24 usable questions

Full paired screen, 500 questions, QwQ, two identical runs per system
(2026-09-02/03). Supersedes the batch-1 figures in D-095, which were optimistic
on a smaller sample.

**Only 262 of 500 produced a score at all.** The 162 `retrieval` questions
sampled here score NOTHING -- their truth is `kind="entity"` with `papers=[]`,
so `set_f1` (needs papers) and `judged_set_f1` (needs a universe) both decline
and no scorer covers them. That is true of ALL 720 in the retrieval bank, not
just the sample. They have never contributed to any arm comparison.

**NOISE FLOOR vs n**, 95th percentile of |mean(a)-mean(b)|, 600 bootstrap draws:

        n      typed    free-SQL
       25      0.110       0.187
       50      0.066       0.112
      100      0.041       0.076
      150      0.029       0.054
      200      0.021       0.039
      250      0.010       0.023

**~150 questions for +/-0.03 on typed; free-SQL needs ~250** for the same.
free-SQL is the noisier instrument by a factor of ~3 (per-question |a-b| 0.213
vs typed's 0.077, 22% of its questions outright unstable against 7%).

**THE FINDING: COUNT QUESTIONS DISCRIMINATE NOTHING.**

    shape     n     discriminating   always-0   always-1   unstable
    count   207          0  ( 0%)         97         96         14
    set      55         24  (44%)         24          2          5

**Zero of 207 count questions can separate two arms.** Every one is answered
correctly every time or wrongly every time. They measure whether the system
works, never which version works better. 57% of our question bank is `count`.
This settles the count-vs-set question the user asked on 2026-09-02: they are
not merely different, one of them is inert.

**By source** (typed):

    gabriel     8     75% discriminating     <-- human truth
    type2       8     12%
    tierB     159     11%      103 of 159 always wrong
    tierA      87      0%       75 of  87 always right

**We own 24 discriminating questions.** Out of 500 screened, out of ~1,500
generated. To assemble the ~150 the noise curve demands, at the current yield
(24/500 = 4.8%) we would have to screen ~3,000 -- more than exist.

**So the bottleneck is not question COUNT, it is question CALIBRATION.** Our
generators produce questions clustered at the extremes: Tier A trivially easy
(86% always right), Tier B and the mined type-2 too hard (65% and 75% always
wrong), retrieval unscoreable, count inert by construction. Generating more of
the same cannot fix this.

**What follows.**
1. Score arms on SET questions only. Report count separately as a capability
   check, never as part of an arm comparison.
2. Fix or retire the retrieval bank -- 720 questions currently costing compute
   and returning nothing.
3. Generation must target the middle of the difficulty band. A question is
   worth generating only if the system sometimes gets it right; the screen
   gives us a cheap filter to enforce that.
4. Gabriel's 8 remain the most informative questions in the project by a factor
   of ~7, which is the strongest argument yet for spending his review time on
   MORE of them rather than on anything we can generate.

## D-097 — D-096 was wrong about two of its four banks; corrected, and the usable set more than doubled

D-096 declared the 720-question retrieval bank unscoreable and Tier A
0% discriminating. Both claims were checked against only three scorers
(`judged_f1`, `set_f1`, `count_correct`) and are wrong -- correcting on the
record rather than leaving a false result standing.

**Retrieval bank: not broken.** It was built exactly as intended -- pick an
entity, write a question that should surface it, check it comes back -- and
`entity_retrieved` (already in `default_scorers()`) has been measuring it the
whole time: 0.784 typed / 0.698 free-SQL, unweighted mean, on the 162 sampled.
The earlier "unscoreable" claim only checked the three paper-shaped scorers and
never noticed `entity_retrieved` covers this shape. What IS true: `papers=[]`
in the stored truth (kind="entity" holds the entity id, never the paper list)
means no PAPER-SET scorer can run, so it never contributed to `set_f1` or
`judged_f1` -- a narrower and correct claim than "unscoreable".

**Root cause of the always-miss group, checked by reading traces, not
guessing.** 35 of 197 sampled retrieval questions always fail. Tool use splits
three ways:

    12   called NO TOOLS AT ALL -- answered from memory, never searched
     7   used `facets` instead of `search` -- FOUND THE RIGHT PAPERS, scored 0
         anyway because entity_retrieved only reads a.entity_ids, not facet hits
    16   searched and genuinely did not find the entity

So of 35 "failures", 7 are a scoring gap (facets results aren't credited) and
12 are the model skipping retrieval outright (`Muon`, appearing in 22 papers
under 10 spellings, returned ZERO entities on `facets(objects=[Muon])` in one
sampled case) -- not a burial-in-noise problem, not a ranking problem. Only 16
are real search misses. This says the next arm worth testing is "never abstain
without searching" and crediting `facets`, not reranking.

**Tier A: the metric was wrong, not the questions.** `label_recall` already
existed (`mentioned_label_recall`, `retrieved_label_recall`) and was never
looked at in D-096's analysis, which used `count_correct` against `truth.value`
-- the wrong field for a "labels"-kind truth. Re-screened on the SAME 500-run
data, no new compute:

    metric                     typed DISCRIM   free-SQL DISCRIM
    count_correct (wrong)          0% (0/87)        0% (0/87)
    mentioned_label_recall        33% (24/73)       29% (21/73)

**And it separates two different capabilities cleanly.** `retrieved_label_recall`
(exact entity ids in the trace) is 99% always-1 on typed -- Tier A retrieval is
essentially solved for typed -- but 51% always-0 on free-SQL: free-SQL is
GENUINELY FAILING TO RETRIEVE the right entities on questions typed answers
almost perfectly. Previously invisible; the count-only view could not show it.
Separately, `count_closeness` (below) on the SAME Tier A questions'
count-shaped half is 0% discriminating both systems -- so for Tier A, "how many"
is trivial and "which ones exactly" is not. Naming, not counting, is where the
system fails.

## New: `count_closeness`, a graded count scorer

`exact_count` is binary: 11 and 500 score identically against a truth of 12,
both zero. D-096 found this made 0 of 207 count questions discriminating --
every one was answered right every time or wrong every time. Added
`count_closeness = 1 - |claimed-true|/true`, floored at 0, exact match at
truth=0. Runs alongside `exact_count`, does not replace it (`default_scorers`).
Recovers 7 typed / 16 free-SQL from zero, concentrated in tierB
(count_closeness DISCRIM 6% typed / 13% free-SQL vs tierA's 0%/0% -- consistent
with tierA's counts being trivial and tierB's being too small, see below).

Tests: `tests/test_eval_harness.py::test_count_closeness_*` (6 cases: relative
not absolute error, zero-truth edge case, prose-number fallback, abstains for
non-count shapes). 862 tests pass project-wide.

## New: `hepcoveragekg/eval/verify_truth.py` -- evidence-backed gold

Two operations, both checked against the real graph on 2026-09-03:

**`trim_question`**: for any SQL-set-truth question naming an `entity_id`,
keeps only papers with >=1 evidence-linked assertion to that entity; drops
papers with none; empties (and the question is discarded) if nothing survives.
Applied to Tier B's 200 questions: 72 unchanged (fully backed already), 124
trimmed (had some unsupported papers), 4 dropped entirely. Gabriel's
human-verified truth is explicitly exempted -- this filter would be a downgrade
there, never an upgrade.

**Important limit, stated in the module docstring so it cannot be forgotten**:
this catches the cheap, common failure (a claim with NOTHING behind it) and is
NOT a substitute for a human reading the paper. A structural spot-check (does
the quote's OWN assertion connect the paper to the concept, not just "is there
a quote") passed all 6 sampled on 2026-09-03. A phrase-level recall check (is
there a paper mentioning the concept that the gold is MISSING) found 1 of 31
distinctive labels with an unaccounted mention, and that one was a generic
phrase multiple systematics would plausibly share. So: not fully trustworthy,
trustworthy enough to filter on.

**`backfill_retrieval_papers`**: fixes the retrieval bank's actual bug --
`papers=[]` where the question text asks "which analyses...". Fills `papers`
from evidence-backed assertions, restricted to `min_papers=3` (128 of 720
qualify; the other 592 resolve to 1-2 papers, all-or-nothing and weak
discriminators per the D-096 screen). Output already evidence-filtered by
construction -- no second trim pass needed. 127 converted, running as a paired
control on QwQ now (54141-54144); result pending.

Tests: `tests/test_verify_truth.py`, 10 cases against an in-memory sqlite
fixture (evidence-backed vs unbacked papers, universe correctness -- a dropped
paper must leave the universe too, never becomes a silent confirmed-negative --
Gabriel exemption, file-level counts).

## Re-screened total, same 500-run data, zero new compute

    system     old (D-096, 3 metrics)   new (this decision, 4 metrics, union)
    typed              24                        50
    free-SQL          ~24                        44

By pool (typed): tierA 24 (via label_recall, was 0), tierB 19 (12 via
evidence-trimmed set_f1 + 7 via count_closeness, some overlap), gabriel 6,
type2 1. The bottleneck named in D-096 -- "our generators cluster at the
extremes" -- turns out to be partly a metric-choice artefact: Tier A was never
at the easy extreme, it was being read through a scorer that couldn't see its
real difficulty.

## Wording-variance experiment (D-098-adjacent, same run)

9 of Gabriel's questions reworded 3 ways each (36 total, meaning checked
against the original condition -- 3 of the first-draft rewordings dropped a
qualifier that carried the condition itself and were rewritten before running,
see `eval/questions/gabriel-reworded-2026-09-03.jsonl` provenance), 6 repeats,
QwQ typed (54135/54136 complete; free-SQL 54137/54138 running).

    average within-one-wording std (repeat noise)   : 0.078
    average across-4-wordings std (phrasing effect)  : 0.095

**Phrasing moves the score by MORE than repeat noise does, on average.** One
case is not noise at all: `gabriel-gf-01-condition` scores 0.03 on its ORIGINAL
wording and 0.53-0.76 on all three independent rewordings. Traced to a specific
mechanism, not guessed: on the original wording the model twice retrieved the
right papers via `facets` (named MV2c10/DeepCSV/DL1r/DeepJet correctly in
prose) and then wrote a summary-style answer with `papers_from="none"` in the
answer contract -- describing what it found instead of committing to a listed
set. The rewordings did not reliably trigger this pattern. Free-SQL half
pending before this is reported as a system-wide finding rather than a
typed-QwQ one.

## D-098 — the evaluation set, rebuilt: 24 usable questions to 163

Closing the work started in D-096/D-097. All numbers from paired controls on
QwQ (two identical runs per system), screened as: a question DISCRIMINATES if
both runs agree (|a-b| <= 0.34) and it lands strictly between always-right and
always-wrong (0.05 <= mean <= 0.95). Anything else cannot separate two arms.

**Result, by bank and system:**

    bank              typed   free-SQL   union
    retrieval-conv       72        35       84
    tierA                24        21       36
    tierB                19        20       34
    gabriel               6         1        7
    type2                 1         1        1
    TOTAL               122        79      163

Against the D-096 noise curve (~150 questions for +/-0.03 typed, ~250 for
free-SQL): **typed is at 81% of target, free-SQL at 32%.** free-SQL needs more
because it is the noisier instrument -- per-question |a-b| 0.213 vs typed 0.077.

**The single biggest win was the retrieval bank, which D-096 called
unscoreable.** Backfilling `truth.papers` from evidence-backed assertions
(`verify_truth.backfill_retrieval_papers`, min 3 papers) converted 127 of 720
into set questions and they discriminate at **63% on typed** -- the highest
rate of ANY bank, Gabriel's included. The 592 skipped resolve to 1-2 papers and
would be all-or-nothing.

NOTE the trade: converting sets `truth.kind` from "entity" to "set", so
`entity_retrieved` no longer applies to the converted copies. The original
720-question file is unmodified, so both measurements remain available -- the
original bank for retrieval capability, the converted 127 for arm comparison.

**Where the growth came from, in order:**

    +72  retrieval bank backfilled (D-097 fix)
    +24  Tier A rescored with label_recall instead of count_correct (D-097 fix)
    +19  Tier B, evidence-trimmed truth + count_closeness
    ~24  the original D-096 count, now understood to have been metric-limited

Two of the four gains were pure scoring corrections on data already collected --
no new model calls. The retrieval conversion needed one QwQ run (free).

## Wording variance, both systems (completes the D-097 stub)

9 Gabriel questions x 4 wordings x 6 repeats, QwQ:

                    repeat sd    wording sd
    typed             0.078        0.095
    free-SQL          0.139        0.108
    ------------------------------------------
    both              0.108        0.101

**Correction to the typed-only reading in D-097**: with free-SQL included,
phrasing and re-running contribute the SAME order of noise (0.101 vs 0.108),
rather than phrasing dominating. The typed-only figure looked like phrasing
mattered more; it does not hold across systems.

Practical consequence: **a reworded question is worth about one extra repeat of
noise, no more.** Rewordings test robustness; they do not multiply a question's
statistical value, and 4 wordings of one question is not 4 questions.

**One case is a real mechanism, not noise.** `gabriel-gf-01-condition` on TYPED
scores 0.028 on its original wording and 0.669 averaged over three independent
rewordings. On FREE-SQL the same comparison is 0.432 vs 0.409 -- no effect. So
it is typed-specific. Traced by reading transcripts: on the original wording
the model twice retrieved the right papers via `facets`, correctly named
MV2c10/DeepCSV/DL1r/DeepJet in prose, then emitted `papers_from="none"` in the
answer contract -- describing what it found instead of committing to a paper
list. The score is 0 because it never named a set, not because it never found
one. Same failure family as the 7 facets-route questions in D-097: the system
finds the answer and fails to hand it over in the contracted form.

**That is now the best-evidenced arm candidate we have** -- not reranking, not
breadth: make the answer contract capture what the trace already contains.

## D-099 — Gabriel's full batch 2 (104/104): 0.54 agreement, and two named causes

Returned complete 2026-09-03. Supersedes the partial reading in D-092, which
saw only the first 51 rows and was therefore biased toward the families he
happened to do first.

**Agreement fell from 0.708 (partial) to 0.54 (full).** The partial view had
covered gf-01/02/04 only; the families he had not yet reached are the ones the
system does worst on.

    question    n   agree   acc   false-YES  miss  unsure
    gf-01-cond 33      25   0.76         0      8      2
    gf-02       6       2   0.33         4      0      0
    gf-04       9       7   0.78         2      0      1
    gf-05       9       1   0.11         8      0      0
    gf-06       1       0   0.00         1      0      0
    gf-07      14       1   0.07        13      0      0
    gf-08      21      11   0.52        10      0      2
    gf-10..15   6       6   1.00         0      0      0
    TOTAL      99      53   0.54        38      8      5

**The error structure is almost perfectly one-directional.** gf-01-cond is the
ONLY question where the system misses (8 misses, 0 false-yes). On every other
set question it OVER-CLAIMS: 38 false positives, 0 misses. Prior conclusions
drawn from the 51-row partial -- which showed a balanced-looking split -- were
an artefact of which families had been reviewed.

**gf-07 (ttZ control region), 13 of 14 wrong. Cause: ttZ/ttbar conflation.**
Of the 13 false positives, the cited evidence mentions ttZ in exactly ONE. Five
cite `$t\bar{t}$` (plain ttbar) and seven cite something unrelated (`tWZ`,
`Wt`, generic top-quark production). The system answers a ttZ question with
ttbar evidence. This is the notation-collision family already known from the
`\mathup` LaTeX problem (gf-07 was the original 2103.06956 case in D-087's
lineage) -- `ttZ` and `tt` are not being kept apart.

**gf-05 (Higgs candidate), 8 of 9 wrong. Cause: CONFIRMS D-087 INDEPENDENTLY.**
Of the 8 false positives, five cite a "candidate" of the WRONG PARTICLE -- top
candidate, boson candidate, quark candidate, 4mu, b+ candidate. The single
correct answer is the only one whose evidence names Higgs. D-087 predicted
exactly this from cosine geometry: bge-base has no `Higgs` token and splits it
into `hi` + `##ggs`, so `Higgs candidate` degenerates to matching `candidate`.
That was a lab measurement on hand-picked phrases; this is the supervisor
independently marking the same failure on real questions he was not told about.
**The encoder work (chATLAS / PhysBERT, D-087) is now evidence-backed as a fix
for a named, supervisor-confirmed failure**, not a speculative improvement.

**The value questions are a clean win: gf-10 through gf-15, 6 of 6 correct.**
These are the ones reframed after his batch-1 complaint ("Not a yes/no
question. What to do here?") to show OUR ANSWER and ask whether it is right.
Every one upheld, with substantive notes (he adds the 2D exclusion contour
detail on gf-12, the data-extracted b-tagging efficiencies on gf-11). The
reframing worked and this question style should be extended.

**A recurring note worth acting on separately**: on gf-11 and gf-13 he marks
the ANSWER correct but observes the EVIDENCE SHOWN does not contain the fact
the answer states ("It is correctly stated in the answer, but missing from the
evidence presented"). Same shape on gf-06. So the answer is right and the
evidence rendering under-reports what the system used -- a presentation gap in
the review sheet, not a system error, but it makes the sheet harder to judge.

**Our critic vs Gabriel, where it ruled: 26/30 = 0.867** (was 0.957 on the
partial 51). Still the strongest component, still not the bottleneck.

**Merged gold set** -> `eval/questions/gabriel-gold-2026-09-03-full.jsonl`:
9 questions, **106 gold rows, 253 judged papers** (was 78/174 at batch 1, and
93/209 at the partial merge). gf-07's universe more than doubles (39 -> 53),
gf-05's 29 -> 38, gf-08's 23 -> 44. One conflict, unchanged from D-092:
2012.01581 on gf-02, batch 2 wins.

## D-100 — the set works, and aggregating question types hid the answer

First arm test on the 164-question discriminating set (D-098). Encoder swap,
typed system, one repeat, duplicate controls. Run on two models; **the QwQ half
is void** -- 121 of 164 records errored (88 API timeouts, 33 harness timeouts)
and the surviving text was garbled ("Okay, the's't the States the user is
asking about"), the server having degraded under 16 concurrent requests on one
A100. qwen3-32b was clean: 159/164 answered, zero errors.

**Aggregate, all 161 scored questions:**

    control-a 0.393   control-b 0.377   noise 0.016
    chATLAS   0.380   delta -0.005      inside noise
    PhysBERT  0.362   delta -0.023      marginally outside, NEGATIVE

Read that alone and the verdict is "the encoders do nothing, PhysBERT slightly
hurts". **That verdict is wrong, and the per-pool split shows why:**

    pool             n   control  chATLAS  PhysBERT   noise
    retrieval-conv  84    0.246    0.257    0.259     0.004
    tierA           36    0.617    0.558    0.497     0.076
    tierB           34    0.487    0.498    0.477     0.037
    gabriel          6    0.409    0.372    0.339     0.162

**On concept -> papers questions BOTH ENCODERS HELP, above a 0.004 noise floor:**
chATLAS +0.011, PhysBERT +0.013, i.e. ~3x the noise. That is the retrieval task
the encoder work was aimed at, and the effect is detected.

**On per-paper questions PhysBERT HURTS, -0.120 against a 0.076 floor.** Tier A
hands the system the paper, so retrieval is free there -- and changing the
encoder changes which entities surface, which can only interfere. chATLAS moves
-0.059 on Tier A, inside that pool's noise.

**The two effects have opposite signs and nearly cancel in the aggregate.** This
is the first hard evidence for the hypothesis the user raised on 2026-09-02 --
that a mechanism can help one question type and hurt another, and that a single
headline number is the wrong instrument. Every arm result reported before this
one was an average over question types that behave differently.

**METHOD RULE, promoted:** report arms PER POOL, never as one number. The pooled
noise floors differ by 40x (0.004 on retrieval-conv, 0.162 on gabriel's six),
so a single "noise floor" for a mixed set is meaningless as well.

**The set is validated, with a caveat.** It detected a real effect that the
aggregate hid, and the retrieval-conv pool's 0.004 floor at n=84 with ONE repeat
is the tightest measurement this project has made. But the Gabriel pool's floor
is 0.162 on six questions -- far too noisy to judge anything -- so "the set
works" is true of its large pools and false of its most valuable one. gf-05, the
question Gabriel's review showed failing on exactly the Higgs/candidate
confusion the encoders were meant to fix, did not survive screening into the
set at all, so this test could not check the prediction most directly.

## D-101 — the PhysBERT verdict in D-100 was a bad draw; retracted

D-100 reported PhysBERT at -0.122 on Gabriel's questions, "ABOVE noise", and
built a mechanism story around it (mean-pooled anisotropic vectors degrading
ranking). That was ONE REPEAT of 9 questions. Rerun at 3 repeats on both models
at the user's request:

                        control  noise   chATLAS   PhysBERT
    qwen3-32b            0.331   0.135    +0.024    -0.032    both inside noise
    QwQ                  0.381   0.076    +0.002    +0.025    both inside noise

**Inside the noise floor on both models, and the sign flips between them.** The
-0.122 was noise. The single clearest "evidence" cited -- gf-03 falling
0.889 -> 0.333 -- inverts at 3 repeats: control 0.593, PhysBERT 0.889, i.e.
PhysBERT is BETTER on that question.

The anisotropy mechanism (D-087, PhysBERT similarities in a 0.62-0.68 band
against bge-base's 0.86-0.89) is a real measured property. **It is not
established that it degrades end-to-end performance.** The two claims were
conflated; only the first is evidenced.

**GABRIEL'S 9 QUESTIONS CANNOT RESOLVE ENCODER-SIZED EFFECTS.** Two identical
controls differ by 0.135 (qwen3-32b) and 0.076 (QwQ) AT THREE REPEATS. Our most
valuable ground truth is our least sensitive instrument. Any arm effect claimed
on this pool alone needs repeats well beyond 3, or it is unreadable.

**What survives unchanged:** the retrieval-conv pool (n=84, noise 0.004,
chATLAS +0.011, PhysBERT +0.013). A large pool with a tiny floor is the only
place this project has detected an encoder effect at all.

**Standing verdict on encoders:** a small positive effect on concept->paper
retrieval; no measurable effect elsewhere; no evidence of harm.

**Method rule, third time this pattern has cost something (cf. D-093, D-096):**
never report an arm from a single repeat on a small pool. The screen classifies
questions at 1 repeat; that is not the same as SCORING an arm at 1 repeat.

**Also observed:** QwQ scored only 6 of 9 Gabriel questions to qwen3-32b's 8 --
gf-04 and gf-08 dropped because QwQ named no papers. The answer-contract
failure again, and it narrows QwQ's Gabriel base further still.

## D-102 — rephrasing the question beats resampling the model, 7 to 1

Asked whether paraphrase-diversity and temperature-diversity are the same idea.
They are not, and the reword run (D-098) already held the answer. Union of FOUR
SAMPLES per question, QwQ typed, 9 Gabriel questions, same compute either way:

                        F1      recall   precision
    single sample     0.385     0.348      0.548
    4 repeats         0.406     0.362      0.548     +0.021
    4 WORDINGS        0.529     0.493      0.710     +0.144
    both (8 samples)  0.536     0.499      0.712     +0.151

**Rewording gains seven times what resampling gains, at equal cost.**

**And it raises PRECISION, 0.548 -> 0.710.** A union normally trades precision
for recall; this does not. So alternative phrasings are not retrieving MORE,
they are retrieving BETTER -- a different phrasing sends a different query and
lands on genuinely more relevant papers.

**Resampling on top of rewording adds almost nothing** (0.529 -> 0.536). Once
the question is diversified, diversifying the decoding is redundant. That is
the direct answer to "is temperature another way of doing the same thing": no,
it is a strictly weaker version, and dominated.

Per-question, the mechanism is visible:

    gf-07              overlap 0/6    the wordings found entirely different papers
    gf-01-condition    overlap 6/22   single 0.167 -> 4 wordings 0.818

gf-01-condition is the question whose ORIGINAL phrasing triggers the
`papers_from="none"` contract failure (D-098). Rewording routes around that bug
without fixing it.

**This is the largest single effect measured in the project** -- larger than the
encoder swap (+0.013 at best), `concept-prompt` (+0.072), `index-values`
(unreplicated). It is multi-query retrieval, a standard RAG technique we had
not tried.

**CAVEATS, and they are not small.** n=9 questions, QwQ typed only, and the four
wordings are the original plus three written BY HAND with meaning verified
against the original condition (3 of the first drafts were rewritten because
they dropped a qualifier that carried the condition). A machine-generated
paraphrase has no such guarantee, and a paraphrase that quietly changes the
question would inflate this number. Cost is 4x per question -- but unlike
repeats, it buys signal rather than averaging noise.

**Arm priority, revised:** (1) `--simple-answer`, already built, targets the
contract failure directly rather than routing around it; (2) multi-query
rewording, needs building, strongest evidence we have; (3) temperature+union,
now known to be dominated -- run only to confirm.

## D-103 — D-102's comparison was unfair; temperature was never tested

D-102 concluded that rewording beats resampling 7 to 1 and called temperature
"a strictly weaker version, and dominated". **The experiment behind that did not
involve temperature.**

`temperature=0.0` is hardcoded at every call site (planner.py:1311, 1582, 1597,
1657; free_sql.py:507) with no override. So the "4 repeats" arm was four GREEDY
runs, differing only by the batch-scheduling nondeterminism described in D-101 --
accidental floating-point noise, not a diversification strategy. The measured
contrast was therefore:

    4 rewordings      deliberate diversification      0.529
    4 greedy repeats  accidental infra noise          0.406

which is rewording versus NOTHING. The +0.144 stands; the claim that it beats
temperature does not, because temperature was never run.

**Reasons temperature may in fact win, none of them tested:**
  - it diversifies at EVERY step, not just the opening query. Rewording changes
    the first call and then proceeds greedily; a single optional `category`
    argument was worth 0.400 on gf-01-met, and that divergence was at round 1 of
    several.
  - it cannot corrupt the question. Three of the four hand-written rewordings
    had to be rewritten because the first drafts dropped a qualifier carrying
    the condition; a generated paraphrase has no such guarantee, and one that
    quietly broadens the question would inflate D-102's number.
  - one parameter to build, against a paraphrase generator plus a
    meaning-preservation check.

**The fair test, three arms at equal compute on Gabriel's nine:** 4 greedy
repeats (0.406, known) / 4 samples at temperature ~0.7 (unknown) / 4 rewordings
(0.529, known). Temperature needs plumbing first -- there is currently no way to
set it.

**Method note.** This is the fourth time this session a comparison has been
read as stronger than its design supported (cf. D-093 path-tool, D-096 metric
choice, D-101 single-repeat PhysBERT). The common shape: a difference is real,
and the ATTRIBUTION of it is asserted rather than measured.

## D-104 — retrieval is at 0.96; the whole loss is the handoff

Eight arms, 164-question set, QwQ typed, control 54163/54164 (noise 0.034).
Reporting `retrieval_reach` (did it FIND the right papers) beside `judged_f1`
(did it SAY so) for the first time:

    arm               retrieval_reach   answer F1    gap
    control                 0.963         0.277     0.685
    subgoal-status          0.966         0.400     0.566
    tool-examples           0.977         0.262     0.715
    path-tool               0.948         0.225     0.723
    state-objective         0.959         0.216     0.743
    simple-answer           0.930         0.179     0.751
    index-values            0.956         0.203     0.753
    reviewer                0.983         0.221     0.762

**Every arm retrieves 93-98% of the right papers. The spread across all eight
is 0.053. The answer then scores 0.18-0.40.** The gap is 0.57-0.76 everywhere.

**Consequence for everything measured before this.** The encoder work, the path
tool, index-values, search-sets, the quote indexing -- all of it targets a
component already at 0.96 with almost no headroom. That is why every retrieval
arm has come back inside noise: there is nothing left to win there. The session
spent its effort on the wrong half of the pipeline.

**subgoal-status, the only arm above noise (+0.071 overall), wins by narrowing
the GAP** (0.685 -> 0.566), not by retrieving better (0.963 -> 0.966, inside
noise). Its measured mechanism is reporting, not retrieval or memory -- not what
it was designed for.

**`--reviewer` diagnosed.** Best retrieval of any arm (0.983) and the worst gap
(0.762). Instrumentation shows why, and it is a dose-response:

    every plan rejected    n=8    rounds 1.25   ids named 0.00   F1 0.000
    some rejected          n=76   rounds 2.21   ids named 2.51   F1 0.192
    none rejected          n=80   rounds 2.19   ids named 7.09   F1 0.244

150 of 387 plans rejected (39%); one question had 9 proposed and 9 rejected. 0
unparsed verdicts and 0 ceiling hits, so the reviewer works as designed and the
DESIGN is wrong: `review_plan` assumed rejections are free because "retry
shouldn't consume a round", but they consume LLM CALLS, and the planner runs out
of budget before finishing retrieval. It does not filter bad plans, it prevents
work. It also pushes the answer toward citing `set_N` rather than listing:
89/164 answers against the control's 75/164.

**`--simple-answer` fixed the willingness and not the selection.** It cut
"named nothing" from 0.468 to 0.139 -- the largest change any arm made to that
number, and exactly its design intent -- while its gap stayed 0.751 and F1 moved
+0.012. Precision +0.002: the extra papers it names are as often wrong as right.
So the handoff hypothesis in its own docstring ("typed retrieval is fine and our
answer contract is awkward") is HALF right: the contract was awkward, and fixing
it does not help, because the model cannot select which of the retrieved papers
actually answer the question.

**`--subgoal-status` on gf-01 is not a bug, it is the arm's own mechanism.**
It scored 0.000 on gf-01, gf-01-condition and gf-01-met while the control scored
0.55/0.17/0.40. Cause: it SUMMARISES instead of enumerating on exactly the
multi-condition questions it targets -- "The graph identifies 23 analyses...
These include papers such as X, Y, Z, and others listed in the facet results"
(347 chars against the control's 3005). The three examples it cites fall outside
the small judged universe. It is not systematically terse: median answer 3394
chars against the control's 2711. Only on its target questions.

**A real metric bug found on the way.** `judged_named_none` returns 1.0 when the
answer named papers but none fell in the judged universe -- conflating "named
nothing" with "named nothing judged". That is what first made this look like a
scoring failure rather than a behaviour.

**WHAT TO TEST NEXT, and it is a different problem.** Not retrieval. The
question is which of ~0.96-recalled papers the answer commits to. `--reviewer`
judges plans and `--critic` judges retrieved rows; nothing judges the FINAL
SELECTION against the question. That is the untouched component and it holds
the entire 0.57-0.76.

## D-105 — the critic has been judging NOTHING; 100% of candidates defaulted

Investigating why `--reviewer` names nothing (user's suspicion, 2026-09-05) led
to a much larger finding. Across four 164-question runs:

    run                 reviews  candidates   kept  defaulted  dropped
    control                 131        7315   7315       7315        0
    reviewer arm            128        7903   7903       7903        0
    subgoal-status          170        9360   9360       9360        0
    simple-answer           166        9258   9258       9258        0

**33,836 candidates. 100% defaulted. Zero drops. The critic never judged a
single candidate**, while every arm was reported as running with a critic.

**Root cause, measured not guessed.** `CRITIC_MODEL=Qwen/Qwen3.5-9B` is served
with `--reasoning-parser qwen3`, so the chain of thought goes to
`reasoning_content` and `content` stays EMPTY until it stops thinking. On this
task it never stops -- a direct call returned `finish_reason=length`,
`completion_tokens=4000`, `content` empty. `_parse` finds no JSON, returns {},
and every candidate takes the deliberate default in `judge_candidates`:

    # A missing verdict defaults to the loosest kept rung, never to a drop.
    # Flagging is recoverable ... while a drop the model never actually made
    # is invisible.

That asymmetry is right, and it is exactly what made this silent. `set_1` and
`set_1_kept` had identical median size (58) in every run and nobody looked.

**Two bugs, one on top of the other.**
1. The critic call hardcoded `MAX_COMPLETION_TOKENS` (800) instead of
   `completion_cap(model)` (4000 for this model) -- D-084 repeating on a path
   the fix was never wired into. Fixed, and NOT SUFFICIENT: it thinks for
   whatever it is given.
2. The real fix is to turn thinking OFF. Judging a candidate against a question
   is CLASSIFICATION, not reasoning. With
   `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` the same
   model returns correct verdicts in under 800 tokens. Verified end-to-end
   through `_build_critic`: b-tagged jet -> exact (kept), Muon -> unrelated
   (dropped), Photon candidate -> unrelated (dropped). 1 of 3 kept, all judged.
   Sent as `extra_body` with a plain retry on failure, so a non-Qwen judge is
   unaffected.

**WHAT THIS INVALIDATES.** Every arm measured on 2026-09-04/05 ran with an inert
critic. They remain internally comparable -- all had the same dead critic -- but
no absolute number from them means what it said, and the critic's own +0.231
(D-062 era, measured with the Llama-8B judge) was never in force in any of them.
D-104's central finding needs re-testing for the same reason: retrieval measured
0.96 with NOTHING filtering the candidate set, so "retrieval is solved" was
measured on an unfiltered set.

**The three-stage view the user asked for, and why stage 2 was invisible:**

    1. retrieved          mean 84-102 entities
    2. judged relevant    SHOULD be set_N_kept -- was identical to set_N
    3. named in the answer      mean 2.3-4.4 arXiv ids

Stage 2 was a no-op, so the entire selection burden fell on the answer step
unaided. That is a better explanation of the 0.57-0.76 gap than anything in
D-104.

Reran control x2 plus subgoal-status, reviewer, simple-answer, tool-examples
with the working critic (54201-54206).

## D-106 — an answer-stage critic, built and fast-tested: right mechanism, wrong threshold

Built `hepcoveragekg/query/answer_critic.py` to sit on the 0.57-0.76 gap D-104
measured between what retrieval finds (0.93-0.98) and what the answer says
(0.18-0.40). Nothing addressed that gap: `--critic` judges ENTITIES against the
SEARCH TEXT, `--reviewer` judges the PLAN, `verify.py` checks claims after the
fact. None of them asks "does THIS PAPER satisfy the question".

    critic         entity  vs  search term     keeps the candidate list sane
    answer_critic  paper   vs  THE QUESTION    decides what the answer asserts

**Deliberately Gabriel's task** -- question, paper, retrieved sentence, yes/no --
so his 253 verdicts measure it directly rather than by proxy. No other component
can be checked that way.

**Fast test, qwen3-32b, four Gabriel questions, judged against his verdicts:**

    question           keep-all F1   critic F1     candidate purity
    gf-01-condition       0.88         0.77          78% gold
    gf-04                 0.88         0.60          78% gold
    gf-05                 0.63         0.40          46% gold
    gf-07                 0.42         0.80          27% gold
    OVERALL               0.661        0.618         -0.044

**It filters well and still loses.** Precision 0.494 -> 0.808, zero defaulted, so
the mechanism works. But recall falls 1.00 -> 0.50 and F1 drops 0.044. As
configured it is not a win.

**The sign tracks candidate purity, and that is the actionable part.** It helps
where the candidate set is mostly wrong (gf-07: 27% gold, 0.42 -> 0.80) and
hurts where it is mostly right (gf-04: 78% gold, 0.88 -> 0.60). gf-07 is the ttZ
question every arm ever run has scored 0.00-0.29 on; this is the first mechanism
to move it at all.

So the next question is a THRESHOLD, not a redesign: apply it only when the
candidate set is large or low-purity, or have it drop only what it is confident
about. To be measured once the current arms finish -- tuning it on four
questions would be exactly the fishing D-102 warns about.

**Design decisions taken from this session's failures.** Quotes are shown, not
just entity labels (gf-01-condition turns on `veto = use`, which lives in the
sentence). Only RETRIEVED entities are used, so it measures the system and not
the corpus. Missing verdicts KEEP the paper -- the `critic.py` asymmetry, since a
drop the model never made is invisible -- but the default rate is counted,
returned, and logged at WARNING above 25%, because that exact asymmetry hid a
completely dead critic for weeks (D-105).

**Caveat on the numbers.** The fast test judged only papers inside Gabriel's
judged universe. In production it faces the whole retrieved set, larger and
noisier, so 0.808 precision is an optimistic ceiling.

Tests: `tests/test_answer_critic.py`, 8 cases (per-paper keep/drop, missing
verdict keeps and is counted, unparseable output alarms, broken judge does not
kill the run, invented ids ignored, chunking covers each paper once, the prompt
carries the quote, evidence reads only retrieved entities).

## D-107 — the arm differences on typed are mostly a formatting artefact, not reasoning

Runs 54201-06 (QwQ, 164 discriminating questions, critic working after D-105).

`judged_set_f1` and `set_f1` both read arXiv ids out of `a.text`. That is the
right rule (D-062: a retrieval footprint must not count as an answer). But the
agent very often writes a correct answer that contains no ids:

  - a placeholder: "the analyses are: [list of papers from the intersection]"
  - the papers by TITLE: "1. Measurement of the production cross section for a
    W boson in association with a charm quark ..." -- 7 correct papers, scored 0
  - a deferred instruction: "run `papers_of` on the intersection of set_3 and
    set_4" -- and then it never ran it
  - the citation tag `<papers_from>set_1_kept and set_3_kept</papers_from>`,
    which `resolve_citations` did not resolve: `a.cited` came back "" in 59 of
    the 60 Gabriel answers, so the legitimate cited-set path never fired

Rate of the resulting hard zero, over the 108 set-scored questions per arm:

    reviewer        71%      simple-answer   19%
    tool-examples   50%      control         35-36%
    subgoal-status  35%

Decomposing score into (prints ids at all) x (quality when it does):

    arm              score   print-rate   when-printed
    ctrl-a           0.222      0.65         0.277
    ctrl-b           0.225      0.64         0.272
    subgoal-status   0.265      0.65         0.353
    reviewer         0.180      0.29         0.312
    simple-answer    0.214      0.81         0.244
    tool-examples    0.221      0.50         0.308

The left column is what we have been reporting as arm quality. The middle
column is a property of the OUTPUT FORMAT. On Gabriel's 9 questions, the two
where all six arms printed ids (gf-01, gf-02) score **0.626 in every single
arm** -- the entire measured spread there came from who happened to print ids.

Consequences:

1. The reviewer's -0.025 is not the reviewer reasoning worse. It stops early
   (1.8 rounds vs 2.4) and then writes a pointer instead of a list. Its
   when-printed quality (0.312) is ABOVE both controls.
2. simple-answer's gain is the opposite artefact: it prints ids more often
   (81%) while being slightly worse per question (0.244).
3. subgoal-status is the only arm that is better on BOTH axes, so D-096's
   +0.038 survives -- it is the one real effect in the set.
4. Every typed arm number reported before this is confounded. They are not
   void (the ranking on the when-printed column is still a measurement), but
   they must be reported as the product, not as answer quality.

Fix, in order: (a) resolve `<papers_from>` properly -- it is already the
designed mechanism and it is silently dead; (b) reject an answer whose text
names a set but prints nothing, and re-ask once; (c) report `print_rate` and
`f1_when_printed` alongside `judged_f1` in every arm table from now on.

Do NOT "fix" this by falling back to `a.papers` -- that is the D-062 footprint
trap, and on gf-08 it would hand back all 24 gold papers out of a 44-paper
universe for free.

## D-108 — three checks on an answer, not one reviewer

Raised as "why not get a reviewer on the answer -- checks it is properly
answered, checks everything is well cited, no empty lists". Right instinct,
and the job splits into three with very different costs:

  1. FORM      does the answer name anything at all?
  2. TRUTH     is each paper it names actually right?
  3. SUPPORT   is every claim tied to evidence we retrieved?

**(1) is deterministic and gets no model call.** `query/answer_gate.py`. The
four shapes are all decidable by looking at the text: a bracketed placeholder,
an empty answer, a promise to run `papers_of`, or prose and titles with no
arXiv id in them. An LLM here would add a call, a latency, a noise floor and
the D-105 failure mode -- a judge defaulting silently -- to a question that has
an exact answer. One retry, then accepted and flagged (`gate_failed`).

Abstentions are EXEMT. "The graph does not record this" names nothing and is a
legitimate answer; gating it would push the system to fabricate coverage, which
is the failure that matters more than the one being fixed.

**(2) is `answer_critic` (D-106), and it needed rewiring to measure anything.**
It filtered `session.answer_papers`, which only a resolved citation fills --
and D-107 measured `cited` empty in 59 of 60 Gabriel answers. It would have
judged an empty list on nearly every question while reporting itself as run:
the D-105 shape again, a mechanism that costs calls and changes no number. It
now reads the ids out of the PROSE by the same rule the scorer uses, and
strikes dropped ids from the text. `answer_before_critic` keeps the original --
a harness that rewrites an answer and then scores it is measuring itself.

**(3) is NOT built yet, deliberately.** `verification_score` already computes
`unsupported_claims` and nothing acts on it (gf-08 control-a: 0.833 with
`unsupported_claims: ["100"]`, run carried on). The reason to wait is D-107's
own finding about the PLAN reviewer: it cut rounds 2.4 -> 1.8 and made the
agent write a pointer instead of a list. A second LLM voice on the OUTPUT has
the same risk -- it can make the agent hedge, and hedging is what produces
prose with no ids. (1) and (2) can only narrow; neither can make it quit.

If the gap survives (1) and (2), (3) is the next arm, and it should be a
CITATION BINDER rather than a reviewer: every arXiv id in the answer must pair
with an evidence id from this run, and an id with nothing behind it is struck.
Deterministic, same as (1).

Running as a 2x2 (54217-20, 164 questions, QwQ, working critic): control,
gate, answer-critic, both. Not one arm: the critic filters what the answer
names, so if the gate changes HOW OFTEN it names anything, the critic alone
would confound the two -- which is the exact confound D-107 just found.

## D-109 — the DIAS clone had drifted out of git, and it had been silent

Found while deploying D-107. The clone was ~60 commits behind and a `git pull`
there REFUSED, which is why nobody had run one:

  - five job scripts existed only in its working tree -- `or_arm_job.sh`,
    `dev200_job.sh`, `frozen_baseline_job.sh`, `overnight.sh`, `arm_8b_job.sh`
  - `gabriel_arm_job.sh` was modified there and nowhere else (SYSTEM/WORKERS)
  - the ENTIRE question set was untracked in both trees

So every arm since 2026-08-30 ran from a script that was not in git, against
question files that were not in git. A run cites its questions by path and by
`questions_hash`; a set that is not versioned makes every score in `eval/runs/`
unreproducible in principle, and it nearly did in practice -- four submitted
jobs died in three seconds on a FileNotFoundError after a checkout removed the
164-question file (54213-16).

Fixed: DIAS's scripts committed verbatim (they are what actually ran, so they
are the record), question sets committed, `logs/` gitignored -- `git add -A` on
DIAS had swept 316 Slurm logs into a branch, after which `git checkout main`
deleted the directory the job scripts write into (54209-12, dead in one
second). The old working tree is preserved on branch `dias-wip-2026-09-05`.

Nothing was lost. It took three failed submissions to find all of it, which is
the argument for the fix rather than against it.

## D-110 — the citation fix works, and it hands the scorer a footprint

Measured on the surviving records of 54217-20 (the servers died mid-run, see
D-111, but the citation counts are per-answer and stand).

D-107 fixed `resolve_citations` so a compound `papers_from` -- "set_1_kept and
set_3_kept" -- resolves instead of dropping in silence. It worked:

    arm         answers   cited   cited %   papers per citation
    old ctrl        152       8       5%          32.6
    control          29       6      21%          38.0
    gate             31       7      23%          42.0
    critic           30       6      20%          10.5
    both             28       7      25%          12.7

The citation path went from 5% of answers to 20-25%. **And what it cites is a
38-42 paper set**, which is the retrieval footprint with a citation wrapped
round it.

THIS IS A RISK I INTRODUCED, and `judged_set_f1`'s own comment predicts it:

    "gpt-5.6-luna scored judged_f1 0.889 on gf-01-condition that way, having
     written no answer at all -- retrieving everything is optimal when the
     yardstick is that short."

The scorer distinguishes a CITED set (legitimate, v3's design) from a
FOOTPRINT (not an answer). That distinction holds only while the model cites a
narrowed set. It is now citing a wide one, and before the fix those answers
scored 0 for naming nothing -- so the fix converts a formatting zero into a
possible precision collapse. Both are wrong; they are wrong in opposite
directions, and only the second is measurable.

The answer-critic is what contains it: it cuts the cited set from 38-42 papers
to 10-13. That is not a side effect, it is the mechanism doing its job on a
footprint, and it is why the 2x2 (control / gate / critic / both) is the right
design rather than one arm -- `--answer-gate` alone may well score WORSE than
the old code by turning silent zeros into wide citations.

The critic's stated reasons are specific and judge the condition rather than
the topic, which is what it was built for:

    2001.06899  "No photon object mentioned in quotes."
    2007.02873  "Discusses top control region, not MB/BDT-CRW."
    2004.14060  "No mention of f_a2 observable."

It returned an all-drop verdict on 4 of 13 questions; the guard keeps every
paper there rather than emptying the answer.

STILL UNMEASURED: whether the papers it keeps are the RIGHT ones. No Gabriel
question survived in the wreckage, so the check against his 253 verdicts --
the one measurement that would settle it -- has not been made.

## D-111 — an arm must refuse to start behind a server that will die first

The D-108 2x2 was submitted behind a vLLM job showing `R 19:11:58` in squeue.
It had a 20-hour limit, so it had 48 minutes left. Forty minutes later the
server hit TIMEOUT and four 13-hour runs spent the rest of their lives
collecting APIConnectionError: 44% of questions errored and about 30 clean
records survived out of 164 per arm.

On the same 52 questions, old control vs the wreckage:

    abstained %    9.6 -> 44.2
    rounds        1.98 -> 1.31
    errored      0.096 -> 0.442

Every guard passed. `wait_for` proved the port answered; `model_served` proved
the id matched. Neither asks how long that stays true. "The server is up" was
never the question -- "the server is up for the next thirteen hours" was.

`server_outlives` now reads %L for the serving job on the answering host and
refuses below NEEDED_HOURS (default 14). Where no job of ours is on that host
it says so and continues rather than blocking on a server it cannot see.

THIS IS THE THIRD INSTANCE OF ONE FAILURE SHAPE and it belongs in the
write-up as such: D-088 the critic 404ing on every call, D-105 the reasoning
judge returning empty content, D-111 the server disappearing underneath. Each
time a missing dependency degraded into a PLAUSIBLE NUMBER instead of an
error, and each time the run completed, scored and reported. A system that
fails loudly is a design requirement here, not a nicety -- the alternative is
what happened in all three cases: weeks of measurements read as findings.

## D-112 — the answer-critic loses to doing nothing, except on dirty candidate sets

The first check of `answer_critic` against labels a physicist wrote. 253
(question, paper) pairs from Gabriel's two batches, 106 yes and 147 no, with
the evidence pooled from all 59 stored run files so every pair carries both
labels and quotes (one run's `entity_ids` covered only 52 of 253). Run on
OpenRouter for $0.099 total while the cluster ran the arms.

    judge               prec  recall      F1   agree  defaulted
    keep everything    0.419   1.000   0.591   0.419      -
    drop everything      -     0.000   0.000   0.581      -
    llama-3.1-8b       0.509   0.264   0.348   0.585      8
    qwen3-32b          0.589   0.500   0.541   0.644     38
    qwen3.8-flash      0.457   0.962   0.620   0.506    218
    gpt-4.1-mini       0.723   0.321   0.444   0.664      0

EVERY VALID JUDGE IS WORSE ON F1 THAN DOING NOTHING. qwen3.8-flash's apparent
win is D-105 again -- 218 of 253 defaulted, 1085 completion tokens per call
spent reasoning, nothing parseable returned. It is "always keep" in a judge's
coat, and `enable_thinking=false` did not take on that endpoint.

THE BOUNDARY IS SHARP AND IT IS PURITY (gold / candidates), gpt-4.1-mini:

    gf-01-met   purity 0.23   +0.292        gf-05   purity 0.42   -0.093
    gf-07       purity 0.19   +0.216        gf-02   purity 0.58   -0.108
    gf-01       purity 0.24   +0.165        gf-03   purity 0.56   -0.143
                                            gf-01-c purity 0.61   -0.203
                                            gf-08   purity 0.55   -0.552
                                            gf-04   purity 0.69   -0.618
                                                             mean  -0.116

Everything below 0.25 gains, everything above 0.4 loses. This confirms the
2026-09-05 fast test (gf-07 0.42 -> 0.80, gf-04 0.88 -> 0.60) on nine
questions instead of two.

AND THE GATE IS NOT AVAILABLE. Purity is defined by the gold, and the judge's
own drop rate does not proxy it:

    gf-07  kept 5/53 = 9%   purity 0.19   gained 0.216
    gf-04  kept 2/26 = 8%   purity 0.69   lost   0.618

Same keep rate, opposite truth. On gf-04 -- "which analyses unfold", where 18
of 26 candidates are right -- it kept TWO. A judge that cannot tell a clean
candidate set from a dirty one cannot be gated on its own confidence, which is
the only signal available at runtime.

DECISION: `--answer-critic` stays OFF by default. It is not withdrawn: the one
place it earns its keep is an answer that is a retrieval FOOTPRINT, which is
exactly what D-110 found the citation fix now produces (38-42 papers cited),
and exactly the gf-07 shape. The cluster arm (54234/54235) tests it against
footprints rather than against this universe, so it remains the right
experiment and this result does not pre-empt it.

SECOND FINDING, and it contradicts a standing assumption. Judge size matters
enormously here: keep-recall 0.264 (8B) -> 0.500 (32B) -> and gpt-4.1-mini
reaches keep-precision 0.723 with zero defaults. The SEARCH critic measured
the opposite -- the 8B was -0.0035 against the 72B, inside noise, which is why
the 8B became standard for everything. The two jobs are not the same
difficulty and the project has been treating them as if they were. Anything
concluded about "the critic" from the 8B's performance on search candidates
does not transfer to judging papers against a question.

New: `hepcoveragekg/eval/judge_gold.py` -- pairs from the gold, a cost
`estimate()` printed before any call, and the confusion matrix in filter terms
(keep-precision, keep-recall, drop rate) rather than as accuracy, because
accuracy here is beaten by a constant.

## D-113 — rank the candidates, do not filter them

Raised by Raul: the critic only says relevant/irrelevant; rank instead, so the
best papers sit at the top, and precision improves even where the answerer
cannot report everything. Tested the same day on Gabriel's 253 verdicts for
$0.07. It is the best result the project has on this problem.

Same judge, same calls, same evidence -- only the OUTPUT changes, from
keep/drop to a grade 0-3, ranked.

                            prec   recall      F1
    keep everything        0.419    1.000    0.591
    binary critic (drop)   0.723    0.321    0.445     <- D-112
    rank, take top-16      0.560    0.788    0.654     <- best
    rank, take top-10      0.617    0.624    0.621

                         p@1    p@3    p@5   p@10   p@16
    random (today)      0.459  0.445  0.449  0.458  0.449
    graded rerank       1.000  0.815  0.778  0.617  0.560
    perfect ranking     1.000  1.000  0.956  0.851  0.698
    share of headroom    100%    67%    65%    41%    44%

p@1 = 1.00 on all nine questions: the top-ranked paper is one Gabriel approved,
every time.

WHY IT WORKS WHERE THE BINARY CRITIC FAILED, and this explains D-112. Grades
against his verdicts, gpt-4.1-mini:

    grade 3   n=38   P(gold) 0.79
    grade 2   n=31   P(gold) 0.52
    grade 1   n=64   P(gold) 0.36
    grade 0   n=120  P(gold) 0.31

Monotone, so the ordering is real -- but the BOTTOM is weak: grade 0 still
holds 31% gold. The judge recognises the best evidence and cannot rule things
out. All the discriminative power is at the top of the scale. A binary critic
discards everything below its threshold and below the threshold is a third
gold, so it bleeds recall; a ranking only needs the top to be right, and the
top is right.

THE CUT-OFF IS ALREADY THERE. Best F1 is at top-16, and `TYPICAL_LISTED = 16`
in scoring.py -- measured from how many papers the planner actually writes. The
answerer's natural truncation is the optimal cut-off, so nothing has to choose a
threshold. That is the difference from D-112, whose whole failure was a
threshold that had to be right globally and could not be.

DEPLOYABLE ON WHAT WE ALREADY SERVE:

    judge           p@1   p@3  F1@16   grade spread 3/2/1/0
    gpt-4.1-mini   1.00  0.81  0.654   38/31/64/120
    qwen3-32b      0.89  0.81  0.645   52/11/45/133
    llama-3.1-8b   0.78  0.67  0.575   36/13/ 1/193

qwen3-32b matches the paid model inside noise, so this needs no metered
endpoint. THE 8B IS WORSE THAN DOING NOTHING (0.575 vs 0.591) and the spread
says why: 14 of 253 papers in the two middle grades. It refuses the scale, so
it is a binary classifier in a grader's prompt, and binary is what already
failed. Consistent with D-112's finding that judge size matters far more here
than for search filtering.

A RUNTIME DIAGNOSTIC FALLS OUT OF THAT, and it needs no gold: if a judge puts
almost nothing in the middle grades, its ranking is not usable. The binary
critic never had such a check -- `defaulted` catches a judge that answers
nothing, not one that answers with one grade.

FREE SIGNAL ALREADY DISCARDED. `critic.py` grades every search candidate
`exact | broader | unrelated` and `KEPT_RUNGS = (EXACT, BROADER)` collapses it,
treating the two as identical. That is a three-level ranking computed on every
run and thrown away. Ordering by rung costs no additional call.

NEXT, in order:
  1. rank by the rung we already have -- zero cost, testable on stored runs
  2. `--rerank` arm: graded answer-critic, order the answer, no dropping
  3. a question tier where ranking IS the task ("name the best example of X"),
     scored p@1/p@3. Raul's point: for those, set_f1 is the wrong metric and we
     currently have no question that exercises p@1 = 1.00.

## D-114 — a run must prove it ran the arm, before any number is read from it

Raul, after three consecutive days of runs that produced numbers while a
dependency was missing: check every running job once an hour, against what it
was supposed to be running, and say so even when the results are bad.

The point is that NONE of these crashed. Every one completed, scored and
reported a plausible figure, and watching `squeue` caught none of them because
in all four cases the job was RUNNING:

    D-088  the critic 404'd on every call            six arms "critic-on"
    D-105  reasoning judge returned empty content    33,836 candidates,
                                                     100% defaulted, 0 drops
    D-111  server hit its wall limit 40 min in       44% of questions errored
    D-108  gate fired 2 times in 164 questions       18% of answers left by a
                                                     path the gate cannot see

`hpc/monitor/armcheck.py` asserts the four things that were each, in turn, the
thing nobody checked:

    FLAGS    the recorded config matches what was asked for
    FIRING   the mechanism actually did something
    HEALTH   the error rate is not eating the run
    RANGE    the scores are inside what this system can produce

FIRING IS AGAINST OPPORTUNITY, NOT AGAINST ZERO, and this is the part that
matters. A `> 0` test passes the D-108 gate happily -- it fired twice while 23
answers named nothing. Each mechanism declares both what it did and how many
chances it had; below 50% is a FAIL. A judge defaulting above 30% is a FAIL,
which is D-105's line drawn where a run can be refused rather than annotated.

Verified against all four historical failures rather than asserted:

    54217   FAIL error rate 23/52 = 44%
    54233   FAIL gate fired 2 of 23 chances = 9%
    54194   FAIL use_critic judged 8548/8548 defaulted = 100%
    54127   FAIL 8548 candidates, 100% defaulted

Exit status is 1 on any failure, so it can gate a report instead of being
something to remember to read. An hourly job now runs it while anything is
queued.

CONSEQUENCE, IMMEDIATE: 54194 -- the `--subgoals` arm whose result was still
outstanding -- had a 100% defaulted critic. It is void, and no longer pending.

## D-115 — the first noise floors, and free-SQL beats typed on set F1

Runs 54242-48, 164 questions, QwQ, one worker per job, both systems with two
identical controls. Every job passed armcheck on flags, error rate (0-2%) and
critic defaulting (10-12%).

THE NOISE FLOORS, which this project has never had for the typed system:

    typed    set_f1     0.0006      <- usable
    typed    judged_f1  0.0843      <- NOT usable for arms
    free-sql set_f1     0.0534
    free-sql judged_f1  0.1494

`judged_f1` runs on Gabriel's nine questions and its floor is 0.084. Every arm
difference this project has reported on that metric -- including D-096's
+0.038 for subgoal-status -- is inside it. `set_f1` is the metric that can
carry an ablation; `judged_f1` can only carry a description.

THE ARMS, against the control mean (* = outside the noise floor):

    typed rerank        judged -0.043   set_f1 -0.030*
    typed rerank+ac     judged -0.042   set_f1 -0.009*

RERANKING HURT, and it is outside the floor, so it is real. It contradicts
D-113's offline +0.063 and the difference is what was reordered: offline the
ranking was over the papers THE ANSWER NAMED; `--rerank` reorders `set_N_kept`
and the `papers_of` rows. Ordering the retrieval is not ordering the answer,
and the offline result does not transfer to the place it was wired into.

TYPED vs FREE-SQL, and free-SQL wins on the metric with a floor:

    arm            set_f1   reach   via text  via a.papers  papers/ans
    typed a         0.228   0.969      67          31          38.7
    typed b         0.228   0.983      74          24          41.2
    free-sql a      0.308   0.523      87          11           4.5
    free-sql b      0.254   0.548      89           9           6.0

It is not a scoring artefact, which was the first thing checked. `set_f1`
falls back to `a.papers` when the text names nothing, and that fallback is the
D-062 footprint trap -- but free-SQL barely uses it (9-11 of ~98) and its
fallback set is 4.5-6.0 papers, an answer. TYPED uses it 24-31 times on a
38-41 PAPER FOOTPRINT. So the typed system is being scored on its retrieval
footprint a quarter of the time, and that is where its set F1 goes.

Retrieval reach 0.97 against 0.52 makes the same point from the other side:
the typed system finds nearly twice as much and converts it worse.

BY QUESTION TYPE the two systems are not close, and they split cleanly:

    tierA (36)   typed 0.73-0.77   free-sql 0.51-0.60    typed wins
    tierB (34)   typed 0.21-0.28   free-sql 0.48-0.55    free-SQL wins
    retrieval    0.21-0.24         0.23-0.27             flat

CORRECTION TO THE RUN RECORD: `named_ids` is written by `from_session`, the
planner's path. The free-SQL system builds its Answer separately and never
sets it, so `answer_names_papers` reads 12-15% for free-SQL when the scorer
found ids in the text of 87-89 of its set answers. The field is a planner-only
measurement and any cross-system table using it is wrong.

## D-116 — the model was calling answer() all along, as an XML tag

Chasing why 70-78% of answers never reached `answer()`. They did. We were not
listening.

Verbatim tail of run 54242, gabriel-gf-01-condition:

    <answer text="18 analyses (papers) use b-tagged jets in their event
            selection, identified via facets matching the 'BJet' tag."
            papers_from="the facets result" reason="answered"
            answerable="true"/>

That is an `answer` call in everything but syntax. `_TEXT_TOOL_CALL` matches
`<tool_call>{json}</tool_call>` and nothing else, so it matched none of these
and the whole message fell through to `after_plan`, where `last_content`
becomes the answer -- raw, uncited, no `answer_papers`, no gate, no critic.

Measured over three 164-question runs, of the ~150 answers each that never
reached `answer()`:

    <answer ...> XML tag, missed      115 / 107 / 115   = 71-77%
    no call at all, plain prose        34 /  40 /  33   = 22-27%
    bare {"name":"answer"} json         1 /   3 /   2   =  1-2%

So the model called `answer` on roughly 78% of questions. We recorded 7%.

ONE MISSED SYNTAX IS THE ROOT OF FOUR SEPARATE FINDINGS:

    D-107  `cited` empty in 59 of 60 Gabriel answers
    D-107  answers that "name nothing" -- they named plenty, in a tag
    D-108  the gate firing twice in 164 questions
    D-113  the answer-critic reaching 9% of its chances

Every one of those was read as a property of the system. All four are one
regex. The citation mechanism, the gate, the answer-critic and `answer_papers`
were each doing exactly what they were built to do, on the 7% of answers that
reached them.

FIXED: `_recover_tool_calls` takes the run's tool names and recovers the XML
form, but only when the JSON form found nothing (a message holding both is the
model correcting itself, and the explicit call wins) and only for tags that
name an actual tool -- without that check any `<sub scale="1">` in prose
becomes a call.

WHAT THIS INVALIDATES. Nothing measured is wrong, but much of it was measured
on a system whose answer contract was unreachable 93% of the time. Every typed
arm since the contract landed needs re-reading in that light, and the D-115
comparison in particular: free-SQL scored higher set F1 while the typed
system's answers were being taken from its prose rather than its answer call.
Whether that gap survives the fix is the first thing to re-run.

## D-117 — the prompt told the model not to write the one thing we score

Raul: "maybe changing something in the prompt to ask for a specified
notation?" -- and "we can still be checking for other notations that may be
still being used". Both, and the second is the one that keeps working.

WHAT THE PROMPT SAYS. `tools_for` rewrites the `answer` schema for v2 and v3:

    "the answer, in prose. Cite the set of papers in `papers_from` rather than
     writing arXiv ids into this text."

`text` is the ONLY field `set_f1` and `judged_set_f1` read. So the system has
been instructed away from the one thing that scores, and pointed instead at a
citation that resolved on 7% of answers (D-116). Every "the answer named
nothing" finding in D-107 was partly this.

AN ARM, NOT A FIX. `--name-ids` reverses the instruction and gives an example.
It is not made the v3 default: v3 is what every recent result was measured on,
and rewording it silently is the D-062 failure -- a one-line prompt change that
moved the control while an arm was being read. `test_the_v1_tool_schema_is_frozen`
caught exactly that on the first attempt here, which is the second time that
test has earned itself.

Prior evidence the lever works: in D-107's decomposition `--simple-answer`,
which asks for a literal id list instead of a citation, had the highest print
rate of any arm -- 0.81 against 0.65 for the control.

AND KEEP WATCHING THE NOTATIONS, because the prompt will not hold. Four forms
observed, from two models:

    json         `{"text": ...}` anywhere, however wrapped     QwQ, qwen3-32b
    attrs        `<answer text="..." reason="answered"/>`      QwQ
    tag_per_arg  `<papers_from>set_1</papers_from>`            qwen3-32b
    prose        ids in the text, no structure                 both

`answer_syntax()` classifies each answer and the result travels in the run
record, so a fifth form arrives as a number rather than as a week of confusing
results. `harvest_answer_args` reads the arguments BY NAME out of any of them,
which is what survives a new model; a recogniser built from a list of observed
forms does not.

THE ORDER MATTERS. The harvest is a parser: it costs nothing, cannot regress,
and works whatever the model does. The prompt is a behaviour change that has to
be measured and can be undone by the next model. So the harvest is the floor
and the prompt is the arm, not the other way round.

## D-118 — Gabriel's questions, replayed: retrieval and the handoff lose the same amount

Wave-1 controls (54251/54252), every tool call replayed against the DIAS
database because stored previews were empty (see the instrumentation branch).
188 gold instances over 9 questions x 2 controls, one errored record excluded.

    named correctly                    59   31%
    reached, not named (handoff)       67   36%
    in the graph, never reached        62   33%
    not in the graph                    0

D-104's "retrieval is at 0.96, the whole loss is the handoff" was measured on
synthetic questions whose gold comes from the graph. On the physicist's
questions retrieval misses a third -- and of those 62 misses, 44 sit on papers
that CARRY an entity with the concept. The graph has it; search does not find
it. 10 are in a quote but no entity; 8 are absent.

By tool, gold papers reached: facets 79, search 52, subjects_of 38, papers_of 7.
facets found more gold than search and was invisible to every reach metric.

PER QUESTION -- the failure is different each time, which is the finding:

  gf-05  Higgs candidate     reach 2/16. 13 of 14 misses carry `H->bb candidate`,
                             `Diphoton system (H→γγ candidate)`, `Large-R jet
                             (H→bḇ candidate)`. Question says Higgs, labels say H,
                             in four notations. D-087 on a real question.
  gf-08  ee OR mumu          reach 24/24, named 1. subjects_of returned 201 and
                             224 rows; the model saw 25. Pure truncation.
  gf-07  ttZ + CR            reach 8/10, named 0 (or 3 wrong). search 27,
                             subjects_of 52 rows. Truncation + wrong picks.
  gf-01-condition / -met     reach 11/11 and 3/3, named 1 each, no step over 25
                             rows. "18 analyses use b-tagged jets" plus three
                             examples. Summarise-instead-of-list.
  gf-02  ABCD/sideband/matrix reach 6/11, named 6 -- perfect handoff. All 5
                             misses are `Matrix method ...` labels. The model
                             searched ABCD; the question named three concepts.
  gf-04  unfolding           reach 10/18, named 10 -- perfect handoff. Misses:
                             3 concept-absent, 3 quote-only (TUnfold in text),
                             2 entity-present. Mostly a graph gap.
  gf-03  HistFitter          reach 4/5, named 4. One quote-only miss.
  gf-01  b-jets AND MET,     reach 7/8, named 6, EIGHT false positives -- and all
         searches            eight are category=search, like the true positives.
                             The facet tags are on all 14; Gabriel says 8 do not
                             REQUIRE both in the selection. Tag ≠ selection
                             (D-099, now located). No query arm fixes this.

TWO CONSEQUENCES. The handoff loss is not universal: gf-02 and gf-04 hand off
perfectly, gf-08 loses 23/24, and the difference is whether the result fit in
25 rows. "The handoff" is really TRUNCATION, a much narrower claim. And every
arm now has a named failure:

    surface-form miss, entity present   44   encoder / aliases / multi-search
    truncation at 25 rows               31   rerank, papers_from, max_rows
    summarise instead of list           12   --name-ids, gate
    quote-only extraction gap           10   --index-quotes
    tag ≠ selection (FP)                 8   graph precision, not a query fix

## D-119 — the kind filter starves some questions and feeds others; fall back, don't choose

Found by the offline retrieval bench (report 1, 2026-09-08). Replayed against
the DIAS DB, "Higgs" with `kind=detector_object` -- as the model typed it --
reaches 2 of gf-05's 16 gold papers; the same query without the kind reaches
13. The Higgs-candidate entities on those papers are typed event_region (38),
physics_process (39), observable (38), result (29): only 9 are detector_object.
The model's guess was reasonable; the graph typed the concept differently, and
the model cannot see that.

But the SAME filter helps elsewhere: gf-08 +2, gf-01 +3. Over the five distinct
kinded searches in the controls, with-kind reaches 61 of 79 gold papers,
without 71. So neither policy is right.

`--kind-fallback`: run the kinded search, then the unfiltered one, append what
is new, tell the model how many entities of other kinds matched, hand the union
to the critic. Predicted offline, before building: 61 -> 74 of 79 (+13), gf-05
2 -> 13, one extra local search per kinded query, zero LLM calls.

An arm, default off: wave 2 must stay comparable with wave 1. To be confirmed
live on Gabriel's questions via OpenRouter before promotion.

## D-120 — decide the order where the cut is, and on every tool that found the gold

D-118 replayed which tool actually reached Gabriel's gold papers: facets 79,
search 52, subjects_of 38, papers_of 7. The rerank hook (D-113) covered
papers_of only -- the tool that found the fewest.

The cut is one place for every tool: `_render_rows` at `max_rows=25`, and rows
arrive in retrieval order. Search rows already carry the critic's rung as
`bears_on`; with 60 hits and a 25-row window an `exact` hit ranked 40th by
BM25 is dropped while an `unrelated` one ranked 3rd is shown to the model.

Under `--rerank` now: entity rows are stable-sorted by rung at the render site
(exact, broader, unjudged, unrelated -- retrieval order breaks ties), and the
graded paper ranker also runs on `facets` and `contents_of` results, not only
`papers_of`. Membership, counts and sets are unchanged; only which side of the
window a row lands on. Off by default; `--max-rows` is now a knob so the
window itself can be an axis.

Still uncovered: `subjects_of` rows are entities with no critic verdict (the
critic runs only on search). gf-08's 224-row truncation was there. Options are
a rung by paper overlap with the kept set, or the graded judge on labels; not
built until the cheaper levers (`--max-rows`, rung ordering) are measured.

## D-121 — on Gabriel's questions the index is not the bottleneck; every retrieval miss is in the query

Offline bench, no LLM, concept queries, limit 60, DIAS database, 2x2 over the
index variants:

    index                  gold reached   of the 31 papers never reached live
    values=0 quotes=0        105/106              30/31
    values=1 quotes=0        105/106              30/31
    values=0 quotes=1        105/106              30/31
    values=1 quotes=1        104/106              30/31

`--index-values` and `--index-quotes` change nothing here. The 10 misses D-118
called "quote-only" were quote-only FOR THE KEYWORD; the papers themselves are
reachable through other entities they carry. D-085's 31% (values) was a real
effect on a different question set and stands; it does not transfer to these.

With D-119 this closes classes A and D together: the surface forms are there,
the retriever finds them, and the 62 "in the graph, never reached" gold papers
are lost between the question and the search call --

    the `kind` filter starving a query          gf-05   13    --kind-fallback
    one facet where the question named three    gf-02    5    multi-concept search
    the model typing one concept, not the set   (as-typed 88% vs concept 99%)

"What is missing for retrieval to be perfect" on these questions is therefore
not coverage or embeddings: it is that the agent issues one narrow query where
the question implies several broad ones. Query formulation, not the index.

## D-122 — class B predicted offline: the window is the lever, ordering cannot be judged without the critic

Every wave-1 control step whose result exceeded 25 rows and contained gold,
replayed against the DIAS DB. Gold papers VISIBLE to the model (inside the
first `max_rows` rows), summed over those steps:

    max_rows        25      50     100     250
    visible         81      94     101     108   of 108

    gf-08 subjects_of (202 rows, 20 gold):   3 ->  8 -> 13 -> 20
    gf-07 subjects_of ( 52 rows,  8 gold):   3 ->  7 ->  8 ->  8

`--max-rows 100` recovers 20 of the 27 gold papers the 25-row window hides;
250 recovers all. The window is the cheapest lever in the project and was not
a knob until today.

Rung/kept ORDERING showed no effect in this replay -- and that is not evidence
against it. The replay ran with `critic=None`, so no `*_kept` set exists and
the kept-signal had nothing to act on; and on `search` rows all gold was
already inside the window (8/8, 24/24), so no ordering can add. D-113's offline
result (F1 0.591 -> 0.654) stands; whether the live mechanism realises it can
only be measured live, with the critic on.

Order of live tests on the OpenRouter lane, cheapest first: `--max-rows 50` and
`100` on gf-08/gf-07; then `--rerank` on top; then `--subgoals` on gf-02.

## D-119 addendum — the same three collapses on a second model

OpenRouter control, qwen3-32b + llama-3.1-8b critic, Gabriel's 9 x 3 repeats
(run 20260907T233950-hepkg-1914), the comparison target for --kind-fallback:

    question          gold   f1    reach   named-gold/named   kinds the model typed
    gf-05 Higgs cand.   16  0.07   0.21        0.7 / 2.3      detector_object x3, physics_process x1
    gf-08 ee OR mumu    24  0.11   0.52        1.5 / 3.0      detector_object, channel, selection_requirement
    gf-02 ABCD/matrix   11  0.65   0.48        5.3 / 5.3      (facets, one value)
    gf-03 HistFitter     5  0.89   0.80        4.0 / 8.0
    gf-04 unfolding     18  0.67   0.65        9.3 /10.3

Same shape as QwQ on the cluster (D-118): gf-05 collapses on the kind filter,
gf-08 on truncation, gf-02 hands off perfectly at half the reach. The classes
are properties of the question-to-query step, not of the answering model --
which is what makes them fixable by mechanism rather than by model choice.
Typed judged_f1 0.435 +/- 0.086 over 27 records; the noise floor to beat.

## D-119 addendum 2 — confirmed firing live, and the gain on the searches actually issued

The in-flight OpenRouter fallback arm (20260908T002432-hepkg-9786) predates
the counter, so it was verified by replay: its gf-03 search ("HistFitter",
kind=statistical_method) gives set_1 = 11 with the fallback off and 77 with it
on (56 appended); the live record shows 77. It fired.

Replaying the run's own kinded searches against the DIAS DB, fallback off -> on:

    gf-03  HistFitter      statistical_method   4/5   -> 5/5    +56 entities
    gf-05  Higgs boson     detector_object      5/16  -> 13/16  +57
    gf-04  unfolding       statistical_method   14/18 -> 18/18  +42

Three of three searches gain, two reach every gold paper. The cost is ~50
extra entities per kinded search for the critic to judge -- which is what the
critic is for.
