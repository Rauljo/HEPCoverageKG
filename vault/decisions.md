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
