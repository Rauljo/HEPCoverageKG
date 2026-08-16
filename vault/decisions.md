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
