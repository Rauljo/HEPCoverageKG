# Overview — current state

*Living document: always reflects the present. History lives in `decisions.md` and `logs/`.*
*Last updated: 2026-09-08*

> **State on 2026-09-08 (one week to submission).** The map of the system, its arms and what
> has been measured is [`system-inventory.md`](system-inventory.md); read it before the sections
> below, which describe how the graph was built. The evaluation has two lanes: DIAS (QwQ-32B,
> 164 questions, waves 1-3) and OpenRouter (qwen3-32b, Gabriel's 9 questions x 3 repeats, ~$0.25 a
> run). Findings that shape the write-up: the relevance critic is worth nothing on the 164
> (D-133); arm differences were formatting until `answer()` was recovered from XML and ids
> were asked for in the text (D-107, D-116, D-117); the losses split into retrieval, handoff
> and truncation (D-118, D-135); the stack that moves them -- kind-fallback, max-rows 100,
> answer-gate, enum-expand with a set cap, rerank on a 32B judge, the ranked answer -- reaches
> 0.562 against a 0.452 control on the 9 (D-139), with gf-08 alone worth +-0.05 of any such
> comparison (D-138). On the 164 the stack wins count and the Gabriel-10 judged score and
> ties or loses set_f1 depending on whether the control's footprint fallback is counted (D-142,
> D-143); on small-truth generated questions with qwen3-32b it doubles recall at equal precision
> and a judge-based strike removes gold (D-144). The first results section is drafted from
> [`analysis/controls-and-gabriel.md`](analysis/controls-and-gabriel.md): the controls by question type
> and Gabriel's nine questions one by one. On 2026-09-09: the results chapter's baseline
> section drafted with the user; free-SQL's plain control and constrained decoding built and
> measured (D-155, D-156) -- constrained selection is the answer-side mechanism for set questions,
> recall doubles at the footprint's precision. Logs: [`logs/2026-09-08.md`](logs/2026-09-08.md),
> [`logs/2026-09-09.md`](logs/2026-09-09.md).

> **→ [`system.md`](system.md) (v1.8, 2026-08-07) is now the reference document for what gets built
> next**: the query system and the agent layers on top of the graph, the evaluation design, the
> technology stack, and the order of work (S-01 … S-57). Read it after this file.
> **→ [`ground-truth.md`](ground-truth.md) (v1.0, 2026-08-14) is the methods reference for the
> evaluation gold**: why the supervisor's answers cannot serve as truth, the reference reader, the
> cascade and its two judges, what the fixes were measured to be worth, and the three things the
> finished gold is for. The evaluation chapter is written from that file (D-054 … D-059).
> **2026-08-02 — the order of work is agreed** (§5 Phase 3, blocks 0–7). **Step 0 is asking Gabriel
> for questions**, today. Harness designed but not built:
> [`ideas/eval-harness-design.md`](ideas/eval-harness-design.md). Two pipeline ideas recorded as
> ablation axes and deliberately *not* built yet (S-53 decomposition → set algebra, S-54 paraphrase
> fusion). **S-56: run the D-038 signature parse now** — prose reads but cannot be counted.
> **2026-08-02 (afternoon) — the harness is BUILT and running** (`hepcoveragekg/eval/`, **335 tests**).
> First real run: 10 questions × 3 repeats, **count_correct 0.75**, abstention 1.00, and **zero spread
> across repeats** — so the noise floor is ~0 and any ablation difference will be readable.
> **The generated question set does not work yet, and that is the day's finding**: truth must be
> *invariant to where the concept boundary is drawn* (**S-58**); 52% of concepts survive that test and
> **the 48% discarded is a direct measurement of how incomplete deduplication is**. Six question tiers
> defined (**S-59**), with per-paper questions as the backbone. **Dedup is now upstream of the question
> set.** Deep alias proposals exist (126,335 pairs) but must **not** be confirmed — ~50% precision,
> physics errors (**D-044**); they are useful as an ambiguity *filter* (**S-60**).
> See [`logs/2026-08-02.md`](logs/2026-08-02.md).
> **2026-08-03 — three question tiers exist, and the biggest finding is a gap they exposed.**
> **A coverage map could not read a paper**: every template went concept -> papers and nothing was
> its inverse, so *"which generators does analysis X use?"* produced 60 `unknown_entity_id` errors.
> `contents_of` (S-63, D-047) fixed it — `count_correct` 0.06 -> 0.50.
> **Tier A** 1,352 per-paper questions (exact truth, no dedup, reaches the 31% of assertions stored
> as free text) · **Tier B** 436 from 109 invariance-tested concepts · **retrieval** 720 whose truth
> is an entity id and which ambiguity cannot spoil. **S-66**: confirming merges GROWS the question
> set while filtering with them shrinks it, so reviewing the 111 pairs is what unlocks Tier B.
> Four measurement bugs, all metrics anchored to one code path. See
> [`logs/2026-08-03.md`](logs/2026-08-03.md).
> The sections below still describe how the graph itself is built (M1 done, M2–M4 open).
> **2026-08-01 — the evaluation is designed end to end** (§4 rewritten, S-32 … S-52). The reframing:
> **three layers get measured separately**, and almost everything being built is *the agent against
> the graph*, where **the graph itself is exact ground truth** — no judge needed for most of it.
> Gabriel's time buys **judge calibration and disagreement adjudication, never labels** (<2 hours).
> The **reference reader** (a third model, one pass per paper, constrained to our schema) is the only
> thing that measures *the graph against the papers*. See [`logs/2026-08-01.md`](logs/2026-08-01.md).
> **Two unblocks found** (D-041, D-042): Gabriel's extraction pipeline runs on our own vLLM
> (`--provider openai`), so more papers are bounded by cluster time, not by him; and
> `HEPKG_promopt_tests/GROUND_TRUTH.md` already holds **hand-written ground truth for ~17 papers** on
> final states — stale since 2026-07-05, but real.
> **2026-07-31 — the query layer works end to end.** *"How many analyses used Pythia?"* returns
> **58 papers / 344 facts / 367 assertions**, matching ground truth exactly, 100% grounded, 202
> evidence quotes, 6 seconds. Built: `query/{schema_card,templates,retrieve,planner,verify,graph}.py`,
> **280 tests**. The planner is a LangGraph state machine with checkpointing (S-16 fulfilled;
> LangChain deliberately not a dependency). See [`logs/2026-07-31.md`](logs/2026-07-31.md).
> One A100 on the cluster is failing with uncorrectable ECC (D-040) — serving TP=1 on a healthy card.
>
> **What changed on 2026-07-29**: the project's framing moved from *gap finding* to **coverage
> review** — helping a physicist see what has been covered while reviewing or planning. Gap finding
> is one use of that, now future work. **M3 became the critical path**: aggregate coverage questions
> are the backbone of the evaluation and none are answerable until final-state signatures exist
> (D-038 — and the long-standing "compile them from `count` qualifiers" plan was based on a wrong
> reading of the data; no such qualifier exists).

## What this project is

A typed knowledge graph mapping which HEP final states have been measured, by which
experiment (ATLAS/CMS), at which energy — to surface coverage gaps for BSM physics.

**Two tracks now exist, and the center of gravity has shifted (2026-07-23):**

1. **KG construction from given bundles (the current critical path).** Gabriel's acquisition
   pipeline (`HEPKG_promopt_tests`) extracts 60 ATLAS/CMS papers into versioned "bundles"
   (schema `hepkg-acquisition-v0.2`). My job = **import** those bundles into one queryable
   graph, preserving status/evidence/history. The facts are *given*; correctness isn't my job;
   the output is the graph. Graded by the repo's integration fixtures + count targets.
   Milestones 1–4. Design: `ideas/bundle-importer-design.md` (D-021..D-026).
2. **My own RAG extraction pipeline (repositioned, not retired).** The original
   harvest → RAG extraction → KG stack (below) still works and is verified, but it's no longer
   the way the KG gets built — it moves to the *payoff/quality arms* (gap finder, critic panel,
   held-out validation) and to a baseline comparison ("my extraction vs Gabriel's bundles").
   Exactly how prominent it stays is **still open** (possibly a supervisor conversation).

Patterns ported from DeepCollector (referenced, not forked). Supervisor provided
`graph_schema.py`/`graph_extractor.py` (kept untouched at repo root as reference).

## Milestone 1 — the bundle importer (active build)

- **Input:** `HEPKG_promopt_tests/pilot/bundles/*.json.gz` (60 papers, read-only).
- **Store:** SQLite as system of record; graph (NetworkX/Neo4j Community) as a projection from
  it, later (D-022).
- **Validation:** `jsonschema` shape gate + hand-written semantic layer; Pydantic deferred (D-023).
- **✅ MILESTONE 1 COMPLETE (2026-07-24):** all 8 build steps done, **108 tests green**. The CLI
  imports all 60 bundles and reproduces **14,188 / 11,309 / 2,555 / 324** exactly
  (`verify-counts` → target met), reimport is a no-op, and all 7 contract pass conditions hold.
  Run it: `python -m hepcoveragekg.cli import <bundles-dir>` then `verify-counts`.
- **Acceptance (the M1 gate, MET):** reproduce **14,188 / 11,309 / 2,555 / 324** by status across
  the 60 bundles, AND pass the 4 `examples/integration/` fixtures against the 7 pass conditions.
- **Key modeling:** paper=`arxiv_id` vs bundle=`bundle_id`; entity split (merged node +
  per-paper `entity_occurrence`, merge-never-abort); qualifiers lossless JSON (no promoted
  columns); completeness findings isolated; accepted-view = a filter (D-024).
- **Identity/conflict (D-027, corrects D-024):** `bundle_id` is **identity-only**
  (schema+paper+source+normalization) and is *stable across re-extractions* — so the whole-bundle
  hash is only the "byte-identical → no-op" fast path. **Conflict detection is per assertion**:
  new id → insert (corrections land here); only `status` differs → update + log; anything else
  differs → abort. Status transitions are preserved in `assertion_status_history` (D-028).
- **Modules:** `kg/` (store/schema.sql/queries/export/graph) · `ingest/` (reader/validate/
  canonical/importer) · `aliases/` (normalize; D-025) · `cli.py` · tests (D-026).
- **Build progress:** ✅ step 1 (store: 15 tables/view/indexes) · ✅ step 2 (`canonical.py`
  fingerprints, pinned byte-for-byte to the supervisor's ids) · ✅ step 3 (`reader.py` + vendored
  schema + `errors.py`; 60/60 bundles pass the shape gate) · ✅ step 4 (`validate.py`: semantic
  gate — 0 violations and 0 warnings across all 60 bundles) · ✅ step 5 (`importer.py`:
  **all 60 imported through the real importer reproduce 14,188 / 11,309 / 2,555 / 324 exactly**;
  order-independent entity merge) · ✅ step 6 (re-import guard: identical→no-op, conflict→abort,
  changed→skip+warn; all 60 imported twice = no change) · ✅ step 7 (**all 7 contract pass
  conditions green**; minimal `queries.py` with the trace query) · ✅ step 8 (`cli.py` + count gate
  — **M1 COMPLETE**) · **108 tests green**.
- **Verified against the data:** 0 signatures populated; assertion_ids never collide; entity_ids
  collide by design (347 shared, 334 divergent — merge); 487 IDs → 223 concepts on spelling
  alone (→ aliases layer); one quote backs up to 37 assertions (→ many-to-many mandatory).

## RAG pipeline layer status (repositioned track)

| Layer | Status |
|---|---|
| `harvesting/html_harvester.py` | **Working, verified** against arXiv 2307.01094. Tables not extracted (D-008). |
| `harvesting/paper_list.py`, `chain.py` | Stubs. Paper source may be a supervisor-provided dataset. |
| `config/schema.py` | Working. Ontology + 6/15 predicates + vocabs + plausibility bounds. NB: this is the *old RAG ontology*, distinct from the bundle JSON schema and from `kg/schema.sql`. |
| `extraction/state.py` | **Working, verified**: per-paper CatalogState + typed dataclasses. |
| `extraction/rag_engine.py` | **Working, verified end-to-end** (Groq + self-hosted vLLM); all 6 predicates populate. |
| `extraction/generate.py` | Working. Backend-agnostic (any OpenAI-compatible endpoint). |
| `kg/` | Stubs — now being filled by the **importer** design, not the old merger plan (D-024, D-026). |
| `tests/` | Empty stubs → M1 adds `test_import_counts.py`, `test_integration_fixtures.py`. |

## Infrastructure

- **LLM**: self-hosted vLLM on UCL DIAS cluster (`ssh dias`), 3× A100 80GB PCIe. Effective
  per-model limit **2 cards (~158GB)** (TP=3 fails head-divisibility; PCIe, no NVLink; GPU0+1
  same-NUMA fast pair, GPU2 cross-NUMA). Big model TP=2 on GPU0+1 + a 2nd small model TP=1 on
  GPU2 concurrently. Serving `NousResearch/Meta-Llama-3.1-8B-Instruct` via Apptainer
  `vllm-openai-v0.8.5.sif`; script `hpc/serve_vllm.sh` (needs `VLLM_API_KEY`); 24h job cap.
- **Access**: SSH tunnel `ssh -f -N -L 8000:<node>:8000 dias` (node changes per resubmission);
  `.env` points at `http://localhost:8000/v1`. Groq fallback commented in `.env` (D-012).
- **Embeddings**: BAAI/bge-small-en-v1.5, local CPU — **now also the aliases-layer embedding
  tier** (D-025), so the extraction stack feeds the importer's canonicalization.
- **Env**: Python 3.11.8 (pyenv), venv at repo root, deps in `requirements.txt` (+ `jsonschema`).

## Next steps (rough order)

*The live, themed to-do list is now [`backlog.md`](backlog.md) (Extraction / Graph / Agents +
the four milestones). Idea *designs* are in [`ideas/index.md`](ideas/index.md), also themed.*

**★ Deliverables for next Tuesday (agreed in the 2026-07-24 team meeting):**
1. **Build the knowledge graph** — the Neo4j graph-DB projection from the SQLite store (D-022).
2. **Get queries working** on the graph — depth TBD (simple filters → multi-hop physics signatures).
3. **Aliases layer working** — `aliases/normalize.py` Tier 1, producing the **first draft of the
   alias list** (recovers the b_jet/bjet/b-jet-style splits). = milestone 2 + D-025 + D-022 together.

Then:

1. ✅ **Milestone 1 DONE** — importer complete (`ideas/bundle-importer-design.md`), 108 tests,
   target reproduced. DB populated at `data/processed/hepkg.db`.
2. Then aliases Tier 1 (`aliases/normalize.py`), deterministic export, trace-query polish.
3. Send Gabriel the 7 onboarding questions (signatures + corpus access unblock the most).
4. Milestones 2–4 (trace; accepted-view + a real physics query; re-import after review).
5. Later arms: gap finder + held-out validation (payoff), critic panel (quality) — these lean on
   the repositioned RAG pipeline.

## Pending questions

- **RAG-pipeline repositioning** — importer is the M1–4 backbone; how much the RAG stack features
  vs. becomes payoff/baseline infrastructure is not formally decided (D-021 note).
- The 7 Gabriel questions (`reports/2026-07-23-hepkg-repo-onboarding.md`): signatures plan,
  qualifier normalization ownership, disguised-edge relinking, category vocabulary, six-way
  "no accepted result", corpus access, retrieval alignment.
- Promoted qualifier columns — deferred; add via generated columns when a query needs them.
- Repo may merge into a supervisor repository — vault stays self-contained in `vault/`.

## Dissertation shape (emerging, planning-stage)

Three acts, each separately evaluable: **(1) baseline pipeline** — now the **bundle importer**
(harvest→extract still exists but repositioned) → **(2) quality**: critic panel + reconciliation
(`ideas/multi-agent-extension.md`) → **(3) payoff**: deterministic gap enumeration + reasoning
layer producing ranked, literature-checked gap hypotheses (`ideas/gap-hypothesis-system.md`,
`ideas/held-out-gap-validation.md`). Critical path stays the baseline: import the bundles,
reproduce the counts, then the aliases layer (load-bearing for correct coverage axes) and the
gap finder.
