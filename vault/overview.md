# Overview — current state

*Living document: always reflects the present. History lives in `decisions.md` and `logs/`.*
*Last updated: 2026-07-29*

> **→ [`system.md`](system.md) (v1, 2026-07-29) is now the reference document for what gets built
> next**: the query system and the agent layers on top of the graph, the evaluation design, the
> technology stack, and the order of work (S-01 … S-22). Read it after this file.
> The sections below still describe how the graph itself is built (M1 done, M2–M4 open).
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
