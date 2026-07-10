# Overview — current state

*Living document: always reflects the present. History lives in `decisions.md` and `logs/`.*
*Last updated: 2026-07-10*

## What this project is

A typed knowledge graph mapping which HEP final states have been measured, by which
experiment (ATLAS/CMS), at which energy — to surface coverage gaps for BSM physics.
Three-layer pipeline: **harvest** (fixed list of HTML pages) → **RAG extraction**
(per-paper, per-predicate) → **knowledge graph** (SQLite + NetworkX, Neo4j later).
Patterns ported from DeepCollector (referenced, not forked). Supervisor provided
`graph_schema.py`/`graph_extractor.py` (kept untouched at repo root as reference).

## Layer status

| Layer | Status |
|---|---|
| `harvesting/html_harvester.py` | **Working, verified** against arXiv 2307.01094 (ATLAS SUSY). Handles custom-macro title/abstract fallback + MathML annotation stripping. Tables not extracted (D-008). |
| `harvesting/paper_list.py`, `chain.py` | Stubs. No paper list exists yet — may come as a supervisor-provided downloaded dataset instead of live fetches. |
| `config/schema.py` | Working. Ontology + 6 of 15 predicates with query templates (rest in `DEFERRED_PREDICATES`) + vocabs + plausibility bounds. |
| `extraction/state.py` | **Working, verified**: CatalogState (per-paper hybrid index + assertion accumulation with confidence-beats), typed dataclasses. |
| `extraction/rag_engine.py` | **Working, verified end-to-end** against the real paper via both Groq and self-hosted vLLM: all 6 predicates populate, multi-value splitting works. |
| `extraction/generate.py` | Working. Backend-agnostic (any OpenAI-compatible endpoint via env vars). |
| `kg/` (store, graph, merger, queries) | Stubs. Design planned (see D-013..D-016 and `ideas/open-vocab-reconciliation.md`). |
| `tests/` | Empty stubs. |

## Infrastructure

- **LLM**: self-hosted vLLM on UCL DIAS cluster (`ssh dias`), GPU partition (3× A100 80GB).
  Serving `NousResearch/Meta-Llama-3.1-8B-Instruct` via Apptainer image
  `~/hepcoveragekg_setup/images/vllm-openai-v0.8.5.sif`. Submission script: `hpc/serve_vllm.sh`
  (requires `VLLM_API_KEY` env var). Jobs live 24h max, then need resubmission.
- **Access**: SSH tunnel `ssh -f -N -L 8000:<node>:8000 dias` — node changes per
  resubmission (`squeue -u $USER -h -o "%N"`); `.env` always points at `http://localhost:8000/v1`.
- **Fallback**: Groq config kept commented in `.env` — switching backends is env-vars-only (D-012).
- **Embeddings**: BAAI/bge-small-en-v1.5, local CPU, loaded via `Settings.embed_model`.
- **Env**: Python 3.11.8 (pyenv), venv at repo root, deps in `requirements.txt`.

## Known model quirks (measured, not guessed)

- llama-3.1-8b sometimes ignores "one JSON object" on choose-all-that-apply questions and
  emits one object per candidate term — `rag_engine._extract_json_objects` handles both shapes.
- Self-reported confidence is uncalibrated (~everything 1.00). Plausibility/vocab checks do
  the real filtering; don't lean on confidence downstream without revisiting.

## Next steps (rough order)

1. Populate `paper_list.py` + implement `chain.py` (blocked on paper-source decision).
2. Resolve final-state representation (`ideas/final-state-representation.md`) — most
   important pre-scale fix — and `result_type` vocabulary (`ideas/result-type-vocabulary.md`, needs supervisor).
3. Build `kg/store.py` + `kg/graph.py`, then `kg/merger.py` per the reconciliation design.
4. Implement consistency-check + leftovers passes (D-018, D-019).
5. Tests.

## Pending questions

- Final-state composite node vs object fan-out → `ideas/final-state-representation.md`
- `result_type` vocabulary → `ideas/result-type-vocabulary.md` (supervisor input needed)
- Hybrid `vocab_policy` per predicate: designed, leaning yes, **not explicitly signed off** → `ideas/open-vocab-reconciliation.md`
- Repo may be merged into a supervisor-created repository in the coming weeks — vault is
  self-contained in `vault/` to move atomically.
