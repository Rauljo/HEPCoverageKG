# HEPCoverageKG — Progress Log

Status as of 2026-07-07. This records what's been built and, more importantly,
*why* — the decisions and dead ends along the way, so future sessions (or
the dissertation write-up) don't have to re-derive them.

## Project goal

A typed knowledge graph mapping which HEP final states have been measured,
by which experiment (CMS/ATLAS), at which energy — to surface coverage gaps
for BSM physics models. Three-layer pipeline: harvest (InspireHEP → arXiv →
HEPData) → RAG extraction → knowledge graph (SQLite + NetworkX, Neo4j
later). Built by referencing (not forking) a prior project, DeepCollector,
porting its patterns where they still apply and rebuilding where they don't.

## Repository layout

```
HEPCoverageKG/
├── hepcoveragekg/
│   ├── harvesting/
│   │   ├── base_harvester.py    # fetch/parse abstract interface
│   │   ├── html_harvester.py    # generic fetch+parse, arXiv-verified
│   │   ├── paper_list.py        # fixed hand-picked (source, URL) list - stub, not populated yet
│   │   └── chain.py             # drives html_harvester per paper - stub, not implemented yet
│   ├── extraction/
│   │   ├── state.py             # CatalogState + HybridRetriever + typed dataclasses
│   │   ├── rag_engine.py        # cellular RAG loop
│   │   └── generate.py          # plain LLM call, backend-agnostic
│   ├── kg/                      # not started: store.py, graph.py, merger.py, queries.py are empty stubs
│   ├── config/
│   │   ├── schema.py            # ontology + PREDICATE_SCHEMA + plausibility/missing-value config
│   │   └── settings.py          # empty stub, not started
│   └── pipeline.py               # empty stub, not started
├── hpc/
│   └── serve_vllm.sh             # SLURM job serving vLLM on the DIAS cluster
├── tests/                        # empty stubs, no tests written yet
├── graph_schema.py, graph_extractor.py   # supervisor-provided reference files, kept untouched at root
├── requirements.txt, .env(.example), .gitignore, .python-version, README.md
```

## Architecture decisions made this session

- **Reference DeepCollector, don't fork it.** Port patterns (schema-as-config,
  cellular RAG, confidence-beats, hybrid retrieval) fresh into a new codebase,
  rather than branching DeepCollector itself.
- **Harvesting scope, deliberately narrow for now**: a *fixed*, hand-picked
  list of HTML pages, not a dynamic crawl/discovery process. This may expand
  later, but isn't needed for the first working pipeline.
- **Papers may not come from live arXiv fetches at all** — the supervisor may
  provide a downloaded dataset instead. Didn't block on this: harvesting's
  `fetch()`/`parse()` split, and `extraction/state.py`'s dependence on just a
  `HarvestedPaper` object (not on *how* it was obtained), means this can be
  decided later without rework.
- **`html_harvester.py` collapsed from three separate per-source harvesters**
  (inspire/arxiv/hepdata) **into one generic harvester** driven by a fixed
  page list, since all three sources reduce to "fetch a known URL, parse its
  HTML" rather than needing distinct dynamic-query logic per source.
- **`base_harvester.py`'s interface was changed from DeepCollector's own**:
  DeepCollector's `execute_harvest(state: CatalogState)` mutates one shared,
  incrementally-discovered catalog across a whole project's worth of
  datasets. We have no equivalent shared state at harvest time — each paper
  is fetched/parsed independently, and its retrieval index is only built
  later, per paper, in the extraction layer. Interface is now a plain
  `fetch(url) -> html` / `parse(html, url) -> document` pair.
- **Table extraction deliberately not implemented** in `html_harvester.py`.
  HEPData (a separate stage in the harvest chain) is the authoritative
  source for structured numeric results; arXiv HTML tables are secondary and
  can be added later only if HEPData coverage has gaps.
- **`CatalogState` (ported from DeepCollector's `core/state.py`) is rescoped**
  from "one project, many datasets, many flat fields" to "one paper, one
  Result entity, many predicate assertions." `self.catalog` keeps its name
  for continuity but now holds `list[CoverageAssertion]` instead of
  DeepCollector's `list[CatalogItem]`.
- **The supervisor's `graph_schema.py`/`graph_extractor.py` were kept as
  untouched reference files at repo root**, same treatment as DeepCollector:
  their ontology (entity kinds, predicates) and dataclasses were ported into
  `config/schema.py`/`extraction/state.py`; their deterministic regex
  extractor logic was *not* ported (RAG-based extraction is the chosen
  method), only its controlled-vocabulary term lists were reused as seed
  values.
- **`PaperExtraction`'s primary id flipped from `cds_id` to `arxiv_id`**
  (required, with `cds_id` optional) — our harvest chain is arXiv-first, not
  CDS-based like the supervisor's original.
- **Predicates are split into single-value vs multi-value** (`multi_value`
  flag in `PREDICATE_SCHEMA`). A result has exactly one collision energy
  (two different answers are a conflict to arbitrate via confidence-beats),
  but can target several physics processes at once (two different answers
  are both true facts to keep, not a conflict). `CatalogState.get_assertion`/
  `update_assertion` key on `(predicate, object_label)` for multi-value
  predicates, and on `predicate` alone for single-value ones.
- **Dedup principle for the future `kg/merger.py`**: dedup happens at the
  *vocabulary entity* level (merge "SUSY" and "supersymmetry" into one node),
  never at the *Result/Paper* level. Two different papers reporting the same
  final state must both remain as separate `Result` nodes, each linked to
  the same shared entity node — that multiplicity *is* the coverage signal
  the whole project measures, not noise to collapse away.
- **`generate.py` is backend-agnostic by design**: any OpenAI-compatible
  chat completions endpoint works (Groq, self-hosted vLLM, others), switched
  purely via `.env` (`LLM_BASE_URL`/`LLM_MODEL_NAME`/`LLM_API_KEY`), never a
  code change. This was a direct response to wanting zero marginal cost
  without losing the option to go back to a hosted API easily.
- **`.venv` lives at the repo root**, not inside the `hepcoveragekg` package
  folder — keeps the runtime environment separate from source code.

## Known open gaps / deliberately deferred

- `result_type` (on `PaperExtraction`) has no defined controlled vocabulary
  yet, anywhere — not in this project, not in the supervisor's own files.
  Likely needs values like search / measurement / combination / observation,
  and probably matters for coverage-gap logic (only "search"-type results
  should count toward a BSM coverage gap). Needs resolving with the
  supervisor before `kg/queries.py` is built.
- `result_has_final_state`'s controlled vocabulary is currently a flat
  detector-object pick list, but a real final state is a *multiplicity
  composition* (e.g. "2 electrons + 1 jet + MET"), not a single label. This
  is the most important pre-extraction-at-scale fix still outstanding.
- Confidence-bounds behavior on malformed LLM confidence values (hard-reject
  vs auto-normalize) was resolved pragmatically (normalize in
  `rag_engine.py`, validate strictly in `state.py`'s dataclasses as a
  backstop) but the two philosophies (DeepCollector's forgiving normalize vs
  the supervisor's strict raise) were never explicitly reconciled beyond that.
- `EXTRACTION_METHODS` currently only distinguishes technique (`llm` /
  `regex` / `metadata`). A future requirement was flagged: distinguishing
  plain single-pass RAG retrieval from an eventual orchestrated/multi-agent
  retrieval expansion. Not built - noted for whenever that expansion happens.
- llama-3.1-8b-instant (both via Groq and self-hosted) does not reliably
  follow "respond with one JSON object" instructions for controlled-vocab
  "choose all that apply" questions - it sometimes emits one JSON object per
  candidate term instead, each with its own confidence. `rag_engine.py`'s
  parser handles both shapes, but it's worth knowing this model's behavior
  isn't fully predictable prompt-to-prompt.
- The model's self-reported confidence is not well calibrated in testing -
  almost everything came back at 1.00 regardless of how well-supported the
  claim actually was. Plausibility/vocab-membership checks are doing the
  real filtering work right now, not the confidence number. Worth
  re-examining before leaning on confidence for anything downstream (e.g.
  merge tie-breaking in `kg/merger.py`).
- `paper_list.py` and `chain.py` are still empty stubs - no fixed paper list
  exists yet, and the harvest-chain driver (InspireHEP → arXiv → HEPData per
  paper) hasn't been implemented.
- `kg/` (store.py, graph.py, merger.py, queries.py), `config/settings.py`,
  and `hepcoveragekg/pipeline.py` are all empty stubs - the knowledge graph
  layer hasn't been started at all yet.
- No tests written yet (`tests/` are empty stub files).

## What's implemented and verified

### Harvesting (`hepcoveragekg/harvesting/`)
- `base_harvester.py`: abstract `fetch`/`parse` interface.
- `html_harvester.py`: `HTMLHarvester.harvest(url)` fetches (requests
  session, rotating User-Agent, retry/backoff) and parses arXiv's
  LaTeXML/ar5iv HTML rendering into a `HarvestedPaper(source_url, title,
  abstract, sections)`. Built and tested against a real paper, arXiv
  2307.01094 (ATLAS SUSY search, "2 same-sign or 3 leptons"):
  - Handles custom-macro papers (e.g. ATLAS's `\AtlasTitle`/`\AtlasAbstract`)
    where LaTeXML can't resolve the macro and title+abstract collapse into
    one paragraph with the literal macro name as an inline separator - falls
    back to splitting on that pattern when the standard `ltx_title_document`/
    `ltx_abstract` tags aren't present.
  - Strips redundant MathML `<annotation>`/`<annotation-xml>` fallback
    encodings before extracting text - without this, numeric values like
    luminosity/energy came out garbled (duplicated TeX/plain-text renderings
    concatenated together).
  - Extracts only top-level `<section class="ltx_section">` elements
    (subsections nest inside their parent, so parent text already includes
    them); explicitly excludes the bibliography section.
  - Verified against only one ATLAS paper - CMS papers may use different
    custom macros, untested.

### Extraction (`hepcoveragekg/extraction/`)
- `state.py`:
  - `EvidenceSpan`, `EntityRef`, `CoverageAssertion`, `PaperExtraction`
    dataclasses (ported from the supervisor's `graph_schema.py`), each with
    a `validate()` method.
  - `HybridRetriever`: combines vector (embedding) and BM25 (keyword)
    retrieval over the same chunks, deduped by node id.
  - `CatalogState`: builds a per-paper `VectorStoreIndex` + BM25 index from
    a `HarvestedPaper`'s title/abstract/sections (chunked via
    `SentenceSplitter`), and accumulates `CoverageAssertion`s via
    `get_assertion`/`update_assertion` (confidence-beats: fill if missing,
    confirm-if-higher-confidence if the new value agrees, refine-if-
    confidence-at-least-as-high if it disagrees - keyed by predicate alone
    for single-value predicates, by `(predicate, object_label)` for
    multi-value ones). Also has `capture_confidence_metrics()` for
    completeness/avg-confidence reporting.
  - Verified end-to-end against the real harvested paper with the
    BAAI/bge-small-en-v1.5 embedding model: indexing (13 sections → 58
    chunks), hybrid retrieval (correctly surfaced the right section for an
    energy/luminosity query), and confidence-beats logic (fill / confirm /
    reject-lower-confidence-disagreement) all confirmed working.
- `generate.py`: `generate(prompt, max_tokens, temperature) -> str`, a thin
  wrapper around the `openai` Python client pointed at a configurable
  `LLM_BASE_URL`/`LLM_MODEL_NAME`/`LLM_API_KEY`. Works against both Groq
  (its API is itself OpenAI-compatible via `https://api.groq.com/openai/v1`)
  and a self-hosted vLLM server, with zero code changes between the two.
- `rag_engine.py`: `RAGEngine.run(state, result_label)` loops over
  `config/schema.py`'s `PREDICATE_SCHEMA`, retrieves context per predicate
  via the paper's hybrid retriever, builds a prompt, calls `generate()`,
  parses the JSON response, validates it (plausibility bounds for numeric
  qualifier fields, controlled-vocabulary matching for categorical fields),
  and applies the result via `CatalogState.update_assertion`. Handles two
  response shapes from the LLM: a single JSON object (comma-joined value for
  multi-pick answers) and llama-3.1-8b-instant's tendency to instead emit
  one JSON object per candidate vocab term (a per-term verdict) - the latter
  turned out to carry richer information (per-term confidence) so the parser
  uses it directly rather than fighting the model into one format.
  Verified end-to-end against the real paper, both via Groq and via the
  self-hosted vLLM server: all 6 `EXTRACTED_PREDICATES` populated
  correctly, multi-value predicates correctly split into separate
  assertions per matched term.

### Config (`hepcoveragekg/config/schema.py`)
- Ontology (`ENTITY_KINDS`, `ASSERTION_PREDICATES`, `ASSERTION_STATUSES`,
  `EXTRACTION_SUPPORT`, `EXTRACTION_METHODS`, `ALLOWED_EXPERIMENTS`) ported
  from the supervisor's `graph_schema.py`.
- Controlled vocabularies (`DETECTOR_OBJECT_VOCAB`, `PHYSICS_PROCESS_VOCAB`,
  `OBSERVABLE_VOCAB`, `BACKGROUND_VOCAB`) seeded from the supervisor's
  `graph_extractor.py` regex term lists.
- `PREDICATE_SCHEMA`: 6 of 15 ontology predicates have RAG query templates
  (`result_has_final_state`, `result_has_collision_system`,
  `result_uses_dataset`, `result_targets_process`,
  `result_measures_observable`, `result_estimates_background`); the other 8
  are listed in `DEFERRED_PREDICATES` (kinematic selections, systematics,
  numeric result values, region links, background-estimation methods) -
  deliberately out of scope for this first version.
- `PLAUSIBILITY_THRESHOLDS` (energy 0.9-14 TeV, luminosity 0.001-3000 fb⁻¹)
  and `MISSING_DATA_PLACEHOLDERS`, adapted from DeepCollector's pattern.

## LLM hosting: what happened and why

1. Started with Groq (`llama-3.1-8b-instant`), free-tier hosted API - this
   is what the extraction layer was originally built and tested against.
2. User wants zero marginal cost even at full scale, so decided to move to
   self-hosted inference. `generate.py` was already isolated enough that
   this only ever meant changing *what's behind* the `generate()` function,
   never its callers.
3. Considered running locally via Ollama on the user's Mac (Apple M5, 16GB)
   - rejected; user has HPC access instead and didn't want to tie up their
     own machine.
4. User has SLURM access to UCL Physics & Astronomy's DIAS cluster (3x
   A100, confirmed 80GB each - cluster docs undersold this as 40GB),
   already SSH-key-configured (`ssh dias` works directly through the UCL
   gateway).
5. Chose vLLM as the serving framework (DeepCollector's own `settings.py`
   already had an anticipated-but-unused `VLLM_HIGH_THROUGHPUT` profile for
   exactly this scenario). Chose Apptainer (already configured on this HPC
   account) + vLLM's official Docker image over a conda/pip install,
   specifically to sidestep vLLM's well-known CUDA/PyTorch/flash-attention
   version fragility against the cluster's older build toolchain
   (GCCcore 11.2.0-era).
6. Practical issues hit and resolved, in order:
   - `meta-llama/Llama-3.1-8B-Instruct` is gated on HuggingFace; the
     account's cached token wasn't approved for it (confirmed via a direct
     HTTP 401 on the weights file). Switched to the ungated mirror
     `NousResearch/Meta-Llama-3.1-8B-Instruct` (identical weights) rather
     than waiting on Meta's manual approval.
   - `/tmp` is node-local, not shared across the cluster - SLURM job output
     must be written under the shared home directory instead.
   - The `vllm/vllm-openai:latest` image failed with "NVIDIA driver too old
     (found version 12040)" - the image's bundled CUDA (12.9) exceeded what
     the cluster's driver (550.163.01, CUDA 12.4 max) supports; containers
     share the host's kernel driver, so this can't be fixed by any image
     config, only by picking an image built against an older CUDA. Checked
     vLLM's Dockerfile history directly on GitHub across versions and found
     `v0.8.5` (April 2025) is the newest release still on CUDA 12.4.1
     before `v0.9.0` moved to 12.8.1.
   - Model loading then failed with `unable to mmap ... Cannot allocate
     memory` on a ~5GB safetensors shard - the SLURM job's `--mem 32G`
     cgroup limit was too tight, since mmap'd file pages count against it.
     Fixed by raising to `--mem 128G`.
   - After both fixes, the server started cleanly and served real requests
     end-to-end through the full extraction pipeline.
7. Access pattern settled on: SLURM job runs vLLM on a GPU compute node
   (not directly reachable from outside the cluster); SSH tunnel through the
   `dias` login node (`ssh -L 8000:<node>:8000 dias`) forwards a local port
   to it, working from anywhere the user can SSH to the cluster. The
   compute node's hostname changes on every resubmission, so the tunnel
   command needs redoing each time (`.env` itself never changes, since it
   always points at the local tunnel endpoint).
8. `hpc/serve_vllm.sh` was committed to the repo as the reusable submission
   script, with the API key parameterized via a required `VLLM_API_KEY` env
   var rather than hardcoded (the actually-running cluster copy still has
   the real key hardcoded, but that copy isn't version-controlled).

## Next steps (not yet decided/started)

- Populate `paper_list.py` with an actual fixed paper list (blocked on
  either the supervisor's dataset or a hand-picked InspireHEP list).
- Implement `chain.py` to drive InspireHEP → arXiv → HEPData per paper.
- Resolve `result_type`'s vocabulary and the final-state multiplicity
  modeling gap before running extraction at scale.
- Build `kg/store.py` (SQLite schema) and `kg/graph.py` (NetworkX
  construction), then `kg/merger.py` (entity-level dedup) and
  `kg/queries.py` (coverage-gap queries).
- Write tests (currently all stubs).
