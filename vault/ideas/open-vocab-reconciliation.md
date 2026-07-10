# Open vocabulary + batch reconciliation

**Status**: discussed — core mechanism adopted (D-016); per-predicate `vocab_policy` designed and leaning yes, **not explicitly signed off**.
**Discussed**: 2026-07-09, 2026-07-10.

## Problem

Extraction snaps LLM answers to closed vocab lists and **silently drops** non-matches
(`rag_engine._extract_vocab_terms`). A paper searching for leptoquarks (not in
`PHYSICS_PROCESS_VOCAB`) contributes zero assertions — invisible coverage, the worst
failure mode for a coverage-mapping project. The seed vocabs came from
`graph_extractor.py`'s regex demo lists, not a curated taxonomy; the real BSM landscape
(leptoquarks, HNLs, dark photons, vector-like quarks, ALPs, SUSY sub-modes…) far exceeds them.

## Proposed design (per-predicate, not global)

Add `vocab_policy` to each PREDICATE_SCHEMA entry:
- **`closed`** — `detector_object` only: what a detector reconstructs is a physically
  complete fixed set; opening it admits noise, discovers nothing.
- **`open_with_vocab`** — physics_process, background, observable (observable is the
  borderline case, user to decide): prompt says *"choose from this list; if none genuinely
  apply, answer `NEW: <short label>`"*. Vocab anchors common cases (precision kept),
  escape hatch adds recall. `NEW:` labels become **provisional** nodes (flagged, so
  queries can include/exclude them).

## Reconciliation loop (adopted, D-016)

Every N papers, a batch script (~150 lines, NOT an agent — no planning/tool-choice exists in the task):
1. Collect node labels of one entity kind.
2. Nominate candidate pairs: bge-small embedding similarity + string similarity (difflib).
   Nomination needs recall not precision — false nominations cost one LLM call + one human glance.
3. LLM adjudicates each pair — prompt grounded with each label's stored **EvidenceSpan
   snippets** (the sentences it was extracted from) plus the vocab-term descriptions.
4. **User confirms** → alias row appended (`"SUSY" → supersymmetry`). Non-destructive:
   raw labels stay on assertions forever; canonicalization resolves through the alias
   table at build time; undo = delete a row. Genuinely-new labels get **promoted into
   the vocabulary** (data change with provenance — an LLM never edits prompt templates).

Ported from UniversalOracle (deepcollector/kb/merger.py), keeping its tiered shape:
hard blocks first → cheap signals → auto-accept only strong matches → LLM only for the
ambiguous middle → pair cache + call budget + errors-default-to-rejected. Changed: our
signals are label+evidence (their URL/dataset-shape fingerprints don't exist for short
physics labels); offline batch not inline; reversible aliases not golden-record collapses;
human confirm not auto-apply. We need our own domain hard-blocks (e.g. two different
collision energies never merge).

## Explicitly rejected (kept for the write-up)

- **Agent-system framing** — it's a deterministic pipeline with one LLM-judgment step.
- **Auto-applied merges** — NELL's semantic-drift lesson; graphs rot quietly.
- **LLM-tuned prompts as the feedback loop** — unauditable behavior drift; feedback writes alias/vocab *data* instead.
- **Physics-specific embeddings (SPECTER/SciBERT-class)** — nomination only needs recall,
  which is threshold-tunable; upgrade ladder if misses are *observed*: (1) embed label
  together with its evidence snippet (no new model), (2) only then a domain embedder.
- **External physics-theory RAG for the judge** — one-line vocab descriptions + evidence
  snippets cover it; hard pairs route to a bigger model before anyone builds a theory index.

## Open

- Explicit sign-off on `vocab_policy` values per predicate (esp. observable).
- N (papers per reconciliation batch) — decide when there's a paper list.
- Alias table storage (likely a table in the SQLite store) — decide with `kg/store.py`.
