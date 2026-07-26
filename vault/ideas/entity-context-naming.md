# Entity-context naming (reuse existing entities at extraction time)

**Status**: discussed — Raul's proposal, 2026-07-25. The *upstream* counterpart to the aliases
layer: prevent id duplication at extraction instead of reconciling it after. Related to
[[open-vocab-reconciliation]] (the `NEW:` mechanism) and [[researcher-feedback-loop]]; in tension
(softly) with D-017.

## The idea

Give the LLM, in its context, the entities **already created**, so it **reuses an existing id**
(`hepkg:object:b_jet`) instead of inventing a new spelling (`bjet`, `b-jet`). Fixes the id split
(and the namespace mess) at the source.

## Why it's defensible (the D-017 distinction)

D-017 rejected feeding prior *answers* into later *measurements* (anchoring, correlated errors,
broken provenance). This is **softer and different**: it feeds the entity **vocabulary** (node
names) for *naming consistency*, not prior *facts* into a *measurement*. It's giving the model a
controlled vocabulary to snap to — the same role `vocabulary.py` plays, but for ids and actually
*used at construction*, not just checked.

## The clean framing: `NEW:` applied to ids

It's the open-vocab escape-hatch pattern one level down: *"here are the entities we have; reuse an
id if it matches, else answer `NEW: <label>`."* Reuse keeps ids consistent (precision); the `NEW:`
hatch preserves recall.

## Guards (or it backfires)

- **False-covered risk** (the open-vocab `unmapped_object_requirements` lesson): the model
  "helpfully" snaps a genuinely-new object onto an existing id, hiding novelty — the worst failure
  for a coverage project. So the `NEW:` hatch is mandatory, and the model must be able to say
  "this is new."
- **Scale**: 5,114 entities do not fit in a prompt. Retrieve the **K most similar existing
  entities** (bge-small embeddings) and offer only those as candidates → retrieval-augmented
  naming.
- **Pairs with guided decoding**: constrain the emitted id to `existing_candidates ∪ {NEW}` — then
  a duplicate spelling literally cannot be produced.

## Relations

- **Upstream vs downstream**: this fixes duplication at extraction; the **aliases layer** cleans up
  what still slips through. Both wanted — belt and suspenders.
- Whose pipeline: a suggestion for Gabriel's acquisition pipeline, and directly usable in Raul's
  own cellular RAG.
- Open: measure it — does entity-context naming reduce the 487→223 duplication, and at what recall
  cost (false-covered rate)? An ablation, ties to the guided-decoding evaluation.
