# Final-state representation

**Status**: discussed, open — the most important pre-scale fix (blocks running extraction at scale).
**Discussed**: 2026-07-06 (flagged in schema review), 2026-07-09.

## The gap

`result_has_final_state` currently reuses the flat `DETECTOR_OBJECT_VOCAB` pick-list, so a
paper searching "2 electrons + 1 jet + MET" produces separate independent edges to
`electron`, `small_r_jet`, `missing_transverse_momentum` — losing the **multiplicities**
("2 electrons") and the **combination** (that these objects together form one signature).
A real final state is a multiplicity composition (MUSiC-style event class), and it's the
core coverage-mapping field.

## The two candidate representations

1. **Composite node**: one node per full signature, e.g. `final_state:2e+1j+met`, that
   Results point at. Coverage counting = direct query on that node.
2. **Object fan-out (current)**: keep individual object edges; *reconstruct* signatures at
   query time from each Result's set of object edges (+ count qualifiers if added).

User's position (2026-07-09): unsure which; comfortable deferring because "it shouldn't be
hard to take the final nodes and construct the final state taking each of them" — i.e.
reconstruction from object edges is an acceptable fallback, so fan-out isn't a dead end.

## What's already settled regardless (D-015)

Identical compositions across papers share one node/pattern — canonical ID = deterministic
serialization of the composition (e.g. sorted `{object: count}`), so identical always
collapses automatically, whichever representation wins.

## Landmine to avoid when resolving

Flavor-inclusive vs flavor-specific: "2 leptons" ≠ "2 electrons" (lepton ⊃ electron, muon).
Extraction/canonicalization that conflates these wrongly merges (or wrongly splits) final
states. The eventual representation needs an explicit stance — possibly a small hierarchy
(lepton → electron/muon) rather than a flat object list.

## Likely shape of the fix (not decided)

Extend the extraction target for this predicate to structured `{object, count}` pairs
(schema gap noted since 2026-07-06), then derive the canonical composite ID from that
structure. Decide before populating the paper list and running at scale.
