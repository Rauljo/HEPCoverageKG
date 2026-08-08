# Discovered grouping: letting the graph propose its own vocabulary

**Status**: discussed (2026-08-08). User's proposal, refining an earlier framing of "new tags via
self-correction". Fills the rung named in S-68 and never built. Depends on the facet layer
(D-052..D-055) for both its input and its answer key. **Sequenced after the supervisor's 15
questions are scored** — see [Where this sits](#where-this-sits).

## The proposal

Rather than hand-writing patterns to extend the closed facet vocabulary, let the system **find its
own groups** in what the vocabulary fails to classify, and propose them as new tags. The
"self-correcting mechanism" the user has in mind: the graph notices what it cannot classify and
suggests classifications, rather than waiting for someone to write a regex.

## Why this is the right shape

It fills the gap in the abstraction ladder (S-68), which has always had a hole in the middle:

```
raw label   ->   alias cluster   ->   families   ->   facet keys
   227             194              NEVER BUILT          18       (detector objects)
```

Facets arrived **top-down**: a curated menu, with patterns assigning membership. Discovered groups
would work **bottom-up**: cluster what is there, then name the clusters. A discovered group is
exactly a candidate for a new facet tag, so the two meet in the middle.

It also replaces a bad plan. The alternative considered first was a local pattern file we maintain
by hand; measured yield was **3 patterns → 5 labels**, against 282 untagged systematic uncertainties
where labels and occurrences are nearly 1:1. Closing that tail by hand is ~150 regexes. Not a good
use of the time, and not a contribution.

## The material is there (measured 2026-08-08)

The 282 untagged `systematic_uncertainty` labels are not noise. Their content words:

```
73  background      37  scale        22  lepton      13  branching
39  normalization   25  jets         22  shape       13  mass
```

**Shape vs normalisation** is a textbook systematic split and upstream's 17-tag vocabulary has
neither; `background normalisation uncertainty` is a standard category. So the untagged tail
contains real missing groups, and they are visible from the data alone.

## The constraint that keeps it safe: propose, never adopt

Discovered tags go into a **local** vocabulary. Parity always runs against pure `facets-v1`.
Anything genuinely good is proposed **to the supervisor** for the shared menu.

Two reasons, the first decisive:

- **The vocabulary's value is that it is shared.** `facets-v1` is a common language with the
  supervisor — it is why our 60 cards can be compared against his and match exactly (D-052). A
  vocabulary that grows itself is no longer his, and we lose the only independent check we have.
- **His own lab already measured the alternative.** From `vocabulary.py`: *"a closed object
  vocabulary gives high expectation precision (84%/93%) with zero invented types"*. Auto-adopting
  machine-named groups reintroduces invented types — the exact failure the closed vocabulary was
  chosen to eliminate.

The loop, which is the aliases layer's shape (D-025) applied one rung up:

```
facets gaps  ->  cluster the untagged  ->  name each cluster  ->  propose
                                                                    |
                    re-derive, measure coverage delta  <-  confirm  <-
```

Better safety profile than the aliases layer, and worth saying explicitly: a bad group is visible in
one command and undone by re-deriving, whereas a bad alias merge fuses two entities and corrupts
every downstream count silently (S-70).

## The experiment — this is the part that makes it a contribution

**Test whether automatic grouping works, using the curated vocabulary as ground truth.**

82% of entities already carry tags assigned by a hand-built closed vocabulary. So:

1. Take the **tagged** entities and hide their tags.
2. Run the discovery layer on them as if they were unclassified.
3. Ask whether it recovers the same groups.

A held-out evaluation with a real answer key, for a question usually argued rather than measured.

**The failure mode it should expose.** Embedding clustering groups things that *sound* alike:

```
"jet energy scale uncertainty"        -> upstream: JES
"jet energy resolution uncertainty"   -> upstream: JER
```

Near-identical as text; genuinely different systematics that upstream correctly separates. Expect
discovery to merge them — a grouping **coarser than the right one**. Whether that happens and how
often is precisely what the held-out test measures.

The reportable result is then a sentence with a number in it:
*"automatic grouping recovers N% of a curated vocabulary and systematically under-splits along
axis X"* — which generalises past HEP, and is a far better claim than "we added some tags".

## Design notes for when it is built

- **Three distinct causes of a miss**, measured (D-052) and needing different repairs. Discovery only
  addresses the third:
  1. *Notation* — `$E_T^{miss}$` folds to `e t miss`, so `\bmet\b` never fires. A fold repair, not a
     new tag. Lives in the vendored file we deliberately cannot edit — an unresolved tension.
  2. *Narrow pattern* — `$b$-jet identification calibration` should be `FlavourTagging`; upstream's
     pattern requires the word "tag".
  3. *Genuinely absent concept* — `OS-SS subtraction`, `h_damp variation`. **Discovery's target.**
- **Storage already supports it.** `entity_facet.vocabulary` lets versions coexist and be diffed;
  `facets derive --vocabulary X` writes a second set (fixed 2026-08-08 — it had been a silent no-op,
  parameterised DELETE with a hardcoded INSERT).
- **Union, never override.** A local vocabulary may only add tags, so `v1+local` is always a superset
  of `v1` and the delta is unambiguous.
- **The runtime stays deterministic.** The model proposes patterns at design time; no model call ever
  enters the query path. That property is what makes facets cheap and reproducible, and is the whole
  distinction between this layer and the aliases layer.

## Where this sits

**After the supervisor's 15 questions are scored.** The held-out test needs the tagged data as ground
truth, which we already have and which is not going anywhere. The *motivation* needs a measured gap:
if the agent scores badly on Tier 1 because it never reaches for facets at all, then discovering new
tags fixes nothing, and we would have built a research contribution answering a question the system
was not asking.

Related: [open-vocab-reconciliation.md](open-vocab-reconciliation.md) (the rung below),
[final-state-representation.md](final-state-representation.md) (grouping as a later layer, same
argument), [researcher-feedback-loop.md](researcher-feedback-loop.md) (the same propose/confirm
discipline, and the same refusal to let an LLM self-edit).
