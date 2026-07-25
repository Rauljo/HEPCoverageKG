# Researcher feedback loop + self-correction

**Status**: discussed — raised in the 2026-07-24 team meeting (Raul's proposal). Future
expansion; not scheduled. Related to [[open-vocab-reconciliation]] (human-confirm step),
[[multi-agent-extension]] (corpus-consistency critic), and the bundle contract's `expert_decision`.

## The idea

A paper's assertions, taken together, **are a structured summary of that paper**. So:

1. **Expose** the graph as per-paper summaries (a list / view / page where a researcher can find
   *their own* paper and see what the system extracted from it).
2. **Invite correction** — let researchers (ideally the actual authors — the best possible domain
   experts) contribute and fix errors on their papers.
3. **Learn from the corrections** — an agent examines the stream of corrections and tries to
   **self-correct / evolve the system** so the *same class* of error stops happening.

This is a community-scale, author-sourced version of the human-in-the-loop already in the design
(`expert_decision`, the reconciliation human-confirm), and a bounded, defensible
"the-KG-improves-its-own-construction" angle.

## Why it's attractive

- **The correctors are the authors** — the highest-quality labels you could get for "did this
  paper measure X?", essentially free.
- Corrections become three reusable assets at once: (a) an **evaluation gold set**, (b)
  **alias / vocab data** for the canonicalization layer, (c) worked **examples** for few-shot
  prompts or guided-decoding constraints.
- The "summary" framing is a real product: per-paper structured summaries are useful to
  researchers independent of the coverage payoff — which is what makes them *want* to correct.

## The load-bearing caveat (D-016's lesson applies directly)

"An agent self-corrects / evolves the system" must **NOT** mean an LLM rewriting its own prompts —
that is exactly the **unauditable behaviour-drift** trap D-016 rejected (and NELL's semantic-drift
lesson). Discipline:

- Corrections are stored as **data** (a correction record, an alias row, a vocab addition, a
  gold-set example) — never as silent prompt edits.
- "Evolve to avoid future errors" = **cluster the corrections** (same embed→group→adjudicate
  tooling as [[open-vocab-reconciliation]] / [[leftovers-pass]]) → surface *systematic* error
  types ("47 corrections all fix a `detector_object` vs `object_definition` mislabel") → a
  **human decides** the fix (a prompt change, a schema rule, a guided-decoding constraint).
  The agent **surfaces the pattern; a human applies the fix.**
- This is the direct payoff route for the errors already found (the b-jet id split, the
  `predicate_typing` kind-flips): author corrections would be the training/eval signal that a
  guided-decoding fix is then measured against.

## Open questions / risks

- **Author bias**: authors may "correct" toward flattering readings. Low risk for factual
  extraction (what was measured), higher for interpretation — track provenance of every correction,
  and treat corrections as *review signal*, not automatic truth (mirror the corpus-consistency
  critic's "raise, never silently overwrite" rule).
- **Incentive / adoption**: researchers only correct if the summaries are useful to them first.
- **Interface**: needs the graph queryable + a per-paper view — depends on the graph DB + query
  layer being built first.
- **Boundary with Sunny**: his verification agents are the *internal* reviewer; this is the
  *external/author* reviewer. Both feed the same `status` seam — coordinate the format.
