# Gap-hypothesis system (reasoning layer over coverage gaps)

**Status**: discussed (2026-07-14) — direction agreed in principle; blocked on populated
graph + `kg/queries.py`. Distinct from [multi-agent-extension.md](multi-agent-extension.md):
the critics protect input quality, this generates the research output itself.

## User's proposal

An agent that watches new info entering the KG and reasons about what gaps *would make
sense* related to it — noting candidates down — plus a second system that reasons deeply on
those candidates against the whole KG and physics theory.

## Why it matters more than the critics

`kg/queries.py`'s deterministic enumeration is mandatory but blind: the combinatorial
(final state × experiment × energy) space is huge and most empty cells are boring
(kinematically forbidden, trivial, indirectly covered). The interesting gap is empty AND
physically sensible AND theoretically motivated — distinguishing those requires reasoning.
This system IS the dissertation's headline-results generator. **If forced to choose between
finishing the critic panel and building this: build this.**

## Two-stage design

- **Stage A — ingestion-time trigger (cheap, log-only)**: when paper N merges, graph
  traversal enumerates the *neighborhood* of its new nodes/edges (same final state at the
  other experiment / other energy; ±1-object adjacent signatures; same process, other final
  states). Empty neighbors → **gap-candidates log** with provenance (triggered-by paper,
  relation). Barely needs an LLM — it's deterministic traversal. Mirrors the leftovers-log
  pattern.
- **Stage B — offline deep-reasoning pass (batch)**: cluster candidates (same
  embed-group-adjudicate tooling, third use), then per candidate/cluster reason: physically
  sensible? BSM-motivated (physics fact sheet + theory context; **route to a bigger model
  than the 8B — the one task in the project that genuinely needs it**)? Corpus artifact
  (see constraint 2)? Output: ranked annotated gap hypotheses → human review → results
  chapter.

## Two hard constraints

1. **The LLM never *generates* gaps, only reasons about enumerated ones.** Every candidate
   originates from an explicit graph query result; the LLM ranks/filters/motivates. Free
   gap-finding hallucinates absences the graph doesn't show and misses ones it does.
2. **Corpus-incompleteness caveat (examiner will ask)**: a gap in our graph = "not measured
   in the ingested papers," NOT "not measured by anyone." With hundreds of papers vs
   thousands of real analyses, most raw gaps are corpus artifacts. Mitigations, use both:
   (a) scope all claims as "gaps within the ingested corpus"; (b) Stage B does a
   **literature check per surviving candidate** (InspireHEP search: does an un-ingested
   paper cover this?). That literature check — query graph, search external DB, consult
   theory, synthesize — is the first genuinely tool-using agency in the project; the
   "agent" label is earned here.

## Beyond final-state gaps (team meeting, 2026-07-24)

The gap axis so far is (final state × experiment × energy). But "un-looked-at" has **more axes
than final state** — expansions discussed, in rough order of ambition:

1. **Subject / topic gaps via literature pull.** Point agents at a *subject*, pull the relevant
   literature (InspireHEP/arXiv), ingest it into the graph, then enumerate gaps *within that
   subject*. This generalizes the constraint-2 external literature check (and the pull-retrieval
   in [[held-out-gap-validation]]) from "confirm one flagged cell" to "map a whole topic." Same
   asymmetric-reliability caveat: pull can confirm coverage, never assert a genuine gap.
2. **The theory side of the graph (the ambitious version).** Build a *second, theoretical* graph —
   BSM models, their predicted signatures, parameter spaces — and cross it with the experimental
   graph. A gap is then **theory-predicted-but-unmeasured**, a far more principled definition than
   "empty cell in the experimental grid." Agents work both graphs at once. Uses the physics fact
   sheet ([[multi-agent-extension]]) as the seed of the theory side.
3. **Scoped-theory fallback (the tractable version — likely the realistic one).** The full theory
   graph is probably too big for the timeframe. Instead pick **one theory area** (e.g. a specific
   BSM family — leptoquarks, a SUSY sub-model), build just that slice of the theory side, and find
   gaps *there* against experimental coverage. Dissertation-sized, still a genuine theory×experiment
   coverage contribution, and de-risks the ambitious version.

**Framing note**: (2)/(3) turn the project from "empty cells in an experimental grid" into a
**theory-vs-measurement coverage map** — which is also the cleanest answer to the Contur comparison
(Contur = measurement-side; a theory-predicted-vs-measured map is the complementary direction).
Sequencing unchanged: all of this is act three, after the experimental graph exists.

## Relations

- Builds on `kg/queries.py` (deterministic enumeration is the substrate) and the physics
  fact sheet from the critic panel.
- Contur comparison belongs in this doc's related work: Contur answers coverage from the
  measurement side; the write-up must position gap hypotheses against it.
- The theory-side expansion (above) pairs naturally with [[researcher-feedback-loop]] (authors
  correcting the experimental side while agents extend the theory side).
- Sequencing: act three — after baseline pipeline and populated graph.
