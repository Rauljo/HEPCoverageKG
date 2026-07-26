# Ideas — index

**Status counts: 4 seed · 10 discussed · 3 adopted · 0 rejected**

*Statuses: seed (one-liner, never discussed) · discussed (has a doc) · adopted (graduated to a D-nnn decision; doc is its pre-history) · rejected (kept forever, with rationale).*
*Organized by theme (each idea placed by its primary concern; several span themes — cross-links in the docs). Actionable to-dos live in [`../backlog.md`](../backlog.md), not here.*
*Convention: whenever any idea doc is touched, refresh the counts line above.*

## Extraction & retrieval

| Idea | Status | Doc / note |
|---|---|---|
| Consistency-check pass | adopted (D-018) | [consistency-check-pass.md](consistency-check-pass.md) — post-extraction cross-field flag-only pass |
| Leftovers / predicate-suggestion pass | adopted (D-019) | [leftovers-pass.md](leftovers-pass.md) — log-only capture of un-predicated claims; cluster to find schema gaps |
| Entity-context naming | discussed | [entity-context-naming.md](entity-context-naming.md) — reuse existing entities at extraction (`NEW:` hatch); upstream fix for id duplication |
| Pydantic / guided decoding | discussed | [pydantic-validation.md](pydantic-validation.md) — NO to retrofitting the contract dataclasses; YES to vLLM guided decoding as a measured ablation (fixes signatures + typing + kind-flips) |
| `result_type` vocabulary | discussed (open) | [result-type-vocabulary.md](result-type-vocabulary.md) — needs supervisor; gates what counts toward a gap |
| Grounding & evaluation | discussed | [grounding-and-evaluation.md](grounding-and-evaluation.md) — Rivet/SimpleAnalysis/pyhf as external truth; n=3 runs; upper-bound eval |
| HEPData harvesting | seed | Future numeric-results source (D-008); not current scope |
| arXiv table extraction | seed | Only if structured numerics are needed and HEPData isn't built (D-008) |

## Graph & canonicalization

| Idea | Status | Doc / note |
|---|---|---|
| **Bundle importer (milestone 1)** | **adopted (D-021..D-030)** | [bundle-importer-design.md](bundle-importer-design.md) — the durable build reference; M1 done |
| Open vocabulary + batch reconciliation | discussed | [open-vocab-reconciliation.md](open-vocab-reconciliation.md) — the mechanism (D-016) and the **`aliases/` layer** over bundle entity_ids (D-025); Tier 1 built; vocab_policy still needs sign-off |
| Final-state representation | discussed (open) | [final-state-representation.md](final-state-representation.md) — multiplicity composition; compile from `count`/`subchannel` qualifiers (signature field is empty). Blocks M3 |
| Neo4j migration | seed | Neo4j Community (local) as a projection from SQLite (D-022) — switch on ergonomics/visualization, not scale |

## Agents & reasoning (the payoff)

| Idea | Status | Doc / note |
|---|---|---|
| Gap-hypothesis system | discussed | [gap-hypothesis-system.md](gap-hypothesis-system.md) — enumerate → reason → verify; the headline generator. Expanded 2026-07-24: beyond final-state (literature-pull, theory-side graph, scoped fallback) |
| Held-out gap validation | discussed | [held-out-gap-validation.md](held-out-gap-validation.md) — train/test split for gaps; matrix-completion eval; intrinsic complement to the external literature check |
| Multi-agent extension | discussed | [multi-agent-extension.md](multi-agent-extension.md) — agentic verification vs single-pass baseline; the critic panel is mostly **Sunny's** thread; blocked on baseline |
| Researcher feedback loop + self-correction | discussed | [researcher-feedback-loop.md](researcher-feedback-loop.md) — expose per-paper summaries → authors correct → cluster → human-applied fixes (never LLM self-editing). Team meeting 2026-07-24 |
| Observability / artefact tracing | seed | Keep intermediate artefacts so an error's *cause* is recoverable (AgentRivet/Langfuse) — provenance, not scoring |
