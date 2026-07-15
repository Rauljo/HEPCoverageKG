# Multi-agent extension (as an evaluated research question)

**Status**: discussed — direction agreed in principle (2026-07-14), no design finalized, blocked on baseline completion.
**Absorbs** the former seed "orchestrated-agent extraction provenance."

## Motivation (user, 2026-07-14)

The user wants a multi-agent system in the dissertation — stated motive: "to add complexity
to it," exploring finance-agent repos (FINSABER, TradingAgents, ai-hedge-fund, MiroFish) to
learn what multi-agent systems are capable of.

## Reframing agreed

"Add complexity" inverted into a **measured research question**: *does agentic verification
improve evidence-grounded extraction from HEP papers, and at what cost?* The existing
single-pass pipeline is not the boring part — it's the **baseline / control arm**. The
agentic layer is the treatment arm; both get evaluated on extraction quality (precision/
recall vs a gold set) and cost (LLM calls). A negative result is still a result. This keeps
D-002 (no orchestration agent in the core pipeline) intact — agents are a comparative
extension, not a replacement.

## Candidate agent-shaped components (pick 1–2, not all)

1. **Extract → verify loop**: extractor + independent critic checking each assertion
   against retrieved evidence. The consistency-check pass (D-018) is the one-step embryo.
   Template + citation: **AgentRivet** (HEP, published, extraction with code-review and
   physics-review agents).
2. **Adaptive re-querying**: a supervisor reacting to failed validation by choosing which
   section to re-query with what reformulated question — genuine runtime-contingent agency,
   unlike the fixed six-question loop.
3. **Debate adjudication for hard reconciliation pairs**: two models argue same/not-same,
   third adjudicates; contained, comparable against the single-judge design (D-016).

## Critic-panel architecture (user proposed critics 2 & 3, 2026-07-14)

Three narrow, **flag-only** critics feeding one `needs_review` queue; each separately
ablatable (baseline → +1 → +2 → +3, measuring per-critic flag precision on the gold set):

1. **Fidelity critic** — assertion vs retrieved evidence (the core loop; build FIRST).
2. **Physics-coherence critic** (user's idea) — grounded in a **curated static physics
   fact sheet** in the prompt (typical backgrounds per final-state family, standard run
   configs, observable types per result type) — versioned in the repo, auditable.
   Theory-RAG only as a measured upgrade if the fact sheet plateaus. **Role restriction:
   checks extraction fidelity/internal coherence, never physics orthodoxy** — a critic
   scoring against its priors would censor surprising-but-correct extractions, i.e. the
   exact signal a coverage-mapping project exists to find. AgentRivet's physics-review
   step is the precedent/citation.
3. **Corpus-consistency critic** (user's idea, reframed) — consults the existing KG, but as
   a **novelty detector / review-router, never a validity judge**. Naive version ("graph
   says most searches use 139 fb⁻¹, this claims 3 → probably wrong") is NELL's
   beliefs-feeding-beliefs drift and punishes novelty (rare final states are the product,
   not noise). Reframed: novelty = raise review priority — the assertion is either an
   extraction error or a rare data point, and both deserve human eyes. Graph signals can
   only RAISE attention, never lower status. Cold-start useless → build LAST, once a few
   hundred papers are ingested. Bounded, defensible "KG improves its own construction" angle.

Ship with two critics if time bites; fidelity critic is the research question's heart.

## Guardrails

- **Sequencing**: baseline first (kg layer, paper list, final-state fix, measured simple
  pipeline) — agents added before a baseline exists can be exhibited but not evaluated.
- **Vocabulary discipline** (from Anthropic's "Building Effective Agents"): most
  "multi-agent" systems are *workflows* (fixed pipelines with LLM steps); true agency =
  model-directed control flow. Use the terms precisely in the write-up.
- **Finance repos**: skim ONE for coordination patterns (role decomposition, debate,
  orchestrator-worker) — they don't evaluate agent-vs-single-model and much persona
  machinery is theatrical. Dissertation-relevant sources: AgentRivet, the fellow student's
  agents-in-science review.
- **Provenance** (the absorbed seed): when agent-derived extraction exists,
  `EXTRACTION_METHODS` must distinguish it from plain single-pass RAG, so the comparison
  arms stay separable in the graph itself.

## Open

- Which 1–2 components to build; evaluation gold-set design; when the baseline counts as done.
