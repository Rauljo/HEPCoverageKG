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

## Authority ordering: blocker vs advisory (from AgentRivet, 2026-07-15)

Principle (AgentRivet's Code-Reviewer vs Physics-Reviewer split): **if something outside the
LLM pipeline can check a claim, that claim wins automatically and may overwrite; if nothing
external can check it, it is only advisory — flag/quarantine/lower-confidence, never
overwrite.** "External" does the work — a compiler earns authority by not caring what any LLM
thinks. AgentRivet's Physics-Reviewer kept demanding an impossible observable; because it was
*advisory*, the Coder could correctly refuse — as a blocker the loop would never terminate.
Our mapping: HEPData / SimpleAnalysis / pyhf = **blocker** (ATLAS published it, external →
overwrites the extractor); every LLM critic above = **advisory** (grades its own homework off
the same paper → flag only). This is the *principled reason* our critics are flag-only (D-018)
and makes grounding load-bearing — see [grounding-and-evaluation.md](grounding-and-evaluation.md).
Inverting it fails **silently**: a confident critic overwriting a grounded count errors
nothing, hangs nothing, and the map is simply wrong.

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
- **Critics must re-read the SOURCE, not just the extracted fields** (AgentRivet's
  Physics-Reviewer blind spot: it judged the Coder against the Analyst's *summary*, so it could
  never catch the Analyst dropping a definition — and silent omission was the dominant failure).
  A fidelity critic reading only assertions replicates that blind spot exactly.
- **Test any critic on a known-clean case** (AgentRivet's Claude-Opus never once returned
  "approved" even when instructed to — hedging, not grading). If our critic never says "clean,"
  everything lands in quarantine and a quarantine of everything tells us nothing.

## Compute feasibility on DIAS (discussed 2026-07-15)

**Usable ceiling: 160GB VRAM per model** (2× A100 80GB — a single model tensor-parallels
across at most 2 of the 3 cards; see overview infra note). The 3rd card runs a separate
TP=1 server.

- **Model ladder for the size-comparison experiment**: `8B → 32B → 70B`. All fit; 70B in
  BF16 (~140GB) is *tight* on 160GB (~20GB KV cache) → prefer **4-bit** (~40GB, ~120GB KV,
  comfortable). 120B-class only quantized (~62GB). 405B is out (and overkill for grounding-
  hard extraction anyway). Comparison is sequential (one model at a time) so you only need
  the largest single model resident.
- **Bigger embeddings are NOT a compute concern**: even 7B-class embedders (~14GB) are
  trivial and run once-per-paper cached. The whole compute story is the LLM.
- **Agents ≠ models**: N concurrent agents on ONE shared model = one set of weights + KV
  cache. Concurrency *helps* GPU utilization (vLLM batches independent requests — parallel
  agents are exactly what it wants). Resource consumed is KV cache (peak sequences ×
  context), not weights. Default design: **one model, many roles** (a "physics critic" is a
  prompt, not a checkpoint).
- **The real ceiling is concurrent *distinct large models*** (~160GB): 70B+32B doesn't fit;
  even 70B-BF16 + 8B has no KV room. Mitigations: share model within a capability tier;
  quantize; big-model-on-2-cards + small-on-1; hosted API (D-012) for a frontier agent (zero
  local VRAM). Route by tier, don't give each agent its own checkpoint.
- **At high agent concurrency the bottleneck leaves the GPU**: external tool rate limits
  (InspireHEP won't love hundreds of concurrent queries), orchestrator CPU/RAM, and the
  shared-node + 24h-wall reality. An *always-on* swarm fights DIAS's shared-batch nature —
  batch/offline runs fit, persistent services don't.
- **Orchestration must be async / high-concurrency**: agent chains are sequential
  dependencies (critic waits on extractor), so run many paper-trajectories at once to keep
  vLLM batched — else a 70B sits at single-digit % utilization waiting between calls.

## Open

- Which 1–2 components to build; evaluation gold-set design; when the baseline counts as done.
- ~~Verify whether the 2-GPU-per-model limit is TP head-divisibility or a SLURM cap~~
  **RESOLVED 2026-07-15** (3-GPU probe job): no SLURM cap — all 3 allocatable in one job;
  the 2-per-model limit is TP=3 head-divisibility + PCIe/NUMA topology (no NVLink; GPU0+1
  same-NUMA fast pair, GPU2 cross-NUMA slow). Two-server trick (TP=2 big on GPU0+1, TP=1
  small on GPU2) is confirmed available. See overview infra note.
