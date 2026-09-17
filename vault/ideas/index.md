# Ideas — index

**Status counts: 5 seed · 18 discussed · 3 adopted · 0 rejected**

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
| **Evaluation harness** | discussed | [eval-harness-design.md](eval-harness-design.md) — runs *any* system behind one `System` protocol, so baselines and ablations share one runner; test-set lock and variance-first built in (S-22, S-10, S-36, S-52) |
| HEPData harvesting | seed | Future numeric-results source (D-008); not current scope |
| arXiv table extraction | seed | Only if structured numerics are needed and HEPData isn't built (D-008) |

## Graph & canonicalization

| Idea | Status | Doc / note |
|---|---|---|
| **Bundle importer (milestone 1)** | **adopted (D-021..D-030)** | [bundle-importer-design.md](bundle-importer-design.md) — the durable build reference; M1 done |
| Open vocabulary + batch reconciliation | discussed | [open-vocab-reconciliation.md](open-vocab-reconciliation.md) — the mechanism (D-016) and the **`aliases/` layer** over bundle entity_ids (D-025); Tier 1 built; vocab_policy still needs sign-off |
| **Discovered grouping layer** | discussed | [discovered-grouping-layer.md](discovered-grouping-layer.md) — fills the never-built middle rung of S-68: cluster what the closed vocabulary *fails* to classify and propose new tags. Propose-never-adopt (parity against v1 must survive). Held-out test **using the curated vocabulary as ground truth** is the contribution; expect under-splitting (JES/JER). Sequenced after the 15 questions are scored |
| **Final-state representation** | **discussed — rewritten 2026-07-29 (D-038)** | [final-state-representation.md](final-state-representation.md) — the old entry was **wrong**: no fan-out, and **no `count` qualifier exists**. The signature is one node of English prose; 126 distinct labels / 138. Fix = LLM parses prose → structure, code serialises the id; grouping is a separate later layer. **Blocks M3, which now blocks the evaluation** |
| Neo4j migration | seed | Neo4j Community (local) as a projection from SQLite (D-022) — switch on ergonomics/visualization, not scale |

## Agents & reasoning (the payoff)

| Idea | Status | Doc / note |
|---|---|---|
| Gap-hypothesis system | discussed | [gap-hypothesis-system.md](gap-hypothesis-system.md) — enumerate → reason → verify; the headline generator. Expanded 2026-07-24: beyond final-state (literature-pull, theory-side graph, scoped fallback); 2026-07-27: author-declared gaps (future-work mining) + the Swanson/Popper framing of the whole arm |
| Held-out gap validation | discussed | [held-out-gap-validation.md](held-out-gap-validation.md) — train/test split for gaps; matrix-completion eval; intrinsic complement to the external literature check. 2026-07-27: the gap-*matching* problem (reuses aliases Tiers 2–3) + temporal holdout from declared-future-work labels |
| Multi-agent extension | discussed | [multi-agent-extension.md](multi-agent-extension.md) — agentic verification vs single-pass baseline; the critic panel is mostly **Sunny's** thread; blocked on baseline. 2026-07-27: "mixture of LLMs" is an *ensemble*, not an agency, argument |
| Researcher feedback loop + self-correction | discussed | [researcher-feedback-loop.md](researcher-feedback-loop.md) — expose per-paper summaries → authors correct → cluster → human-applied fixes (never LLM self-editing). Team meeting 2026-07-24 |
| **Self-correcting planning arms** | discussed | [self-correcting-planning-arms.md](self-correcting-planning-arms.md) — four REMOVABLE arms after PoG (NeurIPS 2024), specced 2026-08-31: `--subgoals` (decompose, ≤3), `--subgoal-status` (the memory block, contains it — PoG's highest-value mechanism and the fix for gf-01's forgotten conditions), `--structured-verdict`, `--backtrack` (critic-filtered unfollowed candidates, competing with a fresh search). Departs from PoG on breadth: they minimise counts for Hits@1, we filter by relevance because recall is our weaker side |
| Anchored candidates | discussed | [anchored-candidates.md](anchored-candidates.md) -- separate exploring from nominating; the judge sees what the planner nominated, not everything it touched (D-197) |
| Routing by intent | discussed | [routing-by-intent.md](routing-by-intent.md) -- a dispatcher over measured arms: shape x intent x effort, rerank-explain for sets, built to not need sweeping at 3,000 papers |
| App: follow-up questions | seed | [app-follow-up-questions.md](app-follow-up-questions.md) -- a thread that carries the previous turn's pool forward; the human-in-the-loop version of tighten-never-loosen |
| **Typed-interface arms** | discussed | [typed-interface-arms.md](typed-interface-arms.md) — three REMOVABLE arms on the tool layer, specced 2026-08-31 from the trace anatomy: `--symmetric-hops` (the backward hop requires a predicate, the forward hop does not), `--strict-refs` (92% of typed errors are references to things that do not exist yet), `--path-tool` (one call for the conjunctive multi-hop that free-SQL writes as a JOIN). Each default OFF, one guarded branch, deletable. None expected to work until measured |
| Grade the reached at answer time | discussed | [grade-the-reached-at-answer-time.md](grade-the-reached-at-answer-time.md) — D-135's 14% bucket (gf-07): papers that entered via subjects_of/describe and never met the ranker; grade them once at the ranked-answer hook. Not built: 7-20 judge calls at answer time on a path near the record budget, ~+0.03 expected |
| Answer routing by question shape | discussed | [answer-routing-by-shape.md](answer-routing-by-shape.md) — the name-ids instruction is set-specific and costs per-paper and count answers (D-153); choose the answer instruction by shape. Not built |
| Observability / artefact tracing | seed | Keep intermediate artefacts so an error's *cause* is recoverable (AgentRivet/Langfuse) — provenance, not scoring |
- [plan-ahead-batching](plan-ahead-batching.md) -- **proposed** 2026-09-17. Let the model chain dependent calls in one round by naming the set a same-turn search will save; the executor already allows it (927 ok / 82 wrong-name), the prompt forbids it. Cheap version is one prompt line.
