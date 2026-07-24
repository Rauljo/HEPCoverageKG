# Ideas — index

**Status counts: 4 seed · 8 discussed · 3 adopted · 0 rejected**

*Statuses: seed (one-liner, never discussed) · discussed (has a doc) · adopted (graduated to a D-nnn decision; doc is its pre-history) · rejected (kept forever, with rationale).*
*Convention: whenever any idea doc is touched, refresh the counts line above.*

| Idea | Status | Doc / note |
|---|---|---|
| **Bundle importer (milestone 1)** | **adopted (D-021..D-026)** | [bundle-importer-design.md](bundle-importer-design.md) — full design + build order; the durable build reference |
| Open vocabulary + batch reconciliation | discussed | [open-vocab-reconciliation.md](open-vocab-reconciliation.md) — mechanism adopted as D-016; **now also the `aliases/` layer over Gabriel's bundle entity_ids (D-025)** — reusable project-wide (coverage axes, queries, gap finder); per-predicate `vocab_policy` still needs sign-off |
| Consistency-check pass | adopted (D-018) | [consistency-check-pass.md](consistency-check-pass.md) |
| Leftovers / predicate-suggestion pass | adopted (D-019) | [leftovers-pass.md](leftovers-pass.md) |
| Final-state representation | discussed (open) | [final-state-representation.md](final-state-representation.md) — most important pre-scale fix |
| `result_type` vocabulary | discussed (open) | [result-type-vocabulary.md](result-type-vocabulary.md) — needs supervisor |
| HEPData harvesting | seed | Future expansion, not current scope (D-008 clarification 2026-07-10) — would be the structured numeric-results source if ever needed |
| arXiv table extraction | seed | Only if structured numerics are needed and HEPData isn't built (D-008) |
| Multi-agent extension | discussed | [multi-agent-extension.md](multi-agent-extension.md) — reframed as an evaluated research question (agentic verification vs single-pass baseline); absorbs the former extraction-provenance seed; blocked on baseline |
| Gap-hypothesis system | discussed | [gap-hypothesis-system.md](gap-hypothesis-system.md) — reasoning layer over enumerated coverage gaps; the headline-results generator; takes priority over critics 2–3 if time forces a choice |
| Pydantic for validation | discussed | [pydantic-validation.md](pydantic-validation.md) — NO to retrofitting the supervisor-contract dataclasses; YES to vLLM guided decoding at the LLM boundary as a measured ablation. **Importer (D-023): jsonschema is the gate, Pydantic deferred — generate from schema later if the query layer wants typed access** |
| Grounding & evaluation | discussed | [grounding-and-evaluation.md](grounding-and-evaluation.md) — ground-truth sources (Rivet/SimpleAnalysis/pyhf), grounding-as-blocker, eval methodology; from the AgentRivet read; partially revisits D-008 framing |
| Held-out gap validation | discussed | [held-out-gap-validation.md](held-out-gap-validation.md) — build KG on a corpus subset, check which enumerated gaps the held-out papers fill; intrinsic (cheap, no-LLM) complement to the gap-hypothesis external literature check; matrix-completion eval framing |
| Observability / artefact tracing | seed | Keep intermediate agent artefacts so an error's *cause* is recoverable, not just its symptom (AgentRivet/Langfuse); the point is provenance, not scoring |
| Neo4j migration | seed | SQLite + NetworkX now, **Neo4j Community (local) later as a projection from SQLite** (D-022) — switch on ergonomics/visualization/service, not scale |
