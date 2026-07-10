# Ideas — index

**Status counts: 4 seed · 3 discussed · 2 adopted · 0 rejected**

*Statuses: seed (one-liner, never discussed) · discussed (has a doc) · adopted (graduated to a D-nnn decision; doc is its pre-history) · rejected (kept forever, with rationale).*
*Convention: whenever any idea doc is touched, refresh the counts line above.*

| Idea | Status | Doc / note |
|---|---|---|
| Open vocabulary + batch reconciliation | discussed | [open-vocab-reconciliation.md](open-vocab-reconciliation.md) — mechanism adopted as D-016; per-predicate `vocab_policy` still needs explicit sign-off |
| Consistency-check pass | adopted (D-018) | [consistency-check-pass.md](consistency-check-pass.md) |
| Leftovers / predicate-suggestion pass | adopted (D-019) | [leftovers-pass.md](leftovers-pass.md) |
| Final-state representation | discussed (open) | [final-state-representation.md](final-state-representation.md) — most important pre-scale fix |
| `result_type` vocabulary | discussed (open) | [result-type-vocabulary.md](result-type-vocabulary.md) — needs supervisor |
| HEPData harvesting | seed | Future expansion, not current scope (D-008 clarification 2026-07-10) — would be the structured numeric-results source if ever needed |
| arXiv table extraction | seed | Only if structured numerics are needed and HEPData isn't built (D-008) |
| Orchestrated-agent extraction provenance | seed | When multi-agent retrieval expansions ever exist, `EXTRACTION_METHODS` should distinguish their provenance from plain single-pass RAG |
| Neo4j migration | seed | Kickoff plan: SQLite + NetworkX now, Neo4j later |
