# Held-out corpus validation of the gap finder

**Status**: discussed (2026-07-22). User's proposal. Distinct from — and a cheap intrinsic
complement to — the external literature check in [gap-hypothesis-system.md](gap-hypothesis-system.md)
constraint 2. Evaluation methodology; slots into [grounding-and-evaluation.md](grounding-and-evaluation.md).
Blocked on the same prerequisites as the gap finder (populated graph + `kg/queries.py`).

## User's proposal

Take the first X papers → run extraction/retrieval → build a partial KG → run the gap finder
→ then look for those gaps against the *full* dataset of papers.

## Two things bundled here (separate them)

- **(A) Held-out corpus as self-validation** *(the valuable one)*: build KG on a subset,
  enumerate gaps, check which gaps get filled once the remaining papers are ingested. A
  train/test split for gaps.
- **(B) Gap-directed targeted retrieval**: only search the full dataset for the specific
  flagged gaps instead of extracting everything up front. See dedicated section below — it is
  actually the *mechanism* (A) uses to check fills cheaply, not a separate idea.

## Why (A) is strong

The unsolved problem in the gap-hypothesis system is constraint 2 (a gap = "not in the
ingested papers," not "not measured by anyone"), whose logged mitigation is an *external*
InspireHEP check. Held-out validation is the **intrinsic** version, for free:

- Gap found on subset X that is **filled by the held-out papers** → the finder pointed at an
  empty-but-physically-real cell. Positive signal: the held-out papers are an **oracle for
  "physically sensible,"** exactly the empty-AND-sensible-vs-boring-empty distinction the
  finder exists to make (see gap-hypothesis constraint 1 / "most empty cells are boring").
- Gap that **persists on the full corpus** → genuine within-corpus gap or a truly boring cell.

This is a standard **matrix-completion / link-prediction** evaluation framing (examiner-legible,
trusted). The check itself is deterministic graph lookup — **no LLM, cheap**.

## The caveat to state up front (do not hide it)

Held-out validation measures the **enumeration + ranking machinery**, NOT the scientific
novelty of the headline gaps. It is almost the opposite objective: it rewards finding cells
*someone did measure*; the dissertation payoff is cells *no one* has. Framing must be: "held-out
fill-rate shows the finder surfaces physically-sensible empty cells rather than random ones;
genuine-discovery claims still rest on the external literature check." Both, not either/or.

## Practical design notes

- **Random splits, not "first X"** — arbitrary paper ordering biases the result; multiple
  random splits also feed the n=3 non-determinism/variance concern (grounding-and-evaluation).
- **Learning curve**: held-out fill-rate vs X — near-free, examiner-friendly plot; shows how
  coverage saturates as the corpus grows.
- Natural metrics: precision-like (fraction of predicted gaps that are corpus artifacts, i.e.
  filled by held-out) and a ranking metric (do high-ranked "sensible" gaps get filled more
  often than low-ranked ones?).

## (B) Gap-directed targeted retrieval — expanded (2026-07-22)

Not a separate idea: (B) is *how* (A) checks a fill when you do NOT want to fully extract the
held-out papers. (A)'s check has two implementations — full-extract-then-graph-lookup (cheap
lookup, full extraction cost paid) or **targeted-search the held-out papers for that one gap**
(= B). They compose.

**Direction flip.** Baseline pipeline is **push** (`paper → assertions`: "what did this paper
measure?"). (B) is **pull** (`hypothesis → evidence`: "did anyone measure *this cell*?"). The
pull direction is exactly the gap-hypothesis constraint-2 literature-check agent, pointed at the
*internal* corpus instead of InspireHEP. Same machinery, internal target.

**Active-retrieval loop.** Reframes the gap finder: enumerate gaps on current graph → pull corpus
for those gaps → ingest confirmations → new nodes spawn new neighborhood gaps (the Stage-A
ingestion trigger in gap-hypothesis-system) → repeat. "Gap-directed extraction" vs "exhaustive
extraction" is a clean ablation.

**Asymmetric reliability — the load-bearing caveat.** The dissertation's headline claim is about
absence, so this matters:
- Finding a fill is **reliable** (positive confirmation) — and that is exactly what (A) needs.
- Finding nothing is **NOT reliable**: "no evidence surfaced" conflates *genuinely uncovered*
  with *retriever missed the passage*. Every absence claim then inherits per-query retrieval
  recall. Exhaustive extraction has the same recall problem but spreads it uniformly; (B)
  concentrates it into precisely the absence claims that are hardest to defend.
  → **Use (B) to confirm fills, never to assert a gap is genuine.**

**Low build cost.** Not a new engine: the existing per-predicate RAG query *inverted* into a
grounded yes/no ("did paper P measure final-state F at √s?"), reusing `generate.py` + the hybrid
index — a new query template, not a new subsystem.

**Scope call.** In this corpus (fixed, harvested, hundreds of papers, fully extracted anyway for
the KG) (B) buys nothing operationally — exhaustive extraction *is* the product; worth building
only as an efficiency ablation ("recover Y% of coverage for Z% of compute"). (B) becomes
*necessary* only for a corpus too large to exhaustively extract (all of InspireHEP/arXiv), where
pull is the only feasible mode — that is (B)'s **scaling-path** framing, its main reason to
appear in the write-up.

## Relations

- Complements gap-hypothesis-system.md constraint 2 (intrinsic vs external literature check).
- Belongs to the evaluation methodology in grounding-and-evaluation.md.
- (B) is a possible efficiency variant, deprioritized.
