# Leftovers / predicate-suggestion pass

**Status**: adopted → D-019. Not yet implemented.
**Discussed**: 2026-07-10 (arose from "what about discovering new predicates?").

## Design

Per **section** of each paper (skipping structural boilerplate: detector description,
acknowledgements), one LLM call: the section text + the facts already extracted
(serialized from `state.catalog` — the model has no memory; we construct its context) +
*"what other important claims does this section make about the result that none of these
capture?"* Answers append to a suggestions **log file**. No graph writes, no schema edits.
Optionally one aggregation call per paper to dedup its own suggestions. ~10 calls/paper —
free on self-hosted inference.

Why per-section, not whole-paper: 8B models degrade on 40k+-token contexts (the paper fits
in the 131k window but doesn't get *reasoned over* at that length), and retrieval can't
substitute — you cannot retrieve toward unknown unknowns; the leftovers pass exists
precisely for what no query anticipates.

## Reading the log at scale: cluster first (stolen from AutoSchemaKG's conceptualization)

At ~300 papers the log is ~3,000 lines and the signal — *recurrence* — is invisible to
linear reading because repetition never repeats word-for-word ("defines signal regions
with kinematic cuts" / "selection criteria for SR-A/SR-B" / "event selection per region").
So: embed each suggestion (bge-small), group similar ones, LLM labels each group, review
*groups* sorted by size — "47 suggestions across 41 papers: signal/control region
definitions" answers "what is the schema most missing?" in seconds. This is the same
embed→group→adjudicate tooling as the reconciliation script (D-016): one mechanism,
two levels of the schema (values and predicates), human gate at both.

## What recurrence triggers (humans only)

A big cluster → implement the matching predicate from `DEFERRED_PREDICATES`, or bring a
genuinely new predicate proposal to the supervisor. Automated predicate application was
explicitly rejected (D-019): OpenIE's lesson is that noticing is easy and canonicalization
drowns you; a discovered predicate arrives with no object_kind/template/validation/query
role, so a human designs it anyway; and the predicate set is the supervisor contract.
Side benefit: the clustered log is an evidence-backed future-work section for the dissertation.
