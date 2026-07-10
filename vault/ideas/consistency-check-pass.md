# Consistency-check pass

**Status**: adopted → D-018. Not yet implemented.
**Discussed**: 2026-07-10 (arose from "what about keeping memory between the 6 calls?").

## Design

After the six primary predicate extractions finish, one extra LLM call sees all extracted
facts together and judges cross-field consistency — e.g. *"energy says 13 TeV, luminosity
says 20 fb⁻¹: mutually consistent for one analysis?"* (13 TeV + 139 fb⁻¹ = Run 2 pairing).

**The one crucial rule**: its output can only **flag** — demote an assertion to
`needs_review`, lower confidence — **never rewrite** an answer.

Trivial cases (energy↔luminosity per LHC run) don't need an LLM: a deterministic lookup
table of standard run configurations covers them first.

## Why this instead of memory between the primary calls (D-017)

Chaining the six extractions (prepending earlier answers to later prompts) was considered
and rejected: it anchors later calls to earlier errors (correlated failures defeat the
plausibility/vocab cross-checks, which assume independent measurements — the deliberate
"cellular" isolation in cellular RAG); it breaks EvidenceSpan provenance (an answer driven
by an injected prior fact is grounded in another assertion, not the paper — fatal to the
dissertation's evidence-grounded-method claim); and it makes results depend on arbitrary
predicate ordering.

General principle (shared with the leftovers pass and the merge judge): accumulated facts
are fine as the *subject* of a downstream check, dangerous as *input* to primary measurements.
