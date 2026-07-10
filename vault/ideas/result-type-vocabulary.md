# `result_type` vocabulary

**Status**: discussed, open — needs supervisor input.
**Discussed**: 2026-07-06 (schema review), 2026-07-07.

## The gap

`PaperExtraction.result_type` is a bare string with no controlled vocabulary anywhere —
not in our schema, not in the supervisor's own `graph_schema.py` (plain `str`, no Literal).

## Candidate vocabulary (from discussion, not settled)

- **search** — hunts for BSM physics; null result → exclusion limit. The type that
  directly answers the coverage question.
- **measurement** — precision measurement of a known SM process; may touch the same
  final state without probing for new physics.
- **combination** — statistical combination of prior results (possibly cross-experiment).
- **observation** — discovery-significance claim.

## Why it matters more than metadata

It likely **gates coverage-gap logic**: a BSM coverage gap should probably count only
`search`-type results — a measurement touching the same final state is not evidence that
anyone searched there for new physics. So the vocabulary (and which types count) must be
settled before `kg/queries.py` is designed.

## Action

Raise with the supervisor: proposed vocabulary above + the gating question ("do only
searches count toward a gap, or should measurements count with a different weight?").
