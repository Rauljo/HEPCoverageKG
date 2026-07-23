# Pydantic for validation

**Status**: discussed (2026-07-15) — direction leaning: NO to internal retrofit, YES to
LLM-boundary guided decoding as a measured experiment. Not implemented.

## Context

Pydantic is already a dependency (`pydantic==2.13.4`; DeepCollector's merger used it for LLM
response schemas). Current validation is two separate jobs:
- **Job A — internal structures**: `state.py` dataclasses (EvidenceSpan, EntityRef,
  CoverageAssertion, PaperExtraction) use plain `@dataclass` + manual `validate()` raising
  ValueError.
- **Job B — LLM output**: `rag_engine.py` hand-parses messy JSON (`_extract_json_objects`,
  `_parse_response`, confidence normalization, markdown-fence stripping, per-term-verdict quirk).

## Verdict per job

**Job A (retrofit the dataclasses) — recommend NOT doing it.**
1. **Supervisor-contract divergence** (the real reason): these dataclasses are ported from
   the supervisor's `graph_schema.py`, whose comment says it "intentionally stay[s]
   dependency-free." Repo may merge into the supervisor's (D-020) → converting the shared
   schema contract to Pydantic invites merge friction.
2. Marginal: by construction time data is already parsed/normalized; validate() are short
   invariant checks. Pydantic shaves boilerplate, changes no outcomes.
3. Threatens D-011's deliberate split (normalize-in-engine vs validate-strictly-in-dataclass) —
   Pydantic validators coerce inside the model, collapsing the layer separation. If ever done,
   decide deliberately whether normalization moves into the model.

**Job B (LLM boundary via vLLM guided decoding) — recommend YES, as a measured experiment.**
We self-host vLLM, which supports constraining generation to a JSON schema (Pydantic model →
guided-decoding backend). Could retire the whole hand-parsing class (malformed JSON, fences,
per-term-verdict quirk) — robustness by construction, not validation after the fact.
Caveats:
- **Don't lose the signal**: the per-term-verdict behavior carried MORE info (per-term
  confidence) than the single-object format. Design the schema to match what the model wants
  to do (explicit list of per-term verdicts), not to force the lossy single-object shape.
- **Constraining can hurt an 8B's reasoning** (forced JSON suppresses "thinking"; small
  models feel it more). Keep a free-text `rationale` field so it can still reason.
- **It's evaluable**: freeform-parse (current) vs guided-decoding — extraction
  precision/recall + parse-failure rate, head to head. A results-table ablation, not plumbing.

## Field-level in, object-level out (AgentRivet, 2026-07-15)

AgentRivet treats model outputs as untrusted until schema-validated, and gets both
structured-output constraint and post-hoc validation from one Pydantic class. Two caveats it
confirms: **validation ≠ confidence** (well-formed ≠ true; AgentRivet has no confidence
anywhere, only its review loop), and a whole-object Pydantic model over the *assembled* record
is a **trap** — one bad field fails the entire construction, reintroducing exactly the
monolithic failure cellular RAG exists to avoid. Pattern: **validate per cell on the way in**
(`TypeAdapter(int).validate_python(v)`, or one tiny model per field), reserve whole-object
Pydantic for the **derived** layer where all-or-nothing is genuinely wanted. Our design is
already close (per-field extraction + per-field checks in rag_engine; object-level `validate()`
only at the end). Concrete: `PLAUSIBILITY_THRESHOLDS` could become Pydantic `Field(ge=…, le=…)`
constraints rather than the bespoke `_validate_plausibility` — if guided decoding is adopted.

## Open

- Run the guided-decoding ablation once a gold set exists (ties to multi-agent baseline eval).
- If Job A is ever reconsidered, clear it with the supervisor first (shared contract file).
