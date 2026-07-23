# HEPKG repo onboarding — findings, questions, and plans (2026-07-23)

A working synthesis after reading through the supervisor's repo (`HEPKG_promopt_tests`):
what the system is, the problems worth surfacing, questions for Gabriel, notes for the
team, suggestions for Sunny, and the gap-finder evaluation plan. Sections are written so
the relevant ones can be lifted directly into messages.

---

## 1. The setup, as I now understand it

- **The repo is Gabriel's acquisition pipeline**: arXiv HTML → structured "bundles", driven
  through Claude Sonnet (`claude -p`), schema `hepkg-acquisition-v0.2`. Pilot = 60 papers,
  **14,188 assertions**, 6,047 entities, 8,482 evidence records.
- **A bundle** = one paper's contribution: entities (nodes), assertions (typed edges),
  evidence (quote + hash-anchored location), statuses (lifecycle), qualifiers (open
  key/value refinements), plus QA findings and expert decisions as metadata.
- **Two student threads, mirror images of each other:**
  - **Me (Raul) — KG construction**: import the bundles into one queryable knowledge graph,
    preserving status/evidence/history. I treat the facts as *given*; correctness isn't my
    job. Output = the graph. Graded by the repo's integration fixtures + expected counts.
  - **Sunny — verification agents**: build agents that do "rung 6" of the review ladder
    (does the quote support the claim? is the typing right? what's missing?) — the part the
    deterministic checker can't. Output = verdicts (accept/reject/correct/missing), graded
    against human experts.
  - **The seam is the `status` field only.** Sunny's verdicts → (human-confirmed) → Gabriel's
    pipeline → updated bundles → I re-import (milestone 4) → statuses change in my graph.
    No shared code; coordinate only through the bundle contract.
- **My milestones**: (1) load & reconcile all 60 bundles idempotently, reproducing
  14,188 / 11,309 / 2,555 / 324 by status; (2) trace any assertion to its quote/section/paper;
  (3) accepted-view filter + a real physics query with OR-logic correct; (4) re-import a
  reviewed paper and land promotions + corrections without disturbing the rest.
- **The contract (`hepkg-integration.md`) gives me a test suite for free**: known-good,
  correction, conflicting, and invalid fixtures under `examples/integration/`, plus 7 explicit
  pass conditions and a suggested table layout. That is effectively my acceptance checklist.

---

## 2. Problems / findings worth surfacing

### 2.1 Signatures are unpopulated — a structure the schema defines but the data lacks
- **0 of 14,188 assertions have a populated `signature`.** The logical-tree machinery
  (`all_of`/`any_of`/`object_count`) exists in the schema and is wired through the code, but
  no bundle uses it.
- **Root cause (traced in code):** Claude Sonnet does not reliably emit the strict
  `SignatureNode` grammar; it produces bare cut-expressions instead. `coercion.py`'s
  `_demote_malformed_signatures` then converts every malformed attempt into `object_value`
  "so the claim survives instead of killing the paper" — its docstring even names 2307.01094.
  Net: **1,067 dict-valued `object_value`s are demoted signature attempts**, and final states
  live as `object_id` (138) or free-text `object_value` (23), never as signatures.
- **Impact:** milestone 3 ("which analyses require exactly 2 electrons and MET > 200 GeV,
  OR-signatures handled correctly") assumes a structure the pilot data doesn't currently have.
- **Candidate fixes (to propose, not decide):** constrained/guided decoding (force the
  grammar), or — cheaper and robust — a **deterministic compile-step** from the flat
  multiplicity dicts (which the model *does* produce well) into `SignatureNode` trees.

### 2.2 Qualifiers — heavily used, open, and inconsistently named
- **55.3% of assertions (7,841) carry qualifiers**; several hundred distinct keys with a long
  tail. The important ones are meaningful (`role: signal/control`, `level: reconstructed/particle`,
  `count`). The tail is messy: the same concept appears under many keys/casings
  (`cl`/`CL`/`confidence_level`; a dozen pT-threshold spellings; `flavor`/`flavour`;
  `sqrts`/`sqrt_s`).
- **Consequence for the importer:** store them **losslessly as JSON** (the contract requires
  "don't drop unknown keys"), promote only the 2–3 query-critical keys (`role`, `level`) to
  indexed form, and normalize other keys only on demand when a query needs them.

### 2.3 Some qualifiers are disguised edges (latent graph structure)
- Qualifier keys `channel`, `background`, `collision_system`, `observable` **exactly match
  entity kinds**, and their values name entities that exist as nodes — e.g. *"this W+jets
  background is estimated in the dilepton channel"* stored as `channel: "dilepton"` instead of
  a link to the dilepton node. These are relationships flattened into strings.
- **Opportunity:** re-link them into proper edges at import time → a richer, more queryable
  graph. **But** it needs entity resolution (the same matching/ambiguity problem) and is a
  graph-modeling decision that brushes against Gabriel's extraction boundary → ask first.

### 2.4 Data-consistency check: the "contradictions" were mostly my tooling
- Checked whether qualifiers contradict structured fields (collision energy, the cleanest
  case). Apparent disagreements (7 vs 13, 5.02 vs 502) turned out to be **parsing artifacts** —
  5.02 TeV is encoded three ways (`pp502tev`, `pp-5p02tev`, `"5.02 TeV"`) — plus legitimate
  multi-energy papers. **Real energy disagreements ≈ 0.** Qualifier↔structured consistency is
  good, so "contradiction detection" is lower priority than it first looked. Lesson worth
  keeping: **value normalization is the fiddly part**, harder than key-matching.

### 2.5 Retrieval model (from a separate analysis of the pipeline)
- Gabriel's extraction uses **deterministic section routing** (PaperMap role → heading regex →
  char cap), **not embeddings**. Note the terminology trap: "hybrid" in that repo is a
  *prompt-input mode* for the multiplicity prompt, **not** a hybrid retriever.
- Section routing handles **~96%+** of the 1,017-paper corpus cleanly; only **~38 papers
  (3.7%)** hit the whole-text fallback (8 pure letter-style). Implication: **do not** replace
  section routing wholesale with embedding retrieval — the only defensible home for a hybrid
  retriever is a **targeted rescue of those ~38 fallback papers**, and it should be decided as
  an A/B on those, not on spec.

### 2.6 Practical: the source corpus isn't in the repo
- The **paper list is** (`pilot/pilot60.json` + the bundle filenames are the arXiv IDs), but
  the **source HTML corpus lives on Gabriel's machine** — needed to run or prototype extraction.

---

## 3. Questions for Gabriel

1. **Signatures.** No bundle populates `signature`; Sonnet's malformed attempts are demoted to
   `object_value`. Is valid structured-signature extraction planned for a future schema version?
   For milestone 3 *now*, should I parse the free-text final states or wait for populated
   signatures? And — would you be open to constrained/guided decoding, or a deterministic
   multiplicity-dict → `SignatureNode` compile-step, to fix the demotion?
2. **Qualifier key normalization** (`cl`/`CL`/`confidence_level`, pT-threshold spellings, US/UK
   `flavour`): is normalization intended upstream in your pipeline, or is it the graph
   consumer's job?
3. **Disguised-edge qualifiers** (`channel`, `background`, `collision_system`): several look like
   references to entities that already exist as nodes. Should the graph re-link those as proper
   edges, or keep them as qualifier strings by design?
4. **`category`/result-type vocabulary** (`search`/`measurement`/`detector`/`performance`/`other`):
   confirming this is the intended set — and how it should gate coverage (only `search` counts
   toward a BSM gap, or measurements too, weighted differently?).
5. **The six-way "no accepted result" distinction** (absent / excluded category / failed
   extraction / quarantined / awaiting review / no match): confirming these are the categories I
   should preserve, since they map directly onto how I'll have to report coverage gaps.
6. **Corpus access**: could I get the source HTML corpus, so I can prototype/replay extraction
   locally (e.g. to test the signature-compile-step idea)?
7. **If the repo-merge is happening**: is the intent that my retrieval aligns to your section
   routing? (My read is yes for ~96% of papers; the only candidate for embedding/hybrid
   retrieval is the ~38 letter-style fallback papers.)

---

## 4. Notes for the team

- **Shared mental model to confirm:** two threads (KG construction / verification), mirror
  images, meeting only at the `status` field via the bundle contract. If anyone disagrees with
  that boundary, better to catch it now.
- **Signature-emptiness is a shared concern:** it shapes my milestone 3 *and* reflects on the
  structure Sunny's agents are judging (multiplicity/typing). Worth a joint awareness.
- **Reassuring data-quality signal:** qualifier↔structured energy values are consistent once
  parsed correctly; the apparent contradictions were tooling artifacts, not extraction errors.

---

## 5. Suggestions for Sunny

- **The 964 contested assertions are the shared hard set** — your richest review population, and
  also where my data-quality edge cases concentrate. Worth treating as common ground.
- **`predicate_typing` is the dominant failure signature** (e.g. 2006.05880: 149 contested, all
  typing). Two hooks from my side:
  - Qualifiers like `role: signal/control` and `level: reconstructed/particle` **encode typing
    information** — your typing-judge could use them as features rather than reading prose alone.
  - The **disguised-edge finding** (`channel`/`background`/`collision_system` qualifiers) is
    typing-adjacent; a shared look might help both of us.
- **Constrained/guided decoding** (the fix I'd propose for signatures) likely helps any
  *re-extraction* your agents do — same "force the model onto a strict format" lever.
- **Let's fix the verdict → status → re-import seam early**: the exact format of the decisions
  that eventually flow into my graph (milestone 4) is our one real interface — cheaper to agree
  on now than to retrofit.

---

## 6. Evaluation plan for the gap finder (held-out corpus validation)

*(From `ideas/held-out-gap-validation.md` — the method for evaluating the coverage-gap finder.)*

- **The method:** build the KG on a *subset* of papers, enumerate coverage gaps, then check which
  of those gaps get **filled** once the held-out papers are added. A train/test split for gaps —
  a standard **matrix-completion / link-prediction** framing (examiner-legible, deterministic,
  no LLM needed for the check).
- **Why it's strong:** the hard part of gap-finding is telling a *physically-sensible* empty cell
  from a *boring* one. The held-out papers are a free **oracle for "sensible"**: a gap the
  held-out set fills was an empty-but-real cell the finder correctly surfaced. This is the
  *intrinsic* counterpart to the external InspireHEP literature-check — same goal, zero cost.
- **The caveat to state up front (don't hide it):** held-out validation measures the
  **enumeration + ranking machinery**, not the scientific novelty of the headline gaps — it
  rewards finding cells someone *did* measure, which is almost the opposite of the dissertation's
  payoff (cells *no one* has). Correct framing: *"held-out fill-rate shows the finder surfaces
  physically-sensible empty cells rather than random ones; genuine-discovery claims still rest on
  the external literature check."* Both, not either/or.
- **Design notes:** use **random splits** (not "first X" — ordering biases it; multiple random
  splits also feed the non-determinism/variance analysis); plot a **learning curve** (held-out
  fill-rate vs corpus size — a cheap, examiner-friendly figure); metrics = a precision-like
  fraction (predicted gaps that are corpus artifacts) + a ranking metric (do high-ranked gaps get
  filled more than low-ranked ones?).
- **The efficiency variant (gap-directed / "pull" retrieval):** search the corpus only for the
  specific flagged gaps instead of extracting everything. For *this* fixed corpus it buys nothing
  operationally (exhaustive extraction is the product); it matters mainly as the **scaling-path**
  story — the only feasible mode for a corpus too large to fully extract (all of InspireHEP).
  Load-bearing caveat: "pull" can reliably *confirm a fill* but must **never** be used to *assert
  a gap is genuine* (absence-of-evidence inherits per-query retrieval recall).

---

## 7. Immediate next steps

1. **Start milestone 1** — the importer, built against the `examples/integration/` fixtures and
   the count target (14,188 / 11,309 / 2,555 / 324).
2. **Send Gabriel the section-3 questions** (signatures + corpus access are the two that unblock
   the most).
3. **Get corpus access** if I want to prototype the signature-compile-step.
4. Keep the held-out validation plan on file for when the graph + `kg/queries.py` exist — it's an
   evaluation concern, not a blocker now.
