# Literature — annotated, project-relevance-first

*Everything we discuss gets an entry: what it is, why it matters HERE, where it came up. This accumulates the related-work chapter passively.*

---

## OpenIE — Open Information Extraction (Banko et al. 2007; ReVerb, OLLIE, Stanford OpenIE)
**What**: Extract (subject, relation, object) triples with no predefined schema — relation phrases lifted verbatim from text. Canonical pre-LLM attempt at schema-free extraction.
**Why it matters here**: Its failure mode is the cautionary tale for predicate discovery: uncanonicalized relation phrases explode ("was born in" / "is a native of" / "'s birthplace is" = three relations). Direct justification for our fixed predicate schema + log-only discovery (D-019) and for treating canonicalization as the hard part.
**Where discussed**: 2026-07-10 session (predicate-discovery discussion).

## NELL — Never-Ending Language Learner (Carlson et al. 2010; Mitchell et al. 2018, CMU)
**What**: Ran ~a decade reading the web 24/7, bootstrapping a growing KB — learned facts fed later learning, with periodic human correction.
**Why it matters here**: Two lessons imported directly: (1) *semantic drift* — beliefs feeding beliefs correlates errors; the same phenomenon argued against memory between our extraction calls (D-017); (2) its human-in-the-loop gate is our human-confirm step in reconciliation (D-016). Our design ≈ a NELL-style loop at dissertation scale with a much tighter human gate — citable framing for the write-up.
**Where discussed**: 2026-07-10 session.

## AutoSchemaKG (Bai et al. 2025, arXiv 2505.23628)
**What**: Flagship LLM dynamic schema induction: extracts triples AND induces the schema simultaneously from raw text; a *conceptualization* stage clusters raw relation phrases (embeddings + LLM) into canonical predicate types. Built "ATLAS" KGs (name collision with the LHC experiment — careful in the write-up) from 50M documents; induced schemas reach ~95% semantic alignment with human-crafted ones, zero manual intervention.
**Why it matters here**: (1) Read the 95% number in reverse: given our expert schema, full induction's marginal value is the disagreeing ~5% — which our leftovers log captures far more cheaply. (2) Their regime (no ontology, web scale, unbounded goal, big models) is the opposite of ours on every axis — the argument for NOT adopting it wholesale. (3) One technique stolen outright: conceptualization-style clustering as post-processing for the leftovers suggestion log (D-019).
**Where discussed**: 2026-07-10 session (recent-research question).

## LLM-empowered KG construction: A survey (Bian et al. 2025, arXiv 2510.20345)
**What**: Organizes the field into *schema-based* (structure, normalization, consistency) vs *schema-free* (open discovery) paradigms; concludes they're complementary — ideal systems combine schema-free discovery with schema-based formalization.
**Why it matters here**: That conclusion is literally our architecture (fixed predicate core + NEW:/leftovers discovery channel + human-gated formalization) — lets the dissertation position the design as an instance of current best practice rather than ad-hoc plumbing. Also mentions LKD-KGC (lightweight schema induction by clustering entity types from document summaries) as the lighter end of the spectrum.
**Where discussed**: 2026-07-10 session.

## AgentRivet (arXiv 2606.13535)
**What**: LLM multi-agent workflow that reads HEP journal publications, extracts analysis details, and generates the missing Rivet routines (only ~39% of collider measurements have one), with built-in code- and physics-review steps. Reports competent generated routines; main physics-error source = publication ambiguity; complex observables hardest.
**Why it matters here**: The closest existing system to this dissertation — same extraction problem (LLM reads HEP paper → structured analysis record), different target (Rivet C++ vs KG triples). Use for: related-work anchor, failure-mode expectations (publication ambiguity), review-step comparison (our consistency-check pass, D-018), and evaluation-design template. **Not yet read in full — Tier 1.**
**Where discussed**: 2026-07-14 (reading-list triage; abstract fetched).

## MUSiC — CMS Model Unspecific Search
**What**: CMS's model-independent search: classifies events into exclusive event classes by final-state object multiplicity and scans them against SM expectation.
**Why it matters here**: The schema's `result_has_final_state` is explicitly "MUSiC-style"; MUSiC's event-class definition is the leading candidate for our canonical final-state ID (see `ideas/final-state-representation.md`). Read with one question: what exactly constitutes an event class, incl. flavor-inclusive vs -specific handling. **Not yet read — Tier 1.** Companion needed: ATLAS's model-independent general search, for the other experiment's conventions.
**Where discussed**: 2026-07-14 (reading-list triage).

## Rivet + Contur (recommended addition, not yet on the user's list)
**What**: Rivet = analysis-preservation toolkit whose routine database is a machine-readable record of published analyses' selections. Contur = uses Rivet routines to determine which BSM models are already constrained by existing measurements.
**Why it matters here**: Contur answers the flip side of our coverage-gap question — an examiner will ask how this KG differs from what Rivet/Contur already provide. Likely THE physics-side related work. AgentRivet cites both.
**Where discussed**: 2026-07-14 (reading-list triage).

## Scientific KG and ontology generation with open LLMs (Digital Discovery, RSC, 2026, DOI 10.1039/D5DD00275C)
**What**: Scientific-domain KG/ontology construction using self-hosted open models.
**Why it matters here**: Existence proof that the open-weights self-hosted regime (ours: Llama-3.1-8B on DIAS) is a publishable methodology for scientific KG construction — useful citation for the infrastructure choices (D-012).
**Where discussed**: 2026-07-10 session.

---

# Reading queue (triaged 2026-07-14)

**Tier 1 — read now, unblocks work**: MUSiC paper; AgentRivet (2606.13535); one example ATLAS + one CMS paper (also: CMS paper doubles as harvester test input; both inform result_type vocab).
**Tier 2 — dissertation positioning**: LLM-KG survey (2510.20345); AutoSchemaKG (2505.23628, conceptualization section); fellow student's "agents in science" literature review (know the group's baseline, avoid duplication).
**Tier 3 — skim/defer**: DeepLearning.AI KG-with-agents course (background only; architecture is decided, D-002); llm-wiki-newsroom (README skim).
**Tier 4 — dropped for the dissertation** (off-domain; no orchestration agent per D-002; GitHub stars ≠ relevance): FINSABER, TradingAgents (2412.20138), ai-hedge-fund, MiroFish.
**To add, not on the original list**: ATLAS model-independent general search (MUSiC's ATLAS counterpart); Rivet + Contur (see entry above); entity-resolution/canonicalization evaluation methodology (for reporting reconciliation merge precision/recall).
