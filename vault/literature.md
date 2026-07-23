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

## AgentRivet (Costa, Doglioni, Gütschow, Pilkington, Sinha — arXiv 2606.13535, Jun 2026, Manchester/UCL; code gitlab.com/hepcedar/AgentRivet)
**What**: Four-agent LLM pipeline arXiv-ID → Rivet routine. **Analyst** (paper → Pydantic-validated structured analysis info) → **Coder** (→ Rivet4 C++) → **Code Reviewer** (syntax/compile, *blocker*) + **Physics Reviewer** (fidelity, *advisory*) in an iterative loop, over a shared **Memory** (a per-run disk cache, NOT a knowledge base — no entity resolution/canonicalisation/cross-paper anything). Provider-agnostic `generate()`. Benchmarked on 2 measurements × 3 LLMs × 3 runs. **Read in full 2026-07-14/15** — transfer lessons filed across the idea docs.
**Why it matters here**: closest existing system to this dissertation, and from a *strictly more favourable* regime — measurements have precise fiducial definitions **because Rivet demands it**; our searches have no equivalent discipline (reco-level, region-dependent, often only in aux tables). Both halves belong in related work. Key transferred lessons: blocker-vs-advisory authority (→ [multi-agent-extension.md](multi-agent-extension.md)); silent omission is the dominant, invisible failure and critics must re-read source (same); source-ambiguity (not model incapacity) bounds achievable accuracy (→ [grounding-and-evaluation.md](grounding-and-evaluation.md)); n=3-runs confidence-vs-variance check; field-level-in/object-level-out Pydantic (→ [pydantic-validation.md](pydantic-validation.md)); false-covered escape hatch (→ [open-vocab-reconciliation.md](open-vocab-reconciliation.md)).
**Positioning notes**: their single-pass extraction is cheaper and preserves cross-field coherence — our cellular RAG degrades gracefully (one bad field, and we know which) at the cost of coherence; both deliberate, defend honestly. Their "Memory looks like a store but is a cache" = *our landscape/KG layer is the thing their architecture assumes exists and doesn't build*. Their eval is thin (prompts tuned on 2 papers, eval on 2 others; no cross-check vs official routines; grounding "ablation" n=1/arm, unnamed) — our plan is a stronger answer to the same no-ground-truth bind. Cost USD 1.20–2.20/routine (4 agents + loop) — moot for us (self-hosted, D-012) but motivates the cheap/strong two-tier model split.
**Where discussed**: 2026-07-14 (triage), 2026-07-15 (full read → transfer memo processed into vault).

## Supervisor's HEPKG acquisition pipeline (`HEPKG_promopt_tests`, Facini; local repo)
**What**: The supervisor's own prompt-engineering pipeline, arXiv HTML → structured acquisition JSON, driven through `claude -p` (schema `hepkg-acquisition-v0.2`). Multi-stage chain: classify → papermap → selection (Prompt 2) → multiplicity (Prompt 3) → review. **Retrieval = deterministic structural section routing, NOT embeddings**: papers parsed into named sections; the relevant sections are selected per question by **PaperMap role** (`_routed_sections`) or, failing that, a **heading regex**, then char-capped and handed to the LLM. No vector index / BM25 / rerank anywhere. **Terminology trap**: "hybrid" in that repo = a *prompt input mode* for Prompt 3 (selection-JSON **+** event-selection section text vs JSON-only) — the FINDINGS production recommendation for multiplicity (iter4 10/10) — NOT a hybrid retriever. Distinct from our own genuine hybrid retriever (BGE dense index in `extraction/state.py`).
**Why it matters here**: our repo may merge into the supervisor's (overview "Pending questions"), so their retrieval model is the baseline we'd align to. **Empirical fallback measurement (from their `census.json`, corpus = 1,017 papers)**: the whole-text `Body` dump (parser's letter-style fallback, `_cap(Body, 60000)`) can fire on at most **38 papers (3.7%)** — those with no analysis-content `<section>` markup (`"no ltx_section elements"`); of which **8 (0.8%)** are pure letter-style (no named sections at all): `2002.12223, 2004.01678, 2004.04545, 2008.05928, 2009.14537, 2011.07812, 2304.08962, 2407.10549`. **Upper bound, not actual**: letter-style papers still get `Body ¶i` paragraph pseudo-sections that PaperMap can route, so the raw dump fires on ≤38, realistically fewer.
**Retrieval-design implication**: section routing already handles ~96%+ of the corpus cleanly → do NOT replace it with embedding retrieval wholesale. The only defensible home for hybrid retrieval is a **targeted rescue of those ~38 fallback papers**; and if retrieving in this domain, use *hybrid* (BM25+dense) not dense-only — exact technical tokens (region names, cut values, symbols) reward the sparse half. Decide it as an A/B on the 38, not on spec. Full discussion: this thread (2026-07-23).
**Where discussed**: 2026-07-23 (walked the supervisor repo's retrieval + measured the letter-style fallback).

## MUSiC — CMS Model Unspecific Search
**What**: CMS's model-independent search: classifies events into exclusive event classes by final-state object multiplicity and scans them against SM expectation.
**Why it matters here**: The schema's `result_has_final_state` is explicitly "MUSiC-style"; MUSiC's event-class definition is the leading candidate for our canonical final-state ID (see `ideas/final-state-representation.md`). Read with one question: what exactly constitutes an event class, incl. flavor-inclusive vs -specific handling. **Not yet read — Tier 1.** Companion needed: ATLAS's model-independent general search, for the other experiment's conventions.
**Where discussed**: 2026-07-14 (reading-list triage).

## Rivet + Contur
**What**: Rivet = analysis-preservation toolkit whose routine database is a machine-readable record of published analyses' selections. Contur = uses Rivet routines to determine which BSM models are already constrained by existing measurements (BSM model → generate events → run many Rivet routines → does the BSM contribution distort any measured distribution?).
**Why it matters here**: Contur answers the flip side of our coverage-gap question — the examiner *will* ask how this KG differs from Rivet/Contur. **The programme-level connection (real future-work framing)**: Contur's reach is bounded by Rivet's coverage — the ~39% is the fraction of measurements Contur can even see, and AgentRivet exists to raise that ceiling. Meanwhile Rivet's ~230 "high-priority" missing routines are flagged by hand — **an event-class coverage map would give a principled priority ordering** (which missing routines sit in classes nobody else covers). So AgentRivet raises this programme's ceiling and our map could tell AgentRivet where to aim: two projects pointing at each other. Also grounding-side: Rivet is *particle-level* (measurements); searches need reco-level SimpleAnalysis/pyhf — see [grounding-and-evaluation.md](grounding-and-evaluation.md).
**Where discussed**: 2026-07-14 (triage), 2026-07-15 (AgentRivet read).

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
