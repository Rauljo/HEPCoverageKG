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

# Gap-finding payoff arm — literature (search 2026-07-26)

*From the Elicit (3 searches) + Gemini deep-research run for the payoff arm (gap finder + held-out
validation + theory-side). Triaged against the idea docs; design lives in [gap-hypothesis-system](ideas/gap-hypothesis-system.md)
et al. **Reading order is preserved below** — Tier 1 = core (read top-to-bottom); Tier 2 = support.
Agent systems marked **[AGENT]**.*

**Two cross-cutting facts that shape the whole chapter:**
1. **Nearly all prior art is biomedical LBD / drug-repurposing** (A–B–C co-occurrence link prediction).
   The open niche this project fills = gaps-as-**coverage-grid** (final-state × experiment × energy) +
   a **theory-vs-measurement** axis. Nobody here does coverage mapping — that's the positioning win.
2. **Two distinct "gap" notions run through these; do not conflate them** — *text-declared/implied*
   gaps (GAPMAP) vs *structurally-absent graph entries* (ResearchLink, fact-discovery, PU, AGATHA).
   This project is firmly the structural kind; its Stage-B literature check borrows from the first.

## Reading order

**Tier 1 — core (read in order):**
1. Swanson 1986 — LBD origin
2. GAPMAP (2510.25055) — literal "gap mapping" with LLMs
3. ResearchLink (Borrego 2025) — hypothesis gen as link prediction over a sci KG
4. Fact discovery from KGE (Bhagaskoro 2024) — enumerate complement + rank
5. KG-CoI (2411.02382) — KG-grounded LLM hypothesis gen + hallucination check
6. **[AGENT]** GeneAgent (2405.16205) — self-verification agent over domain DBs (= Stage-B verifier)
7. AGATHA (2002.05635) — temporal-holdout validation
8. Open-world KG eval (2209.08858) — metrics when "absent" ≠ "false"
9. nPUGraph (2306.07512) — PU learning for absence
10. **[AGENT]** SciAgents (2409.05556) — multi-agent graph reasoning (= theory-side vision)

**Tier 2 — support:** **[AGENT]** BioKGBench (2407.00466) · SKiM-GPT · Dyport · Daowd 2022 · LinkExplorer.

**Reference / parked** (skim or cite, don't read cover-to-cover): Rossi 2020 (LP methods survey);
Zamini 2022 & Peng 2023 (general KG/KGC surveys, overlap the Bian survey above); ADVISE 2023
(evidence-gap-map methodology); SciGraph-LLM 2026 & SemNet Explorer 2026 (evidence-grounded KG
*construction/reporting* — belong next to the grounding track, not the payoff).

---

## [1] Swanson 1986 — "Fish Oil, Raynaud's Syndrome, and Undiscovered Public Knowledge" (Perspectives in Biology and Medicine, DOI 10.1353/pbm.1986.0087)
**What**: The founding paper of literature-based discovery (LBD). Swanson showed two disjoint literatures — fish oil's effects on blood viscosity/platelets, and those same effects in Raynaud's — implied an untested connection (fish oil may treat Raynaud's), later clinically confirmed. Coined "undiscovered public knowledge."
**Why it matters here**: The conceptual root and canonical citation for the whole payoff arm — a "gap" as knowledge *latent in but never stated by* the corpus. Grounds the claim that empty cells can be *implied* by what's present. Every downstream system here (AGATHA, SKiM-GPT, Daowd) is a descendant. **Boundary to draw**: Swanson's ABC is co-occurrence-implied *links*; this project's gap is a structurally-absent *cell in a typed coverage grid* — cite for lineage, then state the difference. [gap-hypothesis-system](ideas/gap-hypothesis-system.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [2] GAPMAP (Salem et al. 2025, arXiv 2510.25055)
**What**: Uses LLMs to identify research knowledge gaps in biomedical literature, distinguishing *explicit* gaps (declared "we don't know X") from *implicit* (context-inferred); introduces TABI (Toulmin-Abductive Bucketed Inference), a structured reasoning scheme; ~1,500 docs, open- and closed-weight models.
**Why it matters here**: Closest-*named* prior art to the gap finder. But its gaps are *text-declared/inferred*, not *structurally-absent grid cells* — reading it sharpens the contribution boundary (their gap = what authors say is missing; ours = what the coverage graph shows is missing). TABI's structure-reasoning-then-bucket pattern is a candidate template for Stage-B reasoning over enumerated candidates; its human-in-the-loop verification finding echoes constraint 1. [gap-hypothesis-system](ideas/gap-hypothesis-system.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [3] ResearchLink (Borrego et al. 2025, Knowledge-Based Systems, DOI 10.1016/j.knosys.2025.113280)
**What**: Domain-independent hypothesis generation framed as **link prediction over a scientific KG**, combining path features + KG embeddings + text embeddings + bibliometric signals; evaluated on CSKG-600, a new expert-labeled hypothesis dataset; beats TransH/TransD/RotatE (78.7% P@20).
**Why it matters here**: The closest methodological sibling to Stage B — it operationalizes "an empty edge that is plausible" as ranked link prediction, exactly the enumerate-then-rank substrate. CSKG-600 is a model for building a gold set of "sensible vs boring" gaps. Its domain-independence claim is the one this project tests in HEP. [gap-hypothesis-system](ideas/gap-hypothesis-system.md) / [held-out-gap-validation](ideas/held-out-gap-validation.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [4] Fact discovery from KG embeddings (Bhagaskoro et al. 2024, EDBT, DOI 10.48786/edbt.2024.57)
**What**: Defines "fact discovery" — find *all* (or as many as possible) missing facts from a KGE model, not just answer a given query. Core obstacle: the complement graph is too large to score exhaustively, so it studies **sampling methods** to generate candidate facts, then ranks the most plausible by the KGE, with guidance on which sampling to use when.
**Why it matters here**: The most direct analogue to the deterministic-enumeration problem — the (final-state × experiment × energy) space *is* a combinatorial complement graph too big to score naively. This paper's sample-then-rank is a candidate mechanism for making enumeration tractable, and its "which sampling when" analysis maps onto deciding which empty cells to even consider. [gap-hypothesis-system](ideas/gap-hypothesis-system.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [5] KG-CoI — Knowledge Grounded Chain of Ideas (Xiong et al. 2024, arXiv 2411.02382)
**What**: Enhances LLM hypothesis generation by injecting structured KG facts, organizing output as a chain of ideas, and adding a **KG-supported hallucination-detection module**; improves accuracy and cuts hallucinations vs ungrounded LLM.
**Why it matters here**: Essentially Stage-B constraints 1–2 realized — the LLM reasons *over graph facts* (never free-generates) and its claims are checked against the KG. The hallucination-detection-against-KG loop is the template for "the LLM ranks/motivates enumerated gaps, grounded, without inventing absences." [gap-hypothesis-system](ideas/gap-hypothesis-system.md) constraints 1 & 2.
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [6] GeneAgent (Wang et al. 2024, arXiv 2405.16205) — **[AGENT]**
**What**: First-of-its-kind **self-verification language agent** for gene-set knowledge discovery: it autonomously interacts with biological databases, checks its own generated claims against them, and reduces hallucination; benchmarked on 1,106 gene sets, beats GPT-4, confirmed by manual review.
**Why it matters here**: The closest published analogue to the **Stage-B tool-using literature-check agent** — the one place the "agent" label is *earned* (constraint 2). Its verify-each-claim-against-an-external-DB loop is directly the pattern for confirming a flagged coverage gap against InspireHEP ("did an un-ingested paper measure this cell?"). Self-verification-as-data (not self-editing) aligns with the D-016 caveat in [researcher-feedback-loop](ideas/researcher-feedback-loop.md). [gap-hypothesis-system](ideas/gap-hypothesis-system.md) constraint 2.
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [7] AGATHA (Sybrandt et al. 2020, arXiv 2002.05635)
**What**: Deep-learning biomedical hypothesis-generation system that learns a data-driven ranking criterion to recommend new connections, **massively validated by a temporal holdout** — predict connections first published after 2015 using only pre-2015 data — across the 20 most common relation types.
**Why it matters here**: The examiner-legible template for held-out validation — the temporal holdout *is* the "build the KG on a subset, check which gaps the held-out papers fill" design, and its framing as ranking-quality (not novelty) matches the held-out caveat exactly. Cite as precedent that holdout-fill-rate is an accepted way to validate a gap/hypothesis ranker. [held-out-gap-validation](ideas/held-out-gap-validation.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [8] Rethinking KG Evaluation Under the Open-World Assumption (Yang et al. 2022, arXiv 2209.08858)
**What**: Shows standard KGC metrics (MRR, Hits@K) misbehave when unknown triples are treated as false (closed-world), because many "unknown" triples are missing-but-true; proves theoretically and empirically this can even *reverse* model rankings, and suggests fixes.
**Why it matters here**: The methodological warning for held-out scoring — an empty cell scored as "negative" may just be missing, so naive fill-rate/precision can mislead in exactly the way described. Read *before* fixing the held-out metric so the eval doesn't inherit the closed-world trap; it's the formal backing for the load-bearing "absence ≠ false" caveat. [held-out-gap-validation](ideas/held-out-gap-validation.md) / [grounding-and-evaluation](ideas/grounding-and-evaluation.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [9] nPUGraph — Noisy Positive-Unlabeled Learning for Speculative KG Reasoning (Wang et al. 2023, arXiv 2306.07512)
**What**: Formulates KG reasoning with both false-negative (true facts excluded) and false-positive (wrong facts included) issues as a **noisy positive-unlabeled learning** problem; a variational framework jointly estimates the correctness of collected and uncollected facts.
**Why it matters here**: The formal ML treatment of the absence problem — "not in the graph" is *unlabeled*, not *negative*. The citable framework for constraint 2 and for defending any absence claim: PU learning is how the field rigorously handles "evidence of absence vs absence of evidence." [gap-hypothesis-system](ideas/gap-hypothesis-system.md) constraint 2 / [held-out-gap-validation](ideas/held-out-gap-validation.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## [10] SciAgents (Ghafarollahi & Buehler 2024, arXiv 2409.05556, Advanced Materials) — **[AGENT, multi-agent]**
**What**: **Multi-agent** system that autonomously advances scientific discovery by combining LLMs, data-retrieval tools, and in-situ learning over **large-scale ontological knowledge graphs**, uncovering hidden interdisciplinary links and generating + refining research hypotheses; applied to bio-inspired materials to propose novel materials.
**Why it matters here**: The model for the ambitious theory-side vision — agents traversing an ontological graph to find and motivate cross-domain connections is exactly the theory×experiment cross ([gap-hypothesis-system](ideas/gap-hypothesis-system.md) "beyond final-state", options 2/3). Precedent that multi-agent graph reasoning can yield genuinely novel, expert-credible hypotheses. Its scale/ambition also flags the tractability risk that motivates the *scoped-theory fallback*.
**Where discussed**: 2026-07-26 (gap-finding literature search).

---

### Tier 2 — support

## BioKGBench (Lin et al. 2024, arXiv 2407.00466) — **[AGENT]**
**What**: Benchmark + agent (BKGAgent) for AI-scientist evaluation: disentangles "understanding literature" into scientific-claim verification + KGQA, and defines KGCheck — an agent task using KGQA+RAG to find factual errors in large KGs (found 90+ real errors).
**Why it matters here**: Template for the coherence/review-router and the author-feedback loop — an agent that *audits* a KG for errors rather than extends it. [researcher-feedback-loop](ideas/researcher-feedback-loop.md) / graph-consistency check.
**Where discussed**: 2026-07-26 (gap-finding literature search).

## SKiM-GPT (Freeman et al. 2025, BMC Bioinformatics, DOI 10.1186/s12859-025-06350-7)
**What**: RAG system combining SKiM (Serial KinderMiner) A–B–C co-occurrence LBD with a frontier LLM that *evaluates* user-defined hypotheses given retrieved PubMed abstracts; transparent/human-verifiable, κ=0.84 vs four expert biologists.
**Why it matters here**: The external-literature-check + human-review pattern made auditable — LBD surfaces candidates, LLM scores them against retrieved evidence with a written justification. Directly constraint 2's external check. [gap-hypothesis-system](ideas/gap-hypothesis-system.md) constraint 2 / [researcher-feedback-loop](ideas/researcher-feedback-loop.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## Dyport (Tyagin & Safro 2024, BMC Bioinformatics, DOI 10.1186/s12859-024-05812-8)
**What**: A benchmarking technique for biomedical hypothesis-generation systems that scores predicted discoveries by *dynamic importance* (not just binary correctness), built on curated temporal knowledge. (Same group as AGATHA.)
**Why it matters here**: Pairs with AGATHA for the evaluation chapter — moves beyond hit/miss toward importance-weighted gap ranking, relevant to "do high-ranked sensible gaps get filled more than low-ranked ones?" [held-out-gap-validation](ideas/held-out-gap-validation.md) / [grounding-and-evaluation](ideas/grounding-and-evaluation.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## KGC + LBD for cancer drug repurposing (Daowd et al. 2022, DOI 10.1007/978-3-031-09342-5_3)
**What**: Applies KG completion (FocusE-TransE) to LBD, using **time-slicing** to build incomplete KGs from cancer literature, then discovery patterns on the augmented KG to replicate known drug-repurposing findings.
**Why it matters here**: A worked instance of the held-out / train-test-split-for-gaps design — build on old data, verify predictions against later discoveries; a concrete recipe for [held-out-gap-validation](ideas/held-out-gap-validation.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

## LinkExplorer / SAFRAN (Ott et al. 2022, DOI 10.1101/2022.01.09.475537)
**What**: Software suite for predicting, **explaining**, and exploring links in large biomedical KGs; rule-based engine SAFRAN gives explainable link predictions with a web UI.
**Why it matters here**: Gaps must be *motivated*, not just ranked — SAFRAN's explainable, rule-based predictions model how to give a human-legible reason a cell is a plausible gap (vs a black-box embedding score). [gap-hypothesis-system](ideas/gap-hypothesis-system.md).
**Where discussed**: 2026-07-26 (gap-finding literature search).

---

# Reading queue (triaged 2026-07-14)

**Tier 1 — read now, unblocks work**: MUSiC paper; AgentRivet (2606.13535); one example ATLAS + one CMS paper (also: CMS paper doubles as harvester test input; both inform result_type vocab).
**Tier 2 — dissertation positioning**: LLM-KG survey (2510.20345); AutoSchemaKG (2505.23628, conceptualization section); fellow student's "agents in science" literature review (know the group's baseline, avoid duplication).
**Tier 3 — skim/defer**: DeepLearning.AI KG-with-agents course (background only; architecture is decided, D-002); llm-wiki-newsroom (README skim).
**Tier 4 — dropped for the dissertation** (off-domain; no orchestration agent per D-002; GitHub stars ≠ relevance): FINSABER, TradingAgents (2412.20138), ai-hedge-fund, MiroFish.
**To add, not on the original list**: ATLAS model-independent general search (MUSiC's ATLAS counterpart); Rivet + Contur (see entry above); entity-resolution/canonicalization evaluation methodology (for reporting reconciliation merge precision/recall).
