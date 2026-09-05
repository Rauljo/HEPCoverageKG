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
1. ✅ Swanson 1986 — LBD origin (**the *Library Quarterly* theory paper**, not the fish-oil one — read 2026-07-27)
2. ✅ GAPMAP (2510.25055) — literal "gap mapping" with LLMs (read 2026-07-27)
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

## [1] Swanson 1986 — "Undiscovered Public Knowledge" (*The Library Quarterly* 56(2):103–118, DOI 10.1086/601720, JSTOR 4307965) — **READ IN FULL 2026-07-27**

**Careful — there are TWO Swanson 1986 papers, cite the right one:**
- **"Undiscovered Public Knowledge"**, *Library Quarterly* 56(2):103–118 — **the theory paper**
  (Popper, World 3, the logic of UPK, three examples). **This is the one read.** Cite for the *concept*.
- **"Fish Oil, Raynaud's Syndrome, and Undiscovered Public Knowledge"**, *Perspectives in Biology and
  Medicine* 30(1):7–18, DOI 10.1353/pbm.1986.0087 — **the case study** for a biomedical audience.
  It is ref. [21] of the LQ paper, listed there as "in press". Cite for the *worked discovery*.
- Local copy of the LQ scan: `data/raw/papers/swanson-1986-undiscovered-public-knowledge.pdf` (gitignored).

**What (the argument, in its own order)**
1. **Thesis (p. 103)**: knowledge can be *public yet undiscovered* if independently created fragments
   are logically related but never retrieved, brought together, and interpreted. Puzzle-piece image:
   independently designed pieces can assemble into "an unseen, unknown, and unintended pattern."
2. **Popperian setup (pp. 104–106)**: against positivism — theories are conjectures, never verified,
   only tested; they "remain forever conjectural." Crucially, *positivism is an unsuitable model of
   science, but attacking positivism is not attacking science* (a distinction Swanson insists on).
   If theories can't be verified, what makes one better? Three criteria: **testable > untestable;
   tested-and-passed > tested-and-failed; withstood more criticism > less.** Perception is
   expectation-laden (Popper's *searchlight*, not *bucket*); a mistake is "a clash between expectation
   and reality"; the humanities suffer because clear confrontation with reality is harder to evoke.
   Money line for the write-up: *"the sine qua non of science is not objectivity or even 'truth' …
   but a systematically self-critical attitude."*
3. **World 3 (pp. 106–107)**: World 1 = physical; World 2 = subjective/mental; World 3 = objective
   knowledge (problems, theories, products of mind). World 3 is man-made yet exceeds its makers'
   intentions — **prime numbers existed, awaiting discovery, once the number system was invented**.
   Therefore World 3 "must contain ever increasing quantities of undiscovered knowledge," and only a
   small portion is known to anyone.
4. **Three examples (pp. 108–113)** — the heart:
   - **Ex. 1, Black swans: a hidden refutation.** "All swans are white" can be refuted but never
     verified. Shift from World 1 (find a black swan) to World 3 (find a *report* of one). A reliable
     published report can exist, incidental to some other article and missed by indexers, while the
     hypothesis it refutes stands. **A refutation can itself be undiscovered public knowledge.**
   - **Ex. 2, A missing link in the logic of discovery.** The **ABC syllogism**: (i) A causes B,
     (ii) B causes C, published independently by authors unaware of each other ⇒ (iii) A causes C
     exists objectively as an *undiscovered* hypothesis; i and ii are *indirect tests* of it. Live
     instance: A = dietary fish oil, B = reduced platelet aggregability / blood viscosity, C =
     improvement in Raynaud's patients. **The operational disjointness criterion he actually used:
     the two literatures share no common authors and no mutual citations** (nn. 3–5).
   - **Ex. 3, Hidden cumulative strength of individually weak tests.** Many independent weak tests can
     jointly constitute a strong one; via Popper's severity measure p(e,h) − p(e), the *joint*
     probability of many kinds of evidence is smaller than each separately. Case: tobacco–lung cancer,
     where fragmentary case-control/mortality/animal evidence existed but no article assembled it —
     so *the degree to which the hypothesis had withstood tests was itself undiscovered*.
5. **The essential uncertainty of information retrieval (pp. 113–115)** — the strongest section for
   this project. Documents are reachable only via "points of access"/searchable attributes, which
   cannot encode relevance to problems not yet formulated when the document was written. Any request
   for *all* information on a theory presumes a universal hypothesis: *"all pieces of recorded
   information relevant to a given theory can be described and found by constructing some specific
   function of searchable attributes"* — **the "search function"**. That hypothesis can never be
   verified (verification needs direct inspection of everything published, a task that never
   terminates because the corpus grows meanwhile) but **can be refuted by a single relevant document
   lacking the attributes**. Hence *"an information search is essentially incomplete, or, if it were
   complete, we could never know it."* The **universal-thesaurus infinite regress**: a thesaurus
   encoding all relationships in advance would have to contain a complete representation of World 3,
   and you would need a second thesaurus to search the first, *ad infinitum*. Signature formulation:
   **"A search function is a conjecture or a theory about the contents of World 3, whereas a
   scientific theory is a conjecture about World 1."** Falsifiable, not verifiable — like any theory.
6. **Interactive searching (p. 116)**: because search functions are criticizable they are improvable;
   the output of one search is the base for constructing a better one. Multistage interactive search
   > one-shot (and, he notes, online services were then used mostly one-shot).
7. **Close (pp. 116–117)**: the "central problem of IR" = finding everything bearing on the testing and
   criticism of a theory; its specific form depends on which puzzle pieces are already retrieved —
   "the logic of undiscovered public knowledge". UPK is open-ended yet has "a certain order, form, and
   structure that may be worth systematic study." **World 3 as an endless frontier**, alongside World 1.
8. **Author's own hedge (n. 8)**: he does *not* claim literature-based discovery is unprecedented or
   unusual; and cleanly separating lab-based from literature-based knowledge is itself problematic,
   since all scientific argument leans on literature-grounded background knowledge.

**Why it matters here** — four transfers, in order of value:
1. **It is the canonical citation for our hardest caveat.** [[gap-hypothesis-system]] constraint 2
   ("a gap = not measured *in the ingested papers*") and the asymmetric-reliability caveat in
   [[held-out-gap-validation]] ("finding a fill is reliable; finding nothing is not") are *special
   cases of Swanson's essential incompleteness of IR*. Swanson turns our biggest vulnerability from
   an apology into a stated epistemological position with a 40-year-old canonical citation.
2. **It supplies the Popperian frame for the whole payoff arm.** Our gap enumeration IS a search
   function = a conjecture about World 3: falsifiable, never verifiable. So the correct posture is
   **refutation-based**: an enumerated gap is a conjecture that the Stage-B literature check and the
   held-out split *attempt to refute*; surviving refutation is the claim, never "verified absence."
   This makes held-out validation ([[held-out-gap-validation]]) a *severity-of-test* argument, and
   Ex. 3 gives the vocabulary for aggregating many individually weak plausibility signals into one
   strong one — directly usable for gap ranking.
3. **Boundary to draw — two different absences (the positioning sentence).** *Swanson's gap is
   epistemic: the knowledge exists in the literature but no one assembled it. This project's headline
   gap is ontic: the measurement was never made — the cell is empty because nobody ran the analysis.*
   Cite for lineage, then state that difference. **But the project contains both**: Stage-B's
   literature check is exactly Swanson-shaped (assemble scattered evidence to decide whether a cell is
   *really* empty), so UPK is our verification stage even though it is not our discovery target.
   Swanson's *disjointness test* (no shared authors, no mutual citations) is a cheap, concrete measure
   we could compute on the HEP citation/author graph to score whether two sub-literatures are
   genuinely non-interacting.
4. **It is an argument against building the alias thesaurus up front.** The universal-thesaurus
   infinite regress (p. 115) is a principled reason our `aliases/` layer is *bottom-up and empirical*
   (cluster what the corpus actually contains, human-gated) rather than a pre-authored controlled
   vocabulary — a nice unexpected citation for [[open-vocab-reconciliation]] / the Tier 2–3 design.

**Where discussed**: 2026-07-26 (gap-finding literature search — triage); **2026-07-27 (read in full,
notes + discussion; entry rewritten from the paper itself).**

## [2] GAPMAP (Salem et al. 2025, arXiv 2510.25055) — **READ 2026-07-27**
**What**: Uses LLMs to identify research knowledge gaps in biomedical literature, distinguishing *explicit* gaps (declared "we don't know X") from *implicit* (context-inferred); introduces TABI (Toulmin-Abductive Bucketed Inference), a structured reasoning scheme; ~1,500 docs, open- and closed-weight models.
**From the read (2026-07-27)**:
- Frames gaps as **"known unknowns"** and justifies the whole enterprise as *prioritising new studies
  and directing funding toward consequential open problems* — **lift this as the motivation paragraph
  of the dissertation intro**; it is the cleanest published statement of why gap-finding is worth doing.
- Offers a **taxonomy of gap types**, with *explicit vs implicit* cut **within** each category —
  the two-axis structure worth borrowing for our own gap typology.
- Claims to be the **first work on implicit gap identification**; prior art was explicitly-declared
  gaps only. Related work traces: hedging/uncertainty **cue-word matching** → supervised ML →
  BERT-family classifiers (good scores, but scalability + generalisability still open).
- Extracted gaps come **with their supporting evidence** — same evidence-attached discipline as our
  assertions; makes the output auditable rather than an assertion of absence.
- **RoBERTa** used to validate implicit-gap extraction.
- **Two findings that transfer straight to our prompting/model choices**: performance rose sharply
  with **larger LLMs**, and **3-shot ≫ zero-shot** for *implicit* gap extraction specifically
  (consistent with routing Stage-B to the big model, [[gap-hypothesis-system]]).
- Conclusion they draw: *"using a mixture of LLMs instead of relying only on the top performing ones
  can provide a robust gap recommender system."*
**Why it matters here**: Closest-*named* prior art to the gap finder. But its gaps are *text-declared/inferred*, not *structurally-absent grid cells* — reading it sharpens the contribution boundary (their gap = what authors say is missing; ours = what the coverage graph shows is missing). TABI's structure-reasoning-then-bucket pattern is a candidate template for Stage-B reasoning over enumerated candidates; its human-in-the-loop verification finding echoes constraint 1. [gap-hypothesis-system](ideas/gap-hypothesis-system.md).
**Two ideas it triggered (Raul, 2026-07-27)** — both filed:
1. **Mine "Future Work" / "Outlook" sections as a gap source** → new section in [[gap-hypothesis-system]]
   ("Author-declared gaps"). This is GAPMAP's *explicit* channel applied to our corpus.
2. **LLM/encoder equivalence check for held-out gap matching** → [[held-out-gap-validation]]; reuse
   the aliases Tier 2–3 machinery rather than building a second matcher.
**Caveat on the "mixture of LLMs" line (do not over-claim)**: that is an **ensemble/diversity** result,
not an **agency** result. Prompt-variants of one checkpoint give correlated errors, so it does *not* by
itself justify the multi-agent arm — keep [[multi-agent-extension]]'s vocabulary discipline (workflow
vs agency). Cite GAPMAP for "an ensemble beats the single best model at gap identification"; the agent
justification must come from **tool use / self-verification** (GeneAgent), not from this.
**Where discussed**: 2026-07-26 (triage); **2026-07-27 (read + discussion)**.

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

---

# Query-system arm — literature (search 2026-07-29)

*Triage for the system in [`system.md`](system.md): GraphRAG, RAG/QA evaluation, abstention,
text-to-SQL/Cypher, agentic KG curation. **Triaged from abstracts and search results only — none of
these read in full yet.** Verify before citing. Reading order preserved: Tier 1 = read these, each
one attaches to a specific decision; Tier 2 = support.*

## Tier 1

### Know Your Limits: A Survey of Abstention in LLMs (arXiv 2407.18418)
**What**: Survey of the abstention literature — when and how models should decline to answer.
**Why it matters here**: **S-13 is not a novel idea, it is a field**, and we did not know that. The
deletion protocol measures abstention; this gives the vocabulary, the framing, and the prior work to
position against. Highest-value single read on this list — it converts our "nice metric" into a
literature-grounded contribution.
**Where discussed**: 2026-07-29 (query-system lit search).

### AbstentionBench + Do LLMs Know When to NOT Answer? (COLING 2025) + Abstain-QA
**What**: Benchmarks and methodology for abstention. Abstain-QA = 2,900 MCQs, half unanswerable,
with an explicit "I don't know" option. **AUCM — the Answerable/Unanswerable Confusion Matrix** — is
a ready-made evaluation frame. Finding: models frequently fail to recognise unanswerability and
fabricate instead; reasoning models often "know" internally but do not express abstention.
**Why it matters here**: AUCM is directly reusable for the deletion protocol (S-13) — do not invent
our own scoring frame. The "knows internally but answers anyway" result also bears on whether
confidence (which the 72B calibrates well) can drive abstention.
**Where discussed**: 2026-07-29.

### CypherBench (ACL 2025, arXiv 2412.18702)
**What**: First large-scale text-to-Cypher benchmark — 11 property graphs, 7.8M entities, >10k
questions. Argues RDF KGs are inefficient for LLMs (schemas exceed context, resource identifiers,
overlapping/ambiguous relations, no normalization) and that **property-graph views** are the fix.
**Why it matters here**: **Read this precisely because it challenges S-05** (SQL-only querying,
Neo4j as view). If graph querying is the better route, this is where the evidence would be. Also the
benchmark to quote if we ever add Cypher generation. Note the schema-size argument cuts our way for
now — our schema is small and closed.
**Where discussed**: 2026-07-29.

### Graph Retrieval-Augmented Generation: A Survey (arXiv 2408.08921)
**What**: The anchor GraphRAG survey. Formalizes the workflow as Graph-Based Indexing →
Graph-Guided Retrieval → Graph-Enhanced Generation.
**Why it matters here**: Positions the whole system in a named field. Our architecture maps onto
those three stages, so it gives the related-work chapter its spine. Companion:
**RAG with Graphs (arXiv 2501.00309)**, which decomposes GraphRAG into query processor / retriever /
organizer / generator / data source — component vocabulary that maps almost one-to-one onto S-03/04.
**Where discussed**: 2026-07-29.

### Text-to-SQL benchmark numbers — BIRD, Spider 2.0
**What**: BIRD = 12,751 pairs over 95 real databases (~33GB, 37 domains). Reported figures:
SENSE-13B 86.6% on Spider / 63.4% on BIRD; **Spider 2.0 — GPT-4o 10.1%, versus 86.6% on Spider 1.0**;
enterprise variants 39.1 EX (BIRD-Ent) / 60.5 EX (Spider-Ent).
**Why it matters here**: **This is the empirical justification for S-04** (templates filled with
retrieved ids, not free-form generation). "Free-form text-to-SQL is unreliable on real schemas" stops
being our opinion and becomes a cited number with a 10.1% next to it. Also gives the
constrained-vs-free-form ablation a benchmark vocabulary.
**Where discussed**: 2026-07-29.

### KARMA — multi-agent LLM KG enrichment (OpenReview k0wyi4cOGy)
**What**: Nine specialised agents that automatically update and expand a knowledge graph **from
scientific papers**.
**Why it matters here**: The closest published system to our paper-reading expansion loop (§2.3) —
same input (papers), same output (graph growth), same multi-agent shape. Must be in related work,
and read early enough to steal from rather than reinvent. Compare its verification story against
S-08 (provenance) and S-07 (proposals, never self-edits).
**Where discussed**: 2026-07-29.

## Tier 2 — support

### In-context Clustering-based Entity Resolution with LLMs (arXiv 2506.02509)
**What**: Packs many records into one prompt for direct in-context **clustering**, instead of
pairwise questioning. Design-space exploration of prompting strategies (match / compare / select).
**Why it matters here**: Our Tier 3 is strictly **pairwise** (126k candidate pairs, one call each).
This is the named alternative and it is cheaper. Read to decide whether to defend pairwise as a
deliberate choice or switch — either way the write-up needs the sentence.
**Where discussed**: 2026-07-29.

### Cost-Efficient RAG for Entity Matching: A Blocking-based Exploration (arXiv 2602.05708)
**What**: Blocking/filtering as the cost lever in LLM entity matching.
**Why it matters here**: Our embedding+Jaccard candidate stage plus `guards.py` **is blocking** — we
built it without using the word. This supplies the standard vocabulary and the comparison baseline
for our 126,335 → 14,656 → 9,570 reduction (D-035, D-036).
**Where discussed**: 2026-07-29.

### Can we trust LLM Self-Explanations for Entity Resolution? (arXiv 2606.01210)
**What**: Whether the explanation an LLM gives for a match/non-match can be trusted.
**Why it matters here**: We store `explanation` on every adjudication and measured **1 contradiction
in 2,198 answers** (verdict vs its own reasoning) on the 72B. This paper is exactly that question,
and would let us report that number against a prior baseline instead of in isolation.
**Where discussed**: 2026-07-29.

### Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards (EMNLP 2025 industry) — FaithJudge
**What**: Hallucination/faithfulness leaderboard across summarization, QA, data-to-text;
**FaithJudge** = LLM-as-judge grounded in human-annotated hallucination examples.
**Why it matters here**: S-14 checks faithfulness **mechanically** (every claim must appear in a
retrieved row), which is stronger than LLM-judging — but only if we can say what LLM-judging does
and why ours is different. This is that citation.
**Where discussed**: 2026-07-29.

### The multi-layer RAG evaluation argument
**What**: Recurring finding in the 2025–26 RAG-eval literature: the retriever can miss relevant
material while the generator answers coherently from partial context, so **faithfulness stays high
and no retrieval-stage metric surfaces the regression**. Hence evaluation across retrieval /
generation / end-to-end.
**Why it matters here**: This is precisely the argument for §4.4 (per-layer metrics) — end-to-end
accuracy says *that* something broke, never *what*. Find the strongest paper stating it and cite it
there. Also see *Benchmarking Hallucination Evaluation for RAG Under an Abstention Policy*, which
sits exactly at our abstention × RAG-eval intersection.
**Where discussed**: 2026-07-29.

### Awesome-GraphRAG (github.com/DEEP-PolyU/Awesome-GraphRAG)
**What**: Curated, maintained list of GraphRAG surveys, papers, benchmarks and code.
**Why it matters here**: The cheap way to stay current in a field moving this fast, and to find the
2026 work that post-dates these searches. Not a citation — a tool.
**Where discussed**: 2026-07-29.

## Second pass — the three remaining gaps (search 2026-07-29, same session)

### LLM systematic-review / evidence-synthesis automation — **the closest framing to the actual goal**
**What**: A live field with a blunt consensus. *LLMs for conducting systematic reviews: on the rise,
but **not yet ready for use** — a scoping review* (J Clin Epidemiol 2025) is the headline; a 2026
meta-analysis of 18 studies (2023–25) measures screening performance; **otto-SR** automates
screening + extraction + risk-of-bias and claims it can reproduce and update existing reviews.
Distribution of use across studies: **41% literature search, 38% screening, 30% data extraction**,
89% GPT-family.
**Why it matters here**: This is the application framing the supervisor actually described — help a
researcher see what has been covered. It gives the dissertation a **named field with a stated gap**
("not ready for use", tasks done in isolation, no persistent structure), and our answer to it is
specific: a *persistent typed graph* rather than a per-review one-shot pipeline. The 41/38/30 split
also shows nobody is building the coverage map — they automate the steps, not the artefact.
Also: **Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis**
(arXiv 2602.10881) — a failure-mode taxonomy for evidence extraction, i.e. prior work for the M4
diff idea (S-09). **Promote to Tier 1.**
**Where discussed**: 2026-07-29.

### CleanGraph — Human-in-the-loop KG Refinement and Completion (arXiv 2405.03932)
**What**: Interactive accept/reject of model-proposed corrections to a knowledge graph; keeps expert
control while cutting manual correction cost.
**Why it matters here**: **This is §2.4 and S-09 already built by someone else.** Read before
designing the editing UI. The general HITL loop it sits in — detect → elicit feedback → adapt →
quality-control — is the shape our edit-diagnosis flow should follow, and the literature notes that
captured corrections can feed back into prompts and retrieval constraints, which is exactly the
"find other things wrong for the same reason" step. **Tier 1 for the editing stage.**
**Where discussed**: 2026-07-29.

### Multi-agent systems — one survey each, then stop
**What**: *LLM-based Multi-Agents: A Survey of Progress and Challenges* (arXiv 2402.01680) as the
anchor; **A Survey on Evaluation of LLM-based Agents (arXiv 2503.16416)** for the evaluation side.
Architecture reviews catalogue ~18 reusable patterns from 57 sources; 2025–26 work has moved from
fixed topologies to runtime-adaptive ones.
**Why it matters here**: Cite and move on, per the "light" weighting — with **one finding worth
acting on**: *multi-agent overhead grows superlinearly*, and centralized coordination only pays on
parallelizable tasks. That is a direct argument for keeping the agent count low and for the
cost-per-question metric in §4.4, which we included on instinct. The evaluation survey is the more
useful of the two for us.
**Where discussed**: 2026-07-29.

## Still not searched

- HEP-specific literature tooling beyond Rivet/Contur/AgentRivet (already in this file).
- Calibration and confidence estimation in LLMs — relevant to the 72B calibration result and to
  whether confidence can drive abstention (see *Know Your Limits*, Tier 1).

---

# Evaluation arm — literature (session 2026-08-01)

*Entered while designing §4 of [`system.md`](system.md) (S-32 … S-52). Split into **systems we
compare against** and **methods we borrow**. Items marked ⚠ are from memory and need verifying
against the source before they go in the write-up.*

## Systems to compare against

### chATLAS + chATLAS_Benchmark — **the in-domain baseline, and the corpus source**
**What**: a RAG assistant over ATLAS documentation, with a published benchmark on PyPI
(`chATLAS_Benchmark`) and an MCP server. Paper draft read 2026-07-31; **Gabriel is an author**, and
its future-work section names **graph-based retrieval** as the next step.
**Why it matters here**: (1) it is the honest domain baseline — a real system, same physics, same
users; (2) **the 2,969-paper corpus on DIAS came from the chATLAS EOS area**
(`/eos/atlas/atlascerngroupdisk/phys-mlf/Chatlas/hep-papers-html/`), so same corpus lineage and its
questions may be answerable over documents we hold; (3) *it names our project as its own future
work*, which is the single best positioning sentence available to the dissertation.
**Open, and worth an hour before betting on it** (S-45): does the benchmark target **published
analyses** or **ATLAS internal documentation** (CDS notes, TWikis, indico — papers cannot answer
those)? And does it score retrieval, answers, or both? Retrieval-only is still valuable: an
externally-defined number on someone else's questions is the independent check we cannot manufacture.
**Where discussed**: 2026-07-31 (found), 2026-08-01 (scoped as a baseline). *Not previously in the
vault — recorded late.*

### Plan-on-Graph (PoG) — Chen et al., NeurIPS 2024 — **the closest prior work to our query layer**
**What**: a *prompting* (training-free) KG-augmented LLM agent on Freebase. Decomposes the question
into sub-objectives, then loops: explore paths → update memory → reflect on whether to self-correct.
Three named mechanisms — **Guidance** (decomposition), **Memory** (searched subgraph + reasoning
paths + **sub-objective status**), **Reflection** (is this enough? if not, which already-seen entity
do we backtrack to?). Beats ToG on CWQ 63.2 vs 57.1 (GPT-3.5) and 75.0 vs 67.6 (GPT-4).
**Why it matters here**: it independently names **our worst failure**. Their §1 limitation 3,
*"forgetting partial conditions"* — the model remembered the song was Taylor Swift's but forgot it
had to have won an AMA — **is gf-01**: precision 0.90 on single-condition questions, 0.30 on the
three-part one. Their ablation says the fix (sub-objective status held in memory) is the most
valuable of their four mechanisms (−4.3 on CWQ when removed, vs −3.8 reflection, −3.1 guidance,
−1.9 adaptive breadth). We have no memory of that kind at all.
**The efficiency result, which is the counter-intuitive one**: adding decomposition, memory and
reflection made PoG *cheaper* than ToG, not dearer — 13.3 LLM calls vs 22.6, **output tokens down
76%** (353 vs 1,486), 4× faster. Avoided dead-end exploration more than pays for the extra
deliberation. A direct, testable prediction for our own reviewer arm.
**Where it does NOT transfer, and this needs saying in the write-up**: (1) their relation-exploration
step exists because a Freebase entity has hundreds of relations — we have ~20 predicates and the
schema card lists them, so that call buys nothing; (2) **they assume entity linking is solved**
(§2: *"we assume any entity mentioned in q ... are labeled and linked"*) while **56% of our errors
are `unknown_entity_id`** — PoG begins after our hardest step; (3) their metric is Hits@1 on one
entity, so their pervasive *"as few as possible"* prompting is correct for them and wrong for us,
where recall is the weaker side in every arm measured. See
[`ideas/self-correcting-planning-arms.md`](ideas/self-correcting-planning-arms.md).
**The gap we sit in**: PoG is prompting-KGQA over Freebase with pre-linked entities, scored Hits@1.
Ours is a *typed coverage* KG over HEP papers, scored set-F1, with a **free-SQL control that beats
the typed agent** — a comparison they do not make and cannot speak to.
**Where discussed**: 2026-08-31 (read; four arms specced from it).

### GraphRAG (Microsoft) — the KG competitor, and the project's obvious challenge
**What**: builds a graph from a corpus automatically — LLM extracts entities and relations, clusters
them into communities, summarises each — then answers via **local search** (entity neighbourhood) or
**global search** (from community summaries). Aimed explicitly at *global/sensemaking* questions
that vanilla RAG cannot answer. Ships its own query system, so no re-implementation is needed.
`entity_types` is configurable and extraction prompts are editable (⚠ it also ships a prompt-tuning
step — verify the exact command).
**Why it matters here**: it is the **off-the-shelf answer to "why not just run GraphRAG?"**, which is
the obvious challenge to this whole project and currently unanswered. Its graph is *induced and
untyped* — relations are free-text descriptions — where ours is typed, fixed-schema, evidence-linked
and deduplicated. So the head-to-head asks a real question: **does hand-curating a typed schema buy
anything, or does an induced graph get most of the way there?** (S-44)
**Expected result, and the honest framing**: we should win on counting/coverage (free-text relations
cannot be counted reliably, and global search answers from summaries, which have already discarded
the counts) and **lose on open-ended sensemaking**, where community summaries are genuinely good and
we have no equivalent. *Reporting the loss is what makes the win credible.*
**Caveat recorded from the user (2026-08-01, largely correct)**: they serve different purposes, so
beating it at counting proves nothing anyone doubted — hence the reframing above rather than a
"we win" claim. And if it were successfully *reprompted into* a coverage map it would be a
reimplementation of our system and would measure nothing.
**Related**: **LightRAG** (cheaper, same idea), **RAPTOR** (hierarchical summarisation rather than a
graph, same target class). ⚠ citations to be pinned.

## Methods borrowed

### Question generation from logical forms — Spider, BIRD (text-to-SQL); GrailQA (KGQA)
**What**: benchmarks built by generating questions *from* queries/logical forms, then paraphrasing
into natural language — because it is the only way to get exact labels at scale over a database.
**Why here**: the precedent for **S-33** (generate questions backwards from the graph). Also supplies
the standard caveat we must state: generated questions can only ask what the database can answer, so
they are a dev set, not the headline. ⚠ pin exact citations.

### TREC-style pooling — evaluation when the collection cannot be exhaustively judged
**What**: judge the *union* of what all participating systems returned, plus a sample outside the
pool; report metrics with intervals. Standard IR practice since nobody has ever labelled a large
collection completely.
**Why here**: **S-43** — the answer to "how do we label anything at 2,969 papers". Also the reason a
multi-system comparison *helps* the labelling rather than multiplying it: more systems, better pool.
The user's refinement — judge the **quote**, not the answer — makes each judgement ~10 seconds and
requires no physics reasoning.

### LLM-as-judge — MT-Bench / Chatbot Arena (Zheng et al. 2023)
**What**: the standard reference on using strong LLMs as evaluators. Findings we obey directly:
pairwise comparison is far more reliable than absolute 1–5 scoring; **position bias** is large (swap
the order, keep only consistent verdicts); **self-preference** is real; judge–human agreement lands
around human–human agreement, which is what makes calibration meaningful rather than hopeful.
**Why here**: **S-41**, and the shape of **S-40** — Gabriel's time buys a **κ against the judge**, not
labels. That is what converts "we used a judge" into a measured instrument.

### CheckList — behavioural testing via invariants (Ribeiro et al. 2020)
**What**: test NLP systems with capability-targeted invariance and directional tests instead of a
single accuracy number.
**Why here**: **S-34** metamorphic relations (paraphrase invariance, monotonicity,
inclusion–exclusion, order invariance). Every violation is a guaranteed bug, and none of them need a
label.

### Chain-of-thought unfaithfulness (Turpin et al. 2023, *Language Models Don't Always Say What They Think*)
**What**: models' stated reasoning can be systematically unfaithful to the computation that produced
the answer — steered by factors never mentioned, with a fluent rationale attached.
**Why here**: **S-50**. It is why "make the model justify every tool call" was rejected as a *metric*,
and why the adopted version (`missing:`) is designed to be **mechanically checkable** — we verify the
claim against the retrieved rows rather than trusting the explanation.

### Semantic entropy (Farquhar et al. 2024, *Nature*)
**What**: sample an answer repeatedly and measure entropy over *meanings* (not token strings);
high entropy predicts hallucination without any ground truth.
**Why here**: **S-52** stability. Unusually clean in our setting because counting answers are numbers,
so "same meaning" needs no entailment model — `58, 58, 58, 45, 58` is already the signal.

### Lost in the middle — long-context degradation (Liu et al. 2023)
**What**: retrieval-augmented models attend poorly to information in the middle of long inputs.
**Why here**: **S-37** — the argument for the reference reader working **one paper at a time** rather
than being handed all 60 (≈700k–900k tokens) at once. Per-paper also gives per-paper provenance and
moves the arithmetic out of the model and into Python.

### Distant supervision — noisy labels used honestly
**Why here**: **S-51**. `same_id` is a *noisy positive* signal (347 shared ids vs **334 divergent**),
not ground truth. The honest use is to measure the label source's own precision on a sample and
report results against a stated noise level — not to pretend it is gold.

### PhysBERT — a physics-specific text embedding model
Thellert et al., *APL Machine Learning* (2024). arXiv 2408.09574. HF:
`thellert/physbert_cased`, `thellert/physbert_uncased`.

BERT pre-trained **from scratch** on ~1.2M arXiv physics papers, with a
physics-specific WordPiece vocabulary, then fine-tuned with SimCSE for sentence
embeddings. Tested here 2026-09-01 — see [D-087](decisions.md).

**Why it matters to us, and it is a tokenizer story.** Its vocabulary is 30,522
tokens, *the same size as bge-base*. It is not a bigger model; it spent an equal
budget on physics rather than general English. `Higgs`, `boson`, `quark`,
`luminosity`, `pseudorapidity`, `calorimeter` each survive as one token, where
bge-base and mpnet split `Higgs` into `hi` + `##ggs`. On our Higgs-candidate
discrimination that is +0.116 separation against bge-base's +0.006 — the best
measured. On ttZ control regions it loses badly to the chATLAS encoder
(+0.097 vs +0.248), which is the expected split: PhysBERT read papers, the
chATLAS model read twiki and chat, and region names are twiki vocabulary.

**Contrast with SciBERT** (Beltagy et al. 2019, `allenai/scibert_scivocab_uncased`),
same architecture and same mean-pooling in our test, trained on general
scientific text: **last of twelve, -0.010**. The relevant domain is physics, not
science. This is the cleanest evidence we have that in-domain pre-training, not
capacity, is what the coverage-map retrieval needs.

**Caveat for us:** the HF checkpoints are raw `transformers` feature-extraction
models with no trained pooling head, so we mean-pool. Mean-pooled BERT is
anisotropic (all cosines compressed into ~0.62-0.68), which is fine for ranking
and wrong for any tuned threshold — see the alias-merge warning in D-087.

---

# From the OneNote reading notes (folder `Dissertation/OneNote notes/`, processed 2026-09-05)

*Six papers with substantive notes that had no vault entry, plus three that had only a passing mention. Notes are Raul's own, taken Jul–Sep 2026; the "Why it matters here" lines are the transfer.*

## StructGPT (Jiang, Zhou, Dong, Ye, Zhao, Wen — EMNLP 2023, pp. 9237–9251)
**What**: An *Iterative Reading–Reasoning* (IRR) framework for LLMs over structured data, built on an invoking–linearisation–generation cycle. Splits the loop into two named functions: **reading** (collect relevant evidence through interfaces) and **reasoning** (infer the answer, or plan the next step).
**Why it matters here**: the earliest clean statement of the loop our planner runs. The read/reason split is the vocabulary our harness lacks — worth adopting in the methodology chapter, because it names why the critic sits where it does (it judges the *reading*, not the *reasoning*).
**Where discussed**: notes 2026-09-01; folded into related work 2026-09-05.

## KG-Agent (Jiang, Zhou, Zhao, Song, Zhu, Zhu, Wen — arXiv 2402.11163, 2024)
**What**: Autonomous agent over a KG with four components — instruction-tuned LLM, multifunctional toolbox, KG-based executor, knowledge memory. **Toolbox is the transferable part**: *extraction* (get_relation, get head/tail entities, entities by type or constraint), *logic* (count, intersection, union, condition verification, terminate-with-answer), *semantic* (relation retrieval by NN, entity disambiguation by NN). Training data synthesised by taking a known result, its SQL, and the reasoning path that reaches it, as ground truth. Explicitly motivated by removing the human-crafted plan and by making **small models** sufficient without a closed API.
**Why it matters here**: (1) their toolbox is a checklist against ours — we should confirm we expose `count`, `intersection` and `union` as *tools* rather than hoping the SQL arm reconstructs them; (2) the "small models, no closed API" motivation is ours exactly (D-012, self-hosted); (3) the reasoning-program synthesis recipe (result → SQL → path) is a ready-made way to generate our few-shot examples from the graph we already have.
**Where discussed**: notes 2026-08-31; related work 2026-09-05.

## Plan-on-Graph (Chen, Tong, Jin, Sun, Ye, Xiong — arXiv 2410.23875, 2024) — **closest architectural precedent**
**What**: Self-correcting adaptive planning over KGs. Decomposes the question into **sub-objectives each carrying a condition**; explores adaptively (relations first, then entities fulfilling them — not fixed breadth); maintains **memory** of subgraph + reasoning paths + per-sub-objective status; **reflects** to decide whether to keep exploring or backtrack.
Diagnoses three failures of prior work: **predefined path breadth** (fixed number of neighbours → silent loss), **irreversible exploration** (no backtracking), **forgetting partial conditions** (multi-condition questions answered against only some).
**Why it matters here**: the third failure **is our conjunction problem (D-058)**, arrived at independently — that is a citable convergence, not a coincidence, and belongs in the write-up. The first failure is why our exploration breadth cannot be capped. Their two-step relation-then-entity exploration is a concrete critique of our current tools: **are we unrolling neighbours along graph structure, or querying rows from nowhere?** If the latter, the critic is judging the wrong object — it should judge *path relevance*, not row relevance. Their prompt is in the notes.
**Difference to defend**: their benchmarks have one answer entity reachable by a path, so exploration may stop on finding it. Ours are set-valued, so pruning to find *an* answer is exactly how you return a silently incomplete set.
**Where discussed**: notes 2026-08-30; related work + positioning 2026-09-05.

## Wikontic — building KGs from text aligned with the Wikidata ontology
**What**: Combines **open** IE (no predefined entity/relation names) with the structural rigour of **closed** IE by leveraging an external ontology. Six components; the load-bearing ones are candidate triplet extraction, ontology-aware triplet refinement (hand the LLM the ontology, make it rewrite), subject/object refinement (embeddings + LLM), and storage/retrieval where **embeddings are used to decide which entities to look for at query time**.
Their framing of the problem: most KG approaches use the graph as an *auxiliary retrieval tool* rather than a high-quality knowledge resource, and synonymy plus redundant and inconsistent representations are what destroy the graph's advantages.
**Evaluation**: MINE (how much factual information the system retains); KG quality via structural compactness (non-redundancy, deduplication) and downstream multi-hop QA testing correctness *and* completeness. **Result worth stealing: qualifiers carry substantial information** — structured evidence, the same qualifiers we store.
**Why it matters here**: (1) the open-vs-closed framing is the cleanest statement of why our fixed schema plus discovery channel is the right shape; (2) "auxiliary tool vs knowledge resource" is the sentence our positioning argues; (3) their qualifier finding directly supports D-024's lossless-qualifier decision; (4) their multi-hop completeness test is a template — if anything is missing en route, multi-hop questions fail, which is a *usable* completeness probe for us.
**Raul's own ideas in the notes worth keeping**: LLM to find entity hierarchies; two-step extract-then-refine; get an ontology from INSPIRE-HEP; evaluate by artificially removing nodes; evidence + KG rather than full text + KG (checked against HippoRAG — *not* novel, they did it for evaluation only, but we could too).
**Where discussed**: notes 2026-07-28; related work 2026-09-05.

## MatKG — autonomously generated materials-science KG
**What**: Large KG built automatically from materials literature. Huge dataset, simpler target than ours, **deduplication is basic**.
**Why it matters here**: mostly a scale-and-contrast citation — same shape, different field, and its light treatment of deduplication is the gap our aliases layer occupies. **One idea taken**: use Levenshtein edit distance rather than n-grams with Jaccard (worth testing against the current Tier 1).
**Where discussed**: notes 2026-07-28.

## RAPTOR (Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning — ICLR 2024)
**What**: Recursive abstractive processing — builds a tree by summarising chunks, then summarising the summaries, and retrieves at whichever level of abstraction fits the query.
**Why it matters here**: it is about **indexes over text, not graphs**. Raul's note that doing the equivalent *on a graph* could be novel is worth pursuing carefully — it is close to Microsoft GraphRAG's community summaries, so the novelty claim must be made against that, not against RAPTOR. And the summarisation is lossy, which is the objection we raise to community-summary GraphRAG for counting questions: whatever we build, the underlying assertions must stay reachable.
**Where discussed**: notes 2026-07-30.

## Self-RAG (Asai et al., ICLR 2024)
**What**: Trains a model to decide when to retrieve and to critique the relevance and sufficiency of what it retrieved, via reflection tokens.
**Why it matters here**: Raul's note is the correct read — **they train, we orchestrate**. The open question he flags is whether prompting an orchestrated critic reproduces the behaviour without fine-tuning. That is testable with what we already have, and the answer belongs in the critic ablation.
**Where discussed**: notes 2026-07-31.

## Building effective agents (Anthropic engineering, 2024)
**What**: Practitioner taxonomy separating *workflows* (predefined code paths) from *agents* (model directs its own process). Workflows: prompt chaining, routing, parallelisation (sectioning and voting), orchestrator–workers, evaluator–optimiser. Agents for open-ended problems where the number of steps cannot be predicted. Appendix argues tool definitions deserve as much prompt engineering as the main prompt.
**Why it matters here**: gives standard names to what our arms already are — the typed planner is closest to **orchestrator–workers**, the critic loop is **evaluator–optimiser**, and repeats-with-unanimity is **voting**. Using those names makes the ablation legible to a reader. The tool-definition point is directly actionable: our free-SQL arm's failures may be tool-description failures rather than model failures.
**Where discussed**: notes 2026-08-02.

## GraphRAG survey — additional detail from the notes
**Beyond what was already recorded**: the **semantic parsing vs information-retrieval** distinction (generate a logical form and execute it, versus retrieve and let the model compose) — this is the correct technical name for what we do, now used in the background chapter. Retrieval granularity taxonomy (nodes, triplets, paths, subgraphs, hybrid). Iterative retrieval split into **non-adaptive** (fixed steps, threshold) and **adaptive** (model decides whether to continue) — we are adaptive, and the survey lists refs 22/92/94/100/203/213/229/248 as doing what we do. Indexing: graph, text and vector indexes as a multi-layered system; **indexing quotes** is called out, which we do. Evaluation limitations worth quoting: no standardised retrieval-faithfulness metric because reliable retrieval ground truth is hard, and **position bias in LLM-as-judge**, mitigated by reasoning models with explicit CoT.
**Future-work items that are ours**: scalable retrieval on real-world (not toy) KGs; hierarchical retrieval architectures for the grouping layer; and benchmarks that annotate **ground-truth retrieval elements, not only final answers** — which is precisely the half of our evaluation that does not yet exist.
**Where discussed**: notes 2026-08-02.
