# Backlog — pending, actionable

*The to-do list, by theme. The "why/design" lives in `ideas/`; this is the "what next".
Check off or move to a `D-nnn` decision as things get built. Newest concerns near the top of
each theme.*

Themes: **Extraction & retrieval** · **Graph & canonicalization** · **Agents & reasoning**.

---

## Extraction & retrieval

- [ ] **★ ASK GABRIEL for the source HTML corpus (the paper dataset).** Unblocks the most: the
      cellular-vs-family comparison, the baseline-extraction study ("my RAG vs his bundles"), and
      any local prototyping. The pilot paper *list* is in-repo, but the HTML lives on his machine.
- [ ] **Cellular vs family-grouped extraction — an ablation.** Gabriel asks 7 family prompts
      (each covering several predicates); Raul's cellular RAG asks per-predicate. Measure quality +
      failure behaviour. Hypothesis: cellular reduces correlated failures (a fragile family prompt
      currently kills all its predicates — the retry imbalance) and sharpens retrieval, at the cost
      of coherence + more calls. → this is a core baseline-vs-treatment experiment. See
      [[multi-agent-extension]].
- [ ] **Log out-of-family predicates instead of discarding them.** Verified in code: a family
      payload with *any* out-of-family predicate is **hard-rejected** (`raise`), which triggers a
      **retry** and *loses* the observation. So (a) it's a cause of the retry imbalance, and (b)
      real signal is thrown away. Log them (leftovers-pass pattern) → recurrence shows where the
      family boundaries leak. Suggestion for Gabriel. See [[leftovers-pass]].
- [ ] **Entity-context naming** — feed existing entities so the model reuses ids (`NEW:` hatch +
      retrieval of K candidates). Upstream fix for the id duplication. [[entity-context-naming]].
- [ ] **Guided / constrained decoding ablation** — the shared fix for empty signatures + typing
      quarantines + kind-flips; measure vs free-parse. [[pydantic-validation]].
- [ ] **Feed `entity.aliases` into the aliases layer** as a Tier-2 nomination signal — but only
      nominate (never auto-merge); weight by cross-paper recurrence / evidence grounding, since the
      model can invent aliases. (Graph-adjacent.)
- [ ] Grounding & evaluation actions (Rivet/SimpleAnalysis machine labels, n=3 runs) —
      [[grounding-and-evaluation]]. Consistency-check pass [[consistency-check-pass]] (D-018).
- [ ] Retrieval model: only the ~38 letter-style fallback papers justify a hybrid retriever (A/B,
      not wholesale) — from the acquisition-repo read (`literature.md`).

## Graph & canonicalization

- [ ] **★★ FROM GABRIEL'S PR #1 (2026-07-27)** — full analysis in
      `reports/2026-07-27-hepkg-pr1-analysis.md`. No schema break (bundles still v0.2; `canonical`/
      `facets` are export-only). Three actions:
    1. **Correct importer entity identity → (bundle_id, entity_id)**, never bare entity_id (the
       contract now says cross-paper id-equality is *accidental collision*, not entity resolution;
       `hepkg:object:muon` spans 42 papers). Revisits **D-024/D-029** — `entity_occurrence` already
       has the right grain; demote the merged `entity` node from identity to convenience.
    2. **Reposition the aliases layer as the `resolves_to` residual layer**: consume Gabriel's
       `canonicalize_entity` / `facet_tags` (objects→`BJet`, generators→family+version) as a strong
       prior; own the null tail + cross-paper node resolution. Our Tier-1 dot/spelling work is now
       partly redundant *for detector-objects/generators*, still needed elsewhere.
    3. **Consume the derived data**: port `vocabulary.py` (`objects-v2`/`generators-v1`/`facets-v1`)
       over imported entities, so we get `canonical` + `facets` + the per-paper `analysis_facets.jsonl`
       card (a ready coverage backbone). Alt: import from the compiled export.
    - Ask Gabriel: is the vocab frozen, and will future bundles carry `canonical`/`facets` *in the
      bundle* (schema bump → update our gate) or stay export-only? Also add the `column` qualifier to
      the final-state compile.
- [ ] **Aliases layer — Tiers 2–3.** Tier 1 (normalize) done + draft list produced. Next: n-grams
      + exact number/version guard (proposed); embeddings (bge-small, candidate-gen only); LLM
      adjudication grounded in evidence + human confirm. [[open-vocab-reconciliation]] (D-025).
      *(LLM tier note: DIAS GPUs are Raul's alone — spinning up vLLM is just "ssh in + run", no
      queue/contention to worry about.)*
    - Tier 1.5 DONE 2026-07-26: deterministic US/UK spelling + data-driven plural (`aliases/spelling.py`).
    - Tier-1 refinement done 2026-07-26: strip the `.` version dot too (`pythia8.210`=`pythia8-210`).
    - **Tier-2 rule TODO**: the `p`-as-decimal convention in energies — `2p76tev`=2.76 TeV,
      `5p02tev`=5.02 TeV — a `p` between digits means a decimal; not handled by Tier 1.
- [ ] **Canonical standardization** (distinct from clustering — Raul, 2026-07-26). The canonical is
      currently "most-papers existing id", so canonicals are inconsistently spelled *across* clusters
      (`pythia-8.186` vs `pythia8.210` vs `pythia-8-212`). Add a per-kind step that synthesizes a
      *consistent* canonical form — best derived from the clean **label** ("Pythia 8.186"), since the
      normalized slug has lost separator positions (`pythia8210` is ambiguous). Non-destructive
      (originals kept; this sets the display/resolution id). **Bonus**: standardized canonicals
      unlock a generator→version **hierarchy** (all `pythia8.*` = one "Pythia 8" family) for
      coarser coverage queries — a *link*, not a merge (versions stay distinct).
- [ ] **★ MAÑANA: review the NEW Tier-1.5 spelling/plural bridges** (19 of them, `method='spelling'`,
      status `proposed`) in the regenerated `data/processed/draft-aliases.md` (255 clusters / 597 ids)
      → then `aliases confirm` to promote them. Tier 1 (328 normalize edges) is already `auto`.
- [ ] **Literature: scientific-concept deduplication / entity resolution** (for the aliases layer,
      Raul 2026-07-26). Search proper (Elicit-style, house-style `literature.md` entries). Likely
      landmark strands: record linkage / entity resolution (Fellegi–Sunter, blocking, `dedupe`);
      entity *linking/normalization* (BLINK; **SapBERT** self-aligned biomedical concept linking;
      UMLS/MetaMap normalization); ontology/entity **alignment & matching** (OAEI, Silk/LIMES,
      Wikidata/DBpedia alignment). Goal: a principled, ideally cheaper, method for Tiers 2–3 —
      esp. how others tell true synonyms from dangerous look-alikes (our w/z, s/t, version trap).
      **Two consumers, one search** (2026-07-27): the aliases layer *and* the held-out gap-matcher
      ("is this fill the same gap?") — see [[held-out-gap-validation]]. SapBERT-style cross-encoders
      are the strand GAPMAP used (RoBERTa) for its implicit-gap validation.
- [ ] **Build final-state signatures by LLM-parsing the label prose** (D-038, 2026-07-29 — replaces
      "compile from the `count`/`subchannel` qualifiers", which was based on a wrong reading of the
      data: **no `count` qualifier exists**, 0 of 161). The signature field is empty *and* so are the
      structured ingredients; the information lives only in the English label. Blocks the M3 physics
      query, which now blocks the evaluation. [[final-state-representation]].
- [ ] **Neo4j projection** from SQLite (D-022) — for traversal + visualization.
- [ ] **Query layer** — simple filters → multi-hop physics signatures (M2 done as trace; M3 = the
      physics query).
- [ ] **Coherence / review-router** — on ingestion, surface contradictions + typing-flips for a
      human; *raises attention, never judges* (novelty is the product). Shares the gap-finder
      trigger.
- [ ] `result_type` vocabulary — needs supervisor; gates what counts toward a gap. [[result-type-vocabulary]].
- [ ] **Report to Gabriel** (all one root cause — *schema is validated, not used to construct*):
      vocabulary flags labels but doesn't canonicalize ids (b_jet split); the id *namespace* is
      ungoverned (34 for ~22 kinds; `cs`/`collision_system`); derive the id from the controlled
      `kind` + normalized slug. Also: single-vs-multi-value cardinality isn't modelled (below).

## Agents & reasoning (the payoff)

- [ ] **★ Gap-finding literature review** — build the related-work chapter for the payoff arm
      (gap finder + held-out validation + theory-side). An Elicit paper list already exists (Raul
      has it); **start from scratch when doing this** — re-run the search, then triage → integrate.
      Cover these three method themes (kept HEP-free so they don't collapse onto Contur/Rivet/MUSiC,
      which are already in `literature.md`): **(1)** finding/ranking gaps from a KG — link
      prediction / matrix-completion / LLM reasoning to tell plausible-but-unobserved from trivially
      absent; **(2)** the open-world / absence problem — "not in corpus" ≠ "doesn't exist", negative
      evidence, PU learning, asymmetric retrieval reliability; **(3)** LLM/agentic scientific
      hypothesis & gap generation grounded in a graph/DB (not free-generated), incl. hallucination
      constraints. Also sweep literature-based discovery (Swanson / undiscovered public knowledge)
      and evidence-gap-map / systematic-map methodology. Output: `literature.md` entries in house
      style (what / why-it-matters-HERE / where-discussed), each routed to
      [[gap-hypothesis-system]] / [[held-out-gap-validation]] / [[grounding-and-evaluation]] /
      [[researcher-feedback-loop]]; flag any that challenge a current decision. Triage before
      writing anything into the vault.
    - **Weight agent systems as first-class** (Raul, 2026-07-26): the payoff arm's agentic
      contribution lives at Stage-B tool-using verification (gap-hypothesis constraint 2) and the
      multi-agent theory×experiment vision — so the from-scratch run must surface *tool-using /
      self-verifying / multi-agent* systems, not just static link-prediction/embedding methods.
      **Reading progress**: ✅ Swanson 1986 (LQ theory paper) and ✅ GAPMAP read 2026-07-27 — both
      `literature.md` entries rewritten from the papers. Next in order: ResearchLink (Borrego 2025).
      Core reads to date (agent systems in **bold**): Swanson 1986 · GAPMAP (2510.25055) ·
      ResearchLink (Borrego 2025) · fact-discovery-from-KGE (Bhagaskoro 2024) · KG-CoI (2411.02382) ·
      **GeneAgent (2405.16205 — self-verification agent over domain DBs = the Stage-B verifier)** ·
      AGATHA (2002.05635, temporal holdout) · open-world KG eval (2209.08858) · nPUGraph (2306.07512,
      PU learning for absence) · **SciAgents (Ghafarollahi 2024 — multi-agent graph reasoning =
      theory-side vision)**. Support: GeneAgent's cousins BioKGBench (2407.00466, agent KG-checking),
      SKiM-GPT, Dyport, Daowd 2022, LinkExplorer. Note: nearly all prior art is biomedical LBD; the
      open niche is gaps-as-*coverage-grid* (final-state×experiment×energy) + theory-vs-measurement.
- [ ] **Gap finder** — deterministic enumerate → LLM reason (quality) → InspireHEP verify;
      never invents gaps. [[gap-hypothesis-system]].
- [ ] **Beyond final-state gaps** — subject/literature-pull; theory-side graph
      (theory-predicted-vs-measured); scoped-theory fallback. (In [[gap-hypothesis-system]].)
- [ ] **Mine "Future Work" / "Outlook" sections for author-declared gaps** (Raul 2026-07-27, from
      GAPMAP). A new *prior*/ranking signal over structurally-enumerated cells — never the enumerator
      (constraint 1). Log un-enumerated declared gaps separately: they indicate a **missing coverage
      axis**, leftovers-log style. Cheap — a new prompt over the existing section routing. Big
      by-product: declared-then-later-filled = a **timestamped labelled gap set** (see below).
      [[gap-hypothesis-system]] § Author-declared gaps.
- [ ] **Held-out validation** — train/test split for gaps; matrix-completion framing;
      learning curve. [[held-out-gap-validation]].
    - **Gap-matching ("is this fill the same gap?")** — exact string match under-counts fills and the
      bias flatters the finder. **Reuse aliases Tiers 2–3**: BGE blocking → pairwise adjudication
      (cross-encoder à la RoBERTa/SapBERT *vs* LLM prompt, report agreement) → human gate on
      disagreements. Precision-first here (a false fill *erases* a real gap); consider exact-match on
      experiment/√s, fuzzy only on the final-state label. Shares the dedup literature search above.
    - **Temporal holdout** (AGATHA-style) using declared-future-work→later-fill pairs as labels;
      run *alongside* the random split, not instead.
- [ ] **Researcher feedback loop** — per-paper summaries → authors correct → cluster → human-applied
      fixes (never LLM self-editing). [[researcher-feedback-loop]].
- [ ] **Single-vs-multi-value conflicts.** Two same-predicate assertions can conflict (two
      collision energies) or coexist (two targeted processes); Gabriel's schema doesn't encode
      cardinality, so it doesn't catch within-pass conflicts (only cross-pass, via adjudication →
      "contested"). Raul's D-010 `multi_value` flag + D-018 consistency pass handle it — a place the
      design is more principled; use as a graph-level consistency check.
- [ ] Critic panel (fidelity / physics-coherence / corpus-consistency) — mostly **Sunny's** thread;
      coordinate at the `status` seam. [[multi-agent-extension]].

---

## Evaluation (designed 2026-08-01 — `system.md` §4, S-32 … S-52)

*Ordered by value-per-day. **Everything in the first block needs no human and no judge** — the graph
is exact ground truth for it (S-32). The thing to resist is starting with Gabriel.*

- [x] **Harness** (S-22) — `hepcoveragekg/eval/`, 335 tests. Runs any `System`; run identity, test-set lock, variance-first, `eval score` rescoring.
- [~] **Backwards question generation** (S-33) — v1 and v2 both produced unusable truth (1/36, then 0/36). **Rebuild around S-58 invariance + S-59 tiers.**
  - [ ] **Tier A (per-paper)** — the backbone, ~540 questions, dedup-independent. Pre-check the 22 intra-paper duplicates (D-046).
  - [ ] **Tier B/C** — invariance-tested concepts (242 well-posed) and ordinal questions.
  - [ ] **Tier D/E** — safe-unique anchors and anchor+hop. **Re-measure safe-unique with the alias proposals folded into the broad reading first** — 3,733 is an upper bound.
  - [ ] **Tier F** — intersections; measure the yield before committing.
- [ ] **Per-question item analysis** (S-62) — group by `qid` across runs/systems; a question everything fails is a suspect question. Report mode only, no re-runs.
- [ ] **Fold `aliases_proposed.json` into the broad reading** (S-60) — filter, never merges.
- [ ] **Two cheap baselines** (S-44) — the 70B with no corpus (1 h), and plain RAG over the same 60 papers (½ day, reuses bge + existing BM25). A number on day one.
- [ ] **Plan type-checking against the schema card** (S-35) — guard *and* metric; the 2026-07-31 wrong-direction failure is **confirmed still live**, 3/3 in the seed run.
- [x] **`contents_of`** (D-047) — the paper -> contents inverse hop. Built, tested, wired as a planner tool.
- [ ] **Count inside `contents_of`** — counting is the weak spot (0.50). Diagnosed: for *result-subject* predicates the planner does not reach for `contents_of` at all (7 of 16 failures), and where it does it tallies rows **by eye** and lands one or two out. Arithmetic belongs in SQL, which is why `count` exists.
- [ ] **★★ IMPORT the facet + signature layers** (`origin/kg-evaluation-questions`:`pilot/analysis_facets.jsonl`, 60 rows/49 KB; `hepkg_acquisition/signatures.py`). Blocks 9 of the supervisor's 15 questions AND M3. The long-open "vocabulary port".
- [ ] **`facets(filter)` tool** — set operations over the closed enums; the cheapest exactness win available.
- [~] **Critic that flags before it filters** (S-69) — **specified in D-060**: three rungs, ranked+chunked, both controls, and breadth as a tail-relevance stopping rule. Validated symmetrically: his Tier 1 catches under-use of facets, our Tier B catches over-use. *Blocked on the re-baseline below.*
- [ ] **Re-baseline A / B / retrieval on today's code** — the existing runs are all `6945b02-dirty` and predate `7ad9cd6` (the facet layer + both facet tools). Nothing built after them can be compared against them.
- [ ] **Sufficiency critic** (D-060) — "is this evidence *enough*", beside `verify.py` and never inside it. Catches answering from 3 rows of 60, reasoning over a truncated `_render_rows` result as if complete, and cheap `not_in_graph` after one phrasing. After the candidate critic.
- [ ] **The planner searches for arXiv ids** — 206 of 218 Tier A searches returned zero rows, every one of them a paper id passed to `search` (`search("2604.27044")`), which `contents_of` explicitly forbids. Independent of the critic; cheap.
- [ ] **Run the supervisor's 15 questions** — the first externally-authored measurement.
- [ ] **Label the 111 review pairs** — then the adjudicator ablation: current prompt vs `explanation` moved BEFORE the verdict (free CoT) vs a reasoning model, all on the same pairs.
- [ ] **Feed `entity.aliases` into candidate generation** — the funnel narrows at 1.3% of pairs considered, not at adjudication. `MET` / `E_T^miss` are sitting in the data unused.
- [ ] **Fix `list_papers`** (D-043) — it disagrees with `count` (59 vs 58) because `papers_of` is untied to the assertion's bundle. Re-run affected numbers after.
- [ ] **Measure alias precision properly** (D-044) — a random labelled sample, not 14 eyeballed. Then decide the human gate.
- [ ] **Metamorphic checks** (S-34) — paraphrase invariance, monotonicity, inclusion–exclusion, order invariance. Violations are guaranteed bugs, no labels.
- [ ] **Stability / repeat sampling** (S-52).
- [~] **Growth curve, subsample version** (S-47 B) — **entity half done (D-061): no saturation, ~86 new entities/paper flat from 10 to 60, because only 1–22% of entities appear in more than one paper.** Extrapolates to ~255k entities at 2,969 papers. Still open: the *answer-quality* half — does accuracy change with corpus size?
- [ ] **Hops-to-node** (S-48) — Recall@k + MRR, stratified by rare/common, many-spellings/one, **merged/singleton** (tests whether dedup helps or hurts retrieval).
- [ ] **Tool-selection + tool-necessity tests** (S-49) — remove a tool, see if it routes around; unused tools are a finding.
- [ ] **Ablations** (S-36) — dedup, BM25/dense/RRF, guards, `SEARCH_BREADTH` 6/20/60, `PURPOSE` full vs minimal. Settles the open aliases and breadth questions **with evidence**.
- [ ] **`missing:` field ablation** (S-50) — with/without, measure rounds-per-question and accuracy.

*Then, with a dependency:*

- [ ] **Reference reader** (S-37–S-39) — one pass, per paper, constrained to our schema, on a **third** model (not Sonnet, not Qwen). 3× for self-agreement; verbatim quote required per claim. Then **graph vs graph** at three strictness levels — and **measure the matcher itself** (50 matches + 50 non-matches by hand).
- [ ] **GraphRAG** (S-44 arm 3) — as shipped with physics-tuned `entity_types`; the "was curation worth it?" arm. Budget: indexing is slow.
- [ ] **Gabriel, once, <2 hours total** (S-40) — ~50 blind pairwise (judge κ) · ~40 graph-vs-graph disagreements · ~20 quote adjudications · ~10 checklist sanity checks · ~20 naturalness spot-checks.
- [ ] **chATLAS_Benchmark scoping** (S-45) — 1 h: published analyses or internal docs? retrieval or answers? Decides whether it is usable.
- [ ] **Growth curve, version A** (S-47 A) — 60 → 300 → 1,000, re-extracted **on our own vLLM** (D-041). Bounded by cluster time, not by Gabriel.
- [ ] **Deletion protocol** (S-13) — select by *fact specificity*, not paper membership.
- [ ] **Deep-research snapshot** (S-12) and **frontier-model-with-RAG arm** (S-44 arm 6, separates "our system is good" from "our model is good").

*Ablation axes — the list the study sweeps (S-36). Nothing here is built before the harness, or the
thing being measured moves while it is measured:*

- [ ] dedup on/off · BM25 / dense / RRF · guards on/off · `SEARCH_BREADTH` 6/20/60 · `PURPOSE` full vs minimal
- [ ] **question decomposition → set algebra** (S-53) — expect gains only on multi-constraint questions
- [ ] **paraphrase fusion of retrievals via RRF** (S-54) — expect gains in retrieval recall; build once, it also serves S-34 paraphrase-invariance
- [ ] **agreement-as-confidence** shown in the answer (S-54) — a product feature, measured as calibration
- [ ] `missing:` field (S-50)

*Blocking checks, cheap, do early:*

- [ ] **Step 0 — one message to Gabriel** (today): ~30–50 questions, no answers, hard ones marked · is `GROUND_TRUTH.md` current? · chATLAS access + benchmark scope. **Longest-latency item in the project.**
- [ ] **D-038 signature parse into a derived table** (S-56) — one afternoon; prose reads but cannot be counted (126 distinct labels of 138), and query-time bucketing would put the count outside `verify.py`'s reach.
- [ ] **Tag Gabriel's questions against M3 the moment they arrive** (S-57) — decides whether S-56 is urgent or merely useful.

- [ ] **`GROUND_TRUTH.md` field names** vs current pipeline output (~30 min) — it is stale since 2026-07-05 (D-042). Ask Gabriel if he considers it current.
- [ ] **`html_harvester.py` against the DIAS corpus** — built for arXiv pages; this is a different layout.
- [ ] **Who writes the checklists** (S-42) — deliberately open, needs the frozen question set first.

---

## Milestones (the contract's four)

- [x] **M1 — Load & reconcile** — importer, 60 bundles → 14,188/11,309/2,555/324. Done.
- [~] **M2 — Trace** — `trace_assertion` built (contract condition 5 passes); polish later.
- [ ] **M3 — Accepted view + a real physics query** — "2 electrons + MET>200", OR-logic. Status
      filter (cheap) + final-state signatures (the real work). **Now on the critical path** — the
      aggregate coverage questions the evaluation rests on are unanswerable without it.
      **Corrected 2026-07-29 (D-038)**: there is no "compile from `count` qualifiers" — no such
      qualifier exists (0 of 161), and the signature is not fanned out across object edges, it is
      **one node holding English prose**. So: LLM parses the prose into `{object, count, comparator}`,
      code serialises that into the canonical id. 161 items, all 60 papers, **126 distinct labels
      out of 138**. See [[final-state-representation]].
- [ ] **M4 — Re-import after review** — the apply-updates reconciler (status promotions + correction
      chains + `assertion_status_history` writes); guard + table exist, reconciler deferred (step 6).
      **PR #1 update**: real review decisions now exist (2001.06899 accepted; 2006.05880 = 287
      decisions staged in `runs/.../decisions.json`) — once a *reviewed* bundle is compiled, M4 has a
      real target beyond the `corrected_bundle` fixture.
  - [ ] **M4 diff → failure-mode learning (2026-07-29, user's idea — do not lose).** The reconciler
        already has to compute *what changed* between the extracted and the reviewed bundle. Keep
        that diff as data, not as a transient step: it is **free labelled extraction error**,
        corrected by physicists. Two uses: (a) derive a taxonomy of real failure modes instead of
        guessed ones; (b) use each observed mode as a **detector** — sweep the corpus for other
        assertions with the same shape and flag them, so one human review generalises.
        **Economy**: this is the same machinery as the new system's user-editing stage ("what led to
        this being wrong, and what else is wrong for the same reason"). Sourced from Gabriel's
        review here, from a UI edit there. Build once, at M4.
        **Caveat**: 1 accepted paper + 287 decisions is enough to build and demonstrate the
        mechanism, **not** enough to claim a validated failure taxonomy. Say so in the write-up.
