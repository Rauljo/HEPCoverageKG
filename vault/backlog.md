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

- [ ] **Aliases layer — Tiers 2–3.** Tier 1 (normalize) done + draft list produced. Next: n-grams
      + exact number/version guard (proposed); embeddings (bge-small, candidate-gen only); LLM
      adjudication grounded in evidence + human confirm. [[open-vocab-reconciliation]] (D-025).
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
- [ ] **★ Raul: READ the draft alias list** (`data/processed/draft-aliases.md` — 245 clusters /
      559 ids) → then `aliases confirm` (materialize `entity_canonical`) and decide if Tier 1 is
      auto-trustworthy going forward.
- [ ] **Compile final-state signatures from the `count`/`subchannel` qualifiers** — the signature
      field is empty, but the ingredients aren't. Blocks the M3 physics query. [[final-state-representation]].
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
- [ ] **Held-out validation** — train/test split for gaps; matrix-completion framing;
      learning curve. [[held-out-gap-validation]].
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

## Milestones (the contract's four)

- [x] **M1 — Load & reconcile** — importer, 60 bundles → 14,188/11,309/2,555/324. Done.
- [~] **M2 — Trace** — `trace_assertion` built (contract condition 5 passes); polish later.
- [ ] **M3 — Accepted view + a real physics query** — "2 electrons + MET>200", OR-logic. Needs the
      final-state-from-qualifiers compile + status filter.
- [ ] **M4 — Re-import after review** — the apply-updates reconciler (status promotions + correction
      chains + `assertion_status_history` writes); guard + table exist, reconciler deferred (step 6).
      Needs a reviewed bundle (pilot has 0 expert_decisions) — ask Gabriel, or test on `corrected_bundle`.
