# Grounding & evaluation (ground truth, authority, methodology)

**Status**: discussed (2026-07-15, from the AgentRivet close-reading). Several week-0/1
actions; partially revisits D-008's framing (see caveat). Not implemented.

## Why this is a theme now

We have **no external check on a coverage label** — no compiler, no downstream artefact that
can disagree with the extractor (AgentRivet had a Coder + compile step; we have nothing). The
only source of truth external to the LLM is physics preservation data. So grounding isn't
"light metadata" — where it exists, it's the only thing that can catch a wrong extraction.

## Ground-truth sources (distinct from D-008's descoped "HEPData as data source")

**Important scope distinction**: D-008 descoped HEPData *table extraction as an input data
source* (pulling numeric results into the graph). This is a **different use** — the same
artefacts as **evaluation ground truth** and as authority for a few fields. Not a conflict,
but flag it explicitly so the two uses don't get confused.

- **Measurements** → **Rivet**. `ANALYSIS.info` (YAML: Beams, Energies, Luminosity_fb,
  InspireID, Keywords, Description) gives 3–4 schema fields as *validated structured data,
  free, no LLM*. `ANALYSIS.cc` is the selection logic, unambiguous. Caveats: (a) the .cc
  encodes multiplicities in imperative control flow — running an LLM over it is "extraction
  with a better source," not a bypass; (b) **Rivet is particle-level only** (dressed leptons,
  truth-flavour b-jets, no trigger/ID) while MUSiC classes are *reconstructed*-level — usually
  1:1 for counting but NOT always; flag efficiency-driven fiducial-vs-reco multiplicity
  differences. This particle-vs-reco gap is *why searches aren't in Rivet*.
- **Searches** → **SimpleAnalysis / pyhf / HEPData cutflows**. ATLAS SimpleAnalysis is C++
  encoding **reconstructed-level** signal-region definitions — object multiplicities at exactly
  the level MUSiC classes live at, no particle-level mismatch. This is the search-side ground
  truth that makes evaluation credible.

**Honest use = evaluation, not extraction.** Where routines exist we machine-label instead of
hand-label → much bigger validation set, cheap. But **write down the skew**: the Rivet eval set
is measurements only, and only the ~39% that were well-preserved (clean subset) → it's an
**upper bound on performance, not an estimate of it**.

## Our test paper

arXiv 2307.01094 = ATLAS **SUSY-2020-27** (identified 2026-07-15). Signal regions + one WZ+jets
control region, R-parity conserving & violating interpretations, 139 fb⁻¹. Vintage strongly
suggests full pyhf likelihoods on HEPData; SimpleAnalysis plausible.
> **ACTION (wk 0–1)**: verify pyhf/SimpleAnalysis availability directly on the ATLAS
> SUSY-2020-27 page / its HEPData record (both JS-rendered — browser check). If present, it may
> replace hand-labelling with machine ground truth → changes the whole eval design. Revisit the
> "SimpleAnalysis = light metadata" assumption → possibly load-bearing.

## Grounding as authority (blocker), and log conflicts as first-class

Per the blocker/advisory principle (see [multi-agent-extension.md](multi-agent-extension.md)):
grounded external sources are **blockers** that overwrite the extractor; LLM critics are only
advisory. BUT the grounding step must **surface disagreements, not silently prefer one side** —
AgentRivet found a CMS paper disagreeing with its *own* HEPData record (normalisation
mismatch). Extraction-vs-record conflict may mean our extractor is wrong OR that paper and
record disagree; both are worth knowing. Log the conflict as a first-class object.

## Evaluation methodology (from AgentRivet's empirical findings)

- **n=3 runs per model per paper** on the starter cases (LLMs are non-deterministic;
  AgentRivet caught errors appearing in 1 run of 3, invisible to a single run). **The check
  that matters**: if a field returns **confidence 0.9 in all 3 runs but 3 different values**,
  confidence is measuring *fluency, not correctness*. We already OBSERVED uncalibrated
  confidence (~everything 1.00, overview model-quirks) — this is how to measure it rigorously.
- **Don't hand-correct fields before computing accuracy** — else the number is the accuracy of
  the human, not the system (AgentRivet fixes compile errors and counts them but never fixes
  physics errors, keeping the two failure modes from contaminating each other's measurement).
- **Distinguish retrieval-uncertainty from source-underdetermination in the quarantine queue**.
  A low confidence means either "I'm unsure I retrieved the right passage" (our problem,
  fixable) or "the paper genuinely doesn't specify this" (not our problem, not fixable). These
  currently look identical. `EXTRACTION_SUPPORT` (absent/unclear) partly captures it — leverage
  or extend it so the queue is interpretable.
- **Dominant failure is source ambiguity, not model incapacity** — a better model can't fix
  what the paper doesn't say; this bounds achievable accuracy at how clearly papers are
  written. And **silent omission** (a well-formed record that simply left something out) is the
  empirically dominant, hardest-to-see failure → the `unmapped_object_requirements` flag
  (open-vocab doc) is the concrete mitigation for the coverage-critical version of it.

## Our eval is positioned to be stronger than AgentRivet's (fair to state)

AgentRivet: prompts tuned on 2 papers, evaluated on 2 others; no cross-check against official
Rivet routines (expert eyeballing, n=2); grounding "ablation" is n=1/arm, unnamed. Our plan
(hand-labelled held-out set + external check against published manual gap/coverage surveys +
a proper with/without-grounding ablation) answers the same bind better. Say so without
disparagement.
