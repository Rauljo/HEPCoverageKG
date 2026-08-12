"""
The supervisor's 15 evaluation questions, as a question set.

Source: HEPKG_promopt_tests `docs/onboarding/kg-evaluation-questions.md`
@ origin/kg-evaluation-questions, dated 2026-08-03. These are the TEST split.

Three things about them that decide how they are handled.

**His gold is graph-agreement gold, not physics truth.** His own header says so:
the answers were computed by filtering the pilot export, so they measure whether
we agree with his pipeline, not whether either of us agrees with the papers.
Treating them as truth would make the evaluation circular. They are recorded as
`gabriel_gold` in provenance and are NOT loaded as `truth` — the truth field is
filled by the reference reader (`eval/reader.py`) reading the papers.

**Scope is derived from the question TEXT, mechanically.** A question is about
one paper only if an arXiv id appears in the question itself. Reading scope off
the *gold* instead is circular in the worst way: Q8 asks "which analyses require
exactly 2 electrons OR exactly 2 muons", and its gold is "Exactly one:
2001.06899". Scoping the check to that one paper could only ever confirm the
answer — a second paper doing the same thing would be unfindable by
construction. `paper_scope()` below is a regex, not a judgement call, because
this exact error was made once already.

**Three of his Tier 4 premises are wrong.** Q12, Q13 and Q14 are labelled "not
in the graph", but the answers sit in stored evidence quotes:
    Q12  "Masses of the t2 up to 875 GeV are excluded at 95% CL ..."
    Q13  "... less than 40 GeV, which results in a signal efficiency of ~80%
          with the ttbar rejection factor of ~4.8"
    Q14  "These results improve upon the existing limits ... by ~300 GeV"
Scored as he intended, a correct answer would be marked wrong for failing to
abstain. They carry `premise_disputed` so both scorings can be reported.
"""
from __future__ import annotations

import re
from typing import Optional

# An arXiv id in the QUESTION text is what makes a question single-paper.
_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")


def paper_scope(text: str) -> Optional[list[str]]:
    """Papers a question names, or None meaning 'sweep the whole corpus'.

    Mechanical on purpose. The alternative — deciding scope by reading the gold
    answer — restricts the search to the papers the answer already names, so the
    check can only ever confirm what it was given.
    """
    found = _ARXIV_RE.findall(text)
    return sorted(set(found)) or None


# qid, tier, text, his gold (recorded, never trusted), notes
SUPERVISOR_QUESTIONS: list[dict] = [
    # --- Tier 1: facet-index lookups. Should need zero LLM calls. -----------
    {
        "qid": "gf-01", "per_paper": 'Is this analysis a SEARCH (rather than a measurement) whose event selection uses b-tagged jets AND missing transverse momentum?', "tier": 1, "shape": "set",
        "text": "Which searches select b-jets and missing transverse momentum?",
        "gabriel_gold": {"kind": "set", "value": 18, "papers": [
            "2004.14060", "2006.05880", "2010.14293", "2012.03799", "2012.08600",
            "2102.01444", "2106.01676", "2106.14246", "2201.11585", "2202.08676",
            "2211.08028", "2302.05225", "2307.01094", "2403.01556", "2506.13565",
            "2508.13900", "2510.07527", "2511.11853"]},
        "gold_note": ("his doc names 8 and then '...'. The remaining 10 are DERIVED, not "
                      "guessed: his gold is one filter over analysis_facets.jsonl "
                      "(objects superset {BJet,MET}, category=search), and we reproduce "
                      "all 60 cards exactly (D-052). The derived set is 18 -- his stated "
                      "count -- and contains all 8 he named."),
    },
    {
        "qid": "gf-02", "per_paper": 'Does this analysis estimate a background using an ABCD method, or an ABCD-style sideband or matrix method over independent regions?', "tier": 1, "shape": "set",
        "text": "Which analyses use an ABCD background estimate?",
        "gabriel_gold": {"kind": "set", "papers": [
            "2004.01678", "2011.07812", "2012.01581",
            "2404.06204", "2511.11853", "2604.27044"]},
    },
    {
        "qid": "gf-03", "per_paper": 'Does this analysis use the HistFitter framework for its statistical analysis?', "tier": 1, "shape": "set",
        "text": "Which SUSY searches did their statistics in HistFitter?",
        "gabriel_gold": {"kind": "set", "papers": [
            "2006.05880", "2011.07812", "2106.01676", "2211.08028"]},
    },
    {
        "qid": "gf-04", "per_paper": 'Does this analysis unfold its measured distributions -- that is, correct them back to particle level or truth level?', "tier": 1, "shape": "set",
        "text": "Which measurements unfold their distributions?",
        "gabriel_gold": {"kind": "set", "value": 10, "papers": [
            "2001.06899", "2110.11231", "2207.12246", "2208.12095", "2308.02285",
            "2309.14442", "2312.04450", "2401.05299", "2402.08486", "2404.06204"]},
        "gold_note": ("his doc says '10 papers incl.' and names 6. The other 4 are "
                      "DERIVED the same way (statistical_methods contains Unfolding, "
                      "category=measurement); the derived set is 10 and contains all 6 "
                      "he named."),
    },
    {
        "qid": "gf-05", "per_paper": 'Does this analysis RECONSTRUCT a Higgs-boson candidate as a physical object it selects on (not merely study Higgs production or decay)?', "tier": 1, "shape": "set",
        "text": "Which analyses reconstruct a Higgs-boson candidate as a detector object?",
        "gabriel_gold": {"kind": "set", "papers": ["2006.05880", "2504.13081"]},
        "gold_note": "his trap: string-matching 'Higgs' drowns in Higgs PROCESS papers",
    },

    # --- Tier 2: graph traversal -------------------------------------------
    {
        "qid": "gf-06", "tier": 2, "shape": "set",
        "text": ("Which regions of 2006.05880 require at least three leptons, "
                 "and what role does each play?"),
        "gabriel_gold": {"kind": "count", "value": 11},
        "gold_note": "4 SR^Z, 2 CR, 5 VR",
    },
    {
        "qid": "gf-07", "per_paper": 'Does this analysis have a ttZ background, and a control region used to normalise that background?', "tier": 2, "shape": "freeform",
        "text": ("In papers with a ttZ background, which control region normalises it, "
                 "and what quote defines that region?"),
        "gabriel_gold": {"kind": "none"},
        "gold_note": ("a SWEEP despite sitting in Tier 2; his gold gives one example "
                      "(2006.05880: CR^Z_ttZ, Table 6 caption), not the set"),
    },
    {
        "qid": "gf-08", "per_paper": 'Does this analysis require exactly two electrons OR exactly two muons as ALTERNATIVE selections -- that is, parallel ee and mumu channels, rather than requiring both?', "tier": 2, "shape": "set",
        "text": ("Which analyses require exactly 2 electrons OR exactly 2 muons "
                 "as alternatives?"),
        "gabriel_gold": {"kind": "set", "papers": ["2001.06899"]},
        "gold_note": ("THE ONE TO DOUBT. His gold is 'exactly one'. The rule producing "
                      "it needs a `subchannel` qualifier that exists in 5 papers of 60, "
                      "while 46 papers have a region requiring both an electron and a "
                      "muon. Parallel ee/mumu channels are standard; 1-of-60 is not "
                      "credible as a fact about the literature."),
    },
    {
        "qid": "gf-09", "tier": 2, "shape": "freeform",
        "text": ("Show the correction history of assertion A83 in 2001.06899: "
                 "who decided, when, what changed?"),
        "gabriel_gold": {"kind": "none"},
        "gold_note": ("UNANSWERABLE from any paper AND from our graph: 0 expert_decision "
                      "rows, 0 assertions with revision_of, in our DB and in his own "
                      "committed bundles. Needs him."),
        "blocked": "no provenance data in the corpus",
    },

    # --- Tier 3: evidence-text mining --------------------------------------
    {
        "qid": "gf-10", "tier": 3, "shape": "freeform",
        "text": "What signal efficiency does the pT^miss < 40 GeV cut retain in 2001.06899?",
        "gabriel_gold": {"kind": "value", "value": "approximately 80%"},
        "gold_note": "and ttbar rejection ~4.8; both in one stored quote",
    },
    {
        "qid": "gf-11", "tier": 3, "shape": "freeform",
        "text": ("Which b-tagging algorithm and working point does 2001.06899 use, "
                 "and with what performance?"),
        "gabriel_gold": {"kind": "value",
                         "value": "CSVv2 medium; ~10% (c) / ~60% (b) efficiency, 1% light mistag"},
        "gold_note": ("half unreachable: the performance IS in a quote, but 'CSVv2' "
                      "appears in 0 evidence quotes AND 0 source blocks of our harvest"),
    },

    # --- Tier 4: he says these are out of graph. Three are not. ------------
    {
        "qid": "gf-12", "tier": 4, "shape": "freeform",
        "text": "What is the observed 95% CL lower limit on the t2 mass in 2006.05880?",
        "gabriel_gold": {"kind": "none"},
        "gold_note": "he says: numerical results live in figures/tables",
        "premise_disputed": ('stored quote: "Masses of the $\\tilde{t}_{2}$ up to 875 GeV '
                             'are excluded at 95% CL for a chi^0_1 mass of about 350 GeV"'),
    },
    {
        "qid": "gf-13", "tier": 4, "shape": "freeform",
        "text": "Why did 2001.06899 choose 40 GeV for the MET cut?",
        "gabriel_gold": {"kind": "none"},
        "gold_note": "he says: optimisation rationale is prose with no ontology slot",
        "premise_disputed": ('the same quote as Q10 gives the rationale: "...less than '
                             '40 GeV, which results in a signal efficiency of ~80% with '
                             'the ttbar rejection factor of ~4.8"'),
    },
    {
        "qid": "gf-14", "tier": 4, "shape": "freeform",
        "text": ("How do 2006.05880's stop-mass limits compare to the previous "
                 "ATLAS result?"),
        "gabriel_gold": {"kind": "none"},
        "gold_note": "he says: needs the paper's comparison discussion",
        "premise_disputed": ('stored quote: "These results improve upon the existing '
                             'limits on the $\\tilde{t}_{1}$ mass in this model by '
                             'approximately 300 GeV [20]"'),
    },
    {
        "qid": "gf-15", "tier": 4, "shape": "freeform",
        "text": "What are the observed and expected yields in SR^Z_1A of 2006.05880?",
        "gabriel_gold": {"kind": "none"},
        "gold_note": ("premise HOLDS: we store the table caption ('Observed and expected "
                      "numbers of events in the 3l signal regions') and none of the numbers"),
    },
]

# `needs` per tier, for failure attribution (S-57).
_TIER_NEEDS = {
    1: ["sql"],
    2: ["hops"],
    3: ["hops"],
    4: ["not_in_graph"],
}


def build_records(split: str = "test") -> list[dict]:
    """The 15 as loader-ready records.

    `truth` is deliberately EMPTY. His gold goes to provenance; the truth field
    is filled later by the reference reader, which reads the papers rather than
    the graph. Loading his gold as truth would score agreement with his pipeline
    and call it accuracy.
    """
    records = []
    for q in SUPERVISOR_QUESTIONS:
        scope = paper_scope(q["text"])
        needs = list(_TIER_NEEDS[q["tier"]])
        if q["qid"] == "gf-08":
            needs.append("signatures")     # blocked upstream on M3 (S-57)
        records.append({
            "qid": q["qid"],
            "text": q["text"],
            "source": "gabriel",
            "split": split,
            "shape": q["shape"],
            "needs": needs,
            "difficulty": {1: "easy", 2: "medium", 3: "hard", 4: "hard"}[q["tier"]],
            "truth": {"kind": "none"},
            "truth_source": "none",
            # What the reader is actually asked, one paper at a time.
            #
            # Not a change of question: a sweep IS 60 instances of "does THIS
            # paper do X". But the phrasing decides the answer. Same window,
            # same model, only the wording differs:
            #   "Which SUSY searches did their statistics in HistFitter?"
            #      -> no, "the text does not mention SUSY searches in HistFitter"
            #   "Does this analysis use the HistFitter framework?"
            #      -> yes, "implemented in the HistFitter [168] framework"
            # The reader prompt already INSTRUCTS "does THIS analysis do it", and
            # the model anchored on the question's phrasing anyway. Instructing
            # around a corpus-wide question does not work; decomposing it does.
            # Hand-written and recorded beside the original so both are auditable.
            "per_paper": q.get("per_paper"),
            "provenance": {
                "tier": q["tier"],
                "question_as_asked": q["text"],
                "paper_scope": scope,                 # None => sweep all 60
                "scope_rule": "arXiv id present in the question text",
                "gabriel_gold": q.get("gabriel_gold"),
                "gold_note": q.get("gold_note"),
                "gold_is": "graph-agreement, not physics truth (his header says so)",
                "premise_disputed": q.get("premise_disputed"),
                "blocked": q.get("blocked"),
                "source_doc": ("HEPKG_promopt_tests docs/onboarding/"
                               "kg-evaluation-questions.md @ kg-evaluation-questions"),
            },
        })
    return records


def sweep_questions() -> list[dict]:
    """The ones that must be checked against all 60 papers."""
    return [r for r in build_records() if r["provenance"]["paper_scope"] is None]


def single_paper_questions() -> list[dict]:
    """The ones that name their paper(s) in the question text."""
    return [r for r in build_records() if r["provenance"]["paper_scope"] is not None]
