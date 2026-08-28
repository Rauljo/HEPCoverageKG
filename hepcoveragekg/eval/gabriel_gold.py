"""Gabriel's per-paper verdicts, turned into query-layer questions with human gold.

WHY THIS EXISTS.

His 199 verdicts are the only ground truth in this project written by someone who
knows the physics. Every other gold set is derived from the graph by SQL, which
means it can only ever confirm that the query layer reads back what extraction
put in. A question set scored against a physicist measures something different
and better.

But the verdicts arrived in the wrong shape. They are per-(question, paper)
yes/no rows produced by the READER -- one paper at a time. The knobs worth
ablating (the critic, ranked vs shuffled candidates, the 8B judge against the
72B) all live in the QUERY layer, which never ran on them. So running the
ablation over these rows as they stand would produce eight byte-identical arms.

The conversion: collect the papers he marked yes for one question, and that set
becomes the gold answer to a set-valued query-layer question -- "which analyses
use an ABCD-style background estimate?" Now the knobs bite.

THE GOLD IS PARTIAL AND THE SHAPE OF THE HOLE MATTERS.

He was only ever shown papers our system had surfaced. So a paper with no
verdict is not a negative, it is unknown, and the papers most likely to be
unknown are exactly the ones our retrieval missed. That biases recall upward and
leaves precision clean. `Truth.universe` carries the judged papers so the scorer
can restrict to them, and `judged_coverage` reports how much of the corpus that
is -- see `scoring.judged_set_f1`.

WHAT IS DROPPED, AND WHY.

  `unsure` verdicts   -- 14 rows. Four of them are Gabriel telling us the
      evidence is insufficient though the paper is probably true (D-066), which
      is a different axis from yes/no. Scoring them either way invents a verdict
      he declined to give.

  open questions      -- gf-06, gf-10..gf-15 ask for a value or a list, not a
      set of papers, and carry one row each. Not enough to score.

  golds above MAX_LISTABLE -- no prose answer lists 20 papers, so a large gold
      would measure brevity. The scorer abstains; this refuses to emit them at
      all so the abstention is visible here rather than as a silent gap later.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

# Where the returned sheet and the sheet it answers live.
VERDICTS = Path("eval/review/returned/gabriel-batch1-full-2026-08-19.tsv")
SHEET = Path("eval/review/gabriel-review.tsv")

# The most papers a gold may hold before the set scorer abstains. Kept in step
# with scoring.MAX_LISTABLE, imported rather than repeated.
from .scoring import MAX_LISTABLE  # noqa: E402

#: Per-paper reader questions rewritten as corpus-wide query-layer questions.
#: The physics must not drift -- these are his questions, reworded from "does
#: THIS paper" to "which papers", and nothing else.
AS_SET_QUESTION: dict[str, str] = {
    "gf-01": "Which analyses are searches (rather than measurements) whose event "
             "selection uses both b-tagged jets and missing transverse momentum?",
    "gf-01-condition": "Which analyses use b-tagged jets (jets identified as "
                       "containing a b-hadron) in their event selection?",
    "gf-02": "Which analyses estimate a background using an ABCD method, or an "
             "ABCD-style sideband or matrix method over independent regions?",
    "gf-03": "Which analyses use the HistFitter framework for their statistical "
             "analysis?",
    "gf-04": "Which analyses unfold their measured distributions -- that is, "
             "correct them back to particle level or truth level?",
    "gf-05": "Which analyses reconstruct a Higgs-boson candidate as a physical "
             "object they select on (rather than merely studying Higgs "
             "production or decay)?",
    "gf-07": "Which analyses have a ttZ background and a control region used to "
             "normalise that background?",
    "gf-08": "Which analyses require exactly two electrons OR exactly two muons "
             "as alternative selections -- parallel ee and mumu channels, rather "
             "than requiring both?",
}

#: How many conditions each question conjoins. The conjunction problem (D-058,
#: D-069) predicts precision falls with this, so it travels with the question
#: rather than being reconstructed by eye at analysis time.
CONDITIONS: dict[str, int] = {
    "gf-01": 3, "gf-01-condition": 1, "gf-02": 1, "gf-03": 1,
    "gf-04": 1, "gf-05": 2, "gf-07": 2, "gf-08": 2,
}


def read_verdicts(path: Path | str = VERDICTS) -> dict[str, dict[str, list[str]]]:
    """{qid: {paper: [every verdict given]}}. Skips the sheet's comment header.

    A LIST, not a value. 199 rows cover only 186 distinct (question, paper)
    pairs: the sheet asked about 13 papers twice, and on 7 of them Gabriel
    answered differently the second time -- three of those are a flat yes
    against a no. Keying a dict by paper would silently keep whichever row came
    last and put a coin-flip into the gold. All 7 are on gf-01, the three-part
    conjunction, which is consistent with it being genuinely ambiguous rather
    than him being careless."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("row\t"))
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(lines[start:], delimiter="\t"):
        qid = (row.get("question_id") or "").strip()
        paper = (row.get("paper") or "").strip()
        verdict = (row.get("verdict") or "").strip().lower()
        if qid and paper and verdict:
            out[qid][paper].append(verdict)
    return {q: dict(papers) for q, papers in out.items()}


#: Verdicts ranked by strength. A row is one PIECE OF EVIDENCE, so the paper
#: takes the best verdict any of its rows earned.
_STRENGTH = {"no": 0, "unsure": 1, "yes": 2}


def resolve(given: list[str]) -> Optional[str]:
    """One verdict for a paper from however many rows the sheet gave it.

    **The strongest wins.** A sheet row is (question, paper, ONE cited
    sentence), and the review builder failed to dedupe by (qid, paper) -- so 13
    papers were asked about twice, each time with a different sentence. All 13
    duplicates carry different quotes; none is a repeat.

    That makes "no" a statement about the SENTENCE, not the paper: it means this
    quote does not establish the claim. If another quote does establish it, the
    paper is yes. Reading the pair as a contradiction and dropping the paper --
    which is what this function did first -- throws away five of gf-01's eight
    gold papers on the strength of a misreading.

    `unsure` sits between: no row established the claim, but he declined to
    reject it either. Those stay out of the gold and out of the universe, same
    as a single `unsure`.
    """
    if not given:
        return None
    return max(given, key=lambda v: _STRENGTH.get(v, -1))


def disagreements(verdicts: Optional[dict] = None) -> list[tuple[str, str, list[str], str]]:
    """(qid, paper, verdicts, resolved) where the rows did not all agree.

    Not errors -- different evidence for the same question, which is the review
    builder's missing dedupe showing through. Kept visible because the resolved
    label depends on a rule, and a rule applied to ground truth should be
    inspectable rather than buried."""
    verdicts = verdicts if verdicts is not None else read_verdicts()
    return sorted((qid, paper, given, resolve(given))
                  for qid, papers in verdicts.items()
                  for paper, given in papers.items()
                  if len(set(given)) > 1)


def build(verdicts: Optional[dict] = None, *, split: str = "dev") -> list[dict]:
    """The question records. Deterministic: sorted qids, sorted paper lists.

    SPLIT IS `dev`, AND THAT IS A REAL COST. These are the best labels in the
    project -- a physicist's, not SQL's -- and the obvious instinct is to lock
    them as `test` so the headline number stays clean (S-10). But the immediate
    use is choosing between eight ablation arms, and choosing on a set is
    developing against it. Marking them `test` and then running the ablation
    anyway would launder that.

    So: batch 1 is spent on selection and lives in `dev`. The held-out human
    evaluation has to be batch 2 -- the questions Gabriel has not returned yet --
    and it must not be looked at until the arm is frozen."""
    verdicts = verdicts if verdicts is not None else read_verdicts()
    questions: list[dict] = []

    for qid in sorted(AS_SET_QUESTION):
        judged = {p: resolve(given) for p, given in (verdicts.get(qid) or {}).items()}
        # `unsure` stays in neither list: not gold, and not a judged negative
        # either, because he declined to call it.
        yes = sorted(p for p, v in judged.items() if v == "yes")
        no = sorted(p for p, v in judged.items() if v == "no")
        if not yes or len(yes) > MAX_LISTABLE:
            continue

        questions.append({
            "qid": f"gabriel-{qid}",
            "text": AS_SET_QUESTION[qid],
            "source": "gabriel",
            "split": split,
            "shape": "set",
            "needs": ["sql"],
            "difficulty": "hard" if CONDITIONS.get(qid, 1) > 1 else "medium",
            "truth": {
                "kind": "set",
                "papers": yes,
                # Both verdicts, so the scorer can charge a false positive for a
                # paper he rejected without charging one for a paper he never saw.
                "universe": sorted(set(yes) | set(no)),
            },
            "truth_source": "gabriel",
            "provenance": {
                "generated_at": "Gabriel batch 1, returned 2026-08-19",
                "reader_qid": qid,
                "reader_question": "per-paper yes/no; reworded to a corpus-wide set",
                "conditions": CONDITIONS.get(qid, 1),
                "judged_yes": len(yes),
                "judged_no": len(no),
                "judged_unsure": sum(1 for v in judged.values() if v == "unsure"),
                "judged_multi_row": sum(
                    1 for p in judged
                    if len(set((verdicts.get(qid) or {}).get(p, []))) > 1),
                # The number that keeps a good score honest: recall is measured
                # over the papers he saw, which is not the corpus.
                "judged_papers": len(yes) + len(no),
            },
            "group": f"gabriel-{qid}",
        })
    return questions


def write(path: Path | str, questions: Optional[Iterable[dict]] = None) -> Path:
    questions = list(questions) if questions is not None else build()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    return path


def summary(questions: Iterable[dict]) -> str:
    rows = ["  qid                       gold  judged  unsure  multirow  cond",
            "  " + "-" * 62]
    for q in questions:
        p = q["provenance"]
        rows.append(f"  {q['qid']:25s} {p['judged_yes']:4d}  {p['judged_papers']:6d}"
                    f"  {p['judged_unsure']:6d}  {p['judged_multi_row']:8d}"
                    f"  {p['conditions']:4d}")
    return "\n".join(rows)
