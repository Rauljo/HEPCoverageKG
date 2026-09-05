"""What an arm actually changed, paper by paper, against a control.

WHY A MEAN IS NOT ENOUGH. Gabriel's nine questions cannot resolve an effect
smaller than about 0.14: four identical control runs spanned 0.259-0.399
(D-101). So an arm that genuinely fixes a known bug can be invisible in the
aggregate, and an arm that fixes nothing can look good by luck. But his gold is
per (question, PAPER) -- 253 verdicts -- so "did this arm stop naming
2103.06956 for ttZ, which he judged wrong" is answerable directly, without
waiting for a mean to separate from noise.

WHAT THIS IS NOT. It does not decide whether an arm works. Looking at 253
verdicts after the fact and keeping the ones that improved is how noise becomes
a finding -- with this many comparisons some will flip by chance either way.
The `predicted` argument exists to make that explicit: name the papers or
questions the arm was SUPPOSED to fix BEFORE running it, and the report
separates predicted hits from everything else. An unpredicted flip is a
hypothesis for the next run, never evidence for this one.

WHAT IT IS FOR. Three things a mean cannot give:
  - which specific papers moved, in each direction (a fix that also breaks two
    other papers is a different result from a clean fix)
  - what changed in the TRACE to cause it -- the first tool call, the entities
    retrieved, the papers finally named
  - whether the flip is stable across repeats or a one-off

The trace half matters because run-to-run divergence here is infrastructural,
not sampled: temperature is 0.0 everywhere, and identical runs still differ on
the first tool call in 5 of 9 questions (one optional `category` argument
present or absent), which is enough to double the score. So "the arm fixed it"
and "this run happened to make the better first call" look identical in a mean
and different in a trace.
"""
from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")


@dataclass
class PaperFlip:
    qid: str
    paper: str
    verdict: str            # gabriel's truth: "gold" or "judged-negative"
    control_named: int      # how many control repeats named it
    arm_named: int          # how many arm repeats named it
    repeats: int

    @property
    def direction(self) -> str:
        """FIXED = an arm names a gold paper the control missed, or drops a
        judged-negative the control wrongly named. BROKE = the reverse."""
        gained = self.arm_named > self.control_named
        if self.verdict == "gold":
            return "FIXED" if gained else "BROKE"
        return "BROKE" if gained else "FIXED"

    @property
    def stable(self) -> bool:
        """Unanimous in both runs -- not one repeat out of three."""
        return (self.control_named in (0, self.repeats)
                and self.arm_named in (0, self.repeats))

    @property
    def unlocked(self) -> bool:
        """The control NEVER got this paper and the arm sometimes did.

        Not "stable", and it must not be filtered out as noise: on gf-05
        (2026-09-04) the control scored 0.000 in every repeat of every run ever
        made, and chATLAS named 11 of 16 gold papers on one repeat in three.
        Requiring unanimity on both sides discarded the single most informative
        event in that experiment -- a question moving from NEVER answered to
        SOMETIMES answered is a signal about capability, even when it is not yet
        a reliable fix. The mirror case (control always had it, arm sometimes
        loses it) is a regression and is reported the same way."""
        return self.control_named == 0 and 0 < self.arm_named

    @property
    def lost(self) -> bool:
        """The control ALWAYS got this paper and the arm sometimes did not."""
        return self.control_named == self.repeats and self.arm_named < self.repeats


def _records(path: str | Path) -> dict:
    out = collections.defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith('{"_meta"'):
            continue
        r = json.loads(line)
        out[r["qid"]].append(r)
    return out


def _named(record: dict) -> set:
    """Papers the ANSWER asserts -- prose ids, or the cited set. Never the
    retrieval footprint, for the reason set_f1 stopped using it (D-062)."""
    named = set(ARXIV.findall(record["answer"].get("text") or ""))
    if named:
        return named
    if record["answer"].get("cited"):
        return set(record["answer"].get("papers") or [])
    return set()


def flips(control_path, arm_path, questions: dict) -> list[PaperFlip]:
    """Every paper whose naming changed, judged against Gabriel's verdicts.

    `questions` maps qid -> the question record, so gold and universe come from
    the same file the run was scored against.
    """
    ctrl, arm = _records(control_path), _records(arm_path)
    out: list[PaperFlip] = []
    for qid in sorted(set(ctrl) & set(arm) & set(questions)):
        truth = (questions[qid].get("truth") or {})
        gold = set(truth.get("papers") or [])
        universe = set(truth.get("universe") or []) or gold
        if not gold:
            continue
        c_runs, a_runs = ctrl[qid], arm[qid]
        reps = min(len(c_runs), len(a_runs))
        if not reps:
            continue
        c_count = collections.Counter()
        a_count = collections.Counter()
        for r in c_runs[:reps]:
            c_count.update(_named(r) & universe)
        for r in a_runs[:reps]:
            a_count.update(_named(r) & universe)
        for paper in sorted(universe):
            cn, an = c_count[paper], a_count[paper]
            if cn == an:
                continue
            out.append(PaperFlip(qid=qid, paper=paper,
                                verdict="gold" if paper in gold else "judged-negative",
                                control_named=cn, arm_named=an, repeats=reps))
    return out


def trace_diff(control_path, arm_path, qid: str) -> dict:
    """What the two runs DID differently on one question -- first call, tools,
    entities, papers named. The evidence for *why* a flip happened."""
    ctrl, arm = _records(control_path), _records(arm_path)
    def summarise(runs):
        first, tools, ents, papers = [], [], [], []
        for r in runs:
            steps = r["answer"].get("steps") or []
            if steps:
                first.append(f"{steps[0].get('tool')}({json.dumps(steps[0].get('args'), sort_keys=True)})")
            tools.append([s.get("tool") for s in steps])
            ents.append(len(r["answer"].get("entity_ids") or []))
            papers.append(sorted(_named(r)))
        return {"first_calls": first, "tools": tools,
                "entities_retrieved": ents, "papers_named": papers}
    return {"control": summarise(ctrl.get(qid, [])), "arm": summarise(arm.get(qid, []))}


def report(control_path, arm_path, questions: dict,
           predicted: Optional[Iterable[str]] = None, stable_only: bool = True) -> str:
    """Human-readable diff. `predicted` is the qids (or 'qid:paper' pairs) the
    arm was expected to fix -- stated in advance, per the module docstring."""
    predicted = set(predicted or ())
    fl = flips(control_path, arm_path, questions)
    # `unlocked` and `lost` survive the stability filter on purpose -- see the
    # properties. A capability appearing or disappearing is the finding; waiting
    # for it to be unanimous discards it while it is still interesting.
    if stable_only:
        fl = [f for f in fl if f.stable or f.unlocked or f.lost]
    fixed = [f for f in fl if f.direction == "FIXED"]
    broke = [f for f in fl if f.direction == "BROKE"]

    def hit(f):
        return f.qid in predicted or f"{f.qid}:{f.paper}" in predicted

    lines = [f"stable flips: {len(fixed)} fixed, {len(broke)} broke"]
    if predicted:
        pf = [f for f in fixed if hit(f)]
        pb = [f for f in broke if hit(f)]
        lines.append(f"  PREDICTED targets: {len(pf)} fixed, {len(pb)} broke  "
                     f"(of {len(predicted)} predicted)")
        lines.append(f"  unpredicted:       {len(fixed)-len(pf)} fixed, "
                     f"{len(broke)-len(pb)} broke  <- hypotheses, not evidence")
    for label, group in (("FIXED", fixed), ("BROKE", broke)):
        if not group:
            continue
        lines.append(f"\n{label}")
        for f in sorted(group, key=lambda x: (x.qid, x.paper)):
            mark = " *PREDICTED*" if hit(f) else ""
            if f.unlocked:
                mark = " UNLOCKED (control never got it)" + mark
            elif f.lost:
                mark = " LOST (control always got it)" + mark
            lines.append(f"  {f.qid:26} {f.paper:12} {f.verdict:16} "
                         f"control {f.control_named}/{f.repeats} -> arm {f.arm_named}/{f.repeats}{mark}")
    return "\n".join(lines)
