"""Score the answer-critic against the labels a physicist wrote (D-106, D-110).

THE ONE COMPONENT WE CAN CHECK DIRECTLY. Gabriel was shown a question, a paper,
and the sentence we retrieved from it, and answered yes or no -- 253 such
verdicts across nine questions. `answer_critic.judge_papers` is asked to do the
same thing on the same evidence. So his sheet is not a proxy for this
measurement, it IS this measurement, and nothing else in the project has that.

WHAT COUNTS AS A LABEL. `truth.universe` is every paper he looked at for a
question; `truth.papers` is the subset he said yes to. The complement is a real
NO, not an absence -- which is the whole reason `judged_set_f1` exists (D-072).

WHERE THE EVIDENCE COMES FROM. The critic must see what the SYSTEM retrieved,
not everything the paper contains: judging on material the run never surfaced
would measure the corpus. So the entity ids come from a stored run record, and
the judge sees exactly the labels and quotes that run had in hand.

WHAT IT COSTS. Nothing like an arm. There is no agent loop, no retrieval and no
tool call -- one chunked judging pass over 253 papers, which is why this is the
right thing to run on a metered endpoint while the cluster does the arms.
`estimate()` prints the bill before any of it is spent.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class Pair:
    """One (question, paper) that Gabriel judged."""
    qid: str
    question: str
    paper: str
    gold: bool                       # what he said
    labels: list = field(default_factory=list)
    quotes: list = field(default_factory=list)


@dataclass
class Outcome:
    """The confusion matrix, in the terms that matter for a filter."""
    tp: int = 0      # critic kept,    he said yes
    fp: int = 0      # critic kept,    he said no    -- a false keep
    tn: int = 0      # critic dropped, he said no    -- a correct drop
    fn: int = 0      # critic dropped, he said yes   -- THE COST
    defaulted: int = 0
    missing: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def agreement(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def keep_precision(self) -> float:
        """Of the papers it kept, how many he agreed with."""
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def keep_recall(self) -> float:
        """Of the papers he said yes to, how many survived."""
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def drop_rate(self) -> float:
        d = self.n
        return (self.tn + self.fn) / d if d else 0.0

    def to_dict(self) -> dict:
        return {"n": self.n, "tp": self.tp, "fp": self.fp, "tn": self.tn,
                "fn": self.fn, "defaulted": self.defaulted,
                "agreement": round(self.agreement, 4),
                "keep_precision": round(self.keep_precision, 4),
                "keep_recall": round(self.keep_recall, 4),
                "drop_rate": round(self.drop_rate, 4)}


def pairs_from(questions: Iterable[dict], evidence_for: Callable) -> list:
    """Every judged (question, paper), with the evidence the run retrieved.

    `evidence_for(qid, papers) -> {paper: (labels, quotes)}` so the caller
    decides which run's retrieval is being replayed.
    """
    out = []
    for q in questions:
        truth = q.get("truth") or {}
        universe = [str(p) for p in (truth.get("universe") or [])]
        yes = {str(p) for p in (truth.get("papers") or [])}
        if not universe:
            continue
        ev = evidence_for(q["qid"], universe) or {}
        for paper in universe:
            labels, quotes = ev.get(paper, (set(), []))
            out.append(Pair(qid=q["qid"], question=q.get("text") or "",
                            paper=paper, gold=paper in yes,
                            labels=sorted(labels), quotes=list(quotes)))
    return out


def estimate(pairs: list, price_in: float, price_out: float,
             chunk: int = 8, out_tokens: int = 40) -> dict:
    """The bill, before any of it is spent.

    Tokens are counted at 4 characters each -- the same crude rule the budget
    guard refuses to trust for a STOP, and adequate for a decision about
    whether to press the button at all. It rounds up, never down.
    """
    from hepcoveragekg.query import answer_critic as AC

    by_q: dict = {}
    for p in pairs:
        by_q.setdefault((p.qid, p.question), []).append(p)
    chars = 0
    calls = 0
    for (_qid, question), group in by_q.items():
        for i in range(0, len(group), chunk):
            batch = group[i:i + chunk]
            blocks = "\n\n".join(
                AC._render(p.paper, p.labels, p.quotes) for p in batch)
            chars += len(AC.PROMPT) + len(question) + len(blocks) + 40
            calls += 1
    tok_in = -(-chars // 4)                      # ceil
    tok_out = calls * out_tokens * (chunk // 2 or 1)
    usd = tok_in / 1e6 * price_in + tok_out / 1e6 * price_out
    return {"papers": len(pairs), "calls": calls, "prompt_tokens": tok_in,
            "completion_tokens_est": tok_out, "usd_est": round(usd, 4)}


def run(pairs: list, chat: Callable, chunk: int = 8) -> tuple:
    """Judge every pair and score it against the gold. Returns (Outcome, rows)."""
    from hepcoveragekg.query import answer_critic as AC

    by_q: dict = {}
    for p in pairs:
        by_q.setdefault((p.qid, p.question), []).append(p)

    out = Outcome()
    rows = []
    for (qid, question), group in sorted(by_q.items()):
        evidence = {p.paper: (p.labels, p.quotes) for p in group}
        review = AC.judge_papers(chat, question, evidence, chunk=chunk)
        verdict = {v.paper_id: v for v in review.verdicts}
        for p in group:
            v = verdict.get(p.paper)
            if v is None:
                out.missing += 1
                continue
            if v.defaulted:
                out.defaulted += 1
            if v.keep and p.gold:
                out.tp += 1
            elif v.keep and not p.gold:
                out.fp += 1
            elif not v.keep and not p.gold:
                out.tn += 1
            else:
                out.fn += 1
            rows.append({"qid": qid, "paper": p.paper, "gold": p.gold,
                         "kept": v.keep, "why": v.why,
                         "defaulted": v.defaulted,
                         "quotes": len(p.quotes), "labels": len(p.labels)})
        logger.info("%s: judged %d papers, kept %d",
                    qid, len(group), sum(1 for v in review.verdicts if v.keep))
    return out, rows
