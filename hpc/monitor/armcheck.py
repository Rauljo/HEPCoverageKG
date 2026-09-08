#!/usr/bin/env python3
"""Is this run actually running the arm we asked for? (D-114)

WHY THIS EXISTS. Three consecutive days of runs completed, scored, and
reported numbers while something they depended on was not there:

    D-088  the critic 404'd on every call; six arms labelled critic-on
    D-105  a reasoning judge returned empty content; 33,836 candidates,
           100% defaulted, zero drops, four full runs
    D-111  the server hit its wall limit 40 minutes in; 44% of questions
           errored and the arms reported anyway
    D-108  the gate arm fired twice in 164 questions, because 18% of answers
           left through a path the gate could not see

None of these was a crash. Every one produced a plausible number. Watching
`squeue` catches none of them, because in all four cases the job was RUNNING.

So this checks the four things that were each, in turn, the thing nobody
checked:

    FLAGS      what the run RECORDED matches what we asked for
    FIRING     the mechanism the arm exists to test actually did something
    HEALTH     the error rate is not eating the run
    RANGE      the scores are inside what this system can produce

Exit status is 1 if any check FAILS, so it can gate a report.

    python hpc/monitor/armcheck.py 54232 54233 --expect gate=1 critic=1
    python hpc/monitor/armcheck.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

#: An error rate above this is not a run with some bad questions in it, it is a
#: broken dependency. D-111 sat at 0.44; the stacked-GPU batch at 0.13, which
#: was already too much and was reported anyway.
MAX_ERROR_RATE = 0.10

#: judged_f1 outside this band means the scorer is reading something other than
#: an answer. 0.0 across the board was the D-111 signature; above 0.9 on this
#: corpus has never happened and would mean the gold leaked into the input.
SCORE_BAND = (0.02, 0.90)

#: A mechanism is checked against its OPPORTUNITY, never against zero.
#:
#: "Greater than zero" is not enough and D-108 proves it: the gate arm fired
#: twice in 164 questions while 23 answers named nothing, and a `> 0` test
#: passes that happily. What the arm is worth depends on the mechanism acting
#: on the cases it exists for, so each entry says how to count BOTH -- what it
#: did, and how many chances it had.
#:
#:   fired(answer)        -> did the mechanism act on this record
#:   chances(answer, rec) -> should it have
MECHANISMS = {
    "use_critic": (
        "critic",
        lambda a: sum(r.get("candidates", 0) for r in a.get("reviews", [])),
        # every search is a chance to judge candidates
        lambda a, r: sum(1 for s in a.get("steps", [])
                         if s.get("tool") in ("search", "concept")),
    ),
    "answer_gate": (
        "gate",
        lambda a: int(bool(a.get("gate_retried") or a.get("gate_failed"))),
        # an answer that names nothing and is not an abstention IS the case the
        # gate exists for. If those are not being caught, the arm is a no-op.
        lambda a, r: int(r.get("shape") == "set"
                         and not a.get("error")
                         and not a.get("named_ids")
                         and not a.get("cited")
                         and a.get("answered")),
    ),
    "answer_critic": (
        "answer-critic",
        lambda a: (a.get("answer_review") or {}).get("candidates", 0),
        lambda a, r: int(bool(a.get("named_ids") or a.get("cited"))),
    ),
    "kind_fallback": (
        "kind-fallback",
        lambda a: a.get("kind_fallback_added", 0),
        # every kinded search is a chance; zero appended over many is a no-op
        lambda a, r: a.get("kinded_searches", 0),
    ),
    "enum_expand": (
        "enum-expand",
        lambda a: a.get("enum_added", 0),
        lambda a, r: a.get("enum_concepts", 0),
    ),
    "ranked_answer": (
        "ranked-answer",
        lambda a: int(bool(a.get("ranked_answer_asked"))),
        # a chance = an answer() reached with a usable ranking in hand
        lambda a, r: int(bool(a.get("ranked_answer_shown"))),
    ),
    "rerank": (
        "rerank",
        lambda a: sum(x.get("graded", 0) for x in a.get("rankings", [])),
        # ONLY THE CALLS IT COULD ACT ON. `rank_papers` returns early below two
        # papers -- there is no order to change -- and on 54248 thirty of
        # forty-three `papers_of` calls came back with fewer than two. Counting
        # those as chances put the rate at 49% against a 50% threshold and
        # reported a working mechanism as broken. Against calls it could act
        # on, it fires on every one.
        lambda a, r: sum(1 for s in a.get("steps", [])
                         if s.get("tool") == "papers_of" and (s.get("rows") or 0) >= 2),
    ),
}

#: Below this share of its opportunities, a mechanism is not doing its job even
#: though it is switched on and has fired at least once. The D-108 gate managed
#: 2 of 23 = 9%.
MIN_FIRE_RATE = 0.5

#: RATES ARE NOT JUDGED ON A HANDFUL OF RECORDS. Checked against a live run two
#: minutes after launch, this reported 36/36 critic verdicts defaulted and
#: called the run broken -- and the equivalent finished run had exactly the same
#: first record and averaged 15% over 164. Early searches default and later ones
#: do not, so a threshold applied to the first two records is noise. A check
#: that cries wolf hourly is a check that gets ignored, which is the failure it
#: exists to prevent, one level up.
MIN_RECORDS_FOR_RATES = 20
MIN_CHANCES_FOR_RATES = 10

#: A judge that defaults most of its verdicts has not judged. D-105 ran at 100%
#: for weeks; `answer_critic` warns above 25% at runtime and this is the same
#: line drawn where a run can be refused rather than merely annotated.
MAX_DEFAULT_RATE = 0.30


def load(path):
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            if line.startswith('{"_meta"'):
                meta = json.loads(line)["_meta"]
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass                       # a half-written last line, mid-run
    return meta, rows


def check(path, expect: dict) -> list:
    """Returns [(ok, label, detail)]; ok is None for 'nothing to say yet'."""
    meta, rows = load(path)
    cfg = meta.get("config") or {}
    out = []
    n = len(rows)
    out.append((None, "records", f"{n} of {meta.get('n_questions', '?')}"))
    if not n:
        out.append((False, "progress", "no records yet"))
        return out

    # 1. FLAGS -- what we asked for is what it recorded
    for key, want in expect.items():
        got = cfg.get(key)
        ok = bool(got) == bool(want)
        out.append((ok, f"flag {key}",
                    f"asked {bool(want)}, recorded {got!r}"))

    # 2. FIRING -- against the chances it had, not against zero
    for key, (label, fired, chances) in MECHANISMS.items():
        if not cfg.get(key):
            continue
        total = sum(fired(r.get("answer") or {}) for r in rows)
        acted = sum(1 for r in rows if fired(r.get("answer") or {}))
        chance = sum(chances(r.get("answer") or {}, r) for r in rows)
        if chance < MIN_CHANCES_FOR_RATES and n < MIN_RECORDS_FOR_RATES:
            # No information yet. Saying nothing is right; saying FAIL is worse
            # than saying nothing, because it trains the reader to skip the line.
            out.append((None, f"{label} fired",
                        f"{total} so far, {chance} chances -- too early to judge"))
        elif total == 0 and chance:
            out.append((False, f"{label} fired",
                        f"NEVER, over {chance} chances -- the arm is a no-op"))
        elif total == 0:
            out.append((None, f"{label} fired", "no chances yet"))
        elif chance and acted / chance < MIN_FIRE_RATE:
            out.append((False, f"{label} fired",
                        f"{acted} of {chance} chances = {acted / chance:.0%} "
                        f"(min {MIN_FIRE_RATE:.0%}) -- switched on but not working"))
        else:
            rate = f"{acted}/{chance} chances" if chance else f"{acted} records"
            out.append((True, f"{label} fired", f"{total} over {rate}"))

    # 2b. A JUDGE THAT DEFAULTS HAS NOT JUDGED (D-105)
    for key, get in (("use_critic", lambda a: a.get("reviews", [])),
                     ("answer_critic",
                      lambda a: [a["answer_review"]] if a.get("answer_review") else [])):
        if not cfg.get(key):
            continue
        seen = defaulted = 0
        for r in rows:
            for rv in get(r.get("answer") or {}):
                seen += rv.get("candidates", 0)
                defaulted += rv.get("defaulted", 0)
        if seen and n >= MIN_RECORDS_FOR_RATES:
            share = defaulted / seen
            out.append((share <= MAX_DEFAULT_RATE, f"{key} judged",
                        f"{defaulted}/{seen} defaulted = {share:.0%} "
                        f"(max {MAX_DEFAULT_RATE:.0%})"))
        elif seen:
            share = defaulted / seen
            out.append((None, f"{key} judged",
                        f"{defaulted}/{seen} = {share:.0%} over {n} records "
                        f"-- too early to judge (need {MIN_RECORDS_FOR_RATES})"))

    # 3. HEALTH
    errs = sum(1 for r in rows if (r.get("answer") or {}).get("error"))
    rate = errs / n
    out.append((rate <= MAX_ERROR_RATE if n >= MIN_RECORDS_FOR_RATES else None,
                "error rate",
                f"{errs}/{n} = {rate:.0%} (max {MAX_ERROR_RATE:.0%})"
                + ("" if n >= MIN_RECORDS_FOR_RATES else " -- too early")))

    # 4. RANGE
    scored = [r["scores"].get("judged_f1", r["scores"].get("set_f1"))
              for r in rows]
    scored = [s for s in scored if s is not None]
    if scored:
        mean = sum(scored) / len(scored)
        lo, hi = SCORE_BAND
        out.append((lo <= mean <= hi, "score in range",
                    f"mean {mean:.3f} over {len(scored)} (band {lo}-{hi})"))
    else:
        out.append((None, "score in range", "nothing scored yet"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*", help="Slurm job ids")
    ap.add_argument("--all", action="store_true", help="every run file today")
    ap.add_argument("--expect", nargs="*", default=[],
                    help="key=1 / key=0 flags the run must have recorded")
    ap.add_argument("--runs", default="eval/runs")
    a = ap.parse_args()

    expect = {}
    for pair in a.expect:
        k, _, v = pair.partition("=")
        expect[k] = v not in ("0", "false", "False", "")

    paths = []
    for job in a.jobs:
        paths += sorted(glob.glob(os.path.join(a.runs, f"*{job}.jsonl")))
    if a.all and not paths:
        paths = sorted(glob.glob(os.path.join(a.runs, "*.jsonl")))[-8:]
    if not paths:
        print("no run files found", file=sys.stderr)
        return 2

    worst = 0
    for p in paths:
        print(f"\n=== {os.path.basename(p)} ===")
        for ok, label, detail in check(p, expect):
            mark = "    " if ok is None else ("PASS" if ok else "FAIL")
            print(f"  {mark}  {label:22} {detail}")
            if ok is False:
                worst = 1
    print("\n" + ("SOMETHING IS WRONG -- do not report these numbers"
                  if worst else "all checks passed"))
    return worst


if __name__ == "__main__":
    sys.exit(main())
