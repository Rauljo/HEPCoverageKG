"""Teaching the small critic where the large one draws the line.

THE MEASUREMENT THAT MOTIVATES IT. Both critics judged the same 202 search steps
on conceptB-200:

    8B    exact 11.1%   broader 20.5%   unrelated 68.4%   -> keeps 31.6%
    72B   exact 13.0%   broader 10.4%   unrelated 76.7%   -> keeps 23.4%

The 8B is the PERMISSIVE one. It reaches for "broader" twice as often, and the
72B -- which scores 0.144 on counting against the 8B's 0.080 -- is stricter. So
the fix is not another rule about what "broader" means. It is showing the 8B the
specific candidates it waved through and the large model did not.

WHY THESE EXAMPLES AND NOT WRITTEN ONES. The prompt already carries three
hand-written examples and the 8B still keeps a third of everything. These are
1,969 real disagreements with the 72B's own stated reason attached -- "This is
about tWZ modelling uncertainty, which is not directly related to the PDF
modelling uncertainty asked about". That is the line being drawn, in the
larger model's words, on candidates the small one actually got wrong.

WHAT IT IS. Distillation of a 72B judgement into an 8B prompt, and it should be
reported as that. It does NOT make the 8B a 72B; it transfers one specific
behaviour -- where to stop calling things related -- and only that.

THE RISK, which is the mirror of the last one. Teaching strictness can overshoot:
a critic that drops everything scores zero on recall and the arm looks fine on
precision. `broader` exists because a wider version of what was asked is still
worth keeping, and examples that are all "unrelated" would teach the model to
drop. So the block carries KEEP examples too, drawn from candidates BOTH critics
kept, and the ratio is stated rather than left to chance.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Iterable

MAX_DROP = 3
MAX_KEEP = 2


def _reviews(paths: Iterable[str]) -> dict:
    out: dict = collections.defaultdict(dict)
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "_meta" in rec:
                continue
            for rev in (rec.get("answer") or {}).get("reviews") or []:
                key = (rec["qid"], rev.get("search_text"))
                # The QUESTION travels with the review. The critic judges a
                # candidate against the question, not against the search text,
                # so an example showing only the search text is unreadable --
                # the reason will talk about something the reader cannot see.
                out[key]["__question__"] = rec.get("question", "")
                for d in rev.get("dropped") or []:
                    out[key][d["id"]] = d.get("why", "")
    return dict(out)


def disagreements(small_paths, big_paths) -> list[dict]:
    """Candidates the LARGE critic dropped and the small one kept."""
    small, big = _reviews(small_paths), _reviews(big_paths)
    got = []
    for key in set(small) & set(big):
        qid, search_text = key
        question = big[key].get("__question__") or small[key].get("__question__", "")
        for eid, why in big[key].items():
            if eid == "__question__":
                continue
            if eid not in small[key] and why:
                got.append({"question": question, "search_text": search_text,
                            "entity_id": eid, "why": why, "qid": qid})
    # One per search text, so a single verbose question cannot supply them all.
    seen, unique = set(), []
    for item in sorted(got, key=lambda x: x["entity_id"]):
        if item["search_text"] in seen:
            continue
        seen.add(item["search_text"])
        unique.append(item)
    return unique


def render(drops: list[dict], keeps: list[dict] | None = None) -> str:
    """The block appended to the small critic's prompt."""
    if not drops:
        return ""
    lines = ["", "WHERE THE LINE SITS -- judgements from a larger judge on real candidates:"]
    for d in drops[:MAX_DROP]:
        lines.append(f'  question: {str(d.get("question") or "")[:80]}')
        lines.append(f'    candidate "{d["entity_id"].split(":")[-1]}" -> unrelated, '
                     f'because {d["why"][:110]}')
    if keeps:
        lines.append("")
        lines.append("And candidates it KEPT -- a wider version of what was asked "
                     "is still worth counting:")
        for k in keeps[:MAX_KEEP]:
            lines.append(f'  searching {str(k["search_text"])[:60]!r}')
            lines.append(f'    "{k["entity_id"].split(":")[-1]}" -> keep')
    return "\n".join(lines)
