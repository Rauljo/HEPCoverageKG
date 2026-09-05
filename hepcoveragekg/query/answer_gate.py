"""Does the written answer actually SAY the answer? (D-107)

The scorer reads arXiv ids out of `a.text`, and refusing to fall back to the
retrieval footprint is right -- D-062 measured what happens when a 58-paper
footprint is allowed to stand in for an answer. But runs 54201-06 showed the
other side of that rule: 23 of 60 Gabriel answers scored a hard 0 while the run
had already found the papers, because the prose named none of them.

Four shapes, all observed verbatim in those runs:

    placeholder   "...are: [list of papers from the intersection]."
    titles only   "1. **Measurement of the production cross section for a W
                   boson in association with a charm quark** (uses unfolding
                   ...)" -- seven correct papers, scored 0
    deferred      "To finalize, run `papers_of` on the intersection of these
                   two result sets." -- and it never ran it
    dangling cite `papers_from` naming a set that does not exist

THIS IS DELIBERATELY NOT AN LLM. Every one of those is decidable by looking at
the answer: are there ids in the text, did a citation resolve, is there a
promise to do something instead of a result. A judge would add a call, a
latency, a noise floor and a 25%-default failure mode (D-105) to a question
that has an exact answer. The semantic check -- "does this answer the question
that was asked" -- is a different job and belongs to `answer_critic`.

The gate never rewrites the answer. It returns one correction message, the
model gets one more round, and whether it needed asking is recorded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

#: An arXiv id as this corpus writes them. Same pattern the scorer uses, on
#: purpose: the gate must pass exactly when `judged_set_f1` would find ids.
ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")

#: A bracketed stand-in where the list should be. Anchored on the bracket so
#: ordinary prose in square brackets ("[1]", "[see fig. 3]") does not trip it.
PLACEHOLDER = re.compile(
    r"\[\s*(?:list|the list|insert|papers?|ids?|arxiv|results?|to be|tbd|\.\.\.)"
    r"[^\]]{0,80}\]", re.IGNORECASE)

#: A promise to do the work instead of the work. `papers_of` is the tool the
#: model defers to in practice; the others are how it phrases the deferral.
DEFERRED = re.compile(
    r"(?:to finalize|to complete|next step|you (?:can|should|would)|"
    r"one (?:can|would)|run|call|execute|query)\b[^.]{0,60}"
    r"`?(?:papers_of|describe|contents_of|intersection)`?", re.IGNORECASE)

MESSAGE = (
    "That answer does not name any papers. The scorer -- and the reader -- "
    "only ever sees the text you write, so a set reference, a placeholder or "
    "an instruction to run another tool is not an answer.\n\n{problem}\n\n"
    "Write the arXiv ids into the text, one per paper, e.g. "
    "\"2004.14060, 2006.05880, 2012.03799\". You may keep the prose and the "
    "titles; add the ids. If you cannot name any, answer with "
    "reason=not_in_graph and say why."
)


@dataclass
class GateResult:
    """What the gate found. `ok` means the answer stands as written."""
    ok: bool
    problem: str = ""
    named: int = 0
    #: Which shape tripped it, for the run record. One of "", "placeholder",
    #: "deferred", "silent" (prose that simply never names an id).
    kind: str = ""

    @property
    def message(self) -> str:
        return "" if self.ok else MESSAGE.format(problem=self.problem)


def check(text: str, cited: str = "", answerable: bool = True,
          reason: str = "") -> GateResult:
    """Judge one answer's FORM. Never touches the graph or the question.

    `cited` is `session.answer_cited`: a resolved citation is a real answer even
    when the prose holds no ids, because `resolve_citations` has put the papers
    where the scorer can read them. An abstention is exempt -- "not in the
    graph" is a legitimate answer that names nothing, and gating it would push
    the system to fabricate coverage, which is the failure that matters more.
    """
    text = text or ""
    if reason == "not_in_graph" or not answerable:
        return GateResult(True)
    if cited:
        return GateResult(True)
    ids = set(ARXIV.findall(text))
    if ids:
        return GateResult(True, named=len(ids))
    if not text.strip():
        return GateResult(False, "The answer text is empty.", kind="silent")
    hit = PLACEHOLDER.search(text)
    if hit:
        return GateResult(
            False, f"You wrote a placeholder where the list should be: "
                   f"{hit.group(0)!r}.", kind="placeholder")
    hit = DEFERRED.search(text)
    if hit:
        return GateResult(
            False, f"You described the work instead of doing it: "
                   f"{hit.group(0).strip()!r}. Make the call, then answer.",
            kind="deferred")
    return GateResult(
        False, "The answer is written entirely in prose and titles -- there is "
               "not one arXiv id in it.", kind="silent")
