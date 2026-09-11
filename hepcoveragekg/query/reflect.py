"""Post-answer reflection, judged against the graph rather than against the
model's opinion of its own answer (POST_REFLECT=1).

WHY THIS SHAPE, AND NOT THE OBVIOUS ONE. The obvious post-hoc reflection is
the one Reflexion (Shinn et al., 2023) popularised: show the model its own
trajectory and let it write verbal feedback for the next attempt. Two results
say that is the wrong default here. Pan et al. (TACL 2024) survey the whole
correction landscape and conclude that intrinsic self-correction -- feedback
the model produces about itself, with no external signal -- is unreliable,
because "using the same model to verify its outputs offers minimal
improvement over the original generation", and they warn that post-hoc
correction risks "degrading initially correct responses". Huang et al. (ICLR
2024) measure exactly that: without an external signal, self-correction often
makes reasoning worse. Reflexion itself does not contradict this -- its
Evaluator is an EXTERNAL signal (unit tests, an exact-match grader, an
environment reward), and the reflection only translates that signal into
words.

This project has external signals of its own, and they are cheap: the graph
says which papers exist, the trace says which entities were actually
retrieved, and the question names its own conditions. So the audit below is
COMPUTED, never asked:

  condition not covered   the question enumerates a condition (D-131's
                          splitter, the one behind --enum-expand) and NOT ONE
                          retrieved entity covers it. gf-01 asks for searches
                          using b-tagged jets AND missing transverse
                          momentum; precision is 0.90 on single-condition
                          questions and 0.30 on that one, and the traces show
                          runs answering it having retrieved nothing for the
                          second condition.
  invented paper          the answer names an arXiv id the graph does not
                          hold (D-157 found runs of consecutive invented ids
                          once the draft was allowed into a candidate list).

WHAT IT MAY DO, AND WHAT IT MAY NOT. It may send the run back to retrieve,
naming the specific condition that has nothing behind it. It may NOT edit the
answer, score it, or ask the model whether the answer is good: an answer that
is already right cannot be made wrong by this mechanism, only extended, which
is the property Pan et al.'s warning asks for. This is Devil's Advocate's
post-action alignment check (Wang et al., 2024) with the alignment function
computed from the graph instead of asked of the model, and it is PoG's
reflection step (Chen et al., NeurIPS 2024) -- "is the current information
sufficient?", then go back to a specific entity -- with the sufficiency test
made mechanical.

BOUNDED, for the reason the widening ladder is bounded. Each condition is
offered at most once, there is a hard ceiling of MAX_REFLECTIONS per run, and
it requires rounds to spend. A run that is out of conditions, out of
reflections or out of rounds answers as it was going to.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

#: How many times one run may be sent back. Two: the traces stop at ~3 rounds
#: of a budget of 6, so there is room, but a mechanism that can fire on every
#: round is a loop with extra steps.
MAX_REFLECTIONS = 2

#: Rounds that must remain. One is not enough -- the run needs a round to
#: retrieve and a round to answer.
MIN_ROUNDS_LEFT = 2

_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}\b")

UNCOVERED = "uncovered_condition"
INVENTED = "invented_paper"


@dataclass
class Defect:
    """One thing the graph says is wrong with the answer about to be given."""

    kind: str
    detail: str
    suggestion: str


def _retrieved_labels(conn, entity_ids) -> list:
    """Labels of the entities this run actually retrieved.

    The ids themselves are slugs ("hepkg:object:b-tagged-jet") and would half
    work, but a label is what the concept splitter was tuned against, and the
    alias column carries the spellings that make "MET" meet "missing
    transverse momentum".
    """
    ids = sorted({str(e) for e in (entity_ids or []) if e})
    if not ids or conn is None:
        return []
    out = []
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        marks = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                f"SELECT label, aliases FROM entity_occurrence "
                f"WHERE entity_id IN ({marks})", chunk).fetchall()
        except Exception:  # noqa: BLE001 -- an audit must not kill a run
            try:
                rows = conn.execute(
                    f"SELECT label, NULL FROM entity_occurrence "
                    f"WHERE entity_id IN ({marks})", chunk).fetchall()
            except Exception:  # noqa: BLE001
                return out
        for label, raw in rows:
            if label:
                out.append(str(label))
            if raw:
                out.append(str(raw))
    return out


def condition_groups(question: str, limit: int = 4) -> list:
    """The question's conditions, as groups of alternatives.

    AND and OR are not the same demand and the enumeration splitter (D-131)
    does not distinguish them, because for RETRIEVAL both are worth searching.
    For an audit the difference decides correctness: gf-04 asks for analyses
    that "correct them back to particle level or truth level", and a run that
    retrieved particle level has covered that condition. Treating the two
    halves as separate demands made the audit fire on a sound answer, which is
    the false positive this mechanism can least afford.

    So: split on AND and commas into groups, then split each group on OR into
    alternatives. A group is covered when ANY alternative is.
    """
    from hepcoveragekg.query import planner as _p

    t = re.sub(r"\(.*?\)", "", question or "")
    t = re.sub(r"\b(rather than|instead of)\b.*$", "", t, flags=re.IGNORECASE)
    groups = []
    for chunk in re.split(r",\s+|\s+and\s+", t):
        alts = []
        for part in re.split(r",\s*or\s+|\s+or\s+|\bor\b", chunk):
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-\+\u2192>]*", part)
                     if w.lower() not in _p._ENUM_STOP]
            if 1 <= len(words) <= 5:
                phrase = " ".join(words)
                if len(phrase) > 3 and phrase not in alts:
                    alts.append(phrase)
        if alts and alts not in groups:
            groups.append(alts)
    return groups[:limit]


def uncovered_conditions(question: str, conn, session) -> list:
    """Conditions the question names that nothing retrieved covers.

    Reported by the group's first alternative, which is the wording the
    question leads with.
    """
    from hepcoveragekg.query import planner as _p

    groups = condition_groups(question or "")
    if len(groups) < 2:
        # A single-condition question has nothing to be partially covered:
        # if it retrieved nothing at all that is an abstention, which is the
        # ladder's business, not this one.
        return []
    labels = _retrieved_labels(conn, getattr(session, "known_entity_ids", set()))
    if not labels:
        return []
    blob = " ".join(labels)
    return [g[0] for g in groups if not any(_p._covers(blob, alt) for alt in g)]


def invented_papers(text: str, conn) -> list:
    """arXiv ids in the answer that the graph does not hold."""
    named = list(dict.fromkeys(_ARXIV.findall(text or "")))
    if not named or conn is None:
        return []
    marks = ",".join("?" * len(named))
    try:
        held = {str(r[0]) for r in conn.execute(
            f"SELECT arxiv_id FROM paper WHERE arxiv_id IN ({marks})", named).fetchall()}
    except Exception:  # noqa: BLE001
        return []
    return [p for p in named if p not in held]


def audit(question: str, text: str, conn, session, already_offered=()) -> list:
    """Everything the graph says is wrong, minus what was already raised."""
    defects = []
    for concept in uncovered_conditions(question, conn, session):
        if concept in already_offered:
            continue
        defects.append(Defect(
            UNCOVERED, concept,
            f'search for "{concept}" and use what it returns before answering'))
    invented = invented_papers(text, conn)
    if invented and INVENTED not in already_offered:
        defects.append(Defect(
            INVENTED, ", ".join(invented[:6]),
            "remove the ids that are not in the graph and name only papers a "
            "tool actually returned"))
    return defects


MESSAGE = """\
Before that answer is accepted, a check against the graph itself found \
something you have not covered.

{findings}

You have {rounds_left} rounds left. {instruction} Then answer.

This is not a judgement of your answer -- it is what the graph says is \
missing from what you retrieved."""


def message(defects: list, rounds_left: int) -> str:
    lines, instructions = [], []
    for d in defects:
        if d.kind == UNCOVERED:
            lines.append(f"  - the question asks about \"{d.detail}\", and not one "
                         f"entity you retrieved covers it")
        else:
            lines.append(f"  - the answer names ids the graph does not hold: {d.detail}")
        instructions.append(d.suggestion)
    return MESSAGE.format(findings="\n".join(lines), rounds_left=rounds_left,
                          instruction="; ".join(instructions).capitalize() + ".")


def should_reflect(question: str, text: str, conn, session, rounds_left: int,
                   enabled: bool) -> Optional[list]:
    """The defects worth sending the run back for, or None.

    Never fires on an abstention (the ladder's job), never twice on the same
    condition, never without rounds to act.
    """
    if not enabled or rounds_left < MIN_ROUNDS_LEFT:
        return None
    if getattr(session, "reflections_used", 0) >= MAX_REFLECTIONS:
        return None
    offered = getattr(session, "reflect_offered", None)
    if offered is None:
        offered = set()
    defects = audit(question, text, conn, session, already_offered=offered)
    if not defects:
        return None
    return defects


def enabled_from_env() -> bool:
    return os.environ.get("POST_REFLECT", "") == "1"


# ---------------------------------------------------------------------------
# THE ASKED VERSION (REFLECT_MODE=model): Devil's Advocate's post-action
# alignment, on the objective rather than on the answer.
# ---------------------------------------------------------------------------
#
# The computed audit above is high-precision and, measured on real traces,
# fires almost only on invented ids: at 44 retrieved entities per record the
# coverage test is always satisfied, even where the ANSWER ignores a condition
# (D-177). Coverage of the graph is not the same thing as fulfilment of the
# objective, and only a reader can tell them apart.
#
# So this asks, and the thing it is asked about is chosen to stay out of the
# circularity Pan et al. (TACL 2024) warn about. It does NOT ask "is your
# answer right" -- that is self-evaluation, unreliable and able to spoil a
# correct answer. It asks "given the question and THESE RETRIEVED ROWS, which
# parts of the question do the rows already establish, and which have nothing
# behind them yet" -- a question about the trace, which the model can check
# against material in front of it, and which is what Devil's Advocate's
# alignment function G_align(S_t, a_t, S_t+1, tau_i) does after every action.
#
# The verdict has consequences, which is the half our earlier progress block
# (D-173, D-176) lacked: a run that tries to answer while the verdict says a
# part has nothing behind it is sent back once with that part named. That is
# the backtracking half of the same paper, with "go back to the previous
# state" replaced by "go back and retrieve the missing part", because our
# actions are queries and re-querying is cheap.

COMPLETENESS_PROMPT = """\
You are checking how much of a question has been ANSWERED SO FAR by what has
been retrieved. You do not answer the question, you do not plan, and you do
not judge whether the retrieved rows are correct.

THE QUESTION
{question}

WHAT HAS BEEN RETRIEVED SO FAR
{retrieved}

{previous}Break the question into its parts. For each, say whether the rows above
already establish it, or whether nothing retrieved bears on it yet.

Reply in exactly this form and nothing else:

PART: <the part of the question, in a few words> -- HAVE: <what the rows give
for it, or "nothing yet">
(one line per part)
READY: yes|no
NEXT: <the single most useful thing to retrieve next, concretely -- a search
text, an entity, a predicate. Write "-" if READY is yes.>"""


@dataclass
class Verdict:
    ready: bool
    parts: list
    next_step: str
    raw: str = ""

    @property
    def missing(self) -> list:
        return [p for p, have in self.parts if "nothing" in have.lower()]


_PART = re.compile(r"^\s*PART:\s*(.+?)\s*--\s*HAVE:\s*(.+?)\s*$", re.M | re.I)
_READY = re.compile(r"^\s*READY:\s*(yes|no)\b", re.M | re.I)
_NEXT = re.compile(r"^\s*NEXT:\s*(.+?)\s*$", re.M | re.I)


def retrieval_summary(conn, session, max_labels: int = 40) -> str:
    """What the run has actually got, as the completeness call sees it.

    The steps say what was asked and how much came back; the labels say what
    the rows ARE, which is what a part of the question has to be matched
    against. Both are capped: this is re-sent every round.
    """
    lines = []
    for s in list(getattr(session, "steps", None) or [])[-8:]:
        tool = getattr(s, "tool", "")
        if tool in ("push", "reflect"):
            continue
        err = getattr(s, "error", "")
        lines.append(f"  {tool}({getattr(s, 'args', {})}) -> "
                     + (f"ERROR {err}" if err else f"{getattr(s, 'rows', 0)} rows"))
    labels = []
    seen = set()
    for lab in _retrieved_labels(conn, getattr(session, "known_entity_ids", set())):
        key = lab.lower()[:60]
        if key not in seen and not lab.startswith("["):
            seen.add(key)
            labels.append(lab[:70])
        if len(labels) >= max_labels:
            break
    if labels:
        lines.append("  entities retrieved: " + "; ".join(labels))
    return "\n".join(lines) or "  (nothing retrieved yet)"


def completeness(chat, question: str, conn, session, previous: str = "") -> Optional[Verdict]:
    """One call: which parts of the question the retrieved rows establish.

    Returns None on any failure, which means "do not interfere" -- a parse
    error must never block an answer.
    """
    body = COMPLETENESS_PROMPT.format(
        question=question,
        retrieved=retrieval_summary(conn, session),
        previous=(f"YOUR LAST CHECK SAID\n{previous.strip()}\n\n" if previous.strip() else ""))
    try:
        response = chat([{"role": "user", "content": body}], None)
        text = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 -- a check must not kill a run
        logger.warning("completeness call failed (%s); not interfering", str(exc)[:80])
        return None
    parts = [(m.group(1).strip(), m.group(2).strip()) for m in _PART.finditer(text)]
    ready_m = _READY.search(text)
    next_m = _NEXT.search(text)
    if not parts and not ready_m:
        logger.info("completeness verdict unparseable, not interfering: %r", text[:120])
        return None
    ready = (ready_m.group(1).lower() == "yes") if ready_m else True
    return Verdict(ready=ready, parts=parts,
                   next_step=(next_m.group(1).strip() if next_m else "-"), raw=text)


def render_verdict(v: Verdict) -> str:
    """The block the planner is shown between rounds."""
    lines = [f"  - {p}: {have}" for p, have in v.parts] or ["  - (no parts identified)"]
    tail = ("\n  Everything the question asks for has something behind it."
            if v.ready else f"\n  Still missing. Most useful next: {v.next_step}")
    return ("\n\nHOW MUCH OF THE QUESTION IS ANSWERED SO FAR (checked after the last "
            "round, from what you retrieved):\n" + "\n".join(lines) + tail)


BOUNCE = """\
A check of what you have retrieved, made before this answer is accepted, says part of the question has nothing behind it yet:

{missing}

You have {rounds_left} rounds left. Retrieve that first: {next_step}. Then answer.

This is not a judgement of your answer -- it is about what the rows you retrieved do and do not cover."""


def bounce_message(v: Verdict, rounds_left: int) -> str:
    missing = "\n".join(f"  - {p}: {have}" for p, have in v.parts
                         if "nothing" in have.lower()) or "  - (see the check above)"
    return BOUNCE.format(missing=missing, rounds_left=rounds_left,
                         next_step=v.next_step if v.next_step != "-" else
                         "search for the part named above")


def mode() -> str:
    """`graph` (the computed audit), `model` (the asked check), or `off`."""
    raw = os.environ.get("REFLECT_MODE", "").strip().lower()
    if raw in ("graph", "model", "both"):
        return raw
    return "graph" if os.environ.get("POST_REFLECT", "") == "1" else "off"


def completeness_from_raw(raw: str) -> Optional[Verdict]:
    """Rebuild a verdict from the text it produced, for re-display."""
    parts = [(m.group(1).strip(), m.group(2).strip()) for m in _PART.finditer(raw or "")]
    ready_m = _READY.search(raw or "")
    next_m = _NEXT.search(raw or "")
    if not parts and not ready_m:
        return None
    return Verdict(ready=(ready_m.group(1).lower() == "yes") if ready_m else True,
                   parts=parts, next_step=(next_m.group(1).strip() if next_m else "-"),
                   raw=raw)
