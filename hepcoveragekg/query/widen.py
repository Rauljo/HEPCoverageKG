"""What to try instead, when a run is about to give up while holding material.

THE PROBLEM. Measured on the 2026-08-28 arms: 105 abstentions, 93% of them with
rounds still unspent against a budget of six, 100% of them holding retrieved
entities -- 59 entities across 30 papers on average. Those runs score 0.000 on
counting. They spend 44 seconds to produce nothing, then stop with half their
budget unused.

THE OBJECTION, and it is the right one. Forcing the loop to continue could make
it "go in loops infinitely without actually finding anything". Two things keep
that from happening, and neither is a promise about the model's behaviour:

  the ladder is FINITE.  Four rungs, each offered at most once per run, tracked
      in `session.widenings_used`. When it is exhausted the abstention goes
      through. There is no fifth thing to suggest, so there is nothing to loop
      on.
  every rung is CONCRETE.  Never "try harder" -- always a specific call the run
      has not made, built from its own trace. Generic exhortation is what
      produces flailing; a named alternative produces a different query.

Combined with the duplicate guard in `graph.execute` (an exact repeat is
answered from the record, never re-executed) and a requirement of two rounds
remaining, the worst case is four extra tool calls, inside a budget that is
already allocated and today goes unspent.

WHAT THIS DELIBERATELY DOES NOT DO. It does not make abstention harder to reach
in general, and it never fires on a real answer. "No paper here covers that" is
this project's output, and a system taught never to abstain fabricates coverage
instead -- which is a worse failure than the one being fixed, because it is
invisible. So a sound abstention still passes: the ladder runs out, and the
refusal is recorded as earned rather than as giving up early.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

#: Rungs, in the order they are offered. Cheapest and most likely first.
DROP_KIND = "drop_kind"
COMPATIBLE_PREDICATE = "compatible_predicate"
SHORTER_TEXT = "shorter_text"
TERMINAL_PAPERS_OF = "papers_of"

LADDER = (DROP_KIND, COMPATIBLE_PREDICATE, SHORTER_TEXT, TERMINAL_PAPERS_OF)

#: Rounds that must remain for a suggestion to be worth making. One is not
#: enough: the model needs a round to act on it and a round to answer.
MIN_ROUNDS_LEFT = 2

#: Words that narrow a search without naming the thing being searched for.
#: Dropping them widens the text without changing its subject.
_QUALIFIERS = re.compile(
    r"\b(exactly|at least|at most|precisely|only|both|either|alternative|"
    r"requires?|requiring|uses?|using|with|the|a|an|of|in|for)\b", re.I)


@dataclass
class Suggestion:
    rung: str
    message: str


def _searches(session) -> list:
    return [s for s in session.steps
            if s.tool == "search" and not s.error]


def _empty_hops(session) -> list:
    return [s for s in session.steps
            if s.tool in ("subjects_of", "objects_of") and not s.error and s.rows == 0]


def _shorten(text: str) -> str:
    """The same search with its qualifiers removed.

    'exactly two electrons' -> 'electrons'. The narrowing words are what make a
    label match fail; the noun is what the graph actually stores.
    """
    stripped = _QUALIFIERS.sub(" ", text)
    stripped = re.sub(r"\b(one|two|three|four|\d+)\b", " ", stripped, flags=re.I)
    return re.sub(r"\s+", " ", stripped).strip()


def next_suggestion(session, rounds_left: int) -> Optional[Suggestion]:
    """The next untried widening, or None when there is nothing honest to offer.

    None is a real outcome and the common one at the end: it means the ladder is
    spent, and the abstention should be allowed through.
    """
    if rounds_left < MIN_ROUNDS_LEFT:
        return None
    used = getattr(session, "widenings_used", set())

    # 1. A search that was narrowed by `kind`. The kind filter is the most
    #    common reason a search under-returns, and dropping it is one call.
    if DROP_KIND not in used:
        for step in _searches(session):
            kind = (step.args or {}).get("kind")
            text = (step.args or {}).get("text")
            if kind and text:
                return Suggestion(DROP_KIND, (
                    f"You have not tried search({text!r}) WITHOUT kind={kind!r}. "
                    f"That filter is often why a search under-returns: an entity "
                    f"the papers describe as something else is invisible to it."))

    # 2. A hop that could not have matched. `templates._why_empty` already put
    #    the compatible predicates in the tool's own reply, but the model has
    #    since read many more rows; repeating it at the decision point is where
    #    it can still be acted on.
    if COMPATIBLE_PREDICATE not in used:
        for step in _empty_hops(session):
            note = step.preview or ""
            if "could not have been otherwise" in note:
                return Suggestion(COMPATIBLE_PREDICATE, (
                    f"One of your hops was impossible, not empty: {note.strip()} "
                    f"You have not tried the predicates it named."))

    # 3. The search text itself. A label in the graph reads "ee+p (electron pair
    #    plus tagged forward proton)", which no phrasing of "exactly two
    #    electrons" will match; "electrons" might.
    if SHORTER_TEXT not in used:
        for step in _searches(session):
            text = (step.args or {}).get("text") or ""
            shorter = _shorten(text)
            if shorter and shorter.lower() != text.lower() and len(text.split()) > 1:
                return Suggestion(SHORTER_TEXT, (
                    f"Your search text was {text!r}. The graph stores labels as "
                    f"the papers write them, so the qualifiers are what fail to "
                    f"match, not the noun. You have not tried search({shorter!r})."))

    # 4. The terminal move. If anything at all was retrieved, this returns
    #    something, and something is what an abstention is claiming not to have.
    if TERMINAL_PAPERS_OF not in used:
        sets = [name for name, ids in (session.sets or {}).items() if ids]
        if sets:
            held = sum(len(ids) for ids in session.sets.values())
            return Suggestion(TERMINAL_PAPERS_OF, (
                f"You are holding {held} retrieved entities in {sets}. You have "
                f"not tried papers_of on them. If those entities are relevant, "
                f"the papers they occur in are the answer; if they are not, say "
                f"which ones you rejected and why."))

    return None


def should_widen(session, *, reason: str, rounds_left: int,
                 enabled: bool) -> Optional[Suggestion]:
    """Whether to push back on this abstention, and with what.

    Every condition is a reason a suggestion would be useless or unsafe, not a
    reluctance to make one:

      enabled        it is an arm, never a default, until measured
      not_in_graph   never interferes with a real answer
      rounds_left    a suggestion with no round to act on is noise
      retrieved      nothing retrieved means nothing to widen FROM, and the
                     abstention is very likely correct
      ladder         when it is spent there is nothing honest left to say
    """
    if not enabled or reason != "not_in_graph":
        return None
    if not (session.sets or session.known_entity_ids):
        return None
    return next_suggestion(session, rounds_left)


#: A run that ANSWERS after looking once. Distinct from the abstention case and
#: far more common: measured on 207 Qwen records, 135 made exactly one search,
#: 69 made none, and only 3 made two. It looks once and concludes.
#:
#: That matters because the round it declines to spend is the valuable one. On
#: conceptB-200, runs that took 4 rounds scored count_correct 0.158 against
#: 0.063 for runs that took 3 -- and Qwen stops at 3.25 on average, well short
#: of a budget of 6 it is never denied.
MAX_SEARCHES_FOR_THIN = 1


def should_push_further(session, *, reason: str, rounds_left: int,
                        enabled: bool) -> Optional[Suggestion]:
    """Whether an ANSWER was reached too cheaply to trust, and what to try.

    The mirror of `should_widen`. That one catches a run giving up with rows in
    hand; this catches a run concluding without having looked properly. Same
    ladder, same finiteness, same once-per-rung discipline -- only the trigger
    differs.

    THE RISK IS THE OPPOSITE ONE, and it is why the trigger is narrow. Pushing
    an abstention can only turn a refusal into an answer; pushing an ANSWER can
    turn a good one into a worse one, because the model may replace a precise
    set with a broader one it likes less. So this fires only where the answer
    was reached without looking: one search or none. A run that searched twice
    and concluded has done the work, and is left alone.
    """
    if not enabled or reason != "answered":
        return None
    searches = len(_searches(session))
    if searches > MAX_SEARCHES_FOR_THIN:
        return None
    if not (session.sets or session.known_entity_ids):
        return None
    return next_suggestion(session, rounds_left)


PUSH_MESSAGE = (
    "You answered after {searches} search(es), with {rounds_left} rounds left.\n\n"
    "{suggestion}\n\n"
    "If your answer already covers what that would find, repeat it unchanged -- "
    "a confirmed answer is a better answer. If it does not, revise it."
)

WIDEN_MESSAGE = (
    "Before you answer 'not in the graph' -- you have {rounds_left} rounds left "
    "and you are holding retrieved rows.\n\n{suggestion}\n\n"
    "Either try that, or answer 'not in the graph' again and say what you "
    "rejected and why. A refusal with a reason is a good answer here; a refusal "
    "with an untried route is not."
)
