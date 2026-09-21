"""The brake.

Every other mechanism in this package pushes the planner to keep going. The
sub-goal block tells it what it has not covered, the status block tells it what
it has, the completeness check tells it what is still missing -- three ways of
saying *continue*. Nothing in the loop has ever said *stop*, and on a model with
the stamina to use the whole budget that omission is expensive.

Measured on the supervisor's nine (Table~4.8): qwen3.8-flash runs 11.4 rounds,
issues 22.8 tool calls and touches 55.3 of the 60 papers in the corpus. It
reaches 0.98 of the gold -- and its retrieval precision is 0.48. With the status
block it reaches *all* of the gold at 0.46. The agent has read the library and
under half of what it judged is right, so the answer critic is carrying the
entire task and the graph traversal is no longer what finds the papers. QwQ
looks selective by comparison (0.68 reach at 0.67 precision) only because it
quits at 3.2 rounds; it is not choosing better, it is stopping sooner.

So the missing signal is not "what have I covered" but "is more searching still
paying". This block supplies it, and it is deliberately DETERMINISTIC -- no
model call, no judge. Two reasons. A brake implemented as another LLM opinion
would cost a call per round and confound "a stopping rule helped" with "a second
model looked at it", which is the confound `REVIEWER_MODEL` exists to avoid.
And the quantity that matters is arithmetic: how many papers the last round
added that the run did not already hold. A model does not need to be asked.

The instruction is conditional on purpose. An unconditional "stop when
saturated" would cost recall on the questions where the remaining gold is
genuinely hard to reach, which is the failure mode the sub-goal family was built
to fix -- and a brake that undoes the accelerator is not an improvement. So the
block reports the arithmetic, states that stopping is a correct action rather
than a surrender, and leaves the decision with the planner.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: The marker every rendered block carries, so `plan()` can strip the previous
#: round's copy. The status block is replaced and never appended for exactly
#: this reason: the message list grows each round, so appending would leave one
#: stale coverage report per round and the planner would have to work out which
#: of them is current.
MARKER = "COVERAGE SO FAR"

#: A round adding fewer than this many papers the run did not already hold is
#: not paying for itself. Two rather than one: a single new paper can be the one
#: that matters, and a threshold that fires on it would brake a productive run.
MIN_NEW_PAPERS = 2

#: Consecutive unproductive rounds before the block says so. One quiet round is
#: ordinary -- a `describe` or a `quotes` call reads what is already held and
#: adds nothing by design. Two in a row is a pattern.
SATURATION_ROUNDS = 2


def enabled() -> bool:
    """Whether the brake is fitted on this run."""
    return os.environ.get("SUFFICIENCY", "") == "1"


def papers_for(conn, entity_ids) -> set:
    """The papers reachable from the entities retrieved so far.

    The same quantity `_papers_of` computes for the record at the end of a run,
    asked once per round instead. It is a local sqlite read over an id list, so
    the cost is not comparable to the model call this deliberately avoids.
    """
    ids = [str(e) for e in (entity_ids or []) if e]
    if not conn or not ids:
        return set()
    out = set()
    # Chunked because sqlite caps variables per statement (999 by default) and
    # a saturating run is precisely the one that accumulates enough entities to
    # hit it -- the failure would arrive only on the runs this measures.
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        marks = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                "SELECT DISTINCT paper_id FROM entity_occurrence "
                "WHERE entity_id IN (%s)" % marks, tuple(chunk)).fetchall()
        except Exception as exc:  # noqa: BLE001 -- a hint must not kill a run
            logger.warning("sufficiency: paper lookup failed: %s", exc)
            return out
        out.update(str(r[0]) for r in rows if r and r[0])
    return out


def update(state, conn, entity_ids) -> dict:
    """Recompute coverage for this round and fold it into `state`.

    Returns the stats the block renders. Mutates `state` so the next round can
    tell what changed: the marginal yield is the whole point and it is not
    recoverable from a single round's view.
    """
    held = papers_for(conn, entity_ids)
    before = set(state.get("sufficiency_papers") or ())
    new = held - before
    stale = int(state.get("sufficiency_stale") or 0)
    # A round with no tool calls at all is not evidence of saturation -- there
    # was nothing for it to add. Only rounds that actually searched count.
    if before or new:
        stale = stale + 1 if len(new) < MIN_NEW_PAPERS else 0
    state["sufficiency_papers"] = held
    state["sufficiency_stale"] = stale
    return {"papers": len(held), "new": len(new), "stale_rounds": stale,
            "saturated": stale >= SATURATION_ROUNDS}


def render(stats: dict, rounds_left: int) -> str:
    """The block the planner sees. Reports, then permits -- never orders."""
    papers = stats.get("papers") or 0
    new = stats.get("new") or 0
    lines = [MARKER,
             "You hold %d paper%s. The last round added %d the run did not "
             "already have." % (papers, "" if papers == 1 else "s", new)]

    if stats.get("saturated"):
        lines.append(
            "Your recent rounds have added almost nothing new. More searching "
            "of the same kind is not finding more papers.")
        lines.append(
            "If every part of the question now has evidence behind it, ANSWER "
            "NOW with what you hold. Stopping because you have enough is the "
            "correct move and is scored as success, not as giving up: an "
            "answer naming the right papers beats a longer search naming more "
            "of them. If some specific part of the question still has nothing "
            "behind it, say which part and search for THAT -- not more of what "
            "you have already covered.")
    elif new < MIN_NEW_PAPERS:
        # Reported but not escalated. One quiet round is ordinary, and a block
        # that cried saturation on it would train the planner to ignore this
        # section by the time the signal is real.
        lines.append(
            "That is a thin round. If the next one is as thin, you have "
            "probably found what is there.")
    else:
        lines.append(
            "Keep going while rounds are still productive, and stop when they "
            "are not.")

    if rounds_left <= 1:
        lines.append(
            "This is your last round. Answer with what you hold.")
    return "\n".join(lines)


def replace_block(messages: list, block: str) -> list:
    """Swap the previous round's coverage report for this one.

    REPLACED, NEVER APPENDED. The message list grows every round, so appending
    would leave one stale coverage report per round and the planner would have
    to work out which of them is current -- while the older copies, which say
    the run is still productive, actively contradict the newest one. The
    sub-goal block carries the same rule for the same reason.
    """
    kept = [m for m in (messages or [])
            if not (m.get("role") == "system" and MARKER in (m.get("content") or ""))]
    return kept + [{"role": "system", "content": block}]
