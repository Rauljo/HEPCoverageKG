"""The widening ladder, and the guarantee that it stops.

Raul's objection to persistence was that it would make the agent "go in loops
infinitely without actually finding anything". The answer is not a promise about
the model -- it is that there are four rungs, each offered at most once, and
when they are spent the abstention goes through.
"""
from __future__ import annotations

import pytest

from hepcoveragekg.query import planner, widen


def _session(steps=(), sets=None, used=()):
    s = planner.Session(question="q")
    s.steps = list(steps)
    s.sets = dict(sets or {})
    s.widenings_used = set(used)
    return s


def _step(tool, args, rows=0, preview="", error=None):
    return planner.Step(1, tool, args, rows=rows, preview=preview, error=error)


def test_the_ladder_is_finite_and_then_lets_the_abstention_through():
    """THE guarantee. Offer suggestions until they run out and confirm they do."""
    s = _session(
        steps=[_step("search", {"text": "exactly two electrons", "kind": "selection_requirement"}, rows=8),
               _step("subjects_of", {"predicate": "region_requires_object"}, rows=0,
                     preview="0 rows, and it could not have been otherwise: ...")],
        sets={"set_1": ["e1", "e2"]})
    offered = []
    for _ in range(20):                      # far more turns than rungs
        sugg = widen.next_suggestion(s, rounds_left=5)
        if sugg is None:
            break
        offered.append(sugg.rung)
        s.widenings_used.add(sugg.rung)
    assert sugg is None, "the ladder must run out"
    assert len(offered) <= len(widen.LADDER)
    assert len(offered) == len(set(offered)), "a rung must never be offered twice"


def test_each_suggestion_names_a_concrete_untried_call():
    """Never 'try harder'. Generic exhortation is what produces flailing."""
    s = _session(steps=[_step("search", {"text": "exactly two electrons",
                                         "kind": "selection_requirement"}, rows=8)],
                 sets={"set_1": ["e1"]})
    sugg = widen.next_suggestion(s, rounds_left=5)
    assert sugg.rung == widen.DROP_KIND
    assert "exactly two electrons" in sugg.message
    assert "selection_requirement" in sugg.message


def test_it_will_not_fire_without_room_to_act():
    """A suggestion with no round left to use it is noise."""
    s = _session(steps=[_step("search", {"text": "x y", "kind": "k"}, rows=3)],
                 sets={"set_1": ["e1"]})
    assert widen.next_suggestion(s, rounds_left=1) is None
    assert widen.next_suggestion(s, rounds_left=widen.MIN_ROUNDS_LEFT) is not None


def test_it_never_touches_a_real_answer():
    """It fires on 'not_in_graph' and nothing else. A system taught never to
    abstain fabricates coverage, which is worse than the problem being fixed."""
    s = _session(steps=[_step("search", {"text": "x y", "kind": "k"}, rows=3)],
                 sets={"set_1": ["e1"]})
    assert widen.should_widen(s, reason="answered", rounds_left=5, enabled=True) is None
    assert widen.should_widen(s, reason="not_in_graph", rounds_left=5, enabled=True)


def test_it_is_off_unless_the_arm_is_on():
    s = _session(steps=[_step("search", {"text": "x y", "kind": "k"}, rows=3)],
                 sets={"set_1": ["e1"]})
    assert widen.should_widen(s, reason="not_in_graph", rounds_left=5, enabled=False) is None


def test_an_abstention_with_nothing_retrieved_is_left_alone():
    """Nothing retrieved means nothing to widen FROM, and the abstention is very
    likely correct. Pushing there would be pushing toward invention."""
    s = _session(steps=[_step("search", {"text": "x", "kind": "k"}, rows=0)])
    assert widen.should_widen(s, reason="not_in_graph", rounds_left=5, enabled=True) is None


def test_shortening_drops_the_qualifiers_and_keeps_the_noun():
    """The graph stores 'ee+p (electron pair plus tagged forward proton)'. No
    phrasing of 'exactly two electrons' matches that; 'electrons' might."""
    assert widen._shorten("exactly two electrons") == "electrons"
    assert widen._shorten("at least three leptons") == "leptons"
    assert "HistFitter" in widen._shorten("uses the HistFitter framework")


def test_the_terminal_rung_needs_something_to_fall_back_on():
    s = _session(steps=[], sets={})
    assert widen.next_suggestion(s, rounds_left=5) is None
    s2 = _session(steps=[], sets={"set_1": ["e1", "e2", "e3"]})
    sugg = widen.next_suggestion(s2, rounds_left=5)
    assert sugg.rung == widen.TERMINAL_PAPERS_OF and "3 retrieved entities" in sugg.message
