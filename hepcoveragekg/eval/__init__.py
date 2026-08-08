"""
The evaluation harness.

Design: `vault/ideas/eval-harness-design.md`. Decisions: `vault/system.md` §4
(S-22 harness first, S-10 freeze/split/tag, S-32 three layers, S-36 ablations,
S-52 variance).

One entry point in (`questions`), one entry point out (run records), and
everything between is swappable: a new system is an adapter in `systems.py`, a
new metric is a function in `scoring.py`, and neither touches the other.
"""
from . import questions, report, runner, scoring, systems  # noqa: F401

__all__ = ["questions", "systems", "runner", "scoring", "report"]
