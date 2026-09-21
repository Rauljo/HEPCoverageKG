# Overnight arm sweep — predictions, written BEFORE the runs

Set: `discriminating-2026-09-03.jsonl` (164 questions). Model: QwQ-32B-AWQ,
typed, critic on. Control: jobs 54163/54164, same set, same conditions.
Control noise floor on this set: **0.034**.

Predicting first is the guard against fishing: with 164 questions x 4 pools x
per-paper flips, some slice will move by chance for every arm. A hit on a
prediction is evidence; anything else is a hypothesis for the next run (D-102,
and enforced by `arm_diff(predicted=...)`).

| arm | predicted effect | predicted target | why |
|---|---|---|---|
| `--reviewer` | negative | none | -0.061 typed when last measured; PoG's reflection is their 2nd-weakest, and our baseline is not weak |
| `--state-objective` | none | none | +0.017 last time, inside noise on every set so far |
| `--subgoals` | negative | none | worst arm of the whole project; PoG's own `w/o Memory` variant |
| `--subgoal-status` | UNKNOWN | gf-01, gf-01-condition | three contradictory readings (+0.104, then 0.403, then -0.065). Targets multi-condition questions, which is what it is for |
| `--path-tool` | none | gf-07 | three different numbers across three runs; ttZ needs a 2-hop walk, so gf-07 is the only place it should show |
| `--index-values` | positive on tierA | tierA | tierA answers are entity lists, and value surface forms are what it adds |
| `--index-quotes` | none | none | +0.047 once, never replicated |
| `--tool-examples` | UNKNOWN | gf-04, gf-08 | never run. Same family as `--simple-answer`: if the failure is the handoff, showing worked calls may help |

## Already running
`--simple-answer` (54174/54175) — predicted to fix **gf-04 and gf-08**, where
the model names 10+ papers and scores zero. Its docstring predicts the size:
typed 0.287 vs free-SQL 0.669 on the same questions, typed having the better
retrieval process and losing it at the handoff.
