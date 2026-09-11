# Planning mechanisms on the three question sets (2026-09-11)

Every arm below sits on the same base and changes one thing. Differences are
**paired by question** and reported with the paired standard error, because
the between-question spread is far larger than any effect here.

## 1. Gabriel's nine set questions -- judged F1

Base `--answer-critic`, QwQ-32B, control pooled over three runs
(54355+54356+54378, 42 records); each arm 18 records.

| arm | judged F1 | P | R | rounds | LLM calls | vs control |
|---|---|---|---|---|---|---|
| control | 0.522 | 0.715 | 0.475 | 3.06 | 3.1 | -- |
| sub-goal status, inline (54365) | 0.578 | 0.706 | 0.565 | 4.00 | 4.0 | **+0.056** (se 0.046) |
| sub-goal status, own call (54436) | 0.584 | 0.757 | 0.533 | 3.83 | 6.6 | **+0.062** (se 0.026) |
| plan reviewer (54432) | 0.584 | 0.791 | 0.534 | 2.83 | 2.8 | **+0.062** (se 0.021) |

Three things worth saying:

- **This is the first planning mechanism to move anything.** Every arm in the
  16-arm screen was inside the noise floor. Here all three are positive, and
  two are 2.4-3.0 paired standard errors out.
- **Splitting the status into its own call buys precision, not size.** The
  effect is the same (+0.062 vs +0.056) but the standard error halves, at 6.6
  LLM calls against 4.0. Raul asked for this split ("first ask about
  completion after an execution, and then write the next plan") and it is the
  right shape: the estimate is tighter because the status stops competing with
  the plan for the same generation.
- **The reviewer gets the same gain for free.** 2.8 calls, fewer than the
  control's 3.1, because a reviewed plan needs fewer rounds (2.83 vs 3.06). It
  also buys the gain differently -- precision 0.791 against the control's
  0.715, where the status arms buy recall. Those are different mechanisms and
  could compose.

n=9 questions. Nothing here resolves a small difference and it is not meant to.

## 2. The 36 per-paper questions -- label recall

| arm | retrieved | mentioned | fuzzy | rounds | calls | vs baseline (mentioned) |
|---|---|---|---|---|---|---|
| baseline (54392) | 0.972 | 0.331 | 0.781 | 2.69 | 2.7 | -- |
| reviewer (54394) | 1.000 | 0.358 | 0.817 | 2.62 | 2.7 | +0.027 (se 0.031) |
| all three (54393) | 1.000 | 0.338 | 0.768 | 2.79 | 3.1 | +0.007 (se 0.051) |
| persist (54403) | 0.986 | 0.332 | 0.776 | 2.60 | 2.6 | +0.001 (se 0.037) |
| reviewer + objective (54406) | 1.000 | 0.325 | 0.779 | 2.65 | 2.7 | -0.007 (se 0.056) |
| status, own call (54408) | 0.988 | 0.330 | 0.780 | 3.11 | 5.2 | -0.002 (se 0.062) |
| status, inline (54395) | 1.000 | 0.299 | 0.740 | 2.97 | 3.0 | -0.032 (se 0.042) |
| status, third run (54435) | 1.000 | 0.307 | 0.695 | 2.92 | 2.9 | -0.024 (se 0.068) |

**Nothing moves, and the reason is visible in the first column.** Retrieval is
at 0.97-1.00: the right rows are in hand on essentially every question. Of the
labels retrieved, 30-36% reach the answer. Every mechanism here acts on
planning, and planning is not the bottleneck on this set -- the handoff from
retrieved rows to written text is. Raul's hypothesis that these mechanisms
"may work better to get specific answers rather than sets" is not supported:
they do nothing here, and they do something on the sets.

54435 is reported as unverified. It carried `--subgoal-status` and the same
config hash as 54395, and nothing in the record said whether it was the
question-scope variant it was submitted as. That is fixed going forward --
every record now carries `subgoal_scope` and `reflect_mode`.

## 3. The seven batch-2 value questions -- facts stated

Base plain QwQ-32B, 3 repeats, 21 records per arm.

| arm | facts stated | silent | rounds | vs baseline |
|---|---|---|---|---|
| baseline (54421) | 0.36 | 0 | 2.90 | -- |
| **sub-goal status (54425)** | **0.44** | 0 | 4.33 | **+0.075** (se 0.041) |
| quotes indexed (54427) | 0.35 | 0 | 2.71 | -0.008 (se 0.047) |
| persist + both indexes (54430) | 0.35 | 0 | 2.81 | -0.008 (se 0.028) |
| both indexes (54428) | 0.34 | 0 | 2.76 | -0.024 (se 0.032) |
| values indexed (54426) | 0.31 | 0 | 2.67 | -0.056 (se 0.052) |
| reviewer+subgoals+status (54422) | 0.29 | 5 | 2.71 | -0.068 (se 0.076) |
| persist (54429) | 0.25 | 2 | 3.05 | -0.110 (se 0.059) |

- **Sub-goal status is again the only arm that helps**, and by the largest
  margin of the three sets. It is also the most expensive in rounds (4.33 vs
  2.90) -- it keeps the run going when the run would otherwise stop short,
  which is exactly what a value question needs.
- **The three together are worse than the status alone** (-0.068 against
  +0.075) and produce five silent answers. Composition is not free.
- **Indexing values and quotes does nothing on this metric.** The earlier
  finding that the indexed field bounds what is reachable (labels 26%, +quotes
  52%, +values 70%, both 96%) is about what the graph *contains*; it does not
  transfer to whether the answer *states the number*.
- **Two questions are at zero in every arm** (gf-10, gf-13). They are not
  noise-limited, they are unreachable, and they should be reported as such
  rather than averaged away.

## What this says for the write-up

One mechanism survives all three sets: **sub-goal status**, positive on the
sets (+0.06) and on the values (+0.075), flat on the per-paper questions. The
**plan reviewer** matches it on the sets at lower cost and by a different
route (precision, not recall). Everything else -- persist, the widening
ladder, state-objective, index variants, and all three composed -- is flat or
negative.
