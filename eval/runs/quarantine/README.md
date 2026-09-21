# Quarantined runs

Runs kept rather than deleted, but held out of `eval/runs/` because their
stored `scores` would be read as a model's number and are not one.

## 20260830T205015-hepkg-48507.jsonl

deepseek-v4-flash-0731, typed, 4000-token cap, Llama-8B critic, seed 20260815.
The re-run intended to undo the 800-token truncation.

    all 24 records        judged_f1 = 0.218
    the 8 that completed  judged_f1 = 0.655
    9 timed out           scored 0 by construction
    (original at 800 tok  judged_f1 = 0.512)

**Why it cannot be quoted either way.** The 0.218 is not deepseek's score --
nine records are zeros because the client never received a response
(`TimeoutError: no answer within 600s`, `rounds=0, calls=0`, one record burning
3.4 hours through retries; the stacked-abandoned-request failure described at
planner.py:88-99). Not a model outage: a direct call to the same model returned
in 4.0s the same afternoon.

The 0.655 is not deepseek's score either, and this is the more dangerous
number. Those 8 records are selected by *having completed*, and the questions
that timed out (gf-02, gf-05, gf-08 twice each) are plausibly the ones needing
the longest generations -- so the subset is filtered toward the easy end by
exactly the mechanism under test. Quoting it would be selection bias in the
direction we want to believe.

What it does suggest, weakly: raising the cap moved deepseek UP, consistent
with the qwen3-32b result (free-SQL 0.199 -> 0.592). Unresolved, not refuted.

**Dropped 2026-08-31** at the supervisor's-student's direction -- paid models
to be revisited later. The timeout cause is undiagnosed and must be found
before any hosted re-run is trusted.
