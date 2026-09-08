# eval/analysis -- the scripts behind the 2026-09-07/08 decisions

Moved here from a session scratchpad so the numbers in `vault/decisions.md`
(D-118 .. D-142) can be reproduced. Each takes run files from `eval/runs/`.

- `compare_arms.py CONTROL.jsonl ARM.jsonl [label]` -- per-question judged_f1,
  reach, gold named and set sizes, arm vs control on Gabriel's 9 (3 repeats).
- `score_wave2.py DIR` / `score_wave3.py DIR` -- the 164-question cluster
  waves, paired by qid; wave 3 prints each column with its n (D-141: set_f1
  exists on 98 records, count_correct on 20, judged_f1 on 10).
- `gabriel_replay.py` -- D-118 loss decomposition, replaying a run's tool
  calls against a DB copy.
- `retrieval_bench.py`, `bench_index_variants.py` -- D-121 offline index reach.
- `score_2x2.py` -- D-123 decomposition 2x2.
- `run_judge.py`, `rank_judge.py`, `rank_judge_listwise.py`, `build_pairs.py`
  -- D-124 / D-129 judge-vs-Gabriel and ranker-capacity studies.

Paths inside some scripts point at the scratchpad they were written in
(`$SP/wave1`); pass the directory explicitly.
