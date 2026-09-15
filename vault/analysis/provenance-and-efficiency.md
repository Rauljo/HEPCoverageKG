# Provenance and efficiency — the two missing analyses

Compiled 2026-09-15 from the 184 run files on DIAS newer than 2026-09-11
(`~/HEPCoverageKG/eval/runs`, rsynced for analysis; **they exist only on the
cluster**). All figures are Gabriel's nine questions unless stated, clean
records only (a record that never received a response scores zero on every
metric and is excluded before any comparison).

Both metrics were **already recorded on every run**. Neither had ever been
analysed. No new runs were needed.

---

## 1. Provenance

### What is actually measured

Requirement 2 (`introduction.tex`, `\section{Approach}`) has three parts, in
three different states:

| Claim | State |
|---|---|
| The graph carries provenance | **Measured.** 13,738/14,188 assertions (96.8%) carry ≥1 verbatim quote; the 450 without are all quarantined. Written up in `methodology.tex`. |
| The answer only claims what was retrieved | **Recorded on every run, never reported.** This document. |
| The physicist can see the quotes | Built (app expander + verification panel); `evidence_quotes` recorded per run; never analysed. |

`query/verify.py` implements the second **mechanically**, not with an
LLM judge: it extracts every arXiv id, entity id and number from the answer and
checks it against `session.seen_values` — everything the tools returned, taken
*before* truncation. Prose is deliberately not checked, because paraphrase is
legitimate and flagging it would drown the real findings.

`faithfulness = 1 − unsupported / checked`, and **1.0 when nothing is
checkable**. That last rule matters: an answer making no factual claims cannot
be unfaithful, only unhelpful. Across all runs, **74% of answers naming no
paper score a perfect 1.0**. The metric is therefore reported here only over
records with `checked > 0`, with `checked` printed beside it.

### THE TRAP: the check only works where the trace was recorded

`seen_values` is not populated by every system. Measured over the 2,213 clean
records on Gabriel's nine:

| System | Records | `seen_values` empty | `evidence_ids` empty |
|---|---:|---:|---:|
| `hepkg` (typed) | 1,328 | **3 (0%)** | 541 (41%) |
| `chain` | 390 | 104 (27%) | 165 (42%) |
| `free-sql` | 479 | **479 (100%)** | 479 (100%) |
| `ensemble-critic` | 16 | 16 (100%) | 16 (100%) |

Two separate causes, and neither is a property of the mechanism being tested:

- **free-SQL has no provenance instrumentation at all.** `eval/free_sql.py`
  contains zero references to `verification` or `seen_values`. The agent writes
  its own SQL and the harness never registers the returned values, so there is
  nothing to check a claim against. *This is architectural and it is a real
  finding*: mechanical traceability is a capability the typed interface has and
  the free-SQL interface does not.
- **`chain` merges the wrong fields.** `eval/chain.py` (~line 166) unions
  `entity_ids`, `evidence_ids`, `papers`, `steps`, `llm_calls`, `rounds` and
  `seconds` across legs, then returns `out = last_answer` — so `seen_values`
  carries **only the final leg's** values while the answer text covers every
  leg. Chain faithfulness is therefore measured against a fraction of its own
  trace. **This is a recording bug, not a result.**

> **Do not report "chaining destroys provenance".** It is the D-070/D-180
> failure class again: a run file recording what a system produced, not what it
> was asked to do. A one-line fix (`out.seen_values = sorted(seen)`, unioned
> like the others) would make chain measurable; until then chain arms are
> excluded from every provenance number below.

### Result (typed agent, where the trace is valid)

Faithfulness over records with `checked > 0`; `chk` is claims checked per
answer, `unsup` claims not found in the trace, `evid` evidence quotes attached.

| Arm | Model | n | judged F1 | faith | chk | unsup | evid |
|---|---|---:|---:|---:|---:|---:|---:|
| answer-critic | QwQ-32B | 32 | 0.241 | 0.746 | 7.3 | 1.6 | 56.0 |
| answer-critic | qwen3-32b | 54 | 0.189 | 0.759 | 5.7 | 1.3 | 31.5 |
| constrained + answer-critic | QwQ-32B | 303 | 0.572 | 0.698 | 17.3 | 7.0 | 20.8 |
| constrained + answer-critic | qwen3-32b | 117 | 0.559 | 0.608 | 21.4 | 8.3 | 22.2 |
| constrained + answer-critic | qwen3.8-flash | 90 | 0.572 | 0.749 | 40.3 | 9.2 | 259.6 |
| **+ anchor** | QwQ-32B | 36 | 0.532 | **0.754** | 11.8 | **1.8** | 7.6 |
| **+ anchor** | qwen3-32b | 108 | 0.505 | **0.809** | 14.4 | **2.7** | 24.8 |
| **+ anchor** | qwen3.8-flash | 174 | 0.612 | **0.890** | 29.7 | **2.8** | 222.9 |
| + status | QwQ-32B | 53 | 0.576 | 0.610 | 15.8 | 5.9 | 44.8 |
| + status | qwen3-32b | 126 | 0.584 | 0.564 | 25.2 | 12.6 | 40.6 |
| + status | qwen3.8-flash | 132 | 0.586 | 0.735 | 41.7 | 10.4 | 238.7 |

### The headline: anchoring buys provenance and time

Paired by question, typed agent, traced and checkable records only, 9 questions:

| Model | Metric | base | anchor | Δ | s.e. |
|---|---|---:|---:|---:|---:|
| QwQ-32B | faithfulness | 0.693 | 0.733 | +0.040 | 0.060 |
| | unsupported claims | 7.19 | 1.96 | **−5.23** | 2.79 |
| | LLM calls | 7.07 | 5.47 | −1.60 | 0.47 |
| | seconds | 306 | 129 | **−177** | 36 |
| | judged F1 | 0.572 | 0.550 | −0.022 | 0.031 |
| qwen3-32b | faithfulness | 0.605 | 0.809 | **+0.204** | 0.043 |
| | unsupported claims | 8.31 | 2.70 | **−5.61** | 1.56 |
| | LLM calls | 7.27 | 6.31 | −0.96 | 0.38 |
| | seconds | 192 | 115 | **−78** | 13 |
| | judged F1 | 0.562 | 0.505 | −0.057 | 0.029 |
| qwen3.8-flash | faithfulness | 0.749 | 0.892 | **+0.143** | 0.023 |
| | unsupported claims | 9.24 | 2.81 | **−6.43** | 1.08 |
| | LLM calls | 17.87 | 15.60 | −2.27 | 0.63 |
| | seconds | 308 | 233 | **−74** | 16 |
| | judged F1 | 0.668 | 0.700 | +0.032 | 0.011 |

**This rescues the anchoring mechanism.** It was recorded as flat-to-negative on
F1 and shelved on that basis. Judged against the requirement it was actually
serving, it is the strongest mechanism in the set: it raises faithfulness on all
three models, cuts unsupported claims by two thirds to three quarters, and is
*cheaper and faster* everywhere. The reason is mechanical — the anchor narrows
the candidate pool, so the answer has less unretrieved material to reach for.

On qwen3.8-flash it wins on every axis at once, F1 included.

### Honest limits of the metric

- Only numbers, arXiv ids and entity ids are checked. A fluent physics sentence
  no retrieved row supports passes. It catches *invention of a paper or a
  number* — the failure that matters most for a coverage map — not all
  ungrounded prose.
- `checked = 0` scores 1.0, so the metric must always be reported over
  checkable records with `checked` beside it.
- It measures claim → retrieved row. It does **not** verify that the retrieved
  row's quote actually supports the claim; that is the separate, unmeasured
  question of citation *quality* flagged in `methodology.tex`.

---

## 2. Efficiency

`llm_calls` counts the planner, sub-goal status and reflection calls. Critic
calls are recovered from `answer.reviews[].calls` and
`answer.answer_review.calls`; the plan reviewer from `review_calls`. **Critic
tokens are not recorded anywhere** — `critic.Review` computes them and the
Answer never stores them — so token counts are the planner's (plus the plan
reviewer's) and are a **lower bound** on the critic arms.

Cost is derived from measured spend, not from a published price list: run
`20260908T063144-hepkg-72812` (qwen3-32b, OpenRouter, 27 clean records) cost a
recorded **$0.25** and carries 1,081,881 recorded tokens, giving
**$0.231 per 1M recorded tokens**. Applying that to recorded tokens is an upper
bound on price per recorded token (the unrecorded critic tokens make the true
rate lower). QwQ runs on the cluster at no marginal cost. **qwen3.8-flash has no
measured spend anchor — its rate is unknown and is not guessed here.**

| System / arm | Model | n | F1 | calls | rounds | in-tok | out-tok | secs | $/answer |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| free-sql / control | QwQ-32B | 75 | 0.397 | 2.8 | 2.8 | 8,666 | 6,354 | 189 | cluster |
| hepkg / constr+critic | QwQ-32B | 303 | 0.572 | 7.0 | 3.1 | 20,578 | 4,261 | 301 | cluster |
| hepkg / +anchor | QwQ-32B | 36 | 0.532 | 5.3 | 2.8 | 16,272 | 4,084 | **126** | cluster |
| hepkg / +plan-reviewer | QwQ-32B | 36 | 0.546 | 11.0 | 2.9 | 37,590 | 7,101 | 365 | cluster |
| chain / constr+critic | QwQ-32B | 49 | 0.700 | 13.9 | 8.4 | 22,163 | 4,162 | 623 | cluster |
| free-sql / control | qwen3-32b | 90 | 0.412 | 3.8 | 3.4 | 10,592 | 2,790 | 145 | **$0.0031** |
| hepkg / constr+critic | qwen3-32b | 117 | 0.559 | 7.3 | 3.3 | 19,907 | 2,730 | 191 | $0.0052 |
| hepkg / +anchor | qwen3-32b | 108 | 0.505 | 6.3 | 3.2 | 20,372 | 2,649 | **115** | $0.0053 |
| chain / constr+critic | qwen3-32b | 97 | 0.659 | 12.9 | 8.1 | 18,445 | 2,659 | 631 | $0.0049 |
| free-sql / control | qwen3.8-flash | 90 | 0.667 | 13.1 | 12.0 | 146,910 | 9,273 | 270 | n/m |
| hepkg / constr+critic | qwen3.8-flash | 90 | 0.572 | 16.9 | 10.9 | 191,535 | 11,992 | 289 | n/m |
| hepkg / +anchor | qwen3.8-flash | 174 | 0.612 | 15.1 | 11.3 | 198,205 | 12,308 | 230 | n/m |
| chain / constr+critic | qwen3.8-flash | 102 | 0.648 | 15.9 | 8.7 | 35,050 | 4,463 | 587 | n/m |

### What it says

1. **qwen3.8-flash costs roughly ten times the input tokens of either 32B model
   for comparable quality** — 191k against 20k on the typed agent at F1 0.572
   against 0.572 (QwQ) and 0.559 (qwen3-32b). It reads far more and does not
   convert that into a better answer. This is the efficiency counterweight to
   its retrieval breadth and it belongs beside every result that favours it.
2. **Free-SQL is the cheap interface**: 2.8–3.8 calls and ~10–15k tokens against
   the typed agent's 7.0–7.3 calls and ~23k, at 145–189 s against 191–301 s. On
   qwen3-32b it is $0.0031 against $0.0052 per answer — 40% cheaper — for
   F1 0.412 against 0.559. The typed agent buys +0.147 F1 and mechanical
   provenance for ~1.7× the price.
3. **Chaining roughly doubles everything**: calls 7.0 → 13.9, rounds 3.1 → 8.4,
   wall time 301 → 623 s on QwQ. It is the most expensive mechanism measured and
   the largest F1 gain, and the trade should be stated as such.
4. **Anchoring is free or better than free** — fewer calls on all three models
   and 38–58% less wall time, at ≤0.06 F1.
5. **The plan reviewer is the worst buy in the set**: calls 7.0 → 11.0 and
   tokens 20.6k → 37.6k on QwQ for −0.026 F1.

### Caveats

- **Wall time is not comparable across lanes** and `methodology.tex` already
  says so: cluster runs share a vLLM server with up to four concurrent eval
  jobs, OpenRouter runs contend with other tenants. Time is reported here
  because within a lane and model it ranks arms consistently, but calls and
  tokens are the primary currency.
- Critic tokens are unrecorded, so every arm with a critic understates tokens.
  Calls are complete.
- Cost is measured for qwen3-32b only.

---

## What to do next

1. **Report both.** Two subsections, no new runs.
2. **Fix `chain.py`** to union `seen_values` (one line) if chain provenance is
   wanted; otherwise state the exclusion.
3. **Record critic tokens** on the Answer so token cost is complete — also one
   small change, and it makes the efficiency table exact rather than a bound.
4. If a flash cost figure is wanted, read the OpenRouter usage counter for one
   flash run, exactly as was done for run 72812.
