# Citation-grounded generation: baseline vs the three answer-side changes

Compiled 2026-09-10 from D-143, D-145, D-146, D-153 (+3 addenda), D-155 (parts 1-5),
D-156 (parts 1-4). Only measured pairs appear; "--" means not run. Every arm in
the cluster table is QwQ-32B-AWQ, no critic. The three changes:

- typed + name-ids: the answer instruction "write the arXiv ids" (--name-ids).
- free-SQL + id instruction: "put the ids in `papers`, that is what gets scored"
  (the original free-SQL prompt; the baseline table now uses the plain variant).
- constrained: one extra call at the answer exit that picks ids from an enum of
  every reachable paper under a JSON schema (CONSTRAINED_IDS=1, vLLM guided_json);
  set questions only (shape gate), so per-paper and count rows are n/a.

## Table 1 -- cluster lane (DIAS, QwQ, no critic)

| Question type / metric | typed control | typed + name-ids | typed + constrained | free-SQL plain | free-SQL + id instr. | free-SQL + constrained |
|---|---|---|---|---|---|---|
| **Retrieval, 84 q x 2 (168)** | wave 2 | 54288 | 54296 | 54289 | wave 2 | -- |
| set_f1 | 0.200 | 0.221 | 0.271 | 0.264 | 0.268 | -- |
| named-only F1 (P / R) | 0.202 (0.18 / 0.32) | 0.224 (0.21 / 0.36) | 0.274 (0.21 / 0.69) | 0.306 (0.26 / 0.53) | 0.301 (0.24 / 0.54) | -- |
| essay-only set_f1 | 0.132 | 0.188 | 0.269 | 0.264 | 0.260 | -- |
| paired delta vs control (se) | -- | +0.024 (0.024) | +0.066 (0.023) | -- | +0.004 | -- |
| silent answers | 58 | 27 | 3 | 23 | 23 | -- |
| count written where a list was asked | 29 | 6 | -- | 4 | -- | -- |
| partial lists ("such as") | 21 | 5 | -- | 7 | -- | -- |
| complete lists | 15 | 16 | 78 | 64 | -- | -- |
| ids per answer | 8.6 | 19.4 | 31.2 | 16.0 | 23.1 | -- |
| **Concept-set, 14 q x 2 (28)** | wave 2 | 54261->54272 (critic ON) | -- | 54293 | wave 2 | -- |
| set_f1 | -- | -- | -- | 0.575 | 0.372 | -- |
| named-only F1 (P / R) | 0.422 | 0.416 -> 0.332 | -- | 0.644 (0.63 / 0.76) | (0.36 / 0.50) | -- |
| essay-only | 0.271 | 0.268 -> 0.309 | -- | 0.575 | -- | -- |
| **Count, 20 q x 2 (40)** | wave 2 | 54261->54272 (critic ON) | n/a (gated) | 54293 | wave 2 | n/a |
| exact / closeness | 0.30 / 0.36 | 0.20 -> 0.25 / 0.34 -> 0.30 | n/a | 0.53 / 0.59 | 0.35 / 0.54 | n/a |
| **Per-paper, 36 q x 2 (72)** | wave 2 | 54288 | n/a (gated) | 54289 | wave 2 | n/a |
| retrieved label recall | 0.97 | 0.97 | n/a | -- | -- | n/a |
| mentioned label recall (fuzzy) | 0.77 | 0.66 | n/a | 0.76 | 0.78 | n/a |
| **Gabriel's 9, judged_f1** | 54290 (x3) | -- (54294 cancelled) | 54292 (x3) | 54293 (x2) | wave 2 (x2) | 54297 (x2) |
| judged_f1 | 0.288 | -- | 0.554 | 0.303 | 0.303 | 0.245 |
| papers named / gold among them | 6.5 / 2.4 | -- | 15.5 / 5.5 | 8.7 / 2.8 | 7.4 / 2.8 | 7.1 / 1.9 |
| judged wrong / outside judged set | 1.2 / 2.9 | -- | 3.0 / 7.1 | -- | -- | -- |
| silent answers | 7 / 27 | -- | 0 / 27 | 6 / 18 | 5 / 18 | 4 / 18 |
| candidates offered / picked | -- | -- | 27.6 / 15.3 | -- | -- | 5.4 / 5.4 |
| stack + constrained (54295) | | | 0.543 (0.611 on 24, gf-08 context errors) | | | |

Notes for Table 1.
- The Gabriel typed control 54290 (0.288, 7 silent of 27) is lower than the wave-2
  no-critic control the baseline table uses (0.438, 18 answers). The constrained
  pair is same-session (54290 vs 54292), so its +0.27 is read against 54290;
  against the baseline-table control it is +0.12. Record spread 0.315 at n=27.
- Name-ids without the critic on Gabriel's 9 was not run on the cluster (54294
  cancelled). The critic-on single-repeat pair on the merged code is 0.330 ->
  0.476 (D-153); the honest cross-lane range for name-ids alone is +0.03 to +0.10.
- Concept-set and count rows for typed name-ids come from the critic-on pair
  54261 -> 54272 (D-153), the only run of name-ids on those types.
- The free-SQL "with id instruction" wave-2 rows ran on older code and another day;
  the concept-set and count gaps (0.372 -> 0.575, 0.35 -> 0.53) are larger than the
  prompt can explain and are treated as run-to-run variation (D-155 part 4).
- Free-SQL + constrained: candidates are the SQL result rows, all of them picked,
  none on gf-05/07/08; the mechanism has nothing to select from (D-156 part 2).

## Table 2 -- OpenRouter lane (qwen/qwen3-32b, Gabriel's 9, 3 repeats, 27 records, judged_f1)

| Arm | control | + the change | delta | named / gold (control -> change) | silent (control -> change) |
|---|---|---|---|---|---|
| typed, no critic -> + name-ids | 0.383 (84932) | 0.486 (2681) | +0.10 | 7.9 / 3.2 -> 12.8 / 4.3 | 3 -> 3 |
| typed, critic on -> + name-ids | 0.435 (1914) | 0.481 (99172) | +0.05 | 7.1 / 3.4 -> 10.9 / 4.3 | 4 -> 2 |
| free-SQL plain -> + id instruction | 0.453 (3469) | 0.571 (85223) | +0.12 | 7.8 / 3.4 -> 9.4 / 4.4 | 4 -> 1 |
| constrained | -- | -- | | not run: providers do not enforce the schema | |

Record spread 0.24-0.30 at n=27, so each single delta is within one spread;
the sign is the same on all four lane/critic pairs measured.

## LaTeX

```latex
\begin{table}[htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrr}
\toprule
& & \multicolumn{3}{c}{Typed} & \multicolumn{3}{c}{Free-SQL} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8}
Question type & Metric & control & +name-ids & +constr. & plain & +id instr. & +constr. \\
\midrule
\multirow{6}{*}{Retrieval (168)}
 & set F1 & 0.200 & 0.221 & \textbf{0.271} & 0.264 & 0.268 & -- \\
 & named-only F1 (P/R) & 0.202 (0.18/0.32) & 0.224 (0.21/0.36) & 0.274 (0.21/0.69) & 0.306 (0.26/0.53) & 0.301 (0.24/0.54) & -- \\
 & essay-only F1 & 0.132 & 0.188 & 0.269 & 0.264 & 0.260 & -- \\
 & silent answers & 58 & 27 & 3 & 23 & 23 & -- \\
 & complete lists & 15 & 16 & 78 & 64 & -- & -- \\
 & ids per answer & 8.6 & 19.4 & 31.2 & 16.0 & 23.1 & -- \\
\midrule
\multirow{2}{*}{Concept-set (28)}
 & named-only F1 & 0.422 & 0.332$^{\dagger}$ & -- & 0.644 & 0.36/0.50 (P/R) & -- \\
 & essay-only F1 & 0.271 & 0.309$^{\dagger}$ & -- & 0.575 & -- & -- \\
\midrule
Count (40) & exact / closeness & 0.30 / 0.36 & 0.25 / 0.30$^{\dagger}$ & n/a & 0.53 / 0.59 & 0.35 / 0.54 & n/a \\
\midrule
Per-paper (72) & label recall (fuzzy) & 0.77 & 0.66 & n/a & 0.76 & 0.78 & n/a \\
\midrule
\multirow{4}{*}{Gabriel's 9}
 & judged F1 & 0.288$^{\ddagger}$ & -- & \textbf{0.554} & 0.303 & 0.303 & 0.245 \\
 & named / gold & 6.5 / 2.4 & -- & 15.5 / 5.5 & 8.7 / 2.8 & 7.4 / 2.8 & 7.1 / 1.9 \\
 & silent answers & 7/27 & -- & 0/27 & 6/18 & 5/18 & 4/18 \\
 & candidates / picked & -- & -- & 27.6 / 15.3 & -- & -- & 5.4 / 5.4 \\
\bottomrule
\end{tabular}
\caption{The three answer-side changes against their controls on the cluster lane
(QwQ-32B-AWQ, no critic). Typed name-ids: the instruction to write arXiv ids in the
answer. Free-SQL id instruction: ``put the ids in \texttt{papers}, that is what gets
scored''. Constrained: a final call that selects ids from an enumeration of every
reachable paper under a JSON schema; applied to set questions only, so count and
per-paper rows are n/a. $^{\dagger}$From the critic-on pair 54261$\to$54272, the only
run of name-ids on those types; its control values there were 0.416 / 0.268 and
0.20 / 0.34. $^{\ddagger}$Same-session control for the constrained pair; the
baseline table's typed control on these questions is 0.438. Free-SQL rows with the
id instruction come from an earlier run on older code; the concept-set and count
gaps exceed what the prompt can explain.}
\label{tab:citation-grounding}
\end{table}

\begin{table}[htbp]
\centering
\small
\begin{tabular}{lrrrll}
\toprule
Arm (OpenRouter, qwen3-32b) & control & +change & $\Delta$ & named / gold & silent \\
\midrule
Typed, no critic $\to$ +name-ids & 0.383 & 0.486 & +0.10 & 7.9/3.2 $\to$ 12.8/4.3 & 3 $\to$ 3 \\
Typed, critic on $\to$ +name-ids & 0.435 & 0.481 & +0.05 & 7.1/3.4 $\to$ 10.9/4.3 & 4 $\to$ 2 \\
Free-SQL plain $\to$ +id instruction & 0.453 & 0.571 & +0.12 & 7.8/3.4 $\to$ 9.4/4.4 & 4 $\to$ 1 \\
\bottomrule
\end{tabular}
\caption{The two prompt changes on the OpenRouter lane, Gabriel's nine questions,
three repeats (27 records), judged F1. Constrained selection was not run on this
lane because the providers do not enforce the schema. Record spread 0.24--0.30.}
\label{tab:citation-grounding-or}
\end{table}
```
