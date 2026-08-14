# Ground truth — how the evaluation gold is built

**Version 1.0 · 2026-08-14**

*Methods reference for the evaluation chapter. Living: update it and bump the version whenever the
procedure changes. Changelog at the bottom.*

*Decisions behind it: D-054 … D-059 in [`decisions.md`](decisions.md). Code:
`hepcoveragekg/eval/`.*

---

## 1. The problem: why the supervisor's answers cannot be the gold

The 15 evaluation questions came with answers. Those answers **cannot be used as ground truth**, and
his own document says why: they were computed by **filtering the pilot export**. They are
*graph-agreement gold, not physics truth*.

The consequence is precise and fatal. Scoring our graph against them measures whether **two
pipelines built from one extraction agree with each other**. Wherever the extraction dropped
something, both pipelines drop it, and both score 100% while both are wrong. The error is invisible
by construction because it is common to both sides.

Three of his Tier-4 premises are already known to be wrong on exactly this mechanism (Q12, Q13,
Q14 are labelled "not in the graph" while the answers sit in stored evidence quotes).

**So the truth field is filled by reading the papers, and his answers are kept as a separate
`gabriel_gold` field in provenance — never loaded as `truth`.**

## 2. The question set

Source: `HEPKG_promopt_tests docs/onboarding/kg-evaluation-questions.md`, dated 2026-08-03. The
TEST split. Held in `hepcoveragekg/eval/supervisor.py`.

**Scope is derived from the question TEXT, mechanically.** A question is about one paper only if an
arXiv id appears *in the question itself*. `paper_scope()` is a regex, not a judgement call.

Reading scope off the *gold* instead is circular in the worst way. Q8 asks "which analyses require
exactly 2 electrons OR exactly 2 muons", and its gold is "exactly one: 2001.06899". Scoping the
check to that one paper could only ever confirm the answer — a second paper doing the same thing
would be **unfindable by construction**. This error was made once already, which is why the rule is
now mechanical.

**Two gold lists were recovered rather than accepted as missing.** gf-01 and gf-04 were recorded as
"a count, no papers", which made them unscoreable. Both are derivable from the analysis cards by the
same filter he used. Deriving them revealed the best result in the set (gf-04, 9/10) and the worst
(gf-01, 2/18) — both previously invisible.

## 3. The reference reader

`hepcoveragekg/eval/reader.py`. An independent second opinion, read from the papers.

### 3.1 What it is allowed to see

**`source_block` and nothing else.** Never `entity`, `assertion`, `entity_facet` or any derived
table. The moment the reader sees the graph it stops being independent and the exercise becomes a
second measurement of the same extraction.

`source_block` holds the whole paper as LaTeXML parsed it, in **paragraph-sized pieces** — 31,051
blocks across 60 papers, ~518 per paper. A block is *not* a section; the section is a label on each
block (`section_title`). Block kinds:

| kind | count | avg chars |
|---|---|---|
| paragraph | 18,109 | 566 |
| bibliography | 6,498 | 197 |
| caption | 2,487 | 413 |
| heading | 1,326 | 34 |
| **table** | **1,168** | **862** |
| list_item | 922 | 223 |
| equation | 449 | 262 |
| abstract | 32 | 1,518 |

Tables are present, which is how gf-15 recovered "observed 3, expected 5.7 ± 1.0" from a results
table. Figures are **not** — only their captions.

Bibliography, author lists and acknowledgments are dropped before reading. For 2001.06899 that is
134,271 chars stored → **72,013 chars actually read (54%)**, almost all of the difference being the
reference list.

### 3.2 Reading windows

Surviving blocks are repacked into windows of **≤12,000 characters** (109 blocks → 10 windows for
2001.06899). A window may span several sections, and its label reflects that.

The size is derived, not guessed: **3.46 chars/token measured** with the served model's own
tokenizer on this corpus (physics prose is dense with LaTeX and tokenises badly). Against an 8,192
context, 12,000 chars ≈ 3,470 tokens — half the ceiling, deliberately.

*An assumed 4 chars/token cost half of the first run's calls to context overflow, and
systematically: overflow kills the biggest windows, which are the content-rich ones holding the
answers. The run then reported confident negatives on answers already located by hand.*

### 3.3 Every claim carries a quote, checked mechanically

A model saying "yes, this paper uses an ABCD estimate" is an opinion. Saying so **and citing a
sentence that provably appears in the paper** is evidence.

Verification is a **word-level 8-gram match** against `source_block.text` (`MIN_MATCH_TOKENS = 8`).
Substring matching was tried first and rejected true quotes — the dangerous direction, because a
rejected true quote silently suppresses a correct answer and nothing looks wrong.

This catches **invention, not misreading**. Measured, only **43–45%** of verified quotes actually
answer the question asked. Hence the honest name for the reader's output: **corroborated gold**, not
truth — and hence the judge stage below, and ultimately the human.

### 3.4 Self-consistency, with disagreement surfaced rather than resolved

Each window is read **3 times at temperature 0.3**. Non-zero on purpose: at temperature 0 a model
repeats itself, and three identical samples would look like consensus while measuring nothing.

Verdicts are grouped **per window**, and the paper-level answer is:

- **True** — some window's samples agree it is there, with a verified quote
- **False** — every window agreed it is not, and at least one call was usable
- **None (split)** — a window's samples disagreed, or nothing usable came back

`None` is never collapsed into "no". It covers two things that must not hide inside a False: the
model genuinely disagreeing with itself, and *us never having successfully asked*. A dead endpoint
reporting "this paper does not do X" for all 60 papers is evidence of absence manufactured from an
outage.

**Grouping per window, not per paper, is load-bearing.** Repeats within one window are samples of
the same question and should agree. Verdicts from different windows answer different questions —
window 5 holds the ABCD sentence, window 12 is the detector description, and "not in this passage"
is correct for 22 of 23 windows. Pooling made unanimity impossible *exactly when the answer was
found*, so successful reads were marked unusable while papers where nothing was found agreed
trivially and reported a confident False. **Finding the answer was what made the result unusable.**

Sweep outcome across 420 paper-reads: **262 no, 87 split, 71 yes**.

### 3.5 Scale

420 paper-reads, **19,251 calls**, 0.2% hard errors, 97.3% usable, 1h43m on Mistral-Small-24B.

## 4. The cascade

Retrieve-then-verify, one level up from the query system's own design:

```
Mistral-Small-24B   recall     reads every window, gathers candidate sentences
QwQ-32B-AWQ         recall     the same, independently — a reasoning model
merge               union      not a vote
QwQ judge           precision  decides whether the gathered evidence answers the question
```

**The union is not a vote.** The two models fail in opposite directions (D-057): the 24B is a
literal reader and under-finds; QwQ reasons and over-finds. Requiring agreement would discard
exactly the evidence the second model was added to recover. The judge downstream is what stops the
union being credulous.

**Reason-before-verdict field ordering.** The JSON asks for the reasoning first and the verdict last.
Free, and it fixed 3 of 5 false positives. Worth stating honestly: a stated reason that *follows* the
token order is not thereby a faithful account of the computation (Turpin et al. 2023) — this buys
accuracy, not interpretability.

**The judge runs on a different endpoint from the reader**, because a model marking its own homework
is the self-preference problem (MT-Bench).

**Two judges, not one.** An *existence* question ("does this paper use b-tagging?") is settled by the
existence of a sentence. An *extraction* question ("what signal efficiency does the cut retain?") is
not — the paper can plainly have such a cut, with the number sitting right there, while the answer we
extracted is a different number entirely. `check_extraction` therefore judges **the value we
returned**, and names a better one if the evidence holds it.

## 5. Failures found and fixed

Recorded in full in D-056, D-058 and D-059. The pattern worth carrying into the writeup: **every one
of these reported success.**

### 5.1 The conjunction problem — one fault, five forms (D-058)

Most questions ask for several things at once. The harness was built for single-fact existence
questions and mishandled conjunctions in five places, **two of which presented as passes**:

1. The reader sees one window; no window shows all three parts → **ask one condition at a time, AND
   computed over the whole paper**
2. The judge handed one sentence and the whole question → **the judge sees every condition's quote**
3. The extractor stopping at the first hit → **EXTRACTION reads every window**
4. A third of an answer scoring as a pass → **still open; needs per-part gold**
5. Telling the recall stage to assess completeness made it return *nothing* → **the gather prompt
   collects, the judge decides**

The generalisation: **every one is deciding sufficiency at a stage that cannot see all the
evidence.** Gather wide, decide once, at the only point where everything is visible.

### 5.2 Two parser faults deleting evidence (D-059)

`parse_reply` discarded **599 complete, well-formed answers**:

- the JSON extractor took the last **brace-free** object, so a quote carrying `${\approx}4.8$`
  selected `{\approx}` and the whole reply was thrown away
- the 24B copies LaTeX into JSON strings verbatim (`"$\mathup{{{t}}}$"`), where `\m` is not a legal
  escape

**The loss was one-directional: 475 recovered YES, 0 recovered NO.** A "no" reply carries no quote,
so there is no LaTeX to choke on and it always parsed. The bug could only ever *delete* evidence —
and the harness reported the silence as "the paper does not say".

Raw replies are stored on every verdict precisely so a parser fix is retroactive: `reparse()`
recovered all of it with no GPU cost. The one path that did **not** store raw replies (the gf-01
conditions run) had to be re-read from scratch; it now stores them.

## 6. What the fixes were actually worth

Measured with one variable changed at a time — after three invalid comparisons of my own (wrong
baseline file; different judge models on the two sides; `repeats` changed from 2 to 3 at the same
time as the parser).

| | before | after |
|---|---|---|
| sweep, 60 papers, same stage and judge | 15 gold | **16 gold, +12 false positives** |
| gf-01 (conditions, repeats=2) | 6/18 | **8/18, zero false positives** |
| gf-11 | 0 evidence in 60 calls | **complete match to gold** |
| single-paper value judge | 7/7 "supported" (uninformative) | **5 upheld, 2 rejected** |

The parser fix is **close to score-neutral on the corpus sweep and worsens precision there** (gf-08:
7 found → 16, every extra one wrong). It is **decisive** where the answer is LaTeX-dense: gf-01's
conjunction, where a lost YES on any single condition fails the whole paper, and the single-paper
extraction questions.

**`repeats` is not a free knob.** Consensus requires the repeats within a window to agree, so a third
sample can only make a yes *harder* to reach. Raising it silently tightens the threshold.

**The conclusion that reframes the next stage: recall is no longer the bottleneck — judge precision
is.** Every sweep question except gf-02 gained candidates and converted none. gf-05 stands at 1 hit
against 18 false positives.

The value judge's best moment, and the argument for it: on gf-10 it rejected our answer (a 1.5%
systematic uncertainty) and produced *"the cut retains a signal efficiency of approximately 80%"* —
**his gold, exactly**. The evidence had been right all along; the extraction was wrong, which is the
failure an existence judge structurally cannot see.

## 7. The human adjudication

Models produce corroborated gold. **A physicist turns it into gold.**

### 7.1 Batch 1 — 202 items

| category | n |
|---|---|
| we cited it and the judge upheld it | 76 |
| we cited it and the judge rejected it | 63 |
| we found nothing (hybrid-retrieval candidates shown so a "no" is answerable) | 63 |

Design rules, each one earned:

- **Blinded.** Items shuffled within each question so claimed and missed are indistinguishable. In
  the app the reveal is *disabled until he answers* — enforced rather than requested.
- **Judge-rejected items are included.** Selecting on the final answer would drop all 63 downgrades,
  and those are exactly where the judge overruled the reader. Without them we could measure the
  reader and never the judge.
- **gf-01 enters at the condition level**, where the failure actually is.
- **Nothing is sampled away.** ~70% of the comparable claims are papers his list does not contain.
  Those are not known errors — they are the only evidence that our system finds things his does not.
- **The sheet says so outright**: *a paper your list does not contain is not automatically our
  error; judge the sentence against the paper, not against your list.* Without that, a reviewer meets
  an unfamiliar paper, reads it as our mistake, and every genuine discovery becomes a false positive.
- **Candidate sentences come from the real hybrid retriever** (BM25 + bge-base dense + RRF), not word
  overlap.
- **The TSV is one physical line per row.** 31 cells originally held newlines; Excel and Sheets
  handle quoted TSV unreliably, and a one-row shift would reattach every later verdict to the wrong
  paper — silently, and undetectably afterwards.

### 7.2 Batch 2 — 87 splits

The reads where samples of *one* passage disagreed. They fell through every bucket in `build()` —
not `claimed` (answer is None, never judged), not `negatives` (answer is not False) — so only the 11
that happened to sit in his gold reached batch 1, by accident.

They are the **most informative items available**: a claim the judge upheld is usually easy, a split
is a real question our own machinery could not settle. Batch 1 measures the easy cases.

Issued **strictly additively**: rows from 1001, separate files, its own browser-storage key, and
`build()` deliberately untouched with a test asserting it stays that way. Batch 1's answers are keyed
on row number, so renumbering it would silently reattach verdicts to different papers.

The reveal reports the tally of the window that **actually disagreed** — not `votes`, which counts
every sample of every window and would read "46 said false, 2 said true" for a paper where one
window split 2–1.

## 8. What this gold is for

Three uses, in the order they should be done.

**8.1 Separate graph coverage from query quality.** Right now a miss is ambiguous: did the query
fail, or was the fact never in the graph to retrieve? The graph stores `evidence` — 8,482 quoted
sentences with block and character offsets, linked to assertions — so a gold quote can be checked
against it directly.

First reading, using our verified quotes as a stand-in until the human labels land: **50 of 76 (66%)
are also cited by the graph**. By question: gf-01 and gf-03 100%, gf-08 69%, gf-05 47%, **gf-04 25%**
— six of gf-04's eight evidence sentences are invisible to *any* query.

The proper decomposition is three categories, mapping onto the three layers built:

1. **absent from the graph** → extraction/coverage failure
2. **present but not reachable as queried** (wrong alias, missing facet) → representation failure
3. **present and reachable but not returned** → query/ranking failure

That turns one accuracy number into an **error budget with an owner per layer**. It must be built
*before* the ablation, or the ablation misattributes coverage failures to the query.

*Caveat: "the graph cites this sentence" is a strict test — the graph may hold the same fact citing a
different sentence. 66% is a lower bound on semantic coverage.*

**8.2 Expand the gold with what the graph finds and the papers confirm.** If a paper is retrieved
repeatedly and is not in gold, it may belong there — and that is a positive result, because it means
the graph surfaces what a paper-by-paper read missed. The mechanism is real and nameable: **alias
normalisation and typed structure**. A paper writing "the combined secondary vertex (Version 2)"
rather than "CSVv2" is invisible to string retrieval but not to the alias layer — gf-11 is exactly
that case.

**The hazard, and the guard.** Adding to gold whatever our query system retrieves inflates our own
score by construction. So: the **graph may propose**, but only the **reference reader** (papers only,
no graph access) and **the supervisor** may accept. Gold is versioned and frozen before scoring, and
results are reported against both v1 and v2 — never silently against the expanded one. A graph-found
paper with no supporting sentence in the paper is an extraction error, not a discovery; the quote
verifier already tells those apart.

**8.3 Measure the judge.** Now the weakest link. His labels give the first real precision figure for
the support judge, which everything downstream depends on.

## 9. Known limitations

- **Corroborated, not true.** Quote verification catches invention, not misreading. Only the human
  labels close that gap.
- **The reader sees what LaTeXML extracted.** Anything lost upstream is invisible to the reader too,
  so the reader is not a fully independent check on extraction — only on the *graph built from* it.
- **Figures are captions only.**
- **Partial extractions still score as passes** (D-058 form 4). Needs per-part gold.
- **gf-05 is a designed trap** ("string-matching Higgs drowns in Higgs PROCESS papers"). Our 26
  claims against his 2 may be the trap being sprung rather than a discovery. His verdicts will be
  diagnostic either way.
- **gf-12 is disputed, not solved.** We answer "875 GeV" from "masses up to 875 GeV are excluded at
  95% CL". Whether that is a *lower limit* is a physics judgement; the models refused the conversion
  for two days before one accepted it.
- **Q9 provenance is unexplained** — `expert_decision` has 0 rows anywhere in the graph.

---

## Changelog

**v1.0 — 2026-08-14.** First version. Covers the whole ground-truth procedure as built between
2026-08-07 and 2026-08-14: why the supervisor's answers cannot serve as gold; the question set and
its circularity guards; the reference reader (what it may read, window sizing, quote verification,
per-window consensus); the four-stage cascade and the two judges; the conjunction problem (D-058) and
the parser faults (D-059); the controlled measurements; both human-adjudication batches; and the
three uses of the finished gold. Related: D-054 … D-059, `logs/2026-08-12.md`.
