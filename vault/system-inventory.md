# What this system is, and what has actually been measured

One page per question a methodology section has to answer. Written 2026-09-07,
because the repo had reached a point where the arms could not be read off it.

**The rule for reading this file**: `evidence` is what a number can be quoted
from. Where it says *void*, the mechanism exists and works, and no valid
measurement of it exists — see §0 for why that is true of so much.

---

## §0. Why most of the record is void

Four failures, all the same shape: a missing dependency degraded into a
**plausible number** instead of an error, so the run completed, scored, and was
believed.

| | what was broken | what it invalidates |
|---|---|---|
| D-088 | critic 404'd on every call | 6 arms, Sept 1 |
| D-105 | reasoning judge returned empty; 100% of verdicts defaulted | 4 full runs, ~34k candidates |
| D-111 | server hit its wall limit 40 min in; 44% errored | the first answer-gate 2x2 |
| **D-116** | **`answer()` reached on 6% of questions** | **every arm touching the answer contract, all of them** |

D-116 is the big one. The model was calling `answer` as `<answer .../>`,
`<answer>{json}</answer>` and (another model) `<papers_from>set_1</papers_from>`
— none of which the recovery matched. So the citation resolver, the gate, the
answer-critic and `answer_papers` were each working correctly **on 6% of
answers**. After the fix: 75–96%.

Add the noise floors (D-115) and the picture is complete:

    typed    set_f1     0.0006     <- the only metric that can carry an ablation
    typed    judged_f1  0.0843     <- Gabriel's 9 questions; most arm gaps are inside this
    free-sql set_f1     0.0534
    free-sql judged_f1  0.1494

**Consequence for the write-up**: the list of mechanisms built is long, the
list measured is short, and saying so plainly is stronger than implying twelve
ablations that were all inside noise.

---

## §1. The two systems

|  | typed | free-SQL |
|---|---|---|
| interface | 14 typed tools over the graph | writes SQL directly |
| candidate critic | **yes** | **none, by design** |
| retrieval reach | **0.97** | 0.52 |
| set F1 (D-115) | 0.228 | **0.254–0.308** |
| scored on a footprint | 24–31 of ~98 answers, 38–41 papers each | 9–11 of ~98, 4.5 papers each |

The comparison's honest statement: **the typed system retrieves nearly twice as
much and converts it worse**, and a quarter of its set answers are graded on a
retrieval footprint rather than an answer. "Typed beats free-SQL" would also
partly mean "the critic helps", because free-SQL has none.

---

## §2. Mechanisms, by status

### Working, and the evidence is valid

| mechanism | flag / knob | evidence |
|---|---|---|
| value indexing | `--index-values` | D-085: 31% of the graph was unsearchable. Largest single effect in the project, and a graph-quality result as well as a query one |
| encoder swap | `ALIASES_EMBED_MODEL` | D-087: bge has no `Higgs` token, splits `hi`+`##ggs`. 12 models swept. Intrinsic, model-independent |
| answer-critic as a **ranker** | offline | D-113: F1 0.654 vs 0.591 for doing nothing, p@1 = 1.00 on all nine questions. Against Gabriel's 253 labels, no agent loop |
| answer-critic as a **filter** | offline | D-112: 0.445, WORSE than keeping everything. Gains only below purity 0.25, and purity is not knowable at runtime |
| temperature | `PLANNER_TEMPERATURE` | 0.0 → 0.651, 0.4 → 0.622, 0.7 → 0.513. Monotonically worse |
| question-set screening | — | D-095→098: 86% of questions cannot separate two arms; 24 usable → 163 |
| noise floors | `--repeats` | D-115, above |

### Built, never validly measured (all blocked on D-116)

| mechanism | flag | note |
|---|---|---|
| candidate critic | `--critic` | +0.231 is the largest number in the project and it predates D-105 |
| critic ordering | `--critic-seed` | shuffled vs ranked; position bias |
| critic set substitution | `--force-critic-set` | flat-but-faster; 39% of counts ran over the RAW set, diluting it |
| contract v2 / v3 | `--contract` | +0.059 measured with half of v3 switched off |
| sub-goals | `--subgoals` | 54194 had a 100% defaulted critic — void |
| sub-goal status | `--subgoal-status` | **the only arm that ever beat its noise floor on both axes** (D-096); never re-measured live |
| plan reviewer | `--reviewer` | D-107: does not reason worse, it stops early (1.8 vs 2.4 rounds) and writes a pointer instead of a list |
| widening ladder | `--persist` | four finite rungs, each a concrete untried call from the run's own trace. **This is not backtracking** — nothing returns to an earlier state |
| answer gate | `--answer-gate` | fired twice in 164 questions before D-116 |
| answer-critic (live) | `--answer-critic` | fired on 9% of chances before D-116 |
| rerank | `--rerank` | D-115: −0.030, but it reordered ENTITIES by critic rung, and 96–98% of `papers_of` results are under `max_rows` so nothing truncates. The offline result was never tested |
| name the ids | `--name-ids` | D-117. v3 currently tells the model NOT to write ids into `text`, and `text` is the only field the set scorers read |
| literal id list | `--simple-answer` | highest print rate of any arm, 0.81 vs 0.65 (D-107) |
| worked tool examples | `--tool-examples` | |
| few-shot | `--fewshot`, `--fewshot-plan` | distillation vs hand-written never decided |
| multi-hop tool | `--path-tool` | **this IS the "2-hop tool"** |
| stated objective | `--state-objective` | |
| push further | `--push-further` | |
| reduced prompt | `--minimal-prompt` | |
| quote indexing | `--index-quotes` | |
| paraphrases | `--paraphrases` | D-102 said rephrasing beat resampling 7:1; D-103 retracted it as an unfair comparison |
| free-SQL: sample framing | `--concept-prompt` | |
| free-SQL: materialise matches | `--search-sets` | the memory table — free-SQL's equivalent of citing a set name |

### Never built

- true **backtracking** (return to an earlier retrieved state)
- **routing** per question type
- **typed + free-SQL combined**
- **Cypher**
- cost/latency as a reported axis — the data is in every run record, nothing reads it

---

## §3. Metrics (put this BEFORE the mechanisms in the write-up)

Half of what looked like an agent finding turned out to be a scoring artefact
(D-107, D-110, D-115). A reader who does not understand the metrics first will
misread every arm.

- **`judged_set_f1` + the universe restriction** (D-072). Gabriel only saw
  papers the system surfaced, so an unjudged paper is *unknown, not wrong*.
  Precision is clean; recall is optimistic **by construction**.
- **`retrieval_reach` vs `judged_f1`** — found it vs said it. The gap is 0.57–0.76
  and it is the thesis (D-104).
- **the footprint trap** (D-062): `set_f1` falls back to `a.papers` when the text
  names nothing, and `a.papers` is everything the system touched. gpt-5.6-luna
  scored 0.889 having written no answer at all.
- **`MAX_LISTABLE` / `TYPICAL_LISTED`**: a metric that punishes truncation is
  worse than one that says it cannot judge.
- **abstention**: `answerable` + `reason ∈ {answered, not_in_graph, out_of_scope}`.
  A system taught never to abstain fabricates coverage, which is worse than the
  false negatives it fixes, because it is invisible.

---

## §4. Evaluating the evaluation

`hpc/monitor/armcheck.py` (D-114) asserts, per run: the recorded config matches
what was asked for; the mechanism fired **against its opportunities, not against
zero**; the error rate is not eating the run; scores are in range. Verified
against all four historical failures rather than asserted.

That is a methods contribution in its own right, and arguably more transferable
than any single arm.
