# Evaluation harness — design

**Status**: **BUILT 2026-08-02** — `hepcoveragekg/eval/`, **24 tests** (317 total green).
Block 1, item 1 of `system.md` §5 Phase 3. This doc is the design; deviations from it are noted
under *As built* at the bottom.
**Realises**: S-22 (harness before everything), S-10 (freeze/split/tag), S-32 (three layers),
S-36 (ablations), S-52 (variance). Feeds every other item in §4.

---

## The one structural decision

**The harness does not run "our system". It runs *any* system that can answer a question.**

```python
class System(Protocol):
    name: str
    config: dict                      # everything that could change the answer
    def answer(self, q: Question) -> Answer: ...
```

Everything becomes an adapter behind that: the planner, plain RAG, BM25-only, the no-corpus LLM,
GraphRAG, later chATLAS. Get this wrong and the baseline arms need a second harness, the numbers are
not comparable, and the comparison tables have to be assembled by hand.

The corollary is that **an ablation is just a `System` with a different `config`** — no separate
machinery, and the ablation study is a loop over configs rather than a project.

## What a run must capture, or the result is unattributable

Every record carries **what produced it**, not only what came out:

```
run_id · timestamp · git_sha · system.name · config_hash · repeat_index
```

Without `git_sha` + `config_hash`, two runs a week apart cannot be compared and the ablation study
is worthless. This is the single cheapest thing to get right and the most expensive to retrofit.

## Data shapes

### Question (frozen, dated, **in git**)

`data/` is gitignored, so questions live at **`eval/questions/`** — they are the artefact that must
never drift.

```jsonc
{
  "qid": "gen-0001",
  "text": "How many analyses used Pythia?",
  "source": "generated",              // generated | gabriel
  "split": "dev",                     // dev | test          (S-10)
  "shape": "count",                   // count|set|describe|compare|crosstab|quotes|freeform
  "needs": ["sql"],                   // sql|hops|signatures|merged_entities|not_in_graph   (S-10, S-57)
  "difficulty": "easy",
  "truth": {"kind": "count", "value": 58, "papers": ["2307.01094", ...]},
  "truth_source": "sql",              // sql | reader | gabriel | none
  "provenance": {"query": "...", "generated_at": "2026-08-02"},
  "group": null,                      // metamorphic group    (S-34)
  "relation": null                    // e.g. "paraphrase_of:gen-0001", "subset_of:gen-0007"
}
```

`needs` is what makes a low score **attributable** — was the query layer wrong, or was the data never
there? It is also the M3 tag (S-57): `"signatures"` says the question is blocked upstream.

### Answer / run record

Mostly what `Session` already holds — steps, thoughts, sets, tokens, seconds, verification — plus
`answer`, `evidence_ids`, and the run identity above. One JSONL line per (question × repeat × system).

## Layout

```
hepcoveragekg/eval/
  questions.py    load, validate, freeze-check, dev/test lock
  systems.py      the System protocol + adapters
  runner.py       run(questions, system, repeats) -> JSONL
  scoring.py      scorers, each  (Question, Answer) -> dict[str, float]
  metamorphic.py  relation checks over GROUPS of questions      (S-34)
  report.py       tables, per-tag breakdown, run-vs-run with spread
eval/questions/   dev-YYYY-MM-DD.jsonl, test-YYYY-MM-DD.jsonl   (in git)
eval/runs/        <run_id>.jsonl                                (gitignored; summaries kept)
```

CLI as a subcommand of the existing `hepcoveragekg.cli`:
`eval run | score | report | compare`.

## Scorers (each independent, each may abstain)

| scorer | needs | gives |
|---|---|---|
| `exact` / `set_f1` | `truth` | end-to-end correctness |
| `faithfulness` | nothing | already built (`verify.py`) — claims traced to retrieved rows (S-14) |
| `plan_validity` | schema card | % tool calls type-correct first try (S-35) |
| `tool_choice` | expected tool | did it use the right one (S-49) |
| `retrieval` | target entity | Recall@k, MRR (S-48) |
| `abstained` | nothing | did it decline, and was declining right (S-13) |
| `cost` | nothing | LLM calls, tokens, seconds |

**A scorer that cannot apply returns nothing rather than zero.** Scoring an unlabelled question as 0
silently drags every average down and the cause is invisible.

## Two mechanisms that enforce the protocol in code

**The test-set lock (S-10).** `runner` refuses `split == "test"` unless `--unlock-test` is passed, and
every unlock is appended to `eval/TEST_OPENED.log` with date, git sha and reason. Discipline that
depends on remembering is discipline that fails in week 3.

**Variance first (S-52).** `repeats` is a first-class argument and the report prints
**mean ± spread**, never a bare number. Two runs compared without spread should be a warning in the
output, because a 3% "improvement" inside a ±5% band is noise.

## Build order (~1.5–2 days)

1. `questions.py` — schema, loader, validator, lock. **Write ~10 questions by hand** to shape it.
2. `systems.py` — the protocol, plus a **stub** that answers nothing, plus the planner adapter (thin;
   `planner.stream` already returns a `Session`).
3. `runner.py` — repeats, JSONL out, run identity captured.
4. `scoring.py` — start with the two that need no labels: `faithfulness` (exists) and `cost`.
5. `report.py` — table + per-tag + run-vs-run with spread.
6. **Prove the loop**: stub vs planner on the 10 hand-written questions. The numbers must differ.
   *This is S-22's point — watch the number move before the question set exists.*
7. Only then: backward question generation (S-33) at volume.

## Explicitly out of scope for the harness

Judges, checklists, pooling adjudication, the reference reader. All of them **consume** run records
and produce scores, so they are scorers or downstream tools — none belongs inside the runner. Keeping
them out is what keeps blocks 1–2 free of humans and API keys.

## Open

- Whether run records go in git (reproducibility) or stay local (size). Leaning: summaries in git,
  full JSONL local.
- Whether `config_hash` covers the prompt text (it should — `PURPOSE` full vs minimal is an axis).
- Question **id stability** across regenerations: ids must survive a re-run of the generator, or
  cross-run comparison breaks.

---

## As built (2026-08-02)

`hepcoveragekg/eval/{questions,systems,runner,scoring,report}.py` + `cli.py eval
run|report|compare|describe`. Seed set: `eval/questions/dev-2026-08-02-seed.jsonl`, **10 hand-written
questions**, truth pulled from SQL (Pythia **58**, Herwig **22**, Sherpa **30** papers — the 58
matches the 2026-07-31 end-to-end result exactly).

**The loop is proven.** `eval run --system stub --repeats 2` scores 0 on `answered`, `count_correct`
and `set_f1` — and **1.0 on `abstention_correct`**, because declining is the right answer to the one
`not_in_graph` question (S-13). A test asserts stub-vs-correct moves the number; until that passed,
nothing downstream would have meant anything.

**Deviations from the design above**, all small:
- Scoring runs **inline in the runner**, not as a second pass. Scorers are pure functions of
  (question, answer), so deferring bought nothing and cost a re-read of every record.
- `git_sha` appends **`-dirty`** when the tree is not clean, and the report prints a warning. A sha
  that does not describe what ran is worse than no sha, because it looks authoritative.
- A `.summary.json` sidecar per run — small enough to keep in git while `eval/runs/*.jsonl` is
  ignored.
- `tool_use` (plan shape) is in; **`plan_validity` (S-35) and `tool_choice` (S-49) are not** — both
  need something the questions do not yet carry (the schema card's type table, an expected tool).

**Open items from the design, now resolved**: run records stay local, summaries in git ·
`config_hash` covers `minimal_prompt`, so the `PURPOSE` ablation hashes differently · id stability is
enforced by `save()` sorting on `qid`, with a test.
