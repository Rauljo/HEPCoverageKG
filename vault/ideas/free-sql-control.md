# The free-SQL control — the experiment that can prove us wrong

*Status: designed, not built. 2026-08-28.*

## What it is for

The project's claim is that **typed tools over a typed graph beat a model turned loose
on the data**. Nothing built so far can falsify that, because nothing has been turned
loose. Every arm measured to date varies something *inside* the typed layer — the critic,
the ordering, the contract, the judge. None of them asks whether the typed layer earns
its place at all.

This is the control that does. Same model, same database, same questions, same scorers.
**The only difference is the tool surface**: instead of nine typed tools, one tool that
takes SQL.

If free SQL matches the typed agent, the typed layer is scaffolding and the dissertation
should say so. That outcome is a finding, not a failure — and it is the first thing an
examiner will ask about.

## What each side gets

|  | typed agent | free-SQL agent |
|---|---|---|
| model | 72B | **same** |
| database | `hepkg.db` | **same** |
| questions, repeats, scorers | | **same** |
| tools | `search`, `contents_of`, `neighbors_of`, `subjects_of`, `count`, `facets`, `describe`, … | **one**: `sql(query)` |
| schema knowledge | the schema card: kinds, predicates, facet vocabularies | **the DDL**, plus row counts and a sample of distinct values per column |
| entity resolution | canonical ids, alias clusters, facet tags | the same tables exist in the DB and it may join them **if it works out how** |

**The fairness rule: it may see everything, it just gets no help using it.**
`entity_canonical`, `same_as` and `entity_facet` are real tables and free SQL can query
them. Hiding them would make the control a straw man — the claim under test is that the
*tools* and the *vocabularies* help, not that we possess data nobody else does.

The one thing it genuinely cannot have is the retrieval index, because that is not in the
database. Noted as a limitation rather than papered over: on concept questions the typed
agent has a real advantage the control cannot match, and that advantage is part of what is
being measured.

## The tool

    sql(query: str) -> rows

- **Read-only, enforced by the connection** (`file:...?mode=ro`), not by inspecting the
  string. String checks are bypassable; the connection is not.
- One statement, `SELECT` or `WITH` only. Rejected otherwise, with the reason returned to
  the model so it can retry.
- `LIMIT` forced to ≤ 200 rows, and the truncation reported in the result. A silent
  truncation would make the model confidently count a capped set — the exact failure
  `count` was built to avoid.
- Wall-clock cap per query. A cartesian join over `assertion` × `entity_occurrence` is one
  plausible token away.
- Errors go back verbatim. A SQL agent that cannot see its own syntax error is being
  tested on one-shot SQL, which is not the question.

## What we expect, so that being wrong is informative

Predictions written before the run, because a prediction made afterwards is a
rationalisation:

1. **Free SQL wins on exact per-paper questions (Tier A).** "Which generators does 2106.01676
   use?" is one join and the labels are literal. If the typed agent does not at least match
   it here, the typed layer is costing accuracy, not buying it.
2. **Free SQL loses badly on concept questions (Tier B).** "How many analyses used Pythia?"
   requires knowing that Pythia 8.230, PYTHIA8 and Pythia8.2 are one generator. That is
   `entity_canonical`, and the model has to discover both that the table exists and that it
   needs it. **This is the thesis, stated as a measurable gap.**
3. **Free SQL loses on Gabriel's gold**, for the same reason plus the conjunctions.
4. **Free SQL is faster and cheaper per question.** One round trip against six planner
   rounds. If it also matched on quality, that would settle the argument against us.

If (1) and (2) both hold, the honest claim becomes narrow and defensible: *typed tools buy
nothing on literal lookups and a great deal on questions that cross spellings* — which is
exactly what a coverage map is made of.

## What it must not become

Not a strawman. Three specific ways this experiment could be rigged without anyone
intending it, all of which must be checked before the numbers are believed:

- **Prompt asymmetry.** The typed agent's PURPOSE prompt is the product of weeks of
  iteration; a first-draft SQL prompt is not a fair opponent. The SQL prompt gets at least
  one round of iteration on *dev questions only*, and the transcripts get read.
- **Round asymmetry.** The typed agent gets six rounds. The SQL agent gets six too, so it
  can look at the schema, fail, and recover.
- **Scorer asymmetry.** `set_f1` scores papers the answer names. A SQL agent returning a
  table rather than prose could score zero for reasons of format. The answer must be
  extracted the same way from both, and any question where the format decides the score is
  reported separately.

## Cost

Half a day to build, and it is mostly the tool plus a prompt. Then one run per question
set. Cheap relative to what it settles.

## Links

[[eval-harness-design]] — the socket it plugs into is already there: a System is anything
with `answer(question)`, so this is a new adapter, not new machinery.
