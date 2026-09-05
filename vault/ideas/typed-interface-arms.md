# Typed-interface arms — three removable changes to the tool layer

**Status**: specced 2026-08-31, none built. Each is an **arm**, not a change to the
baseline: default OFF, one flag, one guarded branch, deletable without touching the
frozen path. Motivated by the trace anatomy below; **none of them is expected to work
until measured.**

**Realises**: S-36 (ablations), S-52 (variance). Pairs with
[free-sql-control.md](free-sql-control.md) — the control that beat us is what suggested
all three.

---

## The evidence these come from

Trace anatomy over 81,790 typed steps (Qwen2.5-72B, all cluster runs) and 1,377
free-SQL statements.

**The typed planner does follow the graph.** 52% of its step transitions are
"named something, then walked an edge from it"; only 4% are "searched, then searched
again". 45% of its edge-walks reference a stored set symbolically (`set_1_kept`)
rather than retyping ids. The design is working as intended.

**But the walk is one edge per round, and it dies at round three.**

    round 1  37,939 calls   search 57%, contents_of 36%      find
    round 2  23,625 calls   count 56%, subjects_of 23%       aggregate
    round 3  10,707 calls   papers_of 52%, count 27%         resolve to papers
    round 4   2,601 calls   nothing above 28% -- the tail

Rounds 1-3 are a near-deterministic recipe. After that the distribution flattens and
the volume collapses. Median rows: **25 on the first step, 1 on every later step.**
The planner is not accumulating a subgraph; it is confirming one thing at a time and
stopping.

**free-SQL wins by collapsing that recipe into one statement.** 36% of its records
are a single SQL call answering the whole question; 68% of statements carry a JOIN,
39% carry two or more. It is not exploring more widely — it is expressing the same
2-3 hop path declaratively instead of procedurally. It scores 0.669 (sol) / 0.592
(qwen3-32b) against typed 0.423.

**And 92% of typed errors are references to things that do not exist yet:**

    unknown_entity_id                            1,289   56% of errors
    "no set named 'set_1'. Available: none yet"    833   36% of errors

The second is the interesting one: the model has learnt the set idiom and reaches
for `set_1` *before running the search that would create it*.

---

## Arm A — `--symmetric-hops`: make the backward hop predicate-optional

**The asymmetry.** `describe` (forward: what X points at) takes `predicate` as
OPTIONAL, so a bare call returns every outgoing edge. `subjects_of` (backward: what
points at X) REQUIRES it. So the model can ask "what is this?" but cannot ask "what
points at this?" without first guessing a predicate name. It is called 6,619 times,
always with a predicate, because the schema forbids anything else.

**The change.** When the flag is on, `predicate` moves out of `required` for
`subjects_of`; omitted, it returns incoming edges grouped by predicate.

**Removability.** The flag selects between two tool-schema dicts and guards one
branch in the handler. Off = today's schema and today's SQL, unchanged.

**The risk, stated first.** Fan-in is unbounded in a way fan-out is not: a popular
object (`hepkg:object:jet`) may have thousands of incoming assertions, and an
unfiltered backward hop could return a large fraction of the graph. It must reuse
the existing row cap, and it should return a *predicate histogram* when the cap is
hit rather than an arbitrary truncation — "there are 2,100 incoming edges across 9
predicates, here they are" is a useful answer; 200 arbitrary rows is not.

**Measure**: zero-row rate on `subjects_of` (4% today), backward hops per record,
`unknown_entity_id` rate, `judged_f1`. **Kill it if** the row cap fires on more than
~20% of calls, which would mean the tool mostly returns histograms and not evidence.

## Arm B — `--strict-refs`: close the 92% error class

Two behaviours behind one flag. They are separable and should be split if the
combined arm moves the number, so the credit is attributable.

**B1: a set reference before any search is guidance, not an error.** Today it costs
a whole round and returns a scolding string. It should return the same message
*without consuming the round*, since the model has demonstrably understood the idiom
and simply mis-ordered it.

**B2: entity ids must come from a set.** A literal `entity_ids` list is refused with
a pointer to the set that holds them. This is the one mechanism that would cut
sol's 41% id-invention rate, because an id you never retype is an id you cannot
invent.

**Removability.** Both are validation branches in front of existing handlers. Off =
the current permissive path.

**The risk.** B2 must apply to ENTITY ids only. Paper ids legitimately arrive as
literals from `papers_of`/`contents_of` results, and `contents_of(paper_ids=[...])`
is a correct call with no set behind it. Getting that boundary wrong would break the
paper-resolution step that round 3 depends on. B2 is also the most likely of the
three to *hurt*: it removes a degree of freedom from the model, and D-062's lesson
is that taking a decision away from the model has to be earned by measurement.

**Measure**: `unknown_entity_id` rate, "no set named" rate, rounds spent on errored
steps, `judged_f1`. Run on **sol and Qwen2.5-72B both** — the invention rates differ
4x (41% vs 11%), so a gain on one says nothing about the other.

## Arm C — `--path-tool`: one call for a multi-hop path

**The structural difference between the two systems**, and the arm most likely to
move the number.

**What the winning SQL actually looks like** — this is a real free-SQL statement,
and it is NOT a chain:

```sql
SELECT DISTINCT eo.paper_id FROM assertion a_est
  JOIN assertion a_def ON a_est.subject_id = a_def.subject_id
  JOIN entity_occurrence eo ON a_def.object_id = eo.entity_id
 WHERE a_est.predicate = 'result_estimates_background' AND a_est.object_id IN (...)
```

Two assertions joined **on a shared subject**: find the things that satisfy
predicate A *and* predicate B. That is a conjunction, not a walk — and it is exactly
the shape of the multi-condition questions we are worst at. gf-01 ("searches using
b-jets AND missing transverse momentum") is the question where precision fell from
0.90 on single-condition questions to 0.30.

So the tool is conjunctive first, chained second:

```
path(constraints=[{predicate, object_set|object_ids}, ...],
     project="subjects"|"papers",
     mode="all"|"any")            # all = intersect, any = union
```

One call returns the subjects satisfying every constraint, optionally projected to
papers. The 3-act recipe (find, aggregate, resolve) collapses into search + path.

**Removability.** The tool is appended to the tool list only when the flag is on, so
with it off the model never sees it and the prompt is byte-identical. That matters:
adding a tool changes the prompt for every question, so an unflagged version would
contaminate the baseline.

**The risk.** It may simply not be used — `describe` already offers all forward
neighbours and is called in 3% of steps, so a capability existing is not the same as
a model reaching for it. If `path` goes unused, that is a finding about prompting,
not about the graph, and it should be reported as such rather than quietly dropped.

**Measure**: calls to `path` per record (is it used at all?), steps per record
(should fall from 2.1), median rows on later steps (should stop collapsing to 1),
and `judged_f1` on the multi-condition split specifically — gf-01's 35 per-condition
rows in batch 2 are the natural test bed.

---

## How to run them

One axis at a time, against the frozen baseline, on the same code. D-062 lost a week
to a one-line prompt change that moved the control while an arm was being read.

Order by cost-to-benefit: **A** (small, low risk), then **C** (the real hypothesis),
then **B** (most likely to hurt, and needs two models to interpret).

Powered on `dev-2026-08-03-paperA-200.jsonl`, not the eight supervisor questions:
the noise floor on n=27 is ~0.06 and these effects may be smaller than that. The
supervisor's gold is the headline, but it cannot resolve these.

**None of this is adopted.** If all three come back null, the conclusion is that the
typed interface is not the bottleneck and the free-SQL margin is about something
else — most likely that one declarative statement costs one round where the typed
path costs three, which is a *budget* finding, not an *interface* finding.
