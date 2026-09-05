# Self-correcting planning — four removable arms, after PoG

**Status**: specced 2026-08-31, none built. Same rule as
[typed-interface-arms.md](typed-interface-arms.md): default OFF, one flag, one
guarded branch, deletable without touching the frozen path.

**Source**: Plan-on-Graph (PoG), Chen et al., NeurIPS 2024 — see
[`../literature.md`](../literature.md). Two arms already built on 2026-08-31
(`--reviewer`, `--state-objective`) came before reading it; these four are what
the paper adds that we do not already have.

---

## Why these four, and in this order

PoG's own ablation ranks its mechanisms, and the ranking is the argument:

    w/o Memory            -4.3    <- the biggest
    w/o Reflection        -3.8
    w/o Guidance          -3.1
    w/o Adaptive Breadth  -1.9

**Memory is worth most, and we have none.** `Session.sets` holds every entity
the graph handed back, but nothing tracks WHICH PART OF THE QUESTION IS ANSWERED
so far.

And PoG's §1 names our worst failure independently. Their limitation 3,
"forgetting partial conditions": *the LLM remembered the song was by Taylor
Swift but forgot the condition about winning an AMA*. That is gf-01 — precision
0.90 on single-condition questions, **0.30** on the three-part one, and the
reason batch 2 contains 35 hand-decomposed per-condition rows.

## Arm D — `--subgoals`: decompose, and nothing else

Split the question into **at most 3** sub-objectives, once, at the start. They
may depend on each other; the list is fixed thereafter and never updated.

Three, not "as few as possible" and not more: **93% of runs finish in ≤4
rounds**, so four sub-objectives leaves one round each. This arm is also the
first thing in the project for which the round budget genuinely binds — raising
`max_rounds` alone does nothing today (only 2.8% of runs reach the cap), but it
must rise WITH this arm or decomposition simply starves retrieval.

Useless on its own by PoG's own account (that is their `w/o Memory` variant,
their second-worst result). It exists to be the baseline Arm E is measured
against, and to answer "is the gain from splitting the question, or from
remembering the split?"

## Arm E — `--subgoal-status`: the memory block (contains D)

Arm D, plus a status line per sub-objective, **rewritten every round** and
carried in the prompt:

    1. searches using b-tagged jets      18 papers found, set_1_kept
    2. ...that also require MET          not yet established
    3. intersect 1 and 2                 not started

**Rewritten, not appended.** The message list already grows each round; adding a
fresh block per round would leave five stale copies in context, and the model
would have to work out which is current. One block, replaced in place.

**Folded into the planner's existing call**, not a separate one. PoG spends a
whole LLM call on this (their A.3). We already folded `GOT`/`AIM` the same way,
and the measured cost of an extra call on a reasoning model is ~31s. Split it
later if the folded version is sloppy.

Storage is not the hard part -- `PlannerState` is a TypedDict on LangGraph and
already checkpointed, so this is one more plain-data field. The hard part is
that **state the model never sees does nothing**: it only reads `messages`.

**E − D isolates the memory effect.** That subtraction is the whole point of
running D at all.

## Arm F — `--structured-verdict`: a reason field on every judgement

PoG returns JSON from every decision prompt: `{"A":…, "R":…}` for answers,
`{"Add":…, "Reason":…}` for reflection. Ours returns `APPROVE`/`REVISE` plus
free prose, parsed by regex, and an unparseable verdict silently approves.

Structured output makes the verdict countable rather than readable, and removes
the fail-open ambiguity: today "the reviewer approves a lot" and "the reviewer
is broken" look identical unless `review_unparsed` is checked by hand.

Cheapest of the four and the least likely to move the score. Worth it for
measurability, not for performance.

## Arm G — `--backtrack`: named candidates, competing with a fresh search

When the model is stuck, the widening ladder currently offers a **generic
route**. PoG instead offers the specific things already retrieved and never
followed, and asks which to return to.

We can build that list for free: `session.sets` holds what was retrieved,
`session.steps` records what was passed onward, and the difference is the
unfollowed remainder. A real trace shows why it would help:

    round 1   search(f_a3^ggH, observable) -> 60 rows -> set_1
    round 2   count over set_1_kept
    round 3-6 papers_of("hepkg:observable:f_a3_ggH") -> unknown_entity_id  x4

Sixty real entities retrieved, then four rounds inventing ids instead of using
any of them.

**THE FEAR, WHICH THE PAPER'S OWN NUMBERS SUPPORT.** Backtracking is not
reliably good: of PoG's 24% of cases that backtrack, the answer is right 48%
(CWQ), 64% (WebQSP), **36%** (GrailQA). On GrailQA it leads somewhere wrong
nearly two times in three. And our own widening/persist arms came back null or
negative on Qwen.

So backtracking is offered as ONE OF TWO NAMED OPTIONS -- revisit these, or
start a new search -- with the choice recorded. `backtracked` vs
`searched_fresh` then becomes a measurement instead of an argument.

Candidates come from `set_N_kept`, the ones **the critic already judged
relevant**, not the raw 60. Which raises a live inconsistency: `force-critic-set`
is off by default, so 39% of counts currently run over the RAW set and discard
those same verdicts. Leaning on `_kept` here makes that harder to justify.

## Where we deliberately depart from PoG

**Their "as few as possible" framing is inverted for anything touching recall.**
Every PoG prompt asks for the minimum -- fewest sub-objectives, fewest
relations, fewest entities -- because their metric is Hits@1 on ONE entity,
where extra answers are pure noise. Ours is set F1 over papers, and recall is
the weaker side in every arm measured:

    QwQ typed            precision 0.567   recall 0.527
    QwQ free-SQL                   0.543          0.426
    Qwen2.5-72B typed              0.572          0.473
    qwen3-32b free-SQL             0.704          0.576

There is a sharper reason than the metric. The model writes 11 arXiv ids at the
median; 100% of what it names was genuinely retrieved, and only 22.8% of gold
papers are among them. It is a SELECTION failure, not an over-production one, so
"return the minimum" forces it to drop candidates it already cannot rank.

Same design principle as theirs -- match the instruction to the metric -- and
the opposite conclusion. **Breadth is controlled by relevance (the critic,
+0.231, our largest measured effect), never by a count.**

**Two PoG mechanisms do not transfer at all.** Their relation-exploration step
exists because a Freebase entity has hundreds of relations; we have ~20
predicates and the schema card lists them all, so that LLM call buys nothing.
And they assume entity linking is solved -- *"we assume any entity mentioned in
q ... are labeled and linked"* -- while 56% of our errors are
`unknown_entity_id`. PoG starts after our hardest step.

## What to measure

Beyond `judged_f1`: sub-objectives produced, how many end the run still "not
started" (the forgetting rate), backtrack-vs-fresh-search taken and which paid,
and the accept/reject rate of any judge -- a mechanism that fires on everything
or nothing is a no-op at double the price, which `force-critic-set` already
demonstrated once (417/426 traces changed, 148/150 counts identical).

Powered on `dev-2026-08-03-paperA-200.jsonl`. The eight supervisor questions
cannot resolve these: the noise floor at n=24 is ~0.06.

**A caution from the first reviewer run, 2026-08-31.** Rejections do not consume
a round, by design -- but they do consume wall-clock, and the per-record harness
timeout is 600s. One record spent 523s across 15 LLM calls; another timed out
with nothing. The 20-cycle runaway ceiling never fires because the timeout bites
first at roughly 8-10 cycles, so the ceiling as written is decorative. Any arm
that adds a self-correction loop needs its cycle cap set BELOW what the timeout
allows, or the cap is not the thing doing the capping.
