# The system — design and decisions

**Version 1.8 · 2026-08-07**

*The reference document for everything built on top of the knowledge graph. Living: update it and
bump the version whenever a decision here changes. Changelog at the bottom.*

*Scope note: **S-nn** numbers below are **system-level** decisions (architecture, evaluation,
tooling). **D-nnn** in [`decisions.md`](decisions.md) stays what it has always been — decisions
about the schema, the store, and the extraction/aliases code. Where a system decision forces a
schema or code change, it gets a D-nnn entry there as well.*

---

## 1. What the system is

The knowledge graph (SQLite as system of record, Neo4j as a projection) is a **coverage map** of a
corpus of HEP papers. On top of it sits a set of agent systems that talk to each other.

The purpose, as framed with the supervisor on 2026-07-29: **help a physicist see what has been
covered** — while reviewing the literature, planning an analysis, or checking whether an idea is
new. Finding *gaps* is one use of that, not the whole point of it.

**The load-bearing idea: the query system is the single entry point.** Not because a chat interface
is nice, but because every other component goes through the same door:

- the **user** asks questions
- the **paper-reading loop** asks what the graph is missing
- the **deduplication and grouping layers** ask for context
- a future **gap finder** asks whether a candidate gap is really absent

One thing to build well, one thing to measure, and any improvement anywhere shows up in the same
number.

### S-01 — the query system is the only entry point
Every consumer — human or agent — reaches the graph through it. No component gets its own private
retrieval path.
*Why*: it collapses N interfaces into one, and it makes answer quality a usable proxy for the
health of every layer underneath.

### S-02 — faithful base, LLM layers strictly on top
The base must stay as close as possible to what the papers actually said. Everything interpretive is
a **separate, labelled layer above it**, never a rewrite of the base.
*Why*: the user's own formulation (2026-07-28) — "get everything in the basic sql/graph, be as close
as possible to the extraction and the real physics, then build layers on top that are completely LLM
made."
*Note, and it matters*: the genuinely faithful layer is **`entity_occurrence`** (the per-paper
record). `entity` is already a derived rollup — modal label, consensus-only attributes — and is
easy to mistake for the base. It is not.

---

## 2. Architecture

### 2.1 Layers

| Layer | What it is | Who writes it |
|---|---|---|
| `entity_occurrence`, `assertion`, `evidence` | what each paper said, per paper | importer (faithful) |
| `entity` | rollup across papers | derived, deterministic |
| `entity_canonical`, `same_as` | identity — which things are the same | Tiers 1/1.5 deterministic, 2/3 LLM |
| final-state signatures | structured `{object, count, comparator}` | LLM parse + code naming (D-038) |
| groups / hierarchies | coverage families, flavour hierarchy | LLM, later |
| user subgraph | per-session retrieved + newly read facts | query system + reader |

### 2.2 The query system

Free-text question in, evidence-backed answer out. Four stages:

**a. Route.** Two separate jobs, deliberately not merged:
- *what shape of question is this* → LLM classifies into one of ~6 templates (count, filter,
  compare, trace, list, …)
- *what is this question about* → hybrid retrieval (embeddings + BM25) over entity labels, kinds
  and predicates. **Deterministic, no LLM.**

**b. Build the query.** Fill the chosen template with the **retrieved ids** — never with text the
model invented. The model picks the shape; it cannot conjure an entity that does not exist.

**c. Execute and check.** Mechanical checks first (does it parse, is it read-only, does it run, do
the columns match the question, are the ids real). Only what survives goes to an LLM to judge
relevance. Failure returns the error to the generator for a bounded number of retries.

**d. Present.** Text summary, table, plot, or graph view. Every claim carries its evidence.

**Join trap, found 2026-07-30 while building the templates.** To get from a `result` to its paper,
use **`entity_occurrence.paper_id`**, never the `paper_reports_result` predicate. The predicate
links only the *headline* result of each paper — **60 of 272** — while the occurrence table reaches
all 272. A `result` is one specific finding, and papers report between 1 and 44 of them (the
headline analysis, plus normalisation factors, per-region numbers, limits).
*Why this matters beyond one query*: it is the semantically obvious join, so it is exactly what a
model writing free-form SQL would choose, and it fails **silently** — no error, no empty result,
just 22% of the truth looking entirely plausible. A direct, concrete argument for S-04 (templates
over free-form generation): the trap gets encoded once, in a tested template, instead of being
rediscovered — or not — on every question.

#### S-03 — the router is two jobs, not one
Shape classification (LLM) and content retrieval (deterministic) stay separate and are measured
separately.
*Why*: merged, a wrong answer tells you nothing about which half failed.

#### S-04 — queries are templates filled with retrieved ids, not free-form generation
*Why*: free-form text-to-SQL lands around 60–70% on arbitrary schemas. This schema is closed —
predicates and kinds are a known list — so most of that difficulty is self-inflicted. Also gives a
clean measured comparison for the write-up (constrained vs free-form).

#### S-05 — SQL for querying, Neo4j for visualisation
All querying goes to SQLite. Neo4j stays what it was defined as (D-022): a projection, used for the
graph view the user sees.
Multi-hop: most apparent multi-hop in this schema is **two joins**, not traversal. Where real depth
is needed, add a **hand-written, tested template** the agent parameterises. `WITH RECURSIVE` covers
variable depth if it comes to that.
*Revisit only when the evaluation shows questions failing specifically for lack of traversal* —
measure first, add Cypher generation second.
*Why*: two query languages doubles the failure surface and doubles what has to be evaluated, for
questions that are overwhelmingly filters and counts.

#### S-24 (2026-07-30) — the router may answer "no template fits", and that is a measurement
The shape classifier returns one of the templates **or `none`**. `none` routes to free-form query
generation, and the answer is **flagged** as free-form so the evaluation can score the two paths
separately.
*Why*: closed templates cannot cover every question, and pretending otherwise means the router
silently forces a bad fit — the worst outcome, because it looks like an answer. This is the same
shape as abstention (S-13): the system saying "this is outside what I can do cleanly" is
information, not failure.
*The by-product is the design input*: the log of questions where no template fitted **is** the
specification for templates 7, 8, 9. Recurring shapes get promoted from free-form to tested SQL.
*Constraint*: the free-form path uses the **same rails** — read-only connection, `sqlglot` parse and
write check, evidence collected from whatever it matched. Free-form means unconstrained *SQL*, never
unconstrained *access*.

#### S-25 (2026-07-30) — dual-path answering: template and free-form, compared
Run both paths and have a third step compare them: does the free-form answer contain anything the
template missed, and does the template answer contain anything free-form lost.
*Why it is worth the cost*: it turns the central S-04 claim into a **per-question measurement**
instead of a one-off ablation. "On N% of questions the unconstrained agent found something the
templates could not express" is a result either way it comes out.
*Two safeguards, both non-negotiable*:
  1. **The free-form answer must have executed.** Its rows must come from SQL that actually ran and
     returned real records — never from the model's prose. Otherwise the comparison rewards fluent
     invention.
  2. **Compare structurally first, LLM second.** Do the two return the same rows? Does one strictly
     contain the other? Only the *qualitative* residue ("is this extra actually interesting?") goes
     to a model. An LLM asked which answer is better will pick the richer one, and richer is exactly
     what a hallucinated join looks like.
*When to run both*: **always on the evaluation set** (that is where the comparison is the point);
in the live system, only when the router abstained or the user asks to dig deeper — multi-agent
overhead grows superlinearly (see [[literature]]) and most questions do not need two passes.

#### S-29 (2026-07-31) — the model never handles entity ids; `search` saves a named set
`search` stores its hits under a handle (`set_1`) and returns *"saved as set_1 (31 entities);
pass set_1 as object_set -- do not retype the ids"*. Every id-taking tool accepts `object_set` /
`entity_set` as well as explicit ids.
**Why, measured on the first live runs.** Asked "how many analyses used Pythia?", the model:
1. batched `search` and `count` in ONE round, so `count` was written before any ids existed, and it
   **invented** `gen-223` → confident `{papers: 0}` → answered *"the graph records no analyses that
   used Pythia"*. Truth: 58. The trace looked healthy — grounded, no errors, faithfulness clean,
   because an answer with no numbers has nothing to check. **A wrong answer wearing a good trace.**
2. told the ids were unknown, it corrected itself perfectly — right predicate, right ids — but wrote
   the call as **text** rather than a structured call, so vLLM's parser missed it and the loop
   recorded raw JSON as the final answer.
3. the recovered text turned out to be **truncated at 6,521 characters**, mid-identifier: it had
   looped writing ids until the token cap.
Only there does the cause appear: **making a model retype 31 long identifiers**. 863 completion
tokens for one call, every one a chance to mistype or invent.
**Effect**: 122s → 6s, 863 → 139 completion tokens, 1 invented id → 0, answer wrong → **58 / 344 /
367, exactly right, 100% grounded, 202 evidence quotes**. It also sequenced itself correctly without
the batching guard firing, because "use the set name search gave you" implies waiting for search.
**Kept as defence anyway**, all counted in the trace: invented ids rejected (`invented_ids`),
text-mode calls parsed back out (`recovered_calls`), sets listed (`sets`). If the model reaches for
raw ids again we will see it rather than infer it.
**Unexpected second payoff**: the sets are the unit of *memory*. `set_1` survives compaction and
survives a turn, so "how many of **those** also used Herwig?" has a real object to point at rather
than prose to re-derive.

#### S-30 (2026-07-31) — state describes the run; config carries the machinery
Checkpointing forces the graph state to be **plain, serialisable data**. Learned three times in ten
minutes when turning it on:
- **undeclared keys vanish.** LangGraph keeps only the keys its schema names, so values passed
  between nodes without being in the `TypedDict` are silently dropped — 29 tests failed at once.
- **callables cannot be checkpointed.** `chat`, the tool executor and the tool schemas were in
  state; every checkpoint tried to serialise a function. They belong in **config**, which is not
  persisted.
- **SDK objects cannot either.** Tool calls are normalised to `{id, name, arguments}` dicts and the
  message to a plain string before entering state.
Also: our trace dataclasses are **registered with the serialiser**. LangGraph deserialises
unregistered types with a warning today and will *refuse* them in a future release — which would
break resume silently and late.
*The hand-rolled loop allowed all three, because nothing ever had to survive the process.*

#### S-31 (2026-07-31) — context growth: compaction and recall, deferred until measurable
One question already costs **11,902 prompt tokens over 3 rounds** against an 8,192 window, because
each round re-sends everything before it. Two mechanisms, neither built yet:

**Compaction, split by what a tool returns** — a first pass that proposed collapsing *everything*
deterministically was wrong:
| result | treatment |
|---|---|
| `search` → a set | deterministic collapse is **lossless**: the set holds the ids, so 31 rows become one line |
| `describe` / `list` / `crosstab` | the rows **are** the content; a row count is useless |
| the model's reasoning | keep — dense, and not reconstructable from a `Step` |
| last 1–2 rounds | verbatim, always |
*Trap*: the chat format requires an assistant `tool_calls` message be followed by `tool` messages
with matching ids, so compaction rewrites content **in place** rather than deleting messages.

**Recall instead of summarising, across questions.** Relevance across turns is long and sparse —
question five rarely needs question two's rows, but occasionally needs exactly them. Summarising
guesses wrong; carrying everything drowns. So: sets and question/answer pairs stay visible, and a
`recall(query)` tool fetches the rest on demand. **The checkpointer is already the store** —
`get_state_history()` per `thread_id`.
*What LangGraph does NOT give*: search over that history, the tool itself, and — the one people
assume — **any relief from context growth**. Checkpointing solves persistence, not pruning.
**Deferred deliberately**: compaction quality is testable (same questions, compaction on and off,
compare answers), and guessing a scheme before that exists is how the deterministic-everything
version got proposed. Interim fix: raise `--max-model-len` to 32k — one flag, and the card has 97k
of KV cache.

#### S-27 (2026-07-30) — one planner, batched operations; hierarchy only under context pressure
The agent is a **single planner** that emits a **batch of operations per round**, sees every result
together, and plans again. One LLM call per round — not one per operation, and no separate
"executor" agent (tool calling already emits structured calls; a translator agent would be a second
call per round doing a job the schema does).

*Measured, and it settles the design*: a template query takes **0.29 ms**; the heaviest (crosstab,
386 cells) 16 ms; running them across 8 threads is **slower** (0.51 ms) because connection setup
dominates. One LLM call ≈ **3,000–17,000 queries**. So parallelising *operations* buys nothing —
the model is the entire cost.

**Why a boss-and-workers hierarchy would be added is context, not speed.** A single planner that
fans out already has full visibility and loses nothing to summarisation. Workers exist to
*compress* — each explores a branch and returns a summary so the boss never sees raw rows. You pay
N extra LLM calls to buy context headroom, and nothing else.
*Trigger*: add hierarchy when the planner's context is the constraint. Log context size per round.
*Note first*: `--max-model-len` is currently **8192**, which is a setting, not a limit — the AWQ
model has ~120 GB of KV cache headroom on 2×A100. Raise it to 32k before building a hierarchy to
work around a number we chose.
*Ablation built in (user's idea)*: a **max-concurrent-places** attribute caps how many branches the
planner may explore at once. That makes 1-vs-many a dial rather than a second architecture, so the
comparison is a parameter sweep instead of a rewrite.
*Also for SQLite*: a connection cannot be shared across threads (`ProgrammingError`). Moot while
batching, but it would bite immediately if anything ever threads.

#### S-28 (2026-07-30) — clusters collapse by default, groups expand by default
They are different claims and therefore have opposite defaults.

**A cluster** (`entity_canonical` / `same_as`) is an **identity** claim — "these *are* the same
thing". If correct, members are interchangeable, so collapsing loses nothing. Default: **collapse**;
drilling in is for *auditing the merge*.

**A group** is **not** an identity claim — "these are different things that belong together".
PYTHIA 6 and PYTHIA 8 genuinely differ. Collapsing loses information. Default: **expand**; the group
is primarily a **retrieval handle** (the thing that turns the word "Pythia" into 56 ids).
*A group may be the unit of an answer only when the question was posed at group granularity*:
"how many analyses used Pythia, any version?" → group-level is right; "which Pythia versions?" →
must expand.

**And a merge that changes a number must be shown next to the number.** Not `58 papers` but
`58 papers, collapsing 56 entities into 12 clusters`, inspectable. Otherwise a wrong merge is
silently wrong forever; shown, it costs a visible dispute instead.
*Correction to an earlier claim*: the agreement-by-paper-count figures (1 paper 79.0%, 2 papers
91.7%, 3+ 100%) are **Tier 1/1.5 merges only** — that is what the trial set measured. Tiers 2/3 have
not run at scale, so applying the paper-count rule to them is a **hypothesis to test**, not a
finding.

#### S-26 (2026-07-30) — learned lessons are promoted into structure, not into prose
When the system discovers something — a failure mode, a data quirk, a trap — ask **"can this be
code?"** before writing it anywhere. Order of preference, strongest first:

| Where it goes | When | Strength |
|---|---|---|
| a **template / code** | mechanically enforceable | absolute; a test proves it |
| the **schema card** | a fact about the data | generated, so it cannot drift |
| a **guard / validator** | a check | fails loudly |
| the **prompt** | judgement that cannot be mechanised | advisory only, obeyed *most* of the time |

*Worked example*: "join to papers through `entity_occurrence`" could have been a line in PURPOSE and
would have been followed most of the time. As a template it is followed **every** time. It therefore
lives in `templates.py`, and appears in the prompt only for the free-form path, where there is no
template to hold it.

**The prompt may still be amended, but by proposal only.** The system logs a candidate line with the
evidence that motivated it; a human accepts; the prompt is **versioned** and every evaluation run
records which version it used.
*Why the guard*: the prompt is the control surface — an agent editing it changes every future answer
at once, and silently. Three specific failures: evaluation numbers stop being comparable across
runs; the prompt bloats into thousands of tokens of half-contradictory caveats nobody can audit; and
one bad case generalised into a rule poisons everything after it. Same principle as S-07, in a more
dangerous place.
*Same idea as the convenience detector*: recurring chains become tools, recurring mistakes become
templates and guards. Prose absorbs only what neither can hold.

#### S-06 — validation is mechanical first, LLM second
An LLM judging its own query agrees with itself. Parse → read-only check (`sqlglot`) → execute →
sanity-check the shape of the result. Only then ask a model whether it answered the question.

### 2.3 Paper-reading expansion

When the graph is not enough — the user says so, or the system detects it — agents pick papers worth
reading, skim them, extract what was asked for, and add it to the **user's** subgraph. Presented the
same four ways.

This is the piece that separates **"the physics isn't there"** from **"we never extracted it"**,
which both of the other features depend on.

#### S-07 — agents never edit the extraction code; they file proposals
A positive user evaluation produces a **proposal record** (add predicate, split a kind, fix a
mapping) for human approval, against a versioned extraction. Never a silent self-edit.
*Why*: an agent rewriting the extractor makes the graph change under the evaluation, so nothing
replicates — including your own results from the week before. As a proposal queue it also yields a
real metric: how many proposals were accepted, and did the accepted ones improve the answers.
*Consistent with* [[researcher-feedback-loop]] — "human-applied fixes, never LLM self-editing".

#### S-08 — anything the reader adds carries provenance
Newly read facts are marked distinct from the curated base (`trust`: extraction / deterministic /
llm / human), on `entity_canonical` / `RESOLVES_TO` and on new assertions.
*Why*: without it the evaluation silently scores the system on facts it invented earlier in the same
session. **Decide the field before building the reader** — retrofitting provenance is miserable.
Also enables a conservative vs full view of the graph.

### 2.4 User subgraph and editing

Each session builds a subgraph from what was retrieved plus what was read. The user can edit it —
labels, content, structure. An agent then asks **why that was wrong**, and looks for other things
wrong for the same reason, re-retrieving where needed.

#### S-09 — the edit-diagnosis machinery is the same as M4's review diff
M4 (re-import after review) already has to compute what changed between the extracted and the
reviewed bundle. Keep that diff **as data**: it is free labelled extraction error, corrected by
physicists. Use it twice — (a) derive a real taxonomy of failure modes instead of a guessed one,
(b) turn each mode into a **detector** that sweeps the corpus for the same shape, so one human
review generalises.
*Why*: identical problem to the user-editing feature, just fed from Gabriel's review instead of a
click in the UI. Build once, at M4.
*Caveat for the write-up*: 1 accepted paper + 287 staged decisions is enough to build and
demonstrate the mechanism, **not** enough to claim a validated taxonomy.

### 2.5 Deduplication and grouping (already running)

The aliases layer (Tiers 1 → 3) continues as the identity layer. Above it, a **grouping** layer
builds coverage families and hierarchies as alternative views the router can choose between; from a
group the system can always descend to canonical → occurrence.

Current state as of 2026-07-29, carried forward: base pipeline built and measured; Qwen2.5-72B-AWQ
gives usable calibration (≥0.99 band = 99.9% accurate, 0.7–0.9 band = 62.7%); Tier 1/1.5
**over-merges** — ~19% of its cross-label merges are disputed and spot-checks favour the model, with
disagreement concentrated on single-paper entities (1 paper 79.0% agreement, 3+ papers 100%).
Decisions still open there are listed in §7.

### 2.6 Gap finding

Deferred to future work (§5). Reframed 2026-07-29: after the supervisor conversation this is one use
of the coverage map, not the point of the project — the title no longer hangs on it.
When it returns, the cheap defensible version is **structurally empty cells** — build the grid of (process × final state × energy …), find combinations nobody
covered, and use the LLM only to say which empty cells are physically meaningful. Author-declared
gaps ("we leave X to future work") give free labels. See [[gap-hypothesis-system]].

---

## 3. Technology stack

| Job | Choice | Note |
|---|---|---|
| LLM serving | **vLLM**, self-hosted (DIAS) | consider a small model alongside the 72B — routing does not need 72B, and the query system makes far more calls than dedup did |
| Structured output | **vLLM guided decoding** | constrains to a *schema*, not just valid JSON |
| Orchestration | **LangGraph** | orchestration only — see S-16 |
| Embeddings | **BAAI/bge-base-en-v1.5** | already measured and chosen (D-035) |
| Vector storage | **numpy array in memory** | ~5,000 × 768 floats = 15 MB. **No vector database** |
| Lexical | **BM25** (`rank_bm25`) + existing trigram Jaccard | embeddings miss exact tokens: "SRA", "MET", arXiv ids |
| Databases | **SQLite** (record) + **Neo4j** (view) | agent connections **read-only** (`mode=ro`) |
| Query safety | **sqlglot** parse + write-statement check | before anything executes |
| Early UI | **Streamlit** / Gradio | ~1 day; stop reading JSONL by hand |
| Final UI | **FastAPI** + static page | |
| Graph view | **cytoscape.js** | Neo4j Browser is a DB tool, not a product |
| Tracing | **JSONL per session** | consistent with everything else; Langfuse later only if a UI is wanted |

#### S-16 — LangGraph orchestrates; everything else stays hand-written
Nodes, edges, state and checkpointing are LangGraph's. **LLM calls, prompts, retrieval and SQL stay
in our own code.** It does not go near `adjudicate.py` or the aliases layer, which are finished and
measured.
*Why*: human-in-the-loop is central to this design, sessions are long-lived, and resumable state
across a pause is the one genuinely fiddly thing to hand-roll. The earlier objection ("you lose
sight of the prompt") applies to LangChain, not to LangGraph — in LangGraph the LLM call is code you
write inside a node.
*Rejected*: **LangChain** — replaces working, debuggable code with abstraction; hides the exact
prompt and response that the faithfulness metric depends on; fast API churn against a dissertation
that must still run at marking time.

#### S-17 — no vector database
At 60 papers and ~5,000 entities, a numpy dot product is the whole implementation.

#### S-18 — hybrid retrieval, dense + sparse
*Known limit, already measured*: abbreviations defeat both — `MET` vs `missing transverse momentum`
scores 0.496 semantically and 0.091 lexically. That needs a dictionary, not a threshold (D-035).

#### S-19 — guided decoding for every structured output
Especially the final-state parse (D-038) and query generation. The `is_match` / `same` / `is_same`
key drift that needed `_VERDICT_KEYS` to patch is exactly what this removes.
See [[pydantic-validation]].

#### S-20 — real tool calling, and expose the query system over MCP
Define retrieval and query steps as proper tools the model calls, rather than hand-rolling
JSON-out-then-dispatch. Then publish the query system as an **MCP server**.
*Why*: it is the right design anyway — and it matches S-01 exactly, since MCP is the standard for
"here is a tool other systems can call."

#### S-21 — the agent's blast radius is bounded
Read-only DB connection; generated SQL parsed and checked before execution; generated plotting code
runs in a restricted namespace with no filesystem or network access.

---

## 4. Evaluation

**The evaluation is the contribution.** Most systems of this kind ship with no idea whether they
work. Everything below is designed so that a wrong answer tells you *which layer* was wrong.

### 4.0 Three layers, measured separately

#### S-32 (2026-08-01) — separate "the agent queried badly" from "the graph was already wrong"
Three different things get evaluated, and treating them as one contaminates every number:

| | what "correct" means | who decides | cost |
|---|---|---|---|
| **L1 — did the agent query the graph correctly?** | matches what SQL says | **the graph itself** | free, exact, unlimited |
| **L2 — does the graph match the papers?** | matches what the paper says | the papers (§4.7) | one reading pass |
| **L3 — is the answer useful to a physicist?** | Gabriel nods | only Gabriel | scarce |

*Why this is the unlock*: **almost everything being built is L1** — planner, retrieval, dedup,
templates, abstention, the LangGraph loop. All of it is "did the agent get out of the graph what is
in the graph", and for that **the graph is perfect ground truth**. Thousands of exact labels, no
human, no judge. The instinct "we have no ground truth, so we need a judge" skips over this.

*The consequence that matters*: today, when an answer is wrong, **we cannot tell whether the agent
asked badly or the fact was never extracted.** Two different problems, two different fixes, and one
mixed number. §4.7 separates them, and afterwards every L1 result is reported conditionally —
*"the query layer is right on 94%, against a graph measured at 89% assertion recall"* — two honest
numbers instead of one contaminated one.

### 4.1 The question set — the single biggest risk

Written by the user and the supervisor, in progress as of 2026-07-29.

#### S-10 — freeze the questions, split dev/test, tag every one
A **dev set** to iterate against and a **test set** not opened until the end. Frozen, in git, dated.
Each question tagged with what it needs: plain SQL / graph hops / final-state signatures / merged
entities / not in the graph at all.
*Why freeze*: questions written after the system exists are questions the system can answer. That
alone can invalidate every number.
*Why split*: four stages of tuning against one set measures how well the set was fitted.
*Why tag*: a low score otherwise cannot be attributed — was the query layer wrong, or was the data
never there? The tags also say immediately which questions are blocked on M3.

#### S-11 — the set must contain dedup-sensitive questions
Deliberately include questions whose answer **changes depending on whether entities were merged** —
counting questions are the natural form ("how many analyses used this generator?").
*Why*: without them, the evaluation cannot tell dedup configurations apart, and the open aliases
decisions (§7) go back to being settled by taste. This costs nothing while the questions are being
written and cannot be fixed afterwards without re-freezing.

### 4.2 Three-arm comparison

Same questions, three systems:

1. **LLM alone**, no corpus — shows the questions are not general knowledge
2. **LLM + plain text search** over the same papers — the honest RAG baseline
3. **This system**

Arm 2 is the one that matters: beating it is what shows the *graph* did something.

#### S-12 — deep research is a real baseline, corpus-restricted and dated
Against the same questions, restricted to what is answerable from the 60 papers, with every
transcript saved and dated.
*Why it belongs*: on **aggregate** questions the KG has a structural advantage that no amount of
reading gives — retrieval *samples* (20–50 sources, no completeness guarantee), a graph
*enumerates*. "How many analyses tested X" is exactly that. Since coverage counting is the product,
these questions are the backbone of the set.
*Two guardrails*: restrict to the corpus, or it beat your library rather than your system; and it is
not reproducible, so report it as a dated snapshot, never as a fixed baseline.
*Expect to lose on single-fact lookups* ("what limit did this one paper set?"). Reporting that
honestly is stronger than claiming to win everywhere — it shows where structure helps and where it
does not.

### 4.3 Deletion protocol

Remove a known part of the graph, then ask a question that needed it. **Two separate measurements:**

#### S-13 — abstention and recovery are measured separately
- **Abstention** — does it say it does not know, instead of inventing? Rare, hard, and worth a
  number of its own.
- **Recovery** — let it read papers: does it get back exactly what was deleted? Perfect ground
  truth, because we removed it. A complete benchmark for the reader, free, repeatable.

*Why this is the best idea in the plan*: it is the same trick that already worked twice — labels
pulled out of signal that was already there (`same_id` for the trial set, the review diff for
failure modes).

### 4.4 Per-layer metrics

End-to-end accuracy says *that* something broke, never *what*. So, per layer:

| Layer | Metric |
|---|---|
| Router | did it retrieve the nodes the answer needed? |
| Query | did it execute first try? how many repairs? |
| Answer | **every claim traces to a retrieved row** — checkable mechanically, because evidence quotes are stored |
| End to end | correct or not |
| Cost | LLM calls and wall-clock per question |
| Stability | same question twice — same answer? |

#### S-14 — faithfulness is checked mechanically, not by an LLM
Every number and claim in an answer must appear in a row that was actually retrieved. Most systems
cannot do this. This one can, and it should be a headline metric.

### 4.5 Human evaluation

#### S-15 — a small batch early, the full batch late
~20 answers rated by Gabriel **after the query system first works**, not at the end.
*Why*: three further layers should not be built on our own guess about what counts as a good answer.
Domain experts are also slow to schedule — book it early even if the full run is late.

---

### 4.6 Ground truth we manufacture ourselves (L1 — no humans, no judges)

#### S-33 (2026-08-01) — generate questions backwards from the graph; Gabriel's questions are the test set
Run a query **first**, get the answer, then have the LLM write the question that answer belongs to:

```
SQL   count analyses linked by sample_uses_generator to cluster(Pythia)  →  58
LLM   "How many analyses used Pythia?"
```

The label is the **query result**, never something a model invented. Hundreds of them, across every
template shape, entity kind and hop depth. Precedent: this is how the text-to-SQL benchmarks
(Spider, BIRD) and the graph-QA ones (GrailQA) are built — questions generated from logical forms,
because that is the only way to get exact labels at scale.
*Two corrections applied at generation time*: paraphrase into physicist language, then a second pass
checks the paraphrase still means the same thing. Gabriel spot-checks ~20 for naturalness — ten
minutes, and he never needs to know an answer.
*The bias, which must be stated in the write-up*: **generated questions can only ask what the graph
can answer.** So this is the **dev set** — large, exact, drives development and every ablation.
**Gabriel's questions stay the test set** and carry the headline number. Do not let the generated
set produce the reported result.

#### S-34 (2026-08-01) — metamorphic relations: correctness checks that need no answer
Some things are checkable from how two answers must *relate*, with no label at all:

| relation | must hold |
|---|---|
| paraphrase invariance | same question, five wordings → same answer |
| monotonicity | `count(Pythia OR Herwig)` ≥ `count(Pythia)` |
| inclusion–exclusion | `count(A) + count(B) − count(A∧B)` = `count(A∨B)` |
| order invariance | `compare(A,B)` ≡ `compare(B,A)` |
| deletion monotonicity | remove papers → counts only go down |

Every violation is a **guaranteed bug**, no interpretation required. Precedent: Ribeiro et al.,
*CheckList* — behavioural testing of NLP systems through invariants rather than accuracy.
*Why it earns its place*: "we tested N metamorphic relations, M held" is a real result, costs one
day, and it catches the class of bug that end-to-end accuracy hides.

#### S-35 (2026-08-01) — type-check the plan against the schema card: a guard **and** a metric
The schema card already records, from real data, which kinds sit on each end of every predicate
(`result_has_systematic  result -> systematic_uncertainty 98%`). So a call pointing that predicate
at a `detector_object` is **wrong before it runs** — the wrong end of the arrow.
*This is the 2026-07-31 silent failure*: the model used `result_has_systematic` backwards, got
nothing, and concluded nothing exists against 1,148 rows, with a clean trace and no guard triggered.
**Two uses**: reject the call and tell the model which end is which (guard); and report *"plan
validity: X% of tool calls type-correct on first attempt"* — a per-layer measurement of whether the
model understands the schema, independent of whether the answer was right.

#### S-36 (2026-08-01) — ablations are evaluation, not decoration
Once S-33 supplies exact labels, switch things off one at a time: dedup, BM25, dense, RRF, the
guards, `SEARCH_BREADTH` at 6/20/60, `PURPOSE` full vs minimal.
*Why it matters more than end-to-end accuracy*: it shows **each component earns its place**, it is
the most defensible kind of result available, and it finally settles the open aliases questions and
the search-breadth question **with evidence instead of taste**.

### 4.7 The reference reader — L2, and the shared target for every baseline

**The user's idea, and the strongest single item in the plan**, because everything else measures the
agent against the graph; this measures the graph against the papers.

#### S-37 (2026-08-01) — one pass, per paper, constrained to our schema, on a third model
A large model reads the corpus **one paper at a time** and records, per paper, a fact plus the
verbatim sentence supporting it. Aggregation is then **arithmetic in Python**, never model output.

*Why per paper rather than all 60 at once*: 60 HEP papers ≈ 700k–900k tokens; models that can hold
that degrade badly in the middle of long contexts, and the arithmetic — where models are weakest —
would be done by the model. Per-paper gives per-paper provenance, parallelises, and has no context
limit.

*Why a third model.* Measured setup: **extraction is Claude Sonnet** (`run_pilot.sh`:
`--provider claude-cli --model sonnet`), **querying is Qwen2.5-72B**. They already differ, which is
good. The reference reader must be **neither** — Gemini or GPT — so that a disagreement is real
information rather than a model agreeing with itself. Shared blind spots are the whole risk: a fact
one model systematically fails to notice while reading is a fact it will not think to ask for.

*It is a strong reference, never "ground truth"* — if it is 90% right, a perfect system scores 90%.
So bound it before using it: run each paper 3× and report self-agreement; **require a verbatim quote
for every claim and discard claims without one** (this also blocks answering from pretraining memory
rather than the text); have Gabriel check ~20 *quotes*, not 20 answers.

*Do not let it be both reference and baseline.* The exhaustive per-paper pass is the reference; a
one-shot long-context read is a separate, legitimate baseline arm.

*Cost, measured honestly*: ~2,969 papers × ~15k tokens ≈ **45M tokens per pass** — a few dollars on
a cheap fast API, ~$130 on a mid one, or roughly a day of DIAS time on the 70B. **Reading the whole
corpus is affordable.** An earlier estimate treating this as prohibitive was wrong.

##### Amendment (2026-08-02) — **targeted verification is the default; the full second graph is optional**
*User's objection, and it is correct*: a question-agnostic pass over every paper **is re-running
extraction**. Doing it deliberately, by a different model and a different method, is what makes it an
independent reference — the same logic as two annotators labelling one dataset to get an agreement
figure — but the **full** version is more than the evaluation needs.

**Default: verify only what the questions touch.** For each question, take what the graph answered
and check *those* claims against the papers — the papers the graph says *yes* to give precision
exactly; a sample of the *no*s gives recall with a confidence interval (this is S-43 pooling). Roughly
**an order of magnitude cheaper**, duplicates nothing, and across ~50 questions it accumulates into a
partial second graph anyway, so most of S-39 arrives as a by-product rather than as a project.

**The full pass becomes optional** — worth it only if a standalone "how good is the extraction?"
chapter is wanted. Decide after seeing how much the targeted version already reveals.

#### S-38 (2026-08-01) — the reader covers far more than counting; only open synthesis escapes it
`paper → fact + quote` answers anything that is a **function of per-paper facts**:

| question type | how |
|---|---|
| counting | count the yeses |
| **sets** ("which analyses…") | the yeses *are* the set → precision/recall |
| **existence / abstention** | zero yeses = nobody |
| **per-paper lookup** | the extracted value + quote |
| **crosstabs, comparisons** | group or diff the per-paper labels |

What it genuinely cannot do is **open-ended synthesis** ("where is coverage thin?"). That class goes
to §4.8 checklists — and the reader still supplies *those*, so it covers the whole question set in
two modes rather than one.
*Practical*: freeze the questions **first**, derive the per-paper fields they need, then do **one**
pass extracting all of them. Otherwise the corpus is re-read per question for no reason.

#### S-39 (2026-08-01) — graph against graph, at three strictness levels, with the matcher measured
The reader's output has the **same shape as an `assertion` row** — paper, fact, quote. So the two
can be diffed: facts in ours but not the reader's are candidate inventions; **facts in the reader's
but not ours are extraction gaps**, which is the *silent omission* failure the vault already names
as dominant and which nothing else in the plan would catch.

*The threat (user, 2026-08-01)*: one sentence can legitimately become
`sample_uses_generator(ttbar_sample, "Pythia 8.212")` **or** `paper_uses_generator(2307.01094,
"Pythia")`. Neither is wrong. A naive diff calls that disagreement and the numbers become nonsense.

Two defences:
1. **Cut it off at source** — constrain the reader to our predicate list, entity kinds **and
   granularity rules** ("the subject of `sample_uses_generator` is a sample, not a paper"). Most
   shape differences never occur. This is why the reader does not get to invent a schema.
2. **Report three strictness levels**, never one:

| level | matches on |
|---|---|
| strict | same paper + same predicate + both entities in the same canonical cluster |
| relaxed | same paper + same predicate + object cluster matches (ignore which sample) |
| loose | same paper + the object entity appears at all |

**The gap between strict and loose *is* the measurement of representation-difference vs real
disagreement** — *"of 340 apparent mismatches, 280 were shape, 60 were real"*. That is a finding, not
a workaround.

*How agreement is computed*: push both sides through the aliases layer (normalisation →
`semantics.embed_cached()`), match entities by canonical id → normalised string → embedding
similarity above a threshold, predicates exactly (because the reader was constrained), then set
intersection per paper.
**The matcher itself must be measured** — sample 50 matches and 50 non-matches and check by hand.
Skip this and the "extraction recall" number is really measuring the string matcher, invisibly.

### 4.8 Judges and expert time

#### S-40 (2026-08-01) — expert time buys calibration and disagreements, never labels
Gabriel supplies interesting questions, **not answers**. So he is never asked "what is the right
answer?" (ten minutes, one label). He is asked things that take seconds and scale:

| ask | cost | buys |
|---|---|---|
| 30–50 blind pairwise preferences | <1 h | **judge calibration** — Cohen's κ against the automatic judge |
| ~20 quote adjudications ("does this sentence say that?") | 10 s each | precision for **every** system at once (§4.9) |
| ~40 **graph-vs-graph disagreements**, both quotes shown | ~40 min | resolves L2 where all the information is |
| ~10 checklist sanity checks | 20 min | makes §4.8 scoring self-sustaining |

*Why calibration is the key move*: it converts *"we used an LLM judge, trust us"* into *"our judge
agrees with the domain expert at κ = 0.7 on 40 items, comparable to reported human–human agreement,
so we apply it to 300."* A measured instrument, not an assertion.
*Why disagreements, not random samples*: everywhere the two graphs agree, the answer is already
known. Sample where the uncertainty is.

#### S-41 (2026-08-01) — judging protocol: decompose, allow ties, use a different model family
From the LLM-as-judge literature (Zheng et al., MT-Bench / Chatbot Arena):
- **pairwise beats absolute scoring** — judges are unreliable at "rate 1–5", decent at "which is better"
- **position bias is real** — run both orders, count only verdicts that survive the swap
- **self-preference is real** — do not judge Qwen-72B output with Qwen-72B
- **give the judge the retrieved rows** — then it judges *grounding*, which it can do, not physics, which it cannot

#### S-42 (2026-08-01) — for prose questions, score both answers against one checklist; never compare them
*The problem (user, 2026-08-01)*: answer A discusses ATLAS SUSY searches, answer B discusses CMS top
measurements. Both sensible. "Which is better" is unanswerable and pairwise collapses.

*The fix*: build the checklist **before** anyone sees an answer, from the graph or the reference
reader, then score both against the same list.

```
Q: "What's been done at 13 TeV with b-jets?"
   □ the ATLAS b-jet SUSY searches (12)      □ a count, not "several"
   □ the CMS ttbar measurements (8)          □ nothing outside the corpus
   A → 3/4     B → 1/4
```

Nobody compares anything; "they answered different things" simply means one missed the list. And
Gabriel's role shrinks to *"is this the right list?"* — ten checklists, twenty minutes, once.
*Also*: allow **both bad / both fine / not comparable** as outcomes wherever pairwise is used. Ties
are data; forcing a binary is what created the problem.
*Open*: who writes the checklists, how many items, how they are weighted. Needs the frozen question
set first — **deliberately not settled here.**

#### S-43 (2026-08-01) — pooling with quote adjudication, so labelling survives scale
At 2,969 papers nothing can be labelled exhaustively. Standard practice in information retrieval
(TREC-style pooling): judge the **union of what all systems returned**, plus a random sample of what
none returned.
*User's refinement, adopted*: the unit of judgement is the **quote**, not the answer — *"does this
sentence actually say that?"* Ten seconds, no physics reasoning, and it yields precision for every
system in the pool from a single pass.
*Two additions*:
- **sample the "no"s too** (~50 papers no system flagged) — pooling gives precision, not recall
- **three verdicts, not two** — `supports` / `does not support` / **`wrong quote, but the paper probably does say this`**. Otherwise a retrieval bug and an extraction bug are recorded identically.

Consequence: for any one question, **the reader scales with the question, not the corpus** — verify
the papers the graph says *yes* to (exact precision) plus a sample of the *no*s (recall with a
confidence interval). Hundreds of papers, not thousands.

### 4.9 The baseline ladder

#### S-44 (2026-08-01) — build the two cheap baselines immediately; GraphRAG asks "was curation worth it?"

| arm | cost | why |
|---|---|---|
| 1. the 70B, **no corpus** | 1 h | proves the questions are not general knowledge — makes everything else mean something |
| 2. **plain RAG over the same 60 papers** | ½ day | the honest baseline — see the construction note below |
| 2b. **BM25 alone** | 1 h | famously hard to beat; a convincing win over it is a stronger claim than beating a fancier system |
| 3. **GraphRAG / LightRAG** | ~2 days | the KG competitor — **this is the contribution** |
| 4. **chATLAS** | ask Gabriel | real domain system, same corpus lineage, he is an author |
| 5. deep research (S-12) | cheap | dated snapshot, corpus-restricted |
| 6. **frontier model + plain RAG vs local 70B + the graph** | ½ day | separates *"our system is good"* from *"our model is good"* |
| — | | reference reader (§4.7) is the shared target, **not an arm** |

*Arms 1 and 2 need nothing that does not already exist — do them first.* A number on day one is
worth more than a better number in a month, because the gap becomes watchable.

*Construction of arm 2 (2026-08-02)*: there is no canonical library everyone cites, but there **is** a
canonical construction — `chunk → embed → vector store → top-k → stuff into the prompt` — which
GraphRAG's own paper uses as its baseline under the name **"naive RAG"**. Matching it makes our
numbers comparable to their published ones. **Write it by hand (~80 lines) rather than using
LlamaIndex, for one specific reason: use the same encoder as our system (bge).** A baseline on
different embeddings measures "bge vs their encoder" mixed in with "graph vs chunks", and the two
cannot be separated afterwards.

**On GraphRAG, the framing matters (user objection, 2026-08-01, largely correct).** It builds an
*induced* graph for sensemaking; ours is a *typed* coverage map for counting. They serve different
purposes, and beating it at counting proves nothing anyone doubted. So the question is **not** "who
wins" but:

> **Does hand-curating a typed schema buy anything, or does an automatically induced graph get most
> of the way there?**

That is the obvious challenge to the whole project — *"why not just run GraphRAG?"* — and there is
currently no answer to it. Report **per question type, and expect to lose somewhere**: we should win
on counting and coverage (free-text relations cannot be counted reliably), and **lose on open-ended
sensemaking**, where community summaries are genuinely good and we have no equivalent. Reporting the
loss is what makes the win credible.

*What configuration can and cannot do*: `entity_types` is settable and the extraction prompts are
editable (it ships a prompt-tuning step), so it can be pushed toward physics. It **cannot** be made
to produce typed closed relations, per-assertion evidence quotes, or exact counts — global search
answers from *community summaries*, and a summary has already discarded the counts. That is
structural, not a settings problem.
*And the trap*: if it were successfully turned into a coverage map, it would be a reimplementation
of this system and the comparison would measure nothing. Its value is being **the off-the-shelf
thing someone would reach for instead of building this.** Run it as shipped (domain-tuned prompts);
optionally also with our entity types forced in, which separates *"curation matters"* from
*"GraphRAG's defaults are wrong for physics"*.
*Fairness obligation*: same papers, same LLM, physics-tuned entity types. A rigged comparison is
worse than none.

#### S-45 (2026-08-01) — no existing benchmark measures aggregate coverage; say so, and run what does apply
Multi-hop sets (HotpotQA, MuSiQue) and paper-QA sets (QASPER) are runnable but measure generic
retrieval-and-reasoning, not a coverage map; document-ranking benchmarks (BEIR) do not map, because
retrieval here ranks **entity surface forms**, not documents.
**chATLAS_Benchmark is the one in-domain option**, and the 2,969 papers came from the chATLAS EOS
area — same corpus lineage, so its questions may be answerable over documents we hold. *Check first,
it is an hour's work and it changes the plan either way*: (a) does it target published analyses or
ATLAS internal documentation (CDS/TWiki/indico — papers cannot answer those), and (b) does it score
retrieval, answers, or both? Retrieval-only is still valuable: an externally-defined number on
someone else's questions is the independent check we cannot manufacture.
*For the write-up*: a paragraph saying no benchmark measures aggregate coverage — with the survey
behind it and what was run instead — reads as someone who found a real gap. Silently skipping
benchmarks reads as avoidance.

### 4.10 Scale

#### S-46 (2026-08-01) — "does it hold at 2,969 papers?" is five questions; four need no labels
1. **Retrieval degrades** — 8,449 surface forms → ~420,000; measure precision@k at 60/300/1,000/2,969. **Free**, because S-33 questions know their own target entity. Most likely thing to break.
2. **Latency and cost** — BM25 is 0.11 s/query with no inverted index. A curve.
3. **Context pressure** — tokens per question vs corpus size; this is what makes S-31 compaction a decision rather than a guess.
4. **Dedup degrades** — candidate pairs already grow faster than linearly (126,335 at 60 papers).
5. **Answer accuracy** — ← the only one needing ground truth, and S-43 pooling is how it is obtained.

#### S-47 (2026-08-01) — the coverage-growth curve; run the subsample version first
**Version B (today, no dependencies)**: random subsets of the existing 60 — 10/20/30/40/50/60,
several draws each for error bars, a `WHERE paper_id IN (…)` filter. The *shape* answers *"is 2%
coverage enough to say anything?"* — still climbing steeply at 60 means every aggregate answer is a
floor; flattening by 40–50 means the question is near-saturated. **It differs per question type**
("which generators exist?" saturates fast; "how many analyses used Pythia?" never does), and knowing
which is which says exactly which answers need a coverage caveat attached.
**Version A (as papers land)**: 60 → 300 → 1,000, re-extracted. Adds a **free correctness check**
(counts can only rise — a violation is a guaranteed bug), a **representativeness test** (does the
58/60 rate hold at 1,000? if not, the 60 are a biased sample and no claim about "the field" stands),
and it **is** the demonstration of the paper-reading loop.
*Related*: the honest scoping line the system should print anyway — *"58 analyses in this graph used
Pythia; the graph covers 60 of 2,969 available papers."* A coverage map that does not know its own
coverage is the one thing it must not be.

#### S-48 (2026-08-01) — hops-to-node: retrieval measured with exact, free labels
Pick a random node, generate a question for it (S-33), then measure — the target is known because
the question was built from it:
- **Recall@k** — did the target appear in the top *k* at all? *The most important one*: anything never retrieved is invisible to the planner however good it is.
- **MRR** — mean of 1/rank. Interpretable: 0.5 means second on average.
- **rounds to reach it** — 1 is ideal.
- **NDCG** only where relevance is genuinely *graded* (all 31 Pythia clusters relevant, exact-version match more so). For one correct answer, MRR is the right metric and NDCG is overkill.

*The value is in stratifying the draw*: rare (1 paper) vs common (58); many spellings (`jet` has 40)
vs one; **merged clusters vs singletons — which tests directly whether dedup helps or hurts
retrieval**, an open aliases question currently settled by taste. Re-run at 60/300/1,000 and it is
S-46(1) made concrete: *"median rank of the target degrades from 2 to 9 as the corpus grows 16×."*

### 4.11 Component tests

#### S-49 (2026-08-01) — tool-selection accuracy, and tool *necessity*
Questions where exactly one tool is obviously right, then check it is used:

| question | must call | failure caught |
|---|---|---|
| "How many X?" | `count` | uses `list` and counts by hand — **the model doing arithmetic**, where it is weakest |
| "What sentence says this?" | `quotes` | never fetches evidence → unsupported answers |
| "Compare A and B" | `compare`/`crosstab` | two lookups with no shared basis |
| "What is X?" | `describe` | over-searching |

Exact ground truth, because the question was designed. Nothing currently measures this layer.
*The sharper version — **necessity***: remove one tool and see whether the model routes around it or
fails. If three of nine tools are never called, or removing them changes nothing, the API is too
big — and a smaller toolset means better selection. **Simplification backed by a number.**

#### S-50 (2026-08-01) — make the model state what is *missing*, not why it acted
*Rejected*: "justify every tool call." Model explanations are frequently unfaithful to the actual
computation (Turpin et al., *Language Models Don't Always Say What They Think*), so a justification
is an artefact, **never a metric**; and per-call prose is expensive under batched calls (S-27).
*Adopted, narrowly*: on continuation rounds only, one short field — **`missing:` what the previous
results did not contain.**
Three reasons this shape survives the objection: it is **cheap** (one string, rounds ≥ 2); it
targets the model's actual weakness, **knowing when to stop**; and it is **mechanically checkable** —
verify whether what it claims is missing really is absent from the retrieved rows. A model that says
"I still need the CMS papers" when CMS papers are already in `set_1` has just revealed it is not
reading its own results, and that is now countable.
*Watch for*: forcing a justification can make a model **more** committed to a wrong path — it states
a reason then follows it. Hence a with/without ablation rather than an assumption.

#### S-51 (2026-08-01) — deduplication has no free ground truth; use three noisy sources and one label-free measure
*Correction (user, 2026-08-01)*: `same_id` is **not** ground truth. The vault already records 347
shared ids against **334 divergent** — same id does not mean same thing. An earlier claim that
held-out `same_id` pairs give clean recall was wrong.

| source | bias | use |
|---|---|---|
| `same_id` pairs | noisy — extraction merged things it should not | sample 50, **measure its own precision**, then use as a distantly-supervised label set with the noise stated |
| normalisation-identical (`b-jet`/`b jet`/`bjet`) | trivially easy | a **floor** — missing these means something is broken |
| LLM judge on the pair + contexts | model bias | the only one that scales; calibrate on ~30 human labels |

Use all three and **report where they disagree** — the disagreement rate measures how ill-defined
the task is, which is itself worth knowing given the open aliases decisions.
**But the measurement that actually decides it needs no labels at all**: run the dedup-sensitive
counting questions with dedup **on and off**, at 10/20/40/60 papers. A widening gap means dedup is
load-bearing; wrong answers appearing only when it is on means it is dangerous. That plot settles
the question and sidesteps labelling entirely.

#### S-52 (2026-08-01) — stability as an unsupervised correctness signal
Ask the same question k times at temperature > 0 and measure agreement. `58, 58, 58, 45, 58` says
something real without knowing the truth. Precedent: semantic entropy (Farquhar et al., *Nature*
2024) — disagreement over *meanings* predicts hallucination. Unusually clean here because counting
answers are numbers, so "same meaning" needs no NLI.

### 4.12 Pipeline features deliberately held for the ablation study

*Both were designed on 2026-08-02 and are **not** being built before the harness — see the
sequencing note at the end of S-54. They live here rather than in §2 because their justification is
a measurement that does not exist yet.*

#### S-53 (2026-08-02) — question decomposition compiles to **set algebra**, not to sub-answers
Deep-research systems split a question and recombine the sub-**answers**, which means a model reads
two paragraphs and merges them — the step where they leak errors. We have something better, because
`search` already produces **named sets** (S-29), so a decomposed question recombines as SQL:

```
"Which analyses used Pythia AND searched for top squarks at 13 TeV?"
  set_1 = search("Pythia") · set_2 = search("top squark") · set_3 = search("13 TeV")
  answer = count(papers in set_1 ∩ set_2 ∩ set_3)          ← arithmetic, not judgement
```

*The free consequence*: once decomposed, S-34's inclusion–exclusion relations apply directly
(`count(A∧B) ≤ min(count(A), count(B))`), so the decomposition makes the answer **checkable**, not
just answerable.
*Risk — over-splitting*: a question that should not be decomposed becomes two partial answers that
never recombine, and the decomposition step is one more LLM call that fails quietly. Decompose only
when the question has genuine parts; and **when the intersection is empty but each part is not, say
so** — *"58 used Pythia, 12 searched top squarks, none did both"* is a far better answer than *"none"*.

#### S-54 (2026-08-02) — paraphrases: **fuse the retrievals, report the agreement, never merge answers**
Two halves of the same idea, with opposite risk profiles.

**Safe, and valuable — fuse retrievals.** Paraphrase the question, run `search` on each, fuse the
entity sets with **RRF** (already built). Answer once, from the fused set. This is query expansion
(RAG-Fusion is exactly this), and it is safe because the answer is still computed by SQL over one
set. It should improve retrieval recall, which is the weakest link at scale (S-46).

**Risky — merging disagreeing answers.** Combining three different answers to "include more info" is
taking a union of things we cannot tell apart, adding items with no evidence behind them. That is how
a confidently wrong composite gets built with a clean trace. **Rejected.**

**Instead, agreement becomes a reported confidence**, i.e. S-52 promoted from internal metric to a
product feature: `58, 58, 58, 45, 58` → answer 58, flag the instability; `58, 12, 44, 9, 31` → say
plainly that this is not a reliable answer. For a coverage map, *"here is the number and here is how
stable it is"* is a feature, not a hedge.

*Sequencing (applies to S-53 too)*: **both are ablation axes, not pre-harness work.** Building them
before the testbed exists means the thing being measured moves while it is measured.
*"Decomposition: +6% on multi-constraint questions, no change elsewhere"* is a result; *"we added it
and things seem better"* is not. **Efficiency note**: paraphrase generation is needed for S-34
paraphrase-invariance testing anyway — build it once, it serves both.

### 4.13 Question truth — what learned from building it (2026-08-02)

*The harness works. The first two generated question sets did not, and the reason turned out to be
the most useful thing the day produced.*

#### S-58 (2026-08-02) — truth must be **invariant to where the concept boundary is drawn**
Backwards generation (S-33) needs to know *which entity ids constitute the thing being asked about*.
**That is exactly the concept-resolution problem the aliases layer exists to solve** — so a generator
built on a graph with incomplete deduplication keeps re-deriving a worse version of deduplication.
Two attempts, two different failures, both invisible in the summary table:

| version | concept = | result |
|---|---|---|
| v1 | one canonical **cluster** | **1 of 36 correct** |
| v2 | entities sharing a **head word** | **0 of 36 correct** |

*v1*: `search` returns a **set** and `count` counts over all of it — correct, because a physicist
asking about Pythia means every Pythia. But `Pythia 8` and `Pythia 8.230` are separate clusters, so a
cluster-level truth is unreachable by a concept-level question. Measured on `region_requires_object`:
one `Electron` cluster = 20 papers, all electron-ish entities = 50, the system answered 52.
*v2*: head words are meaningless for descriptive labels —
`b-tagged jet` → `'b'`, `Signal Region (SR1)` → `'signal'`, `95% CL excluded mass limit` → `'95'`.
It works only for **branded single-token names** (Pythia, POWHEG, Sherpa), which is why Pythia = 58
came out right and almost nothing else did.

**The principle, replacing both heuristics**: compute the answer under the **narrowest** reading (one
entity) and the **broadest** (everything a widening rule links), and keep the question **only if they
agree**. If they differ, no answer is defensible and the question should never have been asked. It
does not resolve ambiguity — it detects it and refuses.
*Measured*: **242 well-posed, 224 ambiguous — 52% survive.** e.g. `Sherpa 2.2.1` narrow 15 / broad 22
→ dropped; `Powheg Box v2` 18 / 21 → dropped.
*And the discard rate is itself a result*: **48% of graph concepts cannot be counted unambiguously**
— a direct measurement of how incomplete deduplication is, and the number the aliases work should be
measured against. It should fall as dedup improves.
*Consequence for the plan*: **the aliases layer is upstream of the question set**, not parallel to it.

#### S-59 (2026-08-02) — six question tiers, ordered by what their truth is anchored on

| tier | shape | truth anchored on | tests | needs dedup? |
|---|---|---|---|---|
| **A** | per-paper | **`paper_id` — a hard identifier** | recall inside a paper | no |
| **B** | concept counting | invariance-tested concept (S-58) | aggregation, coverage | **yes** |
| **C** | ordinal ("more than?") | the *direction*, not the magnitude | robust ranking | partly |
| **D** | unique-anchor lookup | a safe-unique entity | **hard retrieval** | no |
| **E** | unique anchor **+ hop** | a safe-unique entity | **multi-step reasoning** | no |
| **F** | intersection | two Tier-B concepts | cross-paper set logic | **yes** |
| **D-del** | D with the fact removed (S-55) | what we deleted | **abstention** | no |

**Tier A is the backbone** and should have been from the start: `paper_id` is unambiguous, so truth is
exact and completely dedup-independent. ~60 papers × 9 predicates ≈ 540 questions.
**Tier C** recovers questions Tier B must discard: absolute counts are unknowable under ambiguity but
the *ordering* often is not, because widening moves both sides together. Keep it when
`sign(narrow_A − narrow_B) == sign(broad_A − broad_B)`. Worked example: Sherpa 15/22 vs HERWIG++
3/17 — Sherpa wins under both readings, so *"do more analyses use Sherpa or Herwig?"* is safe.
**Tier E is the most valuable new shape**: a safe-unique entity pins one paper unambiguously, then the
question hops from it (*"the analysis that defined region X — which generators did it use?"*). Exact
truth **and** genuine multi-step reasoning, which counting questions cannot test at all.
**Tier D-del closes the 2026-08-01 question** about designing genuinely unanswerable questions:
delete a safe-unique fact and its Tier D question becomes truly unanswerable, not merely answerable
from elsewhere.
*Open*: Tier F intersections shrink fast — yield needs measuring, not assuming.

#### S-60 (2026-08-02) — noisy alias proposals are an ambiguity **filter**, never merges
`data/processed/aliases_proposed.json` (28 July, 68 MB) already holds the Tier 2/2.5/3 output:
**126,335 pairs adjudicated, 4,238 marked match (3.4%)**. It was never confirmed into `same_as`, and
it must not be — see D-044: precision is roughly half, and the errors are physics errors
(dilepton ≡ four-lepton, leading ≡ subleading).

**But it is immediately useful in the other direction.** Fold it into the **broad** reading of S-58:

```
narrow = the one entity
broad  = everything a spelling rule links  ∪  everything the LLM proposed merging
keep the question only if narrow == broad
```

A false-positive proposal then only **drops a question**; it can never produce a wrong answer. *The
50% precision that makes the file unusable for merging is harmless for filtering, because the filter
errs in the safe direction.* This is what makes the question set **not blocked on dedup** — smaller
than it would be with good dedup, and growing as dedup improves, which is another before/after
measurement.
*Why the spelling rule alone is not enough*: it widens by substring, so `b-jet` and
`Bottom (b)-tagged jet` are never linked. Semantic variants are exactly what Tiers 2/3 catch and a
string rule cannot.

#### S-61 (2026-08-02) — bound every generation, and bound every question
Two guards, from one failure: a single question consumed **an hour** of a 53-question run and
produced nothing, while the server sat healthy generating at 26 tok/s.
- **`MAX_COMPLETION_TOKENS = 800`** in the planner. The arithmetic is the point, because the failure
  looks like a hang rather than an error: the client allows 120 s with 3 retries, generation runs at
  ~26 tok/s, so **any completion past ~3,100 tokens times out, is retried, and regenerates** —
  6 rounds × 4 attempts × 120 s ≈ 48 minutes. S-29 exists precisely so the model never retypes ids,
  so a long completion now means something has gone wrong; truncating makes it a *visible bad answer*
  instead of silence.
- **A per-question timeout in the runner** (180 s default). The harness must not depend on the
  planner being well behaved: a run is a long unattended job, and **one pathological question must
  cost a minute, not an evening.** Same reasoning as flushing every record.
*Third occurrence of the same family*: 2026-07-31 the model wrote 6,521 characters of ids until
truncation. **Long generations are this system's recurring failure mode**, and until now nothing
bounded them.
*And the fix broke it differently first*: running `answer` in a worker thread made SQLite refuse the
main-thread connection — 53 instant `ProgrammingError`s. Fixed with `check_same_thread=False`, which
is safe **only** because the connection is `mode=ro` and callers use it serially.

#### S-62 (2026-08-02) — per-question item analysis across systems
*User's idea, and it is the right instrument*: group run records by `qid` across every run and system.

```
qid                stub  planner  planner-nodedup  RAG    verdict
gen-count-4258b5f7   0      0            0          0     suspect the QUESTION
gen-count-0ae50ffe   0      1            0          1     real signal about a system
```

**A question every system fails is a suspect question; a question one system fails is a finding.**
The plumbing already supports it — every record carries `qid`, `system` and `config_hash` — so it is
a report mode over `eval/runs/`, needing no re-runs. It would have flagged both generator failures
immediately instead of after two full runs.


#### S-63 (2026-08-03) — the graph needs the INVERSE hop: paper -> its contents
Every template started from a concept and found papers; nothing went the other way. The consequence
was not a poor answer but no answer: *"which generators does analysis 2001.06899 use?"* produced 60
`unknown_entity_id` errors across 45 questions, because the planner had nowhere to put an arXiv id
(D-047). For a **coverage map** the most natural question about one paper is *"what does this analysis
cover?"*, so the gap was a product gap before it was an evaluation gap.
*Why it stayed hidden*: every question shape built before Tier A starts from a concept. A question set
does not only score a system, **it reveals where the system cannot go** — and this took one run.

#### S-64 (2026-08-03) — Tier A is the backbone, and it is the only tier that reaches free text
1,352 per-paper questions. `paper_id` is a **hard identifier**, so the truth never moves when
deduplication changes, and no invariance test is needed.
*The part that matters beyond convenience*: three of the highest-volume predicates are 97-98%
free-text VALUES (`object_has_selection`, `region_has_selection`, `result_reports_quantity`), so
**31% of all assertions have no entity to deduplicate and no concept to count over.** Concept
questions cannot touch them; a per-paper question reads them straight out. Value-carrying predicates
get counting questions only — there is no entity id to match a named answer against.
*Truth is deduplicated WITHIN the paper* (D-046), or a region stored under two ids makes a correct
answer score wrong.

#### S-65 (2026-08-03) — retrieval questions: the one measurement ambiguity cannot spoil
720 questions whose truth is a single **entity id**, scored only on whether that entity came back.
A *count* depends on where the concept boundary is drawn; *"did this entity appear in what was
retrieved"* does not — so this applies to the **81% of concepts Tier B must discard**, which is most
of the graph.
599 are rare (<=2 papers) and **480 have one spelling in the whole corpus**: where retrieval should
fail, and where a coverage map most needs to work.
*Stated as a hypothesis, not a result*: difficulty is labelled by paper count. If failures do not
concentrate in that bucket the label is wrong and belongs on spelling count instead.

#### S-66 (2026-08-03) — applying merges GROWS the question set; using them as a filter shrinks it
Counterintuitive and worth stating plainly, because the instinct is backwards. A *more relaxed*
aliases layer widens `broad`, so more concepts disagree with `narrow` and **fewer** Tier B questions
survive — measured: 45% survival on spelling alone, **19% once the proposals are folded in**.
What grows the set is **confirming correct merges**: `_papers_for` expands through
`entity_canonical`, so a confirmed merge makes `narrow` and `broad` converge and the question becomes
well-posed.
**So reviewing the merge proposals is not quality assurance, it is what unlocks the question set** —
and the size of the usable Tier B set becomes a direct measure of deduplication quality, which is
S-11 with a number attached.
*The cost of the current state*: Tier B selects for RARE concepts — 69 of 109 appear in exactly two
papers, and *"how many analyses used Pythia?"* (58) does **not** survive, because Pythia has 56
spellings. So it cannot yet stand in for the coverage-aggregation questions that motivate the
project.

#### S-67 (2026-08-03) — rewording: two modes, opposite goals, mechanical guards
Template questions are robotic (*"In how many papers is Pythia 8 recorded as generator?"*). An LLM
pass adds natural wording **without ever letting a model near the answer** — the truth is inherited.

| mode | goal | guard |
|---|---|---|
| **faithful** (Tier B) | same meaning, better English | the entity name **survived** |
| **vague** (retrieval) | the question someone asks who does NOT know the name | no token of the name **leaked** |

*Both guards are mechanical on purpose*: "keep the name" and "do not name it" are exactly the
instructions a model half-follows, and a hoped-for constraint is not a constraint. A rewrite failing
its guard is **discarded, never repaired** — a silently-drifted question keeps the OLD truth, which is
worse than a clumsy one.
*Measured*: faithful 5/5 kept; vague ~33% with bounded retries. The low vague rate is the guard
working — a leaked name turns the retrieval test into string matching and measures BM25 instead of
retrieval.
*Which model writes them depends on the claim*: for measuring whether rewording adds ambiguity (the
signal being how much MORE the same system retrieves) any model does. For claiming the system handles
natural phrasing, the rewriter must be a **different family** from the answerer (S-37), or questions
get phrased in the vocabulary the answerer already finds easiest.


### 4.14 One ladder of abstraction (2026-08-07)

#### S-68 (2026-08-07) — aliases, grouping and facets are RUNGS, not three systems
The supervisor's `analysis_facets.jsonl` (60 cards, 49 KB, committed on
`origin/kg-evaluation-questions`) maps every messy label onto a **closed vocabulary** -- 18 object
keys, 22 generators, 13 background methods. That looked at first like a better deduplication. It is
not: it is a **different rung on the same ladder**, and seeing it that way unifies four things this
project had been treating separately.

```
raw entity ids           227   "b-tagged jet (DeepCSV)" · "Reconstructed b-tagged jet"
                                 |  aliases layer -- same thing, different spelling
canonical                194   one node per concept-as-written
                                 |  grouping layer -- same family, different variant   (NOT BUILT)
families                 ---   all b-tag working points together
                                 |  facet mapping -- same category
facet key                 18   BJet
```
*(detector objects; the whole graph is 5,114 entities -> 4,772 canonical, 342 merged so far)*

**Every rung is the right answer to some question, and both measured failures are wrong-rung
errors.** Tier B scored 0.058 by answering too coarse -- 60 near-neighbours counted as one thing.
The supervisor's trap question (Q5, "which analyses reconstruct a Higgs-boson *candidate*") fails
the other way, too fine -- string-matching "Higgs" drowns in Higgs *process* papers where
`objects ∋ HiggsCandidate` gives exactly two.

*Facets are NOT deduplication.* Deduplication **discovers** that two labels denote one thing and
keeps what it does not merge; the facet mapping **classifies** into a predefined bucket and
deliberately discards detail. `b-tagged jet (tight WP)` and `b-tagged jet (relaxed WP)` were
correctly *rejected* as a merge by the adjudicator, and both map to `BJet` anyway -- right for
"which searches use b-jets", wrong for anything about working points.

*And it covers less than half the graph*: a facet field classifies **2,375 of 5,114 entities (46%)**.
The largest uncovered kind is `event_region` (771), then `sample` (551), `result` (272), `channel`
(208) -- precisely where the free vocabulary is worst (`SR^Z_1A`, `MB/BDT-CRW`). Those keep the raw
and canonical rungs and will want a family rung (signal / control / validation), which is what the
supervisor's Q6 asks about.

#### S-69 (2026-08-07) — retrieve every rung; let the critic pick the level
**Do not route on abstraction level before retrieving.** A router must decide "is this a facet
question or an entity question?" *before it knows what is there*, and a wrong guess returns nothing
with no signal that anything went wrong.

```
question ─┬─→ entity search   60 candidates, with each hit's canonical / family / facet position
          └─→ facet search    matching enum keys
                     ↓  critic: which of these bear on THIS question?
                     ↓  planner: count / answer over the survivors, ORIGINAL rows returned
```

Returning the original rows is what keeps the evidence chain and the mechanical faithfulness check
(S-14) intact -- a facet key is used as a *filter*, never as the answer.

**The validation is symmetric and already built**, which is the strongest argument for this design:

| set | catches |
|---|---|
| supervisor's Tier 1 | the critic **under**-uses facets → drowns in string matches (his Q5) |
| our Tier B | the critic **over**-uses facets → answers `Modelling` where a specific systematic was asked |

Neither set alone finds both failures. A critic that always takes the cheap facet answer is caught by
Tier B; one that ignores facets is caught by Tier 1.

*Flag before filter* (D-016, D-018): a wrong discard is silent -- nothing in the trace shows what was
removed. Stage 1 marks rows relevant/not and removes nothing, and only once the flags are shown to
agree with what the answer actually used does it cut.

#### S-70 (2026-08-07) — keeping the raw rung changes how dangerous a merge is
The aliases layer has been cautious because a wrong merge silently corrupts every count that follows
-- which is why 342 of 5,114 entities are merged and nothing is written to `same_as`.

**If the raw rung is always retained and retrievable, a wrong merge only corrupts that rung.** The
specific entities remain queryable and countable. A bad merge stops being a corruption of the graph
and becomes a **wrong option on a menu**, which the critic can decline and which the question sets
can measure.

That is a real loosening: not that merges become free -- a wrong one still misleads whenever the
critic picks that rung -- but "this could silently ruin the graph" becomes "this could give a wrong
answer to some questions", which is recoverable.
*Consequence for grouping*: it stops being a second kind of merge that must be got right, and becomes
one more rung offered alongside the others.

---

## 5. Order of work

Roughly one month of building, then 15 days for final tests and >50 pages of writing.

**Phase 1 — foundations**
1. **M3 final-state signatures** — **now an upstream dependency (2026-07-30).** Gabriel is working
   on populating signatures in the acquisition pipeline, so **do not build the parse yet** (D-038
   describes the fallback, not the plan).
   *The risk this creates*: the backbone of the evaluation now sits on a deliverable we do not
   control. Mitigation, in order:
   - **Ask for the JSON shape now, not the data.** The shape is what unblocks us — the query layer
     and the coverage templates can be written against it before a single row exists.
   - **Ask for a date**, and whether the existing 60 bundles get re-extracted or only new papers
     carry signatures. Re-extraction means a re-import, which is M4 territory.
   - **Fallback trigger**: if no signature data has landed by the time the query system is working
     (~end of Phase 2), run the D-038 parse on the 138 labels as a stopgap. One afternoon, and the
     evaluation is not held hostage.
   *Context*: 161 assertions, all 60 papers, **126 distinct labels out of 138**, `signature` column
   0 populated. Note `assertion.signature` is a **third object shape**
   (`CHECK (object_id + object_value + signature = 1)`), so signatures arriving upstream means those
   assertions change shape — another reason re-import matters.
2. **Accepted-view filter** — **PARKED (S-23)**, see below.
3. **Evaluation harness + frozen question set** (S-10, S-11).

#### S-23 (2026-07-30) — query everything for now; the accepted-view filter is deferred
No status filter on the query path yet. Every assertion is queryable regardless of status.
*Why*: human review has barely happened — one paper accepted (2001.06899) and 287 decisions staged
for another. Filtering to "accepted" today would filter on the *extraction pipeline's* own status,
which is not the same thing as "a human checked this", while silently hiding most of the corpus.
Better to query everything and be honest that nothing is reviewed yet.
*Cost of deferring*: low — it is a `WHERE` clause. But build the query path so the filter is a
**flag that can be switched on**, not something bolted in later.
*Watch for*: numbers measured before and after the filter exists are **not comparable**. If review
lands mid-project, re-run the affected evaluation rather than comparing across the change.

#### S-22 — the harness is built *before* the query system
Point it at a stub that answers nothing, and watch the number move from day one.
*Why*: about a day's work, and it is the difference between "we measured it" and "we measured it at
the end." It also forces the question freeze, which is the highest-value cheap thing available.

**Phase 2 — the query system**
Router (S-03) → templates (S-04) → execution and checks (S-05, S-06, S-21) → cited answers (S-14).
Tool calling (S-20). Tracing to JSONL. Streamlit page alongside it, not after — one day, and it ends
the era of reading JSONL by hand.

**Phase 3 — measurement** *(agreed with the user 2026-08-02; §4 is the detail)*

**Step 0, today, before anything else — one message to Gabriel.** He is the longest-latency item in
the project and none of him is parallelisable. Ask for **~30–50 questions, no answers**, varied, with
a mark on the ones he expects to be hard; plus *"is `GROUND_TRUTH.md` current?"* (D-042) and
*"chATLAS access / what does the benchmark cover?"* (S-45). **One message — he is slow, so ask once.**
*Second reason for asking first, which is the user's and is better than the obvious one*: his
questions are the check on whether **our generated questions are the right shape at all** — the
direct answer to the S-33 bias.

| block | ~days | contents |
|---|---|---|
| **1 — testbed skeleton** | 2–3 | harness **first** (it defines the question format) → backward question generation, **tagged** (S-33, S-10) → **variance baseline** (S-52) → **plan type-checking** (S-35) → cheap baselines (S-44 arms 1, 2, 2b) → **the D-038 signature parse** (see below) |
| **2 — enrich it** | 2–3 | metamorphic composites from block 1's questions (S-34) → growth curve **version B** (S-47, free) → hops-to-node + tool-selection (S-48, S-49) → **soft** deletion (below) |
| **3 — background, from ~day 3** | — | targeted verification pass (S-37 amendment) · GraphRAG indexing — a long job, start it and leave it |
| **4 — when the questions land** | 1 | **tag against M3 first** (below) → check answerability against the verification records → **close the set** |
| **5 — ablation study** | 3–4 | dev set only, variance known, one axis at a time. Axes include S-53 and S-54. |
| **6 — Gabriel's two hours** | — | judge calibration → full-scale judged comparison (S-40, S-42) |
| **7 — chATLAS** | — | if access lands |

Four rules that make the difference between an ablation study and a week of chasing noise:

- **Fix known bugs before measuring.** S-35 is a *fix*, not only a metric. Starting the search for
  "the best pipeline version" with a live defect means every comparison inherits it.
- **Measure run-to-run variance before any ablation** (S-52). If identical runs swing ±5%, a 3%
  "improvement" is noise. The variance sets the smallest difference that is readable at all.
- **Dev/test discipline** — *generated questions are DEV, tune freely; **Gabriel's are TEST, opened
  once**.* Reading his questions to check alignment and type coverage is fine; iterating the system
  against them is not, and it would invalidate the only realistic set we have (S-10).
- **The thing to resist is starting with Gabriel.** Blocks 1–2 need no human and no judge.

#### S-55 (2026-08-02) — deletion is a query-time filter first, a rebuild only if needed
Real deletion means rebuilding entity rollups, canonical clusters and the retrieval index. S-23
already requires the accepted-view to be a **switchable filter**, so:
`WHERE paper_id NOT IN (…)` — reversible, no rebuild, no risk to the store.
*It is not a complete deletion*: the entity survives in the search index, so the system can still
find the **name** while the **facts** are gone. That tests something arguably more interesting — does
it say *"I found this but nothing is recorded"*, or does it invent? Do the soft version first; hard
deletion with a rebuild only if the soft one proves insufficient.
*Choosing what to delete* (the 2026-07-31 lesson): select by **fact specificity**, not paper
membership — entities appearing in exactly one paper — and **apply dedup first**, or one spelling is
"deleted" while its twin remains.

#### S-56 (2026-08-02) — run the D-038 signature parse now; prose reads but does not count
*The question (user, 2026-08-02)*: with no `signature` column populated, can the model not just read
the final state out of the quotes?
**For a single paper, yes** — retrieve the assertion, read the prose, answer. Those questions are not
blocked, and the earlier framing ("none are answerable") was too strong.
**For aggregation, no** — and aggregation is the product. 161 assertions carry **126 distinct labels
out of 138**: almost every paper words it differently, and prose cannot be `GROUP BY`-ed.
*The tempting shortcut, rejected*: hand the model all 161 and let it bucket them at query time. It
fits in 32k and it would work — but **the count would then come from the model, not from rows**:
different buckets on different runs, no `COUNT(*)` to point at, and `verify.py` cannot check a number
that appears in no retrieved row. That is exactly the "confident answer, clean trace, nothing behind
it" failure the design exists to prevent.
**Decision**: run the D-038 batch parse into a derived table **in block 1**, keeping the prose as the
evidence it was parsed from — counts come from SQL and each one traces back to its sentence.
Auditable *and* countable. One afternoon, and it stops the evaluation backbone resting on a
deliverable we do not control.

#### S-57 (2026-08-02) — tag Gabriel's questions against M3 the moment they arrive
Before running anything: how many of his questions need final-state signatures? The vault's own
framing makes aggregate coverage questions the backbone, and `assertion.signature` is **0 populated**.
If a large share need signatures, that is not a query-layer problem and no evaluation work fixes it —
**it is the trigger for S-56**, and his questions are what decides whether the parse is urgent or
merely useful.

**Phase 4 — the targeted extensions**
Paper-reading loop with recovery measured against the deletion set (S-13), provenance from the start
(S-08). Then the user subgraph and editing, reusing the M4 diff machinery (S-09).

**Gap finding**: future work.

### Schedule notes
- Something will slip. Cluster problems have occurred every week so far (MIG partition, stale
  script on DIAS, SSH timeouts, a failed job) — that is the base rate, not bad luck.
- **Freeze the code on day 1 of the 15, run the full evaluation on days 1–3, write from real
  numbers.** Leaving the final run to day 12 and finding a problem is the single largest risk in the
  plan.
- **Start writing now.** `vault/` is already most of a methods chapter — 38 numbered decisions with
  context and measured justification, plus the encoder bake-off, the guard fix that passed 29 tests
  and was still wrong, the identity fix, and the calibration result. Converting that to prose is a
  much smaller job than writing from scratch, and it fits in the gaps while jobs run.

---

## 6. Open questions

**System**
- How the answer format is chosen (text / table / plot / graph) — LLM, or by question type.
- How the user's subgraph is stored: a real table, a Neo4j subgraph, or just the ids returned in a
  session.
- Sandbox specifics for the plotting agent.
- Whether the demo runs on the 72B (24h SLURM cap, node address changes) or a smaller always-up
  model. The user is content to reconfigure by hand for now; revisit before the final demo.

**Carried over from the deduplication work (settle with evaluation, per §5 Phase 3)**
- Merge threshold — gate on paper count, confidence, or both.
- **Whether Tier 1/1.5's existing merges should be re-examined** — ~21% of single-paper merges look
  wrong, and those merges are live in the graph.
- Reject-aware clustering (the transitive-closure fix).
- Symbol-swap guards; the "same kind + identical label" free tier (164 pairs).
- Label → entity_id writeback into `same_as`.
- Whether groups become first-class nodes.
- Porting Gabriel's `vocabulary.py` — as a **veto signal and grouping**, never as a merge signal.

**Data**
- The 23 `result_has_final_state` rows carrying a bare `object_value` instead of a node.
- Ask Gabriel whether `objects-v2` / `facets-v1` are frozen, and whether bundles will carry
  `canonical` / `facets`.

---

## 7. Relationship to the existing milestones

The contract milestones are not a detour from this system — they are its substrate.

| | |
|---|---|
| **M2** — trace | partial; `trace_assertion` built. Becomes one of the query templates. |
| **M3** — accepted view + physics query | **critical path.** The accepted-view filter and final-state signatures are what the query layer stands on. Nothing in §4 is measurable without it. |
| **M4** — re-import after review | the correction loop the user-editing stage reuses (S-09). |

---

## Changelog

**v1.8 — 2026-08-07. One ladder, not three systems. S-68 … S-70.**
The supervisor landed `signatures-v1` (confirming D-038 — signature empty on all 14,188 assertions,
the semantics sitting in flat qualifiers) and a **15-question tiered test set with computed gold**,
plus `analysis_facets.jsonl`: 60 closed-vocabulary cards, 18 object keys.
**S-68** reframes aliases, grouping and facets as **rungs of one abstraction ladder** rather than
three systems. Both measured failures are wrong-rung errors: our Tier B answered too coarse (0.058),
his Q5 trap catches answering too fine. Facets are *not* better deduplication — they classify and
discard detail where dedup discovers and preserves — and cover **46% of entities**, missing
`event_region` (771) entirely.
**S-69** retrieve every rung and let the critic pick the level, rather than routing before retrieval.
The validation is symmetric and already built: his Tier 1 catches under-use of facets, our Tier B
catches over-use.
**S-70** keeping the raw rung means a wrong merge corrupts one rung rather than the graph — which
loosens the caution that has held the aliases layer at 342 merges.
**Blocking**: 9 of his 15 questions need the facet and signature layers imported, which is the
"vocabulary port" open since 2026-07-27.

**v1.7 — 2026-08-03. Three question tiers, and a missing primitive. S-63 … S-67.**
**`contents_of` (S-63, D-047)**: the graph could go concept -> papers and had **no inverse**, so
*"which generators does this paper use?"* was unanswerable — 60 `unknown_entity_id` errors across 45
questions. A coverage map that could not read a paper. Adding it moved `count_correct` 0.06 -> 0.50
and `tool_error_rate` 0.386 -> 0.083.
**Tier A** (S-64) 1,352 per-paper questions, exact truth, no deduplication needed, and the only tier
that reaches the **31% of assertions stored as free text**. **Tier B** 436 questions from 109
invariance-tested concepts. **Retrieval** (S-65) 720 questions whose truth is an entity id, immune to
ambiguity, covering the 81% Tier B discards.
**S-66**: applying merges GROWS the question set while using them as a filter shrinks it — so
reviewing the proposals is what unlocks Tier B, not merely quality assurance. **S-67** rewording with
opposite guards per mode.
**Four measurement bugs, all the same mistake** — a metric anchored to one code path rather than to
the behaviour — each found by an impossible-looking number rather than a crash. Chief among them
D-048: `_evidence_for` silently returned **every evidence row in the graph** (11 rows of output,
8,369 evidence ids, no error).
Infrastructure: **D-049** compute-0-1 accepts jobs and runs nothing; `langgraph` was missing from the
DIAS venv; the circuit breaker worked and then corrupted its own output file.

**v1.6 — 2026-08-02 (afternoon). The harness is built; the questions are not. S-58 … S-62.**
`hepcoveragekg/eval/` — 6 modules, **335 tests**, runs *any* `System` so baselines and ablations share
one runner. First real run: 10 questions x 3 repeats, **count_correct 0.75**, abstention 1.00, and
**zero spread across repeats**, so the noise floor is ~0 and any ablation difference will be readable.
**The day's finding is that both generated question sets failed** — 1/36, then 0/36 — because
**truth must be invariant to where the concept boundary is drawn (S-58)**. A cluster is not a concept;
head words are meaningless for descriptive labels. The cause is circular: writing a question with a
known answer requires knowing which ids constitute the thing asked about, *which is the aliases
problem*. 52% of concepts survive the invariance test, and **the 48% discarded is a direct measurement
of how incomplete deduplication is**. Six tiers defined (**S-59**), per-paper questions as the backbone,
and **Tier E (unique anchor + hop)** as the first shape that tests multi-step reasoning at all.
**S-60** the deep alias proposals become an ambiguity *filter*, never merges — a false positive then
only drops a question and can never produce a wrong answer, so the question set is not blocked on
dedup. **S-61** bound every generation and every question, after one question consumed an hour of a
53-question run. **S-62** per-question item analysis across systems.
Related: **D-043** `list_papers` disagrees with `count` · **D-044** the proposals must not be confirmed
(~50% precision, physics errors, 4,144 of 4,238 at confidence exactly 0.9) · **D-045** same `entity_id`
across papers is mostly the same concept, reversing a claim made that morning · **D-046** 22
intra-paper duplicates as a free dedup benchmark.
**Deduplication is now upstream of the question set**, not parallel to it.

**v1.5 — 2026-08-02. The order of work, agreed; S-53 … S-57.** §5 Phase 3 rewritten into blocks with
**step 0 = ask Gabriel today** (longest-latency item; and his questions are the check on whether our
*generated* ones are the right shape — the user's reason, better than the obvious one).
Four rules added that separate an ablation study from a week chasing noise: **fix known bugs before
measuring**, **measure run-to-run variance first**, **dev/test discipline** (generated = dev, his =
test, opened once), and *the thing to resist is starting with Gabriel*.
New: **S-53** decomposition compiles to **set algebra** rather than sub-answers — possible only
because `search` produces named sets (S-29), and it makes the answer *checkable* via
inclusion–exclusion · **S-54** paraphrases: **fuse the retrievals** (RRF, already built), **report the
agreement**, and **never merge disagreeing answers** · **S-55** deletion is a query-time filter first,
a rebuild only if needed · **S-56** run the D-038 signature parse now — prose *reads* but does not
*count* (126 distinct labels of 138), and letting the model bucket them at query time would put the
count outside `verify.py`'s reach · **S-57** tag Gabriel's questions against M3 the moment they land.
**Amendments to yesterday, both from the user**: the reference reader defaults to **targeted
verification, not a full second pass** — a question-agnostic sweep *is* re-running extraction, and
~10× more than the evaluation needs (S-37 amendment); and arm 2 is built **by hand on our own
encoder**, matching GraphRAG's "naive RAG" construction, plus a **BM25-only** arm (S-44).
**S-53 and S-54 are deliberately not built yet** — they are ablation axes, and building them before
the harness would move the thing being measured.

**v1.4 — 2026-08-01. The evaluation chapter (§4 rewritten, S-32 … S-52).** A full session spent on
"how do we evaluate this with no curated ground truth". The answer that reframed everything:
**three layers get measured separately (S-32)**, and *almost everything being built is L1 — the
agent against the graph — where the graph itself is perfect ground truth.* No judge required for
most of it.
New: questions generated backwards from the graph (**S-33**) · metamorphic invariants (**S-34**) ·
plan type-checking, which turns the 2026-07-31 silent failure into a guard *and* a metric
(**S-35**) · ablations as evaluation (**S-36**).
**The reference reader (S-37…S-39)** — the user's idea, and the strongest item in the plan, because
it is the only thing that measures the *graph against the papers*: one pass per paper, constrained
to our schema, on a **third** model (extraction is Sonnet, query is Qwen — so the reader must be
neither), producing a second graph that can be diffed against ours at three strictness levels.
Expert time buys **calibration and disagreements, never labels** (**S-40**); prose answers are
scored against a **checklist**, never compared (**S-42**, fixing the user's A-vs-B objection);
**pooling with quote adjudication** for scale (**S-43**).
Baselines (**S-44**): two cheap arms buildable immediately, and **GraphRAG reframed** — not "who
wins" but *"does hand-curating a typed schema buy anything?"*, which is the obvious challenge to the
whole project and currently unanswered. Benchmarks surveyed (**S-45**): none measure aggregate
coverage; chATLAS_Benchmark is the one in-domain option and needs an hour of checking.
Scale (**S-46**–**S-48**), component tests (**S-49**–**S-52**).
**Two corrections to earlier claims, both from the user**: `same_id` is *not* dedup ground truth
(347 shared vs **334 divergent**) — **S-51**; and reading all 2,969 papers is *affordable*
(~45M tokens ≈ a few dollars on a cheap API, or a day of DIAS) — the earlier "too expensive"
estimate was wrong (**S-37**).
**Two finds while checking the pipeline** (see D-041, D-042): extraction accepts
`--provider openai`, so it runs against our own vLLM — the growth curve is not blocked on Gabriel;
and `HEPKG_promopt_tests/GROUND_TRUTH.md` **already holds hand-written ground truth for ~17 papers**
on final states, unmaintained since 2026-07-05.

**v1.3 — 2026-07-31.** The query layer runs end to end against the real model and graph:
*"How many analyses used Pythia?"* → **58 papers / 344 facts / 367 assertions**, matching ground
truth exactly, 100% of claims grounded, 202 evidence quotes, 6 seconds. **280 tests.**
**S-29** named result sets (the model never handles ids) · **S-30** state describes the run, config
carries the machinery · **S-31** compaction and recall, designed and deliberately deferred.
Built: `retrieve.py` (8,449 surface forms, hybrid rank-fusion, incremental cache shared with the
aliases layer), `planner.py`, `verify.py` (mechanical faithfulness), `graph.py` (**S-16 fulfilled** —
LangGraph orchestrates, LangChain deliberately not a dependency; checkpointing verified, 7
checkpoints per run, diagram generated from the compiled graph).
Hardware: **D-040** — one A100 on compute-gpu-0-1 fails with uncorrectable ECC on the first
inference request; reproduced twice, reported, guarded against by bus id.

**v1.2 — 2026-07-30.** Query layer began. **S-24** free-form escape hatch (the router may answer
"no template fits", and the log of those *is* the specification for the next templates);
**S-25** dual-path answering, template vs free, with the two safeguards; **S-26** learned lessons
are promoted into structure, not prose; **S-27** one planner, batched operations, hierarchy only
under context pressure (measured: one LLM call ≈ 3,000–17,000 queries); **S-28** clusters collapse,
groups expand, and merges are shown beside the numbers they change.
Also: the schema card now prints **real kind distributions** instead of a "mixed" verdict behind a
60% threshold (~300 extra tokens, and it was hiding an 86/14 split); `PURPOSE` exists in **full and
minimal variants with section markers**, so how much prompt instruction actually helps is an
ablation rather than a guess.
**Built**: `query/schema_card.py`, `query/templates.py`, `query/prompts.py`. 215 tests.
**Two silent traps found in the pilot data and encoded in tested SQL** — join to papers via
`entity_occurrence` (the obvious predicate reaches 60 of 272), and count facts not rows (up to 2x
over-report). Both return a plausible wrong answer rather than an error, and both are now
self-discovered evidence for S-04.

**v1.1 — 2026-07-30.** S-23: the accepted-view filter is parked; query everything until review
actually exists. Recorded that `assertion.signature` is a **third object shape** in the schema
(`CHECK (object_id + object_value + signature = 1)`), 0 populated — so the parsed final state cannot
be written back onto an existing assertion and needs its own derived table, consistent with S-02.
Added [`hep-notation.md`](hep-notation.md).

**v1 — 2026-07-29.** First version. Captures the system as designed across the 2026-07-29 session:
purpose reframed from gap-finding to coverage review (supervisor conversation); query system as sole
entry point; router/template/validation design; SQL-only querying; evaluation design (three arms,
deletion protocol, per-layer metrics, frozen and tagged question set); technology stack including
the LangGraph decision; ordering; open questions. S-01 … S-22. Related: D-038 (final states),
D-032 … D-037 (deduplication base).

**v1.7 — 2026-08-02 (evening).** Aliases layer rebuilt: graph context wired into the real pipeline (it was only in the trial-set evaluator), entity_id keying, LaTeX cleaning, streamed output, paradox counting. Re-run: 10,839 pairs, 2,262 matches, **stability 100%** at temperature 0. **Transitive closure over-merges** — 75 statistical methods in one cluster; 15% of rejections contradicted. **Dedup emits pairs, grouping does families (user).** Whole-cluster checking: 79 of 122 small clusters should be split, ~2 minutes. Sampling at temperature 0.7 gives a behavioural confidence signal greedy decoding hides. Details in `logs/2026-08-02.md`.
