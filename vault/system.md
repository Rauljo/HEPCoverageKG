# The system — design and decisions

**Version 1 · 2026-07-29**

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

**Phase 3 — measurement**
Abstention via deletion (S-13). Three-arm comparison (S-12) plus the deep-research snapshot. First
human batch (S-15). Then **curate deduplication and grouping against real evaluation** — the
machinery already exists, so this is mostly running and measuring, and it is where the open aliases
decisions finally get settled by evidence.

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
