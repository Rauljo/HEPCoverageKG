# HEPCoverageKG

A typed knowledge graph of ATLAS and CMS papers, and an agentic query system that
answers **coverage questions** about them: *which analyses use b-tagged jets in
their event selection?*, *how many papers estimate a ttZ background from a control
region?*, *what does paper 2405.18661 measure?* Every paper in an answer comes with
the quotes from that paper that justify it.

This is the code of the MSc dissertation *Agentic GraphRAG over a knowledge graph of
high-energy physics literature* (Raul Jose Salgado, UCL, 2026), supervised by
Gabriel Facini. The dissertation is the documentation; this file says where things
are and how to run them.

## What is here

The work has three parts, and the repository follows them.

**1. The graph.** Per-paper extractions (JSON "bundles" produced by the supervisor's
[HEPKG](https://github.com/gfacini/HEPKG_promopt_tests) pipeline) are validated and
imported into a SQLite database under a fixed, closed ontology. Every assertion keeps
the quote it came from. Two layers are derived on top: an **aliases layer** that merges
the same concept written in different ways (`b_jet`, `bjet`, `b-jet`), and a **facet
layer** of closed vocabularies. A Neo4j projection is exported for traversal and
visualisation. The schema is drawn in [`docs/`](docs/README.md). The graph used in
the dissertation holds a 60-paper pilot.

**2. The query system.** A physicist's question is turned into a sequence of graph
queries by an LLM agent running a ReAct loop (plan, execute, observe, until it
answers). Two agents exist:

- the **typed agent** has 12 fixed tools (one hybrid search tool, ten SQL templates,
  and `answer`) and runs on LangGraph;
- the **free-SQL agent** has three tools (`search`, `sql`, `answer`) and writes its
  own SQL.

Mechanisms added on top of the base loop, each built against a failure the
evaluation found: constrained decoding of the paper list, a search critic and an
**answer critic** (a judge that decides paper by paper before the answer is written),
task decomposition, a plan reviewer, task-status reflection, and **chained
sub-goals** (each sub-goal is its own short run).

**3. The evaluation and the app.** A question bank generated from the graph plus nine
supervisor-written questions with hand-judged truth, a runner that records every
tool call, and scorers for sets, counts and labels. `app.py` is a Streamlit
demonstrator: ask a question, read the answer, the papers, the quotes behind each
paper and the subgraph they stand on.

## Layout

| Path | What it is |
|---|---|
| `hepcoveragekg/ingest/` | bundle validation and import (`validate.py`, `importer.py`, `canonical.py`) |
| `hepcoveragekg/kg/` | the SQLite store and schema (`schema.sql`, `store.py`), Neo4j export |
| `hepcoveragekg/aliases/` | the aliases layer: deterministic tiers, candidate generation, LLM adjudication |
| `hepcoveragekg/facets/` | the facet and signature layers |
| `hepcoveragekg/query/` | the typed agent: `planner.py` (loop and tool schemas), `graph.py` (tools), `prompts.py`, `schema_card.py`, `fewshot.py`, `critic.py`, `answer_critic.py`, hybrid retrieval |
| `hepcoveragekg/eval/` | the harness: `runner.py`, `systems.py`, `free_sql.py` (the free-SQL agent and its prompt), `chain.py` (chained sub-goals), question generation and scoring |
| `hepcoveragekg/ui/` | what the app needs: model registry, run wrapper, provenance lookup, graph view |
| `app.py` | the Streamlit demonstrator |
| `eval/questions/` | the question files; `eval/runs/` the run summaries; `eval/review/` the supervisor's judgements |
| `hpc/` | Slurm job scripts for the UCL DIAS cluster (vLLM serving, evaluation arms) |
| `docs/` | entity-relationship diagrams of the schema |
| `tests/` | pytest suite (79 files) |
| `vault/` | the project record: decisions, ideas, logs, literature notes, written during the work |

## Setup

Python 3.11. Models are called through any OpenAI-compatible endpoint.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill it in
```

Environment variables:

| Variable | Used for |
|---|---|
| `OPENROUTER_API_KEY` | the hosted models in the app (Qwen3-32B, Qwen3.8-flash, the Qwen3.5-9B judge) |
| `LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY` | a self-hosted vLLM server for the evaluation harness |
| `QWQ_BASE_URL` | the self-hosted QwQ-32B endpoint the app offers (defaults to `LLM_BASE_URL`) |

The `.env` file is git-ignored. Never commit it.

## Build the graph

All commands run from the repository root with `PYTHONPATH=.`; `--db` defaults to
`data/processed/hepkg.db`.

```bash
python -m hepcoveragekg.cli import  data/raw/bundles/     # validate + import the bundles
python -m hepcoveragekg.cli aliases build                  # tiers 1 and 1.5, offline
python -m hepcoveragekg.cli aliases deep                   # tier 2, needs an LLM endpoint
python -m hepcoveragekg.cli facets  derive                 # facet tags
python -m hepcoveragekg.cli graph   export data/neo4j/     # CSVs for the Neo4j projection
```

## Run the app

```bash
streamlit run app.py
```

The sidebar picks the model, the agent and the effort. *Standard* runs constrained
decoding with the answer critic; *High* runs chained sub-goals. The defaults per model
are the configurations the dissertation kept.

## Run the evaluation

```bash
python -m hepcoveragekg.cli eval run  eval/questions/<file>.jsonl --system typed --answer-critic
python -m hepcoveragekg.cli eval report  eval/runs/<run>.jsonl
python -m hepcoveragekg.cli eval compare eval/runs/<a>.jsonl eval/runs/<b>.jsonl
```

`eval run --help` lists every mechanism as a flag. Every run writes one record per
answer (text, papers, tool calls, tokens) and a `.summary.json`; the summaries of
the dissertation's runs are in `eval/runs/`. The Slurm scripts in `hpc/` are how the
arms were actually run on the cluster.

## Tests

```bash
pytest
```

## Where the prompts are

The dissertation cites these files directly:

| What | File |
|---|---|
| intent block and abstain clause, both agents | [`hepcoveragekg/query/prompts.py`](hepcoveragekg/query/prompts.py) |
| description of the graph, generated from the database | [`hepcoveragekg/query/schema_card.py`](hepcoveragekg/query/schema_card.py) |
| worked examples | [`hepcoveragekg/query/fewshot.py`](hepcoveragekg/query/fewshot.py) |
| typed agent's tool schemas, stated-objective block | [`hepcoveragekg/query/planner.py`](hepcoveragekg/query/planner.py) |
| the free-SQL agent and its whole prompt | [`hepcoveragekg/eval/free_sql.py`](hepcoveragekg/eval/free_sql.py) |
| the database schema | [`hepcoveragekg/kg/schema.sql`](hepcoveragekg/kg/schema.sql) |
