"""The free-SQL control: the same model, the same database, one tool that takes SQL.

Nothing built so far can falsify "typed tools over a typed graph beat a model
turned loose on the data", because nothing has been turned loose. Every arm to
date varies something INSIDE the typed layer. This varies the layer itself.

Design and predictions: vault/ideas/free-sql-control.md. The short version --
it sees every table, including `entity_canonical` and `same_as`. Hiding them
would make it a straw man, since the claim under test is that the tools and the
vocabularies help, not that we hold data nobody else has. What it does not get
is the retrieval index, which is not in the database; that is a real advantage
for the typed agent and is stated as a limitation rather than papered over.

THREE WAYS THIS COULD BE RIGGED WITHOUT ANYONE INTENDING IT, and what is done
about each:

  prompt      The typed agent's PURPOSE prompt has had weeks of iteration. A
              first-draft SQL prompt is not a fair opponent. This one states the
              schema, the dialect, the row cap, and the one thing that actually
              decides these questions -- that spellings are not unified -- and
              it may be iterated on DEV questions before any reported run.
  rounds      Six, the same as the planner, so it can look, fail, and recover.
  scoring     The same scorers, and `papers` is filled from paper_id columns in
              the result exactly as the planner fills it from retrieved
              entities. A SQL agent must not score zero for returning a table.

SAFETY IS STRUCTURAL, NOT TEXTUAL. The connection is opened read-only, so a
write cannot succeed even if a string check is fooled. The statement checks on
top of that are there to give the model a useful error, not to be the barrier.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .questions import Question
from .systems import Answer, env_config as _env_config

MAX_ROWS = 200
MAX_ROUNDS = 6
STATEMENT_SECONDS = 15.0

#: One statement, and it must read. `WITH` leads a CTE, which is still a read.
#
# There is deliberately NO keyword blacklist. The first version carried one, and
# it rejected `WHERE label LIKE '%update%'` and `LIKE '%drop%'` -- ordinary reads
# over a corpus whose labels contain English words. A control handicapped by a
# false positive is not measuring what it claims to.
#
# Nothing is lost by removing it. The connection is opened read-only, so a write
# raises whatever the string says, and a statement that must begin with SELECT or
# WITH cannot be a write in SQLite. The barrier was never the regex.
_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.I)

ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": (
            "Give the final answer and stop. Put the arXiv ids you are asserting "
            "in `papers` -- that is what gets scored, so an answer that describes "
            "the right papers without listing them cannot be credited."),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the answer, in prose"},
                "papers": {"type": "array", "items": {"type": "string"},
                           "description": "the arXiv ids the answer asserts, e.g. "
                                          "['2106.01676', '2001.06899']"},
            },
            "required": ["text"],
        },
    },
}

def plain_answer_tool() -> dict:
    """The answer tool WITHOUT the scoring instruction (2026-09-09).

    The shipped description says "Put the arXiv ids you are asserting in
    `papers` -- that is what gets scored", which is the typed system's
    --name-ids instruction under another name, given to the control from day
    one while the typed control's `answer` said the opposite (D-117). For a
    fair baseline pair the control must not carry it either; this is the
    counterpart of the typed v3 contract: a `papers` field that exists and is
    described neutrally, and no word about scoring. Selected with
    FREESQL_PLAIN_ANSWER=1 and recorded in the run config.
    """
    import copy
    t = copy.deepcopy(ANSWER_TOOL)
    t["function"]["description"] = "Give the final answer and stop."
    t["function"]["parameters"]["properties"]["papers"]["description"] = (
        "optionally, the arXiv ids the answer refers to")
    return t


def constrained_papers(client, model, question: str, draft: str, touched: set,
                       context: dict | None = None) -> tuple[list, str]:
    """Constrained selection for the free-SQL side (CONSTRAINED_IDS=1; D-156).

    Same mechanism as the typed planner's: one more call whose reply must be
    {"papers": [...]} with items restricted by schema to the arXiv ids the
    run's own queries returned (`touched`), enforced by vLLM `guided_json`,
    plain JSON as the fallback. Returns (picked ids, mode).
    """
    cands = sorted(str(p) for p in touched)[:300]
    if not cands:
        return [], ""
    ctx = context or {}
    listing = "\n".join(f"{p}: {(ctx.get(p) or '')[:140]}" for p in cands)
    schema = {"type": "object",
              "properties": {"papers": {"type": "array", "items": {"type": "string", "enum": cands}}},
              "required": ["papers"], "additionalProperties": False}
    messages = [
        {"role": "system", "content": "You select papers from a candidate list. Reply with JSON only, "
                                      "of the form {\"papers\": [\"2106.01676\", ...]}, using only ids from the list."},
        {"role": "user", "content": (
            f"Question: {question}\n\nDraft answer:\n{(draft or '')[:3000]}\n\n"
            f"Candidate papers your queries returned (arXiv id: where it came from):\n{listing}\n\n"
            "List every candidate that answers the question. Leave out candidates that merely "
            "mention the concept without satisfying the question.")}]
    content, mode = "", ""
    # Standard form first: vLLM 0.18 accepts the legacy `guided_json` field
    # and ignores it (D-159); `response_format` json_schema is enforced.
    attempts = (
        ("json_schema", dict(response_format={"type": "json_schema", "json_schema": {
            "name": "papers", "schema": schema, "strict": True}},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})),
        ("guided_json", dict(extra_body={"guided_json": schema, "chat_template_kwargs": {"enable_thinking": False}})),
        ("json_object", dict(response_format={"type": "json_object"})),
    )
    for mode_, kw in attempts:
        try:
            r = client.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=2000, **kw)
            content, mode = r.choices[0].message.content or "", mode_
            break
        except Exception:  # noqa: BLE001
            continue
    else:
        return [], ""
    picked: list = []
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            for p in (json.loads(m.group(0)).get("papers") or []):
                p = str(p).strip()
                if p in set(cands) and p not in picked:
                    picked.append(p)
        except (ValueError, AttributeError):
            pass
    return picked, mode


def critic_selects_papers(conn, question: str, touched: set, named=(), chat=None) -> tuple[list, dict]:
    """The judge decides free-SQL's list (CRITIC_SELECTS=1 + CONSTRAINED_IDS=1; D-166).

    Candidates are the arXiv ids free-SQL's queries returned plus any it
    named that the graph holds. Free-SQL retrieves no entities, so the
    judge's material is built per paper from the graph: every quote the
    paper has (paper-wide evidence, D-165), and as "retrieved" labels the
    paper's own entity labels that share a word with the question, with
    their aliases -- the corpus vocabulary the quote ranking needs. Same
    prompt, same judge as the typed side. Returns (kept ids, review dict).
    """
    from hepcoveragekg.query import answer_critic as AC
    from hepcoveragekg.query import planner as _p
    cands = sorted({str(p) for p in touched} | {str(p) for p in (named or [])})[:300]
    if not cands or conn is None:
        return [], {}
    try:
        held = {str(r[0]) for r in conn.execute(
            f"SELECT arxiv_id FROM paper WHERE arxiv_id IN ({','.join('?' * len(cands))})", cands).fetchall()}
        cands = [c for c in cands if c in held]
    except Exception:  # noqa: BLE001 -- no paper table: judge them all
        pass
    if not cands:
        return [], {}
    terms = AC._question_terms(question)
    evidence = AC.evidence_by_paper_wide(conn, cands)
    for pid in cands:
        labels, quotes, aliases = evidence.get(pid, (set(), [], set()))
        try:
            rows = conn.execute("SELECT label, aliases FROM entity_occurrence WHERE paper_id = ?", (pid,)).fetchall()
        except Exception:  # noqa: BLE001
            rows = conn.execute("SELECT label, NULL FROM entity_occurrence WHERE paper_id = ?", (pid,)).fetchall()
        for label, raw in rows:
            if label and AC._overlap(str(label), terms):
                labels.add(str(label))
                try:
                    for a_ in (json.loads(raw) if isinstance(raw, str) and raw else (raw or [])):
                        if a_:
                            aliases.add(str(a_))
                except ValueError:
                    pass
        evidence[pid] = (labels, quotes, aliases)
    if chat is None:
        client, model = _p._critic_client()
        cap = _p.completion_cap(model)
        chat = lambda messages: _p.answer_critic_call(client, model, messages, cap)  # noqa: E731
    review = AC.judge_papers(chat, question, evidence)
    kept = set(review.kept)
    picked = [p for p in cands if p in kept]
    summary = review.summary() if hasattr(review, "summary") else {"kept": len(kept), "candidates": len(cands)}
    return picked, summary


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Find entities by meaning as well as by spelling: BM25 over labels "
            "fused with dense embeddings. Returns entity_id, label and kind, "
            "best first. Use it when you do not know how the papers spell "
            "something -- 'missing transverse momentum' finds MET and ETmiss, "
            "which no LIKE will. Feed the entity_ids into SQL."),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "what to look for, in your own words"},
                "kind": {"type": "string",
                         "description": "optional entity kind filter -- usually leave it out"},
                "limit": {"type": "integer", "description": "how many hits (default 20)"},
            },
            "required": ["text"],
        },
    },
}

SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "sql",
        "description": (
            "Run one read-only SQL SELECT against the knowledge graph and get "
            f"rows back. At most {MAX_ROWS} rows; add your own LIMIT for less. "
            "Errors come back to you verbatim so you can fix the query."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "one SELECT statement"}},
            "required": ["query"],
        },
    },
}


CYPHER_TOOL = {
    "type": "function",
    "function": {
        "name": "cypher",
        "description": (
            "Run one read-only Cypher query (MATCH ... RETURN) against the Neo4j "
            "projection of the SAME knowledge graph and get rows back. At most "
            f"{MAX_ROWS} rows; add your own LIMIT for less. Errors come back "
            "verbatim so you can fix the query."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "one Cypher read query"}},
            "required": ["query"],
        },
    },
}

_KIND_LABELS = {
    "detector_object": "DetectorObject", "systematic_uncertainty": "Systematic",
    "physics_process": "Process", "statistical_method": "StatMethod",
    "background_method": "BackgroundMethod", "event_region": "Region",
    "model_parameter": "ModelParameter", "object_definition": "ObjectDefinition",
    "selection_requirement": "Selection", "collision_system": "CollisionSystem",
    "result_quantity": "ResultQuantity", "bsm_model": "BSMModel",
    "generator": "Generator", "background": "Background", "observable": "Observable",
    "sample": "Sample", "channel": "Channel", "dataset": "Dataset",
    "benchmark": "Benchmark", "result": "Result", "paper": "PaperNode",
}


def cypher_brief(conn) -> str:
    """The Neo4j projection (kg/export.py) as the model must picture it.

    Same graph, different shape: the SQLite `assertion` rows become typed
    relationships between Occurrence nodes, `entity_occurrence` becomes
    HAS_OCCURRENCE, and canonical concepts are nodes labelled by kind.
    """
    preds = [(r[0], r[1]) for r in conn.execute(
        "SELECT predicate, COUNT(*) FROM assertion GROUP BY predicate ORDER BY 2 DESC LIMIT 24")]
    n_paper = conn.execute("SELECT COUNT(*) FROM paper").fetchone()[0]
    n_occ = conn.execute("SELECT COUNT(*) FROM entity_occurrence").fetchone()[0]
    labels = ", ".join(sorted(set(_KIND_LABELS.values())))
    return f"""GRAPH (Neo4j, database from the same SQLite; read with the `cypher` tool)

NODES
  (:Paper {{arxiv_id, title, category}})            {n_paper} papers. arxiv_id is the paper id, e.g. '2106.01676'. category is 'search' or 'measurement'.
  (:Occurrence {{id, bundle_id, entity_id, kind, label}})   {n_occ} entity occurrences, one per (paper, entity). id = bundle_id + ':' + entity_id.
  concept nodes, ONE label each by kind: {labels}
      each has {{id, kind, label}}; id is the entity id, e.g. 'hepkg:method:abcd-method'; label is the human name.
  (:LiteralValue {{id, value, type}})               numbers and strings that assertions point at.

RELATIONSHIPS
  (p:Paper)-[:HAS_OCCURRENCE]->(o:Occurrence)         the paper's occurrences
  (o:Occurrence)-[:RESOLVES_TO]->(c)                  occurrence -> its canonical concept node (after alias merging)
  (p:Paper)-[:MENTIONS]->(c)                          paper -> concept, directly (shortcut of the two above)
  (o1:Occurrence)-[:PREDICATE]->(o2:Occurrence|LiteralValue)   an assertion inside one paper; the TYPE is the predicate:
      {", ".join(f"{k}({v})" for k, v in preds)}
      relationship properties: assertion_id, family, status, support, qualifiers

SHAPES THAT WORK
  THE ONE RULE: ids from `search` are OCCURRENCE-level entity ids (Occurrence.entity_id). They are
  usually NOT the id of the concept node -- aliases were merged, so 'hepkg:object:bjet' resolves to
  the concept 'hepkg:object:b-jet'. Always go through the occurrence and RESOLVES_TO:
    MATCH (o:Occurrence)-[:RESOLVES_TO]->(c) WHERE o.entity_id IN [ ...ids from search... ]
    WITH DISTINCT c MATCH (p:Paper)-[:MENTIONS]->(c) RETURN DISTINCT p.arxiv_id
  (this returns every paper that mentions any merged spelling; matching c.id directly misses most)
  papers with a hop through a predicate (subject and object are occurrences of the SAME paper):
    MATCH (p:Paper)-[:HAS_OCCURRENCE]->(r:Occurrence)-[:result_uses_statistical_method]->(m:Occurrence)-[:RESOLVES_TO]->(c)
    WHERE c.id IN [ ...canonical ids, e.g. from a previous RESOLVES_TO... ] OR m.entity_id IN [ ...ids from search... ]
    RETURN DISTINCT p.arxiv_id
  papers mentioning two concepts (AND), after resolving each id list to concepts:
    MATCH (o1:Occurrence)-[:RESOLVES_TO]->(a) WHERE o1.entity_id IN [...] WITH collect(DISTINCT a) AS A
    MATCH (o2:Occurrence)-[:RESOLVES_TO]->(b) WHERE o2.entity_id IN [...] WITH A, collect(DISTINCT b) AS B
    MATCH (p:Paper)-[:MENTIONS]->(a) WHERE a IN A WITH p, B MATCH (p)-[:MENTIONS]->(b) WHERE b IN B AND p.category='search'
    RETURN DISTINCT p.arxiv_id
  exploring by label text (labels are free text; both Occurrence and concept nodes carry `label`):
    MATCH (p:Paper)-[:HAS_OCCURRENCE]->(o:Occurrence) WHERE toLower(o.label) CONTAINS 'b-tag' RETURN DISTINCT p.arxiv_id
  counting: RETURN count(DISTINCT p.arxiv_id)"""


_CYPHER_WRITE = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+(dbms|db\.create|apoc\.(create|load|export|periodic)))\b", re.I)


def run_cypher(driver, database: str, query: str, max_rows: int = MAX_ROWS,
               seconds: float = STATEMENT_SECONDS) -> SqlResult:
    """One read query against Neo4j, read-only session, row-capped, time-capped.
    Node and relationship values are rendered as their property maps."""
    text = (query or "").strip().rstrip(";")
    if not text:
        return SqlResult(error="empty query")
    if _CYPHER_WRITE.search(text):
        return SqlResult(error="read-only: MATCH/RETURN/WITH/UNWIND/CALL db.* only")
    if driver is None:
        return SqlResult(error="cypher is not available: no Neo4j connection "
                               "(set NEO4J_PASSWORD in .env)")
    try:
        from neo4j import Query  # type: ignore
        with driver.session(database=database, default_access_mode="READ") as sess:
            res = sess.run(Query(text, timeout=seconds))
            keys = list(res.keys())
            rows: list[tuple] = []
            for rec in res:
                rows.append(tuple(_plain(v) for v in rec.values()))
                if len(rows) > max_rows:
                    break
    except Exception as exc:                      # noqa: BLE001 -- verbatim to the model
        return SqlResult(error=f"{type(exc).__name__}: {str(exc)[:400]}")
    truncated = len(rows) > max_rows
    return SqlResult(rows=rows[:max_rows], columns=keys, truncated=truncated)


def _plain(v):
    """Neo4j nodes/relationships/paths -> a compact string; scalars unchanged."""
    if hasattr(v, "items") and hasattr(v, "labels"):        # Node
        d = dict(v.items()); return ":".join(sorted(v.labels)) + " " + json.dumps(d, ensure_ascii=False)
    if hasattr(v, "items") and hasattr(v, "type"):          # Relationship
        return f"-[:{v.type}]->"
    if isinstance(v, (list, tuple)):
        return json.dumps([_plain(x) for x in v], ensure_ascii=False)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return v


def neo4j_driver_from_env():
    """A driver from NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD, or None.
    The password is read from the environment (sourced .env) and never logged."""
    import os
    pw = os.environ.get("NEO4J_PASSWORD", "")
    if not pw:
        return None, os.environ.get("NEO4J_DATABASE", "neo4j")
    from neo4j import GraphDatabase  # type: ignore
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", "neo4j"))
    return GraphDatabase.driver(uri, auth=(user, pw)), os.environ.get("NEO4J_DATABASE", "neo4j")


CYPHER_EXAMPLES = """
WORKED EXAMPLES (Cypher)

Q: Which analyses use the HistFitter framework?
  search("HistFitter") -> hepkg:method:histfitter, hepkg:method:profile-likelihood-fit-histfitter
  cypher: MATCH (o:Occurrence)-[:RESOLVES_TO]->(c) WHERE o.entity_id IN ['hepkg:method:histfitter','hepkg:method:profile-likelihood-fit-histfitter']
          WITH DISTINCT c MATCH (p:Paper)-[:MENTIONS]->(c) RETURN DISTINCT p.arxiv_id

Q: How many analyses estimate a W+jets background?
  search("W+jets background") -> ids
  cypher: MATCH (p:Paper)-[:HAS_OCCURRENCE]->(r)-[:result_estimates_background]->(b:Occurrence)
          WHERE b.entity_id IN [...] RETURN count(DISTINCT p.arxiv_id)
"""


def schema_brief(conn) -> str:
    """The DDL, plus what the typed agent gets for free: how big things are and
    what the values look like.

    A schema without sample values is not a fair starting point. The typed agent
    is handed closed vocabularies -- `objects`, `process_families` -- while a SQL
    agent facing `entity.kind` has no way to guess that the value is
    `detector_object` and not `object` or `Detector Object`. Showing a handful
    of real values costs nothing and removes a difficulty that is about guessing
    our column conventions rather than about querying a graph.
    """
    parts: list[str] = ["TABLES (SQLite)"]
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        " ORDER BY name")]
    for table in tables:
        cols = [(r[1], r[2]) for r in conn.execute(f"PRAGMA table_info({table})")]
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        parts.append(f"\n{table}  ({n} rows)")
        parts.append("  " + ", ".join(f"{c} {t}" for c, t in cols))

    parts.append("\nVALUES YOU WILL NEED")
    for table, column in (("entity", "kind"), ("assertion", "predicate"),
                          ("paper", "category")):
        try:
            vals = [str(r[0]) for r in conn.execute(
                f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
                f" GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT 18")]
        except sqlite3.Error:
            continue
        parts.append(f"  {table}.{column}: " + ", ".join(vals))

    # The typed agent is handed closed vocabularies and a retrieval index that
    # matches across spellings. A SQL agent seeing only column names has to
    # GUESS that a b-tagged jet is filed as `detector_object` and written
    # "b-tagged jet (MV2c10, 77%)". Guessing our filing conventions is not the
    # skill under test, so a few real labels per kind are shown.
    parts.append("\nWHAT LABELS LOOK LIKE (entity.label, by kind)")
    kinds = [r[0] for r in conn.execute(
        "SELECT kind FROM entity GROUP BY kind ORDER BY COUNT(*) DESC LIMIT 10")]
    for kind in kinds:
        labels = [str(r[0])[:52] for r in conn.execute(
            "SELECT label FROM entity WHERE kind = ? AND label IS NOT NULL"
            " ORDER BY LENGTH(label) LIMIT 3", (kind,))]
        if labels:
            parts.append(f"  {kind}: " + " | ".join(labels))

    parts.append(
        "\nentity has NO paper_id column. Papers come from entity_occurrence,\n"
        "  which is (paper_id, entity_id, kind, label) -- one row per paper that\n"
        "  mentions the entity. entity.label is a rollup and can differ from the\n"
        "  per-paper label.")
    parts.append(
        "\nWHAT MAKES THIS HARD\n"
        "  Labels are written as the papers write them and are NOT unified.\n"
        "  'Pythia 8.230', 'PYTHIA8' and 'Pythia 8.2' are three rows and one\n"
        "  generator. `entity_canonical` and `same_as` exist and may help.\n"
        "  A paper is identified by paper.arxiv_id.\n"
        "  ONE CONCEPT LIVES UNDER SEVERAL KINDS. 'b-tagged jet' appears as a\n"
        "  detector_object, an event_region, a channel and a\n"
        "  systematic_uncertainty. Filtering on entity.kind before you have\n"
        "  checked which kinds actually hold your term is the fastest way to\n"
        "  get 0 rows from a graph that has the answer.")
    return "\n".join(parts)


WORKED_EXAMPLES = """
WORKED EXAMPLES (the join paths that actually work here)

Q: how many analyses used Pythia?
   search("Pythia")                       -> entity_ids for every spelling
   SELECT COUNT(DISTINCT eo.paper_id) FROM entity_occurrence eo
    WHERE eo.entity_id IN ('hepkg:generator:pythia8_v8p212', ...)
   -- or, when the word is distinctive enough that spelling does not vary:
   SELECT COUNT(DISTINCT paper_id) FROM entity_occurrence
    WHERE kind='generator' AND lower(label) LIKE '%pythia%'

Q: which analyses use b-tagged jets in their event selection?
   -- find out which KIND holds the term before filtering on it
   SELECT kind, COUNT(*) FROM entity WHERE lower(label) LIKE '%b-tag%' GROUP BY kind
   -- then
   SELECT DISTINCT eo.paper_id FROM entity_occurrence eo
    WHERE eo.kind='detector_object' AND lower(eo.label) LIKE '%b%tag%'

Q: which papers estimate a background with an ABCD method?
   SELECT DISTINCT a.paper_id FROM assertion a
     JOIN entity e ON e.entity_id = a.object_id
    WHERE a.predicate='background_uses_method' AND lower(e.label) LIKE '%abcd%'

SPELLINGS ARE NOT UNIFIED, and there are two ways to cope:
   `search` finds them by meaning -- prefer it when the concept has many names.
   `entity_canonical` maps an entity to its cluster, so joining through it
   groups spellings that were merged:
     SELECT ec.canonical_id, COUNT(DISTINCT eo.paper_id)
       FROM entity_occurrence eo JOIN entity_canonical ec USING (entity_id)
      GROUP BY ec.canonical_id
   It covers 663 of 5,114 entities, so it is a help and not a guarantee.
"""

#: WHAT `search` IS FOR. The shipped prompt says "search to find the entity_ids
#: a question is about, then sql to count or join over them" -- and the agent
#: does exactly that, which is the failure. Measured on 2026-08-30: it searched,
#: saw 20 hits, pasted 3 ids, and capped recall at 3 of 279 matching entities
#: (gf-05: 1 of 15 gold). On gf-08 it pasted two ids that do not exist and
#: scored 0 from 13 gold. The single question where it wrote a pattern instead
#: of an id list scored 0.89 against 0.43 on the same concept, and across the
#: eight questions pattern-style averaged 0.785 against 0.372 for id-paste.
#:
#: The graph holds 56 spellings of Pythia and 279 Higgs-ish entities, so a
#: search result is a SAMPLE OF THE VOCABULARY, never the answer set. This block
#: says so.
CONCEPT_BLOCK = """
WHAT A SEARCH RESULT IS
`search` shows you HOW THIS CORPUS SPELLS A CONCEPT. It is a sample, not the
answer, and it is truncated: the graph holds 56 distinct entities for Pythia and
over 200 that mention a Higgs candidate. The rows you are shown are examples.

So do NOT paste the ids you were shown into `IN (...)`. That silently caps your
answer at the handful you happened to see. Instead, read the spellings, then
write a query that matches the FAMILY -- `label LIKE`, a kind filter, a
subquery. Paste ids only when the search tells you it found them all.

Casting wide is cheap here and missing papers is not: a paper you name that
turns out irrelevant costs little, a paper you never reach cannot be recovered.
"""


SYSTEM_PROMPT = """You answer questions about a knowledge graph of 60 high-energy
physics papers. You have two tools and may use either, in any order, as often as
you like:

  search(text)  -- find entities by MEANING. Use it when you do not know how the
                   papers spell something.
  sql(query)    -- read anything, aggregate anything, join anything.

The usual shape is: `search` to find the entity_ids a question is about, then
`sql` to count or join over them. Neither tool alone is enough for most
questions.

{schema}
{examples}

HOW TO WORK
- Call `sql` to look. Look more than once.
- WHEN A QUERY RETURNS 0 ROWS, SUSPECT YOUR FILTERS BEFORE YOU CONCLUDE THE
  DATA IS ABSENT. Drop the narrowest condition and run it again. In particular,
  find out which kinds hold your term before filtering on kind:
      SELECT kind, COUNT(*) FROM entity WHERE label LIKE '%b-tag%' GROUP BY kind
  is one call and it turns a guess into a fact.
- Counting questions want a number. Questions asking which analyses or which
  papers want arXiv ids, and you must WRITE THE IDS OUT in your final answer.
- Questions asking WHICH ENTITY -- which generator, which region, which
  systematic -- want entity.entity_id, so SELECT it and name it. An answer that
  identifies the right thing without saying which row it is cannot be checked.
- When you have the answer, reply in prose with no further tool call.
- If the graph does not contain the answer, say so plainly. A wrong answer is
  worse than "not in the graph"."""


@dataclass
class SqlResult:
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str = ""
    truncated: bool = False

    def render(self) -> str:
        if self.error:
            return f"ERROR: {self.error}"
        if not self.rows:
            return "0 rows."
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join("" if v is None else str(v) for v in row)
                         for row in self.rows)
        # Truncation is REPORTED. A silent cap makes the model count a capped
        # set with total confidence -- the failure `count` exists to avoid.
        note = (f"\n({len(self.rows)} rows shown; MORE EXIST and were cut off at "
                f"{MAX_ROWS} -- use COUNT(*) if you need the total)"
                if self.truncated else f"\n({len(self.rows)} rows)")
        return f"{head}\n{body}{note}"


#: A join against a whole search set is not a query, it is the absence of one.
#: Measured: half the statements that referenced `search_N` carried no predicate
#: on it, returned a median of 25 rows against 5 for the filtered half, and cost
#: 0.102 of precision. Refused rather than discouraged, because the prompt
#: already discourages it and was ignored half the time.
_SET_REF = re.compile(r"\bsearch_(\d+)\b", re.I)
#: `entity_id` is EXCLUDED on purpose: `ON eo.entity_id = s.entity_id` is the
#: join key, present in every such query, and counting it as a filter made the
#: guard accept exactly the queries it exists to refuse.
_SET_FILTER = re.compile(r"\b(\w+)\.(label|kind)\s*(LIKE|=|IN|GLOB|<|>)", re.I)


def unfiltered_set_join(query: str) -> str:
    """The message to return instead of rows, or "" if the query is fine."""
    names = set(_SET_REF.findall(query or ""))
    if not names:
        return ""
    if _SET_FILTER.search(query or ""):
        return ""
    # INSPECTING the set is not answering with it. `SELECT COUNT(*) FROM
    # search_1` or `SELECT kind, COUNT(*) ... GROUP BY kind` is exactly the
    # move the guard is meant to encourage, so a query that touches no other
    # table is always allowed.
    # `[a-z_]+` stopped at the digit, so "search_1" read as "search_" and never
    # matched the set name -- the inspection queries the guard is meant to
    # ENCOURAGE were the ones it refused.
    others = set(re.findall(r"(?:FROM|JOIN)\s+([a-z_][a-z0-9_]*)", query or "", re.I))
    others = {t.lower() for t in others} - {f"search_{n}" for n in names}
    if not others:
        return ""
    n = sorted(names)[0]
    return (f"Not run. This joins ALL of search_{n} with no filter, which asks "
            f"for every entity the search touched -- and a search is always "
            f"wider than the question. Narrow it, e.g. "
            f"WHERE s.kind = '...' or WHERE s.label LIKE '%...%'. "
            f"If you believe every row qualifies, say so and use a COUNT first.")


def run_sql(conn, query: str, max_rows: int = MAX_ROWS,
            seconds: float = STATEMENT_SECONDS) -> SqlResult:
    """One statement, read-only, row-capped, time-capped."""
    text = (query or "").strip().rstrip(";")
    if not text:
        return SqlResult(error="empty query")
    if ";" in text:
        return SqlResult(error="one statement at a time, please")
    if not _READ_ONLY.match(text):
        return SqlResult(error="only SELECT (or WITH ... SELECT) is allowed")
    refusal = unfiltered_set_join(text)
    if refusal:
        return SqlResult(error=refusal)

    # A cartesian join over assertion x entity_occurrence is one plausible token
    # away, and this runs unattended for hours.
    deadline = time.monotonic() + seconds
    conn.set_progress_handler(
        lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        cur = conn.execute(text)
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in (cur.description or [])]
    except sqlite3.Error as exc:
        return SqlResult(error=str(exc))
    except Exception as exc:                      # noqa: BLE001 -- incl. the abort
        return SqlResult(error=f"query aborted: {type(exc).__name__}")
    finally:
        conn.set_progress_handler(None, 0)

    truncated = len(rows) > max_rows
    return SqlResult(rows=list(rows[:max_rows]), columns=columns, truncated=truncated)


#: An entity id, e.g. `hepkg:generator:pythia8-210`.
_ENTITY_ID = re.compile(r"hepkg:[a-z_]+:[A-Za-z0-9_.\-]+")


def entities_in(result: SqlResult) -> list[str]:
    """Entity ids anywhere in the result.

    The retrieval tier's truth IS an entity id, and `entity_retrieved` scores
    `Answer.entity_ids`. Collecting only papers scored all 16 of those questions
    0.000 for a reason that has nothing to do with SQL -- the same
    scorer-asymmetry the design doc warned about and this implementation then
    walked into. A control must not lose on a field the harness forgot to fill.
    """
    found: set[str] = set()
    for row in result.rows:
        for value in row:
            if isinstance(value, str):
                found.update(_ENTITY_ID.findall(value))
    return sorted(found)


def papers_in(result: SqlResult) -> list[str]:
    """arXiv ids anywhere in the result.

    The analogue of the planner's retrieval footprint -- what the run TOUCHED,
    as opposed to what it wrote down. Both are recorded for both systems, so
    `set_f1` can score the answer and `retrieval_reach` the footprint, and
    neither system is scored on a different quantity from the other.
    """
    found: set[str] = set()
    for row in result.rows:
        for value in row:
            if isinstance(value, str) and re.fullmatch(r"\d{4}\.\d{4,5}", value.strip()):
                found.add(value.strip())
    return sorted(found)


#: Offered once each, when a run is about to answer having found nothing. The
#: planner gets a widening ladder (query/widen.py) for exactly this failure, so
#: the control gets one too -- persistence on one side only would rig the next
#: comparison, and 191 of 207 runs stopped after a single query.
FREE_LADDER = (
    "You have not tried `search` yet. It finds entities by meaning, so it "
    "returns things no LIKE will: 'missing transverse momentum' finds MET and "
    "ETmiss. Try it before concluding the graph is empty.",
    "Your last query returned nothing. Drop its narrowest condition -- usually "
    "the kind filter or the most specific LIKE -- and run it again. "
    "`SELECT kind, COUNT(*) FROM entity WHERE label LIKE '%...%' GROUP BY kind` "
    "turns a guess about where a term lives into a fact.",
    "You have not looked through entity_canonical. Spellings that were merged "
    "into one cluster are only grouped by joining through it.",
)


class FreeSQLSystem:
    """A System, so it plugs into the same runner as everything else.

    Two tools now, not one. `search` is the SAME BM25+dense index the planner
    uses, and giving it away is deliberate: with SQL alone the control was
    losing partly because it could not match 'missing transverse momentum'
    against a label reading 'ETmiss', which is a fact about retrieval technology
    and not about typed graphs.

    That changes the claim under test, for the better. Not "typed tools beat raw
    SQL" -- easy to win and easy to dismiss as giving the baseline a worse search
    engine -- but **"typed structure beats flat access to the same raw
    materials"**. Both sides now have the corpus, the index and the model. What
    the control still does not have is the typed layer itself: predicate-aware
    hops, the critic, the empty-hop diagnostics, automatic canonical expansion,
    and the answer contract. That difference is the experiment.
    """

    def __init__(self, conn, index=None, *, name: str = "free-sql",
                 model: Optional[str] = None, chat=None,
                 max_rounds: int = MAX_ROUNDS, persist: bool = True,
                 reviewer: bool = False, state_objective: bool = False,
                 index_values: bool = False, index_quotes: bool = False,
                 search_sets: bool = False,
                 concept_prompt: bool = False, subgoals: bool = False,
                 subgoal_status: bool = False,
                 languages: tuple = ("sql",)) -> None:
        self.name = name
        self._conn = conn
        self._index = index
        # QUERY LANGUAGES. ("sql",) is the control as it always was;
        # ("cypher",) queries the Neo4j projection of the same graph;
        # ("sql", "cypher") offers both and records which one the model used.
        self._languages = tuple(languages)
        import os as _os
        self._plain_answer = _os.environ.get("FREESQL_PLAIN_ANSWER", "") == "1"
        self._constrained = _os.environ.get("CONSTRAINED_IDS", "") == "1"
        self._driver = None
        self._database = "neo4j"
        if "cypher" in self._languages:
            self._driver, self._database = neo4j_driver_from_env()
        self._chat = chat
        self._max_rounds = max_rounds
        self._persist = persist
        briefs = []
        if "sql" in self._languages:
            briefs.append(schema_brief(conn))
        if "cypher" in self._languages:
            briefs.append(cypher_brief(conn))
        self._schema = "\n\n".join(briefs)
        self._search_sets = bool(search_sets)
        # IMPLIED BY search_sets, because the shipped line ("find the entity_ids
        # ... then sql over them") flatly contradicts a queryable set table. The
        # flag stands alone too, so the prompt change is attributable without it.
        self._concept_prompt = bool(concept_prompt or search_sets)
        self._subgoals = bool(subgoals or subgoal_status)
        self._subgoal_status = bool(subgoal_status)
        self._set_n = 0
        self._reviewer = bool(reviewer)
        self._state_objective = bool(state_objective)
        self._review_stats = {}
        self.config = {
            "kind": "free-sql",
            "model": model or "",
            "max_rounds": max_rounds,
            "max_rows": MAX_ROWS,
            "statement_seconds": STATEMENT_SECONDS,
            "tools": list(self._languages) + (["search"] if index is not None else []),
            "languages": list(self._languages),
            "plain_answer": self._plain_answer,
            "constrained_ids": self._constrained,
            "neo4j": bool(self._driver),
            "persist": bool(persist),
            "reviewer": bool(reviewer),
            "state_objective": bool(state_objective),
            "index_values": bool(index_values),
            "index_quotes": bool(index_quotes),
            "search_sets": bool(search_sets),
            "concept_prompt": bool(concept_prompt or search_sets),
            "subgoals": bool(subgoals or subgoal_status),
            "subgoal_status": bool(subgoal_status),
            # The environment half, shared with the planner so the two cannot
            # record different things. Without it a free-SQL run named neither
            # its judge nor its encoder (D-089).
            **_env_config(),
        }

    def _client_pair(self):
        """(client, model) for the constrained call; the same endpoint the answerer uses."""
        from ..query.planner import _client
        made = _client()
        if isinstance(made, tuple):
            return made[0], (made[1] if len(made) > 1 else self.config.get("model", ""))
        import os as _os
        return made, _os.environ.get("LLM_MODEL_NAME", "")

    def _client_chat(self):
        if self._chat is not None:
            return self._chat
        from ..query import budget as budget_mod
        from ..query.planner import _client, completion_cap
        client, model = _client()
        self.config["model"] = model
        # THE SAME REASONING-AWARE CAP as the planner (D-084). This loop built
        # its own client and so had its own copy of the 800 default: qwen3-32b
        # scored 0.199 here purely on truncation, and 0.592 once the cap moved.
        cap = completion_cap(model)

        def chat(messages, tools):
            kwargs = dict(model=model, messages=messages, temperature=0.0,
                          max_tokens=cap)
            if tools:
                kwargs["tools"] = tools
            return client.chat.completions.create(**kwargs)

        # The SAME hard cap the planner gets. This system builds its own client,
        # so it would otherwise run uncapped against a metered endpoint -- the
        # control spending money the treatment cannot.
        self._chat = budget_mod.guard(chat, budget_mod.from_env(model))
        return self._chat

    def _review_fields(self) -> dict:
        """Reviewer counters, shaped for the Answer record."""
        st = self._review_stats
        return {
            "review_calls": st.get("calls", 0),
            "reviews_rejected": st.get("rejected", 0),
            "review_prompt_tokens": st.get("prompt_tokens", 0),
            "review_completion_tokens": st.get("completion_tokens", 0),
            "review_unparsed": st.get("unparsed", 0),
            "review_ceiling_hit": bool(st.get("ceiling_hit", False)),
        }

    def _review(self, question, choice, tool_calls, steps):
        """Judge the proposed SQL before it runs. None = reviewer unavailable."""
        from ..query import reviewer as R
        from ..query.planner import clean_content as _clean

        st = self._review_stats
        if st.get("cycles", 0) >= R.MAX_REVIEW_CYCLES:
            st["ceiling_hit"] = True
            return None
        plan = "\n".join(
            f"  {c.function.name}({c.function.arguments})" for c in tool_calls)
        history = "\n".join(f"  {d.get('tool')}({d.get('args')}) -> {d.get('rows')} rows"
                             for d in steps[-8:])
        verdict = R.review_plan(
            chat=self._chat, question=question, schema=self._schema,
            objective=_clean(choice.content), plan=plan, kind="sql",
            history=history)
        st["calls"] = st.get("calls", 0) + 1
        st["prompt_tokens"] = st.get("prompt_tokens", 0) + verdict.prompt_tokens
        st["completion_tokens"] = st.get("completion_tokens", 0) + verdict.completion_tokens
        if not verdict.parsed:
            st["unparsed"] = st.get("unparsed", 0) + 1
        if not verdict.approved:
            st["rejected"] = st.get("rejected", 0) + 1
            st["cycles"] = st.get("cycles", 0) + 1
        return verdict

    #: How many result sets stay queryable at once. The model can hold a few
    #: threads; twenty temp tables is a second schema to reason about, and the
    #: oldest are the least likely to be revisited.
    MAX_LIVE_SETS = 5
    #: The full set is materialised, not the displayed 20 -- but a search that
    #: matches half the graph is a bad search, not a useful set.
    SET_CAP = 400

    def _search(self, args: dict) -> tuple[str, list[str]]:
        """The planner's own retrieval, rendered for a model that will then
        write SQL against the ids.

        WITH `search_sets`, THE MODEL STOPS RETYPING IDS. Measured on the
        2026-08-30 run: it searched, saw 20 hits, and pasted 3 of them into an
        `IN (...)` list -- out of 279 entities that actually matched. Recall was
        capped there (gf-05: 1 of 15 gold). On gf-08 it typed two ids that do
        not exist (`hepkg:channel:mu mu`, note the space) and got a silent zero
        from 13 gold. The one question where it wrote a LIKE pattern instead of
        an id list scored 0.89 against 0.43 on the same concept.
        So the whole match set is materialised as a temp table it can FILTER in
        SQL, which is the thing this agent is supposed to be good at. Same 20
        rows are displayed, so the context cost is unchanged; what changes is
        what it can REFERENCE.
        """
        from ..query import retrieve
        text = args.get("text", "")
        kind = args.get("kind") or None
        shown = int(args.get("limit") or 20)
        want = max(shown, self.SET_CAP) if self._search_sets else shown
        try:
            hits = retrieve.search(self._index, text, conn=self._conn,
                                   kind=kind, limit=want)
        except Exception as exc:                      # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}", []
        if not hits:
            return "0 hits.", []
        head = hits[:shown]
        lines = ["entity_id | kind | label"]
        lines += [f"{h.entity_id} | {h.kind} | {h.label}" for h in head]
        body = "\n".join(lines)
        if not self._search_sets:
            return body + f"\n({len(head)} hits)", [h.entity_id for h in head]
        name = self._materialise(hits)
        # THE SPREAD, NOT JUST THE SIZE. Measured 2026-09-01: given the table,
        # the model joined it with no WHERE in 10 of 20 statements -- pulling
        # every one of ~400 entities and collapsing precision 0.704 -> 0.602.
        # It could not see that the set was heterogeneous. A search for "Higgs
        # candidate" returns `electron candidate` and `muon candidate` too,
        # because every paper has candidate objects.
        import collections as _c
        kinds = _c.Counter(h.kind or "?" for h in hits)
        spread = ", ".join(f"{k} {n}" for k, n in kinds.most_common(5))
        body += (f"\n({len(head)} shown of {len(hits)} matching -- ALL {len(hits)} "
                 f"saved as {name}(entity_id, kind, label).\n"
                 f" spread: {spread}\n"
                 f" This set is WIDER than your question. Filter it in SQL, do not "
                 f"paste ids and do not join it whole:\n"
                 f"   SELECT DISTINCT eo.paper_id FROM {name} s\n"
                 f"     JOIN entity_occurrence eo ON eo.entity_id = s.entity_id\n"
                 f"    WHERE s.kind = '...' AND s.label LIKE '%...%'\n"
                 f" Check the filter first: SELECT COUNT(*) FROM {name} WHERE ...)")
        return body, [h.entity_id for h in hits]

    def _materialise(self, hits) -> str:
        """The match set as a temp table. Read-only main db, writable temp."""
        self._set_n += 1
        name = f"search_{self._set_n}"
        self._conn.execute(f"DROP TABLE IF EXISTS temp.{name}")
        self._conn.execute(
            f"CREATE TEMP TABLE {name}(entity_id TEXT, kind TEXT, label TEXT)")
        self._conn.executemany(
            f"INSERT INTO {name} VALUES (?,?,?)",
            [(h.entity_id, h.kind or "", h.label or "") for h in hits])
        old = self._set_n - self.MAX_LIVE_SETS
        if old > 0:
            self._conn.execute(f"DROP TABLE IF EXISTS temp.search_{old}")
        return name

    def _drop_sets(self) -> None:
        """Between questions. Temp tables live on the CONNECTION, and the same
        connection answers every question in a run -- so without this, question
        12 could join against question 3's search set and score on it."""
        for i in range(1, self._set_n + 1):
            try:
                self._conn.execute(f"DROP TABLE IF EXISTS temp.search_{i}")
            except Exception:                          # noqa: BLE001
                pass
        self._set_n = 0

    def answer(self, q: Question) -> Answer:
        chat = self._client_chat()
        # PER QUESTION, not per system. The stats live on the instance because
        # the helper needs them, and the same instance answers every question in
        # a run -- so without this reset the review ceiling would be reached
        # once and then suppress reviewing for every later question, silently
        # turning the arm off partway through.
        self._review_stats = {}
        self._drop_sets()
        examples = WORKED_EXAMPLES
        if self._languages == ("cypher",):
            examples = CYPHER_EXAMPLES
        elif "cypher" in self._languages:
            examples = WORKED_EXAMPLES + CYPHER_EXAMPLES
        system = SYSTEM_PROMPT.format(schema=self._schema, examples=examples)
        if self._plain_answer:
            system = system.replace(
                "- Counting questions want a number. Questions asking which analyses or which\n"
                "  papers want arXiv ids, and you must WRITE THE IDS OUT in your final answer.\n",
                "- Counting questions want a number. Questions asking which analyses or which\n"
                "  papers want the papers.\n")
        if self._languages == ("cypher",):
            system = system.replace("sql(query)    -- read anything, aggregate anything, join anything.",
                                    "cypher(query) -- read anything, aggregate anything, traverse anything.")
            system = system.replace("`sql`", "`cypher`")
        elif "cypher" in self._languages:
            system = system.replace(
                "  sql(query)    -- read anything, aggregate anything, join anything.\n",
                "  sql(query)    -- read anything, aggregate anything, join anything (SQLite).\n"
                "  cypher(query) -- the SAME graph as a Neo4j property graph; use whichever "
                "language fits the question. Both are always available.\n")
        if self._concept_prompt:
            # Replace the instruction that teaches the failure, rather than
            # appending a contradiction of it.
            system = system.replace(
                "The usual shape is: `search` to find the entity_ids a question "
                "is about, then\n`sql` to count or join over them. Neither tool "
                "alone is enough for most\nquestions.",
                CONCEPT_BLOCK.strip())
        if self._state_objective:
            from ..query.planner import OBJECTIVE_BLOCK
            system += OBJECTIVE_BLOCK
        goals, status = [], ""
        if self._subgoals:
            from ..query import subgoals as _sg
            goals = _sg.decompose(chat, q.text)
            system += (_sg.render(goals) if self._subgoal_status
                       else _sg.goals_only(goals))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": q.text},
        ]
        started = time.time()
        touched: set[str] = set()
        where: dict = {}          # paper id -> the query that first returned it
        entities: set[str] = set()
        calls = prompt_tokens = completion_tokens = 0
        rounds = 0
        widenings = 0
        # EVERY query and what it returned. Without this, "free SQL lost" is
        # unfalsifiable: a fair defeat and a broken prompt look identical from
        # the score alone, and the first thing anyone will ask about this
        # control is whether it was a straw man. The transcripts are the answer,
        # so they have to exist before the run rather than after the argument.
        steps: list[dict] = []

        try:
            for rounds in range(1, self._max_rounds + 1):
                answer_tool = plain_answer_tool() if self._plain_answer else ANSWER_TOOL
                tools = ([SQL_TOOL] if "sql" in self._languages else []) \
                    + ([CYPHER_TOOL] if "cypher" in self._languages else []) \
                    + [answer_tool] + ([SEARCH_TOOL] if self._index is not None else [])
                # THE LAST ROUND IS FOR ANSWERING. The planner got this fix and
                # this loop did not, which is why deepseek-v4-flash scored 0.000
                # here while scoring 0.512 on the planner: 6 of 6 rounds on every
                # question, 51 papers retrieved by SQL, and no answer written.
                # The control cannot be a fair control if it is the only side
                # that can run out of turns holding the answer.
                if rounds >= self._max_rounds:
                    tools = [answer_tool]
                    messages.append({"role": "user", "content": (
                        f"Round {rounds} of {self._max_rounds} -- your last. No "
                        "more queries: answer now from what you already have"
                        + ("." if self._plain_answer else ", and put the arXiv ids in `papers`."))})
                reply = chat(messages, tools)
                calls += 1
                usage = getattr(reply, "usage", None)
                if usage:
                    prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0

                choice = reply.choices[0].message
                if self._subgoal_status and goals:
                    from ..query import subgoals as _sg
                    fresh = _sg.extract_status(choice.content or "")
                    if fresh and fresh != status:
                        status = fresh
                        # REPLACED, not appended: one live status block.
                        messages = [m for m in messages
                                    if not (m.get("role") == "system"
                                            and "SUB-OBJECTIVES" in (m.get("content") or ""))]
                        messages.append({"role": "system",
                                         "content": _sg.render(goals, status)})
                tool_calls = getattr(choice, "tool_calls", None)
                if not tool_calls:
                    # About to answer having found nothing, with rounds to
                    # spare. Same gate as the planner's: concrete, finite, once
                    # each, and it stops when the ladder does.
                    rounds_left = self._max_rounds - rounds
                    if (self._persist and not touched and not entities
                            and rounds_left >= 2 and widenings < len(FREE_LADDER)):
                        messages.append({"role": "user", "content": (
                            f"Before you answer -- you have {rounds_left} rounds "
                            f"left and have found nothing.\n\n"
                            f"{FREE_LADDER[widenings]}\n\n"
                            "Either try that, or answer and say what you looked "
                            "for and where.")})
                        widenings += 1
                        continue
                    text_out = choice.content or ""
                    extra = {}
                    if self._constrained and touched:
                        if _os.environ.get("CRITIC_SELECTS", "") == "1":
                            picked, rv = critic_selects_papers(self._conn, q.text, touched)
                            extra = {"constrained_candidates": len(touched), "constrained_ids": picked,
                                     "constrained_mode": "critic", "answer_review": rv}
                        else:
                            cl, mdl = self._client_pair()
                            picked, mode = constrained_papers(cl, mdl, q.text, text_out, touched, where)
                            extra = {"constrained_candidates": len(touched), "constrained_ids": picked, "constrained_mode": mode}
                        if picked:
                            text_out = text_out.rstrip() + "\n\nPapers: " + ", ".join(picked)
                    return Answer(
                        text=text_out, answered=bool(text_out),
                        papers=sorted(touched), steps=steps, entity_ids=sorted(entities), **extra,
                        llm_calls=calls, rounds=rounds,
                        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                        seconds=time.time() - started, **self._review_fields())

                # THE PLAN REVIEWER, before anything runs. Unlike the typed
                # side this also judges the SQL itself -- tables, joins, whether
                # a count counts the right thing -- because here a wrong query
                # is not a wrong hop, it is a wrong answer that looks right.
                # A rejection does NOT consume a round: `rounds` only advances
                # in the for-loop, so re-planning happens inside this iteration.
                if self._reviewer and any(c.function.name in ("sql", "cypher") for c in tool_calls):
                    verdict = self._review(q.text, choice, tool_calls, steps)
                    if verdict is not None and not verdict.approved:
                        messages.append({
                            "role": "assistant", "content": choice.content or "",
                            "tool_calls": [{"id": c.id, "type": "function",
                                            "function": {"name": c.function.name,
                                                         "arguments": c.function.arguments}}
                                           for c in tool_calls]})
                        for c in tool_calls:
                            messages.append({
                                "role": "tool", "tool_call_id": c.id,
                                "content": ("Not run. A reviewer checked this "
                                            "against your stated objective and "
                                            "the schema:\n\n" + verdict.feedback)})
                        continue

                messages.append({
                    "role": "assistant", "content": choice.content or "",
                    "tool_calls": [{"id": c.id, "type": "function",
                                    "function": {"name": c.function.name,
                                                 "arguments": c.function.arguments}}
                                   for c in tool_calls]})
                for call in tool_calls:
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except ValueError:
                        args = {}
                    if call.function.name == "answer":
                        asserted = [str(x).strip() for x in (args.get("papers") or [])]
                        asserted = [x for x in asserted if re.fullmatch(r"\d{4}\.\d{4,5}", x)]
                        text_out = str(args.get("text") or "")
                        extra = {}
                        if self._constrained and (touched or asserted):
                            if _os.environ.get("CRITIC_SELECTS", "") == "1":
                                picked, rv = critic_selects_papers(self._conn, q.text, touched, asserted)
                                extra = {"constrained_candidates": len(set(touched) | set(asserted)), "constrained_ids": picked,
                                         "constrained_mode": "critic", "answer_review": rv}
                            else:
                                cl, mdl = self._client_pair()
                                picked, mode = constrained_papers(cl, mdl, q.text, text_out, touched, where)
                                extra = {"constrained_candidates": len(touched), "constrained_ids": picked, "constrained_mode": mode}
                            if picked:
                                text_out = text_out.rstrip() + "\n\nPapers: " + ", ".join(picked)
                        return Answer(
                            text=text_out,
                            answered=bool(text_out.strip() or asserted), **extra,
                            # What it ASSERTED, not what its queries touched. The
                            # planner's `papers_from` cites a set; this is the
                            # same gesture, and without it the control would be
                            # forced to retype ids while the planner is not.
                            papers=sorted(asserted) if asserted else sorted(touched),
                            cited="answer.papers" if asserted else "",
                            steps=steps, entity_ids=sorted(entities),
                            llm_calls=calls, rounds=rounds,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            seconds=time.time() - started, **self._review_fields())
                    if call.function.name == "search":
                        body, hits = self._search(args)
                        entities.update(hits)
                        steps.append({"tool": "search", "round": rounds,
                                      "args": args, "rows": len(hits),
                                      "error": None, "preview": body[:200]})
                        messages.append({"role": "tool", "tool_call_id": call.id,
                                         "content": body[:6000]})
                        continue
                    query = args.get("query", "")
                    lang = "cypher" if call.function.name == "cypher" else "sql"
                    if lang not in self._languages:
                        result = SqlResult(error=f"{lang} is not available in this arm; use "
                                                 f"{' or '.join(self._languages)}")
                    elif lang == "cypher":
                        result = run_cypher(self._driver, self._database, query)
                    else:
                        result = run_sql(self._conn, query)
                    for _p in papers_in(result):
                        where.setdefault(_p, " ".join(query.split())[:140])
                    touched.update(papers_in(result))
                    entities.update(entities_in(result))
                    steps.append({
                        "tool": lang, "round": rounds, "args": {"query": query},
                        "rows": len(result.rows), "error": result.error or None,
                        "truncated": result.truncated,
                        "preview": result.render()[:200],
                    })
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": result.render()[:6000]})
        except Exception as exc:                  # noqa: BLE001
            return Answer(text="", answered=False, seconds=time.time() - started,
                          papers=sorted(touched), steps=steps, entity_ids=sorted(entities),
                          llm_calls=calls, rounds=rounds,
                          error=f"{type(exc).__name__}: {exc}")

        # Out of rounds with no prose. Not an error -- it looked and never
        # concluded, which is a real behaviour and should score as one.
        return Answer(text="", answered=False, papers=sorted(touched), steps=steps, entity_ids=sorted(entities),
                      llm_calls=calls, rounds=self._max_rounds,
                      prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                      seconds=time.time() - started, **self._review_fields())
