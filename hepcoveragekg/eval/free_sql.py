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
from .systems import Answer

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
                 max_rounds: int = MAX_ROUNDS, persist: bool = True) -> None:
        self.name = name
        self._conn = conn
        self._index = index
        self._chat = chat
        self._max_rounds = max_rounds
        self._persist = persist
        self._schema = schema_brief(conn)
        self.config = {
            "kind": "free-sql",
            "model": model or "",
            "max_rounds": max_rounds,
            "max_rows": MAX_ROWS,
            "statement_seconds": STATEMENT_SECONDS,
            "tools": ["sql"] + (["search"] if index is not None else []),
            "persist": bool(persist),
        }

    def _client_chat(self):
        if self._chat is not None:
            return self._chat
        from ..query import budget as budget_mod
        from ..query.planner import MAX_COMPLETION_TOKENS, _client
        client, model = _client()
        self.config["model"] = model

        def chat(messages, tools):
            return client.chat.completions.create(
                model=model, messages=messages, tools=tools,
                temperature=0.0, max_tokens=MAX_COMPLETION_TOKENS)

        # The SAME hard cap the planner gets. This system builds its own client,
        # so it would otherwise run uncapped against a metered endpoint -- the
        # control spending money the treatment cannot.
        self._chat = budget_mod.guard(chat, budget_mod.from_env(model))
        return self._chat

    def _search(self, args: dict) -> tuple[str, list[str]]:
        """The planner's own retrieval, rendered for a model that will then
        write SQL against the ids."""
        from ..query import retrieve
        try:
            hits = retrieve.search(self._index, args.get("text", ""), conn=self._conn,
                                   kind=args.get("kind") or None,
                                   limit=int(args.get("limit") or 20))
        except Exception as exc:                      # noqa: BLE001
            return f"ERROR: {type(exc).__name__}: {exc}", []
        if not hits:
            return "0 hits.", []
        lines = ["entity_id | kind | label"]
        lines += [f"{h.entity_id} | {h.kind} | {h.label}" for h in hits]
        return "\n".join(lines) + f"\n({len(hits)} hits)", [h.entity_id for h in hits]

    def answer(self, q: Question) -> Answer:
        chat = self._client_chat()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(
                schema=self._schema, examples=WORKED_EXAMPLES)},
            {"role": "user", "content": q.text},
        ]
        started = time.time()
        touched: set[str] = set()
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
                tools = ([SQL_TOOL, ANSWER_TOOL]
                         + ([SEARCH_TOOL] if self._index is not None else []))
                # THE LAST ROUND IS FOR ANSWERING. The planner got this fix and
                # this loop did not, which is why deepseek-v4-flash scored 0.000
                # here while scoring 0.512 on the planner: 6 of 6 rounds on every
                # question, 51 papers retrieved by SQL, and no answer written.
                # The control cannot be a fair control if it is the only side
                # that can run out of turns holding the answer.
                if rounds >= self._max_rounds:
                    tools = [ANSWER_TOOL]
                    messages.append({"role": "user", "content": (
                        f"Round {rounds} of {self._max_rounds} -- your last. No "
                        "more queries: answer now from what you already have, "
                        "and put the arXiv ids in `papers`.")})
                reply = chat(messages, tools)
                calls += 1
                usage = getattr(reply, "usage", None)
                if usage:
                    prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0

                choice = reply.choices[0].message
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
                    return Answer(
                        text=choice.content or "", answered=bool(choice.content),
                        papers=sorted(touched), steps=steps, entity_ids=sorted(entities),
                        llm_calls=calls, rounds=rounds,
                        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                        seconds=time.time() - started)

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
                        return Answer(
                            text=str(args.get("text") or ""),
                            answered=bool(str(args.get("text") or "").strip() or asserted),
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
                            seconds=time.time() - started)
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
                    result = run_sql(self._conn, query)
                    touched.update(papers_in(result))
                    entities.update(entities_in(result))
                    steps.append({
                        "tool": "sql", "round": rounds, "args": {"query": query},
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
                      seconds=time.time() - started)
