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
_READ_ONLY = re.compile(r"^\s*(select|with)\b", re.I)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|pragma|vacuum|replace)\b", re.I)

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

    parts.append(
        "\nWHAT MAKES THIS HARD\n"
        "  Labels are written as the papers write them and are NOT unified.\n"
        "  'Pythia 8.230', 'PYTHIA8' and 'Pythia 8.2' are three rows and one\n"
        "  generator. `entity_canonical` and `same_as` exist and may help.\n"
        "  A paper is identified by paper.arxiv_id.")
    return "\n".join(parts)


SYSTEM_PROMPT = """You answer questions about a knowledge graph of 60 high-energy
physics papers by writing SQL against it.

{schema}

HOW TO WORK
- Call `sql` to look. Look more than once: check what a column actually contains
  before you trust a WHERE clause, and widen it when a match returns nothing.
- Counting questions want a number. Questions asking which analyses or which
  papers want arXiv ids, and you must WRITE THE IDS OUT in your final answer.
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
    if _FORBIDDEN.search(text):
        return SqlResult(error="only read-only SELECT is allowed")

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


class FreeSQLSystem:
    """A System, so it plugs into the same runner as everything else."""

    def __init__(self, conn, *, name: str = "free-sql", model: Optional[str] = None,
                 chat=None, max_rounds: int = MAX_ROUNDS) -> None:
        self.name = name
        self._conn = conn
        self._chat = chat
        self._max_rounds = max_rounds
        self._schema = schema_brief(conn)
        self.config = {
            "kind": "free-sql",
            "model": model or "",
            "max_rounds": max_rounds,
            "max_rows": MAX_ROWS,
            "statement_seconds": STATEMENT_SECONDS,
            "tools": ["sql"],
        }

    def _client_chat(self):
        if self._chat is not None:
            return self._chat
        from ..query.planner import MAX_COMPLETION_TOKENS, _client
        client, model = _client()
        self.config["model"] = model

        def chat(messages, tools):
            return client.chat.completions.create(
                model=model, messages=messages, tools=tools,
                temperature=0.0, max_tokens=MAX_COMPLETION_TOKENS)
        self._chat = chat
        return chat

    def answer(self, q: Question) -> Answer:
        chat = self._client_chat()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(schema=self._schema)},
            {"role": "user", "content": q.text},
        ]
        started = time.time()
        touched: set[str] = set()
        calls = prompt_tokens = completion_tokens = 0
        rounds = 0
        # EVERY query and what it returned. Without this, "free SQL lost" is
        # unfalsifiable: a fair defeat and a broken prompt look identical from
        # the score alone, and the first thing anyone will ask about this
        # control is whether it was a straw man. The transcripts are the answer,
        # so they have to exist before the run rather than after the argument.
        steps: list[dict] = []

        try:
            for rounds in range(1, self._max_rounds + 1):
                reply = chat(messages, [SQL_TOOL])
                calls += 1
                usage = getattr(reply, "usage", None)
                if usage:
                    prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens += getattr(usage, "completion_tokens", 0) or 0

                choice = reply.choices[0].message
                tool_calls = getattr(choice, "tool_calls", None)
                if not tool_calls:
                    return Answer(
                        text=choice.content or "", answered=bool(choice.content),
                        papers=sorted(touched), steps=steps,
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
                    query = args.get("query", "")
                    result = run_sql(self._conn, query)
                    touched.update(papers_in(result))
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
                          papers=sorted(touched), steps=steps,
                          llm_calls=calls, rounds=rounds,
                          error=f"{type(exc).__name__}: {exc}")

        # Out of rounds with no prose. Not an error -- it looked and never
        # concluded, which is a real behaviour and should score as one.
        return Answer(text="", answered=False, papers=sorted(touched), steps=steps,
                      llm_calls=calls, rounds=self._max_rounds,
                      prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                      seconds=time.time() - started)
