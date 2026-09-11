"""The ensemble arm: the typed stack and free-SQL answer the same question, and
the answer is what they name together (D-146: on Gabriel's 9 free-SQL is the
more precise system and the typed stack the higher-recall one, tied on
judged_f1; the question is what their union or intersection is worth).

ENSEMBLE_MODE = union (default) | intersection. Both sides' answers are kept in
`steps` (tagged with the side) so either can be re-read from the record.
"""
from __future__ import annotations
import os, re, time
from .systems import Answer

_ARX = re.compile(r"\b\d{4}\.\d{4,5}\b")


def _named(a: Answer) -> set[str]:
    ids = set(_ARX.findall(a.text or ""))
    if a.cited:                      # a structured citation counts as naming
        ids |= {p for p in (a.papers or []) if _ARX.fullmatch(str(p))}
    return ids


class EnsembleSystem:
    def __init__(self, typed, sql, mode: str | None = None, conn=None):
        self._typed = typed
        self._sql = sql
        self._conn = conn
        self.mode = (mode or os.environ.get("ENSEMBLE_MODE", "union")).lower()
        if self.mode not in ("union", "intersection", "critic"):
            raise ValueError(
                f"ENSEMBLE_MODE must be union, intersection or critic, got {self.mode!r}")
        if self.mode == "critic" and conn is None:
            raise ValueError("ENSEMBLE_MODE=critic needs a database connection")
        self.name = f"ensemble-{self.mode}"
        self.config = {"kind": "ensemble", "mode": self.mode,
                       "typed": getattr(typed, "config", {}), "sql": getattr(sql, "config", {})}

    def _candidates(self, a: Answer, b: Answer, named: set) -> list:
        """Every paper either system could defensibly be asserting (D-167).

        The typed side's is its retrieval footprint -- the papers reachable
        from the entities it retrieved, which is the pool its own constrained
        step would have chosen from -- and free-SQL's is what its queries
        returned. Both plus whatever either one named, kept to ids the graph
        holds, so an invented id can never become a candidate (D-157).
        """
        pool = set(named) | {str(p) for p in (b.papers or [])}
        ents = sorted({str(e) for e in (a.entity_ids or []) if e})
        if ents:
            marks = ",".join("?" * len(ents))
            try:
                pool |= {str(r[0]) for r in self._conn.execute(
                    f"SELECT DISTINCT paper_id FROM entity_occurrence "
                    f"WHERE entity_id IN ({marks})", ents).fetchall()}
            except Exception:  # noqa: BLE001 -- a missing footprint is not fatal
                pass
        ids = sorted(p for p in pool if _ARX.fullmatch(str(p)))
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        try:
            held = {str(r[0]) for r in self._conn.execute(
                f"SELECT arxiv_id FROM paper WHERE arxiv_id IN ({marks})", ids).fetchall()}
        except Exception:  # noqa: BLE001
            return ids
        return [p for p in ids if p in held]

    def answer(self, q) -> Answer:
        started = time.time()
        a = self._typed.answer(q)
        b = self._sql.answer(q)
        na, nb = _named(a), _named(b)
        both = na & nb
        extra: dict = {}
        if self.mode == "critic":
            # ONE JUDGEMENT OVER BOTH POOLS (D-167). Neither system's own
            # selection decides anything here: the judge reads every candidate
            # either side could asserted, with that paper's evidence, and its
            # kept set is the answer. The two systems contribute reach.
            from hepcoveragekg.eval.free_sql import critic_selects_papers
            cands = self._candidates(a, b, na | nb)
            picked, review = critic_selects_papers(self._conn, q.text, set(cands))
            ids = set(picked)
            extra = {"constrained_candidates": len(cands),
                     "constrained_ids": list(picked),
                     "constrained_mode": "critic",
                     "answer_review": review}
            text = (f"Papers (judged from the typed and SQL candidates): "
                    f"{', '.join(picked) if picked else 'none'}. "
                    f"Typed named {len(na)}, SQL named {len(nb)}, both {len(both)}, "
                    f"candidates {len(cands)}.")
        else:
            ids = (na | nb) if self.mode == "union" else (na & nb)
            text = (f"Papers ({self.mode} of the typed and SQL answers): "
                    f"{', '.join(sorted(ids)) if ids else 'none'}. "
                    f"Typed named {len(na)}, SQL named {len(nb)}, both {len(both)}.")
        steps = ([dict(s, side="typed") for s in (a.steps or [])]
                 + [dict(s, side="sql") for s in (b.steps or [])]
                 + [{"tool": "ensemble", "side": "merge", "args": {"mode": self.mode},
                     "rows": len(ids), "error": None,
                     "preview": f"typed: {' '.join(sorted(na))} | sql: {' '.join(sorted(nb))}"[:400],
                     "typed_text": (a.text or "")[:1500], "sql_text": (b.text or "")[:1500]}])
        err = ""
        if a.error and b.error:
            err = f"typed: {a.error} | sql: {b.error}"
        return Answer(text=text, answered=bool(ids) or (a.answered or b.answered),
                      papers=sorted(ids), cited="ensemble" if ids else "", **extra,
                      steps=steps, entity_ids=sorted(set(a.entity_ids or []) | set(b.entity_ids or [])),
                      llm_calls=a.llm_calls + b.llm_calls, rounds=max(a.rounds, b.rounds),
                      prompt_tokens=a.prompt_tokens + b.prompt_tokens,
                      completion_tokens=a.completion_tokens + b.completion_tokens,
                      seconds=time.time() - started, error=err)
