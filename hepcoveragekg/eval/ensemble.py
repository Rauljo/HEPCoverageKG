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
    def __init__(self, typed, sql, mode: str | None = None):
        self._typed = typed
        self._sql = sql
        self.mode = (mode or os.environ.get("ENSEMBLE_MODE", "union")).lower()
        if self.mode not in ("union", "intersection"):
            raise ValueError(f"ENSEMBLE_MODE must be union or intersection, got {self.mode!r}")
        self.name = f"ensemble-{self.mode}"
        self.config = {"kind": "ensemble", "mode": self.mode,
                       "typed": getattr(typed, "config", {}), "sql": getattr(sql, "config", {})}

    def answer(self, q) -> Answer:
        started = time.time()
        a = self._typed.answer(q)
        b = self._sql.answer(q)
        na, nb = _named(a), _named(b)
        ids = (na | nb) if self.mode == "union" else (na & nb)
        both = na & nb
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
                      papers=sorted(ids), cited="ensemble" if ids else "",
                      steps=steps, entity_ids=sorted(set(a.entity_ids or []) | set(b.entity_ids or [])),
                      llm_calls=a.llm_calls + b.llm_calls, rounds=max(a.rounds, b.rounds),
                      prompt_tokens=a.prompt_tokens + b.prompt_tokens,
                      completion_tokens=a.completion_tokens + b.completion_tokens,
                      seconds=time.time() - started, error=err)
