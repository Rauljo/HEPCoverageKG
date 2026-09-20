"""Provenance per paper: the sentences that put a paper on the list.

The same material the answer critic reads when it judges a paper (D-165,
D-166): every quote the graph holds for the paper, ranked by the terms of the
question and the aliases of the entities involved, exactly as
`answer_critic._render` ranks them for the judge. Read from the graph, not
from the run's trace, so it works for either agent -- the free-SQL agent
records no evidence ids and still gets its quotes here.

No model is called. Provenance is a lookup, and a lookup is what makes it
verifiable: the section title and the verbatim sentence are enough for a
physicist to open the paper at the right place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..query import answer_critic as AC


@dataclass
class Quote:
    text: str
    section: str
    score: float


@dataclass
class PaperEvidence:
    paper_id: str
    title: str
    labels: list[str] = field(default_factory=list)     # the paper's entities that match the question
    quotes: list[Quote] = field(default_factory=list)    # best first
    verdict: str = ""                                     # the judge's reason, when it judged this paper


def _title(conn, paper_id: str) -> str:
    row = conn.execute("SELECT title FROM paper WHERE arxiv_id = ?", (paper_id,)).fetchone()
    return (row[0] if row else "") or ""


def _matching_labels(conn, paper_id: str, terms: set) -> tuple[list[str], set]:
    """The paper's own entity labels that share a word with the question, with
    their aliases -- the vocabulary the ranking needs (as in
    `free_sql.critic_selects_papers`)."""
    labels: list[str] = []
    aliases: set = set()
    rows = conn.execute("SELECT label, aliases FROM entity_occurrence WHERE paper_id = ?",
                        (paper_id,)).fetchall()
    for label, raw in rows:
        if not label or not AC._overlap(str(label), terms):
            continue
        if label not in labels:
            labels.append(str(label))
        try:
            for a in (json.loads(raw) if isinstance(raw, str) and raw else (raw or [])):
                if a:
                    aliases.add(str(a))
        except ValueError:
            pass
    return labels, aliases


def _quotes(conn, paper_id: str) -> list[tuple[str, str]]:
    rows = conn.execute(
        """SELECT DISTINCT ev.quote, ev.section_title FROM assertion a
           JOIN assertion_evidence ae ON ae.assertion_id = a.assertion_id
           JOIN evidence ev ON ev.evidence_id = ae.evidence_id
           WHERE a.paper_id = ? AND ev.quote IS NOT NULL AND LENGTH(ev.quote) > 20""",
        (paper_id,)).fetchall()
    seen: set = set()
    out = []
    for quote, section in rows:
        q = " ".join(str(quote).split())
        if q in seen:
            continue
        seen.add(q)
        out.append((q, section or ""))
    return out


def rank_quotes(question: str, quotes: list[tuple[str, str]], labels: list[str],
                aliases: set, n: int = 5) -> list[Quote]:
    """The judge's ranking (`answer_critic._render`), returning the quotes
    with their sections instead of a prompt block."""
    qterms = AC._question_terms(question)
    cterms = set(qterms)
    for lab in labels:
        cterms |= AC._question_terms(lab)
    cterms |= {a.lower() for a in aliases if a}
    texts = [q for q, _ in quotes]
    lex = {q: AC._overlap(q, qterms) * 2 + AC._overlap(q, cterms) + (1 if AC._REQ.search(q) else 0)
           for q in texts}
    dense = AC._dense_scores(question, texts) if texts else None
    if dense is not None:
        top = max(lex.values()) or 1
        score = {q: d + 0.1 * lex[q] / top for q, d in zip(texts, dense)}
    else:
        score = {q: float(lex[q]) for q in texts}
    ranked = sorted(quotes, key=lambda qs: -score[qs[0]])
    return [Quote(text=q, section=s, score=round(score[q], 3)) for q, s in ranked[:n]]


def provenance(conn, question: str, papers: list[str], review: dict | None = None,
               n_quotes: int = 5) -> list[PaperEvidence]:
    """For each paper the answer names, the sentences that support it.

    `review` is the answer critic's record when it ran; its per-paper reasons
    are attached so the user sees why the judge kept the paper. The quotes are
    recomputed either way -- the judge does not store them, and the lookup is
    deterministic and free.
    """
    terms = AC._question_terms(question)
    reasons: dict[str, str] = {}
    for v in (review or {}).get("kept_papers", []) or []:
        if isinstance(v, dict):
            reasons[str(v.get("id", ""))] = str(v.get("why", "") or "")
    out: list[PaperEvidence] = []
    for pid in papers:
        pid = str(pid)
        labels, aliases = _matching_labels(conn, pid, terms)
        quotes = _quotes(conn, pid)
        out.append(PaperEvidence(
            paper_id=pid, title=_title(conn, pid), labels=labels[:8],
            quotes=rank_quotes(question, quotes, labels, aliases, n=n_quotes),
            verdict=reasons.get(pid, "")))
    return out
