"""
HEPCoverageKG aliases: graph context for adjudication.

Tier 3 previously judged two bare strings. That threw away most of what the
graph knows, and the strings alone are often not enough -- "SRA b tag" and
"SRC b tag" are indistinguishable as text but their selections differ.

Two things are added here, both read straight from what the papers said:

  every wording, not one.  `entity` keeps a single merged label (the mode across
      occurrences), so 312 entity_ids silently lose alternative wordings before
      the model sees them -- hepkg:background:diboson has six. We read
      `entity_occurrence` instead and show all of them.

  evidence quotes.  The literal sentence in which a paper defined or used the
      entity. This is the most discriminating signal available: two systematics
      can have near-identical names and completely different defining sentences.
      Available for 97% of entities (4,984 of 5,114).

Structural neighbours (what else the entity connects to) are deliberately NOT
included. Two different systematics in the SAME paper share nearly all their
neighbourhood, so that context would make near-duplicates look more alike --
the exact failure mode being fixed. Add it later as a measured ablation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

MAX_LABELS = 6
MAX_ALIASES = 8
QUOTE_CHARS = 320

# 79% of entities have <=3 distinct quotes and 97% have <=10, so a cap of 10
# sends everything for almost every entity. The cap exists only for the tail:
# hepkg:object:electron has 252 quotes (43,625 chars), and a pair of such
# entities would be ~22k tokens, well past an 8k context.
MAX_QUOTES = 10

# Below this a span is almost always a table cell rather than a statement.
MIN_QUOTE_CHARS = 40

# Which section a quote came from decides how much it is worth. A sentence in
# "Event selection" DEFINES the object; one in "Results" merely mentions it. For
# the frequent objects that get truncated, this ordering is what determines
# whether the surviving quotes are definitions or noise.
_SECTION_RANK = (
    (0, ("object reconstruction", "event reconstruction", "object selection",
         "event selection", "reconstruction and selection", "categorization")),
    (1, ("simulated", "simulation", "datasets", "data and", "samples")),
    (2, ("background estimation", "analysis strategy", "fiducial")),
    (3, ("systematic", "experimental uncertaint", "theoretical uncertaint")),
    (4, ("results", "interpretation", "stxs", "summary")),
    (5, ("introduction", "abstract", "conclusion")),
)
_UNRANKED = 4  # unknown sections sit mid-table, ahead of prose but behind definitions


def section_rank(section: str | None) -> int:
    """Lower is more definitional. Unknown sections land mid-table."""
    text = (section or "").strip().lower()
    if not text:
        return _UNRANKED
    for rank, keywords in _SECTION_RANK:
        if any(k in text for k in keywords):
            return rank
    return _UNRANKED


@dataclass
class EntityContext:
    entity_id: str
    kind: str
    labels: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    papers: int = 0
    quotes: list[tuple[str, str]] = field(default_factory=list)  # (section, quote)

    def render(self, name: str) -> str:
        """Compact prompt block. Omits empty sections rather than printing 'none',
        so the model is never shown a field it cannot use."""
        out = [f"{name}:", f'  primary name: "{self.labels[0] if self.labels else ""}"',
               f"  kind: {self.kind}"]
        if len(self.labels) > 1:
            other = "; ".join(f'"{lbl}"' for lbl in self.labels[1:])
            out.append(f"  also written as: {other}")
        if self.aliases:
            out.append("  aliases: " + "; ".join(f'"{a}"' for a in self.aliases))
        if self.attributes:
            out.append(f"  attributes: {json.dumps(self.attributes, ensure_ascii=False)}")
        out.append(f"  appears in {self.papers} paper(s)")
        if self.quotes:
            out.append("  quotes from the papers:")
            for section, q in self.quotes:
                prefix = f"[{section}] " if section else ""
                out.append(f'    - {prefix}"{q}"')
        return "\n".join(out)


def _truncate(text: str, limit: int = QUOTE_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def fetch(conn, entity_id: str) -> EntityContext:
    """Everything the graph knows about one entity, from the per-paper rows."""
    rows = conn.execute(
        "SELECT kind, label, aliases, attributes, paper_id"
        " FROM entity_occurrence WHERE entity_id = ?",
        (entity_id,),
    ).fetchall()

    if not rows:  # not imported (or a canonical with no occurrences)
        r = conn.execute(
            "SELECT kind, label FROM entity WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        return EntityContext(entity_id, r["kind"] if r else "", [r["label"]] if r else [])

    # Most frequent wording first: it is the one a reader would recognise.
    counts: dict[str, int] = {}
    aliases: list[str] = []
    attributes: dict = {}
    papers: set = set()
    for r in rows:
        if r["label"]:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        papers.add(r["paper_id"])
        for a in json.loads(r["aliases"] or "[]"):
            if a not in aliases:
                aliases.append(a)
        attributes.update(json.loads(r["attributes"] or "{}"))

    labels = [lbl for lbl, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    return EntityContext(
        entity_id=entity_id,
        kind=rows[0]["kind"],
        labels=labels[:MAX_LABELS],
        aliases=[a for a in aliases if a not in labels][:MAX_ALIASES],
        attributes=attributes,
        papers=len(papers),
        quotes=quotes(conn, entity_id),
    )


def quotes(conn, entity_id: str, limit: int = MAX_QUOTES) -> list[tuple[str, str]]:
    """Distinct (section, quote) pairs where this entity is subject or object.

    Sorted so the most definitional sections come first, because for the handful
    of very frequent objects the list gets truncated and we want the surviving
    quotes to be the ones that say what the thing IS.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT ev.quote AS quote, ev.section_title AS section
        FROM assertion a
        JOIN assertion_evidence ae ON ae.assertion_id = a.assertion_id
        JOIN evidence ev ON ev.evidence_id = ae.evidence_id
        WHERE a.subject_id = ? OR a.object_id = ?
        """,
        (entity_id, entity_id),
    ).fetchall()

    # Within a section, prefer LONGER quotes. Evidence spans include table cells
    # and fragments ("29/30/30", "Loose (90% efficiency)"), and sorting
    # alphabetically floated those to the top -- so the 10 quotes we kept for
    # `electron` were mostly numeric scraps rather than the sentences that define
    # it. Length is a crude but effective proxy for "is this a statement".
    ranked = sorted(
        rows,
        key=lambda r: (section_rank(r["section"]), -len(r["quote"] or ""),
                       (r["section"] or "").strip()),
    )

    seen: set = set()
    out: list[tuple[str, str]] = []
    fragments: list[tuple[str, str]] = []
    for r in ranked:
        q = _truncate(r["quote"])
        if not q or q in seen:
            continue
        seen.add(q)
        entry = ((r["section"] or "").strip(), q)
        # Hold very short spans back: they are usually table cells, and are only
        # worth sending if the entity has nothing better.
        (out if len(q) >= MIN_QUOTE_CHARS else fragments).append(entry)
        if len(out) >= limit:
            return out
    return (out + fragments)[:limit]
