"""
HEPCoverageKG query layer: the schema card.

A compact description of what the graph actually contains, written for a model
to read. Every downstream step needs it -- the router has to know which
predicates exist before it can pick a question shape, and template filling has
to know which kinds a predicate connects.

**Generated, never hand-written.** A hand-maintained description of a schema is
wrong the first time the schema changes, and then every generated query is wrong
at once, silently. Deriving it from the database means the card is either
correct or the database is empty.

Three things are reported, and each earns its place in the prompt:

  predicates, with their dominant subject and object kinds.  Predicate names are
      regular (`<subject>_<verb>_<object>`), but the typing is only *mostly*
      consistent -- measured against the pilot, `result_reports_quantity` has 6
      distinct object kinds and `background_uses_method` has 4 subject kinds. So
      the card reports the dominant kind **with the share it covers**, which is
      honest about the fuzziness instead of asserting a type that does not hold.

  counts.  A predicate with 1,335 rows and one with 4 support very different
      questions, and a model that cannot see the difference will confidently
      write a query returning nothing. Counts are the cheapest possible warning.

  entity kinds.  The vocabulary a question gets resolved against.

Deliberately excluded: the import/provenance tables (`bundle_import`,
`source_snapshot`, `activity`, ...). They are real, but no user question is about
them, and every extra table is another way for a generated query to go wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Tables the query layer is allowed to see. Provenance and import bookkeeping is
# deliberately absent -- see the module docstring.
QUERYABLE_TABLES = (
    "paper",
    "entity",
    "entity_occurrence",
    "assertion",
    "evidence",
    "assertion_evidence",
    "entity_canonical",
    "same_as",
)

# Show the actual share rather than a verdict whenever a predicate is not
# essentially pure. An earlier version hid everything above a 60% threshold
# behind silence, which was misleading in both directions: `result_measures_
# observable` is 86% observable and said nothing, while `result_reports_quantity`
# is 41/25/15/10 across four kinds and said only "mixed". The percentages cost
# ~300 tokens for the whole card -- cheap enough that embedding a judgement of
# ours in place of the number was never worth it.
PURE_SHARE = 0.99
MAX_KINDS_SHOWN = 4

# Predicates rarer than this are still listed, but grouped at the end: they are
# real, and a question may need them, but they should not crowd the prompt.
RARE_MAX_ROWS = 20


@dataclass
class Predicate:
    name: str
    family: str
    rows: int
    subject_kinds: list[tuple[str, int]]  # (kind, count), most common first
    object_kinds: list[tuple[str, int]]
    literal_share: float  # fraction whose object is a value, not a node

    @property
    def is_literal(self) -> bool:
        """Objects are mostly literal values (numbers, strings) rather than nodes.

        Worth flagging: a query joining to `entity` on such a predicate returns
        almost nothing, which is a silent failure rather than an error.
        """
        return self.literal_share >= 0.5

    @staticmethod
    def _side(kinds: list[tuple[str, int]]) -> str:
        """One side of a predicate: the kind alone when it is essentially pure,
        otherwise every kind with its share."""
        total = sum(n for _, n in kinds)
        if not total:
            return "unknown"
        if kinds[0][1] / total >= PURE_SHARE:
            return kinds[0][0]
        parts = [f"{k} {n / total:.0%}" for k, n in kinds[:MAX_KINDS_SHOWN]]
        return "/".join(parts)

    def render(self) -> str:
        obj = "<value>" if self.is_literal else self._side(self.object_kinds)
        line = f"  {self.name:<32} {self._side(self.subject_kinds):<26} -> {obj}"
        line = f"{line:<96} ({self.rows} rows)"
        if 0.0 < self.literal_share < 0.5:
            line += f"  [{self.literal_share:.0%} literal objects]"
        return line


@dataclass
class SchemaCard:
    papers: int = 0
    entities: int = 0
    assertions: int = 0
    evidence_share: float = 0.0
    kinds: list[tuple[str, int]] = field(default_factory=list)
    predicates: list[Predicate] = field(default_factory=list)
    facet_fields: list[tuple[str, list[str], int, int]] = field(default_factory=list)

    def render(self) -> str:
        """The prompt block. Ordered most-useful-first, because a model reading a
        long card weights the top of it more heavily."""
        common = [p for p in self.predicates if p.rows > RARE_MAX_ROWS]
        rare = [p for p in self.predicates if p.rows <= RARE_MAX_ROWS]

        out = [
            "GRAPH CONTENTS",
            f"  {self.papers} papers, {self.entities} entities, {self.assertions} assertions.",
            f"  {self.evidence_share:.0%} of assertions carry at least one verbatim evidence quote.",
            "",
            "ENTITY KINDS (what a question can be about)",
        ]
        out += [f"  {kind:<28} {n}" for kind, n in self.kinds]
        out += ["", "PREDICATES (subject kind -> object kind)"]
        out += [p.render() for p in common]
        if rare:
            out += ["", "  rare (<= 20 rows; usable, but check the result is not empty):"]
            out += [f"    {p.name} ({p.rows})" for p in rare]
        out += self._facet_block()
        return "\n".join(out)

    def _facet_block(self) -> list[str]:
        """Field NAMES and coverage. Deliberately no example values.

        An earlier version listed every value in every field (~590 tokens). Two
        problems killed it, and the second is the serious one.

        It anchors. Whichever values were chosen as "representative" would
        quietly become the working vocabulary, and that editorial choice would
        shape the distribution of tool calls with no measurement behind it.

        It leaks the answers. The values that read as most natural to list --
        BJet, MET, ABCD, Unfolding, SUSY -- are FIVE OF THE SEVEN distinct keys
        in the supervisor's Tier 1 gold answers. Any Tier 1 score would then be
        partly measuring that the answer key was pasted into the prompt.

        Field names are a different category: they are STRUCTURE, required to
        form a `facets` call at all and not discoverable from any result. Values
        are CONTENT, and content comes from the graph -- search hits carry the
        facet keys of whatever they matched, at rank 1 on 8 of 8 tested
        phrasings (D-054). The rest of this card follows the same rule: it lists
        entity kinds and predicate names, never example entity labels.

        Coverage stays. It is a number about reliability, not an answer, and
        without it a facet result reads as a complete index when a quarter of
        entities carry no tag at all.
        """
        if not self.facet_fields:
            return []
        out = [
            "",
            "FACET FIELDS (closed vocabularies, filterable with the `facets` tool)",
        ]
        for field_name, _values, matched, total in self.facet_fields:
            share = f"{matched / total:.0%} of entities tagged" if total else "empty"
            out.append(f"  {field_name:<22} {share}")
        out += [
            "  Keys are fixed strings, not free text. Search results carry the keys of",
            "  whatever they matched -- take keys from there rather than guessing.",
        ]
        return out


def _distribution(rows: list) -> list[tuple[str, int]]:
    """Kind counts for one side of a predicate, most common first."""
    return sorted(
        ((r["kind"] or "unknown", r["n"]) for r in rows),
        key=lambda kv: (-kv[1], kv[0]),
    )


def build(conn) -> SchemaCard:
    """Read the database and describe it."""
    card = SchemaCard()

    card.papers = conn.execute("SELECT COUNT(*) FROM paper").fetchone()[0]
    card.entities = conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0]
    card.assertions = conn.execute("SELECT COUNT(*) FROM assertion").fetchone()[0]
    with_ev = conn.execute(
        "SELECT COUNT(DISTINCT assertion_id) FROM assertion_evidence"
    ).fetchone()[0]
    card.evidence_share = (with_ev / card.assertions) if card.assertions else 0.0

    card.kinds = [
        (r["kind"], r["n"])
        for r in conn.execute(
            "SELECT kind, COUNT(*) n FROM entity WHERE kind IS NOT NULL"
            " GROUP BY kind ORDER BY n DESC"
        )
    ]

    # One pass per predicate rather than a single grouped query: the dominant
    # subject kind and the dominant object kind are independent questions, and
    # doing them together would report the most common *pair*, which is not the
    # same thing and is misleading when either side is mixed.
    for prow in conn.execute(
        "SELECT predicate, family, COUNT(*) n,"
        "       SUM(object_value IS NOT NULL) * 1.0 / COUNT(*) lit"
        " FROM assertion GROUP BY predicate, family ORDER BY n DESC"
    ).fetchall():
        subs = conn.execute(
            "SELECT e.kind AS kind, COUNT(*) AS n FROM assertion a"
            " JOIN entity e ON e.entity_id = a.subject_id"
            " WHERE a.predicate = ? GROUP BY e.kind",
            (prow["predicate"],),
        ).fetchall()
        objs = conn.execute(
            "SELECT e.kind AS kind, COUNT(*) AS n FROM assertion a"
            " JOIN entity e ON e.entity_id = a.object_id"
            " WHERE a.predicate = ? GROUP BY e.kind",
            (prow["predicate"],),
        ).fetchall()
        card.predicates.append(
            Predicate(
                name=prow["predicate"],
                family=prow["family"] or "",
                rows=prow["n"],
                subject_kinds=_distribution(subs),
                object_kinds=_distribution(objs),
                literal_share=prow["lit"] or 0.0,
            )
        )

    card.facet_fields = _facet_fields(conn)
    return card


def _facet_fields(conn) -> list[tuple[str, list[str], int, int]]:
    """(field, every value in use, entities tagged, entities of that kind).

    Values come from `entity_facet`, i.e. what the vocabulary actually MATCHED
    in this corpus, not the full enum. Listing keys that match nothing here
    would invite calls that correctly return zero papers, and a model reading an
    empty result cannot tell "no paper does this" from "the key was wrong".

    Silently absent when the layer has not been derived: the query layer must
    keep working on a database that predates the facets tables.
    """
    from hepcoveragekg.facets import CARD_FIELDS

    try:
        rows = conn.execute(
            "SELECT field, value FROM entity_facet GROUP BY field, value ORDER BY field, value"
        ).fetchall()
    except Exception:          # table absent -> no facet block, not a crash
        return []
    if not rows:
        return []

    values: dict[str, list[str]] = {}
    for r in rows:
        values.setdefault(r["field"], []).append(r["value"])

    out: list[tuple[str, list[str], int, int]] = []
    for field_name in sorted(values):
        kinds = sorted(k for k, f in CARD_FIELDS.items() if f == field_name)
        if not kinds:
            continue
        marks = ",".join("?" * len(kinds))
        row = conn.execute(
            "SELECT COUNT(*) AS total, SUM(tagged) AS matched FROM ("
            "  SELECT MAX(CASE WHEN f.entity_id IS NULL THEN 0 ELSE 1 END) AS tagged"
            "    FROM entity_occurrence eo"
            "    LEFT JOIN entity_facet f"
            "      ON f.paper_id = eo.paper_id AND f.entity_id = eo.entity_id"
            "     AND f.field = ?"
            f"  WHERE eo.kind IN ({marks})"
            "   GROUP BY eo.paper_id, eo.entity_id)",
            (field_name, *kinds),
        ).fetchone()
        out.append((field_name, values[field_name],
                    row["matched"] or 0, row["total"] or 0))
    return out


def render(conn) -> str:
    """Convenience: build and render in one call."""
    return build(conn).render()
