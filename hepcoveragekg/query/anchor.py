"""Anchored candidates: separate exploring from nominating (D-197).

The harness has always defined retrieval as "every paper reachable from any
entity any call returned", and the constrained selection judges *that* set. So
there is no such thing as a free look. A `subjects_of` issued to *read* about
465 regions turns fifty papers into candidates; a `facets` on Electron, which
the question never mentioned, adds twenty more; and the judge is then asked to
sort 55 papers when the planner's own round-2 intersection had already found
the 18 that matter (8 of the 11 gold at precision 0.80, on gf-01). qwen3.8-flash
does exactly this: it computes A AND B correctly, then -- because an agentic
prior says thoroughness is more calls -- checks A, B, and every neighbour of
both, and hands the judge the corpus.

This module keeps a second, smaller set beside the footprint: the entities
returned by calls that were *finding* things on the question's own terms. Only
those become candidates. Reads never do. The footprint is still recorded, so
"papers retrieved" keeps its meaning across the chapter; what changes is what
the judge is shown.

THE ANCHOR IS THE PLANNER'S OWN FIRST INTERSECTION. No question parsing: the
first `facets` call fixes the constraint values, the first round's `search`
calls fix the terms. A later finding-call whose values are a strict subset of
the anchor is a *relaxation* (the AND split into ORs); one whose values fall
outside it is a *drift* (Electron). Both run, both are recorded, neither
nominates, and the tool message says so in one line. Widening stays possible
-- an id named in `answer()` is a candidate regardless -- but it has to be a
decision, not a side effect of looking around.

Deliberately arithmetic. The drift test is set comparison on the planner's own
arguments; it costs nothing and cannot be confounded with a second model's
opinion. Two blocks already *advise* the planner to stop and the trace shows
them ignored: advice does not compete with a trained prior, changing what
counts does.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

#: Calls that look for things. Everything else reads about things already held.
FINDING = frozenset({"facets", "facet_entities", "search", "papers_of"})

#: Finding calls whose rows ARE papers. These nominate the papers they return,
#: not the entities on those rows: on gf-01 the intersection `facets [BJet,
#: MET] mode=all` returned 18 papers, but its rows carry the BJet and MET
#: entities, which occur across 45 papers -- and a candidate list built as
#: papers-of-entities turned the AND the planner had computed back into an OR
#: at the judge's door (measured: candidates 45 of a 45 footprint, judged
#: precision 0.26). `search` returns entities and nominates at that level.
PAPER_LEVEL = frozenset({"facets", "facet_entities", "papers_of"})

#: Reads whose row counts have no ceiling and can return the corpus from one
#: broad input (465 and 775 rows on gf-01). Capped only when EXPANSION_CAP is
#: set, so the anchoring effect can be measured without it first.
EXPANSION_TOOLS = frozenset({"subjects_of", "contents_of"})

_STOP = frozenset("""a an the of in on at to for with and or by from as is are
which what that this these those use uses used using analyses analysis paper
papers search searches measurement measurements event selection their its rather
than both whose does do did any all""".split())

NOMINATE = "nominate"     # on the anchor: rows become candidates
ANCHOR = "anchor"         # this call SET the anchor (and nominates)
RELAX = "relax"           # strict subset of the anchor's values: AND became OR
DRIFT = "drift"           # values or terms outside the anchor
READ = "read"             # not a finding call; informs, never nominates
TIGHTEN = "tighten"       # a superset of the anchor: the planner re-stated the question with MORE conditions


def enabled() -> bool:
    return os.environ.get("ANCHOR", "") == "1"


def expansion_cap() -> int:
    """0 means off. Read fresh so a job script can set it after import."""
    try:
        return int(os.environ.get("EXPANSION_CAP", "0") or 0)
    except ValueError:
        return 0


def terms(text) -> set:
    """Content words of a search string, lower-cased, stopwords out."""
    words = re.findall(r"[a-z0-9][a-z0-9_^\-]*", str(text or "").lower())
    return {w for w in words if w not in _STOP and len(w) > 1}


def _values(args: dict) -> set:
    v = args.get("values")
    if v is None:
        return set()
    if not isinstance(v, list):
        v = [v]
    return {str(x).strip().lower() for x in v if str(x).strip()}


def classify(session, round_no: int, name: str, args: dict) -> str:
    """Which of the five things this call is, given what the run has anchored.

    Mutates the session's anchor when the call is the one that sets it. The
    anchor for `facets` is the first call's value set; for `search` it is the
    union of the terms of every search in the anchor's round, because a
    conjunctive question is typically resolved as one search per condition.
    """
    args = args or {}
    if name not in FINDING:
        return READ

    if name in ("facets", "facet_entities"):
        vals = _values(args)
        mode = str(args.get("mode") or "all").lower()
        if session.anchor_values is None:
            if not vals:
                return READ
            session.anchor_values = set(vals)
            session.anchor_mode = mode
            session.anchor_round = round_no
            return ANCHOR
        if not vals:
            return READ
        outside = vals - session.anchor_values
        if outside and vals > session.anchor_values:
            # TIGHTEN, NEVER LOOSEN (v7). The anchor is a snapshot of what the
            # planner knew at its first exact call, and on a question it had
            # to traverse the graph to understand, that snapshot is premature:
            # `facets [BJet]` then, after reading, `facets [BJet, MET]
            # mode=all`. Under v6 the second call -- the one that actually
            # states the conjunction -- was drift. A strict superset is the
            # planner re-stating the question with more conditions, so it
            # becomes the anchor and the pool is rebuilt from it alone.
            session.anchor_values = set(vals)
            session.anchor_mode = mode
            session.anchor_tightened = getattr(session, "anchor_tightened", 0) + 1
            session.nominated_papers = set()
            return TIGHTEN
        if outside:
            return DRIFT
        if vals < session.anchor_values:
            return RELAX
        # Same values: still a relaxation if the anchor demanded ALL of them
        # and this call asks for ANY.
        if session.anchor_mode == "all" and mode == "any" and len(vals) > 1:
            return RELAX
        return NOMINATE

    if name == "search":
        t = terms(args.get("text"))
        if not session.anchor_terms or round_no <= (session.anchor_round or round_no):
            # The anchor's round: every search in it is resolving a condition.
            if session.anchor_round is None:
                session.anchor_round = round_no
            session.anchor_terms |= t
            if t:
                groups = getattr(session, "anchor_groups_terms", None)
                if groups is None:
                    session.anchor_groups_terms = groups = []
                groups.append(set(t))
            return ANCHOR if t else READ
        if not (t & session.anchor_terms):
            return DRIFT
        # CONJUNCTIVE ANCHOR: a later search that covers only one of the
        # conditions the anchor round resolved separately is the AND split
        # into an OR, however on-topic its terms. "DeepJet ParticleNet" is
        # about b-tagging and says nothing about missing momentum.
        groups = getattr(session, "anchor_groups_terms", None) or []
        if _conjunctive(session) and len(groups) >= 2:
            covered = sum(1 for g in groups if g & t)
            if covered < len(groups):
                return RELAX
        return NOMINATE

    if name == "papers_of":
        ids = set(str(e) for e in (args.get("entity_ids") or []) if e)
        if not ids:
            return READ
        if _conjunctive(session):
            # The papers of some entities are one side of the AND. Only a call
            # that states every condition (facets mode=all, or a search that
            # covers each) can nominate under a conjunctive anchor.
            return RELAX
        # Papers of something the run found on-anchor are on-anchor.
        if ids & session.nominated_entity_ids:
            return NOMINATE
        return DRIFT

    return READ


def papers_in_rows(rows) -> set:
    """Paper ids on result rows, for the tools whose rows are papers."""
    out = set()
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        for k in ("paper_id", "arxiv_id"):
            v = r.get(k)
            if v:
                out.add(str(v))
    return out


def _conjunctive(session) -> bool:
    """Did the planner's own anchor state more than one condition?"""
    vals = getattr(session, "anchor_values", None) or set()
    groups = getattr(session, "anchor_groups_terms", None) or []
    return len(vals) >= 2 or len(groups) >= 2


def observe(session, round_no: int, name: str, args: dict, rows, note: str,
            collect) -> str:
    """Classify one executed call and fold its rows into the nominated set.

    `collect` is `planner._collect_ids`, passed in so this module does not
    import the planner. Returns the tag, which the caller stores on the Step
    and uses to decide whether the tool message gets a feedback line.
    """
    tag = classify(session, round_no, name, args)
    if tag in (ANCHOR, NOMINATE, TIGHTEN):
        # SEARCH RESOLVES VOCABULARY; IT DOES NOT DEFINE CANDIDATES (v5). A
        # semantic search returns its top-k whatever the match, so `search
        # kind=event_region "signal region with b jets and missing momentum"`
        # hands back 60 regions across the corpus however well its text
        # covers the conditions -- measured: pools of 55 under v4 from six
        # on-anchor calls, five of them searches. Only the structural lookups
        # (`facets`, `papers_of`, `facet_entities`) are selective, so only they
        # nominate papers. Search rows go to the entity tier, which the pool
        # uses only when no structural call was ever made.
        if name in PAPER_LEVEL:
            session.nominated_papers |= papers_in_rows(rows)
        else:
            session.nominated_entity_ids |= collect(rows, note)
        session.anchor_nominating += 1
    elif tag == DRIFT:
        session.anchor_drift += 1
    elif tag == RELAX:
        session.anchor_relax += 1
    return tag


def feedback(tag: str, session, name: str, args: dict, n_rows: int) -> str:
    """The one line appended to a tool message when a call did not nominate.

    Says what happened and how to override it, and nothing else. The planner
    is being told what counted, not asked to feel bad about it.
    """
    if tag == RELAX:
        anchor = ", ".join(sorted(session.anchor_values or ()))
        return ("[not nominated: this asks for fewer conditions than the question "
                "(anchor: %s). Its %d rows are visible but its papers are not "
                "candidates; only a lookup stating every condition adds candidates.]"
                % (anchor, n_rows))
    if tag == DRIFT:
        if name in ("facets", "facet_entities"):
            what = "values %s are outside the question's anchor {%s}" % (
                sorted(_values(args)), ", ".join(sorted(session.anchor_values or ())))
        elif name == "search":
            what = "search terms share nothing with the question's anchor terms"
        else:
            what = "these entities were not found on the question's terms"
        return ("[not nominated: %s. Its %d rows are visible but its papers are "
                "not candidates; only a lookup on the question's own terms adds "
                "candidates.]" % (what, n_rows))
    return ""


def candidate_pool(session, conn) -> tuple[set, bool]:
    """The paper candidate set and whether the run fell back to the footprint.

    Papers of the entities nominated by entity-returning calls, plus the
    papers nominated directly by paper-returning ones. Fallback when nothing
    was nominated at either level -- a run that never made a finding call on
    its own terms has no anchor to hold it to, and an empty list would score
    as silence for a mechanism that never engaged. Recorded so it counts.
    """
    from hepcoveragekg.query.sufficiency import papers_for
    ents = getattr(session, "nominated_entity_ids", set()) or set()
    papers = set(getattr(session, "nominated_papers", set()) or ())
    # PAPERS FIRST, ENTITIES ONLY AS A FALLBACK TIER. The round-1 searches that
    # resolve "b-tagged jets" and "missing transverse momentum" return the
    # BJet and MET entities across ~45 papers; they are vocabulary resolution,
    # not a claim about which papers answer. Once the planner has made a
    # paper-returning call on the anchor (the intersection), that is its
    # claim and the entity tier must not widen it back to the OR. Measured
    # on gf-01: pool 55 of a 60 footprint with the union, 18 + the on-anchor
    # paper searches without it.
    if papers:
        return papers, False
    if ents:
        return set(papers_for(conn, sorted(ents))), False
    return set(), True


def candidate_ids(session) -> tuple[list, bool]:
    """The entity ids whose papers may be candidates, and whether we fell back.

    Falls back to the full footprint when nothing was nominated at either
    level -- a run that never made a finding call on its own terms has no
    anchor to hold it to, and an empty candidate list would score as silence
    for a mechanism that never engaged. The fallback is recorded.
    """
    ents = sorted(getattr(session, "nominated_entity_ids", set()) or [])
    papers = getattr(session, "nominated_papers", set()) or set()
    if ents or papers:
        return ents, False
    return sorted(getattr(session, "known_entity_ids", set()) or []), True


def candidate_papers(session, by_entities: set) -> set:
    """The paper candidate set: papers of nominated entities, plus papers
    nominated directly by paper-returning calls. Empty only on fallback."""
    return set(by_entities) | set(getattr(session, "nominated_papers", set()) or ())


def cap_expansion(name: str, result) -> bool:
    """Truncate a corpus-sized read. Returns whether it did."""
    cap = expansion_cap()
    if not cap or name not in EXPANSION_TOOLS:
        return False
    rows = getattr(result, "rows", None)
    if not rows or len(rows) <= cap:
        return False
    total = len(rows)
    try:
        result.rows = rows[:cap]
        note = (getattr(result, "note", "") or "").strip()
        extra = ("showing %d of %d rows; add a predicate or narrow the set to "
                 "see the rest" % (cap, total))
        result.note = (note + " " + extra).strip() if note else extra
    except Exception as exc:  # noqa: BLE001 -- a cap must not kill a run
        logger.warning("expansion cap could not be applied to %s: %s", name, exc)
        return False
    return True


# --------------------------------------------------------------------------
# FREE-SQL. One tool, so the constraint set is read off the query text: the
# string literals are what the planner is filtering on. `LIKE '%b-tag%' AND
# ... LIKE '%missing%'` anchors two conditions; a later query carrying only
# '%missing%' is the AND split into an OR, and one carrying '%electron%' is
# drift. A query with no literals at all is a read (or a dump) and never
# nominates -- `SELECT * FROM paper` is exactly the call that sweeps a corpus.

def new_state():
    """A holder with the anchor fields, for a loop that has no Session."""
    from types import SimpleNamespace
    return SimpleNamespace(nominated_entity_ids=set(), nominated_papers=set(), known_entity_ids=set(), anchor_groups_terms=[], anchor_tightened=0,
                           anchor_values=None, anchor_mode="all", anchor_terms=set(),
                           anchor_round=None, anchor_nominating=0, anchor_drift=0,
                           anchor_relax=0, anchor_fallback=False, anchor_groups=0)


_ID_LITERAL = re.compile(r"^(hepkg:[a-z_]+:[A-Za-z0-9_.\-]+|\d{4}\.\d{4,5})$")
_STRUCTURAL = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)+$")


def sql_literals(query: str) -> list:
    """The content-term set of each string literal in a query, empties dropped.

    An id literal (`'hepkg:object:bjet'`, `'2004.14060'`) is returned as a
    one-element set holding the id itself, so `classify_sql` can recognise a
    query that fetches papers for the entities an earlier on-anchor query
    found -- the two-step shape free-SQL writes for almost every question.
    """
    out = []
    for lit in re.findall(r"'((?:[^']|'')*)'", str(query or "")):
        lit = lit.strip()
        if _ID_LITERAL.match(lit):
            out.append({lit})
            continue
        if _STRUCTURAL.match(lit):
            # 'result_uses_statistical_method', 'detector_object': a predicate
            # or a kind, i.e. the schema, not a condition from the question.
            continue
        t = terms(lit.replace("%", " ").replace("_", " "))
        if t:
            out.append(t)
    return out


def note_search(holder, text, hits) -> None:
    """A free-SQL `search` call: contributes terms and the ids it found, but
    does NOT open the anchor round -- the first SQL query does that. Otherwise
    the query that acts on the search's result is judged against the search
    and found wanting (its literals are a predicate and an id)."""
    holder.anchor_terms |= terms(text)
    holder.nominated_entity_ids |= {str(h) for h in (hits or []) if h}


def classify_sql(holder, round_no: int, query: str) -> str:
    """Same five tags as `classify`, for a SQL query."""
    groups = sql_literals(query)
    if not groups:
        return READ
    if holder.anchor_round is None:
        ids = [g for g in groups if len(g) == 1 and _ID_LITERAL.match(next(iter(g)))]
        words = [g for g in groups if g not in ids]
        known = getattr(holder, "nominated_entity_ids", set())
        if not words and not (ids and all(next(iter(g)) in known for g in ids)):
            return READ                      # ids from nowhere, before any anchor
        for g in words:
            holder.anchor_terms |= g
        holder.anchor_groups = max(len(words), 1)
        holder.anchor_round = round_no
        return ANCHOR
    known = getattr(holder, "nominated_entity_ids", set()) | getattr(holder, "nominated_papers", set())
    ids = [g for g in groups if len(g) == 1 and _ID_LITERAL.match(next(iter(g)))]
    words = [g for g in groups if g not in ids]
    if ids and not words:
        # Fetching by id: on-anchor iff the ids came from an on-anchor result.
        return NOMINATE if all(next(iter(g)) in known for g in ids) else DRIFT
    if round_no <= (holder.anchor_round or round_no):
        # Still resolving conditions in the anchor's round.
        for g in words:
            holder.anchor_terms |= g
        holder.anchor_groups = max(holder.anchor_groups, len(words))
        return ANCHOR
    if any(not (g & holder.anchor_terms) for g in words):
        if len(words) > holder.anchor_groups and any(g & holder.anchor_terms for g in words):
            # New condition added to the old ones: tighten.
            for g in words:
                holder.anchor_terms |= g
            holder.anchor_groups = len(words)
            holder.anchor_tightened = getattr(holder, "anchor_tightened", 0) + 1
            holder.nominated_papers = set()
            return TIGHTEN
        return DRIFT
    if len(words) < holder.anchor_groups:
        return RELAX
    return NOMINATE


def sql_feedback(tag: str, holder, n_rows: int) -> str:
    anchor = ", ".join(sorted(holder.anchor_terms))[:120]
    if tag == RELAX:
        return ("[not nominated: this filters on fewer conditions than the question "
                "(anchor terms: %s). Its %d rows are visible but its papers are not "
                "candidates; only a query stating every condition adds candidates.]"
                % (anchor, n_rows))
    if tag == DRIFT:
        return ("[not nominated: a filter term here is outside the question's anchor "
                "terms (%s). Its %d rows are visible but its papers are not candidates; "
                "only a query on the question's own terms adds candidates.]" % (anchor, n_rows))
    return ""
