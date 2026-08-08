"""
HEPCoverageKG query layer: the standing prompt blocks.

Three artefacts, kept apart because they are needed at different moments and
they have different lifetimes:

  PURPOSE      what this graph is and how to reason about it. Hand-written,
               revised by the author -- it encodes intent, which no amount of
               reading the database can recover.
  schema_card  what the graph *contains*. Generated (see schema_card.py).
  ddl()        the literal CREATE TABLE statements. Generated. Only the
               free-form path gets this: the constrained path fills tested
               templates and has no use for column names, and every table shown
               is another way for a generated query to go wrong (S-24).
"""
from __future__ import annotations

from hepcoveragekg.query.schema_card import QUERYABLE_TABLES

# ---------------------------------------------------------------------------
# DRAFT -- for the author to revise. This states intent, and intent is the one
# thing that cannot be derived from the data.
# ---------------------------------------------------------------------------
PURPOSE = """\
You are answering questions about a knowledge graph built from published
high-energy-physics papers (ATLAS and CMS analyses at the LHC).

WHAT THIS GRAPH IS FOR
It is a coverage map of the experimental literature. It exists to answer
questions a physicist has while reviewing what has been done or planning
something new: what has been measured, how often, by whom, in which final
states, with which methods -- and where the literature is thin.

HOW IT IS ORGANISED
  paper      one published article.
  result     ONE specific finding inside a paper. A paper has between 1 and 44
             of them: the headline analysis, plus individual numbers such as
             normalisation factors and per-region measurements. `result` is the
             hub -- most facts hang off it.
  entity     anything a result refers to: a generator, a systematic uncertainty,
             a physics process, a detector object, a region, a final state.
  assertion  one recorded fact, `subject -- predicate -> object`, backed by a
             verbatim quote from the paper.

HOW TO COUNT
"How many analyses..." almost always means DISTINCT PAPERS. Say which you are
reporting. Note that one fact can be recorded as several assertions when a paper
states it more than once, so the number of database rows is not the number of
findings.

THE SAME THING IS WRITTEN MANY WAYS
Every paper words things its own way -- the graph holds 56 distinct entities for
Pythia and 126 distinct phrasings of final states. A question about "Pythia" is
about all of them. Never answer from a single entity when the question is about
a concept.

EVIDENCE -- ANSWER ONLY FROM THE GRAPH, NEVER FROM MEMORY
You know a great deal of particle physics. None of it is admissible here. This
graph is a record of what specific papers said, and a question about it is a
question about that record -- not about the field.

So: before stating anything, retrieve it. Every name, number, count and claim in
your answer must come from a row you actually queried, and 97% of facts carry
the sentence they came from, so there is no excuse for an unsupported one.

If you find yourself writing something you know to be true of physics but did
not retrieve, stop and retrieve it. If it is not there, say it is not there.
A statement that is correct about the world but absent from this corpus is the
worst failure available to you: it looks right, it reads as authoritative, and
nothing downstream can catch it.

WHEN THE GRAPH CANNOT ANSWER
Say so plainly, and say what IS there. "The graph records 12 analyses using this
generator but does not record their tunes" is a good answer. Inventing the tunes
is not. A question the graph cannot answer is information about coverage, which
is the point of the project.

FACET TAGS
Some entities also carry facet tags: values from a small closed vocabulary,
assigned by matching patterns against the entity's own label. No model was
involved -- it is a lookup table. Because the values are fixed strings, they
support exact set operations ("papers with both X and Y") without depending on
how each paper happened to word things.

Two limits, both of which matter when reading a facet result:
  The vocabulary is CLOSED. Roughly a quarter of entities match no pattern and
  carry no tag. They keep their label and every fact -- they are simply absent
  from any facet answer, silently. A facet result is a floor, not a total.
  A tag names a FAMILY, not a specific method. Papers sharing one tag routinely
  did measurably different things -- a modified version, a two-dimensional
  version, one built on a different variable. The tag says where to look; the
  label says what is actually there, so read the labels before concluding that
  two papers did the same thing.

Search results carry the facet tags of whatever they matched, so that is where
the keys come from. A key that is not in the vocabulary is reported as unknown
rather than as zero papers, so an empty facet result means what it says.

NOTATION
l = lepton (electron or muon); v = neutrino, seen as missing energy;
g = photon; m = muon; t = tau; s = cross-section; a bar means antiparticle
(t tbar = top + antitop); -> means "decays to". MET / ETmiss = missing
transverse energy. fb^-1 = how much data was collected. 13 TeV = collision
energy.
"""


# ---------------------------------------------------------------------------
# The same thing, stripped to what the model CANNOT work out for itself.
#
# There is a real split inside PURPOSE. Some of it is *facts about this data* --
# a result is not a paper, there are 56 Pythia entities, rows are not findings.
# A model has no way to infer those and will be confidently wrong without them.
# The rest is *behavioural instruction*, and that is the part where over-
# specification makes a model rigid and formulaic instead of letting it reason.
#
# Which of the two is doing the work is an open question, not a matter of taste,
# so it is a variable rather than a decision: run both against the question set
# and let the numbers say. This is the lower bound.
# ---------------------------------------------------------------------------
PURPOSE_MINIMAL = """\
You are answering questions about a knowledge graph built from published
high-energy-physics papers (ATLAS and CMS analyses at the LHC). It is a coverage
map of that literature: what has been measured, how often, by whom, and where
the literature is thin.

Four things about this data that you cannot guess:

  A `result` is ONE finding inside a paper, not the paper itself. Papers hold
  between 1 and 44 of them.

  "How many analyses" means distinct PAPERS. One finding can occupy several
  database rows when a paper states it twice, so rows are not findings.

  The same thing is written many ways -- 56 distinct entities for Pythia, 126
  phrasings of final states. A question about a concept is about all of them,
  never about one node.

  97% of facts carry the verbatim sentence they came from.

Answer only from what you retrieve. If the graph does not have it, say so.
"""

# Section markers, so the harness can ablate individual blocks of PURPOSE rather
# than only choosing between full and minimal.
PURPOSE_SECTIONS = (
    "WHAT THIS GRAPH IS FOR",
    "HOW IT IS ORGANISED",
    "HOW TO COUNT",
    "THE SAME THING IS WRITTEN MANY WAYS",
    "EVIDENCE",
    "WHEN THE GRAPH CANNOT ANSWER",
    "FACET TAGS",
    "NOTATION",
)

# FACET TAGS is written to DESCRIBE the layer, never to recommend it. The line
# it stays behind: no sentence tells the model which route to take.
#
# That restraint is the whole measurement. The supervisor's scoring says an LLM
# call on a Tier 1 question is a soft fail *even when the answer is right* --
# so which rung the agent picks IS the thing being graded. Writing "prefer the
# cheapest route" into the prompt would not produce an agent that chooses well;
# it would hard-code the exam answer and delete the result.
#
# Whether tier-appropriate choice emerges or has to be instructed is a real
# question, so it is an ablation arm rather than a decision:
#   A  no FACET TAGS section
#   B  this section, descriptive          <- default
#   C  this section + an explicit "prefer the cheapest route" line
# B ~ C means the model works it out. C >> B is the more interesting finding.
PREFER_CHEAPEST_ROUTE = """\
When a question can be answered from facet tags alone, use them rather than
retrieving and reading labels: it is exact, and it costs nothing.
"""


def ddl(conn) -> str:
    """The CREATE TABLE statements for the tables the query layer may read.

    Generated from `sqlite_master` rather than copied from schema.sql, so it
    cannot describe a database that is not the one being queried -- the same
    reasoning as the schema card.

    Only the free-form path (S-24) should receive this. Handing column names to
    the constrained path invites it to reason about SQL it will never write.
    """
    rows = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
        f" AND name IN ({','.join('?' * len(QUERYABLE_TABLES))})"
        " ORDER BY name",
        QUERYABLE_TABLES,
    ).fetchall()
    return "\n\n".join(r["sql"] for r in rows if r["sql"])


# Written out for the free-form path because they are precisely what an
# unconstrained query gets wrong, and both fail silently -- see templates.py.
FREEFORM_WARNINGS = """\
TWO TRAPS, both of which return a plausible wrong answer rather than an error:

1. To get from a result to its paper, join `entity_occurrence` on
   (bundle_id, entity_id) and read `paper_id`. Do NOT use the
   `paper_reports_result` predicate: it links only the headline result of each
   paper, 60 of 272, so it silently loses 78% of results.

2. To count findings, count DISTINCT (subject, predicate, object) -- resolving
   subject and object through `entity_canonical` first. Do NOT use COUNT(*):
   one fact can occupy several rows when a paper states it twice, which
   over-reports by up to 2x.

The connection is read-only. Any statement that writes will be rejected.
"""


def freeform_context(conn) -> str:
    """Everything the unconstrained path needs: purpose, real DDL, and the traps."""
    return "\n\n".join([PURPOSE, "DATABASE SCHEMA", ddl(conn), FREEFORM_WARNINGS])
