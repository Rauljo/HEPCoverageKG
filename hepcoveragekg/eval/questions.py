"""
Evaluation harness: the question set.

Questions are the artefact that must never drift (system.md S-10). Three
properties are load-bearing, and all three are enforced here rather than by
remembering:

  frozen and dated.  A question written after the system exists is a question
      the system can answer. Files are named by date and live in git; the
      loader records a content hash so a silently edited set is detectable.

  split.  Generated questions are DEV -- tune against them freely. Gabriel's
      are TEST, and the runner refuses to touch them without an explicit
      unlock that gets logged. Four rounds of tuning against one set measures
      how well the set was fitted, and discipline that depends on memory fails
      by week three.

  tagged.  `needs` says what a question requires of the graph. Without it a low
      score cannot be attributed -- was the query layer wrong, or was the data
      never there? It is also the M3 tag (S-57): `signatures` means the
      question is blocked upstream and no amount of query work fixes it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

QUESTION_DIR = Path("eval/questions")
UNLOCK_LOG = Path("eval/TEST_OPENED.log")

# The template shapes a question can take. Mirrors query/templates.py; a shape
# outside this list means the question set has drifted from the system.
SHAPES = {"count", "set", "describe", "compare", "crosstab", "quotes", "freeform"}

# What a question demands of the graph. The point is attribution of failure.
NEEDS = {
    "sql",              # a single template, no reasoning
    "hops",             # multi-step: one result feeds the next
    "signatures",       # BLOCKED on M3 -- assertion.signature is 0 populated
    "merged_entities",  # the answer changes if deduplication is off (S-11)
    "not_in_graph",     # correct behaviour is abstention (S-13)
}

SPLITS = {"dev", "test"}
SOURCES = {"generated", "gabriel", "handwritten"}
# Where a truth value came from, in descending order of authority.
TRUTH_SOURCES = {"sql", "reader", "gabriel", "none"}


class QuestionError(ValueError):
    """A malformed question. Raised at load time, never at run time -- a bad
    record should stop the run before compute is spent, not halfway through."""


@dataclass
class Truth:
    """The expected answer, when one exists.

    `kind` decides which scorer applies: a count is compared numerically, a set
    by F1 over paper ids, and `none` means only the label-free scorers run.
    """
    # count | set | subset | labels | value | none.
    # `labels` means: the answer must name these entities (one paper's
    # generators, systematics, regions...). See generate.paper_questions.
    # `subset` means: these papers MUST appear; extra papers are not counted
    # against the system (see generate.known_positive_questions).
    kind: str = "none"
    value: Any = None
    papers: list[str] = field(default_factory=list)
    # Entity ids the answer must cover, for `labels`-kind questions. Ids rather
    # than the label text, because matching prose against a label like
    # "Simultaneous binned maximum-likelihood fit to SR m_bb distributions..."
    # would measure wording. The readable labels live in provenance.
    items: list[str] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return self.kind != "none"


@dataclass
class Question:
    qid: str
    text: str
    source: str = "generated"
    split: str = "dev"
    shape: str = "count"
    needs: list[str] = field(default_factory=lambda: ["sql"])
    difficulty: str = "medium"
    truth: Truth = field(default_factory=Truth)
    truth_source: str = "none"
    provenance: dict = field(default_factory=dict)
    # Metamorphic grouping (S-34). `group` ties related questions together;
    # `relation` says how this one relates to the group's anchor.
    group: Optional[str] = None
    relation: Optional[str] = None

    @property
    def blocked_on_m3(self) -> bool:
        """S-57: needs final-state signatures, which are 0 populated."""
        return "signatures" in self.needs

    def to_dict(self) -> dict:
        return asdict(self)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuestionError(message)


def parse(record: dict) -> Question:
    """One JSON record -> a validated Question."""
    _require("qid" in record, "record has no qid")
    qid = record["qid"]
    _require(bool(record.get("text", "").strip()), f"{qid}: empty text")

    truth_raw = record.get("truth") or {}
    truth = Truth(
        kind=truth_raw.get("kind", "none"),
        value=truth_raw.get("value"),
        papers=list(truth_raw.get("papers") or []),
        items=list(truth_raw.get("items") or []),
    )

    q = Question(
        qid=qid,
        text=record["text"],
        source=record.get("source", "generated"),
        split=record.get("split", "dev"),
        shape=record.get("shape", "count"),
        needs=list(record.get("needs") or ["sql"]),
        difficulty=record.get("difficulty", "medium"),
        truth=truth,
        truth_source=record.get("truth_source", "none"),
        provenance=dict(record.get("provenance") or {}),
        group=record.get("group"),
        relation=record.get("relation"),
    )

    _require(q.split in SPLITS, f"{qid}: split {q.split!r} not in {sorted(SPLITS)}")
    _require(q.source in SOURCES, f"{qid}: source {q.source!r} not in {sorted(SOURCES)}")
    _require(q.shape in SHAPES, f"{qid}: shape {q.shape!r} not in {sorted(SHAPES)}")
    _require(q.truth_source in TRUTH_SOURCES,
             f"{qid}: truth_source {q.truth_source!r} not in {sorted(TRUTH_SOURCES)}")
    unknown = set(q.needs) - NEEDS
    _require(not unknown, f"{qid}: unknown needs {sorted(unknown)}")

    # A truth value with no stated origin is untraceable, and an origin with no
    # value is a bookkeeping error. Both are cheap to catch here and expensive
    # to notice in a results table.
    if q.truth.known:
        _require(q.truth_source != "none", f"{qid}: has a truth but truth_source is 'none'")
    else:
        _require(q.truth_source == "none",
                 f"{qid}: truth_source is {q.truth_source!r} but no truth value is given")

    return q


@dataclass
class QuestionSet:
    questions: list[Question]
    path: Optional[Path] = None
    content_hash: str = ""

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    def filter(self, split: Optional[str] = None, shape: Optional[str] = None,
               needs: Optional[str] = None, group: Optional[str] = None) -> "QuestionSet":
        out = [
            q for q in self.questions
            if (split is None or q.split == split)
            and (shape is None or q.shape == shape)
            and (needs is None or needs in q.needs)
            and (group is None or q.group == group)
        ]
        return QuestionSet(out, self.path, self.content_hash)

    def groups(self) -> dict[str, list[Question]]:
        """Metamorphic groups -- the unit `metamorphic.py` checks over."""
        out: dict[str, list[Question]] = {}
        for q in self.questions:
            if q.group:
                out.setdefault(q.group, []).append(q)
        return out

    def summary(self) -> dict:
        def tally(key) -> dict[str, int]:
            counts: dict[str, int] = {}
            for q in self.questions:
                counts[key(q)] = counts.get(key(q), 0) + 1
            return dict(sorted(counts.items()))

        needs_counts: dict[str, int] = {}
        for q in self.questions:
            for n in q.needs:
                needs_counts[n] = needs_counts.get(n, 0) + 1

        return {
            "total": len(self.questions),
            "split": tally(lambda q: q.split),
            "shape": tally(lambda q: q.shape),
            "source": tally(lambda q: q.source),
            "difficulty": tally(lambda q: q.difficulty),
            "needs": dict(sorted(needs_counts.items())),
            "with_truth": sum(1 for q in self.questions if q.truth.known),
            "blocked_on_m3": sum(1 for q in self.questions if q.blocked_on_m3),
            "groups": len(self.groups()),
        }


def load(path: str | Path, *, allow_test: bool = False,
         reason: str = "") -> QuestionSet:
    """Read a JSONL question file.

    `allow_test` is the lock (S-10). Test questions are the only realistic set
    we have; iterating against them would invalidate the headline number, so
    opening them is deliberate, and every opening is written to
    `eval/TEST_OPENED.log` with the date and a reason.
    """
    path = Path(path)
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    questions = [parse(r) for r in records]

    seen: set[str] = set()
    for q in questions:
        # Duplicate ids silently overwrite each other in every downstream join.
        _require(q.qid not in seen, f"duplicate qid {q.qid!r} in {path}")
        seen.add(q.qid)

    held = [q.qid for q in questions if q.split == "test"]
    if held and not allow_test:
        raise QuestionError(
            f"{path} contains {len(held)} test questions. Pass allow_test=True "
            "(CLI: --unlock-test) with a reason. See system.md S-10."
        )
    if held and allow_test:
        _record_unlock(path, len(held), reason)

    return QuestionSet(questions, path, hashlib.sha256(raw).hexdigest()[:12])


def _record_unlock(path: Path, count: int, reason: str) -> None:
    """Append-only. The value is that opening the test set leaves a trace no
    one has to remember to leave."""
    from datetime import date

    from . import runner  # local import: runner imports nothing from here

    UNLOCK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with UNLOCK_LOG.open("a") as fh:
        fh.write(f"{date.today().isoformat()}\t{runner.git_sha()}\t{path}\t"
                 f"{count} questions\t{reason or 'no reason given'}\n")


def save(questions: Iterable[Question], path: str | Path) -> Path:
    """Write a question file. Sorted by qid so regeneration produces a stable
    diff rather than a reshuffle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(questions, key=lambda q: q.qid)
    with path.open("w") as fh:
        for q in ordered:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
    return path
