"""
Evaluation harness: running a question set against a system.

Two things this file exists to guarantee.

**Run identity.** Every record carries `git_sha` and `config_hash`. It is the
cheapest thing to get right and the most expensive to retrofit: without it, two
runs a week apart cannot be compared, and an ablation study over configurations
that were not recorded is a week of work that produces no attributable result.

**Repeats are first class.** `repeats` is an argument, not something bolted on
later, because run-to-run variance has to be known *before* any ablation is
interpreted (S-52). If identical runs swing by 5%, a 3% "improvement" is noise,
and a harness that reports single numbers invites chasing it.

Records are JSONL, one line per (question x repeat). Append-only and streamed,
so a run killed at question 180 of 200 leaves 180 usable records rather than
nothing.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .questions import Question, QuestionSet
from .systems import Answer, System, config_hash

# Bound on one question. Raised from 180 s on 2026-08-15: see `run`. The point
# of the bound is to stop a hung socket, not to cap honest work, so it wants to
# sit well above the slowest arm's tail rather than near its mean.
DEFAULT_TIMEOUT = float(os.environ.get("EVAL_QUESTION_TIMEOUT", 600))

RUN_DIR = Path("eval/runs")


def git_sha() -> str:
    """The commit the run was produced by. 'unknown' rather than an exception:
    a missing git checkout must not stop a run, only weaken its provenance."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        sha = out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
    if not sha:
        return "unknown"
    try:
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        dirty = ""
    # A dirty tree means the sha does not describe what actually ran. Say so in
    # the record rather than reporting a commit that is not the code.
    return f"{sha}-dirty" if dirty else sha


@dataclass
class RunMeta:
    run_id: str
    started_at: str
    git_sha: str
    system: str
    config: dict
    config_hash: str
    questions_path: str
    questions_hash: str
    n_questions: int
    repeats: int
    host: str = field(default_factory=platform.node)
    model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL_NAME", ""))
    finished_at: str = ""
    n_records: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Record:
    """One question, one repeat, one system."""
    run_id: str
    qid: str
    repeat: int
    system: str
    config_hash: str
    git_sha: str
    question: str
    shape: str
    split: str
    needs: list[str]
    difficulty: str
    answer: dict
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def new_run_id(system_name: str) -> str:
    """Unique per process, not merely per second.

    The timestamp alone collides: four evaluation jobs submitted together start
    within the same second, generate the SAME id, and then all open the same path
    with mode "w" -- each truncating the others. On 2026-08-04 that left one file
    where there should have been four, and the corruption is invisible because the
    file still parses.

    A Slurm job id when there is one, otherwise the pid. Both are unique among
    processes that could possibly be running at the same instant.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    unique = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    return f"{stamp}-{system_name}-{unique}"


def _answer_with_timeout(system: System, q, seconds: float):
    """One question, abandoned if it takes too long.

    Added after 2026-08-02, when a single question consumed an hour of a 53
    question run and produced nothing. The planner-side cause is fixed
    (`MAX_COMPLETION_TOKENS`), but the harness should not depend on that: a run
    is a long unattended job, and **one pathological question must cost a minute,
    not an evening.** The same reasoning as flushing every record -- durability
    is a property the runner owns.

    The abandoned thread is left running rather than killed, because Python
    cannot safely kill a thread blocked in a socket read. It is a daemon, so it
    cannot hold the process open, and vLLM will finish or drop the request on its
    own. A leaked thread for a few minutes is a much smaller problem than a run
    that never ends.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    from .systems import Answer

    started = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(system.answer, q)
        try:
            return future.result(timeout=seconds)
        except FutureTimeout:
            return Answer(
                text="", answered=False, seconds=time.time() - started,
                error=f"TimeoutError: no answer within {seconds:.0f}s "
                      "(abandoned; see runner._answer_with_timeout)",
            )
    finally:
        pool.shutdown(wait=False)


def run(questions: QuestionSet, system: System, *, repeats: int = 1,
        out_dir: str | Path = RUN_DIR,
        on_record: Optional[Callable[[Record], None]] = None,
        scorers: Optional[Iterable] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        abort_after_consecutive_errors: int = 20) -> Path:
    """Run every question `repeats` times and write JSONL. Returns the path.

    Scoring happens here, inline, rather than as a separate pass: the scorers
    are pure functions of (question, answer), so deferring them buys nothing and
    costs a re-read of every record.

    `timeout` bounds a single question. `None` disables it, which is right for a
    stub or an offline system and wrong for anything touching a model.

    **A timeout is not neutral across arms.** At 180 s it cut 33 of 436 Tier B
    questions out of the CRITIC arm on 2026-08-15 and 1 out of the control --
    because concept questions go through `search`, which the critic makes slow,
    while per-paper questions go through `contents_of`, which it does not touch.
    A wall that lands almost entirely on the arm under test, in the tier under
    test, biases the comparison it is supposed to protect. Set it from the
    SLOWEST arm, not the fastest.

    `abort_after_consecutive_errors` stops a run whose model endpoint has died.
    Without it an overnight job outlives its server and then spends `timeout`
    seconds on every remaining question -- 1,352 questions at 180 s is three days
    of a process hanging on a socket. Consecutive rather than total, because a
    handful of scattered failures is normal and a solid wall of them is not.
    """
    from . import scoring

    scorers = list(scorers) if scorers is not None else scoring.default_scorers()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = new_run_id(system.name)
    path = out_dir / f"{run_id}.jsonl"

    meta = RunMeta(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_sha=git_sha(),
        system=system.name,
        config=dict(system.config),
        config_hash=config_hash(system.config),
        questions_path=str(questions.path) if questions.path else "(in memory)",
        questions_hash=questions.content_hash,
        n_questions=len(questions),
        repeats=repeats,
    )

    n = 0
    consecutive_errors = 0
    aborted = False
    started = time.time()
    with path.open("w") as fh:
        fh.write(json.dumps({"_meta": meta.to_dict()}) + "\n")
        fh.flush()
        for repeat in range(repeats):
            for q in questions:
                answer = (_answer_with_timeout(system, q, timeout) if timeout
                          else system.answer(q))
                record = Record(
                    run_id=run_id, qid=q.qid, repeat=repeat, system=system.name,
                    config_hash=meta.config_hash, git_sha=meta.git_sha,
                    question=q.text, shape=q.shape, split=q.split,
                    needs=list(q.needs), difficulty=q.difficulty,
                    answer=asdict(answer),
                    scores=scoring.score_all(q, answer, scorers),
                )
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                fh.flush()  # a killed run keeps everything it had finished
                n += 1
                if on_record:
                    on_record(record)

                consecutive_errors = consecutive_errors + 1 if answer.error else 0
                if (abort_after_consecutive_errors
                        and consecutive_errors >= abort_after_consecutive_errors):
                    # Flag and break -- do NOT finalise here. `_rewrite_meta`
                    # reads the file and writes it back, and doing that while
                    # this handle is still open interleaves with its buffer:
                    # the run file came out with lines like `{"r{"run_id":...`
                    # and every reader then failed on a JSONDecodeError.
                    logging.getLogger(__name__).error(
                        "%d consecutive errors -- the model endpoint is probably gone. "
                        "Stopping with %d records kept.", consecutive_errors, n)
                    aborted = True
                    break
            if aborted:
                break

    meta.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta.n_records = n
    _rewrite_meta(path, meta)
    _write_summary(path, meta, time.time() - started)
    return path


def _rewrite_meta(path: Path, meta: RunMeta) -> None:
    """Patch the header line now that the run is finished. The header is written
    first (so a crashed run is still identifiable) and completed last."""
    lines = path.read_text().splitlines()
    lines[0] = json.dumps({"_meta": meta.to_dict()})
    path.write_text("\n".join(lines) + "\n")


def _write_summary(path: Path, meta: RunMeta, wall_seconds: float) -> None:
    """A small sidecar that is cheap to keep in git while the JSONL is not."""
    summary = meta.to_dict() | {"wall_seconds": round(wall_seconds, 1)}
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def load_records(path: str | Path) -> tuple[dict, list[Record]]:
    """Read a run file back. Returns (meta, records)."""
    path = Path(path)
    meta: dict = {}
    records: list[Record] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "_meta" in obj:
            meta = obj["_meta"]
            continue
        records.append(Record(**obj))
    return meta, records


def iter_runs(out_dir: str | Path = RUN_DIR) -> Iterator[Path]:
    yield from sorted(Path(out_dir).glob("*.jsonl"))


def rescore(run_path: str | Path, questions: QuestionSet,
            scorers: Optional[Iterable] = None) -> int:
    """Recompute every score from the stored answers, in place.

    Scoring runs inline during a run, which was the right call for speed and
    the wrong one for everything else: **scorers change constantly and model
    runs are expensive.** Fixing one metric definition should not cost 30
    requests to a 72B, and on 2026-08-02 it did -- twice, within an hour of the
    harness first working.

    This is cheap because the full `Answer` is stored, not just its score. The
    run record is the raw material; scores are a view over it.

    Returns the number of records rescored. Question ids that are no longer in
    the set are left untouched rather than dropped -- a run against an older
    question file stays readable.
    """
    from . import scoring

    scorers = list(scorers) if scorers is not None else scoring.default_scorers()
    by_qid = {q.qid: q for q in questions}

    run_path = Path(run_path)
    lines = [l for l in run_path.read_text().splitlines() if l.strip()]
    out: list[str] = []
    n = 0
    from .systems import Answer

    for line in lines:
        obj = json.loads(line)
        if "_meta" in obj:
            out.append(line)
            continue
        q = by_qid.get(obj["qid"])
        if q is not None:
            answer = Answer(**obj["answer"])
            obj["scores"] = scoring.score_all(q, answer, scorers)
            n += 1
        out.append(json.dumps(obj, ensure_ascii=False))

    run_path.write_text("\n".join(out) + "\n")
    return n
