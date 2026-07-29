"""
HEPCoverageKG aliases: trial set for evaluating an adjudicator.

Free evaluation pairs, built from signals already in the database. No hand
labelling: each source carries its own expected answer, so one run yields
recall AND precision instead of a vibe.

Three sources, deliberately kept separate because they are NOT equally reliable:

  tier1_positive  (expect SAME)
      Two entity_ids that Tiers 1/1.5 merged by deterministic string rules, but
      whose LABELS differ. Merging them was a decision we already trust, so a
      model that calls them different is losing real synonyms. Measures RECALL.

  same_id_positive  (expect SAME, with a caveat)
      One entity_id carrying different labels in different papers. Mostly true
      synonyms -- but the contract reports 752 shared ids with conflicting
      kind/label, so a disagreement here is genuinely ambiguous: either the model
      is wrong, or we have found an id collision. Treat misses as REVIEW ITEMS,
      not as errors. Harder than tier1_positive: the ids are identical and only
      the wording differs, so it is pure semantic judgement.

  negative  (expect DIFFERENT)
      Pairs from DIFFERENT Tier-1 clusters within one kind. Not guaranteed
      distinct -- Tier 1 only catches spelling variants, so two clusters can
      still be the same concept (that is the whole reason Tiers 2/3 exist).
      So a "same" verdict here is a candidate discovery, not automatically a
      false positive. Measures PRECISION, with that caveat stated.

Pairs whose labels are byte-identical are excluded everywhere: any model gets
those right and they would inflate the score.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

DEFAULT_NEGATIVES = 400
RANDOM_SEED = 20260729  # fixed: the trial set must be identical across runs


def _rows(conn, sql: str, params: tuple = ()) -> list:
    return conn.execute(sql, params).fetchall()


def tier1_positive(conn) -> list[dict]:
    """Different entity_ids merged by Tier 1/1.5, with differing labels."""
    sql = """
        SELECT DISTINCT ea.kind AS kind, ea.label AS label_a, eb.label AS label_b,
               s.entity_id_a AS id_a, s.entity_id_b AS id_b, s.method AS method
        FROM same_as s
        JOIN entity ea ON ea.entity_id = s.entity_id_a
        JOIN entity eb ON eb.entity_id = s.entity_id_b
        WHERE s.status IN ('confirmed','auto')
          AND ea.label IS NOT NULL AND eb.label IS NOT NULL
          AND ea.label <> eb.label
    """
    return [
        {
            "source": "tier1_positive",
            "expected": "same",
            "kind": r["kind"],
            "term_a": r["label_a"],
            "term_b": r["label_b"],
            "id_a": r["id_a"],
            "id_b": r["id_b"],
            "note": f"merged by tier '{r['method']}'",
        }
        for r in _rows(conn, sql)
    ]


def same_id_positive(conn) -> list[dict]:
    """One entity_id, different labels in different papers."""
    sql = """
        SELECT DISTINCT a.kind AS kind, a.entity_id AS eid,
               a.label AS label_a, b.label AS label_b,
               a.paper_id AS paper_a, b.paper_id AS paper_b
        FROM entity_occurrence a
        JOIN entity_occurrence b
          ON a.entity_id = b.entity_id AND a.label < b.label
        WHERE a.label IS NOT NULL AND b.label IS NOT NULL
    """
    return [
        {
            "source": "same_id_positive",
            "expected": "same",
            "kind": r["kind"],
            "term_a": r["label_a"],
            "term_b": r["label_b"],
            "id_a": r["eid"],
            "id_b": r["eid"],
            "note": f"same entity_id in {r['paper_a']} and {r['paper_b']}",
        }
        for r in _rows(conn, sql)
    ]


def negative(conn, n: int = DEFAULT_NEGATIVES, seed: int = RANDOM_SEED) -> list[dict]:
    """Same kind, different Tier-1 clusters. Expected different, not guaranteed."""
    sql = """
        SELECT e.entity_id AS eid, e.kind AS kind, e.label AS label,
               COALESCE(ec.canonical_id, e.entity_id) AS canon
        FROM entity e
        LEFT JOIN entity_canonical ec ON ec.entity_id = e.entity_id
        WHERE e.label IS NOT NULL AND TRIM(e.label) <> ''
    """
    by_kind: dict[str, list] = {}
    for r in _rows(conn, sql):
        by_kind.setdefault(r["kind"], []).append(r)

    rng = random.Random(seed)
    out: list[dict] = []
    kinds = [k for k, v in by_kind.items() if len(v) >= 2]
    # round-robin over kinds so one huge kind cannot dominate the sample
    attempts = 0
    while len(out) < n and attempts < n * 50 and kinds:
        kind = kinds[len(out) % len(kinds)]
        pool = by_kind[kind]
        a, b = rng.choice(pool), rng.choice(pool)
        attempts += 1
        if a["canon"] == b["canon"] or a["label"] == b["label"]:
            continue  # same cluster, or identical text -> not a negative
        out.append(
            {
                "source": "negative",
                "expected": "different",
                "kind": kind,
                "term_a": a["label"],
                "term_b": b["label"],
                "id_a": a["eid"],
                "id_b": b["eid"],
                "note": "different tier-1 clusters, same kind",
            }
        )
    return out


def build(conn, n_negative: int = DEFAULT_NEGATIVES) -> list[dict]:
    """All three sources, deduplicated on the unordered label pair."""
    pairs = tier1_positive(conn) + same_id_positive(conn) + negative(conn, n_negative)
    seen: set = set()
    unique: list[dict] = []
    for p in pairs:
        key = (p["kind"], *sorted((p["term_a"], p["term_b"])))
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def write(pairs: list[dict], out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return out_path


def load(path: Path | str) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
