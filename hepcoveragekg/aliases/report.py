# =============================================================================
# HEPCoverageKG aliases: the draft alias list (the human-review deliverable)
#
# Read-only. Renders the clusters proposed by the tiers into a markdown table
# (and a flat CSV) for eyeballing before anything is confirmed. The auto-picked
# canonical is shown but not binding.
# =============================================================================
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence, Union

from hepcoveragekg.aliases import cluster, store


def _member_info(conn) -> dict[str, tuple[str, str]]:
    """entity_id -> (kind, label)."""
    return {r["entity_id"]: (r["kind"], r["label"]) for r in conn.execute(
        "SELECT entity_id, kind, label FROM entity")}


def cluster_rows(conn, statuses: Sequence[str] = ("proposed", "auto", "confirmed")) -> list[dict]:
    """One dict per cluster, largest first, canonical marked."""
    counts = store.paper_counts(conn)
    overrides = store._overrides(conn)
    info = _member_info(conn)
    rows = []
    for members in store.clusters(conn, statuses):
        canonical = cluster.pick_canonical(members, counts, overrides)
        kind = info.get(canonical, ("?", ""))[0]
        members_sorted = sorted(members, key=lambda e: (-counts.get(e, 0), e))
        rows.append({
            "kind": kind,
            "canonical": canonical,
            "n_ids": len(members),
            "n_papers": sum(counts.get(m, 0) for m in members),
            "members": [
                {"entity_id": m, "papers": counts.get(m, 0), "label": info.get(m, ("", ""))[1],
                 "is_canonical": m == canonical}
                for m in members_sorted
            ],
        })
    rows.sort(key=lambda r: (-r["n_ids"], -r["n_papers"], r["canonical"]))
    return rows


def to_markdown(rows: list[dict]) -> str:
    total_ids = sum(r["n_ids"] for r in rows)
    lines = [
        "# Aliases — draft list (for review)",
        "",
        f"**{len(rows)} clusters** collapsing **{total_ids} entity ids** into {len(rows)} concepts. "
        "The `→` row is the auto-picked canonical (most papers); override or reject as needed. "
        "Nothing is applied to the graph until confirmed.",
        "",
    ]
    for r in rows:
        lines.append(f"### `{r['canonical']}`  ·  {r['kind']}  ·  {r['n_ids']} ids / {r['n_papers']} papers")
        lines.append("")
        lines.append("| | entity_id | papers | label |")
        lines.append("|---|---|---|---|")
        for m in r["members"]:
            mark = "→" if m["is_canonical"] else ""
            label = (m["label"] or "").replace("|", "\\|")[:70]
            lines.append(f"| {mark} | `{m['entity_id']}` | {m['papers']} | {label} |")
        lines.append("")
    return "\n".join(lines)


def write_report(conn, out_dir: Union[str, Path], statuses=("proposed", "auto", "confirmed")) -> dict:
    """Write draft-aliases.md and draft-aliases.csv into out_dir. Returns paths + counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = cluster_rows(conn, statuses)

    md_path = out_dir / "draft-aliases.md"
    md_path.write_text(to_markdown(rows), encoding="utf-8")

    csv_path = out_dir / "draft-aliases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_canonical", "kind", "entity_id", "papers", "is_canonical", "label"])
        for r in rows:
            for m in r["members"]:
                w.writerow([r["canonical"], r["kind"], m["entity_id"], m["papers"],
                            int(m["is_canonical"]), m["label"]])
    return {"clusters": len(rows), "ids": sum(r["n_ids"] for r in rows),
            "markdown": md_path, "csv": csv_path}
