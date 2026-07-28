# =============================================================================
# HEPCoverageKG: aliases layer tests (Tier 1)
#
# Pure logic (normalize, cluster) unit-tested; the store tier tested end-to-end
# on a small seeded DB. Verifies the safe behaviours: same-concept variants
# merge, versioned ids stay apart, cross-kind never merges, non-destructive,
# and nothing resolves until confirmed.
# =============================================================================
from __future__ import annotations

from hepcoveragekg.aliases import cluster, normalize, report, run, store
from hepcoveragekg.kg import store as kg_store


# --- Tier 1 normalization (pure) ---------------------------------------------


def test_normalize_folds_separators_and_case_keeps_digits():
    assert normalize.normalize_slug("hepkg:object:b_jet") == "bjet"
    assert normalize.normalize_slug("hepkg:object:b-jet") == "bjet"
    assert normalize.normalize_slug("hepkg:object:bjet") == "bjet"
    assert normalize.normalize_slug("hepkg:object:large-R-jet") == "largerjet"
    # digits preserved -> versions stay distinct
    assert normalize.normalize_slug("hepkg:generator:sherpa_2_2_1") == "sherpa221"
    assert normalize.normalize_slug("hepkg:generator:sherpa_2_2_2") == "sherpa222"
    # the '.' version dot is a separator too: 8.210 / 8-210 / 8210 all unify
    assert normalize.normalize_slug("hepkg:generator:pythia8.210") == "pythia8210"
    assert normalize.normalize_slug("hepkg:generator:pythia8-210") == "pythia8210"
    assert normalize.normalize_slug("hepkg:generator:pythia8210") == "pythia8210"
    # but distinct versions still stay apart
    assert normalize.normalize_slug("hepkg:generator:pythia8.212") != normalize.normalize_slug(
        "hepkg:generator:pythia8.230")


def test_propose_groups_variants_but_not_versions_or_kinds():
    ents = [
        ("hepkg:object:b_jet", "detector_object"),
        ("hepkg:object:b-jet", "detector_object"),
        ("hepkg:object:bjet", "detector_object"),
        ("hepkg:generator:sherpa_2_2_1", "generator"),
        ("hepkg:generator:sherpa_2_2_2", "generator"),
        ("hepkg:process:bjet", "physics_process"),  # same slug, different kind
    ]
    props = normalize.propose(ents)
    merged = {frozenset((p.entity_id_a, p.entity_id_b)) for p in props}
    # the three b-jets connect to one anchor; nothing else merges
    ids = {i for pair in merged for i in pair}
    assert ids == {"hepkg:object:b_jet", "hepkg:object:b-jet", "hepkg:object:bjet"}
    assert all(p.method == "normalize" and p.score == 1.0 for p in props)


# --- clustering (pure) -------------------------------------------------------


def test_connected_components_groups_transitively():
    comps = cluster.connected_components([("a", "b"), ("b", "c"), ("x", "y")])
    assert {frozenset(c) for c in comps} == {frozenset({"a", "b", "c"}), frozenset({"x", "y"})}
    assert cluster.connected_components([("a", "b")], nodes=["solo"]) == [{"a", "b"}]  # singleton omitted


def test_pick_canonical_most_papers_then_lexicographic():
    counts = {"b-jet": 10, "b_jet": 9, "bjet": 8}
    assert cluster.pick_canonical(["b_jet", "b-jet", "bjet"], counts) == "b-jet"
    # tie -> lexicographic
    assert cluster.pick_canonical(["m", "a"], {"m": 5, "a": 5}) == "a"
    # override wins
    assert cluster.pick_canonical(["b_jet", "b-jet"], counts, {"b_jet": "b_jet"}) == "b_jet"


# --- end to end on a seeded DB -----------------------------------------------


def _seed(conn, entities):
    """entities = [(entity_id, kind, [papers...])]."""
    conn.execute("INSERT INTO paper (arxiv_id, title, experiments) VALUES ('p','t','[]')")
    conn.execute(
        "INSERT INTO source_snapshot (source_hash, paper_id, title, source_path, experiments)"
        " VALUES ('sh','p','t','/x','[]')"
    )
    conn.execute(
        "INSERT INTO bundle_import (bundle_id, paper_arxiv_id, source_hash, bundle_content_hash,"
        " schema_version, imported_at) VALUES ('b','p','sh','h','v','t')"
    )
    for eid, kind, papers in entities:
        conn.execute("INSERT INTO entity (entity_id, kind, label) VALUES (?,?,?)", (eid, kind, eid))
        for paper in papers:
            conn.execute(
                "INSERT OR IGNORE INTO entity_occurrence (bundle_id, entity_id, paper_id, kind, label)"
                " VALUES (?,?,?,?,?)",
                (f"b{paper}", eid, paper, kind, eid),
            )


def _db_with(entities):
    conn = store.connect(":memory:")
    # seed needs the bundle FKs; entity_occurrence.bundle_id references bundle_import,
    # so create one bundle per paper id used
    papers = {p for _, _, ps in entities for p in ps}
    conn.execute("INSERT INTO paper (arxiv_id, title, experiments) VALUES ('p','t','[]')")
    conn.execute(
        "INSERT INTO source_snapshot (source_hash, paper_id, title, source_path, experiments)"
        " VALUES ('sh','p','t','/x','[]')"
    )
    for p in sorted(papers):
        conn.execute(
            "INSERT INTO bundle_import (bundle_id, paper_arxiv_id, source_hash, bundle_content_hash,"
            " schema_version, imported_at) VALUES (?, 'p','sh','h','v','t')",
            (f"b{p}",),
        )
    for eid, kind, ps in entities:
        conn.execute("INSERT INTO entity (entity_id, kind, label) VALUES (?,?,?)", (eid, kind, eid))
        for p in ps:
            conn.execute(
                "INSERT INTO entity_occurrence (bundle_id, entity_id, paper_id, kind, label)"
                " VALUES (?,?,?,?,?)",
                (f"b{p}", eid, p, kind, eid),
            )
    return conn


ENTITIES = [
    ("hepkg:object:b_jet", "detector_object", ["1", "2", "3"]),        # 3 papers
    ("hepkg:object:b-jet", "detector_object", ["1", "2", "3", "4"]),   # 4 papers -> canonical
    ("hepkg:object:bjet", "detector_object", ["5", "6"]),              # 2 papers
    ("hepkg:generator:sherpa_2_2_1", "generator", ["1"]),
    ("hepkg:generator:sherpa_2_2_2", "generator", ["2"]),
]


def test_build_writes_proposals_but_does_not_resolve():
    conn = _db_with(ENTITIES)
    run.build(conn)
    # b-jet trio proposed; sherpa versions not
    assert conn.execute("SELECT COUNT(*) FROM same_as WHERE status='proposed'").fetchone()[0] == 2
    # nothing resolves yet
    assert conn.execute("SELECT COUNT(*) FROM entity_canonical").fetchone()[0] == 0
    # entity table untouched
    assert conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0] == 5


def test_build_is_idempotent():
    conn = _db_with(ENTITIES)
    run.build(conn)
    n1 = conn.execute("SELECT COUNT(*) FROM same_as").fetchone()[0]
    run.build(conn)
    assert conn.execute("SELECT COUNT(*) FROM same_as").fetchone()[0] == n1


def test_report_shows_the_bjet_cluster_with_right_canonical():
    conn = _db_with(ENTITIES)
    run.build(conn)
    rows = report.cluster_rows(conn, statuses=("proposed",))
    assert len(rows) == 1
    r = rows[0]
    assert r["n_ids"] == 3
    assert r["canonical"] == "hepkg:object:b-jet"  # most papers (4)
    assert r["n_papers"] == 3 + 4 + 2


def test_confirm_then_materialize_resolves():
    conn = _db_with(ENTITIES)
    run.build(conn)
    store.confirm(conn)  # proposed -> auto
    mapped = store.materialize_canonical(conn)
    assert mapped == 3  # the three b-jets
    canon = {r["entity_id"]: r["canonical_id"] for r in conn.execute(
        "SELECT entity_id, canonical_id FROM entity_canonical")}
    assert set(canon.values()) == {"hepkg:object:b-jet"}


def test_aliases_schema_also_degrades_on_old_sqlite(monkeypatch):
    """The aliases DDL is executed separately from the import store's, so it
    needs the same STRICT degradation -- otherwise `aliases build` fails on the
    cluster even though the importer works."""
    import re as _re
    from hepcoveragekg.aliases.store import _SCHEMA_PATH

    keyword = _re.compile(r"\)\s*STRICT\s*;")
    assert keyword.search(_SCHEMA_PATH.read_text(encoding="utf-8"))

    monkeypatch.setattr(kg_store.sqlite3, "sqlite_version_info", (3, 36, 0))
    assert not keyword.search(kg_store.read_schema(_SCHEMA_PATH))

    # a full connect() must work end to end under the old version
    conn = store.connect(":memory:")
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"same_as", "entity_canonical", "entity", "assertion"} <= tables
