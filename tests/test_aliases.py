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


# --- Tier 1 label rules (pure) -----------------------------------------------


def test_normalize_label_folds_case_and_separators_but_keeps_the_decimal_point():
    assert normalize.normalize_label("B-Tagged Jet") == "btaggedjet"
    assert normalize.normalize_label("b tagged jet") == "btaggedjet"
    assert normalize.normalize_label("  b_tagged_jet  ") == "btaggedjet"
    # the dot survives: labels carry cut values, and folding it would merge
    # two different thresholds into one.
    assert normalize.normalize_label("pT > 2.0 GeV") != normalize.normalize_label("pT > 20 GeV")


def test_propose_labels_merges_identical_labels_across_unrelated_ids():
    """The real MadGraph case: one label, ids the slug rule cannot connect."""
    ents = [
        ("hepkg:generator:madgraph5_amcnlo", "generator", "MadGraph5_aMC@NLO"),
        ("hepkg:generator:mg5-amcnlo", "generator", "MadGraph5_aMC@NLO"),
        ("hepkg:generator:madgraph5-amcatnlo", "generator", "MadGraph5_aMC@NLO"),
        ("hepkg:generator:sherpa", "generator", "Sherpa"),
    ]
    assert normalize.propose(ents[:1] and [(i, k) for i, k, _ in ents]) == []  # ids never connect

    props = normalize.propose_labels(ents)
    ids = {i for p in props for i in (p.entity_id_a, p.entity_id_b)}
    assert ids == {
        "hepkg:generator:madgraph5_amcnlo",
        "hepkg:generator:mg5-amcnlo",
        "hepkg:generator:madgraph5-amcatnlo",
    }
    assert all(p.method == "label_exact" and p.score == 1.0 for p in props)


def test_propose_labels_never_merges_across_kinds():
    ents = [
        ("hepkg:object:bjet", "detector_object", "b-tagged jet"),
        ("hepkg:process:bjet", "physics_process", "b-tagged jet"),
    ]
    assert normalize.propose_labels(ents) == []


def test_label_norm_only_covers_what_exact_missed():
    ents = [
        ("hepkg:object:a", "detector_object", "b-tagged jet"),
        ("hepkg:object:b", "detector_object", "b-tagged jet"),   # exact with a
        ("hepkg:object:c", "detector_object", "B Tagged Jet"),   # only after folding
    ]
    props = normalize.propose_labels(ents)
    by_method = {p.method: [] for p in props}
    for p in props:
        by_method[p.method].append(frozenset((p.entity_id_a, p.entity_id_b)))

    assert by_method["label_exact"] == [frozenset({"hepkg:object:a", "hepkg:object:b"})]
    # c joins, and the a/b edge is not re-emitted under the weaker method
    assert frozenset({"hepkg:object:a", "hepkg:object:b"}) not in by_method["label_norm"]
    assert frozenset({"hepkg:object:a", "hepkg:object:c"}) in by_method["label_norm"]


def test_propose_labels_ignores_blank_labels():
    ents = [
        ("hepkg:object:a", "detector_object", ""),
        ("hepkg:object:b", "detector_object", "   "),
    ]
    assert normalize.propose_labels(ents) == []


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
    papers = {p for *_, ps in entities for p in ps}
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
    for eid, kind, label, ps in entities:
        conn.execute("INSERT INTO entity (entity_id, kind, label) VALUES (?,?,?)", (eid, kind, label))
        for p in ps:
            conn.execute(
                "INSERT INTO entity_occurrence (bundle_id, entity_id, paper_id, kind, label)"
                " VALUES (?,?,?,?,?)",
                (f"b{p}", eid, p, kind, label),
            )
    return conn


# Labels are real wordings, not echoes of the id: the Tier 1 label rules key on
# them, so a fixture that set label = entity_id would make the id rule and the
# label rule indistinguishable.
ENTITIES = [
    ("hepkg:object:b_jet", "detector_object", "b-tagged jet", ["1", "2", "3"]),       # 3 papers
    ("hepkg:object:b-jet", "detector_object", "b-tagged jet", ["1", "2", "3", "4"]),  # 4 -> canonical
    ("hepkg:object:bjet", "detector_object", "B Tagged Jet", ["5", "6"]),             # 2 papers
    ("hepkg:generator:sherpa_2_2_1", "generator", "Sherpa 2.2.1", ["1"]),
    ("hepkg:generator:sherpa_2_2_2", "generator", "Sherpa 2.2.2", ["2"]),
]


def test_build_writes_proposals_but_does_not_resolve():
    conn = _db_with(ENTITIES)
    run.build(conn)
    by_method = dict(conn.execute(
        "SELECT method, COUNT(*) FROM same_as WHERE status='proposed' GROUP BY method"))
    # the id rule connects the b-jet trio; the sherpa versions stay apart
    assert by_method["normalize"] == 2
    # b_jet and b-jet share a byte-identical label; bjet only matches once folded
    assert by_method["label_exact"] == 1
    assert by_method["label_norm"] == 1
    # nothing resolves yet
    assert conn.execute("SELECT COUNT(*) FROM entity_canonical").fetchone()[0] == 0
    # entity table untouched
    assert conn.execute("SELECT COUNT(*) FROM entity").fetchone()[0] == 5


def test_label_rules_find_a_cluster_the_id_rule_cannot_see():
    """Same label, ids with nothing in common — the MadGraph shape (D-050)."""
    conn = _db_with(ENTITIES + [
        ("hepkg:generator:madgraph5_amcnlo", "generator", "MadGraph5_aMC@NLO", ["1"]),
        ("hepkg:generator:mg5-amcnlo", "generator", "MadGraph5_aMC@NLO", ["2"]),
    ])
    run.build(conn)
    store.confirm(conn)
    store.materialize_canonical(conn)
    canon = dict(conn.execute("SELECT entity_id, canonical_id FROM entity_canonical"))
    assert canon["hepkg:generator:madgraph5_amcnlo"] == canon["hepkg:generator:mg5-amcnlo"]
    # and the version-numbered generators are still not touched
    assert "hepkg:generator:sherpa_2_2_1" not in canon


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
