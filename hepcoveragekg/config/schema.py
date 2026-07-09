# =============================================================================
# HEPCoverageKG: Schema-as-config for RAG extraction
# Ontology (entity kinds, predicates) ported from graph_schema.py (supervisor).
# Controlled vocabularies seeded from graph_extractor.py's term lists (supervisor).
# Query-template / plausibility pattern ported from DeepCollector's config/schema.py.
# =============================================================================

SCHEMA_VERSION = "public-results-graph-v0.1"  # kept in sync with graph_schema.py

# --- Ontology: the typed vocabulary of the knowledge graph -------------------
ENTITY_KINDS = [
    "paper", "result", "dataset", "collision_system", "detector_object",
    "object_definition", "event_region", "selection_requirement", "observable",
    "physics_process", "background", "background_method", "result_quantity",
    "systematic_uncertainty",
]

ASSERTION_PREDICATES = [
    "paper_reports_result", "result_uses_dataset", "result_has_collision_system",
    "result_targets_process", "result_has_final_state", "result_defines_object",
    "result_defines_region", "region_requires_object", "region_vetoes_object",
    "object_has_selection", "result_estimates_background", "background_uses_method",
    "result_measures_observable", "result_reports_quantity", "result_has_systematic",
]

ASSERTION_STATUSES = ["accepted", "quarantined", "rejected", "needs_review"]
EXTRACTION_SUPPORT = ["explicit", "implicit", "absent", "unclear"]

# How an assertion's (subject, predicate, object) triple was produced. "llm"
# covers the current single-pass RAG design; "regex"/"metadata" match the
# values graph_extractor.py's deterministic baseline uses.
EXTRACTION_METHODS = ["llm", "regex", "metadata"]

ALLOWED_EXPERIMENTS = ["ATLAS", "CMS"]

# --- Controlled vocabularies per entity kind ----------------------------------
# Seeded from graph_extractor.py's TermPattern lists (its regex baseline) -
# used here as the closed label sets a RAG query is allowed to choose from.
DETECTOR_OBJECT_VOCAB = [
    "electron", "muon", "photon", "small_r_jet", "large_r_jet", "b_tagged_jet",
    "hadronic_tau", "missing_transverse_momentum", "track", "vertex",
]

PHYSICS_PROCESS_VOCAB = [
    "higgs_boson", "top_quark", "w_boson", "z_boson", "diboson",
    "supersymmetry", "dark_matter", "heavy_ion", "standard_model",
]

OBSERVABLE_VOCAB = [
    "cross_section", "differential_cross_section", "mass",
    "transverse_momentum", "limit", "branching_fraction",
]

BACKGROUND_VOCAB = [
    "multijet_background", "fake_background", "top_background",
    "zjets_background", "wjets_background", "diboson_background",
]

# --- Per-predicate RAG extraction definitions ---------------------------------
# DeepCollector's CATALOG_SCHEMA pattern (description + query template), keyed
# by predicate instead of flat column, since one paper's result fans out into
# many typed (subject, predicate, object) assertions rather than one flat row.
# "{result}" is substituted with the paper's result title/id at query time;
# "{vocab}" is substituted with the predicate's controlled_vocabulary.
#
# "multi_value" tells CatalogState.update_assertion how to arbitrate: False
# means two different answers for the same predicate are a CONFLICT to
# resolve via confidence-beats (e.g. a result has one collision energy);
# True means two different answers are both valid facts to keep side by side
# (e.g. a result can target several physics processes at once).
PREDICATE_SCHEMA = {
    "result_has_final_state": {
        "object_kind": "detector_object",
        "multi_value": True,
        "description": "MUSiC-style final-state signature (object multiplicities) targeted by the result - the core coverage-mapping field.",
        "query": "State the final-state signature searched for in {result} as object multiplicities (e.g. '2 leptons + missing transverse momentum'). Use only objects from: {vocab}.",
        "controlled_vocabulary": DETECTOR_OBJECT_VOCAB,
    },
    "result_has_collision_system": {
        "object_kind": "collision_system",
        "multi_value": False,
        "description": "Center-of-mass collision energy at which the result's data was collected.",
        "query": "What center-of-mass energy (in TeV) was used for {result}? Reply with a single number.",
        "controlled_vocabulary": None,
        "qualifier_field": "energy_tev",
    },
    "result_uses_dataset": {
        "object_kind": "dataset",
        "multi_value": False,
        "description": "Integrated luminosity of the dataset the result is based on.",
        "query": "What integrated luminosity (in fb^-1) did {result} use? Reply with a single number.",
        "controlled_vocabulary": None,
        "qualifier_field": "integrated_luminosity_fb",
    },
    "result_targets_process": {
        "object_kind": "physics_process",
        "multi_value": True,
        "description": "Physics process(es) the result targets or searches for.",
        "query": "Which physics process(es) does {result} target? Choose all that apply from: {vocab}.",
        "controlled_vocabulary": PHYSICS_PROCESS_VOCAB,
    },
    "result_measures_observable": {
        "object_kind": "observable",
        "multi_value": True,
        "description": "Observable(s) reported by the result.",
        "query": "Which observable(s) does {result} report? Choose all that apply from: {vocab}.",
        "controlled_vocabulary": OBSERVABLE_VOCAB,
    },
    "result_estimates_background": {
        "object_kind": "background",
        "multi_value": True,
        "description": "Background(s) estimated in the result's analysis.",
        "query": "Which background(s) does {result} estimate? Choose all that apply from: {vocab}.",
        "controlled_vocabulary": BACKGROUND_VOCAB,
    },
}

EXTRACTED_PREDICATES = list(PREDICATE_SCHEMA.keys())

# Predicates present in the ontology but without a query template yet - deferred
# past this first version (kinematic selections, systematics, numeric result
# values, background-estimation methods, region definitions/links).
DEFERRED_PREDICATES = [
    "result_defines_object", "result_defines_region", "region_requires_object",
    "region_vetoes_object", "object_has_selection", "background_uses_method",
    "result_reports_quantity", "result_has_systematic",
]

# --- Plausibility & missing-value handling (ported pattern from DeepCollector)
PLAUSIBILITY_THRESHOLDS = {
    "energy_tev": {"min": 0.9, "max": 14.0},
    "integrated_luminosity_fb": {"min": 0.001, "max": 3000.0},
}

MISSING_DATA_PLACEHOLDERS = {
    "", "[missing]", "unknown", "n/a", "not found", "not specified",
    "not available", "not mentioned", "none", "missing", "nan", "null",
    "[implausible]", "[error]", "[retrieval_error]",
}
