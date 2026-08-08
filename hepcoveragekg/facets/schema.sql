-- =============================================================================
-- HEPCoverageKG facets layer (D-052). Derived, non-destructive: owned by the
-- facets package, never modifies entity / entity_occurrence / assertion. Undo
-- is DELETE WHERE vocabulary = '<version>'.
--
-- The `vocabulary` column on every table does two jobs: re-deriving is
-- delete-then-insert scoped to one version, and when upstream ships facets-v2
-- both versions can be held at once and diffed — so "did the new vocabulary
-- improve coverage" is a query rather than an argument.
-- =============================================================================

-- One row per (paper, entity, field, value). Tags and renames share this table:
-- the difference between them is a cardinality constraint, not a shape.
--
--   descriptive kinds -> ZERO OR MORE tags ("Data-driven jet smearing" gets
--       both JetSmearing and DataDriven)
--   detector_object / generator -> ZERO OR ONE canonical name, plus `version`
--       when the label carried one ("MG5_aMC@NLO 2.6.2" -> family + 2.6.2)
--
-- KEYED PER OCCURRENCE, NOT PER ENTITY. A facet is a property of the label, and
-- `entity.label` is a rollup — the schema calls entity_id a "convenience rollup
-- ONLY, not identity". 312 entity_ids carry different labels in different
-- papers, and the difference changes the tag:
--     2107.12553 wrote "b-tagged small-R jet"   -> no canonical match
--     the rollup label is "b-tagged jet"        -> BJet
-- Deriving from the rollup invents a BJet for a paper that never claimed one.
-- Upstream derives per bundle for the same reason; this is what card parity
-- turns out to mean in practice (D-052).
--
-- `field` uses the analysis-card field names, so the card is a projection with
-- no translation step.
--
-- NOTE the name. `entity_canonical` (aliases layer) is a DIFFERENT THING: it
-- says two entities are the same entity. This says a label belongs in an enum.
-- Different rungs of the ladder (S-68); joining the wrong one is a near-
-- invisible bug.
CREATE TABLE IF NOT EXISTS entity_facet (
    paper_id    TEXT NOT NULL,
    entity_id   TEXT NOT NULL REFERENCES entity(entity_id),
    field       TEXT NOT NULL,
    value       TEXT NOT NULL,
    version     TEXT,               -- generators only; NULL everywhere else
    vocabulary  TEXT NOT NULL,
    PRIMARY KEY (paper_id, entity_id, field, value, vocabulary)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_entity_facet_value ON entity_facet(field, value);
CREATE INDEX IF NOT EXISTS idx_entity_facet_entity ON entity_facet(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_facet_vocab ON entity_facet(vocabulary);

-- Rebuilt selection-cut trees. The originals keep signature = NULL; this sits
-- beside them, marked derived, and a native signature always wins.
-- is_or_group: this assertion is one member of a flavour-OR. Every member of a
-- group stores the SAME any_of tree — that shared identity is how alternatives
-- are told apart from separate cuts.
CREATE TABLE IF NOT EXISTS assertion_signature_derived (
    assertion_id  TEXT PRIMARY KEY REFERENCES assertion(assertion_id),
    signature     TEXT NOT NULL CHECK (json_valid(signature)),
    is_or_group   INTEGER NOT NULL DEFAULT 0 CHECK (is_or_group IN (0, 1)),
    vocabulary    TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sig_derived_or ON assertion_signature_derived(is_or_group);

-- The analysis card: one paper reduced to sets of closed-vocabulary values.
--
-- A VIEW, deliberately. The card is fully computable from entity_facet, so
-- storing a copy would add a second source of truth that can drift — a failure
-- this project has already hit twice. As a view it cannot go stale, and set
-- queries over it are still plain SQL.
--
-- The view drops entity_id; the un-projected query (same filter, without the
-- DISTINCT over entities) is what returns the labels that caused each match.
-- Both matter: the card answers "which papers use ABCD", the un-projected form
-- answers "and how did each of them do it" — six papers, six different ABCD
-- variants. Never show the card alone (S-69).
--
-- `category` and `experiments` live on `paper` and are joined at query time.
CREATE VIEW IF NOT EXISTS analysis_card AS
    SELECT DISTINCT
        paper_id   AS paper_id,
        field      AS field,
        value      AS value,
        vocabulary AS vocabulary
    FROM entity_facet;
