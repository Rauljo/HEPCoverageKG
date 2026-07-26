-- =============================================================================
-- HEPCoverageKG aliases layer (D-025). Derived, non-destructive: these tables
-- live in the same DB as the import store but are OWNED by the aliases package
-- and never modify entity / entity_occurrence / assertion. Undo is DELETE.
-- See vault/ideas/open-vocab-reconciliation.md and bundle-importer-design.md.
-- =============================================================================

-- One proposed-or-confirmed equivalence between two entities of the same kind.
-- method: how it was nominated (normalize | ngram | embed | llm).
-- status: proposed (awaiting review) | confirmed | rejected | auto (trusted tier).
CREATE TABLE IF NOT EXISTS same_as (
    entity_id_a  TEXT NOT NULL REFERENCES entity(entity_id),
    entity_id_b  TEXT NOT NULL REFERENCES entity(entity_id),
    method       TEXT NOT NULL,
    score        REAL NOT NULL,
    status       TEXT NOT NULL DEFAULT 'proposed'
                 CHECK (status IN ('proposed','confirmed','rejected','auto')),
    created_at   TEXT NOT NULL,
    PRIMARY KEY (entity_id_a, entity_id_b, method)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_same_as_a ON same_as(entity_id_a);
CREATE INDEX IF NOT EXISTS idx_same_as_b ON same_as(entity_id_b);
CREATE INDEX IF NOT EXISTS idx_same_as_status ON same_as(status);

-- Derived resolution: each entity -> its cluster's canonical representative.
-- Materialized from confirmed + auto same_as edges only (proposals never resolve).
-- canonical_source: 'auto' (most papers, tie -> lexicographic) or 'override' (human pick).
CREATE TABLE IF NOT EXISTS entity_canonical (
    entity_id         TEXT PRIMARY KEY REFERENCES entity(entity_id),
    canonical_id      TEXT NOT NULL REFERENCES entity(entity_id),
    cluster_size      INTEGER NOT NULL,
    canonical_source  TEXT NOT NULL DEFAULT 'auto'
                      CHECK (canonical_source IN ('auto','override'))
) STRICT;

CREATE INDEX IF NOT EXISTS idx_entity_canonical_canon ON entity_canonical(canonical_id);

-- Optional human override of a cluster's canonical representative. Consulted when
-- materializing entity_canonical, so the auto pick is never binding.
CREATE TABLE IF NOT EXISTS canonical_override (
    cluster_member  TEXT NOT NULL REFERENCES entity(entity_id),  -- any member id
    canonical_id    TEXT NOT NULL REFERENCES entity(entity_id),  -- the chosen rep
    PRIMARY KEY (cluster_member)
) STRICT;
