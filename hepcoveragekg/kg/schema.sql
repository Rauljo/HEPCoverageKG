-- =============================================================================
-- HEPCoverageKG bundle-import store (milestone 1).
-- See vault/ideas/bundle-importer-design.md (D-024). SQLite, STRICT tables.
-- JSON is stored as TEXT, guarded by json_valid(). Idempotent DDL
-- (CREATE ... IF NOT EXISTS). Foreign keys are enforced per-connection by
-- store.connect() (PRAGMA foreign_keys = ON), so records must be inserted
-- parent-first. Two columns are renamed off SQLite reserved words:
--   source_block.block_order  <- bundle field "order"
--   qa_finding.check_name     <- bundle field "check"
-- [inj] marks provenance keys injected during flattening (not native schema
-- fields). Soft pointers carry no FK to avoid the paper<->bundle cycle and
-- insert-order fragility; they are validated in the importer's semantic layer.
-- =============================================================================

-- ---------------------------------------------------------- provenance / bookkeeping

CREATE TABLE IF NOT EXISTS paper (
    arxiv_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    category          TEXT,
    experiments       TEXT CHECK (experiments IS NULL OR json_valid(experiments)),
    doi               TEXT,
    inspire_id        TEXT,
    cds_id            TEXT,
    latest_bundle_id  TEXT,   -- soft pointer (avoids paper<->bundle_import cycle)
    source_hash       TEXT,   -- soft pointer (denormalized convenience)
    paper_map         TEXT CHECK (paper_map IS NULL OR json_valid(paper_map))
) STRICT;

CREATE TABLE IF NOT EXISTS source_snapshot (
    source_hash            TEXT PRIMARY KEY,
    paper_id               TEXT NOT NULL REFERENCES paper(arxiv_id),
    normalization_version  TEXT,
    title                  TEXT NOT NULL,
    experiments            TEXT CHECK (experiments IS NULL OR json_valid(experiments)),
    problems               TEXT CHECK (problems IS NULL OR json_valid(problems)),
    source_path            TEXT NOT NULL,
    conversion_qa          TEXT CHECK (conversion_qa IS NULL OR json_valid(conversion_qa))
) STRICT;

CREATE TABLE IF NOT EXISTS source_block (
    source_hash    TEXT NOT NULL REFERENCES source_snapshot(source_hash),  -- [inj]
    block_id       TEXT NOT NULL,
    kind           TEXT NOT NULL,
    block_order    INTEGER NOT NULL,   -- bundle field "order"
    section_id     TEXT,
    section_title  TEXT,
    dom_id         TEXT,
    text           TEXT NOT NULL,
    text_hash      TEXT NOT NULL,
    table_rows     TEXT CHECK (table_rows IS NULL OR json_valid(table_rows)),
    attributes     TEXT CHECK (attributes IS NULL OR json_valid(attributes)),
    PRIMARY KEY (source_hash, block_id)
) STRICT;

CREATE TABLE IF NOT EXISTS bundle_import (
    bundle_id            TEXT PRIMARY KEY,
    paper_arxiv_id       TEXT NOT NULL REFERENCES paper(arxiv_id),
    source_hash          TEXT NOT NULL REFERENCES source_snapshot(source_hash),
    bundle_content_hash  TEXT NOT NULL,
    schema_version       TEXT NOT NULL,
    imported_at          TEXT NOT NULL,
    usage                TEXT CHECK (usage IS NULL OR json_valid(usage)),
    warnings             TEXT CHECK (warnings IS NULL OR json_valid(warnings))
) STRICT;

-- ---------------------------------------------------------- entities (merged node + per-paper occurrences)

CREATE TABLE IF NOT EXISTS entity (
    entity_id     TEXT PRIMARY KEY,   -- convenience rollup ONLY, not identity — see entity_occurrence
    kind          TEXT NOT NULL,
    label         TEXT NOT NULL,
    aliases       TEXT CHECK (aliases IS NULL OR json_valid(aliases)),
    attributes    TEXT CHECK (attributes IS NULL OR json_valid(attributes)),
    external_ids  TEXT CHECK (external_ids IS NULL OR json_valid(external_ids))
) STRICT;

CREATE TABLE IF NOT EXISTS entity_occurrence (
    bundle_id     TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- [inj]
    entity_id     TEXT NOT NULL REFERENCES entity(entity_id),
    paper_id      TEXT NOT NULL,   -- [inj]
    kind          TEXT NOT NULL,
    label         TEXT NOT NULL,
    aliases       TEXT CHECK (aliases IS NULL OR json_valid(aliases)),
    attributes    TEXT CHECK (attributes IS NULL OR json_valid(attributes)),
    external_ids  TEXT CHECK (external_ids IS NULL OR json_valid(external_ids)),
    PRIMARY KEY (bundle_id, entity_id)
) STRICT;

-- ---------------------------------------------------------- activities (extraction/QA provenance)

CREATE TABLE IF NOT EXISTS activity (
    activity_id       TEXT PRIMARY KEY,
    bundle_id         TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- [inj]
    kind              TEXT NOT NULL,
    model             TEXT,
    provider          TEXT,
    prompt_id         TEXT,
    software_version  TEXT NOT NULL,
    input_hashes      TEXT CHECK (input_hashes IS NULL OR json_valid(input_hashes)),
    output_hashes     TEXT CHECK (output_hashes IS NULL OR json_valid(output_hashes)),
    metadata          TEXT CHECK (metadata IS NULL OR json_valid(metadata))
) STRICT;

-- ---------------------------------------------------------- assertions (the typed edges)

CREATE TABLE IF NOT EXISTS assertion (
    assertion_id       TEXT PRIMARY KEY,
    bundle_id          TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- [inj]
    paper_id           TEXT,   -- [inj] soft, denormalized for convenience
    subject_id         TEXT NOT NULL,
    predicate          TEXT NOT NULL,
    family             TEXT NOT NULL,
    object_id          TEXT,
    object_value       TEXT CHECK (object_value IS NULL OR json_valid(object_value)),
    signature          TEXT CHECK (signature IS NULL OR json_valid(signature)),
    status             TEXT NOT NULL,
    support            TEXT,
    extraction_method  TEXT,
    activity_id        TEXT REFERENCES activity(activity_id),
    revision_of        TEXT,   -- soft self-pointer (validated in semantic layer)
    notes              TEXT,
    qualifiers         TEXT CHECK (qualifiers IS NULL OR json_valid(qualifiers)),
    -- importer-maintained: content hash of the claim EXCLUDING status. On a
    -- re-import, an incoming assertion whose identity_hash differs from the
    -- stored one changed something other than status -> conflict (D-027).
    identity_hash      TEXT,
    -- exactly one object shape must be present
    CHECK ( (object_id IS NOT NULL) + (object_value IS NOT NULL) + (signature IS NOT NULL) = 1 ),
    FOREIGN KEY (bundle_id, subject_id) REFERENCES entity_occurrence(bundle_id, entity_id),
    FOREIGN KEY (bundle_id, object_id)  REFERENCES entity_occurrence(bundle_id, entity_id)
) STRICT;

-- ---------------------------------------------------------- evidence + many-to-many junction

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id            TEXT PRIMARY KEY,
    bundle_id              TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- [inj]
    source_hash            TEXT NOT NULL,
    block_id               TEXT NOT NULL,
    dom_id                 TEXT,
    quote                  TEXT NOT NULL,
    quote_hash             TEXT NOT NULL,
    char_start             INTEGER NOT NULL,
    char_end               INTEGER NOT NULL,
    section_title          TEXT NOT NULL,
    normalization_version  TEXT,
    FOREIGN KEY (source_hash, block_id) REFERENCES source_block(source_hash, block_id)
) STRICT;

CREATE TABLE IF NOT EXISTS assertion_evidence (
    assertion_id  TEXT NOT NULL REFERENCES assertion(assertion_id),
    evidence_id   TEXT NOT NULL REFERENCES evidence(evidence_id),
    PRIMARY KEY (assertion_id, evidence_id)
) STRICT;

-- ---------------------------------------------------------- external artifacts

-- Content-derived ids can recur across bundles (the same external link cited by
-- two papers), so the key is (bundle_id, artifact_id) -- one row per bundle.
CREATE TABLE IF NOT EXISTS artifact (
    artifact_id  TEXT NOT NULL,
    bundle_id    TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- [inj]
    kind         TEXT NOT NULL,
    url          TEXT NOT NULL,
    source       TEXT NOT NULL,
    label        TEXT,
    metadata     TEXT CHECK (metadata IS NULL OR json_valid(metadata)),
    PRIMARY KEY (bundle_id, artifact_id)
) STRICT;

-- ---------------------------------------------------------- QA findings (metadata, never scientific edges)

-- finding_id is content-derived, so the same finding recurs across papers (e.g.
-- "object met is outside the vocabulary" fires in every paper with a MET object
-- -- 18 such collisions in the pilot). Key is (bundle_id, finding_id): one row
-- per bundle, preserving which papers each finding fired on.
CREATE TABLE IF NOT EXISTS qa_finding (
    finding_id     TEXT NOT NULL,
    bundle_id      TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- [inj]
    check_name     TEXT NOT NULL,   -- bundle field "check"
    severity       TEXT NOT NULL,
    verdict        TEXT NOT NULL,
    message        TEXT NOT NULL,
    assertion_ids  TEXT CHECK (assertion_ids IS NULL OR json_valid(assertion_ids)),
    evidence_ids   TEXT CHECK (evidence_ids IS NULL OR json_valid(evidence_ids)),
    PRIMARY KEY (bundle_id, finding_id)
) STRICT;

-- ---------------------------------------------------------- completeness (ISOLATED: never a domain edge)

CREATE TABLE IF NOT EXISTS completeness_finding (
    finding_id                TEXT NOT NULL,
    bundle_id                 TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- [inj]
    rule_id                   TEXT NOT NULL,
    rule_version              TEXT NOT NULL,
    trigger_fired             INTEGER NOT NULL,   -- boolean 0/1
    verdict                   TEXT NOT NULL,
    note                      TEXT,
    trigger_evidence_ids      TEXT CHECK (trigger_evidence_ids IS NULL OR json_valid(trigger_evidence_ids)),
    expectation_evidence_ids  TEXT CHECK (expectation_evidence_ids IS NULL OR json_valid(expectation_evidence_ids)),
    PRIMARY KEY (bundle_id, finding_id)   -- content-derived id can recur across bundles
) STRICT;

-- ---------------------------------------------------------- expert decisions (review audit trail)

CREATE TABLE IF NOT EXISTS expert_decision (
    decision_id             TEXT PRIMARY KEY,
    bundle_id               TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- [inj]
    action                  TEXT NOT NULL,
    assertion_id            TEXT NOT NULL REFERENCES assertion(assertion_id),
    corrected_assertion_id  TEXT REFERENCES assertion(assertion_id),
    reviewer                TEXT NOT NULL,
    rationale               TEXT NOT NULL,
    decided_at              TEXT NOT NULL
) STRICT;

-- ---------------------------------------------------------- status history (append-only)

-- Records only OBSERVED CHANGES between our imports -- not Gabriel's internal
-- lifecycle, since we only ever see per-bundle snapshots. Stays empty after a
-- first import (nothing changes); fills when a re-import promotes or demotes an
-- existing assertion. Initial statuses are deliberately not logged: any past
-- state is reconstructable by walking these rows backwards from the current
-- assertion.status. Needed for milestone 4 ("show promotions land").
CREATE TABLE IF NOT EXISTS assertion_status_history (
    assertion_id  TEXT NOT NULL REFERENCES assertion(assertion_id),
    old_status    TEXT NOT NULL,
    new_status    TEXT NOT NULL,
    bundle_id     TEXT NOT NULL REFERENCES bundle_import(bundle_id),  -- import that observed it
    observed_at   TEXT NOT NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_status_history_assertion
    ON assertion_status_history(assertion_id);

-- ---------------------------------------------------------- views

-- The accepted view is a filter, never a destructive rebuild. Superseded
-- proposals drop out; corrections (expert_accepted with revision_of set) stay.
CREATE VIEW IF NOT EXISTS accepted_view AS
    SELECT * FROM assertion WHERE status = 'expert_accepted';

-- ---------------------------------------------------------- indexes (query columns)

CREATE INDEX IF NOT EXISTS idx_assertion_status      ON assertion(status);
CREATE INDEX IF NOT EXISTS idx_assertion_predicate   ON assertion(predicate);
CREATE INDEX IF NOT EXISTS idx_assertion_subject     ON assertion(subject_id);
CREATE INDEX IF NOT EXISTS idx_assertion_object      ON assertion(object_id);
CREATE INDEX IF NOT EXISTS idx_assertion_revision_of ON assertion(revision_of);
