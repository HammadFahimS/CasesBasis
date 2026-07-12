-- Question Bank (UWorld-style practice) — two tables.
--
-- qbank_cases: the assessment content, seeded once on startup from the bundled
--   backend/data/qbank_cases.json (validated against schemas.qbank.QBankCase).
--   The filterable tags are denormalised into columns so the session builder can
--   count/filter without parsing JSON; the full document rides in case_json.
-- qbank_attempts: one row per answered checkpoint, per resident — the durable
--   per-checkpoint history behind the UWorld-style performance stats. Kept
--   separate from the content so re-seeding content never touches user progress.
CREATE TABLE IF NOT EXISTS qbank_cases (
    case_id        TEXT PRIMARY KEY,
    stem           TEXT NOT NULL,
    subject        TEXT NOT NULL DEFAULT 'General',
    subtopic       TEXT NOT NULL DEFAULT '',
    system         TEXT NOT NULL DEFAULT 'General',
    exam           TEXT NOT NULL DEFAULT '',
    difficulty     INTEGER NOT NULL DEFAULT 3,
    high_yield     INTEGER NOT NULL DEFAULT 0,
    checkpoint_count INTEGER NOT NULL DEFAULT 0,
    case_json      TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qbank_cases_subject ON qbank_cases(subject);
CREATE INDEX IF NOT EXISTS idx_qbank_cases_system  ON qbank_cases(system);

CREATE TABLE IF NOT EXISTS qbank_attempts (
    id            TEXT PRIMARY KEY,
    resident_id   TEXT REFERENCES residents(id) ON DELETE CASCADE,
    case_id       TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    correct       INTEGER NOT NULL DEFAULT 0,
    chosen_index  INTEGER,
    mode          TEXT NOT NULL DEFAULT 'tutor',
    subject       TEXT NOT NULL DEFAULT '',
    system        TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qbank_attempts_resident
    ON qbank_attempts(resident_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_qbank_attempts_checkpoint
    ON qbank_attempts(resident_id, checkpoint_id);
