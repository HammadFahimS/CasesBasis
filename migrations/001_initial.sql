CREATE TABLE IF NOT EXISTS residents (
    id           TEXT PRIMARY KEY,
    email        TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    last_seen    TEXT
);

CREATE TABLE IF NOT EXISTS magic_links (
    token       TEXT PRIMARY KEY,
    resident_id TEXT NOT NULL REFERENCES residents(id) ON DELETE CASCADE,
    expires_at  TEXT NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS web_sessions (
    id             TEXT PRIMARY KEY,
    resident_id    TEXT REFERENCES residents(id) ON DELETE SET NULL,
    diagnosis      TEXT NOT NULL,
    difficulty     TEXT NOT NULL DEFAULT 'medium',
    status         TEXT NOT NULL DEFAULT 'active',
    case_json      TEXT NOT NULL,
    state_json     TEXT NOT NULL DEFAULT '{}',
    graph_snapshot TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    evaluated_at   TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES web_sessions(id) ON DELETE CASCADE,
    resident_id     TEXT REFERENCES residents(id) ON DELETE SET NULL,
    evaluation_json TEXT NOT NULL,
    overall_score   INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES web_sessions(id) ON DELETE CASCADE,
    resident_id TEXT,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_cache (
    cache_key  TEXT PRIMARY KEY,
    case_json  TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_web_sessions_resident ON web_sessions(resident_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_session   ON evaluations(session_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_resident  ON evaluations(resident_id);
CREATE INDEX IF NOT EXISTS idx_events_session        ON events(session_id);
