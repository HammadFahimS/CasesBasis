-- Browser-side telemetry for a voice session: page errors and audio outcomes
-- (watchdog fires, skipped lines, play() rejections). Written by
-- /api/voice/client-log so a "reply shown but not heard" run can be diagnosed
-- from the server instead of from a console nobody was watching.
CREATE TABLE IF NOT EXISTS voice_client_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    resident_id TEXT NOT NULL,
    t           TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_voice_client_events_session
    ON voice_client_events(session_id, id);
