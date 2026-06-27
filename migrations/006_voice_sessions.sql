-- Durable storage for voice-driven (simulated dynamic case) sessions.
-- Separate from web_sessions (the manual ClinicalCase path): a voice run is a
-- SimulationEngine snapshot plus two transcript channels and the diagnostics
-- cards produced so far, so a parked run can be fully rehydrated and resumed.
--   snapshot_json         — engine.snapshot() (case_id + RunState)
--   history_json          — nurse-channel transcript [{role,text}, ...]
--   patient_history_json  — patient-channel transcript
--   cards_json            — diagnostics-drawer cards produced (full visual_payload)
-- status: active (live) | paused (parked) | ended (concluded, debriefed)
CREATE TABLE IF NOT EXISTS voice_sessions (
    id                   TEXT PRIMARY KEY,
    resident_id          TEXT REFERENCES residents(id) ON DELETE SET NULL,
    case_id              TEXT NOT NULL,
    status               TEXT NOT NULL DEFAULT 'active',
    snapshot_json        TEXT NOT NULL DEFAULT '{}',
    history_json         TEXT NOT NULL DEFAULT '[]',
    patient_history_json TEXT NOT NULL DEFAULT '[]',
    cards_json           TEXT NOT NULL DEFAULT '[]',
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    ended_at             TEXT
);

CREATE INDEX IF NOT EXISTS idx_voice_sessions_resident
    ON voice_sessions(resident_id, status, updated_at DESC);
