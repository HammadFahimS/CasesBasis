-- Simulated dynamic cases authored via the CRUD editor.
-- case_json holds the full schemas.simulated_case.SimulatedCase document;
-- layout_json holds editor node positions ({state_id: {x, y}}) so the clinical
-- schema stays presentation-free. is_gold marks protected seed cases that
-- cannot be edited or deleted (only duplicated).
CREATE TABLE IF NOT EXISTS simulated_cases (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    specialty   TEXT,
    difficulty  TEXT NOT NULL DEFAULT 'medium',
    case_json   TEXT NOT NULL,
    layout_json TEXT NOT NULL DEFAULT '{}',
    is_gold     INTEGER NOT NULL DEFAULT 0,
    created_by  TEXT REFERENCES residents(id) ON DELETE SET NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_simulated_cases_updated ON simulated_cases(updated_at DESC);
