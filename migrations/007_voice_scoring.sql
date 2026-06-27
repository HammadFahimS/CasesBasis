-- Score a concluded voice run so it shows up in the resident's analytics
-- alongside manual-sim cases. Computed once at debrief from backend.sim_scoring
-- (recognition, safety, medication criteria) and stored on the run itself.
--   overall_score    — 0-100, percent of assessment criteria met
--   assessment_json   — the full Criterion list (for the review drawer)
--   duration_seconds  — case-clock seconds elapsed at conclusion
--   evaluated_at      — when the debrief concluded the run
ALTER TABLE voice_sessions ADD COLUMN overall_score INTEGER;
ALTER TABLE voice_sessions ADD COLUMN assessment_json TEXT;
ALTER TABLE voice_sessions ADD COLUMN duration_seconds INTEGER DEFAULT 0;
ALTER TABLE voice_sessions ADD COLUMN evaluated_at TEXT;
