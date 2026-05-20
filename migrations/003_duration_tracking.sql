-- Track total active time (in seconds) per session.
-- Accumulated on pause and evaluate; excludes time spent paused.
ALTER TABLE web_sessions ADD COLUMN duration_seconds INTEGER DEFAULT 0;
