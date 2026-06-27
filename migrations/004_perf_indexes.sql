-- Indexes on created_at columns used in ORDER BY clauses improve
-- sort performance as the sessions and evaluations tables grow.
CREATE INDEX IF NOT EXISTS idx_web_sessions_created  ON web_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_created   ON evaluations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_web_sessions_status   ON web_sessions(status);
