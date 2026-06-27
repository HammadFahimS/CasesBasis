-- Add resident_code (human-readable ID, e.g. R-A4729) and last_name for
-- the Resident ID + Last Name login flow. Both are nullable so existing
-- email-based residents are unaffected.

ALTER TABLE residents ADD COLUMN resident_code TEXT;
ALTER TABLE residents ADD COLUMN last_name     TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_residents_code
    ON residents (resident_code)
    WHERE resident_code IS NOT NULL;
