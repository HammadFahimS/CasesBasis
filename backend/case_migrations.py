"""Case-JSON schema migration runner.

Every authored case is stored as a ``case_json`` blob in ``simulated_cases``.
When the :mod:`schemas.simulated_case` Pydantic models gain new fields, old
blobs silently lack those fields — Pydantic uses the field default, so purely
*additive* additions are safe at parse-time.  But a *rename*, *removal*, or
a field that needs a computed backfill requires an explicit transform applied
before the blob is validated.

HOW TO ADD A MIGRATION:

1. Write a transform function::

       def _v1_to_v2(raw: dict) -> dict:
           raw = dict(raw)
           for a in raw.get("actors", []):
               a.setdefault("voice_id", "")   # example: new field with default
           return raw

2. Register it::

       _MIGRATIONS[1] = _v1_to_v2

3. Bump CASE_SCHEMA_VERSION to 2.

``migrate_case_json`` chains all registered transforms in version order, so
a blob at version 0 gets migrated through every step automatically.  It is
idempotent: running it on an already-current blob is a no-op (just stamps the
version).

The migration is applied on every ``get_case()`` read — no DB writes are
needed for additive changes.  The ``migrate-cases`` CLI command does an
explicit write-back pass for transforms that require persisted data changes.
"""

from __future__ import annotations

from typing import Callable

CASE_SCHEMA_VERSION = 1

# Migration registry — keys are the *from* version, values transform the raw dict.
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    # Example (uncomment and edit when adding the first real migration):
    # 0: _v0_to_v1,
}


def migrate_case_json(raw: dict) -> dict:
    """Return a case dict upgraded to CASE_SCHEMA_VERSION.

    Applies all registered transforms in ascending order between the blob's
    current ``schema_version`` and ``CASE_SCHEMA_VERSION``.  The returned dict
    has ``schema_version`` stamped; the input is never mutated."""
    v = int(raw.get("schema_version") or 0)
    if v >= CASE_SCHEMA_VERSION and not _MIGRATIONS:
        return raw
    result = dict(raw)
    for ver in sorted(k for k in _MIGRATIONS if v <= k < CASE_SCHEMA_VERSION):
        result = _MIGRATIONS[ver](result)
    result["schema_version"] = CASE_SCHEMA_VERSION
    return result
