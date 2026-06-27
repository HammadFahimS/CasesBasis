from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResidentAccount(StrictModel):
    id: str
    email: str
    display_name: str
    created_at: str
    last_seen: str | None = None
