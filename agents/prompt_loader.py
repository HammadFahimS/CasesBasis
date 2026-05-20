from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=None)
def load(agent: str, name: str) -> str:
    """Load and cache a prompt from agents/prompts/{agent}/{name}.md"""
    path = _PROMPTS_DIR / agent / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def render(template: str, **kwargs: object) -> str:
    """
    Substitute {{variable_name}} placeholders in a prompt template.

    Uses double-brace syntax so literal { } in JSON schema examples are unaffected.
    Raises KeyError if a placeholder has no matching kwarg.
    """
    def replacer(m: re.Match) -> str:
        key = m.group(1)
        if key not in kwargs:
            raise KeyError(f"Prompt template variable '{{{key}}}' not provided")
        return str(kwargs[key])

    return re.sub(r"\{\{(\w+)\}\}", replacer, template)
