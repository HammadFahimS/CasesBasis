from __future__ import annotations

from collections import Counter
from typing import Iterable


def normalize_history(items: Iterable[object]) -> list[str]:
    return [text for item in items if (text := str(item).strip())]


def most_common(items: Iterable[object]) -> str | None:
    counter = Counter(normalize_history(items))
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def most_common_values(items: Iterable[object], *, limit: int) -> list[str]:
    return [value for value, _ in Counter(normalize_history(items)).most_common(limit)]
