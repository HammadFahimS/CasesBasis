"""Sample statistics for the latency harness — pure, dependency-free, testable.

Latency distributions are long-tailed: a mean is dominated by the slow tail and a
single fast run proves nothing. Everything here works off **raw samples** so the
comparison step can bootstrap a confidence interval rather than eyeball two
averages and declare victory.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile. ``p`` is a fraction (0.5 = median)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


@dataclass
class Summary:
    """Descriptive stats for one probe's samples. ``p95`` is the number that
    matters — it is what a resident actually feels on a bad turn."""

    n: int
    p50: float
    p95: float
    minimum: float
    maximum: float
    stdev: float

    def to_dict(self) -> dict:
        return {
            "n": self.n, "p50": round(self.p50, 2), "p95": round(self.p95, 2),
            "min": round(self.minimum, 2), "max": round(self.maximum, 2),
            "stdev": round(self.stdev, 2),
        }


def summarize(samples: list[float]) -> Summary:
    if not samples:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(samples)
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n if n > 1 else 0.0
    return Summary(
        n=n,
        p50=percentile(samples, 0.50),
        p95=percentile(samples, 0.95),
        minimum=min(samples),
        maximum=max(samples),
        stdev=math.sqrt(var),
    )


def bootstrap_ci_median_diff(
    before: list[float], after: list[float],
    iters: int = 2000, alpha: float = 0.10, seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval on ``median(after) - median(before)``.

    Resamples both sample sets with replacement and takes the empirical
    percentiles of the difference. Negative means *after* is faster. If the
    interval straddles zero, the change is indistinguishable from run-to-run
    noise — which is the whole reason this function exists.

    Seeded so a given pair of runs always produces the same verdict.
    """
    if not before or not after:
        return (0.0, 0.0)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(iters):
        ra = [before[rng.randrange(len(before))] for _ in before]
        rb = [after[rng.randrange(len(after))] for _ in after]
        diffs.append(percentile(rb, 0.5) - percentile(ra, 0.5))
    return percentile(diffs, alpha / 2), percentile(diffs, 1 - alpha / 2)


@dataclass
class Verdict:
    """The comparison outcome for one probe."""

    probe: str
    before_p50: float
    after_p50: float
    delta_ms: float
    delta_pct: float
    ci_low: float
    ci_high: float
    label: str            # IMPROVED | REGRESSED | NOISE | NEW | GONE
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "probe": self.probe,
            "before_p50": round(self.before_p50, 2),
            "after_p50": round(self.after_p50, 2),
            "delta_ms": round(self.delta_ms, 2),
            "delta_pct": round(self.delta_pct, 1),
            "ci": [round(self.ci_low, 2), round(self.ci_high, 2)],
            "label": self.label,
            "note": self.note,
        }


def judge(
    probe: str, before: list[float], after: list[float],
    min_effect_ms: float = 5.0, seed: int = 0,
) -> Verdict:
    """Decide whether a change is real, using two independent gates.

    A change counts only if the bootstrap CI excludes zero **and** the shift
    clears ``min_effect_ms``. The second gate stops us celebrating a 0.4 ms
    "win" on a pure-Python probe where the CI is tight enough to resolve noise
    we do not care about.
    """
    b50, a50 = percentile(before, 0.5), percentile(after, 0.5)
    delta = a50 - b50
    pct = (delta / b50 * 100.0) if b50 else 0.0
    lo, hi = bootstrap_ci_median_diff(before, after, seed=seed)

    if lo <= 0.0 <= hi:
        label, note = "NOISE", "CI includes zero — re-run with more samples if it matters"
    elif abs(delta) < min_effect_ms:
        label, note = "NOISE", f"below the {min_effect_ms:g}ms effect floor"
    elif delta < 0:
        label, note = "IMPROVED", ""
    else:
        label, note = "REGRESSED", ""
    return Verdict(probe, b50, a50, delta, pct, lo, hi, label, note)
