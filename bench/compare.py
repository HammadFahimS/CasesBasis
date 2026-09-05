"""Compare two latency runs and say whether the change was real.

    python -m bench.compare bench/results/<before>.json bench/results/<after>.json

The verdict is deliberately conservative. A bootstrap confidence interval on the
median difference has to exclude zero *and* the shift has to clear an absolute
effect floor before anything is called IMPROVED. Two runs of identical code on a
shared network will differ; this is the guard against reading that as progress.

Exit code is 1 if any probe REGRESSED, so this can gate a change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bench.stats import judge

_MARK = {
    "IMPROVED": "✓", "REGRESSED": "✗", "NOISE": "·", "NEW": "+", "GONE": "-",
}


def _samples(run: dict[str, Any]) -> dict[str, list[float]]:
    return {
        p["name"]: p.get("samples", [])
        for p in run.get("probes", [])
        if p.get("samples")
    }


def compare(before: dict, after: dict, min_effect_ms: float = 5.0) -> list:
    from bench.stats import Verdict

    b, a = _samples(before), _samples(after)
    verdicts = []
    for name in sorted(set(b) | set(a)):
        if name not in b:
            verdicts.append(Verdict(name, 0, 0, 0, 0, 0, 0, "NEW", "not in baseline"))
        elif name not in a:
            verdicts.append(Verdict(name, 0, 0, 0, 0, 0, 0, "GONE", "absent from new run"))
        else:
            verdicts.append(judge(name, b[name], a[name], min_effect_ms=min_effect_ms))
    return verdicts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench.compare", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--min-effect-ms", type=float, default=5.0,
                    help="ignore shifts smaller than this (default 5)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    verdicts = compare(before, after, args.min_effect_ms)

    if args.json:
        print(json.dumps([v.to_dict() for v in verdicts], indent=2))
    else:
        bl, al = before.get("label", "?"), after.get("label", "?")
        bs, as_ = before.get("git", {}).get("sha", "?"), after.get("git", {}).get("sha", "?")
        print(f"\n{bl} ({bs})  →  {al} ({as_})\n")
        print(f"  {'probe':<16} {'before':>10} {'after':>10} {'delta':>11} {'':>8}  90% CI")
        print(f"  {'-' * 72}")
        for v in verdicts:
            if v.label in ("NEW", "GONE"):
                print(f"  {_MARK[v.label]} {v.probe:<14} {v.note}")
                continue
            print(
                f"  {_MARK[v.label]} {v.probe:<14}"
                f" {v.before_p50:>9.1f}ms {v.after_p50:>9.1f}ms"
                f" {v.delta_ms:>+9.1f}ms {v.delta_pct:>+6.1f}%"
                f"  [{v.ci_low:+.1f}, {v.ci_high:+.1f}]"
                + (f"  {v.note}" if v.note else "")
            )
        real = [v for v in verdicts if v.label in ("IMPROVED", "REGRESSED")]
        if not real:
            print("\n  No probe moved beyond noise.")
        else:
            net = sum(v.delta_ms for v in real)
            print(f"\n  Net across probes that actually moved: {net:+.1f} ms")

        for run, name in ((before, "before"), (after, "after")):
            if run.get("git", {}).get("dirty"):
                print(f"  WARNING: '{name}' run was taken on a dirty tree")

    return 1 if any(v.label == "REGRESSED" for v in verdicts) else 0


if __name__ == "__main__":
    sys.exit(main())
