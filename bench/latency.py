"""Latency benchmark runner for the live voice room.

    python -m bench.latency --label baseline
    python -m bench.latency --label after-step-1 --n 15
    python -m bench.latency --pure-only          # no network, no keys needed

Records raw samples to ``bench/results/`` so :mod:`bench.compare` can bootstrap a
confidence interval later. Commit the result files — the point is the history.

The benchmark builds its session **in memory** (one DB read for the case, no
writes) and snapshot/restores the engine around mutating probes, so running it
never leaves sessions behind or deteriorates the patient under measurement.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).parent / "results"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """True when *tracked* files differ from HEAD.

    Recorded because a benchmark taken on a modified tree cannot be reproduced
    from its SHA. Untracked files are excluded deliberately: scratch notes and
    result files would otherwise flag every run, and a warning that fires
    constantly is a warning nobody reads.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def build_context(case_id: str):
    """A live in-memory session on the gold case — no DB writes, no persistence."""
    from backend.nurse_brain import make_intent_matcher
    from backend.simulated_case_repo import get_case
    from backend.simulation_engine import SimulationEngine
    from backend.voice_session import VoiceSession
    from schemas.simulated_case import SimulatedCase

    from bench.probes import Context, ensure_stt_fixture, ensure_stt_pcm_fixture

    row = get_case(case_id)
    if row is None:
        raise SystemExit(
            f"case {case_id!r} is not in the DB. Seed it first:\n"
            f"    python -m backend.manage reseed-case {case_id}"
        )
    case = SimulatedCase.model_validate(row["case"])
    engine = SimulationEngine(case, intent_matcher=make_intent_matcher())
    session = VoiceSession(id="bench", resident_id="bench", case=case, engine=engine)

    audio, content_type, note = ensure_stt_fixture()
    pcm, pcm_note = ensure_stt_pcm_fixture()
    ctx = Context(
        session=session, stt_audio=audio, stt_content_type=content_type, stt_pcm=pcm,
    )
    for n in (note, pcm_note):
        if n:
            ctx.notes.append(n)
    return ctx


def run(
    probe_names: list[str], case_id: str, n: int, warmup: int, label: str,
) -> dict[str, Any]:
    from bench.probes import PROBES, capture_region

    ctx = build_context(case_id)
    results: list[dict[str, Any]] = []

    for name in probe_names:
        fn = PROBES[name]
        print(f"  {name:<16} ", end="", flush=True)
        # Warm-up samples are run and discarded: the first call of any network
        # probe pays a TLS handshake, and steady-state is what a mid-case turn
        # actually costs. `tts_ttfb_cold` measures the handshake on purpose.
        if warmup and name != "tts_ttfb_cold":
            try:
                fn(ctx, warmup)
            except Exception:
                pass
        res = fn(ctx, n)
        results.append(res.to_dict())
        if res.skipped:
            print(f"skipped — {res.skipped}")
        elif res.error and not res.samples:
            print(f"ERROR — {res.error}")
        else:
            s = res.to_dict()["stats"]
            suffix = f"  (errored after {s['n']}: {res.error})" if res.error else ""
            print(f"p50 {s['p50']:>8.1f} ms   p95 {s['p95']:>8.1f} ms   n={s['n']}{suffix}")

    import os
    return {
        "label": label,
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": {"sha": _git_sha(), "branch": _git_branch(), "dirty": _git_dirty()},
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "config": {
            "case_id": case_id, "n": n, "warmup": warmup,
            "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
            "tts_model": _tts_model(),
        },
        "elevenlabs_region": capture_region(),
        "notes": ctx.notes,
        "probes": results,
    }


def _tts_model() -> str:
    try:
        from backend.voice_service import tts_model
        return tts_model()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv
    load_dotenv()

    from bench.probes import PROBES, PURE_PROBES

    ap = argparse.ArgumentParser(prog="bench.latency", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="unlabelled",
                    help="name this run, e.g. 'baseline' or 'after-step-1'")
    ap.add_argument("--n", type=int, default=10, help="samples per probe (default 10)")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded samples before each probe (default 1)")
    ap.add_argument("--case", default="myasthenic_crisis", help="case id to benchmark")
    ap.add_argument("--probes", default="",
                    help=f"comma-separated subset of: {', '.join(PROBES)}")
    ap.add_argument("--pure-only", action="store_true",
                    help="run only the no-network probes")
    ap.add_argument("--out", type=Path, default=None, help="explicit output path")
    ap.add_argument("--tts-model", default="",
                    help="ElevenLabs TTS model for this run (sets ELEVENLABS_TTS_MODEL), "
                         "e.g. eleven_flash_v2_5 to compare against the v4 Turbo default")
    args = ap.parse_args(argv)
    if args.tts_model:
        import os
        os.environ["ELEVENLABS_TTS_MODEL"] = args.tts_model

    if args.pure_only:
        names = list(PURE_PROBES)
    elif args.probes:
        names = [p.strip() for p in args.probes.split(",") if p.strip()]
        unknown = [p for p in names if p not in PROBES]
        if unknown:
            ap.error(f"unknown probe(s): {', '.join(unknown)}")
    else:
        names = list(PROBES)

    print(f"\nlatency bench — label={args.label} n={args.n} warmup={args.warmup}\n")
    payload = run(names, args.case, args.n, args.warmup, args.label)

    region = payload["elevenlabs_region"]
    if "x-region" in region:
        print(f"\n  elevenlabs x-region: {region['x-region']}")
    for note in payload["notes"]:
        print(f"  note: {note}")
    if payload["git"]["dirty"]:
        print("  WARNING: tree is dirty — this run is not reproducible from its SHA")

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.out:
        path = args.out
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in args.label)
        path = RESULTS / f"{stamp}-{payload['git']['sha']}-{safe}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    print(f"compare with:  python -m bench.compare <baseline.json> {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
