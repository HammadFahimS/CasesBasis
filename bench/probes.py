"""The probes — one per leg of a voice turn.

Each probe measures a *single* stage in isolation so a change can be attributed.
The composite ``turn_total`` is deliberately kept alongside the parts: if the
parts improve and the composite does not, the win was somewhere we are not
looking, and that is worth knowing before shipping.

Tiers
-----
``pure``     no network — engine, router, dispatch. Sub-millisecond baselines that
             prove where the time is *not*. Always runnable, even in CI.
``network``  hits the real provider. Needs ``ELEVENLABS_API_KEY`` / the configured
             ``LLM_PROVIDER`` credentials. Skipped with a recorded reason when the
             key is absent, so a partial run still produces a usable report.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

FIXTURES = Path(__file__).parent / "fixtures"

# Utterances chosen to spread across the op vocabulary rather than to be easy:
# a vitals read, a high-alert med *with* a dose (the S3 park path), a sent lab,
# an EMS history beat, a bedside RT measurement, and a diagnosis attempt.
UTTERANCES: list[tuple[str, str]] = [
    ("read_vitals", "What are her vitals right now?"),
    ("order_medication", "Give her two milligrams of lorazepam IV."),
    ("order_investigation", "Let's get an ABG please."),
    ("ask_history", "What happened before she got here?"),
    ("bedside", "Can you check her negative inspiratory force?"),
    ("diagnose", "I think this is a myasthenic crisis."),
]

# The line we synthesise once to use as STT input, and re-synthesise for TTS
# timings. Long enough to be a realistic turn, short enough not to dominate.
TTS_SAMPLE = "Blood pressure is 148 over 92, heart rate 112, sats 91 percent on room air."
STT_SAMPLE_TEXT = "Give her two milligrams of lorazepam intravenously please."


@dataclass
class ProbeResult:
    name: str
    tier: str
    samples: list[float] = field(default_factory=list)   # milliseconds
    skipped: str | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from bench.stats import summarize
        out: dict[str, Any] = {
            "name": self.name, "tier": self.tier,
            "samples": [round(s, 3) for s in self.samples],
            "stats": summarize(self.samples).to_dict(),
        }
        if self.skipped:
            out["skipped"] = self.skipped
        if self.error:
            out["error"] = self.error
        if self.meta:
            out["meta"] = self.meta
        return out


@dataclass
class Context:
    """Everything the probes share: a live session and cached fixtures.

    The session is built once and reused. Probes that mutate engine state
    (dispatch, turn_total) snapshot/restore around each sample so run N+1 starts
    from the same clinical state as run N — otherwise the case deteriorates
    under the benchmark and later samples measure a different problem.
    """

    session: Any
    stt_audio: bytes | None = None
    stt_content_type: str = "audio/mpeg"
    stt_pcm: bytes | None = None          # raw pcm_16000 for the realtime path
    notes: list[str] = field(default_factory=list)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _have_elevenlabs() -> bool:
    return bool(os.environ.get("ELEVENLABS_API_KEY", "").strip())


def _have_llm() -> bool:
    """True when the configured provider has whatever credential it needs.

    Ollama is local and needs none; everything else needs a key. Mirrors
    ``agents.base`` rather than duplicating its table, so a new provider there
    does not silently make this lie.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    if provider == "ollama":
        return True
    env = {
        "openai": "OPENAI_API_KEY", "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY", "openrouter": "OPENROUTER_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
    }.get(provider)
    return bool(env and os.environ.get(env, "").strip())


def _snapshot(ctx: Context):
    return ctx.session.engine.snapshot()


def _restore(ctx: Context, snap) -> None:
    ctx.session.engine.restore(snap)


# ── pure probes ──────────────────────────────────────────────────────────────

def probe_route(ctx: Context, n: int) -> ProbeResult:
    """Router resolution only. The C2 two-axis decision, no LLM, no engine write."""
    from backend.nurse_brain import NurseIntent
    from backend.routing import route

    res = ProbeResult("route", "pure")
    intents = [
        NurseIntent(op="read_vitals"),
        NurseIntent(op="ask_history", target=None),
        NurseIntent(op="order_medication", action_text="lorazepam 2mg"),
    ]
    for i in range(n):
        intent = intents[i % len(intents)]
        transcript = UTTERANCES[i % len(UTTERANCES)][1]
        t0 = time.perf_counter()
        route(intent, ctx.session.engine, transcript)
        res.samples.append(_ms(t0))
    return res


def probe_dispatch(ctx: Context, n: int) -> ProbeResult:
    """Deterministic engine dispatch. This is the clinical core, and it should be
    invisible in the budget — if it is not, that is the finding."""
    from backend.nurse_brain import NurseIntent, dispatch

    res = ProbeResult("dispatch", "pure")
    snap = _snapshot(ctx)
    try:
        for _ in range(n):
            intent = NurseIntent(op="read_vitals")
            t0 = time.perf_counter()
            dispatch(intent, ctx.session.engine, "What are her vitals?")
            res.samples.append(_ms(t0))
    finally:
        _restore(ctx, snap)
    return res


def probe_engine_tick(ctx: Context, n: int) -> ProbeResult:
    """One clock advance — what the SessionRuntime pays every 2 s, per session."""
    from backend.voice_session import advance_clock

    res = ProbeResult("engine_tick", "pure")
    snap = _snapshot(ctx)
    try:
        for _ in range(n):
            t0 = time.perf_counter()
            advance_clock(ctx.session)
            res.samples.append(_ms(t0))
    finally:
        _restore(ctx, snap)
    return res


# ── network probes ───────────────────────────────────────────────────────────

def probe_llm_interpret(ctx: Context, n: int) -> ProbeResult:
    """The first of the two serial LLM calls: utterance → structured intent.

    This one is on the hot path *and* correctness-critical — a misclassified op
    is a wrong clinical action, not just a slow one.
    """
    from backend.nurse_brain import interpret

    res = ProbeResult("llm_interpret", "network")
    if not _have_llm():
        res.skipped = f"no credential for LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'ollama')}"
        return res
    for i in range(n):
        transcript = UTTERANCES[i % len(UTTERANCES)][1]
        t0 = time.perf_counter()
        try:
            interpret(transcript, ctx.session.engine, ctx.session.history)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            break
        res.samples.append(_ms(t0))
    return res


def probe_llm_speak(ctx: Context, n: int) -> ProbeResult:
    """The second serial LLM call: engine outcome → spoken phrasing.

    Step 1 of the latency plan proposes deleting this for deterministic ops.
    That change is only defensible against a recorded number, which is this one.
    """
    from backend.nurse_brain import NurseIntent, speak

    res = ProbeResult("llm_speak", "network")
    if not _have_llm():
        res.skipped = f"no credential for LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'ollama')}"
        return res
    engine = ctx.session.engine
    vitals = engine.current_vitals().model_dump(exclude_none=True)
    for i in range(n):
        transcript = UTTERANCES[i % len(UTTERANCES)][1]
        intent = NurseIntent(op="read_vitals")
        t0 = time.perf_counter()
        try:
            speak(transcript, intent, None, vitals, ctx.session.history, engine=engine)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            break
        res.samples.append(_ms(t0))
    return res


def probe_turn_total(ctx: Context, n: int) -> ProbeResult:
    """The whole grounded turn: advance → interpret → route → dispatch → voice.

    Kept beside its parts on purpose. If ``llm_interpret`` and ``llm_speak``
    both fall and this does not, the saving went somewhere else.
    """
    from backend.nurse_brain import run_turn

    res = ProbeResult("turn_total", "network")
    if not _have_llm():
        res.skipped = f"no credential for LLM_PROVIDER={os.getenv('LLM_PROVIDER', 'ollama')}"
        return res
    for i in range(n):
        transcript = UTTERANCES[i % len(UTTERANCES)][1]
        snap = _snapshot(ctx)
        t0 = time.perf_counter()
        try:
            turn = run_turn(ctx.session, transcript)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            _restore(ctx, snap)
            break
        res.samples.append(_ms(t0))
        res.meta.setdefault("ops", []).append(turn.intent.op)
        _restore(ctx, snap)
    return res


def probe_tts_ttfb(ctx: Context, n: int) -> ProbeResult:
    """Time to the **first PCM chunk** — when the resident first hears anything.

    The number that matters for perceived latency. ``_el_tts_stream`` is a
    generator, so the request does not leave until the first ``next()``; timing
    around that call is honest.
    """
    from backend.voice_service import synthesize_stream

    res = ProbeResult("tts_ttfb", "network")
    if not _have_elevenlabs():
        res.skipped = "ELEVENLABS_API_KEY not set"
        return res
    total: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            _, gen = synthesize_stream(TTS_SAMPLE)
            next(gen)
            res.samples.append(_ms(t0))
            nbytes = 0
            for chunk in gen:
                nbytes += len(chunk)
            total.append(_ms(t0))
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            break
    if total:
        from bench.stats import percentile
        res.meta["full_stream_p50_ms"] = round(percentile(total, 0.5), 2)
    return res


def probe_tts_ttfb_cold(ctx: Context, n: int) -> ProbeResult:
    """TTFB with the connection pool deliberately discarded before each sample.

    Isolates what commit 9fb71c0 bought. The gap between this and ``tts_ttfb``
    is the TLS handshake you avoid by keeping the client warm — and it is also
    what every resident pays on their first turn.
    """
    import backend.voice_service as vs

    res = ProbeResult("tts_ttfb_cold", "network")
    if not _have_elevenlabs():
        res.skipped = "ELEVENLABS_API_KEY not set"
        return res
    for _ in range(n):
        try:
            if vs._client is not None:
                vs._client.close()
        except Exception:
            pass
        vs._client = None
        t0 = time.perf_counter()
        try:
            _, gen = vs.synthesize_stream(TTS_SAMPLE)
            next(gen)
            res.samples.append(_ms(t0))
            for _chunk in gen:
                pass
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            break
    return res


def probe_stt_blocking(ctx: Context, n: int) -> ProbeResult:
    """The blocking Scribe v1 upload path — audio blob in, transcript out.

    Uses a cached fixture synthesised on first run (see :func:`ensure_stt_fixture`)
    so every run transcribes byte-identical audio.
    """
    from backend.voice_service import transcribe

    res = ProbeResult("stt_blocking", "network")
    if not _have_elevenlabs():
        res.skipped = "ELEVENLABS_API_KEY not set"
        return res
    if not ctx.stt_audio:
        res.skipped = "no STT fixture (synthesis failed)"
        return res
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            text = transcribe(ctx.stt_audio, ctx.stt_content_type)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            break
        res.samples.append(_ms(t0))
        res.meta["last_transcript"] = text
    res.meta["audio_bytes"] = len(ctx.stt_audio)
    return res


PCM_RATE = 16000
PCM_BYTES_PER_SEC = PCM_RATE * 2          # 16-bit mono
_STREAM_CHUNK_MS = 100                    # mirrors the browser's ScriptProcessor cadence


async def _realtime_tail_ms(pcm: bytes, keyterms: list[str]) -> tuple[float, str]:
    """Stream ``pcm`` at real-time pace, commit, and time the tail.

    The pacing is the whole point. Dumping the audio instantly and committing
    would measure ElevenLabs processing a cold 4-second buffer — the opposite of
    how the browser uses this socket. Streaming at 1x means the recogniser has
    already consumed almost everything by the time the resident stops, so what
    we time from ``commit`` is exactly what they wait through: the tail.

    That makes this directly comparable to ``stt_blocking``, which also starts
    its clock at end-of-speech.
    """
    import asyncio
    import base64
    import json

    from backend.voice_service import open_realtime_stt

    ws = await open_realtime_stt(keyterms)
    try:
        step = PCM_BYTES_PER_SEC * _STREAM_CHUNK_MS // 1000
        for off in range(0, len(pcm), step):
            await ws.send(json.dumps({
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm[off:off + step]).decode("ascii"),
            }))
            await asyncio.sleep(_STREAM_CHUNK_MS / 1000.0)

        t0 = time.perf_counter()
        await ws.send(json.dumps({
            "message_type": "input_audio_chunk", "audio_base_64": "", "commit": True,
        }))
        # Drain buffered partials until the committed transcript lands.
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if data.get("message_type") in (
                "committed_transcript", "committed_transcript_with_timestamps",
                "final_transcript", "final_transcript_with_timestamps",
            ):
                return _ms(t0), (data.get("text") or "").strip()
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def probe_stt_realtime(ctx: Context, n: int) -> ProbeResult:
    """Scribe v2 Realtime: time from end-of-speech to committed transcript.

    The counterpart to ``stt_blocking``. Both clocks start when the resident
    stops talking, so the two p50s are directly comparable — which is the only
    reason we can claim the streaming path is an improvement rather than a
    differently-shaped cost.
    """
    import asyncio

    res = ProbeResult("stt_realtime", "network")
    if not _have_elevenlabs():
        res.skipped = "ELEVENLABS_API_KEY not set"
        return res
    if not ctx.stt_pcm:
        res.skipped = "no PCM fixture (synthesis failed)"
        return res

    keyterms: list[str] = []
    try:
        from backend.routers.voice import _stt_prompt
        keyterms = [t.strip() for t in _stt_prompt(ctx.session.case).split(",") if t.strip()][:32]
    except Exception:
        pass

    for _ in range(n):
        try:
            ms, text = asyncio.run(_realtime_tail_ms(ctx.stt_pcm, keyterms))
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            break
        res.samples.append(ms)
        res.meta["last_transcript"] = text
    res.meta["audio_seconds"] = round(len(ctx.stt_pcm) / PCM_BYTES_PER_SEC, 2)
    res.meta["keyterms"] = len(keyterms)
    return res


# ── registry ─────────────────────────────────────────────────────────────────

PROBES: dict[str, Callable[[Context, int], ProbeResult]] = {
    "route": probe_route,
    "dispatch": probe_dispatch,
    "engine_tick": probe_engine_tick,
    "llm_interpret": probe_llm_interpret,
    "llm_speak": probe_llm_speak,
    "turn_total": probe_turn_total,
    "tts_ttfb": probe_tts_ttfb,
    "tts_ttfb_cold": probe_tts_ttfb_cold,
    "stt_blocking": probe_stt_blocking,
    "stt_realtime": probe_stt_realtime,
}

PURE_PROBES = ("route", "dispatch", "engine_tick")


def ensure_stt_fixture() -> tuple[bytes | None, str, str | None]:
    """Return cached STT input audio, synthesising it once if absent.

    Bootstrapping the fixture from our own TTS keeps a binary out of git while
    guaranteeing every run measures the same bytes. Returns
    ``(audio, content_type, note)``.
    """
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / "stt_sample.mp3"
    if path.exists():
        return path.read_bytes(), "audio/mpeg", None
    if not _have_elevenlabs():
        return None, "audio/mpeg", "cannot synthesise STT fixture without ELEVENLABS_API_KEY"
    try:
        from backend.voice_service import synthesize
        audio = synthesize(STT_SAMPLE_TEXT, fmt="mp3")
    except Exception as exc:
        return None, "audio/mpeg", f"STT fixture synthesis failed: {exc}"
    path.write_bytes(audio)
    return audio, "audio/mpeg", f"synthesised STT fixture → {path}"


def ensure_stt_pcm_fixture() -> tuple[bytes | None, str | None]:
    """Raw ``pcm_16000`` of the same sentence, for the realtime socket.

    Synthesised directly in PCM rather than decoded from the MP3 fixture — no
    decoder dependency, and the realtime path gets exactly the format the
    browser sends it.
    """
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / "stt_sample_16k.pcm"
    if path.exists():
        return path.read_bytes(), None
    if not _have_elevenlabs():
        return None, "cannot synthesise PCM fixture without ELEVENLABS_API_KEY"
    try:
        from backend.voice_service import synthesize
        pcm = synthesize(STT_SAMPLE_TEXT, fmt="pcm_16000")
    except Exception as exc:
        return None, f"PCM fixture synthesis failed: {exc}"
    path.write_bytes(pcm)
    return pcm, f"synthesised PCM fixture → {path}"


def capture_region() -> dict[str, Any]:
    """Record which ElevenLabs edge we are hitting.

    The latency guide puts NA/EU at 100-150 ms TTFB and S/NE Asia at 150-200 ms,
    and says to check ``x-region``. Cheap to capture, and it decides whether
    pinning ``api.us.elevenlabs.io`` is worth doing at all.
    """
    if not _have_elevenlabs():
        return {"skipped": "ELEVENLABS_API_KEY not set"}
    try:
        import backend.voice_service as vs
        resp = vs._http().get(
            f"{vs.ELEVENLABS_BASE_URL}/user/subscription",
            headers={"xi-api-key": vs._elevenlabs_key()},
        )
        return {
            "x-region": resp.headers.get("x-region", "(absent)"),
            "status": resp.status_code,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
