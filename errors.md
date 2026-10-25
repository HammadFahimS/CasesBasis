# errors.md — Live Sim Regression Log

Hand this to Claude Opus 4.8. It documents every change made to the live simulation in
the last two sessions, what was broken before, what was attempted, and what is still wrong.

---

> **Resolved 2026-10-25 — "reply shown, never heard" (the long-running silent-TTS bug).**
> Root cause: `/api/voice/tts` sniffed the first two bytes of the PCM stream for an MP3
> frame sync (`FF Ex`, added in 1b9e81a). PCM-16 whose first sample is -1 starts `FF FF`,
> so ~1 line in 6 was served as `audio/mpeg`; `<audio>` cannot decode raw PCM
> (`MEDIA_ERR_SRC_NOT_SUPPORTED`) and `_playBlob` resolved as if the line had played —
> no retry, no message. Repeating the question re-synthesised with a different first
> sample, which is why it "fixed itself". Fixed by serving the PCM the request pins
> (`output_format` in the query string) and making an `<audio>` failure reject so the
> client's retry + failure report run. The server's stream-retry and non-streaming MP3
> fallbacks (aimed at the same symptom; never on its path) were removed: one retry, at the
> client, where it is logged. Tests: `tests/test_tts_route.py`, `tests/client/audio_harness.js` §6.
> Change 7 below (first-chunk probe) is kept; its non-streaming fallback is gone.

## Baseline commit (what worked before any of this)

```
4b10d4e  fix: patient channel grounding — symptom questions now reach authored beats
```

All changes below were made on top of this commit. Nothing has been committed yet.

---

## Session 1 changes (previous context window — committed in 73148d5)

### Change 1 — VAD high-pass filter (`backend/static/voice_room.html`, `startServer()`)

**What changed:**
```diff
-      micSource.connect(analyser);
+      const vadHp = audioCtx.createBiquadFilter();
+      vadHp.type = 'highpass'; vadHp.frequency.value = 250; vadHp.Q.value = 0.7;
+      micSource.connect(vadHp);
+      vadHp.connect(analyser);
```

**Intent:** Cut sub-250 Hz mechanical noise so table bumps don't trip the RMS voice-activity detector.

**Status:** Committed in 73148d5. No regression observed from this change alone — the analyser is
never connected to the destination so no audio output changes. However, the filter is an extra node
that could introduce an initial transient reading on the analyser before the IIR settles, potentially
causing one false VAD trigger right after mic permission is granted.

---

## Session 2 changes (current session — NOT committed, `git diff HEAD`)

### Change 2 — Drawer CSS (`backend/static/voice_room.html`, `.drawer` rule)

**What changed:**
```diff
- position: fixed; top: 50%; left: 0; transform: translateY(-50%);
- z-index: 6; display: flex; align-items: stretch; max-height: 88vh;
+ position: fixed; top: 72px; bottom: 260px; left: 0;
+ z-index: 6; display: flex; align-items: stretch;

- .drawer.collapsed { transform: translate(calc(-100% + 38px), -50%); }
+ .drawer.collapsed { transform: translateX(calc(-100% + 38px)); }
```

**Intent:** Stop the drawer from overlapping the bottom controls area (talk button, vitals strip).

**Known regressions visible in screenshot:**
- The drawer is now shorter and top-anchored, which feels wrong compared to the original centred design.
- On very tall screens the bottom of the drawer stops at `260px` from the bottom regardless of content.
- The `.dr-panel` content may feel cramped on shorter screens.
- Original centred design was `top: 50%; transform: translateY(-50%); max-height: 88vh`. The collapsed
  state also used `transform: translate(calc(-100% + 38px), -50%)` (two axes); removing the Y axis
  may affect the animation or snap position.

**Suggested revert or alternative:**
Restore the original CSS. Instead of changing the drawer dimensions, constrain the vitals/controls
`z-index` so the drawer slides behind them when open, or reduce `max-height` using a CSS clamp that
accounts for the controls area. The original design was intentional.

---

### Change 3 — `_beep` reuses shared audioCtx (`backend/static/voice_room.html`)

**What changed:**
```diff
- const ctx = new (window.AudioContext || window.webkitAudioContext)();
+ const ctx = (audioCtx && audioCtx.state !== 'closed') ? audioCtx
+           : new (window.AudioContext || window.webkitAudioContext)();
+ const ownCtx = ctx !== audioCtx;
  // ...
- setTimeout(() => { try { ctx.close(); } catch(e){} }, (dur + 0.1) * 1000);
+ if (ownCtx) setTimeout(() => { try { ctx.close(); } catch(e){} }, (dur + 0.1) * 1000);
```

**Intent:** Avoid creating a new AudioContext on every alarm tick (was creating one every 4 s).

**Risk:** The alarm beep oscillator is now connected directly to `audioCtx.destination`, which is
the same destination as the ambience gain node and the PCM TTS playback. If `audioCtx` is busy
playing TTS at the same time an alarm fires, both will play through the same context — which is
correct, but the alarm will not respect the ambience mute toggle. Minor concern.

---

### Change 4 — `_vitalsReady` alarm guard (`backend/static/voice_room.html`)

**What changed:**
```diff
+ let _alarmTimer = null, _vitalsReady = false;

  function tickAlarm(anyWarn, anyCrit){
+   if (!_vitalsReady) return;
    // ...
  }

  function applyVitals(v){
    if (!v) return;
+   _vitalsReady = true;
```

**Intent:** Prevent the alarm from firing at page load (before the session starts) because the
placeholder vitals `spo2 = 92` immediately triggers `spo2Warn = true`.

**Status:** Correct logic. No regressions expected from this change.

---

### Change 5 — `_playPCMStream` empty-stream detection (`backend/static/voice_room.html`)

**What changed:**
```diff
- await new Promise(async (resolve) => {
+ let fetchAborted = false;
+ await new Promise(async (resolve, reject) => {
    // ...
    } catch(e) {
-     if (e.name !== 'AbortError') console.warn('[pcm stream]', e);
+     if (e.name === 'AbortError') fetchAborted = true;
+     else console.warn('[pcm stream]', e);
    }
    if (lastSrc) {
      // ...
    } else {
-     if (playResolve){ playResolve = null; resolve(); }
+     if (playResolve){ playResolve = null; }
+     if (fetchAborted) resolve();
+     else reject(new Error('voice service unavailable'));
    }
```

**Intent:** When ElevenLabs returns an empty PCM stream (auth failure, bad voice ID, quota), the
old code silently resolved the Promise so the caller never knew TTS failed. The new code rejects
so `playHandoff` can show an error.

**Risk:** This changes the behaviour of `stopAudio()`. `stopAudio` calls `playResolve()` (which is
`resolve`). If the rejection path has already run, `resolve()` is a no-op (Promise settled), so
`stopAudio` is safe. BUT: if there is any timing edge where `stopAudio()` fires first and sets
`playResolve = null`, then the rejection path sees `playResolve === null` and calls `reject()` with
no guard — the unhandled rejection is caught by `playTTS`'s `catch(e)` and re-thrown. This may or
may not cause problems depending on timing. Original behaviour was to always resolve silently.

---

### Change 6 — `playHandoff` error display (`backend/static/voice_room.html`)

**What changed:**
```diff
- try { await playTTS(INTRO_TEXT, 'nurse'); } catch(e){}
+ try { await playTTS(INTRO_TEXT, 'nurse'); }
+ catch(e){
+   setNow(NURSE_CH, e.message || 'voice service unavailable');
+   NURSE_CH.root.classList.remove('collapsed');
+ }
```

**Intent:** Show a visible error in the transcript when the nurse handoff TTS fails instead of
silently doing nothing.

**Status:** Correct. No regressions expected. The nurse transcript already shows the typed handoff
text, so showing an error below it is additive.

---

### Change 7 — ElevenLabs first-chunk probe (`backend/routers/voice.py`, `api_voice_tts`)

**What changed:**
```diff
  media_type, chunks = synthesize_stream(text, voice, ...)

  if media_type == "audio/pcm":
+     try:
+         first_chunk: bytes = next(chunks)
+     except StopIteration:
+         first_chunk = b""
+     except Exception as exc:
+         logger.warning("ElevenLabs TTS failed at stream start: %s", exc)
+         first_chunk = b""
+
+     if not first_chunk:
+         # Fall back to Together AI MP3
+         audio = synthesize(text, voice, ...)
+         return Response(content=audio, media_type="audio/mpeg")
+
+     logger.info("[LATENCY] tts_first=...")
+
      def _pcm_gen():
-         first = True
          try:
+             yield first_chunk
              for chunk in chunks:
-                 if first:
-                     first = False
-                     logger.info("[LATENCY]...")
                  yield chunk
```

**Intent:** If ElevenLabs returns no bytes (wrong API key, quota, voice ID error), fall back to
Together AI MP3 instead of sending an empty `audio/pcm` body.

**Status:** Working. Server logs confirm `POST /api/voice/tts 200 OK` with audio playing.

**Risk:** `next(chunks)` advances the ElevenLabs generator; the `httpx.stream()` connection is held
open by the generator's frame while we probe. This is fine as long as `chunks` is iterated to
completion (or exhausted). The `_pcm_gen` correctly yields `first_chunk` first then continues the
generator. BUT: if ElevenLabs succeeds but the first chunk is somehow empty bytes (e.g. a metadata
header), `if not first_chunk` falls through to Together AI fallback unnecessarily. This seems
unlikely with ElevenLabs PCM but worth noting.

---

### Change 8 — STT retry (`backend/voice_service.py`, `transcribe()`)

**What changed:**
```diff
+ import time
  # ...
+ last_exc: Exception | None = None
+ for attempt in range(2):
+     if attempt:
+         time.sleep(1.5)
      try:
          resp = httpx.post(...)
          resp.raise_for_status()
          return (resp.json().get("text") or "").strip()
+     except httpx.HTTPStatusError as exc:
+         last_exc = exc
+         if status < 500:
+             raise   # 4xx not transient
+         logger.warning(...)
+     except httpx.RequestError as exc:
+         last_exc = exc
+         logger.warning(...)
+ raise last_exc
```

**Intent:** Together AI's transcription endpoint was returning 503. Retry once after 1.5 s.

**Status:** Correct. The 503 is a Together AI service outage — entirely external. The retry adds
1.5 s latency per utterance when Together is degraded. No code regressions.

---

### Change 9 — STT friendly error message (`backend/static/voice_room.html`, `stt()`)

**What changed:**
```diff
- if (!res.ok) throw new Error(data.detail || 'transcription failed');
+ if (!res.ok){
+   const raw = data.detail || 'transcription failed';
+   const msg = /503|service unavailable/i.test(raw) ? 'speech-to-text service temporarily unavailable...'
+             : /502|bad gateway/i.test(raw)         ? 'speech-to-text service unreachable'
+             : raw;
+   throw new Error(msg);
+ }
```

**Intent:** Show a readable error instead of the raw httpx exception string.

---

## Issues visible in the attached screenshot

1. **Drawer height/position looks wrong.** The drawer now spans from `top: 72px` to `bottom: 260px`
   instead of being vertically centred. On the screenshot it appears noticeably shorter/different
   from the original design. The EXAM checkboxes are visible but the drawer feels cramped and
   anchored oddly relative to the header.

2. **Drawer collapsed animation may be broken.** The original collapsed transform was
   `translate(calc(-100% + 38px), -50%)` (translating on both X and Y axes because the drawer was
   `top: 50%`). It was changed to `translateX(calc(-100% + 38px))` only. With `top: 72px` the
   drawer is no longer `top: 50%`, so the Y component was correct to remove — but this should be
   verified by actually collapsing/opening the drawer.

3. **Vitals strip may be clipped.** In the screenshot the vitals strip at the very bottom shows
   `104 bpm | 123/83 mmHg | 35 /min | 100 %` with a "Processing…" state indicator visible. These
   appear intact but crowded. Confirm the vitals strip is not overlapping the address chips above it
   or being clipped by the viewport.

4. **SpO2 showing 100% (stable).** The baseline SpO2 for the myasthenic crisis case is 99%, so 100%
   with cosmetic jitter is plausible. However in the previous screenshot it showed 77% (critical),
   which was anomalous. No root cause was identified for that anomaly — it may have been a stale/
   resumed session. Worth confirming the case loads fresh vitals correctly.

5. **Together AI STT returning 503.** External service outage. A retry with 1.5 s delay was added
   (Change 8). If Together AI continues to be degraded, the only workaround is switching the user
   to the Native speech backend (SpeechRecognition API, zero cost) while Together recovers.

---

## Recommended approach for Claude Opus 4.8

1. **Revert Change 2 (drawer CSS)** entirely. The original `top: 50%; transform: translateY(-50%);
   max-height: 88vh` is fine visually. The actual overlap complaint was that the open drawer covers
   the vitals strip horizontally (the drawer is 36vw wide and the vitals are centred). The right fix
   is either:
   - Increase `z-index` of the vitals strip to be above the drawer, OR
   - Offset the vitals strip left-margin when the drawer is open (JS class toggle), OR
   - Simply accept the overlap — the drawer is user-opened and the user can collapse it.
   DO NOT try to fix it by changing the drawer's height/position.

2. **Keep Changes 3–9.** They are correct and address real bugs (alarm before session, silent TTS
   failure, ElevenLabs fallback, STT retry).

3. **Investigate the "fuzzing" sound further.** Even with Changes 3 and 4, if `audioCtx` is
   created in `startServer()` and then `_beep` reuses it, the alarm oscillator connects directly
   to `audioCtx.destination` (same graph as ambience). Check if the ambience `_ambienceGain` node
   (master gain 0.85) is summing with the alarm oscillator's direct connection, causing unexpected
   amplitude. The ambience beep at 880 Hz fires every `60/HR * 1000` ms — at HR=104 that is
   every 577 ms. Combined with the alarm 1047 Hz double-beep every 4 s, this may sound busy/loud.
   Consider lowering `noiseGain.gain.value` from 0.28 to 0.08 and the ambience beep peak from 0.06
   to 0.03.

4. **The VAD filter (Change 1)** should be tested. The `vadHp` highpass filter IIR state starts at
   zero and takes several samples to settle — this could cause a brief spike in the RMS reading
   right after mic permission is granted, triggering a false-positive VAD start. Consider adding a
   50–100 ms warm-up delay before starting the VAD loop after `_initAmbience`.
