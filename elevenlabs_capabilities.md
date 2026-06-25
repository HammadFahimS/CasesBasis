# ElevenLabs — Capabilities & How They Map to CasesBasis

> Research note for the CasesBasis voice sim. We hold an **ElevenLabs Startup Grant**
> (credits on `hammad.fahim57@gmail.com`; key in `.env` as `ELEVENLABS_API_KEY`).
> This documents what ElevenLabs offers and, for each piece, **what it means for our app**.
> Verified against the live docs in **June 2026** — re-check model IDs/prices before relying on them.

---

## 0 · TL;DR for our use case

CasesBasis runs a **real-time, multi-voice ED simulation**: the resident speaks; a cast
(patient, nurse, EMS, family, RT, consultant) speaks back; an engine drives the case on a
wall-clock. Today our voice stack is **Together (Cartesia Sonic-2) for TTS** + **Together Whisper
(or Deepgram) for STT**, behind a single `voice_service.synthesize(text, voice, fmt)` seam, with a
per-speaker voice map in `backend/actors.py`.

ElevenLabs is relevant to us in **three escalating tiers**:

1. **Drop-in TTS upgrade** — swap `synthesize()` to ElevenLabs for noticeably more lifelike,
   emotionally-controllable voices per actor. **Lowest effort, highest immediate payoff.** Our
   architecture (one TTS seam + a speaker→voice map) is already shaped for this.
2. **STT upgrade (Scribe)** — replace Whisper/Deepgram transcription; `scribe_v2_realtime` (~150ms)
   for live capture, diarization + word timestamps.
3. **ElevenAgents (Conversational AI)** — a full STT→LLM→TTS turn-taking agent runtime. This is the
   product the grant email pushed. It could power the live room, **but** it's architecturally a
   big change and partly **competes with our engine** (see §7 — we'd keep the engine as the source
   of clinical truth and use Agents only as a voice I/O layer, if at all).

**Recommendation:** start with **Tier 1** (TTS swap, expressive voices per actor), evaluate **Scribe**
for STT, and treat **ElevenAgents as a research spike** — our `SimulationEngine` must stay the single
source of truth, which constrains how much of Agents we can adopt.

---

## 1 · Models

### Text-to-Speech

| Model | `model_id` | Latency | Max chars/req | Languages | Notes |
|---|---|---|---|---|---|
| **Eleven v3** (expressive) | `eleven_v3` | not real-time-rated | 5,000 (~5 min) | 70+ | Most expressive; **audio tags** (`[whispers]`, `[sighs]`), multi-speaker dialogue, emotional range. Best **quality**, not for tight latency. |
| **Multilingual v2** | `eleven_multilingual_v2` | — | 10,000 | 29 | High quality, stable on long-form, emotionally aware. API default. |
| **Flash v2.5** | `eleven_flash_v2_5` | **~75 ms** | 40,000 | 32 | **Ultra-low latency**, ~50% cheaper/char. The conversational/real-time pick. May fumble complex number normalization. |
| **Flash v2** | `eleven_flash_v2` | ~75 ms | 30,000 | English only | Real-time, English-only. |
| Turbo v2.5 | `eleven_turbo_v2_5` | — | — | 32 | **Deprecated** — use Flash v2.5 instead. |

### Speech-to-Text (Scribe)

| Model | `model_id` | Latency | Languages | Notes |
|---|---|---|---|---|
| **Scribe v2** | `scribe_v2` | batch | 90+ | Word-level timestamps, **diarization (≤32 speakers)**, entity detection, keyterm prompting (≤1000 terms — great for drug names), dynamic audio tagging. |
| **Scribe v2 Realtime** | `scribe_v2_realtime` | **~150 ms** | 90+ | Streaming, VAD, manual commit control. **The live-room STT pick.** |
| Scribe v1 | `scribe_v1` | — | — | **Deprecated — migrate before 9 Jul 2026.** |

> **For us:** the live room needs **low latency** → **Flash v2.5** for TTS and **Scribe v2 Realtime**
> for STT. Reserve **Eleven v3** for non-realtime moments where expressiveness matters most — e.g. the
> **nurse's opening handoff**, the **debrief coaching note**, or pre-rendered cues.

---

## 2 · Text-to-Speech API (the seam we'd plug into)

**Endpoint:** `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}`
Auth header: `xi-api-key: <ELEVENLABS_API_KEY>`. Returns binary audio.
Streaming variant: `.../stream`; real-time **WebSocket** endpoint exists for token-by-token synthesis.

**`voice_settings`** (the per-actor expressiveness knobs):
- `stability` (0–1, def 0.5) — lower = more emotional variation/range; higher = flatter, consistent.
- `similarity_boost` (0–1, def 0.75) — adherence to the source voice.
- `style` (0+, def 0) — style exaggeration; **non-zero adds latency**.
- `use_speaker_boost` (bool, def true) — similarity boost; slight latency cost.
- `speed` (0.7–1.2, def 1.0) — pacing; extremes degrade quality.

**Output formats:** `mp3_*` (e.g. `mp3_44100_128`), `pcm_*`, `wav`, `opus`, `ulaw`, `alaw`.
(We currently request `mp3`. For phone-grade/streaming, μ-law/PCM are available.)

**Latency controls:** `optimize_streaming_latency` (0–4; 4 = max speed, disables normalization).

**Continuity / stitching:** `previous_text`/`next_text` and `previous_request_ids`/`next_request_ids`
(≤3) keep prosody natural across consecutive generations — relevant if we ever chunk a long line.

**Determinism:** `seed` (0–4294967295) for repeatable audio (useful for tests/snapshots).

> **Maps cleanly to our code:** `backend/voice_service.py::_tts_request()` is a single `httpx.post` to
> Together/Cartesia. An ElevenLabs path is the same shape — swap URL, auth header, and body. The
> `synthesize(text, voice, fmt)` signature stays; only `voice` becomes an ElevenLabs **`voice_id`**
> instead of a Cartesia name.

---

## 3 · Voices — giving each actor a distinct identity

This is the feature most aligned with our **multi-actor** design (`backend/actors.py` already maps
`nurse / patient / ems / family / rt / consult` → a voice).

- **Voice Library** — 10,000+ community voices (paid API). Pick a fitting `voice_id` per actor.
- **Voice Design** — generate a brand-new voice from a **text prompt** (age, gender, accent, tone;
  20–1000 chars), returns 3 previews. *Ideal for us:* describe each role
  ("anxious breathless 29-year-old woman", "calm senior neurologist", "brisk paramedic") and mint a
  bespoke `voice_id` per actor/case.
- **Instant Voice Cloning** — from <2 min of audio (most tiers).
- **Professional Voice Cloning** — higher fidelity, needs more audio + Creator plan.

> **Action item:** replace the Cartesia voice-name strings in `actors.py::SPEAKERS`
> (`TOGETHER_*_TTS_VOICE` env vars) with ElevenLabs `voice_id`s — designed once via Voice Design,
> stored in `.env` as `ELEVENLABS_*_VOICE_ID`. Keep the registry as the single source of truth.

---

## 4 · Expressiveness — emotion & delivery control

Our `actor_voice.py` / `patient_voice.py` already coach tone via prompt ("breathless", "say it
aloud naturally"). ElevenLabs gives two extra levers:

- **Eleven v3 audio tags** (inline): `[whispers]`, `[sighs]`, `[laughs]`, `[excited]`, `[sarcastic]`,
  `[curious]`, `[nervous]`, plus SFX (`[applause]`) and experimental (`[strong French accent]`).
  Effectiveness depends on the chosen voice's range. **Great fit for the patient** ("[breathless]
  I... can't catch my breath") and **emotional cues**.
- **Voice settings** per actor — e.g. patient at **low stability** (more affect), consultant at
  **higher stability** (measured, consistent).
- **No SSML breaks in v3** — use `...` for hesitation; v2 supports `<break time="1.0s"/>` (≤3s).

> **Caveat:** audio tags are a **v3** feature, and v3 isn't the low-latency model. So for the live
> room (Flash v2.5) we'd lean on `voice_settings` + punctuation; reserve tag-driven drama for
> pre-rendered/non-realtime lines.

---

## 5 · Speech-to-Text (Scribe) — for the resident's mic

Currently we transcribe with Together Whisper (`/api/voice/stt` → `voice_service.transcribe`).
Scribe upgrades:

- **`scribe_v2_realtime` (~150ms)** — streaming transcription with VAD; lets us show live partials
  and cut the resident's perceived turn latency.
- **Keyterm prompting (≤1000 terms)** — **bias the recognizer toward clinical vocabulary**
  (drug names: "rocuronium", "pyridostigmine", "succinylcholine"; "NIF", "vital capacity"). Directly
  improves our intent-matching accuracy, which feeds `nurse_brain.interpret`.
- **Diarization + word timestamps** — less critical for single-resident mic, but useful for the
  Room log / debrief transcript.

---

## 6 · Pricing & the grant

- **Credit model:** 1 char = 1 credit for v2/multilingual; **Flash/Turbo ~0.5–1 credit/char** (cheaper).
- **Plans:** Creator ~$22/mo = 100k chars + up to 30 custom voices + cloning. (Higher tiers scale chars.)
- **ElevenAgents:** **~$0.08/min** of conversation (reduced May 2026; PAYG available).
- **Our grant** covers credits on the account — so experimentation cost is effectively sponsored.
  Still, **Flash v2.5** keeps burn low for the chatty live room.

> A typical sim turn is a 1–3 sentence spoken line (~100–300 chars). At Flash rates that's ~50–300
> credits/line — cheap. The debrief coaching note (v3, a paragraph) is the priciest single call.

---

## 7 · ElevenAgents (Conversational AI) — the big option, with a caveat

**What it is:** a managed runtime that wires **STT → LLM → TTS** with a **proprietary turn-taking /
interruption** model, so you get a natural full-duplex voice agent without orchestrating the loop.

- **Bring-your-own LLM** ("Choose from supported LLMs or bring your own custom model") + **system
  prompt** + **Knowledge Base / RAG** + **server & client tools (function calling)**.
- **Access:** WebSocket API, **Python/JS SDKs**, native Swift/Kotlin/React Native; deploy to web
  widget, mobile, or **telephony (Twilio/SIP)**.
- **Built-in:** interruption handling, timeouts, monitoring dashboard, evals.

**Why it's tempting for us:** instant barge-in/interruptions, lower-latency full-duplex feel, and the
tool-calling hook could invoke our engine ops.

**Why it's NOT a drop-in for CasesBasis (important):**
- Our **`SimulationEngine` is the single source of truth** for every clinical fact, the wall-clock,
  gated results, state transitions, and scoring. Agents wants to own the conversation; we can't let
  it invent clinical content.
- The platform is **optimized for a single agent**, whereas our defining feature is **multiple
  distinct actors with addressee routing** (patient vs nurse vs EMS…). Multi-character would mean
  either one agent that voice-switches, or multiple agents we orchestrate — both non-trivial.
- Our **turn pipeline** (`nurse_brain.run_turn`: advance_clock → interpret → dispatch → speak) and
  **cards/gating/cue engine** are bespoke and tested offline. Agents would subsume `interpret`/`speak`
  but must defer to `dispatch`.

**If we pursue it:** the only sane shape is **engine-as-tools** — an ElevenAgent handles mic/voice +
turn-taking, and **every clinical action is a tool call into our engine** (`order_investigation`,
`perform_exam`, `medicate`, `diagnose`…), with the engine's return text spoken back. Actor switching
stays our concern (route to the right agent/voice). Treat this as a **spike**, not a migration.

---

## 8 · Recommended path for CasesBasis

1. **Now — TTS swap (Tier 1).** Add an ElevenLabs branch to `voice_service._tts_request` using
   **Flash v2.5**; design one **`voice_id` per actor** via Voice Design; store as `ELEVENLABS_*_VOICE_ID`
   and resolve them in `actors.py`. Keep Together as a fallback (mirrors the existing soft-fallback in
   `synthesize`). **Biggest perceived quality jump for least risk.**
2. **Next — per-actor `voice_settings`.** Low `stability` for the patient (affect), higher for
   consultant/nurse. Optionally render a few **non-realtime lines** (opening handoff, debrief) with
   **Eleven v3 + audio tags** for extra drama.
3. **Evaluate — Scribe.** Swap `transcribe()` to **`scribe_v2_realtime`** with **keyterm prompting**
   seeded from the case's drugs/investigations → better intent matching.
4. **Spike (optional) — ElevenAgents** as a voice-I/O shell with **engine-as-tools**. Time-boxed
   research only; do not let it own clinical truth.

---

## 9 · Open questions / risks

- **Latency budget:** measure Flash v2.5 + Scribe realtime end-to-end vs our current Cartesia/Whisper
  path before committing the live room.
- **v3 availability/limits** on the grant tier (5k char cap, not real-time) — confirm before using for
  anything interactive.
- **Voice consistency across cases:** decide if `voice_id`s are fixed per role (simpler) or designed
  per case persona (richer, more management).
- **Fallback story:** keep the room speaking if ElevenLabs 4xx's — reuse the existing
  `synthesize()` soft-fallback pattern (currently retries the default voice).
- **Scribe v1 deprecation (9 Jul 2026)** — if we ever touched v1, go straight to v2.
- **Cost guardrails:** even sponsored, prefer Flash for chatty turns; reserve v3 for set-pieces.

---

## 10 · Sources

- [Models | ElevenLabs Docs](https://elevenlabs.io/docs/overview/models)
- [Text to Speech | ElevenLabs Docs](https://elevenlabs.io/docs/overview/capabilities/text-to-speech)
- [TTS best practices (audio tags, pauses, model choice)](https://elevenlabs.io/docs/overview/capabilities/text-to-speech/best-practices)
- [Text to Speech convert API reference](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
- [Voices (library, cloning, voice design)](https://elevenlabs.io/docs/overview/capabilities/voices)
- [ElevenAgents overview](https://elevenlabs.io/docs/eleven-agents/overview)
- [Agents Platform quickstart](https://elevenlabs.io/docs/agents-platform/quickstart)
- [Eleven v3 announcement](https://elevenlabs.io/blog/eleven-v3)
- [Pricing](https://elevenlabs.io/pricing) · [API pricing](https://elevenlabs.io/pricing/api) · [Agents pricing](https://elevenlabs.io/pricing/agents)
- [Lowered API/Agents pricing + PAYG (May 2026)](https://elevenlabs.io/blog/weve-lowered-api-agents-pricing-and-introduced-pay-as-you-go)
