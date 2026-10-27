# Latency bench

Measure each leg of a voice turn, record the raw samples, and only believe an
optimisation once the comparison says the shift is bigger than the noise.

```bash
python -m bench.latency --label baseline            # full run (needs API keys)
python -m bench.latency --pure-only                 # no network, no keys
python -m bench.latency --label after-step-1 --n 15
python -m bench.latency --label flash --probes tts_ttfb --tts-model eleven_flash_v2_5

python -m bench.compare bench/results/<before>.json bench/results/<after>.json
```

Commit the files in `results/` — the history is the point. `fixtures/` is
generated and gitignored.

## Probes

| Probe | Tier | What it measures |
|---|---|---|
| `route` | pure | `routing.route()` — C2 addressee resolution |
| `dispatch` | pure | `nurse_brain.dispatch()` — deterministic engine step |
| `engine_tick` | pure | one `advance_clock`, what SessionRuntime pays every 2 s |
| `llm_interpret` | network | utterance → structured intent (LLM call 1 of 2) |
| `llm_speak` | network | engine outcome → spoken phrasing (LLM call 2 of 2) |
| `turn_total` | network | the whole `run_turn` composite |
| `tts_ttfb` | network | **time to first PCM chunk** — when the resident first hears anything |
| `tts_ttfb_cold` | network | same, with the connection pool discarded first |
| `stt_blocking` | network | Scribe v1 blob upload → transcript |

`turn_total` is kept alongside its parts deliberately: if `llm_interpret` and
`llm_speak` both fall and the composite does not, the saving went somewhere we
are not looking.

`tts_ttfb_cold` exists to keep commit `9fb71c0` (warm pooled connections)
honest. The gap between it and `tts_ttfb` is the TLS handshake you avoid by
staying warm — and it is what every resident pays on their first turn.

## Reading a comparison

```
✓ llm_speak         412.0ms     0.1ms    -411.9ms  -99.9%  [-441.2, -388.0]
· tts_ttfb          186.0ms   181.0ms      -5.0ms   -2.7%  [-22.4, +11.9]  CI includes zero
```

`✓ IMPROVED` `✗ REGRESSED` `· NOISE`. A change is only called real when the
bootstrap 90% CI on the median difference **excludes zero** *and* the shift
clears `--min-effect-ms` (default 5 ms). Both gates matter: the CI stops you
reading a lucky draw as progress, the floor stops you shipping a statistically
real 2 ms win as an achievement.

`bench.compare` exits non-zero if anything regressed, so it can gate a change.

## Method notes, and where it is weak

- **Raw samples are stored**, not just percentiles, so the bootstrap has
  something to resample. Do not strip them from the result files.
- **p95, not mean.** Latency is long-tailed; the mean hides the bad turn the
  resident actually notices.
- **Warm-up samples are discarded** (`--warmup`, default 1) so network probes
  measure steady state rather than a TLS handshake. `tts_ttfb_cold` opts out.
- **The engine is snapshot/restored** around mutating probes, so sample 10
  starts from the same clinical state as sample 1 — otherwise the patient
  deteriorates under the benchmark and later samples measure a different case.
- **The session is in-memory.** One DB read for the case, no writes, no
  persistence, nothing left behind.
- **This measures the server, not the browser.** Mic capture (push-to-talk: the
  resident's key release is the utterance boundary, so there is no VAD hangover),
  Web Audio scheduling and the client's own round-trips are not in these
  numbers. Real end-to-end is worse than `turn_total + tts_ttfb`, and closing
  that gap needs instrumentation in `voice_room_functional.html`.
- **Shared network, single host.** Runs taken at different times of day over
  different connections are not strictly comparable. When a verdict is
  marginal, re-run both sides back to back rather than trusting the number.
