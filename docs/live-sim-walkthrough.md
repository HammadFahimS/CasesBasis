# Live Sim — End-to-End Walkthrough

A full voice-room walkthrough of the **Myasthenic Crisis** case that exercises
every feature: patient/EMS/husband/RT voices, addressee routing, gated
investigations, the medication dose-confirm loop, the dosing-nuance rubric, the
`diagnose` op, consults, the critical-error branch, the cue engine, status
chips, reaching the terminal state, the debrief, and analytics.

LLM phrasing varies, so the **deterministic** facts are anchored below (who
answers, which card, state changes, gating). Each step notes what it's testing.

## Two things about timing

- **The clock is real time.** Gated results only return after the wall-clock
  delay: **VC/NIF 60s · VBG 120s · husband arrives 300s · CT chest 300s.**
  Order it, keep working, ask again later.
- **Addressing:** lead with a name to reach staff —
  **"Nurse… / Husband… / EMS… / RT… / Neuro… / ICU…"**. Anything *unaddressed*
  goes to the **patient**.

---

## 0 · Start
- Pick a voice backend in the modal → **Tap to start**.
- *Expect:* the **nurse** voices a bedside handoff — names EK, 29F, respiratory
  distress. ✓ opening narration + nurse TTS voice.

## 1 · Talk to the patient (unaddressed → patient)
- **Say:** "Hi, can you tell me what's been happening?"
- *Expect:* the **patient** answers breathlessly, short sentences — dyspnoea,
  weakness. Line tagged **Patient** in the Room log. ✓ patient channel +
  unaddressed routing.

## 2 · EMS handover (addressee → EMS voice)
- **Say:** "EMS, what's the story?"
- *Expect:* the **paramedic** gives the handover (called 20 min ago, tripoding,
  O₂ for comfort) — tagged **EMS**, distinct voice. ✓ EMS actor + addressee
  routing + provider-voiced history.

## 3 · Husband, early (tests the 300s gate)
- **Say:** "Husband, what's her background?"
- *Expect (before 5 min):* the **nurse** says he's not here yet / a few minutes
  behind. ✓ gated history doesn't leak; falls back to nurse. *(Back to him at
  step 8.)*

## 4 · Neuro exam (advances Baseline → Recognition)
- **Say:** "Nurse, examine her neurological system."  *(or click Exam →
  Neurological)*
- *Expect:* an **Exam card** (bilateral facial weakness, **left ptosis**,
  fatigable weakness, neck flexion 3/5). State flips to **Recognition**.
  ✓ exam card + the baseline→recognition transition.

## 5 · ECG + CXR (cards with visuals)
- **Say:** "Nurse, get me an ECG." then "pull up a chest film."
- *Expect:* an **ECG card** with a tracing (sinus tach ~105) that wakes the
  lead-II monitor; a **CXR card** (no acute process). ✓ investigation cards +
  visuals.

## 6 · Bedside measures via RT (60s gate + RT voice)
- **Say:** "RT, get me a vital capacity and a NIF."
- *Expect (after ~60s):* the **Respiratory Therapist** reports **VC < 20 mL/kg**
  and **NIF ~ −25** in the RT voice; cards land in Investigations. ✓ RT actor +
  bedside gating.

## 7 · Blood gas (120s gate)
- **Say:** "Nurse, send a venous blood gas." → keep working → later: "Nurse,
  what's the gas show?"
- *Expect:* early = **pending**; after 120s = **pH 7.31, pCO₂ 55** (type-2 resp
  failure). ✓ gating + the result coming back later.

## 8 · Husband collateral (now arrived → family voice)
- **Say (after ~5 min):** "Husband, has she been getting weaker? Any family
  history?"
- *Expect:* the **husband** gives the 3-month fatigable-weakness story (and, if
  prompted, the "mother had a neuromuscular condition" hint) in the **Family**
  voice. ✓ the recognition arc that was dark before Phase C.

## 9 · Commit a diagnosis (diagnose op + chip + scoring)
- **Say:** "I think this is a myasthenic crisis."
- *Expect:* the nurse acknowledges **neutrally** (doesn't confirm/grade); a
  **Working dx: myasthenic crisis** chip appears top-left. ✓ diagnose op +
  working-dx chip + recognition criterion (scored at debrief).

## 10 · Neurology consult (consult voice + card)
- **Say:** "Nurse, call neurology." *(or "Neuro, any advice?")*
- *Expect:* the **neurology consultant** answers in a distinct voice (intubate
  early, IVIG/plasma exchange, **no pyridostigmine**); a **Consult card** in the
  Consults tab. ✓ consult routing + voice.

## 11 · Decide to intubate (Recognition → Intubation)
- **Say:** "She's tiring — let's prepare for intubation, RSI."
- *Expect:* state flips to **Intubation — paralytic dosing matters**. ✓ action
  transition (keyword *and* soft-intent matcher).

## 12 · The dose-confirm loop (the case's crux)
- **Say:** "Nurse, give rocuronium." *(no dose)*
- *Expect:* nurse **asks you to confirm the dose**; an **Orders card** shows
  **⚠ confirm dose**. ✓ high-alert hold.
- **Say:** "Rocuronium, a reduced dose, 0.6 mg per kilo."
- *Expect:* the Orders card flips to **✓ given**; because rocuronium is correct
  in MG, state advances to **Post-intubation**. The **Airway chip** now reads
  **intubated · ventilated**. ✓ medication mechanic + dosing-nuance ("reduced"
  earns the rubric point) + airway chip + correct-paralytic transition.

> **Optional — the critical-error branch (run once to see the failure):**
> instead of step 12, say *"Give standard-dose succinylcholine, 1.5 mg/kg."* →
> critical error → **PEA arrest**: the monitor shows **flatline dashes** (not
> stale numbers), SpO₂ 82. Then *"Start CPR and secure the airway."* → after
> ~2 min → **ROSC → Post-intubation.** ✓ critical branch + PEA null-vitals fix +
> ROSC path.

## 13 · Crisis therapy + disposition (reach the terminal state)
- **Say:** "Start IVIG and high-dose steroids." then "Admit her to the ICU."
- *Expect:* "admit"/"IVIG"/"plasma exchange" drives **Post-intubation →
  Disposition (terminal)** → the case **ends** → the **debrief** opens. ✓ the
  disposition fix (was manual-only). **Don't** say "pyridostigmine" — that's the
  trap (fails the avoid criterion).

## 14 · Cue engine (optional — tests proactive prompts)
- At any state, **just wait ~80 seconds** without acting, then take any action.
- *Expect:* the room **prompts you** in the appropriate voice (e.g. nurse:
  "Doctor, should we get collateral from the husband?"). Fires once per cue.
  ✓ cue engine.

## 15 · Debrief
- *Expect:* the debrief shows an **Assessment checklist** — **Recognised the
  diagnosis ✓**, **Avoided critical errors ✓** (✗ if you ran the sux branch),
  **Chose a non-depolariser ✓**, **Reduced the rocuronium dose ✓**, **Avoided
  pyridostigmine ✓** — plus teaching points and a coaching note. ✓ scoring seam +
  medication rubric surfaced.

## 16 · Analytics (the loop closes)
- Go to **/analytics**.
- *Expect:* the run appears in the library with a **🎙 Voice** badge and its
  **score on the trend chart**; clicking **Review** opens the assessment
  checklist + score donut; **Delete** removes it. ✓ Phase D end-to-end.

---

## Fastest clean "happy path"

If you just want an all-green run: **4 → 5 → 6 → 7 → 9 → 11 → 12 (reduced
rocuronium) → 13 (IVIG + admit ICU)**, never saying "pyridostigmine" or
"standard succinylcholine." That lights up every assessment criterion ✓ and
lands a high score in analytics.

## If something misbehaves live

Note the **step number** and what you heard/saw (addressee mis-route, a gate
that feels off, a wrong/duplicate voice). Those are the things only a real run
surfaces.
