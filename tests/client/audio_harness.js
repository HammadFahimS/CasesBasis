// Audio hardening (2026-09-05): replies must be heard even when the UI update
// throws, when one playback stalls, when a live cue lands mid-turn, and when the
// AudioContext will not run — and every one of those must reach the client log.
const sb = require('./sandbox').create(process.argv[2], {
  globals: { __ttsWatchdog: { base: 300, perChar: 0, max: 300 } },   // fast watchdog for the test
});
const { calls, fetchLog, FakeWS, FakeAudioContext, routes, responses, find, tick, assert } = sb;
const MODE = process.argv[3] || 'functional';

const ttsPosts = () => fetchLog.filter(f => /\/api\/voice\/tts/.test(f[0]));
const ttsTexts = () => ttsPosts().map(f => f[2] && f[2].text);
const logged = () => [].concat(...fetchLog.filter(f => /client-log/.test(f[0])).map(f => (f[2] && f[2].events) || []));
const has = (kind, re) => logged().some(e => e.kind === kind && (!re || re.test(e.detail)));
const ev = (code, extra) => Object.assign({ code, key: ' ', repeat: false, target: { tagName: 'BODY' }, preventDefault(){} }, extra || {});
const fire = (fns, e) => fns.forEach(f => f(e));

let reply = '';
routes.turn = async () => responses.json({ reply, speaker: 'nurse' });

(async () => {
  const kd = find('document', 'keydown'), ku = find('document', 'keyup');
  const boot = MODE === 'cool' ? find('#talkBtn', 'pointerdown') : find('#micOrb', 'pointerdown');
  assert(boot.length === 1, 'room has a start control');
  boot[0]({ pointerId: 1, preventDefault(){} });
  await tick(120);
  assert(FakeWS.all.some(w => /\/ws\/voice\//.test(w.url)), 'live event channel is open');

  // Speak one utterance through the real push-to-talk path and wait for delivery.
  async function say(text, settle){
    fire(kd, ev('Space')); await tick(250); fire(ku, ev('Space')); await tick(15);
    if (MODE !== 'cool'){ const ws = FakeWS.all.filter(w => /\/ws\/stt/.test(w.url)).pop(); ws.onmessage({ data: JSON.stringify({ final: true, text }) }); }
    await tick(settle || 200);
  }

  // 1. a normal turn is voiced
  reply = 'Blood pressure 138 over 84, heart rate 92.';
  await say('vitals please');
  assert(ttsTexts().includes(reply), 'a normal reply is fetched for speech');

  // 2. the engine update throws → the reply is still voiced, the error is logged
  reply = 'Four liters of oxygen via nasal prongs. I will get that started.';
  routes.turn = async () => responses.json({ reply, speaker: 'nurse', get vitals(){ throw new Error('boom in applyEngine'); } });
  await say('four liters of oxygen');
  assert(ttsTexts().includes(reply), 'a UI error after the reply does not mute it');
  routes.turn = async () => responses.json({ reply, speaker: 'nurse' });

  // 3. one playback stalls → the watchdog frees the queue and the next reply plays
  routes.tts = async () => responses.pcm(800, true);
  reply = 'The chest X-ray is up on the screen for you to review.';
  await say('chest x-ray', 450);
  assert(ttsTexts().includes(reply), 'the stalled line was requested');
  assert(calls.some(c => c[0] === 'gain.zero'), 'watchdog killed the stalled line');
  routes.tts = async () => responses.pcm();
  reply = 'Oxygen saturation is 84 percent.';
  await say('what is the sat now');
  assert(ttsTexts().includes(reply), 'the reply after a stall is still voiced (queue not wedged)');

  // 4. a live cue that lands while the turn is in flight is voiced, then the reply
  routes.turn = async () => { await tick(200); return responses.json({ reply, speaker: 'nurse' }); };
  reply = 'Epinephrine is in.';
  const live = FakeWS.all.filter(w => /\/ws\/voice\//.test(w.url)).pop();
  const p = say('epinephrine 0.5 IM', 500);
  await tick(300);
  live.onmessage({ data: JSON.stringify({ seq: 1, kind: 'cue', speaker: 'rt', text: 'Sats are down to seventy-eight and still dropping.', t: 90 }) });
  await p;
  const texts = ttsTexts();
  assert(texts.includes('Sats are down to seventy-eight and still dropping.'), 'a mid-turn live cue is voiced');
  assert(texts.indexOf('Sats are down to seventy-eight and still dropping.') < texts.lastIndexOf(reply), 'the cue plays before the reply that was still in flight');
  routes.turn = async () => responses.json({ reply, speaker: 'nurse' });

  // 5. the AudioContext will not run → the line is fetched as MP3 and played via <audio>
  const ctx = FakeAudioContext.last; ctx.state = 'suspended'; FakeAudioContext.resumeHangs = true;
  routes.tts = async (body) => (body && body.format === 'mp3') ? responses.mp3() : responses.pcm();
  reply = 'Tube is through. Good chest rise on the right.';
  const before = calls.filter(c => c[0] === 'audio.play').length;
  await say('intubate', 1900);
  const last = ttsPosts().pop();
  assert(last && last[2].text === reply && last[2].format === 'mp3', 'a stuck AudioContext falls back to an MP3 request');
  assert(calls.filter(c => c[0] === 'audio.play').length === before + 1, 'the MP3 is played through <audio>');
  ctx.state = 'running'; FakeAudioContext.resumeHangs = false; routes.tts = async () => responses.pcm();

  // 6. page errors reach the log too
  find('window', 'error').forEach(f => f({ message: 'Cannot read properties of null', filename: 'room.js', lineno: 7 }));

  await tick(2300);   // the log batches every 2 s
  assert(has('ui.error', /^engine: .*boom in applyEngine/), 'the engine-update error was logged');
  assert(has('tts.watchdog', /chest X-ray/), 'the watchdog fire was logged with the line');
  assert(has('audio.ctx', /suspended/), 'the stuck AudioContext was logged');
  assert(has('js.error', /Cannot read properties of null @room\.js:7/), 'window errors are logged');
  assert(!has('tts.skipped'), 'no reply was skipped in this run');
  console.log('\nALL PASS' + (MODE === 'cool' ? ' (cool)' : '')); process.exit(0);
})().catch(e => { console.error('HARNESS ERROR', e); console.error(calls.slice(-12)); console.error(fetchLog.slice(-6)); process.exit(1); });
