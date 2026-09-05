// Drive the functional room's real inline script with stubbed browser globals and
// exercise push-to-talk: hold -> capture starts + PCM socket armed; release -> recorder
// stops + commit sent; short tap -> discarded; Space in a text field -> ignored.
const { calls, fetchLog, FakeWS, find, tick, assert } = require('./sandbox').create(process.argv[2]);

const MODE = process.argv[3] || 'functional';
(async () => {
  if (MODE === 'cool') return coolScenario();
  const orbDown = find('#micOrb', 'pointerdown'); assert(orbDown.length === 1, 'orb has a pointerdown handler');
  const kd = find('document', 'keydown'), ku = find('document', 'keyup');
  assert(kd.length >= 1 && ku.length >= 1, 'document has keydown/keyup handlers');
  const bodyEv = (code, extra) => Object.assign({ code, key: code === 'Space' ? ' ' : code, repeat: false, target: { tagName: 'BODY' }, preventDefault(){ calls.push(['preventDefault']); } }, extra || {});
  const fire = (fns, ev) => fns.forEach(f => f(ev));

  // 1. first orb press begins the case + mic (startServer)
  orbDown[0]({ pointerId: 1, preventDefault(){} });
  await tick(80);
  assert(calls.some(c => c[0] === 'ws.new' && /\/ws\/stt/.test(c[1])), 'STT socket opened once live');
  assert(!calls.some(c => c[0] === 'recorder.start'), 'nothing recorded before a press');

  // 2. Space held -> capture starts; auto-repeat ignored
  fire(kd, bodyEv('Space')); fire(kd, bodyEv('Space', { repeat: true }));
  await tick(10);
  assert(calls.filter(c => c[0] === 'recorder.start').length === 1, 'hold starts exactly one recording');
  const sttSock = () => FakeWS.all.filter(w => /\/ws\/stt/.test(w.url)).pop();
  const ws = sttSock();
  // 3. release after a real hold -> recorder stops, commit sent, transcript awaited then turn posted
  await tick(250);
  fire(ku, bodyEv('Space'));
  await tick(10);
  assert(calls.some(c => c[0] === 'recorder.stop'), 'release stops the recorder');
  assert(ws.sent.some(x => typeof x === 'string' && JSON.parse(x).t === 'commit'), 'release commits the streaming transcript');
  ws.onmessage({ data: JSON.stringify({ final: true, text: 'give 0.5 milligrams of epinephrine IM' }) });
  await tick(50);
  assert(fetchLog.some(f => /\/api\/voice\/turn/.test(f[0])), 'final transcript reaches /api/voice/turn');
  const turnsBefore = fetchLog.filter(f => /\/api\/voice\/turn/.test(f[0])).length;

  // 4. short tap -> discarded, no turn
  await tick(300);
  const startsBefore = calls.filter(c => c[0] === 'recorder.start').length;
  fire(kd, bodyEv('Space')); await tick(20); fire(ku, bodyEv('Space')); await tick(100);
  assert(calls.filter(c => c[0] === 'recorder.start').length === startsBefore + 1, 'tap still opens the recorder');
  assert(fetchLog.filter(f => /\/api\/voice\/turn/.test(f[0])).length === turnsBefore, 'a sub-200 ms tap sends nothing');
  const sttPosts = fetchLog.filter(f => /\/api\/voice\/stt/.test(f[0])).length;
  assert(sttPosts === 0, 'no batch STT upload for the tap');

  // 5. Space inside the command box is left alone
  const s2 = calls.filter(c => c[0] === 'recorder.start').length;
  fire(kd, bodyEv('Space', { target: { tagName: 'INPUT' } })); await tick(10);
  assert(calls.filter(c => c[0] === 'recorder.start').length === s2, 'Space while typing does not open the mic');

  // 6. socket closed -> press still records and release falls back to batch upload
  await tick(300);
  const socketsBefore = FakeWS.all.length;
  ws.readyState = 3; ws.onclose && ws.onclose();
  await tick(10);
  fire(kd, bodyEv('Space')); await tick(250); fire(ku, bodyEv('Space')); await tick(100);
  assert(fetchLog.some(f => /\/api\/voice\/stt/.test(f[0])), 'with the socket down, release uploads the blob to /api/voice/stt');
  await tick(600);
  assert(FakeWS.all.length > socketsBefore && sttSock() !== ws, 'socket reconnects on its own after a drop');

  // 7. losing window focus releases a held key
  const stops = calls.filter(c => c[0] === 'recorder.stop').length;
  fire(kd, bodyEv('Space')); await tick(250);
  find('window', 'blur').forEach(f => f());
  await tick(10);
  assert(calls.filter(c => c[0] === 'recorder.stop').length === stops + 1, 'window blur releases the mic');
  console.log('\nALL PASS'); process.exit(0);
})().catch(e => { console.error('HARNESS ERROR', e); console.error(calls.slice(-10)); process.exit(1); });

async function coolScenario(){
  const bodyEv = (code, extra) => Object.assign({ code, key: ' ', repeat: false, target: { tagName: 'BODY' }, preventDefault(){ calls.push(['preventDefault']); } }, extra || {});
  const fire = (fns, ev) => fns.forEach(f => f(ev));
  const talkDown = find('#talkBtn', 'pointerdown'); assert(talkDown.length === 1, 'talk button has a pointerdown handler');
  const kd = find('document', 'keydown'), ku = find('document', 'keyup');
  assert(kd.length >= 1 && ku.length >= 1, 'document has keydown/keyup handlers');
  talkDown[0]({ pointerId: 1, preventDefault(){} });
  await tick(80);
  assert(!calls.some(c => c[0] === 'recorder.start'), 'nothing recorded before a press');
  fire(kd, bodyEv('Space')); fire(kd, bodyEv('Space', { repeat: true })); await tick(10);
  assert(calls.filter(c => c[0] === 'recorder.start').length === 1, 'hold starts exactly one recording');
  await tick(250); fire(ku, bodyEv('Space')); await tick(30);
  assert(calls.some(c => c[0] === 'recorder.stop'), 'release stops the recorder');
  assert(fetchLog.some(f => /\/api\/voice\/stt/.test(f[0])), 'release uploads the blob to /api/voice/stt');
  await tick(400);
  const turns = () => fetchLog.filter(f => /\/api\/voice\/turn/.test(f[0])).length, stts = () => fetchLog.filter(f => /\/api\/voice\/stt/.test(f[0])).length;
  const t0 = turns(), s0 = stts();
  fire(kd, bodyEv('Space')); await tick(20); fire(ku, bodyEv('Space')); await tick(100);
  assert(stts() === s0 && turns() === t0, 'a sub-200 ms tap sends nothing');
  const r0 = calls.filter(c => c[0] === 'recorder.start').length;
  fire(kd, bodyEv('Space', { target: { tagName: 'INPUT' } })); await tick(10);
  assert(calls.filter(c => c[0] === 'recorder.start').length === r0, 'Space while typing does not open the mic');
  const stops = calls.filter(c => c[0] === 'recorder.stop').length;
  fire(kd, bodyEv('Space')); await tick(250); find('window', 'blur').forEach(f => f()); await tick(30);
  assert(calls.filter(c => c[0] === 'recorder.stop').length === stops + 1, 'window blur releases the mic');
  console.log('\nALL PASS (cool)'); process.exit(0);
}
