// Drive the functional room's real inline script with stubbed browser globals and
// exercise push-to-talk: hold -> capture starts + PCM socket armed; release -> recorder
// stops + commit sent; short tap -> discarded; Space in a text field -> ignored.
const vm = require('vm'), fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const calls = [];                     // (what, detail) trace
const listeners = [];                 // {target, type, fn}
function stub(name){
  const f = function(){ return stub(name + '()'); };
  return new Proxy(f, {
    get(t, k){
      if (typeof k === 'symbol' || k === 'then') return undefined;
      if (k === 'toString' || k === 'valueOf') return () => '';
      if (k === 'length') return 0;
      if (k === 'querySelectorAll' || k === 'getElementsByClassName' || k === 'children') return () => [];
      if (k === 'querySelector') return (sel) => stub(sel);
      if (k === 'getElementById') return (id) => stub('#' + id);
      if (k === 'closest') return (sel) => stub(name + '.closest(' + sel + ')');
      if (k === 'getAttribute') return () => null;
      if (k === 'contains') return () => false;
      if (k === 'addEventListener') return (type, fn) => { listeners.push({ target: name, type, fn }); };
      if (k === 'classList') return { add(){}, remove(){}, toggle(){}, contains(){ return false; } };
      if (k === 'style') return stub(name + '.style');
      if (k === 'dataset') return stub(name + '.dataset');
      if (k === 'tagName') return 'DIV';
      if (k === 'value' || k === 'textContent' || k === 'innerHTML' || k === 'placeholder') return '';
      if (k === 'disabled' || k === 'hidden' || k === 'isContentEditable') return false;
      return stub(name + '.' + String(k));
    },
    set(){ return true; },
    construct(){ return stub('new ' + name); },
    apply(){ return stub(name + '()'); },
  });
}
class FakeRecorder {
  constructor(stream, opts){ this.opts = opts; calls.push(['recorder.new', opts && opts.mimeType]); FakeRecorder.last = this; }
  static isTypeSupported(){ return true; }
  start(){ calls.push(['recorder.start']); }
  stop(){ calls.push(['recorder.stop']); setTimeout(() => { const blob = new Blob([new Uint8Array(6000)], { type: 'audio/webm' }); this.ondataavailable && this.ondataavailable({ data: blob }); this.onstop && this.onstop(); }, 0); }
}
class FakeWS {
  constructor(url){ this.url = url; this.readyState = 0; this.sent = []; FakeWS.last = this; calls.push(['ws.new', url]); setTimeout(() => { this.readyState = 1; this.onopen && this.onopen(); }, 20); }
  send(x){ this.sent.push(x); if (typeof x === 'string') calls.push(['ws.send', x]); else calls.push(['ws.pcm', x.byteLength]); }
  close(){}
}
const fetchLog = [];
const sandboxBase = {
  console, setTimeout, clearTimeout, setInterval, clearInterval, Promise, JSON, Math, Date, Object, Array, String, Number, Boolean, RegExp, Error, TypeError, Map, Set, Symbol, parseInt, parseFloat, isNaN, encodeURIComponent, decodeURIComponent,
  Blob, URLSearchParams, AbortController, Uint8Array, Int16Array, Float32Array, ArrayBuffer, Proxy, Reflect, Intl,
  MediaRecorder: FakeRecorder, WebSocket: FakeWS,
  Audio: class { play(){ setTimeout(() => this.onended && this.onended(), 5); return Promise.resolve(); } pause(){} },
  performance: { now: () => Date.now() },
  location: { search: '?ui=functional&speech=models&case=foreign_body_airway', protocol: 'https:', host: 'x', href: '' , pathname: '/simulator/voice-room' },
  navigator: { mediaDevices: { getUserMedia: async () => stub('stream') }, userAgent: 'node' },
  fetch: async (url, opts) => { fetchLog.push([url, opts && opts.method]); return { ok: true, status: 200, json: async () => ({ session_id: 'S1', intro: '', vitals: {}, roster: [], menu: {}, history: [], cards: [] }), headers: { get: () => '' }, blob: async () => new Blob([]) }; },
  requestAnimationFrame: () => 0, cancelAnimationFrame(){},
  URL: { createObjectURL: () => 'blob:x', revokeObjectURL(){} },
  localStorage: { getItem: () => null, setItem(){}, removeItem(){} },
  sessionStorage: { getItem: () => null, setItem(){}, removeItem(){} },
  speechSynthesis: undefined, SpeechRecognition: undefined, webkitSpeechRecognition: undefined, webkitAudioContext: undefined,
};
const doc = stub('document');
const sandbox = new Proxy(sandboxBase, {
  has: () => true,
  get(t, k){
    if (k in t) return t[k];
    if (k === 'document') return doc;
    if (k === 'addEventListener') return (type, fn) => { listeners.push({ target: 'window', type, fn }); };
    if (k === 'window' || k === 'globalThis' || k === 'self') return sandbox;
    if (k === 'undefined') return undefined;
    if (typeof k === 'symbol') return undefined;
    return stub(String(k));
  },
  set(t, k, v){ t[k] = v; return true; },
});
const ctx = vm.createContext(sandbox);
vm.runInContext(src, ctx, { filename: 'room.js' });

const find = (target, type) => listeners.filter(l => l.target.includes(target) && l.type === type).map(l => l.fn);
const tick = (ms) => new Promise(r => setTimeout(r, ms));
function assert(cond, msg){ if (!cond){ console.error('FAIL:', msg); console.error(calls); process.exit(1); } console.log('ok  -', msg); }

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
  const ws = FakeWS.last;
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
  ws.readyState = 3; ws.onclose && ws.onclose();
  await tick(10);
  FakeWS.last = null;
  fire(kd, bodyEv('Space')); await tick(250); fire(ku, bodyEv('Space')); await tick(100);
  assert(fetchLog.some(f => /\/api\/voice\/stt/.test(f[0])), 'with the socket down, release uploads the blob to /api/voice/stt');
  await tick(600);
  assert(FakeWS.last && /\/ws\/stt/.test(FakeWS.last.url), 'socket reconnects on its own after a drop');

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
