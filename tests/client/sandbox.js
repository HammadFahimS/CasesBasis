// A Node `vm` sandbox that runs a voice room's real inline script against stubbed
// browser globals. Shared by the push-to-talk and audio-hardening harnesses.
//
// The DOM is a Proxy that answers every property with another stub, so the page
// boots without a browser; the parts the scenarios care about are real fakes:
// MediaRecorder, WebSocket, AudioContext, Audio, fetch (per-route, swappable).
const vm = require('vm'), fs = require('fs');

function create(scriptPath, opts){
  opts = opts || {};
  const calls = [];                     // (what, detail) trace
  const listeners = [];                 // {target, type, fn}
  const fetchLog = [];                  // [url, method, body]
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
    constructor(stream, o){ this.opts = o; calls.push(['recorder.new', o && o.mimeType]); FakeRecorder.last = this; }
    static isTypeSupported(){ return true; }
    start(){ calls.push(['recorder.start']); }
    stop(){ calls.push(['recorder.stop']); setTimeout(() => { const blob = new Blob([new Uint8Array(6000)], { type: 'audio/webm' }); this.ondataavailable && this.ondataavailable({ data: blob }); this.onstop && this.onstop(); }, 0); }
  }
  class FakeWS {
    constructor(url){ this.url = url; this.readyState = 0; this.sent = []; FakeWS.last = this; FakeWS.all.push(this); calls.push(['ws.new', url]); setTimeout(() => { this.readyState = 1; this.onopen && this.onopen(); }, 20); }
    send(x){ this.sent.push(x); if (typeof x === 'string') calls.push(['ws.send', x]); else calls.push(['ws.pcm', x.byteLength]); }
    close(){}
  }
  FakeWS.all = [];
  // Web Audio: sources "play" for 30 ms then end; the clock does not advance, so the
  // rooms' safety timeouts stay short. resume() can be made to hang (a stuck context).
  class FakeAudioContext {
    constructor(){ this.state = FakeAudioContext.initialState; this.currentTime = 0; this.sampleRate = 48000; this.destination = {}; FakeAudioContext.last = this; calls.push(['ctx.new']); }
    resume(){ calls.push(['ctx.resume']); if (FakeAudioContext.resumeHangs) return new Promise(() => {}); this.state = 'running'; return Promise.resolve(); }
    createGain(){ return { connect(){}, gain: { value: 1, setValueAtTime(){ calls.push(['gain.zero']); } } }; }
    createBuffer(ch, n, rate){ return { duration: n / rate, getChannelData(){ return new Float32Array(n); } }; }
    createBufferSource(){ const s = { buffer: null, connect(){}, start(){ calls.push(['src.start']); setTimeout(() => s.onended && s.onended(), 30); } }; return s; }
    createAnalyser(){ return { connect(){}, fftSize: 0, frequencyBinCount: 256, getByteTimeDomainData(){}, getFloatTimeDomainData(){} }; }
    createMediaStreamSource(){ return { connect(){} }; }
    createBiquadFilter(){ return { connect(){}, type: '', frequency: { value: 0 }, Q: { value: 0 } }; }
    createScriptProcessor(){ return { connect(){}, onaudioprocess: null }; }
  }
  FakeAudioContext.initialState = 'running';
  FakeAudioContext.resumeHangs = false;
  class FakeAudio {
    constructor(url){ this.url = url; FakeAudio.last = this; calls.push(['audio.new']); }
    play(){ calls.push(['audio.play']); if (FakeAudio.rejectWith){ const n = FakeAudio.rejectWith; const err = new Error(n); err.name = n; return Promise.reject(err); } setTimeout(() => this.onended && this.onended(), 5); return Promise.resolve(); }
    pause(){}
  }
  FakeAudio.rejectWith = null;

  // ── fetch: one swappable implementation per route ──
  const json = (obj) => ({ ok: true, status: 200, json: async () => obj, headers: { get: () => 'application/json' }, blob: async () => new Blob([]) });
  const pcm = (bytes, stall) => {
    let sent = false;
    return { ok: true, status: 200, headers: { get: (h) => /content-type/i.test(h) ? 'audio/pcm' : '' }, json: async () => ({}), blob: async () => new Blob([]),
      body: { getReader(){ return { read: () => { if (stall) return new Promise(() => {}); if (!sent){ sent = true; return Promise.resolve({ done: false, value: new Uint8Array(bytes || 800) }); } return Promise.resolve({ done: true }); } }; } } };
  };
  const mp3 = () => ({ ok: true, status: 200, headers: { get: (h) => /content-type/i.test(h) ? 'audio/mpeg' : '' }, json: async () => ({}), blob: async () => new Blob([new Uint8Array(10)]) });
  const routes = {
    session: async () => json({ session_id: 'S1', intro: '', vitals: {}, roster: [], menu: {}, history: [], cards: [], live_runtime: true }),
    turn:    async () => json({ reply: 'Four liters of oxygen via nasal prongs. I will get that started.', speaker: 'nurse' }),
    action:  async () => json({ reply: 'Ordered.', speaker: 'nurse' }),
    tts:     async () => pcm(),
    stt:     async () => json({ text: 'give oxygen' }),
    log:     async (body) => json({ stored: (body && body.events && body.events.length) || 0 }),
    debrief: async () => json({}),
    other:   async () => json({}),
  };
  function routeOf(url){
    if (/\/api\/voice\/session/.test(url)) return 'session';
    if (/\/api\/voice\/turn/.test(url)) return 'turn';
    if (/\/api\/voice\/action/.test(url)) return 'action';
    if (/\/api\/voice\/tts/.test(url)) return 'tts';
    if (/\/api\/voice\/stt/.test(url)) return 'stt';
    if (/\/api\/voice\/client-log/.test(url)) return 'log';
    if (/\/api\/voice\/debrief/.test(url)) return 'debrief';
    return 'other';
  }
  async function fetchImpl(url, o){
    let body = null;
    try { body = o && typeof o.body === 'string' ? JSON.parse(o.body) : (o && o.body) || null; } catch(e){ body = o.body; }
    fetchLog.push([url, o && o.method, body]);
    return routes[routeOf(url)](body, url, o);
  }

  const sandboxBase = Object.assign({
    console, setTimeout, clearTimeout, setInterval, clearInterval, Promise, JSON, Math, Date, Object, Array, String, Number, Boolean, RegExp, Error, TypeError, Map, Set, Symbol, parseInt, parseFloat, isNaN, encodeURIComponent, decodeURIComponent,
    Blob, URLSearchParams, AbortController, Uint8Array, Int16Array, Float32Array, ArrayBuffer, DataView, Proxy, Reflect, Intl,
    MediaRecorder: FakeRecorder, WebSocket: FakeWS, AudioContext: FakeAudioContext, Audio: FakeAudio,
    performance: { now: () => Date.now() },
    location: { search: opts.search || '?ui=functional&speech=models&case=foreign_body_airway', protocol: 'https:', host: 'x', href: '', pathname: '/simulator/voice-room' },
    navigator: { mediaDevices: { getUserMedia: async () => stub('stream') }, userAgent: 'node', sendBeacon: (url, blob) => { calls.push(['beacon', url]); return true; } },
    fetch: fetchImpl,
    requestAnimationFrame: () => 0, cancelAnimationFrame(){},
    URL: { createObjectURL: () => 'blob:x', revokeObjectURL(){} },
    localStorage: { getItem: () => null, setItem(){}, removeItem(){} },
    sessionStorage: { getItem: () => null, setItem(){}, removeItem(){} },
    speechSynthesis: undefined, SpeechRecognition: undefined, webkitSpeechRecognition: undefined, webkitAudioContext: undefined,
  }, opts.globals || {});
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
  vm.runInContext(fs.readFileSync(scriptPath, 'utf8'), ctx, { filename: 'room.js' });

  const find = (target, type) => listeners.filter(l => l.target.includes(target) && l.type === type).map(l => l.fn);
  const tick = (ms) => new Promise(r => setTimeout(r, ms));
  function assert(cond, msg){ if (!cond){ console.error('FAIL:', msg); console.error(calls.slice(-25)); console.error(fetchLog.slice(-8)); process.exit(1); } console.log('ok  -', msg); }
  return { calls, listeners, fetchLog, FakeWS, FakeRecorder, FakeAudioContext, FakeAudio, routes, responses: { json, pcm, mp3 }, sandbox, find, tick, assert };
}
module.exports = { create };
