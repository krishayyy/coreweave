// Search-loop display: the same case, searched two ways, on one clock.
//
// Canvas2D with no dependencies, deliberately -- a live demo should not need a
// CDN to be reachable, and the moment the premise is abandoned needs
// frame-level control.
//
// The rule the renderer follows is that nothing ever pops. Belief fields
// cross-fade between periods, the aircraft is interpolated between track
// samples, and the graph advances continuously. Discrete updates are the single
// thing that most makes software read as cheap.

const PALETTE = [
  [0.00, [0, 0, 0, 0]],
  [0.30, [18, 16, 52, 0]],
  [0.44, [66, 36, 104, 70]],
  [0.58, [156, 60, 82, 165]],
  [0.74, [236, 116, 40, 224]],
  [0.88, [255, 186, 48, 246]],
  [1.00, [255, 248, 222, 255]],
];

const DWELL_MS = 1500;
const REVISE_MS = 4200;
const FADE_FRAC = 0.55;
const HOLD_MS = 4000;          // how long the final frame sits before looping

const el = (id) => document.getElementById(id);
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const clamp01 = (t) => Math.min(1, Math.max(0, t));

function buildLut() {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    let a = PALETTE[0], b = PALETTE[PALETTE.length - 1];
    for (let s = 0; s < PALETTE.length - 1; s++) {
      if (v >= PALETTE[s][0] && v <= PALETTE[s + 1][0]) { a = PALETTE[s]; b = PALETTE[s + 1]; break; }
    }
    const t = b[0] === a[0] ? 0 : (v - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 4; c++) lut[i * 4 + c] = lerp(a[1][c], b[1][c], t);
  }
  return lut;
}

const loadImage = (src) => new Promise((res, rej) => {
  const img = new Image();
  img.onload = () => res(img);
  img.onerror = () => rej(new Error(`failed to load ${src}`));
  img.src = src;
});

function colorize(img, lut) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) {
    const v = px[i] * 4;
    px[i] = lut[v]; px[i + 1] = lut[v + 1]; px[i + 2] = lut[v + 2]; px[i + 3] = lut[v + 3];
  }
  ctx.putImageData(data, 0, 0);
  return c;
}

function tintSwept(img) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) {
    const on = px[i] > 127;
    px[i] = 36; px[i + 1] = 58; px[i + 2] = 78; px[i + 3] = on ? 132 : 0;
  }
  ctx.putImageData(data, 0, 0);
  return c;
}

/** Shade relief into a cool dark ramp; raw greyscale competes with the field. */
function tintTerrain(img) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) {
    const s = Math.pow(px[i] / 255, 1.45);
    px[i] = 16 + s * 62; px[i + 1] = 20 + s * 70; px[i + 2] = 28 + s * 84; px[i + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
  return c;
}

/** One map pane: terrain, swept ground, belief, aircraft, markers. */
class Pane {
  constructor(root, assets, run, isOurs) {
    this.root = root;
    this.canvas = root.querySelector(".map");
    this.ctx = this.canvas.getContext("2d");
    this.a = assets;
    this.run = run;
    this.ours = isOurs;
    this.resize();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width; this.h = r.height;
  }

  fit() {
    const size = Math.min(this.w, this.h);
    return { x: (this.w - size) / 2, y: (this.h - size) / 2, size };
  }

  /** Frame index for a period, clamped: a finished search holds its last frame. */
  frameFor(period) {
    const n = this.a.belief.length;
    return Math.min(Math.max(period - 1, 0), n - 1);
  }

  draw(period, phase, geo, scenario, showTruth) {
    const i = this.frameFor(period);
    const live = period - 1 < this.a.belief.length;
    const next = Math.min(i + 1, this.a.belief.length - 1);
    const frame = this.frames[i];
    const { x, y, size } = this.fit();
    const ctx = this.ctx;
    const scale = size / geo.cols;
    const fade = live ? easeInOut(clamp01((phase - (1 - FADE_FRAC)) / FADE_FRAC)) : 0;
    const revising = live && frame && frame.revised;
    const dip = revising ? 1 - 0.42 * Math.sin(Math.PI * clamp01(phase / 0.55)) : 1;

    ctx.clearRect(0, 0, this.w, this.h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    ctx.fillStyle = "#0d1015";
    ctx.fillRect(x, y, size, size);
    ctx.globalAlpha = 0.34 * (revising ? 0.5 : 1);
    ctx.drawImage(this.a.hillshade, x, y, size, size);

    ctx.globalAlpha = 0.85;
    ctx.drawImage(this.a.swept[i], x, y, size, size);
    if (fade > 0) { ctx.globalAlpha = 0.85 * fade; ctx.drawImage(this.a.swept[next], x, y, size, size); }

    ctx.globalCompositeOperation = "screen";
    ctx.globalAlpha = (1 - fade) * dip;
    ctx.drawImage(this.a.belief[i], x, y, size, size);
    ctx.globalAlpha = fade * dip;
    ctx.drawImage(this.a.belief[next], x, y, size, size);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;

    if (live && frame) this.aircraft(frame.track || [], x, y, scale, phase);
    this.marker(x, y, scale, scenario.ipp, "#7ee7ff", "cross");
    if (showTruth) this.marker(x, y, scale, scenario.truth, "#ff5a52", "x");

    this.root.querySelector(".revising")?.classList.toggle("on", revising && phase < 0.6);
  }

  aircraft(track, x, y, scale, phase) {
    if (track.length < 2) return;
    const ctx = this.ctx;
    const head = clamp01(phase / 0.86) * (track.length - 1);
    const tail = Math.max(0, Math.floor(head) - 14);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (let k = tail; k < Math.floor(head); k++) {
      const [r0, c0] = track[k], [r1, c1] = track[k + 1];
      const age = (k - tail) / Math.max(Math.floor(head) - tail, 1);
      ctx.beginPath();
      ctx.moveTo(x + (c0 + 0.5) * scale, y + (r0 + 0.5) * scale);
      ctx.lineTo(x + (c1 + 0.5) * scale, y + (r1 + 0.5) * scale);
      ctx.strokeStyle = `rgba(126,231,255,${(0.42 * age).toFixed(3)})`;
      ctx.lineWidth = 1.4; ctx.stroke();
    }
    const i0 = Math.floor(head), i1 = Math.min(i0 + 1, track.length - 1), f = head - i0;
    const ax = x + (lerp(track[i0][1], track[i1][1], f) + 0.5) * scale;
    const ay = y + (lerp(track[i0][0], track[i1][0], f) + 0.5) * scale;
    ctx.beginPath(); ctx.arc(ax, ay, 2.6, 0, Math.PI * 2);
    ctx.fillStyle = "#bff2ff"; ctx.fill();
    ctx.beginPath(); ctx.arc(ax, ay, 6.5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(126,231,255,0.30)"; ctx.lineWidth = 1; ctx.stroke();
  }

  marker(x, y, scale, rc, color, kind) {
    const ctx = this.ctx;
    const px = x + (rc[1] + 0.5) * scale, py = y + (rc[0] + 0.5) * scale;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    const s = 6;
    if (kind === "cross") {
      ctx.moveTo(px - s, py); ctx.lineTo(px + s, py);
      ctx.moveTo(px, py - s); ctx.lineTo(px, py + s);
      ctx.stroke();
      ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.stroke();
    } else {
      ctx.moveTo(px - s, py - s); ctx.lineTo(px + s, py + s);
      ctx.moveTo(px + s, py - s); ctx.lineTo(px - s, py + s);
      ctx.stroke();
    }
  }
}

class Display {
  constructor(run, ours, rival) {
    this.run = run;
    this.paneB = new Pane(el("paneB"), ours, run, true);
    this.paneA = new Pane(el("paneA"), rival, run, false);
    this.paneB.frames = run.frames;
    this.paneA.frames = run.versus.frames;
    this.graph = el("graph");
    this.gctx = this.graph.getContext("2d");
    this.total = Math.max(run.frames.length, run.versus.frames.length);
    this.started = 0;
    this.sizeGraph();
    window.addEventListener("resize", () => {
      this.paneA.resize(); this.paneB.resize(); this.sizeGraph();
    });
  }

  sizeGraph() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.graph.getBoundingClientRect();
    this.graph.width = Math.round(r.width * dpr);
    this.graph.height = Math.round(r.height * dpr);
    this.gctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.gw = r.width; this.gh = r.height;
  }

  clock(now) {
    let t = (now - this.started);
    const span = (i) => (this.run.frames[i] && this.run.frames[i].revised ? REVISE_MS : DWELL_MS);
    for (let i = 0; i < this.total; i++) {
      const s = span(i);
      if (t < s) return { period: i + 1, phase: clamp01(t / s), done: false };
      t -= s;
    }
    return { period: this.total, phase: 1, done: t > HOLD_MS };
  }

  draw(now) {
    const { period, phase, done } = this.clock(now);
    if (done) { this.started = now; return; }
    const geo = this.run.geo, sc = this.run.scenario;
    const finished = period >= this.total;

    this.paneB.draw(period, phase, geo, sc, finished);
    this.paneA.draw(period, phase, geo, sc, finished);
    this.chrome(period, phase, finished);
    this.drawGraph(period, phase);
  }

  chrome(period, phase, finished) {
    el("clock").textContent = `P${String(period).padStart(2, "0")}`;
    const frame = this.run.frames[Math.min(period - 1, this.run.frames.length - 1)];
    const revising = period - 1 < this.run.frames.length && frame && frame.revised;
    const status = el("status");
    status.className = revising ? "alert" : "live";
    status.textContent = revising ? "PREMISE FAILING" : "SEARCHING";

    const ours = this.run.result, rival = this.run.versus;
    const setVerdict = (pane, res, label) => {
      const v = pane.querySelector(".verdict");
      const reached = res.found && period >= res.periods_to_find;
      if (reached) {
        v.className = "verdict on found";
        v.textContent = `SUBJECT LOCATED · PERIOD ${res.periods_to_find}`;
      } else if (finished && phase > 0.05) {
        v.className = "verdict on lost";
        v.textContent = "NOT LOCATED";
      } else {
        v.className = "verdict";
      }
    };
    setVerdict(el("paneB"), ours);
    setVerdict(el("paneA"), rival);
  }

  drawGraph(period, phase) {
    const ctx = this.gctx, w = this.gw, h = this.gh;
    const pad = { l: 34, r: 12, t: 8, b: 18 };
    ctx.clearRect(0, 0, w, h);

    const X = (p) => pad.l + (p - 1) / Math.max(this.total - 1, 1) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - v * (h - pad.t - pad.b);

    ctx.strokeStyle = "#1c2128"; ctx.lineWidth = 1;
    for (const v of [0, 0.5, 1]) {
      ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
      ctx.fillStyle = "#565e6b"; ctx.font = "9px ui-monospace, monospace";
      ctx.fillText(`${(v * 100).toFixed(0)}%`, 6, Y(v) + 3);
    }

    const line = (frames, colour, upto) => {
      ctx.beginPath();
      let started = false;
      for (const f of frames) {
        if (f.period > upto) break;
        const px = X(f.period), py = Y((f.truth_percentile ?? 0) / 100);
        started ? ctx.lineTo(px, py) : (ctx.moveTo(px, py), started = true);
      }
      if (!started) return;
      ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.stroke();
    };
    const upto = period - 1 + phase;
    line(this.run.versus.frames, "#8b93a1", upto);
    line(this.run.frames, "#ffb020", upto);

    // Mark the moment the premise was abandoned.
    for (const f of this.run.frames) {
      if (!f.revised || f.period > upto) continue;
      ctx.strokeStyle = "rgba(255,90,82,0.55)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(f.period), pad.t); ctx.lineTo(X(f.period), h - pad.b); ctx.stroke();
    }

    ctx.font = "9.5px ui-monospace, monospace";
    ctx.fillStyle = "#8b93a1"; ctx.fillText("conventional", w - pad.r - 150, pad.t + 10);
    ctx.fillStyle = "#ffb020"; ctx.fillText("this system", w - pad.r - 62, pad.t + 10);
  }

  run_() {
    this.started = performance.now();
    const tick = (now) => { this.draw(now); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
}

async function boot() {
  const run = await (await fetch("public/run/run.json")).json();
  if (!run.versus) throw new Error("run.json has no comparison arm; export with --versus");
  const lut = buildLut();
  const base = "public/run/";

  const load = async (frames) => ({
    belief: await Promise.all(frames.map(async (f) => colorize(await loadImage(base + f.belief), lut))),
    swept: await Promise.all(frames.map(async (f) => tintSwept(await loadImage(base + f.swept)))),
  });

  const hillshade = tintTerrain(await loadImage(base + "hillshade.png"));
  const ours = { hillshade, ...(await load(run.frames)) };
  const rival = { hillshade, ...(await load(run.versus.frames)) };

  el("op").textContent = `OPERATION ${run.scenario.id}`;
  el("subject").textContent = run.scenario.case_file.split(",").slice(0, 2).join(",");

  new Display(run, ours, rival).run_();
}

boot().catch((err) => { el("status").textContent = "LOAD FAILED"; console.error(err); });
