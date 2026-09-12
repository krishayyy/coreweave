// Search-loop display.
//
// Canvas2D with no dependencies, deliberately: a live demo should not need a CDN
// to be reachable, and every transition here is hand-controlled. The rule the
// whole file follows is that nothing ever pops -- belief fields cross-fade,
// hypothesis weights ease, and the clock counts rather than jumps. Discrete
// updates are what make software read as cheap.

// Alpha stays near zero across the bottom third. Probability mass has a very
// long tail, and painting the tail at all turns the whole map into a glow --
// the field stops reading as a concentration and starts reading as a texture.
const PALETTE = [
  [0.00, [0, 0, 0, 0]],
  [0.30, [18, 16, 52, 0]],
  [0.44, [66, 36, 104, 70]],
  [0.58, [156, 60, 82, 165]],
  [0.74, [236, 116, 40, 224]],
  [0.88, [255, 186, 48, 246]],
  [1.00, [255, 248, 222, 255]],
];

const DWELL_MS = 1500;        // ordinary period
const REVISE_MS = 4200;       // period where the premise is abandoned
const FADE_FRAC = 0.55;       // portion of a dwell spent cross-fading

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

/** Map an 8-bit mask through the palette into its own canvas, once at load. */
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

/** Shade relief into a cool dark ramp; raw greyscale is far too bright. */
function tintTerrain(img) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) {
    const v = px[i] / 255;
    const s = Math.pow(v, 1.45);            // deepen the shadows
    px[i] = 16 + s * 62;
    px[i + 1] = 20 + s * 70;
    px[i + 2] = 28 + s * 84;                // hold a cool cast in the highlights
    px[i + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
  return c;
}

/** Swept ground: cool, dark, and persistent -- the scar tissue of the search. */
function tintSwept(img) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) {
    const on = px[i] > 127;
    px[i] = 36; px[i + 1] = 58; px[i + 2] = 78;
    px[i + 3] = on ? 132 : 0;
  }
  ctx.putImageData(data, 0, 0);
  return c;
}

class Display {
  constructor(run, assets) {
    this.run = run;
    this.a = assets;
    this.canvas = el("map");
    this.ctx = this.canvas.getContext("2d");
    this.weights = new Map();       // eased hypothesis weights
    this.logged = new Set();
    this.started = 0;
    this.resize();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width; this.h = r.height;
  }

  /** Letterbox the square field into the stage. */
  fit() {
    const size = Math.min(this.w, this.h);
    return { x: (this.w - size) / 2, y: (this.h - size) / 2, size };
  }

  /** Where in the timeline are we? Returns frame index and intra-frame phase. */
  clock(now) {
    let t = now - this.started;
    for (let i = 0; i < this.run.frames.length; i++) {
      const span = this.run.frames[i].revised ? REVISE_MS : DWELL_MS;
      if (t < span || i === this.run.frames.length - 1) {
        return { i, phase: clamp01(t / span), last: i === this.run.frames.length - 1 };
      }
      t -= span;
    }
    return { i: this.run.frames.length - 1, phase: 1, last: true };
  }

  draw(now) {
    const { i, phase, last } = this.clock(now);
    const frame = this.run.frames[i];
    const next = this.run.frames[Math.min(i + 1, this.run.frames.length - 1)];
    const { x, y, size } = this.fit();
    const ctx = this.ctx;
    const scale = size / this.run.geo.cols;

    // Cross-fade belief between this frame and the next across the back half of
    // the dwell, so the field is always in motion and never cuts.
    const fade = easeInOut(clamp01((phase - (1 - FADE_FRAC)) / FADE_FRAC));

    // On the revision period the old field is dissolved out before the new one
    // blooms, and everything else is dimmed so the map is the only thing moving.
    const revising = frame.revised;
    const dip = revising ? 1 - 0.42 * Math.sin(Math.PI * clamp01(phase / 0.55)) : 1;

    ctx.clearRect(0, 0, this.w, this.h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    // Terrain. Painted dark and cool rather than as raw greyscale: a bright
    // hillshade competes with the belief field and reads as a hiking map
    // instead of an instrument.
    ctx.fillStyle = "#0d1015";
    ctx.fillRect(x, y, size, size);
    ctx.globalAlpha = 0.34 * (revising ? 0.5 : 1);
    ctx.drawImage(this.a.hillshade, x, y, size, size);
    ctx.globalAlpha = 1;

    // swept ground
    ctx.globalAlpha = 0.85;
    ctx.globalCompositeOperation = "source-over";
    ctx.drawImage(this.a.swept[i], x, y, size, size);
    if (fade > 0) { ctx.globalAlpha = 0.85 * fade; ctx.drawImage(this.a.swept[Math.min(i + 1, this.a.swept.length - 1)], x, y, size, size); }

    // belief, additively so overlapping mass reads as brighter rather than flatter
    ctx.globalCompositeOperation = "screen";
    ctx.globalAlpha = (1 - fade) * dip;
    ctx.drawImage(this.a.belief[i], x, y, size, size);
    ctx.globalAlpha = fade * dip;
    ctx.drawImage(this.a.belief[Math.min(i + 1, this.a.belief.length - 1)], x, y, size, size);

    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;

    // The aircraft, with a short trail behind it. Stroking the entire serpentine
    // at once reads as scan lines across the map rather than as something
    // flying, so only the recent tail is drawn.
    const track = frame.track || [];
    if (track.length > 1) {
      const head = clamp01(phase / 0.86) * (track.length - 1);
      const tail = Math.max(0, Math.floor(head) - 14);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let k = tail; k < Math.floor(head); k++) {
        const [r0, c0] = track[k];
        const [r1, c1] = track[k + 1];
        const age = (k - tail) / Math.max(Math.floor(head) - tail, 1);
        ctx.beginPath();
        ctx.moveTo(x + (c0 + 0.5) * scale, y + (r0 + 0.5) * scale);
        ctx.lineTo(x + (c1 + 0.5) * scale, y + (r1 + 0.5) * scale);
        ctx.strokeStyle = `rgba(126,231,255,${(0.42 * age).toFixed(3)})`;
        ctx.lineWidth = 1.4;
        ctx.stroke();
      }
      const i0 = Math.floor(head), i1 = Math.min(i0 + 1, track.length - 1);
      const f = head - i0;
      const ar = lerp(track[i0][0], track[i1][0], f);
      const ac = lerp(track[i0][1], track[i1][1], f);
      const ax = x + (ac + 0.5) * scale, ay = y + (ar + 0.5) * scale;
      ctx.beginPath();
      ctx.arc(ax, ay, 2.6, 0, Math.PI * 2);
      ctx.fillStyle = "#bff2ff";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(ax, ay, 6.5, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(126,231,255,0.30)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    this.marker(x, y, scale, this.run.scenario.ipp, "#7ee7ff", "cross");
    if (last && this.run.result.found) {
      this.marker(x, y, scale, this.run.scenario.truth, "#ff5a52", "x");
    }

    this.chrome(i, frame, next, phase, last);
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

  chrome(i, frame, next, phase, last) {
    el("clock").textContent = `P${String(frame.period).padStart(2, "0")}`;

    const status = el("status");
    status.className = frame.revised ? "alert" : "live";
    status.textContent = frame.revised ? "PREMISE FAILING" : "SEARCHING";
    el("revising").classList.toggle("on", frame.revised && phase < 0.6);

    // Hypothesis weights ease toward their target rather than snapping.
    const list = el("hyp-list");
    const target = new Map(frame.hypotheses.map((h) => [h.label, h]));
    for (const [label, h] of target) {
      const cur = this.weights.get(label) ?? 0;
      this.weights.set(label, lerp(cur, h.weight, 0.08));
    }
    const rows = [...target.values()].slice(0, 5);
    if (list.childElementCount !== rows.length) {
      list.innerHTML = rows.map(() =>
        '<li><span class="w"></span><span class="track"><span class="fill"></span></span><span class="label"></span></li>'
      ).join("");
    }
    // Bars are scaled against the leader rather than against 1.0: posteriors
    // here sit well below 0.5, and an absolute scale renders every bar as a
    // stub that shows nothing.
    const top = Math.max(...rows.map((h) => this.weights.get(h.label) ?? 0), 1e-6);
    rows.forEach((h, k) => {
      const li = list.children[k];
      const w = this.weights.get(h.label) ?? 0;
      li.classList.toggle("lead", k === 0);
      li.querySelector(".w").textContent = w.toFixed(2);
      const fill = li.querySelector(".fill");
      fill.style.width = `${((w / top) * 100).toFixed(1)}%`;
      fill.classList.toggle("nominated", h.origin !== "library");
      li.querySelector(".label").textContent = h.label;
    });

    // Evidence log, append-only.
    const log = el("log-list");
    const add = (key, text, cls) => {
      if (this.logged.has(key)) return;
      this.logged.add(key);
      const li = document.createElement("li");
      li.className = cls || "";
      li.innerHTML = `<span class="t">P${String(frame.period).padStart(2, "0")}</span>${text}`;
      log.prepend(li);
      while (log.childElementCount > 6) log.lastElementChild.remove();
    };
    for (const e of this.run.scenario.late_evidence || []) {
      if (e.period <= frame.period) add(`ev${e.period}`, e.text, "new");
    }
    if (frame.trigger_fired) add(`tr${frame.period}`, frame.trigger_reason, "rev");
    for (const [n, nom] of (frame.nominations || []).entries()) {
      add(`nm${frame.period}-${n}`, `New account: ${nom.narrative || nom.label}`, "rev");
    }
    add(`sw${frame.period}`, `Swept ${frame.track_km.toFixed(0)} km, no contact.`);

    const verdict = el("verdict");
    if (last && phase > 0.3) {
      verdict.classList.add("on");
      verdict.textContent = this.run.result.found
        ? `SUBJECT LOCATED — PERIOD ${this.run.result.periods_to_find} — `
          + `${this.run.scenario.truth_distance_km.toFixed(1)} KM FROM THE PLANNING POINT`
        : "SUBJECT NOT LOCATED WITHIN THE PERIOD BUDGET";
    }
  }

  run_() {
    this.started = performance.now();
    const tick = (now) => { this.draw(now); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
}

async function boot() {
  const run = await (await fetch("public/run/run.json")).json();
  const lut = buildLut();
  const base = "public/run/";

  const [hillshade, ...rest] = await Promise.all([
    loadImage(base + "hillshade.png"),
    ...run.frames.map((f) => loadImage(base + f.belief)),
    ...run.frames.map((f) => loadImage(base + f.swept)),
  ]);
  const n = run.frames.length;
  const assets = {
    hillshade: tintTerrain(hillshade),
    belief: rest.slice(0, n).map((img) => colorize(img, lut)),
    swept: rest.slice(n).map(tintSwept),
  };

  el("op").textContent = `OPERATION ${run.scenario.id}`;
  el("subject").textContent = run.scenario.case_file.split(",").slice(0, 2).join(",");

  new Display(run, assets).run_();
}

boot().catch((err) => {
  el("status").textContent = "LOAD FAILED";
  console.error(err);
});
