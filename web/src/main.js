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

/** Darken and cool the aerial imagery so the belief field reads over it.
 *
 * Left at full brightness the photograph competes with the data; crushed to
 * grey it stops looking like ground. This keeps it recognisably the real
 * mountain -- glaciers, timber, logging roads -- while ceding the bright end of
 * the range to the overlay. */
function tintTerrain(img) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height);
  const px = data.data;
  for (let i = 0; i < px.length; i += 4) {
    // Keep most of the photograph and only pull saturation and exposure down.
    // Crushing it further makes the ground unreadable, which defeats the point
    // of using real imagery at all.
    const lum = (px[i] * 0.30 + px[i + 1] * 0.59 + px[i + 2] * 0.11);
    const desat = 0.42;                  // toward luminance
    const exposure = 0.78;
    px[i] = (px[i] * (1 - desat) + lum * desat) * exposure * 0.96;
    px[i + 1] = (px[i + 1] * (1 - desat) + lum * desat) * exposure;
    px[i + 2] = (px[i + 2] * (1 - desat) + lum * desat) * exposure * 1.08;
    px[i + 3] = 255;
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
    ctx.globalAlpha = revising ? 0.55 : 1;
    ctx.drawImage(this.a.hillshade, x, y, size, size);
    ctx.globalAlpha = 1;

    // The conventional pane is cooled and held a step back. Same ground, but
    // the eye should know within a second which side is the argument.
    if (!this.ours) {
      ctx.save();
      ctx.globalCompositeOperation = "multiply";
      ctx.fillStyle = "rgba(88,112,152,0.46)";
      ctx.fillRect(x, y, size, size);
      ctx.globalCompositeOperation = "saturation";
      ctx.fillStyle = "rgba(128,128,128,1)";
      ctx.globalAlpha = 0.35;
      ctx.fillRect(x, y, size, size);
      ctx.restore();
    }

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
    if (showTruth) this.marker(x, y, scale, scenario.truth, "#ff5a52", "x", phase);

    const rev = this.root.querySelector(".revising");
    if (rev) rev.classList.toggle("on", revising && phase < 0.6);
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
    const heading = Math.atan2(track[i1][0] - track[i0][0], track[i1][1] - track[i0][1]);

    ctx.save();
    ctx.shadowColor = "rgba(126,231,255,0.55)";
    ctx.shadowBlur = 10;
    ctx.translate(ax, ay);
    ctx.rotate(heading);
    ctx.beginPath();
    ctx.moveTo(6, 0); ctx.lineTo(-3.6, 3.4); ctx.lineTo(-1.8, 0); ctx.lineTo(-3.6, -3.4);
    ctx.closePath();
    ctx.fillStyle = "#d9f6ff";
    ctx.fill();
    ctx.restore();

    ctx.beginPath(); ctx.arc(ax, ay, 8.5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(126,231,255,0.22)"; ctx.lineWidth = 1; ctx.stroke();
  }

  marker(x, y, scale, rc, color, kind, reveal) {
    const ctx = this.ctx;
    const px = x + (rc[1] + 0.5) * scale, py = y + (rc[0] + 0.5) * scale;

    // The subject's location arrives: a ring expands once and settles.
    if (reveal !== undefined) {
      const t = clamp01(reveal * 1.6);
      const e = easeInOut(t);
      ctx.beginPath();
      ctx.arc(px, py, 6 + (1 - e) * 26, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,90,82,${(0.5 * (1 - e)).toFixed(3)})`;
      ctx.lineWidth = 1.5; ctx.stroke();
    }

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
  constructor(demo, basemap) {
    this.demo = demo;
    this.basemap = basemap;
    this.lut = buildLut();
    this.cache = new Map();
    this.logged = new Set();
    this.lastPeriod = 1;
    this.graph = el("graph");
    this.gctx = this.graph.getContext("2d");
    this.paneB = new Pane(el("paneB"), null, demo, true);
    this.paneA = new Pane(el("paneA"), null, demo, false);
    this.started = 0;
    this.sizeGraph();
    window.addEventListener("resize", () => {
      this.paneA.resize(); this.paneB.resize(); this.sizeGraph();
    });
    this.buildCaseList();
  }

  buildCaseList() {
    const host = el("cases");
    host.innerHTML = "";
    this.demo.cases.forEach((c, i) => {
      const b = document.createElement("button");
      const found = c.ours.found;
      b.className = found ? "" : "miss";
      b.type = "button";
      b.setAttribute("role", "option");
      b.setAttribute("aria-selected", "false");
      b.innerHTML =
        `<span>${c.id} · ${c.subkind.replace(/_/g, " ")}</span>` +
        `<span class="tag">${found ? "P" + c.ours.periods_to_find : "missed"}</span>`;
      b.onclick = () => this.select(i);
      // A list you can only reach with a mouse is a list half the room cannot
      // use, and a judging table is exactly where someone drives by keyboard.
      b.onkeydown = (e) => {
        const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
        if (step === undefined) return;
        e.preventDefault();
        const next = (i + step + this.demo.cases.length) % this.demo.cases.length;
        host.children[next].focus();
        this.select(next);
      };
      host.appendChild(b);
    });
  }

  async select(i) {
    this.index = i;
    [...el("cases").children].forEach((b, k) => {
      b.classList.toggle("on", k === i);
      b.setAttribute("aria-selected", String(k === i));
    });
    const c = this.demo.cases[i];
    if (!this.cache.has(i)) {
      const load = async (frames) => ({
        belief: await Promise.all(frames.map(async (f) =>
          colorize(await loadImage("public/run/" + f.belief), this.lut))),
        swept: await Promise.all(frames.map(async (f) =>
          tintSwept(await loadImage("public/run/" + f.swept)))),
      });
      this.cache.set(i, {
        ours: { hillshade: this.basemap, ...(await load(c.ours.frames)) },
        conv: { hillshade: this.basemap, ...(await load(c.conventional.frames)) },
      });
    }
    const a = this.cache.get(i);
    this.paneB.a = a.ours; this.paneB.frames = c.ours.frames;
    this.paneA.a = a.conv; this.paneA.frames = c.conventional.frames;
    this.total = Math.max(c.ours.frames.length, c.conventional.frames.length);
    this.case_ = c;
    this.logged = new Set();
    el("log").innerHTML = "";
    this.started = performance.now();

    el("op").textContent = `OPERATION ${c.id}`;
    el("subject").textContent = c.case_file.split(",").slice(0, 2).join(",");
    const hrs = this.demo.period_hours;
    const fmt = (r) => r.found ? `${(r.periods_to_find * hrs / 24).toFixed(1)} days`
                               : "not located";
    el("mDist").textContent = `${c.truth_distance_km.toFixed(1)} km away`;
    el("mConv").textContent = fmt(c.conventional);
    el("mOurs").textContent = fmt(c.ours);
    if (c.ours.found && !c.conventional.found) {
      const budget = c.conventional.frames.length;
      el("mSaved").textContent =
        `${((budget - c.ours.periods_to_find) * hrs / 24).toFixed(1)}+ days`;
    } else if (c.ours.found && c.conventional.found) {
      el("mSaved").textContent =
        `${((c.conventional.periods_to_find - c.ours.periods_to_find) * hrs / 24).toFixed(1)} days`;
    } else {
      el("mSaved").textContent = "neither located";
    }
  }

  sizeGraph() {
    // Measure the wrapper, never the canvas: measuring an element you are
    // about to resize from that measurement is how the deck grew to 598px
    // inside a 196px row.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const host = this.graph.parentElement.getBoundingClientRect();
    const w = Math.max(host.width, 1);
    const h = Math.max(host.height - 15, 1);
    this.graph.width = Math.round(w * dpr);
    this.graph.height = Math.round(h * dpr);
    this.gctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.gw = w; this.gh = h;
  }

  clock(now) {
    let t = now - this.started;
    const frames = this.case_.ours.frames;
    for (let i = 0; i < this.total; i++) {
      const span = frames[i] && frames[i].revised ? REVISE_MS : DWELL_MS;
      if (t < span) return { period: i + 1, phase: clamp01(t / span), done: false };
      t -= span;
    }
    return { period: this.total, phase: 1, done: t > HOLD_MS };
  }

  draw(now) {
    if (!this.case_) return;
    const { period, phase, done } = this.clock(now);
    if (done) {
      // Restarting the case restarts its story. The log is append-only and
      // keyed by event id, so without clearing it the second cycle onward
      // showed the whole previous run frozen in place while the clock counted
      // up from P01 again -- the two halves of the screen disagreeing about
      // what period it was.
      this.started = now;
      this.logged = new Set();
      el("log").innerHTML = "";
      return;
    }
    const c = this.case_;
    const finished = period >= this.total;
    const sc = { ipp: c.ipp, truth: c.truth };

    this.paneB.draw(period, phase, this.demo.geo, sc, finished);
    this.paneA.draw(period, phase, this.demo.geo, sc, finished);
    this.chrome(period, phase, finished);
    this.drawGraph(period, phase);
  }

  logEvent(key, text, cls) {
    if (this.logged.has(key)) return;
    this.logged.add(key);
    const li = document.createElement("li");
    li.className = cls || "";
    li.innerHTML = `<span class="t">P${String(this.lastPeriod).padStart(2, "0")}</span>`
      + `<span>${text}</span>`;
    el("log").prepend(li);
    while (el("log").childElementCount > 14) el("log").lastElementChild.remove();
  }

  chrome(period, phase, finished) {
    const c = this.case_;
    this.lastPeriod = period;
    el("clock").textContent = `P${String(period).padStart(2, "0")}`;
    const hrs = this.demo.period_hours;
    el("elapsed").textContent = `${(period * hrs / 24).toFixed(1)} DAYS`;
    const f = c.ours.frames[Math.min(period - 1, c.ours.frames.length - 1)];
    const revising = period - 1 < c.ours.frames.length && f && f.revised;
    const status = el("status");
    // Once the subject is located the header said SEARCHING while the verdict
    // over the map said LOCATED, which is the one frame that has to be
    // unambiguous.
    const located = c.ours.found && period >= c.ours.periods_to_find;
    status.className = located ? "live found" : (revising ? "alert" : "live");
    status.textContent = located ? "LOCATED"
                       : (revising ? "PREMISE FAILING" : "SEARCHING");

    // Evidence, disconfirmation and proposals, as they land.
    for (const e of c.late_evidence || []) {
      if (e.period <= period) this.logEvent(`ev${e.period}`, e.text, "evidence");
    }
    const cf = c.ours.frames[Math.min(period - 1, c.ours.frames.length - 1)];
    if (cf && period - 1 < c.ours.frames.length) {
      if (cf.trigger_reason && cf.revised)
        this.logEvent(`tr${cf.period}`, cf.trigger_reason, "revise");
      for (const [n, nom] of (cf.nominations || []).entries())
        this.logEvent(`nm${cf.period}-${n}`, `New account: ${nom.narrative || nom.label}`, "revise");
      this.logEvent(`sw${cf.period}`, `Swept ${Math.round(cf.track_km || 0)} km, no contact.`);
    }

    // Set only the state classes. Assigning className outright used to drop
    // the `right` modifier that positions this system's verdict over the
    // second map, so both verdicts landed in the same place and the payoff
    // frame rendered as two strings interleaved on top of each other.
    const setV = (v, res) => {
      const located = res.found && period >= res.periods_to_find;
      const lost = !located && finished && phase > 0.05;
      v.classList.toggle("on", located || lost);
      v.classList.toggle("found", located);
      v.classList.toggle("lost", lost);
      if (located) {
        v.textContent = `SUBJECT LOCATED · ${(res.periods_to_find * hrs / 24).toFixed(1)} DAYS`;
      } else if (lost) {
        v.textContent = "NOT LOCATED";
      }
    };
    setV(el("verdictB"), c.ours);
    setV(el("verdictA"), c.conventional);
  }

  drawGraph(period, phase) {
    const ctx = this.gctx, w = this.gw, h = this.gh, c = this.case_;
    const pad = { l: 34, r: 14, t: 10, b: 16 };
    ctx.clearRect(0, 0, w, h);
    if (w < 40 || h < 30) return;

    const css = getComputedStyle(document.documentElement);
    const tok = (n, f) => (css.getPropertyValue(n).trim() || f);
    const X = (p) => pad.l + (p - 1) / Math.max(this.total - 1, 1) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - v * (h - pad.t - pad.b);
    const upto = period - 1 + phase;

    ctx.strokeStyle = tok("--rule", "#1c2128"); ctx.lineWidth = 1;
    ctx.font = `500 10px ${tok("--mono", "monospace")}`;
    ctx.textBaseline = "middle";
    for (const v of [0, 0.5, 1]) {
      ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
      ctx.fillStyle = tok("--label", "#828c9c");
      ctx.fillText(`${(v * 100).toFixed(0)}`, 8, Y(v));
    }

    const path = (frames) => {
      const pts = [];
      for (const f of frames) {
        if (f.period > upto) break;
        pts.push([X(f.period), Y((f.truth_percentile ?? 0) / 100)]);
      }
      return pts;
    };
    const ours = path(c.ours.frames);
    const conv = path(c.conventional.frames);

    // Area under our line. The gap between the two curves is the claim, so
    // the claim is the thing given weight rather than left as whitespace.
    if (ours.length > 1) {
      const g = ctx.createLinearGradient(0, pad.t, 0, h - pad.b);
      g.addColorStop(0, "rgba(255,176,32,0.22)");
      g.addColorStop(1, "rgba(255,176,32,0.02)");
      ctx.beginPath();
      ctx.moveTo(ours[0][0], Y(0));
      for (const [px, py] of ours) ctx.lineTo(px, py);
      ctx.lineTo(ours[ours.length - 1][0], Y(0));
      ctx.closePath();
      ctx.fillStyle = g; ctx.fill();
    }

    const stroke = (pts, colour, width) => {
      if (pts.length < 2) return;
      ctx.beginPath();
      pts.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.strokeStyle = colour; ctx.lineWidth = width;
      ctx.lineJoin = "round"; ctx.lineCap = "round";
      ctx.stroke();
    };
    stroke(conv, tok("--dim", "#8b93a1"), 1.6);
    stroke(ours, tok("--warm", "#ffb020"), 2.4);

    // Where the premise was abandoned.
    for (const f of c.ours.frames) {
      if (!f.revised || f.period > upto) continue;
      ctx.save();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = "rgba(255,90,82,0.5)"; ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(X(f.period), pad.t); ctx.lineTo(X(f.period), h - pad.b);
      ctx.stroke();
      ctx.restore();
    }

    // Leading dot on each line, so the eye knows where "now" is.
    for (const [pts, colour, r] of [[conv, tok("--dim", "#8b93a1"), 2.2],
                                    [ours, tok("--warm", "#ffb020"), 3]]) {
      if (!pts.length) continue;
      const [px, py] = pts[pts.length - 1];
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = colour; ctx.fill();
    }
  }

  run_() {
    const tick = (now) => { this.draw(now); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
}

async function boot() {
  const demo = await (await fetch("public/run/demo.json")).json();
  const basemap = tintTerrain(await loadImage("public/run/imagery.jpg"));
  const d = new Display(demo, basemap);
  // Exposed so the final frame can be inspected directly rather than by
  // waiting for the clock to reach it.
  window.display = d;
  await d.select(0);
  d.run_();
}

boot().catch((err) => { el("status").textContent = "LOAD FAILED"; console.error(err); });
