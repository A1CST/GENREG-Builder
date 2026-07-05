// Animation page — dataset preview grid + the Animation Evo client.
//
// Dataset: /api/animations (10 clips, 24 frames of 64x64 grayscale, base64).
// Training: WS /animevo onto animation_evo.HUB (mutation-only evolutionary
// shape classifier). The grid doubles as the live-prediction view: each clip's
// caption shows the champion's shape call for the frame currently displayed.
(function () {
  "use strict";

  const FPS = 12;
  const NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const SHAPE_COLORS = ["#78c8ff", "#ffaa5a", "#8ceb8c", "#f078c8", "#faf078",
                        "#b49aff", "#5ae0d2", "#ff8c8c", "#c8b478", "#9ab4c8"];

  // ── state ──────────────────────────────────────────────────────────
  let ws = null;
  let clips = null;          // /api/animations payload
  let drawers = [];          // per-clip frame painters
  let predEls = [];          // per-clip prediction <span>s
  let cells = [];            // per-clip <figure> (for current-clip highlight)
  let frame = 0;             // global playback frame counter
  let started = null;        // "started" event (shapes, clips, labels, ...)
  let predByClip = [];       // predByClip[ci] = champion's 24 preds for clip ci
  const fit = { clip: [], roll: [] };   // per-clip fitness + rolling epoch avg

  // ── dataset grid (also the live-prediction display) ───────────────
  function decodeClip(clip) {
    const raw = atob(clip.data);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return bytes;
  }

  function buildCell(clip, grid) {
    const cell = document.createElement("figure");
    cell.className = "anim-cell";
    const canvas = document.createElement("canvas");
    canvas.width = clip.size;
    canvas.height = clip.size;
    const cap = document.createElement("figcaption");
    const name = document.createElement("span");
    name.className = "anim-shape";
    name.textContent = clip.shape;                 // the SHAPE is the label
    const path = document.createElement("span");
    path.textContent = ` · ${clip.name} path`;     // motion path, secondary
    const predEl = document.createElement("span");
    predEl.className = "anim-pred";
    cap.appendChild(name);
    cap.appendChild(path);
    cap.appendChild(document.createElement("br"));
    cap.appendChild(predEl);
    cell.appendChild(canvas);
    cell.appendChild(cap);
    grid.appendChild(cell);
    predEls.push(predEl);
    cells.push(cell);

    const ctx = canvas.getContext("2d");
    const bytes = decodeClip(clip);
    const px = clip.size * clip.size;
    const img = ctx.createImageData(clip.size, clip.size);
    for (let i = 3; i < img.data.length; i += 4) img.data[i] = 255;

    return function drawFrame(f) {
      const off = (f % clip.frames) * px;
      for (let i = 0; i < px; i++) {
        const v = bytes[off + i];
        img.data[i * 4] = v;
        img.data[i * 4 + 1] = v;
        img.data[i * 4 + 2] = v;
      }
      ctx.putImageData(img, 0, 0);
    };
  }

  function updatePredictions() {
    if (!started) return;
    const fpc = started.frames_per_clip;
    for (let ci = 0; ci < predEls.length && ci < started.clips.length; ci++) {
      const clipPred = predByClip[ci];
      const el = predEls[ci];
      if (!clipPred) { el.textContent = ""; el.className = "anim-pred"; continue; }
      const f = frame % fpc;                      // frame shown in this tile
      const p = clipPred[f];
      const truth = started.labels[ci * fpc];     // clip's shape (constant)
      el.textContent = "→ " + started.shapes[p];
      el.className = "anim-pred " + (p === truth ? "ok" : "bad");
    }
  }

  async function initGrid() {
    const grid = $("anim-grid");
    try {
      const r = await fetch("/api/animations");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      clips = await r.json();
    } catch (err) {
      grid.textContent = `failed to load animations: ${err.message}`;
      return;
    }
    drawers = clips.map((c) => buildCell(c, grid));
    predByClip = new Array(clips.length).fill(null);
    drawers.forEach((d) => d(0));
    setInterval(() => {
      frame++;
      drawers.forEach((d) => d(frame));
      updatePredictions();
    }, 1000 / FPS);
  }

  // ── websocket ──────────────────────────────────────────────────────
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/animevo`);
    ws.onopen = () => setConn(true);
    ws.onclose = () => { setConn(false); setTimeout(connect, 2000); };
    ws.onmessage = (m) => { try { handle(JSON.parse(m.data)); } catch (e) {} };
  }

  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

  function setConn(ok) {
    $("av-dot").className = "dot" + (ok ? " ok" : " bad");
    $("av-conn").textContent = ok ? "connected" : "disconnected";
  }

  function setRunning(running) {
    $("av-start").disabled = running;
    $("av-stop").disabled = !running;
  }

  function handle(ev) {
    switch (ev.type) {
      case "job":
        setRunning(ev.running);
        break;
      case "started":
        started = ev;
        predByClip = new Array(ev.clips.length).fill(null);
        fit.clip = []; fit.roll = [];
        cells.forEach((c) => c.classList.remove("current"));
        $("st-params").textContent = ev.params_per_genome.toLocaleString();
        $("st-result").textContent = "—";
        $("st-clip").textContent = "—";
        $("st-roll").textContent = "—";
        $("st-mean").textContent = "—";
        $("av-report").hidden = true;
        $("av-status").textContent =
          `training — one clip/generation, shuffled order, frames in sequence · ` +
          `pop ${ev.pop}, enc ${ev.enc}, hid ${ev.hidden}, ${ev.shapes.length} shapes`;
        setRunning(true);
        break;
      case "gen":
        fit.clip.push(ev.clip_acc); fit.roll.push(ev.roll_acc);
        predByClip[ev.clip] = ev.pred;            // champion's calls on this clip
        cells.forEach((c, i) => c.classList.toggle("current", i === ev.clip));
        $("st-gen").textContent = `${ev.gen} / ${ev.generations}`;
        $("st-clip").textContent =
          `${ev.clip_name} · ${(ev.clip_acc * 100).toFixed(0)}%`;
        $("st-roll").textContent = ev.roll_acc.toFixed(3);
        $("st-mean").textContent = ev.pop_mean.toFixed(3);
        updatePredictions();
        drawFitness();
        break;
      case "done":
        cells.forEach((c) => c.classList.remove("current"));
        if (started && ev.pred) {                 // fill every tile from champion
          const fpc = started.frames_per_clip;
          predByClip = started.clips.map((_, ci) =>
            ev.pred.slice(ci * fpc, (ci + 1) * fpc));
          updatePredictions();
        }
        $("st-result").textContent = `${ev.reason} · ${(ev.full_acc * 100).toFixed(1)}%`;
        $("av-status").textContent =
          `${ev.reason} at gen ${ev.gen} — champion accuracy over ALL clips ` +
          `${ev.full_acc.toFixed(4)} (${ev.seconds}s)`;
        setRunning(false);
        report(ev);
        break;
      case "error":
        $("av-status").textContent = `error: ${ev.message}`;
        setRunning(false);
        break;
    }
  }

  // ── fitness chart ──────────────────────────────────────────────────
  function drawFitness() {
    const svg = $("av-fitness");
    if (!svg || !fit.clip.length) return;
    const W = svg.clientWidth || 640, H = 200, m = { t: 10, r: 10, b: 22, l: 44 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const n = fit.clip.length;
    const X = (i) => m.l + (n < 2 ? 0 : (iw * i) / (n - 1));
    const Y = (v) => m.t + ih - ih * v;               // accuracy in [0,1]

    seg(svg, m.l, m.t, m.l, m.t + ih, "var(--border)", 1);
    seg(svg, m.l, m.t + ih, m.l + iw, m.t + ih, "var(--border)", 1);
    for (const v of [0.25, 0.5, 0.75, 1.0]) {
      const y = Y(v);
      seg(svg, m.l, y, m.l + iw, y, "var(--tlm-idle)", 1);
      text(svg, m.l - 6, y + 3, v.toFixed(2), "end");
    }
    text(svg, m.l + iw / 2, H - 4, "generation (one clip each)", "middle");

    line(svg, fit.clip, X, Y, "var(--tlm-s2)");       // noisy per-clip fitness
    line(svg, fit.roll, X, Y, "var(--tlm-s1)");       // rolling epoch average
  }

  function line(svg, ys, X, Y, color) {
    const p = document.createElementNS(NS, "polyline");
    p.setAttribute("points", ys.map((v, i) => `${X(i)},${Y(v)}`).join(" "));
    p.setAttribute("fill", "none");
    p.setAttribute("stroke", color);
    p.setAttribute("stroke-width", "1.6");
    svg.appendChild(p);
  }

  function seg(svg, x1, y1, x2, y2, color, w) {
    const l = document.createElementNS(NS, "line");
    l.setAttribute("x1", x1); l.setAttribute("y1", y1);
    l.setAttribute("x2", x2); l.setAttribute("y2", y2);
    l.setAttribute("stroke", color); l.setAttribute("stroke-width", w);
    svg.appendChild(l);
  }

  function text(svg, x, y, str, anchor) {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.setAttribute("text-anchor", anchor);
    t.setAttribute("fill", "var(--muted)");
    t.setAttribute("font-size", "9");
    t.textContent = str;
    svg.appendChild(t);
  }

  // ── final champion report ──────────────────────────────────────────
  function report(ev) {
    if (!started || !ev.confusion) return;
    $("av-report").hidden = false;
    $("av-report-sub").textContent =
      `best genome after ${ev.gen} generations — ${(ev.full_acc * 100).toFixed(1)}% across all clips ` +
      `(measured clip-by-clip)`;

    // confusion table
    const shapes = started.shapes;
    const tbl = $("av-confusion");
    tbl.innerHTML = "";
    const hr = tbl.insertRow();
    hr.appendChild(document.createElement("th"));
    for (const s of shapes) {
      const th = document.createElement("th");
      th.textContent = s;
      hr.appendChild(th);
    }
    ev.confusion.forEach((row, i) => {
      const tr = tbl.insertRow();
      const th = document.createElement("th");
      th.textContent = shapes[i];
      tr.appendChild(th);
      row.forEach((v, j) => {
        const td = tr.insertCell();
        td.textContent = v || "";
        if (v) td.style.color = i === j ? "var(--tlm-s1, #78c8ff)" : "#f07878";
      });
    });

    // per-clip accuracy table
    const pc = $("av-perclip");
    pc.innerHTML = "";
    started.clips.forEach((c, ci) => {
      const tr = pc.insertRow();
      tr.insertCell().textContent = c;
      const shape = shapes[started.labels[ci * started.frames_per_clip]];
      tr.insertCell().textContent = shape;
      const td = tr.insertCell();
      const a = ev.per_clip_acc[ci];
      td.textContent = (a * 100).toFixed(1) + "%";
      td.style.color = a >= 0.99 ? "var(--tlm-s1, #78c8ff)" : a < 0.5 ? "#f07878" : "";
    });

    // encoder PCA scatter
    const cv = $("av-pca");
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#05070a";
    ctx.fillRect(0, 0, cv.width, cv.height);
    ev.encoder_2d.forEach((p, i) => {
      const lab = started.labels[i];
      ctx.fillStyle = SHAPE_COLORS[lab % SHAPE_COLORS.length];
      const x = cv.width / 2 + p[0] * (cv.width / 2 - 8);
      const y = cv.height / 2 + p[1] * (cv.height / 2 - 8);
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    });
    const legend = $("av-pca-legend");
    legend.innerHTML = "";
    shapes.forEach((s, i) => {
      const chip = document.createElement("span");
      chip.className = "tlm-chip";
      const dot = document.createElement("i");
      dot.style.background = SHAPE_COLORS[i % SHAPE_COLORS.length];
      chip.appendChild(dot);
      chip.appendChild(document.createTextNode(s));
      legend.appendChild(chip);
    });
  }

  // ── controls ───────────────────────────────────────────────────────
  function wireControls() {
    $("av-start").addEventListener("click", () => {
      send({
        op: "start",
        pop: +$("av-pop").value,
        enc: +$("av-enc").value,
        hidden: +$("av-hidden").value,
        elite_frac: +$("av-elite").value,
        mutation: +$("av-mut").value,
        max_gens: +$("av-maxgens").value,
        patience: +$("av-patience").value,
        seed: +$("av-seed").value,
      });
      $("av-status").textContent = "starting…";
    });
    $("av-stop").addEventListener("click", () => send({ op: "stop" }));
  }

  function init() {
    initGrid();
    wireControls();
    connect();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
