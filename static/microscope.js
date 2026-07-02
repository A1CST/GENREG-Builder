// GENREG microscope: a magnified "lab" view of a genome — weight matrices as a
// heatmap (value -> diverging color), per-neuron saturation bars, a hover
// readout for individual weights, and a live mutation loop so you can watch the
// genome change in real time.
//
// This is currently ILLUSTRATIVE: it evolves a synthetic genome with the same
// relative, self-adapting mutation the engine uses (w += N(0,1)*ms*(|w|+eps)),
// so the motion is representative. Call GENREG.scope.setGenome(layers) to feed a
// real genome once training is wired up.

(() => {
  const canvas = document.getElementById("scope-canvas");
  const stage = document.getElementById("scope-stage");
  const readout = document.getElementById("scope-readout");
  const genBadge = document.getElementById("scope-gen");
  const btnToggle = document.getElementById("scope-toggle");
  const btnStep = document.getElementById("scope-step");
  if (!canvas || !stage) return;   // markup absent -> skip, never throw
  const ctx = canvas.getContext("2d");

  const EPS = 1e-3;
  const STEP_MS = 140;             // ~7 mutation steps/sec — "realtime, but slower"

  // -- genome model --------------------------------------------------------
  // Each layer: W is a Float64Array of shape [rows(out) x cols(in)], plus the
  // most recent activation per output neuron (for saturation).
  function gauss() {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function makeLayer(rows, cols) {
    const W = new Float64Array(rows * cols);
    for (let i = 0; i < W.length; i++) W[i] = gauss() * 0.6;
    return {
      rows, cols, W,
      ms: 0.05,                          // self-adapting mutation scale (a gene)
      act: new Float64Array(rows),       // last activations (for saturation)
    };
  }

  // A small illustrative net: 16 -> 24 -> 12 -> 4.
  let genome = { layers: [makeLayer(24, 16), makeLayer(12, 24), makeLayer(4, 12)], gen: 0 };
  let input = null;                      // fixed input so motion reflects weights, not noise
  function seedInput(n) { input = new Float64Array(n); for (let i = 0; i < n; i++) input[i] = gauss(); }
  seedInput(genome.layers[0].cols);

  // forward pass (tanh); records per-neuron activation for saturation readout
  function forward() {
    let x = input;
    for (const L of genome.layers) {
      const y = new Float64Array(L.rows);
      for (let r = 0; r < L.rows; r++) {
        let s = 0;
        const base = r * L.cols;
        for (let c = 0; c < L.cols; c++) s += L.W[base + c] * x[c];
        y[r] = Math.tanh(s);
      }
      L.act = y;
      x = y;
    }
  }

  // one relative, self-adapting mutation step (mirrors the engine's law)
  function mutate() {
    for (const L of genome.layers) {
      // the search rate is itself a gene: nudge ms, keep it sane
      L.ms *= Math.exp(0.08 * gauss());
      L.ms = Math.min(0.25, Math.max(0.01, L.ms));
      for (let i = 0; i < L.W.length; i++) {
        L.W[i] += gauss() * L.ms * (Math.abs(L.W[i]) + EPS);
      }
    }
    genome.gen++;
    forward();
  }

  // -- color maps ----------------------------------------------------------
  const CENTER = [22, 27, 34];
  const POS = [240, 136, 62];   // warm  (positive weight)
  const NEG = [78, 161, 255];   // blue  (negative weight)
  function lerp(a, b, t) { return Math.round(a + (b - a) * t); }
  function valueColor(v, vmax) {
    const t = Math.max(-1, Math.min(1, v / (vmax || 1)));
    const end = t >= 0 ? POS : NEG;
    const a = Math.abs(t);
    return `rgb(${lerp(CENTER[0], end[0], a)},${lerp(CENTER[1], end[1], a)},${lerp(CENTER[2], end[2], a)})`;
  }
  function satColor(s) {
    // 0 (calm, green) -> 1 (saturated, red)
    const g = [63, 185, 80], r = [248, 81, 73];
    return `rgb(${lerp(g[0], r[0], s)},${lerp(g[1], r[1], s)},${lerp(g[2], r[2], s)})`;
  }

  // -- rendering -----------------------------------------------------------
  let cells = [];   // hit-test rects: {x,y,cell,rows,cols,li}
  let hover = null; // {li,r,c}

  function stats() {
    let sum = 0, n = 0, max = 0, satN = 0, satTot = 0;
    for (const L of genome.layers) {
      for (let i = 0; i < L.W.length; i++) { const a = Math.abs(L.W[i]); sum += a; max = Math.max(max, a); n++; }
      for (let r = 0; r < L.rows; r++) { satTot++; if (Math.abs(L.act[r]) > 0.9) satN++; }
    }
    return { mean: sum / Math.max(1, n), max, satPct: satN / Math.max(1, satTot) };
  }

  function render() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // lab background
    ctx.fillStyle = "#07090d";
    ctx.fillRect(0, 0, w, h);

    const padX = 12, padTop = 10, gap = 22, titleH = 14, satW = 14, satGap = 8;
    const layers = genome.layers;
    const maxCols = layers.reduce((m, L) => Math.max(m, L.cols), 1);
    const totalRows = layers.reduce((s, L) => s + L.rows, 0);
    const availW = w - padX * 2 - satW - satGap;
    const availH = h - padTop - gap * layers.length - titleH * layers.length;
    const cell = Math.max(2, Math.floor(Math.min(availW / maxCols, availH / totalRows)));

    const s = stats();
    const vmax = Math.max(0.35, s.max * 0.9);

    cells = [];
    const watchSet = new Set(watch.map((x) => x.key));
    let y = padTop;
    ctx.textBaseline = "alphabetic";
    for (let li = 0; li < layers.length; li++) {
      const L = layers[li];
      // title
      ctx.fillStyle = "rgba(125,135,148,0.85)";
      ctx.font = '11px "Cascadia Code", Consolas, monospace';
      ctx.textAlign = "left";
      ctx.fillText(`L${li + 1}  ${L.cols}→${L.rows}   ms ${L.ms.toFixed(3)}`, padX, y + 10);
      y += titleH;

      const gx = padX, gy = y;
      // weight heatmap
      for (let r = 0; r < L.rows; r++) {
        for (let c = 0; c < L.cols; c++) {
          ctx.fillStyle = valueColor(L.W[r * L.cols + c], vmax);
          ctx.fillRect(gx + c * cell, gy + r * cell, cell - (cell > 3 ? 1 : 0), cell - (cell > 3 ? 1 : 0));
        }
      }
      // saturation bar per output neuron (one row = one neuron)
      const sx = gx + L.cols * cell + satGap;
      for (let r = 0; r < L.rows; r++) {
        const sat = Math.abs(L.act[r]);
        ctx.fillStyle = satColor(sat);
        const bw = Math.max(2, satW * (0.35 + 0.65 * sat));
        ctx.fillRect(sx, gy + r * cell, bw, cell - (cell > 3 ? 1 : 0));
      }

      // tracked-neuron markers: an accent tick left of the row + a row outline
      ctx.lineWidth = 1;
      for (let r = 0; r < L.rows; r++) {
        if (!watchSet.has(`${li}:${r}`)) continue;
        ctx.fillStyle = "#4ea1ff";
        ctx.fillRect(gx - 6, gy + r * cell, 3, Math.max(2, cell - (cell > 3 ? 1 : 0)));
        ctx.strokeStyle = "rgba(78,161,255,0.7)";
        ctx.strokeRect(gx - 0.5, gy + r * cell + 0.5, L.cols * cell + satGap + satW, cell - 1);
      }

      // hovered-cell highlight
      if (hover && hover.li === li) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.strokeRect(gx + hover.c * cell + 0.5, gy + hover.r * cell + 0.5, cell - 1, cell - 1);
      }

      cells.push({ x: gx, y: gy, cell, rows: L.rows, cols: L.cols, li, satX: sx, satW });
      y = gy + L.rows * cell + gap;
    }

    if (genBadge) {
      genBadge.textContent = (external ? "● live · gen " : "demo · gen ") + genome.gen;
      genBadge.style.color = external ? "#3fb950" : "";
    }
    if (readout && !hover) {
      readout.textContent = `μ|w| ${s.mean.toFixed(3)}   max|w| ${s.max.toFixed(2)}   saturated ${(s.satPct * 100).toFixed(0)}%`;
    }
    updateWatchValues();   // keep the tracked-neuron list live
  }

  // -- hover readout -------------------------------------------------------
  function hitTest(mx, my) {
    for (const g of cells) {
      if (mx >= g.x && mx < g.x + g.cols * g.cell && my >= g.y && my < g.y + g.rows * g.cell) {
        return { li: g.li, r: Math.floor((my - g.y) / g.cell), c: Math.floor((mx - g.x) / g.cell) };
      }
    }
    return null;
  }
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    hover = hitTest(e.clientX - rect.left, e.clientY - rect.top);
    if (hover) {
      const L = genome.layers[hover.li];
      const v = L.W[hover.r * L.cols + hover.c];
      const sat = Math.abs(L.act[hover.r]);
      readout.textContent = `L${hover.li + 1} w[${hover.r},${hover.c}] = ${v >= 0 ? "+" : ""}${v.toFixed(3)}   neuron sat ${sat.toFixed(2)}`;
    }
    if (!playing) render();   // when paused, refresh so the highlight tracks
  });
  canvas.addEventListener("mouseleave", () => { hover = null; if (!playing) render(); });

  // -- tracked-neuron watch list -------------------------------------------
  const watch = [];                 // {key, li, r}
  const watchEls = new Map();       // key -> value <span>
  const listEl = document.getElementById("scope-watch-list");
  const countEl = document.getElementById("scope-watch-count");
  const emptyEl = document.getElementById("scope-watch-empty");

  function neuronAt(mx, my) {
    for (const g of cells) {
      if (my < g.y || my >= g.y + g.rows * g.cell) continue;
      const inGrid = mx >= g.x && mx < g.x + g.cols * g.cell;
      const inSat = mx >= g.satX && mx < g.satX + g.satW;
      if (inGrid || inSat) return { li: g.li, r: Math.floor((my - g.y) / g.cell) };
    }
    return null;
  }

  function neuronStats(li, r) {
    const L = genome.layers[li];
    if (!L || r >= L.rows) return null;
    let sum = 0; const base = r * L.cols;
    for (let c = 0; c < L.cols; c++) sum += Math.abs(L.W[base + c]);
    return { sat: Math.abs(L.act[r]), meanW: sum / L.cols };
  }

  function updateWatchValues() {
    for (const wch of watch) {
      const el = watchEls.get(wch.key);
      if (!el) continue;
      const s = neuronStats(wch.li, wch.r);
      el.textContent = s ? `sat ${s.sat.toFixed(2)}  μ|w| ${s.meanW.toFixed(2)}` : "—";
    }
  }

  function buildWatchList() {
    if (!listEl) return;
    listEl.innerHTML = "";
    watchEls.clear();
    for (const wch of watch) {
      const li = document.createElement("li");
      li.className = "scope-watch-item";
      const label = document.createElement("span");
      label.className = "w-label";
      label.textContent = `L${wch.li + 1} · n${wch.r}`;
      const val = document.createElement("span");
      val.className = "w-val";
      const rm = document.createElement("button");
      rm.className = "w-remove"; rm.title = "Stop tracking"; rm.textContent = "×";
      rm.addEventListener("click", () => removeWatch(wch.key));
      li.append(label, val, rm);
      listEl.appendChild(li);
      watchEls.set(wch.key, val);
    }
    if (countEl) countEl.textContent = String(watch.length);
    if (emptyEl) emptyEl.hidden = watch.length > 0;
    updateWatchValues();
  }

  function addWatch(li, r) {
    const key = `${li}:${r}`;
    if (watch.some((x) => x.key === key)) return;
    watch.push({ key, li, r });
    buildWatchList();
    render();
  }

  function removeWatch(key) {
    const i = watch.findIndex((x) => x.key === key);
    if (i < 0) return;
    watch.splice(i, 1);
    buildWatchList();
    render();
  }

  canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const n = neuronAt(e.clientX - rect.left, e.clientY - rect.top);
    if (n) addWatch(n.li, n.r);
  });

  // -- animation loop ------------------------------------------------------
  // `external` = a real genome is being pushed in by the trainer; pause the
  // illustrative self-mutation so we display live training weights, not a demo.
  let playing = true, external = false, last = 0, acc = 0;
  function loop(t) {
    if (playing && !external) {
      if (last) acc += t - last;
      last = t;
      let changed = false;
      while (acc >= STEP_MS) { mutate(); acc -= STEP_MS; changed = true; }
      if (changed) render();
    } else {
      last = t;
    }
    requestAnimationFrame(loop);
  }

  function setPlaying(on) {
    playing = on;
    acc = 0; last = 0;
    if (btnToggle) btnToggle.textContent = on ? "Pause" : "Play";
  }
  if (btnToggle) btnToggle.addEventListener("click", () => setPlaying(!playing));
  if (btnStep) btnStep.addEventListener("click", () => { setPlaying(false); mutate(); render(); });

  new ResizeObserver(render).observe(stage);
  forward();
  buildWatchList();
  render();
  requestAnimationFrame(loop);

  // Convert an incoming genome (engine format: layers of {rows, cols, w:[...]})
  // into the microscope's internal layer objects.
  function adoptLayers(layers, keepWatch) {
    genome = {
      layers: layers.map((L) => ({
        rows: L.rows, cols: L.cols,
        W: L.W ? L.W : Float64Array.from(L.w || []),
        ms: L.ms || 0.05,
        act: new Float64Array(L.rows),
      })),
      gen: 0,
    };
    seedInput(genome.layers[0].cols);
    if (!keepWatch) { watch.length = 0; buildWatchList(); }
    forward();
    render();
  }

  // public hook for the trainer to stream real genomes
  window.GENREG = window.GENREG || {};
  window.GENREG.scope = {
    // stream a live training genome; keeps the watch list across generations
    setGenome(layers, gen) {
      external = true;
      const sameShape = genome.layers.length === layers.length &&
        genome.layers.every((L, i) => L.rows === layers[i].rows && L.cols === layers[i].cols);
      adoptLayers(layers, sameShape);   // keep tracked neurons if the shape is unchanged
      if (typeof gen === "number") genome.gen = gen;
      if (genBadge) {
      genBadge.textContent = (external ? "● live · gen " : "demo · gen ") + genome.gen;
      genBadge.style.color = external ? "#3fb950" : "";
    }
      render();
    },
    // hand control back to the illustrative demo (or just freeze if false stays)
    setExternal(on) { external = !!on; if (btnToggle) btnToggle.disabled = !!on; },
    setPlaying,
    render,
  };
})();
