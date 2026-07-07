/* pure.js — PURE model-assembly node graph.
 *
 * You BUILD a model here by dragging nodes onto the canvas and wiring them
 * together. Each node is a piece of the model — a layer (its dims/units live
 * inside it), or a constraint you wire into the pipeline (e.g. an Energy node
 * placed before the Genome, where you set what costs energy). Select a node to
 * edit its settings in the right-hand Properties panel.
 *
 * NODE_TYPES below is a data-driven starter catalog — edit it freely; the whole
 * editor (ports, properties, wiring) is generated from it. The assembled graph
 * is exposed on window.PureGraph.getGraph() for the eventual GA wiring.
 */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const STORE = "pure.graph.v1";

  // ── node catalog (edit to taste — everything is generated from this) ─────
  // Each type: title, i/o port names, accent color, and typed properties.
  // Property types: number | text | select(options) | checkbox.
  const C = "#e0a02a";                 // shared "constraint" accent (the GENREG spice)
  const NODE_TYPES = {
    // ── data source: a video file, sampled into frames ───────────────────
    data: {
      group: "data", title: "Data", color: "#d98b34", inputs: [], outputs: ["data"],
      badge: (p) => p.filename ? (p.filename.length > 11 ? p.filename.slice(0, 10) + "…" : p.filename) : "no file",
      props: [
        { key: "fps", type: "number", label: "Assumed FPS", default: 30, min: 1, max: 240 },
        { key: "start_frame", type: "number", label: "Start frame", default: 0, min: 0, max: 1000000 },
        { key: "skip", type: "number", label: "Skip frames (stride−1)", default: 0, min: 0, max: 1000 },
        { key: "max_frames", type: "number", label: "Max frames", default: 64, min: 1, max: 100000 },
        { key: "size", type: "number", label: "Frame size (NxN)", default: 32, min: 4, max: 256 },
        { key: "grayscale", type: "checkbox", label: "Grayscale", default: true },
      ],
    },

    // ── data source: feeds the Input node ────────────────────────────────
    synthetic: {
      group: "data", title: "Synthetic", color: "#4bbf9a", inputs: [], outputs: ["data"],
      badge: (p) => p.kind,
      props: [
        { key: "kind", type: "select", label: "Kind", default: "sine",
          options: ["sine", "square", "ramp", "noise", "image"] },
        { key: "frequency", type: "number", label: "Frequency", default: 3, min: 0, max: 256, step: 0.5,
          when: { key: "kind", in: ["sine", "square", "ramp"] } },
        { key: "amplitude", type: "number", label: "Amplitude", default: 1, min: 0, max: 100, step: 0.1,
          when: { key: "kind", in: ["sine", "square", "ramp", "noise"] } },
        { key: "phase", type: "number", label: "Phase", default: 0, min: 0, max: 6.283, step: 0.1,
          when: { key: "kind", in: ["sine", "square", "ramp"] } },
        { key: "length", type: "number", label: "Samples", default: 64, min: 2, max: 4096,
          when: { key: "kind", in: ["sine", "square", "ramp", "noise"] } },
        { key: "pattern", type: "select", label: "Pattern", default: "gradient",
          options: ["gradient", "checker", "circle", "stripes"], when: { key: "kind", in: ["image"] } },
        { key: "size", type: "number", label: "Size (NxN)", default: 16, min: 2, max: 128,
          when: { key: "kind", in: ["image"] } },
        { key: "loop", type: "checkbox", label: "Loop frames", default: true, when: { key: "kind", in: ["image"] } },
      ],
    },

    // ── structure: the model itself ──────────────────────────────────────
    input: {
      group: "structure", title: "Input", color: "var(--tlm-s1)", inputs: ["data"], outputs: ["out"],
      badge: (p) => `${p.dims}d`,
      props: [{ key: "dims", type: "number", label: "Dimensions", default: 16, min: 1, max: 8192 }],
    },
    layer: {
      group: "structure", title: "Layer", color: "var(--accent)", inputs: ["in"], outputs: ["out"],
      badge: (p) => `${p.evolve_units ? "≤" + p.units : p.units} · ${p.activation}` + (p.k_sparsity ? ` · k${p.evolve_k ? "*" : p.k}` : ""),
      props: [
        { key: "units", type: "number", label: "Units (max if evolving)", default: 24, min: 1, max: 8192 },
        { key: "evolve_units", type: "checkbox", label: "Evolve unit count", default: false },
        { key: "activation", type: "select", label: "Activation", default: "tanh",
          options: ["identity", "tanh", "relu", "sigmoid", "evolved"] },
        { key: "bias", type: "checkbox", label: "Bias", default: true },
        { key: "k_sparsity", type: "checkbox", label: "k-sparsity", default: false },
        { key: "evolve_k", type: "checkbox", label: "Evolve k", default: false, when: { key: "k_sparsity", in: [true] } },
        { key: "k", type: "number", label: "k (active units)", default: 4, min: 1, max: 8192,
          when: [{ key: "k_sparsity", in: [true] }, { key: "evolve_k", in: [false] }] },
      ],
    },
    genome: {
      group: "structure", title: "Genome", color: "#b072f0", inputs: ["net"], outputs: ["out"],
      badge: () => "organism",
      props: [
        { key: "label", type: "text", label: "Label", default: "genome" },
        { key: "note", type: "text", label: "Note", default: "" },
      ],
    },
    output: {
      group: "structure", title: "Output", color: "var(--tlm-s2)", inputs: ["in"], outputs: [],
      badge: (p) => `${p.classes} out`,
      props: [{ key: "classes", type: "number", label: "Outputs", default: 10, min: 1, max: 8192 }],
    },

    // ── constraints: wire one into the pipeline to activate it, set its
    //    behaviour here. (No engine behind them yet — configuration only.) ──
    energy: {
      group: "constraint", title: "Energy", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `dec ${p.decay}`,
      props: [
        { key: "action_cost", type: "number", label: "Cost per action", default: 0.5, min: 0, max: 50, step: 0.05 },
        { key: "recover_gain", type: "number", label: "Recover on consequence", default: 4.0, min: 0, max: 50, step: 0.5 },
        { key: "decay", type: "number", label: "Decay", default: 0.9, min: 0.5, max: 1, step: 0.01 },
        { key: "floor", type: "number", label: "Floor (cull)", default: 0.2, min: 0, max: 1, step: 0.01 },
        { key: "e_max", type: "number", label: "Ceiling", default: 1.5, min: 0.5, max: 10, step: 0.1 },
      ],
    },
    temporal_budget: {
      group: "constraint", title: "Temporal Budget", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `${p.max_steps} steps`,
      props: [
        { key: "max_steps", type: "number", label: "Steps per episode", default: 256, min: 1, max: 100000 },
        { key: "idle_penalty", type: "number", label: "Idle penalty / step", default: 0.1, min: 0, max: 10, step: 0.01 },
      ],
    },
    consequential_drive: {
      group: "constraint", title: "Consequential Drive", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `spike ${p.spike}`,
      props: [
        { key: "spike", type: "number", label: "Spike on world-change", default: 1.0, min: 0, max: 10, step: 0.1 },
        { key: "decay", type: "number", label: "Decay / step", default: 0.95, min: 0.5, max: 1, step: 0.01 },
        { key: "sustain_floor", type: "number", label: "Sustain floor", default: 0.2, min: 0, max: 5, step: 0.05 },
      ],
    },
    capacity_cost: {
      group: "constraint", title: "Capacity Cost", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `${p.cost_per_dim}/dim`,
      props: [{ key: "cost_per_dim", type: "number", label: "Fitness cost / dimension", default: 0.001, min: 0, max: 1, step: 0.0005 }],
    },
    observation_cost: {
      group: "constraint", title: "Observation Cost", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `${p.cost_per_read}/read`,
      props: [{ key: "cost_per_read", type: "number", label: "Cost per environment read", default: 0.05, min: 0, max: 10, step: 0.01 }],
    },
    prediction_error: {
      group: "constraint", title: "Prediction Error", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `${p.norm}·${p.weight}`,
      props: [
        { key: "norm", type: "select", label: "Norm", default: "L1", options: ["L1", "L2"] },
        { key: "weight", type: "number", label: "Weight", default: 1.0, min: 0, max: 10, step: 0.1 },
      ],
    },
    information_gradient: {
      group: "constraint", title: "Information Gradient", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `pull ${p.strength}`,
      props: [{ key: "strength", type: "number", label: "Pull toward uncertainty", default: 1.0, min: 0, max: 10, step: 0.1 }],
    },
    stimulus_stagnation: {
      group: "constraint", title: "Stimulus Stagnation", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `w${p.window}`,
      props: [
        { key: "window", type: "number", label: "Info-gain window", default: 32, min: 1, max: 10000 },
        { key: "penalty", type: "number", label: "Repetition penalty", default: 0.5, min: 0, max: 10, step: 0.05 },
      ],
    },
    predictive_variance: {
      group: "constraint", title: "Predictive Variance", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `w ${p.weight}`,
      props: [{ key: "weight", type: "number", label: "Confidence-spread weight", default: 1.0, min: 0, max: 10, step: 0.1 }],
    },
    homeostatic_proximity: {
      group: "constraint", title: "Homeostatic Proximity", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `set ${p.setpoint}`,
      props: [
        { key: "setpoint", type: "number", label: "Setpoint", default: 0.5, min: 0, max: 1, step: 0.01 },
        { key: "tolerance", type: "number", label: "Tolerance", default: 0.2, min: 0, max: 1, step: 0.01 },
        { key: "weight", type: "number", label: "Weight", default: 1.0, min: 0, max: 10, step: 0.1 },
      ],
    },
    consolidation_threshold: {
      group: "constraint", title: "Consolidation Threshold", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `debt ${p.debt_threshold}`,
      props: [
        { key: "debt_threshold", type: "number", label: "Energy-debt trigger", default: 1.0, min: 0, max: 100, step: 0.1 },
        { key: "consolidation_steps", type: "number", label: "Consolidation steps", default: 16, min: 1, max: 10000 },
      ],
    },
    hazard_signal: {
      group: "constraint", title: "Hazard Signal", color: C, inputs: ["in"], outputs: ["out"],
      badge: (p) => `sens ${p.sensitivity}`,
      props: [
        { key: "sensitivity", type: "number", label: "Threat sensitivity", default: 1.0, min: 0, max: 10, step: 0.1 },
        { key: "weight", type: "number", label: "Weight", default: 1.0, min: 0, max: 10, step: 0.1 },
      ],
    },

    // ── objective: what the model evolves toward. Wire the model Output in,
    //    plus any constraints. Dynamic — always keeps one spare input port. ──
    // Passes the model's Output data through its "data" output so another model
    // can be chained after it (each model scored by its own Fitness).
    fitness: {
      group: "objective", title: "Fitness", color: "#d0679a", inputs: [], outputs: ["data"],
      dynamicInputs: true, badge: (p) => p.objective,
      props: [
        { key: "objective", type: "select", label: "Objective", default: "reconstruct",
          options: ["reconstruct", "reconstruct_source", "predict_next"] },
        { key: "metric", type: "select", label: "Error", default: "mse", options: ["mse", "mae"] },
      ],
    },
  };

  // ── geometry ─────────────────────────────────────────────────────────────
  const NODE_W = 194, HEAD_H = 30, PORT_TOP = 24, PORT_H = 26, PORT_R = 6;
  const portCY = (i) => HEAD_H + PORT_TOP + i * PORT_H;

  // ── state ─────────────────────────────────────────────────────────────────
  let graph = { nodes: [], edges: [], pan: { x: 40, y: 30 } };
  let uid = 1;
  let selected = null;
  let canvas = null, world = null, svg = null, propsHost = null;

  const nodeById = (id) => graph.nodes.find((n) => n.id === id);
  const typeOf = (n) => NODE_TYPES[n.type];
  const dataFiles = {};                 // nodeId → selected File (runtime only, not persisted)

  // Frame math for a Data node: source frames from duration×fps, then start
  // offset, stride (skip), and the max cap. Returns the pieces for display.
  function dataFrames(p) {
    const src = Math.max(0, Math.floor((p.duration || 0) * (p.fps || 30)));
    const afterStart = Math.max(0, src - (p.start_frame || 0));
    const stride = (p.skip || 0) + 1;
    const afterSkip = Math.ceil(afterStart / stride);
    const eff = Math.min(p.max_frames || 0, afterSkip);
    return { src, afterStart, stride, afterSkip, eff };
  }

  // Actually DECODE the selected video into frame vectors (this is what makes
  // the Data node real, not mock). Seeks the video to each timestamp, WAITS for
  // the frame to actually paint (requestVideoFrameCallback, else a double rAF),
  // draws into an NxN canvas, reads pixels → grayscale/RGB in [-1,1]. If every
  // frame comes out uniform/black it errors instead of feeding black frames.
  function grabFrame(v, t, N, ctx) {
    return new Promise((resolve) => {
      let settled = false;
      const paint = () => { if (settled) return; settled = true;
        ctx.drawImage(v, 0, 0, N, N); resolve(ctx.getImageData(0, 0, N, N).data); };
      const afterSeek = () => { v.removeEventListener("seeked", afterSeek);
        if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => paint());
        else requestAnimationFrame(() => requestAnimationFrame(() => paint())); };
      v.addEventListener("seeked", afterSeek);
      const dur = v.duration || 0.001, tgt = Math.min(Math.max(t, 0), Math.max(0, dur - 1e-3));
      // nudge so a seek event always fires (t=0 on an already-at-0 video won't)
      v.currentTime = (Math.abs(v.currentTime - tgt) < 1e-3) ? tgt + 1e-3 : tgt;
      setTimeout(() => { if (!settled) { settled = true; ctx.drawImage(v, 0, 0, N, N); resolve(ctx.getImageData(0, 0, N, N).data); } }, 800); // safety
    });
  }
  // Acquire frames for a Data node: try the SERVER (imageio+ffmpeg — handles any
  // format incl. mkv/avi/mov) first; fall back to browser decode (mp4/webm/ogg)
  // if the server route is unavailable. Returns normalized frames in [-1,1].
  const frameCache = new WeakMap();     // File → {key, result} (avoid re-decoding each Run)
  async function acquireFrames(p, file) {
    if (!file) return { error: "no video file selected on the Data node" };
    const key = [p.size, p.start_frame, p.skip, p.max_frames, p.grayscale ? 1 : 0].join(",");
    const cached = frameCache.get(file);
    if (cached && cached.key === key) return cached.result;
    let result = null;
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("size", p.size); fd.append("start", p.start_frame);
      fd.append("skip", p.skip); fd.append("max", p.max_frames);
      fd.append("gray", p.grayscale ? "1" : "0");
      const r = await fetch("/api/pure/frames", { method: "POST", body: fd });
      if (r.ok) {
        const d = await r.json();
        if (d.error) result = { error: d.error };
        else if (d.frames && d.frames.length) result = { frames: d.frames.map((f) => f.map((v) => v / 255 * 2 - 1)), size: d.size, gray: d.gray, source: "server" };
      } else if (r.status !== 404) {
        try { const d = await r.json(); if (d.error) result = { error: d.error }; } catch (_) {}
      }
    } catch (_) { /* network error → browser fallback below */ }
    if (!result) result = await extractVideoFrames(p, file);   // browser fallback (mp4/webm/ogg)
    if (result.frames) frameCache.set(file, { key, result });
    return result;
  }
  function extractVideoFrames(p, file) {
    return new Promise((resolve) => {
      if (!file) { resolve({ error: "no video file selected on the Data node" }); return; }
      const N = Math.max(2, p.size | 0), url = URL.createObjectURL(file);
      const v = document.createElement("video");
      v.muted = true; v.playsInline = true; v.preload = "auto";
      v.style.position = "fixed"; v.style.left = "-9999px"; v.style.width = "2px"; v.style.height = "2px";
      document.body.appendChild(v);
      const cv = document.createElement("canvas"); cv.width = N; cv.height = N;
      const ctx = cv.getContext("2d", { willReadFrequently: true });
      let done = false;
      const finish = (r) => { if (done) return; done = true; try { URL.revokeObjectURL(url); } catch (_) {} try { v.remove(); } catch (_) {} resolve(r); };
      v.addEventListener("error", () => finish({ error: "could not decode this video (" + ((v.error && v.error.message) || "unsupported format") + ")" }));
      v.addEventListener("loadeddata", async () => {
        try {
          if (!v.videoWidth) { finish({ error: "video has no visual track (zero dimensions)" }); return; }
          const dur = v.duration || 0, fps = p.fps || 30, stride = (p.skip || 0) + 1, maxF = p.max_frames || 64, start = p.start_frame || 0;
          const frames = []; let anyVariation = false;
          for (let i = 0; i < maxF; i++) {
            const t = (start + i * stride) / fps; if (dur && t >= dur) break;
            const img = await grabFrame(v, t, N, ctx);
            const vec = []; let mn = Infinity, mx = -Infinity;
            if (p.grayscale) for (let k = 0; k < img.length; k += 4) { const g = (0.299 * img[k] + 0.587 * img[k + 1] + 0.114 * img[k + 2]) / 255 * 2 - 1; vec.push(g); if (g < mn) mn = g; if (g > mx) mx = g; }
            else for (let k = 0; k < img.length; k += 4) { const rr = img[k] / 255 * 2 - 1, gg = img[k + 1] / 255 * 2 - 1, bb = img[k + 2] / 255 * 2 - 1; vec.push(rr, gg, bb); mn = Math.min(mn, rr, gg, bb); mx = Math.max(mx, rr, gg, bb); }
            if (mx - mn > 0.02) anyVariation = true;
            frames.push(vec);
          }
          if (!frames.length) { finish({ error: "no frames decoded — check FPS / start frame vs the video's duration" }); return; }
          if (!anyVariation) { finish({ error: "frames decoded but every one is uniform/black — the browser couldn't paint this video's frames (try a smaller/re-encoded mp4)" }); return; }
          finish({ frames, size: N, gray: !!p.grayscale });
        } catch (e) { finish({ error: "frame decode failed: " + e.message }); }
      });
      v.src = url; v.load();
    });
  }

  // The chain ACTUALLY wired from Input → hidden layers → Output, in data-flow
  // order. Nodes not on this path (e.g. a Layer dropped on the canvas but never
  // connected) are excluded. Both the genome visual and the engine build from
  // this, so an unwired layer never shows as connected or gets trained.
  function wiredPath() {
    const input = graph.nodes.find((n) => n.type === "input");
    if (!input) return [];
    const path = [input], seen = new Set([input.id]);
    let cur = input;
    for (let guard = 0; guard < graph.nodes.length + 2; guard++) {
      const targets = graph.edges.filter((e) => e.from.node === cur.id).map((e) => nodeById(e.to.node)).filter(Boolean);
      const nextLayer = targets.find((t) => t.type === "layer" && !seen.has(t.id));
      if (nextLayer) { path.push(nextLayer); seen.add(nextLayer.id); cur = nextLayer; continue; }
      const toOut = targets.find((t) => t.type === "output");
      if (toOut) path.push(toOut);
      break;
    }
    return path;
  }

  // Input ports: static from the type, or dynamic (one per wired edge + one
  // spare) for nodes like Fitness that grow ports as you wire constraints in.
  function nodeInputs(n) {
    const t = typeOf(n);
    if (!t.dynamicInputs) return t.inputs;
    const inc = graph.edges.filter((e) => e.to.node === n.id).sort((a, b) => a.to.idx - b.to.idx);
    const labels = inc.map((e) => { const s = nodeById(e.from.node); return s ? typeOf(s).title.toLowerCase().slice(0, 9) : "in"; });
    labels.push("＋");                              // always one open port
    return labels;
  }
  const nodeOutputs = (n) => typeOf(n).outputs;
  const rowsN = (n) => Math.max(nodeInputs(n).length, nodeOutputs(n).length, 1);
  const nodeHN = (n) => HEAD_H + rowsN(n) * PORT_H + 8;

  // Densify the input-port indices of dynamic nodes so removing a wired
  // constraint collapses the gap (ports stay 0..k, spare at the end).
  function normalizeDynamic() {
    for (const n of graph.nodes) {
      if (!typeOf(n).dynamicInputs) continue;
      graph.edges.filter((e) => e.to.node === n.id).sort((a, b) => a.to.idx - b.to.idx)
        .forEach((e, i) => { e.to.idx = i; });
    }
  }

  let genomeHost = null;
  function save() {
    try { localStorage.setItem(STORE, JSON.stringify(graph)); } catch (_) {}
    renderGenome();
  }
  function load() {
    try {
      const g = JSON.parse(localStorage.getItem(STORE) || "null");
      if (g && Array.isArray(g.nodes)) {
        graph = g;
        graph.pan = graph.pan || { x: 40, y: 30 };
        uid = graph.nodes.reduce((m, n) => Math.max(m, (n.id | 0) + 1), 1);
        return true;
      }
    } catch (_) {}
    return false;
  }

  function defaults(type) {
    const p = {};
    for (const d of NODE_TYPES[type].props) p[d.key] = d.default;
    return p;
  }

  function addNode(type, x, y) {
    const t = NODE_TYPES[type];
    if (!t) return null;
    const n = { id: uid++, type, x: x, y: y, props: defaults(type) };
    graph.nodes.push(n);
    buildNode(n);
    positionNode(n);
    save();
    return n;
  }

  function removeNode(n) {
    graph.edges = graph.edges.filter((e) => e.from.node !== n.id && e.to.node !== n.id);
    graph.nodes = graph.nodes.filter((x) => x !== n);
    if (selected === n) selected = null;
    renderAll();
    save();
  }

  // ── ports (world coords) ───────────────────────────────────────────────────
  const outPos = (n, i) => ({ x: n.x + NODE_W, y: n.y + portCY(i) });
  const inPos = (n, i) => ({ x: n.x, y: n.y + portCY(i) });

  // ── node DOM ───────────────────────────────────────────────────────────────
  function buildNode(n) {
    const t = typeOf(n);
    const el = document.createElement("div");
    el.className = "pg-node";
    el.style.width = NODE_W + "px";
    el.style.height = nodeHN(n) + "px";
    el.style.setProperty("--pg-accent", t.color);
    n._el = el;

    const head = document.createElement("div");
    head.className = "pg-node-head";
    head.style.background = t.color;
    const title = document.createElement("span");
    title.textContent = t.title;
    const badge = document.createElement("span");
    badge.className = "pg-node-badge";
    head.appendChild(title); head.appendChild(badge);
    el.appendChild(head);
    n._badge = badge;

    // ports + labels
    nodeInputs(n).forEach((name, i) => {
      const dot = document.createElement("div");
      dot.className = "pg-port pg-in";
      dot.style.top = (portCY(i) - PORT_R) + "px";
      dot.dataset.node = n.id; dot.dataset.idx = i;
      el.appendChild(dot);
      const lbl = document.createElement("div");
      lbl.className = "pg-port-label pg-in-label";
      lbl.style.top = (portCY(i) - 9) + "px";
      lbl.textContent = name;
      el.appendChild(lbl);
    });
    nodeOutputs(n).forEach((name, i) => {
      const dot = document.createElement("div");
      dot.className = "pg-port pg-out";
      dot.style.top = (portCY(i) - PORT_R) + "px";
      dot.dataset.node = n.id; dot.dataset.idx = i;
      el.appendChild(dot);
      const lbl = document.createElement("div");
      lbl.className = "pg-port-label pg-out-label";
      lbl.style.top = (portCY(i) - 9) + "px";
      lbl.textContent = name;
      el.appendChild(lbl);
    });

    // interactions
    head.addEventListener("mousedown", (e) => startNodeDrag(e, n));
    el.addEventListener("mousedown", (e) => { if (e.target === el) selectNode(n); });
    el.addEventListener("click", (e) => { e.stopPropagation(); selectNode(n); });
    el.querySelectorAll(".pg-out").forEach((d) =>
      d.addEventListener("mousedown", (e) => startWire(e, n, +d.dataset.idx)));

    world.appendChild(el);
    updateBadge(n);
  }

  function updateBadge(n) {
    const t = typeOf(n);
    if (n._badge) n._badge.textContent = t.badge ? t.badge(n.props) : "";
  }
  function positionNode(n) {
    if (n._el) { n._el.style.left = n.x + "px"; n._el.style.top = n.y + "px"; }
  }

  // ── wires ──────────────────────────────────────────────────────────────────
  function wirePath(a, b) {
    const dx = Math.max(38, Math.abs(b.x - a.x) * 0.5);
    return `M ${a.x} ${a.y} C ${a.x + dx} ${a.y}, ${b.x - dx} ${b.y}, ${b.x} ${b.y}`;
  }
  function renderWires(temp) {
    svg.replaceChildren();
    for (const e of graph.edges) {
      const fn = nodeById(e.from.node), tn = nodeById(e.to.node);
      if (!fn || !tn) continue;
      const d = wirePath(outPos(fn, e.from.idx), inPos(tn, e.to.idx));
      const hit = document.createElementNS(NS, "path");
      hit.setAttribute("d", d); hit.setAttribute("class", "pg-wire-hit");
      hit.addEventListener("click", (ev) => { ev.stopPropagation(); removeEdge(e); });
      const vis = document.createElementNS(NS, "path");
      vis.setAttribute("d", d); vis.setAttribute("class", "pg-wire");
      svg.appendChild(hit); svg.appendChild(vis);
    }
    if (temp) {
      const p = document.createElementNS(NS, "path");
      p.setAttribute("d", wirePath(temp.a, temp.b));
      p.setAttribute("class", "pg-wire pg-wire-temp");
      svg.appendChild(p);
    }
  }
  function removeEdge(e) {
    graph.edges = graph.edges.filter((x) => x !== e);
    renderAll(); save();
  }

  // ── dragging: pan / node / wire ─────────────────────────────────────────────
  let drag = null;
  function worldFromEvent(e) {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left - graph.pan.x, y: e.clientY - r.top - graph.pan.y };
  }
  function startNodeDrag(e, n) {
    e.preventDefault(); e.stopPropagation();
    selectNode(n);
    const w = worldFromEvent(e);
    drag = { kind: "node", n, dx: w.x - n.x, dy: w.y - n.y };
  }
  function startPan(e) {
    if (e.target !== canvas && e.target !== world && e.target !== svg) return;
    selectNode(null);
    drag = { kind: "pan", sx: e.clientX, sy: e.clientY, px: graph.pan.x, py: graph.pan.y };
  }
  function startWire(e, n, idx) {
    e.preventDefault(); e.stopPropagation();
    drag = { kind: "wire", from: { node: n.id, idx }, a: outPos(n, idx) };
  }
  function onMove(e) {
    if (!drag) return;
    if (drag.kind === "node") {
      const w = worldFromEvent(e);
      drag.n.x = Math.round(w.x - drag.dx);
      drag.n.y = Math.round(w.y - drag.dy);
      positionNode(drag.n); renderWires();
    } else if (drag.kind === "pan") {
      graph.pan.x = drag.px + (e.clientX - drag.sx);
      graph.pan.y = drag.py + (e.clientY - drag.sy);
      applyPan();
    } else if (drag.kind === "wire") {
      renderWires({ a: drag.a, b: worldFromEvent(e) });
    }
  }
  function onUp(e) {
    if (!drag) return;
    if (drag.kind === "wire") {
      const port = e.target.closest && e.target.closest(".pg-port.pg-in");
      if (port && +port.dataset.node !== drag.from.node) {
        const to = { node: +port.dataset.node, idx: +port.dataset.idx };
        graph.edges = graph.edges.filter((x) => !(x.to.node === to.node && x.to.idx === to.idx));
        graph.edges.push({ from: drag.from, to });
        renderAll(); save();                        // rebuild so dynamic ports refresh
      } else {
        renderWires();                              // dropped on empty space — clear temp wire
      }
    }
    if (drag.kind === "node" || drag.kind === "pan") save();
    drag = null;
  }
  function applyPan() {
    world.style.transform = `translate(${graph.pan.x}px, ${graph.pan.y}px)`;
  }

  // ── selection + properties panel ────────────────────────────────────────────
  function selectNode(n) {
    if (selected && selected._el) selected._el.classList.remove("selected");
    selected = n;
    if (n && n._el) n._el.classList.add("selected");
    const badge = document.getElementById("pg-sel");
    if (badge) badge.textContent = n ? typeOf(n).title : "none";
    renderProps();
  }

  function renderProps() {
    if (!propsHost) return;
    propsHost.replaceChildren();
    if (!selected) {
      const hint = document.createElement("div");
      hint.className = "tlm-status";
      hint.textContent = "Select a node to edit its settings. Drag from an output port to an input port to wire nodes; click a wire to remove it.";
      propsHost.appendChild(hint);
      return;
    }
    const n = selected, t = typeOf(n);
    const idline = document.createElement("div");
    idline.className = "pg-prop-id";
    idline.textContent = `${t.title} · #${n.id}`;
    propsHost.appendChild(idline);

    // a field shows when its `when` is satisfied — a single {key,in} or an
    // array of them (all must pass).
    const condOk = (c) => (c.in || []).includes(n.props[c.key]);
    const shown = (d) => !d.when || (Array.isArray(d.when) ? d.when.every(condOk) : condOk(d.when));
    const controls = (key) => t.props.some((x) => x.when &&
      (Array.isArray(x.when) ? x.when.some((c) => c.key === key) : x.when.key === key));

    if (n.type === "data") renderDataFile(n);        // video picker at the top
    for (const d of t.props) {
      if (!shown(d)) continue;                     // conditional field (e.g. kind-specific)
      const field = document.createElement("div");
      field.className = "field";
      if (d.type === "checkbox") {
        const lab = document.createElement("label");
        lab.className = "check";
        const cb = document.createElement("input");
        cb.type = "checkbox"; cb.checked = !!n.props[d.key];
        const ctl = controls(d.key);
        cb.addEventListener("change", () => { n.props[d.key] = cb.checked; updateBadge(n); save();
          if (ctl) renderProps(); else updatePreview(); });   // toggle may reveal/hide fields
        lab.appendChild(cb); lab.appendChild(document.createTextNode(" " + d.label));
        field.appendChild(lab);
      } else {
        const lab = document.createElement("label"); lab.textContent = d.label;
        field.appendChild(lab);
        let inp;
        if (d.type === "select") {
          inp = document.createElement("select");
          for (const o of d.options) {
            const opt = document.createElement("option");
            opt.value = o; opt.textContent = o; if (n.props[d.key] === o) opt.selected = true;
            inp.appendChild(opt);
          }
          const ctl = controls(d.key);
          inp.addEventListener("change", () => {
            n.props[d.key] = inp.value; updateBadge(n); save();
            if (ctl) renderProps(); else updatePreview();   // reveal/hide dependent fields
          });
        } else {
          inp = document.createElement("input");
          inp.type = d.type === "number" ? "number" : "text";
          if (d.min != null) inp.min = d.min;
          if (d.max != null) inp.max = d.max;
          if (d.step != null) inp.step = d.step;
          inp.value = n.props[d.key];
          inp.addEventListener("input", () => {
            n.props[d.key] = d.type === "number"
              ? Math.max(d.min ?? -Infinity, Math.min(d.max ?? Infinity, parseFloat(inp.value) || 0))
              : inp.value;
            updateBadge(n); save(); updatePreview();
          });
        }
        field.appendChild(inp);
      }
      propsHost.appendChild(field);
    }

    synthPreviewHost = null; dataFramesOut = null; dataFramesSub = null;
    if (n.type === "synthetic") {
      const cap = document.createElement("div");
      cap.className = "pg-prop-id"; cap.style.marginTop = "8px"; cap.textContent = "Preview";
      propsHost.appendChild(cap);
      synthPreviewHost = document.createElement("div");
      synthPreviewHost.className = "pg-preview";
      propsHost.appendChild(synthPreviewHost);
      drawSynthPreview(n);
    }
    if (n.type === "data") renderDataFrames(n);      // computed total-frames readout

    const del = document.createElement("button");
    del.className = "runs-btn pg-del";
    del.textContent = "Delete node";
    del.addEventListener("click", () => removeNode(n));
    propsHost.appendChild(del);
  }

  // ── Data node: video picker + computed total-frames readout ─────────────────
  function renderDataFile(n) {
    const wrap = document.createElement("div"); wrap.className = "field";
    const lab = document.createElement("label"); lab.textContent = "Video file";
    wrap.appendChild(lab);
    const file = document.createElement("input");
    file.type = "file"; file.accept = "video/*,.mkv,.avi,.mov,.flv,.wmv,.m4v,.mpg,.mpeg,.webm";
    file.style.fontSize = "11px"; file.style.color = "var(--muted)";
    file.addEventListener("change", () => {
      const f = file.files && file.files[0];
      if (!f) return;
      dataFiles[n.id] = f;
      n.props.filename = f.name;
      updateBadge(n); save();
      const url = URL.createObjectURL(f);
      const v = document.createElement("video");
      v.preload = "metadata"; v.muted = true;
      v.addEventListener("loadedmetadata", () => { n.props.duration = v.duration || 0; try { URL.revokeObjectURL(url); } catch (_) {} save(); if (selected === n) renderProps(); });
      v.addEventListener("error", () => { n.props.duration = 0; if (selected === n) renderProps(); });
      v.src = url;
    });
    wrap.appendChild(file);
    if (n.props.filename) {
      const cur = document.createElement("div"); cur.className = "field-hint";
      cur.textContent = "selected: " + n.props.filename + (n.props.duration ? ` · ${n.props.duration.toFixed(1)}s` : "");
      wrap.appendChild(cur);
    }
    propsHost.appendChild(wrap);
  }
  let dataFramesOut = null, dataFramesSub = null;
  function updateDataFrames(n) {
    if (!dataFramesOut) return;
    if (!n.props.filename) { dataFramesOut.textContent = "select a video to compute frames"; if (dataFramesSub) dataFramesSub.textContent = ""; return; }
    if (!n.props.duration) { dataFramesOut.textContent = "reading video…"; return; }
    const fr = dataFrames(n.props);
    dataFramesOut.textContent = `source ~${fr.src.toLocaleString()} · stride ${fr.stride} → ${fr.afterSkip.toLocaleString()} · using ${fr.eff.toLocaleString()}`;
    if (dataFramesSub) dataFramesSub.textContent = `${fr.eff.toLocaleString()} frames of ${n.props.size}×${n.props.size}${n.props.grayscale ? " grayscale" : ""} → the model`;
  }
  function renderDataFrames(n) {
    const box = document.createElement("div");
    box.className = "pg-prop-id"; box.style.marginTop = "8px"; box.textContent = "Frames";
    propsHost.appendChild(box);
    dataFramesOut = document.createElement("div"); dataFramesOut.className = "pu-metric";
    propsHost.appendChild(dataFramesOut);
    dataFramesSub = document.createElement("div"); dataFramesSub.className = "pg-chart-cap";
    propsHost.appendChild(dataFramesSub);
    updateDataFrames(n);

    if (n.props.filename && dataFiles[n.id]) {       // verify decode without training
      const btn = document.createElement("button");
      btn.className = "runs-btn"; btn.textContent = "Preview frame"; btn.style.marginTop = "6px";
      const pv = document.createElement("canvas"); pv.width = 96; pv.height = 96;
      pv.className = "pg-preview-canvas"; pv.style.marginTop = "6px";
      const pmsg = document.createElement("div"); pmsg.className = "pg-chart-cap";
      btn.addEventListener("click", async () => {
        pmsg.style.color = "var(--muted)"; pmsg.textContent = "decoding first frame…";
        const res = await acquireFrames(Object.assign({}, n.props, { start_frame: 0, skip: 0, max_frames: 1 }), dataFiles[n.id]);
        if (res.error) { pmsg.textContent = res.error; pmsg.style.color = "var(--red)"; return; }
        const N = res.size, f = res.frames[0], c = 96 / N, ctx = pv.getContext("2d");
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
          const idx = res.gray ? (y * N + x) : (y * N + x) * 3;
          const g = Math.max(0, Math.min(255, Math.round((f[idx] + 1) / 2 * 255)));
          ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(x * c, y * c, Math.ceil(c), Math.ceil(c));
        }
        pmsg.textContent = `decoded ${N}×${N} OK — this is what feeds the model`;
      });
      propsHost.appendChild(btn); propsHost.appendChild(pmsg); propsHost.appendChild(pv);
    }
  }

  // ── synthetic data preview (waveform sparkline or image pattern) ────────────
  let synthPreviewHost = null;
  function updatePreview() {
    if (selected && selected.type === "synthetic" && synthPreviewHost) drawSynthPreview(selected);
    if (selected && selected.type === "data") updateDataFrames(selected);
  }
  function synthSamples(p) {
    const n = Math.max(2, Math.min(4096, p.length | 0));
    const amp = p.amplitude != null ? p.amplitude : 1, out = [];
    for (let i = 0; i < n; i++) {
      const t = i / n, x = 2 * Math.PI * (p.frequency || 0) * t + (p.phase || 0);
      let v;
      if (p.kind === "sine") v = Math.sin(x);
      else if (p.kind === "square") v = Math.sign(Math.sin(x)) || 1;
      else if (p.kind === "ramp") v = 2 * (((p.frequency || 0) * t + (p.phase || 0) / (2 * Math.PI)) % 1) - 1;
      else v = Math.random() * 2 - 1;              // noise
      out.push(v * amp);
    }
    return out;
  }
  function imagePixel(pattern, xr, yr, N) {
    if (pattern === "gradient") return xr;
    if (pattern === "checker") { const c = 8; return ((Math.floor(xr * c) + Math.floor(yr * c)) % 2) ? 1 : 0; }
    if (pattern === "stripes") return (Math.floor(xr * 8) % 2) ? 1 : 0;
    const dx = xr - 0.5, dy = yr - 0.5; return (dx * dx + dy * dy) < 0.16 ? 1 : 0;   // circle
  }
  function drawSynthPreview(n) {
    if (!synthPreviewHost) return;
    synthPreviewHost.replaceChildren();
    const p = n.props;
    if (p.kind === "image") {
      const N = Math.max(2, Math.min(128, p.size | 0)), px = 128;
      const cv = document.createElement("canvas"); cv.width = px; cv.height = px;
      cv.className = "pg-preview-canvas";
      const ctx = cv.getContext("2d"); const cell = px / N;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const v = imagePixel(p.pattern, x / (N - 1 || 1), y / (N - 1 || 1), N);
        const g = Math.round(v * 255);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
      }
      synthPreviewHost.appendChild(cv);
      const lbl = document.createElement("div"); lbl.className = "pg-preview-cap";
      lbl.textContent = `${p.pattern} · ${N}×${N}${p.loop ? " · looped" : ""}`;
      synthPreviewHost.appendChild(lbl);
    } else {
      const W = 240, H = 84, s = synthSamples(p);
      const mx = Math.max(1e-6, ...s.map((v) => Math.abs(v)));
      const pts = s.map((v, i) => `${(i / (s.length - 1)) * W},${H / 2 - (v / mx) * (H / 2 - 4)}`).join(" ");
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", "100%");
      svg.setAttribute("class", "pg-preview-svg");
      const mid = document.createElementNS(NS, "line");
      mid.setAttribute("x1", 0); mid.setAttribute("y1", H / 2); mid.setAttribute("x2", W); mid.setAttribute("y2", H / 2);
      mid.setAttribute("stroke", "var(--tlm-grid)");
      const poly = document.createElementNS(NS, "polyline");
      poly.setAttribute("points", pts); poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", "#4bbf9a"); poly.setAttribute("stroke-width", "1.5");
      svg.appendChild(mid); svg.appendChild(poly);
      synthPreviewHost.appendChild(svg);
      const lbl = document.createElement("div"); lbl.className = "pg-preview-cap";
      lbl.textContent = `${p.kind} · ${p.length} samples`;
      synthPreviewHost.appendChild(lbl);
    }
  }

  // ── full render ─────────────────────────────────────────────────────────────
  function renderAll() {
    normalizeDynamic();
    world.querySelectorAll(".pg-node").forEach((e) => e.remove());
    graph.nodes.forEach((n) => { buildNode(n); positionNode(n); });
    applyPan();
    renderWires();
    if (selected && selected._el) selected._el.classList.add("selected");   // keep highlight
  }

  // ── genome visual: dot-columns derived from the graph's structure ────────────
  // One column per dimensional structure node (Input / Layer / Output), left to
  // right by position. Non-wired node-dot style; the model you assemble above is
  // rendered as the organism here.
  const GVW = 980, GVH = 220, GM_TOP = 46, GM_BOT = 36;
  const GBAND = GVH - GM_TOP - GM_BOT, GMAX = 13, GR_MAX = 9, GR_MIN = 3;
  const SIZED = { input: "dims", layer: "units", output: "classes" };

  function orderedStructure() {
    // only the nodes actually wired Input → … → Output (matches the engine).
    return wiredPath()
      .filter((n) => n.type in SIZED)
      .map((n) => ({ label: typeOf(n).title, color: typeOf(n).color, n: Math.max(0, n.props[SIZED[n.type]] | 0) }));
  }
  // Returns the visible dots as {y, idx} where idx is the unit index (so live
  // activations can be mapped to the right dot even when the column is capped).
  function colLayoutG(n) {
    const cy = GM_TOP + GBAND / 2;
    if (n <= 0) return { dots: [], gap: false, r: GR_MAX };
    if (n === 1) return { dots: [{ y: cy, idx: 0 }], gap: false, r: GR_MAX };
    const shown = Math.min(n, GMAX);
    const r = Math.max(GR_MIN, Math.round(GR_MAX - (GR_MAX - GR_MIN) * (shown - 2) / (GMAX - 2)));
    const step = GBAND / (shown - 1);
    if (n <= GMAX) {
      const dots = []; for (let i = 0; i < n; i++) dots.push({ y: GM_TOP + i * step, idx: i });
      return { dots, gap: false, r };
    }
    const half = Math.floor(GMAX / 2), dots = [];
    for (let i = 0; i < half; i++) dots.push({ y: GM_TOP + i * step, idx: i });            // top → first units
    for (let i = 0; i < half; i++) dots.push({ y: GM_TOP + GBAND - i * step, idx: n - 1 - i }); // bottom → last units
    dots.sort((a, b) => a.y - b.y);
    return { dots, gap: true, r };
  }
  // neuron saturation color: barely used (yellow) → close to full saturation (red).
  // t = |activation| clamped to 1, so "saturated" means genuinely near ±1.
  function satColor(t) {
    const a = [242, 208, 36], b = [229, 72, 77], m = (i) => Math.round(a[i] + (b[i] - a[i]) * t);
    return `rgb(${m(0)},${m(1)},${m(2)})`;
  }
  function svgn(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  // activations: optional array (one Float array per column) — dots fire by
  // live activation. wLayers: optional weight matrices per column-transition —
  // connection lines are colored by weight sign/strength (and brightened from
  // firing neurons). Both absent = static skeleton with faint connections.
  function renderGenome(activations, wLayers, dims, columns) {
    if (!genomeHost) return;
    genomeHost.replaceChildren();
    const cols = columns || orderedStructure();   // engine may supply the active model's columns
    if (!cols.length) {
      const hint = document.createElement("div");
      hint.className = "tlm-status";
      hint.textContent = "Add Input / Layer / Output nodes above to see the assembled genome.";
      genomeHost.appendChild(hint);
      return;
    }
    const svg = svgn("svg", { viewBox: `0 0 ${GVW} ${GVH}`, width: "100%",
      preserveAspectRatio: "xMidYMid meet", role: "img",
      "aria-label": "Assembled genome — neuron columns and their connections" });
    svg.style.display = "block";
    const colX = cols.map((_, i) => (GVW * (i + 1)) / (cols.length + 1));
    const layouts = cols.map((col, ci) => {
      // during a run an evolve-units layer reports fewer active units → the
      // column shrinks to its effective width.
      const effN = (dims && dims[ci] != null) ? Math.min(col.n, Math.max(1, dims[ci] | 0)) : col.n;
      const L = colLayoutG(effN);
      return { col, cx: colX[ci], dots: L.dots, gap: L.gap, r: L.r, effN, vals: activations && activations[ci] };
    });

    // 1) connections (drawn first, behind the neurons)
    for (let ci = 0; ci < layouts.length - 1; ci++) {
      const A = layouts[ci], B = layouts[ci + 1];
      const L = wLayers && wLayers[ci];
      let maxW = 1e-6; if (L) for (let k = 0; k < L.w.length; k++) { const a = Math.abs(L.w[k]); if (a > maxW) maxW = a; }
      const maxAv = A.vals ? Math.max(1e-6, ...A.vals.map((v) => Math.abs(v))) : 1;
      for (const a of A.dots) for (const b of B.dots) {
        let stroke = "var(--tlm-grid)", op = 0.1;
        if (L && a.idx < L.inn && b.idx < L.outn) {
          const w = L.w[a.idx * L.outn + b.idx];
          op = 0.04 + 0.7 * (Math.abs(w) / maxW);
          if (A.vals) op *= 0.35 + 0.65 * Math.min(1, Math.abs(A.vals[a.idx] || 0) / maxAv);   // firing source
          stroke = w >= 0 ? "#4ea1ff" : "#f85149";
        }
        svg.appendChild(svgn("line", { x1: A.cx, y1: a.y, x2: B.cx, y2: b.y,
          stroke, "stroke-width": 1, "stroke-opacity": Math.min(0.85, op) }));
      }
    }

    // 2) neurons + captions
    layouts.forEach((Lo) => {
      const { col, cx, dots, gap, r, effN, vals } = Lo;
      dots.forEach((d) => {
        let fill = "var(--panel-2)";
        if (vals && d.idx < vals.length) fill = satColor(Math.min(1, Math.abs(vals[d.idx])));   // yellow→red by saturation
        svg.appendChild(svgn("circle", { cx, cy: d.y, r, fill, stroke: col.color, "stroke-width": 2 }));
      });
      if (gap) {
        const gy = GM_TOP + GBAND / 2;
        for (let k = -1; k <= 1; k++) svg.appendChild(svgn("circle", { cx, cy: gy + k * 7, r: 1.4, fill: "var(--muted)" }));
      }
      const cap = svgn("text", { x: cx, y: GM_TOP - 24, "text-anchor": "middle",
        fill: "var(--text)", "font-size": 13, "font-weight": 600, "font-family": "var(--mono, monospace)" });
      cap.textContent = col.label;
      svg.appendChild(cap);
      const evolving = effN < col.n;                 // showing fewer than the max → evolved width
      const sub = svgn("text", { x: cx, y: GVH - GM_BOT + 26, "text-anchor": "middle",
        fill: evolving ? "var(--accent)" : "var(--muted)", "font-size": 11, "font-family": "var(--mono, monospace)" });
      sub.textContent = (effN === 1 ? "1 unit" : `${effN} units`) + (evolving ? ` / ${col.n}` : "");
      svg.appendChild(sub);
    });
    genomeHost.appendChild(svg);
  }

  const wire = (a, ai, b, bi) => graph.edges.push({ from: { node: a.id, idx: ai }, to: { node: b.id, idx: bi } });

  // Basic single model: Synthetic → Input → Layer → Output → Fitness, with an
  // Energy constraint wired into Fitness to show the constraint→fitness pattern.
  function tmplBasic() {
    const sy = addNode("synthetic", 20, 100);
    const inp = addNode("input", 250, 100);
    const lay = addNode("layer", 480, 100);
    const out = addNode("output", 710, 100);
    const fit = addNode("fitness", 710, 320);
    const en = addNode("energy", 480, 340);
    wire(sy, 0, inp, 0); wire(inp, 0, lay, 0); wire(lay, 0, out, 0);
    wire(out, 0, fit, 0); wire(en, 0, fit, 1);
  }

  // Autoencoder: encoder (16→4 latent, scored by reconstruct) then decoder
  // (4→16, scored by reconstruct_source — against the ORIGINAL input, so it
  // truly reconstructs the source through the bottleneck).
  function tmplAutoencoder() {
    const sy = addNode("synthetic", 20, 100);
    const in1 = addNode("input", 230, 100); in1.props.dims = 16;
    const l1 = addNode("layer", 440, 100); l1.props.units = 12;
    const out1 = addNode("output", 650, 100); out1.props.classes = 4;      // latent
    const fit1 = addNode("fitness", 650, 320); fit1.props.objective = "reconstruct";
    const in2 = addNode("input", 880, 100); in2.props.dims = 4;            // = latent
    const l2 = addNode("layer", 1090, 100); l2.props.units = 12;
    const out2 = addNode("output", 1300, 100); out2.props.classes = 16;    // back to original
    const fit2 = addNode("fitness", 1300, 320); fit2.props.objective = "reconstruct_source";
    wire(sy, 0, in1, 0); wire(in1, 0, l1, 0); wire(l1, 0, out1, 0); wire(out1, 0, fit1, 0);
    wire(fit1, 0, in2, 0);                                                 // latent passes through
    wire(in2, 0, l2, 0); wire(l2, 0, out2, 0); wire(out2, 0, fit2, 0);
    [in1, l1, out1, fit1, in2, l2, out2, fit2].forEach(updateBadge);
  }

  function templateGraph(kind) {
    graph = { nodes: [], edges: [], pan: { x: 40, y: 30 } }; uid = 1; selected = null;
    if (kind === "autoencoder") tmplAutoencoder(); else tmplBasic();
    renderAll(); renderGenome(); save();
  }
  function seedDefault() { templateGraph("basic"); }

  // ════════════════════════════════════════════════════════════════════════════
  // Engine — the PURE baseline: a plain GA over the assembled network. Synthetic
  // data flows Input -> hidden layers -> Output; the Fitness node's objective is
  // what the population evolves toward. No energy / self-adaptation / evolved
  // activations yet — those are the constraints, added later, measured against
  // this. Runs in-browser, streaming best/mean fitness per generation.
  // ════════════════════════════════════════════════════════════════════════════
  const Engine = (function () {
    let running = false, raf = null, state = null;
    const rng = (seed) => { let s = seed >>> 0; return () => { s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
    const randn = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
    const act = (k, x) => k === "identity" ? x : k === "relu" ? (x > 0 ? x : 0) : k === "sigmoid" ? 1 / (1 + Math.exp(-x)) : Math.tanh(x);
    const fitLen = (a, n) => { if (a.length === n) return a.slice(); const o = []; for (let i = 0; i < n; i++) o.push(a[Math.floor(i * a.length / n)] || 0); return o; };

    // Build the spec for ONE model segment (input → its layers → output → fitness).
    function segSpec(input, layers, output, fit, source) {
      const D = Math.max(1, input.props.dims | 0), O = Math.max(1, output.props.classes | 0);
      const specs = layers.map((l) => { const u = Math.max(1, l.props.units | 0);
        return { units: u, evolveUnits: !!l.props.evolve_units,
          kSparse: !!l.props.k_sparsity, evolveK: !!(l.props.k_sparsity && l.props.evolve_k),
          k: Math.max(1, Math.min(u, (l.props.k | 0) || 1)), geneOff: null, uGeneOff: null }; });
      const arch = [D, ...specs.map((s) => s.units), O];
      const acts = [...layers.map((l) => l.props.activation === "evolved" ? "tanh" : l.props.activation), "identity"];
      const columns = [{ label: "Input", color: NODE_TYPES.input.color, n: D },
        ...layers.map((l) => ({ label: "Layer", color: NODE_TYPES.layer.color, n: Math.max(1, l.props.units | 0) })),
        { label: "Output", color: NODE_TYPES.output.color, n: O }];
      return { D, O, arch, acts, sparsity: [...specs, null], columns,
        synth: source && source.type === "synthetic" ? source.props : null,
        objective: fit ? fit.props.objective : "reconstruct", metric: fit ? fit.props.metric : "mse",
        constraints: fit ? graph.edges.filter((e) => e.to.node === fit.id).map((e) => { const s = nodeById(e.from.node); return s && s.type !== "output" ? s.type : null; }).filter(Boolean) : [] };
    }
    // Parse the graph into an ordered chain of model segments. A segment runs
    // Input → layers → Output → Fitness; the Fitness "data" output passes through
    // to the next Input, so several models can be trained in one chain, each with
    // its own fitness.
    function parseModels() {
      const source = graph.nodes.find((n) => n.type === "synthetic" || n.type === "data");
      let input = null;
      if (source) input = graph.edges.filter((e) => e.from.node === source.id).map((e) => nodeById(e.to.node)).find((x) => x && x.type === "input");
      if (!input) input = graph.nodes.find((n) => n.type === "input");
      if (!input) return { error: "add an Input node (and a Synthetic/Data source)" };
      const specs = [], seenIn = new Set();
      while (input && !seenIn.has(input.id)) {
        seenIn.add(input.id);
        const layers = []; let cur = input, output = null; const seen = new Set([input.id]);
        for (let g = 0; g < graph.nodes.length + 2; g++) {
          const targets = graph.edges.filter((e) => e.from.node === cur.id).map((e) => nodeById(e.to.node)).filter(Boolean);
          const nextLayer = targets.find((t) => t.type === "layer" && !seen.has(t.id));
          if (nextLayer) { layers.push(nextLayer); seen.add(nextLayer.id); cur = nextLayer; continue; }
          output = targets.find((t) => t.type === "output") || null; break;
        }
        if (!output) return specs.length ? { error: `model ${specs.length + 1}: wire Input → … → Output` } : { error: "wire Input → … → Output (the chain isn't connected)" };
        const fit = graph.edges.filter((e) => e.from.node === output.id).map((e) => nodeById(e.to.node)).find((x) => x && x.type === "fitness");
        specs.push(segSpec(input, layers, output, fit, specs.length === 0 ? source : null));
        input = fit ? graph.edges.filter((e) => e.from.node === fit.id).map((e) => nodeById(e.to.node)).find((x) => x && x.type === "input") : null;
      }
      return { source, specs };
    }
    // Back-compat: the first model's spec (for single-model callers/tests).
    function buildSpec() { const m = parseModels(); return m.error ? { error: m.error } : m.specs[0]; }
    function makeNet(arch) { const layers = []; let off = 0;
      for (let i = 0; i < arch.length - 1; i++) { const inn = arch[i], outn = arch[i + 1];
        layers.push({ inn, outn, wOff: off, bOff: off + inn * outn }); off += inn * outn + outn; }
      return { layers, paramCount: off };
    }
    // k-sparsity: keep the k largest-magnitude activations, zero the rest.
    function topK(o, k) {
      if (k >= o.length) return;
      const idx = Array.from(o.keys()).sort((a, b) => Math.abs(o[b]) - Math.abs(o[a]));
      for (let i = k; i < idx.length; i++) o[idx[i]] = 0;
    }
    function layerK(sp, w) {
      if (!sp || !sp.kSparse) return null;
      return sp.evolveK ? Math.max(1, Math.min(sp.units, Math.round(w[sp.geneOff]))) : Math.min(sp.k, sp.units);
    }
    // evolvable hidden-neuron count: units allocated up to the max; a gene picks
    // how many are active — the rest are masked (output 0, contribute nothing).
    function layerUnits(sp, w) {
      if (!sp || !sp.evolveUnits) return null;
      return Math.max(1, Math.min(sp.units, Math.round(w[sp.uGeneOff])));
    }
    function layerOut(net, spec, w, a, li) {
      const L = net.layers[li], o = new Float32Array(L.outn);
      for (let j = 0; j < L.outn; j++) { let s = w[L.bOff + j];
        for (let i = 0; i < L.inn; i++) s += w[L.wOff + i * L.outn + j] * a[i];
        o[j] = act(spec.acts[li], s); }
      const sp = spec.sparsity[li];
      const u = layerUnits(sp, w); if (u != null) for (let j = u; j < o.length; j++) o[j] = 0;   // width mask
      const k = layerK(sp, w); if (k != null) topK(o, k);
      return o;
    }
    function forward(net, spec, w, x) { let a = x;
      for (let li = 0; li < net.layers.length; li++) a = layerOut(net, spec, w, a, li);
      return a;
    }
    // per-layer activations for the firing visual: [input, layer1, …, output].
    function activate(net, spec, w, x) {
      const outs = [Array.from(x)]; let a = x;
      for (let li = 0; li < net.layers.length; li++) { a = layerOut(net, spec, w, a, li); outs.push(Array.from(a)); }
      return outs;
    }
    function synth1D(p, n, r) { const amp = p.amplitude != null ? p.amplitude : 1, out = [];
      for (let i = 0; i < n; i++) { const t = i / n, x = 2 * Math.PI * (p.frequency || 0) * t + (p.phase || 0); let v;
        if (p.kind === "square") v = Math.sign(Math.sin(x)) || 1;
        else if (p.kind === "ramp") v = 2 * (((p.frequency || 0) * t + (p.phase || 0) / (2 * Math.PI)) % 1) - 1;
        else if (p.kind === "noise") v = r() * 2 - 1; else v = Math.sin(x);
        out.push(v * amp); } return out;
    }
    // Target for a sample: reconstruct = this model's own input; reconstruct_source
    // = the ORIGINAL pre-encoder input carried down the chain (so a decoder is
    // scored against the source, not the latent it receives).
    function makeTarget(objective, x, x0, O) {
      return objective === "reconstruct_source" ? fitLen(x0, O) : fitLen(x, O);
    }
    // Every sample carries x0 = the original source features (= x at model 0),
    // preserved unchanged as it passes through later models.
    function buildData(spec, r) {
      const D = spec.D, O = spec.O, B = 32, data = [];
      if (state.frames) {                            // REAL decoded video frames, not mock
        return state.frames.frames.map((f) => { const x = fitLen(f, D);
          return { x: Float32Array.from(x), x0: Float32Array.from(x), y: Float32Array.from(makeTarget(spec.objective, x, x, O)) }; });
      }
      const p = spec.synth || { kind: "sine", frequency: 3, amplitude: 1, phase: 0, length: 64 };
      if (p.kind === "image") {
        const N = Math.max(2, p.size | 0), base = [];
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) base.push(imagePixel(p.pattern, x / (N - 1 || 1), y / (N - 1 || 1), N));
        const vec = fitLen(base, D);
        for (let b = 0; b < B; b++) { const x = vec.map((v) => v + (p.loop ? (r() * 0.06 - 0.03) : 0));
          data.push({ x: Float32Array.from(x), x0: Float32Array.from(x), y: Float32Array.from(makeTarget(spec.objective, x, x, O)) }); }
      } else {
        const L = Math.max(D + O + 8, (p.length | 0) || 64), sig = synth1D(p, L, r);
        for (let b = 0; b < B; b++) { const start = Math.floor(r() * L), x = [];
          for (let i = 0; i < D; i++) x.push(sig[(start + i) % L]);
          let y;
          if (spec.objective === "predict_next") { y = []; for (let i = 0; i < O; i++) y.push(sig[(start + D + i) % L]); }
          else y = makeTarget(spec.objective, x, x, O);
          data.push({ x: Float32Array.from(x), x0: Float32Array.from(x), y: Float32Array.from(y) }); }
      }
      return data;
    }
    function evalFit(net, spec, w, data, metric) { let tot = 0, cnt = 0;
      for (const s of data) { const o = forward(net, spec, w, s.x);
        for (let j = 0; j < o.length; j++) { const e = o[j] - s.y[j]; tot += metric === "mae" ? Math.abs(e) : e * e; cnt++; } }
      return -(tot / Math.max(1, cnt));
    }
    // Build one runnable model (net + genes + population + its dataset). For
    // model 0 the data comes from the source; for later models it's derived from
    // the previous model's trained output (the pass-through).
    function initModel(spec, prev) {
      const net = makeNet(spec.arch);
      let geneOff = net.paramCount; const genes = [];
      spec.sparsity.forEach((sp) => { if (!sp) return;
        if (sp.evolveUnits) { sp.uGeneOff = geneOff; genes.push({ off: geneOff, units: sp.units }); geneOff++; }
        if (sp.kSparse && sp.evolveK) { sp.geneOff = geneOff; genes.push({ off: geneOff, units: sp.units }); geneOff++; } });
      const paramCount = geneOff;
      const mutScale = new Float32Array(paramCount).fill(state.mut);
      genes.forEach((gg) => { mutScale[gg.off] = Math.max(state.mut, gg.units * 0.2); });
      let data;
      if (prev) {                                    // pass-through: prev outputs → this input;
        data = prev.data.map((s) => {                // x0 (original source) carried unchanged
          const o = forward(prev.net, prev.spec, prev.best, s.x), x = fitLen(Array.from(o), spec.D);
          return { x: Float32Array.from(x), x0: s.x0, y: Float32Array.from(makeTarget(spec.objective, x, Array.from(s.x0), spec.O)) };
        });
      } else {
        data = buildData(spec, state.r);
      }
      const pop = [];
      for (let i = 0; i < state.P; i++) { const g = new Float32Array(paramCount);
        for (let k = 0; k < net.paramCount; k++) g[k] = randn(state.r) * 0.5;
        genes.forEach((gg) => { g[gg.off] = 1 + Math.floor(state.r() * gg.units); }); pop.push(g); }
      return { spec, net, mutScale, data, pop, gens: state.gensPerModel, gen: 0, best: null, bestFit: -Infinity, hist: [] };
    }
    function stepModel(m) {
      const scored = m.pop.map((g) => ({ g, f: evalFit(m.net, m.spec, g, m.data, m.spec.metric) })).sort((a, b) => b.f - a.f);
      const elite = Math.max(1, Math.floor(state.P * state.survive));
      m.best = scored[0].g; m.bestFit = scored[0].f;
      const mean = scored.reduce((s, x) => s + x.f, 0) / state.P;
      const next = [];
      for (let i = 0; i < elite; i++) next.push(scored[i].g);
      while (next.length < state.P) { const parent = scored[Math.floor(state.r() * elite)].g, child = Float32Array.from(parent);
        for (let k = 0; k < child.length; k++) if (state.r() < 0.5) child[k] += randn(state.r) * m.mutScale[k]; next.push(child); }
      m.pop = next; m.gen++; m.hist.push(m.bestFit);
      return { gen: m.gen, best: m.bestFit, mean };
    }
    function sampleModel(m) { if (!m.best || !m.data.length) return null;
      const s = m.data[0], acts = activate(m.net, m.spec, m.best, s.x);
      const wLayers = m.net.layers.map((L) => ({ inn: L.inn, outn: L.outn, w: m.best.subarray(L.wOff, L.wOff + L.inn * L.outn) }));
      const dims = [m.spec.D];
      m.net.layers.forEach((L, li) => { const u = layerUnits(m.spec.sparsity[li], m.best); dims.push(u != null ? u : L.outn); });
      // per-frame reconstruction grid: best prediction for EVERY frame (only
      // when the output is a square frame — the decoder, not a latent encoder).
      const N = Math.round(Math.sqrt(m.spec.O));
      const recon = { N: (N >= 2 && N * N === m.spec.O) ? N : 0, frames: [] };
      if (recon.N) {
        const K = Math.min(48, m.data.length);
        for (let i = 0; i < K; i++) { const d = m.data[i], o = forward(m.net, m.spec, m.best, d.x);
          recon.frames.push({ y: Array.from(d.y), o: Array.from(o) }); }
      }
      return { y: Array.from(s.y), o: acts[acts.length - 1], activations: acts, wLayers, dims, columns: m.spec.columns, recon };
    }
    function loop() {
      if (!running || !state) return;
      const m = state.models[state.cur];
      let t = { gen: m.gen, best: m.bestFit, mean: 0 };
      for (let i = 0; i < 3 && m.gen < m.gens; i++) t = stepModel(m);
      state.onTick(t, sampleModel(m), m.hist, { cur: state.cur, total: state.specs.length });
      if (m.gen >= m.gens) {
        if (state.cur + 1 < state.specs.length) {    // freeze, move to the next model
          state.cur++;
          state.models[state.cur] = initModel(state.specs[state.cur], m);
          raf = requestAnimationFrame(loop); return;
        }
        running = false; state.onDone(); return;
      }
      raf = requestAnimationFrame(loop);
    }
    async function start(opts) {
      const parsed = parseModels();
      if (parsed.error) { opts.onError(parsed.error); return; }
      // if the source is a Data (video) node, decode its frames NOW so training
      // runs on the real video, not synthetic fallback.
      let frames = null;
      if (parsed.source && parsed.source.type === "data") {
        opts.onStatus && opts.onStatus("decoding video frames…");
        const res = await acquireFrames(parsed.source.props, dataFiles[parsed.source.id]);
        if (res.error) { opts.onError(res.error); return; }
        frames = res;
        opts.onStatus && opts.onStatus(`decoded ${res.frames.length} frames (${res.size}×${res.size}${res.gray ? " gray" : ""}) — training on the real video`);
      }
      state = { specs: parsed.specs, models: [], cur: 0, r: rng(opts.seed || 1234), frames,
        P: Math.max(4, opts.pop || 200), gensPerModel: opts.gens || 1000, mut: 0.08, survive: 0.2,
        onTick: opts.onTick, onDone: opts.onDone };
      state.models[0] = initModel(parsed.specs[0], null);
      opts.onStart && opts.onStart(parsed.specs[0], parsed.specs.length, !!frames);
      running = true; loop();
    }
    function stop() { running = false; if (raf) cancelAnimationFrame(raf); }
    return { start, stop, isRunning: () => running, buildSpec, parseModels };
  })();

  // ── training readout: fitness sparkline + best-output-vs-target plot ─────────
  function drawFitChart(hist) {
    const cv = document.getElementById("pu-fitchart"); if (!cv || !cv.getContext) return;
    const ctx = cv.getContext("2d"); if (!ctx) return; const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    if (!hist || hist.length < 2) return;
    const mn = Math.min(...hist), mx = Math.max(...hist), rng = (mx - mn) || 1;
    ctx.strokeStyle = "#4ea1ff"; ctx.lineWidth = 1.5; ctx.beginPath();
    hist.forEach((v, i) => { const x = (i / (hist.length - 1)) * W, y = H - ((v - mn) / rng) * (H - 4) - 2;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  function drawOutPlot(sample) {
    const cv = document.getElementById("pu-outplot"); if (!cv || !cv.getContext) return;
    const ctx = cv.getContext("2d"); if (!ctx) return; const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    if (!sample) return;
    // if the output is a square frame (video/image), show target vs reconstruction
    // as side-by-side images — so you can actually SEE the model working.
    const N = Math.round(Math.sqrt(sample.o.length));
    if (N >= 4 && N * N === sample.o.length) {
      const drawImg = (arr, ox) => {
        const cell = H / N;
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
          const g = Math.max(0, Math.min(255, Math.round((arr[y * N + x] + 1) / 2 * 255)));
          ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(ox + x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
        }
      };
      drawImg(sample.y, 0);              // target (left)
      drawImg(sample.o, W - H);          // reconstruction (right)
      ctx.fillStyle = "#7d8794"; ctx.font = "9px monospace";
      ctx.fillText("target", 2, H - 2); ctx.fillText("output", W - H + 2, H - 2);
      return;
    }
    const all = sample.y.concat(sample.o), mn = Math.min(...all), mx = Math.max(...all), rng = (mx - mn) || 1;
    const line = (arr, color) => { ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
      arr.forEach((v, i) => { const x = arr.length === 1 ? W / 2 : (i / (arr.length - 1)) * W, y = H - ((v - mn) / rng) * (H - 4) - 2;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); };
    line(sample.y, "#7d8794");     // target
    line(sample.o, "#3fb950");     // best output
  }
  // Reconstruction grid: for every frame, target (top row) and the model's best
  // reconstruction (bottom row), as N×N images.
  function drawRecon(sample) {
    const cv = document.getElementById("pu-recon"); if (!cv || !cv.getContext) return;
    const ctx = cv.getContext("2d"); if (!ctx) return;
    const rec = sample && sample.recon;
    if (!rec || !rec.N || !rec.frames.length) { cv.width = 10; cv.height = 10; ctx.clearRect(0, 0, 10, 10); return; }
    const N = rec.N, F = rec.frames.length, cell = 52, pad = 5, labelH = 13;
    cv.width = pad + F * (cell + pad);
    cv.height = labelH + cell + pad + labelH + cell + pad;
    ctx.clearRect(0, 0, cv.width, cv.height);
    const drawFrame = (arr, ox, oy) => { const c = cell / N;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const g = Math.max(0, Math.min(255, Math.round((arr[y * N + x] + 1) / 2 * 255)));
        ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(ox + x * c, oy + y * c, Math.ceil(c), Math.ceil(c));
      } };
    ctx.fillStyle = "#7d8794"; ctx.font = "10px monospace";
    ctx.fillText("targets", 2, labelH - 2);
    rec.frames.forEach((f, i) => drawFrame(f.y, pad + i * (cell + pad), labelH));
    ctx.fillStyle = "#3fb950"; ctx.fillText("reconstruction", 2, labelH + cell + pad + labelH - 2);
    rec.frames.forEach((f, i) => drawFrame(f.o, pad + i * (cell + pad), labelH + cell + pad + labelH));
  }

  // ── public API for later GA wiring ──────────────────────────────────────────
  window.PureGraph = {
    getGraph: () => JSON.parse(JSON.stringify({
      nodes: graph.nodes.map((n) => ({ id: n.id, type: n.type, x: n.x, y: n.y, props: n.props })),
      edges: graph.edges,
    })),
    clear: () => { graph = { nodes: [], edges: [], pan: { x: 40, y: 30 } }; uid = 1; selectNode(null); renderAll(); save(); },
    template: (kind) => { templateGraph(kind); selectNode(null); },
    addNode,
    types: () => NODE_TYPES,
  };
  window.PureEngine = Engine;

  // ── boot ────────────────────────────────────────────────────────────────────
  function addButton(bar, type) {
    const b = document.createElement("button");
    b.className = "pg-add"; b.textContent = "+ " + NODE_TYPES[type].title;
    b.style.setProperty("--pg-accent", NODE_TYPES[type].color);
    b.addEventListener("click", () => {
      const n = addNode(type, -graph.pan.x + 60 + (graph.nodes.length % 6) * 26,
                               -graph.pan.y + 60 + (graph.nodes.length % 6) * 26);
      selectNode(n);
    });
    bar.appendChild(b);
  }
  const TB_GROUPS = [["data", "Data"], ["structure", "Structure"], ["objective", "Objective"], ["constraint", "Constraints"]];
  function buildToolbar(bar) {
    TB_GROUPS.forEach(([g, label]) => {
      const keys = Object.keys(NODE_TYPES).filter((t) => NODE_TYPES[t].group === g);
      if (!keys.length) return;
      const lab = document.createElement("span");
      lab.className = "pg-tb-label" + (g === "constraint" ? " pg-tb-con" : "");
      lab.textContent = label + ":";
      bar.appendChild(lab);
      keys.forEach((t) => addButton(bar, t));
    });
    const spacer = document.createElement("span"); spacer.style.flex = "1"; bar.appendChild(spacer);
    const tsel = document.createElement("select");
    tsel.className = "runs-btn"; tsel.title = "Load a correctly-wired example graph";
    [["", "Template…"], ["basic", "Basic model"], ["autoencoder", "Autoencoder"]].forEach(([v, l]) => {
      const o = document.createElement("option"); o.value = v; o.textContent = l; tsel.appendChild(o);
    });
    tsel.addEventListener("change", () => {
      const k = tsel.value; tsel.value = "";
      if (k && confirm(`Load the ${k} template? This replaces the current graph.`)) { templateGraph(k); selectNode(null); }
    });
    bar.appendChild(tsel);
    const clear = document.createElement("button");
    clear.className = "runs-btn"; clear.textContent = "Clear";
    clear.addEventListener("click", () => { if (confirm("Clear the whole graph?")) window.PureGraph.clear(); });
    bar.appendChild(clear);
  }

  // ── resizable panels (left run / right properties / genome split) ───────────
  function initResizers() {
    const layout = document.querySelector(".tlm-layout");
    if (!layout) return;
    const targets = {
      left: { el: layout.querySelector("aside.sidebar.left"), dim: "width", dir: 1, min: 150, max: 560, key: "pure.size.left" },
      right: { el: layout.querySelector("aside.sidebar.right"), dim: "width", dir: -1, min: 170, max: 620, key: "pure.size.right" },
      genome: { el: document.querySelector(".pg-genome-panel"), dim: "height", dir: -1, min: 60, key: "pure.size.genome",
                maxFn: (el) => (el.parentElement ? el.parentElement.clientHeight - 130 : 600) },
      recon: { el: document.querySelector(".pg-recon-panel"), dim: "height", dir: -1, min: 50, key: "pure.size.recon",
                maxFn: (el) => (el.parentElement ? el.parentElement.clientHeight - 130 : 600) },
    };
    // apply saved sizes
    for (const k in targets) {
      const t = targets[k];
      if (!t.el) continue;
      let v = null;
      try { v = localStorage.getItem(t.key); } catch (_) {}
      if (v) t.el.style[t.dim] = v;
    }
    document.querySelectorAll("[data-resize]").forEach((h) => {
      const t = targets[h.dataset.resize];
      if (!t || !t.el) return;
      h.addEventListener("mousedown", (e) => {
        e.preventDefault();
        const horiz = t.dim === "width";
        const startPos = horiz ? e.clientX : e.clientY;
        const startSize = t.el.getBoundingClientRect()[t.dim];
        h.classList.add("pg-resizing");
        document.body.style.userSelect = "none";
        document.body.style.cursor = horiz ? "col-resize" : "row-resize";
        function mv(ev) {
          const pos = horiz ? ev.clientX : ev.clientY;
          const max = t.maxFn ? t.maxFn(t.el) : t.max;
          const size = Math.max(t.min, Math.min(max, startSize + t.dir * (pos - startPos)));
          t.el.style[t.dim] = size + "px";
        }
        function up() {
          document.removeEventListener("mousemove", mv);
          document.removeEventListener("mouseup", up);
          h.classList.remove("pg-resizing");
          document.body.style.userSelect = "";
          document.body.style.cursor = "";
          try { localStorage.setItem(t.key, t.el.style[t.dim]); } catch (_) {}
        }
        document.addEventListener("mousemove", mv);
        document.addEventListener("mouseup", up);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    canvas = document.getElementById("pure-graph");
    propsHost = document.getElementById("pure-props");
    genomeHost = document.getElementById("pure-genome-viz");
    const bar = document.getElementById("pg-toolbar");
    if (!canvas) return;

    world = document.createElement("div"); world.className = "pg-world";
    svg = document.createElementNS(NS, "svg"); svg.setAttribute("class", "pg-svg");
    world.appendChild(svg);
    canvas.appendChild(world);
    if (bar) buildToolbar(bar);

    if (!load()) seedDefault();
    renderAll();
    renderGenome();
    selectNode(null);
    initResizers();

    canvas.addEventListener("mousedown", startPan);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    document.addEventListener("keydown", (e) => {
      if ((e.key === "Delete" || e.key === "Backspace") && selected) {
        const tag = (document.activeElement && document.activeElement.tagName) || "";
        if (tag !== "INPUT" && tag !== "SELECT" && tag !== "TEXTAREA") { e.preventDefault(); removeNode(selected); }
      }
    });

    // Run / Stop — drive the in-browser GA engine over the assembled graph.
    const statusEl = document.getElementById("pu-status");
    const metricEl = document.getElementById("pu-metric");
    const runBtn = document.getElementById("pu-run");
    const stopBtn = document.getElementById("pu-stop");
    const resetBtn = document.getElementById("pu-reset");
    const numVal = (id, dflt) => { const el = document.getElementById(id); const v = el ? parseInt(el.value, 10) : NaN; return isNaN(v) ? dflt : v; };
    function setRunning(on) {
      if (runBtn) runBtn.disabled = on;
      if (stopBtn) stopBtn.disabled = !on;
    }
    if (runBtn) runBtn.addEventListener("click", () => {
      if (Engine.isRunning()) return;
      const gens = numVal("pu-gens", 1000);
      setRunning(true);
      Engine.start({
        pop: numVal("pu-pop", 200), gens, seed: numVal("pu-seed", 1234),
        onStatus: (msg) => { if (statusEl) statusEl.textContent = msg; },
        onStart: (spec, nModels, realVideo) => { setRunning(true); if (statusEl) statusEl.textContent =
          (realVideo ? "training on REAL video · " : "training on synthetic data · ") + (nModels > 1 ? `${nModels} models — ` : "") + `network ${spec.arch.join("→")}, objective ${spec.objective}` + (spec.constraints.length ? `, constraints: ${spec.constraints.join(", ")}` : ""); },
        onError: (msg) => { setRunning(false); if (statusEl) statusEl.textContent = "cannot run: " + msg; },
        onTick: (t, sample, hist, prog) => {
          const modelTag = prog && prog.total > 1 ? `model ${prog.cur + 1}/${prog.total} · ` : "";
          if (metricEl) metricEl.textContent = `${modelTag}gen ${t.gen}/${gens} · best ${(-t.best).toExponential(2)} · mean ${(-t.mean).toExponential(2)} err`;
          drawFitChart(hist); drawOutPlot(sample); drawRecon(sample);
          renderGenome(sample && sample.activations, sample && sample.wLayers, sample && sample.dims, sample && sample.columns);   // active model fires
        },
        onDone: () => { setRunning(false); if (statusEl) statusEl.textContent = "run complete — all models trained; edit the graph and Run again."; },
      });
    });
    if (stopBtn) stopBtn.addEventListener("click", () => { Engine.stop(); setRunning(false); if (statusEl) statusEl.textContent = "stopped — press Run to resume from a fresh population, or Reset to clear."; });
    if (resetBtn) resetBtn.addEventListener("click", () => {
      Engine.stop(); setRunning(false);
      if (metricEl) metricEl.textContent = "idle";
      drawFitChart([]); drawOutPlot(null); drawRecon(null); renderGenome();   // clear charts + static genome
      if (statusEl) statusEl.textContent = "reset — the graph is kept; press Run to train a fresh population.";
    });
  });
})();
