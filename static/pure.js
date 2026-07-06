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
      badge: (p) => `${p.units} · ${p.activation}` + (p.k_sparsity ? ` · k${p.evolve_k ? "*" : p.k}` : ""),
      props: [
        { key: "units", type: "number", label: "Units", default: 24, min: 1, max: 8192 },
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
    fitness: {
      group: "objective", title: "Fitness", color: "#d0679a", inputs: [], outputs: ["fit"],
      dynamicInputs: true, badge: (p) => p.objective,
      props: [
        { key: "objective", type: "select", label: "Objective", default: "reconstruct",
          options: ["reconstruct", "predict_next"] },
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

    synthPreviewHost = null;
    if (n.type === "synthetic") {
      const cap = document.createElement("div");
      cap.className = "pg-prop-id"; cap.style.marginTop = "8px"; cap.textContent = "Preview";
      propsHost.appendChild(cap);
      synthPreviewHost = document.createElement("div");
      synthPreviewHost.className = "pg-preview";
      propsHost.appendChild(synthPreviewHost);
      drawSynthPreview(n);
    }

    const del = document.createElement("button");
    del.className = "runs-btn pg-del";
    del.textContent = "Delete node";
    del.addEventListener("click", () => removeNode(n));
    propsHost.appendChild(del);
  }

  // ── synthetic data preview (waveform sparkline or image pattern) ────────────
  let synthPreviewHost = null;
  function updatePreview() {
    if (selected && selected.type === "synthetic" && synthPreviewHost) drawSynthPreview(selected);
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
  function renderGenome(activations, wLayers) {
    if (!genomeHost) return;
    genomeHost.replaceChildren();
    const cols = orderedStructure();
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
      const L = colLayoutG(col.n);
      return { col, cx: colX[ci], dots: L.dots, gap: L.gap, r: L.r, vals: activations && activations[ci] };
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
      const { col, cx, dots, gap, r, vals } = Lo;
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
      const sub = svgn("text", { x: cx, y: GVH - GM_BOT + 26, "text-anchor": "middle",
        fill: "var(--muted)", "font-size": 11, "font-family": "var(--mono, monospace)" });
      sub.textContent = col.n === 1 ? "1 unit" : `${col.n} units`;
      svg.appendChild(sub);
    });
    genomeHost.appendChild(svg);
  }

  // The canonical, correctly-wired graph. Data flows left-to-right through the
  // model; the Output feeds the Fitness node (so its objective can score the
  // model), and a constraint is wired into Fitness to show the pattern.
  //   Synthetic → Input → Layer → Output → Fitness
  //                                Energy ─┘  (constraint into a Fitness port)
  function templateGraph() {
    graph = { nodes: [], edges: [], pan: { x: 40, y: 30 } }; uid = 1; selected = null;
    const sy = addNode("synthetic", 20, 100);
    const inp = addNode("input", 250, 100);
    const lay = addNode("layer", 480, 100);
    const out = addNode("output", 710, 100);
    const fit = addNode("fitness", 710, 320);
    const en = addNode("energy", 480, 340);
    graph.edges.push(
      { from: { node: sy.id, idx: 0 }, to: { node: inp.id, idx: 0 } },   // synthetic → input.data
      { from: { node: inp.id, idx: 0 }, to: { node: lay.id, idx: 0 } },  // input → hidden
      { from: { node: lay.id, idx: 0 }, to: { node: out.id, idx: 0 } },  // hidden → output
      { from: { node: out.id, idx: 0 }, to: { node: fit.id, idx: 0 } },  // output → fitness (scored)
      { from: { node: en.id, idx: 0 }, to: { node: fit.id, idx: 1 } });  // energy constraint → fitness
    renderAll(); renderGenome(); save();
  }
  function seedDefault() { templateGraph(); }

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

    function buildSpec() {
      const inp = graph.nodes.find((n) => n.type === "input");
      const out = graph.nodes.find((n) => n.type === "output");
      if (!inp || !out) return { error: "add an Input and an Output node" };
      const path = wiredPath();
      if (path[path.length - 1] !== out) return { error: "wire Input → … → Output (the chain isn't connected)" };
      const layers = path.filter((n) => n.type === "layer");   // only wired-in layers, in flow order
      const synth = graph.nodes.find((n) => n.type === "synthetic");
      const fit = graph.nodes.find((n) => n.type === "fitness");
      const D = Math.max(1, inp.props.dims | 0), O = Math.max(1, out.props.classes | 0);
      const specs = layers.map((l) => { const u = Math.max(1, l.props.units | 0);
        return { units: u, kSparse: !!l.props.k_sparsity, evolveK: !!(l.props.k_sparsity && l.props.evolve_k),
          k: Math.max(1, Math.min(u, (l.props.k | 0) || 1)), geneOff: null }; });
      const arch = [D, ...specs.map((s) => s.units), O];
      const acts = [...layers.map((l) => l.props.activation === "evolved" ? "tanh" : l.props.activation), "identity"];
      const sparsity = [...specs, null];             // one per net transition; last (→output) has none
      return { D, O, arch, acts, sparsity, synth: synth ? synth.props : null,
        objective: fit ? fit.props.objective : "reconstruct", metric: fit ? fit.props.metric : "mse",
        constraints: fit ? graph.edges.filter((e) => e.to.node === fit.id).map((e) => { const s = nodeById(e.from.node); return s && s.type !== "output" ? s.type : null; }).filter(Boolean) : [] };
    }
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
    function forward(net, spec, w, x) { let a = x;
      for (let li = 0; li < net.layers.length; li++) { const L = net.layers[li], o = new Float32Array(L.outn);
        for (let j = 0; j < L.outn; j++) { let s = w[L.bOff + j];
          for (let i = 0; i < L.inn; i++) s += w[L.wOff + i * L.outn + j] * a[i];
          o[j] = act(spec.acts[li], s); }
        const k = layerK(spec.sparsity[li], w); if (k != null) topK(o, k);
        a = o; }
      return a;
    }
    // per-layer activations for the firing visual: [input, layer1, …, output].
    function activate(net, spec, w, x) {
      const outs = [Array.from(x)]; let a = x;
      for (let li = 0; li < net.layers.length; li++) { const L = net.layers[li], o = new Float32Array(L.outn);
        for (let j = 0; j < L.outn; j++) { let s = w[L.bOff + j];
          for (let i = 0; i < L.inn; i++) s += w[L.wOff + i * L.outn + j] * a[i];
          o[j] = act(spec.acts[li], s); }
        const k = layerK(spec.sparsity[li], w); if (k != null) topK(o, k);
        a = o; outs.push(Array.from(o)); }
      return outs;
    }
    function synth1D(p, n, r) { const amp = p.amplitude != null ? p.amplitude : 1, out = [];
      for (let i = 0; i < n; i++) { const t = i / n, x = 2 * Math.PI * (p.frequency || 0) * t + (p.phase || 0); let v;
        if (p.kind === "square") v = Math.sign(Math.sin(x)) || 1;
        else if (p.kind === "ramp") v = 2 * (((p.frequency || 0) * t + (p.phase || 0) / (2 * Math.PI)) % 1) - 1;
        else if (p.kind === "noise") v = r() * 2 - 1; else v = Math.sin(x);
        out.push(v * amp); } return out;
    }
    function buildData(spec, r) {
      const D = spec.D, O = spec.O, B = 32, data = [];
      const p = spec.synth || { kind: "sine", frequency: 3, amplitude: 1, phase: 0, length: 64 };
      if (p.kind === "image") {
        const N = Math.max(2, p.size | 0), base = [];
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) base.push(imagePixel(p.pattern, x / (N - 1 || 1), y / (N - 1 || 1), N));
        const vec = fitLen(base, D);
        for (let b = 0; b < B; b++) { const x = vec.map((v) => v + (p.loop ? (r() * 0.06 - 0.03) : 0));
          data.push({ x: Float32Array.from(x), y: Float32Array.from(fitLen(x, O)) }); }
      } else {
        const L = Math.max(D + O + 8, (p.length | 0) || 64), sig = synth1D(p, L, r);
        for (let b = 0; b < B; b++) { const start = Math.floor(r() * L), x = [];
          for (let i = 0; i < D; i++) x.push(sig[(start + i) % L]);
          let y;
          if (spec.objective === "predict_next") { y = []; for (let i = 0; i < O; i++) y.push(sig[(start + D + i) % L]); }
          else y = fitLen(x, O);
          data.push({ x: Float32Array.from(x), y: Float32Array.from(y) }); }
      }
      return data;
    }
    function evalFit(net, spec, w, data, metric) { let tot = 0, cnt = 0;
      for (const s of data) { const o = forward(net, spec, w, s.x);
        for (let j = 0; j < o.length; j++) { const e = o[j] - s.y[j]; tot += metric === "mae" ? Math.abs(e) : e * e; cnt++; } }
      return -(tot / Math.max(1, cnt));
    }
    function step() {
      const st = state, scored = st.pop.map((g) => ({ g, f: evalFit(st.net, st.spec, g, st.data, st.spec.metric) })).sort((a, b) => b.f - a.f);
      const elite = Math.max(1, Math.floor(st.P * st.survive));
      st.best = scored[0].g; st.bestFit = scored[0].f;
      const mean = scored.reduce((s, x) => s + x.f, 0) / st.P;
      const next = [];
      for (let i = 0; i < elite; i++) next.push(scored[i].g);
      while (next.length < st.P) { const parent = scored[Math.floor(st.r() * elite)].g, child = Float32Array.from(parent);
        for (let k = 0; k < child.length; k++) if (st.r() < 0.5) child[k] += randn(st.r) * st.mutScale[k]; next.push(child); }
      st.pop = next; st.gen++; st.hist.push(st.bestFit);
      return { gen: st.gen, best: st.bestFit, mean };
    }
    function sampleOutput() { const st = state; if (!st.best || !st.data.length) return null;
      const s = st.data[0], acts = activate(st.net, st.spec, st.best, s.x);
      // weight matrices per layer transition (column k → column k+1), for the
      // genome visual's connection lines.
      const wLayers = st.net.layers.map((L) => ({ inn: L.inn, outn: L.outn,
        w: st.best.subarray(L.wOff, L.wOff + L.inn * L.outn) }));
      return { y: Array.from(s.y), o: acts[acts.length - 1], activations: acts, wLayers };
    }
    function loop() {
      if (!running || !state) return;
      let t = { gen: state.gen, best: state.bestFit, mean: 0 };
      for (let i = 0; i < 3 && state.gen < state.gens; i++) t = step();
      state.onTick(t, sampleOutput(), state.hist);
      if (state.gen >= state.gens) { running = false; state.onDone(); return; }
      raf = requestAnimationFrame(loop);
    }
    function start(opts) {
      const spec = buildSpec();
      if (spec.error) { opts.onError(spec.error); return; }
      const r = rng(opts.seed || 1234), net = makeNet(spec.arch), data = buildData(spec, r);
      // append one gene per evolving-k layer, past the weight params.
      let geneOff = net.paramCount; const kGenes = [];
      spec.sparsity.forEach((sp) => { if (sp && sp.kSparse && sp.evolveK) { sp.geneOff = geneOff; kGenes.push({ off: geneOff, units: sp.units, k0: Math.min(sp.k, sp.units) }); geneOff++; } });
      const paramCount = geneOff, mut = 0.08;
      const mutScale = new Float32Array(paramCount).fill(mut);
      kGenes.forEach((kg) => { mutScale[kg.off] = Math.max(mut, kg.units * 0.2); });   // k moves in unit steps
      const P = Math.max(4, opts.pop || 200), pop = [];
      for (let i = 0; i < P; i++) { const g = new Float32Array(paramCount);
        for (let k = 0; k < net.paramCount; k++) g[k] = randn(r) * 0.5;
        kGenes.forEach((kg) => { g[kg.off] = kg.k0; }); pop.push(g); }
      state = { spec, net, data, r, P, pop, gens: opts.gens || 1000, gen: 0, mut, mutScale, survive: 0.2,
        best: null, bestFit: -Infinity, hist: [], onTick: opts.onTick, onDone: opts.onDone };
      opts.onStart && opts.onStart(spec);
      running = true; loop();
    }
    function stop() { running = false; if (raf) cancelAnimationFrame(raf); }
    return { start, stop, isRunning: () => running, buildSpec };
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
    const all = sample.y.concat(sample.o), mn = Math.min(...all), mx = Math.max(...all), rng = (mx - mn) || 1;
    const line = (arr, color) => { ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
      arr.forEach((v, i) => { const x = arr.length === 1 ? W / 2 : (i / (arr.length - 1)) * W, y = H - ((v - mn) / rng) * (H - 4) - 2;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); };
    line(sample.y, "#7d8794");     // target
    line(sample.o, "#3fb950");     // best output
  }

  // ── public API for later GA wiring ──────────────────────────────────────────
  window.PureGraph = {
    getGraph: () => JSON.parse(JSON.stringify({
      nodes: graph.nodes.map((n) => ({ id: n.id, type: n.type, x: n.x, y: n.y, props: n.props })),
      edges: graph.edges,
    })),
    clear: () => { graph = { nodes: [], edges: [], pan: { x: 40, y: 30 } }; uid = 1; selectNode(null); renderAll(); save(); },
    template: () => { templateGraph(); selectNode(null); },
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
    const tmpl = document.createElement("button");
    tmpl.className = "runs-btn"; tmpl.textContent = "Template";
    tmpl.title = "Load a correctly-wired example: Synthetic → Input → Layer → Output → Fitness";
    tmpl.addEventListener("click", () => { if (confirm("Load the wired example? This replaces the current graph.")) { templateGraph(); selectNode(null); } });
    bar.appendChild(tmpl);
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
      Engine.start({
        pop: numVal("pu-pop", 200), gens, seed: numVal("pu-seed", 1234),
        onStart: (spec) => { setRunning(true); if (statusEl) statusEl.textContent =
          `running — network ${spec.arch.join("→")}, objective ${spec.objective}` + (spec.constraints.length ? `, constraints: ${spec.constraints.join(", ")}` : ""); },
        onError: (msg) => { if (statusEl) statusEl.textContent = "cannot run: " + msg; },
        onTick: (t, sample, hist) => {
          if (metricEl) metricEl.textContent = `gen ${t.gen}/${gens} · best ${(-t.best).toExponential(2)} · mean ${(-t.mean).toExponential(2)} err`;
          drawFitChart(hist); drawOutPlot(sample);
          renderGenome(sample && sample.activations, sample && sample.wLayers);   // fires + connections
        },
        onDone: () => { setRunning(false); if (statusEl) statusEl.textContent = "run complete — best genome converged; edit the graph and Run again."; },
      });
    });
    if (stopBtn) stopBtn.addEventListener("click", () => { Engine.stop(); setRunning(false); if (statusEl) statusEl.textContent = "stopped — press Run to resume from a fresh population, or Reset to clear."; });
    if (resetBtn) resetBtn.addEventListener("click", () => {
      Engine.stop(); setRunning(false);
      if (metricEl) metricEl.textContent = "idle";
      drawFitChart([]); drawOutPlot(null); renderGenome();   // clear charts + static genome
      if (statusEl) statusEl.textContent = "reset — the graph is kept; press Run to train a fresh population.";
    });
  });
})();
