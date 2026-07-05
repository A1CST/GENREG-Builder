/* GENREG — Tree-of-Models page.
 * One WebSocket (/treelm) carries config → training events → generation.
 * All charts are hand-rolled SVG: an icicle map of the routing tree (x = byte
 * range 0–255, y = depth), a live fitness line chart for the node currently
 * evolving, and grouped bars of mean accuracy per depth.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

  // validated palette (dark surface #161b22) — roles, not raw hex, live in CSS
  const S1 = () => css("--tlm-s1");        // blue   — series 1 / best
  const S2 = () => css("--tlm-s2");        // aqua   — series 2 / mean
  const RAMP = ["#184f95", "#1c5cab", "#256abf", "#2a78d6",
                "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6"]; // blue seq, dark-safe end first
  const rampColor = (v) => RAMP[Math.max(0, Math.min(RAMP.length - 1,
    Math.floor(v * RAMP.length)))];
  const inkFor = (v) => (v >= 5 / 8 ? "#0b1420" : "#eef3fa"); // label ink on ramp fills

  // ── state ────────────────────────────────────────────────
  let ws = null, running = false, modelReady = false;
  let nodes = new Map(), nodeOrder = [], totalNodes = 0, doneCount = 0;
  let maxGens = 60;
  let fit = { id: null, kind: "", gens: [], best: [], mean: [] };
  let icicleQueued = false;
  let sweep = null;

  // ── websocket ────────────────────────────────────────────
  function connect() {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/treelm`);
    ws.onopen = () => { setConn(true); send({ op: "encoders" }); };
    ws.onclose = () => { setConn(false); running = false; setButtons(); setTimeout(connect, 1500); };
    ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (_) {} };
  }
  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
  function setConn(ok) {
    $("tlm-dot").className = "dot" + (ok ? " ok" : " bad");
    $("tlm-conn").textContent = ok ? "connected" : "disconnected";
  }

  function handle(ev) {
    switch (ev.type) {
      case "model":
        modelReady = !!ev.available;
        if (modelReady && ev.info) showEval(ev.info);
        setButtons();
        break;
      case "status": $("tlm-status").textContent = ev.message; break;
      case "run": if (ev.id) $("tlm-status").textContent = `run ${ev.id}`; break;
      case "job":                    // sent after snapshot replay on (re)attach
        running = !!ev.running;
        setButtons();
        break;
      case "corpus":
        $("tlm-corpus").textContent =
          `${ev.source} — ${ev.train_samples.toLocaleString()} train / ${ev.test_samples.toLocaleString()} test windows`;
        break;
      case "tree": onTree(ev); break;
      case "node_start": onNodeStart(ev); break;
      case "node_gen":
        onNodeGen(ev);
        if (String(ev.id).startsWith("encoder"))
          $("enc-status").textContent =
            `evolving ${ev.id} — gen ${ev.gen}, best ${(+ev.best).toFixed(3)}` +
            (ev.dim ? `, dim ${ev.dim}` : "");
        break;
      case "node_done": onNodeDone(ev); break;
      case "eval": showEval(ev); break;
      case "done":
        running = false;
        if (!ev.encoder_only) modelReady = true;
        setButtons();
        $("tlm-status").textContent = `done in ${ev.seconds}s` + (ev.saved ? (ev.encoder_only ? " — encoder saved" : " — model saved") : "");
        if (ev.encoder_only) $("enc-status").textContent = `done in ${ev.seconds}s — saved as ${ev.run_id}`;
        break;
      case "stopped":
        running = false; setButtons();
        $("tlm-status").textContent = ev.encoder_only ? "encoder training stopped" : `stopped after ${ev.trained_nodes} nodes`;
        if (ev.encoder_only) $("enc-status").textContent = "stopped";
        break;
      case "encoders":
        renderEncoders(ev.list || []);
        break;
      case "error":
        running = false; setButtons();
        $("tlm-status").textContent = "error: " + ev.message;
        break;
      case "trace":
        TraceView.mount($("tr-strip"), $("tr-detail"), ev);
        setButtons();
        break;
      case "sweep_start":
        sweep = { total: ev.total, results: [], done: false };
        $("sweep-status").textContent =
          `0/${ev.total} candidates` + (ev.truncated ? " (grid truncated to 24)" : "");
        renderSweep();
        break;
      case "sweep_progress":
        $("sweep-status").textContent = `${ev.i + 1}/${ev.total}: ${ev.name}`;
        break;
      case "sweep_result":
        if (sweep) { sweep.results.push(ev); renderSweep(); }
        break;
      case "sweep_done":
        running = false; setButtons();
        if (sweep) {
          sweep.results = ev.results; sweep.done = true;
          renderSweep();
        }
        $("sweep-status").textContent = (ev.stopped ? "stopped — " : "done — ") +
          `${ev.results.length} candidates, saved as ${ev.id}`;
        $("tlm-status").textContent = "sweep " + (ev.stopped ? "stopped" : "complete");
        break;
      case "generated": {
        const out = $("gen-out");
        out.textContent = "";
        const b = document.createElement("b");
        b.textContent = ev.prompt;
        out.appendChild(b);
        out.appendChild(document.createTextNode(ev.text));
        $("gen-btn").disabled = !modelReady;
        break;
      }
    }
  }

  // ── training events ─────────────────────────────────────
  function onTree(ev) {
    nodes = new Map(); nodeOrder = [];
    ev.nodes.forEach((n) => {
      nodes.set(n.id, Object.assign({ status: "idle", acc: null, samples: 0, coverage: 0 }, n));
      nodeOrder.push(n.id);
    });
    totalNodes = ev.nodes.length; doneCount = 0;
    maxGens = ev.config.generations;
    $("st-params").textContent = ev.total_params.toLocaleString();
    $("st-nodes").textContent = `0 / ${totalNodes}`;
    fit = { id: null, kind: "", gens: [], best: [], mean: [] };
    drawFitness(); drawDepthBars(); queueIcicle();
  }

  function onNodeStart(ev) {
    const n = nodes.get(ev.id);
    if (n) { n.status = "active"; n.samples = ev.samples; n.coverage = ev.coverage; }
    fit = { id: ev.id, kind: ev.kind, gens: [], best: [], mean: [] };
    $("fit-node").textContent = ev.kind === "encoder"
      ? (ev.id === "encoder-speed"
        ? `context encoder — phase 2: speed/time constraint (fitness ÷ cost), ${ev.samples.toLocaleString()} samples`
        : `context encoder — nearest-centroid next-byte fitness, ${ev.samples.toLocaleString()} samples`)
      : `${ev.kind} ${ev.id} — ${ev.tokens} token${ev.tokens > 1 ? "s" : ""}, ${ev.samples.toLocaleString()} samples (${ev.coverage.toFixed(1)}%)`;
    drawFitness(); queueIcicle();
  }

  function onNodeGen(ev) {
    if (ev.id !== fit.id) return;
    fit.gens.push(ev.gen); fit.best.push(ev.best); fit.mean.push(ev.mean);
    drawFitness();
  }

  function onNodeDone(ev) {
    const n = nodes.get(ev.id);
    if (n) { n.status = "done"; n.acc = ev.acc; n.samples = ev.samples || 0; n.coverage = ev.coverage || 0; }
    if (typeof ev.progress === "number") doneCount = ev.progress;
    $("st-nodes").textContent = `${doneCount} / ${totalNodes}`;
    $("tlm-status").textContent = `evolving… ${doneCount}/${totalNodes} nodes frozen`;
    drawDepthBars(); queueIcicle();
  }

  function showEval(ev) {
    const pct = (v) => (v * 100).toFixed(1) + "%";
    $("st-acc").textContent = pct(ev.accuracy);
    $("st-bigram").textContent = pct(ev.bigram_accuracy);
    const d = ev.accuracy - ev.bigram_accuracy;
    const el = $("st-delta");
    el.textContent = (d >= 0 ? "+" : "") + (d * 100).toFixed(1) + " pts";
    el.className = "tlm-tile-value " + (d >= 0 ? "tlm-up" : "tlm-down");
    $("st-tps").textContent = Math.round(ev.tokens_per_sec).toLocaleString();
    if (ev.total_params) $("st-params").textContent = ev.total_params.toLocaleString();
  }

  // ── tooltip ──────────────────────────────────────────────
  const tip = $("tlm-tooltip");
  function tipShow(html, x, y) {
    tip.innerHTML = html;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(x + 14, innerWidth - r.width - 10) + "px";
    tip.style.top = Math.min(y + 14, innerHeight - r.height - 10) + "px";
  }
  function tipHide() { tip.hidden = true; }

  // ── icicle (routing tree map) ────────────────────────────
  function queueIcicle() {
    if (icicleQueued) return;
    icicleQueued = true;
    requestAnimationFrame(() => { icicleQueued = false; drawIcicle(); });
  }

  function drawIcicle() {
    const svg = $("icicle");
    svg.textContent = "";
    if (!nodeOrder.length) return;
    const W = svg.clientWidth || 900, ROW = 36, PAD = 18;
    const maxDepth = Math.max(...nodeOrder.map((id) => nodes.get(id).depth));
    const H = (maxDepth + 1) * ROW + PAD;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.style.height = H + "px";
    const x = (t) => (t / 256) * (W - 2) + 1;

    nodeOrder.forEach((id) => {
      const n = nodes.get(id);
      const rx = x(n.t0) + 1, rw = Math.max(x(n.t1) - x(n.t0) - 2, 1); // 2px surface gap
      const ry = n.depth * ROW, rh = ROW - 2;
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", rx); rect.setAttribute("y", ry);
      rect.setAttribute("width", rw); rect.setAttribute("height", rh);
      rect.setAttribute("rx", 3);
      if (n.status === "done" && n.acc !== null) rect.setAttribute("fill", rampColor(n.acc));
      else if (n.status === "active") { rect.setAttribute("fill", css("--tlm-active")); rect.classList.add("tlm-pulse"); }
      else rect.setAttribute("fill", css("--tlm-idle"));
      rect.classList.add("tlm-node");
      rect.addEventListener("mousemove", (e) => tipShow(nodeTip(n), e.clientX, e.clientY));
      rect.addEventListener("mouseleave", tipHide);
      svg.appendChild(rect);

      if (rw > 46 && n.status === "done" && n.acc !== null) {
        const t = document.createElementNS(NS, "text");
        t.setAttribute("x", rx + rw / 2); t.setAttribute("y", ry + rh / 2 + 4);
        t.setAttribute("text-anchor", "middle");
        t.setAttribute("fill", inkFor(n.acc));
        t.setAttribute("class", "tlm-node-label");
        t.textContent = (n.acc * 100).toFixed(0) + "%";
        svg.appendChild(t);
      }
    });

    // byte-range axis under the last row
    const ax = document.createElementNS(NS, "text");
    ax.setAttribute("x", 1); ax.setAttribute("y", H - 4);
    ax.setAttribute("class", "tlm-axis-label");
    ax.textContent = "byte 0";
    svg.appendChild(ax);
    const ax2 = document.createElementNS(NS, "text");
    ax2.setAttribute("x", W - 1); ax2.setAttribute("y", H - 4);
    ax2.setAttribute("text-anchor", "end");
    ax2.setAttribute("class", "tlm-axis-label");
    ax2.textContent = "byte 255";
    svg.appendChild(ax2);
  }

  function nodeTip(n) {
    const kind = n.leaf ? "leaf" : "router";
    const acc = n.acc === null ? "—" : (n.acc * 100).toFixed(1) + "%";
    const accName = n.leaf ? "prediction acc" : "routing acc";
    const span = n.sample ? n.sample : `bytes ${n.t0}–${n.t1 - 1}`;
    return `<b>${kind} ${n.id}</b><br>${span} (${n.tokens} token${n.tokens > 1 ? "s" : ""})<br>` +
      `${n.samples ? n.samples.toLocaleString() + " samples (" + n.coverage.toFixed(1) + "%)<br>" : ""}` +
      `${accName}: ${acc} · ${n.status}`;
  }

  // ── fitness line chart ───────────────────────────────────
  function drawFitness() {
    const svg = $("fitness");
    svg.textContent = "";
    const W = svg.clientWidth || 460, H = 220;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.style.height = H + "px";
    const m = { l: 38, r: 10, t: 10, b: 24 };
    // encoder stage may run more/fewer generations than tree nodes
    const xMax = Math.max(maxGens - 1, fit.gens[fit.gens.length - 1] || 0, 1);
    const X = (g) => m.l + (g / xMax) * (W - m.l - m.r);
    const Y = (v) => m.t + (1 - v) * (H - m.t - m.b);

    frame(svg, W, H, m, X, Y, xMax, "generation");
    if (!fit.gens.length) return;

    const line = (vals, color) => {
      const p = document.createElementNS(NS, "polyline");
      p.setAttribute("points", fit.gens.map((g, i) => `${X(g)},${Y(vals[i])}`).join(" "));
      p.setAttribute("fill", "none"); p.setAttribute("stroke", color);
      p.setAttribute("stroke-width", 2); p.setAttribute("stroke-linejoin", "round");
      svg.appendChild(p);
    };
    line(fit.mean, S2());
    line(fit.best, S1());

    // hover crosshair + tooltip
    const hover = document.createElementNS(NS, "rect");
    hover.setAttribute("x", m.l); hover.setAttribute("y", m.t);
    hover.setAttribute("width", W - m.l - m.r); hover.setAttribute("height", H - m.t - m.b);
    hover.setAttribute("fill", "transparent");
    const cross = document.createElementNS(NS, "line");
    cross.setAttribute("y1", m.t); cross.setAttribute("y2", H - m.b);
    cross.setAttribute("class", "tlm-crosshair"); cross.setAttribute("visibility", "hidden");
    svg.appendChild(cross); svg.appendChild(hover);
    hover.addEventListener("mousemove", (e) => {
      const box = svg.getBoundingClientRect();
      const gx = ((e.clientX - box.left) - m.l) / (W - m.l - m.r) * xMax;
      let i = 0;
      for (let k = 1; k < fit.gens.length; k++)
        if (Math.abs(fit.gens[k] - gx) < Math.abs(fit.gens[i] - gx)) i = k;
      cross.setAttribute("x1", X(fit.gens[i])); cross.setAttribute("x2", X(fit.gens[i]));
      cross.setAttribute("visibility", "visible");
      tipShow(`gen ${fit.gens[i]}<br>` +
        `<i class="tlm-dotchip" style="background:${S1()}"></i>best ${(fit.best[i] * 100).toFixed(1)}%<br>` +
        `<i class="tlm-dotchip" style="background:${S2()}"></i>mean ${(fit.mean[i] * 100).toFixed(1)}%`,
        e.clientX, e.clientY);
    });
    hover.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tipHide(); });
  }

  // ── depth accuracy bars ──────────────────────────────────
  function depthAgg() {
    const agg = new Map();
    nodes.forEach((n) => {
      if (n.status !== "done" || n.acc === null) return;
      if (!agg.has(n.depth)) agg.set(n.depth, { r: [], l: [] });
      agg.get(n.depth)[n.leaf ? "l" : "r"].push(n.acc);
    });
    return [...agg.entries()].sort((a, b) => a[0] - b[0]).map(([d, v]) => ({
      depth: d,
      router: v.r.length ? v.r.reduce((s, x) => s + x, 0) / v.r.length : null,
      leaf: v.l.length ? v.l.reduce((s, x) => s + x, 0) / v.l.length : null,
      nr: v.r.length, nl: v.l.length,
    }));
  }

  function drawDepthBars() {
    const svg = $("depthbars");
    svg.textContent = "";
    const rows = depthAgg();
    const W = svg.clientWidth || 460, H = 220;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.style.height = H + "px";
    const m = { l: 38, r: 10, t: 10, b: 24 };
    const Y = (v) => m.t + (1 - v) * (H - m.t - m.b);
    frame(svg, W, H, m, null, Y, null, "depth");
    if (!rows.length) return;

    const slot = (W - m.l - m.r) / rows.length;
    const bw = Math.min(26, slot / 2 - 4);
    rows.forEach((r, i) => {
      const cx = m.l + slot * (i + 0.5);
      const bars = [["router", S1(), -bw - 1], ["leaf", S2(), 1]];
      bars.forEach(([key, color, off]) => {
        if (r[key] === null) return;
        const h = Math.max((H - m.t - m.b) * r[key], 1);
        const rect = document.createElementNS(NS, "rect");
        rect.setAttribute("x", cx + off); rect.setAttribute("y", Y(r[key]));
        rect.setAttribute("width", bw); rect.setAttribute("height", h);
        rect.setAttribute("rx", 3); rect.setAttribute("fill", color);
        rect.addEventListener("mousemove", (e) => tipShow(
          `<b>depth ${r.depth}</b><br><i class="tlm-dotchip" style="background:${color}"></i>` +
          `${key}s: ${(r[key] * 100).toFixed(1)}% mean (${key === "router" ? r.nr : r.nl} nodes)`,
          e.clientX, e.clientY));
        rect.addEventListener("mouseleave", tipHide);
        svg.appendChild(rect);
      });
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", cx); t.setAttribute("y", H - m.b + 15);
      t.setAttribute("text-anchor", "middle");
      t.setAttribute("class", "tlm-axis-label");
      t.textContent = r.depth;
      svg.appendChild(t);
    });

    // table view (accessibility relief)
    const tbl = ["<table><tr><th>depth</th><th>routers mean</th><th>leaves mean</th></tr>"];
    rows.forEach((r) => tbl.push(
      `<tr><td>${r.depth}</td><td>${r.router === null ? "—" : (r.router * 100).toFixed(1) + "%"}</td>` +
      `<td>${r.leaf === null ? "—" : (r.leaf * 100).toFixed(1) + "%"}</td></tr>`));
    tbl.push("</table>");
    $("depth-table").innerHTML = tbl.join("");
  }

  // shared chart frame: hairline grid, y ticks 0–100%, baseline
  function frame(svg, W, H, m, X, Y, xMax, xLabel) {
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      const g = document.createElementNS(NS, "line");
      g.setAttribute("x1", m.l); g.setAttribute("x2", W - m.r);
      g.setAttribute("y1", Y(v)); g.setAttribute("y2", Y(v));
      g.setAttribute("class", v === 0 ? "tlm-baseline" : "tlm-grid");
      svg.appendChild(g);
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", m.l - 6); t.setAttribute("y", Y(v) + 4);
      t.setAttribute("text-anchor", "end");
      t.setAttribute("class", "tlm-axis-label");
      t.textContent = (v * 100) + "%";
      svg.appendChild(t);
    });
    if (X && xMax) {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", W - m.r); t.setAttribute("y", H - 6);
      t.setAttribute("text-anchor", "end");
      t.setAttribute("class", "tlm-axis-label");
      t.textContent = `${xLabel} 0–${xMax}`;
      svg.appendChild(t);
    }
  }

  // ── controls ─────────────────────────────────────────────
  // (routing-trace rendering lives in trace_view.js, shared with /runs)
  function setButtons() {
    $("tlm-start").disabled = running || !ws || ws.readyState !== 1;
    $("sweep-btn").disabled = running || !ws || ws.readyState !== 1;
    $("tlm-stop").disabled = !running;
    $("gen-btn").disabled = !modelReady || running;
    $("tr-btn").disabled = !modelReady || running;
    $("enc-train").disabled = running || !ws || ws.readyState !== 1;
    $("enc-stop").disabled = !running;
  }

  // ── encoder trainer modal ────────────────────────────────
  $("enc-modal-open").addEventListener("click", () => { $("enc-modal").hidden = false; });
  $("enc-modal-close").addEventListener("click", () => { $("enc-modal").hidden = true; });
  $("enc-modal").addEventListener("click", (e) => { if (e.target === $("enc-modal")) $("enc-modal").hidden = true; });

  $("enc-train").addEventListener("click", () => {
    running = true; setButtons();
    $("enc-status").textContent = "starting…";
    const cfg = gatherConfig();
    delete cfg.encoder_id;                 // always trains a fresh encoder
    send(Object.assign({ op: "train_encoder", notes: "standalone encoder" }, cfg));
  });
  $("enc-stop").addEventListener("click", () => send({ op: "stop" }));

  function renderEncoders(list) {
    const sel = $("cf-encoder-id");
    const keep = sel.value;
    sel.innerHTML = '<option value="">evolve fresh each run</option>';
    list.forEach((e) => {
      const o = document.createElement("option");
      o.value = e.id;
      o.textContent = `${e.id} · dim ${e.context_dim} · acc ${e.nc_accuracy != null ? (e.nc_accuracy * 100).toFixed(1) + "%" : "?"}`;
      sel.appendChild(o);
    });
    if ([...sel.options].some((o) => o.value === keep)) sel.value = keep;

    const box = $("enc-list");
    if (!list.length) { box.textContent = "none saved yet"; return; }
    box.innerHTML = "";
    const tbl = document.createElement("table");
    tbl.className = "tlm-sweep-table";
    tbl.innerHTML = "<tr><th>encoder</th><th>NC acc</th><th>dim</th><th>window</th><th>embed</th><th>flags</th><th></th></tr>";
    list.forEach((e) => {
      const tr = document.createElement("tr");
      const flags = [e.time_constrained ? "time" : null, e.diversity ? "div" : null,
                     e.evolve_dims ? "dims" : null].filter(Boolean).join("+") || "—";
      tr.innerHTML =
        `<td></td><td>${e.nc_accuracy != null ? (e.nc_accuracy * 100).toFixed(1) + "%" : "?"}</td>` +
        `<td>${e.context_dim ?? "?"}</td><td>${e.context_window ?? "?"}</td>` +
        `<td>${e.embed_dim ?? "?"}</td><td>${flags}</td><td></td>`;
      tr.children[0].textContent = e.id;
      const use = document.createElement("button");
      use.className = "runs-btn";
      use.textContent = sel.value === e.id ? "selected" : "use";
      use.addEventListener("click", () => {
        sel.value = e.id;
        $("enc-modal").hidden = true;
        $("tlm-status").textContent = `encoder ${e.id} selected — Start Training will reuse it`;
      });
      tr.children[6].appendChild(use);
      tbl.appendChild(tr);
    });
    box.appendChild(tbl);
  }

  function gatherConfig() {
    const v = (id) => parseFloat($(id).value);
    return {
      token_mode: $("cf-tokenmode") ? $("cf-tokenmode").value : "byte",
      vocab_size: $("cf-vocab") ? v("cf-vocab") || 2048 : 2048,
      branching_factor: v("cf-branch"), context_window: v("cf-ctxwin"),
      cluster_tokens: $("cf-cluster").checked,
      device: $("cf-device").value,
      ridge_seed: $("cf-ridge").checked,
      encoder_fitness: $("cf-encfit").value,
      sa_mutation: $("cf-sa").checked,
      node_resample: $("cf-resample").checked,
      encoder_split_rotate: $("cf-encrotate").checked,
      encoder_depth: parseInt($("cf-encdepth").value, 10) || 1,
      encoder_hidden: v("cf-enchidden") || 0,
      embed_dim: v("cf-embed"), context_dim: v("cf-ctxdim"),
      routing_layers: v("cf-layers"), max_samples: v("cf-samples"),
      encoder_generations: v("cf-encgens"),
      encoder_pop_size: v("cf-encpop") || 0,
      encoder_samples: v("cf-encsamples") || 2000,
      encoder_id: $("cf-encoder-id").value || undefined,
      encoder_speed_generations: $("cf-encspeed-on").checked ? v("cf-encspeed") : 0,
      encoder_time_budget: v("cf-encbudget"),
      encoder_time_constrained: $("cf-enctime").checked,
      encoder_evolve_dims: $("cf-encdims").checked,
      encoder_diversity: $("cf-encdiv").checked,
      encoder_diversity_budget: v("cf-encdivbudget"),
      encoder_novelty: $("cf-encnov").checked,
      encoder_novelty_strength: v("cf-encnovstr"),
      encoder_seed_embeddings: $("cf-encseed").checked,
      pop_size: v("cf-pop"), generations: v("cf-gens"),
      mutation_rate: v("cf-mut"), elite_frac: v("cf-elite"),
    };
  }

  // token-mode selector: show the vocab-size field only in word mode
  (() => {
    const mode = $("cf-tokenmode"), field = $("cf-vocab-field");
    if (!mode || !field) return;
    const sync = () => { field.hidden = mode.value !== "word"; };
    mode.addEventListener("change", sync);
    sync();
  })();

  $("tlm-start").addEventListener("click", () => {
    running = true; setButtons();
    $("tlm-status").textContent = "starting…";
    $("gen-out").textContent = "Training… generation unlocks when the tree is frozen.";
    send(Object.assign({ op: "start" }, gatherConfig()));
  });

  // ── config sweep ─────────────────────────────────────────
  const SWEEPABLE = [
    { key: "routing_layers", label: "routing layers", suggest: "0,1,2" },
    { key: "branching_factor", label: "branching", suggest: "4,16" },
    { key: "cluster_tokens", label: "cluster token split (0/1)", suggest: "0,1" },
    { key: "ridge_seed", label: "ridge seed (0/1)", suggest: "0,1" },
    { key: "encoder_split_rotate", label: "rotate ridge split (0/1)", suggest: "0,1" },
    { key: "sa_mutation", label: "self-adaptive mutation (0/1)", suggest: "0,1" },
    { key: "context_dim", label: "context dim", suggest: "64,128,256" },
    { key: "embed_dim", label: "embed dim", suggest: "32,64" },
    { key: "context_window", label: "context window", suggest: "8,16,32" },
    { key: "generations", label: "generations / node", suggest: "40,80" },
    { key: "pop_size", label: "population", suggest: "40,80" },
    { key: "encoder_generations", label: "encoder gens", suggest: "40,80" },
    { key: "encoder_pop_size", label: "encoder population (0 = shared)", suggest: "0,120" },
    { key: "encoder_speed_generations", label: "speed gens", suggest: "0,40" },
    { key: "encoder_time_budget", label: "speed budget", suggest: "0.15,0.5,1" },
    { key: "encoder_time_constrained", label: "time-constrained encoder (0/1)", suggest: "0,1" },
    { key: "encoder_diversity", label: "encoder head diversity (0/1)", suggest: "0,1" },
    { key: "encoder_novelty", label: "encoder novelty bonus (0/1)", suggest: "0,1" },
    { key: "encoder_seed_embeddings", label: "seed embeddings (0/1)", suggest: "0,1" },
    { key: "encoder_depth", label: "encoder depth (1/2)", suggest: "1,2" },
    { key: "encoder_novelty_strength", label: "novelty strength", suggest: "0.25,0.5,1" },
    { key: "encoder_evolve_dims", label: "evolve context dim (0/1)", suggest: "0,1" },
    { key: "encoder_diversity_budget", label: "diversity budget", suggest: "0.15,0.5,1" },
  ];

  function buildSweepParams() {
    const box = $("sweep-params");
    SWEEPABLE.forEach((p) => {
      const row = document.createElement("div");
      row.className = "tlm-sweep-row";
      const lbl = document.createElement("label");
      lbl.className = "check";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.dataset.key = p.key;
      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(" " + p.label));
      const vals = document.createElement("input");
      vals.type = "text";
      vals.value = p.suggest;
      vals.disabled = true;
      vals.className = "tlm-sweep-vals";
      vals.dataset.key = p.key;
      cb.addEventListener("change", () => { vals.disabled = !cb.checked; });
      row.appendChild(lbl);
      row.appendChild(vals);
      box.appendChild(row);
    });
  }
  buildSweepParams();

  $("sweep-btn").addEventListener("click", () => {
    const spec = {};
    document.querySelectorAll("#sweep-params input[type=checkbox]").forEach((cb) => {
      if (!cb.checked) return;
      const vals = document.querySelector(`#sweep-params input.tlm-sweep-vals[data-key="${cb.dataset.key}"]`)
        .value.split(",").map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n));
      if (vals.length) spec[cb.dataset.key] = vals;
    });
    if (!Object.keys(spec).length) {
      $("sweep-status").textContent = "check at least one parameter and give it values";
      return;
    }
    running = true; setButtons();
    sweep = null;
    $("sweep-results").textContent = "";
    $("sweep-status").textContent = "starting sweep…";
    send(Object.assign({ op: "sweep", sweep: spec }, gatherConfig()));
  });

  function renderSweep() {
    const box = $("sweep-results");
    box.textContent = "";
    if (!sweep || !sweep.results.length) return;
    const rows = sweep.done
      ? sweep.results
      : sweep.results.slice().sort((a, b) => b.accuracy - a.accuracy);
    const tbl = document.createElement("table");
    tbl.className = "tlm-sweep-table";
    tbl.innerHTML = "<tr><th>#</th><th>candidate</th><th>accuracy</th><th>vs bigram</th><th>time</th><th>run</th></tr>";
    rows.forEach((r, rank) => {
      const tr = document.createElement("tr");
      if (rank === 0) tr.className = "best";
      const d = r.accuracy - r.bigram;
      tr.innerHTML =
        `<td>${rank + 1}</td><td></td>` +
        `<td>${(r.accuracy * 100).toFixed(1)}%</td>` +
        `<td class="${d >= 0 ? "tlm-up" : "tlm-down"}">${(d * 100).toFixed(1)} pts</td>` +
        `<td>${r.seconds != null ? r.seconds + "s" : "—"}</td><td></td>`;
      tr.children[1].textContent = r.name;
      if (r.run_id) {
        const a = document.createElement("a");
        a.href = `/runs#${r.run_id}`;
        a.target = "_blank";
        a.textContent = "open ↗";
        tr.children[5].appendChild(a);
      }
      tbl.appendChild(tr);
    });
    box.appendChild(tbl);
  }

  $("tlm-stop").addEventListener("click", () => send({ op: "stop" }));

  $("gen-btn").addEventListener("click", () => {
    $("gen-btn").disabled = true;
    $("gen-out").textContent = "generating…";
    send({
      op: "generate",
      prompt: $("gen-prompt").value,
      length: parseInt($("gen-len").value, 10) || 300,
      temperature: parseFloat($("gen-temp").value) || 0,
    });
  });

  $("tr-btn").addEventListener("click", () => {
    $("tr-btn").disabled = true;
    $("tr-detail").textContent = "tracing…";
    send({
      op: "trace",
      prompt: $("tr-prompt").value,
      length: parseInt($("tr-steps").value, 10) || 48,
      temperature: parseFloat($("tr-temp").value) || 0,
    });
  });

  addEventListener("resize", () => { queueIcicle(); drawFitness(); drawDepthBars(); });

  connect();
  setButtons();
})();
