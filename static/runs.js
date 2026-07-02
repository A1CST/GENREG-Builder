// GENREG runs dashboard: per-environment tech tree of runs, run details, and an
// inference viewer that replays a run's saved checkpoint on a board canvas.
(() => {
  const $ = (id) => document.getElementById(id);
  const tabsEl = $("env-tabs"), treeEl = $("tree"), detailEl = $("detail");

  let runs = [];            // all runs
  let byEnv = {};           // env -> [runs]
  let currentEnv = null;
  let currentId = null;
  let replayTimer = null;

  async function load() {
    try {
      runs = await (await fetch("/api/runs")).json();
    } catch (_) { runs = []; }
    byEnv = {};
    for (const r of runs) (byEnv[r.environment] = byEnv[r.environment] || []).push(r);
    for (const e in byEnv) byEnv[e].sort((a, b) => (a.created || "").localeCompare(b.created || ""));
    renderTabs();
    if (!currentEnv || !byEnv[currentEnv]) currentEnv = Object.keys(byEnv)[0] || null;
    renderTree();
  }

  function renderTabs() {
    const envs = Object.keys(byEnv);
    tabsEl.innerHTML = "";
    if (!envs.length) { tabsEl.innerHTML = '<span class="tree-empty">no environments</span>'; return; }
    for (const e of envs) {
      const b = document.createElement("button");
      b.className = "env-tab" + (e === currentEnv ? " active" : "");
      b.textContent = `${e} · ${byEnv[e].length}`;
      b.addEventListener("click", () => { currentEnv = e; renderTabs(); renderTree(); });
      tabsEl.appendChild(b);
    }
  }

  function statusDot(status) {
    const cls = status === "finished" ? "ok" : status === "stopped" ? "warn" : status === "running" ? "run" : "bad";
    return `<span class="st-dot ${cls}" title="${status}"></span>`;
  }

  function renderTree() {
    const list = (currentEnv && byEnv[currentEnv]) || [];
    if (!list.length) { treeEl.innerHTML = '<div class="tree-empty">No runs for this environment.</div>'; return; }
    treeEl.innerHTML = `<div class="tree-root">▣ ${currentEnv}</div>`;
    const branches = document.createElement("div");
    branches.className = "tree-branches";
    treeEl.appendChild(branches);
    for (const r of list) {
      const node = document.createElement("div");
      node.className = "tree-node" + (r.id === currentId ? " selected" : "");
      const cons = (r.constraints || []).length ? ` · ${r.constraints.length}c` : "";
      const best = r.best && r.best.score != null ? `score ${r.best.score}` : "—";
      node.innerHTML =
        `${statusDot(r.status)}<div class="tn-main">` +
        `<div class="tn-title">${r.created ? r.created.replace("T", " ") : r.id}</div>` +
        `<div class="tn-sub">${r.device || "cpu"} · pop ${r.population ?? "?"} · ${r.generations ?? "?"}g${cons} · ${best}` +
        `${r.has_checkpoint ? ' · <span class="ckpt">ckpt</span>' : ""}</div></div>`;
      node.addEventListener("click", () => showDetail(r.id));
      branches.appendChild(node);
    }
  }

  // -- detail --------------------------------------------------------------
  async function showDetail(id) {
    currentId = id;
    renderTree();
    detailEl.innerHTML = '<div class="detail-empty">Loading…</div>';
    let d;
    try { d = await (await fetch(`/api/runs/${id}`)).json(); } catch (_) { d = null; }
    if (!d || d.error) { detailEl.innerHTML = '<div class="detail-empty">Failed to load run.</div>'; return; }
    const cfg = (d.config && d.config.config) || {};
    const isTree = (d.config && d.config.environment) === "tree";
    const started = (d.config && d.config.started) || {};
    const summ = d.summary || {};
    const best = summ.best || {};
    const rows = [
      ["environment", d.config && d.config.environment], ["device", cfg.device || "cpu"],
      ["population", cfg.population], ["generations", cfg.generations],
      ["hidden (H)", cfg.params && cfg.params.hidden], ["evolve H", cfg.evolve_hidden ? "yes" : "no"],
      ["crossover", cfg.sexual === false ? "off (mutation only)" : "on"],
      ["elite", cfg.elite], ["breed top %", cfg.parent_frac != null ? Math.round(cfg.parent_frac * 100) : null],
      ["constraints", (cfg.constraints || []).join(", ") || "none"],
      ["snake board", cfg.snake ? `${cfg.snake.w}×${cfg.snake.h}` : null],
      ["status", summ.status || (d.config && d.config.status)],
      ["created", d.config && d.config.created], ["finished", summ.finished],
      ["notes", started.notes],
    ].filter(([, v]) => v != null && v !== "");

    detailEl.innerHTML = `
      <div class="detail-head">
        <div><div class="dh-title">${d.id}</div><div class="dh-sub">${(d.config && d.config.environment) || ""}</div></div>
        <div class="dh-badges">${statusDot(summ.status || "running")}${summ.checkpoint ? '<span class="ckpt">checkpoint</span>' : '<span class="nockpt">no checkpoint</span>'}<button id="export-btn" class="runs-btn" title="Download every detail of this run as a JSON file">⤓ Export JSON</button></div>
      </div>
      <div class="detail-grid">
        <section class="d-card">
          <h3>Configuration</h3>
          <table class="cfg">${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>
        </section>
        <section class="d-card">
          <h3>Result</h3>
          <div class="result">
            <div><span>best score</span><b>${best.score ?? "—"}</b></div>
            <div><span>base fitness</span><b>${best.base ?? "—"}</b></div>
            <div><span>H · leak · bits</span><b>${best.H ?? "—"} · ${best.leak ?? "—"} · ${best.bits ?? "—"}</b></div>
            <div><span>generations</span><b>${d.history ? d.history.length : 0}</b></div>
          </div>
          <canvas id="d-spark" class="d-spark" width="360" height="70"></canvas>
          <div class="spark-legend"><span class="dot best"></span>best <span class="dot mean"></span>mean fitness / gen</div>
        </section>
        <section class="d-card wide">
          <h3>Inference / verify</h3>
          <div class="infer-bar">
            <button id="infer-btn" class="runs-btn" ${summ.checkpoint ? "" : "disabled"}>${isTree ? "▶ Generate sample" : "▶ Play checkpoint"}</button>
            <span class="infer-status" id="infer-status">${summ.checkpoint ? (isTree ? "generates a text sample from the saved tree model" : "runs the saved champion on a fresh game") : "no checkpoint saved for this run"}</span>
          </div>
          <div class="infer-stage" id="infer-stage"><canvas id="infer-canvas"></canvas></div>
        </section>
        ${summ.encoder ? `
        <section class="d-card wide">
          <h3>Encoder evolution</h3>
          <div id="enc-detail"></div>
        </section>` : ""}
        ${summ.sweep_results ? `
        <section class="d-card wide">
          <h3>Sweep results (ranked)</h3>
          <div id="sweep-detail"></div>
        </section>` : ""}
        ${isTree ? `
        <section class="d-card wide">
          <h3>Saved traces</h3>
          <div id="trace-list" class="trace-list">loading…</div>
          <div id="trace-strip" class="tlm-strip" hidden></div>
          <div id="trace-detail" class="tlm-tr-detail"></div>
        </section>` : ""}
      </div>`;

    drawSparkline($("d-spark"), d.history || []);
    const btn = $("infer-btn");
    if (btn) btn.addEventListener("click", () => playInference(id));
    $("export-btn").addEventListener("click", () => exportRun(id, d, isTree));
    if (summ.encoder) renderEncoderDetail(summ.encoder);
    if (summ.sweep_results) renderSweepDetail(summ.sweep_results);
    if (isTree) loadTraces(id);
  }

  // -- encoder evolution card (tree runs: summary.encoder) -------------------
  function renderEncoderDetail(enc) {
    const box = $("enc-detail");
    if (!box) return;
    const pct = (v) => v != null ? (v * 100).toFixed(0) + "%" : null;
    const rows = [
      ["NC accuracy (fitness proxy)", enc.nc_accuracy],
      ["fitness samples", enc.samples],
      ["generations", enc.generations],
      ["population", enc.pop_size],
      ["context dim", enc.evolve_dims
        ? `${enc.start_context_dim} → ${enc.context_dim} (evolved)`
        : enc.context_dim],
      ["time constraint", enc.time_constrained ? `on · budget ${enc.time_budget}` : "off"],
      ["active weights", pct(enc.active_fraction)],
      ["head diversity", enc.diversity ? `on · budget ${enc.diversity_budget}` : "off"],
      ["head redundancy", enc.redundancy],
      ["speed phase", enc.speed_generations
        ? `${enc.speed_generations} gens → acc ${enc.speed_nc_accuracy}, active ${pct(enc.speed_active_fraction)}`
        : "off"],
    ].filter(([, v]) => v != null && v !== "");
    const curves = enc.curves || {};
    const keys = Object.keys(curves).filter((k) => (curves[k] || []).length > 1);
    box.innerHTML =
      `<table class="cfg">${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>` +
      keys.map((k, i) =>
        `<div class="spark-legend" style="margin-top:10px"><span class="dot best"></span>best ` +
        `<span class="dot mean"></span>mean — ${k} fitness per generation` +
        `${(enc.time_constrained || enc.diversity) && k === "encoder" ? " (constrained)" : ""}</div>` +
        `<canvas id="enc-spark-${i}" class="d-spark" width="360" height="70"></canvas>`).join("");
    keys.forEach((k, i) => {
      drawSparkline($(`enc-spark-${i}`),
        curves[k].map((p) => ({ fitness: { best: p.best, mean: p.mean } })));
    });
  }

  // -- export: download every detail of a run as one JSON file ---------------
  async function exportRun(id, d, isTree) {
    const btn = $("export-btn");
    if (btn) { btn.disabled = true; btn.textContent = "exporting…"; }
    const bundle = {
      exported_at: new Date().toISOString(),
      id: d.id,
      config: d.config || null,          // launch config + metadata + status
      summary: d.summary || null,        // final result / eval block
      history: d.history || [],          // per-generation (or per-node) metrics
    };
    if (isTree) {
      try {
        bundle.traces = await (await fetch(`/api/runs/${id}/traces`)).json();
      } catch (_) { bundle.traces = []; }
    }
    const blob = new Blob([JSON.stringify(bundle, null, 2)],
                          { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${d.id}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    if (btn) { btn.disabled = false; btn.textContent = "⤓ Export JSON"; }
  }

  // -- sweep detail (ranked candidate table; rows jump to the candidate run) --
  function renderSweepDetail(results) {
    const box = $("sweep-detail");
    if (!box) return;
    box.textContent = "";
    if (!results.length) { box.textContent = "no candidates completed"; return; }
    const tbl = document.createElement("table");
    tbl.className = "tlm-sweep-table";
    tbl.innerHTML = "<tr><th>#</th><th>candidate</th><th>accuracy</th><th>vs bigram</th><th>time</th><th>run</th></tr>";
    results.forEach((r, rank) => {
      const tr = document.createElement("tr");
      if (rank === 0) tr.className = "best";
      const dlt = r.accuracy - r.bigram;
      tr.innerHTML =
        `<td>${rank + 1}</td><td></td>` +
        `<td>${(r.accuracy * 100).toFixed(1)}%</td>` +
        `<td>${(dlt * 100).toFixed(1)} pts</td>` +
        `<td>${r.seconds != null ? r.seconds + "s" : "—"}</td><td></td>`;
      tr.children[1].textContent = r.name;
      if (r.run_id) {
        const a = document.createElement("a");
        a.href = `#${r.run_id}`;
        a.textContent = "view run";
        a.addEventListener("click", (e) => { e.preventDefault(); showDetail(r.run_id); });
        tr.children[5].appendChild(a);
      }
      tbl.appendChild(tr);
    });
    box.appendChild(tbl);
  }

  // -- saved routing traces (tree runs) --------------------------------------
  async function loadTraces(id) {
    const list = $("trace-list");
    let traces;
    try { traces = await (await fetch(`/api/runs/${id}/traces`)).json(); } catch (_) { traces = []; }
    if (!Array.isArray(traces) || !traces.length) {
      list.textContent = "no traces saved for this run — use Trace on the Tree LM page while this model is loaded";
      return;
    }
    list.textContent = "";
    traces.forEach((t) => {
      const row = document.createElement("div");
      row.className = "trace-row";
      const time = document.createElement("span");
      time.className = "tr-time";
      time.textContent = (t.created || "").replace("T", " ");
      const meta = document.createElement("span");
      meta.className = "tr-meta";
      meta.textContent = `${(t.steps || []).length} steps · temp ${t.temperature}`;
      const prompt = document.createElement("span");
      prompt.className = "tr-prompt";
      prompt.textContent = JSON.stringify(t.prompt);
      const text = document.createElement("span");
      text.className = "tr-text";
      text.textContent = (t.text || "").slice(0, 48);
      row.append(time, meta, prompt, text);
      row.addEventListener("click", () => {
        [...list.children].forEach((c) => c.classList.remove("sel"));
        row.classList.add("sel");
        TraceView.mount($("trace-strip"), $("trace-detail"), t);
      });
      list.appendChild(row);
    });
  }

  function drawSparkline(cv, history) {
    if (!cv) return;
    const g = cv.getContext("2d");
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 360, h = cv.clientHeight || 70;
    cv.width = w * dpr; cv.height = h * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const best = history.map((r) => r.fitness && r.fitness.best).filter((v) => typeof v === "number");
    const mean = history.map((r) => r.fitness && r.fitness.mean).filter((v) => typeof v === "number");
    if (best.length < 2) return;
    const all = best.concat(mean);
    let lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (hi - lo < 1e-6) hi = lo + 1;
    const pad = 5, X = (i, n) => pad + (w - 2 * pad) * (n > 1 ? i / (n - 1) : 0);
    const Y = (v) => (h - pad) - (h - 2 * pad) * (v - lo) / (hi - lo);
    const line = (arr, color) => { g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath(); arr.forEach((v, i) => { const x = X(i, arr.length), y = Y(v); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); };
    line(mean, "#7d8794"); line(best, "#4ea1ff");
  }

  // -- inference viewer ----------------------------------------------------
  function stopReplay() { if (replayTimer) { clearInterval(replayTimer); replayTimer = null; } }

  async function playInference(id) {
    const status = $("infer-status");
    status.textContent = "loading checkpoint…";
    stopReplay();
    let rep;
    try { rep = await (await fetch(`/api/runs/${id}/replay`)).json(); } catch (_) { rep = null; }
    if (rep && rep.env === "tree" && typeof rep.text === "string") {
      status.textContent = `generated ${rep.text.length} bytes · temp ${rep.temperature}`;
      const stage = $("infer-stage");
      stage.innerHTML = "";
      const pre = document.createElement("pre");
      pre.className = "tlm-gen-out";
      const b = document.createElement("b");
      b.textContent = rep.prompt;
      pre.appendChild(b);
      pre.appendChild(document.createTextNode(rep.text));
      stage.appendChild(pre);
      return;
    }
    if (!rep || rep.error || !rep.frames || !rep.frames.length) { status.textContent = rep && rep.error ? rep.error : "no frames"; return; }
    status.textContent = `playing… final base ${rep.base}`;
    const cv = $("infer-canvas");
    let i = 0;
    const draw = () => drawFrame(cv, rep.frames[Math.min(i, rep.frames.length - 1)]);
    draw();
    replayTimer = setInterval(() => {
      i++;
      if (i >= rep.frames.length) { stopReplay(); status.textContent = `done · final base ${rep.base} · score ${(rep.stats || {}).score ?? "—"}`; return; }
      draw();
    }, 1000 / 15);
  }

  // compact board renderers (snake + 2048) for the inference stage
  const TILE = { 0: "rgba(255,255,255,0.05)", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e" };
  function fit(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 300, h = cv.clientHeight || 300;
    cv.width = w * dpr; cv.height = h * dpr; const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h); g.fillStyle = "#0b0f16"; g.fillRect(0, 0, w, h); return { g, w, h };
  }
  function cellRect(w, h, cols, rows, pad) { const c = Math.max(3, Math.floor(Math.min((w - 2 * pad) / cols, (h - 2 * pad) / rows))); return { c, x: (w - c * cols) / 2, y: (h - c * rows) / 2 }; }
  function drawFrame(cv, f) {
    if (!cv || !f) return;
    const { g, w, h } = fit(cv);
    if (f.env === "2048") {
      const r = cellRect(w, h, 4, 4, 18), gap = Math.max(4, Math.round(r.c * 0.12));
      for (let ry = 0; ry < 4; ry++) for (let cx = 0; cx < 4; cx++) {
        const v = f.grid[ry][cx], x = r.x + cx * r.c + gap / 2, y = r.y + ry * r.c + gap / 2, s = r.c - gap;
        g.fillStyle = TILE[v] || TILE[2048]; g.fillRect(x, y, s, s);
        if (v) { g.fillStyle = v <= 4 ? "#776e65" : "#f9f6f2"; g.font = `700 ${Math.round(s * 0.36)}px monospace`; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(String(v), x + s / 2, y + s / 2); }
      }
    } else {
      const r = cellRect(w, h, f.w, f.h, 14);
      g.strokeStyle = "rgba(78,161,255,0.10)"; g.lineWidth = 1;
      for (let c = 0; c <= f.w; c++) { g.beginPath(); g.moveTo(r.x + c * r.c, r.y); g.lineTo(r.x + c * r.c, r.y + f.h * r.c); g.stroke(); }
      for (let rr = 0; rr <= f.h; rr++) { g.beginPath(); g.moveTo(r.x, r.y + rr * r.c); g.lineTo(r.x + f.w * r.c, r.y + rr * r.c); g.stroke(); }
      if (f.food) { g.fillStyle = "#f85149"; g.fillRect(r.x + f.food[0] * r.c + 1, r.y + f.food[1] * r.c + 1, r.c - 2, r.c - 2); }
      (f.snake || []).forEach((cell, idx) => { g.fillStyle = idx === 0 ? "#56d364" : "#3fb950"; g.fillRect(r.x + cell[0] * r.c + 1, r.y + cell[1] * r.c + 1, r.c - 2, r.c - 2); });
    }
  }

  $("runs-refresh").addEventListener("click", load);
  load().then(() => {
    // deep-link: /runs#<run-id> opens that run (used by sweep tables)
    const rid = location.hash.slice(1);
    if (!rid) return;
    const run = runs.find((r) => r.id === rid);
    if (run) { currentEnv = run.environment; renderTabs(); renderTree(); }
    showDetail(rid);
  });
})();
