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

  // -- filter / sort / group toolbar state -----------------------------------
  const searchEl = $("run-search"), statusEl = $("run-status"),
        sortEl = $("run-sort"), favBtn = $("run-fav");
  let favOnly = false;
  const COLLAPSE_KEY = "genreg_runs_collapse";      // "<env>:<group>" -> bool
  let collapsedMap = {};
  try { collapsedMap = JSON.parse(localStorage.getItem(COLLAPSE_KEY)) || {}; } catch (_) {}
  const saveCollapsed = () => {
    try { localStorage.setItem(COLLAPSE_KEY, JSON.stringify(collapsedMap)); } catch (_) {}
  };

  async function load() {
    try {
      runs = await (await fetch("/api/runs")).json();
    } catch (_) { runs = []; }
    byEnv = {};
    for (const r of runs) (byEnv[r.environment] = byEnv[r.environment] || []).push(r);
    renderTabs();
    if (!currentEnv || !byEnv[currentEnv]) currentEnv = Object.keys(byEnv)[0] || null;
    renderTree();
  }

  // -- run metadata (label / favorite / group / tags) ------------------------
  async function saveMeta(id, patch) {
    let meta = null;
    try {
      const res = await fetch(`/api/runs/${id}/meta`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      meta = await res.json();
      if (!res.ok) meta = null;
    } catch (_) { meta = null; }
    if (meta && !meta.error) {
      const r = runs.find((x) => x.id === id);
      if (r) Object.assign(r, { label: meta.label || "", favorite: !!meta.favorite,
                                group: meta.group || "", tags: meta.tags || [] });
      renderTree();
      syncDetailStar(id);
    }
    return meta;
  }

  function syncDetailStar(id) {
    const btn = $("dh-fav");
    if (!btn || currentId !== id) return;
    const r = runs.find((x) => x.id === id);
    const on = !!(r && r.favorite);
    btn.classList.toggle("on", on);
    btn.textContent = on ? "★" : "☆";
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

  // searchable text for one run (filter box matches any of it)
  function hay(r) {
    return [r.id, r.label, r.group, (r.tags || []).join(" "), r.created,
            r.device, (r.constraints || []).join(" "), r.status]
      .join(" ").toLowerCase();
  }

  function sortRuns(list) {
    const mode = sortEl ? sortEl.value : "new";
    const score = (r) => (r.best && typeof r.best.score === "number") ? r.best.score : -Infinity;
    const arr = [...list];
    if (mode === "old") arr.sort((a, b) => (a.created || "").localeCompare(b.created || ""));
    else if (mode === "score") arr.sort((a, b) => score(b) - score(a));
    else arr.sort((a, b) => (b.created || "").localeCompare(a.created || ""));
    // favorites pinned first (stable within each half)
    return arr.filter((r) => r.favorite).concat(arr.filter((r) => !r.favorite));
  }

  function renderTree() {
    const all = (currentEnv && byEnv[currentEnv]) || [];
    treeEl.innerHTML = "";
    if (!all.length) { treeEl.innerHTML = '<div class="tree-empty">No runs for this environment.</div>'; return; }

    const q = searchEl ? searchEl.value.trim().toLowerCase() : "";
    let list = all;
    if (q) list = list.filter((r) => hay(r).includes(q));
    if (statusEl && statusEl.value) list = list.filter((r) => (r.status || "") === statusEl.value);
    if (favOnly) list = list.filter((r) => r.favorite);

    const root = document.createElement("div");
    root.className = "tree-root";
    root.textContent = `▣ ${currentEnv}` +
      (list.length !== all.length ? ` · ${list.length}/${all.length} shown` : "");
    treeEl.appendChild(root);
    if (!list.length) {
      const d = document.createElement("div");
      d.className = "tree-empty";
      d.textContent = "No runs match the filter.";
      treeEl.appendChild(d);
      return;
    }

    // bucket by group: ungrouped first, then named groups alphabetically
    const groups = new Map();
    for (const r of list) {
      const g = (r.group || "").trim();
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(r);
    }
    const names = [...groups.keys()].sort((a, b) =>
      a === "" ? -1 : b === "" ? 1 : a.localeCompare(b));
    const flat = names.length === 1 && names[0] === "";   // no groups anywhere

    for (const g of names) {
      const runsIn = sortRuns(groups.get(g));
      if (!flat) {
        const key = `${currentEnv}:${g}`;
        // named groups start collapsed (that's their point); ungrouped starts open
        const isClosed = key in collapsedMap ? !!collapsedMap[key] : g !== "";
        const head = document.createElement("div");
        head.className = "group-head" + (isClosed ? " closed" : "");
        head.innerHTML = `<span class="gh-arrow">${isClosed ? "▸" : "▾"}</span>` +
                         `<span class="gh-name"></span><span class="gh-count">${runsIn.length}</span>`;
        head.querySelector(".gh-name").textContent = g || "ungrouped";
        head.addEventListener("click", () => {
          collapsedMap[key] = !isClosed;
          saveCollapsed();
          renderTree();
        });
        treeEl.appendChild(head);
        if (isClosed) continue;
      }
      const branches = document.createElement("div");
      branches.className = "tree-branches";
      treeEl.appendChild(branches);
      for (const r of runsIn) branches.appendChild(runNode(r));
    }
  }

  function runNode(r) {
    const node = document.createElement("div");
    node.className = "tree-node" + (r.id === currentId ? " selected" : "");
    const cons = (r.constraints || []).length ? ` · ${r.constraints.length}c` : "";
    const best = r.best && r.best.score != null ? `score ${r.best.score}` : "—";
    node.innerHTML =
      `${statusDot(r.status)}<div class="tn-main">` +
      `<div class="tn-title"></div>` +
      `<div class="tn-sub">${r.device || "cpu"} · pop ${r.population ?? "?"} · ${r.generations ?? "?"}g${cons} · ${best}` +
      `${r.has_checkpoint ? ' · <span class="ckpt">ckpt</span>' : ""}</div></div>` +
      `<button class="tn-star${r.favorite ? " on" : ""}" title="${r.favorite ? "Unfavorite" : "Favorite"}">${r.favorite ? "★" : "☆"}</button>`;
    const title = node.querySelector(".tn-title");
    const date = r.created ? r.created.replace("T", " ") : r.id;
    title.textContent = r.label || date;              // textContent: labels are user text
    if (r.label) {
      const t = document.createElement("span");
      t.className = "tn-date";
      t.textContent = " · " + date;
      title.appendChild(t);
    }
    if ((r.tags || []).length) {
      const sub = node.querySelector(".tn-sub");
      for (const tg of r.tags) {
        const c = document.createElement("span");
        c.className = "tag-chip";
        c.textContent = tg;
        sub.appendChild(c);
      }
    }
    node.querySelector(".tn-star").addEventListener("click", (e) => {
      e.stopPropagation();
      saveMeta(r.id, { favorite: !r.favorite });
    });
    node.addEventListener("click", () => showDetail(r.id));
    return node;
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
    const isEnc = (d.config && d.config.environment) === "encoder";
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

    const meta = d.meta || {};
    detailEl.innerHTML = `
      <div class="detail-head">
        <div><div class="dh-title">${d.id}</div><div class="dh-sub">${(d.config && d.config.environment) || ""}</div></div>
        <div class="dh-badges">${statusDot(summ.status || "running")}${summ.checkpoint ? '<span class="ckpt">checkpoint</span>' : '<span class="nockpt">no checkpoint</span>'}<button id="dh-fav" class="dh-star${meta.favorite ? " on" : ""}" title="Toggle favorite">${meta.favorite ? "★" : "☆"}</button><button id="export-btn" class="runs-btn" title="Download every detail of this run as a JSON file">⤓ Export JSON</button></div>
      </div>
      <div class="meta-edit">
        <input id="me-label" class="me-in" placeholder="label (shown in the run list)" maxlength="80" spellcheck="false" />
        <input id="me-group" class="me-in" list="run-groups" placeholder="group" maxlength="80" spellcheck="false" />
        <input id="me-tags" class="me-in" placeholder="tags, comma separated" spellcheck="false" />
        <button id="me-save" class="runs-btn">Save</button>
        <span id="me-status" class="me-status"></span>
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
        ${!isEnc ? `
        <section class="d-card wide">
          <h3>Inference / verify</h3>
          <div class="infer-bar">
            <button id="infer-btn" class="runs-btn" ${summ.checkpoint ? "" : "disabled"}>${isTree ? "▶ Generate sample" : "▶ Play checkpoint"}</button>
            <span class="infer-status" id="infer-status">${summ.checkpoint ? (isTree ? "generates a text sample from the saved tree model" : "runs the saved champion on a fresh game") : "no checkpoint saved for this run"}</span>
          </div>
          <div class="infer-stage" id="infer-stage"><canvas id="infer-canvas"></canvas></div>
        </section>` : ""}
        ${(isTree || isEnc) && summ.checkpoint ? `
        <section class="d-card wide">
          <h3>Embedding space — letters <span class="dh-sub">the 256 byte embeddings (PCA → 3D, drag to rotate)</span></h3>
          <div class="emb-info" id="emb-info">loading embedding cloud…</div>
          <div class="emb-stage"><canvas id="emb-canvas"></canvas></div>
          <div class="emb-legend" id="emb-legend"></div>
        </section>
        <section class="d-card wide">
          <h3>Embedding space — words <span class="dh-sub">top corpus words encoded as context vectors (PCA → 3D, drag to rotate)</span></h3>
          <div class="emb-info" id="wemb-info">loading word cloud…</div>
          <div class="emb-stage"><canvas id="wemb-canvas"></canvas></div>
          <div class="emb-legend">dot size = word frequency · most frequent 30 labeled · hover any dot for its word</div>
        </section>` : ""}
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

    // -- metadata editor (values set via .value/.textContent — user text) ----
    $("me-label").value = meta.label || "";
    $("me-group").value = meta.group || "";
    $("me-tags").value = (meta.tags || []).join(", ");
    const groupList = $("run-groups");
    if (groupList) {
      groupList.innerHTML = "";
      for (const g of [...new Set(runs.map((r) => r.group).filter(Boolean))].sort()) {
        const o = document.createElement("option");
        o.value = g;
        groupList.appendChild(o);
      }
    }
    $("dh-fav").addEventListener("click", () => {
      const r = runs.find((x) => x.id === id);
      saveMeta(id, { favorite: !(r ? r.favorite : meta.favorite) });
    });
    $("me-save").addEventListener("click", async () => {
      $("me-status").textContent = "saving…";
      const m = await saveMeta(id, {
        label: $("me-label").value,
        group: $("me-group").value,
        tags: $("me-tags").value,
      });
      $("me-status").textContent = m && !m.error ? "saved ✓" : "save failed";
    });
    for (const mid of ["me-label", "me-group", "me-tags"]) {
      $(mid).addEventListener("keydown", (e) => { if (e.key === "Enter") $("me-save").click(); });
    }
    if (summ.encoder) renderEncoderDetail(summ.encoder);
    if ((isTree || isEnc) && summ.checkpoint) { renderEmbedding(id); renderWordEmbedding(id); }
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
      ["novelty bonus", enc.novelty ? `on · strength ${enc.novelty_strength}` : "off"],
      ["seeded embeddings", enc.seeded_embeddings != null ? (enc.seeded_embeddings ? "co-occurrence PCA" : "random") : null],
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

  // -- embedding space: rotatable 3D scatter of the byte embeddings ----------
  async function renderEmbedding(id) {
    const info = $("emb-info"), cv = $("emb-canvas");
    if (!info || !cv) return;
    let d;
    try { d = await (await fetch(`/api/runs/${id}/embedding`)).json(); } catch (_) { d = null; }
    if (!d || d.error || !d.points) {
      info.textContent = "no embedding available" + (d && d.error ? ` — ${d.error}` : "");
      return;
    }
    const pct = d.explained.map((e) => (e * 100).toFixed(1) + "%").join(" / ");
    let verdict = "";
    if (d.explained[0] > 0.9) verdict = "  ⚠ PC1 holds >90% of variance — embedding has collapsed to ~a line";
    else if (d.effective_rank < 3) verdict = `  ⚠ effective rank ${d.effective_rank} of ${d.embed_dim} — heavy collapse`;
    else if (d.effective_rank < d.embed_dim / 4) verdict = `  ⚠ effective rank ${d.effective_rank} of ${d.embed_dim} — most dims unused`;
    info.textContent = `embed_dim ${d.embed_dim} · PC1/2/3 explain ${pct} · ` +
      `effective rank ${d.effective_rank} · mean vector norm ${d.mean_norm}${verdict}`;

    const labels = d.labels || null;   // word-mode run: embeddings are WORDS
    const cls = (b) => b === 32 ? "space"
      : (b >= 97 && b <= 122) ? "lower" : (b >= 65 && b <= 90) ? "upper"
      : (b >= 48 && b <= 57) ? "digit" : (b > 32 && b < 127) ? "punct" : "other";
    const COL = { space: "#e8f0fe", lower: "#4ea1ff", upper: "#19a974",
                  digit: "#c98500", punct: "#b085f5", other: "#49525e" };
    const LABELED = [32, 101, 116, 97, 111, 110, 105];   // ␣ e t a o n i
    let yaw = 0.7, pitch = 0.35, drag = null;
    const g = cv.getContext("2d");

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth || 600, h = cv.clientHeight || 380;
      cv.width = w * dpr; cv.height = h * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const R = Math.min(w, h) * 0.42;
      const proj = d.points.map((p) => {
        const x1 = p.x * cy + p.z * sy, z1 = -p.x * sy + p.z * cy;
        const y1 = p.y * cp - z1 * sp, z2 = p.y * sp + z1 * cp;
        return { sx: w / 2 + x1 * R, sy: h / 2 - y1 * R, z: z2, b: p.b };
      }).sort((a, b) => a.z - b.z);
      proj.forEach((q) => {
        const t = (q.z + 1.2) / 2.4;                     // depth 0..1
        g.globalAlpha = 0.3 + 0.7 * Math.max(0, Math.min(1, t));
        g.fillStyle = labels ? "#4ea1ff" : COL[cls(q.b)];
        g.beginPath();
        g.arc(q.sx, q.sy, 1.8 + 2.6 * t, 0, 7);
        g.fill();
      });
      g.globalAlpha = 1; g.font = "11px monospace"; g.fillStyle = "#c7d0dc";
      if (labels) {
        // word-mode: label the most frequent tokens (ids 1..30; 0 = <unk>)
        proj.forEach((q) => {
          if (q.b >= 1 && q.b <= 30 && labels[q.b]) g.fillText(labels[q.b], q.sx + 5, q.sy - 4);
        });
      } else {
        LABELED.forEach((b) => {
          const q = proj.find((p) => p.b === b);
          if (q) g.fillText(b === 32 ? "␣" : String.fromCharCode(b), q.sx + 5, q.sy - 4);
        });
      }
    }

    cv.style.cursor = "grab";
    cv.onmousedown = (e) => { drag = { x: e.clientX, y: e.clientY }; e.preventDefault(); };
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      yaw += (e.clientX - drag.x) * 0.01;
      pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - drag.y) * 0.01));
      drag = { x: e.clientX, y: e.clientY };
      draw();
    });
    window.addEventListener("mouseup", () => { drag = null; });
    draw();

    $("emb-legend").innerHTML = [["space", "␣ space"], ["lower", "a–z"],
      ["upper", "A–Z"], ["digit", "0–9"], ["punct", "punctuation"],
      ["other", "control / extended"]]
      .map(([k, lbl]) => `<span class="emb-key"><i style="background:${COL[k]}"></i>${lbl}</span>`)
      .join("");
  }

  // -- word embedding space: frequent words through the trained encoder ------
  async function renderWordEmbedding(id) {
    const info = $("wemb-info"), cv = $("wemb-canvas");
    if (!info || !cv) return;
    let d;
    try { d = await (await fetch(`/api/runs/${id}/words`)).json(); } catch (_) { d = null; }
    if (!d || d.error || !d.points) {
      info.textContent = "no word cloud available" + (d && d.error ? ` — ${d.error}` : "");
      return;
    }
    const pct = d.explained.map((e) => (e * 100).toFixed(1) + "%").join(" / ");
    const baseInfo = `${d.points.length} words · context_dim ${d.context_dim} · ` +
      `PC1/2/3 explain ${pct} · effective rank ${d.effective_rank}`;
    info.textContent = baseInfo;

    const maxN = Math.max(...d.points.map((p) => p.n)) || 1;
    const labeled = new Set(d.points.slice(0, 30).map((p) => p.w));   // freq-ordered
    let yaw = 0.7, pitch = 0.35, drag = null, hovered = null;
    const g = cv.getContext("2d");
    let proj = [];

    function draw() {
      const dpr = window.devicePixelRatio || 1;
      const w = cv.clientWidth || 600, h = cv.clientHeight || 380;
      cv.width = w * dpr; cv.height = h * dpr;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const R = Math.min(w, h) * 0.42;
      proj = d.points.map((p) => {
        const x1 = p.x * cy + p.z * sy, z1 = -p.x * sy + p.z * cy;
        const y1 = p.y * cp - z1 * sp, z2 = p.y * sp + z1 * cp;
        return { sx: w / 2 + x1 * R, sy: h / 2 - y1 * R, z: z2, w: p.w, n: p.n };
      }).sort((a, b) => a.z - b.z);
      for (const q of proj) {
        const t = Math.max(0, Math.min(1, (q.z + 1.2) / 2.4));       // depth 0..1
        g.globalAlpha = q === hovered ? 1 : 0.35 + 0.65 * t;
        g.fillStyle = q === hovered ? "#e3b341" : "#4ea1ff";
        g.beginPath();
        g.arc(q.sx, q.sy, 1.5 + 2.5 * Math.sqrt(q.n / maxN) + 1.5 * t, 0, 7);
        g.fill();
      }
      g.globalAlpha = 1; g.font = "11px monospace";
      for (const q of proj) {
        if (q === hovered || labeled.has(q.w)) {
          g.fillStyle = q === hovered ? "#e3b341" : "#c7d0dc";
          g.fillText(q.w, q.sx + 5, q.sy - 4);
        }
      }
    }

    cv.style.cursor = "grab";
    cv.onmousedown = (e) => { drag = { x: e.clientX, y: e.clientY }; e.preventDefault(); };
    window.addEventListener("mousemove", (e) => {
      if (drag) {
        yaw += (e.clientX - drag.x) * 0.01;
        pitch = Math.max(-1.5, Math.min(1.5, pitch + (e.clientY - drag.y) * 0.01));
        drag = { x: e.clientX, y: e.clientY };
        draw();
      }
    });
    window.addEventListener("mouseup", () => { drag = null; });
    cv.addEventListener("mousemove", (e) => {
      if (drag) return;
      const r = cv.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = null, bd = 14 * 14;   // 14px pick radius, front-most wins ties
      for (const q of proj) {
        const dd = (q.sx - mx) ** 2 + (q.sy - my) ** 2;
        if (dd <= bd) { bd = dd; best = q; }
      }
      if (best !== hovered) {
        hovered = best;
        info.textContent = best
          ? `${baseInfo} · "${best.w}" — ${best.n.toLocaleString()}× in corpus`
          : baseInfo;
        draw();
      }
    });
    cv.addEventListener("mouseleave", () => {
      if (hovered) { hovered = null; info.textContent = baseInfo; draw(); }
    });
    draw();
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
  if (searchEl) searchEl.addEventListener("input", renderTree);
  if (statusEl) statusEl.addEventListener("change", renderTree);
  if (sortEl) sortEl.addEventListener("change", renderTree);
  if (favBtn) favBtn.addEventListener("click", () => {
    favOnly = !favOnly;
    favBtn.classList.toggle("on", favOnly);
    renderTree();
  });
  // deep-link: /runs#<run-id> opens that run (sweep tables, Agent panel).
  // Handles both initial load and hash changes while already on the page;
  // reloads the run list if the id isn't known yet (a just-finished run).
  async function openFromHash() {
    const rid = decodeURIComponent(location.hash.slice(1));
    if (!rid) return;
    let run = runs.find((r) => r.id === rid);
    if (!run) { await load(); run = runs.find((r) => r.id === rid); }
    if (run) { currentEnv = run.environment; renderTabs(); renderTree(); }
    showDetail(rid);
  }
  window.addEventListener("hashchange", openFromHash);
  load().then(openFromHash);
})();
