// GENREG run-config panel: a second floating window (sibling of the Agent
// panel) that tracks the CONFIGURATION + RESULTS of the run belonging to the
// page you are on. Adoption rules (per spec):
//   - arriving on a page CLEARS the panel — it never shows a historical run
//     that finished before you got here;
//   - it adopts a run when you start one on this page (created >= page entry);
//   - it also adopts a run already MID-FLIGHT when you arrive (status
//     running), and fills in the results when that run finishes;
//   - a newer run started on this page replaces the tracked one.
// Data comes from GET /api/active-run?scope=... (the run files on disk), so it
// works no matter which client started the run.
(() => {
  if (document.getElementById("config-panel")) return;

  const SCOPE = { "/": "build", "/tree": "tree", "/diff": "diff" }[location.pathname] || null;
  const SCOPE_LABEL = { build: "engine", tree: "tree LM / encoder", diff: "diffevo" }[SCOPE] || "";
  const LS_POS = "genreg_runpanel_pos", LS_MIN = "genreg_runpanel_min";
  const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) {} };

  // local-time ISO seconds — same format runstore stamps `created` with,
  // so string comparison against page-entry time is safe (same machine).
  const pad = (n) => String(n).padStart(2, "0");
  const nowIso = () => {
    const d = new Date();
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
           `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  };
  const ENTERED = nowIso();

  // -- DOM --------------------------------------------------------------------
  const panel = document.createElement("div");
  panel.id = "config-panel";
  panel.className = "run-panel";
  panel.innerHTML = `
    <div class="rp-head" id="rp-head" title="Drag to move">
      <span class="rp-title">Run Config</span>
      <span class="rp-chip" id="rp-chip" hidden></span>
      <button class="ap-btn rp-push" id="rp-copy" title="Copy this run's config as JSON" hidden>copy</button>
      <button class="ap-btn" id="rp-open" title="Open this run on the runs page" hidden>open run</button>
      <button class="ap-btn" id="rp-min" title="Minimize to a pill">—</button>
    </div>
    <div class="rp-body" id="rp-body">
      <div id="rp-active"></div>
      <div class="rp-histsec" id="rp-histsec"></div>
    </div>`;
  const pill = document.createElement("button");
  pill.id = "config-pill";
  pill.className = "agent-pill run-pill";
  pill.title = "Run-config panel — click to open";
  pill.textContent = "Config";
  document.body.appendChild(panel);
  document.body.appendChild(pill);

  const pos = lsGet(LS_POS, null);
  if (pos && Number.isFinite(pos.x) && Number.isFinite(pos.y)) {
    panel.style.left = Math.max(0, Math.min(innerWidth - 90, pos.x)) + "px";
    panel.style.top = Math.max(0, Math.min(innerHeight - 60, pos.y)) + "px";
    panel.style.right = "auto";
  }
  let minimized = !!lsGet(LS_MIN, false);
  const apply = () => { panel.hidden = minimized; pill.hidden = !minimized; };
  apply();

  panel.querySelector("#rp-min").addEventListener("click", () => {
    minimized = true; lsSet(LS_MIN, true); apply();
  });
  pill.addEventListener("click", () => {
    minimized = false; lsSet(LS_MIN, false); apply();
  });

  // drag by header (same behavior as the Agent panel)
  const head = panel.querySelector("#rp-head");
  head.addEventListener("pointerdown", (e) => {
    if (e.target.closest("button")) return;
    e.preventDefault();
    const r = panel.getBoundingClientRect();
    const dx = e.clientX - r.left, dy = e.clientY - r.top;
    head.setPointerCapture(e.pointerId);
    const move = (ev) => {
      panel.style.left = Math.max(0, Math.min(innerWidth - 90, ev.clientX - dx)) + "px";
      panel.style.top = Math.max(0, Math.min(innerHeight - 40, ev.clientY - dy)) + "px";
      panel.style.right = "auto";
    };
    const up = () => {
      head.releasePointerCapture(e.pointerId);
      head.removeEventListener("pointermove", move);
      head.removeEventListener("pointerup", up);
      const rr = panel.getBoundingClientRect();
      lsSet(LS_POS, { x: Math.round(rr.left), y: Math.round(rr.top) });
    };
    head.addEventListener("pointermove", move);
    head.addEventListener("pointerup", up);
  });

  // -- rendering ----------------------------------------------------------------
  // flatten nested config/summary into [key, value] rows; everything shown.
  function flatten(obj, prefix, out) {
    out = out || [];
    for (const k of Object.keys(obj || {})) {
      const v = obj[k];
      if (v == null || v === "") continue;
      const key = prefix ? `${prefix}.${k}` : k;
      if (Array.isArray(v)) {
        if (v.every((x) => typeof x !== "object" || x === null)) {
          out.push([key, v.join(", ").slice(0, 160)]);
        } else {
          out.push([key, `${v.length} item(s)`]);
        }
      } else if (typeof v === "object") {
        flatten(v, key, out);
      } else {
        out.push([key, String(v).slice(0, 160)]);
      }
    }
    return out;
  }

  function section(title, rows) {
    const s = document.createElement("div");
    s.className = "rp-section";
    const h = document.createElement("div");
    h.className = "rp-sec-head";
    h.textContent = title;
    s.appendChild(h);
    const tbl = document.createElement("table");
    tbl.className = "rp-table";
    for (const [k, v] of rows) {
      const tr = document.createElement("tr");
      const td1 = document.createElement("td"), td2 = document.createElement("td");
      td1.textContent = k; td2.textContent = v;
      tr.append(td1, td2);
      tbl.appendChild(tr);
    }
    s.appendChild(tbl);
    return s;
  }

  function message(text) {
    const d = document.createElement("div");
    d.className = "rp-empty";
    d.textContent = text;
    return d;
  }

  const bodyEl = panel.querySelector("#rp-active");
  const chipEl = panel.querySelector("#rp-chip");
  const openBtn = panel.querySelector("#rp-open");
  const copyBtn = panel.querySelector("#rp-copy");
  let trackedId = null;
  let trackedConfig = null;  // for the header copy button
  let lastPayload = null;    // skip DOM rebuilds when nothing changed
  let openRunId = null;
  openBtn.addEventListener("click", () => {
    if (openRunId) location.href = "/runs#" + encodeURIComponent(openRunId);
  });

  // -- copy a run's config JSON to the clipboard ------------------------------
  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (_) {
      // fallback: hidden textarea + execCommand
      try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        ta.remove();
        return ok;
      } catch (_) { return false; }
    }
  }

  function wireCopy(btn, getConfig) {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const cfg = getConfig();
      if (!cfg) return;
      const ok = await copyText(JSON.stringify(cfg, null, 2));
      const old = btn.textContent;
      btn.textContent = ok ? "copied ✓" : "copy failed";
      setTimeout(() => { btn.textContent = old; }, 1200);
    });
  }
  wireCopy(copyBtn, () => trackedConfig);

  function setChip(status) {
    if (!status) { chipEl.hidden = true; return; }
    chipEl.hidden = false;
    chipEl.textContent = status;
    chipEl.className = "rp-chip " +
      (status === "finished" ? "ok" : status === "running" ? "run" :
       status === "stopped" ? "warn" : "bad");
    pill.classList.toggle("pulse", status === "running");
  }

  function renderIdle() {
    const payload = "idle:" + (SCOPE || "none");
    if (payload === lastPayload) return;
    lastPayload = payload;
    trackedId = null; openRunId = null; trackedConfig = null;
    setChip(null);
    openBtn.hidden = true;
    copyBtn.hidden = true;
    bodyEl.textContent = "";
    bodyEl.appendChild(message(SCOPE
      ? `Waiting for a ${SCOPE_LABEL} run — start training on this page (or arrive while one is mid-flight) and its full configuration + results appear here.`
      : "No trainable project on this page — the Build, Tree LM, and DiffEvo pages have run configs."));
  }

  function renderRun(r) {
    const payload = JSON.stringify(r);
    if (payload === lastPayload) return;
    lastPayload = payload;
    openRunId = r.id;
    trackedConfig = r.config || null;
    openBtn.hidden = false;
    copyBtn.hidden = false;
    setChip(r.status);
    bodyEl.textContent = "";

    const idLine = document.createElement("div");
    idLine.className = "rp-runid";
    idLine.textContent = `${r.id}${r.created ? " · started " + r.created.replace("T", " ") : ""}`;
    bodyEl.appendChild(idLine);

    bodyEl.appendChild(section("Configuration",
      flatten(r.config).filter(([k]) => k !== "op")));

    if (r.summary) {
      const rows = flatten(r.summary).filter(([k]) => k !== "id" && k !== "environment");
      bodyEl.appendChild(section("Results", rows));
    } else if (r.last_metric) {
      const rows = flatten(r.last_metric);
      bodyEl.appendChild(section("Results — running…", rows));
    } else {
      const s = document.createElement("div");
      s.className = "rp-section";
      s.appendChild(message("Results — running, no metrics yet…"));
      bodyEl.appendChild(s);
    }
  }

  // -- config history: last N runs across ALL projects -----------------------
  const LS_HIST = "genreg_runpanel_histn";
  const histSec = panel.querySelector("#rp-histsec");
  let histLimit = [3, 5, 10].includes(lsGet(LS_HIST, 5)) ? lsGet(LS_HIST, 5) : 5;
  let histData = [];
  let histPayload = null;
  const expanded = new Set();   // run ids the user opened

  // one-line result summary per run, program-aware
  function headline(r) {
    const s = r.summary || {};
    const ev = s.eval || {}, b = s.best || {}, enc = s.encoder || {};
    if (typeof ev.accuracy === "number") return `acc ${(ev.accuracy * 100).toFixed(1)}%`;
    if (typeof enc.nc_accuracy === "number") return `nc ${(enc.nc_accuracy * 100).toFixed(1)}%`;
    if (typeof b.final_l1 === "number") return `L1 ${b.final_l1}`;
    if (typeof b.score === "number") return `score ${b.score}`;
    return r.status || "—";
  }

  function renderHistory() {
    histSec.textContent = "";
    const head = document.createElement("div");
    head.className = "rp-sec-head rp-hist-head";
    const title = document.createElement("span");
    title.textContent = "Config history — all projects";
    const sel = document.createElement("select");
    sel.className = "rp-hist-sel";
    sel.title = "How many recent runs to list";
    for (const n of [3, 5, 10]) {
      const o = document.createElement("option");
      o.value = String(n);
      o.textContent = `last ${n}`;
      if (n === histLimit) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => {
      histLimit = parseInt(sel.value, 10) || 5;
      lsSet(LS_HIST, histLimit);
      histPayload = null;
      pollHistory();
    });
    head.append(title, sel);
    histSec.appendChild(head);

    if (!histData.length) {
      histSec.appendChild(message("No runs recorded yet."));
      return;
    }
    for (const r of histData) {
      const item = document.createElement("div");
      item.className = "rp-hitem";
      const hh = document.createElement("div");
      hh.className = "rp-hitem-head";
      hh.title = "Click to expand full config + results";
      const dot = document.createElement("span");
      dot.className = "rp-hdot " +
        (r.status === "finished" ? "ok" : r.status === "running" ? "run" :
         r.status === "stopped" ? "warn" : "bad");
      const env = document.createElement("span");
      env.className = "rp-henv";
      env.textContent = r.environment;
      const time = document.createElement("span");
      time.className = "rp-htime";
      time.textContent = (r.created || "").replace("T", " ").slice(5, 16);
      const res = document.createElement("span");
      res.className = "rp-hres";
      res.textContent = headline(r);
      const copy = document.createElement("button");
      copy.className = "ap-btn rp-hcopy";
      copy.textContent = "copy";
      copy.title = "Copy this run's config as JSON";
      wireCopy(copy, () => r.config);
      hh.append(dot, env, time, res, copy);
      item.appendChild(hh);

      if (expanded.has(r.id)) {
        const detail = document.createElement("div");
        detail.className = "rp-hdetail";
        const idLine = document.createElement("div");
        idLine.className = "rp-runid";
        idLine.textContent = r.id;
        idLine.style.cursor = "pointer";
        idLine.title = "Open this run on the runs page";
        idLine.addEventListener("click", () =>
          location.href = "/runs#" + encodeURIComponent(r.id));
        detail.appendChild(idLine);
        detail.appendChild(section("Configuration",
          flatten(r.config).filter(([k]) => k !== "op")));
        if (r.summary) {
          detail.appendChild(section("Results",
            flatten(r.summary).filter(([k]) => k !== "id" && k !== "environment")));
        } else {
          detail.appendChild(message("still running — no results yet"));
        }
        item.appendChild(detail);
      }
      hh.addEventListener("click", () => {
        if (expanded.has(r.id)) expanded.delete(r.id); else expanded.add(r.id);
        renderHistory();
      });
      histSec.appendChild(item);
    }
  }

  async function pollHistory() {
    let list = null;
    try { list = await (await fetch(`/api/run-history?limit=${histLimit}`)).json(); } catch (_) { return; }
    if (!Array.isArray(list)) return;
    const payload = JSON.stringify(list);
    if (payload === histPayload) return;
    histPayload = payload;
    histData = list;
    renderHistory();
  }

  // -- polling + adoption rules ---------------------------------------------
  async function poll() {
    if (!SCOPE) { renderIdle(); return; }
    let r = null;
    try { r = await (await fetch(`/api/active-run?scope=${SCOPE}`)).json(); } catch (_) { return; }
    if (!r || !r.id) { if (!trackedId) renderIdle(); return; }
    if (r.id === trackedId) { renderRun(r); return; }             // tracked: keep updating
    const startedHere = r.created && r.created >= ENTERED;        // new run since page entry
    const midFlight = r.status === "running";                     // arrived during a run
    if (startedHere || midFlight) { trackedId = r.id; renderRun(r); return; }
    if (!trackedId) renderIdle();   // newest run predates this visit — stay cleared
  }

  renderIdle();
  poll();
  if (SCOPE) setInterval(poll, 4000);
  renderHistory();
  pollHistory();
  setInterval(pollHistory, 12000);   // history is cross-project: poll everywhere
})();
