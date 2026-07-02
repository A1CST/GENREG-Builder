// GENREG training client: owns the /train WebSocket. Reads the Control Panel,
// starts a neuroevolution run on the server, and fans the streamed events out to
// the board (live replay), the Microscope (champion genome), and a status HUD.
(() => {
  const $ = (id) => document.getElementById(id);
  const btnStart = $("btn-train");
  const btnStop = $("btn-train-stop");
  const statusEl = $("train-status");
  const paramsBox = $("constraint-params");
  if (!btnStart || !btnStop) return;

  const constraintBoxes = () => Array.from(document.querySelectorAll('input[name="constraint"]'));
  const checkedConstraints = () => constraintBoxes().filter((c) => c.checked).map((c) => c.value);

  // -- show only the parameter rows relevant to the checked constraints ----
  function updateParamVisibility() {
    const on = new Set(checkedConstraints());
    const env = ($("cp-environment") || {}).value || "snake";
    for (const row of paramsBox.querySelectorAll(".param-row")) {
      const forAttr = row.dataset.for || "";
      let show = forAttr === "always" || forAttr.split(/\s+/).some((c) => on.has(c));
      if (show && row.dataset.env && row.dataset.env !== env) show = false;   // env-specific option
      row.hidden = !show;
    }
  }
  constraintBoxes().forEach((c) => c.addEventListener("change", updateParamVisibility));
  const envSelForParams = $("cp-environment");
  if (envSelForParams) envSelForParams.addEventListener("change", updateParamVisibility);
  updateParamVisibility();

  // when H is evolved, grey out the fixed Net width input and mark it "evolved"
  const evolveBox = $("cp-evolve-hidden");
  function updateHiddenState() {
    const evo = !!(evolveBox && evolveBox.checked);
    const inp = $("pp-hidden"), note = $("pp-hidden-note");
    if (inp) inp.disabled = evo;
    if (note) note.hidden = !evo;
  }
  if (evolveBox) evolveBox.addEventListener("change", updateHiddenState);
  updateHiddenState();

  // -- assemble the config from the Control Panel --------------------------
  function intVal(id, dflt) {
    const el = $(id);
    const n = el ? parseInt(el.value, 10) : NaN;
    return Number.isFinite(n) ? n : dflt;
  }

  function assembleConfig() {
    const params = {};
    for (const inp of paramsBox.querySelectorAll("[data-param]")) {
      const v = parseFloat(inp.value);
      if (Number.isFinite(v)) params[inp.dataset.param] = v;
    }
    const checked = (id) => { const el = $(id); return !!(el && el.checked); };
    return {
      op: "start",
      environment: ($("cp-environment") || {}).value || "snake",
      device: ($("cp-device") || {}).value || "cpu",
      population: intVal("cp-population", 100),
      generations: intVal("cp-generations", 60),
      constraints: checkedConstraints(),
      params,
      snake: { w: intVal("cp-snake-w", 20), h: intVal("cp-snake-h", 15) },
      // evolution controls
      evolve_hidden: checked("cp-evolve-hidden"),
      sexual: !checked("cp-mutation-only"),
      elite: intVal("cp-elite", 2),
      parent_frac: intVal("cp-parent-frac", 25) / 100,
      seed: 0,
    };
  }

  // -- WebSocket -----------------------------------------------------------
  let ws = null;
  let running = false;
  let gotStarted = false;      // did we receive a 'started' before the socket closed?

  function setRunning(on) {
    running = on;
    btnStart.disabled = on;
    btnStop.disabled = !on;
  }

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  function openSocket(onOpen) {
    if (ws && ws.readyState === WebSocket.OPEN) { onOpen(); return; }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/train`);
    ws.onopen = onOpen;
    ws.onmessage = (e) => { try { handleEvent(JSON.parse(e.data)); } catch (_) {} };
    ws.onclose = () => {
      if (!running) return;
      setRunning(false);
      // no 'started' => the /train route wasn't there; the server is likely stale
      setStatus(gotStarted ? "disconnected"
        : "couldn't reach /train — restart the Flask server to load the training route");
    };
    ws.onerror = () => { try { ws.close(); } catch (_) {} };
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "started":
        gotStarted = true;
        setStatus(`starting · ${ev.environment} · pop ${ev.population} · ${ev.generations} gens`);
        hudReset(ev.generations);
        hudShow();
        if (window.GENREG && GENREG.scope) GENREG.scope.setExternal(true);
        break;
      case "generation": {
        const f = ev.fitness || {}, b = ev.best || {};
        if (window.GENREG) {
          if (GENREG.board) GENREG.board.playReplay(ev.replay);
          if (GENREG.scope && ev.genome) GENREG.scope.setGenome(ev.genome.layers, ev.gen);
        }
        hudUpdate(ev);
        setStatus(`gen ${ev.gen}/${ev.generations} · best ${fmt(f.best)} · score ${b.score} · base ${fmt(b.base)}`);
        break;
      }
      case "done":
        setRunning(false);
        hudDone(ev);
        setStatus(`done (${ev.reason}) · champ score ${(ev.best || {}).score ?? "—"}`);
        break;
      case "error":
        setRunning(false);
        setStatus(`error: ${ev.message}`);
        break;
    }
  }

  function fmt(v) { return (v == null || Number.isNaN(v)) ? "—" : (+v).toFixed(2); }

  // -- metrics HUD (floating panel over the game canvas) -------------------
  const hud = $("game-hud");
  const hudGen = $("hud-gen"), sparkEl = $("hud-spark");
  const bestF = [], meanF = [];
  let peakScore = 0;
  const setText = (id, v) => { const el = $(id); if (el) el.textContent = v; };

  function hudReset(gens) {
    bestF.length = 0; meanF.length = 0; peakScore = 0;
    if (hudGen) hudGen.textContent = `gen 0/${gens}`;
    for (const id of ["hud-best", "hud-mean", "hud-median", "hud-score", "hud-peak", "hud-steps", "hud-net"]) setText(id, "—");
    drawSpark();
  }
  function hudShow() { if (hud) hud.hidden = false; }

  function hudUpdate(ev) {
    const f = ev.fitness || {}, b = ev.best || {};
    if (typeof f.best === "number") bestF.push(f.best);
    if (typeof f.mean === "number") meanF.push(f.mean);
    if (hudGen) hudGen.textContent = `gen ${ev.gen}/${ev.generations}`;
    setText("hud-best", fmt(f.best));
    setText("hud-mean", fmt(f.mean));
    setText("hud-median", fmt(f.median));
    const score = b.score != null ? b.score : 0;
    if (score > peakScore) peakScore = score;
    setText("hud-score", score);
    setText("hud-peak", peakScore);
    setText("hud-steps", b.steps != null ? b.steps : "—");
    setText("hud-net", `${b.H != null ? b.H : "—"} · ${b.leak != null ? b.leak : "—"} · ${b.bits != null ? b.bits : "—"}`);
    drawSpark();
  }

  function hudDone(ev) {
    if (hudGen) hudGen.textContent = `done · ${ev.reason || ""}`.trim();
  }

  function drawSpark() {
    if (!sparkEl) return;
    const g = sparkEl.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = sparkEl.clientWidth || 232, h = sparkEl.clientHeight || 52;
    sparkEl.width = Math.round(w * dpr); sparkEl.height = Math.round(h * dpr);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (bestF.length < 2) return;
    const all = bestF.concat(meanF);
    let lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (hi - lo < 1e-6) hi = lo + 1;
    const pad = 4, n = bestF.length;
    const X = (i) => pad + (w - 2 * pad) * (n > 1 ? i / (n - 1) : 0);
    const Y = (v) => (h - pad) - (h - 2 * pad) * (v - lo) / (hi - lo);
    const line = (arr, color) => {
      g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath();
      arr.forEach((v, i) => { const x = X(i), y = Y(v); i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke();
    };
    line(meanF, "#7d8794");
    line(bestF, "#4ea1ff");
  }

  // draggable + hide
  (() => {
    const head = $("hud-head"), close = $("hud-close");
    if (close) close.addEventListener("click", () => { if (hud) hud.hidden = true; });
    if (!head || !hud) return;
    head.addEventListener("pointerdown", (e) => {
      if (e.target === close) return;
      const wrap = hud.parentElement.getBoundingClientRect();
      const box = hud.getBoundingClientRect();
      const offX = e.clientX - box.left, offY = e.clientY - box.top;
      head.setPointerCapture(e.pointerId);
      const move = (ev) => {
        let left = ev.clientX - wrap.left - offX, top = ev.clientY - wrap.top - offY;
        left = Math.max(0, Math.min(wrap.width - box.width, left));
        top = Math.max(0, Math.min(wrap.height - box.height, top));
        hud.style.left = left + "px"; hud.style.top = top + "px"; hud.style.right = "auto";
      };
      const up = () => { head.removeEventListener("pointermove", move); head.removeEventListener("pointerup", up); };
      head.addEventListener("pointermove", move);
      head.addEventListener("pointerup", up);
    });
  })();

  // -- controls ------------------------------------------------------------
  btnStart.addEventListener("click", () => {
    if (running) return;
    const cfg = assembleConfig();
    gotStarted = false;
    setRunning(true);
    setStatus("connecting…");
    openSocket(() => ws.send(JSON.stringify(cfg)));
  });

  btnStop.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ op: "stop" }));
    setStatus("stopping…");
  });

  // -- control panel persistence (localStorage) ----------------------------
  // Remember every control-panel input across refreshes so settings don't reset.
  const CP_KEY = "genreg_controls";
  const cpForm = document.getElementById("control-panel");
  const keyFor = (el) => el.id || (el.name ? `${el.name}:${el.value}` : null);
  const cpControls = () => cpForm ? Array.from(cpForm.querySelectorAll("input, select")).filter(keyFor) : [];

  function saveControls() {
    const data = {};
    for (const el of cpControls()) data[keyFor(el)] = el.type === "checkbox" ? el.checked : el.value;
    try { localStorage.setItem(CP_KEY, JSON.stringify(data)); } catch (_) {}
  }
  function restoreControls() {
    let data;
    try { data = JSON.parse(localStorage.getItem(CP_KEY) || "null"); } catch (_) { return; }
    if (!data) return;
    for (const el of cpControls()) {
      const k = keyFor(el);
      if (!(k in data)) continue;
      if (el.type === "checkbox") el.checked = !!data[k]; else el.value = data[k];
    }
    // fire handlers so dependent UI (board, param visibility, grey-out, PO) re-syncs
    for (const el of cpControls()) {
      const type = (el.tagName === "SELECT" || el.type === "checkbox") ? "change" : "input";
      el.dispatchEvent(new Event(type, { bubbles: true }));
    }
  }
  if (cpForm) {
    cpForm.addEventListener("change", saveControls);
    cpForm.addEventListener("input", saveControls);
  }
  restoreControls();   // run last: all handlers are attached, so restored values propagate
})();
