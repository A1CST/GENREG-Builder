// GENREG front-end: tabbed xterm.js terminals bridged to real PTYs over a WebSocket.

const tabsEl = document.getElementById("tabs");
const panesEl = document.getElementById("panes");
const connDot = document.getElementById("conn-dot");
const connText = document.getElementById("conn-text");

const terms = new Map();   // id -> {id, title, alive, term, fit, paneEl, tabEl}
const held = new Map();    // id -> {tabEl, title, deadline, tick}  (closed, reopenable)
let activeId = null;
let ws = null;
let wsReady = false;

// Each page remembers which terminal tab it had focused (keyed by pathname),
// so switching between project pages doesn't lose "your" terminal.
const ACTIVE_TAB_KEY = "genreg_term_tab:" + location.pathname;
function rememberActiveTab(id) {
  try { localStorage.setItem(ACTIVE_TAB_KEY, String(id)); } catch (_) {}
}
function recallActiveTab() {
  try {
    const v = localStorage.getItem(ACTIVE_TAB_KEY);
    if (v == null) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  } catch (_) { return null; }
}

// The connection dot/text live in the topbar (build/I2) or the injected dock
// header (other pages); guard so a page without them still gets terminals.
function setConn(cls, text) {
  if (connDot) connDot.className = cls;
  if (connText) connText.textContent = text;
}

// -- project tags ----------------------------------------------------------
// Terminals are shared across every project page and long-lived, so a tab has
// no inherent project. The user tags each terminal with the project it is
// driving; the active terminal then shows a loud, colored banner so a reply
// never lands in the wrong project's session. Tags are per-terminal-id and
// global (NOT per-path) because the same terminal appears on every page.
// The tag-able project list. Driven by window.GENREG_PROJECTS, which _nav.html
// emits from the ONE registry in app.py (PROJECT_GROUPS) — so a project added
// to the nav is automatically tag-able here too. The literal below is only a
// fallback for a page that somehow renders without the nav include.
const PROJECTS = (Array.isArray(window.GENREG_PROJECTS) && window.GENREG_PROJECTS.length)
  ? window.GENREG_PROJECTS
  : [
  { key: "build",     label: "Build",     color: "#4ea1ff" },
  { key: "lm",        label: "LM",        color: "#56d364" },
  { key: "mnist",     label: "MNIST",     color: "#e3b341" },
  { key: "cifar",     label: "CIFAR",     color: "#ff7b72" },
  { key: "tsdb",      label: "TSDB",      color: "#39c5cf" },
  { key: "diff",      label: "DiffEvo",   color: "#d2a8ff" },
  { key: "animation", label: "Animation", color: "#ff9e64" },
  { key: "pure",      label: "PURE",      color: "#7ee787" },
  { key: "xray",      label: "X-Ray",     color: "#79c0ff" },
  { key: "radial",    label: "Radial",    color: "#f778ba" },
  { key: "humanoid",  label: "Humanoid",  color: "#ffa657" },
  { key: "resnet",    label: "ResNet",    color: "#d29922" },
  { key: "images",    label: "Images",    color: "#a5a5f5" },
  { key: "video",     label: "Video",     color: "#f0883e" },
  { key: "i2",        label: "I2",        color: "#2ea043" },
  { key: "pia",       label: "PIA",       color: "#db61a2" },
  { key: "history",   label: "History",   color: "#c8a2ff" },
  { key: "runs",      label: "Runs",      color: "#58a6ff" },
];
const PROJ_BY_KEY = new Map(PROJECTS.map((p) => [p.key, p]));
const TAGS_KEY = "genreg_term_project";

function loadTags() {
  try { return JSON.parse(localStorage.getItem(TAGS_KEY)) || {}; }
  catch (_) { return {}; }
}
function saveTags(m) {
  try { localStorage.setItem(TAGS_KEY, JSON.stringify(m)); } catch (_) {}
}
function projFor(id) { return PROJ_BY_KEY.get(loadTags()[id]) || null; }
function setTag(id, key) {
  const m = loadTags();
  if (key) m[id] = key; else delete m[id];
  saveTags(m);
  const t = terms.get(id);
  if (t) refreshTab(t);
  if (id === activeId) renderProjBar();
}

// The active-terminal banner: a full-width colored strip at the top of the
// dock naming the project you are about to type into. Created lazily so it
// works on both the hard-coded docks (build/I2) and the injected one.
let projBarEl = null;
function ensureProjBar() {
  if (projBarEl) return projBarEl;
  const panel = document.getElementById("terminal-panel");
  if (!panel) return null;
  projBarEl = document.createElement("div");
  projBarEl.className = "term-projbar";
  projBarEl.addEventListener("click", () => {
    if (activeId != null) openProjMenu(activeId, projBarEl);
  });
  panel.insertBefore(projBarEl, panel.firstChild);
  return projBarEl;
}
function renderProjBar() {
  const bar = ensureProjBar();
  const p = activeId != null ? projFor(activeId) : null;
  // reinforce the cue on the pane itself so it stays visible while reading
  const at = activeId != null ? terms.get(activeId) : null;
  if (at) {
    at.paneEl.style.setProperty("--pc", p ? p.color : "transparent");
    at.paneEl.classList.toggle("tagged", !!p);
  }
  if (!bar) return;
  if (p) {
    bar.innerHTML = `<span class="pb-dot"></span><b class="pb-name"></b>` +
      `<span class="pb-hint">typing here goes to this project · click to change</span>`;
    bar.querySelector(".pb-dot").style.background = p.color;
    bar.querySelector(".pb-name").textContent = p.label;
    bar.style.setProperty("--pc", p.color);
    bar.classList.add("tagged");
  } else {
    bar.innerHTML = `<span class="pb-hint">no project tag · ` +
      `click here (or right-click a tab) to label this terminal</span>`;
    bar.style.removeProperty("--pc");
    bar.classList.remove("tagged");
  }
}

// The picker: a small floating menu of projects with color swatches.
let projMenuEl = null;
function openProjMenu(id, anchorEl) {
  closeProjMenu();
  const cur = loadTags()[id] || "";
  projMenuEl = document.createElement("div");
  projMenuEl.className = "term-projmenu";
  const rows = PROJECTS.map((p) =>
    `<div class="pm-row${p.key === cur ? " on" : ""}" data-k="${p.key}">` +
    `<span class="pm-sw" style="background:${p.color}"></span>${p.label}</div>`
  ).join("");
  projMenuEl.innerHTML = `<div class="pm-head">Tag terminal ${id}</div>${rows}` +
    `<div class="pm-row pm-clear" data-k="">✕ Untag</div>`;
  document.body.appendChild(projMenuEl);

  const r = anchorEl.getBoundingClientRect();
  const mh = projMenuEl.offsetHeight;
  // prefer above the anchor (the dock sits at the bottom of the viewport)
  let top = r.top - mh - 4;
  if (top < 8) top = Math.min(r.bottom + 4, window.innerHeight - mh - 8);
  projMenuEl.style.top = Math.max(8, top) + "px";
  projMenuEl.style.left =
    Math.min(r.left, window.innerWidth - projMenuEl.offsetWidth - 8) + "px";

  projMenuEl.addEventListener("click", (e) => {
    const row = e.target.closest(".pm-row");
    if (!row) return;
    setTag(id, row.getAttribute("data-k"));
    closeProjMenu();
  });
  setTimeout(() => document.addEventListener("pointerdown", onDocDown, true), 0);
}
function onDocDown(e) {
  if (projMenuEl && !projMenuEl.contains(e.target)) closeProjMenu();
}
function closeProjMenu() {
  if (!projMenuEl) return;
  document.removeEventListener("pointerdown", onDocDown, true);
  projMenuEl.remove();
  projMenuEl = null;
}

const THEME = {
  background: "#0d1117", foreground: "#d7dde5", cursor: "#4ea1ff",
  selectionBackground: "#264f78",
  black: "#0d1117", brightBlack: "#7d8794",
  red: "#f85149", brightRed: "#ff7b72",
  green: "#3fb950", brightGreen: "#56d364",
  yellow: "#d29922", brightYellow: "#e3b341",
  blue: "#4ea1ff", brightBlue: "#79c0ff",
  magenta: "#bc8cff", brightMagenta: "#d2a8ff",
  cyan: "#39c5cf", brightCyan: "#56d4dd",
  white: "#d7dde5", brightWhite: "#ffffff",
};

// -- WebSocket -------------------------------------------------------------
function send(obj) {
  if (ws && wsReady) { try { ws.send(JSON.stringify(obj)); } catch (_) {} }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    wsReady = true;
    setConn("dot ok", "connected");
    // Re-send sizes for any terminals restored from a snapshot.
    for (const t of terms.values()) fitAndReport(t);
  };
  ws.onmessage = (e) => {
    try { handleEvent(JSON.parse(e.data)); } catch (_) {}
  };
  ws.onclose = () => {
    wsReady = false;
    setConn("dot bad", "reconnecting…");
    setTimeout(connect, 1500);
  };
  ws.onerror = () => { try { ws.close(); } catch (_) {} };
}

// -- tab / pane / xterm management ----------------------------------------
function ensureTerm(meta) {
  let t = terms.get(meta.id);
  if (t) {
    if (meta.title != null) t.title = meta.title;
    if (meta.alive != null) t.alive = meta.alive;
    refreshTab(t);
    return t;
  }

  const tabEl = document.createElement("div");
  tabEl.className = "tab";
  tabEl.innerHTML = `<span class="pdot" title="Project tag"></span>` +
    `<span class="title"></span><button class="close" title="Close">×</button>`;
  tabEl.addEventListener("click", (e) => {
    if (e.target.classList.contains("close")) { send({ op: "close", id: meta.id }); e.stopPropagation(); return; }
    if (e.target.classList.contains("pdot")) { openProjMenu(meta.id, e.target); e.stopPropagation(); return; }
    if (e.target.classList.contains("tag-prefix")) { openProjMenu(meta.id, e.target); e.stopPropagation(); return; }
    setActive(meta.id);
  });
  // right-click anywhere on the tab tags it with a project
  tabEl.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    openProjMenu(meta.id, tabEl);
  });
  tabEl.title = "Click to focus · right-click to tag with a project";
  tabsEl.appendChild(tabEl);

  const paneEl = document.createElement("div");
  paneEl.className = "pane";
  panesEl.appendChild(paneEl);

  const term = new Terminal({
    cursorBlink: true, fontFamily: '"Cascadia Code","Consolas",monospace',
    fontSize: 13, theme: THEME, scrollback: 5000, allowProposedApi: true,
    // Seed with the PTY's own dimensions so xterm and the PTY agree even
    // before the first fit() runs (avoids a brief wrap on load).
    cols: meta.cols || 100, rows: meta.rows || 30,
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(paneEl);
  term.onData((data) => send({ op: "input", id: meta.id, data }));

  t = { id: meta.id, title: meta.title || `Terminal ${meta.id}`,
        alive: meta.alive !== false, term, fit, paneEl, tabEl, _cols: 0, _rows: 0 };
  terms.set(meta.id, t);
  refreshTab(t);
  if (activeId === null) setActive(meta.id);
  return t;
}

function refreshTab(t) {
  const p = projFor(t.id);
  t.tabEl.querySelector(".title").innerHTML =
    (p ? `<span class="tag-prefix" title="Click to change project label" style="border-bottom:1px dashed ${p.color}; cursor:pointer; margin-right:4px;">${p.label}</span> · ` : "") + t.title;
  t.tabEl.classList.toggle("dead", !t.alive);
  t.tabEl.classList.toggle("tagged", !!p);
  const dot = t.tabEl.querySelector(".pdot");
  if (dot) {
    dot.style.background = p ? p.color : "transparent";
    dot.style.borderColor = p ? p.color : "var(--border)";
  }
  t.tabEl.style.setProperty("--pc", p ? p.color : "var(--accent)");
}

function setActive(id) {
  activeId = id;
  rememberActiveTab(id);
  for (const t of terms.values()) {
    const on = t.id === id;
    t.tabEl.classList.toggle("active", on);
    t.paneEl.classList.toggle("active", on);
  }
  renderProjBar();
  const t = terms.get(id);
  if (t) {
    // Defer the fit one frame so the pane has been laid out (a synchronous
    // fit right after display:block measures stale/zero dimensions).
    requestAnimationFrame(() => { fitAndReport(t); t.term.focus(); });
  }
}

function removeTerm(id) {
  const t = terms.get(id);
  if (!t) return;
  try { t.term.dispose(); } catch (_) {}
  t.tabEl.remove();
  t.paneEl.remove();
  terms.delete(id);
  if (activeId === id) {
    activeId = null;
    const next = terms.keys().next();
    if (!next.done) setActive(next.value);
  }
}

// -- held (recently-closed) sessions --------------------------------------
// A closed tab isn't gone: the daemon keeps its shell alive for a grace
// window. We tear down the live pane but leave a dim "held" tab with a
// countdown; clicking it re-opens the session with its scrollback + process.
const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.max(0, s % 60)).padStart(2, "0")}`;

function holdTerm(id, title, graceSecs) {
  // drop the live term/pane (the daemon still owns the process)
  const t = terms.get(id);
  if (t) {
    try { t.term.dispose(); } catch (_) {}
    t.paneEl.remove();
    t.tabEl.remove();
    terms.delete(id);
    if (activeId === id) {
      activeId = null;
      const next = terms.keys().next();
      if (!next.done) setActive(next.value);
    }
  }
  releaseHeld(id);   // replace any existing held entry for this id

  const tabEl = document.createElement("div");
  tabEl.className = "tab held";
  tabEl.innerHTML = `<span class="reopen-ico">↻</span><span class="title"></span>` +
                    `<span class="held-timer"></span>`;
  tabEl.title = "Session held — click to reopen";
  tabEl.addEventListener("click", () => send({ op: "reopen", id }));
  tabsEl.appendChild(tabEl);

  const deadline = Date.now() + graceSecs * 1000;
  const h = { tabEl, title, deadline, tick: null };
  held.set(id, h);
  tabEl.querySelector(".title").textContent = title;

  const render = () => {
    const left = Math.round((h.deadline - Date.now()) / 1000);
    if (left <= 0) { releaseHeld(id); return; }   // daemon reaps ~now
    tabEl.querySelector(".held-timer").textContent = mmss(left);
  };
  render();
  h.tick = setInterval(render, 1000);
}

function releaseHeld(id) {
  const h = held.get(id);
  if (!h) return;
  clearInterval(h.tick);
  h.tabEl.remove();
  held.delete(id);
}

function fitAndReport(t) {
  // Only the visible pane can be measured correctly.
  if (t.id !== activeId) return;
  try {
    t.fit.fit();                      // resizes the xterm instance to the pane
    const cols = t.term.cols, rows = t.term.rows;
    if (cols && rows && (cols !== t._cols || rows !== t._rows)) {
      t._cols = cols; t._rows = rows;
      send({ op: "resize", id: t.id, cols, rows });   // keep the PTY in lockstep
    }
  } catch (_) {}
}

// -- event handling --------------------------------------------------------
function handleEvent(ev) {
  switch (ev.type) {
    case "snapshot": {
      const want = recallActiveTab();   // read BEFORE ensureTerm auto-activates tab 1
      for (const meta of ev.terminals) {
        if (meta.detached) {            // closed but still held — show reopenable
          holdTerm(meta.id, meta.title || `Terminal ${meta.id}`,
                   meta.grace_remaining || 0);
          continue;
        }
        const t = ensureTerm(meta);
        t.term.reset();
        if (meta.data) t.term.write(meta.data);
      }
      // Restore this page's remembered tab (if it still exists).
      if (want != null && want !== activeId && terms.has(want)) setActive(want);
      return;
    }
    case "terminal_detached":
      holdTerm(ev.id, (terms.get(ev.id) || {}).title || `Terminal ${ev.id}`,
               ev.grace || 0);
      return;
    case "terminal_restored": {         // reopen brought it back with scrollback
      releaseHeld(ev.id);
      const t = ensureTerm(ev);
      t.term.reset();
      if (ev.data) t.term.write(ev.data);
      setActive(ev.id);
      return;
    }
    case "terminal_created": {
      const t = ensureTerm(ev);
      // A tab we requested via the Claude button: focus it, then type the
      // command only once the PowerShell PROMPT has actually appeared.
      // (A blind timer raced the terminal's device-attributes reply —
      // xterm.js answers ESC[c with ESC[?1;2c, and that response landed on
      // the command line as literal "[?1;2c" garbage before our command.)
      if (pendingClaude > 0 && ev.title === "Claude") {
        pendingClaude--;
        t.waitPrompt = true;
        t.promptBuf = "";
        // fallback: if we never recognize a prompt, fire anyway after 5s
        t.promptTimer = setTimeout(() => launchClaude(t), 5000);
        setActive(t.id);
      }
      if (pendingGemini > 0 && ev.title === "Gemini") {
        pendingGemini--;
        t.waitPrompt = true;
        t.promptBuf = "";
        t.promptTimer = setTimeout(() => launchGemini(t), 5000);
        setActive(t.id);
      }
      return;
    }
    case "terminal_closed": releaseHeld(ev.id); removeTerm(ev.id); return;
  }
  const t = terms.get(ev.id);
  if (!t) return;
  switch (ev.type) {
    case "output":
    case "system":
      t.term.write(ev.data);
      if (t.waitPrompt && ev.type === "output") {
        t.promptBuf = (t.promptBuf + ev.data).slice(-2000);
        // strip ANSI escapes, then look for a shell prompt at the tail
        const clean = t.promptBuf
          .replace(/\x1b\][^\x07]*(\x07|\x1b\\)/g, "")   // OSC sequences
          .replace(/\x1b\[[0-9;?]*[ -\/]*[@-~]/g, "")     // CSI sequences
          .replace(/\r/g, "");
        if (/PS [^>\n]*> ?$/.test(clean.trimEnd())) {
          launchClaude(t);
          launchGemini(t);
        }
      }
      break;
    case "clear": t.term.reset(); break;
  }
}

// Type the claude command into a ready shell: Escape first (PSReadLine
// RevertLine wipes any stray escape-reply characters), then the command.
function launchClaude(t) {
  if (!t.waitPrompt || t.title !== "Claude") return;
  t.waitPrompt = false;
  clearTimeout(t.promptTimer);
  send({ op: "input", id: t.id, data: "\x1b" });
  setTimeout(() => send({ op: "input", id: t.id, data: "claude --dangerously-skip-permissions\r" }), 150);
}

function launchGemini(t) {
  if (!t.waitPrompt || t.title !== "Gemini") return;
  t.waitPrompt = false;
  clearTimeout(t.promptTimer);
  send({ op: "input", id: t.id, data: "\x1b" });
  setTimeout(() => send({ op: "input", id: t.id, data: "agy\r" }), 150);
}

// -- buttons + resize ------------------------------------------------------
document.getElementById("new-tab").addEventListener("click", () => {
  const t = terms.get(activeId);
  let cols = 100, rows = 30;
  if (t) { try { const d = t.fit.proposeDimensions(); if (d) { cols = d.cols; rows = d.rows; } } catch (_) {} }
  send({ op: "create", cols, rows });
});
// Claude launcher: open a fresh tab titled "Claude"; the terminal_created
// handler recognizes it (via pendingClaude) and types the command into it.
let pendingClaude = 0;
document.getElementById("btn-claude").addEventListener("click", () => {
  const t = terms.get(activeId);
  let cols = 100, rows = 30;
  if (t) { try { const d = t.fit.proposeDimensions(); if (d) { cols = d.cols; rows = d.rows; } } catch (_) {} }
  pendingClaude++;
  send({ op: "create", cols, rows, title: "Claude" });
});

let pendingGemini = 0;
document.getElementById("btn-gemini").addEventListener("click", () => {
  const t = terms.get(activeId);
  let cols = 100, rows = 30;
  if (t) { try { const d = t.fit.proposeDimensions(); if (d) { cols = d.cols; rows = d.rows; } } catch (_) {} }
  pendingGemini++;
  send({ op: "create", cols, rows, title: "Gemini" });
});
document.getElementById("btn-clear").addEventListener("click", () => activeId != null && send({ op: "clear", id: activeId }));
document.getElementById("btn-restart").addEventListener("click", () => activeId != null && send({ op: "restart", id: activeId }));
document.getElementById("btn-stop").addEventListener("click", () => activeId != null && send({ op: "stop", id: activeId }));

// -- changelog modal -------------------------------------------------------
// Optional UI: guard every lookup so missing markup (e.g. a stale cached
// template) can never throw and abort the script before connect() runs.
(() => {
  const overlay = document.getElementById("changelog-overlay");
  const body = document.getElementById("changelog-body");
  const btn = document.getElementById("btn-changelog");
  const closeBtn = document.getElementById("changelog-close");
  if (!overlay || !body || !btn || !closeBtn) return;   // markup absent -> skip

  // per-project changelog: pages default to THEIR project's log
  // (documentation/changelogs/CHANGELOG_<X>.md), toggleable to the main log.
  // Prefix matching so sub-pages (/radial/demo, /radial/demo/cousins, ...)
  // resolve to their project too.
  const PROJECT = (() => {
    const p = location.pathname;
    if (p === "/") return "BUILD";
    const rules = [["/radial", "RADIAL"], ["/cifar", "CIFAR"],
                   ["/mnist", "MNIST"], ["/lm_demo", "LM"], ["/lm", "LM"],
                   ["/diff", "DIFFEVO"],
                   ["/animation", "ANIMATION"], ["/pure", "PURE"],
                   ["/xray", "XRAY"], ["/i2", "I2"], ["/tree", "TREE"],
                   ["/evolang", "EVOLANG"], ["/images", "IMAGES"],
                   ["/video", "VIDEO"], ["/humanoid", "HUMANOID"],
                   ["/progress", "PROGRESS"],
                   ["/vision_demo", "VISION_DEMO"],
                   ["/history", "HISTORY"]];
    for (const [pre, proj] of rules)
      if (p === pre || p.startsWith(pre + "/")) return proj;
    return null;   // meta pages (plan/runs/docs) keep the main log
  })();
  let scope = PROJECT ? "project" : "main";
  const title = document.getElementById("changelog-title");
  let toggle = null;
  const head = overlay.querySelector(".modal-head");
  if (PROJECT && head) {
    toggle = document.createElement("button");
    toggle.className = "clg-scope";
    toggle.title = "Switch between this project's changelog and the main (all-projects) changelog";
    head.insertBefore(toggle, closeBtn);
    toggle.addEventListener("click", () => {
      scope = scope === "project" ? "main" : "project";
      load();
    });
  }
  const load = () => {
    body.textContent = "Loading…";
    if (toggle) toggle.textContent = scope === "project" ? "show all projects" : "show this project only";
    if (title) title.textContent = scope === "project" ? "Changelog — this project" : "Changelog — all projects";
    const url = scope === "project"
      ? `/api/docs/file/changelogs/CHANGELOG_${PROJECT}.md`
      : "/changelog";
    fetch(url)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((t) => (body.textContent = t))
      .catch(() => {
        if (scope === "project") {   // missing project log -> fall back to main
          scope = "main";
          load();
        } else {
          body.textContent = "Failed to load changelog.";
        }
      });
  };
  const open = () => {
    overlay.hidden = false;
    load();
  };
  const close = () => (overlay.hidden = true);
  btn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => e.target === overlay && close());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !overlay.hidden) {
      close();
    } else if (e.shiftKey && e.key === "ArrowLeft" && activeId !== null) {
      const t = terms.get(activeId);
      if (t && t.tabEl) {
        const prev = t.tabEl.previousElementSibling;
        if (prev && prev.classList.contains("tab")) {
          tabsEl.insertBefore(t.tabEl, prev);
          e.preventDefault();
        }
      }
    } else if (e.shiftKey && e.key === "ArrowRight" && activeId !== null) {
      const t = terms.get(activeId);
      if (t && t.tabEl) {
        const next = t.tabEl.nextElementSibling;
        if (next && next.classList.contains("tab")) {
          tabsEl.insertBefore(t.tabEl, next.nextSibling);
          e.preventDefault();
        }
      }
    }
  });
})();

// Auto-refit the active terminal whenever its container actually changes size
// (window resize, layout shifts, font load reflow) — more reliable than
// listening to window 'resize' alone.
let roTimer = null;
const ro = new ResizeObserver(() => {
  clearTimeout(roTimer);
  roTimer = setTimeout(() => { const t = terms.get(activeId); if (t) fitAndReport(t); }, 80);
});
ro.observe(panesEl);

// Web/system fonts can finish loading after first paint, changing the cell
// metrics; refit once they're ready so column math is correct.
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(() => { const t = terms.get(activeId); if (t) fitAndReport(t); });
}

// -- terminal quick command presets -----------------------------------------
const PRESETS = [
  { name: "Run pytest (all tests)", cmd: "pytest" },
  { name: "Post alert notice", cmd: "python agent_notify.py \"Alert notice\" \"Description\" --kind alert --source human" },
  { name: "Start LM training run", cmd: "python radial_evo.py" },
  { name: "Show python processes", cmd: "Get-Process -Name python" }
];
let cmdMenu = null;
function closeCmdMenu() {
  if (cmdMenu) { cmdMenu.remove(); cmdMenu = null; }
}
document.addEventListener("click", (e) => {
  if (!e.target.closest("#btn-quick-cmds") && !e.target.closest(".term-cmd-menu")) {
    closeCmdMenu();
  }
});

const termActions = document.querySelector(".term-actions");
if (termActions) {
  const cmdBtn = document.createElement("button");
  cmdBtn.id = "btn-quick-cmds";
  cmdBtn.className = "term-cmd-btn";
  cmdBtn.title = "Execute common presets";
  cmdBtn.textContent = "Presets";
  termActions.appendChild(cmdBtn);

  cmdBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (cmdMenu) { closeCmdMenu(); return; }
    if (activeId === null) return;
    
    cmdMenu = document.createElement("div");
    cmdMenu.className = "term-cmd-menu";
    
    PRESETS.forEach((p) => {
      const item = document.createElement("div");
      item.className = "term-cmd-item";
      item.textContent = p.name;
      item.title = p.cmd;
      item.addEventListener("click", () => {
        send({ op: "input", id: activeId, data: p.cmd + "\r" });
        closeCmdMenu();
      });
      cmdMenu.appendChild(item);
    });

    document.body.appendChild(cmdMenu);

    const r = cmdBtn.getBoundingClientRect();
    cmdMenu.style.top = (r.top - cmdMenu.offsetHeight - 4) + "px";
    cmdMenu.style.left = Math.max(8, r.left + r.width - cmdMenu.offsetWidth) + "px";
  });
}

connect();

// Load notes script dynamically to run on all pages
(() => {
  const s = document.createElement("script");
  s.src = "/static/notes.js?v=" + Date.now();
  document.body.appendChild(s);
})();
