// GENREG front-end: tabbed xterm.js terminals bridged to real PTYs over a WebSocket.

const tabsEl = document.getElementById("tabs");
const panesEl = document.getElementById("panes");
const connDot = document.getElementById("conn-dot");
const connText = document.getElementById("conn-text");

const terms = new Map();   // id -> {id, title, alive, term, fit, paneEl, tabEl}
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
  tabEl.innerHTML = `<span class="title"></span><button class="close" title="Close">×</button>`;
  tabEl.addEventListener("click", (e) => {
    if (e.target.classList.contains("close")) { send({ op: "close", id: meta.id }); e.stopPropagation(); return; }
    setActive(meta.id);
  });
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
  t.tabEl.querySelector(".title").textContent = t.title;
  t.tabEl.classList.toggle("dead", !t.alive);
}

function setActive(id) {
  activeId = id;
  rememberActiveTab(id);
  for (const t of terms.values()) {
    const on = t.id === id;
    t.tabEl.classList.toggle("active", on);
    t.paneEl.classList.toggle("active", on);
  }
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
        const t = ensureTerm(meta);
        t.term.reset();
        if (meta.data) t.term.write(meta.data);
      }
      // Restore this page's remembered tab (if it still exists).
      if (want != null && want !== activeId && terms.has(want)) setActive(want);
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
      return;
    }
    case "terminal_closed": removeTerm(ev.id); return;
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
        if (/PS [^>\n]*> ?$/.test(clean.trimEnd())) launchClaude(t);
      }
      break;
    case "clear": t.term.reset(); break;
  }
}

// Type the claude command into a ready shell: Escape first (PSReadLine
// RevertLine wipes any stray escape-reply characters), then the command.
function launchClaude(t) {
  if (!t.waitPrompt) return;
  t.waitPrompt = false;
  clearTimeout(t.promptTimer);
  send({ op: "input", id: t.id, data: "\x1b" });
  setTimeout(() => send({ op: "input", id: t.id, data: "claude --dangerously-skip-permissions\r" }), 150);
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
  // (documentation/changelogs/CHANGELOG_<X>.md), toggleable to the main log
  const PROJECT = { "/": "BUILD", "/tree": "TREE", "/diff": "DIFFEVO",
                    "/animation": "ANIMATION", "/i2": "I2" }[location.pathname] || null;
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
      .catch(() => (body.textContent = "Failed to load changelog."));
  };
  const open = () => {
    overlay.hidden = false;
    load();
  };
  const close = () => (overlay.hidden = true);
  btn.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  overlay.addEventListener("click", (e) => e.target === overlay && close());
  document.addEventListener("keydown", (e) => e.key === "Escape" && !overlay.hidden && close());
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

connect();
