// GENREG shared terminal dock. The build (/) and I2 pages hard-code the dock
// in their templates; every other project page loads this first to inject the
// exact same markup, then app.js (loaded after) drives the tabs/PTYs over the
// shared /ws daemon bridge. Pages that already have #terminal-panel are left
// untouched.
(() => {
  if (document.getElementById("terminal-panel")) return;

  const resizer = document.createElement("div");
  resizer.className = "resizer h";
  resizer.id = "term-resizer";
  resizer.title = "Drag to resize terminal";

  const panel = document.createElement("section");
  panel.className = "terminal-panel";
  panel.id = "terminal-panel";
  panel.innerHTML = `
    <div class="term-header">
      <div class="tabs" id="tabs"></div>
      <button id="new-tab" class="tab-new" title="New terminal">+ New Tab</button>
      <div class="term-actions">
        <span class="dot" id="conn-dot" title="Terminal connection"></span>
        <span id="conn-text" class="term-conn">connecting…</span>
        <button id="btn-changelog" title="View the GENREG changelog">Changelog</button>
        <button id="btn-claude" title="Open a new terminal running claude --dangerously-skip-permissions">Claude</button>
        <button id="btn-clear" title="Clear this terminal">Clear</button>
        <button id="btn-restart" title="Kill and respawn the shell (fresh state)">Restart</button>
        <button id="btn-stop" title="Stop the shell process (halts a running command)">Stop</button>
      </div>
    </div>
    <main id="panes" class="panes"></main>`;

  const modal = document.createElement("div");
  modal.id = "changelog-overlay";
  modal.className = "modal-overlay";
  modal.hidden = true;
  modal.innerHTML = `
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="changelog-title">
      <div class="modal-head">
        <span id="changelog-title">Changelog</span>
        <button id="changelog-close" class="modal-close" title="Close">×</button>
      </div>
      <pre id="changelog-body" class="modal-body">Loading…</pre>
    </div>`;

  // Every host page (.tlm-body / .runs-body) is a 100vh column flex, so
  // appending to <body> stacks the dock under the page content.
  document.body.appendChild(resizer);
  document.body.appendChild(panel);
  document.body.appendChild(modal);

  // -- height: per-page persistence + drag resizer ---------------------------
  const KEY = "genreg_dock_height:" + location.pathname;
  const CLAMP = { min: 120, max: 0.85 };   // max as fraction of viewport height
  const clampH = (h) => Math.min(window.innerHeight * CLAMP.max, Math.max(CLAMP.min, h));

  let saved = null;
  try { saved = parseInt(localStorage.getItem(KEY), 10); } catch (_) {}
  // Default smaller than the build page's 42vh so charts stay usable.
  panel.style.height = clampH(Number.isFinite(saved) && saved > 0 ? saved : 240) + "px";

  const save = () => {
    try { localStorage.setItem(KEY, String(Math.round(panel.getBoundingClientRect().height))); }
    catch (_) {}
  };

  resizer.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    const base = panel.getBoundingClientRect().height;
    const startY = e.clientY;
    resizer.setPointerCapture(e.pointerId);
    resizer.classList.add("dragging");
    const move = (ev) => { panel.style.height = clampH(base - (ev.clientY - startY)) + "px"; };
    const up = () => {
      resizer.releasePointerCapture(e.pointerId);
      resizer.classList.remove("dragging");
      resizer.removeEventListener("pointermove", move);
      resizer.removeEventListener("pointerup", up);
      save();
    };
    resizer.addEventListener("pointermove", move);
    resizer.addEventListener("pointerup", up);
  });
})();
