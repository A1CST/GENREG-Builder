// GENREG config persistence: remember every input/select/textarea with an id
// across page loads (per page, localStorage), so leaving a page never resets
// its configuration sidebar. The Build page keeps its own equivalent inside
// training.js; this script covers the other project pages (Tree LM, DiffEvo).
// Load AFTER the page's own scripts so restore-dispatched events re-sync
// dependent UI (field visibility, greyed inputs, etc.).
(() => {
  const KEY = "genreg_cfg:" + location.pathname;
  const SKIP_TYPES = new Set(["button", "submit", "file", "password", "hidden"]);

  function controls() {
    return Array.from(document.querySelectorAll("input[id], select[id], textarea[id]"))
      .filter((el) => !SKIP_TYPES.has(el.type)
        && !el.closest("#terminal-panel, #agent-panel, #changelog-overlay"));
  }

  let restoring = false;   // don't save half-restored state mid-restore

  function save() {
    if (restoring) return;
    const data = {};
    for (const el of controls()) {
      data[el.id] = el.type === "checkbox" || el.type === "radio" ? el.checked : el.value;
    }
    try { localStorage.setItem(KEY, JSON.stringify(data)); } catch (_) {}
  }

  function fire(el) {
    // both events: some handlers listen to 'input', others to 'change'
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function restore() {
    let data;
    try { data = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (_) { return; }
    if (!data) return;
    restoring = true;
    const pendingSelects = new Map();   // options populated async (e.g. encoder list)
    for (const el of controls()) {
      if (!(el.id in data)) continue;
      const v = data[el.id];
      if (el.type === "checkbox" || el.type === "radio") {
        el.checked = !!v;
      } else if (el.tagName === "SELECT"
                 && !Array.from(el.options).some((o) => o.value === String(v))) {
        pendingSelects.set(el, String(v));   // value not offered yet — wait for options
        continue;
      } else {
        el.value = v;
      }
      fire(el);
    }
    // re-apply saved select values once their options arrive over the socket
    if (pendingSelects.size) {
      const mo = new MutationObserver(() => {
        for (const [el, v] of Array.from(pendingSelects)) {
          if (Array.from(el.options).some((o) => o.value === v)) {
            el.value = v;
            fire(el);
            pendingSelects.delete(el);
          }
        }
        if (!pendingSelects.size) mo.disconnect();
      });
      for (const el of pendingSelects.keys()) mo.observe(el, { childList: true });
    }
    restoring = false;
  }

  // save on any edit (delegated, capture — catches controls added later too)
  document.addEventListener("input", (e) => { if (e.target && e.target.id) save(); }, true);
  document.addEventListener("change", (e) => { if (e.target && e.target.id) save(); }, true);

  restore();
})();
