// GENREG Agent panel: a floating, draggable, minimizable notice feed present
// on every project page (topmost, above page content). AI assistants post
// here (agent_notify.py / POST /api/agent/notices) and training jobs post
// automatically when they finish. Unread notices badge the panel and its
// minimized pill; a notice arriving live pops the panel open (the "alarm").
(() => {
  if (document.getElementById("agent-panel")) return;

  const LS_POS = "genreg_agent_pos", LS_MIN = "genreg_agent_min", LS_SEEN = "genreg_agent_seen";
  const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } };
  const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) {} };

  const panel = document.createElement("div");
  panel.id = "agent-panel";
  panel.className = "agent-panel";
  panel.innerHTML = `
    <div class="ap-head" id="ap-head" title="Drag to move">
      <span class="ap-title">Agent</span>
      <span class="ap-badge" id="ap-badge" hidden></span>
      <button class="ap-btn" id="ap-read" title="Mark all notices read">mark read</button>
      <button class="ap-btn" id="ap-min" title="Minimize to a pill">—</button>
    </div>
    <div class="ap-body" id="ap-body">
      <div class="ap-empty">No notices yet — run alarms, test results, and AI updates land here.</div>
    </div>`;

  const pill = document.createElement("button");
  pill.id = "agent-pill";
  pill.className = "agent-pill";
  pill.title = "Agent panel — click to open";
  pill.innerHTML = `Agent<span class="ap-badge" id="ap-pill-badge" hidden></span>`;

  document.body.appendChild(panel);
  document.body.appendChild(pill);

  // -- position: persisted, clamped on-screen --------------------------------
  const pos = lsGet(LS_POS, null);
  if (pos && Number.isFinite(pos.x) && Number.isFinite(pos.y)) {
    panel.style.left = Math.max(0, Math.min(innerWidth - 90, pos.x)) + "px";
    panel.style.top = Math.max(0, Math.min(innerHeight - 60, pos.y)) + "px";
    panel.style.right = "auto";
  }

  let minimized = !!lsGet(LS_MIN, false);
  function apply() { panel.hidden = minimized; pill.hidden = !minimized; }
  apply();

  // -- drag by the header -----------------------------------------------------
  const head = panel.querySelector("#ap-head");
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

  // -- notices: poll, render, badge -------------------------------------------
  let notices = [];
  let lastTopId = null;     // top id seen by the PREVIOUS poll (null = first poll)
  const seenId = () => Number(lsGet(LS_SEEN, 0)) || 0;

  function render() {
    const body = panel.querySelector("#ap-body");
    body.innerHTML = "";
    if (!notices.length) {
      body.innerHTML = '<div class="ap-empty">No notices yet — run alarms, test results, and AI updates land here.</div>';
      return;
    }
    const s = seenId();
    for (const n of notices) {
      const row = document.createElement("div");
      row.className = `ap-row k-${n.kind || "info"}` + (n.id > s ? " unread" : "");
      row.innerHTML =
        `<div class="ap-row-head"><span class="ap-kind"></span>` +
        `<span class="ap-row-title"></span><span class="ap-time"></span></div>` +
        (n.body ? `<pre class="ap-row-body"></pre>` : "");
      row.querySelector(".ap-kind").textContent = n.kind || "info";
      row.querySelector(".ap-row-title").textContent =
        n.title + (n.source ? ` — ${n.source}` : "");
      row.querySelector(".ap-time").textContent = (n.ts || "").replace("T", " ");
      if (n.body) row.querySelector(".ap-row-body").textContent = n.body;
      if (n.run_id) {           // run notices deep-link to that run on /runs
        row.classList.add("linked");
        row.title = "Open this run on the runs page";
        row.addEventListener("click", () => {
          location.href = "/runs#" + encodeURIComponent(n.run_id);
        });
      }
      body.appendChild(row);
    }
  }

  function setBadges() {
    const unread = notices.filter((n) => n.id > seenId()).length;
    for (const id of ["ap-badge", "ap-pill-badge"]) {
      const b = document.getElementById(id);
      if (!b) continue;
      b.hidden = !unread;
      b.textContent = unread > 99 ? "99+" : String(unread);
    }
    pill.classList.toggle("pulse", unread > 0);
  }

  async function poll() {
    let items = null;
    try { items = await (await fetch("/api/agent/notices")).json(); } catch (_) {}
    if (!Array.isArray(items)) return;
    notices = items;
    const top = notices.length ? Number(notices[0].id) || 0 : 0;
    // The alarm: a notice arriving while this page is open pops the panel up.
    if (lastTopId !== null && top > lastTopId && minimized) {
      minimized = false;
      lsSet(LS_MIN, false);
      apply();
    }
    lastTopId = top;
    render();
    setBadges();
  }

  // -- buttons ------------------------------------------------------------------
  panel.querySelector("#ap-min").addEventListener("click", () => {
    minimized = true; lsSet(LS_MIN, true); apply();
  });
  pill.addEventListener("click", () => {
    minimized = false; lsSet(LS_MIN, false); apply();
  });
  panel.querySelector("#ap-read").addEventListener("click", () => {
    const top = notices.length ? Number(notices[0].id) || 0 : 0;
    lsSet(LS_SEEN, top);
    render();
    setBadges();
  });

  poll();
  setInterval(poll, 8000);
})();
