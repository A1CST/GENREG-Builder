/* progress.js — renders the /progress dashboard from /api/progress/data.
   All charts are hand-rolled inline SVG (no external libraries). */
(function () {
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  let DATA = null;
  const hiddenProj = new Set();   // project keys toggled off in the line chart
  let idxPD = new Map();          // "project|date" -> [entries]
  let idxD = new Map();           // "date"         -> [entries]
  let impactMeta = {};            // impact key -> {label,color}
  let projMeta = {};              // project key -> label

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // strip lightweight markdown (bold / links / code / file refs) for display
  const mdClean = (s) => esc(s)
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1");

  const MEANINGS = {
    discovery: "New mechanism or computational primitive",
    validation: "Independent experiment confirming it",
    refutation: "Invalidated a previous hypothesis",
    architecture: "Structural redesign",
    engineering: "Implementation / refactor",
    documentation: "Paper / page updates",
    maintenance: "Cleanup, deployment",
  };
  const DISPLAY_ORDER = ["discovery", "validation", "refutation", "architecture",
                         "engineering", "documentation", "maintenance"];

  const G = { W: 960, H: 380, L: 46, R: 18, T: 16, B: 46 };

  const fmtDate = (s) => s.slice(5).replace("-", "/");
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const fmtVal = (g, v) => g.fmt === "pct" ? (v * 100).toFixed(2) + "%" : String(v);

  fetch("/api/progress/data", { cache: "no-store" })
    .then((r) => r.json())
    .then((d) => {
      if (d.error) {
        $("page-main").insertAdjacentHTML("afterbegin",
          `<div class="pg-card" style="color:#ff7b72">Failed to load: ${d.error}</div>`);
        return;
      }
      DATA = d;
      buildIndex();
      renderStats();
      renderGoals();
      renderLine();
      renderImpact();
      renderComposition();
      renderTax();
      renderReadout();
      wireInteractions();
    });

  function buildIndex() {
    idxPD = new Map();
    idxD = new Map();
    (DATA.entries || []).forEach((e) => {
      const k = e.project + "|" + e.date;
      (idxPD.get(k) || idxPD.set(k, []).get(k)).push(e);
      (idxD.get(e.date) || idxD.set(e.date, []).get(e.date)).push(e);
    });
    impactMeta = {};
    DATA.impact_levels.forEach((l) => { impactMeta[l.key] = { label: l.label, color: l.color }; });
    projMeta = {};
    DATA.projects.forEach((p) => { projMeta[p.key] = p.label; });
  }

  // ── summary stats ──────────────────────────────────────────────────────
  function renderStats() {
    const days = DATA.dates.length;
    const totW = DATA.impact_daily.reduce((a, r) => a + r.weighted, 0);
    const stats = [
      [DATA.total_entries, "changelog entries"],
      [days, "active days"],
      [(DATA.total_entries / days).toFixed(1), "entries / day"],
      [totW.toFixed(0), "total weighted score"],
      [(totW / days).toFixed(1), "weighted / day"],
    ];
    $("pg-stats").innerHTML = stats.map(([n, l]) =>
      `<div class="pg-stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");
  }

  // ── goal cards ─────────────────────────────────────────────────────────
  function renderGoals() {
    const box = $("pg-goals");
    if (!DATA.goals.length) { box.innerHTML = "<div class='pg-sub'>No goals defined.</div>"; return; }
    box.innerHTML = DATA.goals.map((g) => {
      const frac = clamp01(g.current / g.target);
      const done = g.status === "complete" || g.current >= g.target;
      const col = done ? "#3fb950" : g.color;
      return `<div class="pg-goal">
        <div class="gh">
          <span class="gname">${g.label}</span>
          <span class="gpct" style="color:${col}">${(frac * 100).toFixed(0)}%</span>
        </div>
        <div class="gmetric">${g.metric}</div>
        <div class="pg-bar"><div style="width:${(frac * 100).toFixed(1)}%;background:${col}"></div></div>
        <div class="gvals">
          <span>now ${fmtVal(g, g.current)}</span>
          <span>goal ${fmtVal(g, g.target)}</span>
        </div>
        <div style="margin-top:8px"><span class="pg-badge ${done ? "done" : "active"}">${done ? "complete" : "active"}</span></div>
        <div class="gnote">${g.note || ""}</div>
      </div>`;
    }).join("");
  }

  // ── helpers for axes ───────────────────────────────────────────────────
  function niceTicks(max, n) {
    const step = Math.max(1, Math.ceil(max / n));
    const out = [];
    for (let v = 0; v <= max + 0.001; v += step) out.push(v);
    return out;
  }
  function axis(maxY, label) {
    const innerH = G.H - G.T - G.B;
    const Y = (v) => G.T + innerH * (1 - v / maxY);
    let s = "";
    niceTicks(maxY, 5).forEach((v) => {
      const y = Y(v).toFixed(1);
      s += `<line x1="${G.L}" y1="${y}" x2="${G.W - G.R}" y2="${y}" stroke="#1c232c" stroke-width="1"/>`;
      s += `<text x="${G.L - 6}" y="${y}" fill="#6e7681" font-size="10" text-anchor="end" dominant-baseline="middle">${v}</text>`;
    });
    if (label) s += `<text x="12" y="${G.T + innerH / 2}" fill="#8b95a1" font-size="10" text-anchor="middle" transform="rotate(-90 12 ${G.T + innerH / 2})">${label}</text>`;
    return { Y, svg: s };
  }
  function xLabels(dates) {
    const n = dates.length;
    const innerW = G.W - G.L - G.R;
    const X = (i) => G.L + (n <= 1 ? innerW / 2 : innerW * i / (n - 1));
    let s = "";
    dates.forEach((d, i) => {
      s += `<text x="${X(i).toFixed(1)}" y="${G.H - G.B + 16}" fill="#6e7681" font-size="9.5" text-anchor="middle">${fmtDate(d)}</text>`;
    });
    return { X, svg: s };
  }

  // ── multi-line: activity per project per day ───────────────────────────
  function renderLine() {
    const projs = DATA.projects.filter((p) => p.total > 0);
    const visible = projs.filter((p) => !hiddenProj.has(p.key));
    let maxY = 1;
    DATA.daily.forEach((row) => visible.forEach((p) => { maxY = Math.max(maxY, row[p.key] || 0); }));
    const A = axis(maxY, "entries");
    const XL = xLabels(DATA.dates);

    let lines = "";
    visible.forEach((p) => {
      const pts = DATA.daily.map((row, i) => `${XL.X(i).toFixed(1)},${A.Y(row[p.key] || 0).toFixed(1)}`).join(" ");
      lines += `<polyline points="${pts}" fill="none" stroke="${p.color}" stroke-width="2" stroke-linejoin="round" opacity="0.92"/>`;
      DATA.daily.forEach((row, i) => {
        const v = row[p.key] || 0;
        if (v > 0) lines += `<circle cx="${XL.X(i).toFixed(1)}" cy="${A.Y(v).toFixed(1)}" r="4" fill="${p.color}" data-kind="line" data-proj="${p.key}" data-date="${DATA.dates[i]}"/>`;
      });
    });
    $("pg-line").innerHTML = A.svg + XL.svg + lines;

    $("pg-line-legend").innerHTML = projs.map((p) =>
      `<span class="pg-leg ${hiddenProj.has(p.key) ? "off" : ""}" data-key="${p.key}">
        <span class="sw" style="background:${p.color}"></span>${p.label} <span class="cnt">${p.total}</span></span>`).join("");
    $("pg-line-legend").querySelectorAll(".pg-leg").forEach((el) => {
      el.addEventListener("click", () => {
        const k = el.dataset.key;
        if (hiddenProj.has(k)) hiddenProj.delete(k); else hiddenProj.add(k);
        renderLine();
      });
    });
  }

  // ── stacked impact bars + weighted line ────────────────────────────────
  function renderImpact() {
    const levels = DATA.impact_levels;
    const n = DATA.dates.length;
    const innerW = G.W - G.L - G.R, innerH = G.H - G.T - G.B;
    let maxCount = 1, maxW = 1;
    DATA.impact_daily.forEach((r) => { maxCount = Math.max(maxCount, r.count); maxW = Math.max(maxW, r.weighted); });

    const A = axis(maxCount, "entries");
    const band = (i) => G.L + innerW * (i + 0.5) / n;
    const barW = Math.min(innerW / n * 0.62, 46);
    const Yc = (v) => G.T + innerH * (1 - v / maxCount);
    const Yw = (v) => G.T + innerH * (1 - v / maxW);

    let bars = "";
    DATA.impact_daily.forEach((r, i) => {
      let yTop = G.H - G.B;
      levels.forEach((lv) => {
        const c = r[lv.key] || 0;
        if (!c) return;
        const h = innerH * c / maxCount;
        yTop -= h;
        bars += `<rect x="${(band(i) - barW / 2).toFixed(1)}" y="${yTop.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" fill="${lv.color}" opacity="0.9"><title>${lv.label} · ${fmtDate(DATA.dates[i])} · ${c}</title></rect>`;
      });
    });

    // right axis (weighted) + weighted line
    let rax = "";
    niceTicks(maxW, 5).forEach((v) => {
      rax += `<text x="${G.W - G.R + 6}" y="${Yw(v).toFixed(1)}" fill="#e3b341" font-size="10" text-anchor="start" dominant-baseline="middle">${v}</text>`;
    });
    const wpts = DATA.impact_daily.map((r, i) => `${band(i).toFixed(1)},${Yw(r.weighted).toFixed(1)}`).join(" ");
    let wline = `<polyline points="${wpts}" fill="none" stroke="#e3b341" stroke-width="2.2"/>`;
    DATA.impact_daily.forEach((r, i) => {
      wline += `<circle cx="${band(i).toFixed(1)}" cy="${Yw(r.weighted).toFixed(1)}" r="3.8" fill="#0d1117" stroke="#e3b341" stroke-width="2" data-kind="date" data-date="${DATA.dates[i]}"/>`;
    });

    const XL = xLabels(DATA.dates);
    $("pg-impact").innerHTML = A.svg + rax + XL.svg + bars + wline;

    $("pg-impact-legend").innerHTML =
      DISPLAY_ORDER.map((k) => {
        const lv = levels.find((l) => l.key === k);
        return `<span class="pg-leg"><span class="sw" style="background:${lv.color}"></span>${lv.label} <span class="cnt">×${lv.weight}</span></span>`;
      }).join("") +
      `<span class="pg-leg"><span class="sw" style="background:#e3b341;height:3px;border-radius:2px"></span>weighted score (right axis)</span>`;
  }

  // ── per-project impact composition ─────────────────────────────────────
  function renderComposition() {
    const levels = DATA.impact_levels;
    const projs = DATA.projects.filter((p) => p.total > 0)
      .sort((a, b) => b.weighted - a.weighted);
    const maxW = Math.max(...projs.map((p) => p.weighted), 1);
    $("pg-composition").innerHTML = projs.map((p) => {
      const segs = DISPLAY_ORDER.map((k) => {
        const lv = levels.find((l) => l.key === k);
        const c = (p.impact && p.impact[k]) || 0;
        if (!c) return "";
        const w = (c / p.total * 100).toFixed(2);
        return `<span style="width:${w}%;background:${lv.color}" title="${lv.label}: ${c}"></span>`;
      }).join("");
      const barLen = (p.weighted / maxW * 100).toFixed(1);
      return `<div class="pg-comp-row">
        <div class="lab">${p.label}</div>
        <div class="pg-comp-bar">${segs}</div>
        <div class="wsc">${p.weighted} pts · ${p.total}e</div>
      </div>
      <div class="pg-comp-row" style="margin-top:-4px;margin-bottom:12px">
        <div class="lab"></div>
        <div style="flex:1;height:4px;background:#10141c;border-radius:2px"><div style="width:${barLen}%;height:100%;background:${p.color};border-radius:2px"></div></div>
        <div class="wsc"></div>
      </div>`;
    }).join("");
  }

  // ── taxonomy table ─────────────────────────────────────────────────────
  function renderTax() {
    const tbody = $("pg-tax").querySelector("tbody");
    tbody.innerHTML = DISPLAY_ORDER.map((k) => {
      const lv = DATA.impact_levels.find((l) => l.key === k);
      return `<tr>
        <td><span class="pg-dot" style="background:${lv.color}"></span>${lv.label}</td>
        <td style="color:#8b95a1">${MEANINGS[k]}</td>
        <td class="num">${lv.weight}</td>
        <td class="num">${lv.total}</td>
      </tr>`;
    }).join("");
  }

  // ── computed read-out prose ────────────────────────────────────────────
  function renderReadout() {
    const days = DATA.dates.length;
    const lv = (k) => DATA.impact_levels.find((l) => l.key === k).total;
    const sci = lv("discovery") + lv("validation") + lv("refutation");
    const low = lv("engineering") + lv("documentation") + lv("maintenance");
    const sciPct = (sci / DATA.total_entries * 100).toFixed(0);
    const lowPct = (low / DATA.total_entries * 100).toFixed(0);

    const byCount = [...DATA.projects].filter((p) => p.total > 0).sort((a, b) => b.total - a.total);
    const byWeight = [...DATA.projects].filter((p) => p.total > 0).sort((a, b) => b.weighted - a.weighted);

    const busiestDay = [...DATA.impact_daily].sort((a, b) => b.count - a.count)[0];
    const heaviestDay = [...DATA.impact_daily].sort((a, b) => b.weighted - a.weighted)[0];

    const rankShift = byCount[0].key !== byWeight[0].key
      ? `By raw count <b>${byCount[0].label}</b> leads; by weighted impact the top is <b>${byWeight[0].label}</b> — the ranking changes when you weight for advancement.`
      : `<b>${byCount[0].label}</b> leads on both raw count and weighted impact.`;

    const dayShift = busiestDay.date !== heaviestDay.date
      ? `Your busiest day (<b>${fmtDate(busiestDay.date)}</b>, ${busiestDay.count} entries) is <b>not</b> your highest-advancement day (<b>${fmtDate(heaviestDay.date)}</b>, weighted ${heaviestDay.weighted}) — proof that volume and progress can diverge.`
      : `Your busiest day (<b>${fmtDate(busiestDay.date)}</b>) was also your highest-advancement day.`;

    $("pg-readout").innerHTML = `
      <p><b>${sciPct}%</b> of entries carry scientific signal (discovery + validation + refutation);
      <b>${lowPct}%</b> are engineering, documentation, or maintenance. A raw entry count would treat both alike —
      the weighted view above does not.</p>
      <p>${rankShift}</p>
      <p>${dayShift}</p>
      <p style="color:#6e7681;font-size:12px">Classification is an automated keyword heuristic over each entry's text —
      good for direction, not a substitute for judgement. Tune the level regexes and weights in
      <code>progress_service.py</code>, and the goal numbers in <code>progress_data/goals.json</code>.</p>`;
  }

  // ── hover tooltip + click-to-open modal on the chart dots ──────────────
  function entriesFor(c) {
    if (c.dataset.kind === "line")
      return { list: idxPD.get(c.dataset.proj + "|" + c.dataset.date) || [],
               head: `${projMeta[c.dataset.proj] || c.dataset.proj} · ${fmtDate(c.dataset.date)}` };
    return { list: idxD.get(c.dataset.date) || [],
             head: `${fmtDate(c.dataset.date)}` };
  }

  function positionTip(tip, x, y) {
    const pad = 16, r = tip.getBoundingClientRect();
    let left = x + pad, top = y + pad;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - pad;   // flip left near right edge
    if (top + r.height > window.innerHeight - 8) top = y - r.height - pad;  // flip up near bottom (taskbar-safe)
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }

  function tipHTML(list, head) {
    const rows = list.slice(0, 6).map((e) => {
      const m = impactMeta[e.impact] || { color: "#8b95a1" };
      return `<div class="ti"><span class="d" style="background:${m.color}"></span><span>${mdClean(e.title)}</span></div>`;
    }).join("");
    const more = list.length > 6 ? `<div class="more">+${list.length - 6} more…</div>` : "";
    const n = list.length;
    return `<div class="th">${head} · ${n} ${n === 1 ? "entry" : "entries"}</div>${rows}${more}<div class="hint">click to open ▸</div>`;
  }

  function openModal(head, list) {
    $("pg-modal-title").textContent = `Changelog — ${head} (${list.length})`;
    $("pg-modal-body").innerHTML = list.map((e) => {
      const m = impactMeta[e.impact] || { label: e.impact, color: "#8b95a1" };
      return `<div class="pg-entry">
        <div class="meta"><span>${e.date}</span><span>·</span><span>${esc(e.author)}</span>
          <span class="ilab" style="background:${m.color}22;color:${m.color};border:1px solid ${m.color}66">${m.label}</span></div>
        <div class="etitle">${mdClean(e.title)}</div>
        <div class="ebody">${mdClean(e.body)}</div>
      </div>`;
    }).join("");
    $("pg-modal").hidden = false;
  }

  function closeModal() { $("pg-modal").hidden = true; }

  function wireInteractions() {
    const tip = $("pg-tip");
    ["pg-line", "pg-impact"].forEach((id) => {
      const svg = $(id);
      svg.addEventListener("mousemove", (e) => {
        const c = e.target.closest("circle[data-kind]");
        if (!c) { tip.style.display = "none"; return; }
        const { list, head } = entriesFor(c);
        tip.innerHTML = tipHTML(list, head);
        tip.style.display = "block";
        positionTip(tip, e.clientX, e.clientY);
      });
      svg.addEventListener("mouseleave", () => { tip.style.display = "none"; });
      svg.addEventListener("click", (e) => {
        const c = e.target.closest("circle[data-kind]");
        if (!c) return;
        tip.style.display = "none";
        const { list, head } = entriesFor(c);
        if (list.length) openModal(head, list);
      });
    });
    $("pg-modal-close").addEventListener("click", closeModal);
    $("pg-modal").addEventListener("click", (e) => { if (e.target.id === "pg-modal") closeModal(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  }
})();
