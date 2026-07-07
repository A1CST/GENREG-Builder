/* GENREG — documentation browser (/docs).
 *
 * Mirrors the runs dashboard layout: type tabs + file list on the left,
 * document viewer on the right. Markdown is rendered client-side by the
 * small renderer below (no external libs); PDFs use the browser's native
 * viewer via <embed>; JSON pretty-prints; anything else gets a download
 * link. Deep-linkable via /docs#<path>.
 */
(() => {
  "use strict";

  const $list = document.getElementById("doc-list");
  const $tabs = document.getElementById("type-tabs");
  const $detail = document.getElementById("doc-detail");
  const $filter = document.getElementById("docs-filter");
  const $sort = document.getElementById("docs-sort");
  const $favToggle = document.getElementById("docs-fav-toggle");
  const $archToggle = document.getElementById("docs-arch-toggle");

  let files = [];            // [{path,name,ext,size,mtime}]
  let activeTab = "all";     // all | md | pdf | json | other
  let selected = null;       // path of the open doc
  let sortMode = "name";     // name | recent
  let favOnly = false;
  let archView = false;

  const TAB_ORDER = ["all", "md", "pdf", "json", "other"];
  const TAB_LABEL = { all: "All", md: "Markdown", pdf: "PDF", json: "JSON", other: "Other" };

  // ------------------------------------------------------ per-doc metadata
  // Favorite/archive flags live client-side (this is a single-user local
  // tool) keyed by doc path, so they survive refreshes without a server change.
  const META_KEY = "genreg_docs_meta";
  let meta = {};
  try { meta = JSON.parse(localStorage.getItem(META_KEY) || "{}"); } catch (e) { meta = {}; }
  const getMeta = (p) => meta[p] || { fav: false, arch: false };
  const setMeta = (p, patch) => {
    meta[p] = { ...getMeta(p), ...patch };
    try { localStorage.setItem(META_KEY, JSON.stringify(meta)); } catch (e) { /* ignore */ }
  };
  const NEW_WINDOW_SECS = 3 * 86400;   // mtime within this window gets a NEW badge
  const isNew = (f) => (Date.now() / 1000 - f.mtime) < NEW_WINDOW_SECS;

  const kind = (f) => {
    if (f.ext === "md" || f.ext === "markdown" || f.ext === "txt") return "md";
    if (f.ext === "pdf") return "pdf";
    if (f.ext === "json") return "json";
    return "other";
  };

  const fmtSize = (n) => {
    if (n >= 1048576) return (n / 1048576).toFixed(1) + " MB";
    if (n >= 1024) return (n / 1024).toFixed(1) + " KB";
    return n + " B";
  };
  const fmtDate = (t) => new Date(t * 1000).toLocaleDateString();
  const fileURL = (p) => "/api/docs/file/" + p.split("/").map(encodeURIComponent).join("/");

  // ------------------------------------------------------------------ list
  async function load() {
    try {
      const r = await fetch("/api/docs");
      files = await r.json();
    } catch (e) {
      $list.innerHTML = '<div class="tree-empty">Could not load /api/docs — is the server current?</div>';
      return;
    }
    renderTabs();
    renderList();
    if (location.hash.length > 1) {
      const p = decodeURIComponent(location.hash.slice(1));
      if (files.some((f) => f.path === p)) open(p);
    }
  }

  // Files in the current archive/favorites scope, before type-tab + search
  // narrow it further — tab counts and the list are both built from this.
  function scopedFiles() {
    return files.filter((f) => {
      const m = getMeta(f.path);
      if (archView ? !m.arch : m.arch) return false;
      if (favOnly && !m.fav) return false;
      return true;
    });
  }

  function renderTabs() {
    const scope = scopedFiles();
    const counts = { all: scope.length, md: 0, pdf: 0, json: 0, other: 0 };
    scope.forEach((f) => counts[kind(f)]++);
    $tabs.innerHTML = "";
    TAB_ORDER.forEach((t) => {
      if (t !== "all" && !counts[t]) return;
      const b = document.createElement("button");
      b.className = "env-tab" + (t === activeTab ? " active" : "");
      b.textContent = `${TAB_LABEL[t]} (${counts[t]})`;
      b.onclick = () => { activeTab = t; renderTabs(); renderList(); };
      $tabs.appendChild(b);
    });
  }

  function visibleFiles() {
    const q = ($filter.value || "").trim().toLowerCase();
    const vis = scopedFiles().filter((f) =>
      (activeTab === "all" || kind(f) === activeTab) &&
      (!q || f.name.toLowerCase().includes(q)));
    vis.sort(sortMode === "recent" ? (a, b) => b.mtime - a.mtime
                                    : (a, b) => a.name.localeCompare(b.name));
    return vis;
  }

  function renderList() {
    const vis = visibleFiles();
    $list.innerHTML = "";
    if (!vis.length) {
      $list.innerHTML = `<div class="tree-empty">No ${archView ? "archived " : ""}documents match.</div>`;
      return;
    }
    const root = document.createElement("div");
    root.className = "tree-root";
    root.textContent = archView ? "documentation/ — archived" : "documentation/";
    $list.appendChild(root);
    const wrap = document.createElement("div");
    wrap.className = "tree-branches";
    vis.forEach((f) => {
      const m = getMeta(f.path);
      const n = document.createElement("div");
      n.className = "tree-node" + (f.path === selected ? " selected" : "");
      n.innerHTML =
        `<span class="doc-ext ${kind(f)}">${f.ext || "?"}</span>` +
        `<div class="tn-main"><div class="tn-title"></div>` +
        `<div class="tn-sub">${fmtSize(f.size)} · ${fmtDate(f.mtime)}` +
        `${isNew(f) ? ' · <span class="doc-new">NEW</span>' : ""}</div></div>` +
        `<div class="tn-actions">` +
        `<button class="tn-act${m.fav ? " active" : ""}" data-act="fav" title="Favorite">${m.fav ? "★" : "☆"}</button>` +
        `<button class="tn-act${m.arch ? " active" : ""}" data-act="arch" title="${m.arch ? "Restore" : "Archive"}">${m.arch ? "Restore" : "Archive"}</button>` +
        `</div>`;
      n.querySelector(".tn-title").textContent = f.name;
      n.onclick = () => open(f.path);
      n.querySelector('[data-act="fav"]').onclick = (e) => {
        e.stopPropagation();
        setMeta(f.path, { fav: !getMeta(f.path).fav });
        renderTabs(); renderList();
      };
      n.querySelector('[data-act="arch"]').onclick = (e) => {
        e.stopPropagation();
        setMeta(f.path, { arch: !getMeta(f.path).arch });
        renderTabs(); renderList();
      };
      wrap.appendChild(n);
    });
    $list.appendChild(wrap);
  }

  // ---------------------------------------------------------------- viewer
  async function open(path) {
    selected = path;
    history.replaceState(null, "", "#" + encodeURIComponent(path));
    renderList();
    const f = files.find((x) => x.path === path);
    if (!f) return;
    const k = kind(f);
    const url = fileURL(path);

    $detail.innerHTML =
      `<div class="detail-head"><div><div class="dh-title"></div>` +
      `<div class="dh-sub">${fmtSize(f.size)} · modified ${fmtDate(f.mtime)}</div></div>` +
      `<div class="dh-badges"><span class="ckpt">${(f.ext || "file").toUpperCase()}</span>` +
      `<a class="runs-btn link" href="${url}" target="_blank" rel="noopener">⤓ Open / download</a></div></div>` +
      `<div id="doc-body" class="doc-body"><div class="detail-empty">Loading…</div></div>`;
    $detail.querySelector(".dh-title").textContent = f.name;
    const $body = document.getElementById("doc-body");

    try {
      if (k === "pdf") {
        $body.innerHTML =
          `<embed class="doc-pdf" src="${url}" type="application/pdf" />` +
          `<div class="doc-note">If the viewer stays blank, use “Open / download” above.</div>`;
      } else if (k === "json") {
        const txt = await (await fetch(url)).text();
        let pretty = txt;
        try { pretty = JSON.stringify(JSON.parse(txt), null, 2); } catch (e) { /* show raw */ }
        const pre = document.createElement("pre");
        pre.className = "doc-pre";
        pre.textContent = pretty;
        $body.innerHTML = "";
        $body.appendChild(pre);
      } else if (k === "md") {
        const txt = await (await fetch(url)).text();
        $body.innerHTML = `<div class="md-body">${renderMarkdown(txt)}</div>`;
      } else {
        $body.innerHTML =
          `<div class="detail-empty">No inline viewer for .${f.ext} files — ` +
          `use <b>⤓ Open / download</b> above to save it.</div>`;
      }
    } catch (e) {
      $body.innerHTML = `<div class="detail-empty">Failed to load: ${String(e)}</div>`;
    }
  }

  // ------------------------------------------------------- markdown renderer
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
      .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener">🖼 $1</a>')
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  function renderMarkdown(src) {
    const lines = src.replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;
    let list = null;                     // "ul" | "ol" while inside a list
    const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };

    while (i < lines.length) {
      const raw = lines[i];
      const line = esc(raw);

      // fenced code block
      const fence = raw.match(/^\s*```(\w*)/);
      if (fence) {
        closeList();
        const buf = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
        i++;                              // skip closing fence
        out.push(`<pre class="doc-pre"><code>${esc(buf.join("\n"))}</code></pre>`);
        continue;
      }

      // table (header row + separator row)
      if (/^\s*\|/.test(raw) && i + 1 < lines.length && /^\s*\|[\s\-:|]+\|?\s*$/.test(lines[i + 1])) {
        closeList();
        const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(esc(c.trim())));
        const head = cells(raw);
        i += 2;
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
        out.push('<div class="doc-table-wrap"><table class="doc-table"><thead><tr>' +
          head.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>" +
          rows.map((r) => "<tr>" + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") +
          "</tbody></table></div>");
        continue;
      }

      // heading
      const h = raw.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        closeList();
        const lvl = h[1].length;
        out.push(`<h${lvl}>${inline(esc(h[2]))}</h${lvl}>`);
        i++;
        continue;
      }

      // horizontal rule
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(raw)) { closeList(); out.push("<hr/>"); i++; continue; }

      // blockquote
      if (/^\s*>\s?/.test(raw)) {
        closeList();
        const buf = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i]))
          buf.push(inline(esc(lines[i++].replace(/^\s*>\s?/, ""))));
        out.push(`<blockquote>${buf.join("<br/>")}</blockquote>`);
        continue;
      }

      // list item (nesting flattened; indented items get a css class)
      const li = raw.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
      if (li) {
        const type = /^[-*+]$/.test(li[2]) ? "ul" : "ol";
        if (list !== type) { closeList(); out.push(`<${type}>`); list = type; }
        let text = li[3];
        // absorb hanging-indent continuation lines into this item
        while (i + 1 < lines.length && /^\s{2,}\S/.test(lines[i + 1]) &&
               !/^(\s*([-*+]|\d+[.)])\s|\s*```)/.test(lines[i + 1]))
          text += " " + lines[++i].trim();
        const cls = li[1].length >= 2 ? ' class="li-nest"' : "";
        out.push(`<li${cls}>${inline(esc(text))}</li>`);
        i++;
        continue;
      }

      // blank line
      if (!raw.trim()) { closeList(); i++; continue; }

      // paragraph: gather consecutive plain lines
      closeList();
      const buf = [inline(line)];
      i++;
      while (i < lines.length && lines[i].trim() &&
             !/^(\s*([-*+]|\d+[.)])\s|#{1,6}\s|\s*```|\s*>|\s*\|)/.test(lines[i]))
        buf.push(inline(esc(lines[i++])));
      out.push(`<p>${buf.join(" ")}</p>`);
    }
    closeList();
    return out.join("\n");
  }

  // ---------------------------------------------------------------- wiring
  document.getElementById("docs-refresh").onclick = load;
  $filter.addEventListener("input", renderList);
  $sort.addEventListener("change", () => { sortMode = $sort.value; renderList(); });
  $favToggle.addEventListener("click", () => {
    favOnly = !favOnly;
    $favToggle.classList.toggle("active", favOnly);
    renderTabs(); renderList();
  });
  $archToggle.addEventListener("click", () => {
    archView = !archView;
    $archToggle.classList.toggle("active", archView);
    renderTabs(); renderList();
  });
  window.addEventListener("hashchange", () => {
    const p = decodeURIComponent(location.hash.slice(1));
    if (p && p !== selected && files.some((f) => f.path === p)) open(p);
  });

  load();
})();
