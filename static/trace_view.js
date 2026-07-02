/* Shared renderer for tree-of-models routing traces.
 * Used by the /tree routing inspector and the /runs "Saved traces" panel:
 *   TraceView.mount(stripEl, detailEl, trace)
 * renders the clickable byte strip (with a JSON export button) into stripEl
 * and the per-step decision path (routers + leaf) into detailEl.
 */
window.TraceView = (() => {
  "use strict";

  const charDisp = (b) => {
    if (b === 32) return "␣";
    if (b === 10) return "\\n";
    if (b === 13) return "\\r";
    if (b === 9) return "\\t";
    if (b >= 33 && b < 127) return String.fromCharCode(b);
    return "\\x" + b.toString(16).padStart(2, "0");
  };

  function barRow(label, frac, value, chosen) {
    const row = document.createElement("div");
    row.className = "tlm-bar-row" + (chosen ? " chosen" : "");
    const l = document.createElement("span");
    l.className = "tlm-bar-label";
    l.textContent = label;
    const bar = document.createElement("div");
    bar.className = "tlm-bar";
    const fill = document.createElement("div");
    fill.className = "tlm-bar-fill";
    fill.style.width = Math.max(frac * 100, 1) + "%";
    bar.appendChild(fill);
    const v = document.createElement("span");
    v.className = "tlm-bar-value";
    v.textContent = value + (chosen ? " ✓" : "");
    row.appendChild(l); row.appendChild(bar); row.appendChild(v);
    return row;
  }

  function renderDetail(box, trace, idx) {
    box.textContent = "";
    const st = trace.steps && trace.steps[idx];
    if (!st) return;

    const head = document.createElement("div");
    head.className = "tlm-tr-head";
    head.innerHTML = `<b>step ${st.i}</b> — context ` +
      `<code></code> → produced <b class="tlm-tr-byte"></b>`;
    head.querySelector("code").textContent = "“" + st.context + "”";
    head.querySelector(".tlm-tr-byte").textContent =
      `${charDisp(st.byte)} (byte ${st.byte})`;
    box.appendChild(head);

    // one block per router on the path, top of the tree first
    st.path.forEach((hop) => {
      const div = document.createElement("div");
      div.className = "tlm-hop";
      const sorted = hop.scores.slice().sort((a, b) => b - a);
      const margin = sorted.length > 1 ? (sorted[0] - sorted[1]) : 0;
      const h = document.createElement("div");
      h.className = "tlm-hop-head";
      h.innerHTML = `<b>router ${hop.id}</b> · depth ${hop.depth} · ` +
        `picked child ${hop.chosen} · margin <b>${margin.toFixed(3)}</b>` +
        (margin < 0.05 ? ' <span class="tlm-warn">⚠ near-tie</span>' : "");
      div.appendChild(h);
      const lo = Math.min(...hop.scores), hi = Math.max(...hop.scores);
      hop.scores.forEach((s, c) => {
        const r = hop.children[c];
        div.appendChild(barRow(
          `bytes ${r.t0}–${r.t1 - 1}`,
          hi - lo > 1e-9 ? (s - lo) / (hi - lo) : 1,
          s.toFixed(3),
          c === hop.chosen));
      });
      box.appendChild(div);
    });

    // leaf distribution
    const leaf = st.leaf;
    const div = document.createElement("div");
    div.className = "tlm-hop";
    const h = document.createElement("div");
    h.className = "tlm-hop-head";
    h.innerHTML = `<b>leaf ${leaf.id}</b> · bytes ${leaf.t0}–${leaf.t1 - 1} ` +
      `(${leaf.tokens} token${leaf.tokens > 1 ? "s" : ""}) · top candidates`;
    div.appendChild(h);
    leaf.top.forEach((t) => {
      div.appendChild(barRow(
        `${charDisp(t.byte)} (${t.byte})`,
        t.prob,
        `${(t.prob * 100).toFixed(1)}% · ${t.score.toFixed(3)}`,
        t.byte === leaf.chosen_byte));
    });
    box.appendChild(div);
  }

  function exportJson(trace) {
    const stamp = (trace.created || "trace").replace(/[:T]/g, "-");
    const blob = new Blob([JSON.stringify(trace, null, 2)],
                          { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `tree-trace-${trace.run_id || stamp}-${stamp}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  function mount(stripEl, detailEl, trace) {
    let sel = 0;

    function renderStrip() {
      stripEl.textContent = "";
      stripEl.hidden = !trace;
      if (!trace) return;

      const lbl = document.createElement("span");
      lbl.className = "tlm-strip-prompt";
      lbl.textContent = trace.prompt;
      lbl.title = "prompt (not traced)";
      stripEl.appendChild(lbl);

      trace.steps.forEach((st, i) => {
        const sp = document.createElement("span");
        sp.className = "tlm-step" + (i === sel ? " sel" : "") +
          (st.byte < 33 || st.byte >= 127 ? " ctl" : "");
        sp.textContent = charDisp(st.byte);
        sp.title = `step ${i} · byte ${st.byte} · leaf ${st.leaf.id}`;
        sp.addEventListener("click", () => {
          sel = i;
          renderStrip();
          renderDetail(detailEl, trace, sel);
        });
        stripEl.appendChild(sp);
      });

      const dl = document.createElement("button");
      dl.className = "tlm-export";
      dl.textContent = "⤓ JSON";
      dl.title = "download this trace as JSON";
      dl.addEventListener("click", () => exportJson(trace));
      stripEl.appendChild(dl);
    }

    renderStrip();
    renderDetail(detailEl, trace, sel);
  }

  return { mount, charDisp };
})();
