/* resnet.js — renders the latest gradient-free evolved ResNet result.
   Reads /api/resnet/result (written by resnet_evo.py to F:\Resnet) and draws
   the val-accuracy curve over rounds, the headline test/val stats, the
   reference lines, and what evolution chose (depth / width / stat). No models
   run here — pure results view. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };

  var REFS = [
    { key: "coates_ng",           label: "Coates-Ng hand-crafted", col: "#7d6a2e" },
    { key: "radial_v1_class_tower", label: "radial v1 class tower", col: "#2f4152" },
    { key: "grammar_v2_record",   label: "grammar-v2 record",      col: "#8a5a2e" }
  ];

  function fmt(x) { return (x == null) ? "—" : (x * 100).toFixed(2) + "%"; }

  function hero(d) {
    var box = $("rn-hero");
    var frozen = d.n_frozen || 0;
    var cells = [
      { k: "Test accuracy", v: fmt(d.test_acc), hi: true,
        sub: "touched once · " + (d.n_test || "?") + " test imgs" },
      { k: "Val (final round)", v: fmt(d.val_final), hi: false,
        sub: "held-back 20% split" },
      { k: "Frozen residual nets", v: String(frozen), hi: false,
        sub: "gradient-free features" },
      { k: "Mean residual gain", v: (d.mean_gain != null ? d.mean_gain.toFixed(3) : "—"),
        hi: false, sub: "skip strength evolved" }
    ];
    box.innerHTML = cells.map(function (c) {
      return '<div class="rn-stat"><div class="k">' + c.k + '</div>' +
             '<div class="v' + (c.hi ? ' hi' : '') + '">' + c.v + '</div>' +
             '<div class="sub">' + c.sub + '</div></div>';
    }).join("");
  }

  function curve(d) {
    var cv = $("rn-curve"), ctx = cv.getContext("2d");
    var r = cv.getBoundingClientRect();
    cv.width = Math.floor(r.width) * 2; cv.height = Math.floor(r.height) * 2;
    ctx.clearRect(0, 0, cv.width, cv.height);
    var hist = d.hist || [];
    var W = cv.width, H = cv.height, pad = 44;
    if (!hist.length) { ctx.fillStyle = "#6c7885"; ctx.font = "24px sans-serif";
      ctx.fillText("no history", pad, H / 2); return; }
    var ys = hist.map(function (h) { return h.val_acc; });
    var lo = 0, hi = Math.max.apply(null, ys.concat(0.75)) * 1.02;

    function X(i) { return pad + (W - pad - 14) * (hist.length === 1 ? 0.5 : i / (hist.length - 1)); }
    function Y(v) { return H - pad - (H - pad - 14) * (v - lo) / (hi - lo); }

    // reference lines
    REFS.forEach(function (ref) {
      var v = (d.references || {})[ref.key];
      if (v == null || v < lo || v > hi) return;
      ctx.strokeStyle = ref.col; ctx.lineWidth = 2; ctx.setLineDash([8, 6]);
      ctx.beginPath(); ctx.moveTo(pad, Y(v)); ctx.lineTo(W - 14, Y(v)); ctx.stroke();
      ctx.setLineDash([]);
    });
    // axes
    ctx.strokeStyle = "#232a33"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(pad, 14); ctx.lineTo(pad, H - pad); ctx.lineTo(W - 14, H - pad); ctx.stroke();
    ctx.fillStyle = "#6c7885"; ctx.font = "22px ui-monospace,monospace";
    ctx.fillText((hi * 100).toFixed(0) + "%", 4, 26);
    ctx.fillText((lo * 100).toFixed(0) + "%", 4, H - pad + 6);
    ctx.fillText("round", W / 2 - 24, H - 8);

    // the val curve
    ctx.strokeStyle = "#5fd39a"; ctx.lineWidth = 3;
    ctx.beginPath();
    hist.forEach(function (h, i) { var x = X(i), y = Y(h.val_acc);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();
    ctx.fillStyle = "#5fd39a";
    hist.forEach(function (h, i) { ctx.beginPath();
      ctx.arc(X(i), Y(h.val_acc), 4, 0, 2 * Math.PI); ctx.fill(); });

    $("rn-curve-sub").textContent = "(" + hist.length + " rounds, " +
      (d.smoke ? "SMOKE subset" : (d.n_train || "?") + " train") + ")";
  }

  function refsList(d) {
    var refs = d.references || {};
    var html = REFS.map(function (ref) {
      var v = refs[ref.key];
      return '<div><span class="sw" style="background:' + ref.col + '"></span>' +
             ref.label + ' — <b>' + fmt(v) + '</b></div>';
    }).join("");
    html += '<div><span class="sw" style="background:#5fd39a"></span>' +
            'this run (test) — <b>' + fmt(d.test_acc) + '</b></div>';
    $("rn-refs").innerHTML = html;
  }

  function bars(elId, counts, labelFmt) {
    var el = $(elId);
    if (!counts) { el.innerHTML = '<div class="rn-brow"><span class="lb">—</span></div>'; return; }
    var keys = Object.keys(counts);
    var max = Math.max.apply(null, keys.map(function (k) { return counts[k]; }).concat(1));
    el.innerHTML = keys.map(function (k) {
      var pct = 100 * counts[k] / max;
      return '<div class="rn-brow"><span class="lb">' + labelFmt(k) + '</span>' +
             '<span class="tr"><span class="fl hi" style="width:' + pct + '%"></span></span>' +
             '<span class="vl">' + counts[k] + '</span></div>';
    }).join("");
  }

  function render(d) {
    hero(d); curve(d); refsList(d);
    bars("rn-depth", d.depth_counts, function (k) { return k + " block" + (k === "1" ? "" : "s"); });
    bars("rn-width", d.width_counts, function (k) { return k + " chan"; });
    bars("rn-stat", d.stat_counts, function (k) { return k; });
    var st = $("rn-status");
    st.className = "rn-status";
    st.innerHTML = "loaded <b>" + (d._source || "result") + "</b>" +
      (d.smoke ? " — <b style='color:#d9a441'>SMOKE run</b> (tiny subset, plumbing check only)" : "") +
      " · " + (d.seconds || "?") + "s · phase " + (d.phase || "?") +
      (d._dir ? " · " + d._dir : "");
  }

  function load() {
    var st = $("rn-status");
    st.className = "rn-status"; st.textContent = "loading latest result…";
    fetch("/api/resnet/result").then(function (r) {
      return r.json().then(function (j) { return { ok: r.ok, j: j }; });
    }).then(function (res) {
      if (!res.ok || res.j.error) {
        st.className = "rn-status err";
        st.textContent = res.j.error || "no result yet";
        return;
      }
      render(res.j);
    }).catch(function (e) {
      st.className = "rn-status err"; st.textContent = "load failed: " + e;
    });
  }

  $("rn-reload").addEventListener("click", load);
  window.addEventListener("resize", function () {
    fetch("/api/resnet/result").then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) { if (j && !j.error) { curve(j); } }).catch(function () {});
  });
  load();
})();
