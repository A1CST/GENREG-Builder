/* radial.js — Radial Map v2 front-end: build the activation-behavior map,
   inspect lenses, run the linear probe. Plain fetch + canvas, no libs. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var mapCv = $("rm-map"), curveCv = $("rm-curve");
  var pts = [], colorBy = "nl", kind = "loops", selected = -1;
  var lastMap = null, lastProbe = null, lastLens = null, lastRot = null;

  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function (r) { return r.json(); });
  }

  function status(msg) { $("rm-status").textContent = msg || ""; }

  /* ---- map rendering ------------------------------------------------- */

  function fit() {
    var r = mapCv.parentElement.getBoundingClientRect();
    mapCv.width = Math.max(50, Math.floor(r.width));
    mapCv.height = Math.max(50, Math.floor(r.height));
    draw();
  }
  window.addEventListener("resize", fit);

  var view = null; // {sx, sy, ox, oy}

  function computeView() {
    if (!pts.length) { view = null; return; }
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.y; });
    var lo = [Math.min.apply(0, xs), Math.min.apply(0, ys)];
    var hi = [Math.max.apply(0, xs), Math.max.apply(0, ys)];
    var pad = 26;
    var s = Math.min((mapCv.width - 2 * pad) / Math.max(1e-9, hi[0] - lo[0]),
                     (mapCv.height - 2 * pad) / Math.max(1e-9, hi[1] - lo[1]));
    view = { s: s,
             ox: pad - lo[0] * s + (mapCv.width - 2 * pad - (hi[0] - lo[0]) * s) / 2,
             oy: pad - lo[1] * s + (mapCv.height - 2 * pad - (hi[1] - lo[1]) * s) / 2 };
  }

  function px(p) { return [p.x * view.s + view.ox, p.y * view.s + view.oy]; }

  function colour(v) {
    // slate -> cyan -> warm, outlier-safe (v already 0..1-ish)
    var t = Math.max(0, Math.min(1, v));
    var r = Math.round(40 + 190 * t * t);
    var g = Math.round(70 + 140 * t);
    var b = Math.round(110 + 90 * (1 - t));
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function draw() {
    var ctx = mapCv.getContext("2d");
    ctx.clearRect(0, 0, mapCv.width, mapCv.height);
    if (!pts.length || !view) return;
    // radial reference rings around the identity lens (index 0 = origin)
    var o = px(pts[0]);
    ctx.strokeStyle = "#151b23";
    for (var rr = 1; rr <= 4; rr++) {
      ctx.beginPath();
      ctx.arc(o[0], o[1], rr * 90, 0, 2 * Math.PI);
      ctx.stroke();
    }
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i], c = px(p);
      ctx.fillStyle = colour(colorBy === "nl" ? p.nl : p.osc * 2);
      ctx.beginPath();
      ctx.arc(c[0], c[1], i === selected ? 5 : 2.2, 0, 2 * Math.PI);
      ctx.fill();
      if (i === selected) { ctx.strokeStyle = "#e6ebf1"; ctx.stroke(); }
    }
    // identity marker
    ctx.strokeStyle = "#8b95a1";
    ctx.beginPath(); ctx.arc(o[0], o[1], 6, 0, 2 * Math.PI); ctx.stroke();
  }

  mapCv.addEventListener("click", function (ev) {
    if (!pts.length || !view) return;
    var r = mapCv.getBoundingClientRect();
    var mx = (ev.clientX - r.left) * (mapCv.width / r.width);
    var my = (ev.clientY - r.top) * (mapCv.height / r.height);
    var best = -1, bd = 1e9;
    for (var i = 0; i < pts.length; i++) {
      var c = px(pts[i]);
      var d = (c[0] - mx) * (c[0] - mx) + (c[1] - my) * (c[1] - my);
      if (d < bd) { bd = d; best = i; }
    }
    if (best >= 0 && bd < 400) selectLens(best);
  });

  function selectLens(i) {
    selected = i;
    draw();
    post("/api/radial/lens", { i: pts[i].i, kind: kind }).then(function (d) {
      if (d.error) { status(d.error); return; }
      lastLens = d;
      $("rm-prog").textContent = "#" + d.i + "   " + d.prog;
      drawCurve(d.xs, d.ys);
    });
  }

  function drawCurve(xs, ys) {
    var ctx = curveCv.getContext("2d");
    var W = curveCv.width, H = curveCv.height;
    ctx.clearRect(0, 0, W, H);
    var ylo = Math.min.apply(0, ys), yhi = Math.max.apply(0, ys);
    if (yhi - ylo < 1e-9) { ylo -= 1; yhi += 1; }
    ctx.strokeStyle = "#232a33";
    ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
    ctx.strokeStyle = "#5fb2d3"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i < xs.length; i++) {
      var x = (i / (xs.length - 1)) * (W - 8) + 4;
      var y = H - 6 - ((ys[i] - ylo) / (yhi - ylo)) * (H - 12);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke(); ctx.lineWidth = 1;
  }

  /* ---- actions -------------------------------------------------------- */

  $("rm-build").onclick = function () {
    kind = $("rm-kind").value;
    var n = parseInt($("rm-n").value, 10) || 1200;
    status("building " + n + "-lens map on '" + kind + "'…");
    $("rm-maplbl").innerHTML = "building…";
    post("/api/radial/map", { n: n, kind: kind }).then(function (d) {
      if (d.error) { status(d.error); return; }
      lastMap = d;
      pts = d.pts; selected = -1;
      computeView(); draw();
      $("rm-maplbl").innerHTML = "<b>" + d.n + "</b> lenses · " + d.kind +
        " · rings centred on identity";
      var c = d.checks, box = $("rm-checks");
      box.innerHTML = "";
      kv(box, "radius vs nonlinearity", (c.radius_vs_nonlinearity_corr >= 0 ? "+" : "") +
         c.radius_vs_nonlinearity_corr);
      kv(box, "determinism err", c.determinism_err);
      kv(box, "identity at origin", c.identity_at_origin ? "yes" : "NO");
      status("map built");
    }).catch(function (e) { status("map failed: " + e); });
  };

  function kv(box, k, v) {
    var d = document.createElement("div");
    d.className = "rm-kv";
    d.innerHTML = "<span>" + k + "</span><b>" + v + "</b>";
    box.appendChild(d);
  }

  $("rm-probe").onclick = function () {
    kind = $("rm-kind").value;
    status("running linear probe on '" + kind + "'…");
    post("/api/radial/probe", { n: 400, kind: kind }).then(function (d) {
      if (d.error) { status(d.error); return; }
      lastProbe = d;
      var t = $("rm-probetbl");
      t.innerHTML = "<tr><th>task</th><th>raw x</th><th>lens bank</th></tr>";
      d.rows.forEach(function (r) {
        var tr = document.createElement("tr");
        var win = r.r2_lens > r.r2_linear + 0.05 ? " class=\"win\"" : "";
        tr.innerHTML = "<td>" + r.task + "</td><td>" + r.r2_linear.toFixed(3) +
          "</td><td" + win + ">" + r.r2_lens.toFixed(3) + "</td>";
        t.appendChild(tr);
      });
      status("probe done (" + d.rows[0].n_lens + " lenses in bank)");
    }).catch(function (e) { status("probe failed: " + e); });
  };

  /* ---- rotation probe -------------------------------------------------- */

  $("rm-rotate").onclick = function () {
    kind = $("rm-kind").value;
    status("rotating the map 1°/step and probing each slice (30-90s)…");
    post("/api/radial/rotate", { n: 800, kind: kind }).then(function (d) {
      if (d.error) { status(d.error); return; }
      lastRot = d;
      drawRotCurve(d);
      var box = $("rm-rotkv");
      box.innerHTML = "";
      kv(box, "best angle", d.best.deg + "° → " + d.best.r2.toFixed(3));
      kv(box, "worst angle", d.worst.deg + "° → " + d.worst.r2.toFixed(3));
      kv(box, "angular spread", d.angular_spread);
      kv(box, "random subset (same size)", d.baseline_random_mean.toFixed(3) +
         " ± " + d.baseline_random_std);
      kv(box, "full bank", d.baseline_full.toFixed(3));
      kv(box, "slice size", d.slice_size + " of " + d.n_lens);
      status("rotation probe done");
    }).catch(function (e) { status("rotation probe failed: " + e); });
  };

  function drawRotCurve(d) {
    var cv = $("rm-rotcurve"), ctx = cv.getContext("2d");
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var ys = d.mean_r2;
    var ylo = Math.min(0, Math.min.apply(0, ys)), yhi = 1.0;
    var Y = function (v) { return H - 6 - ((v - ylo) / (yhi - ylo)) * (H - 12); };
    // random-subset baseline band
    ctx.fillStyle = "rgba(139,149,161,0.15)";
    var b0 = Y(d.baseline_random_mean - d.baseline_random_std);
    var b1 = Y(d.baseline_random_mean + d.baseline_random_std);
    ctx.fillRect(0, b1, W, Math.max(1, b0 - b1));
    // the angle curve
    ctx.strokeStyle = "#5fd39a"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i < ys.length; i++) {
      var x = (i / (ys.length - 1)) * (W - 8) + 4;
      if (i === 0) ctx.moveTo(x, Y(ys[i])); else ctx.lineTo(x, Y(ys[i]));
    }
    ctx.stroke(); ctx.lineWidth = 1;
    ctx.fillStyle = "#7f8b98"; ctx.font = "9px ui-monospace,monospace";
    ctx.fillText("0°", 4, H - 1);
    ctx.fillText("360°", W - 26, H - 1);
  }

  /* ---- export --------------------------------------------------------- */

  function r3(v) { return Math.round(v * 1000) / 1000; }

  function distSummary(vals) {
    var s = vals.slice().sort(function (a, b) { return a - b; });
    var mean = s.reduce(function (a, b) { return a + b; }, 0) / s.length;
    return { min: r3(s[0]), p50: r3(s[Math.floor(s.length * 0.5)]),
             p90: r3(s[Math.floor(s.length * 0.9)]), max: r3(s[s.length - 1]),
             mean: r3(mean) };
  }

  function extremes(P, key, n, low) {
    return P.slice().sort(function (a, b) {
      return low ? a[key] - b[key] : b[key] - a[key];
    }).slice(0, n).map(function (p) {
      var e = { i: p.i, prog: p.prog };
      e[key] = p[key];
      return e;
    });
  }

  $("rm-export").onclick = function () {
    if (!lastMap && !lastProbe && !lastRot) {
      status("nothing to export yet — build a map or run a probe first");
      return;
    }
    var out = {
      format: "radial-map-v2-export",
      exported: new Date().toISOString(),
      note: "map.rows is column-oriented (see map.cols); programs are kept " +
            "only for extreme lenses — rebuild any lens from its index, the " +
            "space is deterministic"
    };
    if (lastMap) {
      var P = lastMap.pts;
      out.map = {
        kind: lastMap.kind,
        n: lastMap.n,
        checks: lastMap.checks,
        cols: ["i", "x", "y", "nl", "osc"],
        rows: P.map(function (p) { return [p.i, p.x, p.y, p.nl, p.osc]; }),
        summary: {
          radius: distSummary(P.map(function (p) { return Math.sqrt(p.x * p.x + p.y * p.y); })),
          nonlinearity: distSummary(P.map(function (p) { return p.nl; })),
          oscillation: distSummary(P.map(function (p) { return p.osc; })),
          most_nonlinear: extremes(P, "nl", 8),
          most_oscillatory: extremes(P, "osc", 8),
          most_linear: extremes(P, "nl", 8, true)
        }
      };
    }
    if (lastProbe) out.probe = lastProbe;
    if (lastRot) out.rotation_probe = lastRot;
    if (lastLens) out.selected_lens = lastLens;
    var name = "radial_map_" + (lastMap ? lastMap.kind + "_" + lastMap.n : "probe") +
      "_" + new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-") + ".json";
    var blob = new Blob([JSON.stringify(out)], { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
    status("exported " + name);
  };

  /* toggle */
  Array.prototype.forEach.call(document.querySelectorAll(".rm-tg"), function (el) {
    el.onclick = function () {
      Array.prototype.forEach.call(document.querySelectorAll(".rm-tg"), function (x) {
        x.classList.remove("on");
      });
      el.classList.add("on");
      colorBy = el.getAttribute("data-c");
      draw();
    };
  });

  fit();
})();
