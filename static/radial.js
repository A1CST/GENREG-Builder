/* radial.js — Radial Map v2 front-end: build the activation-behavior map,
   inspect lenses, run the linear probe. Plain fetch + canvas, no libs. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var mapCv = $("rm-map"), curveCv = $("rm-curve");
  var pts = [], colorBy = "nl", kind = "loops", selected = -1;
  var lastMap = null, lastProbe = null, lastLens = null, lastRot = null,
      lastLadder = null;

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

  /* 3D view state — drag to orbit, wheel to zoom, shift/right-drag to pan */
  var yaw = 0, pitch = 0, zoom = 1, panX = 0, panY = 0, baseScale = 1, maxR = 1;

  function computeView() {
    if (!pts.length) return;
    maxR = 1e-9;
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      maxR = Math.max(maxR, Math.sqrt(p.x * p.x + p.y * p.y + (p.z || 0) * (p.z || 0)));
    }
    baseScale = (Math.min(mapCv.width, mapCv.height) / 2 - 30) / maxR;
    yaw = 0.5; pitch = -0.3; zoom = 1; panX = 0; panY = 0;
  }

  function project(x, y, z) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw);
    var cp = Math.cos(pitch), sp = Math.sin(pitch);
    var x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
    var y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
    var s = baseScale * zoom;
    return [mapCv.width / 2 + panX + x1 * s,
            mapCv.height / 2 + panY + y1 * s, z2];
  }

  function px(p) { return project(p.x, p.y, p.z || 0); }

  function colour(v) {
    // slate -> cyan -> warm, outlier-safe (v already 0..1-ish)
    var t = Math.max(0, Math.min(1, v));
    var r = Math.round(40 + 190 * t * t);
    var g = Math.round(70 + 140 * t);
    var b = Math.round(110 + 90 * (1 - t));
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function ring(ctx, r, plane) {
    ctx.beginPath();
    for (var a = 0; a <= 64; a++) {
      var t = (a / 64) * 2 * Math.PI, c;
      if (plane === "xy") c = project(r * Math.cos(t), r * Math.sin(t), 0);
      else if (plane === "xz") c = project(r * Math.cos(t), 0, r * Math.sin(t));
      else c = project(0, r * Math.cos(t), r * Math.sin(t));
      if (a === 0) ctx.moveTo(c[0], c[1]); else ctx.lineTo(c[0], c[1]);
    }
    ctx.stroke();
  }

  function draw() {
    var ctx = mapCv.getContext("2d");
    ctx.clearRect(0, 0, mapCv.width, mapCv.height);
    if (!pts.length) return;
    // wireframe sphere hint: equator rings in the three planes + radius rings
    ctx.strokeStyle = "#151b23";
    ring(ctx, maxR, "xy"); ring(ctx, maxR, "xz"); ring(ctx, maxR, "yz");
    for (var rr = 1; rr <= 3; rr++) ring(ctx, (maxR * rr) / 4, "xy");
    // depth sort: far points first, near points on top
    var order = [];
    for (var i = 0; i < pts.length; i++) {
      var c = px(pts[i]);
      order.push([c[2], i, c[0], c[1]]);
    }
    order.sort(function (a, b) { return a[0] - b[0]; });
    var zlo = order[0][0], zhi = order[order.length - 1][0], zr = Math.max(1e-9, zhi - zlo);
    for (var k = 0; k < order.length; k++) {
      var d = (order[k][0] - zlo) / zr;          // 0 far .. 1 near
      var j = order[k][1], p = pts[j];
      ctx.globalAlpha = 0.35 + 0.65 * d;
      ctx.fillStyle = colour(colorBy === "nl" ? p.nl : p.osc * 2);
      ctx.beginPath();
      ctx.arc(order[k][2], order[k][3], (j === selected ? 5 : 1.4 + 1.6 * d) * Math.sqrt(zoom > 1 ? Math.min(zoom, 4) : 1), 0, 2 * Math.PI);
      ctx.fill();
      if (j === selected) { ctx.globalAlpha = 1; ctx.strokeStyle = "#e6ebf1"; ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
    // origin marker (0,0,0 = the identity lens): solid red dot + halo ring
    var o = project(0, 0, 0);
    ctx.fillStyle = "#e04b3a";
    ctx.beginPath(); ctx.arc(o[0], o[1], 4.5, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = "#e04b3a";
    ctx.beginPath(); ctx.arc(o[0], o[1], 8, 0, 2 * Math.PI); ctx.stroke();
  }

  /* orbit / zoom / pan / click-select */
  var drag = null;

  function canvasXY(ev) {
    var r = mapCv.getBoundingClientRect();
    return [(ev.clientX - r.left) * (mapCv.width / r.width),
            (ev.clientY - r.top) * (mapCv.height / r.height)];
  }

  mapCv.addEventListener("contextmenu", function (ev) { ev.preventDefault(); });

  mapCv.addEventListener("pointerdown", function (ev) {
    if (!pts.length) return;
    ev.preventDefault();
    drag = { x0: ev.clientX, y0: ev.clientY, yaw: yaw, pitch: pitch,
             panX: panX, panY: panY, moved: false,
             pan: ev.shiftKey || ev.button === 2 || ev.button === 1 };
    mapCv.setPointerCapture(ev.pointerId);
  });

  mapCv.addEventListener("pointermove", function (ev) {
    if (!drag) return;
    var dx = ev.clientX - drag.x0, dy = ev.clientY - drag.y0;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    if (!drag.moved) return;
    if (drag.pan) {
      var r = mapCv.getBoundingClientRect(), kx = mapCv.width / r.width;
      panX = drag.panX + dx * kx; panY = drag.panY + dy * kx;
    } else {
      yaw = drag.yaw + dx * 0.008;
      pitch = Math.max(-1.55, Math.min(1.55, drag.pitch + dy * 0.008));
    }
    draw();
  });

  mapCv.addEventListener("pointerup", function (ev) {
    var wasClick = drag && !drag.moved;
    drag = null;
    if (!wasClick || !pts.length) return;
    var m = canvasXY(ev), best = -1, bd = 1e9;
    for (var i = 0; i < pts.length; i++) {
      var c = px(pts[i]);
      var d = (c[0] - m[0]) * (c[0] - m[0]) + (c[1] - m[1]) * (c[1] - m[1]);
      if (d < bd) { bd = d; best = i; }
    }
    if (best >= 0 && bd < 400) selectLens(best);
  });

  mapCv.addEventListener("wheel", function (ev) {
    if (!pts.length) return;
    ev.preventDefault();
    var f = Math.exp(-ev.deltaY * 0.0012);
    var nz = Math.max(0.2, Math.min(40, zoom * f));
    f = nz / zoom;
    var m = canvasXY(ev);
    // keep the point under the cursor fixed while zooming
    panX = (m[0] - mapCv.width / 2) - ((m[0] - mapCv.width / 2) - panX) * f;
    panY = (m[1] - mapCv.height / 2) - ((m[1] - mapCv.height / 2) - panY) * f;
    zoom = nz;
    draw();
  }, { passive: false });

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
        " · drag = orbit · wheel = zoom · shift-drag = pan · click = inspect";
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

  /* ---- task ladder ------------------------------------------------------ */

  $("rm-ladder").onclick = function () {
    kind = $("rm-kind").value;
    status("climbing the task ladder on '" + kind + "'…");
    post("/api/radial/ladder", { n: 400, kind: kind }).then(function (d) {
      if (d.error) { status(d.error); return; }
      lastLadder = d;
      var t = $("rm-laddertbl");
      t.innerHTML = "<tr><th>task</th><th>raw x</th><th>lens bank</th><th></th></tr>";
      d.rungs.forEach(function (r) {
        var tr = document.createElement("tr");
        tr.innerHTML = "<td>" + r.task + "</td><td>" + r.r2_linear.toFixed(3) +
          "</td><td" + (r.passed ? " class=\"win\"" : "") + ">" +
          r.r2_lens.toFixed(3) + "</td><td>" + (r.passed ? "✓" : "✗") + "</td>";
        t.appendChild(tr);
      });
      $("rm-frontier").textContent = d.frontier
        ? "cleared " + d.cleared + "/" + d.total + " — frontier: " + d.frontier
        : "ladder complete (" + d.cleared + "/" + d.total + ")";
      status("ladder done");
    }).catch(function (e) { status("ladder failed: " + e); });
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
    if (!lastMap && !lastProbe && !lastRot && !lastLadder) {
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
        cols: ["i", "x", "y", "z", "nl", "osc"],
        rows: P.map(function (p) { return [p.i, p.x, p.y, p.z || 0, p.nl, p.osc]; }),
        summary: {
          radius: distSummary(P.map(function (p) {
            return Math.sqrt(p.x * p.x + p.y * p.y + (p.z || 0) * (p.z || 0));
          })),
          nonlinearity: distSummary(P.map(function (p) { return p.nl; })),
          oscillation: distSummary(P.map(function (p) { return p.osc; })),
          most_nonlinear: extremes(P, "nl", 8),
          most_oscillatory: extremes(P, "osc", 8),
          most_linear: extremes(P, "nl", 8, true)
        }
      };
    }
    if (lastProbe) out.probe = lastProbe;
    if (lastLadder) out.ladder = lastLadder;
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

  /* colour-by toggle */
  Array.prototype.forEach.call(document.querySelectorAll(".rm-tg[data-c]"), function (el) {
    el.onclick = function () {
      Array.prototype.forEach.call(document.querySelectorAll(".rm-tg[data-c]"), function (x) {
        x.classList.remove("on");
      });
      el.classList.add("on");
      colorBy = el.getAttribute("data-c");
      draw();
    };
  });

  /* ---- baselines view --------------------------------------------------- */

  var baseLoaded = false;

  Array.prototype.forEach.call(document.querySelectorAll(".rm-vw"), function (el) {
    el.onclick = function () {
      Array.prototype.forEach.call(document.querySelectorAll(".rm-vw"), function (x) {
        x.classList.remove("on");
      });
      el.classList.add("on");
      var showBase = el.getAttribute("data-v") === "base";
      $("rm-view-map").style.display = showBase ? "none" : "";
      $("rm-view-base").style.display = showBase ? "block" : "";
      if (showBase && !baseLoaded) loadBaselines();
      if (!showBase) fit();
    };
  });

  function bar(rows, label, val, scale, cls) {
    if (val == null) return;
    rows.push('<div class="rm-brow"><div class="lb">' + label + '</div>' +
      '<div class="tr"><div class="fl ' + (cls || "") + '" style="width:' +
      Math.max(1, Math.min(100, (val / scale) * 100)).toFixed(1) + '%"></div></div>' +
      '<div class="vl">' + val.toFixed(3) + '</div></div>');
  }

  function lineChart(cv, xs, ys, opts) {
    var ctx = cv.getContext("2d");
    var W = cv.width = cv.clientWidth * 2, H = cv.height = cv.clientHeight * 2;
    ctx.clearRect(0, 0, W, H);
    var lo = opts && opts.lo != null ? opts.lo : Math.min.apply(0, ys);
    var hi = opts && opts.hi != null ? opts.hi : Math.max.apply(0, ys);
    if (hi - lo < 1e-9) { hi = lo + 1; }
    var Y = function (v) { return H - 8 - ((v - lo) / (hi - lo)) * (H - 16); };
    if (opts && opts.ref != null) {
      ctx.strokeStyle = "#4a3a2a"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(0, Y(opts.ref)); ctx.lineTo(W, Y(opts.ref)); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = opts && opts.color ? opts.color : "#5fb2d3";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i < ys.length; i++) {
      var x = (i / Math.max(1, ys.length - 1)) * (W - 12) + 6;
      if (i === 0) ctx.moveTo(x, Y(ys[i])); else ctx.lineTo(x, Y(ys[i]));
    }
    ctx.stroke();
    ctx.lineWidth = 1;
  }

  function classChart(cv, perClass) {
    var ctx = cv.getContext("2d");
    var W = cv.width = cv.clientWidth * 2, H = cv.height = cv.clientHeight * 2;
    ctx.clearRect(0, 0, W, H);
    var ks = Object.keys(perClass);
    var bw = W / ks.length;
    ks.forEach(function (k, i) {
      var v = perClass[k];
      ctx.fillStyle = v < 0.5 ? "#7d3a2e" : "#2f4152";
      ctx.fillRect(i * bw + bw * 0.18, H - 14 - v * (H - 22), bw * 0.64, v * (H - 22));
      ctx.fillStyle = "#7f8b98";
      ctx.font = "16px ui-monospace,monospace";
      ctx.fillText(k, i * bw + bw * 0.34, H - 2);
    });
  }

  function domainCard(d) {
    var p = d.probe, r = d.rotation_probe;
    var rows = [];
    var vals = [p.majority_class_acc, p.raw_linear_acc, p.acc,
                r && r.best && r.best.acc, p.bigram_table_ceiling].filter(function (v) {
      return v != null;
    });
    var scale = Math.max.apply(0, vals) * 1.12;
    bar(rows, "majority class", p.majority_class_acc, scale, "lo");
    bar(rows, "raw linear", p.raw_linear_acc, scale);
    if (p.bigram_table_ceiling != null) bar(rows, "bigram ceiling", p.bigram_table_ceiling, scale, "lo");
    bar(rows, "lens bank", p.acc, scale, "hi");
    if (r && r.best) bar(rows, "best rotation slice", r.best.acc, scale,
                         r.best.acc > p.acc ? "hi" : "");
    var html = '<div class="rm-card"><h2>' + d.domain + '</h2>' +
      '<div class="sub">' + (p.task || p.input_format || "") + " · train " + p.train +
      " / test " + p.test + '</div>' + rows.join("");
    html += '<canvas class="cv-curve"></canvas><div class="cl">accuracy vs lens count (' +
      p.curve.map(function (c) { return c.n_lens; }).join("/") + ' lenses)</div>';
    if (r && r.acc) {
      html += '<canvas class="cv-rot"></canvas><div class="cl">rotation sweep 0-360° · slice ~' +
        r.slice_size + " · spread " + r.angular_spread +
        (r.baseline_random_mean != null ? " · random " + r.baseline_random_mean : "") + '</div>';
    }
    if (p.per_class) html += '<canvas class="cv-cls"></canvas><div class="cl">per-class recall</div>';
    html += '<div class="kf">' + keyFinding(d) + '</div></div>';
    var el = document.createElement("div");
    el.innerHTML = html;
    el = el.firstChild;
    // charts must draw AFTER the card is in the DOM (clientWidth is 0 before)
    chartQueue.push(function () {
      lineChart(el.querySelector(".cv-curve"),
                p.curve.map(function (c) { return c.n_lens; }),
                p.curve.map(function (c) { return c.acc; }),
                { ref: p.raw_linear_acc, color: "#5fd39a" });
      if (r && r.acc) {
        lineChart(el.querySelector(".cv-rot"), r.angles, r.acc,
                  { ref: r.baseline_random_mean });
      }
      if (p.per_class) classChart(el.querySelector(".cv-cls"), p.per_class);
    });
    return el;
  }

  var chartQueue = [];

  function keyFinding(d) {
    var p = d.probe, r = d.rotation_probe;
    var gain = p.acc - p.raw_linear_acc;
    var s = "bank vs raw: <b>" + (gain >= 0 ? "+" : "") + (gain * 100).toFixed(1) + " pts</b>. ";
    if (p.bigram_table_ceiling != null) {
      s += "Lands within <b>" + (p.bigram_table_ceiling - p.acc).toFixed(4) +
        "</b> of the bigram-table ceiling with no table. ";
    }
    if (r && r.best && r.best.acc > p.acc) {
      s += "Best " + (r.slice_size || "") + "-lens slice (<b>" + r.best.acc.toFixed(3) +
        " @ " + r.best.deg + "°</b>) beats the full bank. ";
    }
    if (r && r.map_shape_axis_std) s += "Map shape " + r.map_shape_axis_std.join(" / ") + ".";
    return s;
  }

  function fixesCard(fx) {
    var rows = [];
    ["axis_y", "axis_x", "axis_z", "whitened_y"].forEach(function (k) {
      if (!fx[k]) return;
      bar(rows, k.replace("_", " "), fx[k].angular_spread, 1.0,
          fx[k].angular_spread > 0.5 ? "" : "hi");
    });
    var el = document.createElement("div");
    el.innerHTML = '<div class="rm-card"><h2>pre-baseline fixes (loops)</h2>' +
      '<div class="sub">rotation-probe R² spread per spin axis — lower would mean fewer dead zones</div>' +
      rows.join("") +
      '<div class="kf">All axes hit the same worst-case floor (<b>' +
      (fx.axis_y ? fx.axis_y.worst.r2 : "?") +
      '</b>) and whitening the flat axis changes nothing — the dead zone is ' +
      'behavioral redundancy, not geometry. <b>Free rotation is fine; no anchor needed.</b></div></div>';
    return el.firstChild;
  }

  function loadBaselines() {
    fetch("/api/radial/baselines").then(function (r) { return r.json(); }).then(function (d) {
      baseLoaded = true;
      var box = $("rm-cards");
      box.innerHTML = "";
      if (d.error) { box.textContent = d.error; return; }
      var order = ["mnist", "cifar", "text", "audio"];
      order.concat(Object.keys(d.domains).filter(function (k) {
        return order.indexOf(k) < 0;
      })).forEach(function (k) {
        if (d.domains[k]) box.appendChild(domainCard(d.domains[k]));
      });
      if (d.fixes) box.appendChild(fixesCard(d.fixes));
      if (!box.children.length) {
        box.textContent = "no baseline exports found — run: python radial_baseline.py all";
      }
      requestAnimationFrame(function () {
        chartQueue.forEach(function (f) { f(); });
        chartQueue = [];
      });
    }).catch(function (e) { $("rm-cards").textContent = "failed to load: " + e; });
  }

  fit();
})();
