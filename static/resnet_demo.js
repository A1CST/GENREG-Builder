/* resnet_demo.js — interactive explainer of the residual block used by
   resnet_evo.py:  h <- h + gain * act(a*(mix*h)+b).

   Two linked views on one canvas:
     LEFT  — the block schematic: input -> mix -> act -> (xgain) -> (+) -> out,
             with the identity SKIP arc bypassing the transform. A dot flows
             both paths so the "add a small correction onto the identity" idea
             is visible.
     RIGHT — signal-through-depth: a 1-D feature signal is pushed through
             `depth` blocks. Residual mode adds gain*correction onto the signal;
             plain mode REPLACES it. The energy-vs-depth line (green=residual,
             amber=plain) shows the skip keeping the signal alive while the
             plain stack collapses toward a dead constant.

   Pure canvas, no libs, no models. Illustrative — the real thing runs on the
   pod / locally in resnet_evo.py. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var cv = $("rd2-cv");
  if (!cv) return;
  var ctx = cv.getContext("2d");

  /* ---- state ---- */
  var skip = 1, depth = 5, gain = 0.6, act = "tanh";
  var A = 1.5, B = 0.2;               // fixed a,b so saturation is visible
  var t = 0;

  /* ---- the 8-function GENREG activation catalog (matches resnet_evo) ---- */
  var ACT = {
    id:   function (x) { return x; },
    abs:  function (x) { return Math.abs(x); },
    relu: function (x) { return x > 0 ? x : 0; },
    tanh: function (x) { return Math.tanh(x); },
    gauss:function (x) { return Math.exp(-x * x); },
    sq:   function (x) { return Math.max(-4, Math.min(4, x * x)); },
    soft: function (x) { return Math.log1p(Math.exp(Math.max(-30, Math.min(30, x)))); },
    sin:  function (x) { return Math.sin(x); }
  };

  var N = 64;                         // signal resolution
  function baseSignal(phase) {
    var x = [];
    for (var i = 0; i < N; i++) {
      var u = i / (N - 1) * Math.PI * 2;
      x.push(0.9 * Math.sin(u + phase) + 0.4 * Math.sin(2.3 * u - phase * 0.6));
    }
    return x;
  }

  /* push the signal through `d` residual/plain blocks; return all layers */
  function evolveSignal(x0, useSkip, d, g) {
    var f = ACT[act], layers = [x0.slice()], x = x0.slice(), i, k;
    for (k = 0; k < d; k++) {
      var nx = new Array(N);
      for (i = 0; i < N; i++) {
        var corr = f(A * x[i] + B);
        nx[i] = useSkip ? x[i] + g * corr : corr;   // residual adds; plain replaces
      }
      x = nx; layers.push(x.slice());
    }
    return layers;
  }

  function energy(sig) {                // std = "how much signal is left"
    var m = 0, i;
    for (i = 0; i < N; i++) m += sig[i];
    m /= N;
    var v = 0;
    for (i = 0; i < N; i++) v += (sig[i] - m) * (sig[i] - m);
    return Math.sqrt(v / N);
  }

  /* ---- drawing ---- */
  function fit() {
    var r = cv.getBoundingClientRect();
    cv.width = Math.max(320, Math.floor(r.width)) * 2;
    cv.height = Math.max(200, Math.floor(r.height)) * 2;
  }

  function roundRect(x, y, w, h, rad) {
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
  }

  function box(x, y, w, h, label, sub, accent) {
    roundRect(x, y, w, h, 8);
    ctx.fillStyle = "#101720"; ctx.fill();
    ctx.strokeStyle = accent || "#2f4152"; ctx.lineWidth = 3; ctx.stroke();
    ctx.fillStyle = "#dbe2ea"; ctx.font = "600 26px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(label, x + w / 2, y + h / 2 + (sub ? -6 : 9));
    if (sub) { ctx.fillStyle = "#7f8b98"; ctx.font = "20px ui-monospace,monospace";
      ctx.fillText(sub, x + w / 2, y + h / 2 + 20); }
    ctx.textAlign = "left";
  }

  function arrow(x1, y1, x2, y2, col) {
    ctx.strokeStyle = col; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    var a = Math.atan2(y2 - y1, x2 - x1);
    ctx.beginPath(); ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 12 * Math.cos(a - 0.4), y2 - 12 * Math.sin(a - 0.4));
    ctx.lineTo(x2 - 12 * Math.cos(a + 0.4), y2 - 12 * Math.sin(a + 0.4));
    ctx.closePath(); ctx.fillStyle = col; ctx.fill();
  }

  function flowDot(x, y, on) {
    ctx.beginPath(); ctx.arc(x, y, 7, 0, 2 * Math.PI);
    ctx.fillStyle = on ? "#5fd39a" : "#3a4756"; ctx.fill();
  }

  function drawSchematic(x0, y0, w, h) {
    var midY = y0 + h * 0.52;
    var inX = x0 + 30;
    var mixX = x0 + w * 0.24, boxW = w * 0.15, boxH = 58;
    var actX = x0 + w * 0.46;
    var addX = x0 + w * 0.76, addR = 22;
    var outX = x0 + w - 24;

    // title
    ctx.fillStyle = "#8b95a1"; ctx.font = "22px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("one residual block", x0, y0 + 6);

    // main transform path
    arrow(inX + 8, midY, mixX, midY, "#3a4756");
    box(mixX, midY - boxH / 2, boxW, boxH, "mix", "C x C", "#2f4152");
    arrow(mixX + boxW, midY, actX, midY, "#3a4756");
    box(actX, midY - boxH / 2, boxW, boxH, "act", "a·h + b", "#3a5a44");
    // gain label on the path into the adder
    arrow(actX + boxW, midY, addX - addR, midY, skip ? "#3a5a44" : "#7d5a2e");
    ctx.fillStyle = gain < 0.12 ? "#e0a050" : "#7fd0a8";
    ctx.font = "600 21px ui-monospace,monospace"; ctx.textAlign = "center";
    ctx.fillText("x gain " + gain.toFixed(2), (actX + boxW + addX) / 2, midY - 14);
    ctx.textAlign = "left";

    // the adder node
    ctx.beginPath(); ctx.arc(addX, midY, addR, 0, 2 * Math.PI);
    ctx.fillStyle = "#101720"; ctx.fill();
    ctx.strokeStyle = "#5fd39a"; ctx.lineWidth = 3; ctx.stroke();
    ctx.fillStyle = "#dbe2ea"; ctx.font = "600 30px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("+", addX, midY + 10); ctx.textAlign = "left";
    arrow(addX + addR, midY, outX, midY, "#5fd39a");

    // input & output nodes
    flowDot(inX, midY, true);
    ctx.fillStyle = "#8b95a1"; ctx.font = "20px ui-monospace,monospace"; ctx.textAlign = "center";
    ctx.fillText("h", inX, midY + 34); ctx.fillText("h'", outX, midY + 34); ctx.textAlign = "left";

    // the SKIP arc (identity) — bypasses mix+act, into the adder
    var skipY = y0 + h * 0.14;
    ctx.setLineDash(skip ? [] : [7, 6]);
    ctx.strokeStyle = skip ? "#5fd39a" : "#3a4756"; ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(inX, midY - 8);
    ctx.lineTo(inX, skipY); ctx.lineTo(addX, skipY); ctx.lineTo(addX, midY - addR);
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = skip ? "#5fd39a" : "#6c7885"; ctx.font = "20px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(skip ? "identity skip" : "no skip (plain)", (inX + addX) / 2, skipY - 10);
    ctx.textAlign = "left";

    // animated flow dots along both paths (only the skip when residual)
    var ph = (t * 0.4) % 1;
    // main path dot
    var mainX = inX + (addX - addR - inX) * ph;
    flowDot(mainX, midY, true);
    if (skip) {
      // skip dot travels the arc top
      var sx = inX + (addX - inX) * ph;
      flowDot(sx, skipY, true);
    }
  }

  function drawWaterfall(x0, y0, w, h, layers) {
    ctx.fillStyle = "#8b95a1"; ctx.font = "22px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("signal through " + depth + " block" + (depth === 1 ? "" : "s") +
                 (skip ? "  (residual)" : "  (plain)"), x0, y0 + 6);
    var d = layers.length, rowH = (h - 20) / d, i, k;
    for (k = 0; k < d; k++) {
      var cy = y0 + 22 + rowH * (k + 0.5);
      var sig = layers[k];
      // baseline
      ctx.strokeStyle = "#14191f"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, cy); ctx.lineTo(x0 + w, cy); ctx.stroke();
      // curve
      var col = k === 0 ? "#6ea8dc" : (skip ? "#5fd39a" : "#e0a050");
      ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.beginPath();
      for (i = 0; i < N; i++) {
        var px = x0 + w * i / (N - 1);
        var py = cy - sig[i] * (rowH * 0.42) / 1.6;
        py = Math.max(y0 + 22 + rowH * k + 2, Math.min(y0 + 22 + rowH * (k + 1) - 2, py));
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.fillStyle = "#6c7885"; ctx.font = "18px ui-monospace,monospace";
      ctx.fillText(k === 0 ? "in" : "b" + k, x0 - 4, cy - rowH * 0.32);
    }
  }

  function drawEnergy(x0, y0, w, h, eRes, ePlain) {
    ctx.fillStyle = "#8b95a1"; ctx.font = "22px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("signal energy vs depth", x0, y0 + 6);
    var pad = 34, gx0 = x0 + pad, gw = w - pad - 8, gy1 = y0 + h - 24, gh = h - 48;
    var maxE = Math.max(1.2, Math.max.apply(null, eRes.concat(ePlain)));
    // axes
    ctx.strokeStyle = "#232a33"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(gx0, y0 + 20); ctx.lineTo(gx0, gy1); ctx.lineTo(gx0 + gw, gy1); ctx.stroke();
    ctx.fillStyle = "#6c7885"; ctx.font = "18px ui-monospace,monospace";
    ctx.fillText("std", 2, y0 + 30); ctx.fillText("0", gx0 - 16, gy1 + 4);
    ctx.fillText("depth", gx0 + gw / 2 - 20, y0 + h - 4);

    function line(arr, col, dash) {
      ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.setLineDash(dash || []);
      ctx.beginPath();
      arr.forEach(function (e, i) {
        var px = gx0 + gw * (arr.length === 1 ? 0 : i / (arr.length - 1));
        var py = gy1 - gh * Math.min(1, e / maxE);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke(); ctx.setLineDash([]);
      arr.forEach(function (e, i) {
        var px = gx0 + gw * (arr.length === 1 ? 0 : i / (arr.length - 1));
        var py = gy1 - gh * Math.min(1, e / maxE);
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px, py, 4, 0, 2 * Math.PI); ctx.fill();
      });
    }
    line(ePlain, "#e0a050", [6, 5]);
    line(eRes, "#5fd39a");
    // legend
    ctx.font = "19px sans-serif";
    ctx.fillStyle = "#5fd39a"; ctx.fillText("residual (skip)", gx0 + 6, y0 + 30);
    ctx.fillStyle = "#e0a050"; ctx.fillText("plain (no skip)", gx0 + 6, y0 + 52);
  }

  function caption(eRes, ePlain) {
    var end = skip ? eRes[eRes.length - 1] : ePlain[ePlain.length - 1];
    var alive = end > 0.25;
    var html;
    if (skip) {
      html = "The <b>identity skip</b> carries h straight to the <b>+</b>; the " +
        "mix→act path is only a <b>correction</b>, scaled by gain and added on. ";
      if (gain < 0.12) {
        html += "<span class='warn'>At gain≈0 the block is a near-perfect " +
          "identity</span> — this is exactly how a freshly-added block " +
          "<b>bootstraps</b>: it does nothing until evolution finds a gain&gt;0 that helps.";
      } else {
        html += "Even stacked " + depth + " deep the signal stays alive " +
          "(<span class='ok'>energy holds</span>) — each block refines, none destroys.";
      }
    } else {
      html = "Without the skip, every block <b>replaces</b> h with act(a·(mix·h)+b). ";
      if (alive) {
        html += "With this activation the signal survives a few layers — but push " +
          "depth up and watch the amber line sag.";
      } else {
        html += "<span class='warn'>Stacked " + depth + " deep the signal collapses " +
          "toward a dead constant</span> (energy → 0) — the degradation problem " +
          "the residual skip was invented to fix.";
      }
    }
    $("rd2-cap").innerHTML = html;
  }

  function draw() {
    fit();
    var W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    var x0 = baseSignal(t * 0.15);
    var layRes = evolveSignal(x0, true, depth, gain);
    var layPlain = evolveSignal(x0, false, depth, gain);
    var eRes = layRes.map(energy), ePlain = layPlain.map(energy);
    var layers = skip ? layRes : layPlain;

    // top: schematic (full width)
    drawSchematic(24, 20, W - 48, H * 0.40);
    // divider
    ctx.strokeStyle = "#14191f"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(24, H * 0.44); ctx.lineTo(W - 24, H * 0.44); ctx.stroke();
    // bottom: waterfall (left) + energy (right)
    var by = H * 0.47, bh = H - by - 16;
    drawWaterfall(40, by, W * 0.52, bh, layers);
    drawEnergy(W * 0.56, by, W * 0.42, bh, eRes, ePlain);

    caption(eRes, ePlain);
  }

  function loop() { t += 0.016; draw(); requestAnimationFrame(loop); }

  /* ---- controls ---- */
  function seg(id, cb) {
    var box = $(id);
    box.querySelectorAll("div").forEach(function (d) {
      d.addEventListener("click", function () {
        box.querySelectorAll("div").forEach(function (x) { x.classList.remove("on"); });
        d.classList.add("on"); cb(d.getAttribute("data-v"));
      });
    });
  }
  seg("rd2-skip", function (v) { skip = parseInt(v, 10); });
  $("rd2-depth").addEventListener("input", function () {
    depth = parseInt(this.value, 10); $("rd2-depth-v").textContent = depth;
  });
  $("rd2-gain").addEventListener("input", function () {
    gain = parseFloat(this.value); $("rd2-gain-v").textContent = gain.toFixed(2);
  });
  $("rd2-act").addEventListener("change", function () { act = this.value; });
  $("rd2-boot").querySelector("div").addEventListener("click", function () {
    // force the bootstrap story: residual on, gain -> ~0
    skip = 1; gain = 0.05;
    $("rd2-gain").value = 0.05; $("rd2-gain-v").textContent = "0.05";
    var sk = $("rd2-skip");
    sk.querySelectorAll("div").forEach(function (x) {
      x.classList.toggle("on", x.getAttribute("data-v") === "1");
    });
  });

  window.addEventListener("resize", fit);
  requestAnimationFrame(loop);
})();
