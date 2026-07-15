/* radial_demo.js — dual-panel radial space demo, built to the downloaded
   RADIAL_SPACE_VISUAL_SCHEMATIC.md. No models, purely visual.

   Left panel: the ground-truth column (10x30x10) flows on Y through
   stationary pre-rotated lens grids. Right panel: the inverse — lens columns
   flow through a stationary ground-truth cube. Both share one time variable;
   each panel has its own "+ Y rotation" toggle for the moving layer.

   Lenses live in a dynamic list: lens 1 (Y+15) and lens 2 (X+15 -> Y+45) are
   the schematic pair; "+ Add lens" appends more, each with its own X/Y angle
   sliders, color, visibility, and (from lens 3 on) a remove button.
   Plain canvas, no libs. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var DEG = Math.PI / 180;

  /* ---- schematic constants ---- */
  var SPACING = 1.2, COL = [10, 30, 10], CUBE = [10, 10, 10];
  var TOTAL_H = 30 * SPACING;                 // 36 — wrap height of the column
  var AXIS_LEN = 8, CAM_DIST = 22, MAX_LENSES = 8;

  /* ---- tunable state (sidebar) — defaults are the schematic ---- */
  var flowSpeed = 1.5, spinSpeed = 0.008, winY = 6;
  var showGt = true;
  var paused = false, freeCam = false;
  var camYaw = 0.6, camPitch = 0.0, camZoom = 1;

  /* ---- shared time + per-panel spin ---- */
  var time = 0, yRot1 = 0, yRot2 = 0;

  /* ---- lens list. Colors: gt is green; lenses cycle this palette ---- */
  var GT_RGB = "34, 197, 94";
  var PALETTE = ["59, 130, 246",   // blue   (lens 1, schematic)
                 "234, 179, 8",    // yellow (lens 2, schematic)
                 "168, 85, 247",   // purple
                 "6, 182, 212",    // cyan
                 "236, 72, 153",   // pink
                 "249, 115, 22",   // orange
                 "148, 163, 184",  // slate
                 "132, 204, 22"];  // lime
  function schematicLenses() {
    return [{ x: 0, y: 15, show: true },      // lens 1: Y+15 only
            { x: 15, y: 45, show: true }];    // lens 2: X+15 first, then Y+45
  }
  var lenses = schematicLenses();             // {x, y, show, cube, col}

  /* ---- grid builders ---- */
  function grid(shape) {
    var out = [], x, y, z;
    for (x = 0; x < shape[0]; x++)
      for (y = 0; y < shape[1]; y++)
        for (z = 0; z < shape[2]; z++)
          out.push([(x - shape[0] / 2 + 0.5) * SPACING,
                    (y - shape[1] / 2 + 0.5) * SPACING,
                    (z - shape[2] / 2 + 0.5) * SPACING]);
    return out;
  }

  function rotY(pts, deg) {
    var a = deg * DEG, c = Math.cos(a), s = Math.sin(a);
    return pts.map(function (p) {
      return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c];
    });
  }

  function rotX(pts, deg) {
    var a = deg * DEG, c = Math.cos(a), s = Math.sin(a);
    return pts.map(function (p) {
      return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c];
    });
  }

  var gtCol = grid(COL), gtCube = grid(CUBE);

  /* pre-rotate one lens's grids: X rotation FIRST, THEN Y — order matters */
  function bakeLens(L) {
    L.cube = rotY(rotX(grid(CUBE), L.x), L.y);   // panel 1 stationary grid
    L.col = rotY(rotX(grid(COL), L.x), L.y);     // panel 2 flowing column
  }
  lenses.forEach(bakeLens);

  /* ---- panels ---- */
  function Panel(cvId, tId, rotId) {
    this.cv = $(cvId);
    this.ctx = this.cv.getContext("2d");
    this.tEl = $(tId);
    this.rotEl = $(rotId);
  }

  Panel.prototype.fit = function () {
    var r = this.cv.getBoundingClientRect();
    this.cv.width = Math.max(50, Math.floor(r.width)) * 2;    // 2x retina
    this.cv.height = Math.max(50, Math.floor(r.height)) * 2;
  };

  Panel.prototype.project = function (p) {
    var cy = Math.cos(camYaw), sy = Math.sin(camYaw);
    var x = p[0] * cy + p[2] * sy, z = -p[0] * sy + p[2] * cy, y = p[1];
    if (freeCam) {                            // pitch only exists off-schematic
      var cp = Math.cos(camPitch), sp = Math.sin(camPitch);
      var y2 = y * cp - z * sp; z = y * sp + z * cp; y = y2;
    }
    var scale = this.cv.height * 0.6 / (CAM_DIST / camZoom + z);
    return [this.cv.width / 2 + x * scale, this.cv.height / 2 - y * scale, z];
  };

  Panel.prototype.axis = function (to, color, label) {
    var ctx = this.ctx, a = this.project([0, 0, 0]), b = this.project(to);
    ctx.strokeStyle = color; ctx.lineWidth = 4;               // 2px @ 2x
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "600 44px sans-serif";                         // 22px @ 2x
    ctx.fillText(label, b[0] + 12, b[1] + 12);
  };

  /* flow a column layer: shift on Y, wrap at +/-TOTAL_H/2, cull to the
     visible window, then spin the whole layer about Y by yRot */
  function flow(pts, yShift, yRot) {
    var c = Math.cos(yRot), s = Math.sin(yRot), out = [], i, p, y;
    for (i = 0; i < pts.length; i++) {
      p = pts[i];
      y = p[1] + yShift;
      if (y > TOTAL_H / 2) y -= TOTAL_H;
      if (y < -TOTAL_H / 2) y += TOTAL_H;
      if (y < -winY || y > winY) continue;
      out.push([p[0] * c + p[2] * s, y, -p[0] * s + p[2] * c]);
    }
    return out;
  }

  /* layers: [[points, rgbString], ...] */
  Panel.prototype.draw = function (layers) {
    var ctx = this.ctx, i, j, p;
    ctx.clearRect(0, 0, this.cv.width, this.cv.height);

    this.axis([AXIS_LEN, 0, 0], "#ef4444", "X");
    this.axis([0, AXIS_LEN, 0], "#3b82f6", "Y");
    this.axis([0, 0, AXIS_LEN], "#f59e0b", "Z");

    /* collect all dots, project, depth sort back (far, high z) to front */
    var dots = [];
    for (i = 0; i < layers.length; i++) {
      var pts = layers[i][0], rgb = layers[i][1];
      for (j = 0; j < pts.length; j++) {
        p = this.project(pts[j]);
        dots.push([p[2], p[0], p[1], rgb]);
      }
    }
    dots.sort(function (a, b) { return b[0] - a[0]; });
    for (i = 0; i < dots.length; i++) {
      var z = dots[i][0];
      var alpha = Math.max(0.2, Math.min(1, 0.3 + 0.7 * ((z + 10) / 20)));
      var r = Math.max(1.5, 4 - z * 0.08) * 2;                // @ 2x
      ctx.fillStyle = "rgba(" + dots[i][3] + "," + alpha + ")";
      ctx.beginPath();
      ctx.arc(dots[i][1], dots[i][2], r, 0, 2 * Math.PI);
      ctx.fill();
    }

    /* origin marker: red dot + halo ring */
    var o = this.project([0, 0, 0]);
    ctx.fillStyle = "#ef4444";
    ctx.beginPath(); ctx.arc(o[0], o[1], 10, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = "rgba(239, 68, 68, 0.4)"; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.arc(o[0], o[1], 20, 0, 2 * Math.PI); ctx.stroke();

    this.tEl.textContent = "t = " + time.toFixed(2);
  };

  var p1 = new Panel("rd-cv1", "rd-t1", "rd-rot1");
  var p2 = new Panel("rd-cv2", "rd-t2", "rd-rot2");

  function fitAll() { p1.fit(); p2.fit(); }
  window.addEventListener("resize", fitAll);

  /* ---- the shared animation loop ---- */
  function loop() {
    if (!paused) {
      time += 0.016;
      if (p1.rotEl.checked) yRot1 += spinSpeed;
      if (p2.rotEl.checked) yRot2 += spinSpeed;
    }
    var yShift = (time * flowSpeed) % TOTAL_H;
    var i, L;

    /* panel 1 — ground truth moves, lenses stationary */
    var L1 = [];
    if (showGt) L1.push([flow(gtCol, yShift, yRot1), GT_RGB]);
    for (i = 0; i < lenses.length; i++) {
      L = lenses[i];
      if (L.show) L1.push([L.cube, PALETTE[i % PALETTE.length]]);
    }
    p1.draw(L1);

    /* panel 2 — data stationary, lens columns move */
    var L2 = [];
    if (showGt) L2.push([gtCube, GT_RGB]);
    for (i = 0; i < lenses.length; i++) {
      L = lenses[i];
      if (L.show) L2.push([flow(L.col, yShift, yRot2), PALETTE[i % PALETTE.length]]);
    }
    p2.draw(L2);

    requestAnimationFrame(loop);
  }

  /* ---- dynamic lens controls + panel legends ---- */
  function css(rgb) { return "rgb(" + rgb + ")"; }

  function renderLegends() {
    var html = '<div><span class="sw" style="background:' + css(GT_RGB) +
               '"></span>ground truth</div>';
    for (var i = 0; i < lenses.length; i++)
      html += '<div><span class="sw" style="background:' +
              css(PALETTE[i % PALETTE.length]) + '"></span>lens ' + (i + 1) + '</div>';
    $("rd-leg1").innerHTML = html;
    $("rd-leg2").innerHTML = html;
  }

  function lensSlider(labelHtml, value, oninput) {
    var fld = document.createElement("div");
    fld.className = "rd-fld";
    var lab = document.createElement("label");
    var val = document.createElement("span");
    val.className = "val";
    val.innerHTML = value + "&deg;";
    lab.innerHTML = labelHtml;
    lab.appendChild(val);
    var rng = document.createElement("input");
    rng.type = "range"; rng.min = -90; rng.max = 90; rng.step = 1; rng.value = value;
    rng.addEventListener("input", function () {
      val.innerHTML = rng.value + "&deg;";
      oninput(parseFloat(rng.value));
    });
    fld.appendChild(lab);
    fld.appendChild(rng);
    return fld;
  }

  function renderLensControls() {
    var box = $("rd-lenses");
    box.innerHTML = "";
    lenses.forEach(function (L, i) {
      var card = document.createElement("div");
      card.className = "rd-lens";

      var hd = document.createElement("div");
      hd.className = "hd";
      var chk = document.createElement("input");
      chk.type = "checkbox"; chk.checked = L.show;
      chk.addEventListener("change", function () { L.show = chk.checked; });
      var sw = document.createElement("span");
      sw.className = "sw";
      sw.style.background = css(PALETTE[i % PALETTE.length]);
      var nm = document.createElement("span");
      nm.className = "nm";
      nm.textContent = "Lens " + (i + 1);
      hd.appendChild(chk); hd.appendChild(sw); hd.appendChild(nm);
      if (i >= 2) {                            // schematic lenses 1+2 are fixed
        var rm = document.createElement("span");
        rm.className = "rm"; rm.textContent = "× remove"; rm.title = "Remove this lens";
        rm.addEventListener("click", function () {
          lenses.splice(i, 1);
          renderLensControls(); renderLegends();
        });
        hd.appendChild(rm);
      }
      card.appendChild(hd);

      card.appendChild(lensSlider("X angle", L.x, function (v) { L.x = v; bakeLens(L); }));
      card.appendChild(lensSlider("Y angle", L.y, function (v) { L.y = v; bakeLens(L); }));
      box.appendChild(card);
    });
    $("rd-add").disabled = lenses.length >= MAX_LENSES;
  }

  $("rd-add").addEventListener("click", function () {
    if (lenses.length >= MAX_LENSES) return;
    /* seed each new lens with a distinct default angle pair */
    var n = lenses.length;
    var L = { x: (n * 20) % 90, y: (30 + n * 25) % 90, show: true };
    bakeLens(L);
    lenses.push(L);
    renderLensControls(); renderLegends();
  });

  /* ---- sidebar controls ---- */
  function bindRange(id, vId, fmt, set) {
    var el = $(id), v = $(vId);
    el.addEventListener("input", function () {
      set(parseFloat(el.value));
      v.textContent = fmt(parseFloat(el.value));
    });
  }
  bindRange("rd-flow", "rd-flow-v", function (x) { return x.toFixed(1); },
    function (x) { flowSpeed = x; });
  bindRange("rd-spin", "rd-spin-v", function (x) { return x.toFixed(3); },
    function (x) { spinSpeed = x; });
  bindRange("rd-win", "rd-win-v", function (x) { return x.toFixed(1); },
    function (x) { winY = x; });

  $("rd-pause").addEventListener("change", function () { paused = this.checked; });
  $("rd-show-gt").addEventListener("change", function () { showGt = this.checked; });
  $("rd-free").addEventListener("change", function () {
    freeCam = this.checked;
    p1.cv.style.cursor = p2.cv.style.cursor = freeCam ? "grab" : "default";
    if (!freeCam) { camYaw = 0.6; camPitch = 0; camZoom = 1; }  // relock to schematic
  });

  $("rd-reset").addEventListener("click", function () {
    flowSpeed = 1.5; spinSpeed = 0.008; winY = 6;
    lenses = schematicLenses();
    lenses.forEach(bakeLens);
    paused = false; freeCam = false; showGt = true;
    camYaw = 0.6; camPitch = 0; camZoom = 1;
    time = 0; yRot1 = 0; yRot2 = 0;
    $("rd-flow").value = 1.5; $("rd-flow-v").textContent = "1.5";
    $("rd-spin").value = 0.008; $("rd-spin-v").textContent = "0.008";
    $("rd-win").value = 6; $("rd-win-v").textContent = "6.0";
    $("rd-pause").checked = false; $("rd-free").checked = false;
    $("rd-rot1").checked = false; $("rd-rot2").checked = false;
    $("rd-show-gt").checked = true;
    renderLensControls(); renderLegends();
    p1.cv.style.cursor = p2.cv.style.cursor = "default";
  });

  /* ---- free camera: drag orbit + wheel zoom (off by default; the schematic
     view is locked) — shared camera, both panels stay in sync ---- */
  function attachCam(cv) {
    var drag = null;
    cv.addEventListener("mousedown", function (ev) {
      if (!freeCam) return;
      drag = [ev.clientX, ev.clientY];
      cv.style.cursor = "grabbing";
    });
    window.addEventListener("mousemove", function (ev) {
      if (!drag) return;
      camYaw += (ev.clientX - drag[0]) * 0.008;
      camPitch += (ev.clientY - drag[1]) * 0.008;
      camPitch = Math.max(-1.5, Math.min(1.5, camPitch));
      drag = [ev.clientX, ev.clientY];
    });
    window.addEventListener("mouseup", function () {
      drag = null;
      if (freeCam) cv.style.cursor = "grab";
    });
    cv.addEventListener("wheel", function (ev) {
      if (!freeCam) return;
      ev.preventDefault();
      camZoom *= ev.deltaY < 0 ? 1.1 : 1 / 1.1;
      camZoom = Math.max(0.3, Math.min(5, camZoom));
    }, { passive: false });
  }
  attachCam(p1.cv);
  attachCam(p2.cv);

  renderLensControls();
  renderLegends();
  fitAll();
  requestAnimationFrame(loop);
})();
