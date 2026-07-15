/* radial_cousins.js — Space Cousin Finder, built to the downloaded
   radial_space_cousin_finder.html schematic, extended with sibling/lineage
   search. Nothing runs on page load: "find cousins" and "find siblings" are
   explicit runs; each run is recorded to the runs store (Runs page, under
   the Demo_Radial environment) and its full report is downloadable as JSON.
   Pure front-end math, no models. */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var canvas = $("rc-viz");
  var ctx = canvas.getContext("2d");
  var W, H;

  function fit() {
    canvas.width = canvas.clientWidth * 2;      // 2x retina
    canvas.height = 360 * 2;
    W = canvas.width; H = canvas.height;
    drawViz();
  }
  window.addEventListener("resize", fit);

  var activations = [
    { name: "id",    fn: function (x) { return x; } },
    { name: "sin",   fn: Math.sin },
    { name: "cos",   fn: Math.cos },
    { name: "tanh",  fn: Math.tanh },
    { name: "relu",  fn: function (x) { return Math.max(0, x); } },
    { name: "abs",   fn: Math.abs },
    { name: "sq",    fn: function (x) { return x * x; } },
    { name: "sign",  fn: function (x) { return x > 0 ? 1 : x < 0 ? -1 : 0; } },
    { name: "sinc",  fn: function (x) { return x === 0 ? 1 : Math.sin(x) / x; } },
    { name: "gauss", fn: function (x) { return Math.exp(-x * x); } },
    { name: "soft",  fn: function (x) { return Math.log(1 + Math.exp(x)); } },
    { name: "step",  fn: function (x) { return x >= 0 ? 1 : 0; } },
    { name: "neg",   fn: function (x) { return -x; } },
    { name: "sqrt",  fn: function (x) { return Math.sqrt(Math.abs(x)); } }
  ];

  var N = 6;
  var testData = [];
  for (var i = 0; i < 64; i++) testData.push(-3 + 6 * i / 63);

  /* Deterministic integer hash — de-aliases the generator. The schematic's
     original arithmetic (idx%14, (idx*13)%20, (idx*17)%20) repeats every 140
     indices, so a 216-lens grid contained 76 exact duplicate programs and its
     inner parent only ever took 2 of the 14 values — the validation report
     showed the "cousins" it found were mostly that periodicity, not geometry.
     Hash-mixing keeps the address property (same index, same lens, forever)
     while making every one of the 216 programs distinct. */
  function mix(idx, salt) {
    var h = (idx ^ Math.imul(salt + 1, 0x9e3779b9)) | 0;
    h ^= h >>> 16; h = Math.imul(h, 0x45d9f3b);
    h ^= h >>> 16; h = Math.imul(h, 0x45d9f3b);
    h ^= h >>> 16;
    return h >>> 0;
  }

  function lensAt(ix, iy, iz) {
    var idx = ix * N * N + iy * N + iz;
    var a1 = activations[mix(idx, 1) % activations.length];
    var a2 = activations[mix(idx, 2) % activations.length];
    var scale = 0.5 + (mix(idx, 3) % 2000) / 1000;      // 0.5 .. 2.5
    var bias = ((mix(idx, 4) % 2001) - 1000) / 1000;    // -1 .. 1
    return {
      name: a1.name + "(" + a2.name + ")",
      fn: function (x) { return a1.fn(a2.fn(scale * x + bias)); },
      ix: ix, iy: iy, iz: iz
    };
  }

  function signature(lensFn) {
    return testData.map(function (x) {
      var v = lensFn(x);
      return isFinite(v) ? v : 0;
    });
  }

  function pearson(a, b) {
    var n = a.length, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (var i = 0; i < n; i++) {
      sx += a[i]; sy += b[i];
      sxx += a[i] * a[i]; syy += b[i] * b[i]; sxy += a[i] * b[i];
    }
    var num = n * sxy - sx * sy;
    var den = Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    return den === 0 ? 0 : num / den;
  }

  function rotateY(ix, iy, iz, angle) {
    var x = ix - N / 2 + 0.5, y = iy - N / 2 + 0.5, z = iz - N / 2 + 0.5;
    var c = Math.cos(angle), s = Math.sin(angle);
    var rx = x * c + z * s, rz = -x * s + z * c;
    return [rx + N / 2 - 0.5, y + N / 2 - 0.5, rz + N / 2 - 0.5];
  }

  function nearestGrid(fx, fy, fz) {
    var ix = Math.round(fx), iy = Math.round(fy), iz = Math.round(fz);
    if (ix < 0 || ix >= N || iy < 0 || iy >= N || iz < 0 || iz >= N) return null;
    return [ix, iy, iz];
  }

  /* ---- parents / relations (must mirror lensAt) ---- */
  function parentsOf(idx) {
    return [mix(idx, 1) % activations.length, mix(idx, 2) % activations.length];
  }

  function isMember(idx, f, mode) {
    var p = parentsOf(idx);
    if (mode === "outer") return p[0] === f;
    if (mode === "inner") return p[1] === f;
    return p[0] === f || p[1] === f;
  }

  function sharesParent(ia, ib, mode) {
    var a = parentsOf(ia), b = parentsOf(ib);
    if (mode === "outer") return a[0] === b[0];
    if (mode === "inner") return a[1] === b[1];
    return a[0] === b[0] || a[1] === b[1] || a[0] === b[1] || a[1] === b[0];
  }

  function lensIdx(l) { return l.ix * N * N + l.iy * N + l.iz; }

  /* ---- state ---- */
  var bank = null;        // {lenses, corrM, baseline} — built on first run
  var cousins = null;     // last cousin-search result
  var sibRun = false;     // has a sibling search been run
  var selFamily = -1, relMode = "either";
  var lastRun = { cousins: null, siblings: null };   // {id, report} per kind

  /* the deterministic lens bank + full |r| matrix (216 lenses -> 23k
     pearsons on 64-sample sigs, instant) */
  function ensureBank() {
    if (bank) return bank;
    var lenses = [], x, y, z;
    for (x = 0; x < N; x++)
      for (y = 0; y < N; y++)
        for (z = 0; z < N; z++) {
          var l = lensAt(x, y, z);
          l.sig = signature(l.fn);
          lenses.push(l);
        }
    var M = lenses.length, corrM = new Float32Array(M * M), sum = 0, cnt = 0;
    for (var a = 0; a < M; a++)
      for (var b = a + 1; b < M; b++) {
        var rr = Math.abs(pearson(lenses[a].sig, lenses[b].sig));
        corrM[a * M + b] = rr; corrM[b * M + a] = rr;
        sum += rr; cnt++;
      }
    bank = { lenses: lenses, corrM: corrM, baseline: sum / cnt };
    return bank;
  }

  /* ---- run recording + report download ---- */
  function recordRun(kind, params, stats, logLines, report) {
    lastRun[kind] = { id: null, report: {
      kind: kind, params: params, stats: stats, log: logLines, report: report
    } };
    var btn = kind === "cousins" ? $("rc-dl-cous") : $("rc-dl-sib");
    btn.disabled = false;
    fetch("/api/radial/demo/record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: kind, params: params, stats: stats,
                             log: logLines, report: report })
    }).then(function (r) { return r.json(); }).then(function (d) {
      if (d && d.id) lastRun[kind].id = d.id;
    }).catch(function () { /* report still downloadable client-side */ });
  }

  function downloadReport(kind) {
    var lr = lastRun[kind];
    if (!lr) return;
    if (lr.id) { window.location = "/api/radial/demo/report/" + lr.id; return; }
    /* not recorded (server down) — download straight from the page */
    var blob = new Blob([JSON.stringify(lr.report, null, 2)],
                        { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "demo_radial_" + kind + "_report.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  /* ---- cousin search (explicit run) ---- */
  function runCousins() {
    var B = ensureBank(), lenses = B.lenses;
    var angleDeg = parseInt($("rc-angle").value, 10);
    var angle = angleDeg * Math.PI / 180;
    var thresh = parseFloat($("rc-thresh").value) / 100;

    var pairs = [];
    var cousinMap = new Map();
    for (var i = 0; i < lenses.length; i++) {
      var L = lenses[i];
      var r = rotateY(L.ix, L.iy, L.iz, angle);
      var nearest = nearestGrid(r[0], r[1], r[2]);
      if (!nearest) continue;
      var tIdx = nearest[0] * N * N + nearest[1] * N + nearest[2];
      if (i === tIdx) continue;   // self-map: trivially identical, not a cousin
      var corr = Math.abs(B.corrM[i * lenses.length + tIdx]);
      if (corr >= thresh) {
        pairs.push({ from: L, to: lenses[tIdx], corr: corr });
        var key = Math.min(i, tIdx);
        if (!cousinMap.has(key)) cousinMap.set(key, new Set());
        cousinMap.set(key, cousinMap.get(key).add(i).add(tIdx));
      }
    }

    var familyCount = cousinMap.size;
    var involved = new Set();
    cousinMap.forEach(function (s) {
      s.forEach(function (v) { involved.add(v); });
    });
    var uniqueCount = lenses.length - involved.size + familyCount;
    var redund = (1 - uniqueCount / lenses.length) * 100;

    $("rc-pairs").textContent = pairs.length;
    $("rc-families").textContent = familyCount;
    $("rc-redund").textContent = redund.toFixed(1) + "%";

    pairs.sort(function (a, b) { return b.corr - a.corr; });
    var logLines = [], html = "";
    for (var k = 0; k < pairs.length; k++) {
      var p = pairs[k];
      var line = "[" + p.from.ix + "," + p.from.iy + "," + p.from.iz + "] " +
        p.from.name + " <-> [" + p.to.ix + "," + p.to.iy + "," + p.to.iz + "] " +
        p.to.name + "  r=" + p.corr.toFixed(4);
      if (k < 20) {
        html += "<div>" + line.replace("<->", "&harr;") + "</div>";
        logLines.push(line);
      }
    }
    if (pairs.length > 20)
      html += '<div class="more">... and ' + (pairs.length - 20) + " more pairs</div>";
    $("rc-log").innerHTML = html || '<div class="more">no cousin pairs at this threshold</div>';

    cousins = { pairs: pairs, involved: involved };
    if (sibRun) { renderLineage(); updateSiblings(); }   // cousin columns refresh
    drawViz();

    var stats = { pairs: pairs.length, families: familyCount,
                  redundancy_pct: +redund.toFixed(1), grid: "6x6x6=216" };
    recordRun("cousins",
      { angle_deg: angleDeg, threshold: thresh,
        generator: "hash-v2", self_maps_excluded: true },
      stats, logLines,
      { pairs: pairs.map(function (p) {
          return { from: [p.from.ix, p.from.iy, p.from.iz], from_name: p.from.name,
                   to: [p.to.ix, p.to.iy, p.to.iz], to_name: p.to.name,
                   r: +p.corr.toFixed(6) };
        }) });
  }

  /* ---- sibling / lineage search (explicit run) ---- */
  function famStats(f, mode) {
    var B = bank, M = B.lenses.length;
    var members = [], i, j;
    for (i = 0; i < M; i++) if (isMember(i, f, mode)) members.push(i);
    var sum = 0, cnt = 0;
    for (i = 0; i < members.length; i++)
      for (j = i + 1; j < members.length; j++) {
        sum += B.corrM[members[i] * M + members[j]]; cnt++;
      }
    var cous = 0;
    if (cousins)
      for (i = 0; i < members.length; i++)
        if (cousins.involved.has(members[i])) cous++;
    return { members: members, pairs: cnt,
             meanR: cnt ? sum / cnt : 0, cousins: cous };
  }

  function lineageRows() {
    var rows = [];
    activations.forEach(function (act, f) {
      var outer = 0, inner = 0, i;
      for (i = 0; i < bank.lenses.length; i++) {
        var p = parentsOf(i);
        if (p[0] === f) outer++;
        if (p[1] === f) inner++;
      }
      var st = famStats(f, relMode);
      rows.push({ parent: act.name, outer: outer, inner: inner,
                  members: st.members.length,
                  sibling_r: +st.meanR.toFixed(4),
                  delta_vs_base: +(st.meanR - bank.baseline).toFixed(4),
                  rotation_cousins: cousins ? st.cousins : null });
    });
    return rows;
  }

  function renderLineage() {
    var tb = document.querySelector("#rc-lineage tbody");
    tb.innerHTML = "";
    lineageRows().forEach(function (row, f) {
      var d = row.delta_vs_base;
      var tr = document.createElement("tr");
      if (f === selFamily) tr.className = "sel";
      tr.innerHTML =
        "<td>" + row.parent + "</td><td>" + row.outer + "</td><td>" + row.inner +
        "</td><td>" + row.members + "</td><td>" + row.sibling_r.toFixed(3) +
        '</td><td class="' + (d > 0.02 ? "hi" : d < -0.02 ? "lo" : "") + '">' +
        (d >= 0 ? "+" : "") + d.toFixed(3) + "</td><td>" +
        (row.rotation_cousins === null ? "&mdash;"
          : row.rotation_cousins + "/" + row.members) + "</td>";
      tr.addEventListener("click", function () {
        selFamily = (selFamily === f) ? -1 : f;
        $("rc-parent").value = String(selFamily);
        renderLineage(); updateSiblings(); drawViz();
      });
      tb.appendChild(tr);
    });
  }

  function siblingPairs() {
    var B = bank, M = B.lenses.length, out = [], i, j;
    if (selFamily >= 0) {
      var members = famStats(selFamily, relMode).members;
      for (i = 0; i < members.length; i++)
        for (j = i + 1; j < members.length; j++)
          out.push([members[i], members[j], B.corrM[members[i] * M + members[j]]]);
    } else {
      for (i = 0; i < M; i++)
        for (j = i + 1; j < M; j++)
          if (sharesParent(i, j, relMode))
            out.push([i, j, B.corrM[i * M + j]]);
    }
    out.sort(function (a, b) { return b[2] - a[2]; });
    return out;
  }

  function updateSiblings() {
    var B = bank, lenses = B.lenses;
    var sibs = siblingPairs();
    var i, sum = 0;
    for (i = 0; i < sibs.length; i++) sum += sibs[i][2];
    var meanR = sibs.length ? sum / sibs.length : 0;

    var members = selFamily >= 0
      ? famStats(selFamily, relMode).members.length : lenses.length;
    $("rc-members").textContent = members;
    $("rc-sibpairs").textContent = sibs.length;
    $("rc-sibr").textContent = meanR.toFixed(3) + " / " + B.baseline.toFixed(3);

    /* are the rotation-cousins disproportionately relatives?
       (needs a cousin run; self-maps excluded) */
    var relStat = null;
    if (cousins) {
      var rel = 0, real = 0;
      for (i = 0; i < cousins.pairs.length; i++) {
        var p = cousins.pairs[i];
        var ia = lensIdx(p.from), ib = lensIdx(p.to);
        if (ia === ib) continue;
        real++;
        if (sharesParent(ia, ib, relMode)) rel++;
      }
      relStat = { pct: real ? +(100 * rel / real).toFixed(1) : 0,
                  relatives: rel, real_pairs: real };
      $("rc-relcous").textContent =
        real ? (relStat.pct + "% (" + rel + "/" + real + ")") : "no pairs";
    } else {
      $("rc-relcous").innerHTML = "run cousins first";
    }

    var logLines = [], html = "";
    for (i = 0; i < Math.min(20, sibs.length); i++) {
      var A = lenses[sibs[i][0]], Bl = lenses[sibs[i][1]];
      var line = "[" + A.ix + "," + A.iy + "," + A.iz + "] " + A.name +
        " <-> [" + Bl.ix + "," + Bl.iy + "," + Bl.iz + "] " + Bl.name +
        "  r=" + sibs[i][2].toFixed(4);
      html += "<div>" + line.replace("<->", "&harr;") + "</div>";
      logLines.push(line);
    }
    if (sibs.length > 20)
      html += '<div class="more">... and ' + (sibs.length - 20) +
              " more sibling pairs</div>";
    $("rc-siblog").innerHTML = html || '<div class="more">no sibling pairs</div>';

    return { sibs: sibs, meanR: meanR, members: members,
             logLines: logLines, relStat: relStat };
  }

  function runSiblings() {
    ensureBank();
    sibRun = true;
    var view = updateSiblings();
    renderLineage();
    drawViz();

    var B = bank;
    var parentName = selFamily >= 0 ? activations[selFamily].name : "all";
    var stats = {
      parent: parentName, relation: relMode,
      family_members: view.members, sibling_pairs: view.sibs.length,
      mean_sibling_r: +view.meanR.toFixed(4),
      baseline_r: +B.baseline.toFixed(4),
      cousins_that_are_relatives: view.relStat
    };
    var CAP = 500;
    recordRun("siblings",
      { parent: parentName, relation: relMode, generator: "hash-v2",
        cousin_context: cousins
          ? { angle_deg: parseInt($("rc-angle").value, 10),
              threshold: parseFloat($("rc-thresh").value) / 100 }
          : null },
      stats, view.logLines,
      { lineage_table: lineageRows(),
        sibling_pairs: view.sibs.slice(0, CAP).map(function (s) {
          var A = B.lenses[s[0]], Bl = B.lenses[s[1]];
          return { a: [A.ix, A.iy, A.iz], a_name: A.name,
                   b: [Bl.ix, Bl.iy, Bl.iz], b_name: Bl.name,
                   r: +s[2].toFixed(6) };
        }),
        sibling_pairs_truncated: Math.max(0, view.sibs.length - CAP) });
  }

  /* ---- visualization ---- */
  function drawViz() {
    ctx.clearRect(0, 0, W, H);
    if (!bank) {
      ctx.font = "500 26px sans-serif";
      ctx.fillStyle = "#6c7885";
      ctx.textAlign = "center";
      ctx.fillText("press find cousins or find siblings to run", W / 2, H / 2);
      ctx.textAlign = "start";
      return;
    }
    var lenses = bank.lenses;
    var camAy = 0.7;

    function proj(ix, iy, iz, side) {
      var x = (ix - N / 2 + 0.5) * 2, y = (iy - N / 2 + 0.5) * 2, z = (iz - N / 2 + 0.5) * 2;
      var c = Math.cos(camAy), s = Math.sin(camAy);
      var xr = x * c + z * s, zr = -x * s + z * c;
      var sc = H * 0.35 / (16 + zr);
      var cx = side === "left" ? W * 0.25 : W * 0.75;
      return [cx + xr * sc, H / 2 - y * sc, zr];
    }

    ctx.font = "500 24px sans-serif";
    ctx.fillStyle = "#8b95a1";
    ctx.textAlign = "center";
    ctx.fillText("original", W * 0.25, 30);
    ctx.fillText("rotated", W * 0.75, 30);
    ctx.textAlign = "start";

    var allPts = [], i;
    for (i = 0; i < lenses.length; i++) {
      var l = lenses[i];
      var isCousin = cousins ? cousins.involved.has(i) : false;
      var isFam = selFamily >= 0 && sibRun && isMember(i, selFamily, relMode);
      var a = proj(l.ix, l.iy, l.iz, "left");
      allPts.push({ x: a[0], y: a[1], z: a[2], cousin: isCousin, fam: isFam, side: "left" });
      var b = proj(l.ix, l.iy, l.iz, "right");
      allPts.push({ x: b[0], y: b[1], z: b[2], cousin: isCousin, fam: isFam, side: "right" });
    }
    allPts.sort(function (a, b) { return a.z - b.z; });
    for (i = 0; i < allPts.length; i++) {
      var pt = allPts[i];
      var alpha = 0.3 + 0.7 * ((pt.z + 8) / 16);
      var size = Math.max(2.5, 5 - pt.z * 0.12);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, size, 0, Math.PI * 2);
      if (pt.cousin) {
        ctx.fillStyle = pt.side === "left"
          ? "rgba(234,179,8," + Math.min(1, Math.max(0.3, alpha)) + ")"
          : "rgba(168,85,247," + Math.min(1, Math.max(0.3, alpha)) + ")";
      } else {
        ctx.fillStyle = "rgba(100,100,100," + Math.min(0.4, Math.max(0.1, alpha * 0.4)) + ")";
      }
      ctx.fill();
      if (pt.fam) {                            // selected lineage family: cyan ring
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, size + 3, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(6,182,212," + Math.min(1, Math.max(0.5, alpha)) + ")";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    if (cousins) {
      var topPairs = cousins.pairs.slice(0, 8);
      for (i = 0; i < topPairs.length; i++) {
        var p = topPairs[i];
        var p1 = proj(p.from.ix, p.from.iy, p.from.iz, "left");
        var p2 = proj(p.to.ix, p.to.iy, p.to.iz, "right");
        ctx.beginPath();
        ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p2[0], p2[1]);
        ctx.strokeStyle = "rgba(234,179,8," + (0.15 + p.corr * 0.3) + ")";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
  }

  /* ---- controls ---- */
  $("rc-angle").addEventListener("input", function (e) {
    $("rc-angle-out").innerHTML = e.target.value + "&deg;";
  });
  $("rc-thresh").addEventListener("input", function (e) {
    $("rc-thresh-out").textContent = (e.target.value / 100).toFixed(3);
  });
  $("rc-run").addEventListener("click", runCousins);
  $("rc-run-sib").addEventListener("click", runSiblings);
  $("rc-dl-cous").addEventListener("click", function () { downloadReport("cousins"); });
  $("rc-dl-sib").addEventListener("click", function () { downloadReport("siblings"); });

  var parentSel = $("rc-parent");
  activations.forEach(function (act, f) {
    var o = document.createElement("option");
    o.value = String(f);
    o.textContent = act.name;
    parentSel.appendChild(o);
  });
  parentSel.addEventListener("change", function () {
    selFamily = parseInt(parentSel.value, 10);
    if (sibRun) { renderLineage(); updateSiblings(); drawViz(); }
  });
  var radios = document.querySelectorAll('input[name="rc-rel"]');
  for (var ri = 0; ri < radios.length; ri++) {
    radios[ri].addEventListener("change", function () {
      relMode = this.value;
      if (sibRun) { renderLineage(); updateSiblings(); drawViz(); }
    });
  }

  fit();   // draws the "press run" placeholder — nothing runs automatically
})();
