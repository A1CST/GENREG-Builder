/* /tsdb page — drive the in-browser TSDB port with real pipeline metrics
 * (MNIST or CIFAR), then push the store's limits with a large stress load.
 *
 * Flow: fetch /api/tsdb/data?set=… -> serialize each series into TSDB blocks ->
 * flush the manifest -> read every block BACK out and render from the read-back
 * rows (never from the fetched JSON, so the round-trip is real). "Verify"
 * re-reads and diffs against source; "Push limits" hammers the store with
 * many out-of-order blocks and reports throughput. Both post to the Agent panel.
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const fmt = (x, d = 4) => (x == null ? '—' : Number(x).toFixed(d));
  const kb = (n) => (n < 1024 ? n + ' B'
    : n < 1048576 ? (n / 1024).toFixed(1) + ' KB'
      : (n / 1048576).toFixed(2) + ' MB');

  let DATA = null;   // last fetched payload
  let DB = null;     // the live TSDBMem
  let SET = 'mnist';

  function setConn(ok, msg) {
    const dot = $('ts-dot');
    if (dot) dot.style.background = ok ? '#3fb950' : '#f85149';
    $('ts-conn').textContent = msg;
  }

  // ── build the store from the real pipeline series ──────────────────
  function build(data) {
    const db = new window.TSDBMem(data.set);

    const layerKm = db.keyMap('time,acc');
    db.createKey('layers', 'stage', layerKm, true);
    const layerRows = data.layers.map((r) => ({ time: r.t, acc: r.acc }));
    const layerBuf = db.serialize(layerKm, layerRows);
    db.append('layers', 'stage',
      db.createBlock(layerKm, layerBuf, { series: 'layers' }), layerBuf);

    const pairKm = db.keyMap('time,a,b,acc');
    db.createKey('pairs', 'ovo', pairKm, true);
    const pairRows = data.pairs.map((r) => ({ time: r.t, a: r.a, b: r.b, acc: r.acc }));
    const pairBuf = db.serialize(pairKm, pairRows);
    db.append('pairs', 'ovo',
      db.createBlock(pairKm, pairBuf, { series: 'ovo', n: pairRows.length }), pairBuf);

    if (data.detectors && data.detectors.length) {
      const detKm = db.keyMap('time,cls,acc');
      db.createKey('detectors', 'ovr', detKm, true);
      const detRows = data.detectors.map((r) => ({ time: r.t, cls: r.cls, acc: r.acc }));
      const detBuf = db.serialize(detKm, detRows);
      db.append('detectors', 'ovr',
        db.createBlock(detKm, detBuf, { series: 'ovr' }), detBuf);
    }

    db.flush();
    return db;
  }

  function readKey(db, name, key) {
    const km = db.meta.files[name].keys[key].km;
    const out = [];
    for (const block of db.getBlocks(name, key)) {
      out.push(...db.parse(km, db.bf(name, key, block)));
    }
    return out;
  }

  // ── renders (all fed from read-back rows) ──────────────────────────
  function bars(host, items, labelOf, valOf, max) {
    host.innerHTML = '';
    const hi = max || Math.max(...items.map(valOf), 1e-9);
    items.forEach((it) => {
      const v = valOf(it);
      const row = document.createElement('div');
      row.className = 'tsdb-bar-row';
      row.innerHTML =
        `<span class="tsdb-bar-lbl">${labelOf(it)}</span>` +
        `<span class="tsdb-bar-track"><span class="tsdb-bar-fill" style="width:${((v / hi) * 100).toFixed(1)}%"></span></span>` +
        `<span class="tsdb-bar-val">${fmt(v)}</span>`;
      host.appendChild(row);
    });
  }

  function renderMatrix(rows, classes) {
    const host = $('ts-matrix');
    const M = Array.from({ length: 10 }, () => Array(10).fill(null));
    let lo = 1, hi = 0;
    rows.forEach((r) => {
      const a = r.a | 0, b = r.b | 0;
      M[a][b] = r.acc; M[b][a] = r.acc;
      lo = Math.min(lo, r.acc); hi = Math.max(hi, r.acc);
    });
    const span = Math.max(hi - lo, 1e-6);
    const head = (classes || []).map((c) => c.slice(0, 4));
    let html = '<table class="tsdb-matrix"><tr><th></th>';
    for (let c = 0; c < 10; c++) html += `<th>${head[c] ?? c}</th>`;
    html += '</tr>';
    for (let rr = 0; rr < 10; rr++) {
      html += `<tr><th>${head[rr] ?? rr}</th>`;
      for (let c = 0; c < 10; c++) {
        if (M[rr][c] == null) { html += '<td style="background:rgba(255,255,255,.02)"></td>'; continue; }
        const t = 1 - (M[rr][c] - lo) / span;   // 0 easy .. 1 hard
        const alpha = (0.08 + 0.72 * t).toFixed(3);
        html += `<td title="${(head[rr] ?? rr)} vs ${(head[c] ?? c)}: ${fmt(M[rr][c])}" ` +
          `style="background:rgba(78,161,255,${alpha})">${(M[rr][c] * 100).toFixed(0)}</td>`;
      }
      html += '</tr>';
    }
    host.innerHTML = html + '</table>';
  }

  function renderManifest(db, rowCount) {
    $('ts-bytes').textContent = kb(db.physicalBytes());
    $('ts-rows').textContent = rowCount.toLocaleString();
    let files = 0, keys = 0, blocks = 0;
    for (const fn in db.meta.files) {
      files++;
      for (const kn in db.meta.files[fn].keys) {
        keys++; blocks += db.meta.files[fn].keys[kn].blocks.length;
      }
    }
    $('ts-fkb').textContent = `${files} / ${keys} / ${blocks}`;
    $('ts-manifest').textContent = JSON.stringify(JSON.parse(db.saved), null, 2);
  }

  function verify(db, data) {
    const near = (x, y) => Math.abs(x - y) < 1e-9;
    let bad = 0, checks = 0;
    const L = readKey(db, 'layers', 'stage');
    if (L.length !== data.layers.length) bad++;
    L.forEach((r, i) => { checks++; if (!near(r.acc, data.layers[i].acc)) bad++; });
    const P = readKey(db, 'pairs', 'ovo');
    if (P.length !== data.pairs.length) bad++;
    P.forEach((r, i) => {
      checks += 3; const s = data.pairs[i];
      if (r.a !== s.a) bad++; if (r.b !== s.b) bad++; if (!near(r.acc, s.acc)) bad++;
    });
    if (data.detectors && data.detectors.length) {
      const D = readKey(db, 'detectors', 'ovr');
      if (D.length !== data.detectors.length) bad++;
      D.forEach((r, i) => { checks++; if (!near(r.acc, data.detectors[i].acc)) bad++; });
    }
    return { ok: bad === 0, bad, checks };
  }

  // Record the op as a `tsdb` run AND post the notice (carrying the run_id, so
  // the Agent-panel row deep-links to it on /runs). Returns the new run_id.
  async function recordRun(op, title, body, kind, ok, metrics) {
    try {
      const r = await fetch('/api/tsdb/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ op, title, body, kind, ok, metrics }),
      });
      const j = await r.json();
      return j.run_id || null;
    } catch (e) { return null; }       // recording offline is non-fatal
  }

  function rebuild() {
    if (!DATA) return;
    DB = build(DATA);
    const layers = readKey(DB, 'layers', 'stage');
    const pairs = readKey(DB, 'pairs', 'ovo');
    const dets = DATA.detectors && DATA.detectors.length ? readKey(DB, 'detectors', 'ovr') : [];
    bars($('ts-layers'), layers,
      (r) => DATA.layer_labels[r.time] || ('t=' + r.time), (r) => r.acc);
    renderMatrix(pairs, DATA.classes);
    const detCard = $('ts-det-card');
    if (dets.length) {
      detCard.style.display = '';
      bars($('ts-det'), dets,
        (r) => (DATA.classes[r.cls] || r.cls), (r) => r.acc, 1);
    } else {
      detCard.style.display = 'none';
    }
    renderManifest(DB, layers.length + pairs.length + dets.length);
    $('ts-rt').innerHTML = '<span class="tsdb-note">built · click Verify</span>';
    $('ts-status').textContent =
      `${SET} · ${kb(DB.physicalBytes())} over ${layers.length + pairs.length + dets.length} rows`;
  }

  async function doVerify() {
    if (!DB || !DATA) return;
    const v = verify(DB, DATA);
    const bytes = DB.physicalBytes();
    if (v.ok) {
      $('ts-rt').innerHTML = `<span class="tsdb-ok">${v.checks} checks pass</span>`;
      await recordRun('verify', 'TSDB round-trip verified (' + SET + ')',
        `${v.checks} Float64 read-back checks pass across layers + pairs` +
        (DATA.detectors && DATA.detectors.length ? ' + detectors' : '') +
        ` (${DATA.pairs.length} one-vs-one, joint_val_acc ${fmt(DATA.joint_val_acc)}). ` +
        `Physical store ${kb(bytes)}.`, 'test', true,
        { set: SET, checks: v.checks, bytes, score: v.checks });
      $('ts-status').textContent = 'round-trip verified · recorded run · posted to Agent panel';
    } else {
      $('ts-rt').innerHTML = `<span class="tsdb-bad">${v.bad} mismatch</span>`;
      await recordRun('verify', 'TSDB round-trip FAILED (' + SET + ')',
        `${v.bad} of ${v.checks} read-back checks mismatched.`, 'alert', false,
        { set: SET, bad: v.bad, checks: v.checks, score: v.bad });
      $('ts-status').textContent = 'round-trip FAILED · recorded run · see Agent panel';
    }
  }

  // ── push limits: many out-of-order blocks, timed, checksum-verified ─
  // Deterministic value hash so we can recompute the expected checksum
  // without storing every row — a real end-to-end integrity check at scale.
  function cellVal(t, lane) {
    const x = Math.sin((t + 1) * 12.9898 + lane * 78.233) * 43758.5453;
    return x - Math.floor(x);          // [0,1), fully reproducible
  }

  async function stress() {
    const nBlocks = Math.max(1, Math.min(4000, +$('ts-nblocks').value || 400));
    const perBlk = Math.max(1, Math.min(5000, +$('ts-perblk').value || 250));
    const LANES = 4;                    // schema: time,v0,v1,v2 (v3 == time)
    const out = $('ts-stress-out');
    out.textContent = 'running…';
    $('ts-stress').disabled = true;
    // let the "running…" paint before we block the main thread
    await new Promise((r) => setTimeout(r, 20));

    const db = new window.TSDBMem('stress');
    const km = db.keyMap('time,v0,v1,v2');
    db.createKey('load', 'series', km, true);

    // build blocks with RANDOM begin-times to force mid-array sorted inserts
    const starts = [];
    for (let i = 0; i < nBlocks; i++) starts.push(Math.floor(Math.random() * 1e9));

    let expSum = 0, totalRows = 0;
    const tGen0 = performance.now();
    const buffers = [];
    for (let b = 0; b < nBlocks; b++) {
      const base = starts[b];
      const rows = new Array(perBlk);
      for (let i = 0; i < perBlk; i++) {
        const t = base + i;
        const r = { time: t, v0: cellVal(t, 0), v1: cellVal(t, 1), v2: cellVal(t, 2) };
        expSum += r.v0 + r.v1 + r.v2;
        rows[i] = r;
      }
      totalRows += perBlk;
      buffers.push({ base, buf: db.serialize(km, rows, null, 0, true) });
    }
    const genMs = performance.now() - tGen0;

    const tApp0 = performance.now();
    for (const { base, buf } of buffers) {
      db.append('load', 'series', db.createBlock(km, buf, { base }, base), buf);
    }
    db.flush();
    const appMs = performance.now() - tApp0;

    // read every block back, confirm block order is sorted, sum a checksum
    const tRd0 = performance.now();
    const blocks = db.getBlocks('load', 'series');
    let gotSum = 0, gotRows = 0, sorted = true, prev = -Infinity;
    for (const blk of blocks) {
      if (blk.begin < prev) sorted = false;
      prev = blk.begin;
      const rows = db.parse(km, db.bf('load', 'series', blk));
      for (const r of rows) { gotSum += r.v0 + r.v1 + r.v2; gotRows++; }
    }
    const rdMs = performance.now() - tRd0;

    const bytes = db.physicalBytes();
    const drift = Math.abs(expSum - gotSum);
    const ok = sorted && gotRows === totalRows && drift < 1e-6;
    const mb = bytes / 1048576;
    const wrMBs = (mb / (appMs / 1000)).toFixed(1);
    const rdMBs = (mb / (rdMs / 1000)).toFixed(1);

    out.innerHTML =
      `<span class="${ok ? 'tsdb-ok' : 'tsdb-bad'}">${ok ? 'INTEGRITY OK' : 'INTEGRITY FAIL'}</span>  ` +
      `${nBlocks.toLocaleString()} blocks × ${perBlk} rows = ` +
      `<b>${totalRows.toLocaleString()} rows</b> / ${kb(bytes)}\n` +
      `blocks sorted by begin: ${sorted ? 'yes' : 'NO'} · rows read ${gotRows.toLocaleString()}/${totalRows.toLocaleString()} · checksum drift ${drift.toExponential(1)}\n` +
      `serialize ${genMs.toFixed(0)} ms · append+flush ${appMs.toFixed(0)} ms (${wrMBs} MB/s) · read+parse ${rdMs.toFixed(0)} ms (${rdMBs} MB/s)`;

    await recordRun('stress', 'TSDB stress ' + (ok ? 'passed' : 'FAILED'),
      `${totalRows.toLocaleString()} rows in ${nBlocks} out-of-order blocks ` +
      `(${kb(bytes)}). Sorted-insert held: ${sorted}. Checksum drift ${drift.toExponential(1)}. ` +
      `append ${wrMBs} MB/s, read ${rdMBs} MB/s.`, ok ? 'test' : 'alert', ok,
      { set: SET, rows: totalRows, blocks: nBlocks, bytes,
        write_mbs: +wrMBs, read_mbs: +rdMBs, sorted, score: totalRows });

    $('ts-stress').disabled = false;
  }

  async function load(which) {
    SET = which;
    document.querySelectorAll('.ts-set').forEach((b) =>
      b.classList.toggle('active', b.dataset.set === which));
    setConn(false, 'loading ' + which + '…');
    try {
      const r = await fetch('/api/tsdb/data?set=' + encodeURIComponent(which));
      const data = await r.json();
      if (data.err) throw new Error(data.err);
      DATA = data;
      setConn(true, `${which} · feat v${data.feat_version} · ` +
        `${data.pairs.length} pairs${data.detectors && data.detectors.length ? ' · ' + data.detectors.length + ' detectors' : ''}`);
      rebuild();
    } catch (e) {
      setConn(false, 'error');
      $('ts-status').textContent = 'load failed: ' + e.message;
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    $('ts-rebuild').addEventListener('click', rebuild);
    $('ts-verify').addEventListener('click', doVerify);
    $('ts-stress').addEventListener('click', stress);
    document.querySelectorAll('.ts-set').forEach((b) =>
      b.addEventListener('click', () => load(b.dataset.set)));
    load('mnist');
  });
})();
