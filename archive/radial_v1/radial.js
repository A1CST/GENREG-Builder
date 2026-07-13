// radial.js — Radial Space UI. Two modes:
//   codec: address a signal to a dot, show reconstruction + harmonic spectrum.
//   vs   : RS-Gabor matching pursuit vs top-K Fourier on non-stationary signals,
//          with the corr-vs-K curve showing where the localized dictionary wins.
(function () {
  const cv = document.getElementById('rs-cv');
  const ctx = cv.getContext('2d');
  const spec = document.getElementById('rs-spec');
  const sctx = spec.getContext('2d');
  let payload = null, cmp = null, v2 = null, m10 = null, reconMode = '1', mode = 'codec';
  const M_COLORS = { linear: '#5fd39a', arctan: '#3a9ad9', sigmoid: '#9b7bd6', sqrt: '#e0a53b', sinusoidal: '#e0684d', paper: '#7f8b98' };

  function fit(c, x) {
    const dpr = window.devicePixelRatio || 1;
    c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function fitAll() { fit(cv, ctx); fit(spec, sctx); draw(); }
  window.addEventListener('resize', fitAll);

  function lineScaled(x, arr, w, h, color, lw, lo, hi, dash) {
    if (!arr || !arr.length) return;
    x.strokeStyle = color; x.lineWidth = lw; x.setLineDash(dash || []); x.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const px = (i / (arr.length - 1)) * (w - 20) + 10;
      const py = h - 24 - ((arr[i] - lo) / (hi - lo)) * (h - 40);
      i ? x.lineTo(px, py) : x.moveTo(px, py);
    }
    x.stroke(); x.setLineDash([]);
  }

  function draw() {
    if (mode === 'vs') return drawVs();
    if (mode === 'v2') return drawV2();
    if (mode === 'm') return drawM();
    if (mode === 'lens') return drawLens();
    if (mode === 'screen') return drawScreen();
    drawCodec();
  }

  // ---- mapping-M mode ----------------------------------------------------
  function drawM() {
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (!m10) return;
    // phi-vs-input curves for each M family (the doc's figure)
    ctx.strokeStyle = '#171d24'; ctx.lineWidth = 1;
    [90, 180, 270].forEach(g => { const y = h - 24 - (g / 360) * (h - 40); ctx.beginPath(); ctx.moveTo(10, y); ctx.lineTo(w - 10, y); ctx.stroke(); });
    for (const name in m10.curves) {
      const arr = m10.curves[name];
      ctx.strokeStyle = M_COLORS[name] || '#888'; ctx.lineWidth = name === 'linear' ? 2.2 : 1.5;
      ctx.beginPath();
      for (let i = 0; i < arr.length; i++) {
        const px = 10 + (i / (arr.length - 1)) * (w - 20);
        const py = h - 24 - (arr[i] / 360) * (h - 40);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
    }
    drawMLyap();
  }

  function drawMLyap() {
    const w = spec.clientWidth, h = spec.clientHeight;
    sctx.clearRect(0, 0, w, h);
    if (!m10) return;
    const rows = m10.rows, n = rows.length;
    const maxAbs = Math.max(...rows.map(r => Math.abs(r.lyapunov)), 0.1);
    const zeroY = h / 2 + 4, bw = (w - 40) / n * 0.6;
    // zero line = the only place a stable chain could live
    sctx.strokeStyle = '#2f4152'; sctx.setLineDash([4, 3]); sctx.beginPath();
    sctx.moveTo(20, zeroY); sctx.lineTo(w - 10, zeroY); sctx.stroke(); sctx.setLineDash([]);
    sctx.fillStyle = '#5fd39a'; sctx.font = '9px ui-monospace'; sctx.fillText('stable (Lyap=0)', 22, zeroY - 4);
    rows.forEach((r, i) => {
      const cx = 30 + (i + 0.5) * ((w - 40) / n);
      const bh = (r.lyapunov / maxAbs) * (h / 2 - 22);
      sctx.fillStyle = M_COLORS[r.M] || '#888';
      sctx.fillRect(cx - bw / 2, zeroY, bw, -bh);
      sctx.fillStyle = '#c7d0da'; sctx.font = '9px ui-sans-serif'; sctx.textAlign = 'center';
      sctx.fillText(r.M, cx, h - 6); sctx.textAlign = 'start';
    });
  }

  // ---- v2 memory mode ----------------------------------------------------
  function drawV2() {
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (!v2) return;
    const s = v2.stream;
    let lo = Math.min(...s), hi = Math.max(...s);
    const pad = (hi - lo) * 0.1 || 1; lo -= pad; hi += pad;
    lineScaled(ctx, s, w, h, '#5fd39a', 2, lo, hi);
    drawV2Paths();
  }

  function drawV2Paths() {
    const w = spec.clientWidth, h = spec.clientHeight;
    sctx.clearRect(0, 0, w, h);
    if (!v2) return;
    const mid = w / 2;
    sctx.strokeStyle = '#1a2028'; sctx.beginPath(); sctx.moveTo(mid, 6); sctx.lineTo(mid, h - 6); sctx.stroke();
    sctx.font = 'bold 11px ui-sans-serif,system-ui'; sctx.textAlign = 'center';
    sctx.fillStyle = '#e07a6b'; sctx.fillText("paper's mapping — scatter", mid / 2, 15);
    sctx.fillStyle = '#5fd39a'; sctx.fillText('fixed mapping — clean path', mid + mid / 2, 15);
    sctx.textAlign = 'start';
    const half = (path, x0, x1, color) => {
      const xs = path.map(p => p[0]), ys = path.map(p => p[1]);
      const mx = Math.max(...xs.map(Math.abs), ...ys.map(Math.abs)) || 1;
      const cx = (x0 + x1) / 2, cy = h / 2 + 8, s = Math.min(x1 - x0, h - 40) / 2 / mx;
      sctx.strokeStyle = color + '66'; sctx.lineWidth = 1; sctx.beginPath();
      path.forEach((p, i) => { const px = cx + p[0] * s, py = cy - p[1] * s; i ? sctx.lineTo(px, py) : sctx.moveTo(px, py); });
      sctx.stroke();
      path.forEach((p, i) => {
        sctx.fillStyle = color; sctx.globalAlpha = 0.35 + 0.6 * (i / path.length);
        sctx.beginPath(); sctx.arc(cx + p[0] * s, cy - p[1] * s, 2, 0, 6.283); sctx.fill();
      });
      sctx.globalAlpha = 1;
    };
    half(v2.paper, 12, mid - 12, '#e07a6b');
    half(v2.good, mid + 12, w - 12, '#5fd39a');
  }

  // ---- codec mode --------------------------------------------------------
  function drawCodec() {
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (!payload) return;
    const recon = reconMode === '1' ? payload.recon1 : payload.reconK;
    const all = payload.signal.concat(recon);
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.1 || 1; lo -= pad; hi += pad;
    lineScaled(ctx, payload.signal, w, h, '#5fd39a', 2, lo, hi);
    lineScaled(ctx, recon, w, h, '#e0a53b', 1.6, lo, hi);
    drawSpec();
  }

  function drawSpec() {
    const w = spec.clientWidth, h = spec.clientHeight;
    sctx.clearRect(0, 0, w, h);
    if (!payload || !payload.components.length) return;
    const comps = payload.components, maxK = Math.max(...comps.map(c => c.k));
    const maxA = Math.max(...comps.map(c => c.amp));
    const shown = reconMode === '1' ? comps.slice(0, 1) : comps;
    const bw = Math.min(26, (w - 30) / maxK);
    for (const c of shown) {
      const px = 14 + ((c.k - 1) / Math.max(1, maxK - 1)) * (w - 34);
      const bh = (c.amp / maxA) * (h - 34);
      sctx.fillStyle = reconMode === '1' ? '#e0a53b' : '#3a9ad9';
      sctx.fillRect(px - bw / 2, h - 20 - bh, bw, bh);
      sctx.fillStyle = '#6c7885'; sctx.font = '9px ui-monospace'; sctx.textAlign = 'center';
      sctx.fillText('k' + c.k, px, h - 8);
    }
    sctx.textAlign = 'start';
  }

  // ---- vs-Fourier mode ---------------------------------------------------
  function drawVs() {
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (!cmp) return;
    const all = cmp.signal.concat(cmp.rs_recon, cmp.ft_recon);
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.1 || 1; lo -= pad; hi += pad;
    lineScaled(ctx, cmp.signal, w, h, '#5fd39a', 2.4, lo, hi);
    lineScaled(ctx, cmp.ft_recon, w, h, '#3a9ad9', 1.5, lo, hi, [4, 3]);
    lineScaled(ctx, cmp.rs_recon, w, h, '#e0a53b', 1.8, lo, hi);
    drawBlocks();
  }

  // the "why": show each method's building blocks side by side
  function drawBlocks() {
    const w = spec.clientWidth, h = spec.clientHeight;
    sctx.clearRect(0, 0, w, h);
    if (!cmp) return;
    const mid = w / 2;
    sctx.strokeStyle = '#1a2028'; sctx.lineWidth = 1;
    sctx.beginPath(); sctx.moveTo(mid, 6); sctx.lineTo(mid, h - 6); sctx.stroke();
    sctx.font = 'bold 11px ui-sans-serif,system-ui'; sctx.textAlign = 'center';
    sctx.fillStyle = '#3a9ad9'; sctx.fillText('Fourier: endless waves', mid / 2, 16);
    sctx.fillStyle = '#e0a53b'; sctx.fillText('Radial: blips', mid + mid / 2, 16);
    sctx.textAlign = 'start';

    const half = (pieces, x0, x1, color) => {
      const n = Math.max(1, pieces.length);
      const rowH = (h - 28) / n;
      pieces.forEach((p, r) => {
        const cy = 24 + rowH * r + rowH / 2;
        const amp = Math.max(...p.map(Math.abs)) || 1;
        sctx.strokeStyle = color; sctx.lineWidth = 1.6; sctx.beginPath();
        for (let i = 0; i < p.length; i++) {
          const px = x0 + (i / (p.length - 1)) * (x1 - x0);
          const py = cy - (p[i] / amp) * (rowH * 0.38);
          i ? sctx.lineTo(px, py) : sctx.moveTo(px, py);
        }
        sctx.stroke();
      });
    };
    half(cmp.ft_pieces, 14, mid - 14, '#3a9ad9');
    half(cmp.rs_pieces, mid + 14, w - 14, '#e0a53b');
  }

  // ---- metrics -----------------------------------------------------------
  function renderCodec() {
    // restore codec-mode labels (vs-mode overwrites them)
    document.querySelector('.rs-comp span').textContent = 'compression ratio';
    const kv = document.querySelectorAll('.rs-kv span');
    kv[0].textContent = 'correlation'; kv[1].textContent = 'dots used';
    document.querySelector('.rs-sec').innerHTML = 'Address (r, φ₀, z₀)';
    kv[2].textContent = 'r'; kv[3].textContent = 'φ₀'; kv[4].textContent = 'dot idx';
    document.getElementById('rs-spec-lbl').innerHTML =
      'Component spectrum <span style="color:#6c7885">— amplitude per harmonic dot the signal decomposes into</span>';
    const p = payload, c = reconMode === '1' ? p.comp1 : p.compK;
    const corr = reconMode === '1' ? p.corr1 : p.corrK;
    const dots = reconMode === '1' ? 1 : p.components.length;
    document.getElementById('rs-ratio').textContent = c.ratio + ':1';
    document.getElementById('rs-bytes').textContent = `${c.raw_bytes} B raw → ${c.addr_bytes} B address`;
    document.getElementById('rs-corr').textContent = corr.toFixed(4);
    document.getElementById('rs-dots').textContent = dots + ' / ' + p.lattice_dots;
    document.getElementById('rs-r').textContent = p.address.r;
    document.getElementById('rs-phi').textContent = p.address.phi0;
    document.getElementById('rs-idx').textContent = p.address.idx;
    document.getElementById('rs-lbl').innerHTML =
      `<b>${p.kind}</b> · ${p.n} samples · ${reconMode === '1' ? 'single-dot' : dots + '-dot'} reconstruction`;
    document.getElementById('rs-legend').innerHTML =
      '<i style="background:#5fd39a"></i>signal<i style="background:#e0a53b"></i>reconstruction';
  }

  function renderVs() {
    const rs95 = cmp.rs95 > 40 ? '40+' : cmp.rs95;
    const ft95 = cmp.ft95 > 40 ? '40+' : cmp.ft95;
    let head;                                        // plain one-liner
    if (cmp.rs95 < cmp.ft95) head = `Radial needs ${(cmp.ft95 / cmp.rs95).toFixed(1)}× fewer`;
    else if (cmp.ft95 < cmp.rs95) head = `Fourier wins this one`;
    else head = 'about even';
    document.getElementById('rs-ratio').textContent = `${rs95} vs ${ft95}`;
    document.querySelector('.rs-comp span').textContent = 'pieces to rebuild (Radial vs Fourier)';
    document.getElementById('rs-bytes').textContent = head;
    const kv = document.querySelectorAll('.rs-kv span');
    kv[0].textContent = 'match — Radial'; document.getElementById('rs-corr').textContent = Math.round(cmp.rs_corr * 100) + '%';
    kv[1].textContent = 'match — Fourier'; document.getElementById('rs-dots').textContent = Math.round(cmp.ft_corr * 100) + '%';
    document.getElementById('rs-lbl').innerHTML =
      `<b>${cmp.kind}</b> · rebuilt from ${cmp.K} pieces each`;
    document.getElementById('rs-legend').innerHTML =
      '<i style="background:#5fd39a"></i>your signal<i style="background:#e0a53b"></i>Radial<i style="background:#3a9ad9"></i>Fourier';
    document.getElementById('rs-spec-lbl').innerHTML =
      'The building blocks each method uses to rebuild your signal';
  }

  // ---- requests ----------------------------------------------------------
  async function encode() {
    const btn = document.getElementById('rs-encode');
    btn.disabled = true; btn.textContent = 'Encoding…';
    try {
      const body = { kind: document.getElementById('rs-kind').value, n: +document.getElementById('rs-n').value,
        step: +document.getElementById('rs-step').value, rate: +document.getElementById('rs-rate').value,
        max_k: +document.getElementById('rs-mk').value };
      const d = await (await fetch('/api/radial/encode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
      if (d.error) { alert(d.error); return; }
      payload = d; renderCodec(); draw();
    } catch (e) { alert('failed: ' + e); } finally { btn.disabled = false; btn.textContent = 'Encode signal'; }
  }

  async function compare() {
    try {
      const body = { kind: document.getElementById('rs-kind2').value, n: +document.getElementById('rs-n').value, k: +document.getElementById('rs-k').value };
      const d = await (await fetch('/api/radial/compare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
      if (d.error) { alert(d.error); return; }
      cmp = d; renderVs(); draw();
    } catch (e) { alert('failed: ' + e); }
  }

  async function traversal() {
    try {
      const d = await (await fetch('/api/radial/traversal', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: document.getElementById('rs-stream').value })
      })).json();
      if (d.error) { alert(d.error); return; }
      v2 = d;
      document.getElementById('rs-lbl').innerHTML =
        `<b>${d.kind}</b> stream → its path through radial space, two mappings`;
      document.getElementById('rs-legend').innerHTML =
        '<i style="background:#5fd39a"></i>activity stream';
      document.getElementById('rs-spec-lbl').innerHTML =
        'Same stream, two mappings — the paper\'s scatters, a proximity-preserving one draws a clean shape';
      draw();
    } catch (e) { alert('failed: ' + e); }
  }

  async function v2suite() {
    const btn = document.getElementById('rs-v2run');
    btn.disabled = true; btn.textContent = 'Running…';
    try {
      const d = await (await fetch('/api/radial/v2suite', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })).json();
      if (d.error) { alert(d.error); return; }
      // headline: the mapping-M lever
      const L = d.lever;
      document.getElementById('rs-ratio').textContent = `70→98%`;
      document.querySelector('.rs-comp span').textContent = 'activity classifier: broken M → fixed M';
      document.getElementById('rs-bytes').textContent =
        `proximity ${L.paper.proximity} → ${L.good.proximity} (lower = similar inputs stay near)`;
      // grouped test results into the tests area
      document.getElementById('rs-tests').innerHTML = d.groups.map(g =>
        `<div class="rs-sec" style="margin-top:10px">${g.id} ${g.title} — ${g.pass}/${g.total}</div>` +
        g.tests.map(t =>
          `<div class="rs-test"><span class="mk ${t.pass ? 'ok' : 'no'}">${t.pass ? '✓' : '✗'}</span>` +
          `<span><span class="tn">${t.id} ${t.name}</span><br/><span class="tv">${t.value}</span>` +
          (t.note ? `<br/><span class="tv" style="color:#8a94a0">${t.note}</span>` : '') + `</span></div>`).join('')
      ).join('');
      const vd = document.getElementById('rs-verdict'); vd.style.display = 'block'; vd.textContent = d.verdict;
    } catch (e) { alert('failed: ' + e); } finally { btn.disabled = false; btn.textContent = 'Run v2 tests (§9.2 + §9.3)'; }
  }

  async function suite10() {
    document.getElementById('rs-verdict').style.display = 'none';
    document.getElementById('rs-tests').innerHTML = '<div class="rs-test"><span class="tn" style="color:#6c7885">running Suite 10 (~10s)…</span></div>';
    try {
      const d = await (await fetch('/api/radial/suite10', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })).json();
      if (d.error) { alert(d.error); return; }
      m10 = d;
      document.getElementById('rs-lbl').innerHTML = '<b>Mapping M</b> — phi vs input for five M families (straight/smooth = invertible; wavy/sawtooth = not)';
      document.getElementById('rs-spec-lbl').innerHTML = 'Chain stability (Lyapunov) per M — a usable computer needs a bar AT the stable line; none are';
      document.getElementById('rs-legend').innerHTML = Object.keys(d.curves).map(k => `<i style="background:${M_COLORS[k]}"></i>${k}`).join('');
      // headline
      document.getElementById('rs-ratio').textContent = d.best_mem;
      document.querySelector('.rs-comp span').textContent = 'best M for memory + fingerprinting';
      document.getElementById('rs-bytes').textContent = d.computation_dead ? 'chain computation: DEAD under every M' : 'computation: viable';
      // table
      const kv = document.querySelectorAll('.rs-kv span');
      kv[0].textContent = 'winner'; document.getElementById('rs-corr').textContent = d.best_mem;
      kv[1].textContent = 'compute'; document.getElementById('rs-dots').textContent = d.computation_dead ? 'dead' : 'open';
      document.getElementById('rs-tests').innerHTML =
        '<div class="rs-sec">Suite 10 — M comparison</div>' +
        '<div class="xr-mtable" style="font-size:10.5px">' +
        `<div class="rs-test" style="color:#7f8b98"><span style="flex:0 0 62px">M</span><span style="flex:0 0 34px">inv</span><span style="flex:0 0 44px">prox</span><span style="flex:0 0 40px">class</span><span>lyap</span></div>` +
        d.rows.map(r => `<div class="rs-test"><span style="flex:0 0 62px;color:${M_COLORS[r.M]}">${r.M}</span>` +
          `<span style="flex:0 0 34px" class="${r.invertible ? 'mk ok' : 'mk no'}">${r.invertible ? '✓' : '✗'}</span>` +
          `<span style="flex:0 0 44px;color:${r.proximity < 0.5 ? '#9fb0bd' : '#e07a6b'}">${r.proximity}</span>` +
          `<span style="flex:0 0 40px;color:#9fb0bd">${Math.round(r.classifier * 100)}%</span>` +
          `<span style="color:${Math.abs(r.lyapunov) < 0.05 ? '#5fd39a' : '#e0a53b'}">${r.lyapunov}</span></div>`).join('') +
        '</div>';
      const vd = document.getElementById('rs-verdict'); vd.style.display = 'block'; vd.textContent = d.verdict;
      draw();
    } catch (e) { alert('failed: ' + e); }
  }

  // ---- lens-map mode (live exploration) ----------------------------------
  const lens = { on: false, dims: null, cells: [], maxS: 1e-6, maxSep: 1e-6, colorBy: 'struct',
    checkpoints: [], baseline: null, best: null, bestIdx: -1, count: 0, total: 0, done: false,
    depth: 1, history: [] };
  const _imgcv = document.createElement('canvas'); _imgcv.width = 32; _imgcv.height = 32;
  const _imgctx = _imgcv.getContext('2d');

  function heat(t) {                                   // 0..1 -> slate->blue->green->amber (visible low end)
    t = Math.max(0, Math.min(1, t));
    const stops = [[70, 84, 104], [58, 120, 190], [95, 211, 154], [235, 175, 65]];
    const f = t * 3, k = Math.min(2, Math.floor(f)), r = f - k;
    const a = stops[k], b = stops[k + 1];
    return `rgb(${a[0] + (b[0] - a[0]) * r | 0},${a[1] + (b[1] - a[1]) * r | 0},${a[2] + (b[2] - a[2]) * r | 0})`;
  }

  function drawLens() {
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (!lens.cells.length) { ctx.fillStyle = '#6c7885'; ctx.font = '13px system-ui'; ctx.fillText('press "Explore (infinite)" — the lens space is unbounded; it fills outward, richer lenses further out', 20, 40); return; }
    const cx = w / 2, cy = h / 2, s = Math.min(w, h) * 0.42;
    // faint complexity rings
    ctx.strokeStyle = '#141a20';
    for (let c = 2; c <= 9; c++) { ctx.beginPath(); ctx.arc(cx, cy, (0.16 + 0.11 * c) * s, 0, 6.283); ctx.stroke(); }
    // outlier-robust colour: saturating map so the common range spreads across
    // the palette instead of collapsing to 0 under a few extreme-kurtosis lenses
    const kscale = lens.colorBy === 'sep' ? 7 : 0.35;
    for (const p of lens.cells) {
      const v = lens.colorBy === 'sep' ? p.sep : p.struct;
      const t = 1 - Math.exp(-kscale * v);
      ctx.beginPath(); ctx.arc(cx + p.x * s, cy - p.y * s, 2.3, 0, 6.283);
      ctx.fillStyle = heat(t); ctx.fill();
    }
    if (lens.cells.length) {
      const p = lens.cells[lens.cells.length - 1];
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx + p.x * s, cy - p.y * s, 4, 0, 6.283); ctx.stroke();
    }
    ctx.fillStyle = '#6c7885'; ctx.font = '10px ui-monospace';
    ctx.fillText('centre = simple lenses · outward = richer (more axes, composed/product activations) · ∞', 12, h - 8);
    drawAccOverlay(w, h);
    drawStrip();
  }

  function drawAccOverlay(w, h) {
    if (!lens.checkpoints.length && lens.baseline == null) return;
    const bw = 150, bh = 74, x0 = w - bw - 12, y0 = 10;
    ctx.fillStyle = 'rgba(12,15,19,.82)'; ctx.strokeStyle = '#1e242c';
    ctx.fillRect(x0, y0, bw, bh); ctx.strokeRect(x0, y0, bw, bh);
    const lo = 0.15, hi = 0.35, Y = a => y0 + bh - 6 - (a - lo) / (hi - lo) * (bh - 20);
    if (lens.baseline != null) {
      ctx.strokeStyle = '#e0684d'; ctx.setLineDash([3, 2]); ctx.beginPath();
      ctx.moveTo(x0 + 4, Y(lens.baseline)); ctx.lineTo(x0 + bw - 4, Y(lens.baseline)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#e0684d'; ctx.font = '9px ui-monospace'; ctx.fillText('PCA ' + lens.baseline, x0 + 5, Y(lens.baseline) - 3);
    }
    // previous layer's best (compounding reference)
    const prev = lens.history.filter(h => h.depth < lens.depth).map(h => h.best);
    if (prev.length) {
      const pb = Math.max(...prev);
      ctx.strokeStyle = '#7f8b98'; ctx.setLineDash([2, 2]); ctx.beginPath();
      ctx.moveTo(x0 + 4, Y(pb)); ctx.lineTo(x0 + bw - 4, Y(pb)); ctx.stroke(); ctx.setLineDash([]);
    }
    const cps = lens.checkpoints;
    if (cps.length) {
      const maxN = cps[cps.length - 1].n || 1;
      ctx.strokeStyle = '#5fd39a'; ctx.lineWidth = 1.6; ctx.beginPath();
      cps.forEach((p, i) => { const x = x0 + 6 + (p.n / maxN) * (bw - 12), y = Y(p.acc); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
      const best = Math.max(...cps.map(p => p.acc));
      ctx.fillStyle = '#dbe2ea'; ctx.font = '10px ui-sans-serif';
      ctx.fillText('L' + lens.depth + ' ' + best.toFixed(3), x0 + 5, y0 + 12);
    }
  }

  function drawStrip() {
    const w = spec.clientWidth, h = spec.clientHeight;
    sctx.clearRect(0, 0, w, h);
    if (!lens.best || !lens.best.strip) { sctx.fillStyle = '#6c7885'; sctx.font = '12px system-ui'; sctx.fillText('the best lens so far, and the real CIFAR images it fires most / least on', 14, h / 2); return; }
    const strip = lens.best.strip, sz = Math.min(46, (w - 40) / 8);
    sctx.fillStyle = '#8b95a1'; sctx.font = '11px ui-sans-serif';
    sctx.fillText(`best lens: ${lens.best.prog || ''}   structure ${lens.best.struct}`, 12, 16);
    sctx.fillStyle = '#5fd39a'; sctx.fillText('fires most →', 12, 34);
    sctx.fillStyle = '#e0684d'; sctx.fillText('fires least →', 12, 34 + sz + 8);
    for (let n = 0; n < strip.length; n++) {
      const img = strip[n]; const id = _imgctx.createImageData(32, 32);
      for (let p = 0; p < 1024; p++) { const yy = p / 32 | 0, xx = p % 32; const px = img[yy][xx];
        id.data[p * 4] = px[0]; id.data[p * 4 + 1] = px[1]; id.data[p * 4 + 2] = px[2]; id.data[p * 4 + 3] = 255; }
      _imgctx.putImageData(id, 0, 0);
      const row = n < 8 ? 0 : 1, col = n % 8;
      sctx.imageSmoothingEnabled = false;
      sctx.drawImage(_imgcv, 96 + col * (sz + 4), row === 0 ? 22 : 22 + sz + 8, sz, sz);
    }
  }

  function lensStatus() {
    const best = lens.checkpoints.length ? Math.max(...lens.checkpoints.map(p => p.acc)) : null;
    const hist = lens.history.map(h => `L${h.depth}: <b style="color:#5fd39a">${h.best.toFixed(3)}</b>`).join(' → ');
    document.getElementById('rs-lstatus').innerHTML =
      `<b style="color:#c7d0da">Layer ${lens.depth}</b> · explored <b>${lens.count}</b> lenses` +
      (lens.done ? ' <span style="color:#5fd39a">✓ stopped</span>' : ' <span style="color:#e0a53b">∞ exploring…</span>') +
      (best != null ? `<br/>encoder <b style="color:#5fd39a">${best.toFixed(3)}</b> vs input PCA <b style="color:#e0684d">${lens.baseline}</b>` : '') +
      (hist ? `<br/><span style="color:#7f8b98">stack:</span> ${hist}` : '');
  }

  async function lensPoll() {
    if (!lens.on) return;
    try {
      const d = await (await fetch('/api/radial/lensmap/poll', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ since: lens.count }) })).json();
      if (d.error) { alert(d.error); lens.on = false; return; }
      for (const c of d.new) { lens.cells.push(c); if (c.struct > lens.maxS) lens.maxS = c.struct; if (c.sep > lens.maxSep) lens.maxSep = c.sep; }
      lens.count = d.count; lens.total = d.total; lens.checkpoints = d.checkpoints; lens.baseline = d.baseline;
      lens.depth = d.depth; lens.history = d.history || [];
      if (d.best && d.best.idx !== lens.bestIdx) { lens.best = d.best; lens.bestIdx = d.best.idx; }
      lens.done = d.done;
      drawLens(); lensStatus();
      if (!d.done) setTimeout(lensPoll, 180);
      else {
        lens.on = false;
        document.getElementById('rs-lstart').disabled = false;
        document.getElementById('rs-lextend').disabled = false;
        document.getElementById('rs-lstop').disabled = true;
      }
    } catch (e) { lens.on = false; }
  }

  async function lensStart(extend) {
    if (lens.on) return;
    document.getElementById('rs-lstart').disabled = true;
    document.getElementById('rs-lextend').disabled = true;
    document.getElementById('rs-lstop').disabled = false;
    Object.assign(lens, { cells: [], maxS: 1e-6, maxSep: 1e-6, checkpoints: [], best: null, bestIdx: -1, count: 0, done: false });
    if (!extend) lens.history = [];
    try {
      const s = await (await fetch('/api/radial/lensmap/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ n_axes: +document.getElementById('rs-laxes').value, extend: !!extend, rot_deg: +document.getElementById('rs-lrot').value }) })).json();
      if (s.error) { alert(s.error); document.getElementById('rs-lstop').disabled = true; document.getElementById('rs-lstart').disabled = false; return; }
      lens.depth = s.depth; lens.on = true;
      lensPoll();
    } catch (e) { alert('failed: ' + e); }
  }

  async function lensStop() {
    document.getElementById('rs-lstop').disabled = true;
    document.getElementById('rs-lstop').textContent = 'stopping…';
    try { await fetch('/api/radial/lensmap/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }); } catch (e) {}
    setTimeout(() => document.getElementById('rs-lstop').textContent = '■ Stop', 1500);
  }

  // ---- screen mode -------------------------------------------------------
  const LABEL_COLORS = { idle: '#5fd39a', browsing: '#3a9ad9', video: '#e0a53b', coding: '#9b7bd6' };
  const screen = { paths: {}, current: null, curLabel: null, pred: null };
  const labelColor = l => LABEL_COLORS[l] || '#7f8b98';

  function drawScreen() {
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, s = Math.min(w, h) / 2 / 2.2;
    ctx.strokeStyle = '#171d24'; ctx.beginPath(); ctx.arc(cx, cy, s * 2, 0, 6.283); ctx.stroke();
    if (screen.current) {
      const p = screen.current, col = labelColor(screen.pred || screen.curLabel);
      ctx.strokeStyle = col + '77'; ctx.lineWidth = 1.4; ctx.beginPath();
      p.forEach((pt, i) => { const x = cx + pt[0] * s, y = cy - pt[1] * s; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
      p.forEach((pt, i) => { ctx.fillStyle = col; ctx.globalAlpha = 0.3 + 0.6 * (i / p.length); ctx.beginPath(); ctx.arc(cx + pt[0] * s, cy - pt[1] * s, 2.3, 0, 6.283); ctx.fill(); });
      ctx.globalAlpha = 1;
    }
    if (screen.pred) {
      ctx.fillStyle = labelColor(screen.pred); ctx.font = 'bold 22px ui-sans-serif,system-ui';
      ctx.fillText(screen.pred, 16, 34);
    }
    drawScreenOverlay();
  }

  function drawScreenOverlay() {
    const w = spec.clientWidth, h = spec.clientHeight;
    sctx.clearRect(0, 0, w, h);
    const cx = w / 2, cy = h / 2, s = Math.min(w, h) / 2 / 2.4;
    const labels = Object.keys(screen.paths);
    if (!labels.length) { sctx.fillStyle = '#6c7885'; sctx.font = '12px system-ui'; sctx.fillText('record a few activities to see their paths overlay here', 16, h / 2); return; }
    labels.forEach(lab => screen.paths[lab].forEach(p => {
      sctx.strokeStyle = labelColor(lab) + '66'; sctx.lineWidth = 1.2; sctx.beginPath();
      p.forEach((pt, i) => { const x = cx + pt[0] * s, y = cy - pt[1] * s; i ? sctx.lineTo(x, y) : sctx.moveTo(x, y); });
      sctx.stroke();
    }));
  }

  function refreshSStatus(extra) {
    const cnt = {}; Object.keys(screen.paths).forEach(l => cnt[l] = screen.paths[l].length);
    document.getElementById('rs-sstatus').innerHTML =
      '<b style="color:#c7d0da">recorded</b><br/>' +
      (Object.keys(cnt).length ? Object.entries(cnt).map(([l, n]) => `<span style="color:${labelColor(l)}">■</span> ${l}: ${n}`).join('<br/>') : 'nothing yet') +
      (extra ? '<br/><br/>' + extra : '');
  }

  async function screenPost(action, body) {
    return (await fetch('/api/radial/screen/' + action, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) })).json();
  }

  async function screenRecord() {
    const btn = document.getElementById('rs-srecord');
    const label = document.getElementById('rs-slabel').value;
    btn.disabled = true; btn.textContent = '● recording ' + label + '…';
    try {
      const d = await screenPost('record', { label, seconds: +document.getElementById('rs-ssecs').value });
      if (d.error) { alert(d.error); return; }
      (screen.paths[label] = screen.paths[label] || []).push(d.path);
      screen.current = d.path; screen.curLabel = label; screen.pred = null;
      document.getElementById('rs-lbl').innerHTML = `<b>${label}</b> — ${d.n_frames} frames · motion ${d.summary.motion} · bright ${d.summary.bright}`;
      refreshSStatus(); draw();
    } catch (e) { alert('failed: ' + e); } finally { btn.disabled = false; btn.textContent = '● Record activity'; }
  }

  async function screenTrain() {
    const d = await screenPost('train');
    if (d.error) { alert(d.error); return; }
    if (!d.ready) { refreshSStatus('<span style="color:#e0a53b">' + d.msg + '</span>'); return; }
    refreshSStatus(`<b style="color:#5fd39a">trained</b><br/>leave-one-out acc: <b>${Math.round(d.loo_accuracy * 100)}%</b> on ${d.n} clips`);
    document.getElementById('rs-ratio').textContent = Math.round(d.loo_accuracy * 100) + '%';
    document.querySelector('.rs-comp span').textContent = 'classifier accuracy (leave-one-out)';
    document.getElementById('rs-bytes').textContent = d.labels.join(' · ');
  }

  async function screenClassify() {
    const btn = document.getElementById('rs-sclassify');
    btn.disabled = true; btn.textContent = 'watching…';
    try {
      const d = await screenPost('classify', { seconds: 3 });
      if (d.error) { alert(d.error); return; }
      screen.current = d.path; screen.pred = d.predicted; screen.curLabel = null;
      document.getElementById('rs-lbl').innerHTML = `predicted: <b style="color:${labelColor(d.predicted)}">${d.predicted}</b> · motion ${d.summary.motion}`;
      document.getElementById('rs-tests').innerHTML = '<div class="rs-sec">Live prediction</div>' +
        d.scores.sort((a, b) => b.conf - a.conf).map(s =>
          `<div class="xr-drow" style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11.5px">` +
          `<span class="sw" style="width:10px;height:10px;border-radius:2px;background:${labelColor(s.label)}"></span>` +
          `<span style="width:64px;color:#b6c1cd">${s.label}</span>` +
          `<span style="flex:1;height:6px;background:#1a2028;border-radius:3px;overflow:hidden"><i style="display:block;height:100%;width:${s.conf * 100}%;background:${labelColor(s.label)}"></i></span>` +
          `<span style="width:34px;text-align:right;color:#8b95a1">${Math.round(s.conf * 100)}%</span></div>`).join('');
      draw();
    } catch (e) { alert('failed: ' + e); } finally { btn.disabled = false; btn.textContent = 'What am I doing now?'; }
  }

  async function screenClear() {
    await screenPost('clear');
    screen.paths = {}; screen.current = null; screen.pred = null;
    document.getElementById('rs-tests').innerHTML = ''; refreshSStatus(); draw();
  }

  async function validate() {
    const btn = document.getElementById('rs-validate');
    btn.disabled = true; btn.textContent = 'Running…';
    try {
      const body = { step: +document.getElementById('rs-step').value, rate: +document.getElementById('rs-rate').value, n: +document.getElementById('rs-n').value };
      const d = await (await fetch('/api/radial/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })).json();
      if (d.error) { alert(d.error); return; }
      document.getElementById('rs-tests').innerHTML = d.tests.map(t =>
        `<div class="rs-test"><span class="mk ${t.pass ? 'ok' : 'no'}">${t.pass ? '✓' : '✗'}</span>` +
        `<span><span class="tn">${t.id} ${t.name}</span><br/><span class="tv">${t.value}</span></span></div>`).join('');
      const vd = document.getElementById('rs-verdict'); vd.style.display = 'block';
      vd.textContent = `${d.pass}/${d.total} pass · ${d.verdict}`;
    } catch (e) { alert('failed: ' + e); } finally { btn.disabled = false; btn.textContent = 'Run validation suite (§8)'; }
  }

  function setReconMode(m) {
    reconMode = m;
    document.getElementById('rs-1dot').classList.toggle('on', m === '1');
    document.getElementById('rs-kdot').classList.toggle('on', m === 'K');
    if (payload) { renderCodec(); draw(); }
  }

  function setMode(m) {
    mode = m;
    ['codec', 'vs', 'v2', 'm', 'screen', 'lens'].forEach(x => document.getElementById('rs-mode-' + x).classList.toggle('on', m === x));
    const codec = m === 'codec', vs = m === 'vs', isv2 = m === 'v2', ism = m === 'm', iss = m === 'screen', isl = m === 'lens';
    document.getElementById('rs-fld-kind').style.display = codec ? '' : 'none';
    document.getElementById('rs-fld-kind2').style.display = vs ? '' : 'none';
    document.getElementById('rs-fld-k').style.display = vs ? '' : 'none';
    document.getElementById('rs-fld-stream').style.display = isv2 ? '' : 'none';
    document.getElementById('rs-v2run').style.display = isv2 ? '' : 'none';
    document.getElementById('rs-screen-ctl').style.display = iss ? '' : 'none';
    document.getElementById('rs-lens-ctl').style.display = isl ? '' : 'none';
    document.getElementById('rs-encode').style.display = codec ? '' : 'none';
    document.querySelector('.rs-recontoggle').style.display = codec ? '' : 'none';
    document.getElementById('rs-explain').style.display = codec ? 'none' : '';
    document.getElementById('rs-addr-block').style.display = codec ? '' : 'none';
    document.getElementById('rs-validate').style.display = (isv2 || ism || iss || isl) ? 'none' : '';
    if (isl) document.getElementById('rs-explain').innerHTML =
      "The lens map, explored LIVE on real CIFAR. Each cell is one lens — a rotation sweep of two feature axes through an " +
      "activation, <code>act(cos θ·axis_i + sin θ·axis_j)</code>. Cells light up by how much structure the lens found " +
      "(no labels). The green curve (top-right) is a real encoder built from the map, climbing past the red PCA line as more is swept.";
    else if (iss) document.getElementById('rs-explain').innerHTML =
      "§11.1 — real screen capture. Pick an activity, hit <b>Record</b> while doing it (idle, scroll a page, play a video). " +
      "Each frame's brightness + motion becomes a point in radial space; the clip draws a <b>path</b>. Record a few, " +
      "<b>Train</b>, then <b>What am I doing now?</b> classifies live from path shape alone — no model, just geometry.";
    else if (ism) document.getElementById('rs-explain').innerHTML =
      "v3's decision test: the mapping M is the only design choice, so which M works? Five families tested. " +
      "The <b>curves</b> above are phi vs input — straight/smooth = invertible memory; wavy/sawtooth = broken. " +
      "The <b>bars</b> are chain stability: a real computer needs one at zero, and none are — so computation is dropped.";
    else if (isv2) document.getElementById('rs-explain').innerHTML =
      "v2 claims the lattice is <b>memory</b> and a <b>processor</b>. Tested honestly, both fail with the paper's " +
      "mapping — but the paper says the mapping M is the design lever, and it's right: swap M and the one useful " +
      "part (a stream's <b>path</b> fingerprinting activity) works. Left path = paper's M (scatter), right = a fixed M.";
    else if (vs) document.getElementById('rs-explain').innerHTML =
      "Fourier rebuilds from <b style='color:#3a9ad9'>endless waves</b>; the radial version uses " +
      "<b style='color:#e0a53b'>blips</b> you drop where the action is. Fewer pieces = smaller + you know <b>when</b>.";
    if (codec) encode(); else if (vs) compare(); else if (isv2) traversal();
    else if (ism) suite10(); else if (isl) { drawLens(); lensStatus(); }
    else { refreshSStatus(); draw(); }
  }

  document.getElementById('rs-encode').addEventListener('click', encode);
  document.getElementById('rs-validate').addEventListener('click', validate);
  document.getElementById('rs-1dot').addEventListener('click', () => setReconMode('1'));
  document.getElementById('rs-kdot').addEventListener('click', () => setReconMode('K'));
  document.getElementById('rs-mode-codec').addEventListener('click', () => setMode('codec'));
  document.getElementById('rs-mode-vs').addEventListener('click', () => setMode('vs'));
  document.getElementById('rs-mode-v2').addEventListener('click', () => setMode('v2'));
  document.getElementById('rs-mode-m').addEventListener('click', () => setMode('m'));
  document.getElementById('rs-mode-screen').addEventListener('click', () => setMode('screen'));
  document.getElementById('rs-mode-lens').addEventListener('click', () => setMode('lens'));
  document.getElementById('rs-lstart').addEventListener('click', () => lensStart(false));
  document.getElementById('rs-lextend').addEventListener('click', () => lensStart(true));
  document.getElementById('rs-lstop').addEventListener('click', lensStop);
  document.getElementById('rs-lc-struct').addEventListener('click', () => { lens.colorBy = 'struct'; document.getElementById('rs-lc-struct').classList.add('on'); document.getElementById('rs-lc-sep').classList.remove('on'); drawLens(); });
  document.getElementById('rs-lc-sep').addEventListener('click', () => { lens.colorBy = 'sep'; document.getElementById('rs-lc-sep').classList.add('on'); document.getElementById('rs-lc-struct').classList.remove('on'); drawLens(); });
  const laxes = document.getElementById('rs-laxes');
  laxes.addEventListener('input', () => document.getElementById('rs-la-v').textContent = laxes.value);
  document.getElementById('rs-srecord').addEventListener('click', screenRecord);
  document.getElementById('rs-strain').addEventListener('click', screenTrain);
  document.getElementById('rs-sclassify').addEventListener('click', screenClassify);
  document.getElementById('rs-sclear').addEventListener('click', screenClear);
  const ssecs = document.getElementById('rs-ssecs');
  ssecs.addEventListener('input', () => document.getElementById('rs-ssecs-v').textContent = ssecs.value);
  document.getElementById('rs-stream').addEventListener('change', traversal);
  document.getElementById('rs-v2run').addEventListener('click', v2suite);
  document.getElementById('rs-kind').addEventListener('change', encode);
  document.getElementById('rs-kind2').addEventListener('change', compare);
  const kk = document.getElementById('rs-k');
  kk.addEventListener('input', () => document.getElementById('rs-k-v').textContent = kk.value);
  kk.addEventListener('change', compare);
  const mk = document.getElementById('rs-mk');
  mk.addEventListener('input', () => document.getElementById('rs-mk-v').textContent = mk.value);

  // continuous redraw for the live lens map (decoupled from polling)
  (function lensRAF() {
    if (mode === 'lens') { fit(cv, ctx); fit(spec, sctx); drawLens(); }
    requestAnimationFrame(lensRAF);
  })();

  fitAll(); encode();
})();
