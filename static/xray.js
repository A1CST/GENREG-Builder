// xray.js — watch a solved genome pull tangled MNIST digits into ten clusters.
// Each point tweens from its raw-feature position (t=0, tangled) to its
// post-genome position at the digit's corner (t=1, separated). Colour = the TRUE
// digit, so mistakes land on the wrong-coloured corner and you can see them.
(function () {
  const cv = document.getElementById('xr-cv');
  const ctx = cv.getContext('2d');
  let data = null;         // last transform payload
  let t = 0;               // 0..1 animation phase
  let playing = false, playStart = 0;
  const DUR = 1600;        // ms for a full separation sweep
  let state = { genome: 'r5', mixer: true, pairs: true, npc: 50 };

  function fit() {
    const dpr = window.devicePixelRatio || 1;
    cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', fit);

  const ease = x => x < .5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;

  function draw() {
    const w = cv.clientWidth, h = cv.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (!data || data.error) return;
    const cx = w / 2, cy = h / 2, s = Math.min(w, h) * 0.34;
    const e = ease(t);

    // digit anchors + labels (fade in as points separate)
    ctx.globalAlpha = 0.25 + 0.75 * e;
    data.anchors.forEach(a => {
      const px = cx + a.x * s, py = cy + a.y * s;
      ctx.beginPath(); ctx.arc(px, py, 15, 0, 6.283);
      ctx.strokeStyle = data.colors[a.d] + '66'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.fillStyle = data.colors[a.d]; ctx.font = 'bold 14px ui-sans-serif,system-ui';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(a.d, px, py);
    });
    ctx.globalAlpha = 1;

    // points
    for (const p of data.points) {
      const x = p.x0 + (p.x1 - p.x0) * e;
      const y = p.y0 + (p.y1 - p.y0) * e;
      const px = cx + x * s, py = cy + y * s;
      ctx.beginPath(); ctx.arc(px, py, 2.6, 0, 6.283);
      ctx.fillStyle = data.colors[p.d];
      ctx.globalAlpha = 0.55 + 0.4 * e;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
  }

  function loop(ts) {
    if (playing) {
      const k = Math.min(1, (ts - playStart) / DUR);
      t = k;
      document.getElementById('xr-t').value = Math.round(t * 1000);
      if (k >= 1) playing = false;
    }
    draw();
    requestAnimationFrame(loop);
  }

  function play() { playing = true; playStart = performance.now(); t = 0; }

  function renderMetrics() {
    document.getElementById('xr-acc').textContent = data.acc != null ? (data.acc * 100).toFixed(1) + '%' : '—';
    document.getElementById('xr-perdigit').innerHTML = data.per_digit.map((v, d) =>
      `<div class="xr-drow"><span class="sw" style="background:${data.colors[d]}"></span>` +
      `<span class="lb">${d}</span><span class="xr-bar"><i style="width:${(v || 0) * 100}%;background:${data.colors[d]}"></i></span>` +
      `<span class="pc">${v == null ? '—' : (v * 100).toFixed(0)}</span></div>`).join('');
    document.getElementById('xr-cap').innerHTML =
      `<b>Genome ${data.genome}</b> · ${data.n} digits · raw features (left of the slider) → ` +
      `the genome's read of each digit (its corner). Drag the slider or replay to watch it separate.`;
  }

  function renderGenomeList() {
    document.getElementById('xr-genomes').innerHTML = data.genomes.map(g =>
      `<button class="xr-gbtn ${g.id === state.genome ? 'active' : ''}" data-g="${g.id}" ${g.exists ? '' : 'disabled'}>` +
      `<span>${g.label}</span><span class="a">${g.acc != null ? (g.acc * 100).toFixed(2) + '%' : ''}</span></button>`).join('');
    document.querySelectorAll('#xr-genomes .xr-gbtn').forEach(b =>
      b.addEventListener('click', () => { state.genome = b.dataset.g; apply(); }));
  }

  async function apply() {
    const cap = document.getElementById('xr-cap');
    const first = !data;
    cap.innerHTML = first
      ? '<b>Building features…</b> first run reads MNIST and builds the shared cloud (~10s).'
      : '<b>Applying genome…</b>';
    try {
      const res = await fetch('/api/xray/transform', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          genome: state.genome, use_mixer: state.mixer,
          use_pairs: state.pairs, n_per_class: state.npc
        })
      });
      const d = await res.json();
      if (d.error) { cap.innerHTML = '<b>Error:</b> ' + d.error; return; }
      data = d;
      renderGenomeList(); renderMetrics(); play();
    } catch (e) { cap.innerHTML = '<b>Request failed:</b> ' + e; }
  }

  // controls
  document.getElementById('xr-t').addEventListener('input', e => {
    playing = false; t = (+e.target.value) / 1000;
  });
  document.getElementById('xr-apply').addEventListener('click', apply);
  document.getElementById('xr-anim').addEventListener('click', play);
  document.querySelectorAll('.xr-tg').forEach(tg => tg.addEventListener('click', () => {
    const k = tg.dataset.k; state[k] = !state[k];
    tg.classList.toggle('on', state[k]);
    apply();
  }));
  const npc = document.getElementById('xr-npc');
  npc.addEventListener('input', () => { document.getElementById('xr-npc-v').textContent = npc.value; });
  npc.addEventListener('change', () => { state.npc = +npc.value; apply(); });

  fit(); requestAnimationFrame(loop); apply();
})();
