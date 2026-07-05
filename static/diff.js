/* GENREG — DiffEvo page.
 * One WebSocket (/diffuse) carries config → training events → reconstructions.
 * The bet: a tiny genome can't learn a whole image model, but one reverse-
 * diffusion step is easy. We evolve one shared population per noise level
 * (fitness = denoising MSE over a fresh minibatch each gen, so it generalizes)
 * and stack the per-level champions into a de-noiser.
 * Charts are hand-rolled SVG; images render to <canvas> nearest-neighbor.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
  const S1 = () => css("--tlm-s1");
  const S2 = () => css("--tlm-s2");

  // ── state ────────────────────────────────────────────────
  let ws = null, running = false;
  let img = 12;                              // image side, from `started`
  let fit = { level: null, gens: [], best: [], mean: [] };
  let levelBest = [];                        // best MSE per finished level
  let sigmas = [];

  // ── websocket ────────────────────────────────────────────
  function connect() {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/diffuse`);
    ws.onopen = () => setConn(true);
    ws.onclose = () => { setConn(false); running = false; setButtons(); setTimeout(connect, 1500); };
    ws.onmessage = (e) => { try { handle(JSON.parse(e.data)); } catch (_) {} };
  }
  function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
  function setConn(ok) {
    $("dv-dot").className = "dot" + (ok ? " ok" : " bad");
    $("dv-conn").textContent = ok ? "connected" : "disconnected";
  }

  function handle(ev) {
    switch (ev.type) {
      case "job": running = !!ev.running; setButtons(); break;
      case "started":
        img = ev.img; sigmas = ev.sigmas || []; levelBest = [];
        $("st-params").textContent = ev.params_per_genome;
        $("dv-status").textContent =
          `training — ${ev.mode}${ev.unrolled ? " (unrolled)" : ""} · ${ev.levels} levels · ` +
          `pop ${ev.pop} · ${ev.mode === "denoise" ? ev.sampler + " sampler · " : ""}${ev.n_train} train imgs`;
        running = true; setButtons();
        drawLevelBars();
        break;
      case "level_start":
        fit = { level: ev.level, gens: [], best: [], mean: [] };
        $("fit-level").textContent = `level ${ev.level}/${ev.of} · σ ${ev.sigma}`;
        $("st-level").textContent = `${ev.level} / ${ev.of}`;
        drawFitness();
        break;
      case "gen":
        fit.gens.push(ev.gen); fit.best.push(ev.best_l1); fit.mean.push(ev.mean_l1);
        $("st-mse").textContent = ev.best_ever.toFixed(4);
        $("dv-status").textContent =
          `level ${ev.level} · gen ${ev.gen}/${ev.generations} · best ${ev.best_ever.toFixed(4)}`;
        drawFitness();
        break;
      case "level_done":
        levelBest[ev.level - 1] = ev.best_l1;
        drawLevelBars();
        $("dv-status").textContent =
          `level ${ev.level} done (${ev.reason}) · best L1 ${ev.best_l1.toFixed(4)}`;
        break;
      case "sample": onSample(ev); break;
      case "done":
        running = false; setButtons();
        if (ev.test_improvement !== undefined) {
          $("st-improve").textContent = (ev.test_improvement * 100).toFixed(1) + "%";
          $("st-improve").className = "tlm-tile-value " +
            (ev.test_improvement > 0 ? "tlm-up" : "tlm-down");
        }
        $("dv-status").textContent =
          `${ev.reason} in ${ev.seconds}s` +
          (ev.test_improvement !== undefined
            ? ` · held-out L1 ${ev.test_in_l1} → ${ev.test_out_l1}` : "");
        break;
      case "error":
        running = false; setButtons();
        $("dv-status").textContent = "error: " + ev.message;
        break;
    }
  }

  // ── reconstruction (canvas, nearest-neighbor upscale) ────
  function paint(canvas, flat) {
    if (!canvas || !flat) return;
    const off = document.createElement("canvas");
    off.width = img; off.height = img;
    const octx = off.getContext("2d");
    const id = octx.createImageData(img, img);
    for (let i = 0; i < flat.length; i++) {
      const v = Math.max(0, Math.min(255, Math.round(flat[i] * 255)));
      id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = v;
      id.data[i * 4 + 3] = 255;
    }
    octx.putImageData(id, 0, 0);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function onSample(ev) {
    paint($("dv-clean"), ev.clean);
    paint($("dv-noisy"), ev.noisy);
    paint($("dv-final"), ev.final);
    $("st-in").textContent = ev.in_l1.toFixed(4);
    $("st-out").textContent = ev.out_l1.toFixed(4);
    $("recon-mse").innerHTML =
      `<span class="tlm-card-sub">top level ${ev.top_level} · L1 ${ev.in_l1.toFixed(4)} → ` +
      `<b style="color:${ev.out_l1 < ev.in_l1 ? "var(--green)" : "var(--red)"}">` +
      `${ev.out_l1.toFixed(4)}</b></span>`;
    // reverse-chain filmstrip: one frame per level, noisiest → cleanest
    const strip = $("dv-chain");
    strip.innerHTML = "";
    const l1s = ev.chain_l1 || [];
    (ev.chain || []).forEach((frame, i) => {
      const fig = document.createElement("figure");
      fig.className = "dv-fig sm";
      const c = document.createElement("canvas");
      c.width = 64; c.height = 64;
      fig.appendChild(c);
      const cap = document.createElement("figcaption");
      const lvl = ev.top_level - i;
      // color each step's L1 green if it beat the previous step, red if it diluted
      const worse = i > 0 && l1s[i] > l1s[i - 1];
      cap.innerHTML = `L${lvl}` + (l1s[i] !== undefined
        ? ` · <span style="color:${worse ? "var(--red)" : "var(--green)"}">${l1s[i].toFixed(3)}</span>` : "");
      fig.appendChild(cap);
      strip.appendChild(fig);
      paint(c, frame);
    });
  }

  // ── fitness line chart (best + mean MSE per gen, current level) ──
  function drawFitness() {
    const svg = $("fitness");
    const W = svg.clientWidth || 520, H = 200, m = { t: 10, r: 10, b: 22, l: 44 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!fit.gens.length) return;
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xs = fit.gens, allY = fit.best.concat(fit.mean);
    const xmin = xs[0], xmax = xs[xs.length - 1] || xs[0] + 1;
    const ymax = Math.max(...allY), ymin = Math.min(...allY);
    const yhi = ymax + (ymax - ymin) * 0.08 + 1e-6, ylo = Math.max(0, ymin - (ymax - ymin) * 0.08);
    const X = (v) => m.l + (xmax === xmin ? 0 : (v - xmin) / (xmax - xmin)) * iw;
    const Y = (v) => m.t + (yhi === ylo ? ih : (1 - (v - ylo) / (yhi - ylo)) * ih);

    // axes
    axisLine(svg, m.l, m.t, m.l, m.t + ih);
    axisLine(svg, m.l, m.t + ih, m.l + iw, m.t + ih);
    for (let k = 0; k <= 4; k++) {
      const val = ylo + (yhi - ylo) * k / 4, y = Y(val);
      text(svg, m.l - 6, y + 3, val.toFixed(3), "end", "var(--muted)", 9);
      gridLine(svg, m.l, y, m.l + iw, y);
    }
    text(svg, m.l + iw / 2, H - 4, "generation", "middle", "var(--muted)", 10);

    line(svg, xs, fit.mean, X, Y, S2());
    line(svg, xs, fit.best, X, Y, S1());
  }

  // ── best-MSE-by-level bars ───────────────────────────────
  function drawLevelBars() {
    const svg = $("levelbars");
    const W = svg.clientWidth || 520, H = 200, m = { t: 10, r: 10, b: 30, l: 44 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const n = sigmas.length || levelBest.length;
    if (!n) return;
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const vals = levelBest.slice(0, n);
    const ymax = Math.max(0.001, ...vals.filter((v) => v !== undefined));
    const yhi = ymax * 1.1;
    const Y = (v) => m.t + (1 - v / yhi) * ih;
    axisLine(svg, m.l, m.t, m.l, m.t + ih);
    axisLine(svg, m.l, m.t + ih, m.l + iw, m.t + ih);
    for (let k = 0; k <= 4; k++) {
      const val = yhi * k / 4, y = Y(val);
      text(svg, m.l - 6, y + 3, val.toFixed(3), "end", "var(--muted)", 9);
      gridLine(svg, m.l, y, m.l + iw, y);
    }
    const bw = iw / n;
    for (let i = 0; i < n; i++) {
      const v = vals[i];
      const bx = m.l + i * bw + bw * 0.15, bwid = bw * 0.7;
      if (v !== undefined) {
        const r = document.createElementNS(NS, "rect");
        r.setAttribute("x", bx); r.setAttribute("y", Y(v));
        r.setAttribute("width", bwid); r.setAttribute("height", m.t + ih - Y(v));
        r.setAttribute("fill", S1()); r.setAttribute("rx", 2);
        svg.appendChild(r);
      } else {
        const r = document.createElementNS(NS, "rect");
        r.setAttribute("x", bx); r.setAttribute("y", m.t + ih - 3);
        r.setAttribute("width", bwid); r.setAttribute("height", 3);
        r.setAttribute("fill", "var(--tlm-idle)");
        svg.appendChild(r);
      }
      const lbl = sigmas[i] !== undefined ? `σ${sigmas[i]}` : `L${i + 1}`;
      text(svg, bx + bwid / 2, m.t + ih + 12, lbl, "middle", "var(--muted)", 8);
    }
    text(svg, m.l + iw / 2, H - 3, "noise level (σ)", "middle", "var(--muted)", 10);
  }

  // ── tiny SVG helpers ─────────────────────────────────────
  function line(svg, xs, ys, X, Y, color) {
    let d = "";
    for (let i = 0; i < xs.length; i++) d += (i ? "L" : "M") + X(xs[i]) + " " + Y(ys[i]);
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", d); p.setAttribute("fill", "none");
    p.setAttribute("stroke", color); p.setAttribute("stroke-width", 1.6);
    svg.appendChild(p);
  }
  function axisLine(svg, x1, y1, x2, y2) { seg(svg, x1, y1, x2, y2, "var(--border)", 1); }
  function gridLine(svg, x1, y1, x2, y2) { seg(svg, x1, y1, x2, y2, "var(--tlm-idle)", 1); }
  function seg(svg, x1, y1, x2, y2, color, w) {
    const l = document.createElementNS(NS, "line");
    l.setAttribute("x1", x1); l.setAttribute("y1", y1);
    l.setAttribute("x2", x2); l.setAttribute("y2", y2);
    l.setAttribute("stroke", color); l.setAttribute("stroke-width", w);
    svg.appendChild(l);
  }
  function text(svg, x, y, str, anchor, fill, size) {
    const t = document.createElementNS(NS, "text");
    t.setAttribute("x", x); t.setAttribute("y", y);
    t.setAttribute("text-anchor", anchor); t.setAttribute("fill", fill);
    t.setAttribute("font-size", size); t.textContent = str;
    svg.appendChild(t);
  }

  // ── controls ─────────────────────────────────────────────
  function assembleConfig() {
    return {
      op: "start",
      levels: +$("dv-levels").value,
      window: +$("dv-window").value,
      hidden: +$("dv-hidden").value,
      pop: +$("dv-pop").value,
      minibatch: +$("dv-minibatch").value,
      max_gens: +$("dv-maxgens").value,
      patience: +$("dv-patience").value,
      elite_frac: +$("dv-elite").value,
      self_adaptive: $("dv-sa").checked,
      mutation: +$("dv-mut").value,
      seed: +$("dv-seed").value,
      mode: $("dv-mode").value,
      unrolled: $("dv-unrolled").checked,
      sampler: $("dv-sampler").value,
    };
  }
  function setButtons() {
    $("dv-start").disabled = running;
    $("dv-stop").disabled = !running;
  }

  $("dv-start").addEventListener("click", () => {
    if (running) return;
    send(assembleConfig());
    running = true; setButtons();
    $("dv-status").textContent = "starting…";
  });
  $("dv-stop").addEventListener("click", () => send({ op: "stop" }));
  window.addEventListener("resize", () => { drawFitness(); drawLevelBars(); });

  // sampler applies only to denoise mode; unrolled only to diffuse mode
  function syncModeUI() {
    const denoise = $("dv-mode").value === "denoise";
    $("dv-sampler-field").style.display = denoise ? "" : "none";
    $("dv-unrolled-field").style.display = denoise ? "none" : "";
  }
  $("dv-mode").addEventListener("change", syncModeUI);

  connect();
  setButtons();
  syncModeUI();
})();
