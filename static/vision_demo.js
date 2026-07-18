/* vision_demo.js — renders /vision_demo from /api/vision_demo/data.
   Hand-rolled inline SVG (no libraries). Two staples: union + continued training. */
(function () {
  const $ = (id) => document.getElementById(id);
  const COL = { overall: "#7ee787", shapes: "#79c0ff", letters: "#e3b341", before: "#3a4657" };
  const pct = (x) => (x * 100).toFixed(1);

  fetch("/api/vision_demo/data", { cache: "no-store" })
    .then((r) => r.json())
    .then((d) => {
      if (d.error) {
        $("page-main").insertAdjacentHTML("afterbegin",
          `<div class="vd2-card vd2-err">Not built yet: ${d.error}</div>`);
        return;
      }
      $("vd2-chance").textContent = pct(d.meta.chance) + "%";
      renderUnion(d.union);
      renderBeforeAfter(d.continued);
      renderCurve(d.continued);
      renderContExtras(d.continued);
      if (d.efficiency && d.scratch) renderEfficiency(d);
      renderMeta(d);
    });

  // y maps [floor,1] -> [H-padB, padT]; a zoomed axis so near-ceiling gaps show.
  function yscale(g, floor) {
    const inner = g.H - g.padT - g.padB;
    return (v) => g.padT + inner * (1 - (Math.max(floor, Math.min(1, v)) - floor) / (1 - floor));
  }
  function yaxis(svg, g, floor, Y) {
    let s = "";
    [floor, (floor + 1) / 2, 1].forEach((v) => {
      const y = Y(v).toFixed(1);
      s += `<line x1="${g.padL}" y1="${y}" x2="${g.W - g.padR}" y2="${y}" stroke="#1c232c"/>`;
      s += `<text x="${g.padL - 5}" y="${y}" fill="#6e7681" font-size="9.5" text-anchor="end" dominant-baseline="middle">${(v * 100).toFixed(0)}</text>`;
    });
    return s;
  }

  // ── union: grouped bars, 3 banks x {overall, shapes, letters} ──────────
  function renderUnion(u) {
    const g = { W: 720, H: 300, padL: 34, padR: 14, padT: 22, padB: 56 };
    const floor = 0.85;
    const Y = yscale(g, floor);
    const banks = u.results;                       // shape-bank / letter-bank / FUSED
    const metrics = ["overall", "shapes", "letters"];
    const innerW = g.W - g.padL - g.padR;
    const gw = innerW / banks.length;
    const bw = Math.min(34, gw / 4);
    let svg = yaxis($("vd2-union"), g, floor, Y);
    banks.forEach((b, i) => {
      const gx = g.padL + gw * (i + 0.5);
      metrics.forEach((m, j) => {
        const x = gx + (j - 1) * (bw + 5) - bw / 2;
        const y = Y(b[m]), h = (g.H - g.padB) - y;
        const fused = b.bank.indexOf("FUSED") === 0;
        svg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw}" height="${Math.max(0, h).toFixed(1)}" rx="2" fill="${COL[m]}" opacity="${fused ? 1 : 0.82}"/>`;
        svg += `<text x="${(x + bw / 2).toFixed(1)}" y="${(y - 4).toFixed(1)}" fill="#c7d0da" font-size="9" text-anchor="middle" font-family="ui-monospace,monospace">${pct(b[m])}</text>`;
      });
      const label = b.bank.replace(" (both banks)", "").replace(" only", "");
      svg += `<text x="${gx.toFixed(1)}" y="${g.H - g.padB + 16}" fill="${b.bank.indexOf('FUSED') === 0 ? '#7ee787' : '#c7d0da'}" font-size="11" text-anchor="middle" font-weight="${b.bank.indexOf('FUSED') === 0 ? '700' : '400'}">${label}</text>`;
      svg += `<text x="${gx.toFixed(1)}" y="${g.H - g.padB + 30}" fill="#6e7681" font-size="9" text-anchor="middle">${b.n_feats} feats</text>`;
    });
    svg += `<text x="${g.padL - 5}" y="${g.padT - 8}" fill="#6e7681" font-size="9">acc %</text>`;
    svg += `<text x="${g.W - g.padR}" y="${g.H - 6}" fill="#4a525e" font-size="8.5" text-anchor="end">y-axis 85–100%</text>`;
    $("vd2-union").innerHTML = svg;

    $("vd2-union-legend").innerHTML = ["overall", "shapes", "letters"].map((m) =>
      `<span class="k"><span class="sw" style="background:${COL[m]}"></span>${m}</span>`).join("");

    const fused = banks.find((b) => b.bank.indexOf("FUSED") === 0);
    const sh = banks.find((b) => b.bank.indexOf("shape") === 0);
    const le = banks.find((b) => b.bank.indexOf("letter") === 0);
    $("vd2-union-headline").innerHTML =
      `With one shared readout, the fused model reaches <b>${pct(fused.overall)}%</b> overall — above the shape ` +
      `bank alone (${pct(sh.overall)}%) and the letter bank alone (${pct(le.overall)}%). Both banks stayed frozen; ` +
      `only the linear readout was fit.`;
    // cross-modal transfer, stated observation-first (arguably the headline result)
    $("vd2-crossmodal").innerHTML =
      `The frozen shape model classifies letters at <b>${pct(sh.letters)}%</b> without ever being optimized on ` +
      `letters. Likewise the frozen letter model classifies shapes at <b>${pct(le.shapes)}%</b>. This suggests the ` +
      `evolved feature basis is largely shared between the two visual domains.` +
      `<span class="q">Which raises the more interesting question: what are these features actually representing, ` +
      `if a shape detector reads letters this well?</span>`;
  }

  // ── continued: before -> after grouped bars ────────────────────────────
  function renderBeforeAfter(c) {
    const g = { W: 340, H: 300, padL: 30, padR: 12, padT: 22, padB: 44 };
    const floor = 0.85;
    const Y = yscale(g, floor);
    const metrics = ["overall", "shapes", "letters"];
    const innerW = g.W - g.padL - g.padR;
    const gw = innerW / metrics.length;
    const bw = Math.min(26, gw / 3.2);
    let svg = yaxis($("vd2-ba"), g, floor, Y);
    metrics.forEach((m, i) => {
      const gx = g.padL + gw * (i + 0.5);
      [["before", c.before[m], COL.before], ["after", c.after[m], COL[m]]].forEach(([lab, v, col], k) => {
        const x = gx + (k - 0.5) * (bw + 4) - bw / 2;
        const y = Y(v), h = (g.H - g.padB) - y;
        svg += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw}" height="${Math.max(0, h).toFixed(1)}" rx="2" fill="${col}"/>`;
        svg += `<text x="${(x + bw / 2).toFixed(1)}" y="${(y - 4).toFixed(1)}" fill="#c7d0da" font-size="8.5" text-anchor="middle" font-family="ui-monospace,monospace">${pct(v)}</text>`;
      });
      svg += `<text x="${gx.toFixed(1)}" y="${g.H - g.padB + 16}" fill="#c7d0da" font-size="10" text-anchor="middle">${m}</text>`;
    });
    svg += `<text x="${g.W - g.padR}" y="${g.H - 6}" fill="#4a525e" font-size="8" text-anchor="end">y 85–100%</text>`;
    $("vd2-ba").innerHTML = svg;
  }

  // ── continued: the climb (acc vs new genomes) ──────────────────────────
  function renderCurve(c) {
    const g = { W: 340, H: 300, padL: 30, padR: 12, padT: 18, padB: 40 };
    const floor = 0.90;
    const Y = yscale(g, floor);
    const pts = c.curve;
    const maxN = pts[pts.length - 1].n_new || 1;
    const innerW = g.W - g.padL - g.padR;
    const X = (n) => g.padL + innerW * n / maxN;
    let svg = yaxis($("vd2-curve"), g, floor, Y);
    // x ticks
    [0, maxN].forEach((n) => {
      svg += `<text x="${X(n).toFixed(1)}" y="${g.H - g.padB + 15}" fill="#6e7681" font-size="9" text-anchor="middle">${n}</text>`;
    });
    svg += `<text x="${(g.padL + innerW / 2).toFixed(1)}" y="${g.H - g.padB + 30}" fill="#8b95a1" font-size="9.5" text-anchor="middle">new genomes evolved</text>`;
    ["overall", "letters"].forEach((m) => {
      const line = pts.map((p) => `${X(p.n_new).toFixed(1)},${Y(p[m]).toFixed(1)}`).join(" ");
      svg += `<polyline points="${line}" fill="none" stroke="${COL[m]}" stroke-width="2"/>`;
      const last = pts[pts.length - 1];
      svg += `<circle cx="${X(last.n_new).toFixed(1)}" cy="${Y(last[m]).toFixed(1)}" r="3" fill="${COL[m]}"/>`;
    });
    $("vd2-curve").innerHTML = svg;
  }

  function renderContExtras(c) {
    $("vd2-cont-legend").innerHTML = ["overall", "letters"].map((m) =>
      `<span class="k"><span class="sw" style="background:${COL[m]}"></span>${m}</span>`).join("") +
      `<span class="k"><span class="sw" style="background:${COL.before}"></span>before (base only)</span>`;

    const dLet = (c.after.letters - c.before.letters);
    $("vd2-cont-tiles").innerHTML = [
      [`<span class="from">${pct(c.before.letters)}</span><span class="arrow">→</span>` +
       `<span class="to">${pct(c.after.letters)}%</span> <span class="plus">+${(dLet * 100).toFixed(1)}</span>`,
       "letters (test)", true],
      [c.after.n_new_genomes, "new genomes evolved"],
      [c.params.base_genomes, "base shape genomes (frozen)"],
      [c.params.head.toLocaleString(), "head params (one ridge)"],
    ].map(([n, l, html]) =>
      `<div class="vd2-tile"><div class="n">${html ? `<span class="vd2-delta">${n}</span>` : n}</div><div class="l">${l}</div></div>`).join("");

    $("vd2-cont-headline").innerHTML =
      `The shapes model learned letters by itself: letters climbed <b>${pct(c.before.letters)}% → ${pct(c.after.letters)}%</b> ` +
      `and overall ${pct(c.before.overall)}% → ${pct(c.after.overall)}%, adding ${c.after.n_new_genomes} evolved genomes over the ` +
      `${c.params.base_genomes} frozen shape genomes — no separate letter model was ever trained.`;
  }

  // ── transfer efficiency: continued (grow) vs from-scratch control ──────
  function renderEfficiency(d) {
    const e = d.efficiency, cont = d.continued, scr = d.scratch;
    const tgt = pct(e.target);
    const g2t = (v) => (v === 0 ? "0 <span style='color:#6e7681'>(already above)</span>"
                       : v == null ? "—" : v);
    $("vd2-eff").innerHTML =
      `<table class="vd2-eff"><thead><tr><th>Method</th>` +
      `<th style="text-align:right">Final accuracy</th>` +
      `<th style="text-align:right">Genomes to reach ${tgt}%</th>` +
      `<th style="text-align:right">Genomes evolved</th></tr></thead><tbody>` +
      `<tr class="grow"><td><span class="dot" style="background:#56d364"></span>Grow (continued)</td>` +
      `<td class="num hi">${pct(e.continued.final)}%</td><td class="num hi">${g2t(e.continued.genomes_to_target)}</td>` +
      `<td class="num">${e.continued.final_genomes}</td></tr>` +
      `<tr><td><span class="dot" style="background:#ff9e64"></span>From scratch (control)</td>` +
      `<td class="num">${pct(e.scratch.final)}%</td><td class="num">${g2t(e.scratch.genomes_to_target)}</td>` +
      `<td class="num">${e.scratch.final_genomes}</td></tr></tbody></table>`;

    const cg = e.continued.genomes_to_target, sg = e.scratch.genomes_to_target;
    let note;
    if (cg === 0) {
      note = `The reused shape features already exceed the from-scratch model's best (<b>${pct(e.scratch.final)}%</b>) ` +
        `<b>before a single new genome is evolved</b> — the from-scratch control needs ${sg == null ? "many" : sg} ` +
        `genomes to reach the same point. Reusing the evolved shape representation is the difference.`;
    } else {
      const saved = (sg != null && cg != null) ? (sg - cg) : null;
      note = `To reach <b>${tgt}%</b>, growing evolves <b>${cg}</b> new genomes; from scratch needs ` +
        `<b>${sg == null ? "more than it ran" : sg}</b>` +
        (saved != null ? ` — reusing the shape features saves ~${saved} genomes of evolution` : "") + `.`;
    }
    $("vd2-eff").insertAdjacentHTML("beforeend", `<div class="vd2-eff-note">${note}</div>`);

    // curve: overall accuracy vs new genomes, both methods (warm starts high)
    const g = { W: 720, H: 300, padL: 34, padR: 14, padT: 18, padB: 44 };
    const floor = 0.0;
    const Y = yscale(g, floor);
    const maxN = Math.max(cont.curve[cont.curve.length - 1].n_new,
                          scr.curve[scr.curve.length - 1].n_new, 1);
    const innerW = g.W - g.padL - g.padR;
    const X = (n) => g.padL + innerW * n / maxN;
    let svg = yaxis($("vd2-eff-curve"), g, floor, Y);
    [[cont.curve, "#56d364"], [scr.curve, "#ff9e64"]].forEach(([cv, col]) => {
      const line = cv.map((p) => `${X(p.n_new).toFixed(1)},${Y(p.overall).toFixed(1)}`).join(" ");
      svg += `<polyline points="${line}" fill="none" stroke="${col}" stroke-width="2"/>`;
      const last = cv[cv.length - 1];
      svg += `<circle cx="${X(last.n_new).toFixed(1)}" cy="${Y(last.overall).toFixed(1)}" r="3" fill="${col}"/>`;
    });
    // target line
    svg += `<line x1="${g.padL}" y1="${Y(e.target).toFixed(1)}" x2="${g.W - g.padR}" y2="${Y(e.target).toFixed(1)}" stroke="#4a525e" stroke-dasharray="4 3"/>`;
    svg += `<text x="${g.W - g.padR}" y="${(Y(e.target) - 4).toFixed(1)}" fill="#6e7681" font-size="9" text-anchor="end">target ${tgt}%</text>`;
    svg += `<text x="${(g.padL + innerW / 2).toFixed(1)}" y="${g.H - 8}" fill="#8b95a1" font-size="9.5" text-anchor="middle">new genomes evolved</text>`;
    $("vd2-eff-curve").innerHTML = svg;
    $("vd2-eff-legend").innerHTML =
      `<span class="k"><span class="sw" style="background:#56d364"></span>grow (warm start)</span>` +
      `<span class="k"><span class="sw" style="background:#ff9e64"></span>from scratch</span>`;
  }

  // ── animated inference panels: checkpoints identifying shapes/letters ──
  function renderSamples() {
    fetch("/api/vision_demo/samples", { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        const box = $("vd2-panels");
        if (d.error) { box.innerHTML = `<div class="vd2-err">${d.error}</div>`; return; }
        box.innerHTML = "";
        ["shape", "letter", "union", "continued"].forEach((key, pi) => {
          const p = d.panels[key];
          if (!p || !p.samples || !p.samples.length) return;
          const el = document.createElement("div");
          el.className = "vd2-panel";
          el.innerHTML =
            `<div class="pt">${p.title}</div>` +
            `<div class="stage"><img alt="" /></div>` +
            `<div class="sees"></div><div class="pred"></div>` +
            `<div class="acc">accuracy <b>${(p.acc * 100).toFixed(1)}%</b></div>`;
          box.appendChild(el);
          setTimeout(() => animatePanel(el, p.samples), pi * 320);   // stagger the flips
        });
      });
  }

  function animatePanel(el, S) {
    const img = el.querySelector("img");
    const sees = el.querySelector(".sees");
    const pred = el.querySelector(".pred");
    let i = -1;
    function tick() {
      i = (i + 1) % S.length;
      const f = S[i];
      img.style.opacity = 0;
      setTimeout(() => { img.src = f.uri; img.style.opacity = 1; }, 150);
      const tag = f.modality ? `<span class="modtag ${f.modality}">${f.modality}</span>` : "";
      sees.innerHTML = `actual ${f.true} ${tag}`;
      pred.style.opacity = 0.15; pred.innerHTML = "&hellip;";
      setTimeout(() => {
        pred.style.opacity = 1;
        const cls = f.correct ? "ok" : "no", mk = f.correct ? "✓" : "✗";
        pred.innerHTML = `&rarr; <b>${f.pred}</b> <span class="${cls}">${mk}</span>`;
      }, 720);
    }
    tick();
    setInterval(tick, 2300);
  }

  renderSamples();

  function renderMeta(d) {
    $("vd2-meta").innerHTML =
      `${d.meta.n_classes} classes (${d.meta.shapes} shapes + ${d.meta.letters} letters), chance ${pct(d.meta.chance)}%. ` +
      `Gradient-free, test touched once. Built by <code>mm/vision_demo.py</code> in ${d.meta.seconds}s ` +
      `(union <code>mm/mm_merge.py</code> + continued <code>mm/vision_continue.py</code>). ` +
      `Data: <code>radial_data/vision_demo.json</code>.`;
  }
})();
