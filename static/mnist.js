/* GENREG — MNIST page (the specialist pipeline applied to images).
 * Toggle each evolved layer and watch held-out test accuracy move:
 * statistics layer (built, fixed) -> detector genomes -> output mixer ->
 * pairwise disambiguators. Inference runs server-side over the frozen
 * champions (REST /api/mnist/*). No gradients anywhere in the model.
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);

  const LAYERS = [
    { key: "stats", name: "Statistics", fixed: true,
      desc: "the BUILT layer — zone ink, profiles, gradient histograms, PCA of the raw pixels; computed from the data, never evolved" },
    { key: "det", name: "Detectors", fixed: true,
      desc: "10 one-vs-rest genomes (“is this a 3?”) — linear heads over the stats, soft BCE fitness" },
    { key: "mixer", name: "Mixer", desc: "an evolved 10x10 output genome that calibrates the detector logits (soft log-prob fitness over the true digit)" },
    { key: "pairs", name: "Pairwise", desc: "45 one-vs-one referee genomes (“4 or 9?”) — fire only when the mixer's top-2 are close" },
  ];
  const ARCH = [
    ["Statistics", "the environment, not the organism: 677 fixed dims built from the training images' own statistics — 4x4 + 7x7 zone ink, row/column profiles, gradient-orientation histograms at two cell scales, 64 PCA components of the raw pixels. No labels, no evolution. Same role as the SVD word features in EvoLang."],
    ["Detectors", "10 specialist genomes, one per digit, each with one clean survival condition: “is this my digit, yes or no?” A single evolved linear head (678 params) over the stats layer; fitness = mean log-prob on balanced pos/neg minibatches (soft, dense — never argmax)."],
    ["Mixer", "the output layer: an evolved 10x10 matrix + bias over the detector logits, fitness = mean log-softmax prob of the true digit. Gate: beats raw argmax over the detectors on held-out."],
    ["Pairwise", "the environment decomposed further: 45 genomes, one per digit pair, each trained ONLY on its two digits (“4 or 9?”). At inference they referee the mixer's top-2 when the margin is small — the confusable zone they were bred for. Margin chosen on validation, never test."],
  ];

  let en = { mixer: true, pairs: true };
  let seed = 1, ready = false, onlyErrors = false;

  function renderLayers() {
    const box = $("mn-layers");
    box.innerHTML = "";
    for (const L of LAYERS) {
      const on = L.fixed ? true : en[L.key];
      const row = document.createElement("div");
      row.className = "ev-layer" + (on ? " on" : "");
      row.title = L.name + " — " + L.desc;
      const val = L.fixed ? "FIXED" : (on ? "ON" : "OFF");
      row.innerHTML = `<div class="ev-layer-top"><span class="ev-chip ${on ? "on" : ""}">${val}</span>`
        + `<span class="ev-layer-name">${L.name}</span></div>`;
      if (!L.fixed) {
        row.addEventListener("click", () => {
          en[L.key] = !en[L.key];
          renderLayers(); refresh();
        });
      }
      box.appendChild(row);
    }
  }

  function renderArch() {
    $("mn-arch").innerHTML = ARCH.map(([n, d]) =>
      `<div class="ev-arch-row"><span class="ev-arch-name">${n}</span><span class="ev-arch-desc">${d}</span></div>`).join("");
  }

  function stackLabel() {
    const s = ["Statistics", "Detectors"];
    if (en.mixer) s.push("Mixer");
    if (en.pairs) s.push("Pairwise");
    return s.join("  +  ");
  }

  function qs() {
    return `mixer=${en.mixer ? 1 : 0}&pairs=${en.pairs ? 1 : 0}`;
  }

  function drawDigit(px) {
    const c = document.createElement("canvas");
    c.width = 28; c.height = 28;
    c.className = "mn-digit";
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = px[i];
      img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  function renderGrid(d) {
    const box = $("mn-grid");
    box.innerHTML = "";
    if (!d.items || !d.items.length) {
      box.textContent = onlyErrors ? "(no mistakes on the test set)" : "(no samples)";
      return;
    }
    for (const it of d.items) {
      const cell = document.createElement("div");
      cell.className = "mn-cell " + (it.ok ? "ok" : "bad");
      cell.appendChild(drawDigit(it.px));
      const lab = document.createElement("div");
      lab.className = "mn-lab";
      lab.textContent = it.ok ? String(it.pred) : (it.pred + " (true " + it.true + ")");
      cell.appendChild(lab);
      box.appendChild(cell);
    }
    if (d.acc != null) {
      $("st-acc").textContent = (d.acc * 100).toFixed(2) + "%";
      $("st-errors").textContent = d.n_errors;
    }
  }

  function renderConf(conf) {
    if (!conf) { $("mn-conf").textContent = "—"; return; }
    let max = 1;
    for (let i = 0; i < 10; i++)
      for (let j = 0; j < 10; j++)
        if (i !== j && conf[i][j] > max) max = conf[i][j];
    let h = "<table class='mn-conf-table'><tr><th></th>";
    for (let j = 0; j < 10; j++) h += `<th>${j}</th>`;
    h += "<th class='mn-conf-side'>pred</th></tr>";
    for (let i = 0; i < 10; i++) {
      h += `<tr><th>${i}</th>`;
      for (let j = 0; j < 10; j++) {
        const v = conf[i][j];
        if (i === j) h += `<td class="diag">${v}</td>`;
        else {
          const a = v === 0 ? 0 : 0.15 + 0.6 * (v / max);
          h += `<td style="background:rgba(255,90,90,${a.toFixed(2)})">${v || ""}</td>`;
        }
      }
      h += "</tr>";
    }
    h += "<tr><th class='mn-conf-side'>true</th></tr></table>";
    $("mn-conf").innerHTML = h;
  }

  function refresh() {
    if (!ready) return;
    $("mn-stack").textContent = stackLabel();
    $("mn-grid").textContent = "…";
    fetch("/api/mnist/eval?" + qs())
      .then((r) => r.json())
      .then((d) => {
        if (d.err) { $("mn-grid").textContent = "error: " + d.err; return; }
        $("st-acc").textContent = (d.acc * 100).toFixed(2) + "%";
        $("st-centroid").textContent = (d.centroid_acc * 100).toFixed(2) + "%";
        renderConf(d.confusion);
        sample();
      })
      .catch(() => { $("mn-grid").textContent = "(eval error)"; });
  }

  function sample() {
    fetch(`/api/mnist/sample?${qs()}&seed=${seed}&errors=${onlyErrors ? 1 : 0}`)
      .then((r) => r.json())
      .then((d) => { if (!d.err) renderGrid(d); })
      .catch(() => {});
  }

  function poll() {
    fetch("/api/mnist/status").then((r) => r.json()).then((s) => {
      if (s.err) { setConn(false, "error"); $("mn-status").textContent = "error: " + s.err; return; }
      if (!s.ready) {
        setConn(false, "loading");
        $("mn-status").textContent = s.loading ? "loading pipeline… (first hit builds the statistics layer)" : "starting…";
        setTimeout(poll, 1500);
        return;
      }
      setConn(true, "ready");
      $("st-nf").textContent = s.nf;
      $("st-centroid").textContent = (s.centroid_acc * 100).toFixed(2) + "%";
      if (!s.has_genomes) {
        $("mn-status").textContent = "no trained genomes — run: python -m genreg_train.mnist_pipe";
        $("st-genomes").textContent = "0";
        setTimeout(poll, 5000);       // keep polling; training may be running
        return;
      }
      ready = true;
      $("st-genomes").textContent = s.n_detectors + " det + " + s.n_pairs + " pair"
        + (s.has_mixer ? " + mixer" : "");
      $("st-params").textContent = (s.params / 1000).toFixed(1) + "K";
      $("mn-status").textContent = "ready · " + s.test_n + " held-out test digits"
        + (s.results && s.results.full_test ? " · full-stack test " + (s.results.full_test * 100).toFixed(2) + "%" : "");
      refresh();
    }).catch(() => { setConn(false, "offline"); setTimeout(poll, 2000); });
  }

  function setConn(ok, txt) {
    $("mn-dot").className = "dot" + (ok ? " ok" : " bad");
    $("mn-conn").textContent = txt;
  }

  $("mn-gen").addEventListener("click", () => { seed++; sample(); });
  $("mn-err").addEventListener("click", () => {
    onlyErrors = !onlyErrors;
    $("mn-err").textContent = onlyErrors ? "Show all" : "Show mistakes";
    sample();
  });
  renderLayers(); renderArch(); poll();
})();
