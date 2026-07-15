/* GENREG — CIFAR page (the MNIST specialist pipeline, verbatim, on CIFAR-10).
 * Staged: shows environment stats until a battery has been trained, then
 * behaves exactly like the /mnist page (REST /api/cifar/*).
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  let LABELS = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];

  const LAYERS = [
    { key: "env", name: "Environment", fixed: true,
      desc: "built statistics (zones, profiles, gradient histograms, PCA) + the evolved 5x5x3 detector bank, PCA'd — built/evolved once, never trained end-to-end" },
    { key: "joint", name: "Joint head", fixed: true,
      desc: "the evolved linear classifier genome — centroid warm start, full-train deterministic landscape, magnitude-scaled mutation" },
    { key: "pairs", name: "Pairwise", desc: "45 one-vs-one referee genomes (cat-or-dog?) — fire when the joint head's top-2 are close" },
  ];
  const ARCH = [
    ["Environment", "the same two-part environment as /mnist: built statistics from the data (per-channel zone means, row/column profiles, gradient-orientation histograms on luminance, 128 PCA components) + an EVOLVED detector bank — 5x5x3 conv-kernel genomes with per-neuron evolved activations (8-function catalog), bred on Fisher class-separability, decorrelated, multi-shape mean-pooled (3x3/4x2/2x4). All of it PCA'd to ~1024 dims and frozen."],
    ["Joint head", "one evolved linear genome (env dims x 10): warm-started from the class-centroid head (a pure train statistic), refined on the full-train deterministic landscape with magnitude-scaled mutation and L2 — the recipe that closed the optimisation gap on MNIST."],
    ["Pairwise", "45 one-vs-one referee genomes, each trained only on its two classes, refereeing the joint head's close top-2 calls; margin chosen on validation, never test."],
  ];

  let en = { pairs: true };
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
    const s = ["Environment", "Joint head"];
    if (en.pairs) s.push("Pairwise");
    return s.join("  +  ");
  }

  function qs() { return `mixer=1&pairs=${en.pairs ? 1 : 0}`; }

  function drawImg(px) {
    const c = document.createElement("canvas");
    c.width = 32; c.height = 32;
    c.className = "mn-digit";
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(32, 32);
    for (let i = 0; i < 1024; i++) {
      img.data[i * 4] = px[i * 3];
      img.data[i * 4 + 1] = px[i * 3 + 1];
      img.data[i * 4 + 2] = px[i * 3 + 2];
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
      cell.appendChild(drawImg(it.px));
      const lab = document.createElement("div");
      lab.className = "mn-lab";
      lab.textContent = it.ok ? LABELS[it.pred]
        : (LABELS[it.pred] + " (" + LABELS[it.true] + ")");
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
    for (let j = 0; j < 10; j++) h += `<th>${LABELS[j]}</th>`;
    h += "</tr>";
    for (let i = 0; i < 10; i++) {
      h += `<tr><th>${LABELS[i]}</th>`;
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
    h += "</table>";
    $("mn-conf").innerHTML = h;
  }

  function refresh() {
    if (!ready) return;
    $("mn-stack").textContent = stackLabel();
    $("mn-grid").textContent = "…";
    fetch("/api/cifar/eval?" + qs())
      .then((r) => r.json())
      .then((d) => {
        if (d.err) { $("mn-grid").textContent = d.err; return; }
        $("st-acc").textContent = (d.acc * 100).toFixed(2) + "%";
        $("st-centroid").textContent = (d.centroid_acc * 100).toFixed(2) + "%";
        renderConf(d.confusion);
        sample();
      })
      .catch(() => { $("mn-grid").textContent = "(eval error)"; });
  }

  function sample() {
    fetch(`/api/cifar/sample?${qs()}&seed=${seed}&errors=${onlyErrors ? 1 : 0}`)
      .then((r) => r.json())
      .then((d) => { if (!d.err) renderGrid(d); })
      .catch(() => {});
  }

  function poll() {
    fetch("/api/cifar/status").then((r) => r.json()).then((s) => {
      if (s.err) { setConn(false, "error"); $("mn-status").textContent = "error: " + s.err; return; }
      if (s.labels) LABELS = s.labels;
      if (!s.ready) {
        setConn(false, "loading");
        $("mn-status").textContent = s.loading ? "loading environment… (first hit builds the statistics)" : "starting…";
        setTimeout(poll, 2000);
        return;
      }
      setConn(true, "ready");
      $("st-nf").textContent = s.nf + " (v" + s.feat_version + ")";
      $("st-centroid").textContent = (s.centroid_acc * 100).toFixed(2) + "%";
      if (!s.has_genomes) {
        $("mn-status").textContent = "staged — no trained genomes yet. Run: python -m genreg_train.cifar_pipe --detbank, then --v4";
        $("st-genomes").textContent = "0";
        setTimeout(poll, 8000);
        return;
      }
      ready = true;
      $("st-genomes").textContent = (s.has_joint ? "joint" : "") +
        (s.n_pairs ? " + " + s.n_pairs + " pair" : "") +
        (s.n_detectors ? " + " + s.n_detectors + " det" : "");
      $("st-params").textContent = (s.params / 1000).toFixed(1) + "K";
      $("mn-status").textContent = "ready · " + s.test_n + " held-out test images"
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
