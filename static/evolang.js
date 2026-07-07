/* GENREG — EvoLang page (the WordPipe specialist pipeline).
 * Toggle each evolved specialist genome and watch the generated text change:
 * letters -> real words -> grammatical class order -> context-fit words ->
 * sentences -> real phrases. Generation runs server-side over the trained
 * genomes (REST /api/evolang/*). No gradients anywhere in the model.
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);

  const LAYERS = [
    { key: "vocab", name: "Vocabulary", desc: "emit real words instead of letter noise" },
    { key: "order", name: "Order", desc: "words follow a grammatical class skeleton (next class ← last few classes)" },
    { key: "sel", name: "Selection", desc: "fill each slot with the word that fits its neighbours", tri: true,
      states: ["off", "uni", "bi"], labels: { off: "OFF", uni: "PREV WORD", bi: "BOTH NEIGHBOURS" } },
    { key: "bound", name: "Boundary", desc: "cut the stream into sentences (P(sentence ends) per position)" },
    { key: "commas", name: "Commas", desc: "internal punctuation — P(comma here) from class + clause position" },
    { key: "chunks", name: "Chunks", desc: "emit whole real phrases as units where the pattern matches" },
  ];
  const ARCH = [
    ["Order", "next grammatical class given the last few classes — dense log-prob fitness over ~32 induced POS-like classes. ~3,100 params."],
    ["Selection", "score a candidate word against the previous word via fixed distributional features + a tiny bilinear head. ~580 params."],
    ["Bidirectional", "same, but scored against the previous word AND the next class. ~1,150 params."],
    ["Boundary", "P(a sentence ends here) from the word's class + sentence position. ~580 params."],
    ["Commas", "P(a comma here) from the word's class + clause position — internal punctuation. ~580 params."],
    ["Chunks", "a lexicon of frequent real phrases indexed by class pattern; emit a whole phrase when the skeleton matches — 100% real local adjacencies."],
  ];

  let en = { vocab: true, order: true, sel: "bi", bound: true, commas: true, chunks: true };
  let seed = 3, ready = false;

  function renderLayers() {
    const box = $("ev-layers");
    box.innerHTML = "";
    for (const L of LAYERS) {
      const on = L.tri ? en[L.key] !== "off" : en[L.key];
      const row = document.createElement("div");
      row.className = "ev-layer" + (on ? " on" : "");
      const val = L.tri ? (L.labels[en[L.key]]) : (en[L.key] ? "ON" : "OFF");
      row.innerHTML = `<div class="ev-layer-top"><span class="ev-chip ${on ? "on" : ""}">${val}</span>`
        + `<span class="ev-layer-name">${L.name}</span></div>`
        + `<div class="ev-layer-desc">${L.desc}</div>`;
      row.addEventListener("click", () => {
        if (L.tri) {
          const i = L.states.indexOf(en[L.key]);
          en[L.key] = L.states[(i + 1) % L.states.length];
        } else {
          en[L.key] = !en[L.key];
        }
        renderLayers(); generate();
      });
      box.appendChild(row);
    }
  }

  function renderArch() {
    $("ev-arch").innerHTML = ARCH.map(([n, d]) =>
      `<div class="ev-arch-row"><span class="ev-arch-name">${n}</span><span class="ev-arch-desc">${d}</span></div>`).join("");
  }

  function stackLabel() {
    const s = [];
    if (en.vocab) s.push("Vocabulary");
    if (en.order) s.push("Order");
    if (en.sel === "uni") s.push("Selection");
    else if (en.sel === "bi") s.push("Bidirectional");
    if (en.bound && en.vocab) s.push("Boundary");
    if (en.commas && en.vocab) s.push("Commas");
    if (en.chunks && en.vocab) s.push("Chunks");
    return s.length ? s.join("  +  ") : "nothing";
  }

  function qs() {
    return `vocab=${en.vocab ? 1 : 0}&order=${en.order ? 1 : 0}&bound=${en.bound ? 1 : 0}`
      + `&commas=${en.commas ? 1 : 0}&chunks=${en.chunks ? 1 : 0}&sel=${en.sel}&seed=${seed}`;
  }

  function generate() {
    if (!ready) return;
    $("ev-stack").textContent = stackLabel();
    $("ev-out").textContent = "…";
    fetch("/api/evolang/generate?" + qs())
      .then((r) => r.json())
      .then((d) => { $("ev-out").textContent = d.text || "(no output)"; })
      .catch(() => { $("ev-out").textContent = "(generation error)"; });
  }

  function setConn(ok, txt) {
    $("ev-dot").className = "dot" + (ok ? " ok" : " bad");
    $("ev-conn").textContent = txt;
  }

  function poll() {
    fetch("/api/evolang/status").then((r) => r.json()).then((s) => {
      if (s.err) { setConn(false, "error"); $("ev-status").textContent = "error: " + s.err; return; }
      if (!s.ready) {
        setConn(false, "loading");
        $("ev-status").textContent = s.loading ? "loading pipeline… (first hit builds the corpus)" : "starting…";
        setTimeout(poll, 1500);
        return;
      }
      ready = true;
      setConn(true, "ready");
      $("st-corpus").textContent = (s.corpus_chars / 1e6).toFixed(1) + "M";
      $("st-vocab").textContent = s.vocab;
      $("st-classes").textContent = s.n_classes;
      $("st-chunks").textContent = s.chunk_phrases;
      $("ev-status").textContent = s.has_genomes
        ? "ready · genomes: " + s.trained.join(", ")
        : "no trained genomes — run demo/build_cache.py";
      generate();
    }).catch(() => { setConn(false, "offline"); setTimeout(poll, 2000); });
  }

  $("ev-gen").addEventListener("click", () => { seed++; generate(); });
  renderLayers(); renderArch(); poll();
})();
