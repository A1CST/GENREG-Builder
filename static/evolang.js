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
    { key: "altern", name: "Alternation", desc: "content-function rhythm — a function word needs a content word beside it (kills “the of”, “and and”)" },
    { key: "agree", name: "Agreement", desc: "re-rank so subject/verb & modal/aux agree (“he could be” not “he could is”)" },
    { key: "sem", name: "Semantic", desc: "prefer content words that actually co-occur in the corpus (“the horse stable”, not “the horse democracy”)" },
    { key: "rep", name: "No-repeat", desc: "don’t reuse a content word that appeared in the last few positions (function words may recur)" },
    { key: "open", name: "Opener", desc: "start each sentence with a plausible opener (a pronoun/article/wh, not “Of”/“To”/“Is”)" },
    { key: "close", name: "Closer", desc: "end each sentence on a plausible ender (a noun/verb, not a dangling “of/the/to”)" },
    { key: "bound", name: "Boundary", desc: "cut the stream into sentences (P(sentence ends) per position)" },
    { key: "commas", name: "Commas", desc: "internal punctuation — P(comma here) from class + clause position" },
    { key: "chunks", name: "Chunks", desc: "emit whole real phrases as units where the pattern matches. OFF by default — measurably raises content-word repetition (No-repeat only tracks single words, not whole emitted phrases)." },
    { key: "hyper", name: "Hypernym", desc: "EXPERIMENTAL — nudges toward “is-a” continuations (dog→animal). Wikipedia-trained, validated 10/10 on held-out probes as a STANDALONE relation genome; not yet battery-tested for its effect on generation, so it's off by default. High gamma reads as a taxonomy climb, not natural prose — try it at low mix with Selection/Semantic on." },
    { key: "mero", name: "Meronym", desc: "EXPERIMENTAL — nudges toward “part-of” continuations (wheel→car). Wikipedia-trained, validated 9/10 on held-out probes as a standalone relation genome; not yet battery-tested for its effect on generation." },
    { key: "synant", name: "Synonym/Antonym", desc: "EXPERIMENTAL — given a related pair, biases toward same-or-opposite-meaning continuations. Wikipedia-trained, PARTIAL validation (11/14 probes) — known to still misfire on size-adjective pairs (big/large). Not yet battery-tested for its effect on generation." },
    { key: "sent_type", name: "Sentence type", desc: "EXPERIMENTAL — flips a coin at the corpus question-rate (8.2%) per sentence; if “question”, biases the opener toward question-words (do/will/what/is…) and closes with “?”. Probe: all 18 hand-picked question-openers outscored all 11 statement-openers. Generation check: flagged-question rate landed at 8.2% (matches corpus), ~50% of flagged sentences opened with a real question word (was ~0% before). No formal adj-hit/distinct guardrail sweep yet." },
    { key: "lenplan", name: "Sentence length plan", desc: "EXPERIMENTAL — flips a coin at the corpus long-sentence rate (48.8% of sentences ran past the 14-word median); biases the opener toward/away from long-sentence-opener words (although/because/since…) and reshapes the boundary probability to lean toward finishing before/after the median. Probe passed but WEAKER than Sentence-type: mean long-opener score beat mean short-opener score (+0.53 vs -0.13), but several openers tied on identical scores — the shared function-feature space caps how sharp this one gets. Generation spot-check found NO measurable length-distribution change — wired but inert at safe gammas." },
    { key: "pronominal", name: "Pronominalization", desc: "EXPERIMENTAL — Passage-stage genome, no training. Reuses the No-repeat genome's recency buffer: a content word re-mentioned within the last 15 words gets replaced by a generic pronoun (\"it\") 60% of the time instead of repeating the literal noun. Measured effect: 'it' frequency rose 0.77%→1.11% of words (+44% relative) with this toggle on — a real, measurable generation-time change." },
  ];
  const ARCH = [
    ["Order", "next grammatical class given the last few classes — dense log-prob fitness over ~32 induced POS-like classes. ~3,100 params."],
    ["Selection", "score a candidate word against the previous word via fixed distributional features + a tiny bilinear head. ~580 params."],
    ["Bidirectional", "same, but scored against the previous word AND the next class. ~1,150 params."],
    ["Alternation", "the content/function rhythm — a function word (the, of, and, to) needs a content word beside it; only a few function→function transitions are legal (of→the yes, the→of no). 14 function-type features × a tiny bilinear head (12/12 on held-out minimal pairs, ~200 params). Applied at BOTH the order skeleton (class centroids in function-feature space) and word selection — cuts function→function runs 58%→37% while raising local corpus-bigram fluency."],
    ["Agreement", "grammatical agreement (finiteness / number) — 22 rule-based morphology features × a tiny bilinear head (modal→bare, subject-number→copula; 12/12 on held-out minimal pairs, ~480 params). Applied at both the order skeleton and word selection — roughly halves modal/subject-verb agreement violations while holding fluency."],
    ["Semantic", "meaning-level pair fit — prefer content words that actually co-occur in the corpus within a window. Bilinear head over the 24-d distributional (SVD) word features. Raised real content-adjacency ~33%→40% with no fluency cost. ~580 params."],
    ["No-repeat", "an evolved recency-penalty curve — don't reuse a content word from the last few positions (function words may recur). Drove residual content-word repetition to ~0. Tiny (~10 params), stateful over the generated buffer."],
    ["Opener", "a unary classifier over function-type features that fires at each sentence start — favours plausible openers (pronoun/article/wh) over “Of/To/Is”. Cut bad sentence-openings ~17%→8% at no fluency cost. ~14 params."],
    ["Closer", "a unary ender classifier that reshapes WHERE periods land — end sentences after a noun/verb, not a dangling “of/the/to”. Rate-preserving (frequency-centred), so sentence length holds. Cut bad sentence-endings ~65%→33%. ~14 params."],
    ["Boundary", "P(a sentence ends here) from the word's class + sentence position. ~580 params."],
    ["Commas", "P(a comma here) from the word's class + clause position — internal punctuation. ~580 params."],
    ["Chunks", "a lexicon of frequent real phrases indexed by class pattern; emit a whole phrase when the skeleton matches — 100% real local adjacencies."],
    ["Hypernym (experimental)", "standalone relation genome, NOT trained on this pipeline's corpus — trained on a separate 51M-word Wikipedia corpus (30K vocab, 128-d SVD features), then crosswalked into this pipeline's word list by lookup. Bilinear head learns directional is-a: score(dog,animal) > score(animal,dog). 10/10 on held-out probes, directional held-out accuracy 0.86. Wired as a selection re-rank like Semantic, but untested for its effect on generation quality — that's what this toggle is for."],
    ["Meronym (experimental)", "same recipe as Hypernym for part-of (wheel→car). 9/10 on held-out probes. Also standalone/crosswalked, also untested in generation."],
    ["Synonym/Antonym (experimental)", "same recipe, but asks a narrower question: given a pair already known to be related, is it same-meaning or opposite-meaning? (Training two separate detectors failed — see documentation/GA_SCALING_FIELD_NOTES.pdf for why.) 11/14 on held-out probes; known residual failure on size-adjective pairs."],
  ];

  let en = { vocab: true, order: true, sel: "bi", altern: true, agree: true, sem: true, rep: true, open: true, close: true, bound: true, commas: true, chunks: false, hyper: false, mero: false, synant: false, sent_type: false, lenplan: false, pronominal: false };
  let seed = 3, ready = false;

  function renderLayers() {
    const box = $("ev-layers");
    box.innerHTML = "";
    for (const L of LAYERS) {
      const on = L.tri ? en[L.key] !== "off" : en[L.key];
      const row = document.createElement("div");
      row.className = "ev-layer" + (on ? " on" : "");
      row.title = L.name + " — " + L.desc;   // description as tooltip (no scrollbar)
      const val = L.tri ? (L.labels[en[L.key]]) : (en[L.key] ? "ON" : "OFF");
      row.innerHTML = `<div class="ev-layer-top"><span class="ev-chip ${on ? "on" : ""}">${val}</span>`
        + `<span class="ev-layer-name">${L.name}</span></div>`;
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
    if (en.altern && en.sel !== "off" && en.vocab) s.push("Alternation");
    if (en.agree && en.sel !== "off" && en.vocab) s.push("Agreement");
    if (en.sem && en.sel !== "off" && en.vocab) s.push("Semantic");
    if (en.hyper && en.sel !== "off" && en.vocab) s.push("Hypernym*");
    if (en.mero && en.sel !== "off" && en.vocab) s.push("Meronym*");
    if (en.synant && en.sel !== "off" && en.vocab) s.push("Synonym/Antonym*");
    if (en.sent_type && en.vocab) s.push("Sentence-type*");
    if (en.lenplan && en.vocab) s.push("Length-plan*");
    if (en.pronominal && en.vocab) s.push("Pronominalization*");
    if (en.rep && en.sel !== "off" && en.vocab) s.push("No-repeat");
    if (en.open && en.sel !== "off" && en.vocab) s.push("Opener");
    if (en.close && en.vocab) s.push("Closer");
    if (en.bound && en.vocab) s.push("Boundary");
    if (en.commas && en.vocab) s.push("Commas");
    if (en.chunks && en.vocab) s.push("Chunks");
    return s.length ? s.join("  +  ") : "nothing";
  }

  function qs() {
    return `vocab=${en.vocab ? 1 : 0}&order=${en.order ? 1 : 0}&bound=${en.bound ? 1 : 0}`
      + `&commas=${en.commas ? 1 : 0}&chunks=${en.chunks ? 1 : 0}&sel=${en.sel}`
      + `&agree=${en.agree ? 1 : 0}&altern=${en.altern ? 1 : 0}`
      + `&sem=${en.sem ? 1 : 0}&rep=${en.rep ? 1 : 0}&open=${en.open ? 1 : 0}`
      + `&close=${en.close ? 1 : 0}&hyper=${en.hyper ? 1 : 0}&mero=${en.mero ? 1 : 0}`
      + `&synant=${en.synant ? 1 : 0}&sent_type=${en.sent_type ? 1 : 0}`
      + `&lenplan=${en.lenplan ? 1 : 0}&pronominal=${en.pronominal ? 1 : 0}&seed=${seed}`;
  }

  function renderOutput(text) {
    const out = $("ev-out"), stats = $("ev-repstats");
    // count word frequencies (case-insensitive, punctuation stripped)
    const norm = (t) => t.toLowerCase().replace(/[.,?!;:'"]/g, "");
    const freq = {};
    const toks = text.split(/\s+/).filter(Boolean);
    let content = 0;
    for (const t of toks) {
      const w = norm(t);
      if (!w) continue;
      content++; freq[w] = (freq[w] || 0) + 1;
    }
    const maxc = Math.max(1, ...Object.values(freq));
    // render text with repeated words highlighted (brightness ∝ repeat count)
    out.innerHTML = "";
    for (const t of toks) {
      const w = norm(t), c = freq[w] || 0;
      const span = document.createElement("span");
      span.textContent = t + " ";
      if (w && c >= 2) {
        const a = 0.12 + 0.5 * ((c - 1) / (maxc - 1 || 1));
        span.style.background = "rgba(90,170,255," + a.toFixed(2) + ")";
        span.style.borderRadius = "3px";
        span.title = w + " ×" + c;
      }
      out.appendChild(span);
    }
    // stats line + top repeated words
    const distinct = Object.keys(freq).length;
    const uniqPct = content ? Math.round((distinct / content) * 100) : 0;
    const repeated = Object.entries(freq).filter(([, c]) => c >= 2)
      .sort((a, b) => b[1] - a[1]);
    const top = repeated.slice(0, 8).map(([w, c]) => `${w}<b>×${c}</b>`).join("  ");
    const repTokens = repeated.reduce((s, [, c]) => s + c, 0);
    stats.innerHTML =
      `<span class="ev-repnum">${content}</span> words · `
      + `<span class="ev-repnum">${distinct}</span> distinct · `
      + `<span class="ev-repnum ${uniqPct < 60 ? "bad" : uniqPct < 78 ? "warn" : "good"}">${uniqPct}%</span> unique · `
      + `<span class="ev-repnum ${repTokens > content * 0.35 ? "bad" : ""}">${repTokens}</span> repeat-tokens`
      + (top ? `<div class="ev-toprep">most repeated: ${top}</div>` : "");
  }

  function generate() {
    if (!ready) return;
    $("ev-stack").textContent = stackLabel();
    $("ev-out").textContent = "…"; $("ev-repstats").textContent = "";
    fetch("/api/evolang/generate?" + qs())
      .then((r) => r.json())
      .then((d) => { renderOutput(d.text || "(no output)"); })
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
      if (s.params != null) {
        $("st-params").textContent = (s.params / 1000).toFixed(1) + "K";
        const full = s.full_kb >= 1024 ? (s.full_kb / 1024).toFixed(1) + " MB" : s.full_kb + " KB";
        $("st-deploy").textContent = s.heads_kb + " KB / " + full;
        $("st-deploy").title = "evolved genome heads / full pipeline (features + class tables + chunk lexicon)";
      }
      const relNote = s.rel_coverage_pct ? ` · relation-genome vocab coverage ${s.rel_coverage_pct}%` : "";
      $("ev-status").textContent = s.has_genomes
        ? "ready · genomes: " + s.trained.join(", ") + relNote
        : "no trained genomes — run demo/build_cache.py";
      generate();
    }).catch(() => { setConn(false, "offline"); setTimeout(poll, 2000); });
  }

  $("ev-gen").addEventListener("click", () => { seed++; generate(); });
  renderLayers(); renderArch(); poll();
})();
