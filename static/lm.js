/* GENREG — LM page. Rebuild of the language pipeline. Four genome groups:
 * "punctuation" — 5 binary genomes reading the words BEFORE a mark.
 * "opener" — the mirror image: 2 binary genomes reading ONLY a sentence's
 * first word, to confirm what intent the sentence is headed for.
 * "length" — 1 genome: given a partial sentence, is it complete or does it
 * need to keep growing? Drives dynamic-length generation.
 * "fill" — 1 genome: given the words around a blank, does a candidate word
 * fit? Contrastive, scored against the whole vocabulary at inference.
 * length + fill compose into hangman-style Generate below.
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);

  function fmtPct(x) {
    return (x === null || x === undefined) ? "—" : (x * 100).toFixed(1) + "%";
  }

  function headlineAcc(g) {
    return g.holdout_balanced_acc !== undefined ? g.holdout_balanced_acc : g.holdout_acc;
  }

  function renderGenomeChips(allGenomes) {
    $("lm-genomes").innerHTML = allGenomes.map(g =>
      `<div class="ev-layer on" title="${g.desc}">`
      + `<div class="ev-layer-top"><span class="ev-chip on">${g.group}</span>`
      + `<span class="ev-layer-name">${g.key}</span></div>`
      + `<div class="field-hint" style="margin:2px 0 0">${fmtPct(headlineAcc(g))} acc.</div>`
      + `</div>`).join("");
  }

  function renderSplits(elId, genomes) {
    const el = $(elId);
    if (!genomes || !genomes.length) { el.textContent = "—"; return; }
    el.innerHTML = genomes.map(g => {
      const isContrastive = g.holdout_balanced_acc === undefined;
      const rows = isContrastive ? "" : Object.entries(g.recall || {}).map(([name, v]) =>
        `<div class="ev-arch-row"><span class="ev-arch-name">${name}</span>`
        + `<span class="ev-arch-desc">recall ${fmtPct(v)}</span></div>`).join("");
      const tiles = isContrastive
        ? `<div class="tlm-tile"><div class="tlm-tile-label">Holdout acc.</div><div class="tlm-tile-value">${fmtPct(g.holdout_acc)}</div></div>
           <div class="tlm-tile"><div class="tlm-tile-label">Chance</div><div class="tlm-tile-value">${fmtPct(g.chance)}</div></div>
           <div class="tlm-tile"><div class="tlm-tile-label">Examples</div><div class="tlm-tile-value">${g.n_examples.toLocaleString()}</div></div>`
        : `<div class="tlm-tile"><div class="tlm-tile-label">Balanced acc.</div><div class="tlm-tile-value">${fmtPct(g.holdout_balanced_acc)}</div></div>
           <div class="tlm-tile"><div class="tlm-tile-label">Raw acc. (natural dist.)</div><div class="tlm-tile-value">${fmtPct(g.holdout_raw_acc)}</div></div>
           <div class="tlm-tile"><div class="tlm-tile-label">Examples</div><div class="tlm-tile-value">${g.n_examples.toLocaleString()}</div></div>`;
      return `<div class="tlm-card" style="margin-bottom:10px">
        <div class="tlm-card-head"><h2>${g.key} <span class="tlm-card-sub">${g.desc}</span></h2></div>
        <div class="tlm-tiles">${tiles}</div>
        ${rows}
      </div>`;
    }).join("");
  }

  async function refreshStatus() {
    try {
      const r = await fetch("/api/lm/status");
      const s = await r.json();
      $("lm-dot").className = "dot" + (s.ready ? " ok" : "");
      $("lm-conn").textContent = s.ready ? "ready" : (s.err || "not trained yet");
      if (!s.ready) {
        $("lm-status").textContent = s.err || "loading…";
        return;
      }
      const groups = s.groups || [];
      const allGenomes = groups.flatMap(g => g.genomes);
      const accs = allGenomes.map(headlineAcc).filter(v => v !== undefined);
      const meanAcc = accs.length ? accs.reduce((a, v) => a + v, 0) / accs.length : null;
      $("lm-status").textContent = `${allGenomes.length} genomes across ${groups.length} groups — `
        + `mean acc ${fmtPct(meanAcc)} (chance ${fmtPct(s.chance)})`;
      $("st-count").textContent = allGenomes.length;
      $("st-acc").textContent = fmtPct(meanAcc);
      $("st-chance").textContent = fmtPct(s.chance);
      $("st-vocab").textContent = s.vocab_size ?? "—";
      renderGenomeChips(allGenomes);
      const byGroup = (name) => (groups.find(g => g.group === name) || {}).genomes || [];
      renderSplits("lm-splits-punctuation", byGroup("punctuation"));
      renderSplits("lm-splits-opener", byGroup("opener"));
      renderSplits("lm-splits-length", byGroup("length"));
      renderSplits("lm-splits-fill", byGroup("fill"));
      renderSplits("lm-splits-next", byGroup("next"));
    } catch (e) {
      $("lm-conn").textContent = "offline";
    }
  }

  function renderRanked(el, ranked, contextLabel) {
    el.innerHTML = `<div class="field-hint">${contextLabel}</div>` +
      ranked.map((row, i) =>
        `<div class="ev-arch-row"><span class="ev-arch-name">${i === 0 ? "▸ " : ""}${row.mark === "," ? "comma" : row.mark}</span>` +
        `<span class="ev-arch-desc">${row.intent} — ${fmtPct(row.prob)}</span></div>`).join("");
  }

  async function runRecognize() {
    const text = $("lm-input").value.trim();
    if (!text) return;
    $("lm-result").textContent = "recognizing…";
    try {
      const r = await fetch("/api/lm/recognize?text=" + encodeURIComponent(text));
      const d = await r.json();
      if (d.err) { $("lm-result").textContent = d.err; return; }
      renderRanked($("lm-result"), d.ranked, `context: "${d.context_words.join(" ")}"`);
    } catch (e) {
      $("lm-result").textContent = "request failed";
    }
  }

  async function runRecognizeOpener() {
    const word = $("lm-opener-input").value.trim();
    if (!word) return;
    $("lm-opener-result").textContent = "confirming…";
    try {
      const r = await fetch("/api/lm/recognize_opener?word=" + encodeURIComponent(word));
      const d = await r.json();
      if (d.err) { $("lm-opener-result").textContent = d.err; return; }
      renderRanked($("lm-opener-result"), d.ranked, `first word: "${d.first_word}"`);
    } catch (e) {
      $("lm-opener-result").textContent = "request failed";
    }
  }

  async function runGenerate() {
    const word = $("lm-gen-input").value.trim();
    if (!word) return;
    const temperature = parseFloat($("lm-gen-temp").value) || 0.7;
    $("lm-gen-text").textContent = "generating…";
    $("lm-gen-trace").textContent = "—";
    try {
      const seed = Math.floor(Math.random() * 1e9);
      const r = await fetch(`/api/lm/generate?word=${encodeURIComponent(word)}&temperature=${temperature}&seed=${seed}`);
      const d = await r.json();
      if (d.err) { $("lm-gen-text").textContent = d.err; return; }
      $("lm-gen-text").textContent = d.text;
      $("lm-gen-trace").innerHTML = `<div class="field-hint" style="margin-bottom:6px">intent: ${d.mark_intent} (${d.mark})  ·  ${d.words.length} words  ·  fill order below</div>` +
        d.trace.map((step, i) => {
          if (step.action === "end") {
            return `<div class="ev-arch-row"><span class="ev-arch-name">${i}. end</span><span class="ev-arch-desc">committed "${step.mark}"${step.note ? " (" + step.note + ")" : ""}</span></div>`;
          }
          const prob = step.prob !== undefined ? ` — ${fmtPct(step.prob)}` : "";
          return `<div class="ev-arch-row"><span class="ev-arch-name">${i}. ${step.word}</span><span class="ev-arch-desc">${step.note || "filled"}${prob}</span></div>`;
        }).join("");
    } catch (e) {
      $("lm-gen-text").textContent = "request failed";
    }
  }

  refreshStatus();
  $("lm-run").addEventListener("click", runRecognize);
  $("lm-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runRecognize(); });
  $("lm-opener-run").addEventListener("click", runRecognizeOpener);
  $("lm-opener-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runRecognizeOpener(); });
  $("lm-gen-run").addEventListener("click", runGenerate);
  $("lm-gen-input").addEventListener("keydown", (e) => { if (e.key === "Enter") runGenerate(); });
})();
