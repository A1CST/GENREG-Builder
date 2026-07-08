/* GENREG — EvoLang layer/flow map (/evolang/layers).
 * Static data + a hand-rolled SVG diagram (same lightweight pattern as
 * diff.js/animation.js — NOT the PURE node-graph editor; this is read-only,
 * nothing here is draggable). Three horizontal LAYER bands (Structural /
 * Semantic / Abstraction) stacked top to bottom; within every band, nodes
 * sit under the PIPELINE STAGE column (Skeleton / Fill / Boundary) they
 * actually wire into in wordpipe_service.py's generate(), with a bold
 * backbone across the top showing the real generation flow.
 *
 * This is also where future genomes get PLANNED before they're built —
 * status: 'planned' nodes are design intent, not yet attempted (distinct
 * from 'cut', which was attempted and rejected on real evidence).
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

  // stages: which pipeline step(s) each genome's output feeds into, in
  // wordpipe_service.py generate() — 'skeleton' = Order class-sequence
  // reranks, 'fill' = word-selection reranks/bonus, 'boundary' = punctuation
  //
  // short = always-visible one-line caption (fits the node box).
  // desc  = full reasoning, shown in the hover tooltip.
  const NODES = [
    // ---- structural (form: grammar, position, rhythm) ----
    { id: "content_select", name: "Content selection", layer: "semantic", stages: ["content"], status: "experimental",
      short: "pick the meaning FIRST",
      desc: "WIRED, verified (2026-07-08) — user-directed architecture flip: every other stage is structure-first (Order picks a class skeleton blind, meaning gets bolted on after). This runs BEFORE Order: picks 3-5 mutually-related content words using whichever relation genomes are enabled (Hypernym/Meronym/Synonym-Antonym/Semantic), stochastic sampling (softmax, not argmax). The same evolved Order/Fill genomes then place each reserved word into the first matching class slot instead of running word-selection there — no new training, structure accommodates chosen meaning instead of the reverse. Verified: selected sets score +0.58 higher on relatedness than random (t=8.25, 40 samples). Placement rate 66.5% (133/200 words, 50 samples) — a third of chosen words don't find a matching slot and get dropped. Does NOT fix word-level fluency — Order/Fill are unchanged, only WHICH words fill content slots changed. `/api/evolang/meaning_first`, \"Meaning-first\" button on /evolang." },
    { id: "vocab", name: "Vocabulary", layer: "structural", stages: ["fill"], status: "shipped",
      short: "emit real words, not letters",
      desc: "Emit real English words, not letter noise. The vocabulary/class table every other genome fills slots from." },
    { id: "order", name: "Order", layer: "structural", stages: ["skeleton"], status: "shipped",
      short: "next class from last 4 classes",
      desc: "Next grammatical class given the last 4 (or 8, experimentally) classes. The skeleton every word gets filled into." },
    { id: "sel", name: "Selection", layer: "structural", stages: ["fill"], status: "shipped",
      short: "word that fits the previous word",
      desc: "Pick the word that fits the previous word. Base scorer every rerank below adds onto." },
    { id: "bisel", name: "Bidirectional", layer: "structural", stages: ["fill"], status: "shipped",
      short: "fits prev word + next class",
      desc: "Selection, but scored against the previous word AND the next class." },
    { id: "altern", name: "Alternation", layer: "structural", stages: ["skeleton", "fill"], status: "shipped",
      short: "content/function word rhythm",
      desc: "Content/function rhythm — a function word needs a content word beside it. Wired at BOTH the skeleton and word-fill." },
    { id: "agree", name: "Agreement", layer: "structural", stages: ["skeleton", "fill"], status: "shipped",
      short: "subject/verb, modal/aux agreement",
      desc: "Subject/verb & modal/aux agreement. Wired at both the skeleton and word-fill." },
    { id: "rep", name: "No-repeat", layer: "structural", stages: ["fill"], status: "shipped",
      short: "don't reuse a recent content word",
      desc: "Don't reuse a content word from the last few positions." },
    { id: "open", name: "Opener", layer: "structural", stages: ["fill"], status: "shipped",
      short: "plausible sentence-starting word",
      desc: "Bias the first word of each sentence toward plausible openers." },
    { id: "close", name: "Closer", layer: "structural", stages: ["boundary"], status: "shipped",
      short: "end on a noun/verb, not \"of/the\"",
      desc: "Reshape WHERE periods land — end sentences on a noun/verb, not a dangling of/the/to." },
    { id: "bound", name: "Boundary", layer: "structural", stages: ["boundary"], status: "shipped",
      short: "P(sentence ends here)",
      desc: "P(sentence ends here) from class + position. Cuts the stream into sentences." },
    { id: "commas", name: "Commas", layer: "structural", stages: ["boundary"], status: "shipped",
      short: "P(comma here)",
      desc: "P(comma here) from class + clause position — internal punctuation." },
    { id: "chunks", name: "Chunks", layer: "structural", stages: ["fill"], status: "shipped-off",
      short: "emit whole real phrases",
      desc: "Emit whole real phrases as units. OFF by default — raises content-word repetition." },

    // ---- semantic (meaning: content co-occurrence in a built distributional space) ----
    { id: "sem", name: "Semantic (adjacency)", layer: "semantic", stages: ["fill"], status: "shipped",
      short: "content words that co-occur",
      desc: "Prefer content words that actually co-occur in the corpus within a window. First meaning-level genome." },
    { id: "sem_wide", name: "Wider co-occurrence", layer: "semantic", stages: ["fill"], status: "cut",
      short: "co-occur within ±5, not ±1",
      desc: "CUT (2026-07-07 battery, stage2.py) — content words co-occurring in a ±5 window instead of adjacency's ±1. Marginal gain (cpr 35.5→36.7) at low gamma, hurt at higher gamma; adjacency ±1 already captures most of the signal." },
    { id: "sem_bridge", name: "Lexical bridge", layer: "semantic", stages: ["fill"], status: "cut",
      short: "carry a content word cross-sentence",
      desc: "CUT (2026-07-07 battery, stage2.py) — reward reusing a content word from the previous sentence. Baseline carryover (27.5%) already EXCEEDS the corpus rate (23%); boosting further caused unnatural repetition (distinct dropped 53→46)." },
    { id: "sem_coh_local", name: "Sentence coherence", layer: "semantic", stages: ["fill"], status: "cut",
      group: "coherence", groupLabel: "Coherence (composed)",
      short: "fits the current sentence?",
      desc: "CUT (2026-07-08). Split out of a single \"topical drift\" idea, which conflated two different timescales into one compound question — see Theme consistency. This half: does the candidate fit a TIGHT, fast-moving centroid of the last 8 content words. `genreg_train/coherence.py`: val_acc 0.525, barely above chance. Diagnosis: NOT proof coherence itself is unlearnable — averaging several words' features into one mean-centroid likely destroys the information a linear head needs. Would need nearest-neighbor pooling against individual recent words instead of mean pooling; not attempted this round." },
    { id: "sem_coh_theme", name: "Theme consistency", layer: "semantic", stages: ["fill"], status: "cut",
      group: "coherence", groupLabel: "Coherence (composed)",
      short: "fits the passage's broader theme?",
      desc: "CUT (2026-07-08). The passage-level half of the split-out \"topical drift\" idea — does the candidate fit a SLOW-moving centroid over the last 50 content words (a stand-in for \"the passage\", no document markers exist). Also absorbs what a separate \"Domain purity\" genome would have measured. Same file/run as Sentence coherence: val_acc 0.531, also barely above chance, same mean-pooling diagnosis." },
    { id: "sem_collo", name: "Collocation strength", layer: "semantic", stages: ["fill"], status: "cut",
      short: "is this a real fixed phrase pair?",
      desc: "PARTIAL, left unwired (2026-07-08). Tighter than loose window co-occurrence — is this SPECIFIC word pair a common fixed collocation (\"depend on\", \"consist of\") vs merely topically related. Also absorbs the separate \"Verb-preposition\" idea (fixed subcategorization like \"depend ON\") — no POS tags exist to isolate \"verb+governed prep\" from \"any tight content-function bigram\", so one genome covers both without a redundant near-duplicate. `genreg_train/collocation.py`, trained on the I2 primary: val_acc 0.782, but probe only 6/8 correct (75%) — fails on textbook cases (\"depend on\" scored below \"depend at\"). Real signal, not clean enough to trust in generation yet; would need a sharper feature space or more mining data." },
    { id: "sem_parallel", name: "List parallelism", layer: "semantic", stages: ["fill"], status: "cut",
      short: "coordinated items, same type?",
      desc: "PARTIAL, left unwired (2026-07-08). When two content words are directly coordinated (\"X and Y\"), are they distributionally the same kind of thing. `genreg_train/parallelism.py`, trained on the I2 primary: val_acc 0.768. Probe: 8/10 correct (80%) — but fails on two of the most canonical same-type pairs in the language (\"dog\"/\"cat\", \"king\"/\"queen\"), which is a concerning place to fail. Real, majority-correct signal, left unwired rather than trusted, same treatment as Collocation strength." },

    // ---- abstraction (relations built on top of the semantic space; the newest, least-tested tier) ----
    { id: "hyper", name: "Hypernym", layer: "abstraction", stages: ["fill"], status: "experimental",
      short: "is-a bias (dog→animal)",
      desc: "Is-a bias (dog→animal). VALIDATED standalone (10/10 probes) but OFF by default — no generation-time battery yet." },
    { id: "mero", name: "Meronym", layer: "abstraction", stages: ["fill"], status: "experimental",
      short: "part-of bias (wheel→car)",
      desc: "Part-of bias (wheel→car). VALIDATED-weaker standalone (9/10 probes), OFF by default." },
    { id: "synant", name: "Synonym/Antonym", layer: "abstraction", stages: ["fill"], status: "experimental",
      short: "same- vs opposite-meaning bias",
      desc: "Given a related pair, same- or opposite-meaning bias. PARTIAL (11/14 probes), OFF by default. Confirmed causing 'of the X of the Y' chains when ON at full gamma." },
    { id: "polysemy", name: "Polysemy", layer: "abstraction", stages: ["fill"], status: "cut",
      short: "does this word have >1 sense?",
      desc: "CUT — val_acc looked strong but probes failed; Wikipedia's dominant-sense skew defeats the NN-spread proxy." },
    { id: "register", name: "Register", layer: "abstraction", stages: ["fill"], status: "cut",
      short: "formal narration vs informal dialogue",
      desc: "CUT — quote-span dialogue/narration mining produced an incoherent signal." },
    { id: "analogy", name: "Analogy", layer: "abstraction", stages: ["fill"], status: "cut",
      short: "does A:B match C:D?",
      desc: "CUT AS DESIGNED, not a dead end — a LAYER-3 question (do two pairs share a relation?) asked with layer-2 raw-offset machinery. Needs a genome over Hypernym/Meronym/Synant OUTPUTS, not built yet." },
    { id: "oblig", name: "Open-obligation", layer: "abstraction", stages: ["boundary"], status: "cut",
      short: "track unclosed prep/det phrases",
      desc: "CUT — first attempt at a stateful, composed genome (track unclosed prep/det phrases across a sentence). Bug: double-counted nested prep+det as two obligations instead of one; made things worse. The template for what a real abstraction-tier genome needs to look like." },

    // Sentiment, decomposed. The monolithic version was CUT (mining inverted:
    // war scored above joy) — same discipline as the rest of this project:
    // when a question is too hard, make it smaller. These four PLANNED
    // genomes, composed, are what a single "positive or negative?" genome
    // was trying to be — each a tighter binary question with a cleaner,
    // more mineable corpus signal than the monolith had.
    { id: "sent_good", name: "Good", layer: "abstraction", stages: ["fill"], status: "planned",
      group: "sentiment", groupLabel: "Sentiment (composed)",
      short: "describes something good?",
      desc: "PLANNED (replaces monolithic Sentiment, CUT). Does this word describe something good? Seed cluster: excellent, wonderful, pleased." },
    { id: "sent_bad", name: "Bad", layer: "abstraction", stages: ["fill"], status: "planned",
      group: "sentiment", groupLabel: "Sentiment (composed)",
      short: "describes something bad?",
      desc: "PLANNED (replaces monolithic Sentiment, CUT). Does this word describe something bad? Seed cluster: terrible, awful, angry." },
    { id: "sent_intensity", name: "Intensity", layer: "abstraction", stages: ["fill"], status: "planned",
      group: "sentiment", groupLabel: "Sentiment (composed)",
      short: "how strong/extreme?",
      desc: "PLANNED (replaces monolithic Sentiment, CUT). How strong or extreme is this word, independent of polarity — \"warm\" vs \"scorching\", \"annoyed\" vs \"furious\"." },
    { id: "sent_emotion", name: "Emotion", layer: "abstraction", stages: ["fill"], status: "planned",
      group: "sentiment", groupLabel: "Sentiment (composed)",
      short: "emotion, or neutral fact?",
      desc: "PLANNED (replaces monolithic Sentiment, CUT). Does this word describe an emotional state at all, vs a neutral fact/object — a gate before polarity even applies." },

    // ---- gap analysis (user, 2026-07-08): Skeleton is thin (3 genomes vs
    // Fill's 20+); Fill is missing specific lexical-grammatical relationships;
    // two pipeline stages that DON'T EXIST YET (Revision, Passage) are where
    // the next real structural gaps live. All PLANNED — none built.
    { id: "sent_type", name: "Sentence type", layer: "structural", stages: ["skeleton"], status: "experimental",
      short: "question, or statement?",
      desc: "WIRED, real generation effect measured, FAILS the full guardrail battery (2026-07-08). Probe: all 18 hand-picked question-openers (do/will/what/is...) scored above all 11 statement-openers. Wired: flips a coin at the corpus question-rate (8.19%) per sentence, biases the opener toward this genome's scores, forces '?'. Full battery (60 samples, `battery_round1.py`): dangling-ending rate rose 20.8%→24.8% (+0.040), mean-len +2.9% — the question-rate effect is real (matched corpus almost exactly) but it costs fluency. `sent_type` toggle, OFF by default, stays experimental." },
    { id: "sent_lenplan", name: "Sentence length plan", layer: "structural", stages: ["skeleton"], status: "experimental",
      short: "how long should this sentence be?",
      desc: "WIRED, FAILS the full guardrail battery — WORSE than Sentence type (2026-07-08). Probe passed weakly: mean long-sentence-opener score beat mean short-opener score (+0.53 vs -0.13, val_acc 0.605), but several openers tied on identical scores. An early spot-check (20 samples, length distribution ONLY) found no effect and called this \"practically inert\" — that was wrong. The full battery (60 samples, 4 metrics) shows dangling-ending rate rose 20.8%→26.0% (+0.052) — a real fluency cost the narrower check couldn't see, because it never looked at dangling rate. `lenplan` toggle, OFF by default." },
    { id: "clause_count", name: "Clause count", layer: "structural", stages: ["skeleton"], status: "cut",
      short: "simple or compound sentence?",
      desc: "CUT (2026-07-08). Positive = opener of a sentence with a mid-sentence coordinating conjunction (\", and/but/or/so/yet\"), negative = opener with none. val_acc 0.575, barely above the 0.73 majority-class rate. Decisive probe (data-driven Spearman correlation between the genome's score and each word's TRUE empirical compound-rate, not hand-picked words): -0.009 — no learnable signal at all. Genuinely unlearnable from the opener alone, not a compound question to decompose: whether a sentence goes on to coordinate is a downstream planning decision the first word can't predict. Would need real lookahead/planning state (Revision-stage territory), not a per-word Skeleton genome." },

    { id: "definite", name: "Definiteness", layer: "semantic", stages: ["fill"], status: "cut",
      short: "\"a\" vs \"the\" — new or known entity?",
      desc: "CUT (2026-07-08) — blocked before any training. A direct corpus check (`run_definite_check.py`) found the pipeline's word-level vocabulary (`min_len=2` in the lexicon builder) drops single-letter words, so \"a\" is not in the 4000-word pipeline vocabulary AT ALL — 166,554 real occurrences in the corpus, all silently mapped to <unk>. The pipeline cannot currently choose between \"a\" and \"the\" as distinct words; there's no signal to learn because one side of the distinction doesn't exist in the selectable vocabulary. A vocabulary-construction gap, not a learnability gap (also affects \"I\"). Not fixed here — would shift the top-4000 cutoff and likely require retraining every genome touching the vocab/corpus caches." },
    { id: "transitivity", name: "Transitivity", layer: "structural", stages: ["fill"], status: "cut",
      short: "verb needs an object?",
      desc: "CUT as redundant (2026-07-08) — design analysis only, no training run (`genreg_train/transitivity_analysis.md`). Without POS tags, \"does this word need a following content word\" collapses onto one of two things already tried: the shipped Closer genome (already learns the inverse — \"is this word a plausible sentence-ender\"), or the exact ambiguity that already cut the class-level Verb argument genome (no way to isolate \"verb governing an object\" from any other reason a content word follows). Word-level framing doesn't add label information the class-level attempt didn't already have." },

    { id: "whole_sent", name: "Whole-sentence scorer", layer: "abstraction", stages: ["revision"], status: "experimental",
      group: "revision", groupLabel: "Revision (new stage — composed)",
      short: "read + score the full sentence",
      desc: "BUILT (2026-07-08), FAILS the full guardrail battery. `Service._sentence_score()` in `wordpipe_service.py` reads a complete sentence post-hoc and composes a fitness from already-evolved champions (semantic adjacency, opener/closer fit, alternation-violation count, no-repeat-violation count, a degenerate-length guard). No new training. Full battery (15 samples, 8 sentences x 6 candidates): adj-hit -0.035 and distinct +0.057 are fine, dangling -0.045 is BETTER, but mean sentence length COLLAPSED 14.7→7.8 words (-47%) — the composite score is roughly additive over word count, so it systematically prefers short, safe sentences. Confirms the length bias flagged at build time was severe, not a minor caveat. Now exposed via /api/evolang/revision + the \"Best-of-N (Revision)\" button, but needs a length-normalized scorer before it's usable." },
    { id: "best_of_n", name: "Best-of-N", layer: "abstraction", stages: ["revision"], status: "experimental",
      group: "revision", groupLabel: "Revision (new stage — composed)",
      short: "generate N, keep the best",
      desc: "BUILT (2026-07-08), inherits the Whole-sentence scorer's length-collapse problem (see that node — mean sentence length -47% in the full battery). `Service.generate_revision(en, n_sentences, n_candidates, seed)` generates several candidate sentences per slot with the UNCHANGED existing pipeline (different seeds), scores each with the Whole-sentence scorer, keeps the best. No new training — selection at the sentence level instead of the word level. Exposed via /api/evolang/revision + the \"Best-of-N (Revision)\" button." },

    { id: "pronominal", name: "Pronominalization", layer: "semantic", stages: ["passage"], status: "experimental",
      short: "\"the king\" then \"he\" after",
      desc: "WIRED, real generation effect measured, FAILS the full guardrail battery (2026-07-08) — the first real Passage-stage genome, and it needed NO new training: reuses the No-repeat genome's `recent` buffer as the entity-recency signal. A content word re-mentioned within the last 15 words is replaced by a generic pronoun (\"it\", no gender/number modeling) 60% of the time. Early spot-check found 'it' frequency rose 0.77%→1.11% (+44% relative). The full battery (60 samples, 4 metrics) shows that substitution effect comes with a real fluency cost the narrower check didn't measure: dangling-ending rate rose 20.8%→24.0% (+0.032), mean-len +1.4%. `pronominal` toggle, OFF by default, stays experimental." },
    { id: "discourse_rel", name: "Discourse relation", layer: "abstraction", stages: ["passage"], status: "planned",
      short: "elaborates / contrasts / causes?",
      desc: "PLANNED, confirmed genuinely blocked (2026-07-08). Does this sentence elaborate on, contrast with, or follow causally from the previous one? Needs real cross-sentence semantic state (what was asserted) that `generate()` has no representation for at all — a flat token stream with no persistent claim/entity memory. Needs an actual design (a persistent memory structure threaded through the generation loop), not a training run." },
    { id: "info_status", name: "Information status", layer: "semantic", stages: ["passage"], status: "planned",
      short: "given info first, new info last?",
      desc: "PLANNED, confirmed genuinely blocked (2026-07-08). English information structure: given (already-established) information tends to come before new information. Same underlying gap as Discourse relation — needs persistent cross-sentence state this pipeline doesn't have, not a training run." },
  ];

  const STAGES = [
    { id: "content", label: "Content", sub: "EXPERIMENTAL — pick meaning first" },
    { id: "skeleton", label: "Skeleton", sub: "Order class-sequence" },
    { id: "fill", label: "Fill", sub: "word selection" },
    { id: "boundary", label: "Boundary", sub: "punctuation" },
    { id: "revision", label: "Revision", sub: "whole-sentence, best-of-N" },
    { id: "passage", label: "Passage", sub: "partial — pronominalization only", future: true },
  ];
  const LAYERS = [
    { id: "structural", label: "Structural", sub: "form — grammar, position, rhythm", varName: "--accent" },
    { id: "semantic", label: "Semantic", sub: "meaning — built distributional space", varName: "--layer-semantic" },
    { id: "abstraction", label: "Abstraction", sub: "relations composed on the semantic space", varName: "--layer-abstraction" },
  ];

  // per-status visual treatment: [stroke-dasharray, box opacity]
  const STATUS_STYLE = {
    "shipped": ["none", 1],
    "shipped-off": ["4,3", 1],
    "experimental": ["4,3", 1],
    "planned": ["1,3", 0.55],
    "cut": ["4,3", 0.45],
  };

  const W = 2080;
  const COL_X = { content: 180, skeleton: 480, fill: 780, boundary: 1080, revision: 1380, passage: 1680 };
  const OUTPUT_X = 1950;
  const BACKBONE_Y = 46;
  const BAND_TOP = 110;
  const NODE_W = 178, NODE_H = 52, NODE_GAP = 10;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function text(x, y, s, attrs, parent) {
    const t = el("text", Object.assign({ x, y, "font-family": "var(--mono)" }, attrs), parent);
    t.textContent = s;
    return t;
  }

  // ---- custom tooltip (replaces the slow/unstyled native SVG <title>) ----
  let tipEl = null;
  function ensureTip() {
    if (tipEl) return tipEl;
    tipEl = document.createElement("div");
    tipEl.className = "lyr-tooltip";
    tipEl.style.display = "none";
    document.body.appendChild(tipEl);
    return tipEl;
  }
  function showTip(evt, n) {
    const tip = ensureTip();
    tip.innerHTML = `<b>${n.name}</b> <span class="lyr-tip-status">[${n.status}]</span>` +
      `<div class="lyr-tip-desc">${n.desc}</div>`;
    tip.style.display = "block";
    moveTip(evt);
  }
  function moveTip(evt) {
    if (!tipEl || tipEl.style.display === "none") return;
    const pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    const rect = tipEl.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
    tipEl.style.left = x + "px";
    tipEl.style.top = y + "px";
  }
  function hideTip() {
    if (tipEl) tipEl.style.display = "none";
  }

  // ---- grouping: consecutive genomes sharing a `group` id bundle into ONE
  // compact cluster (small 2-col grid inside a labeled box) instead of each
  // stacking as its own full-size row — keeps a family of related planned/
  // cut genomes (e.g. the 4 sentiment sub-genomes) reading as one concept.
  const GROUP_COLS = 2, SUB_H = 32, SUB_GAP = 6, GROUP_PAD = 10, GROUP_LABEL_H = 16;
  // sub-node width scales to the longest member name in ITS group, so a
  // two-word name (e.g. "Sentence coherence") doesn't overflow a box sized
  // for one-word names (e.g. "Intensity") — every group picks its own width.
  function groupSubWidth(members) {
    const maxLen = Math.max(...members.map((m) => m.name.length));
    return Math.max(84, Math.min(150, maxLen * 6 + 16));
  }

  function stageItems(nodes) {
    const items = [];
    const seen = new Set();
    for (const n of nodes) {
      if (n.group) {
        if (seen.has(n.group)) continue;
        seen.add(n.group);
        items.push({ type: "group", group: n.group, label: n.groupLabel,
                    members: nodes.filter((m) => m.group === n.group) });
      } else {
        items.push({ type: "single", node: n });
      }
    }
    return items;
  }
  function itemHeight(item) {
    if (item.type === "single") return NODE_H;
    const rows = Math.ceil(item.members.length / GROUP_COLS);
    return GROUP_LABEL_H + rows * SUB_H + (rows - 1) * SUB_GAP + GROUP_PAD;
  }

  function build() {
    const svg = $("lyr-svg");
    svg.innerHTML = "";

    // ---- layout: bucket nodes by layer -> stage, compute band heights ----
    const byLayer = {};
    for (const L of LAYERS) {
      byLayer[L.id] = {};
      for (const s of STAGES) byLayer[L.id][s.id] = [];
    }
    for (const n of NODES) for (const s of n.stages) byLayer[n.layer][s].push(n);
    const itemsByLayerStage = {};
    for (const L of LAYERS) {
      itemsByLayerStage[L.id] = {};
      for (const s of STAGES) itemsByLayerStage[L.id][s.id] = stageItems(byLayer[L.id][s.id]);
    }

    const bandHeights = {};
    for (const L of LAYERS) {
      const stackHeights = STAGES.map((s) => {
        const items = itemsByLayerStage[L.id][s.id];
        return items.reduce((sum, it) => sum + itemHeight(it) + NODE_GAP, 0);
      });
      bandHeights[L.id] = Math.max(80, Math.max(1, ...stackHeights) + 40);
    }
    const totalH = BAND_TOP + LAYERS.reduce((s, L) => s + bandHeights[L.id] + 26, 0) + 20;
    svg.setAttribute("viewBox", `0 0 ${W} ${totalH}`);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", totalH);

    // ---- defs: arrowhead marker ----
    const defs = el("defs", {}, svg);
    const marker = el("marker", { id: "arrow", markerWidth: 8, markerHeight: 8, refX: 6, refY: 3,
                                  orient: "auto", markerUnits: "strokeWidth" }, defs);
    el("path", { d: "M0,0 L6,3 L0,6 Z", fill: css("--muted") }, marker);

    // ---- column guide lines (dashed, span full height) ----
    for (const s of STAGES) {
      el("line", { x1: COL_X[s.id], y1: BACKBONE_Y + 20, x2: COL_X[s.id], y2: totalH - 10,
                   stroke: css("--border"), "stroke-width": 1, "stroke-dasharray": "3,4" }, svg);
    }

    // ---- backbone: Skeleton -> Fill -> Boundary -> Revision -> Passage -> Output.
    // Revision/Passage don't exist in generate() yet — dashed/muted, same visual
    // language as a 'planned' node, so the backbone itself is honest about what's
    // live pipeline vs. aspirational.
    const backboneNodes = [...STAGES.map((s) => ({ x: COL_X[s.id], label: s.label, sub: s.sub,
                                                    future: !!s.future })),
                           { x: OUTPUT_X, label: "Output", sub: "generated text", future: false }];
    for (let i = 0; i < backboneNodes.length; i++) {
      const b = backboneNodes[i];
      const bColor = b.future ? css("--muted") : css("--accent");
      el("rect", { x: b.x - 58, y: BACKBONE_Y - 16, width: 116, height: 34, rx: 6,
                   fill: css("--panel-2"), stroke: bColor, "stroke-width": 1.5,
                   "stroke-dasharray": b.future ? "4,3" : "none" }, svg);
      text(b.x, BACKBONE_Y + 2, b.label, { "text-anchor": "middle", "font-size": 12,
                                           "font-weight": 700, fill: b.future ? bColor : css("--text") }, svg);
      text(b.x, BACKBONE_Y + 15, b.sub, { "text-anchor": "middle", "font-size": 8.5,
                                          fill: css("--muted") }, svg);
      if (i < backboneNodes.length - 1) {
        const next = backboneNodes[i + 1];
        const lineFuture = b.future || next.future;
        el("line", { x1: b.x + 58, y1: BACKBONE_Y, x2: next.x - 60, y2: BACKBONE_Y,
                     stroke: lineFuture ? css("--muted") : css("--accent"), "stroke-width": 2,
                     "stroke-dasharray": lineFuture ? "4,3" : "none", "marker-end": "url(#arrow)" }, svg);
      }
    }

    // ---- bands ----
    let y = BAND_TOP;
    for (const L of LAYERS) {
      const h = bandHeights[L.id];
      const color = css(L.varName);
      el("rect", { x: 8, y, width: W - 16, height: h, rx: 8,
                   fill: "none", stroke: color, "stroke-width": 1, opacity: 0.35 }, svg);
      text(24, y + 20, L.label, { "font-size": 13, "font-weight": 700, fill: color }, svg);
      text(24, y + 34, L.sub, { "font-size": 9.5, fill: css("--muted") }, svg);

      for (const s of STAGES) {
        const items = itemsByLayerStage[L.id][s.id];
        const cx = COL_X[s.id];
        let iy = y + 46;
        for (const item of items) {
          const ih = itemHeight(item);
          const cy = iy + ih / 2;
          // connector up to the backbone column
          el("line", { x1: cx, y1: iy, x2: cx, y2: BACKBONE_Y + 18,
                       stroke: color, "stroke-width": 1, opacity: 0.3, "stroke-dasharray": "2,3" }, svg);

          if (item.type === "single") {
            const n = item.node;
            const [dash, opacity] = STATUS_STYLE[n.status] || ["4,3", 0.6];
            const g = el("g", { class: "lyr-node", "data-id": n.id, "aria-label": n.name }, svg);
            g.addEventListener("mouseenter", (e) => showTip(e, n));
            g.addEventListener("mousemove", moveTip);
            g.addEventListener("mouseleave", hideTip);
            el("rect", { x: cx - NODE_W / 2, y: cy - NODE_H / 2, width: NODE_W, height: NODE_H,
                         rx: 6, fill: css("--panel-2"), stroke: color, "stroke-width": 1.5,
                         opacity, "stroke-dasharray": dash }, g);
            text(cx, cy - 13, n.name, { "text-anchor": "middle", "font-size": 10.5,
                                        "font-weight": 600, fill: css("--text"), opacity }, g);
            text(cx, cy - 1, n.short, { "text-anchor": "middle", "font-size": 7.8,
                                        fill: css("--muted"), opacity: Math.min(1, opacity + 0.15) }, g);
            text(cx, cy + 16, n.status, { "text-anchor": "middle", "font-size": 8,
                                          fill: color, opacity: Math.min(1, opacity + 0.25) }, g);
          } else {
            // grouped cluster: labeled box + small 2-col grid of sub-nodes
            const subW = groupSubWidth(item.members);
            const gw = GROUP_COLS * subW + (GROUP_COLS - 1) * SUB_GAP + GROUP_PAD * 2;
            const gx0 = cx - gw / 2, gy0 = iy;
            el("rect", { x: gx0, y: gy0, width: gw, height: ih, rx: 8,
                         fill: "none", stroke: color, "stroke-width": 1, "stroke-dasharray": "3,3",
                         opacity: 0.7 }, svg);
            text(cx, gy0 + 12, item.label, { "text-anchor": "middle", "font-size": 8.5,
                                             "font-weight": 700, fill: color }, svg);
            item.members.forEach((n, mi) => {
              const col = mi % GROUP_COLS, row = Math.floor(mi / GROUP_COLS);
              const sx = gx0 + GROUP_PAD + col * (subW + SUB_GAP);
              const sy = gy0 + GROUP_LABEL_H + row * (SUB_H + SUB_GAP);
              const [dash, opacity] = STATUS_STYLE[n.status] || ["4,3", 0.6];
              const g = el("g", { class: "lyr-node", "data-id": n.id, "aria-label": n.name }, svg);
              g.addEventListener("mouseenter", (e) => showTip(e, n));
              g.addEventListener("mousemove", moveTip);
              g.addEventListener("mouseleave", hideTip);
              el("rect", { x: sx, y: sy, width: subW, height: SUB_H, rx: 5,
                           fill: css("--panel-2"), stroke: color, "stroke-width": 1.2,
                           opacity, "stroke-dasharray": dash }, g);
              text(sx + subW / 2, sy + 13, n.name, { "text-anchor": "middle", "font-size": 8.8,
                                                     "font-weight": 600, fill: css("--text"), opacity }, g);
              text(sx + subW / 2, sy + 25, n.status, { "text-anchor": "middle", "font-size": 6.6,
                                                       fill: color, opacity: Math.min(1, opacity + 0.25) }, g);
            });
          }
          iy += ih + NODE_GAP;
        }
      }
      y += h + 26;
    }
  }

  // ---- export: the same data driving the diagram, as JSON for another AI ----
  const PIPELINE_FLOW = [...STAGES.map((s) => s.id), "output"];

  function buildExport() {
    const byStage = {};
    for (const s of STAGES) byStage[s.id] = [];
    for (const n of NODES) for (const s of n.stages) byStage[s].push(n.id);
    return {
      source: "GENREG /evolang WordPipe pipeline — genome layer & flow map",
      exported: new Date().toISOString(),
      pipeline_flow: PIPELINE_FLOW,
      pipeline_flow_note: "skeleton/fill/boundary are the real order generate() runs in today " +
        "(wordpipe_service.py): Order emits a class skeleton, word-selection fills each slot, " +
        "boundary/commas decide punctuation, text is emitted. revision/passage do NOT exist in " +
        "generate() yet (see each stage's `future` flag) — they're planned future stages, not " +
        "current wiring.",
      status_legend: {
        shipped: "live in the default pipeline",
        "shipped-off": "live, toggled off by default",
        experimental: "wired and available, but not battery-tested for generation effect",
        planned: "design intent — not yet built",
        cut: "attempted and rejected on real evidence — see description for why",
      },
      layers: LAYERS.map((L) => ({ id: L.id, label: L.label, description: L.sub })),
      stages: STAGES.map((s) => ({ id: s.id, label: s.label, description: s.sub,
                                   future: !!s.future, genome_ids: byStage[s.id] })),
      genomes: NODES.map((n) => ({ id: n.id, name: n.name, layer: n.layer, stages: n.stages,
                                   status: n.status, short: n.short, description: n.desc,
                                   group: n.group || null,
                                   group_label: n.groupLabel || null })),
    };
  }

  function exportJSON() {
    const data = buildExport();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "evolang_genome_layers.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  document.addEventListener("DOMContentLoaded", () => {
    build();
    const btn = $("lyr-export");
    if (btn) btn.addEventListener("click", exportJSON);
  });
})();
