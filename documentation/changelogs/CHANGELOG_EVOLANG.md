# Changelog — EvoLang (evolution-native language model)

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.

EvoLang is the fresh start after the n-gram / LM / Tree line was archived
(2026-07-06). See `documentation/EVOLANG_PIVOT.md` for why.

---

- **[2026-07-06] (Claude)** — **WordPipe specialist-pipeline experiment** (`genreg_train/wordpipe.py`;
  full writeup `documentation/WORDPIPE_FINDINGS.md`). Gated test of "evolve a specialist per language
  component." **G1 vocabulary/speller PASS**: lexicon-coverage pressure lifts valid tokens 18.9% →
  52.4% vs a plain char LM. **G2 order discriminator FAIL** (real vs within-window-shuffled): stuck at
  chance 51.9% over 2500 gens though a bigram probe hits 69.2% — can't evolve a 4000-word embedding by
  mutation (gradient-free representation wall). G3 orderer skipped. Boundary: specialists are evolvable
  only with a small/scaffolded representation; fix = order over ~30 POS categories, not 4000 words.

- **[2026-07-06] (Claude)** — **Held-out validation split + autonomous findings.** Training samples
  only the first 90% of the corpus; champion ppl also measured on a fixed 4096-window tail sample
  (`val_ppl` in gen/done events; ppl tile shows train/val). Ran an experiment battery (sweep /
  novelty A/B / long run) — see `documentation/EVOLANG_FINDINGS.md`. Headlines: **K4/H48 best**
  (bigger context doesn't help — overfits); **novelty monotonically hurts perplexity** (variety
  lever, not a ppl improver — keep off by default); **long run val 12.25 ≈ char-level ceiling** for
  one tiny genome (more gens → memorisation, not generalisation). Defaults updated to K4/H48/E12.

- **[2026-07-06] (Claude)** — **Corpus scaled up to the Gutenberg book dump.** Removed the 642-char
  toy string; now reads `project/EEC-main/engine/corpus.txt` (~48.6M chars). Fixed 37-char charset
  (space + lowercase + basic punctuation; digits→'#', other→space). Windows sampled on the fly per
  generation (never materialise the ~49M pairs); flat int16 array (97 MB) cached lazily on first run
  so import stays cheap. Page shows a preview slice only; `started` carries `corpus_chars`. New
  `corpus_ids()` / `_build_ids()` / `_corpus_preview()`; `encode` now cleans + folds. ppl honestly
  higher (varied text) — needs more generations than the toy.

- **[2026-07-06] (Claude)** — **Novelty constraint** added (opt-in, pure reward). Per-genome:
  sample a short passage (batched across the population), walk its words maintaining a 0.00–1.00
  novelty scalar — decays `decay`/word, a word pays `gain` × cooldown-ramp (0→full over `cooldown`
  words since last use), capped at 1.0, only gained. **Multiplies** the genome's soft fitness by
  `(1 + weight × novelty)` (log-space: `base + log1p(weight×novelty)`) — a scaled boost proportional
  to the genome's own fitness; never penalises. Fights repetitive collapse. Checkbox + fields (gain 0.15, decay 0.005,
  cooldown 40, weight 0.5, rollout 80 chars) on `/evolang`. New `novelty_scores()` + `LangPop.
  sample_batch()`; ppl tile reads raw log-prob (base), champion-novelty shown separately. Verified:
  "the"×12 → 0.136, distinct words → 1.00. ~1.5× per-gen cost when enabled.

- **[2026-07-06] (Claude)** — **Project created.** `genreg_train/evolang.py`: one small fixed
  English corpus, a tiny neural next-char predictor per genome (context K chars → evolved
  embedding + positional weights → tanh(H) → V logits), evolved by tournament selection +
  elitism + mandatory energy homeostasis (starve the least-fit 3–15% each gen) with self-adaptive
  Gaussian mutation (floored). Fitness = **soft** mean log-prob of the true next char over a fresh
  minibatch of corpus windows — never argmax, never a count table; no gradient touches a weight.
  Streams over WS `/evolang` via the shared JobHub (survives navigation). New page
  `templates/evolang.html` + `static/evolang.js`: config sidebar, live fitness chart, emerging
  sample (re-sampled every 25 gens), a generate box, and the full corpus on display; wired to the
  shared terminal dock / Agent panel / Run-Config panel. Runs persist to `runs/evolang/<id>/` and
  appear on `/runs`. Deliberately minimal — the smallest honest evolution-native LM to build from.
