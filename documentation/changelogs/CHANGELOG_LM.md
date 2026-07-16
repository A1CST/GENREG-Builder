# Changelog — LM (autoregressive path)

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.
Seeded 2026-07-05 from the main changelog (keyword split, best effort).

---

- **[2026-07-16] (Claude)** — **C4: ear-capacity NULL + the cloze ceiling.**
  128-dim directional ears: 0.1384/0.2850, slightly below 64-dim
  (0.1418/0.2952) - capacity ruled out as the bottleneck. Then the missing
  reference: symbolic tables (tri/bi/uni, vocab-restricted, left context)
  score **0.2494 top-1 / 0.4728 top-5** on the same cloze set. REFRAME:
  the fully evolution-made kid runs at 57-62% of the table ceiling from
  pixels + bred experience alone; the absolute gate bar never existed.
  Note: the kid sees RIGHT context too (tables cannot), so ceiling-plus is
  possible in principle. Module 22 on /lm. Decision for the user: call
  60%-of-ceiling a C-gate pass and open Stage D, or close the gap first.

- **[2026-07-16] (Claude)** — **Stage C3: SEQUENCE EARS - evolution earns on
  the semantic stage for the first time. Cloze 0.1418 top-1 / 0.2952 top-5.**
  Built two directional RS layers on pod #3 (pod #2 died overnight;
  everything restored from shadows, zero loss, new pod staged in 4 min):
  embed_rs_next (words separated by what FOLLOWS: three->five two seven
  six eight four) and embed_rs_prev (by what PRECEDES - grammatical:
  said->killed found separated caught kept). Ear ladder: deaf 0.0668 ->
  sim 0.1212 -> +seq 0.1418; composition genomes froze (10) after three
  zero-earn attempts. Intrinsic gates below SVD but the 30k lesson holds
  (task A/B is the gate). Module 21 on /lm. Next lever: scale directional
  ears (64->128+ dims) or pair-context continuation ears.

- **[2026-07-15] (Claude)** — **Multimodal merge: the kid LETTER recognizer
  (kid_modelA.json, Stage A, 0.996 solo) fused with the vision SHAPE recognizer
  into ONE 36-class classifier.** Late fusion (each frozen bank in its own basis
  at its native scale, concat, one closed-form head): FUSED test 0.9946 (shapes
  1.000, letters 0.9925); fusion beats either bank alone. Cross-modal: the shape
  bank alone reads letters ~0.92 and the letter bank reads shapes 1.0 - evolved
  radial features transfer across modalities. mm_merge.py; merged model at
  multimodal/mm_model.json. No genome retrained; all models backed up first.
- **[2026-07-15] (Claude)** — **Stage C2: EARS attached (the RS-evolved
  semantic layer as listening experience) - cloze DOUBLES to 0.1212 top-1
  / 0.2684 top-5 (baseline 0.0608 -> 0.1085) but the gate still fails and
  evolution earned zero.** The curriculum stays evolution-made (eyes from
  pixels, ears from evolved co-occurrence separation; blank word gets no
  vector). Sharpened diagnosis: RS ears carry SIMILARITY; cloze needs
  SEQUENCE experience - heard word-order, not meaning clusters. Candidate
  next ear: an RS-style EVOLVED CONTINUATION layer (genomes that separate
  contexts by what follows them - the evolutionary counterpart of the
  continuation tables). Module 20 on /lm; Stage D remains gated.

- **[2026-07-15] (Claude)** — **CURRICULUM Stage C: CLOZE - GATE FAILED
  honestly, and the failure is the curriculum's most important lesson.**
  Word-bag baseline 0.0608; evolution earned NOTHING (first space zero
  genomes); final 0.0668 / 0.2150 top-5 (33x chance, no composition).
  Contrast: A (letters) passed in 7s, B (spelling) in 30s - deterministic
  answers in pixels. C is SEMANTIC: 20k phrases of pixels contain no
  answer to "which of 500 words fills this blank" - a kid at this stage
  has YEARS of heard language. Design decision recorded for the user:
  (a) give Stage C listening experience - corpus co-occurrence statistics
  as environment channels (the earlier line's tables, now justified as
  curriculum experience), (b) scale data by orders of magnitude, or both.
  OOM fix en route: 640k-row envs need max_cached=1 + scale-sorted eye
  sweep. Module 19 on /lm. Stage D blocked on the C gate per the
  curriculum's own rule.

- **[2026-07-15] (Claude)** — **CURRICULUM Stage B: WORDS - the model learns
  to SPELL. TEST 0.7066 top-1 / 0.9358 top-5 (V=500, pixels only, 30s).**
  Words as 8-letter tile strips; Stage A's 515 letter genomes frozen as
  the eye. THE MEASURED STORY: orderless letter-bag baseline (eye only)
  0.1910 -> first composition space +45 points by ordering letters ->
  0.7066. The (letter-genome x slot) hand-off carried spelling exactly as
  it carried motion and word order before. 297 new genomes on the frozen
  eye; kid_modelB.json is Stage C's base. Module 18 on /lm. Stage C
  (cloze: name the blank in a short phrase) is next.

- **[2026-07-15] (Claude)** — **THE ENGLISH CURRICULUM (user's design) begins:
  Stage A - LETTERS - gate PASSED at TEST 0.9960 in 7 seconds.** The plan:
  learn language like a kid - A: identify letters; B: identify words
  (word = strip of letter tiles, stage-A genomes frozen as the eye);
  C: cloze (name the blank in a short phrase); D: autoregression - each
  stage its own model warm-started on the previous, no stage advances
  before its gate. Stage A (radial_kid.py): 25k letter tiles (14-26px,
  jitter +-4, noise), 515 R0 genomes, self-stopped at 2 spaces
  [515, 82] exactly like the animation shape twin. Retroactive diagnosis
  of the sentence-strip failure: single-glyph perception earns instantly -
  the vision line needed a childhood, not bigger strips. Module 17 on
  /lm; checkpoint kid_modelA.json is Stage B's perceptual base.

- **[2026-07-15] (Claude)** — **MODULE 17 (vision LM) FIRST ATTEMPT:
  language-grounded-in-vision launched (user's pivot - the glyph line
  resurrected at word level: 12-word strips rendered as 32x32 tiles, NO
  embeddings/tables/one-hots, pixels are the whole environment). Result:
  R0 froze ZERO genomes (round 0 gain +0.0000 vs cap) - the same machinery
  that instantly earns on ID-based words gets nothing from word-tile
  perception. Diagnosis queue: window-MEAN over 12 tiles likely washes out
  per-word variance (try window-max / per-slot fitness), tiles may be too
  small for whole words at 32x32 (try 48px, fewer words), check candidate
  fitness distribution. Also: empty-stack guard needed (crashed at
  torch.stack on zero features). Strip data (36k strips) rendered + cached
  on pod (lm_vis.npz) so iterations are cheap. The thesis is untested, not
  disproven - perception must earn before the language question can be
  asked. radial_lm_vision.py committed; not yet a page module (no result
  to show honestly).

- **[2026-07-15] (Claude)** — **GRAMMAR CONTRACT FIX (user's call: fix the
  environment, don't strap on monitors).** The generation bug's true root:
  feature_vec's GATE normalized its stream with RUNTIME BATCH statistics -
  gated genomes were functions of (row, current batch), violating
  "features are the environment" (and silently making test features depend
  on test-batch stats). The reference-batch inference workaround is
  REJECTED. Fix: gate normalization constants are now ENVIRONMENT-PROVIDED
  - `bake_gate_stats()` freezes each gated genome's gate-stream mean/sd
  over the training environment into its genes at freeze time
  (gt["stats"]); feature_vec uses baked stats when present (evolution-time
  fallback: population batch stats, which is legitimately the population's
  environment during selection). Genomes are pure per-row functions after
  freezing; batch == single-row by construction. TO APPLY: replay
  pipelines must call bake_gate_stats(genomes, train_bank) per space
  before computing features; existing checkpoints get baked on next
  replay. Generation/parity remain PARKED until then per user.

- **[2026-07-15] (Claude)** — **GENERATION SALAD ROOT-CAUSED: the model is
  fine; the single-row step pipeline is buggy.** Parity check (30 held-out
  windows, step argmax vs verified batch preds): 15/30 MISMATCH, and every
  step-side prediction is the same attractor set (your/how/that) while the
  batch side predicts sensible words. All reported metrics (0.5663/0.5671
  etc.) came from the verified batch path and stand. The word-salad samples
  in Module 16's export are the BUG's output, not the model's. Known wart:
  the uni-backoff cont branch builds zeros where training used the unigram
  distribution; a block-diff diagnostic (debug hook in
  lm_generate3._step_logits + gen_diag.py on the pod) is bisecting the
  first divergent feature block on a mismatched row. Fix, re-verify parity
  to 30/30, THEN regenerate. Lesson recorded: any reimplementation of a
  feature pipeline gets a parity test BEFORE its output is trusted.

- **[2026-07-15] (Claude)** — **MODULE 16 - third crank: TEST 0.5663 top-1 /
  0.7280 top-5 - the top-5 usefulness bar (60-70%) is CROSSED.** True
  4-gram table (last three words - never tabulated before) + far skips
  (w-6,w-1)/(w-5,w-2); headroom 84.5% of remaining errors answerable (quad
  alone 68%). Continue-train on the frozen Module-15 base: +80 genomes,
  32s. Blind class 0.2051 -> 0.2938 top-1 / 0.4832 top-5; has-target
  0.736. Cumulative crank 0.2974 -> 0.3992 -> 0.4832 -> 0.5663 at ~1.2M
  params, gradient-free. OOM fix: chained environments must FREE
  intermediate banks (bank3 contains bank2 contains the lean bank - 89GB
  resident before the fix). Module 16 on /lm.

- **[2026-07-15] (Claude)** — **MODULE 15 - second crank: far skip-grams
  (w-4,w-1)/(w-5,w-1) take the lean model to TEST 0.4832 top-1 / 0.6643
  top-5 - the user's "simple neural LM territory" band, gradient-free at
  ~1.2M params.** Loop repeated on the Module-14 model: cont2 rebuild
  verified (0.3987 vs 0.3992), headroom 64.9% of remaining errors
  answerable by the far tables, continue-train +178 genomes in 64s. Blind
  class 0.1106 -> 0.2051, bigram-backoff 0.327 -> 0.476, every slice up,
  no forgetting. Cumulative crank: 0.2974 -> 0.3992 -> 0.4832. Module 15
  on /lm.

- **[2026-07-15] (Claude)** — **MODULE 14 - THE LOOP CLOSES: skip-gram
  environment channels + continue-train take the lean model to TEST 0.3992
  top-1 / 0.5981 top-5 - the best model of the entire LM line, beating
  every fat head (best 0.3296) at ~1.1M params.** Module 13's finding
  applied: the miss class needed NEW environment statistics. Added skipA
  (w-3,w-1) and skipB (w-3,w-2) continuation tables from the independent
  slice (+4,256 channels, cached lm_skip_tables.pkl); HEADROOM diagnostic
  before evolving: 45.9% of blind windows answerable by the new tables.
  Continue-train on the same frozen 285-genome base: +65 genomes in 3
  spaces, 45s pod. Blind class 0.0040 -> 0.1106 top-1 (28x), 0.2521 top-5;
  unigram-backoff 0.058 -> 0.214; has-target 0.480 -> 0.579 - improvement
  EVERYWHERE, zero forgetting. The probe -> environment-gap ->
  continue-train method is validated on language. Module 14 on /lm.

- **[2026-07-15] (Claude)** — **LM modules 12+13: PROBE + CONTINUE-TRAIN (the
  animation Module-3 method ported to language).** PROBE (read-only on the
  RS-30k lean checkpoint): the model NEVER overrides its tables - when the
  continuation table's top-5 contains the target (62% of test) top-1 is
  0.4801; when it doesn't (38%) top-1 is 0.0040, chance. Worst-missed words
  are the commonest (i/you/the/it) - wider-context cases. CONTINUE-TRAIN
  (lm_continue.py): frozen 285-genome warm base, new spaces on a 3x
  table-miss-weighted mix, head refit on the true distribution. THE METHOD
  WORKS (no forgetting, overall 0.2974->0.2993) BUT THE REPAIR DOESN'T:
  miss-class 0.0040->0.0056, evolution dry after 10 genomes. FINDING:
  continue-training cannot conjure signal the environment doesn't carry -
  the miss class needs NEW environment statistics (skip-gram continuation
  tables, syntax-position profiles), not more search. Both modules on /lm
  (renderer now shows breakdown tables + example predictions); registry
  encoding mojibake fixed; two OOM fixes (mix subsampling - the attend
  substrate is ~10GB and cannot be duplicated full-size even on 96GB).

- **[2026-07-15] (Claude)** — NOTE: the full radial-LM arc of 2026-07-15
  (char isolation, word pivot, lean/hybrid shapes, RS embeddings, vocab
  scaling, /lm iteration-log rebuild) is in the MAIN CHANGELOG; this
  project file resumes per-project logging with this entry per AGENTS.md.

- **[2026-07-12] (Claude)** — Round 3: sem_next + grammar_real in a corpus-built
  PPMI/eig feature space; no lookup tables in the model (full-vocab genome scoring
  replaces follower pools at inference). Full details in main CHANGELOG.md same
  date. Key results: grammar_real 56.3% balanced (validated); sem_next beat the
  majority-frequency baseline once (30.1% vs 25.3%) but as a degenerate echo
  genome; with echo negatives evolution truly climbs (0.128->0.233, loops gone)
  yet the static per-word bias reward-hacked into word soup — SS XI's vbias
  Goodhart reproduced. GA repairs shipped: relative per-tensor mutation, fixed
  probe batch, lineage fitness EMA, SS VI bootstrap inits, logfreq-as-environment.
  Energy starved band still unreached (known deviation). Runs:
  `runs/lm/20260712-202708-lm-*`. Next levers: context-gated bias, two-phase
  freeze, transition-only ablation. **Flask restart** needed for /lm.

- **[2026-07-11] (Claude)** — GENREG_RULES compliance pass + hard negatives + rerank
  generation, retrained (crash-recovery: rewrite was complete on disk, run never
  launched; watcher crash root-caused to a cp1252 UnicodeEncodeError in `run_job.py`
  poll(), fixed with stdout errors="replace"). Soft log-prob fitness (§IV.1) in all
  trainers; energy homeostasis (§III) in ga_step (starved=0 observed all run — band
  not reached, tune next round); HARD negatives for `next_word` (same-preceding-word
  followers) + majority-frequency baseline; generate() switched to §VI rerank over
  top-200 follower pools (mined into the artifact). Results: **next_word 21.16% —
  beats 16.67% chance, does NOT beat the 26.92% majority-frequency baseline (honest
  negative)**; fill_word 21.92%; punctuation/opener/length ~unchanged
  (opener_question 0.7015 still best). Qualitative win: no more repetition loops —
  varied, near-grammatical short sentences with correct intent-driven end marks.
  Full details in the main CHANGELOG.md entry of the same date. 10 runs recorded at
  `runs/lm/20260711-233847-lm-*/`. Live /lm needs a **Flask restart**.

- **[2026-07-09] (Claude)** — New group **`next`** (`next_word`) — intent-conditioned,
  autoregressive next-word prediction, reviving the archived pipeline's Selection genome
  concept. **Honest negative result**: 24.25% holdout accuracy, essentially tied with
  `fill_word`'s 24.08% — fixing the train/inference mismatch and adding intent-conditioning
  didn't move the needle. Live check confirms generation is still repetitive. Points to
  negative-sampling strategy (deliberately kept simple/random this round) as the real lever
  for next time. Full details in the main CHANGELOG.md entry of the same date. `generate()`
  switched to use `next_word`. 10 genomes recorded at `runs/lm/20260709-213338-lm-*/`.

- **[2026-07-09] (Claude)** — First generation mechanism (hangman-style, variable-length,
  intent-driven): two new groups, **`length`** (`length_continue`, 61.1% balanced vs 50%
  chance) and **`fill`** (`fill_word`, a contrastive discriminator, 24.1% vs 16.7% chance).
  Generation mechanism itself works (real variable length, correct intent wiring); word choice
  is honestly weak/repetitive — `fill_word`'s accuracy is only modest. Full details, including a
  real design-tension scope reduction from the original plan, in the main CHANGELOG.md entry of
  the same date. New `/api/lm/generate` endpoint + "Generate" section on `/lm`. All 9 genomes
  recorded at `runs/lm/20260709-161129-lm-*/`.

- **[2026-07-09] (Claude)** — New group **"opener"**: `opener_question`/`opener_exclaim` read
  ONLY a sentence's first word to confirm its eventual intent — mirrors the punctuation group
  but forward-looking. **`opener_question` (70.6% balanced) is the strongest genome trained
  yet**, beating `punct_question` (67.2%); `opener_exclaim` (61.4%) also beats `punct_exclaim`
  (58.5%). Confirms the user's intuition that the first word is a stronger, cheaper signal than
  the words before the mark. Also fixed a bug in `record_lm_run.py` that mislabeled all opener
  runs under group "punctuation" on first write. Full details in the main CHANGELOG.md entry of
  the same date. 7 runs (5 punctuation + 2 opener) recorded at
  `runs/lm/20260709-124755-lm-*/`.

- **[2026-07-09] (Claude)** — Split into 5 binary genomes under group **"punctuation"**
  (`punct_end`, `punct_question`, `punct_exclaim`, `punct_semicolon`, `punct_colon`) — fixes
  the collapse from the entry below. All 5 beat 50% chance and none collapsed: exclaim recall
  went 0.15%→62.7%, colon 2.5%→47.9%. Live check: "please bring the following items" now
  correctly gets `:` at 80.8%. Full details in the main CHANGELOG.md entry of the same date.
  5 runs recorded at `runs/lm/20260709-111525-lm-punct_*/`, tagged `group: "punctuation"`.

- **[2026-07-09] (Claude)** — Genome #1 retrained with the train/eval split fixed: **balanced
  accuracy 26.5% (chance 16.7%)**, but collapsed to mostly predicting `?`/`;` (recall on those
  two: 63.7%/67.5%; on `!`/`:`: 0.15%/2.5%). Root cause of the entry below: champion selection
  used raw accuracy on an imbalanced holdout while training batches were class-balanced, so
  training was rewarded for getting WORSE at rare marks. Fixed by making balanced accuracy
  (mean per-class recall) drive both selection and the headline number. Full details in the
  main CHANGELOG.md entry of the same date. Run recorded at
  `runs/lm/20260709-110038-lm-intent02/` (original kept at `.../20260709-024735-lm-intent01/`
  for comparison).

- **[2026-07-09] (Claude)** — Genome #1 (intent recognition) trained: **17.8% held-out, barely
  above 16.7% chance** — an honest first result, not a working recognizer yet. Full details
  (the mining-bug catch/fix, confusion matrix, per-class recall, open question about
  balanced-vs-natural eval) in the main CHANGELOG.md entry of the same date. Run recorded at
  `runs/lm/20260709-024735-lm-intent01/`.

- **[2026-07-09] (Claude)** — **LM name revived, third incarnation.** History: this name
  originally belonged to the char/word-level autoregressive campaign below (archived
  2026-07-06, pivoted to /evolang); /evolang was then archived 2026-07-09
  (`archive/evolang_v1/`, see `documentation/WORDPIPE_FIELD_NOTES.pdf`) after its fluency
  ceiling never moved across every architecture variant tried. `/lm` now names the fresh
  gradient-free genome-pipeline rebuild, starting from nothing but the kept datasets. See the
  main CHANGELOG.md entry for what genome #1 (intent recognition) actually is — unrelated in
  approach to both prior LM lines below (no n-gram tables, no char/word autoregression, no
  reused code).

- **[2026-07-06] (Claude)** — **ARCHIVED — line retired by the EvoLang pivot.** `genreg_lm.py`,
  `genreg_attn.py`, `genreg_enc.py`, `genreg_trustmix.py`, `genreg_distill.py`, `lm_sample.py`,
  `genreg_rerank.py`, `pure_engine.py` moved to `archive/lm_and_tree/`; `lm/attn/enc/encoder/distill`
  run dirs to `runs/_archive/`. The campaign mapped a real boundary (distillation verdict: can't
  gradient-free-train away n-gram tables) and that boundary is exactly why we pivoted. Findings docs
  (`LM_STAGE1_*`, `LM_ENCODER_COMPONENT.md`) kept. Successor: `/evolang`. See
  `documentation/EVOLANG_PIVOT.md`. Code preserved, not deleted.

- **[2026-07-06] (Claude)** — LM: **distillation verdict — you can't gradient-free-train away
  the n-gram tables** (honest negative, maps the boundary). Distilled the 56.4% trust-mix
  teacher into a table-free evolved feedforward neural n-gram (soft-teacher + hard-target hybrid
  fitness, 12k gens). Result: student top-5 **58.0%** (matches/beats the teacher's distribution
  SHAPE) but top-1 only **24.7%** (recovered 44% of the teacher; generation is gibberish). The
  split IS the finding — evolution learns *which chars are plausible* (top-5) but not *which is
  most likely per context* (top-1), and generation needs the latter. Compressing corpus
  statistics into weights is directed high-dim optimization = gradient's job; undirected
  mutation can't do it at precision. Boundary: table-free+gradient-free 34.6% gibberish · n-gram
  tables 56% readable (but 1990s lookup) · table-free+gradients = real LMs (violates rule #1).
  The no-gradient LM sits where BOTH good-LM mechanisms are excluded; evolution's edge is where
  gradients can't go (Intelligence Engine), not raw next-token stats. genreg_train/genreg_distill.py.

- **[2026-07-06] (Claude)** — **TRUST-MIX breakthrough: readable English, held-out top-1 55.2%
  / top-5 84.7%** — passes both docs usability bars (top-1≥30, top-5≥60) the A-series never
  reached. The composition path (not one model doing everything): exact char n-gram channels
  (uni..5g dense + 6g/7g hashed) + the evolved neural model, combined by an EVOLVED
  context-conditional backoff GATE (trust + per-channel evidence κ, Witten-Bell style,
  gradient-free ES). Accuracy climbed with orders+gate: neural-alone 34.6 → 4g-mix 47.2 →
  5g-gated 53.5 → 7g-gated **55.2**. Generation is real English with phrases/punctuation and
  the PROMPT STEERS ("the old man "→"stood the first, by all the other … for his nose and the
  surface is not the same"; picks up Moby Dick vocab whale/jonah/fast-fish) — vs the
  neural-alone gibberish ("dod wing fas coltill carm"). Honest: coherence = high-order n-gram
  statistics; the EVOLVED part is the backoff gate (the neural is ~4% trust, near dead weight —
  count tables win accuracy AND coherence, exactly as the stack notes predicted). A real
  gradient-free char LM. genreg_train/genreg_trustmix.py; full progression in
  LM_STAGE1_FINDINGS.md.

- **[2026-07-06] (Claude)** — LM: **trigram interaction channel = best char result yet,
  34.61% top-1 / 68.78% top-5** (two-phase bootstrap from the 31.9% substrate). +2.7pp over the
  substrate, matching the documented A_101 benchmark (34.00/69.70), closing 37% of the
  substrate→char-trigram-ceiling gap (31.9→39.2, bigram 27.3). The §VI multiplicative gate
  OPENED and paid fitness where the additive copy-attention channel never did — confirming
  multiplicative>additive for pair interaction the recurrent state can't express. Generation
  (t=0.7) shows real words + word-shapes but not sentences (expected at 34%; top-5 barely
  moved). Full table in LM_STAGE1_FINDINGS.md. Next: rollout landscape on top (accuracy is up;
  generation coherence / space-collapse is the exposure gap, still open).

- **[2026-07-06] (Claude)** — LM: **low-rank trigram interaction channel (§VI)** added to
  genreg_lm — the documented word/char-pair primitive, targeting the big untapped gap (char
  substrate 31.9% sits far below the char-TRIGRAM count ceiling 39.2%). Gated multiplicative
  channel `logits += a_lr·(bigram[c_t] + (E1[c_t] ⊙ E2[c_prev]) @ O)` — captures the pair
  interaction the additive prev-char concat structurally cannot (§VI). Wired into
  evaluate/rollout/generate; `trigram_only` flag freezes the substrate for two-phase bootstrap
  (evolve the channel alone, then unfreeze). Gate `a_lr` init 0 (transparent — verified gen-0
  identical to substrate). Two-phase chain running from the 31.9% checkpoint: phase-1 channel
  already opening and climbing (31.8%→32.3% by gen 800, soft improving), vs the copy-attention
  channel which never opened — multiplicative interaction pays rent where additive copy didn't.

- **[2026-07-06] (Claude)** — **Word-level recurrent LM** added to genreg_lm (push toward
  sentence/grammar structure). The synthesis: real word tokens (Tree LM tokenizer) + the
  recurrent sequential substrate + prev-token (genreg_lm) + the blended rollout-survival
  landscape at WORD horizons — so a genome must keep producing grammatically-plausible next
  words to survive its own generation (grammar = what long-horizon rollout rewards). Engine
  changes: token_mode "char"|"word"; per-run vocab (V dynamic); <unk> never a scored target or
  emitted (mask in evaluate/rollout/generate); word bigram/trigram baselines (dict-based
  trigram — a dense table is TB-scale at word vocab); word-mode generation detokenizes via the
  persisted vocab. Two rules-endorsed fixes made word-vocab evolvable (naive V=2048 stalled at
  0% — the giant W_out genome diffuses selection): **weight-tied readout** (collapse the ~V·H
  W_out into a tiny H→D projection scored against the shared embedding table) and **bigram-SVD
  embedding seed** (§VI's endorsed init — words in similar contexts start near each other).
  Smoke: 0.9%→8.4% held-out in 200 gens. Bars: word-bigram 17.6%, word-trigram 22.7% (the
  grammar reference). Chain running (2 substrate sweeps + blended rollout R=6 words).

- **[2026-07-05] (Claude)** — **Encoder separation: final verdict = parity, kept as
  optionality.** The frozen-encoder composed model under the blended rollout landscape ties
  the monolith on every metric (open 31.45% vs 31.3 · closed R=8 27.32% vs 27.2 · gap 0.21 vs
  0.17 nats) — notably reaching parity with the encoder FROZEN (only the readout evolved).
  Per the pre-committed decision rule: separation is a validated non-regressing component —
  same performance, measurably richer state (h2/h4 future decodable), cleaner modularity
  (reusable frozen encoder) — not a breakthrough. enc_char_v1 stays available; the monolith
  remains co-champion. Full 3-round trail in LM_STAGE1_FINDINGS.md.

- **[2026-07-05] (Claude)** — Encoder component rounds 1–2 + the decisive rollout test
  (running). Equal-weight horizons: h1 fell to 29.7%, composed 30.71% — under the bar (state
  budget robbed next-char sharpness). **Weighted horizons (0.7/0.2/0.1): h1 recovered to
  31.49%, composed 31.78% — parity with the monolith within eval noise (bar 31.9% not
  passed), with future-decodability (h2/h4) the monolith never had.** The value hypothesis is
  now testable exactly where richer state should pay: the blended rollout-survival landscape —
  frozen-encoder composed model vs monolith benchmarks (open 31.3 / closed 27.2 / gap 0.17
  nats). `horizon_weights` config added to genreg_enc.py.

- **[2026-07-05] (Claude)** — **Encoder separated into its own model: `enc_char_v1`**
  (`genreg_train/genreg_enc.py`, card `documentation/LM_ENCODER_COMPONENT.md`) — per user +
  §X component-first. Fitness = evolved-head decodability at horizons {1,2,4} from the hidden
  state (equal weight — h=1 specialists sink), breeding the STATE rather than the prediction;
  heads are scaffolding, the frozen deliverable is (E, W_in, b_h, act). Skip-gram baselines
  measured first (skip2 22.5%, skip4 20.1%). genreg_lm gained composed mode
  (`encoder_ckpt` → tensors copied + FROZEN; mutation excludes encoder incl. act ids — §X
  freeze-and-compose, never retrained). Warm-start smoke: h1 preserved at 31.3%, h4 at its
  bar in 30 gens. Encoder sweep → composition pipeline running; composed bar > 31.9%.

- **[2026-07-05] (Claude)** — **Stage 4 VALIDATED — blended rollout-survival landscape.** Pure
  rollout fitness bred hedging (open-loop 31.9%→23.1%; new §XI reward-hack: flatten toward
  marginals to score safely on drifted context). Blended fitness (teacher + own-output
  segments scored together, the DiffEvo unrolled lesson): open-loop HELD at 31.3% while
  own-output top-1 at R=8 went 15.0%→27.2% — **exposure gap 0.72→0.17 nats**, the quantified
  cause of generation soup cut 4×. Samples now contain real words at t=0.5. evaluate_rollout
  blended scoring in genreg_lm.py; complete campaign trail + ranked open levers in
  documentation/LM_STAGE1_FINDINGS.md.

- **[2026-07-05] (Claude)** — LM campaign, stages 3–4: **assembly honest-negatives + rollout
  landscape launched.** Copy-attention channel (α-gated, zero-init) stayed pinned at α≈0 through
  TWO variants — v1 retrieved the matched char (control beat it 30.42% vs 29.78%: complexity
  diluted selection, as documented); v2 fixed to induction values (retrieve the successor,
  target-leak excluded) still unopened at 31.91% held-out (+4.61 over bigram — pure substrate
  ratcheting). Verdict: 64-char windows don't pay attention rent; environment lever (longer/
  repeat-rich windows) backlogged. **Exposure gap measured: champion 31.4% teacher-forced vs
  15.0% consuming its own outputs (0.72 nats)** — the quantified train/inference mismatch
  behind generation soup. Stage 4 running: rollout-survival fitness (score TRUE continuation
  while eating own samples; anchored per §IV.4), curriculum R=2→4→8. evaluate_rollout +
  rollout summary fields added to genreg_lm.py; full trail in LM_STAGE1_FINDINGS.md.

- **[2026-07-05] (Claude)** — **attn_copy_v1 PASSED its bar at 100%** (bar: ≥95% held-out on
  every offset k∈{1,2,5,10,20}, L≤64). Verified on 2,500 fresh-seed episodes: 100% on all
  five offsets, soft −6e-05. The recipe that made retrieval evolvable: multiplicative query
  (per-k conditional maps — additive provably could not switch offsets), weight-tied readout +
  identity values (credit reaches attention placement directly), zero-init relative-position
  bias (transparent no-op; evolution grows one bump per offset), 11-rung mastery-gated ladder
  (length grows before offsets unlock). Full lesson ledger in
  documentation/LM_STAGE1_FINDINGS.md. Both stage-1 components (substrate +0.73 pts over
  bigram; attention 100%) now cleared — assembly next.
- **[2026-07-05] (Claude)** — **lm_char_v1 PASSED its stage-1 bar: held-out 28.03% top-1 vs the
  27.30% char-bigram ceiling (+0.73 pts), majority floor +7.8pp, no train→held-out drop** —
  pure constraint-driven evolution per the model card (soft multiplicative fitness, energy
  homeostasis in-band all run, tournament+maturation, recurrence+prev-char, per-neuron
  activations; EMA-smoothed selection worth +1.6pp exactly as §IV.7 documents). Three resumed
  sweeps, ~24 GPU-min. attn_copy_v1: three landscape lessons logged (query couldn't SEE the
  offset; padded episodes nullified the length curriculum — masking fixed it, k=1 went
  12.5%→86% in 150 gens; weight-tied readout + identity values collapsed two coupled miracles
  into one basin) + v4 multiplicative query (additive can't switch offsets — the §VI lesson)
  running the 11-rung ladder. Findings: documentation/LM_STAGE1_FINDINGS.md.

- **[2026-07-05] (Claude)** — **New project: `lm_char_v1`** (`genreg_train/genreg_lm.py`) — stage 1
  of the true-autoregressive path, built per the pre-drafted model card
  `documentation/LM_STAGE1_SUBSTRATE.md` and GENREG_RULES: recurrent char substrate (emb_t +
  emb_t−1 + tanh(h_prev) → per-neuron 8-catalog activations → logits), soft multiplicative
  fitness (mean log-prob, no argmax anywhere), mandatory energy homeostasis (starved landed
  6–16%/gen in the target band), tournament selection + maturation gate, self-adaptive
  mutation with 0.02 floor + late anneal, fresh windows each generation, full-state
  checkpoints, runs/lm/ persistence (dashboard tab "lm"). Torch inference-mode only — no
  autograd, no pretrained weights. Corpus §VII bars measured first: majority 20.3%, char
  bigram 27.3%, trigram 39.2%. Sweep 1 (3000 gens, pop 400) running.
