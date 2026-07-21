# Changelog — REPLICATE

Per-project log for the REPLICATE line (recognize audio by replicating it —
the first temporal radial space with realtime input). Seeded 2026-07-19; new
REPLICATE entries go at the top of the log below, and also in the master
CHANGELOG.md.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-20] (Claude)** — **Module 48: spatial CONFIGURATION binding =
  NULL (-1.8pt, 0.6517 vs 0.6694).** Per-part windows (bind A@region1 &
  B@region2) don't beat co-located terms because the UNION already encodes
  configuration across genomes (each has a position) and the ridge binds it
  linearly => CIFAR@0.77 is largely linearly separable in union-of-single-
  features space. CONSOLIDATED m44-48: 0.77-0.78 is the single-layer
  gradient-free ceiling; our substrate is its param-efficient frontier; it's
  a representational-depth limit (gradient-free literature tops ~80-82%), not
  a search-time one. Strategic fork to user.

- **[2026-07-20] (Claude)** — **Modules 46-47: 0.77 is the SINGLE-LAYER
  ceiling (both directions).** m46 nonlinear k-means env fed to grammar =
  -5.2pt (0.6173 vs 0.6694, sparse-code/low-order-grammar mismatch — our
  grammar is already nonlinear + reads 2-3 dense channels). m47 canonical
  Coates-Ng pipeline (ZCA+kmeans+triangle+2x2 pool+ridge): K=400 0.702, K=800
  0.729, K=1600 0.752, K=3200 0.775. The full nonlinear dictionary only
  matches us at ~170K params and edges past (+0.55pt) at ~345K vs our ~8-20K.
  Our evolved grammar over LINEAR PCA is the param-efficient frontier of the
  same wall the nonlinear method hits. Perception axis DONE: 0.77 = single-
  layer info ceiling; 90% needs DEPTH (composition into spatial part
  configurations), not a better first layer.

- **[2026-07-20] (Claude)** — **Modules 44-45: perception primitive located.**
  m44 SPATIAL LAYOUT null: reading the substrate's grid (2x2 0.728, 3x3
  0.718) is worse than its scalar (0.770) — genome pool already optimal, so
  the limit is WHAT genomes detect not how they pool. m45 HAND-ENRICHED ENV
  regression: added Sobel-x/y + grad-mag + luminance channels, evolved R0
  over them; matched A/B (identical per-round seeds, both 573 genomes) RAW
  0.6694 vs ENRICHED 0.6525 = enrichment HURTS -1.7pt. Patch-PCA already
  spans edges (linear combos of pixels), so gradient channels add nothing
  and dilute the color-discriminative component budget. Lesson: can't
  hand-engineer perception into a LINEAR environment; the only lever left
  that could beat 0.77 is a NONLINEAR/learned feature patch-PCA can't
  represent. Env made channel-agnostic in radial_evo2.py.

- **[2026-07-20] (Claude)** — **Module 43 (bridge): memory does NOT break the
  clean 0.77 ceiling — structural.** Clean CIFAR as T=8 augmented views, two
  banks evolved fairly: memory 0.5123 vs memless 0.4507 = memory EARNS
  +6.2pt (clean, non-confounded), decay 0.318. But both ≪ clean single-shot
  0.77. Reason: evolving memory requires partial/degraded steps, and partial
  obs of a clean image is strictly less than the full image — memory
  recovers toward full-obs but can't exceed what one clean look holds
  (=0.77). Memory arc verdict: law validated, memory earns in its niche
  (degraded input), NOT the clean-ceiling lever. The 0.77 wall = substrate
  clean-image information; needs a better perception primitive.

- **[2026-07-20] (Claude)** — **Modules 41-42: MEMORY EVOLVES IFF THE
  ENVIRONMENT REQUIRES IT (user's law, validated cleanly).** Energy-
  constrained recurrent-memory genomes (leaky integrator + injected entropy
  + memory-cost-as-energy, bootstrapped no-op). On full-observation CIFAR
  memory REFUSED (2/268, decay 0.002, null below baseline) — a single look
  sees everything. Fix: corrupted-glimpse env (T=8, heavy noise+occlusion,
  no glimpse sufficient). Adoption 2/268 -> 227/259 (88%), decay 0.002->0.35
  — categorical switch, identical machinery. GENREG thesis on state.
  Caveat: does-it-earn confounded (ablate+refit only -0.37pt; bank coverage
  compensates; corruption brutal, 0.29 ceiling). Machinery validated;
  system payoff + transfer-to-clean not yet shown.

- **[2026-07-20] (Claude)** — **Modules 39-40: reconstruction beautiful,
  generative fitness NULL.** (39) Linear decoder (closed form, gradient-free)
  reconstructs held-out CIFAR RECOGNIZABLY from ~2,000 word activations (MSE
  0.0077 vs 0.0618 baseline) — constructive proof the description is
  sufficient; per-word pixel signatures = localized color-texture detectors.
  (40) Words evolved to explain the pixel RESIDUAL reconstruct better (MSE
  0.00540 vs 0.00709) but classify at 0.5464 alone; union 0.7695 vs
  class-informed 0.7703 = null. LAW: reconstruction sufficiency ⟂
  discriminative sufficiency (pixel-MSE = low-freq color/background, not
  class detail). The 0.77 ceiling is NOT description lossiness — it's the
  substrate's linear-separability.

- **[2026-07-20] (Claude)** — **Module 38 (radial space R1 at substrate
  level): NULL.** R0-union grids → evolved R1 genomes. Ridge: R0 0.7698 →
  R0|R1 0.7731 (+0.33 noise); raw skip overfits (0.7665). Vocabulary
  (per-view, 830w): 0.7284 = R0-only 0.7285, identical. The 7-union already
  holds the single-space info; one stacked space is redundant for any
  readout. Depth needs to be DEEP (multi-space fresh), not +1 on a union.
  Best stands at per-view 0.7772.

- **[2026-07-20] (Claude)** — **Modules 36-37: CROSSED THE SUBSTRATE HEAD.**
  Per-view head 0.7772 > substrate ridge 0.7708 — first description-only
  model to beat its perception layer. Module 6's +2.6pt mechanism pinned:
  LINEAR-per-view, not conjunction (m36 cross-view words tracked BELOW
  averaged at 0.7285; m37 per-view linear head 0.7669→0.7772). Same 1,931
  words, each hands the head 3 per-view activations vs the mean; params
  32K→71K. Batched evaluator (sharedop_fast, bit-exact) took the 96GB pod
  44%→96% GPU. Path to 0.7965 clear: per-view scales with word count, vocab
  still climbing. Growing on the 96GB pod.

- **[2026-07-20] (Claude)** — **Module 35 (shared operator semantics,
  private operands): EFFICIENCY HEADLINE.** The user's fix to module 34:
  keep operands fully private, share only the sharp/th/p/b that define an
  operator's meaning (32 genes total). Curve: 467w 0.7111 @ 7,776p · 927w
  0.7424 @ 15,550p · 1,183w 0.7509 @ 19,742p (climbing). vs flat: same
  accuracy tiers at 40-45% of the params (flat 0.7472 needed 44,351p).
  Sharing amortizes (saves 4,879 genes at 1,183w) AND regularizes (count
  sharp→0.164, compare→0.10 — private sharpness was noise). relate +
  conditional now ~20% of the vocabulary. Headline: 71% from 7.8K vocab
  params. Growth continuing.

- **[2026-07-20] (Claude)** — **Module 34 (factored vocabulary): LOSES.**
  word = op + args + binding from shared codebooks, two configs: 160x4 →
  0.6173 @ 7,779p; 640x6 → 0.6442 @ 13,429p — both behind flat (0.7106 @
  13,583p), and compression inverted at the big codebook. Reason was
  already in module 30: the vocabulary has NO redundancy, so there are no
  shared themes to factor — each word is an independent fact and sharing
  removes its freedom to name its own channels. Operator census did
  transfer (compare/count/conditional/relate). Module 33b (substrate
  evolved in a different world) also null: 0.7666.

- **[2026-07-20] (Claude)** — **Pod move + module 33 (second substrate) =
  NULL.** Blackwell pod needs torch cu128 (2.4.1 can't do sm_120). THE
  PATCH-PCA BASIS IS NOT PORTABLE: rebuilt caches cost 1.5pt (0.7672 →
  0.7526, 29% of words shifted) because near-degenerate eigenvalues make
  component rotation arbitrary — ship the cached blocks, don't recompute.
  After shipping 1.8GB the pod reproduced local to 4 decimals. Module 33:
  fresh-substrate words admitted 15 in 25 rounds, TEST 0.7670 vs 0.7672 —
  a second substrate evolved the same way rediscovers the same truths;
  the seed-union law does NOT transfer from words to substrates.

- **[2026-07-20] (Claude)** — **Module 31 (fixed budget): NULL.** 2,600 words
  chosen from a 7-seed / 4,067-word pool = 0.7609 vs the same-size lean
  prune's 0.7635. Choice from a bigger pool buys nothing; the union's gains
  were ACCUMULATION of distinct true things. Holdout still rises with size
  (2,600w 0.7607 → 4,067w 0.7637). Fixed-budget ceiling ≈ 0.76.

- **[2026-07-20] (Claude)** — **Module 30 (prune): nothing is free to cut.**
  5-seed union 0.7672 @ 3,337w / 67,293 params; pruning curve is nearly
  linear in vocabulary size (1,000w 0.7420 … 2,600w 0.7601), so every word
  is load-bearing — the opposite of dense-net pruning. LEAN 2,600w /
  53,259 params / 0.7635. Efficiency ledger is now honest and unflattering:
  the model is BIGGER than the 36,460-param substrate head and still
  behind it. Next: fixed-budget evolution (replace weakest words, don't
  extend).

- **[2026-07-20] (Claude)** — **Module 29: NEW BEST 0.7545 @ 2,487 words /
  49,847 params.** Cross-seed control: class-informed beats label-free by
  +4.1pt (seed 73) and +3.0pt (seed 137) — fitness law CONFIRMED, not a
  single-seed artifact. The two vocabularies differ by 1.1pt, so their
  UNION pays: 0.7472 → 0.7545 (+0.73pt), the seed-union law reproduced at
  the word level. 1.6pt from the substrate head. Third seed in flight.

- **[2026-07-20] (Claude)** — **Modules 27-28: fine scales NULL; standing
  result honestly sized at TEST 0.7465.** 2x2/3x3 patches → 36 words,
  0.7446 (the 4x4+ basis was correctly specified). Protocol fix: size and
  lambda chosen on a 15% holdout the admission never touched (peak k=110,
  reproducing the test peak), then refit on all 50k, test once → 0.7465
  @ 2,127 words / 41,790 params. Class-informed fitness diverges val from
  test at scale; label-free never did.

- **[2026-07-20] (Claude)** — **Module 26 (raw environment): NEW BEST 0.7460
  @ 2,127 words / 41,790 params.** Words finally allowed to read the
  patch-PCA maps (2,160 channels of where/at-what-scale) instead of only
  pooled genome outputs: 0.7415 → 0.7444 (30 words) → 0.7460 (110). Op
  census: count 66 / select 23 — the answer to "what did pooling destroy"
  is CARDINALITY OVER SPACE AND SCALE. Arm-D saturation was about the
  channel set, not the substrate.

- **[2026-07-20] (Claude)** — **Module 25 (hybrid): NEW BEST 0.7415 @ 1,950
  words.** A label-free 0.7017 · B class-informed 0.7232 · C union 0.7410
  (+1.8pt over the best single — the two questions index different
  information) · D +21 top-up words 0.7415. Arm D = saturation test, reads
  CLOSED (21 words, +0.05pt, refutations 23/round). Gap to the substrate
  head (0.7708) now 2.9pt. Run 20260720-070912-replicate_hybrid-101727.

- **[2026-07-20] (Claude)** — **Module 24: the FITNESS was the bottleneck —
  new best 0.7106 @ 619 words / 13,583 params** (label-free 0.7017 @ 1,095
  / 19,068). Class-informed fitness wins at every matched size (+3.5 to
  +4.1pt). Grammar census FLIPS with the question: truth → prod 163 /
  count-select-compare 0; usefulness → count 175 / select 143 / compare
  157 / prod 12. Truth is multiplicative, separability is relational and
  cardinal. Keep both vocabularies (portable language + task vocabulary).

- **[2026-07-20] (Claude)** — **Module 23 (expand the environment): NULL
  (0.7017 → 0.7019) with 192 words over 4 question-tuned vocabularies.**
  Evolution consumed the new material eagerly (8/round, ZERO refutations,
  implied-rejections down to 6-33) and it bought nothing → novel-to-the-
  LANGUAGE ≠ informative about CLASSES. Face census (label-free): crop 113,
  occlude 104, warp 103, color 36 — objects are parts-that-persist-across-
  scale, not colored shapes. PIVOT: the binding constraint is the vocabulary
  FITNESS, not the environment.

- **[2026-07-20] (Claude)** — **Module 22 (significance floor): arbiter gain
  is SOFT EVIDENCE only.** 28 specialists (0.7035) < 14 (0.7044); Z-filter
  shows ZERO specialists clear 1.5 SE on their own gate slice (Z>=1.5 → 0
  kept). Test σ≈0.46pt, so the +0.5-0.8pt effect is inside the ±1.4pt soft
  band. Honest standing number: **0.7017 @ 1,095 words / 19,068 params**
  (rung 6). All internal levers now null or sub-noise → the environment is
  the only real headroom left.

- **[2026-07-20] (Claude)** — **Rung 8 (arbiters over the frozen model):
  EARNS — 0.6966 → 0.6987.** Confusion specialists (2-class ridge over the
  full description on a pair's images + evolved features), routed on
  frozen top-2, gated on a third split: 4/5 kept (cat/dog, deer/horse,
  airplane/ship, bird/deer), 23.7% of test routed. First attempt was a
  crippled null (4 features vs 1,287) — equip an ablation before trusting
  its null. Run 20260720-041227-replicate_arbiter-c24f70.

- **[2026-07-20] (Claude)** — **Rung 7 (real R1, bank hand-off): NULL —
  0.7017 → 0.7009.** The scalar-hand-off fix worked mechanically (R1
  admitted 8/round for 24 rounds, 192 words, ~0 refutations, vs the
  scalar-fed reader's 30-44 refutations/round) but bought nothing: R1
  words are novel w.r.t. the language, redundant w.r.t. the labels. Three
  depth attempts now agree the description is class-saturated → headroom
  is in the ENVIRONMENT, not depth/reader/grammar.

- **[2026-07-20] (Claude)** — **Rung 6 (rich grammar): PAST 70% — TEST
  0.7017, 1,095 words, 19,068 params.** count/select/compare admitted ZERO
  words (smooth pooled channels have no crisp "how many"/"which"); the
  ARITY unlock did it — arity spread 2-8 vs the old cap of 3, novelty
  fitness 0.40-0.52 vs 0.23-0.43. Standing gap (user's question): every
  Oclip module evolves ONE radial space with scalar hand-offs — next is a
  fair R1 over the 1,095-word x 3-glimpse bank.

- **[2026-07-20] (Claude)** — **Rung 5 (supervised nonlinear reader): NULL
  (0.6970 → 0.6970, 43 reader words).** "Multiplicative" is a SUBSTRATE
  law, not a language law: over stable words, products amplify
  idiosyncrasy and get refuted on verify (30-44 refutations/round);
  survivors were attend/gate (conditional reading) and still bought
  nothing; prod readers all collapsed to arity 2. The description is
  linearly saturated → accuracy needs better WORDS, not a smarter reader.
  Run 20260720-035611-replicate_reader-748af4.

- **[2026-07-20] (Claude)** — **Rung 4 (glimpse sweep): 3 looks beat 12
  (0.6944 vs 0.6877), lexicon unchanged.** Non-monotone val curve (1:0.6940,
  3:0.6987 win, 4:0.6862, 9:0.6974, 12:0.6906). Mean-aggregated words are
  DILUTED by near-duplicate looks; but max-diversity also lost (0.6809) —
  looks must decorrelate error without changing what the word means.
  Free ~0.7pt + 4x cheaper inference. Run
  20260720-035026-replicate_glimpse-97cd8c.

- **[2026-07-20] (Claude)** — **Rung 3 (cross-modal): 0.6946 → 0.6970 with
  83 straddling words; the face census is the result.** Given all four
  faces the language picks visual 80 / lang 46 / shape 36 / letter 15 —
  private-language codes pulled 3x more per channel; the letter bank
  (0.6371 alone!) nearly ignored = modules 3/5 absorption seen from
  inside the language. V=3 control (0.6946) > V=12 read (0.6877): fewer
  cleaner glimpses win. Run 20260720-033830-replicate_oclipcross-ef3b2d.

- **[2026-07-20] (Claude)** — **Rung 2b curve: 903 words = TEST 0.6877 @
  15,530 params.** 674→0.6801, 792→0.6831, 903→0.6877; earn rate
  ~+0.3pt/100 words (vs rung 1's +3pt/100) → the environment, not the
  question, now binds. Novelty ops prod 247 / absdiff 104: conjunction is
  what additive lexicons can't say. Next: cross-modal words + richer
  glimpse geometry. Run 20260720-033420-replicate_oclipnovel-977cc3.

- **[2026-07-20] (Claude)** — **Rungs 2 + 2b: 674 words = TEST 0.6801 @
  11,554 params.** Rung 2 (predictive, two disjoint look-sets) saturated
  like rung 1 — 0.99 scores, zero verify failures, +0.6pt (547→0.6759).
  Rung 2b (conditional novelty: variance the whole lexicon can't explain
  × consistency) does NOT saturate — scores 0.23-0.43, "implied"
  rejections dominate, target moves with every admission; +0.4pt so far
  and still climbing (674→0.6801). Novelty words are MULTIPLICATIVE:
  prod 91 / absdiff 35 / gate 1. Runs ...-oclippred-cd5f03 /
  ...-oclipnovel-5b9745.

- **[2026-07-20] (Claude)** — **Rung 1 complete: label-free language scales
  to 505 words = TEST 0.6702 @ 8,584 params.** Curve: 174→0.5972,
  219→0.6159, 308→0.6437, 397→0.6541, 505→0.6702; per-100-word gain
  compressing → flat ceiling ~0.67-0.70. Beats the label-guided
  vocabulary by +6.9pt with zero label contact; val-test ≤1pt throughout.
  Next: predictive-word fitness (consistency question saturated).

- **[2026-07-20] (Claude)** — **Module 13 (Oclip label-free): TIED with
  label-guided — 0.5972 vs 0.6013.** 174 words, zero label contact
  (consistency fitness + orthogonality + disjoint-probe verify; labels
  read the finished language once). Consistency 0.95-0.98, zero verify
  failures, orthogonality did all the shaping. The human taxonomy is
  recovered for free by a truth-seeking language — encoder convergence at
  vocabulary scale; diversity-first law restored. English mapping =
  translation, not inheritance. Run
  20260720-023752-replicate_oclipfree-cfe25a; notice #650.

- **[2026-07-20] (Claude)** — **Module 12 (Oclip L2): flat on test (0.6028
  vs 0.6013) — depth can't outrun a thin lexicon.** 18 compounds (attend-
  dominated) cleared all four gates, +2pt val, +0.15pt test. Linear
  compounds are definitionally head-redundant; nonlinear compounds of 117
  words are few. Bottleneck = L1 vocabulary breadth → widen the flat
  vocabulary before adding layers. Run
  20260720-021443-replicate_oclip2-eb5026; notice #649.

- **[2026-07-20] (Claude)** — **Module 11 (Oclip v1): 117 proto-concepts =
  TEST 0.6013 from 2,504 params total.** The description-only model: words
  admitted through four gates (bottleneck gain, view-consistency ≥0.35,
  orthogonality <0.8, second-split verify). Val-test gap 0.5pt. 11x the
  substrate head's accuracy-per-param; flat k≤4 grammar saturates ~0.60 →
  module 12 = words made of words (the LM half). Run
  20260720-015925-replicate_oclip-9d070f; notice #648.

- **[2026-07-20] (Claude)** — **Module 10 (heavylift): slot competition v1
  REGRESSED (0.7822 → 0.7789 test at 96 swaps) — micro-admissions below
  split noise compound; the day's third Goodhart variant.** Every swap
  looked positive on its own fresh split (+0.0002 cap vs σ≈0.4pt noise).
  Fix: double-split verify gate (winner must beat base on a split it was
  never selected on, cap 0.0008); state reset, rerun in flight. Run
  20260720-013816-replicate_heavylift-5be0ed.

- **[2026-07-19] (Claude)** — **Module 9 (temporal): the original vision,
  right form — 0.7822 at fixed 3,645 width.** Glimpse stream as time, 8
  evolved aggregation policies + per-genome assignment (~120 genes; head
  never grew). Control 0.7779 / evolved 0.7822 / visual-alone 0.7708 —
  +1.14pt at identical head size; ~40k total params vs the fat record's
  138k-182k head (0.7967): ~2.3x accuracy-per-parameter. Census: 19% of
  genomes abandoned the mean (persist/range/pmean/std/max/min/gate).
  Run 20260719-233903-replicate_temporal-56439c; notice #646.

- **[2026-07-19] (Claude)** — **Module 8 verdict: conjunctions over 16-dim
  vocabularies plateau at +0.1-0.2pt (noise band).** Five reads, 54→145
  genomes, no growth; 161 checkpointed (resumable). The vocabulary axis
  itself validated (warp 0.3453 > generic 0.334, label-free). Bottleneck =
  vocabulary WIDTH not genome count. Record stands: 0.7967 (module 6).
  Notice #636.

- **[2026-07-19] (Claude)** — **Module 8 in progress: question-tuned
  vocabularies + conjunctions.** Four aug_kind encoders trained label-free
  (warp 0.3453 kNN best, beats generic 0.334); concat control null (0.7941
  → 0.7943); conjunction residual monotone +0.09 → +0.11 → +0.18pt at
  54/81/102 genomes (0.7960 latest). Resumable-chunk evolution added to
  replicate_compose (state checkpoint + rng advance). Stop rule
  pre-registered: two flat rounds or ~240 genomes.

- **[2026-07-19] (Claude)** — **Module 7 (push80): val-greedy selection
  Goodharts.** New views/frames cached; greedy-on-val package hit val
  0.8004 but TEST 0.7938 < module 6's a-priori 0.7967 (record stands).
  Package selection is itself a fitness — it overfits val (~0.7pt gap).
  Module 8 launched (user's direction): question-tuned vocabularies —
  aug_kind added to evolve_encoder; replicate_vocab.py training
  color/crop/occlude/warp encoders, then conjunction composition vs the
  measurement package. Run 20260719-171459-replicate_push80-da2862.

- **[2026-07-19] (Claude)** — **Module 6 (multiview): NEW GRADIENT-FREE CIFAR
  RECORD 0.7967** (prior 0.7702/0.7741). Views (flip/shifts/zoom) + the
  augmented-frame replay, frozen genomes only, zero new evolution. Ladder:
  visual 0.7708 → +view-mean 0.7884 → +4-views concat 0.7965 → +aug-frame
  0.7873 → GRAND 0.7967 (+2.6pt, five arms agree, both probes earn
  independently). Shared-source diagnosis confirmed causally; the immortal
  airplane→ship miss finally correct. Run
  20260719-152747-replicate_multiview-2d170e.

- **[2026-07-19] (Claude)** — **Module 5 (visunion): vision + the trained-off
  model = clean null.** Visual 0.7708 → +offunion model 0.7704 → +all 3
  seeds' 1,101 composed columns 0.7709. The genomes that earn +0.57..+0.87
  over their own union are fully redundant with vision — nonlinear
  composition does not escape the shared 32×32 source. Fourth convergent
  probe on the 0.77 wall; same airplane→ship miss in every configuration.
  Run 20260719-145657-replicate_visunion-4eef8a; notice #618.

- **[2026-07-19] (Claude)** — **Module 4 result (3-seed): the campaign's first
  real residual.** Union 0.6503-0.6504 → union+genomes 0.6560-0.6591 (seeds
  11/23/37, residual +0.57 to +0.87pt, same sign all seeds; LEAN ~0.53).
  A CIFAR perceiver grown entirely off substrates that never saw CIFAR:
  **0.6591** — above letter-alone 0.6371 and Coates-Ng 0.59. Same rig that
  nulled over the visual bank earns here → evolution-suppression law
  measured in one campaign: headroom exists exactly where the base doesn't
  own the signal. Runs 20260719-144341/-144502/-144537.

- **[2026-07-19] (Claude)** — **Module 4 launched: train off the alphabet+shape
  union (user's call).** No visual features anywhere — the letter+shape union
  IS the environment, genomes grow on top. `replicate_compose.py --blocks
  letter,shape --tag offunion --rounds 60`, full 50k run in flight. Smoke:
  union 0.6060, earning rates 3-5x module 3's (real headroom below the
  foreign-union base).

- **[2026-07-19] (Claude)** — **Module 3 result: NULL — diagnosis complete,
  the 0.77 wall is shared-source error.** 237 cross-modal genomes (conjunction
  ops dominated), per-round fresh-val gains +0.0008-0.0027, final: concat
  0.7741 → +genomes 0.7740 (−0.0001), LEAN 0.5401/237 cols. Round gains were
  split noise that never accumulated. Three probes agree (concat null,
  compose null, union saturation at 0.7702): all faces re-encode the same
  32×32 pixels — re-viewings of one measurement, not independent
  measurements. Next: faces with INDEPENDENT error (multi-view re-reads of
  the substrates, or the audio campaign where the signals really are
  different). Run 20260719-110010-replicate_compose-2fe87d.

- **[2026-07-19] (Claude)** — **Module 3 built: COMPOSE
  (`replicate/replicate_compose.py`).** Cross-modal genomes over the joint
  5,093-channel bank: prod/gate/min/absdiff/attend conjunctions across
  faces, §3.8 operators (relative mutation + floor, index drift + cross-face
  jumps, orthogonal admission |corr|<0.85 on a fixed probe, pop//4
  truncation), fitness = residual gain over [full concat | frozen genomes]
  with free-base auto-ablation, FRESH VAL PER ROUND (first smoke provably
  chased val noise: +genomes 0.6990 < concat 0.7060; fixed: 0.7060). Cap
  live-tunable via `replicate/compose_cap.txt`. Full run in flight vs
  module 2's 0.7714.

- **[2026-07-19] (Claude)** — **Module 2 result (full 50k/10k): faces agree,
  concatenation earns nothing — and the transfer numbers are the story.**
  Eyes 0.7710 / ALL 0.7714 (+0.04pt = noise, not a record; notices #607/#608).
  Letter bank alone on CIFAR **0.6371** (vs raw ridge 0.324), shape bank
  alone 0.5137 (191 feats), private language 0.4472 (64 dims). Shared-error
  diagnosis per the seed-union saturation law → module 3 should evolve
  genomes OVER the joint multi-face bank (cross-modal composition a linear
  head cannot tabulate). Export radial_data/replicate_cifar.json, run
  20260719-012336-replicate_cifar-a2c9eb.

- **[2026-07-19] (Claude)** — **Pivot: multimodal convergence first, audio
  later.** Task reframed: the model gets EVERY face of the concept — visual
  union (eyes), private language (label-free contrastive encoders), letter
  bank (symbols), shape bank (geometry) — and one closed-form head notices
  they agree; the label is never given, the model arrives at its own word.
  `replicate/replicate_cifar.py` = module 2: CIFAR-10 ladder (each face
  alone, visual+lang, visual+letter+shape, ALL) against the 0.7702 anchor;
  honest protocol (per-block train-stat standardize ±8sd clamp, lam on val,
  full-train refit, test once per arm); feature blocks cached in
  `replicate/cache/`. Page lede + `__init__` + route docstring reframed;
  `arms` renderer added to `static/replicate.js`.

- **[2026-07-19] (Claude)** — **REPLICATE project created — seeded from the
  two best frozen substrates.** New project targeting audio: the model will
  recognize a stream by REPLICATING it, given tools that make replication
  easy — the LETTER bank (`ocr/models/letters_v1_model.json`, 0.88 held-out
  fonts) as the token substrate, and the temporal SHAPE bank
  (`radial_data/anim_model_shape.json`, 0.9989 anim test, motion-invariant,
  2 self-sized spaces) as the waveform's home in 3D space. Checkpoints
  copied to `replicate/checkpoints/` (letter_ocr_v1.json /
  shape_temporal.json); `replicate/replicate_seed.py` replays both in their
  native protocols (ocr_model replay on held-out fonts; the anim_ablate
  6-frame feature walk with head refit on train, anim test split) and is the
  project's module 1 — exports `radial_data/replicate_seed.json`, appends
  the module registry, records the run under `runs/replicate/`, posts a
  notice. Page: `/replicate` (append-only /lm pattern, newest-at-bottom,
  snap button) + `/api/replicate/modules` + `/api/replicate/export/<name>`
  (whitelisted `replicate_*.json`); `static/replicate.js` renders modules.
  Nav entry (Sequence group) + changelog-modal mapping added. **Flask
  restart needed** for the new route/endpoints.
