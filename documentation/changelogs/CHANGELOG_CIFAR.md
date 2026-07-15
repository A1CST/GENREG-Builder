# Changelog — CIFAR

Per-project log for the CIFAR line. Seeded 2026-07-14 from the master
CHANGELOG.md (all entries mentioning this project); new CIFAR entries go at
the top of the log below, and also in the master CHANGELOG.md.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-14] (Claude)** — **CIFAR champions wired into the /tsdb page.**
  `/api/tsdb/data?set=cifar` serves the frozen CIFAR champions
  (`demo/cifar_genomes.pkl`, no model load) as Float64 series for the in-browser
  TSDB block store: full layer ladder (centroid `.4007` → argmax → mixer → joint
  → +bias `.5592` → +pairs → full), all 45 one-vs-one pairs, and the 10 per-class
  one-vs-rest detectors with real class names. Selectable via a MNIST/CIFAR
  toggle on /tsdb. See master CHANGELOG for the store/stress write-up.

- **[2026-07-14] (Claude)** — **RunPod H100 online + Schur fast-path rewrite.**
  (1) FITNESS REWRITE (`make_scorer` in `radial_evo2.py`): the frozen base's
  normal equations are Cholesky-factored ONCE per round; every candidate is a
  rank-1 Schur border — O(N·F) per candidate instead of O(N·F²), whole batch
  scored in fused matmuls. Parity vs exact solve: soft err 2e-7, acc err 0;
  **68× per candidate at F=300** (grows with F). Local crossover run resumed
  from checkpoint onto the fast path (hard-kill + atomic-ckpt resume, no
  test-set peek). (2) RUNPOD: H100 80GB / 224 cores / 2TB RAM at
  103.207.149.106 (port changes per restart). Project built in
  `/workspace/genreg-radial` (5 modules + both CIFAR npz); NOTE: everything
  outside /workspace is EPHEMERAL — authorized_keys died with the first
  restart (key must live in RunPod account settings for durability). Pod runs
  the full crossover replicate (seed 13; local runs seed 7). (3) SHADOW COPY
  (user rule: pods lose data): `runpod_shadow/` mirrors every pod file; a
  persistent 5-min sync loop pulls log + checkpoint + results and surfaces
  pod progress + sync failures. Both GPUs busy.
- **[2026-07-14] (Claude)** — Evolution campaign part 9: **GRAMMAR V2 RESULT —
  TEST 0.7035 on full CIFAR.** Natural convergence after 96 rounds / 114 min:
  592 genomes, honest-10k val 0.7104, **10k test (touched once) 0.7035** —
  val-test gap only 0.007, the instrument held. Ladder: raw 0.324 → PCA
  0.360 → hand-crafted Coates-Ng 0.5904 → v1 evolved 0.6198 → v1 tower
  0.6378 → **grammar v2 0.7035** (+6.6 over the v1 ceiling, +11.3 over
  hand-crafted, zero gradients, ~60 KB of genomes + ridge head). What
  evolution chose when nothing was hand-picked: all SIX patch scales
  colonized (4-14); 426 order-2 / 166 order-3 interactions; pooling stats
  mean 292 / std 238 / max 62 — the heavy use of evolved-window STD pooling
  (local texture-variance detectors) is a structure no one designed. The
  expressivity hypothesis is confirmed: the v1 wall fell at round 30 with a
  third of the genomes. Export `radial_data/evo2_cifar.json`.
- **[2026-07-14] (Claude)** — Evolution campaign part 7: **the honest-ruler
  deep run (full 50k, 10k val) — the bet was WRONG, and the answer is
  better.** `run_deep_50k()`: stages 2+ evolved on full CIFAR, fitness val =
  10k held-back train samples, test touched once. The instrument now reads
  true (stage-1 val 0.6349 ≈ its real test 0.6378 — no inflation). Result:
  stages [667, +68, +26, +6], val 0.6429, **TEST 0.6279** — the tower
  converges at ~770 genomes and does NOT move past the 8k tower's 0.6378.
  Reading: the 8k run's 310 overfit-selected stage-2 genomes still helped at
  full scale as BULK diversity (random-feature effect: quantity beat
  precision), while the honest bar freezes only precision picks and the
  genome class is simply near-exhausted at ~0.63-0.64 full-CIFAR. WHERE IT
  ACTUALLY STOPS: this feature-genome class (2 components, depth-1 bends,
  one op, one pool) saturates here; further lift requires EXPRESSIVITY
  (multi-scale patches, deeper programs, genomes-of-genomes), not more
  selection. Ladder top remains 0.6378. Export `evo_deep50_cifar.json`;
  levers/checkpoints worked; run took ~9 min of GPU.
- **[2026-07-14] (Claude)** — **Evolution campaign part 3: THE v1 MILESTONE
  FALLS.** `phase_b_full()` transfers the 647 evolved genomes to full CIFAR
  (50k/10k; identical environment basis so component indices keep meaning;
  head refit closed-form; test once): **0.6198 vs the v1 hand-crafted
  Coates-Ng milestone 0.5904 (+2.9 pts)**, 35 s. The no-gradient CIFAR
  ladder now: raw 0.324 → PCA 0.360 → hand-crafted patches 0.5904 →
  **evolved genomes 0.6198**. Genomes evolved on 8k transfer to 50k without
  re-evolution. Export `radial_data/evo_full_cifar.json`; findings §11.
- **[2026-07-14] (Claude)** — **Evolution campaign part 2: THE CEILING BREAKS
  (Phase B) + the genome map (Phase C).** Phase B in `radial_evo.py`:
  interaction feature genomes over a label-free patch-PCA environment (40
  comps, 6×6×3; "features are the environment"); genome = 2 components + 2
  evolved lens bends + combine op + pool → one scalar/image; soft
  residual-gain fitness, tournament+elitism+energy, freeze-and-compose ≤8
  decorrelated winners per round, closed-form ridge head, test measured ONCE.
  **CIFAR 8k/2k: 24 features → 0.4030 (past the pointwise ceiling 0.3845 in
  4 s); 317 → 0.5565 (past hand-crafted Coates-Ng 0.493); 647 → 0.5840 in
  231 s, converged naturally — within 0.006 of the v1 FULL-50k milestone on
  6× less data, in a ~52 KB model, zero gradients.** Phase C maps the 647
  genomes by behavior: effective dim **405/647 vs the lens bank's 46/748**
  (why it works — evolution manufactures new directions ~9× more efficiently
  than enumeration); population near-spherical (8.3/7.4/6.6). Honest
  negatives: no radial expansion over discovery time (corr −0.15) and op
  families don't cluster. Exports `radial_data/evo_interact_cifar.json`,
  `evo_genome_map.json`; findings §11 updated. Next: same recipe on full
  50k CIFAR vs the 0.5904 milestone.
- **[2026-07-14] (Claude)** — **Evolution campaign (autonomous, CIFAR) part 1:
  Phase A + a MAJOR CORRECTION + the zombie.** New `radial_evo.py`. Phase A
  (genome #2, evolve ON the map: plane angles/offset/size select a lens
  region; batched GPU fitness; random-genome control at equal budget):
  evolved 0.3695 test ≈ random control — and the follow-up diagnostic caught
  a real flaw in OUR baselines: the rotation probes selected the best angle
  ON THE TEST SET (max of 360 noisy test measurements ⇒ winner's curse
  ≈ +0.03). Honest protocol (select on val, test once): **CIFAR slice =
  0.3845**, not 0.4015 — still ≈ the 400-lens bank with 6% of the lenses and
  above random 0.3774, but "slice beats bank AND raw" is retracted; audio's
  0.4025 gets the same caveat. What survives: below-random dead zones
  (linear core), loops spread, redundancy analysis. Findings §10-§11, PDF
  regenerated. ENGINEERING: all the "CUDA wedge" slowness across the session
  was ONE zombie GA process squatting the GPU for hours (task brq24ri2p);
  after killing it Phase A runs in **42 s**; also fixed an fp16 gram overflow
  (z-scored gram entries ~1.5e5 > fp16 max ⇒ singular solves) — grams stay
  fp32. Phase B (interaction feature genomes — the real headroom) next.
- **[2026-07-14] (Claude)** — **The four open questions ANSWERED; PDF updated
  to 6 pages; main fast-forwarded on GitHub.** Q1 (why a disc): MDS axis 1
  correlates +0.99 with LINEARITY; axis 2 is response-curve shape (79% of
  curve variance in 2 PCs) — lens behavior has ~2 dominant degrees of freedom.
  Q2 (dead zone): **it is the linear core itself** — worst slice = 23 `id`
  lenses (nl 0.000, intra-distance 0.34 vs 9.10 random); floor 0.1694 = the
  raw-linear score; same floor on every axis because every great circle passes
  through the origin; REMOVABLE — dropping the 126 near-linear lenses raises
  the floor 0.169→0.822 (spread 0.83→0.18). Q3 (temporal rotation):
  **CONFIRMED** — lens-view agreement across time steps (lens-space
  autocorrelation, δ=1..16, phase-invariant, closed-form) takes audio tones
  from 0.41 (raw) / 0.37 (pointwise) to **1.0000 with 16 features** (caveat:
  identity alone saturates this task — diversity awaits a harder one).
  Q4 (genome bridge): first radial genome `radial_slice_ga.py` (24-lens slice
  picker on CIFAR, soft fitness + energy, batched GPU fitness — one (POP,N,N)
  LU per generation after the sequential version wedged CUDA twice in
  background runs; 305 s total): **0.376 = random-subset level, below the
  map-geometry slice 0.4015** — index-space fitness is flat by nature
  (±0.002); the genome's search space should be the MAP itself (plane/angle/
  region), which is the design brief for genome #2. Findings doc §8 + PDF
  page 6 (`RADIAL_SPACE_FINDINGS.pdf`). GitHub visibility fixed: the branch
  was pushed but the repo front page shows `main` — main fast-forwarded to
  the branch head (1673a49→4224e61) and pushed.
- **[2026-07-13] (Claude)** — **ENTIRE RADIAL BASELINE ROADMAP RUN** (all
  domains + pre-baseline fixes; results in `documentation/RADIAL_BASELINES.md`
  and `radial_data/baseline_{cifar,text,audio}.json` + `prebaseline_fixes.json`).
  Headlines: **CIFAR** raw 0.3820 vs bank 0.3815 — pointwise lenses add ZERO
  (interaction-bound domain); but a 24-lens rotation slice hits 0.4015,
  beating the full bank AND raw. **TEXT** lens bank 0.2651 vs measured
  bigram-table ceiling 0.2661 — the bank recovers the bigram function to
  within 0.001 with NO table; nothing above the ceiling is reachable pointwise.
  **AUDIO** raw 0.4105 vs bank 0.3725 — the bank HURTS; random phase makes
  every pointwise view class-identical (temporal rotation confirmed mandatory);
  biggest angular spread of any domain (0.1625). **FIXES:** Z-axis whitening
  does not shrink dead zones (spread 0.8282 vs 0.8285, same 0.1694 floor);
  x/y/z spin axes all equivalent — free rotation fine, no anchor needed.
  Cross-domain law that fell out: the less pointwise diversity can buy
  (MNIST +3.9 > CIFAR +0.0 > audio −3.8), the more WHICH slice you take
  matters — slice > full bank on the hard domains. Fix en route: radial_map
  was missing `import os` (cifar stream crash mid-sweep). Genome targets are
  now defined by these gaps per the roadmap's closing line.
- **[2026-07-13] (Claude)** — **Radial baselines campaign started** (user's
  roadmap `~/Downloads/radial_space_baselines.md` → copied to
  `documentation/RADIAL_BASELINES.md`). New `radial_baseline.py`: full lens
  bank + closed-form linear head per data domain, no genomes — implemented as
  summed per-lens linear kernels + kernel ridge on CUDA (RTX 4080), input
  format logged as FLAT/POINTWISE (a GAM over pixels). Radial-owned MNIST copy
  (`radial_data/mnist_radial.npz`, 8k/2k, built read-only from corpora/mnist).
  **MNIST BASELINE DONE:** raw-pixel linear 0.8005; lens bank 0.827 (L=8) →
  0.8395 (L=400) — saturates immediately, so the roadmap question is answered:
  COMPOUNDING problem, not coverage (pointwise GAM can't see pixel
  interactions). Per-class: 5 and 2 hardest. Top lenses are gauss/abs folds
  (ink detectors). Rotation probe on MNIST: spread only 0.028 (vs loops' 0.83)
  — a 24-lens slice already hits the ceiling. MNIST-domain map is a flat cigar
  (axis std 6.8/1.7/1.4). Export `radial_data/baseline_mnist.json`. Fixes:
  NaN guards in `_sig` (mostly-zero pixel streams degenerated corrcoef and
  poisoned the MDS — the first rotation run scored 0.0 everywhere); `mnist`/
  `cifar` added as data kinds (page select too). Next per roadmap: CIFAR-10
  baseline, then text/audio; pre-baseline fixes (Z-axis expansion, rotation
  axis lock) still open.
- **[2026-07-13] (Claude)** — **Radial v1 ARCHIVED, rebuilt WAY simpler as the
  activation-behavior map** (user's direction, from their activation-galaxy
  research: characterize activation functions by how they TRANSFORM data — not
  the formula — and the space self-organizes: linear at centre, nonlinearity
  outward, oscillators on their own branches). All nine v1 modules + the old
  six-mode page moved intact to `archive/radial_v1/` (README there records each
  verdict, incl. the no-gradient CIFAR 0.5904; nothing deleted; `radial_data/`
  left in place for the future images baseline). NEW `radial_map.py` (~230
  lines): deterministic index-addressed lens programs (1-3 composed primitives
  `prim(a·x+b)` from a 14-op catalog, lens 0 = identity anchor), behavioral
  signatures on baseline numeric-loop data (response curve over the data's own
  quantile grid + 7 behavior stats), classical-MDS 2D map centred on identity,
  and a closed-form ridge linear probe. New `/radial` page (map scatter with
  identity-centred rings, colour by nonlinearity/oscillation, click-to-inspect
  program + response curve, honest-checks panel, probe table); `app.py` radial
  block replaced by 3 endpoints `/api/radial/{map,lens,probe}`. VERIFIED (CLI):
  determinism err 0.0, radius-vs-nonlinearity corr +0.47/+0.49, and the user's
  linear hypothesis holds — raw-x linear R2 ~0 on square/abs/ripple vs
  **lens-bank linear R2 = 1.000 on all five nonlinear targets** (heldout 60/40,
  loops AND noise streams): the lens diversity alone does the heavy lifting.
  `prog_str` kept ASCII (cp1252 console-safe). Data-kind hook ready for images/
  text baselines next. **Needs a Flask restart to serve the new routes.**
- **[2026-07-13] (Claude)** — **No-gradient CIFAR push** (branch
  `radial-cifar-nogradient`, autonomous). Goal: beat CIFAR-10 with zero gradient
  descent — deterministic/closed-form features + closed-form ridge classifier.
  `radial_cifar.py` harness (reads radial-owned CIFAR copy only). Milestone ladder
  on 8k train / 2k test: raw pixels ridge 0.324, PCA-256 ridge 0.360, **patch
  features (Coates-Ng: random-patch k-means dictionary + ZCA whiten + soft-
  threshold + 2×2 pool) ridge 0.493.** kNN → ridge alone was a big lift (raw 0.29→
  0.32, PCA 0.18→0.36). Next: batch patch extraction for full 50k data, bigger
  dictionary, and the lens map on top of patch features. Milestone 1 committed.
  **Milestone 2:** built radial-owned full CIFAR copy (`cifar_full.npz` 50k/10k),
  batched patch extraction (memory-safe). **Full 50k, patch D=512 pool2 + ridge =
  0.5904** — from 0.49 at 8k. Added feature caching + a `ladder()` that layers the
  lens map (nonlinear multi-axis/composed-activation combos) on top of patch
  features. Running D=512/1024 + lens expansion next.
- **[2026-07-12] (Claude)** — **Live lens-map explorer on `/radial` ("Lens Map"
  mode).** Made radial self-contained: `radial_data/cifar_radial.npz` is a
  radial-OWNED 28MB CIFAR subset (8k train / 2k test) copied once from the cifar
  project; radial reads only its own npz at runtime (CIFAR project untouched, per
  user). `radial_lensmap.py` runs the sweep in a background thread and STREAMS
  each lens as it's computed — no simulation. Each lens = `act(cos θ·axis_i +
  sin θ·axis_j)`; per lens it reports a label-free structure score (|excess
  kurtosis|) and an eval-only class-separation score. Endpoints `/api/radial/
  lensmap/{start,poll}`. New page mode: a heatmap grid that fills in live (cells
  = lenses, colour = structure or class-sep, toggle), the real CIFAR images the
  best lens fires most/least on (drawn from the npz), and a live encoder-accuracy
  curve (PCA over the whole explored bank) climbing past the red PCA baseline.
  Honest behaviour observed: the curve rises, **peaks ~0.28 above the 0.26 PCA
  baseline**, then declines as the sweep wanders into weak-axis noise — the map
  has a sweet spot. Selecting lenses by the label-free score alone fails (kurtosis
  picks spiky/outlier lenses); the gain comes from PCA over the collective bank,
  which is what the encoder uses. Needs Flask restart to serve the routes.
- **[2026-07-12] (Claude)** — **Lens map — UPDATE: it works on CIFAR** (corrects
  the MNIST-only negative below). User's insight: it's a MAP, not a search — the
  space is deterministic, so you explore it ONCE and reuse it; no compass needed.
  And MNIST was the wrong test (PCA already 0.92, no headroom). `radial_lens.py`
  `run_cifar()`: reduce CIFAR to structured PCA axes, then the lens map is a
  rotation sweep — each lens `act(cos θ·axis_i + sin θ·axis_j)`, literally the
  radial "1° rotation → activation" on feature space; enumerate the whole grid
  once (~2400 lenses). **Result (label-free, kNN, equal dims): linear PCA 0.225,
  lens map 0.254 (+2.9 pts), and it compounds with PCA.** Bigger sweeps gave up to
  +5.3. The gain grows as more of the map is swept — real nonlinear territory PCA
  can't reach, found by systematic exploration with no labels and no search. Not a
  finished encoder (CIFAR kNN is low), but the map framing is validated on the
  dataset that has headroom. CIFAR data READ ONLY; nothing written to that project.
- **[2026-07-12] (Claude)** — Tested the **"lens flip"** idea (`radial_lens.py`):
  pin the ground-truth images, vary the MODEL — each radial coordinate is a
  deterministic activation-algebra lens `act_a(P_i·x)·act_b(P_j·x)`; build a
  label-free encoder by greedily NAVIGATING to lenses orthogonal to the frozen
  stack. Tested honestly on MNIST (label-free build, kNN eval). **Result: it does
  not work.** Greedy orthogonal selection **0.548** — worse than random lenses
  0.750 and far below plain linear PCA **0.920**; full lens bank + PCA 0.899
  (still < PCA); lens algebra on PCA coords +0.002 (noise). Two data-independent
  reasons: (1) **orthogonality ≠ information** — the most orthogonal high-variance
  direction is noise, so navigating by orthogonality walks toward noise; (2)
  every lens is a **closed-form function of the pixels**, so it only reaches
  structure computable from the numbers (the shallow half) — the configural
  coherence that separates classes lives on no parameterizable lens axis, exactly
  what the source analysis itself suspected. Standalone experiment; no app/CIFAR
  changes (CIFAR read-only per user). Constructive next step is the coherence /
  real-vs-fake discriminator the analysis landed on, not the lens sweep.
- **[2026-07-12] (Claude)** — `/xray` real-genome overlay added (`genome_overlay`
  in `genome_xray.py`, map toggle in the page). Loads **114 real solved genomes**
  — 45 MNIST pairs, 45 CIFAR pairs, 10 CIFAR detectors (`demo/*_genomes.pkl`) +
  14 WordPipe language classifiers (val_acc from genomes.txt) — and places each
  by a pure task-behaviour fingerprint [skill over chance, standing vs own kind],
  standardised, no domain metadata fed in. **Result: families separate 2.25×**
  (same-kind nearer than different-kind) — so real genomes DO cluster by kind on
  a behavioural map. **Honesty check: 89% of that separation is the competence
  axis alone** (mnist 0.98 > cifar-pair 0.83 > cifar-det 0.73 > language 0.67).
  So the split is real but shallow — the deeper per-class structure that made the
  activation map cluster cleanly needs behaviour the language genomes don't store
  (headline accuracy only). Same method, thinner data. Map toggles between
  "Activations · sanity check" and "Real genomes" views.
- **[2026-07-12] (Claude)** — CIFAR encoder **v7: REAL-VS-FAKE coherence question
  -- right question, substrate can't hold the answer**. Built the "does this
  belong" discriminator with a NATURAL (not systematic) corruption: fake = a
  feathered/alpha-blended patch swapped in from a different image, so there is no
  seam to detect -- pixels all real, only the CONTENT is out of place; flagging
  it requires modeling coherence (`_patch_swap_fake`, mode "realfake" reusing the
  arrange hard-negative fitness; job `cifar_l2_realfake.py`). Result (full 10k):
  standalone L1+realfake 0.3373 (+1.2 sigma) -- the WEAKEST positive question
  (vs warp +2.9, color +2.4, arrange +2.1, crop +1.6). Composed: adds no
  orthogonal axis; champion stays L1+warp+color 0.3655 (6-stack 0.3668 is 0.3
  sigma = noise, and needs the negative `occlude` to get there). One real flicker:
  realfake standalone separated bird/deer to 0.828 (best of any single question)
  -- coherence grabbed something real about that confusion but too little to
  matter. INTERPRETATION (confirms the chat with the user): coherence/relational
  ("patterns within the numbers") is the RIGHT KIND of question, but the frozen
  16-ch/14x14 substrate can barely represent the answer -- to judge coherence you
  must compare distant regions, but L1 gives 16 coarse channels and L2 pools
  spatially again, so relational detail is gone by the code. FOUR independent
  lines now converge: (1) read-head -> ceiling is the retina; (2) composition ->
  only 2 orthogonal axes fit; (3) arrange -> layout redundant; (4) realfake ->
  coherence unanswerable here. The substrate is saturated at ~2 axes (photometric/
  geometric); relational axes need a RICHER substrate. Artifact
  `demo/cifar_l2_realfake.pkl`; log `demo/cifar_l2_realfake.log`.
- **[2026-07-12] (Claude)** — CIFAR encoder **v6: ARRANGEMENT question (real-vs-
  shuffled layout) -- answerable but REDUNDANT; substrate holds ~2 orthogonal
  axes**. Added the part-composition question as a real-vs-shuffled discriminator
  (same trick as WordPipe grammar): positives = 2 augs of the real image, plus
  each anchor's QUADRANT-SHUFFLED self as a hard negative (identical local
  content, wrong layout) -> the code must encode arrangement to push it away
  (`_shuffle_quadrants`, `ContrastiveL2GPU.arrange`/`_emt`/`set_shuf`, mode
  "arrange"; job `cifar_l2_arrange.py`). Result (full 10k): standalone L1+arrange
  0.3418 (+2.1 sigma) -- the layout question IS answerable and pays out alone.
  BUT composed onto the champion it DILUTES: L1+warp+color 0.3655 -> +arrange
  0.3621. Layout at this substrate is entangled with appearance/shape, not a
  separate axis. Champion stays **L1+warp+color = 0.3655 (+7.2 sigma)**. Refined
  law: findable is necessary but compounding needs findable AND ORTHOGONAL, and
  orthogonality is SCARCE -- of 5 answerable questions (warp/color/crop/arrange/
  decorr all +sigma alone) only warp+color are mutually orthogonal. Measurement:
  the 16-ch/14x14 frozen L1 substrate holds ~2 orthogonal answerable axes (shape
  via colour-invariance, appearance via shape-invariance); everything else lands
  in their span -> evidence that a THIRD axis needs a richer SUBSTRATE, not
  cleverer questions. Artifact `demo/cifar_l2_arrange.pkl`; log
  `demo/cifar_l2_arrange.log`.
- **[2026-07-12] (Claude)** — CIFAR encoder **v5: FITNESS-AS-QUESTION -- composing
  answerable questions MOVES the ceiling (0.332 -> 0.366, +7 sigma)**. Reframe
  (user's cross-project law, now recorded in memory `fitness-as-answerable-
  question`): a layer's fitness is a QUESTION whose answer must be findable in
  the data; too easy = redundant (v4), too hard/ill-posed = no gain, the win is
  answerable-but-unanswered. Gave L2 DIFFERENT fitnesses than L1 (all label-free,
  frozen L1) -- first a 4-way fitness sweep (`evolve_l2`; decorr/hardneg/color/
  swav), then a battery of augmentation-QUESTIONS where the question = what the
  view varies (`_augment` kinds color/crop/occlude/warp; jobs
  `cifar_l2_variants.py`, `cifar_l2_battery.py`). Per-question concat gain (full
  10k test, L1 kNN 0.3318, SE 0.0047): **warp +2.9 sigma** (vary shape -> learn
  appearance/texture; cracked bird/deer 0.775), **color +2.4 sigma** (vary colour
  -> learn shape; cracked cat/dog), crop +1.6, decorr +2.1, hardneg +1.3, occlude
  -1.2 (unanswerable), swav +0.2 (too hard, collapsed). COMPOSITION
  (`scratchpad/compose_battery.py`): **L1+warp+color = 0.3655, +7.2 sigma, SUPER-
  additive** (warp 0.0138 + color 0.0111 alone -> 0.0337 together). Adding crop
  DILUTES (redundant with warp) and occlude DRAGS (unanswerable adds noise) --
  exactly the law: stack questions whose answers are findable AND non-redundant.
  First real movement of the retina ceiling in the whole arc, via composition of
  answerable questions -- not capacity (v3), depth (v4/v5-stack), or ensembling.
  Artifacts `demo/cifar_l2_{color,crop,occlude,warp,decorr,hardneg,swav}.pkl`;
  logs `demo/cifar_l2_{variants,battery}.log`.
- **[2026-07-12] (Claude)** — CIFAR encoder **v4: FREEZE-AND-STACK layer 2 -- did
  NOT raise the ceiling (honest negative)**. Froze seed-7 encoder's conv filters
  as layer 1, took its activated feature maps before the collapsing pool (16 ch,
  avg-pooled to 14x14), evolved a SECOND conv layer (32x 3x3x16 filters =
  edges-of-edges) on the frozen maps -> d=16 code, same NT-Xent contrastive
  objective, zero labels (`ContrastiveL2GPU` / `evolve_encoder_l2` /
  `_frozen_l1_maps`; reuses `evo_gpu.l1_maps`; job `jobs/cifar_encoder_l2.py`).
  This is the freeze-and-stack discipline v5 lacked (v5 evolved both layers
  jointly from random and stalled; GENREG rules VI/X). Result (pop 48, 2500
  gens): **L2 kNN 0.3157 vs L1 0.3340** -- slightly BELOW, no gain. Verified
  reproduces exactly; layer 1 confirmed frozen. Coarse structure preserved
  (10/10 supergroup-clean) and it DID separate some fine pairs (deer/horse 0.44,
  bird/frog 0.59) but kept the confusable ones (cat/dog 0.94, bird/deer 0.92) and
  lost net kNN. HONEST DIAGNOSIS of why depth-via-freeze-stack failed here: (1)
  both encoders bottleneck to d=16 and kNN is measured on that code -- the
  ceiling is at the bottleneck, downstream of the added depth; (2) the coarse
  augmentations (flip/shift/occlusion/brightness) set what is learnable at ANY
  depth; (3) we froze a TERMINAL 16-filter encoder tuned to feed its own readout,
  not a wide un-collapsed SUBSTRATE -- greedy layerwise stacking needs a wide
  early layer (64+ ch, no tight code), so we stacked on an endpoint. Implication:
  the lever is the layer-1 SUBSTRATE design + the objective/augmentations, not
  more depth on a collapsed code. Artifact `demo/cifar_encoder_l2.pkl`; log
  `demo/cifar_encoder_l2.log`.
- **[2026-07-12] (Claude)** — CIFAR encoder **v3: READ-HEAD on the frozen retina
  (first labels enter the pipeline)**. Froze the seed-7 encoder (kNN 0.3340) as
  fixed infrastructure, encoded every image through it ONCE, and evolved a small
  read-head genome on the cached 16-d codes -> 10-class, soft cross-entropy on
  labeled data (`evolve_readhead` / `_head_logits`; job `jobs/cifar_readhead.py`;
  first time labels touch anything). Results (pop 160, 2000 gens): **linear probe
  16->10 test 0.2848**; **evolved MLP head 16->64->10 test 0.3380** (verified
  reproduces exactly). Reference: chance 0.10, label-free kNN 0.3340. Reading:
  (a) the nonlinear head (0.338) matches/slightly beats the kNN geometry, while
  (b) the LINEAR probe (0.285) is BELOW kNN -- the 16-d code is locally
  structured but not cleanly linearly separable, so a nonlinear head is needed to
  recover what the label-free geometry already contains. Key point: labels in the
  tiny head recover the representation's quality but do NOT exceed it -- the 16-d
  label-free bottleneck is the ceiling, set entirely without labels; the retina,
  not the head, is the limiting factor. Best per-class: plane 0.47, ship 0.43,
  car 0.42, horse 0.42, frog 0.47 (vehicles + distinctive animals lead; bird 0.09
  lags). Artifacts `demo/cifar_readhead_{linear,mlp}.pkl`; log
  `demo/cifar_readhead.log`.
- **[2026-07-12] (Claude)** — CIFAR encoder **v2: two private languages AGREE +
  scaling + reading the code** (`jobs/cifar_encoder_multi.py`,
  `scratchpad/analyze_encoders.py`). Ran all three follow-ups on the label-free
  contrastive encoder. **(1) Agreement:** a second encoder evolved from a
  different seed (101, kNN 0.3267) vs the first (seed 7, kNN 0.3340) -- both
  10/10 supergroup-clean, and their class-similarity matrices correlate
  **Pearson r=0.977 / Spearman rho=0.958**, with per-image RSA rho=0.755. Both
  independently rank the same top pairs (plane~ship, cat~dog, bird~deer,
  car~truck). CONCLUSION: the emergent semantic geometry is REAL, not a per-run
  artifact -- two random seeds converge on the same structure. **(2) Scaling**
  (d=32, M=24, V=6, hard aug via new `hard_aug`/`_augment` strength): kNN 0.3310
  -- NOT better than base d=16 0.3340 (supergroup-clean dropped 10->7), though
  harder aug sharpened FINE sub-structure (intra-animal cat/dog 0.945 >>
  cat/frog 0.756). Scale is not the lever for coarse structure (consistent with
  the classifier line). **(3) Reading the language:** the 16-d code is
  interpretable -- dims split into an organic<->manmade basis: dims 2/5 fire +on
  animals -on vehicles; dims 12/6/3/4/14 +on vehicles -on animals; others encode
  finer animal axes. `evolve_encoder` now takes `out`/`hard_aug`/`seed` in
  payload. Artifacts `demo/cifar_encoder_{seed7,seed101,scaled}.pkl`; log
  `demo/cifar_encoder_multi.log`.
- **[2026-07-12] (Claude)** — CIFAR **PIVOT: evolve label-free ENCODERS with a
  private language (contrastive), not class separators**. User redirected the
  project: stop training genomes to separate classes; evolve encoders that
  capture similarity/difference structure. New `ContrastiveEncoderGPU` /
  `evolve_encoder` / `_augment` / `_knn_acc` in `cifar_internal.py`; job
  `jobs/cifar_encoder_v1.py`. One encoder genome (M=16 conv filters + per-filter
  activation + pooling -> d=16 UNIT-NORM latent code = its private language),
  fitness = NT-Xent / SimCLR (positives = two augmentations of the same image:
  flip/shift/occlusion/brightness; negatives = other images), gradient-free via
  ga_step. THE ENCODER NEVER SEES A LABEL. Result (pop 48, 2500 gens): positive-
  retrieval 0.27 -> 0.74 (augment-invariance learned); **kNN top-1 in the code =
  0.3340 (chance 0.10)** with labels used ONLY at eval. EMERGENCE: the private
  language split CIFAR-10 into vehicles {plane,car,ship,truck} and animals
  {bird,cat,deer,dog,frog,horse} -- ALL 10/10 classes have their top-3 nearest
  neighbors inside the correct supergroup (car~truck~ship~plane, cat~dog,
  deer~horse), discovered with ZERO labels. Independently verified: kNN reloads
  to 0.3340 exactly; fitness confirmed label-free. Artifact
  `demo/cifar_encoder.pkl` (champ + sim_matrix); log `demo/cifar_encoder_v1.log`.
  This is the direction now -- see also the label-free diversity/occlusion work
  and the I2 evolved-genome-encoder vision. Next: bigger d, more views, compare
  two independently-evolved private languages (do they agree on similarity?).
- **[2026-07-12] (Claude)** — CIFAR internal-language **v5: TWO-LAYER decider
  (filters-of-filters) + full VERIFICATION SWEEP**. Added
  `SingleGenome2LayerBinaryGPU` / `evolve_single2` / `evolve_arbiter2` /
  `champ_logits2`: one genome now carries TWO evolved conv banks -- L1 (8x 5x5x3
  texture filters, maps avg-pooled 28->14) and L2 (8x 3x3x8 filters combining L1
  channels into PART detectors) -- plus both activation sets and the readout,
  evolved jointly. Job `jobs/cifar_arbiter2_catdog.py`. Result (cat vs dog, L1=
  L2=8, pop 48, 1500 gens): 2-layer decider A 0.6315 / B 0.6700 / avg-ensemble
  0.6705 / checker 0.6730 -- **the deeper vocabulary does NOT beat the 1-layer
  0.6705 ceiling** (A even overfits: val 0.6714 -> test 0.6315). Honest caveat:
  both conv layers were evolved jointly FROM RANDOM INIT, which GENREG_RULES VI/X
  explicitly warn is a rough landscape ("cannot be evolved from random init";
  bootstrap/freeze-and-compose instead) -- a two-phase L1-then-L2 run is the
  fairer test and the logical next step. Seed-consistency held again (agree 71.9%
  @ 0.71, disagree @ ~coin-flip). CONVERGING EVIDENCE across v3/v4/v5: the ~0.67
  cat-dog wall is NOT capacity (v3), NOT ensembling/verification (v4), NOT naive
  depth (v5) -- it is a representation/landscape limit of gradient-free shallow
  conv on 32x32 whitened patches. **Full independent verification sweep**
  (`scratchpad/verify_all.py`): every saved champion (v1 0.9215, v3 0.6825, v4
  A/B/checker + agree-split, v5) reloads and re-evaluates to the reported number
  within tolerance; adversarial checks pass (deciders genuinely differ
  ||dK||=9.11; checker sees only decider logits, no label leak; train/test
  disjoint). ALL CHECKS PASS. Artifacts `demo/cifar_internal2.pkl`,
  `demo/cifar_arbiter2.pkl`; log `demo/cifar_arbiter2_catdog.log`.
- **[2026-07-12] (Claude)** — CIFAR internal-language **v4: ARBITER + CHECKER
  genome (self-verification via seed-consistency)**. Two decider genomes evolved
  from different seeds (7, 101) = "the same model from a different seed", plus a
  CHECKER genome (one evolved hidden layer, per-unit activation from the
  8-catalog) that reads both deciders' logits AND their disagreement |sA-sB| and
  verifies the answer before outputting. New `evolve_arbiter` / `evolve_checker`
  / `champ_logits` in `cifar_internal.py`; job `jobs/cifar_arbiter_catdog.py`.
  Results (cat vs dog, dec_gens 2000, checker_gens 1500): decider A 0.6715 / B
  0.6680 / average ensemble 0.6790 / **checker 0.6775** (vs single-decider
  0.6705 baseline) -- ensemble buys +0.7-0.8 pts, small because the two seeds'
  errors are correlated. THE REAL FINDING is the seed-consistency verifier: the
  two seeds AGREE on 72.5% of test images and are 73.4% accurate there
  (confident cases); they DISAGREE on 27.5% where a single decider scores 0.5064
  -- a literal coin flip (the answer changes with the seed) -- and the checker
  manages only 0.5281 there. Verdict: seed-disagreement is a VALID calibrated
  uncertainty detector (73% vs 51%), exactly as hypothesised, but it cannot
  manufacture signal that is not in the decider outputs -- ~27% of cat/dog
  images are genuinely unresolvable at 5x5 local-texture resolution. Re-confirms
  v3 from a new angle: the ceiling is a feature/landscape wall, not capacity or
  ensembling. Also hardened logs to ASCII + PYTHONIOENCODING=utf-8 to avoid the
  cp1252 UnicodeEncodeError crash class. Saved `demo/cifar_arbiter.pkl`; log
  `demo/cifar_arbiter_catdog.log`.
- **[2026-07-11] (Claude)** — CIFAR internal-language **v3: genomes now EVOLVE
  THEIR OWN HIDDEN DIM (masked capacity) — width is not the bottleneck**. Added
  `evolve_single_masked` + per-filter soft gate gene `g=sigmoid(gate)`; fitness
  pays an annealed cost proportional to `sum(g)` (warm-up 30% so filters
  specialise before pruning). Genome carries M_max=24 filters and evolves how
  many it keeps. Jobs `jobs/cifar_internal_masked_{carbird,catdog}.py`. Results
  (pop 64, 2500 gens, cost_lambda=0.015): car-vs-bird test **0.9185** (vs fixed
  M=8 0.9215), cat-vs-dog test **0.6825** (vs fixed M=8 0.6705) — both within
  noise. The gate mechanism WORKS (gates spread 0.5-1.0, genome modulates
  capacity) but NEITHER pair prunes: both keep all 24 filters "on" (many weakly
  at 0.50-0.55) because each marginal filter still beats its tiny cost. Verdict:
  letting the genome choose its width does not change the result — **capacity is
  not the lever, empirically confirming GENREG_RULES #2** (the cat-dog wall is a
  landscape/feature problem, not a hidden-dim problem). cost_lambda=0.015 too
  weak to force real sparsity; a lambda-sweep would trace the accuracy/width
  tradeoff (how few filters the internal language actually needs) but won't lift
  the ceiling. Champions `demo/cifar_internal_masked_{carbird,catdog}.pkl`; logs
  `demo/cifar_internal_masked_*.log`.
- **[2026-07-11] (Claude)** — CIFAR internal-language **v2: the HARD pair, cat vs
  dog — single genome reaches 67.05% test** (job `jobs/cifar_internal_v2.py`).
  Same single-genome architecture as v1, no changes but the class pair (pos=3
  cat, neg=5 dog). Result (pop 64, 2500 gens): val 0.6814 / **test 0.6705** vs
  0.5000 baseline (+17 pts); val->test drop 1.6% (generalizes). Verdict: the
  internal-language architecture HOLDS on visually-entangled classes but the
  ceiling is real — predictions are fuzzy (0.4-0.6, no confident collapse), the
  honest signature of two animal classes seen through 8 evolved filters with no
  gradients. Evolved a more PERIODIC lens set [square,cos,relu,cos,relu,gaussian,
  cos,leaky] (texture over color/shape). Champion `demo/cifar_internal.pkl`
  (v1 car-vs-bird champion preserved at `demo/cifar_internal_carbird.pkl`); log
  `demo/cifar_internal_v2.log`. Next levers to lift the hard-pair ceiling: more
  filters (M=16/32), a 2-layer internal vocabulary, or larger train pool.
- **[2026-07-11] (Claude)** — CIFAR **"internal language": one single end-to-end
  genome, everything evolved, 92.15% test on automobile-vs-bird**. New module
  `genreg_train/cifar_internal.py` + job `jobs/cifar_internal_v1.py`. The
  departure from the bank->features->classifier stack: ONE genome does the whole
  job — its own M=8 evolved 5x5x3 kernels, a per-filter activation from the
  8-catalog, multi-shape mean pooling (the pooled responses ARE its private
  "internal language"), and a linear readout to a single sigmoid score. Nothing
  pre-built: no PCA, no Fisher pre-selection, no logistic regression. Evolved
  jointly, gradient-free, by `mp.ga_step` (tournament + elitism + starvation
  homeostasis + self-adaptive sigma, mag-scaled), soft BCE fitness on balanced
  two-class minibatches, ZCA-whitened patches, champion on a held-out val split.
  Result (pop 64, 2500 gens, GPU): val 0.9210 / **test 0.9215** vs 0.5000
  majority baseline; val->test drop -0.1% (no overfit); champion from gen 2000.
  Genome evolved a HETEROGENEOUS lens set [gaussian,abs,leaky,relu,square,abs,
  sin,cos] — not one repeated activation. Predictions inspected: cars 0.93-1.0,
  birds 0.00-0.04. All three GENREG_RULES verification checks pass. Champion
  saved `demo/cifar_internal.pkl`; log `demo/cifar_internal_v1.log`. Next levers:
  harder pairs (cat vs dog), more filters, or a per-genome 2-layer internal
  vocabulary.
- **[2026-07-09] (Claude)** — CIFAR **encoder-objective study complete** (v9 + v10).
  v9 utility-only greedy encoder (user directive round 1: keep a genome only if it
  increases fitness; fitness = marginal nearest-centroid gain, incremental GPU state):
  gate seized at 41 genomes, ceiling **50.9** — greedy selection against a saturating
  proxy optimises the proxy, not the environment. v10 HYBRID (occlusion-contrastive
  generation x dense-margin utility gate, combined fitness = contrastive x
  sigmoid(50*utility), strict ug>0 admission): 512 genomes / 513 rounds, ceiling
  **65.68 test = occlusion-only's 65.56 from 33% fewer genomes**. FULL TABLE (test
  ceilings): utility-only 50.9 | entropy-novelty 58.4 | +jitter 58.9 | +L2-hierarchy
  59.2 | occlusion-contrastive 65.6 | hybrid 65.7 | label-driven Fisher 67.0.
  CONVERGED CONCLUSION: for 5x5-conv genomes the environment tops out ~65.6-67 and the
  OCCLUSION objective is what earns it; the utility gate compacts, labels add ~1.4.
  The open frontier is the CLASSIFIER CLIMB (pure stack reaches ~56 vs 65+ ceilings).
  New machinery: `UtilityFitGPU` (incremental centroid state, dense margin option),
  `evolve_detbank_utility`, `evolve_detbank_hybrid`, feats v9/v10. Banks:
  `cifar_detbank_v9.pkl`, `cifar_detbank_v10.pkl`.
- **[2026-07-09] (Claude)** — CIFAR v8 pure-stack + continuation stacking: **55.64 test**
  (fully pure system: label-free occlusion-contrastive environment + statistics-free
  evolved classifier). Battery: fold 45.0 -> joint 8k gens 52.98 -> referees gated off ->
  51.64 test; then 3 continuation rounds (8k gens x pop 240 each, evolving onward from the
  champion) -> 55.92 val / 55.64 test. THE HONEST CROSSROADS: the v8 environment has the
  best label-free ceiling (65.6) but a FLAT information spectrum — evolved linear heads
  climb it slowly (~10-pt optimisation gap remains; v5's spikier label-driven environment
  gave up its ceiling far more easily). Candidate next levers, undecided: (a) block-
  coordinate joint evolution (decompose 2048x10 into evolvable bites per the conversational
  directive's principle), (b) intermediate PCA scaling (sqrt-eig) between the unclimbable
  unit and the top-heavy eig, (c) narrower readout space (PCA-512: lower ceiling, much
  easier climb, possibly net positive), (d) brute continuation. Champions
  `demo/cifar_genomes.pkl`; scoreboard: fully-pure 55.64 | pure-classifier-on-labeled-env
  58.84 | label-driven-everything 58.84/67-ceiling.
- **[2026-07-09] (Claude)** — CIFAR **v8: OCCLUSION-CONTRASTIVE fitness (user's call:
  'try occlusion alone') — label-free ceiling 65.6 test, +6.7 over jitter-stability,
  within 1.4 of the label-driven 67.0.** Fitness = novelty x entropy x occlusion-SNR:
  the genome's response must survive a random 12x12 blank-out (a third of the image,
  mean-filled) while discriminating between images — invariance no single local
  measurement can satisfy, so survivors integrate global structure. Ladder now: 54.5
  (entropy) -> 58.4 (768) -> 58.9 (+jitter) -> 59.2 (+L2) -> **65.6 (occlusion)**.
  The purity gap is essentially closed by objective design, not labels. Bank:
  `demo/cifar_detbank_v6.pkl` (v8, occlusion; jitter bank backed up `_v6c.pkl`).
  Pure-stack classifier battery launched on v8.
- **[2026-07-09] (Claude)** — CIFAR v7 (layer-2 diversity genomes): ceiling **59.2 test**
  — composition added only +0.3 over single-layer v6c. LABEL-FREE LADDER FINAL: 54.5
  (256 bank) -> 58.4 (768) -> 58.9 (+stability) -> 59.2 (+512 layer-2 genomes over frozen
  L1; channel-selection genes + 3x3x8 kernels, novelty x entropy x stability, zero labels
  at every level) vs label-driven v5 67.0. HONEST VERDICT: pure novelty-driven evolution
  fills information space uniformly, and hierarchy multiplies the space faster than it
  concentrates class-relevant structure — the plateau is the objective, not the capacity
  (archive never saturated at any level). New machinery kept: `Layer2FitGPU`, `l1_maps`,
  `l2_features_gpu`, `evolve_detbank_l2`, `build_features(7)`, randomized-SVD PCA.
  PROPOSED NEXT OBJECTIVE (not yet built): augmentation-contrastive fitness — a genome
  survives by responding INVARIANTLY to strong augmentations of the same image (crop/
  flip/color-jitter) while DISCRIMINATING between different images (the SimCLR principle,
  label-free, evolution-compatible; our jitter-SNR term was a weak version). Population
  diversity stays as a second term. Banks: `demo/cifar_detbank_v7.pkl` (L2), `_v6.pkl`
  (L1, stability), `_v6b.pkl` (entropy-only).
- **[2026-07-09] (Claude)** — CIFAR v6c (stability term): ceiling 58.9 test (+0.5 over
  v6b's entropy-only). LABEL-FREE LADDER COMPLETE for single-layer genomes: 256-bank 54.5
  -> 768-bank 58.4 -> +stability 58.9, vs label-driven v5 67.0. VERDICT: with one conv
  layer, 768 diversity-evolved detectors support 8 points LESS than 51 label-guided ones —
  diversity fills information space uniformly; labels point at the class-relevant corner.
  Scale is nearly spent (3x genomes = +3.9; objective refinement = +0.5). NEXT (building):
  LAYER-2 diversity genomes — genomes whose inputs are the frozen layer-1 bank's response
  maps (channel-selection genes + 3x3 kernel + activation), same novelty x entropy x
  stability fitness, still zero labels: composition as the source of new information
  (genomes stacked on genomes in the environment itself).
- **[2026-07-09] (Claude)** — CIFAR **v6: the LABEL-FREE environment** (user directive:
  feature genomes evolve for DIVERSITY only — no labels/Fisher/niches/hand-stats/whitening
  in the environment; labels only ever reach classifier genomes; saved to memory as the
  standing rule). New machinery: `evo_gpu.DiversityFitGPU` (behavior = z-normed pooled
  response over an unlabeled probe set; fitness = novelty (1 - mean |cos| to k-nearest in
  population+archive) x information (16-bin response entropy)), `evolve_detbank_diverse`
  (novelty-search archive, fresh population per round, dup-capped admission),
  `build_features(6)` = bank outputs ONLY, GPU randomized-SVD PCA for >12k dims,
  auto-chunked bank_features (768-bank OOM fix). CEILING LADDER (test): 256-genome bank
  54.51 -> 768-genome 58.38 (+3.9 from 3x scale; archive NEVER rejected a candidate — the
  behavior space is unsaturated) vs label-driven v5 67.0. Gap to label-driven ~8.6.
  Running now: v6c = same 768 recipe + STABILITY term (novelty x entropy x jitter-SNR —
  response must be stable for the same image under 1px shift, varied across images; kills
  noise-measurers entropy admits; still zero labels). Banks: `demo/cifar_detbank_v6.pkl`
  (+ `_v6b.pkl` backup).
- **[2026-07-08] (Claude)** — CIFAR **round 2: 58.84% test with a PURE GENOME STACK** (user
  directive after the MNIST warm-start critique: no statistics anywhere in the classifier).
  Environment v5: ZCA patch whitening + NICHED bank evolution (55 niches = 45 class pairs +
  10 one-vs-rest, each with its own Fisher survival condition, round-robin decorrelated
  selection) -> 51 diverse detectors (vs 15 in round 1); regularised linear ceiling 58.83 ->
  **~67.0** test; unit-scale centroid floor 65.71. KEY DISCOVERY (cut + fixed with evidence):
  unit-variance standardisation of deep PCA tails makes the binary-genome landscape
  UNCLIMBABLE (chance at 2048 dims, 85% at top-256 — noise dims amplified to signal
  magnitude drown mutation-based search); fix = eigenvalue-proportional scaling
  (`build_features(pca_scale='eig')`), full ceiling kept, detector learnability restored
  (76% at 600 gens). Pure stack (`--pure --eig`): 10 detectors evolved from random (70-79%
  balanced) -> mixer genome -> algebraic fold (53.2 val) -> joint evolution +5.4 -> referees
  +0.9 test (margin 0.5 — they finally contribute, exactly as gate theory predicts for a
  below-ceiling head) = **58.84 test** (round 1 centroid-start: 58.69). Honest gap to
  ceiling ~8 points = the cost of purity at 8k gens. `_train_binary`/`train_detectors`/
  `train_pairwise` gained mag_scale; run_battery gained warm='fold'/pca_scale; champions
  `demo/cifar_genomes.pkl` (+ `_v5_pure.pkl` backup), bank `demo/cifar_detbank_v5.pkl`.
- **[2026-07-08] (Claude)** — **GPU backend + CIFAR round 1 (58.69% test in 11.5 min)**.
  `genreg_train/evo_gpu.py`: fitness evaluations on the RTX 4080 (torch no_grad, TF32 off,
  verified < 5e-7 vs CPU, 57x on the joint head), evolution stays numpy — wired into
  mnist_pipe (joint/binary) + cifar_pipe (detbank/features) with CPU fallback. CIFAR-10
  fetched via HuggingFace parquet (Toronto throttled), converted to standard batches. First
  campaign (`jobs/cifar_v4.py`): bank collapsed to 15 diverse detectors (diversity is the
  round-2 problem), centroid 57.09 -> joint **58.69** test; referees too weak, margin gate
  correctly chose 0. Details in `documentation/changelogs/CHANGELOG_MNIST.md`. MNIST v4
  battery still finishing on CPU (best val 99.24, ceiling 99.26).
- **[2026-07-08] (Claude)** — NEW PAGE (staged): **/cifar — CIFAR-Pipe**, the MNIST-Pipe
  program verbatim on CIFAR-10, per user instruction BUILT BUT NOT RUN (MNIST v4 battery has
  the machine). `genreg_train/cifar_pipe.py` (32x32-RGB data plumbing; classifier genomes
  imported unchanged from mnist_pipe), `cifar_service.py`, `templates/cifar.html`,
  `static/cifar.js`, `/cifar` + `/api/cifar/*`, nav entry; data in `corpora/cifar10/`.
  Run order when ready: `python -m genreg_train.cifar_pipe --detbank` then `--v4`.
  Meanwhile MNIST round 7 (v4, evolved-detector environment, ceiling 99.20): joint head
  relaunched from the CENTROID warm start (99.02 val at gen 1, vs 94.94 from the detector
  fold) — best val 99.12 and climbing. Needs the same pending Flask restart as /mnist.
