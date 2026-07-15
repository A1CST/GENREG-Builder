# Changelog — MNIST (specialist pipeline on images)

Project log for the /mnist page: the WordPipe/EvoLang decomposition recipe
applied to images, as a proof of the GA-abstraction thesis outside language.
Append new entries at the top.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-14] (Claude)** — **MNIST metrics wired into the new /tsdb page.**
  The frozen champions (`demo/mnist_genomes.pkl`) now feed a browser port of the
  `TSDB.js` Float64 block store as demo data: the pipeline layer accuracies
  (centroid → joint → +bias → +pairs → full) and all 45 one-vs-one specialist
  accuracies are serialized into TSDB blocks, read straight back out, and
  charted (incl. a 10×10 hard-pair map). Read-only — pulls from the pkl with no
  model load. See master CHANGELOG for the full store/port write-up.

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
- **[2026-07-08] (Claude)** — **PUBLISHED: github.com/A1CST/GENREG_MNIST_2.0 — headline
  99.10% ± 0.05% test over 5 full-pipeline seeds (7/11/23/42/101 -> 99.00/99.12/99.14/
  99.12/99.10; every seed re-evolves bank + head + referees; margin gate chose 0 on 4/5).**
  10-cell ablation battery on validation (test reserved for shipped config): evolved bank
  +1.42 over built-stats-only; evolved activation catalog +0.32 over relu-only; centroid
  warm start +16.02 over cold at equal budget (and +1.30 over detector-fold); mag-scaled
  mutation = most of the joint head's actual climb (+0.22 vs +0.04 global-sigma);
  full-train landscape +0.04/+0.14 over 16k pool/per-gen minibatch; L2 +0.08;
  **standard GA baseline 92.60 vs shipped 99.22** on identical environment/budget.
  Repo contents: `genreg_mnist/` (mnist_pipe + evo_gpu, standalone), scripts (download/
  train_full/evaluate/run_seeds/run_ablations/make_charts), checkpoints (bank + 99.03
  record head), results JSONs, 5 charts, docs (METHOD/FAILURES/LIMITATIONS), MIT.
  Cold-run reproducibility verified (clean-clone evaluate.py reproduces 99.03, 97 errors).
  Determinism verified (seed-7 GPU bank == record bank: 66 genomes, fisher 1.757).
  Local working copy: `C:\Users\paytonm\Documents\GENREG-MNIST-2`. Seed/ablation drivers
  also in GENREG repo as `jobs/mnist_seeds.py`, `jobs/mnist_ablations.py`; artifacts in
  `demo/seeds/`. mnist_pipe gained: GPU detbank fitness, `act_catalog=False` ablation
  flag, `evolve_detbank(out=)`, `build_features(v4_bank=, v4_cache=)`, GPU rotating-pool
  joint fitness.
- **[2026-07-08] (Claude)** — **MNIST v4 FINAL: 99.03% TEST — the 98.5% gradient-free
  record is broken and the 99% target hit.** Ladder: centroid floor 98.72 -> evolved
  joint head **99.03** (val 99.24 vs closed-form ceiling 99.26 — the GA captured
  essentially the entire evolved environment; val->test gap 0.21, 2% relative). The 45
  pairwise referees were gated OFF at margin 0.0 on validation — with the joint head at
  the feature ceiling, one-vs-one linear referees in the same space have nothing left to
  add (same verdict the gate reached on CIFAR; the gate is doing its job). Inference
  model: 66 evolved conv detector genomes (~5.0K genes) + one evolved 1024x10 joint head
  (10.25K genes) — ~15.3K evolved params, no gradients anywhere in features OR
  classifier. Battery 4665s on CPU (launched before the GPU backend; a rerun would take
  ~5 min). Champions `demo/mnist_genomes.pkl` (feat_version 4), record backup
  `demo/mnist_genomes_v4_9903.pkl`. `mnist_service` now loads features per the pickle's
  feat_version (pickle first, then features — was building v2 unconditionally).
  Comparison line the user tracks: MANTIS 98.55 (50K params, lstsq readout) -> this
  99.03 (15.3K evolved params, evolved readout); LeNet-5 99.05.
- **[2026-07-08] (Claude)** — **CIFAR-Pipe round 1: 58.69% test in 689s on the GPU** (user
  green-lit training once the GPU backend existed; MNIST v4 kept the CPU). Data via
  HuggingFace parquet (Toronto server throttled to ~5KB/s; converted to the standard batch
  format with PIL, `corpora/cifar10/cifar-10-batches-py/`). Ladder: centroid floor 57.09 ->
  joint 58.69 -> +pairwise 58.69 (margin gate chose 0.0 — CIFAR referees at ~74% pairwise
  val acc are too weak to referee; correctly disabled). KEY FINDING: the detector bank
  collapsed to **15 diverse genomes** from 1152 harvested (MNIST kept 66) — CIFAR
  populations keep rediscovering the same low-level edge/color detectors, so the
  ENVIRONMENT is the binding constraint. Round-2 levers: diversity pressure in the bank
  (niching/fitness sharing, per-class Fisher targets, multi-scale kernels 3x3/7x7, lower
  corr cap), more PCA dims, patch-normalised features. Champions:
  `demo/cifar_genomes.pkl`, bank `demo/cifar_detbank.pkl`, log `demo/cifar_train_v4.log`,
  job `jobs/cifar_v4.py`.
- **[2026-07-08] (Claude)** — **GPU backend built** (`genreg_train/evo_gpu.py`, RTX 4080
  detected, torch 2.6+cu124 already present). Pure arithmetic acceleration: evolution
  (ga_step/selection/mutation/gating) stays numpy-CPU; only FITNESS evaluations run on
  GPU under no_grad, TF32 disabled. Three paths: `JointFitGPU` (full-pool joint head),
  `BinaryFitGPU` (detector/pairwise pools uploaded once), `DetbankFitGPU` +
  `bank_features_gpu` (conv responses + activation catalog + crop-block pools + Fisher).
  Wired: `mnist_pipe.train_joint(gpu=)`, `_train_binary(gpu=)`, `cifar_pipe.evolve_detbank`
  / `bank_features` — all with automatic numpy fallback when CUDA absent. VERIFIED
  equivalent to CPU (max |diff| < 5e-7 on all three) and benchmarked: joint fitness at
  CIFAR scale (45k x 1024, pop 60) 0.62s -> 11ms/gen (**57x**). No behaviour change for
  running CPU jobs; future runs pick it up automatically.
- **[2026-07-08] (Claude)** — **CIFAR-10 staged (built, NOT run — per instruction MNIST
  trains first).** New program: `genreg_train/cifar_pipe.py` — the MNIST-Pipe pipeline
  verbatim on CIFAR-10; only the data plumbing differs (32x32 RGB: per-channel zones/
  profiles, luminance gradient histograms, 128-comp PCA; 5x5x3 = 75-gene conv detector
  kernels, 28x28 response map, same pool shapes/Fisher fitness/decorrelation; classifier
  genomes are LITERALLY mnist_pipe's — train_detectors/train_joint/train_pairwise/predict
  imported with D passed in). Entry points: `--detbank` then `--v4` (or bare = v2 battery).
  Also new: `genreg_train/cifar_service.py`, `templates/cifar.html`, `static/cifar.js`,
  `/cifar` + `/api/cifar/*` routes, nav entry. Data: `corpora/cifar10/`. Champions will go
  to `demo/cifar_genomes.pkl` / `demo/cifar_detbank.pkl`. Page shows environment stats +
  "staged" until a battery runs. Joint-head warm start = centroid head (same as MNIST v4).
  Also this entry: MNIST v4 battery relaunched with the CENTROID warm start after the
  detector-fold start opened at 94.94 vs centroid 98.72/99.02 — joint climbing, best val
  99.12 at gen 1500.
- **[2026-07-08] (Claude)** — **Round 7 (v4): the environment becomes EVOLVED — ceiling
  99.20.** MANTIS recipe (user's prior 98.55 project, github.com/A1CST/MANTIS_MNIST)
  adapted genome-pure: `evolve_detbank` breeds 5x5 conv-kernel genomes with EVOLVED
  per-neuron activations (8-function catalog — the GENREG signature primitive) on Fisher
  class-separability fitness (local survival condition; no end-to-end accuracy, no
  gradients, and lstsq is NOT used as the readout — our evolved joint head is). 48 seeded
  rounds x pop 48 x 60 gens, 1152 harvested, greedy correlation-capped selection on a
  common reference set -> **66-genome bank** (`demo/mnist_detbank.pkl`). Environment v4 =
  bank multi-shape mean pools (3x3/4x2/2x4, 25 dims/genome) + built 677 stats, PCA-1024,
  standardised, cached (`corpora/mnist/feats_v4.npz`). Gates: closed-form ceiling
  98.53 -> **99.20 test** (99.26 val); no-evolution centroid floor 88.93 -> **98.72** —
  the evolved features alone beat the user's 98.5 record before any classifier genome
  runs. Full v4 battery running (`--v4`): detectors -> joint (full-train landscape,
  mag-scaled mutation) -> referees -> margin gate. `evaluate`/`predict` now
  feat_version-aware; pre-v4 champions backed up (`_pre_v4.pkl`).
- **[2026-07-08] (Claude)** — **Round 6: 98.19% test at 1/3 the joint-head storage** —
  evolved per-neuron precision validated (user's two-lever directive: magnitude-scaled
  mutation + evolvable bit width per neuron). Accuracy statistically identical to r5
  (98.19 vs 98.21 = 2 images/10k) while the head quantised itself 12-bit-uniform ->
  [9,9,11,11,12,12,10,11,10,10] (~8.9 KB from 26.5 KB fp32); best val ROSE to 98.54.
  Hard digits (4, 5) kept the most bits — differential precision allocation as designed.
  Precision can evolve BOTH ways (symmetric mutation, cap 16, floor 3; bit_cost 3.1e-5/bit
  only wins ties) but init 12 biases exploration downward; a bits0=16 pure-squeeze variant
  is the obvious comparison run. Champions: `demo/mnist_genomes.pkl` (r6) + `_r6.pkl`
  backup; current BEST test remains r5's 98.21 (`_r5.pkl`). Service param tile now counts
  the joint head. Next compression target if wanted: the 45 pairwise referees (82% of
  params, still fp32).
- **[2026-07-08] (Claude)** — **Round 5: 98.21% test.** The optimisation gap closed in
  three steps: fixed-16k pool (r3: first real climb, but pool memorised — NLL -0.009 on
  pool vs -0.086 on full train), magnitude-scaled mutation (r4, user's lever: per-gene
  |w|-proportional steps + 5%-of-mean floor; lighter genomes, |W|^2 451->327), and the
  decisive full-55k deterministic landscape (r5, `--joint-pool 0`): val 98.24->98.50,
  TEST joint 97.83 / +pairwise **98.21** (margin re-gated to 1.5). Val->test gap 0.95
  -> 0.41. Joint fitness einsum -> GEMM (32x). Backups `_r2/_r5.pkl`. Round 6 running:
  evolved per-neuron precision (`JointQPop`, `--quant`) — 10 bit genes 3..16, quantised
  fitness, bit_cost 0.01; self-compressing champion saved with bits + effective KB.
- **[2026-07-08] (Claude)** — **Round-2 shipped (97.63% test); round-3 optimisation-gap
  campaign logged.** Round 2 (deskew): centroid 90.95 -> det 96.97 -> +mixer 97.06 ->
  +pairwise **97.63** test (val 98.52, margin 6.0). Ceiling probe (closed-form logistic,
  diagnostic only): same 677 features -> **98.53% test**, so the gap is optimisation.
  CUT: RFF lift v3 (noise-dominated, cold 95.98, warm decays); shift augmentation
  (harder task, 95.81 at equal gens). IN PROGRESS: joint refine — `fold_stack` folds
  det+mixer into one 677x10 genome, `train_joint` evolves it from the warm start
  (fixed-minibatch rotation, sigma_lo 5e-4, champion tracked on val = no regression
  possible). Plain + stabilised probes still drifted (train fitness up, val down, |W|
  growing); overnight verdict pending on L2 1e-4 probe (`demo/mnist_joint_probe.log`).
  New CLI: `--joint-only --joint-gens N --augment K`. Layer-4 idea if joint lands:
  re-gate pairwise on the refined head (margin grid extended to 12).
- **[2026-07-07] (Claude)** — **Round-1 results + round-2 (deskew) launched.**
  Round 1 test 10k: centroid floor 88.93 -> detectors(argmax) 95.57 -> +mixer 95.60 ->
  +pairwise **96.83** (val 97.68 at margin 3.0; drop within the held-out rule). Per-digit
  balanced val acc 96.2 (8) - 98.9 (1); the weak detectors are exactly the confusable
  digits (5, 8, 9), which the pairwise layer exists for (+1.23 on test). Mixer gate:
  top-1 ~flat vs argmax but val log-prob -0.39 -> -0.24 (calibration the margin gate
  uses). Battery time 1479s CPU. Champions backed up: `demo/mnist_genomes_r1.pkl`.
  ROUND 2 — enrich the environment, not the organism: `deskew()` (moment-based shear
  correction, unsupervised, vectorised bilinear remap) as statistics-layer v2;
  `build_features(version=...)`, `feat_version` stamped in the pickle. Deskew alone:
  centroid floor 88.93 -> 90.95. Full v2 battery -> `demo/mnist_train_r2.log`.
- **[2026-07-07] (Claude)** — NEW PROJECT: **MNIST-Pipe** — the specialist-pipeline
  recipe transposed to images (user pivot: prove the thesis outside language, target the
  99% range, only 10 outputs). Three layers, exactly the EvoLang structure:
  (1) STATISTICS layer, BUILT never evolved — 677 fixed dims from the training images'
  own statistics (4x4 + 7x7 zone ink, row/col profiles, 8-bin gradient-orientation
  histograms at 4x4 and 7x7 cells, 64 PCA comps of raw pixels; no labels).
  (2) SEMANTIC layer, evolved — 10 one-vs-rest detector genomes ("is this a 3?",
  linear head 678 params each, soft BCE fitness) + 45 one-vs-one pairwise
  disambiguators ("4 or 9?", trained only on their two digits).
  (3) OUTPUT layer, evolved — a 10x10 mixer genome over detector logits (soft
  log-softmax fitness); pairwise genomes referee the mixer's top-2 when the margin is
  small (margin tuned on val, never test). All gradient-free via the shared GA
  machinery (tournament + elitism + starvation + self-adaptive sigma). Baselines per
  GENREG_RULES VII: majority class 11.35%, nearest-centroid-in-stats-space floor
  88.93% (no evolution). Champions -> `demo/mnist_genomes.pkl`; data cached in
  `corpora/mnist/`; training entry `python -m genreg_train.mnist_pipe`.
  New files: `genreg_train/mnist_pipe.py`, `genreg_train/mnist_service.py`,
  `templates/mnist.html`, `static/mnist.js`; routes `/mnist` + `/api/mnist/*` in
  `app.py`; nav entry; styles in `style.css`. Page: layer toggles (Mixer / Pairwise),
  live test accuracy + confusion matrix + digit grid with a "Show mistakes" view.
