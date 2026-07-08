# Changelog — MNIST (specialist pipeline on images)

Project log for the /mnist page: the WordPipe/EvoLang decomposition recipe
applied to images, as a proof of the GA-abstraction thesis outside language.
Append new entries at the top.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

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
