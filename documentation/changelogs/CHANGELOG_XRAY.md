# Changelog — XRAY

Per-project log for the XRAY line. Seeded 2026-07-14 from the master
CHANGELOG.md (all entries mentioning this project); new XRAY entries go at
the top of the log below, and also in the master CHANGELOG.md.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-12] (Claude)** — `/xray` page **reframed** to the user's corrected
  concept: NOT a genome↔address bijection (that claim dropped), but "is the
  deterministic rotation lattice usable as a MAP?" Mechanic per the user's spec:
  a fixed ground-truth dot cloud is rotated **1° per step about one fixed axis**;
  a function's coordinate = how it reads that whole 0–359° sweep (deterministic,
  so `sin` lands the same every pass — that's what makes it a map, no
  invertibility needed). `genome_xray.py` rewritten (`run_map`): 12 activation
  functions in 4 secret families (rectifier / saturating / oscillator /
  even-bowl) are placed on the map by their sweep-signature (correlation distance
  + classical MDS) with the families NEVER used for placement, only for scoring.
  Page rebuilt (`templates/xray.html`, `static/xray.js`): left = the 1°/step
  sweep animating, centre = the map (dots per function, coloured by family,
  hulls to centroid), right = verdict panel (family separation, NN purity, known
  orderings, determinism). Terminal dock added. **Result: map HOLDS** — cross-
  family pairs sit **2.97× farther** than same-family, **83% NN purity**,
  determinism error 0, 3/4 known orderings pass (the miss is real signal: `sin`
  is near-linear over the range so it drifts toward the rectifier/saturating
  region). Corrects the earlier "clustering is planted" verdict, which was an
  artifact of placing *random abstract rotations* with no data attached. Next
  step (user's plan): overlay the project's real solved genomes on the same map.
  **Needs a Flask restart to serve the new routes.**
- **[2026-07-12] (Claude)** — `/xray` **pivoted** away from the abstract maps
  (user: "the maps kinda suck and useless to me... one slice rotating for some
  reason"). The rotating slice and the MDS scatters are gone. New page: **watch a
  real genome pull tangled data into structure.** Ground truth = a balanced
  sample of real MNIST test digits; a solved genome (`demo/mnist_genomes_r5/r6/
  r2/pre_v4.pkl`, all feat_v2 so they share one starting cloud) runs the real
  `mnist_pipe.predict` forward pass; each point tweens from its raw-feature PCA
  position (tangled) to a confidence-weighted blend of ten digit anchors on a
  circle (separated). Colour = TRUE digit, so misclassifications land on the
  wrong-coloured corner and are visible. Controls: genome picker, mixer / pair-
  referee layer toggles, digits-per-class, scrub slider + replay.
  `genome_xray.py` rewritten (`transform`, `_ensure_ground_truth` caches the
  ~11s feature build; loaded genomes cached); endpoint swapped
  `/api/xray/run` → `/api/xray/transform`. r5 hits 97.8% on the sample; toggling
  mixer/pairs shifts confidence more than accuracy (detectors alone already
  strong on easy digits — honest). **Needs a Flask restart to serve the routes.**
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
