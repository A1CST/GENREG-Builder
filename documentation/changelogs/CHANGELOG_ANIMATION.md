# Changelog — Animation

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.
Seeded 2026-07-05 from the main changelog (keyword split, best effort).

---

- **[2026-07-05] (Claude)** — Animation Evo: **per-clip presentation regime** (user: the genome
  must NOT see all 240 frames at once). Now ONE clip per generation — the population is scored
  only on a single clip's 24 frames, presented in temporal order (shape moving along its path);
  clip order is shuffled and reshuffled each epoch. Fitness = accuracy on just that clip
  (stochastic minibatch-of-one-clip, same idea as DiffEvo's fresh-minibatch fitness). No
  240-frame batch anywhere in the training loop; the final champion accuracy + confusion /
  per-clip / encoder-PCA report is computed clip-by-clip (ten 24-frame passes) at the end only.
  Training metrics: noisy per-clip fitness + rolling epoch average. Verified baseline (same
  basic model): plateaus fast — champion 0.179 across all clips at gen 107 (~7s); rolling avg
  oscillates ~0.25 because the population chases whichever shape the current clip shows, so no
  generalist emerges under pure per-clip selection. Honest expected result, left as-is. GUI:
  tiles (this-clip / rolling / pop-mean / champion-all-clips), current-clip highlight on the
  grid, chart plots per-clip fitness + rolling avg. Report: `animations/evo_report.png`.

- **[2026-07-05] (Claude)** — Animation GUI: grid captions now label each clip by its **shape
  name** (circle, plus, crescent, …) — bold, primary — with the motion path as secondary text
  ("· wave path"); previously only the path name showed. `/api/animations` now returns each
  clip's `shape`. Live after the user's next Flask restart.

- **[2026-07-05] (Claude)** — Animation: **every clip now has a UNIQUE shape** (user: no
  duplicate shapes across animations). Five new rasterizers in `animation_data.py` (plus,
  xcross, hexagon, crescent, frame/hollow-square); mapping: line→circle, diagonal→square,
  swoop→triangle, loop→ring, figure8→diamond, zigzag→plus, wave→xcross, spiral→hexagon,
  bounce→crescent, scurve→frame. `animations/` regenerated + verified visually. Animation Evo
  becomes a balanced 10-class task automatically; palettes extended to 10 and the report PNG
  re-laid-out for a 10×10 confusion matrix. New baseline (same basic model, accuracy fitness,
  mutation only): 0.358 at max_gens 500 (~3.4 min), still creeping — not yet a hard plateau;
  best clips line 0.92 / swoop 0.67 / bounce 0.63, worst figure8 0.0. Report:
  `animations/evo_report.png`.

- **[2026-07-05] (Claude)** — Animation: **Animation Evo — deliberately basic mutation-only shape
  classifier** (`genreg_train/animation_evo.py`). Genome: frame(4096) → tanh(enc 12) →
  tanh(hid 24) → shape logits (5 classes); ~49.6k params. Evolution is mutation ONLY (no
  crossover): elites kept, children = single elite parent + fixed-step Gaussian noise. Fitness
  is exactly the stated goal — fraction of frames whose shape is named correctly, despite the
  movement. Kept intentionally basic per the user (an earlier balanced-CE + self-adaptive
  variant that reached 100% was REVERTED — the user wants the honest baseline; it plateaus
  around 0.62, mostly the majority class). Report: `animations/evo_report.png`. Reused JobHub
  from diffuse_service (now parameterised by program name); runs persist to `runs/animevo/`.
  New WS `/animevo` + full /animation page UI (config sidebar, accuracy chart, live per-clip
  predictions on the playing dataset grid, final confusion/PCA/per-clip report) — live after
  the user's next Flask restart. Headless anytime: `python -m genreg_train.animation_evo`.

- **[2026-07-04] (Claude)** — Animation: **10-clip procedural dataset** (`genreg_train/animation_data.py`).
  Each clip is 24 frames of 64×64 grayscale, black background, one white shape moving on a
  unique path: line, diagonal, swoop, loop, figure-eight, zigzag, wave, spiral, bounce, S-curve
  (shapes vary: circle/square/diamond/ring/triangle; anti-aliased sub-pixel motion).
  `python -m genreg_train.animation_data` renders `animations/<name>.npy` (uint8) + preview GIFs
  — already run, files are in `animations/`. Added `/api/animations` (base64-packed frames) and a
  playing preview grid on the /animation page (`static/animation.js`). Route + API need the
  user's next Flask restart; the .npy/.gif files are usable now.

- **[2026-07-04] (Claude)** — New **Animation** tab scaffold: `/animation` route + `templates/animation.html`.
  Page is intentionally mostly blank — just the topbar with cross-links and the shared
  terminal dock / agent panel / config panel scripts (same as the other program pages).
  Added "Animation ↗" nav links to the build, tree, diff, and runs pages.
  Flask server NOT restarted — a restart is needed for the new `/animation` route to be live.

