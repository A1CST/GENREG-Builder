# Changelog — Animation

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.
Seeded 2026-07-05 from the main changelog (keyword split, best effort).

---

## 2026-07-15 (Claude) — Model-footprint panel (params + disk + CPU)

Top-of-page panel showing every model's evolved-parameter count, on-disk
checkpoint size, and CPU-inference / gradient-free status. `anim_footprint.py`
computes it from the checkpoints -> radial_data/anim_footprint.json; new
/api/animation/footprint route; hero stat-tiles + per-model table. The whole
page is 93,217 evolved params across 6 models (~2 MB on disk), CPU-only, no
backprop. Mirrored into the standalone GENREG-RADIAL demo (baked). Flask restart
needed.

## 2026-07-15 (Claude) — Run recording for the dot line (AGENTS.md 3 & 4)

The attention trainers weren't recording runs or alerting on completion. New
`dot_runs.py` writes the five-file run record into `runs/animation/<run-id>/`
(so attention runs appear on /runs) and posts a `kind=run` notice with the
run_id when a run ends. Wired into dot_track.run and dot_shape.main (now
accumulate per-round history). Verified with a real subset train
(run ...-animation-97dbba, 80 history rows, notice #401). Records gitignored
like other runs/; runstore auto-discovers the env.

## 2026-07-15 (Claude) — OOD stress-test module (tracker + classifier)

`dot_ood.py` pushes the frozen tracker + 10-class classifier out of their single
training regime along 10 axes (cached-basis inference, gradient-free). The
TRACKER degrades gracefully with clutter (3→6→10 distractors: 3.36→3.77→4.29px)
and is robust to shape scale, but leans on "red is unique": solid-color bg
7.32px, noise bg 9.78px, near-red decoys 6.46px, tiny cursor 8.21px. The
CLASSIFIER follows crop quality — holds ~0.64 under 6 distractors but collapses
when the crop is corrupted (noise bg 0.13, big cursor occluding center 0.40,
shape too big 0.41 / too small 0.24, heavy noise 0.17). New /api/animation/ood
route + module (per-condition error/accuracy bars + sample failure frames with
the predicted crosshair). Flask restart needed.

## 2026-07-15 (Claude) — Interactive Model-1b reworked (was broken)

User: the interactive was "really bad" — cursor offset up-left of the mouse,
reads near-random. Fixes: (1) canvas coordinate bug (rect-scaling, border on a
wrapper). (2) cursor JITTER training + bigger shapes so a shape reads from
anywhere on its body (was trained cursor-at-center only, so free hover was OOD).
(3) geometric gate — off-shape reads "nothing". (4) THE big one: Env normalises
test maps by the test-batch std, and one scene's field is a narrow batch, so
normalisation was wrong (~0.73 core acc); added `Basis`, a cached patch-PCA
projector that fits once at load and normalises by the training-reference std —
fixes accuracy AND kills the per-request SVD (local CUDA-OOM cause), 3x faster.
(5) interactive uses a distinct-5 classifier (circle/square/triangle/plus/xcross)
with the read = majority over each shape's CORE — stable ~0.85-0.89 per-shape
read + confidence. `dot_shape.py`: `jitter` param, `dot_shape_sub_model.json`
subset checkpoint; subset runs no longer overwrite the 10-class page result.
Flask restart needed.

## 2026-07-15 (Claude) — Attention line: depth/occlusion fix + interactive mode

User spotted Model 1b's misses were an occlusion failure — when a bigger
distractor ENCOMPASSES the target, the model picked the encompassing shape, not
the one the cursor sits on. Fix: `gen_labeled` draws a large distractor centered
on the target (behind it) in 60% of examples (`overlap`), forcing the model to
use occlusion (the target is drawn in front) to read the right shape. Controlled:
all-encompassing test 0.5335 (baseline) -> 0.7220 (overlap-trained), +19 pts, for
~1.5 pts on clean (0.9545 -> 0.9430). `dot_shape.py` saves a reloadable
checkpoint (`dot_shape_model.json`) and the depth numbers into `dot_shape.json`.

NEW interactive mode (`dot_live.py`, `/api/animation/cursor_field`, card on
/animation): server renders a random scene and precomputes the model's read
(tracked point + shape) over a 32x32 grid of cursor positions in one batched
gradient-free pass, so moving the mouse gives an instant live readout (green
crosshair + shape name); Shuffle makes a new scene (~3s). Recurring basis rule
again: the classifier's crop-PCA basis is rebuilt from its own training crops
(first 2000, seed 1). Flask restart needed for the new route + template.

## 2026-07-15 (Claude) — Attention line: fixed the white-target leak, retrained

The tracker's target shape was always WHITE while distractors were colored, so
the model could track "the white shape" and never use the red cursor — the
cursor-as-designator premise was untested. Fix (`_rand_color`, across
dot_track/dot_shape/dot_infer): the target now takes a random non-red color from
the SAME distribution as the distractors, so only the cursor on it marks it.
Retrained: tracker 1.84px mean / 1.47px median (R² 0.9817) static, 3.44px mean /
2.81px median moving over an 80-seq pool (demo shows a representative spread now).
Shape-ID via the tracker's attention, reading shape regardless of color: 0.9485
(10 shapes) / 0.9825 (circle-vs-square) — honest drops from the leaky
0.9955/0.9985; the cursor now does real work. Right foundation for Model 2.

## 2026-07-15 (Claude) — Attention line: WHERE unlocks WHAT

After the clutter wall (the stack cannot attend), a new "Attention & control"
section builds a little agent from the ground up on a red cursor. Three
gradient-free models planned: 1) tracking, 2) control, 3) action.

- **Model 1 · tracking** (`dot_track.py`, `dot_infer.py`) — the lab's first
  REGRESSION task. Red cursor pinned to a moving shape amid 3 colored distractor
  shapes; radial features by greedy residual-boosting, ridge to (x,y). Follows
  the moving cursor at ~1.5px mean error (64px frame, static R² 0.9895),
  isolating it by color-based figure-ground. Green-crosshair demo on the page.
- **Model 1b · recognize** (`dot_shape.py`) — attention-gated classify: the
  frozen tracker's OWN predicted position is a spotlight; crop the 20px window,
  evolve a shape classifier on the crop. "Where" is the tracker's, only "what"
  is learned. **0.9955 on 10 shapes, 0.9985 circle-vs-square** (= the true-crop
  ceiling); distractors fall outside the crop and are ignored for free. New
  `/api/animation/shape` route + Model-1b module (attended point, the crop, the
  model's shape call).
- **Basis bug** fixed: a frozen model's genomes only mean something under THEIR
  OWN training PCA basis. Building the tracker's Env on the shape data re-fit the
  wrong basis (21px / 0.27 acc); rebuild from `gen_dot` restored 1.5px / 0.99.
  Same lesson as the resolution work.

Next: Model 2 (control — hand the model the cursor, train it to MOVE).

## 2026-07-15 (Claude) — Scaling & robustness campaign + continue-training method

Scaling the temporal radial motion model (`radial_anim.py`) with a growing stack
of experiment **modules** down the `/animation` page (Scaling & robustness
section). **Every module has live animated visuals** (canvas sample rows) +
result bars + an honest note; each has a `/api/animation/<x>` route reading a
`radial_data/*.json` export. New capability: **continue-training** (warm-start on
a saved model). Pod work on a fresh RTX PRO 6000 Blackwell (see
runpod memory); all evals gradient-free, frozen genomes+head unless stated.

- **Module 1 — background robustness.** Frames' solid-black background → per-frame
  **random RGB color** (`make_anim_data(bg="randcolor")`, RGB pipeline). Motion
  task, matched settings (grid 8): black **0.854** → random color **0.745**
  (−11 pts, still 7.5× chance). Confirmed the [[r0-fatness-law]] on an
  un-designed task — the color distractor is perception-bound, so R0 grew fatter
  (332 vs 249) and the stack shallower (6 vs 7). `anim_bg_ab.py`.
- **Module 2 — inverted B&W (frozen color model).** The color model, genomes+head
  frozen, on new regimes: plain B&W (black bg/white shape) **0.785** (removing
  color *helps* — it was a distractor); **inverted** (white bg/black shape)
  **0.307** — it keys on a *bright* blob, so flipping polarity breaks R0
  (honest limit), but 3× chance survives. `anim_bg_ood.py`.
- **Module 3 — continue-training repair.** NEW METHOD `anim_continue.py`: load a
  finished model, keep the patch-PCA **basis frozen** (loaded genomes only mean
  something under it), rebuild columns + temporal grid on new data, keep stacking
  new spaces. Repaired the inversion: warm base 6 spaces + 4 new → inverted
  **0.31 → 0.79** (small color give-back 0.76→0.70) = polarity-robust, 31 s.
- **Module 4 — random shape size (frozen color model).** Per-sequence radius scale
  (`gen(size=...)`). Random size costs ~4 pts; small (0.4–0.7×) worst at
  **0.634** but still 6.3× chance; large 0.676 — graceful degradation (tracks a
  moving *region*). `anim_size.py`.
- **Module 5 — resolution scaling (crank).** Made the whole pipeline
  resolution-aware: `Env` reads input size from the data (was hardcoded 32),
  `make_anim_data`/`sample_seqs`/`gen` take `res=` (composite at native 64,
  area-downscale), `run()` infers res. Trained separate models at 32/48/64:
  test **0.79 → 0.82 → 0.84**, and the small-shape floor **recovers 0.625 →
  0.707** (+8 pts) — the Module-4 weakness was largely *resolution-bound*.
  `anim_res.py`.
- **Module 6 — one model, many resolutions.** Fixed `feature_r0` to read spatial
  size from the map (`√L`) not the basis, so one Env handles test-res ≠
  basis-res. Generalization matrix (`anim_multires.py`): a single-resolution
  model does NOT generalize (train64@test32 = 0.43). Continue-training on a
  resolution mix (`anim_continue_res.py` — merges each resolution's frames in the
  resolution-invariant grid space) fixes it: from res-64 → **0.745/0.779/0.821**
  at 32/48/64; low→high (res-32 base + 64) → **0.760/0.762**, gains high-res
  without forgetting low. One model, all resolutions, gradient-free.

Files: `anim_bg_ab.py`, `anim_bg_ood.py`, `anim_continue.py`, `anim_size.py`,
`anim_res.py`, `anim_multires.py`, `anim_continue_res.py`; `radial_anim.py`
(res-aware + `sample_seqs`), `radial_evo2.py` (Env res-aware), `radial_stack.py`
(`feature_r0` per-map dims); `app.py` (+7 `/api/animation/*` routes);
`templates/animation.html` (6 modules). Artifacts on the pod +
`radial_data/anim_*.json` local. NOTE: Flask restart needed to serve the new
routes/template.


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

