# Changelog — ResNet (gradient-free evolved residual networks)

Per-project log for the `/resnet` page and the `resnet_evo.py` pipeline. The
whole lab is gradient-free (GENREG rule #1); this line asks the ResNet question
*inside* that law. Newest entries at the top.

---

## 2026-07-14 (Claude / Opus 4.8) — STACKED spaces: gradient-free stacking BEATS single-space

**Headline: mature-R0 + spatial-grid stacking → TEST 0.6638, beating the
single-space record 0.6593 (+0.45 pt), fully gradient-free.** First time a
stacked evolved-feature tower beats one deep space on this line. One seed;
test touched once; val tracked it the whole way.

Implements the emergent-cap stacking idea (`documentation/stacking.txt`) in
`resnet_evo.py` (`run_stacked`, `--stack`) and then fixes it across three
iterations. The full honest arc:

| Config | Test | vs single-space (0.6593) |
|--------|------|--------------------------|
| Stacked, **scalar** handoff, early R0 | 0.6259 | −3.3 |
| Stacked, **spatial** handoff, early R0 | 0.6413 | −1.8 |
| Stacked, **spatial** handoff, **mature** R0 | **0.6638** | **+0.45 ✓** |

Final winning run: **4 spaces, emergent caps 447 → 54 → 27 → 23 = 551 genomes,
val 0.6756, test 0.6638, 1469 s.** Per-space val: R0 0.6586 → R1 +0.0118 →
R2 +0.0055 → R3 −0.0003 (economy stops the stack). No gradients.

What made it work (both were the user's diagnoses):

1. **Pass REPRESENTATIONS, not answers (the spatial-grid fix).** Originally each
   space pooled its spatial map to ONE scalar and the next space read those
   scalars — so all "where" was destroyed and stacking could only re-express R0
   (deeper spaces flatlined, +0.004 total). Fix: each space now emits a coarse
   **GRID=4 (4×4) spatial map per genome**; the next space is a spatial residual
   genome over that `(n_genomes, 4, 4)` tensor (`feature`/`feature_grid`,
   `new_genome_grid`, `mutate_grid`, `_to_grid`). Spatial structure survives →
   real hierarchical build-up (edges→textures→…). Alone: 0.6259 → 0.6413.

2. **Let R0 MATURE (the windowed-cap fix).** The single-round trip metric killed
   R0 at round 11 (333 genomes, val 0.630) on a −0.0001 dip, even though R0 keeps
   climbing to ~0.66 with brief plateaus. Fix: a **patient windowed cap** —
   trip only when val is flat over `CAP_WINDOW=5` rounds (live threshold in
   `F:\Resnet\cap.txt`, default 0.002) + a 3-empty-round backstop. R0 then
   matured to **447 / val 0.6586**. Combined with the grid fix: → 0.6638.

Per-space **energy economy** (validated Phase-D constants: decay 0.75, out-cost
0.05, restore 0.04, gain 400, floor 0.2, steady-state pop) ran the whole time —
starvation held ~5%/gen (in the 3–15% band), and the emergent caps
(447→54→27→23) show deeper spaces self-sizing leaner, exactly as
`stacking.txt` predicts. No hard genome cap anywhere; scarcity sets the size.

**Rotation — investigated, DEAD END (recorded so we don't retry it).** Tested
the radial "relative motion creates diversity" move on the stacked inputs:
- **Block rotation** (all ~F/2 adjacent feature pairs at once): small real
  bump, peak **+0.008 at 30°/space** (0.6264), washing out by 90°. But that's
  dense feature remixing, not "rotation about an axis."
- **Single-axis on the embedded map** (SVD → rotate the dominant PC0–PC1 plane,
  the faithful radial move): **byte-for-byte identical test 0.6184 at every
  angle 15°→90° — zero effect.** A one-plane rotation of a 333-dim space is
  linearly absorbed by the genomes' own `mix` + the ridge readout. Rotation is
  not a lever for this pipeline; the block bump was remixing, not rotation.
  (`resnet_rot_sweep.py`, `F:\Resnet\rot_sweep_*_summary.json`.)

**Engineering added along the way:**
- **R0 cache** (`F:\Resnet\r0_cache.json`, `_load_r0`/`_save_r0`) — R0 is
  deterministic and downstream-independent, so it's cached and reused
  (~40 s to rebuild columns vs ~5–12 min to re-evolve). Keyed on
  seed/pop/gens/data-size/cap/window so it re-evolves only when R0 genuinely
  differs. Turned the rotation/cap sweeps from ~11 min/point to ~67 s/point.
- **Live-tunable cap** — `F:\Resnet\cap.txt` overrides the trip threshold at the
  start of every round; no restart needed to retune.
- All artifacts on `F:\Resnet` (env `GENREG_RESNET_DIR`), never C:.

**Open next steps:** confirm with a 2nd seed (fast now — R0 cached); combine
across seeds (ensemble); surface the stacked result on the `/resnet` page.

## 2026-07-14 (Claude / Opus 4.8) — first full run (stopped early)

- **Full-CIFAR run, local GPU, stopped at round 48** (STOP lever, user request —
  urgent pivot). Result: **386 residual genomes, val 0.6725, TEST 0.6593**
  (test touched once), 858 residual blocks, ~50 min. Still rising at stop
  (val 0.667→0.673 over the last ~6 rounds — no plateau; this is a floor, not
  the ceiling). Artifacts: `F:\Resnet\resnet_evo_cifar.json` +
  `resnet_evo_ckpt.json`; run recorded under `runs/resnet_evo/`.
- **Parameter count (the headline):** **15,479 evolved params** (12,721 float +
  2,758 categorical) + 3,870 closed-form ridge-head numbers = **~19.3K total**,
  avg **40 params/genome**, avg depth **2.22 blocks**. Shared patch-PCA
  environment (~43K numbers, data statistics, not evolved) sits underneath. For
  scale: a gradient ResNet-20 is ~270K backprop weights — this is ~14× fewer,
  gradient-free.
- **Evolution kept the residual skip working.** Final depth distribution
  **{1:88, 2:150, 3:122, 4:26}** — the mode is 2–3 blocks and 4-deep stacks
  survived selection. If the skip were dead weight everything would have
  collapsed to depth-1; instead deep blocks earn their slot. The bootstrap-as-
  no-op init (rule VI) is doing its job.
- **Interactive residual demo added to `/resnet`** (`static/resnet_demo.js`):
  animated block schematic with the identity-skip arc + a signal-through-depth
  waterfall + energy-vs-depth chart contrasting residual (holds signal) vs plain
  (collapses). Controls: residual/plain, depth, gain, the 8-func activation
  catalog, and a "bootstrap (gain→0)" button that demonstrates the no-op init.
- **Not the ceiling.** The 0.70 grammar-v2 line is a *different* architecture
  class (feature-genome), not a bound on this one. A resumed / longer run (the
  checkpoint is intact) or the emergent-cap stacking idea
  (`documentation/stacking.txt`) are the open next steps.

## 2026-07-14 (Claude / Opus 4.8) — line created

- **New pipeline `resnet_evo.py`** — gradient-free evolved **residual
  networks** on CIFAR-10. Genome = a stem of 2–4 label-free patch-PCA channel
  maps (the environment, per "features are the environment") + a stack of
  residual blocks whose depth is a gene:

      h ← h + gain · act( a · (mix · h) + b )

  The identity **skip is the ResNet gene**; `mix` is the C×C 1×1-conv analog;
  `act` is the 8-function GENREG activation catalog. New blocks are
  **bootstrapped as near-no-op** (gain≈0, identity mix, act=id) then evolved
  (rule VI — stacked layers can't be evolved from random init). The head
  collapses channels to one map and reduces to one scalar via the same evolved
  soft spatial window + stat (mean/max/std) as `radial_evo2`.
- **Reuses the radial machinery unchanged**: `Env` (per-scale patch-PCA maps),
  `make_scorer` (Schur-complement border-ridge fast fitness), `_ridge_soft`,
  `_tprims`, and the comma-GA freeze-and-compose loop + checkpoint/STOP/resume
  contract + output-JSON shape. Only the feature grammar changed.
- **Artifacts off C:** — checkpoint and result JSON default to `F:\Resnet`
  (env override `GENREG_RESNET_DIR`; falls back to `radial_data/` when F: is
  absent, e.g. on the pod). Nothing large lands on C:.
- **Runs integration** — every completed run writes the standard runs/ file
  trio under `runs/resnet_evo/<rid>/`, so it appears on `/runs` (env
  `resnet_evo`, tagged `resnet` / `gradient-free`, smoke runs tagged `smoke`).
- **New page `/resnet`** (`templates/resnet.html`, `static/resnet.js`) — mirrors
  the Radial styling: residual-grammar explainer, headline test/val stats, the
  val-accuracy curve over rounds with Coates-Ng / radial-v1 / grammar-v2
  reference lines, and "what evolution chose" distribution bars (residual depth,
  channel width, read-out stat). Reads `/api/resnet/result`. Added to the shared
  nav (`_nav.html`, key `resnet`).
- **Smoke-tested end-to-end** (`python resnet_evo.py --smoke`, 3k/800 subset,
  2 rounds, pop 16): patch-PCA env → residual genomes (depths 1/2/3 evolved) →
  Schur scorer → ridge read-out all exercised; val 0.233 → 0.283, test 0.305
  on the tiny subset (> 0.10 random); output + checkpoint written to
  `F:\Resnet`; a `runs/resnet_evo` entry created. **No full training run
  launched** (the pod is busy with the seed-farm + scale jobs).
- **Not yet done**: a real full-CIFAR run (`python resnet_evo.py`) to measure
  the residual grammar's ceiling vs the grammar-v2 record (0.7035). Deferred —
  needs a free GPU.
