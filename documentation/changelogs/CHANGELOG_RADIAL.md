# Changelog — RADIAL

Per-project log for the RADIAL line. Seeded 2026-07-14 from the master
CHANGELOG.md (all entries mentioning this project); new RADIAL entries go at
the top of the log below, and also in the master CHANGELOG.md.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-14] (Claude)** — **Push-to-80 stage built + crossover replicate
  results.** (1) `radial_push80.py` implements the user's "push to 80" plan
  (documentation/push to 80.txt) as grammar v3, all three levers as GENES:
  STACKING (stage-2 over the converged v2 substrate, head sees both stages),
  META-GENOME (each term's source is a gene: raw scale-map component OR any
  frozen genome's output — genomes assembling genomes), CONDITIONAL ROUTING
  (optional evolved gate: feature × sigmoid(k·gate(x)), gate = own mini
  feature with evolved sharpness — attention from first principles).
  Downstream energy economy (steady-state, relative contribution), Schur
  scorer, crossover 0.5, honest 10k val, freeze bar 0.0002 / dry-streak 5
  (calibrated by smoke: at a 0.71 substrate marginal gains are thin; a
  full-size round froze a meta-term genome with starvation 5.0/gen, in
  band). Staged on pod + shadow. (2) CROSSOVER REPLICATE (seed 7, local)
  CONVERGED: 503 genomes, val 0.7136, **TEST 0.6936 — above the record's
  val but BELOW its test** (0.7035): crossover climbs faster and converges
  smaller but overfits the selection split more (gap 0.020 vs 0.007). Pod
  seed-13 nearly converged (~0.711 val) — second datapoint pending. Local
  push-80 run launched on the seed-7 substrate.
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
- **[2026-07-14] (Claude)** — Evolution campaign part 10: **crossover A/B —
  it helps.** `crossover()` added to `radial_evo2.py` (uniform per-gene
  recombination of two tournament parents: terms drawn from both, window
  blends half the time, then normal mutation), `run(p_cross=…, ckpt_path=…,
  out_path=…)` so experiments never clobber the record artifacts. Controlled
  smoke (same seed, same 10-round budget, full-50k honest val): mutation-only
  val 0.5401 / test 0.5307 vs +crossover **val 0.5551 / test 0.5459**, ahead
  at ALL ten rounds. Full run relaunched with p_cross=0.5 targeting the
  0.7035 record (fresh ckpt `evo2x_ckpt.json`).
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
- **[2026-07-14] (Claude)** — Evolution campaign part 8: **FULL SEND
  EXPRESSIVITY — grammar v2, nothing hand-crafted** (`radial_evo2.py`,
  user: "the genomes need to evolve those features"). Every structural
  property of the v1 feature-genome class that was a design decision is now
  a GENE: patch SCALE (4-14, environment lazily builds patch-PCA maps per
  scale evolution visits, LRU-cached fp16), interaction ORDER (2-3 terms,
  add/remove-term mutation), program DEPTH (1-2 bends, add/drop-stage
  mutation), and POOLING — no region catalog, an evolved soft spatial
  window (cx, cy, sigma genes: wide sigma IS global pooling, tight IS a
  local region). Human hand: math primitives (8 activations, 3 ops, 3
  stats) + data statistics only. Evolved directly on full 50k with the
  honest 10k val; test once; per-round checkpoints (`evo2_ckpt.json`),
  STOP_EVO lever, resume. Smoke: 24 genomes → val 0.402 with evolution
  already diversifying scales on its own. Full run launched (target: the
  v1-class ceiling 0.6378); monitor armed.
- **[2026-07-14] (Claude)** — Evolution campaign part 6: **the deep run — where
  it actually stops.** `run_deep()` in `radial_evo.py`: stack radial spaces
  until a stage adds < 0.002 val; per-round ATOMIC checkpoints
  (`evo_deep_ckpt.json`) + graceful kill lever (touch
  `radial_data/STOP_EVO`) + resume-from-checkpoint. Run to natural stop, no
  lever needed: **stage 2 = +310 genomes (val 0.661→0.7185 — the earlier
  60-round cap had been leaving +2.4 val pts on the table), stage 3 = +8
  genomes (+0.0015) → tower stops at 2 real stacked stages** (667+310+8 =
  985 genomes). Full-50k transfer of the whole tower (`run_deep_full`):
  **0.6378** — ladder: v1 0.5904 → 0.6198 → 0.6257 → 0.6353 → **0.6378**.
  HONEST CAVEAT: the stop was declared by an exhausted instrument — val
  (2000 samples) was reused for ~970 freeze decisions and overfit (val 0.720
  vs 8k-test 0.5885, worse than the shorter stack's 0.5945 on 8k), yet the
  full-scale test still improved; "where it stops" is partly "where a
  2000-sample val stops seeing" — next lever is a bigger/fresh-per-stage
  val split or evolving on full data. Export `evo_deep_cifar.json`,
  `evo_deep_full_cifar.json`.
- **[2026-07-14] (Claude)** — Evolution campaign part 5: **the plateau safety
  mechanism WORKS (Phase D: stacked radial spaces) + the downstream energy
  economy.** `phase_stack()`: when phase_b hard-converges (a round freezes
  nothing), the frozen genome OUTPUTS become the data for a NEW radial space
  (stage-2 genomes read stage-1 features exactly as stage-1 read patch
  components). Results (test once each): stage-1 0.5840 → **stacked 0.5945**
  on 8k (plateau broken, val 0.661→0.695 still climbing at the round cap);
  stage-2 alone re-expresses 0.5285 from outputs only; **full-50k transfer
  0.6353** (`phase_stack_full`). Ladder: v1 0.5904 → B 0.6198 → C=80 0.6257 →
  **stacked 0.6355**. THEN (user directive): every downstream space's genomes
  live under a real ENERGY ECONOMY — existing costs, outputting costs, any
  valid output restores a little, real energy ONLY from above-median
  contribution to the right answer through the composition; steady-state
  population (genomes persist, starved die, slots go to children — two
  earlier attempts left starvation at 0%: per-genome gains are almost never
  negative so the bar must be RELATIVE, and a comma-style GA never lets the
  existence clock tick; both fixed). Now **7.5% starved/gen (in the 3-15%
  band)** and the economy buys EFFICIENCY: 96% of the stacking lift with 50
  stage-2 genomes instead of 411 (8k stacked 0.5900, full 0.6283). Exports
  `evo_stack_cifar.json`, `evo_stack_full_cifar.json`.
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
- **[2026-07-14] (Claude)** — **Validation-report response**
  (`validation_report.txt` audits the /radial/demo/cousins DEMO — periodic
  140-lens generator, 128-sample probes — not the main line; its
  overparameterization and modulo-duplicate critiques are correct for that
  demo). Ran the same attacks on `radial_map.py`: overparameterization does
  NOT apply (1440 fit rows vs ≤749 features, heldout scoring; 100 features
  still give R² 1.000), generator is aperiodic (duplicate pairs 1.17%;
  deduped 171-lens bank still 1.000), rotation-red-herring already resolved
  by Q2 (the linear core). One critique LANDS and is now logged in findings
  §9: **effective dimensionality of the 748-lens bank is only ~46** (1% sv
  threshold) — the infinite space collapses to a few dozen usable directions
  per domain, explaining early L-curve saturation and the sin32x frontier;
  the map = ~46 real axes + dense redundancy, hence slice CHOICE > COUNT.
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
- **[2026-07-13] (Claude)** — **Cousin finder: explicit runs, Runs-page
  recording, JSON report download.** `/radial/demo/cousins` no longer runs on
  page load (canvas shows a "press find" placeholder); cousins and siblings
  each have their own run button. Every run POSTs to new
  `/api/radial/demo/record`, which writes a standard run folder under
  `runs/Demo_Radial/<rid>/` (config.json / history.jsonl with the log lines /
  summary.json with the stats / report.json with the full result) so it
  appears on the Runs page under the Demo_Radial environment. "download
  report (JSON)" buttons fetch `/api/radial/demo/report/<rid>` (attachment;
  falls back to a client-side blob if the server didn't record). Sibling
  reports include the full lineage table + up to 500 sibling pairs. NOTE:
  new API routes — Flask restart required.
- **[2026-07-13] (Claude)** — **Cousin finder expanded to siblings + lineage**
  (`/radial/demo/cousins`). Every lens is a1(a2(scale*x+bias)) so it has two
  parents (outer a1, inner a2); lenses sharing a parent are siblings. New
  section: parent-activation selector + relation mode (outer/inner/either),
  stat cards (family members, sibling pairs, mean sibling |r| vs population
  baseline, and "cousins that are relatives" — the share of rotation-cousin
  pairs that also share a parent, self-maps excluded), a clickable 14-row
  lineage table per parent activation (outer/inner counts, members, sibling
  |r|, delta vs baseline color-coded, rotation-cousin share), top-20 sibling
  pair log, and cyan family-member rings on both 3D grids. Backed by a full
  216x216 |r| matrix computed per run. Static files only.
- **[2026-07-13] (Claude)** — **Radial demo: Y-rotation fix + dynamic lenses +
  Space Cousin Finder sub-page.** (1) Fixed the "+ Y rotation" checkboxes: the
  fixed-500px canvases overflowed their flex wrappers and covered the toggles,
  eating their clicks (also stretching the render) — canvases now fill their
  wrapper (`fit()` measures real height). (2) Lenses are now a dynamic list:
  "+ Add lens" appends lens 3+ (own X/Y sliders, fresh palette color, show
  toggle, remove button; schematic lenses 1-2 fixed; cap 8); legends render
  dynamically. (3) New sub-page `/radial/demo/cousins` built to the downloaded
  `radial_space_cousin_finder.html` (copied to
  `documentation/RADIAL_SPACE_COUSIN_FINDER_SCHEMATIC.html`): 6x6x6 grid of
  deterministic composed-activation lens programs, Y-rotated onto neighbors,
  Pearson-correlated signatures above threshold = cousin pairs; stat cards
  (pairs/families/redundancy), dual original/rotated 3D view with pair lines,
  top-20 pair log. Route added; demo page links to it.
- **[2026-07-13] (Claude)** — **Radial demo rebuilt to the downloaded visual
  schematic** (copied into `documentation/RADIAL_SPACE_VISUAL_SCHEMATIC.md`).
  `/radial/demo` is now the dual-panel demo: left "ground truth moves" (green
  10x30x10 column flows on Y through stationary blue Y+15deg and yellow
  X+15deg->Y+45deg lens cubes), right "lenses move" (the inverse), shared
  time, per-panel "+ Y rotation" toggles, window culling at +/-6, perspective
  camera locked at Y=0.6rad/dist 22, retina 2x canvases. Added play-around
  knobs on top of the schematic defaults: pause, flow speed, spin speed,
  visible-window size, per-layer visibility, all three lens-angle sliders,
  optional free camera (drag orbit/wheel zoom), and a reset-to-schematic
  button. Template + static JS only; no route change.
- **[2026-07-13] (Claude)** — Radial demo: **yellow offset copy added.** Third
  checkbox on `/radial/demo` overlays a stationary duplicate of the cube
  (like blue) but pre-rotated 30 degrees on the Y axis then 30 degrees on the
  X axis. Template key + corner label updated.
- **[2026-07-13] (Claude)** — **Radial rotation demo page (`/radial/demo`),
  no models.** New `templates/radial_demo.html` + `static/radial_demo.js` +
  route in `app.py`: a 3D XYZ coordinate grid with 8,000 red ground-truth dots
  (20x20x20, one per grid coordinate, centered on the origin). A checkbox
  rotates the data itself — every dot uniformly about the Y axis, not a camera
  effect; a second checkbox overlays a stationary blue copy of the same cube.
  Drag-to-orbit / wheel-to-zoom camera is independent of the data rotation.
  Flask restart (or debug auto-reload) needed to pick up the new route.
- **[2026-07-13] (Claude)** — **Radial space discoveries documented +
  published.** `documentation/RADIAL_SPACE_FINDINGS.md` (the master findings
  doc: the space, the galaxy structure with the radius=strangeness correction,
  the linear-diversity hypothesis result, rotation/redundancy, all five domain
  baselines, model-size analysis, the four quantified genome-starting gaps)
  and `documentation/RADIAL_SPACE_FINDINGS.pdf` (5 pages, generated from live
  code: title/abstract, the six per-domain behavioral maps with shapes and the
  red origin, loops + per-domain rotation curves with the redundancy
  annotation, baseline bar chart with the bigram-ceiling line, details page).
  Branch pushed to origin (GENREG-Builder).
- **[2026-07-13] (Claude)** — Baselines view: **each card now shows the
  domain's actual 3D MAP** (user: "I really wanted the map of the baselines so
  I can see the shapes"). Every card fetches its own 600-lens cloud from
  `/api/radial/map` with that domain's stream and renders a mini 3D scatter —
  auto-spinning, drag to orbit, depth-sorted, coloured by nonlinearity, red
  origin dot, measured axis shape in the caption. `text`/`audio` added to the
  main Map view's data-type select too (both verified: text 4.85/4.10/2.03,
  audio 5.17/3.63/1.85, determinism 0.0). Cards/bars/curves untouched.
- **[2026-07-13] (Claude)** — Radial page: **Baselines view** (sidebar View
  toggle: Map | Baselines). New GET `/api/radial/baselines` serves the roadmap
  exports (`radial_data/baseline_*.json` + `prebaseline_fixes.json`); the page
  renders a card per domain: accuracy bars (majority / raw linear / bigram
  ceiling / lens bank / best rotation slice — green when the slice beats the
  bank), accuracy-vs-lens-count curve with the raw-linear reference dashed,
  the 360° rotation sweep with the random-subset baseline dashed, per-class
  recall bars (red < 0.5), and a key-finding line per domain; plus a
  pre-baseline-fixes card (spread per spin axis, the shared dead-zone floor).
  Charts draw after DOM insertion (clientWidth-0 fix); colour-by toggle
  selector scoped to `[data-c]` so the View toggle doesn't clobber it.
  **Flask restart still pending** (all radial routes).
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
- **[2026-07-13] (Claude)** — Radial sphere: origin (0,0,0) marked with a solid
  **red dot + halo ring**, drawn at the true world origin (`project(0,0,0)`) so
  it stays put while orbiting/zooming. Replaces the grey identity outline.
- **[2026-07-13] (Claude)** — Radial v2: **the map is now a navigable 3D
  sphere.** `build_map()` returns the full top-3 MDS embedding (adds `z` per
  point; the radius/nonlinearity check now uses the 3D norm — corr 0.50). The
  page renders an orbitable point cloud: drag = orbit (yaw/pitch), wheel =
  zoom toward the cursor (0.2x-40x), shift-drag or right-drag = pan, click
  still inspects a lens (only fires on non-drags). Depth cues: far points
  smaller/dimmer, near points on top; wireframe sphere hint (three equator
  rings + inner radius rings); identity marker stays at the origin. Export
  `map.cols` gains `z`. No new endpoints; **Flask restart still pending.**
- **[2026-07-13] (Claude)** — Radial v2 ladder: **temporal rungs REMOVED**
  (user: "no one said add temporal yet, that's a whole nother beast"). The
  ladder is pointwise-only now — 7 rungs, square through sin32x. Results
  unchanged where it matters: loops clears 6/7, frontier sin32x (0.9902).
- **[2026-07-13] (Claude)** — Radial v2: **auto-incrementing task ladder**
  (user: "if R2 hits .998 the task changes to the next harder task").
  `ladder_probe()` in `radial_map.py`: 10 ordered rungs — pointwise with rising
  curvature (square, abs, sin3x, ripple, sin8x, sin16x, sin32x) then TEMPORAL
  (lag1_prod, lag4_mean, delay5 — value at t depends on other samples, which a
  pointwise lens bank cannot see); heldout R2 >= 0.998 auto-advances, first
  miss stops the climb and is the frontier. RESULTS (400 lenses): loops clears
  6/10, frontier **sin32x** (0.9902); noise clears 5/10, frontier **sin16x**
  (0.9492) — the frontier is currently FREQUENCY, not temporality (the temporal
  rungs are never reached), and it is data-dependent: the loops stream's value
  distribution supports higher-frequency fits than noise. Endpoint
  `/api/radial/ladder` (n/kind/threshold), CLI `python radial_map.py ladder`,
  page button "Task ladder (auto-harder)" with per-rung table (✓/✗ + frontier
  line), included in the JSON export as `ladder`. **Flask restart still
  pending** (map/lens/probe/rotate/ladder routes).
- **[2026-07-13] (Claude)** — Radial v2: **rotation probe — spin the map, not
  the data** (user's direction: "rotate the radial axis on the y axis, one
  degree, then run the linear probe again — the data isn't changing").
  `rotation_probe()` in `radial_map.py`: the map is embedded in 3D (top-3 MDS
  of the same behavioral signatures, `_mds()` refactored out), rotated about
  the Y axis 1 deg/step; per angle the closed-form linear probe sees ONLY the
  slice of lenses in the current viewing plane. Honest baselines: same-size
  random subsets + full bank. RESULTS (loops, 800 lenses): at slice ~90 every
  angle saturates (R2~1, ceiling — no signal visible), so default slice is
  ~23 (frac 0.03), where real angular structure appears: **best angle 81 deg
  R2 0.998, worst angle 5 deg R2 0.169, spread 0.83** — while random 23-lens
  subsets average 0.996. Reading: co-planar lenses are behaviorally SIMILAR
  (redundant views), so bad angles are far below chance — the map's geometry
  carries real information, and the diversity-does-the-lifting hypothesis is
  confirmed from the failure side. New endpoint `/api/radial/rotate`
  (n/kind/step_deg/frac), CLI `python radial_map.py rotate`, page button
  "Rotation probe (1 deg/step)" with R2-vs-angle curve (random-baseline band),
  best/worst/spread panel; included in the JSON export as `rotation_probe`.
  **Flask restart still pending** (now also for the rotate route).
- **[2026-07-13] (Claude)** — Radial v2 page: **terminal dock restored + Export
  results button.** The rebuilt page had dropped the shared dock stack — added
  the standard includes (xterm.css/js, addon-fit, termdock.js, app.js,
  agentpanel.js, configpanel.js) so terminals, the Agent panel and Run-Config
  panel are back on `/radial`. New "Export results (JSON)" button downloads a
  compressed-but-informative snapshot: column-oriented map rows (`cols`/`rows`
  arrays, 3-dp), honest checks, distribution summaries (radius/nonlinearity/
  oscillation min/p50/p90/max/mean), the 8 most-nonlinear / most-oscillatory /
  most-linear lenses WITH their program strings (all other programs omitted —
  the space is deterministic, any lens rebuilds from its index), plus the last
  probe table and selected-lens curve. No server changes; the pending Flask
  restart from the v2 rebuild still applies.
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
- **[2026-07-12] (Claude)** — **Lens map is now INFINITE / open-ended** (user:
  "the map is infinite, you can infinitely combine combinatorials and
  activations"). Replaced the fixed 3600-cell grid with a lens-PROGRAM generator:
  each lens combines any number of axes with coefficients, composes activations
  to any depth (`tanh∘sin[3+7+12]`), and can multiply two sub-lenses — richness
  (axis order 2→5, composition depth 1→3, product terms) grows with the lens
  index, so exploration keeps reaching new regions and never "finishes." Runs in
  a background thread until you Stop it (or a 20k safety cap). Each lens still has
  a deterministic address (index seeds its program), so it's an unbounded-but-
  fixed coordinate system, laid out RADIALLY (complexity = radius) — simple lenses
  fill the centre, richer ones spiral outward. Encoder uses a bounded reservoir
  sample (`BANK_CAP` 2600) so RAM stays flat while the map grows forever; the
  accuracy curve honestly PLATEAUS even as lenses keep coming (infinite lenses,
  finite useful info). New `stop()` + endpoint; page: "Explore (infinite)" + Stop
  buttons, radial scatter that fills outward coloured by structure/class-sep, best
  lens shown as its program string. Compounding (Extend, residual) preserved.
  Needs Flask restart.
- **[2026-07-12] (Claude)** — **Lens map now COMPOUNDS.** `radial_lensmap.py`
  keeps a frozen layer STACK: each exploration's PCA encoder is frozen, and a new
  "＋ Extend (stack a layer)" run sweeps a fresh lens map over the previous layer's
  OUTPUT (layer-2 lenses look at layer-1 features, not raw pixels). Stacking is
  RESIDUAL (each layer passes its input through alongside the new lens features),
  so extending is non-destructive — it can only add. Measured (honest): depth-1
  0.277 over raw-pixel PCA 0.206 (+0.071, the heavy lift), depth-2 0.279 (small
  further gain, beats its own input PCA 0.256), depth-3 0.272 (plateaus) — a sweet
  spot in depth, diminishing returns after layer 1; replace-stacking (non-
  residual) instead *degraded* with depth, which is why residual was chosen. UI:
  Extend button (enables after a run), status shows current layer + the whole
  stack's per-depth bests (L1 → L2 → …), and the accuracy overlay draws the
  previous layer's best as a dashed reference so you see whether the new layer
  compounds. `start(extend=True)` keeps the stack; fresh Explore resets it.
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
- **[2026-07-12] (Claude)** — **Compression verdict (corrects earlier "beats
  Fourier" claims).** `radial_compress_bench.py` runs a FAIR head-to-head: RS-
  Gabor matching pursuit vs DFT/DCT/Haar-wavelet top-K, all with 8-bit-quantized
  coefficients and honest byte accounting, on realistic signals. Result: bytes to
  reach 99% corr — speech DFT 36 vs RS 92, music DFT 32 vs RS 135, ECG DFT 62 vs
  RS >60c, transient **RS 18 vs DFT 30**. **RS-Gabor loses to plain DFT on every
  realistic signal and wins only on a lone isolated transient.** The prior "RS
  beats Fourier" wins were against an UNQUANTIZED top-K DFT (a baseline no codec
  uses) on burst-shaped signals; under quantized fair comparison the edge
  vanishes. Conclusion: radial-space compression is dominated by standard
  transform coding and is NOT a differentiator — the real keeper is the activity/
  path-fingerprinting direction (§11.1), not compression.
- **[2026-07-12] (Claude)** — Built **Radial Space v3 §11.1: real screen-capture
  fingerprinting** — actual desktop frames, not simulated streams.
  `radial_screen.py` grabs frames via PIL ImageGrab (~36ms full-res, downscaled
  to 160×90), extracts per-frame features (brightness, edge density,
  colourfulness, frame-to-frame motion), maps each through the winning linear M
  into a radial traversal path, and fingerprints the clip with a 10-dim stat
  vector. Nearest-centroid classifier with leave-one-out accuracy. Endpoints
  `/api/radial/screen/{record,train,classify,status,clear}` (record/classify
  block for N seconds of live capture). New **"Screen"** page mode: pick an
  activity → Record while doing it → paths overlay by label (idle/browsing/video/
  coding) → Train → "What am I doing now?" classifies live from path shape with
  confidence bars. Verified end-to-end: real capture 10fps, idle reads motion
  ~0.06; train/classify plumbing gives LOO 100% on separable clips. The real
  separation numbers come from the user recording actual activity. **Needs Flask
  restart to serve the routes.**
- **[2026-07-12] (Claude)** — Built + tested **Radial Space v3** (Test Suite 10:
  mapping-function characterization) from `~/Downloads/radial_space_theory_v3.pdf`
  — the doc had already folded in my v2 results and defined this as its "next
  action" to decide whether chain computation survives. `radial_mfunc.py`
  implements five M families (linear/arctan/sigmoid/sqrt/sinusoidal + paper) and
  runs 10.1–10.6: invertibility (phi-monotonic), condition number, collision
  precision, chain Lyapunov, proximity, activity classifier, gate composition.
  **Decision reached: memory/fingerprinting SOLVED, chain computation DEAD.** All
  four monotonic M invert and preserve proximity (best = linear: cond 3.9,
  proximity 0.019, classifier 98%); paper's hash and the sinusoidal fold fail
  invertibility/proximity. But **no M gives a stable chain** — Lyapunov runs
  −5.33…+0.83, all far from 0 (best −0.20, still contracting), so chained state
  decays to 0 or goes chaotic within a few steps, and the cos lookup discards
  sign+phase (irreversible). Single lookups are expressive (NAND at 100%) but
  can't be composed into a persistent circuit. Per the doc's own rule: drop chain
  computation, keep compression + memory + activity fingerprinting. New endpoint
  `/api/radial/suite10`; page got a **"Mapping M"** mode: phi-vs-input curves for
  all M families (straight/smooth = invertible, wavy/sawtooth = broken),
  Lyapunov-per-M bars (none at the stable line), and the M-comparison table +
  verdict. **Needs Flask restart.**
- **[2026-07-12] (Claude)** — Built + tested **Radial Space v2** from
  `~/Downloads/radial_space_theory_v2.pdf` (adds §5 Memory + §6 Computation
  claims and the §9.2/§9.3 test batteries). `radial_memory.py` implements the v2
  reference code (`mapping`, `lookup`, `chain`) and runs all suites honestly.
  Results: **9.1 8/8, 9.2 7/8, 9.3 7/9.** The two failures are the load-bearing
  new claims: 9.2.3 proximity preservation FAILS (adjacent inputs land 1.31×
  *farther* than random — "proximity IS similarity" is false, because
  phi=v*2.47 mod 2π scrambles neighbours) and 9.3.5 reversibility FAILS (chain
  uses abs+cos, both many-to-one). The chain is a contraction (Lyapunov -49) that
  collapses every input to 0 — a trivial dynamical system, not a processor;
  single logic gates ARE realizable (cos nonlinearity, XOR incl.) but can't be
  wired together. **Constructive finding (the paper's own §10.3/§11.3): the
  mapping M is the whole lever.** Swapping the broken M for a proximity-
  preserving one fixes both instantly — proximity **1.3→0.02**, activity-stream
  classifier **70%→98%**. So the one real capability — a traversal path that
  fingerprints activity (idle/switching/video) — works, but only under a mapping
  the paper didn't use. New endpoints `/api/radial/v2suite` + `/api/radial/
  traversal`; page got a **"Memory v2"** mode: activity-stream picker, side-by-
  side traversal paths (paper's M scatters vs fixed M draws a clean shape), and
  the grouped suite results with the M-lever headline. **Needs Flask restart.**
- **[2026-07-12] (Claude)** — `/radial` vs-Fourier view **rewritten for clarity**
  (user asked to dumb it down). Added a plain-English explainer banner (endless
  waves vs blips), a "building blocks" picture showing Fourier's full-width waves
  beside Radial's localized blips (the WHY), and a plain scoreboard: "pieces to
  rebuild (Radial vs Fourier)" + "Radial needs N× fewer" + match-% for each.
  Dropped the corr-vs-K curve (jargon). Backend `run_compare` now returns the
  per-piece waveforms and `rs95`/`ft95` (pieces each needs to hit 95%). Examples:
  burst 2 vs 7, two-bursts 3 vs 10. **Still needs Flask restart.**
- **[2026-07-12] (Claude)** — `/radial` **"vs Fourier" mode** added — pushing RS
  past the uniform Fourier basis. `radial_dict.py`: give the wasted radial
  coordinate a temporal job — let radius modulate over the sweep so a dot traces
  a WINDOWED oscillation (Gabor atom `exp(-(t-τ)²/2s²)·cos(ω(t-τ)+φ)`, optional
  chirp). That makes an overcomplete, time-localized dictionary (~3136 atoms);
  `matching_pursuit` greedily fits it. Head-to-head vs top-K Fourier, verified in
  Python BEFORE any UI: **RS-Gabor genuinely beats Fourier at low atom count on
  non-stationary signals** — burst K=3 corr **0.98 vs 0.70**, two-bursts K=2
  **0.91 vs 0.58**, chirp wins K=1–8; Fourier wins pure tone (1.0) and AM/walk at
  low K (honest losses). The dictionary CONTAINS near-global atoms so MP can fall
  back to Fourier — fair comparison. New endpoint `/api/radial/compare`; page got
  a Codec/vs-Fourier mode switch, non-stationary signal picker, atoms-K slider,
  3-way overlay (signal / RS-Gabor / Fourier) and a corr-vs-K curve showing where
  amber (RS) sits above blue (FFT). Verdict: this is Gabor/chirplet matching
  pursuit reached naturally from the radial framing — not new to DSP, but a real
  win over plain RS/Fourier exactly where Fourier is weak. **Needs Flask restart.**
- **[2026-07-12] (Claude)** — New `/radial` page: faithful build of the **Radial
  Space** system from `~/Downloads/radial_space_theory.pdf` (a deterministic
  rotation-lattice signal codec — unrelated to genomes). `radial_space.py`
  implements the paper's §7 reference code exactly (cubic lattice, `rotation_
  matrix`, `sweep`, per-dot sinusoid trajectory `r·cos(wT+φ₀)`, encode→address→
  decode) plus fixes that make it actually work: the paper's correlation-only
  encoder can't set amplitude (corr is scale-invariant), so encode also least-
  squares fits amp/phase; `decompose` is a greedy harmonic matching-pursuit for
  the §10.1 multi-axis extension. Full §8 validation suite runs live — **8/8
  pass**: determinism 0, origin displacement 0, pure-dot round-trip err 0 (120:1),
  phase survives noise (sd=1.0 → 1.1° drift), lattice density 0.330→0.036 as step
  refines. Endpoints `/api/radial/encode` + `/api/radial/validate`; page
  (`templates/radial.html`, `static/radial.js`): signal picker, signal-vs-recon
  overlay, harmonic spectrum, 1-dot/K-dot toggle, address + compression readout,
  suite panel. **Honest verdict (from the suite, shown in-page): a single-axis
  sweep is a ONE-frequency basis** — it nails a pure tone (corr 0.996 quantized,
  1.0 continuous) but a single dot stalls on multi-tone (0.86) and random-walk
  (0.89); only the harmonic decomposition reaches them (multitone→3 dots=1.0,
  walk→32 dots=0.99), and that is a greedy Fourier transform in geometric
  clothing. RS is a correct sinusoid codec; it beats raw storage exactly when the
  signal is sparse in this Fourier basis — the standard transform-coding caveat,
  not a new law. Nav link added. **Needs a Flask restart to serve the routes.**
