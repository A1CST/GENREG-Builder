# Changelog — GENREG

Tracks changes made by AI assistants working in this directory.
Multiple AIs share this workspace, so every change should be logged here with
date, author, and a short description. Append new entries at the top of the
log below; don't rewrite existing entries.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-14] (Claude)** — OPS: **silent pod-job loss caught and fixed.**
  The chained `nohup A & nohup B &` ssh launch only ever started the first
  job — seed-29 and the pop-128 scale test never ran (an hour of farm time
  lost); the earlier "3 processes" check was matching the wrapper's command
  string, not real processes. Both relaunched in separate ssh calls with
  `ssh -f`, verified by LOG-FILE EXISTENCE (the reliable check), both
  confirmed running. Farm state: pod = seed-19 (dry-out, val 0.7015),
  seed-29 (restarted), pop-128 (restarted); local = seed-37 (round 48,
  val 0.666). Lesson for the ops notes: never trust `pgrep -c` across ssh
  for launch verification — check the artifact.
- **[2026-07-14] (Claude)** — Ensemble-tower verdict: **stacking has gone
  noise-level; seed farming is the live lever.** A v3 stage on the 1042-genome
  union (212 keepers, fresh val 0.7459) tested 0.7278 — −0.0035 vs the plain
  union (0.7313), inside 1σ. At this substrate strength additional evolved
  stages neither help nor hurt; complementary DIVERSITY (new seeds) is what
  moves test. Local 4080 now farms seed-37 alongside the pod's seeds 19/29.
  Champion remains the 2-seed union at **0.7313**.
- **[2026-07-14] (Claude)** — **THE SCALING LAW APPEARS: cross-seed ensemble
  0.7313.** The union of the two independently-evolved v2 substrates (503 +
  539 = 1042 genomes, one head refit, 10 s of compute, test once) scores
  **0.7313** vs 0.6936 / 0.6962 individually — **+3.5 pts from pure seed
  diversity, essentially free.** Independent evolutionary runs discover
  complementary features; the union is worth more than either tower.
  Also: pod seed-13 push-80 with fresh val DONE (388 genomes, test 0.7119 —
  the fresh-val stack replicates across hardware). CAMPAIGN PIVOT (scaling
  is the goal): the H100 now farms ensemble fodder — seeds 19 and 29
  (grammar v2, crossover) running alongside the pop-128 scale test (3 jobs);
  locally the ENSEMBLE TOWER is running (v3 stage over the 1042-genome
  union, fresh val slice 3; `run()` now accepts multiple stage-1 ckpts).
  Exports: `ensemble2_cifar.json`, pod artifacts in `runpod_shadow/`.
- **[2026-07-14] (Claude)** — **Autonomous scaling campaign (user away):
  RECORD 0.7144; the tower converges at stage 3; pod at full utilization.**
  (1) FRESH-VAL ROTATION shipped (`run(val_slice=…, v3_ckpts=…)`): each stage
  selects against a different 10k train window, earlier v3 stages fold into
  the substrate by replaying their genomes against the channel bank of their
  time. (2) LOCAL TOWER: stage-3 (fresh window) = 96 genomes, **TEST 0.7144
  in 55 s** (gap on the fresh ruler 0.017 vs stale 0.034 — hygiene fix
  works); stage-4 = 11 genomes, test 0.7133 → tower CONVERGED at stage 3.
  Ladder: 0.5904 → 0.6378 → 0.7035 → 0.7079 → **0.7144** (note: stage
  results are sibling measurements — differences near noise are read
  cautiously). (3) CROSSOVER REPLICATE VERDICT (pod seed-13: 539 genomes,
  val 0.7123, test 0.6962): both crossover seeds beat the record's val and
  miss its test by ~1.5σ — crossover = faster climb, slightly more selection
  overfit; replicated across seeds AND hardware. (4) POD (H100) now runs TWO
  jobs: seed-13 push-80 with fresh val (round 80 in 21 s, val 0.7261) + the
  population-scaling question (v2, pop 128, seed 17). (5) Local GPU runs the
  cross-seed ensemble test (503+539 substrate union). Shadow sync v2 pulls
  ALL pod logs/jsons every 5 min.
- **[2026-07-14] (Claude)** — **NEW TEST RECORD: 0.7079 (push-80 stage, local
  seed-7).** The grammar-v3 stack converged in **77 seconds**: 403 stage-2
  genomes over the 503-genome substrate, val 0.7136→0.7424, **test-once
  0.7079 vs the previous record 0.7035**. Evolution adopted all three levers:
  156/403 keepers gated (conditional routing), 421 meta-terms reading other
  genomes' outputs. Energy economy in band (~6 starved/gen) throughout.
  Honest caveat: cumulative val-test gap now 0.034 (906 freeze decisions
  against the same 10k split across two stages) — a fresh val split per
  stage is the known next hygiene fix. Pod seed-13 in dry-out (~round 93);
  its push-80 fires on completion. Ladder: Coates-Ng 0.5904 → v1 0.6378 →
  v2 0.7035 → **v3 stack 0.7079**.
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
- **[2026-07-14] (Claude)** — Evolution campaign part 4: **environment
  richness knob** (user asked how much lift evolution really provides).
  `phase_b(C_env=…)`: doubling the patch-PCA environment 40→80 components,
  same recipe: 8k test unchanged (0.5840 — saturated by train size at this
  scale; val 0.661 vs 0.651) but **full-50k transfer 0.6257** vs 0.6198
  (+0.6). Current bottlenecks are evolution-data size and genome
  expressivity (single 6×6 patch scale, depth-1 bends), not component
  count. NOTE: `evo_interact_cifar.json` now holds the C=80 genomes
  (667); `phase_b_full` reads `C_env` from the export. Ladder: hand-crafted
  0.5904 → evolved 0.6198 (C=40) → **0.6257 (C=80)**.
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
- **[2026-07-14] (Claude)** — **Cousin finder generator de-aliased per the
  independent validation report** (`documentation/validation_report.txt`).
  The schematic's lens arithmetic (idx%14, (idx*13)%20, (idx*17)%20) repeats
  every 140 indices — 76 of the demo's 216 programs were exact duplicates and
  the inner parent (idx*7+3)%14 only ever took 2 of 14 values — so its
  "cousins" were mostly generator periodicity, as the report proved. lensAt
  parameters are now hash-mixed from the index (still deterministic: same
  index, same lens, forever); trivial self-maps no longer count as cousin
  pairs. Verified by script: 216/216 distinct programs, 14/14 parents used in
  both slots, 197/216 distinct signatures (19 genuine behavioral twins), and
  defaults now yield 5 real cousin pairs vs ~100 inflated (96 were
  self-maps). Run reports tag generator "hash-v2". Static files only.
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
- **[2026-07-13] (Claude)** — Baselines maps: **failed cards are now
  click-to-retry.** Diagnosis of "still only text loads after restart": the
  live server answers all five domains 200/valid-JSON in <0.6s (verified with
  curl) — the visible errors were fetches fired by the still-open tab while
  Flask was down, and the view caches after first load so toggling never
  refetched. Failed map labels are now clickable retries; a failed baselines
  load no longer marks the view as loaded. A plain page refresh (F5) shows
  everything.
- **[2026-07-13] (Claude)** — Baselines maps: **NaN crash fixed + bigger +
  zoomable.** Root cause of "only text loaded, others crashed": `np.corrcoef`
  inside `_sig`/`build_map` emitted invalid-divide NaNs on spiky domain
  streams, and a NaN in `radius_vs_nonlinearity_corr` made jsonify produce
  invalid JSON (bare `NaN`), killing the fetch. Replaced every corrcoef with a
  `_safe_corr` (no numpy divide path at all, degenerate → 0.0); verified all
  five domains build with warnings-promoted-to-errors and json.dumps clean.
  GUI: card maps 320px tall (cards min 440px wide), wheel-zoom added
  (0.3x-12x), and the four domain-map requests now load SEQUENTIALLY (a
  promise chain) instead of hitting Flask concurrently.
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
- **[2026-07-12] (Claude)** — Lens map: **fixed "no render" + added per-layer
  frame rotation.** Render bug was two things: (1) `_load()` did a full SVD of the
  8000×3072 pixel matrix (~800MB U, effectively hung) — replaced with randomized
  top-64 PCA (`_rand_pca`, 0.9s); (2) the scatter colormap normalized `struct`
  (kurtosis) by its max, which a few extreme-kurtosis lenses dragged so high that
  every other point rendered at near-background → invisible; fixed with a visible
  slate low-end + outlier-robust saturating scale (`1-exp(-k·v)`), plus a
  continuous rAF redraw for the live map. NEW: **layer-view rotation** (user's
  point — stacking was feeding samples without rotating the frame). A "Next
  layer's view" control (same 0° / step 15/30/45/90°) applies a Givens frame
  rotation before a layer sweeps, so a stacked layer sees the same features from
  a rotated angle; the rotation is stored per frozen layer so `_layer_apply`
  reproduces it. Endpoint + `start(rot_deg=…)`. Needs Flask restart.
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
- **[2026-07-12] (Claude)** — LM round 3: "features are the environment" —
  sem_next + grammar_real genomes in a PPMI/eig feature space built from the
  training corpus itself (no lookup tables in the model, per user rule: word
  choice at inference = the genome scoring the WHOLE vocab through the fixed
  space; round-2 follower pools not consulted). New `genreg_train/lm_sem.py` +
  `run_lm_sem.py` (SS II templates in the docstring), all training/testing on
  the I2 primary (6 smokes + 2 full runs; smoke mode = job arg). GA fixes that
  came out of the smoke ladder, all diagnosed from flat-fitness evidence:
  (1) relative per-tensor mutation — absolute sigma ~= init scale had made every
  mutation a re-randomization; (2) fixed 2048 probe batch (resampled per 25 gens)
  — batch noise was 9x the genome-to-genome signal, selection was a coin flip;
  (3) fitness EMA per lineage; (4) SS VI bootstrap inits (identity Wq, near-no-op
  bilinear); (5) logfreq as an environment feature behind ONE evolved weight.
  RESULTS (runs at `runs/lm/20260712-202708-lm-*`): grammar_real 56.3% balanced
  (real signal, climbs steadily). sem_next run 1 (no echo pressure): 30.1% vs
  25.3% baseline — first time the majority-frequency bar was beaten — BUT the
  champion was the gen-0 bootstrap and generation self-looped ("the the the").
  Run 2 (+echo negatives): evolution genuinely climbs (0.128->0.233) and loops
  are gone, but final acc is below the (now-harder) 26.6% baseline and the
  evolved per-word bias got REWARD-HACKED into context-free word soup
  ("cat/bitch/dog" boosted everywhere) — the vbias Goodhart from SS XI,
  reproduced in a new architecture exactly as SS IV.4 predicts. Known deviation:
  starved band never reached (energy compresses to ~0.3, culls nothing) despite
  birth-cost + EMA redesign of ga_step_energy. Honest verdict: environment
  validated, grammar genome validated, word-choice genome not yet — next levers
  logged (context-gated bias instead of static; two-phase freeze per SS VI;
  transition-term-only ablation). generate() switched to the genome-only path
  (sem propose + grammar rerank); lm_service loads lm_sem.pkl alongside
  lm_intent.pkl. Live /lm needs a **Flask restart**. Also: run_job.py poll()
  made cp1252-safe (root cause of the 2026-07-11 session crash).
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
- **[2026-07-12] (Claude)** — New `/plan` page: personal day-plan tracker for
  Sunday 2026-07-12. Self-contained template (`templates/plan.html`) + route in
  `app.py`. Each schedule block has a tap-to-cycle status (DONE / PARTIAL /
  SKIPPED) and a notes field; includes fallbacks, an add-row unscheduled break
  log, end-of-day scorecard, and patterns section. All state persists in
  browser localStorage (no backend); monochrome/plain-word status chips, no
  emojis. Added to the shared top nav (`_nav.html`) as "Plan".

- **[2026-07-11] (Claude)** — LM v2: GENREG_RULES compliance pass + hard negatives +
  rerank generation, retrained (crash-recovery session: the code rewrite was found
  complete on disk from a session that crashed before launching the run — root cause
  of the crash found and fixed: `run_job.py`'s log watcher died on a cp1252
  UnicodeEncodeError streaming remote logs; poll() now reconfigures stdout with
  errors="replace"). Changes trained: (1) soft fitness — all three trainers select on
  mean log-prob of the true answer instead of argmax accuracy (§IV.1); (2) energy
  homeostasis in ga_step (§III): rank-percentile energy, starvation culling below 0.2
  — observed starved=0 all run, band (3-15%/gen) NOT reached, needs tuning next
  round; (3) HARD negatives for next_word: drawn from words that actually followed
  the same preceding word in the corpus, plus a majority-frequency baseline as the
  honest bar; (4) rerank generation (§VI): generate() scores only the previous
  word's top-200 follower pool (mined into the artifact), never a full-vocab softmax.
  RESULTS (job 9888e902aea0ae02 on the I2 primary, 80.5M tokens): punctuation/opener/
  length genomes ~unchanged (opener_question 0.7015 still best). **next_word: 21.16%
  vs 16.67% chance BUT vs 26.92% majority-frequency baseline — honest negative: the
  genome does not yet beat "pick the most frequent candidate" on hard negatives.**
  fill_word 21.92%. QUALITATIVE win: generation no longer repetition-loops — rerank
  over follower pools produces varied, near-grammatical short sentences ("The road
  signs for me.") with correct intent-driven end marks. 10 runs recorded at
  `runs/lm/20260711-233847-lm-*/`. Live /lm needs a **Flask restart** to pick up the
  new artifact + rerank generate().
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

- **[2026-07-10] (Claude)** — Characters are now SIDE-PROFILE + episode retimed.
  Generated humanoids switched from front-facing (flip was invisible) to profile
  facing +x: one eye/nose/mouth on the leading edge, hair capped over top+back,
  narrower torso, cap visor and glasses forward — so walk direction, flip, and
  face-offs finally READ. Walk cycle looks like actual walking in profile. Episode
  001 rigs regenerated (same seeds), all 8 shots retimed with holds (12/13/14/11/
  13/12/12/9s = 96s total, was 56.5s) incl. a second futile wall-try beat in the
  shadowban shot; the-glitches-ep1.mp4 re-rendered (96s). Existing saved rigs keep
  the old front-facing look until regenerated.
- **[2026-07-10] (Claude)** — STORYBOARD mechanism + first full episode. New story
  layer: a story = ordered list of saved scenes, rendered into ONE mp4 in a single
  encoder pass (all shots rasterized at the first scene's size/fps; no stitching
  needed). Store runs/video/stories, endpoints /api/anim/stories (+delete) and
  /api/anim/render_story, Storyboard card in the Scenes view (add/reorder/remove
  shots, save, render). CONTENT: "The Glitches — Incident 001: Withdrawal" built as
  proof — rig ensemble (ep1-norm, ep1-corrector, custom ep1-atm with door-tagged cash
  tray, ep1-wall) + 8 saved shots (street title / tolerated no-clip through wall /
  ATM abuse with cash eject + LEDGER box / enforcement-log still / shadowban /
  corrector arrives / deallocation fade / thesis card) + saved story
  the-glitches-ep1 -> rendered the-glitches-ep1.mp4 (56.5s, 1280x720@24) in the
  library. Audio deliberately skipped (user records VO separately). **Flask restart
  required** (anim_service.py + app.py changed).
- **[2026-07-10] (Claude)** — OPEN/CLOSE verbs for objects (doors etc.). Two new part
  tags: "door" (slides by dx/dy) and "hinge" (rotates around its pivot by angle).
  New verbs open/close animate any parts with those tags: openness 0..1 persists
  between actions (open -> stays open -> close returns), amplitudes from the open
  action's args (defaults dx=-60, angle=-100). Door archetype rebuilt: sliding panel
  tagged "door" over a dark opening. Pose model gained per-tag translation — updated
  in BOTH anim_service.py and animrig.js. Objects' verb list is now move/fade/open/
  close; characters can also use open/close on tagged parts. "Open" verb-test button
  on the rig stage. Verified: slide door + hinged crate lid render open/close
  correctly. **Flask restart required.**
- **[2026-07-10] (Claude)** — SCENE TEMPLATES + 8 new object archetypes. "Generate
  scene" in the Scenes sidebar: basic / office / forest / city templates (seeded).
  Each generates a bg palette + procedural prop rigs (named <scene>-<archetype>-<n>,
  saved to the rig library and placed as objects — everything stays editable; same
  name regenerate overwrites the props). New archetypes: tree, bush, rock, desk,
  plant, whiteboard, building (random lit-window grid), streetlight. New endpoint
  POST /api/anim/generate_scene; templates listed via /api/anim/status. Verified:
  all four templates render on-style with a character for scale. **Flask restart
  required** (anim_service.py + app.py changed).
- **[2026-07-10] (Claude)** — Scenes: actors and objects separated. New "Objects"
  card with its own add-dropdown (object-kind rigs only); the Actors card takes
  character rigs only. Objects are restricted to move/fade verbs (action editor
  filters the verb list by target kind and coerces invalid verbs when retargeting);
  object ids prefixed "o", labelled "(obj)" in action dropdowns. Scene JSON is
  unchanged (one placement list; kind comes from the rig), so old scenes and both
  renderers are untouched. Frontend-only, no restart.
- **[2026-07-10] (Claude)** — Scene action SEQUENCING: actions now run top to bottom.
  Each action has a start mode — "after prev" (chains when the previous ends, the new
  default), "with prev" (ties to the previous action's start for parallel execution,
  e.g. two actors talking at once), "at time" (explicit t0; what old scenes migrate
  to, so nothing changes for them) — plus a duration field and ▲▼ reorder buttons.
  t0/t1 are resolved in the UI and stored, so scene JSON and both renderers are
  untouched; scene duration auto-grows to fit the chain. Frontend-only, no restart.
- **[2026-07-10] (Claude)** — Rig import/export: "Export .json" on the rig stage
  downloads the open rig (including unsaved edits) as a formatted JSON file;
  "Import .json" in the sidebar accepts one or more rig files (validated: needs a
  parts list; name/kind/canvas defaulted), saves them to the rig library and opens
  the last one. Frontend-only (animstudio.js + video.html), no restart needed.
- **[2026-07-10] (Claude)** — Rig editor UX round: (1) UNDO/REDO — Ctrl+Z / Ctrl+Y
  (Ctrl+Shift+Z too) in all three /video views: rig edits, scene edits, editor
  timeline; snapshot stacks per document, cleared on open, native text-undo left
  alone while typing in fields. (2) Parts panel + selected-part form now FLOAT over
  the rig stage (collapsible overlay panels) instead of a card below. (3) On-stage
  transform gizmos: selected part gets a rotate handle (round, gold) and a resize
  handle (square, blue) around its pivot; new part properties `rot` (base rotation,
  applies to the subtree like verb rotations) and `scale` (uniform, shape only so
  children don't inherit) — added to BOTH anim_service.py and animrig.js renderers,
  plus rotate/size fields in the part form. Old rigs without rot/scale unchanged.
- **[2026-07-10] (Claude)** — Animation rigs: two-segment limbs. Generated humanoids
  now have elbows and knees (thigh/shin + upper-arm/forearm, capsule segments
  overlapping at the hinge; new tags `leg_*_lower` / `arm_*_lower`). Walk verb bends
  knees during swing and gives elbows a soft counter-bend — updated in BOTH
  anim_service.py and animrig.js. Old single-segment rigs stay compatible (unknown
  tags simply don't rotate). Verified with a 5-phase walk strip render.
- **[2026-07-10] (Claude)** — /video is now an ANIMATION PLATFORM (SCP-Explained flat
  style): three views — **Rigs** (manual part-by-part SVG puppet editor: layered parts
  with parent/pivot/z, semantic tags arms/legs/mouth shapes, drag-to-position, live
  verb test idle/walk/talk/point; plus a procedural generator with 10 seeded archetypes:
  researcher/guard/dclass/suit/civilian + crate/table/door/terminal/containment),
  **Scenes** (actors placed by drag, verb action timeline walk/move/talk/point/fade,
  caption/title/infobox overlays, optional voiceover audio muxed from the library,
  scrub+play preview), **Editor** (the original cut/stitch/convert editor). Scene
  renders: per-frame SVG -> resvg -> ffmpeg pipe -> mp4 lands in the library for the
  timeline. Verb/pose math lives twice ON PURPOSE: anim_service.py (render) and
  static/animrig.js (preview) must stay in sync. New: anim_service.py, animrig.js,
  animstudio.js, /api/anim/* routes; library now accepts audio (mp3/wav/…) for
  voiceover tracks. resvg-py added to requirements. Verified end-to-end (generate ->
  scene -> render -> probe). **Flask restart required.**
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
- **[2026-07-09] (Claude)** — NEW: /video page — ffmpeg-backed video editor (plain
  utility, nothing evolved). Library (upload any format, probe via ffprobe, cached
  thumbnails, download/delete), clip timeline with per-clip in/out cut points +
  reorder + stitch-export (concat filter: mixed codecs/resolutions/framerates
  normalised, silent inputs get a silence track), per-file cut-to-file and container
  conversion (mp4/mkv/webm/mov/m4v/avi/gif), background jobs with live progress
  (`-progress pipe:1`) and cancel. Backend `video_service.py`, routes in app.py
  (`/api/video/*`), files under `runs/video/library`. ffmpeg full build extracted from
  the user's downloaded 7z into `tools/ffmpeg-2026-07-09-.../bin` (gitignored;
  resolution order: tools > PATH > imageio-ffmpeg bundle). All ops verified end-to-end
  with generated test clips. **Flask restart required** to pick up the new routes.
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
- **[2026-07-09] (Claude)** — **New genome group "next" — intent-conditioned, autoregressive
  next-word prediction. Honest negative result: it barely beat fill_word (24.25% vs 24.08%),
  and generation is still repetitive.** Per user directive: revive the core idea from the
  archived pipeline's Selection/Bidirectional-Selection genomes (bilinear contrastive word-
  pair scoring), rebuilt fresh with two real fixes over `fill_word`: (1) properly
  AUTOREGRESSIVE — left context only, matching how `generate()` actually runs (fill_word was
  trained bidirectionally but used with zero-padded right context at inference, a genuine
  train/inference mismatch); (2) INTENT-conditioned — the sentence's target end-mark fed in as
  an explicit small embedding, a signal fill_word never had. Scope, decided via
  AskUserQuestion: next-word prediction ONLY this round (deferred No-repeat/Agreement/
  Alternation), negatives kept simple/random (deferred harder same-context negatives) — both
  explicit choices, not oversights. New `mine_next_word_examples()` (flat word array + parallel
  per-word intent array built via one O(n) pass with vectorized per-sentence backfill, avoiding
  the earlier quadratic-mining mistake), new `NextWordPop` class (word embedding + small 3-way
  intent embedding + bilinear query-vs-candidate scoring). Result: **24.25% holdout accuracy —
  essentially tied with fill_word's 24.08%, not a meaningful lift.** `generate()` switched to
  use `next_word` for word choice (fill_word kept trained/shown for comparison only). Live
  check confirms the numeric near-tie is real, not noise: generation is STILL repetitive
  ("fifa...fifa", "networks...networks", "moon...moon", "watching...watching",
  "antonio...antonio", "ministry...ministry" — different failure instances, same failure mode).
  **Useful negative result, not a wasted round**: fixing the train/inference mismatch and
  adding intent-conditioning did NOT move the needle, which points the finger away from those
  two issues and toward the deliberately-deferred lever — negative sampling. Random negatives
  are likely too easy to beat via a handful of generically "safe" embedding directions,
  regardless of whether the genome is bidirectional or autoregressive, intent-aware or not.
  Next real lever for fluency: harder (same-context-class) negatives, per the option explicitly
  deferred this round. All 10 genomes (5 punctuation + 2 opener + 1 length + 1 fill + 1 next)
  recorded at `runs/lm/20260709-213338-lm-*/`. `/lm` page gets a "next" genome group card;
  Fill card re-labeled as comparison-only. **Flask needs a restart.**

- **[2026-07-09] (Claude)** — **First generation mechanism: hangman-style, variable-length,
  intent-driven — mechanism works, word fluency doesn't yet.** Per user's mental model (planned
  via EnterPlanMode, approved): a human knows what they want to say (intent) before the words
  come; like Hangman, guess pieces to fill blanks, except the number of blanks isn't known
  upfront — length **emerges dynamically**, no separate length-prediction genome, decision made
  explicitly via AskUserQuestion over 3 alternatives (upfront-sample / dynamic-growth /
  capped-then-trim — dynamic growth chosen). Two new genome groups: **`length`**
  (`length_continue` — given a partial sentence, is it complete or does it need to keep
  growing? needed a NEW explicit length feature on `ClassifierPop`, since mean-pooling the
  context window alone can't tell a 3-word prefix from a 10-word one apart; mined via bounded
  prefix-sampling per sentence, 4 negative prefixes + 1 true full-length positive each, to stay
  linear in corpus size) and **`fill`** (`fill_word` — given the words around a blank, does a
  candidate word fit; deliberately a CONTRASTIVE discriminator — true word vs 5 random corrupted
  negatives — not a 4000-way softmax, which would've reintroduced the exact class-imbalance
  collapse the punctuation group hit; new `FillPop` class, bilinear query-vs-embedding scoring,
  same tournament-GA machinery). Real design tension surfaced and resolved: `length_continue`
  is trained on strictly-growing, fully-real prefixes, so true out-of-order multi-blank infill
  (as sketched during planning) would feed it a context distribution it never saw — narrowed
  scope to left-to-right fill with confidence-driven grow/stop and full-vocabulary-scored word
  choice (not fixed-length, not forced-greedy), flagged as a real scope reduction from the
  original plan, not silently shipped as the full vision. Results: `length_continue` **61.1%
  balanced accuracy** (chance 50%), recall even across both classes (continue 67.8%, end
  54.3%) — no collapse. `fill_word` **24.1% holdout accuracy** (chance 16.7% — beats it, but
  only modestly). Live generation confirms the mechanism genuinely works — variable length (3-6
  words across samples), correct intent wiring (question-opener words → `?`, "please" → `!`,
  "never" → `.`) — but word choice is honestly weak: heavy repetition of a handful of generic
  words ("product" recurring across unrelated seeds) rather than context-appropriate fills,
  consistent with fill_word's modest accuracy and looking like a classic contrastive-learning
  shortcut (beating 5 random negatives is often easy via a few generically "safe" embedding
  directions without the genome actually nailing true left/right context). Per the
  `lm-diffusion-vision` memory's stated bar (consistency, not perfection): the SKELETON is
  honest and working; word fluency is the clear next lever — likely needs harder/frequency-
  matched negatives (current ones are uniform-position, possibly too easy), more capacity, or
  longer training, not a different architecture. New `/api/lm/generate` endpoint; `/lm` page
  gets a "Generate" section (seed word + temperature, live per-tick trace) plus length/fill
  genome cards. All 9 genomes (5 punctuation + 2 opener + 1 length + 1 fill) recorded at
  `runs/lm/20260709-161129-lm-*/`. **Flask needs a restart to pick up the new page section and
  endpoint.**

- **[2026-07-09] (Claude)** — **New genome group "opener" — reads a sentence's FIRST word to
  confirm its eventual intent, and it's the strongest genome trained yet.** User's observation:
  the `punctuation` genomes only look at words BEFORE a mark; the sentence's opening word
  ("who/what/where/how/can/don't...") is a separate, cheap, strong signal the pipeline wasn't
  using at all. Added 2 binary genomes mirroring `punct_question`/`punct_exclaim` exactly, but
  reading FORWARD from the sentence start instead of backward from the mark: `opener_question`
  (does the first word predict a `?` ending?), `opener_exclaim` (does it predict `!`?). New
  `mine_opener_examples()` splits the token stream into sentences (delimited by `. ! ?` only —
  commas/semicolons/colons don't reset "first word"), one example per sentence = (first word,
  eventual end mark). ctx_k=1 by design — literally just the opening word — and the existing
  `ClassifierPop`/`train_classifier` code needed zero changes, since mean-pooling one embedding
  is a no-op. Result: **`opener_question` 70.6% balanced accuracy — beats every punctuation-
  group genome including `punct_question`'s 67.2%.** `opener_exclaim` 61.4%, also beats its
  mirror `punct_exclaim` (58.5%). Confirmed live: "who"/"what" → 88-92% confident `?`; "the"/
  "never" → 96-98% confident `.`; "please" → correctly favors `!` (42.8%) but not confidently;
  "wow" → still misses (70.6% `.` instead of `!`) — an honest limitation matching
  `opener_exclaim` being the weaker of the two new genomes, not hidden. New
  `/api/lm/recognize_opener` endpoint + a "Confirm" box on `/lm`; `lm_service.py`'s `status()`
  reshaped from a flat genome list into `groups: [{group, genomes}]` so punctuation and opener
  render as separate sections. Caught and fixed a real bug in `record_lm_run.py` while adding
  this: the run-recording script hardcoded `meta.json`'s `"group"` field to `"punctuation"` for
  EVERY genome regardless of which group it actually belonged to — the first write silently
  mislabeled the 2 opener runs; deleted and rewrote with `group=split_result["group"]`. All 7
  genomes (5 punctuation + 2 opener) now correctly recorded at
  `runs/lm/20260709-124755-lm-*/`. **Flask needs a restart to pick up the new page section and
  endpoint.**

- **[2026-07-09] (Claude)** — **Split the punctuation-intent genome into 5 small binary
  specialists — the collapse (0.15% exclaim recall, 2.5% colon recall) is fixed.** Per user
  directive: the single 6-way classifier's failure mode was structural — one softmax head
  arbitrating 6 wildly different base rates (periods 5.7M vs colons 99K) let two easy, frequent
  classes eat all the capacity. Decomposed into 5 genomes, each with ONE clean binary survival
  condition, all sub-labeled under the group **"punctuation"**: `punct_end` (end . ! ? vs
  continue , ; :), `punct_question`/`punct_exclaim` (which end-mark, two binary heads instead
  of a 3-way softmax), `punct_semicolon`/`punct_colon` (which continue-mark, same pattern) —
  the same decompose-into-specialists recipe that was the one clean win in the archived
  /evolang pipeline. `recognize_mark()` composes the 5 heads' probabilities hierarchically back
  into a single ranked guess over all 6 marks. Each binary problem trains AND evaluates
  class-balanced by construction, so the earlier train/eval mismatch bug (previous two entries)
  structurally can't recur the same way. Result, all 5 beat 50% chance and — the actual fix —
  **none collapsed**: `punct_end` 56.9% (continue 56.6%/end 57.1%, even), `punct_question`
  67.2% (strongest genome, real signal), `punct_exclaim` 58.5% with **exclaim recall 62.7%**
  (was 0.15% before the split), `punct_semicolon` 63.5% (semicolon recall 30.2% — genuinely
  harder, still honest, no collapse), `punct_colon` 55.0% (colon recall 47.9%, up from 2.5%).
  Confirmed live: "please bring the following items" now correctly gets `:` at 80.8% — exactly
  the class the old genome couldn't recognize at all; "i cannot believe you actually did that"
  still ranks `?` over `!` but `!` now carries 30% probability instead of ~0% (partial, honest
  progress, not a full fix). `genreg_train/lm_intent.py` generalized `IntentPop` into
  `ClassifierPop` (n_out-parametric) and added `SPLITS`/`prepare_split`/`recognize_mark`;
  `run_lm_intent.py` trains all 5 off one shared corpus load/mine pass; `lm_service.py`
  composes them for `/api/lm/recognize`; `record_lm_run.py` now writes one run per genome
  under `runs/lm/`, each tagged `group: "punctuation"` via `meta.json` so they're visibly
  grouped on `/runs`; `templates/lm.html`/`static/lm.js` show all 5 genomes with per-class
  recall instead of one confusion matrix. **Flask needs a restart to pick up the new page.**

- **[2026-07-09] (Claude)** — **LM genome #1 retrained with the train/eval split fixed —
  balanced accuracy 26.5% (vs 16.7% chance), but the genome collapsed to mostly two classes.**
  Root cause of the entry below's weak, declining holdout number: champion selection during
  training picked the genome with the best RAW accuracy on the natural (period/comma-heavy)
  holdout distribution, while training batches were class-balanced — a genome that gets BETTER
  at rare marks (semicolon/colon) necessarily spends more guesses on them and its raw accuracy
  on a period-heavy holdout goes DOWN, so training was actively selecting against real learning.
  Fix: added `IntentPop.balanced_accuracy()` (mean per-class recall) and made it drive BOTH
  champion selection and the headline metric, matching the class-balanced training batches;
  raw natural-distribution accuracy is still logged for interpretability but no longer picks
  the winner. Result: held-out balanced accuracy **26.5%**, up from 17.8% and now a real,
  stable improvement over chance (train-batch-acc and holdout-balanced-acc rose together and
  plateaued around gen 250, instead of holdout collapsing below chance as before). **Full
  honesty on what that number means, though**: per-class recall is `.` 13.2%, `!` 0.15%(!),
  `?` 63.7%, `,` 11.8%, `;` 67.5%, `:` 2.5% — the genome didn't become a well-rounded 6-way
  recognizer, it collapsed toward confidently predicting `?` or `;` for almost everything
  (confirmed live: "i cannot believe you actually did that" gets 99.9% `?`, when `!` is
  obviously right) and nearly gave up on `!`/`:` entirely. Balanced accuracy went up because
  those two classes are now genuinely well-recognized, not because the genome got well-rounded.
  Two real, distinguishable signals exist in the corpus (question words, subordinate-clause
  cues) and the genome found them; exclaim and colon apparently need either more capacity,
  better negative sampling, or a per-class weighted fitness (not just balanced batches) to
  stop being sacrificed. Re-fetched artifact + log; new run recorded at
  `runs/lm/20260709-110038-lm-intent02/` alongside the original (kept for comparison, not
  overwritten). `genreg_train/lm_intent.py`, `run_lm_intent.py`, `lm_service.py`,
  `record_lm_run.py`, `templates/lm.html`, `static/lm.js` all updated to surface both
  balanced and raw accuracy. **Flask needs a restart to pick up the display changes.**

- **[2026-07-09] (Claude)** — **LM genome #1 (intent recognition) trained — honest first
  result: 17.8% held-out accuracy, barely above the 16.7% chance floor.** Caught and fixed a
  real bug first: `mine_examples()` rebuilt `[0]*ctx_k + recent` from the ENTIRE accumulated
  word list on every mark occurrence instead of the last `ctx_k` words — O(1) on the 3M-char
  smoke test, silently quadratic on the real 417MB corpus (job never progressed past "mining"
  in 30+ min; killed, fixed with a `collections.deque(maxlen=ctx_k)` ring buffer, re-verified
  at 9.4M tokens/~4s before redispatching). Full run then completed clean: 11M examples mined
  (class counts wildly skewed — periods 5.7M, commas 4.2M, vs semicolons 133K, colons 99K),
  400 generations, pop 150, ~15 min on the I2 primary. Result is a real, not-yet-useful signal:
  the confusion matrix shows the genome over-predicts the rare classes (`;`/`:`) across every
  true row — a train/eval distribution mismatch (class-balanced training batches vs the
  natural-distribution held-out set), not pure noise, but not a working recognizer either.
  Per-class recall: `.` 17.8%, `!` 18.2%, `?` 11.3%, `,` 18.1%, `;` 34.8%, `:` 25.2%. Artifact
  fetched to `corpora/combined/lm_intent.pkl` (+ `.log`); confirmed the `/lm` page's
  `recognize()` path runs end to end (probabilities cluster 0.13–0.23, no confident signal —
  consistent with the accuracy). Recorded as a run at `runs/lm/20260709-024735-lm-intent01/`
  (new `genreg_train/record_lm_run.py` — one-off log→runstore-format converter; no
  `checkpoint.pkl`, since this genome's format isn't the RL-engine's `engine_api` format, the
  real artifact is the fetched `.pkl`) so it shows up as an "lm" tab on `/runs` next to the
  RL-engine environments. Open question for next session: is class-balanced training the wrong
  choice here, or does eval also need to be class-balanced for a fair read of what genome #1
  actually learned?

- **[2026-07-09] (Claude)** — **New `/lm` page — the rebuild starts, genome #1 only.** Per
  user directive: start completely over, no code carried forward from the archived /evolang
  pipeline, one genome at a time. Genome #1 is **intent recognition**: given the last few words,
  recognize which punctuation mark's intent belongs right after them — end-statement (.),
  end-question (?), end-exclaim (!), or "more to say" (comma / semicolon / colon). Recognition
  only, no generation, no grammar, no word selection — deliberately that narrow. New files, all
  self-contained (no import from the archived `wordpipe.py`/`evolang.py`, on purpose):
  `genreg_train/lm_intent.py` (mining + the genome itself — word-embedding → mean-pool → tanh
  hidden → 6-way class logits, tournament selection + elitism + self-adaptive mutation, class-
  balanced batches since commas dwarf semicolons/colons ~150:1), `genreg_train/run_lm_intent.py`
  (I2 job driver, trains on the kept combined corpus), `genreg_train/lm_service.py` (Flask
  wrapper). New `templates/lm.html` / `static/lm.js`, `/api/lm/status` + `/api/lm/recognize`
  in `app.py`, nav link added. Local smoke test (3M-char slice, tiny pop/gens) confirmed the
  pipeline runs end to end; real training (400 gens, pop 150, full 417MB corpus) dispatched to
  the I2 primary (job `7321d408997023cd`) — artifact lands at `corpora/combined/lm_intent.pkl`,
  not yet fetched back as of this entry. **Flask needs a restart to pick up the new route.**

- **[2026-07-09] (Claude)** — **Archived the WordPipe /evolang pipeline** for a ground-up
  rebuild, per user directive: fluency ceiling never moved across every architecture tried
  this session (forward, meaning-first, intent-first/backward, crystallized, clause-obligation
  tracking — see `documentation/WORDPIPE_FIELD_NOTES.pdf` for the full record and every verdict).
  Moved all pipeline code (`genreg_train/wordpipe_service.py`, `wordpipe.py`, `genelib.py`,
  `evolang.py`, all ~45 genome/experiment modules, `templates/evolang*.html`,
  `static/evolang*.js`, `demo/demo.py`, trained genome `.pkl` artifacts) to `archive/evolang_v1/`
  via `git mv`; removed the `/evolang` routes, websocket handler, and nav link from `app.py`/
  `templates/_nav.html`/`static/app.js`. Explicitly kept in place (datasets, not pipeline code):
  `corpora/wikipedia/`, `corpora/combined/` (corpus text + build script), and
  `project/conversational/` (Cornell Movie Dialogs source). See `archive/evolang_v1/README.md`
  for what's archived, what's kept, and known dangling references in unrelated historical
  build scripts. **Flask needs a restart to pick up the app.py route removal.**

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
- **[2026-07-08] (Claude)** — Clause-completeness experiments (crystallize forward-polish
  pass + clause-obligation tracker), both real attempts, neither a clean win. CORRECTION:
  the earlier "verified" intent-first samples ran with `self.champs` silently empty on the
  I2 primary (`demo/genomes.pkl` isn't pushed there, verification script never overrode
  `Service.CACHE`) — real backward genomes + real combined-corpus vocab, but Alternation/
  Agreement/Semantic reranks were completely inert, undisclosed at the time. Caught by the
  crystallize A/B test producing suspiciously byte-identical output; fixed in
  `run_verify_crystallize.py`/`run_clause_obligation.py`. **Crystallize**: mixed result
  (func-func-adjacency -0.026 better, dangling-rate +0.068 worse, no visible readability
  gain) — stays OFF by default. Also fixed a real crash (class/word list misalignment when
  backward growth skips `<unk>`-class positions). **Clause-obligation tracker**: real but
  small effect (never-closed-relative-rate 13.1%→12.5%, plateaus immediately) at a real
  cost (dangling-rate worse) — cut, same failure family as the original open-obligation
  tracker (a correctly-targeted idea, blunt implementation). Clause-completeness remains
  open. Full writeup in `genomes.txt`; flow map updated (`static/evolang_layers.js`).

- **[2026-07-08] (Claude)** — Fixed the changelog modal's "select by project" toggle
  missing on `/evolang`, `/evolang/layers`, and `/mnist`. `static/app.js`'s `PROJECT`
  lookup only mapped 5 routes (`/`, `/tree`, `/diff`, `/animation`, `/i2`) even though
  `documentation/changelogs/` has 8 per-project changelog files including
  `CHANGELOG_EVOLANG.md` and `CHANGELOG_MNIST.md` — on those two pages the toggle button
  never got created (gated on `PROJECT` being non-null), so only the flat main changelog
  was ever reachable. Added the missing three route mappings.

- **[2026-07-08] (Claude)** — Intent-first generation VERIFIED with real output. Full
  swap completed: corpus #2 (Wikipedia -> Combined = Wikipedia + Cornell Movie Dialogs,
  421MB, 24.4% dialogue) is now live (`demo/genomes.pkl` swapped, `evolang.py`'s
  `CORPUS_PATH` repointed, wiki-only backup preserved at `demo/genomes_wiki.pkl`). All 23
  genomes retrained on the combined corpus (I2 primary, job `b791766f592ff012`, ~94 min).
  Real, attributable effect of the dialogue mixing: exclaim-affinity training samples rose
  2,499 -> 133,321 (53x), corpus exclaim-rate in the punctuation-sequence mining rose
  0.11% -> 2.34% of all marks, question-rate 0.07% -> 6.69%. `generate_intent_first()`
  now produces real output with genuine question marks and an exclamation tied to actual
  discourse-skeleton marks ("...who?", "...whom?", "...who!") and combined-corpus-specific
  vocabulary ("youtube", "gimme") appearing — first time this session real intent-carrying
  punctuation has shown up in generation at all. Honest verdict unchanged from the
  architecture's design intent: does not fix fluency (still word salad, same clause-
  completeness gap), because it changes what anchors generation, not the underlying
  Order/Selection word-to-word mechanics. Full numbers for all 23 genomes and the verified
  sample in `genomes.txt`'s "Corpus swap #2" section; flow map (`static/evolang_layers.js`)
  gets a new "Intent" stage (5 new nodes: Punctuation sequence, Exclaim affinity,
  Order-backward, Selection-backward, Intent-first generation composed node).

- **[2026-07-08] (Claude)** — **PUBLISHED github.com/A1CST/GENREG_MNIST_2.0** — the full
  scientific package for MNIST-Pipe: **99.10% ± 0.05% test over 5 full-pipeline seeds**,
  10-cell ablation battery (standard GA baseline 92.60 vs shipped 99.22 on identical
  environment/budget; cold start -16.0; evolved bank +1.4; mag-scaled mutation carries the
  climb), charts (confusion matrix, ladder, seed variance, ablations, detector bank),
  METHOD/FAILURES/LIMITATIONS docs, checkpoints, clean-clone reproducibility verified.
  Local copy `Documents\GENREG-MNIST-2`; drivers `jobs/mnist_seeds.py` +
  `jobs/mnist_ablations.py`; details in CHANGELOG_MNIST.md.
- **[2026-07-08] (Claude)** — Intent-first generation architecture (user idea: "the
  punctuation mark IS the intent" — chosen before any word exists, structure grows
  backward to serve it, not forward hoping to land on one). Combined corpus:
  Wikipedia (316MB) + Cornell Movie Dialogs (17.1MB dialogue x6 repeat, ~24.4% of the
  final 421MB corpus) — Wikipedia alone lacks real question/exclaim register (an
  earlier probe found every emotionally-loaded word — wow/amazing/hooray/alas — was
  out-of-vocabulary; movie dialogue supplies real turn-taking, real "!"/"?" usage).
  New genomes: `intent_punct.py` (punctuation-sequence, tiny autoregressive model
  over {. , ; : ! ?}, reuses `wp.OrderPop`), `sent_type_exclaim.py` (generalizes
  `sent_type` to exclaim vs statement via a second binary genome, same
  decompose-into-binary pattern as Alternation/Agreement). Backward Order+Selection
  (`run_backward_experiment.py`) trained via a cache-reversal trick — zero new
  algorithm, same `wp.run_class_lm`/`wp.run_biselection` fed the corpus read
  backward, same word->class mapping so results stay compatible with the existing
  class table. New `Service.generate_intent_first()`: generates the punctuation
  sequence for the whole response FIRST, then grows each word-span BACKWARD from its
  mark toward the previous mark, with the mark's type (question/exclaim/statement)
  biasing what grows toward it from the first word chosen. `run_retrain_combined.py`
  retrains everything (11 core genomes + 7 structural-decomposition sub-genomes + 3
  intent genomes + fresh backward pair, 23 total) on the combined corpus, dispatched
  to the I2 primary — full swap, not a same-pipeline-new-corpus repeat (that mistake
  already happened once this session with the Wikipedia-only swap: fixed vocabulary,
  changed nothing structural, fluency didn't move). New `/api/evolang/intent_first`
  endpoint + "Intent-first (backward)" button on `/evolang`. Not yet verified with
  real output — training was still running when this was committed; see the next
  changelog entry for results once the primary job finishes.

- **[2026-07-08] (Claude)** — **MNIST-Pipe v4 FINAL: 99.03% TEST — gradient-free record
  broken** (user's prior best 98.5, MANTIS 98.55, LeNet-5 99.05). Evolved conv-detector
  environment (66 genomes) + evolved joint head (centroid warm start, full-train
  landscape, magnitude-scaled mutation): centroid 98.72 -> joint **99.03**; referees
  correctly gated off at margin 0. ~15.3K evolved params at inference, zero gradients in
  features or classifier. Backup `demo/mnist_genomes_v4_9903.pkl`; service now honors the
  pickle's feat_version. Full story: `documentation/changelogs/CHANGELOG_MNIST.md`.
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
- **[2026-07-08] (Claude)** — Structural genome decomposition, full observability (user
  directive, after 4 grammar-fix hypotheses all failed/mixed: "this is a living model...
  break up every single structural genome... gives us full observability... trace back to
  WHY its output is a specific way"). Assessed all shipped structural genomes; 5 had a
  genuine internal compound question: **Selection** -> Sel-backward/Sel-forward/
  Sel-frequency (no retrain — ML/MR/beta were already separate params; verified
  byte-identical output at default weights, both locally and on the I2 primary).
  **Order** -> Order-bigram (K=1, val_ppl 11.344 vs unigram 12.984) + existing K=4
  Order-context; Order-bigram trained but left UNWIRED (its artifact is a full class-LM,
  not a bilinear rerank — doesn't fit the reranks-tuple shape; kept as a standalone
  diagnostic scorer). **Alternation** -> Altern-rhythm (coarse content/function only, val_acc
  0.526, barely above chance) + Altern-func-chain (specific function->function legality,
  val_acc 0.668, much stronger). **Agreement** -> Agree-modal (0.768) + Agree-number
  (0.669). **Semantic** -> Sem-adjacent (distance-1, 0.671) + Sem-window (distance-2..4,
  0.539, confirming loose topical fit is genuinely harder than tight collocation).
  No-repeat/Opener/Closer/Boundary/Commas assessed and left as-is (already single-question,
  no real compound to split). Training split across both machines (Alternation/Agreement on
  the I2 primary, job `c510a82fb3c8d046`, ~23 min; Order-bigram/Semantic locally, ~14 min —
  user was done gaming, local heavy compute allowed again this session, still parallelized
  with the primary per their instruction). All 6 wired sub-genomes are ADDITIVE reranks
  gated on their parent's existing toggle (`_add_struct_order_reranks()`/
  `_add_struct_reranks()` in `wordpipe_service.py`) — transparent internal decomposition
  for traceability, no new `/evolang` UI toggles added. Verified locally: all 7 champions
  load correctly, generation runs without error. Documented in `genomes.txt` (new
  "Structural genome decomposition" section) and `static/evolang_layers.js` (composed
  sub-groups under the existing altern/agree/sem nodes).

- **[2026-07-08] (Claude)** — Corpus swap: Gutenberg -> Wikipedia, per user directive
  ("modern corpus" — the pipeline was trained on 19th-century Gutenberg novels, source of
  every "thou"/"shalt"/"grushenka" in generated output). `genreg_train/evolang.py`'s
  `CORPUS_PATH` repointed to `corpora/wikipedia/wiki_corpus.txt` (316MB, already on disk).
  Retrained all 11 core genomes (Order/Selection/Bidirectional/Boundary/Comma/Agreement/
  Alternation/Semantic/No-repeat/Opener/Closer) entirely on the I2 primary per explicit
  no-local-heavy-compute constraint (user was gaming) — `run_retrain_wiki.py`, job
  `5b3f26a088661145`, ~39 min, all stages clean. `demo/genomes.pkl` swapped to the
  wiki-trained champions; archaic-corpus original preserved at
  `demo/genomes_archaic_backup.pkl`. Verified with 10 real generated samples (also
  dispatched remotely, `run_verify_wiki.py`): vocabulary is genuinely modern now, zero
  archaic words. Grammar is at least as broken as before — heavy "of the X of the Y"
  chains, because the Wikipedia corpus is dominated by short formulaic geography/biography
  stub articles, MORE repetitive than the novel prose was. Vocabulary fixed; fluency did
  not improve — see "Grammar investigation" in `genomes.txt` for next steps. Flagged: this
  session's SVD-feature-dependent artifacts (collocation/parallelism/coherence champions,
  none ever wired) and `battery_round1.py`'s guardrail baselines are now stale relative to
  the new corpus — real follow-up, not done in this pass. Requires a Flask restart to take
  effect (not restarted here, per standing instruction).

- **[2026-07-08] (Claude)** — MNIST-Pipe **round 5: 98.21% test** (from 97.63) — the
  optimisation-gap campaign paid off. What worked, in order: (r3) fixed 16k pool — first
  climb past the warm start but the population MEMORISED the pool (champion NLL on pool
  ~-0.009 vs -0.086 on full train; flat test). (r4) user's **magnitude-scaled mutation**
  (`ga_step(mag_scale=True)`: each gene perturbed proportionally to its own |w| + 5%-of-mean
  floor) — found lighter better-fitting genomes (|W|^2 451->327) but same memorised pool, flat
  test. (r5) **full-55k deterministic landscape** (`--joint-pool 0`; the exact objective the
  98.53 closed-form probe optimises, GA as the only traversal): steady honest climb 98.24 ->
  98.50 val, TEST joint 97.83, +pairwise referees (re-gated margin 1.5) **98.21**. Val->test
  gap collapsed 0.95 -> 0.41. Also: joint fitness einsum -> single GEMM (32x faster/gen;
  55k-pool gen ~0.7s). Backups: `demo/mnist_genomes_r{2,5}.pkl`. 0.29 from the user's 98.5
  gradient-free record. ROUND 6 launched: **evolved per-neuron precision** (user's directive):
  `JointQPop` — 10 bit-width genes (3..16), symmetric linear quantisation INSIDE the fitness
  (straight-through latents), bit_cost 0.01 so precision must pay for itself; champion saved
  as the quantised model + per-neuron bits + effective KB (`--quant`). 12-bit warm start
  already holds val 98.50 (quantisation is free at 12 bits).
- **[2026-07-08] (Claude)** — Meaning-first generation: user-directed architecture flip.
  Diagnosis: the pipeline has been structure-first this whole time (Order picks a class
  skeleton blind, Fill picks whatever word fits, meaning is bolted on afterward as a
  rerank) — very likely why Sentence coherence / Theme consistency failed near-chance,
  since a linear rerank can't retrofit coherence onto a sequence never chosen for its
  content. New `Service.generate_meaning_first()` in `wordpipe_service.py`: a Content
  Selection stage runs BEFORE Order, picking 3-5 mutually-related content words via the
  existing relation genomes (hyper/mero/synant/sem, stochastic sampling not argmax), then
  the same evolved Order/Fill genomes place each reserved word into the first matching
  class slot instead of running word-selection there. No new training. Verified (not
  assumed): selected content sets score +0.58 higher on relatedness than random sets
  (t=8.25, 40 samples); placement rate 66.5% (133/200, 50 samples) — a third of chosen
  words don't find a matching slot and get dropped. Word-level fluency is explicitly
  unchanged — this fixes what gets said, not the surface grammar. New
  `/api/evolang/meaning_first` endpoint + "Meaning-first" button on `/evolang`. New
  "Content" stage added to the flow map (`static/evolang_layers.js`), ahead of Skeleton.

- **[2026-07-08] (Claude)** — Full guardrail battery run on round-1's experimental
  genomes (`genreg_train/battery_round1.py`) — correcting an earlier overstatement that
  "tested" meant fully validated. It didn't: earlier verdicts were probe + a lightweight
  1-2-metric spot-check, not the 4-metric (adj-hit rate, distinct-word ratio, dangling-
  ending rate, mean sentence length, 60-sample) battery the original 13 shipped genomes
  went through. Ran that battery on all four wired toggles + the Revision stage — every
  one regresses at least one guardrail, so none graduate to shipped:
  **Sentence type**: dangling-ending rate 20.8%→24.8%. **Sentence length plan**: dangling
  20.8%→26.0% — worse than Sentence type, and this CORRECTS the earlier "practically
  inert" verdict, which only checked sentence length (unaffected) and missed the real
  dangling-rate cost. **Pronominalization**: dangling 20.8%→24.0% — the "it"-substitution
  effect is real but has a fluency cost the earlier spot-check didn't measure.
  **Revision stage (Best-of-N)**: mean sentence length collapsed 14.7→7.8 words (-47%) —
  confirms the length bias flagged at build time was severe, not a minor caveat.
  `genomes.txt` and `static/evolang_layers.js` updated with the real numbers.

- **[2026-07-08] (Claude)** — /evolang UI updated for the genomes wired this session.
  `sent_type`/`lenplan`/`pronominal` toggles now appear in the genome-stack sidebar
  (LAYERS array in `static/evolang.js`) and the pipeline description card (ARCH array).
  New `/api/evolang/revision` endpoint + "Best-of-N (Revision)" button — calls
  `Service.generate_revision()` with the same layer toggles as Regenerate, plus
  `n_sentences`/`n_candidates` (defaults 6/6, capped at 12). Requires a Flask restart
  to pick up (not restarted here, per standing instruction).

- **[2026-07-08] (Claude)** — Autonomous genome-testing run: rest of the /evolang/layers
  roadmap cleared (genomes #5-10), plus a new Revision pipeline stage built.
  **Sentence coherence** + **Theme consistency**: CUT, both barely above chance
  (val_acc 0.525/0.531) — diagnosis is mean-pooling content-word features into one
  centroid likely destroys the signal a linear head needs, not proof coherence itself
  is unlearnable. **List parallelism**: PARTIAL (val_acc 0.768, probe 8/10) — trained on
  the I2 primary, fails on two of the most canonical same-type pairs ("dog"/"cat",
  "king"/"queen"), left unwired. **Definiteness**: CUT before training — a corpus-fact
  check found "a" isn't even in the pipeline's 4000-word vocabulary (`min_len=2` drops
  single-letter words; 166,554 real occurrences silently mapped to `<unk>`), a
  vocabulary-construction gap, not a learnability one. **Transitivity**: CUT as
  redundant with the shipped Closer genome / already-cut Verb argument genome, design
  analysis only, no training run. **Revision stage BUILT**: `Service.generate_revision()`
  + `_sentence_score()` in `wordpipe_service.py` implement Whole-sentence scorer +
  Best-of-N — generates several candidate sentences with the unchanged pipeline, scores
  each from already-evolved champions, keeps the best; no new training; verified
  mechanically (correctly rank-orders candidates); has a known length bias, not yet
  exposed via UI/API. **Pronominalization VALIDATED + WIRED** — the first real
  Passage-stage genome, needed no training at all (reuses the No-repeat genome's
  `recent` buffer): re-mentioned content words get replaced by "it" 60% of the time;
  measured 'it' frequency rose 0.77%→1.11% of words (+44% relative), a real effect.
  Discourse relation / Information status remain genuinely blocked — no persistent
  cross-sentence state exists anywhere in the pipeline. Full details in `genomes.txt`.

- **[2026-07-08] (Claude)** — Autonomous genome-testing run, genome #4 resolved, round 1 wrap-up.
  **Clause count** (skeleton): CUT. val_acc 0.575 never beat the 0.73 majority-class rate;
  decisive probe (Spearman correlation between genome score and each word's TRUE empirical
  compound-sentence rate, not hand-picked words) came back -0.009 — no learnable signal.
  Diagnosis: genuinely unlearnable from the opening word alone (a downstream planning
  decision, not a lexical property), not a compound question to decompose further.
  Round 1 tally (genomes #1-4): 1 clean win (Sentence type), 1 wired-but-inert (Sentence
  length plan), 1 real-but-unclean signal left unwired (Collocation strength), 1 clean cut
  (Clause count). Remaining Fill-stage roadmap items (Definiteness, Transitivity) flagged
  in `evolang_layers.js` as needing a real per-context design idea before another training
  run — both would likely repeat Clause count's per-word-alone failure mode. Full details
  in `genomes.txt`'s "Round 1 summary".

- **[2026-07-08] (Claude)** — Autonomous genome-testing run, genomes #2 and #3 resolved.
  I2 node bumped to v1.4.6 (`JOB_WHITELIST` now also allows `genreg_train/run_*.py`, so
  future genome runners dispatch without duplicating into `jobs/`). **Collocation strength**
  (absorbs Verb-preposition — no POS tags exist to separate them) trained remotely on the
  I2 primary: val_acc 0.782, probe only 6/8 correct (75%, fails "depend on"/"look at") —
  real signal, not clean enough to wire; left unwired, marked PARTIAL not cut.
  **Sentence length plan** trained locally: probe passed weakly (+0.53 vs -0.13 mean
  opener score, many ties — coarse feature space), wired the same shape as Sentence type
  (opener bias + boundary-probability reshaping toward the 14-word median), but a 20-sample
  generation spot-check showed no measurable length-distribution change — wired but inert
  at safe gammas. Both documented honestly in `genomes.txt` and `static/evolang_layers.js`
  (source of truth for current status). `lenplan` toggle added to `/evolang` (default OFF).

- **[2026-07-08] (Claude)** — I2 job dispatch extended for the autonomous genome-testing run
  (per user reminder to use the primary node): `PUSH_WHITELIST` now also allows
  `genreg_train/*.py` and `project/EEC-main/engine/corpus.txt` (v1.4.5), so the full
  in-pipeline WordPipe stack — not just the standalone Wikipedia relation genomes — can run
  on the primary. Deployed the whole `genreg_train/` package (44 files) + the 49MB novel
  corpus. Hit and fixed a real bug: `genreg_train/__init__.py` eagerly imports an unrelated
  subsystem (`trainer.py` -> `engine_api.py` -> requires `project/genreg-engine-main`, a
  different RL-engine project, not deployed to compute nodes) — any `from genreg_train
  import wordpipe` blew up on the primary with a `RuntimeError`. Fixed with a reusable shim,
  NEW `jobs/_pkg_stub.py`: pre-registers a stub `genreg_train` module in `sys.modules` with
  the correct `__path__` before any real import runs, so Python imports the needed
  submodule directly without executing the package `__init__.py`'s side effects. Every
  future dispatch script imports this first. Verified end-to-end with `jobs/test_wp_import.py`
  on the real primary (corpus builds, vocab loads, no exception).
- **[2026-07-08] (Claude)** — **Autonomous genome-testing run started** (per user: backup
  first, then systematically work the /evolang/layers roadmap — train, probe, wire if it
  passes, decompose or cut if it doesn't). Git checkpoint committed first (`c7997e7`).
  Genome #1, **Sentence type** — VALIDATED + WIRED. NEW `genreg_train/sent_type.py` (mines
  sentence-initial words bucketed by whether the sentence ends in "?" vs "."/"!", trains a
  unary genome — same shape as the existing `sent_open.py` opener genome, but a harder
  discrimination: hard negatives are statement-openers, not the general marginal) +
  `run_sent_type.py` (trainer + probe runner). Probe: all 18 hand-picked question-openers
  (do/will/what/is/how...) scored above all 11 statement-openers — clean separation, not
  just on average. Wired into `wordpipe_service.py`: a coin-flip at the corpus question-rate
  (8.19%) fires once per sentence; if "question", biases the opener toward this genome's
  scores (`SENT_TYPE_GAMMA=3.0`) and forces "?" instead of "." at the close (also fixed the
  final-text cleanup regex, which only capitalized after ". " before — now handles "? " too).
  Generation-time spot-check: flagged-question rate landed at 8.2% (matches the corpus rate
  almost exactly), and roughly half of flagged sentences opened with a real recognized
  question word (do/does/did/have/where/was/is/what/how/whose) — near 0% before this existed.
  New `sent_type` toggle on `/evolang`, OFF by default (no formal adj-hit/distinct guardrail
  sweep run yet, only the spot-check above). Map (`static/evolang_layers.js`) and
  `genomes.txt` updated. Continuing autonomously through the rest of the roadmap.
- **[2026-07-08] (Claude)** — `/evolang/layers`: split the "Sentence coherence"/"Theme
  consistency" pair back out of the earlier grouping fix — their names overflowed the
  sub-node box width sized for one-word Sentiment names. `groupSubWidth()` now sizes each
  GROUP's sub-node width to its own longest member name (clamped 84-150px) instead of a
  single global constant, so two-word names render cleanly without touching the Sentiment
  cluster's sizing. Also added TWO NEW PIPELINE STAGES to the backbone/columns — **Revision**
  (whole-sentence post-hoc scoring) and **Passage** (cross-sentence/discourse) — neither
  exists in `wordpipe_service.py`'s `generate()` yet, so both render dashed/muted in the
  backbone itself (not just as node status), honestly distinguishing "live pipeline stage"
  from "planned future stage." Fixed two places that hardcoded the old 3-stage set
  (`byLayer` init, `buildExport`'s `byStage`) to derive from `STAGES` generically instead,
  so future stage additions don't require hunting for hardcoded assumptions again. Added 12
  new PLANNED genomes from the user's gap analysis: Skeleton gets Sentence type/Sentence
  length plan/Clause count (was only 3 genomes vs Fill's 20+); Fill gets
  Definiteness/Verb-preposition/Transitivity; the new Revision stage gets a bundled
  "Revision (composed)" pair (Whole-sentence scorer + Best-of-N — abstraction-tier, since
  scoring a whole sentence composes judgments from other genomes); the new Passage stage
  gets Pronominalization/Information status (semantic) and Discourse relation (abstraction
  — explicitly the "evolved version" of the already-DEFERRED flat-rule Discourse connector
  genome). No training code written — planning only, per explicit instruction. Verified with
  the same throwaway-server + headless-Chrome method; live Flask untouched.
- **[2026-07-08] (Claude)** — `/evolang/layers`: uplifted the Semantic band per user request
  (adjacency alone "is not enough") — 6 new candidate genomes. Two are RE-SURFACED prior
  battery results, labeled `cut` with their real numbers (Wider co-occurrence: marginal gain
  at ±5 vs ±1, cut; Lexical bridge: baseline already exceeded corpus carryover rate, cut) —
  not relabeled "experimental" despite the ask, since real evidence already exists on them and
  that would misrepresent it. Four are genuinely new, untried ideas, labeled `planned`: Topical
  drift (distance from a running content-word centroid), Collocation strength (specific-pair
  fixed-phrase compatibility, tighter than loose window co-occurrence), List parallelism
  (are coordinated list items the same distributional kind), Domain purity (passage-level
  topic consistency, a different mining approach than the failed quote-span Register genome).
  No training code written — this is the flow map being used as a planning tool ahead of any
  build, per user's explicit ask. Verified with the same throwaway-server + headless-Chrome
  method; live Flask untouched.
- **[2026-07-08] (Claude)** — `/evolang/layers`: real hover tooltips (custom-styled div,
  replaces the slow/unstyled native SVG `<title>`), an always-visible tiny one-line
  description on every node (not just on hover), and a new `status: 'planned'` tier (dotted,
  faint — design intent, not yet attempted, distinct from `cut` which was attempted and
  rejected on evidence). Decomposed the monolithic CUT Sentiment genome into 4 PLANNED
  sub-genomes (Good / Bad / Intensity / Emotion — matches the roadmap already in
  `genomes.txt`'s battery note) — and per user feedback, these 4 render bundled into ONE
  compact "Sentiment (composed)" 2×2 cluster with a labeled dashed boundary, not stacked as 4
  separate full-size rows (which read as noise, not as one concept). New reusable `group`/
  `groupLabel` fields in the node data model + grouped-cluster layout code in
  `evolang_layers.js` — the map is now also where FUTURE genomes get planned before they're
  built. Export JSON updated to include group membership. Verified with the same throwaway-
  server + headless-Chrome-screenshot method as the initial build; live Flask untouched.
- **[2026-07-08] (Claude)** — `/evolang/layers`: added an **Export JSON** button (top-right of
  the card). Serializes the same data driving the diagram — layers, pipeline stages (with
  each stage's genome-id list), and every genome (id/name/layer/stages/status/description,
  cut genomes included) — to a downloaded `evolang_genome_layers.json`, for handing the
  pipeline's current architecture to another AI/tool without a screenshot. Client-side only
  (`Blob` + anchor download), no new backend route.
- **[2026-07-08] (Claude)** — NEW `/evolang/layers`: a genome layer/flow map for the WordPipe
  pipeline. Three horizontal bands (Structural / Semantic / Abstraction — per this session's
  "layer" framework: form vs. built-meaning-space vs. relations composed on that space) each
  positioned by which real generation-pipeline stage they wire into (Skeleton / Fill /
  Boundary), with a backbone showing the actual flow (Skeleton → Fill → Boundary → Output) and
  connector lines from each genome to its stage. Includes the 3 wired-but-experimental relation
  genomes AND the cut ones (sentiment/polysemy/register/analogy/open-obligation) as
  dashed/faded nodes with their one-line cut reason in the hover tooltip — same "report the cut
  list too" discipline as the rest of this project. NEW `static/evolang_layers.js` (static
  hand-authored data array + a lightweight hand-rolled SVG renderer, same pattern as
  diff.js/animation.js — NOT the PURE node-graph editor, this is read-only) and
  `templates/evolang_layers.html`; new route in `app.py` (`evolang_layers_page`); linked from
  the main `/evolang` page next to the WORDPIPE_FINDINGS docs link; two new CSS tokens
  (`--layer-semantic`, `--layer-abstraction`) in `style.css`. Verified end-to-end with a
  temporary local server on a spare port + headless Chrome screenshot (both bands/columns and
  the legend render correctly) — the live site's Flask process was never touched. **Flask
  restart needed** to see this on the live site (new route in `app.py`).
- **[2026-07-08] (Claude)** — MNIST-Pipe **rounds 2-3: 97.63% test shipped; the path to the
  98.5% record is diagnosed**. Round 2 (deskew) TEST: centroid 90.95 -> detectors 96.97 ->
  +mixer 97.06 -> +pairwise **97.63** (margin 6.0 on val; champions in `demo/mnist_genomes.pkl`,
  served by /mnist). DIAGNOSTIC (closed-form logistic, ceiling probe only, per GENREG_RULES
  baselines rule): the SAME v2 features support **98.53% test** — so the remaining gap is GA
  optimisation, not representation. Round-3 attempts, all honestly gated and CUT so far:
  (a) random-Fourier lift v3 (`rff_lift`, kept in code) — GA can't exploit the lifted
  directions, minibatch noise dominates (cold 95.98; warm-start decays; L2 slows, doesn't cure).
  (b) shifted-copy train augmentation (`--augment`, kept) — harder task, worse at equal gens
  (95.81). (c) JOINT REFINE (`train_joint`, `--joint-only`): det+mixer folded algebraically
  into one 677x10 genome (`fold_stack`), warm-started, evolved on joint log-softmax — no
  regression (champion tracked on val) but the population drifts off the warm optimum: train
  fitness climbs while val decays, even with fixed-minibatch rotation + 5e-4 sigma floor.
  Diagnosis: mutation's random walk grows |W| (overfit); final probe of the night = L2 1e-4
  (the exact term the closed-form ceiling needed) + |W|^2 logging ->
  `demo/mnist_joint_probe.log`. `ga_step` gained sigma_lo/sigma_hi params.
  **PROBE VERDICT (read the log): L2 1e-4 does NOT cure it** — fit0 declines -0.17 -> -0.27
  across batch rotations while |W|^2 grows 478 -> 828 through the penalty. Mechanism
  identified: SERIAL PER-BATCH OVERFITTING (each 25-gen window is a deterministic landscape
  over one 4096 batch; the population climbs it, rotation invalidates the climb, repeat) —
  not gen-to-gen noise. Next session's fix: hold ONE fixed ~16k train subset as the fitness
  pool for the whole joint run (deterministic landscape, too large for a 6770-param linear
  genome to overfit — the closed-form fit generalises at this ratio), pop ~60, sigma_lo 5e-4,
  keep L2 1e-4; ~80 min CPU at 4000 gens. Then re-gate the pairwise margin (grid now to 12).
- **[2026-07-07] (Claude)** — `/evolang`: **Chunks toggle now OFF by default** (per user: it was
  causing more repeated tokens — the No-repeat genome only tracks single emitted words, so a
  chunk-emitted multi-word phrase can reintroduce a content word No-repeat would otherwise have
  blocked). Tooltip updated to say so. `static/evolang.js` only; no service/Python change, no
  restart needed beyond a browser refresh.
- **[2026-07-07] (Claude)** — Wired the standalone semantic-relation genomes (Hypernym, Meronym,
  unified Synonym/Antonym) into `/evolang`. NEW `genreg_train/rel_wire.py`: crosswalks the 30K-word
  Wikipedia vocab these were trained on into the pipeline's own 4000-word vocab by direct lookup
  (~91% coverage; uncovered words get a zero feature row, same graceful-degradation pattern as
  every other genome). Wired as selection re-ranks in `wordpipe_service.py` (same mechanism as
  Semantic), gated behind new `hyper`/`mero`/`synant` toggles — **OFF by default**, since these are
  validated as standalone relation detectors but haven't been through the generation-time battery
  (real-effect + guardrail measurement) every other shipped genome went through; tooltips in the
  GUI say so explicitly. New `corpora/wikipedia/build/{hypernym,meronym,synant}_export.py`:
  standalone re-exports of each genome (same seed/mining recipe as the validated runs) that persist
  the champion matrix to `.npz` — nothing was previously saved to disk from the validation-only
  training scripts. **Found and fixed a real bug in the process**: `meronym_export.py`'s
  consolidation accidentally reordered its two mining loops relative to the original validated
  run — same seed AND same mined-pair COUNT (7857), but the mined pairs feed a `Counter` whose
  iteration order becomes the training array's index order, and the GA samples by array position,
  not pair identity — so the reordering silently produced a DIFFERENT trained genome (probes 7/10
  instead of the original 9/10) despite looking deterministic. Fixed the loop order; re-running.
  Hypernym reproduced bit-for-bit identical to its original validated probes (10/10); synant's
  convergence trajectory also matched closely. Smoke-tested the full wiring locally (all three
  genomes load, ~91% coverage, generation runs and visibly changes word choice with the toggles on
  vs off, no exceptions) before any Flask restart. **Flask restart needed to see this on /evolang**
  (never restarted automatically — see standing instruction).
- **[2026-07-07] (Claude)** — MNIST-Pipe **round-1 RESULTS + round-2 launched**. Round 1
  (raw-image statistics layer) on the untouched test 10k: centroid floor 88.93% ->
  detectors(argmax) 95.57% -> +mixer 95.60% -> +pairwise referees **96.83%** (37.4K evolved
  params total; every layer passed its gate; val->test drop 97.68->96.83, within the <10%
  relative rule). The mixer barely moves top-1 but improves val log-prob -0.39->-0.24 —
  calibration that the pairwise margin gate rides on. Round-1 champions backed up to
  `demo/mnist_genomes_r1.pkl`. Round 2 per the thesis (enrich the ENVIRONMENT, not the
  organism): new `deskew()` in `mnist_pipe.py` — moment-based shear correction, unsupervised,
  vectorised bilinear remap — as statistics-layer v2 (`build_features(version=2)`, version
  stamped into the champions pickle). Deskew alone lifts the no-evolution centroid floor
  88.93% -> 90.95%. Full v2 battery running detached -> `demo/mnist_train_r2.log`.
- **[2026-07-07] (Claude)** — NEW PROJECT + PAGE: **/mnist — MNIST-Pipe**, the
  EvoLang/WordPipe specialist-pipeline recipe applied to images (user pivot: prove the
  GA-abstraction thesis outside language; target the 99% range). Statistics layer BUILT
  from the data (677 fixed dims: zone ink, profiles, gradient histograms, PCA — no labels,
  never evolved) -> semantic layer EVOLVED (10 one-vs-rest detector genomes + 45 one-vs-one
  pairwise disambiguators, each a tiny linear head with one clean survival condition, soft
  BCE fitness) -> output layer EVOLVED (10x10 mixer genome, soft log-softmax fitness;
  pairwise genomes referee close top-2 calls, margin tuned on val only). Gradient-free
  throughout (shared tournament/elitism/starvation/self-adaptive-sigma machinery).
  Baselines: majority 11.35%, nearest-centroid floor 88.93%. Files:
  `genreg_train/mnist_pipe.py` (data/features/training/eval), `genreg_train/mnist_service.py`
  (lazy web backend), `templates/mnist.html`, `static/mnist.js`, `/mnist` + `/api/mnist/*`
  routes in `app.py`, nav entry, `style.css` additions. Data in `corpora/mnist/`, champions
  in `demo/mnist_genomes.pkl`, training log `demo/mnist_train.log`. See
  `documentation/changelogs/CHANGELOG_MNIST.md`. NOTE: needs a Flask restart to serve the
  new routes (not restarted per standing rule).
- **[2026-07-07] (Claude)** — NEW `documentation/GA_SCALING_FIELD_NOTES.pdf` — a generalized (not
  language-specific) field-notes paper on scaling gradient-free genetic algorithms, drawn entirely
  from this project's own measured results (structural + semantic genome batteries, the I2
  job-dispatch work). Honest by request: reports the ~2/3 cut rate alongside the shipped wins, with
  a dedicated failures section (validation-accuracy-is-not-the-verdict, label/mining quality as the
  real bottleneck, the analogy "representation-altitude" diagnosis, a metric-saturation artifact)
  and a 10-point extracted-principles list. Typeset PDF (reportlab; Candara/Constantia/Consolas,
  a palette grounded in GA vocabulary — amber=selection, green=validated, rust=cut), 11 pages.
  Explicitly flagged as an empirical log, not a controlled study (see its own Limitations section).
- **[2026-07-07] (Claude)** — I2: **secondary node + signed job dispatch** (compute-only v1, per
  user request to run multiple GA training jobs across machines instead of queueing on one). New
  `--role secondary --primary <url>` on `i2_node.py`: a compute-only node (no content plumbing —
  no pages/genome/ledger) that registers with a primary via a signed handshake and exposes
  `POST /api/i2/admin/job/{submit,cancel}` + `GET /api/i2/admin/job/<id>/{status,log}` +
  `GET /api/i2/admin/jobs`. Auth reuses the EXISTING Ed25519 admin-key trust model
  (`verify_admin_doc`, same key `push_to_primary.py`/maintenance already use) with new domain-
  separated prefixes `i2job\x00`/`i2reg\x00` — no new crypto. Submitted scripts must match
  `JOB_WHITELIST` (mirrors `PUSH_WHITELIST`, now also covering `corpora/wikipedia/build/` and a new
  `jobs/` dir) and must already be on the node's disk (deployed via the existing push mechanism) —
  deliberately no inline/arbitrary-code channel. Jobs run one-at-a-time per node via a background
  worker thread + FIFO queue (subprocess, log to `data_dir/jobs/<id>.log`); different NODES run in
  parallel. New `run_job.py` (mirrors `push_to_primary.py`'s signing) to submit/watch/cancel jobs
  from this machine. **This session**: job history now PERSISTS across node restarts
  (`jobs_index.json` in the node's data dir; a job still `queued`/`running` at save time is marked
  `interrupted` on reload, not silently dropped or left stuck). New **Jobs tab** in the primary's Tk
  console GUI — lists all jobs (id/script/status/times/rc), streams the selected job's log, and can
  cancel a running job, refreshed on the same 2s loop as the rest of the dashboard. Bumped
  `NODE_VERSION` 1.3.0 -> 1.4.0. E2E verified locally (submit -> queued -> running -> log streams ->
  done; unsigned submit correctly rejected; job survives a process restart) before pushing to the
  real primary at 10.0.0.15 via `push_to_primary.py` (confirmed back up on v1.4.0 with the new
  routes live). Scope explicitly OUT of v1 (see plan): no reputation-based auto-promotion of unknown
  nodes (trust = admin key only, same boundary as code deploy), no automatic artifact pull-back, no
  cross-node scheduling (the caller picks which node a job targets). Full content replication for
  secondaries (pages/genome/ledger sync) deferred — this pass is compute-only per user's explicit
  scope choice. See also `documentation/changelogs/CHANGELOG_I2.md`.
- **[2026-07-07] (Claude)** — Semantic-relation genome battery, round 2: reran the in-flight
  hypernym/synant work from RESUME.md and added 5 new relation attempts (meronym, synonym/antonym
  UNIFIED reframe, sentiment, polysemy, analogy) — same discipline as every other battery in this
  file: ship what earns its place on clean PROBES, cut what doesn't, log the honest reasoning.
  Full logs: `corpora/wikipedia/build/{hypernym,synant,synant_unified,batch2,register}.log`.
  **VALIDATED — Hypernym**: directional heldout-acc 0.86, probes 10/10 (dog->animal, france->country,
  hammer->tool, etc. all rank correctly above their reverse). **PARTIAL — Synonym/Antonym UNIFIED**
  (`synant_unified.py`, per user's reframe suggestion): training synonym and antonym as two SEPARATE
  "related vs unrelated" detectors failed the decisive test (separate synonym genome ranked the
  unrelated control `dog/car` ABOVE real synonyms; separate antonym genome failed its OWN flagship
  example, `hot/cold` scored negative). Reframed as ONE genome, no unrelated pairs at all, asking
  only "given a pair already known to be related, same-meaning or opposite-meaning?" Real
  improvement: `hot/cold` flipped to +0.86 (correct), 11/14 probes correct overall, val_acc 0.77.
  Residual: `big/large`/`small/little` (size-adjective synonym pairs) still misclassify as
  antonym-leaning — likely genuine contamination (these words really do co-occur in coordination
  contexts too, e.g. "big and small businesses"). Shipped as PARTIAL, not wired. **VALIDATED-weaker
  — Meronym**: probes 9/10 (only leaf/tree failed), heldout-acc modest 0.35-0.41. **CUT — Sentiment**
  (monolithic): seed-propagation drifted to generic frequent words; probes inverted (`war` +7.85,
  `joy` -7.38 — backwards). **CUT — Polysemy**: val_acc looked strong (0.88) but probes failed
  (bank/spring/bat/light/star/book all scored NEGATIVE) — Wikipedia's dominant-sense skew defeats
  the nearest-neighbor-spread proxy (a word mostly used in ONE sense reads as monosemous regardless
  of its dictionary polysemy). **CUT — Register**: quote-span dialogue/narration mining on the novel
  corpus produced a weak, incoherent signal (val_acc 0.61; `shall`/`which` read informal, `sir`/
  `nevertheless` read formal-negative). **CUT AS DESIGNED — Analogy** (chance-level 0.48-0.53 through
  600+ gens, not a mining problem): per user's diagnosis, analogy is a LAYER-3 relation — "do pair
  (A,B) and pair (C,D) instantiate the same relation?" is a question about layer-2 relation-genome
  OUTPUTS, not a question askable from raw distributional offset vectors. Training it directly on
  `feat(B)-feat(A)` asked a layer-3 question with layer-2 machinery; chance accuracy is the expected
  result of that mismatch, not evidence the concept is wrong. Correct future recipe logged in
  `genomes.txt`: score = agreement between `hypernym_genome(a1,b1)` and `hypernym_genome(a2,b2)`.
  **Next proposed step** (not yet built, pending go-ahead): decompose sentiment the same way the
  working pipeline was decomposed — instead of one "positive or negative?" genome, several tighter
  binary genomes (Good, Bad, Intensity, Emotion), each with a cleaner, tighter corpus signal than the
  monolithic attempt. Nothing from this pass is wired into the live generation pipeline — standalone
  validation only, matching how the round-1 hypernym/synonym/antonym work was scoped. `genomes.txt`
  and `RESUME.md` updated with full verdicts.
- **[2026-07-07] (Claude)** — NEW `genreg_train/agreement.py`: an evolved subject/verb + modal/aux
  AGREEMENT genome. 22 rule-based morphology features per word (suffix -s/-ed/-ing/-e, closed-class
  finiteness/number: FIN_3SG is/was/has, FIN_NON3 are/were/am, BARE be/have/do, PARTICIPLE been/gone…,
  pronoun number, "I"→am) × a tiny evolved 22×22 bilinear head (~480 params, gradient-free, ga_step).
  Fitness = corrupted-pair discriminator with HARD contrastive negatives (same prev, candidate drawn
  from the real next-word distribution — only agreement separates real from fake, so it learns the RULE
  not a bigram table). Result: **12/12 on held-out minimal pairs** ("could be">"could is", "they are">
  "they is", "he is">"he are", "i am">"i is", "has been">"has being", "she was">"she were", "he runs">
  "he run", …). Global discriminator acc 0.587 (near-chance is correct here: most word pairs carry no
  agreement constraint; the genome concentrates its power on the closed-class cases where agreement
  actually decides). First attempt (uniform-random negatives + 17 features that couldn't tell be/is
  apart) scored inflated 0.824 global but only 6/12 pairs — fixed by hard negatives + finiteness feats.
- **[2026-07-07] (Claude)** — `/evolang` Params + Deploy-size tiles now computed LIVE from the loaded
  genomes instead of hardcoded. Service `status()` gained `params` (sum of evolved-head array sizes),
  `heads_kb` (evolved genomes) and `full_kb` (everything the pipeline needs at inference except vocab-
  derived feats). Corrects the stale "~7K / ~140 KB": params is genuinely ~6.8K, but deploy is 27 KB
  (heads) / ~1.3 MB (full pipeline — dominated by the chunk lexicon ~820 KB + SVD word-features ~375 KB),
  not 140 KB. Tile now shows "6.8K" and "27 KB / 1.3 MB"; caption reworded off the false size. Also
  shrank the genome-toggle list (smaller boxes, descriptions moved to hover tooltips — no more scrollbar).
  Flask restart needed for the service change; template/js just need a browser refresh.
- **[2026-07-07] (Claude)** — SEMANTIC-RELATION genomes: decompose "meaning" into tiny per-relation genomes
  (thesis: build the distributional SPACE from corpus stats, evolution learns ONE relation inside it — see
  memory `ga-abstraction-thesis`). Corpus upgrade for this: built `corpora/wikipedia/wiki_corpus.txt` (302MB,
  106K clean articles, 51M words) from the 25GB dump via zetifile's stripper, and `wiki_feats.npz` (30K vocab,
  128-d SVD features — NN check strong: king→queen/prince, france→germany/spain, hot→cold). Novels had no
  Hearst/coordination signal; Wikipedia does (validated by mining). In-progress (see RESUME.md): hypernym
  genome learns DIRECTIONAL type-of at 0.86 (dog→animal > animal→dog); synonym val_acc 0.86; antonym trained
  over a concatenated paradigmatic+coordination-SVD environment (built so evolution can read the coordination
  signal that separates hot/cold from big/large) — probes pending. Honest nuance: the space cleanly gives
  similarity/relatedness; isolating synonym vs co-hyponym vs antonym is the hard part. Build scripts in
  `corpora/wikipedia/build/`. Nothing wired/cached yet — pure validation. Open: keep semantics on Wikipedia
  as a knowledge layer vs retrain whole pipeline on it.
- **[2026-07-07] (Claude)** — New-batch stage 2 (cross-sentence tier): built + evaluated has-verb,
  wider-cooc, lexical-bridge, discourse-connector (`genreg_train/stage2.py` + `corpus_reference()`).
  ALL FOUR CUT/DEFERRED — the deficits mostly don't exist. Corpus refs: verb/sent 0.844, carryover 0.23,
  connector-open 0.069. has-verb: baseline 83.7%≈corpus 84.4% (no deficit). bridge: baseline carryover
  27.5% already EXCEEDS corpus 23%; boosting →58-74% = unnatural repetition (distinct 53→46). wider-cooc:
  cpr 35.5→36.7 marginal, sem±1 already sufficient. connector: only real deficit (0.5% vs 6.9%) but my
  version is a flat rule not the evolved previous-sentence-conditioned predictor the spec wants — deferred,
  not shipped as a rule. META-FINDING: the cross-sentence coherence tier has NO measurable structural
  deficit (baseline matches/exceeds corpus on verb-rate + carryover); the remaining gap is MEANINGFUL
  coherence = semantic understanding, the same wall the conversational attempt hit. Structural sentence
  genomes paid off (real deficits); coherence doesn't yield to boosts. Nothing wired; stage2.py kept for helpers.
- **[2026-07-07] (Claude)** — FIXED the run-on-sentence bug (sentences were 54 words, hitting the 55-word
  cap 43/45 samples). Root cause was NOT the Boundary genome (correctly calibrated: corpus base rate 0.0537
  = 18.6-word sentences). The reserved `<unk>` class (class 32 after the unk-fix) carries most of the
  sentence-boundary signal (rare sentence-final words fall into unk; corpus boundary-rate 0.182) and is
  emitted ~38% of the time — but generation SKIPS unk (not in the word table) with a bare `continue`, which
  also bypassed the boundary check, throwing the period away. Fix: when the fill loop skips an unk class,
  still run the boundary check there (`wordpipe_service.generate`). Result: mean sentence length 54→~14
  words (corpus target 18.6), capped-at-55 dropped 43/45→3/167. Output now reads as sentences, not one
  run-on — and this unblocks the whole sentence-level tier (opener now visibly fires, closer can act).
  Flask restart needed.
- **[2026-07-07] (Claude)** — New-batch battery, stage 1 (sentence-level positional genomes). Added
  `genelib.UnaryPop`/`train_unary` (linear word classifier). NEW `sent_open.py` + `sent_close.py`.
  SHIPPED **Opener** (~14 params): unary classifier over function-type features, fires at each sentence
  start (cur==0) — bad-opener% (sentence begins with of/to/is/aux) 22.9→19.3 at zero fluency cost. Wired
  via the fill `bonus` hook (combined with rep penalty), cached, GUI toggle + `open` flag, arch entry
  (OPEN_GAMMA=4.0). After the boundary bug was fixed (see the newer top-of-file entry) opener became a
  strong win: bad-opener% 17.3→8.3 at zero cost. SHIPPED **Closer** too (`sent_close.py`, ~14 params):
  reshapes WHERE periods land — modulates both boundary paths by exp(CLOSE_GAMMA·(close_score[last_word] −
  emission-weighted-centre)), rate-preserving (sentence length holds), ends sentences after nouns/verbs not
  dangling "of/the/to". bad-close% 65→33 at CLOSE_GAMMA=0.5, adj-hit up. Needed emission-weighted centring
  (median-centring inflated length 14→29). Both cached/toggleable (`open`/`close` flags); params ~7.4K.
  (This entry's original "sentences run ~47 words / Boundary under-firing" finding led to the boundary-bug
  fix logged at the top of the file — root cause was the unk class being skipped, not the Boundary genome.)
- **[2026-07-07] (Claude)** — Ran the constraint-genome battery (trained 8, evaluated each in generation).
  SHIPPED 2, cut 6 — the battery did its job. **Semantic** (`sem_compat.py`, ~580 params): meaning-level
  content co-occurrence, real content-adjacency 33%→40% with adj-hit holding — wired as a selection re-rank
  over the 24-d SVD feats (SEM_GAMMA=2.5). **No-repeat** (`repetition.py`, ~10 params): stateful recency
  penalty, content-word repetition 2.2%→0%, guards stable — wired via a new `bonus` hook on
  `_fill_selected`/`_fill_bisel` (caller-computed per-candidate penalty) + a recent-words buffer in the
  service (function words may recur; content may not). Both cached in genomes.pkl; GUI toggles "Semantic" +
  "No-repeat" (on by default) + `sem`/`rep` query flags + arch entries. Params now ~7.4K. CUT (logged in
  genomes.txt with reasons): prep_complete + det_bind (redundant with Alternation; det cut orphan-dets but
  cost -6 adj-hit at every gamma — coarse feats pick the wrong noun), tense_consist (no effect on mixed-tense),
  verb_arg + pron_ref (class-level windowed = same redundancy wall as clause; Order's C=4 subsumes them).
  DEFER: clause_boundary (beats base rate but is a positional predictor, not wired; overlaps Comma). Cut
  modules kept on disk, not wired/cached. Lesson: survivors enforce what Order/Selection/Alternation
  STRUCTURALLY can't see — semantic content identity + exact-word state. Flask restart needed for app.py/service.
- **[2026-07-07] (Claude)** — CUT the Clause-template genome (`clause.py`) — redundant with Order,
  logged honestly. Trained fine (val_acc 0.61) but measured over 60 samples it can't help: the
  order+alternation skeleton already emits only ~0.2% rare class-trigrams (Order's C=4 next-class
  prediction already yields common 3-class sequences), and biasing with the clause head only LOWERS mean
  trigram log-prob (-5.98→-6.36 at g=3). Root cause: class-level clause validity is subsumed by 4-context
  next-class prediction; the "the at the"/"of and to" cases it targeted are WORD-level (common class-trigram,
  rare word-trigram) which a class-level genome can't see, and the word-trigram space is the intractable one
  we decomposed away from. Module kept (windowed machinery reused by verb_arg/pron_ref), not wired/cached.
  NOTE: the earlier class-trigram-HIT metric read 100% for all gammas — a measurement bug (33 classes over
  8.5M words = nearly every triple occurs ≥1×); switched to frequency-weighted log-prob + rare-rate.
- **[2026-07-07] (Claude)** — Built out the constraint-genome battery from genomes.txt (roadmap), ready
  to train but NOT run yet. NEW `genreg_train/genelib.py` — shared scaffolding: `BilinearPop` +
  `train_pairwise` (pairwise discriminator over fixed features w/ hard negatives) and `WindowPop` +
  `train_windowed`/`window_bias_tensor` (K-class windowed validity discriminator). NEW genome modules,
  each one job, gradient-free, following the agreement/alternation pattern: `prep_complete.py` (prep needs
  content object — trained only on preposition contexts), `det_bind.py` (determiner needs noun — trained
  only on determiner contexts), `sem_compat.py` (adjacent content words co-occur in a ±4 window — first
  meaning-level genome, bilinear over the 24-d SVD feats), `repetition.py` (stateful recency-penalty curve,
  content-gated — don't repeat a content word within N), `verb_arg.py` (windowed, corrupt the subject slot),
  `pron_ref.py` (windowed, corrupt the antecedent slot), `tense_consist.py` (pairwise over tense feats —
  compatible tenses among sentence verbs), `clause_boundary.py` (class+position -> P(clause break), mirrors
  Comma genome). Also NEW `clause.py` (clause-template windowed genome, currently under test). All import
  clean. genomes.txt updated with a build-status table + battery plan. Global/dialogue genomes
  (continuity, conversation/copy, recurrent) DEFERRED per priority (conversational route already rolled
  back — see memory). Expectation logged: several may prove redundant (prep/det vs alternation, tense vs
  agreement, clause-boundary vs comma) — the battery decides; cut and log any that don't earn their place.
- **[2026-07-07] (Claude)** — Took AGREEMENT to the ORDER level too (same move; doesn't break the model).
  Generalized `gen_class_seq`'s single `altern=` into a `reranks` LIST of (classfeat, M, gamma) so multiple
  constraint genomes bias the next-CLASS logits and their biases sum. Service builds `agree_classfeat`
  (freq-weighted gram_feats over each class's members) and applies it when `agree` on (ORDER_AGREE_GAMMA=1.0).
  Verified over 60 samples on top of order+selection alternation: agreement violations (modal+finite,
  he/they + wrong-number verb) roughly HALVED 13→7, fluency held (adj-hit 85.3→85.0), salad nudged further
  down (ff 35.6%→31.3%), distinct 63.4→61.0 (still well above baseline 57). Both alternation and agreement
  now act at the skeleton (their proper home) + selection. Flask restart needed for the service change.
- **[2026-07-07] (Claude)** — Took content-function alternation to the ORDER level (does NOT break the
  model — improves it). `gen_class_seq` now takes an optional `altern=(classfeat, M, gamma)`: the SAME
  evolved alternation head lifted to per-class centroids in function-feature space, biasing the next-CLASS
  logits so a function-heavy class is nudged toward a content-heavy successor. Service builds
  `altern_classfeat` (freq-weighted func_feats over each class's members) and applies it when `altern` is
  on (ORDER_ALTERN_GAMMA=2.0). SWEEP over 40 samples: selection-only alternation cut salad (ff 58%→37%)
  but HURT local fluency (adj-hit 86.0→81.4) — because filling a function-class slot with a content word
  means picking off-distribution words. Order-level fixes exactly that: when the SKELETON alternates,
  selection picks natural high-freq words that occur in real bigrams — adj-hit recovers to 84.7, ff stays
  ~37%, distinct 57→64. Strictly better than selection-only on all three axes; reads visibly cleaner
  ("Many is he been to the man that was at at that could... of of that that" → "Many that know been to
  his go as no reached them could how having room of the round the gold moment..."). Confirms the order
  genome is the right home for the constraint. Flask restart needed for the service change.
- **[2026-07-07] (Claude)** — NEW `genreg_train/altern.py` + wired into `/evolang`: CONTENT-FUNCTION
  ALTERNATION genome — the biggest visible fix so far. 14 function-type features per word (CONTENT vs
  article/prep/coord/subord/aux/modal/copula/pronoun/det/wh/to/neg, closed-class membership) × an evolved
  14×14 bilinear head (~200 params, gradient-free). Fitness = real-vs-hard-negative discriminator (same
  prev, candidate from the real next-word distribution). Learned the alternation RULE: 12/12 on held-out
  minimal pairs with sharp margins (of→the +1.25 vs the→of −2.60, of→of −2.05, to→to −4.45, and→and −0.91,
  in→and −2.19). WIRED as a selection re-rank (gamma=3.0) alongside agreement — refactored
  `_fill_selected`/`_fill_bisel` to take a `reranks` LIST of (feats,M,gamma) via new `_apply_reranks`, so
  genomes compose. MEASURED impact: function→function adjacency 57.5%→37.8% over 30 samples (−34%), clearly
  visible in output ("The to not and had of those" → "The behind so and asked of our possibly have
  nothing"). Honest: within-selection re-rank (can't restructure the order skeleton; ~38% residual is the
  skeleton floor; not every seed improves — seed 2 was a wash). GUI: "Alternation" toggle (on by default)
  + `altern=1` query param + arch-panel entry. Flask restart needed for app.py/service; evolang.js = refresh.
- **[2026-07-07] (Claude)** — Cached + wired the Agreement genome into the `/evolang` pipeline. Trained
  champion stored in `demo/genomes.pkl` (key `agree`). `wordpipe._fill_selected`/`_fill_bisel` take an
  optional `agree=(gram_feats, M, gamma)` re-rank term (gamma=2.5) that adds the agreement score to the
  selection score before softmax sampling — backward-compatible (None = old behaviour). `wordpipe_service`
  precomputes gram_feats over the shared vocab and passes the re-rank when the `agree` flag is set; new
  `agree=1` query param in the `/api/evolang/generate` route. GUI: new "Agreement" layer toggle (on by
  default) + architecture-panel entry in `static/evolang.js`. HONEST pipeline impact: marginal so far —
  modal+finite violations over 20 samples OFF=0 / ON=2 (noise), because the ORDER skeleton rarely emits
  the modal→verb / aux→participle adjacencies where agreement decides. Validated standalone (12/12 pairs)
  and ready to compose, but not a visible fluency win until the order genome produces those junctions.
  NOTE: app.py + wordpipe_service.py changes need a Flask restart; evolang.js needs only a browser refresh.
- **[2026-07-07] (Claude)** — `/docs` browser: favorite (star toggle) and archive per document,
  sort by name or recently-updated, and a NEW badge for files modified in the last 3 days.
  Favorite/archive state is client-side (localStorage, single-user local tool) so no server or
  `/api/docs` schema change was needed — archived docs are hidden from the default view and
  surfaced via an "Archived" toggle; a "Favorites" toggle narrows to starred docs only.

- **[2026-07-07] (Claude)** — `/images` reverse-tab polish: caption length ceiling raised 75->200
  tokens; BLIP was hitting its own early-stop well under budget regardless, so `_caption()` now
  passes `min_new_tokens` + sampling (top_p, temperature, repetition_penalty, no_repeat_ngram_size)
  to actually force the extra length instead of silently truncating. Noted in the UI that >~75-100
  tokens degrades into stock-photo-metadata noise (BLIP's real ceiling, not a bug). Gallery UX:
  "Process" now appends each run's frames under a job header instead of clearing the canvas, and
  each frame renders as an index/thumbnail/prompt row stacked in a single column (was a wrapping
  card grid).

- **[2026-07-07] (Claude)** — `/images` reverse tab: image/video -> prompt via BLIP captioning +
  CLIP-ranked medium/style/lighting/quality tags — `genreg_train/reverse_service.py`,
  `POST /api/images/reverse` (single image or video, frames extracted with imageio/ffmpeg),
  `GET /api/images/file/<path>` to serve results. Output lands in a structured job folder
  `runs/images/reverse/<job_id>/{frames,prompts}/frame_NNNNN.{png,txt}` + `manifest.json`.
  Caption length and modifier-tags-per-category are adjustable from the sidebar.

- **[2026-07-07] (Claude)** — `/images` text-to-image: wired a pretrained Stable Diffusion 1.5
  pipeline (diffusers) into the blank Images page — `genreg_train/sd_service.py` (lazy-loaded
  singleton pipeline, GPU if available), `POST /api/images/generate`, prompt/negative-prompt/
  steps/guidance/size/seed controls in the sidebar, generated PNGs saved under `runs/images/`.
  Not evolved — a plain pretrained-checkpoint generation utility, unlike the rest of GENREG.

- **[2026-07-07] (Claude)** — New project page `/images` — blank scaffold (terminals, run-config
  panel, agent-alerts panel) added to the nav, no canvas content yet.

- **[2026-07-07] (Claude)** — **WordPipe Track A — comma / internal-punctuation genome** (+ two
  make-or-break diagnostics that said "don't build"). Comma specialist: per-position P(comma) from
  (class, clause-position), same shape as boundary; beats base rate (val log-prob −0.159 vs −0.264,
  comma rate 8.1%). Wired into generation + the /evolang page (new **Commas** toggle) + the demo;
  retrained `demo/genomes.pkl` to include it. `build_comma_corpus`, `run_comma`. **Diagnostics
  (caution paid off):** (1) **continuity/topic-memory** idea killed — a running topic-average never
  beats the global mean and is identical on shuffled text (no exploitable continuity in a static
  corpus); (2) **local agreement** skipped — det-noun number agreement already subsumed by selection
  (pipeline gap +0.338 ≥ real +0.199), subject-verb isn't a local adjacent signal (+0.026 even in
  real text → needs parsing). The local producer is near its honest ceiling; the next *capability*
  (memory/coherence) lives in a conversational environment (Track B). WORDPIPE_FINDINGS.md.

- **[2026-07-07] (Claude)** — **/evolang page replaced with the WordPipe specialist pipeline**
  (the current evolution-native LM). Retired the outdated char-model page; new interactive page toggles
  each evolved genome (Vocabulary, Order, Selection prev/both, Boundary, Chunks) and generates live
  over the trained genomes. Backend `genreg_train/wordpipe_service.py` (lazy-loads corpus + `demo/
  genomes.pkl`), REST `/api/evolang/status` + `/api/evolang/generate`; rewrote `templates/evolang.html`
  + `static/evolang.js`. **Needs a Flask restart.** Also validated Track A: **bidirectional selection**
  beats prev-word-only (adj 0.767→0.777, distinct 0.219→0.231, logprob −0.940→−0.911 — kept), and the
  **chunk/phrase genome** is the biggest local-fluency win yet (adj 0.776→**0.851**, +7.5 pts; small
  repetition cost). `build_chunk_index`, `gen_chunked`, `run_gate7`, `run_gate_chunks`.

- **[2026-07-07] (Claude)** — **Visual demo** (`demo/demo.py`, pygame) — watch the language build up
  genome by genome. Trains each specialist live (fitness sparklines emerge) then lets you toggle
  layers and see the output transform in real time: nothing → letter-gibberish, +Vocabulary → real
  words, +Order → grammatical class skeleton, +Selection (prev-word or Bidirectional) → context-fit
  words, +Boundary → sentences. First run trains ~4 min then caches to `demo/genomes.pkl` (instant
  after); `demo/build_cache.py` pre-builds headless. `demo/README.md`.

- **[2026-07-07] (Claude)** — **WordPipe Track A — bidirectional selection.** Word choice depends on
  BOTH neighbours; new specialist scores a candidate against the previous WORD and the next CLASS
  (known from the order skeleton) — two bilinear heads (~1150 params), dense in-class-negative fitness,
  purely local. `BiSelPop`, `class_centroids`, `run_biselection`, `run_gate7`. Framing correction
  (user): local production is NOT a ceiling — memory isn't required by a non-conversational corpus
  world (P1); it belongs to a conversational environment (Track B) where it's forced.

- **[2026-07-07] (Claude)** — **EEC memory investigation** (`genreg_train/eec_memory.py`, mirrors
  `project/EEC-main/engine/mind.py`): survival=lifespan on energy, memory-rent, occlusion/entropy as
  memory-forcing laws, graded by emergent STATE (recurrent gain, horizon) not accuracy. Confirms **P1
  (reachability)**: memory ~5× stronger where it pays (long-range: gain 0.55, M 6, horizon 3.3) than
  our language stream (gain 0.12, M 2, horizon ~1) — the language world is local. Lesson: don't grade
  by loss-vs-baseline; read the state; a capability emerges only in the environment that requires it.

- **[2026-07-07] (Claude)** — **WordPipe G5 — sentence-boundary specialist; the pipeline makes
  SENTENCES.** Fifth specialist: per-position P(sentence ends) from (word's class, sentence-position),
  dense binary-prediction fitness, ~577 params. Beats base-rate boundary prediction (val log-prob
  −0.119 vs −0.174) and learned the corpus rhythm (5.4% boundary rate ≈ 18.6-word sentences). **Full
  4-specialist generation** (order + selection + boundary): produces real sentences (periods, caps),
  **gen sentence length 17.0 vs real 18.6**. Sample: "…the good rich and him news we rested for more
  they would free tea cruel and mad. … she was long graceful. And good the dress and easily free…".
  Real local bigrams + sentence structure from 5 tiny gradient-free genomes (~7K params, ~140 KB).
  Honest limits: heavy "more/good" repetition (dominant induced class + high-freq fill) = the
  repetitive-collapse the EvoLang novelty constraint fights (next lever); no long-range grammar.
  Chose punctuation over agreement (agreement largely subsumed by selection; its distinct long-range
  part needs parsing). `BoundaryPop`, `build_boundary_corpus`, `run_boundary`, `run_gate5`.
  WORDPIPE_FINDINGS.md.

- **[2026-07-07] (Claude)** — **WordPipe G4 — word-selection specialist; +15 pts fluency.** Third
  specialist: fill each class slot with the word that fits the PREVIOUS word (not class-random).
  Applied both G2 lessons — the 4000-word representation is FIXED (SVD of distributional
  co-occurrence, out of the search space) so only a tiny **bilinear head M (24×24 ≈ 577 params)**
  evolves; fitness is DENSE predictive (log-prob of the true word among in-class negatives).
  Standalone beats the frequency baseline (val log-prob −0.931 vs −0.981; top-1 0.688 vs 0.666). In
  the pipeline it **compounds: adj-pair-hit 0.610 → 0.761 (+15 pts)** — selection output is full of
  real bigrams ("he was", "the eyes", "and quickly") where random-fill jars ("is are", "gay health").
  Deploy size now ~140 KB int8 (dominated by the 4000×24 feature table; genomes ~6.6K params total).
  Not fluent yet (no agreement/punctuation/long-range) but each specialist visibly closes the gap —
  the decomposition compounds. `WordSelPop`, `word_features`, `run_selection`, `run_gate34`.
  WORDPIPE_FINDINGS.md.

- **[2026-07-06] (Claude)** — **WordPipe G3 — the two specialists COMPOSE into text.** Chained the
  proven order genome (emits a class skeleton) + vocabulary component (fills each class slot with a
  freq-weighted real word). Decisive test isolates class order: pipeline vs a unigram-class baseline
  using the IDENTICAL fillers. **Pipeline 0.604 vs unigram 0.558** on adjacent-word-pair corpus-hit
  (local English-likeness), and the gap GREW as the order genome trained (0.582/0.561 @300 gens →
  0.604/0.558 @3000) — so the evolved class order drives it, not the fillers. Output is real-words-
  in-a-grammatical-skeleton ("clear of my world", "he up and active"), measurably better than the
  unordered salad, not yet fluent (32-class skeleton coarse; slots filled class-randomly — next
  specialist = word-selection-given-neighbours). **Deploy size as-is ~50 KB** (genomes ~6K params /
  ~24 KB; lexicon ~38 KB — the word list outweighs the net; ~0.005% of GPT-2). `run_gate3`,
  `gen_class_seq`, `build_class_words`. Pipeline vision validated end-to-end; WORDPIPE_FINDINGS.md.

- **[2026-07-06] (Claude)** — **WordPipe G2 CORRECTION — the order specialist DOES evolve.** Prior
  entry's "gradient-free wall / evolution can't" was wrong (and un-GENREG: it blamed the tool for a
  badly-shaped space). Two design errors, fixed in order: **(1) space ballooned** — 4000-word
  embedding (40k params); shrank to **~32 induced POS-like classes** (`induce_word_classes`, k-means
  on anchor-context; classes come out clean: past-participles, past-tense verbs, adjectives),
  embedding → ~256 params. Still chance. **(2) fitness wrong SHAPE** — discriminative "is this window
  real?" is one holistic bit, no per-position gradient → flat landscape (margin fitness didn't help
  either). Reframed the order specialist as a **PREDICTOR** (next-class LM, dense log-prob fitness):
  **climbs, val_ppl 30.9 → 12.75, beats unigram 13.4** (count-based ceiling 10.06; longer run
  pending). Lesson (pure GENREG): when a specialist won't evolve, **shrink the space AND reshape the
  fitness until the landscape is dense** — don't conclude evolution can't. Both levers were needed.
  `run_class_lm`, `run_disc_on(fitness=...)`, `OrderPop.fitness_all`. Updated WORDPIPE_FINDINGS.md.

- **[2026-07-06] (Claude)** — **WordPipe — specialist-pipeline experiment** (`genreg_train/wordpipe.py`,
  user's vision: "the constraint IS the genome's reason for existing"; decompose English into
  components, evolve a specialist per component). Built + gated three specialists, each proven before
  the next: **(G1) vocabulary/speller** — char genome rewarded for lexicon coverage (real words),
  char-prediction-scaffolded so it bootstraps; **(G2) order discriminator** — genome evolved to tell
  real corpus word-order from *within-window shuffled* order (a grammaticality LANDSCAPE, not an
  n-gram table — fakes share the bag-of-words so only ORDER distinguishes them); **(G3) orderer** —
  word generator scored ONLY by the frozen discriminator's P(real), never next-word accuracy (which
  would rebuild the table). Lexicon (20,676 words) + word corpus built from the Gutenberg dump. All
  gradient-free (shared tournament/elitism/energy-homeostasis `ga_step`). **Results
  (`documentation/WORDPIPE_FINDINGS.md`): G1 PASS** — vocabulary genome raises valid tokens 18.9% →
  **52.4%** vs a plain char LM (coverage is an evolvable specialist). **G2 FAIL** — the order
  discriminator stayed at chance (51.9%) across 2500 gens, yet a bigram probe separates real/shuffled
  at **69.2%**, so the signal exists — the discriminator just can't evolve a 4000-word embedding
  (~40k params) by mutation: the gradient-free representation wall. G3 skipped. **Boundary mapped:** a
  specialist is evolvable gradient-free only if its representation is small (chars ✓) or scaffolded;
  word-level order is too big. Fix points where linguistics does — **order over ~30 POS categories,
  not 4000 words.** The pipeline vision holds; the evolvable-specialist line is now sharp.

- **[2026-07-06] (Claude)** — EvoLang: **autonomous experiment battery + findings** (ran while user
  away). Held-out `val_ppl` throughout. (1) Capacity sweep K∈{4,6,8}×H∈{32,48}: **K4/H48 wins
  (val 13.84)**; bigger context does NOT help this tiny genome (K6/K8·H48 overfit) — spend capacity
  on hidden width, not window length. (2) Novelty A/B on the winner: novelty **monotonically hurts
  perplexity** (off 13.19 → w0.3 13.67 → w0.6 14.21 → w1.0 18.11, collapsed) — it's a variety
  (anti-collapse) lever, a Goodhart trade against accuracy, NOT a ppl improver; keep off by default.
  (3) Long run (6000 gens) → **val 12.25** (train 8.99): more gens buy memorisation, not
  generalisation — the bottleneck is model class, not budget. Next levers: composition of specialised
  genomes / recurrent evolved state / landscape shaping (not gradients, not tables). Updated `/evolang`
  defaults to the winner (K4 H48 E12). Full writeup: `documentation/EVOLANG_FINDINGS.md`.

- **[2026-07-06] (Claude)** — EvoLang: **held-out validation split.** Training now samples windows
  only from the first `(1 - holdout_frac)` (default 0.9) of the corpus; the champion's perplexity is
  also measured every 100 gens on a fixed 4096-window sample from the reserved tail. `gen`/`done`
  events carry `val_ppl` alongside train `ppl`; the ppl tile shows "train / val". Makes every EvoLang
  number honest about generalisation (early runs show train 15 / val 19). genreg_train/evolang.py.

- **[2026-07-06] (Claude)** — EvoLang: **swapped the toy corpus for the Gutenberg book dump.**
  Removed the 642-char hand-written string; EvoLang now trains on
  `project/EEC-main/engine/corpus.txt` (~48.6M chars of real English). Fixed small charset (37:
  space + lowercase + basic punctuation, digits→'#', other→space) so the genome stays tiny. Windows
  are sampled **on the fly** per generation — the ~49M (context→next-char) pairs are never
  materialised (that'd be GBs); only the flat int16 id array (97 MB) is cached, lazily on first run
  so Flask import stays ~0.3s. The page shows a **preview** slice, not the whole file (never ships
  49 MB over the socket); `started` now carries `corpus_chars`. Perplexity is honestly higher than
  the toy (real text is far more varied; needs more generations). ~58 ms/gen. genreg_train/evolang.py.

- **[2026-07-06] (Claude)** — EvoLang: **novelty constraint (opt-in, pure reward).** A landscape
  lever that fights repetitive collapse ("the the and and"). Each genome is sampled for a short
  passage (batched across the whole population — one forward per char step, no per-genome loop) and
  a novelty scalar on a 0.00–1.00 scale accrues from its words: each step the scalar decays by
  `decay` (0.005 default); a word pays `gain` (0.15) × a cooldown ramp that goes 0→full over
  `cooldown` (40) words since it was last used — so "the" hammered constantly stays on cooldown → ~0,
  a long-unseen word like "normandy" pays full. Only ever gained, capped at 1.0, never a penalty.
  It **multiplies** the genome's own soft fitness by `(1 + weight × novelty)` (0.5 default → up to
  ×1.5) — a scaled boost done in log-space (`base + log1p(weight×novelty)`), so the absolute lift is
  proportional to how well the genome already predicts. Checkbox to enable + fields for gain / decay
  / cooldown / weight / rollout-length. Perplexity tile stays honest (from raw log-prob, not the
  bonus); new champion-novelty tile. Verified: repeated "the"×12 → 0.136, distinct words → 1.00;
  ~1.5× per-gen cost when on. genreg_train/evolang.py, templates/evolang.html, static/evolang.js.

- **[2026-07-06] (Claude)** — **PIVOT: archived the entire n-gram / LM / Tree line, started EvoLang.**
  Per the user: stop chasing n-gram tables (they're the 1990s, and the distillation verdict proved
  you can't gradient-free-train them away). We are NOT building attention, an optimizer, or a
  distilled table — we're building a new *type* of evolution-native LM. Archived
  `genreg_lm/attn/enc/trustmix/distill`, `lm_sample`, `genreg_rerank`, `pure_engine`, `tree_service`,
  `tree_lm` → `archive/lm_and_tree/` (README included); moved `lm/attn/enc/encoder/tree/distill/pure`
  run dirs → `runs/_archive/`. New **`/evolang`** page + `genreg_train/evolang.py`: one small fixed
  corpus, a tiny neural next-char predictor per genome (context → evolved embedding → tanh → V
  logits), evolved by tournament + elitism + mandatory energy homeostasis, self-adaptive mutation,
  **soft** log-prob fitness — no gradients, no lookup table. Minimal by design ("do nothing else").
  Nav Tree LM → EvoLang; guarded `runstore.py`'s archived `tree_service` imports so `/runs` still
  loads legacy runs. Rationale: `documentation/EVOLANG_PIVOT.md`.

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

- **[2026-07-06] (Claude)** — **PER-LAYER ("tissue") constraints — new engine + validated.**
  User's idea: evaluate each constraint against its WIRED LAYER'S activations instead of the
  whole network, so layers face different survival conditions like tissues — same tournament /
  energy / whole-organism reproduction, landscape just becomes locally uneven. Built
  `genreg_train/pure_engine.py` (multi-layer evolved MLP, per-neuron activations, energy
  homeostasis; the evolving backend PURE's node graph will feed). Decisive 4-condition A/B
  (next-char, matched compute): Energy→L1 crushes L1 power to 0.039 (8x below control) while L2
  stays 0.34; SWAP the wire (Energy→L2) and the collapse follows to L2 (0.103) while L1 stays
  high — differentiation is CAUSAL, controlled by the wiring. Global eval compresses both
  layers uniformly (no differentiation). Held-out top-1 preserved at 20.8% across all
  constrained runs (cooperation intact). Consequential Drive holds a wired layer's dead-neuron
  fraction at 0%; its absence + energy starvation grows 18.8% dead (constraints compose as
  predicted). Card + full table: documentation/PURE_PER_LAYER_CONSTRAINTS.md.

- **[2026-07-06] (Claude)** — Animation shape classifier **SOLVED** — identifies all 10 shapes
  regardless of motion pattern/position, gradient-free (mutation-only evolution). Recipe:
  **centroid normalization** (crop a fixed 20×20 window around each frame's white-pixel centroid —
  generic translation-invariance operator, tracks the shape as it moves, removes position while
  keeping all arrangement) + an **evolved MLP** (400→24→10, soft geo-mean fitness, self-adaptive
  mutation, per-neuron activations). VERIFIED §VII across 3 random 80/20 per-clip splits:
  held-out 1.00 / 0.96 / 1.00, every shape, train→heldout drop ~0% (majority baseline 0.10),
  converges <150 gens (~15s). Saved: `animations/anim_shape_evo_SOLVED.py`, `_verify.py`,
  `anim_shape_evo_card.md`. Findings: plain per-frame MLP memorizes position (train 0.8/heldout 0.15);
  evolved conv+global-maxpool generalizes but plateaus ~2-4 shapes (global pool discards arrangement);
  the per-clip energy optimizer is the weak link (0.44 even on separable features) vs full-batch
  elitist. NOTE: uses full-batch + centroid preprocessor, not the one-clip-per-gen energy regime;
  /animation page not yet updated with the winning pipeline (pending approach confirmation).

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

- **[2026-07-06] (Claude)** — Tree LM page: **copy refresh to reflect byte AND word/token level**
  (user: "reflect what we are actually doing now"). The page was live and functional but its
  copy was byte-only and outdated. Fixed: brand subtitle "byte-level" → "byte & word-level";
  "Context window (bytes)" → "(bytes / tokens)" with a word-mode hint; routing-layers hint
  generalized (hardcoded "256 bytes" → "vocab / branch^layers", + the word-level layers-0
  collapse warning); cluster-split, generate, routing-inspector, and icicle card copy updated
  to say token/vocabulary not byte. treelm.js: the icicle x-axis was hardcoded to /256 (word-
  mode trees rendered off the right edge) — now derives the vocab span from the widest node
  and labels the axis "byte 0..255" or "token 0..N" by mode; encoder status "next-byte" →
  "next-token (ridge / nearest-centroid)"; tooltip "bytes" → "tokens". No functional/training
  change; the page still does live GPU training, generation, sweeps, and the routing inspector.
  Static+template only — reload the browser.

- **[2026-07-06] (Claude)** — **PURE: server-side video decode — ANY format (mkv/avi/mov/…).** The
  browser `<video>` element only decodes mp4/webm/ogg, so the user's MKV failed with "zero
  dimensions". Added a Flask endpoint `POST /api/pure/frames` that decodes server-side via
  **imageio + bundled ffmpeg** (installed `imageio`, `imageio-ffmpeg`; added to requirements.txt) —
  handles essentially any format. Extracts frames per the Data node's size/start/skip/max/gray and
  returns 0-255 pixel arrays. Client `acquireFrames` tries the server first, falls back to browser
  decode if the route is 404 (server not restarted), and caches results per File+params so a big
  video isn't re-uploaded each Run. Broadened the Data node's file picker to accept any video.
  Verified end-to-end: an MKV → `/api/pure/frames` → 6×16×16 grayscale frames with real content
  (not black). **Requires a Flask restart** (new route) — until then MP4 still works via the
  browser fallback.

- **[2026-07-06] (Claude)** — **PURE: fix black-frame video decode + Data-node frame preview.**
  User reported every target frame decoded black. Root cause: `drawImage` ran before the video
  painted the seeked frame (and a `t=0` seek never fires `seeked`). Rewrote `grabFrame` to wait for
  the frame to actually present (`requestVideoFrameCallback`, else double-rAF), nudge the seek so an
  event always fires, append the video (hidden) to the DOM and use a `willReadFrequently` context,
  and decode on `loadeddata`. Critically, if every decoded frame is uniform/black the decode now
  **errors clearly** ("frames decoded but every one is uniform/black…") instead of silently feeding
  black frames. Added a **Preview frame** button in the Data node Properties that decodes + shows
  the first frame so you can verify the video feeds correctly without running training. Browser-
  only path (not headless-testable); module load + synthetic path verified intact. Ready on next
  Flask restart.

- **[2026-07-06] (Claude)** — **PURE: per-frame Reconstruction panel (all frames, not just one).**
  New bottom panel (resizable, `data-resize="recon"`) that shows the best prediction for EVERY
  frame: targets on the top row, the model's reconstruction on the bottom row, aligned as N×N
  images — so there are as many output frames as input frames. `sampleModel` now computes the
  best genome's output for every dataset frame (cap 48) when the model's output is a square frame
  (the decoder; a latent-output encoder shows nothing meaningful). `drawRecon` renders the grid
  in a horizontally-scrollable canvas; cleared on Reset. Headless-verified: 2-model autoencoder on
  32 image frames → the decoder's grid drew all 32 frame reconstructions. Ready on next Flask
  restart.

- **[2026-07-06] (Claude)** — **PURE: video Data node now feeds REAL frames (was mock).** Fixed the
  trust gap the user flagged — a Data (video) node previously did nothing; the engine trained on a
  synthetic sine fallback regardless. Now `extractVideoFrames` actually decodes the selected video
  (seek → draw into an N×N canvas → grayscale/RGB pixels in [-1,1]) and those frames become the
  training dataset. `Engine.start` is async: if the source is a Data node it decodes first
  (status "decoding video frames…"), and errors clearly ("no video file selected" / "could not
  decode") instead of silently falling back to mock. The run status now says **"training on REAL
  video"** vs **"training on synthetic data"** so it's never ambiguous. Output plot upgraded: when
  the output is a square frame it renders **target vs reconstruction as side-by-side images** so
  you can watch the model actually rebuild the frame. Verified headlessly: synthetic path intact +
  labeled; video-without-file gives a clear error, no silent mock. NOTE: the actual browser video
  decode (video/canvas APIs) can't be headless-tested — that path is verified in-browser only.
  STILL TODO (deferred, acknowledged): genome visual showing ALL models in a chain; saving PURE
  runs to the Runs page with playback.

- **[2026-07-06] (Claude)** — **PURE: Autoencoder template + template picker.** The toolbar
  Template button is now a picker (select) with two options: **Basic model** (the single
  Synthetic→Input→Layer→Output→Fitness + Energy example) and **Autoencoder** — a wired
  encoder→decoder chain (16→12→**4** latent, reconstruct → **4**→12→16, reconstruct_source) so the
  decoder rebuilds the original through the bottleneck. `templateGraph(kind)` +
  `window.PureGraph.template(kind)`. Verified both load and parse (autoencoder → 2 models with the
  right per-model objectives). Ready on next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: `reconstruct_source` fitness objective (real autoencoder
  chains).** Added a third Fitness objective, `reconstruct_source`, plus carrying the ORIGINAL
  pre-encoder input (`x0`) through the model chain unchanged. So a decoder wired after an encoder
  can be scored against the source input rather than the latent it receives — making encoder(D→L)
  → decoder(L→D) a true autoencoder. `makeTarget()` centralizes target selection; every sample now
  carries `x0` (= x at model 0, preserved through pass-throughs). Headless-verified: encoder
  8→6→3 (reconstruct) then decoder 3→6→8 (reconstruct_source) reconstructs the original 8-dim
  input through the 3-dim bottleneck to 0.026 error. Ready on next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: Data (video) source node + multi-model chains via Fitness
  pass-through.** (1) New **Data** node (Data group): pick an mp4/webm/ogg file in Properties, set
  Assumed FPS / Start frame / Skip (stride) / Max frames / Frame size / Grayscale, and it shows
  the computed total frames live (source ≈ duration×fps → after stride → after max cap). File kept
  in a runtime map (not persisted); duration read from video metadata. (2) **Fitness node now has
  a pass-through `data` output** (was `fit`) that forwards the model's Output onward, so you can
  chain multiple models — each Input→…→Output→Fitness segment is its own model with its OWN
  fitness objective/metric. Engine rewritten for this: `parseModels` walks the chain into ordered
  segments; models train **sequentially**, each with its own population/genes/data — model k>0's
  inputs are the previous model's trained outputs (freeze-and-compose, GENREG-style). Readout shows
  `model i/N`; the genome visual fires the active segment (engine supplies its columns). Headless-
  verified: 2-model chain trained both (model 1 mse→0.069, model 2 mae→0.164, different fitness
  each), single-model unchanged, frame math correct. NOTE: the engine still generates training data
  from the Synthetic node / a default; feeding actual decoded video frames from the Data node is
  the next step (async decode). Ready on next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: Template button (correctly-wired example).** Toolbar gains a
  **Template** button that loads the canonical graph so the correct wiring is visible: Synthetic →
  Input → Layer → Output → **Fitness** (Output feeds Fitness so its objective can score the model),
  with an **Energy** constraint wired into a Fitness input port to show the constraint→fitness
  pattern. Laid out spatially (Fitness below Output, constraint below the layer) so the wires read
  clearly. `templateGraph()` + `window.PureGraph.template()`; `seedDefault` now delegates to it.
  Confirm-guarded (replaces current graph). Headless-verified: all 5 edges present, fitness
  dynamic ports dense, buildSpec accepts (arch 16→24→10, constraint reported). Ready on next Flask
  restart.

- **[2026-07-06] (Claude)** — **PURE fix: unwired layers no longer appear connected / get trained.**
  Both the genome visual (`orderedStructure`) and the engine (`buildSpec`) were collecting ALL
  input/layer/output nodes by type and sorting by x, so a Layer dropped on the canvas but never
  wired still showed as a column (with connections) and would be built into the network. New
  shared `wiredPath()` follows the actual edges Input → hidden layers → Output and returns only
  the connected chain in true data-flow order; both consumers now build from it. Engine also
  errors clearly ("wire Input → … → Output") when the chain doesn't reach the Output instead of
  training a disconnected graph. Bonus: layer order now follows wiring, not canvas x-position.
  Headless-verified: floating layer excluded from visual + arch; wiring order 6→8→5→6 honored over
  x-order; broken chain errors. Ready on next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: neurons colored by saturation.** During a run the genome
  neurons now color yellow (barely used, |activation|≈0) → red (near full saturation, |activation|≈1)
  instead of the teal firing ramp. Uses the ABSOLUTE activation (clamped to 1), so "saturated"
  means genuinely near ±1 rather than merely high relative to its column (`satColor` replaces
  `fireColor`; dropped the per-column max normalization for fill). Panel caption updated. Verified
  ramp (t=0 yellow → t=1 red). Ready on next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: genome visual shows evolved layer width live.** The engine
  now reports each layer's effective active-neuron count for the best genome (`dims` in the tick
  sample); `renderGenome` renders an evolve-units column at that effective width, so the column
  physically shrinks/grows during a run and the caption shows `N / max` (accent) when below max.
  Connections only touch the active units. Also improved the GA: integer genes (width, k) now seed
  the initial population across [1, units] instead of all pinning at max, so the width/k dimension
  is explored from generation 0. Headless-verified: layer column varied 3→6→4 / 12 across a run,
  back to 12 idle. Ready on next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: evolvable hidden-neuron count on Layer nodes.** Layer
  Properties gain an **Evolve unit count** checkbox; the Units field becomes the max (labeled
  "Units (max if evolving)"). Implemented as a width gene: weights are allocated up to the max, and
  a per-genome gene picks how many units are active — the rest are masked (output 0, contribute
  nothing to the next layer). Generalized the gene machinery so a layer can carry a width gene
  and/or a k gene (both mutate in ~unit steps, clamp in the forward pass); `layerOut` centralizes
  the per-layer path (activation → width mask → k-sparsity). Badge shows `≤N` when evolving.
  Headless-verified: fixed units 95% lower error, evolve-units 93%, evolve-units+evolve-k 76%,
  evolve-units+fixed-k 91% — all learn, genes compose. Ready on next Flask restart. (Note: genome
  visual still shows the max-width column; masked units read 0 → yellow.)

- **[2026-07-06] (Claude)** — **PURE: k-sparsity on Layer nodes (fixed or evolvable).** Layer node
  Properties gain a **k-sparsity** checkbox; when on, a **k (active units)** number field appears,
  or check **Evolve k** to make k a per-genome evolved gene instead of fixed (the k field hides
  when evolving). Implemented in the engine: after a layer's activation, `topK` keeps the k
  largest-magnitude units and zeros the rest. Fixed k reads the node value; evolve-k appends one
  gene per evolving layer past the weights, initialized to the set k, mutated in ~unit steps
  (`mutScale`), clamped to [1, units] per forward pass. Also generalized the property `when`
  mechanism to accept an array of conditions (all must hold) and made checkbox toggles re-render
  dependent fields. Layer badge shows `k4` / `k*` (evolving). Headless-verified: fixed k=3 learns
  to a higher error floor than unconstrained (6.2e-2 vs 3.0e-2 — the constraint genuinely bites),
  evolve-k adds 1 gene and still converges (84% lower). Not deployed live (user has a run in
  progress); ready on next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: genome neurons now show their connections.** The genome
  visual draws the edges between consecutive layers' neurons (fully-connected). Idle = faint
  structural skeleton (all connections, grey). During a run the engine exposes the best genome's
  weight matrices per layer (`wLayers` in the tick sample), and each connection is colored by
  weight sign (blue positive / red negative), opacity by |weight| strength, and brightened when
  its source neuron is firing — so you watch the actual evolved wiring change live. Connections
  drawn behind the neurons; capped columns connect their shown units by real unit index.
  Headless-verified: 120 faint lines idle → 120 weight-colored (81 blue / 39 red) mid-run → faint
  again after Reset. Live after the next Flask restart (not urgent — user has a run in progress).

- **[2026-07-06] (Claude)** — **PURE: Reset button + real-time genome firing.** Run panel gains a
  **Reset** (stops the engine, clears charts/metric, returns the genome visual to static, keeps the
  graph). The genome dot-columns now **fire in real time during a run**: the engine exposes the
  best genome's per-layer activations each generation (`activate()`, added to the tick sample),
  and `renderGenome(activations)` colors each dot by |activation| (dark→bright teal via
  `fireColor`), mapped to the correct unit even in capped/gap columns (`colLayoutG` now returns
  per-dot unit indices). `orderedStructure` reordered to data-flow order (input → layers-by-x →
  output) so columns align with the activation arrays. Headless-verified: dots color with 22
  distinct intensities mid-run; Reset returns them static. Live after the next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: engine wired up + Fitness node (the baseline GA runs).**
  Run/Stop now drive a real in-browser plain GA over the assembled graph: Synthetic data feeds
  the Input, flows through the hidden Layer(s) to the Output, and the population evolves toward
  the Fitness node's objective. New **Fitness node** (new "Objective" toolbar group) with a
  DYNAMIC input port set — it always keeps one open port, growing as you wire constraints in
  (wire the model Output into it plus any constraints; the engine finds nodes by type so exact
  placement is free). Implemented general dynamic-port support (`nodeInputs`/`normalizeDynamic` —
  incoming edges densify, spare "＋" port at the end) + selection-preserving `renderAll`.
  Engine (`Engine` in pure.js, exposed as `window.PureEngine`): seeded RNG, builds an MLP from
  input dims → layer units/activations → output classes, generates a dataset from the Synthetic
  node (waveforms windowed; images flattened), fitness = −MSE/MAE for objective reconstruct or
  predict_next, plain (μ,elite) GA with fixed mutation — NO energy/self-adaptation/evolved-acts
  yet (those are the constraints, added later and measured against this). Constraints wired into
  Fitness are collected and reported but do NOT yet affect the fitness math (baseline first).
  Left panel gains a live training readout: gen/best/mean-error, a best-fitness sparkline, and a
  best-output-vs-target plot. Seed graph now ships the full runnable flow
  (Synthetic→Input→Layer→Output→Fitness). Headless-verified: **error 1.29→0.10 (92% lower) over
  150 gens**; buildSpec + constraint collection correct. Live after the next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: Synthetic data-source node.** New "Synthetic" node (new
  "Data" toolbar group) that produces synthetic data to feed the model. Configured in Properties:
  a `kind` selector (sine · square · ramp · noise · image) whose parameters swap per kind —
  waveforms get frequency/amplitude/phase/samples, image gets pattern
  (gradient/checker/circle/stripes)/size/loop. Added a conditional-field mechanism (`when` on a
  prop) so only the relevant params show, and a **live preview** in the Properties panel (waveform
  sparkline or rendered image pattern) that updates as you edit. The Input node gained a `data`
  input port so Synthetic.out wires into it (Synthetic → Input). Toolbar generalized to render
  groups (Data/Structure/Constraints). Config + preview only — no engine consuming it yet.
  Headless-tested (node/ports, per-kind fields, groups). Live after the next Flask restart.

- **[2026-07-06] (Claude)** — **Standardized top navigation across the whole Flask app** (per
  user — nav buttons changed places / went missing per page). New shared Jinja partial
  `templates/_nav.html` renders ONE canonical set + order on every page:
  Build · Tree LM · DiffEvo · Animation · PURE · I2 · Runs · Docs. Each page includes it with
  `{% set nav_active = '<key>' %}` so the current page's link is marked active (highlighted,
  `aria-current`). Replaced the divergent hand-written nav in index/tree/diff/animation/pure/i2
  and added it to runs/docs (which had no project nav, only a Refresh button — kept). Dropped the
  inconsistent `target="_blank"` (index/i2 opened new tabs, others didn't) so switching is
  in-place everywhere; removed the ↗ arrows. Added `.runs-link.active` CSS. Verified: all 8
  templates render the identical 8-link nav in identical order with exactly one active. Live after
  the user's next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: resizable panels.** Drag-resizers on `/pure` — a column
  resizer on each side (Run panel width, Properties panel width) and a row resizer on the
  graph/genome split (drag to trade space between the node canvas and the genome visual). The
  graph canvas grows/shrinks as its neighbors move; the genome SVG now scales to fill its panel
  (flex column). Sizes clamp (sidebars 150-620px; genome min 60px, max = main height - 130) and
  persist to localStorage (`pure.size.left/right/genome`), reapplied on load. Handles highlight
  accent on hover/drag. `initResizers()` in `pure.js`, `.pg-resizer-col/.pg-resizer-row` CSS.
  Headless-tested (saved-size apply, drag delta, persistence). Live after the next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: genome visual re-added, now derived from the graph.** Split
  the `/pure` main area — node graph on top, a Genome dot-column panel below it (`renderGenome`
  in `pure.js`, `.pg-genome-panel` CSS). One column per dimensional structure node
  (Input/Layer/Output), left-to-right by position, node-dot style, colored to match the nodes;
  oversized layers cap with an ellipsis gap-marker while the caption shows the true unit count.
  Re-renders live on every node add/remove/resize (hooked into `save()`), so editing a node's
  units/dims updates the organism immediately. Headless-tested (seed columns, live layer add,
  ordering). Live after the user's next Flask restart.

- **[2026-07-06] (Claude)** — **PURE pivot: node-graph model assembler.** Rebuilt `/pure` as a
  drag-and-drop node editor (`static/pure.js` rewritten, `pure.html` restructured, graph CSS in
  style.css). You assemble a model by dropping nodes and wiring ports: left panel reduced to
  Run/Stop + global run settings (population/generations/seed); centre is the graph canvas
  (pan by dragging background, drag node headers to move, drag output→input port to wire, click a
  wire to delete); right panel is the Properties of the selected node. Dims/units/activation and
  every constraint's settings live INSIDE their nodes. Node catalog is data-driven: structure
  nodes (Input·Layer·Genome·Output) + the 12 constraint nodes from the user's canonical list —
  Energy, Temporal Budget, Consequential Drive, Capacity Cost, Observation Cost, Prediction Error,
  Information Gradient, Stimulus Stagnation, Predictive Variance, Homeostatic Proximity,
  Consolidation Threshold, Hazard Signal — each with typed, editable properties (e.g. Energy:
  cost-per-action / recover-on-consequence / decay / floor / ceiling). Constraints are wireable
  but NOT yet backed by any engine (per user: don't wire them up yet). Graph persists to
  localStorage; `window.PureGraph.getGraph()` serializes it for the eventual GA. Supersedes the
  previous non-wired genome viz + checkbox constraints panel (those were placeholders). Headless
  tested. Live after the user's next Flask restart.

- **[2026-07-06] (Claude)** — ~~PURE: slim right-hand constraints panel~~ (SUPERSEDED same day by
  the node-graph pivot; the checkbox list was replaced by constraint nodes with the correct list). New right
  sidebar on `/pure` listing the GENREG mechanisms that separate the framework from a plain GA,
  each an activatable checkbox, grouped: Fitness landscape (soft fitness, multiplicative,
  EMA smoothing, position-varying ground truth), Metabolism & selection (energy homeostasis,
  tournament+maturation gate, self-adaptive mutation, mutation floor, late anneal), Architecture
  primitives (per-neuron evolved activation, structural mutation, bootstrap/two-phase), Landscape
  design (basin/constraint deformation, novelty/fitness-sharing). 14 toggles, all OFF by default
  (PURE = the naked baseline). Data-driven render in `pure.js`, per-checkbox hint tooltips,
  state persisted to localStorage, header badge shows "baseline"/"N active", and a
  `pure:constraints` DOM event + `window.PureConstraints` API (`get/isActive/active/list`) expose
  the active set for the eventual wiring into the GA. CSS added for the slim panel (214px).
  Headless-tested (render, toggle, persist, count, API). Live after the user's next Flask restart.

- **[2026-07-06] (Claude)** — **PURE: genome visualizer, stage 1 (non-wired FFN skeleton)**
  (per user — wants a visual of the best genome when building models). New `static/pure.js` +
  `templates/pure.html` main card: renders the genome as three columns of circle "neurons" —
  Encoder · Hidden · Output, one column per layer — in the classic node-dot style, NO edges yet
  (wiring + per-genome weight coloring are the next stages; node geometry is retained on
  `host._nodes` for that). SVG, theme-var colored (encoder=tlm-s1, hidden=accent, output=tlm-s2),
  scales to container. Layer sizes are live-editable from three sidebar inputs
  (`PureGenome.setShape({encoder,hidden,output})`); oversized layers cap at ~21 visible dots with
  a centered ellipsis gap-marker while the caption still reports the true unit count (a 4096-input
  layer draws ~20 dots, not 4096). Headless render-tested (node counts, gap-marker, single-x
  columns, ascending y, captions). Live after the user's next Flask restart.

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

- **[2026-07-05] (Claude)** — **Fixed the Claude terminal button** — it typed
  `[?1;2cclaude --dangerously-skip-permissions` (PowerShell "Missing type name after '['"
  parse error). Cause: the button blind-fired on a 600 ms timer, racing the terminal's
  device-attributes handshake — the shell queries ESC[c during startup, xterm.js auto-replies
  ESC[?1;2c, and that reply landed on the not-yet-ready command line as literal text, then our
  command concatenated onto it. Fix (app.js): the command now waits for the actual PowerShell
  PROMPT in the terminal output (ANSI-stripped tail match, 5 s blind fallback), sends Escape
  first (PSReadLine RevertLine clears any stray reply chars), then types the command. Verified
  in a DOM harness: no input before the prompt, escape-then-command after it, fallback fires.
  Static-only — reload the browser.

- **[2026-07-05] (Claude)** — **PURE: new program page scaffold** (per user — pivot to a
  baseline-first campaign). `/pure` route in `app.py` + `templates/pure.html`: blank page in the
  house style (topbar + nav, config sidebar placeholder, empty main card) with the shared
  daemon-backed terminal dock (xterm/termdock/app.js/agentpanel). PURE will hold the very first
  baseline model — a textbook GA with nothing added — that every GENREG bell and whistle gets
  measured against, added one at a time. No model/WS backend yet, deliberately (user: blank page
  then stop). PURE link added to the build page nav. Live after the user's next Flask restart.

- **[2026-07-05] (Claude)** — **Encoder separated into its own model: `enc_char_v1`**
  (`genreg_train/genreg_enc.py`, card `documentation/LM_ENCODER_COMPONENT.md`) — per user +
  §X component-first. Fitness = evolved-head decodability at horizons {1,2,4} from the hidden
  state (equal weight — h=1 specialists sink), breeding the STATE rather than the prediction;
  heads are scaffolding, the frozen deliverable is (E, W_in, b_h, act). Skip-gram baselines
  measured first (skip2 22.5%, skip4 20.1%). genreg_lm gained composed mode
  (`encoder_ckpt` → tensors copied + FROZEN; mutation excludes encoder incl. act ids — §X
  freeze-and-compose, never retrained). Warm-start smoke: h1 preserved at 31.3%, h4 at its
  bar in 30 gens. Encoder sweep → composition pipeline running; composed bar > 31.9%.

- **[2026-07-05] (Claude)** — **Published to GitHub: A1CST/GENREG-LAB (private).** Full program
  snapshot pushed — engine, Tree LM, DiffEvo, Animation, LM campaign, panels, docs — with the
  I2 program excluded per user decision (also caught + fixed a real hazard: the old .gitignore
  had a trailing comment ON the pattern line, so `i2_admin_key.json` didn't actually match and
  the PRIVATE KEY was staged — pattern fixed, key never pushed, verified absent from the
  remote tree). Also excluded: runs/ (577MB runtime artifacts, untracked in a follow-up
  commit), dist/, i2_store/, agent_store/, corpora/. Local remote name `genreg-lab`; origin
  (GENREG-Builder) untouched. Final remote tree: 557 files, zero i2/key/runs leaks (verified
  via API).

- **[2026-07-05] (Claude)** — **Per-project changelogs.** Each project page's Changelog button
  now defaults to a project-scoped log with a "show all projects" toggle (app.js modal, all
  pages incl. Animation). Files live in `documentation/changelogs/CHANGELOG_{BUILD,TREE,
  DIFFEVO,ANIMATION,I2,LM}.md` — served live via the existing docs API (no Flask restart) and
  visible on /docs. Seeded from the main changelog by keyword split (best effort — a few
  cross-referencing entries appear in multiple projects). **Convention from now on: every
  change is logged in the MAIN CHANGELOG.md AND appended to the matching project file(s).**
  Verified: files serve 200 live; DOM test confirms project-default + toggle behavior.

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
- **[2026-07-05] (Claude)** — I2: **ZetiFile — the latent encyclopedia** (per user: local Wikipedia,
  new page, always first in search). New `zetifile.py`: serves articles from a LOCAL Wikipedia
  **multistream** dump (`corpora/wikipedia/`, gitignored, NEVER pushed to the network — code ships
  everywhere, corpus stays on this machine). Random access: seek to the block offset -> decompress
  ONE bz2 stream -> extract the page; ~260ms per article over HTTP, no extraction of the 22GB dump
  ever. Title index = SQLite (19,103,098 titles incl. redirects; exact + prefix + FTS5 word search;
  built in ~3 min from the index file). Wikitext stripped to clean blocks (headings/paras/lists;
  templates, refs, tables, file links removed; redirects resolved). Routes `/api/i2/zetifile/{info,
  search,article}` on primary + child; the child serves from its local corpus and **injects ZetiFile
  hits ahead of primary results in /api/i2/search** — ZetiFile is always the first search result
  group (score 100000+). Browser: new `i2://zetifile` app page (serif wordmark + home search,
  result list, wiki-style article view with sections; no emojis), deep links `i2://zetifile/<title>`,
  search-results integration. Articles beyond the still-downloading portion of the dump explain
  themselves ("sits at byte X, only Y downloaded"). Corpus download (enwiki multistream ~22GB +
  index) running in background via resumable curl. E2E tested (info/search/article/search-first).
  **Deployed**: whitelist widened for `zetifile.py`, pushed to primary (reports available:false
  there — by design, no corpus), dist/i2_child refreshed (child on this machine serves it).
  **Follow-ups**: corpus download completed (26.4GB; deep-offset articles verified). ZetiFile
  added to the network LANDING PAGE (per user): republished the primary's `home` with a
  ZetiFile bullet + `[[zetifile]]` link; flushed the child's page cache via the new
  flush-cache API and verified the updated home decodes through the child.
  **REDESIGN (per user — the corpus must feed the LATENT SPACE, not be a serving path)**:
  ZetiFile v2 = on-demand latentization. The dump is feedstock only. On first read of
  `i2://zetifile/<slug>` the corpus-holding child extracts the article, renders it as page
  markup, genome-encodes it, signs it with its node key, caches it as an ORDINARY page doc
  (`Child._zetifile_latentize`), and best-effort publishes it to the primary (`zetifile/` is
  an open ingest namespace in `publish()`) — after that the article IS a network latent,
  served/verified/cached by the standard page pipeline everywhere, offline included. Articles
  render in the canvas page renderer (app deep-links removed); search targets are latent page
  names `zetifile/<slug>` (slug = readable prefix + title-hash6; 19.1M-row slug table backfilled).
  E2E: offline latentize 86KB -> ratio 0.368 in 2.95s verified; push to primary verified;
  corpus-less child received + verified the latent from the network; search-first intact.
  Child v1.6.0. Deployed: primary pushed (incl. zetifile.py), dist/i2_child refreshed —
  restart the child to activate.

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
- **[2026-07-05] (Claude)** — I2: **YouTube-style social layer across the media pages** (per user:
  comments, likes/downvotes, thumbnails, details, up-next). Backend (`i2_service.py`): likes
  extended to **signed up/down votes** (value 1/-1/0, one vote per DID, up/down mutually
  exclusive; legacy like/unlike wire format + `count` field unchanged); new **signed comments**
  (`i2_store/comments/<target>.json`, sig covers `sha256(text)`, ≤1000 chars, ≤500/target,
  newest-first, works on any content id); **video thumbnails** (optional poster frame at upload,
  jpeg/png/webp ≤200KB, stored genome-coded under key id `<vid>:thumb`, served via
  `/api/i2/video/<id>/thumb`); **view counter** (`/viewed` POST, unsigned/node-local, stated
  openly). Routes added to primary (`i2_node.py`) + child proxy (`i2_child.py`, v1.5.0).
  Browser (`static/i2.js`): shared `voteRow`/`commentSection` components (theme-var driven);
  Latent Stream grid gets real thumbs + views + age; watch page gets channel row, 👍/👎, view
  count, description panel, full comment section, view-ranked Up-next with thumbs; upload
  auto-captures a 480px poster frame (canvas, preview shown); Woven posts get votes + 💬
  (inline thread for images, jump-to-watch for videos); Gallery cells get votes + age.
  Integration-tested against an ephemeral primary (votes up/down/switch/clear, comments incl.
  forged-sig 400, views, thumb coded-at-rest round-trip + 404) — all pass. **Deployed**: pushed
  to primary 10.0.0.15 (restarted, endpoints verified live; backup `_backup/push-20260705-094110`),
  `dist/i2_child` refreshed in place to v1.5.0 (running child needs a restart).
  **Follow-up (per user, hard rule): NO emojis anywhere in the UI** — vote buttons are now
  monochrome ▲/▼, the comments chip is plain text ("N comments"), image tiles/Gallery headers
  use ▦, search placeholder and DMV headers de-emojied; typographic marks (✓ ✗ ⚠ ▶) stay.
  Re-pushed to primary + dist child refreshed.

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

- **[2026-07-05] (Claude)** — I2 child: **flush-cache control** (child v1.4.0). New
  `Child.flush_cache(categories)` — wipes cached pages + sealed media blobs (all or per
  category), keeps genome/identity/config/purge state; returns files + bytes freed. Exposed
  as a confirm-guarded "Flush cache" button on the GUI Cache tab and as same-origin
  `POST /api/i2/child/flush-cache` (optional `{"categories": [...]}`) for headless children.
  Tested: full + selective flush, route incl. bad-category 400; genome untouched.
  Deployed to `dist/i2_child` by copying the code files in place (NOT via `package_i2.py`,
  whose rebuild would have wiped the live child's `i2_cache/` inside that folder);
  `__pycache__` cleared. The running child picks it up on its next restart.

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

- **[2026-07-05] (Claude)** — Tree LM evolve-embed: **mixed-dimension seeding — NEW BEST
  word-level result, −1.67 pts vs bigram.** The drift problem (64→66 in 100 gens) is fixed by
  seeding the population with REAL heuristic genomes at several capacity levels ([start/2,
  start, 2×start, 4×start] — built per level with the co-occurrence embedding seed), so
  selection compares whole capacity levels head-to-head from generation 0 instead of growing
  one random column at a time. Measured (run `20260705-000924-tree-1ce73a`, config J + evolve-
  embed from 64, ladder [32,64,128,256]): the whole population converged to embed **256**
  within 10 generations; encoder ridge acc 0.148 (drift version: 0.129); final **13.20% vs
  word-bigram 14.88%**, beating hand-set embed 128 (−2.15). Campaign delta progression:
  −7.9 → −6.1 → −4.5 → −2.15 → **−1.67**. Since evolution chose a dim above the manual clamp,
  the sidebar **Embed dim max is raised 128 → 256** (parse_config + tree.html, with a word-mode
  hint). `_evolve_encoder_embed` now takes (genome, dim) seed pairs.

- **[2026-07-05] (Claude)** — I2: **cache keys are now network-issued and RAM-only (v2, per user:
  "never at rest, changes each time")**. New primary endpoint `POST /api/i2/node/cachekey`
  (`i2_node.py`) + `i2_service.issue_cache_master()`: child sends a DID-signed, freshness-checked
  request (same trust bar as heartbeats); primary derives `HMAC(node-key, did)` — deterministic,
  nothing stored — and records the issuance (never the key) on the ledger. Child (`i2_child.py`):
  master fetched at connect, held ONLY in RAM; every cache write seals with a one-off key
  `HKDF(master, fresh 16B salt, kind|mid)` → AES-256-GCM (`I2EC\x02` header carries the salt, so
  decryption is stateless). No key file exists at rest anymore: v1 blobs are re-sealed and
  `cache_key.json` deleted on first issuance; identity-less children keep the v1 local-key
  fallback. If no key is obtainable (offline restart), sealed files are kept (KeyError ≠ tamper)
  and nothing is ever written plaintext. `i2_bench.py cache` extended (10 checks, all pass);
  `transfer` now runs a child end-to-end: key issuance 13ms, cold fetch+verify+seal 77.7 MB/s,
  warm sealed-cache hit 322 MB/s, restart+re-key 272 MB/s (8MB video, loopback). **Deployed**:
  pushed v1.3.0 to primary 10.0.0.15 via signed admin push (self-restarted; endpoint verified
  live; backup `_backup/push-20260704-232331`). Children need the updated `i2_child.py` to use it.
  (1) `i2_child.py`: cached media blobs (`videos/*.bin`, `images/*.bin`) are now sealed with
  AES-256-GCM under a per-child random 256-bit key (`<cache>/keys/cache_key.json`) — the
  genome-CTR coding alone gave no at-rest confidentiality since its key derives from the
  *public* genome hash + content id. Legacy plaintext cache files still serve and are migrated
  to sealed form on first hit; tampered/foreign-key files are dropped and refetched (GCM auth,
  AAD binds kind+id). (2) New `i2_bench.py`: `verify` (codec round-trip + measured bits/byte —
  **2.271 b/B, 3.52×** on corpus samples, beats zlib-9 at 4.14 b/B; claim confirmed), `cache`
  (seal/unseal/tamper/wrong-key self-test, all pass), `transfer` (spawns ephemeral primary,
  measures genome/page/blob transfer: video blob ~114 MB/s wire + 1.5 GB/s AES decode over
  loopback; text limited by the pure-Python coder at ~126 KB/s decode → compression wins on
  links slower than ~90 KB/s, decoder is the bottleneck above that). (`encoder_evolve_embed`,
  per user) + one word-level test. New variable-embed GA (`_evolve_encoder_embed`): each
  individual carries its own embed_dim; growing appends an embedding column AND its mixer row
  in every position block, shrinking drops the lowest-magnitude column, crossover in the shared
  embed subspace; bounds [8, max(4×start, 256)]; CPU (ragged genomes), shallow encoder only,
  mutually exclusive with evolve-dims; evolved dim recorded in the encoder card
  (`embed_dim`/`start_embed_dim`/`evolve_embed`). Also fixed `ctx_best` scoring to use the
  final encoder (correct for any evolved dims). **Test result (run 20260704-233101-tree-b6a79f,
  config J + evolve-embed from 64, 100 gens):** evolution drifted 64 → 66 only and plateaued at
  0.129 ridge acc by gen 20; final 11.95% vs bigram 14.88% (−2.92 pts) — WORSE than hand-set
  embed 128 (−2.15). Honest read: one-column-at-a-time growth has no fitness gradient (a fresh
  random column helps only after it learns something), so pure-accuracy evolution can't climb
  to 128+ in 100 gens — matches the EEC finding that size only changes direction under
  pressure. Promising follow-up: seed the population at MIXED dims (64/128/192/256) so
  selection picks the level directly instead of drifting one column at a time.

- **[2026-07-04] (Claude)** — Tree LM word-level: **first test campaign (10 configs, RTX 4080)
  — best −2.15 pts vs word-bigram; full findings in
  `documentation/WORD_LEVEL_TREE_LM.md`.** Delta progression −7.9 → −2.15 across rounds by:
  embed dim 24→128 (the binding constraint at word level — 24 dims can't carry which-of-2048-
  words identity), killing the encoder speed phase (TRAP: `encoder_speed_generations` defaults
  to 40 in the trainer and DEGRADES the evolved encoder 0.148→0.10 ridge acc before tree
  building — UI runs safe, scripted/sweep runs that omit the key are not), and window 8 (beats
  4 and 16 at this capacity). Best run `20260704-231826-tree-ee861e`: 12.7% real-word accuracy
  vs bigram 14.9%; generation is grammatical phrase structure ("in the world of the right of
  the world, and she went not…"). Honest read: the remaining gap is structural (exact count
  table vs compressed continuous encoder — the I2 lesson again); the lever that should PASS
  bigram is count-backoff blending at the leaves (documented as next work). Known issue:
  routing_layers 0 (flat 2048-way leaf) collapses to 1.7% — GA regresses off its ridge seed on
  a 526k-param genome; use routed trees at word level.

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

- **[2026-07-04] (Claude)** — Tree LM: **word/token-level prediction** (user: byte level can't
  get sentence structure — "let the model develop actual semantic embeddings"). Implemented as
  a **Token level** mode on the existing tree page (the engine was already vocab-parameterized;
  a v2 page would have duplicated ~2000 lines for two fields), byte mode untouched/default.
  - **Tokenizer** (`WordVocab`/`load_word_tokens`): corpus decoded UTF-8 with typographic
    punctuation folded to ASCII (curly quotes had shattered into junk vocab entries), tokens =
    words (apostrophes kept: "don't"), numbers, single punctuation; top-K vocabulary, id 0 =
    `<unk>` (2048 → 85.6% coverage of the 10.4M-token corpus); detokenizer with spacing rules.
    Cached per vocab size (~5 s once).
  - **`<unk>` policy**: never a training/eval TARGET (samples with unk targets dropped,
    oversampled to compensate — keeps accuracy honest on real words; without this the model
    just predicted unk forever) and never EMITTED at generation (masked in the leaf); it
    remains valid context. Applies to tree + encoder trainers.
  - **Config**: `token_mode` byte|word + `vocab_size` (256–8192); routing-layers leaf math now
    uses the real vocab; word-vocab reuse guard on saved encoders (byte encoders can't be
    reused in word runs). Bigram baseline now (V,V) int32. Co-occurrence embedding seed
    switches from full SVD to a Gaussian random projection above V=1024 (SVD would take
    minutes at 2048+). Saved models persist their vocabulary (`vocab_blob` in model.npz) so
    /runs replay + generation detokenize correctly; `_load_model` returns (root, encoder,
    vocab). Generation, trace inspector (word labels in samples/top-k), letters cloud (labels
    words when the run is word-mode), and word cloud (words ARE tokens — encoded by id) all
    mode-aware. UI: "Token level" selector + vocabulary-size field on /tree.
  - Verified: clean vocab top-15, tokenize/detokenize round-trip, end-to-end CPU smoke run
    (train → eval → generate readable words, zero <unk> in output → saved-model replay via
    /runs works with the persisted vocab). GPU word-level test campaign running; results to
    follow in the Agent panel. **Needs a Flask restart.**

- **[2026-07-04] (Claude)** — Run-Config panel: **config history + copy buttons.** New
  "Config history — all projects" section at the bottom of the panel: the newest 3/5/10 runs
  (selector persists in localStorage) ACROSS every project — tree, encoder, diffevo, engine
  envs mixed by recency. Each entry: status dot, project, time, and a program-aware one-line
  result (tree → `acc 35.2%`, encoder → `nc %`, diffevo → `L1 0.12`, engine → `score N`);
  click to expand the full flattened config + results inline (run id in the expansion deep-
  links to /runs#id); every entry has a **copy** button that puts the run's config on the
  clipboard as pretty JSON (navigator.clipboard with execCommand fallback, "copied ✓"
  feedback). The tracked active run also gained a **copy** button in the panel header.
  Backend: `runstore.recent_runs(limit≤50)` + `GET /api/run-history?limit=` (7–12 ms), polled
  every 12 s on ALL pages (history is cross-project, unlike the scoped active-run view).
  Verified: real-data cross-project history (tree+2048 interleaved) + 8/8 DOM-harness checks
  (render, headlines, expand/collapse, clipboard JSON round-trip, feedback, limit
  persistence). **Needs a Flask restart** (new route).

- **[2026-07-04] (Claude)** — Tree LM: **fixed "cannot reshape array of size N into shape
  (…)" when generating from a run on /runs.** Root cause: a run that REUSES a saved encoder
  (encoder_id) adopts the encoder's dims in memory (tree_service ~1759) AFTER config.json was
  written with the sidebar dims, so replay/embedding rebuilt the encoder with the wrong shape
  and `unpack_genome`'s reshape blew up. Two-part fix: (1) `_save_model` now records the
  encoder's REAL dims (`encoder_dims`) in model.npz and `_load_model` prefers them — new runs
  are self-describing; (2) legacy files: `_adopt_encoder_dims()` copies dims from the run's
  encoder_id (same adoption the trainer did) before loading — wired into infer_run,
  embedding_cloud, and word_cloud. Verified on a real reused-encoder run: reproduced the exact
  reshape error via the old path, then generated 400 clean chars (no U+FFFD — the UTF-8
  sampler visibly working) and both embedding clouds render. **Needs a Flask restart.**

- **[2026-07-04] (Claude)** — **Run-Config panel** — a second floating window (sibling of the
  Agent panel: draggable header, minimizes to a "Config" pill, position/state persist) that
  tracks the run belonging to THE PAGE YOU'RE ON, showing two sections: the run's FULL
  configuration (every setting, nested keys flattened e.g. `params.hidden`) and its RESULTS
  (live last-generation metrics while running; the full summary — accuracy/best/eval/etc. —
  once finished). Adoption rules exactly per spec: navigating to a page CLEARS the panel (it
  never shows a run that finished before you arrived); it adopts a run you start on that page
  (created ≥ page-entry, compared in local-time ISO to match runstore stamps); it adopts a run
  already mid-flight when you arrive and fills in results when it completes; a newer run
  replaces the tracked one. Status chip (running/finished/stopped/error) + "open run" button
  deep-linking to /runs#id. Scope per page: / → engine envs, /tree → tree+encoder, /diff →
  diffevo; other pages show an explanatory idle note. Backend: `runstore.latest_run(envs)` —
  newest run dir by id timestamp, reads only that run's config/summary + history tail
  (last_metric), measured 7.6 ms/call — served by `GET /api/active-run?scope=`, polled every
  4 s (works regardless of which client started the run, since it reads the run files on
  disk). Files: `static/configpanel.js` (all 6 pages), styles in style.css. Verified: 10/10
  jsdom state-machine checks (old-run cleared, adopt-on-start, config+live metrics, results
  fill-in on finish, chip transitions, mid-flight adoption, non-project message, newer-run
  replacement) + real-data latest_run for all three scopes. **Needs a Flask restart** (route).

- **[2026-07-04] (Claude)** — Runs dashboard: **word-level embedding cloud** alongside the
  letter one (user: "i need both word and letter"). The byte cloud can only ever show letters —
  the tree LM is byte-level, so 256 byte vectors is the whole embedding TABLE; words exist only
  as encoder computations. New `tree_service.word_cloud()`: takes the corpus's ~180 most
  frequent words (cached count over the 49 MB corpus; single letters dropped except a/i),
  encodes each as `' word'` through the run's trained context encoder (batched, left-padded
  like encode()'s own padding), PCAs the context vectors to 3D with the same variance/effective-
  rank diagnostics. `runstore.word_embedding()` + route `GET /api/runs/<rid>/words`. Run detail
  page now shows TWO cards for tree/encoder runs: "Embedding space — letters" (unchanged) and
  "Embedding space — words": drag-to-rotate scatter, dot size = word frequency, top-30 words
  labeled, hover any dot for its word + corpus count. Verified on a real tree run: 180 words,
  2.8 s cold / 0.07 s cached, effective rank 21 (real structure), coords unit-scaled.
  **Needs a Flask restart** (new route + service function).

- **[2026-07-04] (Claude)** — Agent board: **fixed duplicate notice ids across processes.**
  Flask and CLI posts each cached an in-memory id counter, so two writers handed out the same
  id (two id-5 notices existed in the live feed — which double-counted the unread badge).
  `post()` now re-scans the file's max id under the lock on every write; the existing feed was
  renumbered sequentially. Verified: simulated two fresh processes → sequential ids.
  Applies live for CLI posts; Flask picks it up on its next restart.

- **[2026-07-04] (Claude)** — **Config panels no longer reset when you leave the page.**
  Diagnosed the "configuration panel resets" report: the Build page's Control Panel already
  persisted (training.js `genreg_controls` — verified working with a jsdom repro of two page
  loads over the real markup), but the Tree LM and DiffEvo **Configuration sidebars had no
  persistence at all**. New `static/cfgpersist.js` (loaded on /tree and /diff after the page
  scripts): saves every input/select/textarea with an id (skipping the terminal dock / Agent
  panel / changelog modal) to localStorage per page on any edit, restores on load, then fires
  input+change so dependent UI re-syncs (e.g. the DiffEvo sampler field shown in denoise mode);
  selects whose options arrive async over the socket (the tree Encoder dropdown) re-apply their
  saved value via a MutationObserver once the option exists; a `restoring` flag prevents saving
  half-restored state. Verified end-to-end in jsdom: tree fields, diff fields, dependent
  visibility, and the async encoder select all restore across page loads. Static-only — browser
  reload picks it up, no Flask restart.

- **[2026-07-04] (Claude)** — Agent panel: **run notices are clickable** — any notice carrying a
  run_id (the auto run-alarms) navigates to `/runs#<run_id>`, landing on that exact run.
  runs.js deep-linking upgraded: hash handling is now a shared `openFromHash()` that also fires
  on `hashchange` (works when already on /runs) and reloads the run list first if the id isn't
  known yet (a just-finished run). Linked rows show a pointer cursor + hover highlight.

- **[2026-07-04] (Claude)** — Agent panel: **crashed runs now alarm too.** The run-alarm hooks
  only fired on done/sweep_done/stopped, so a run that died mid-training ended silently. Added
  `error` to the hooked event types in both job hubs and the engine `/train` socket;
  `post_run_event` turns it into a red `alert` notice ("<program> run FAILED" + the exception
  message, run_id when one exists). Verified: error event → alert posted.

- **[2026-07-04] (Claude)** — Agent panel: removed the robot emoji per user request — the header
  is just "Agent" and the minimized pill is a small uppercase "Agent" chip (mono font, same
  badge/pulse behavior). Docs and CLI docstring updated to match.

- **[2026-07-04] (Claude)** — **Agent panel** — a floating, draggable, minimizable notice feed
  on EVERY page (topmost, above content, below modals), the shared channel where AIs and
  automated jobs post updates/test results/alarms. New `agent_board.py` stores notices as JSON
  lines in `agent_store/notices.jsonl` (id/ts/kind/source/title/body/run_id; kinds info · test ·
  run · alert; trimmed to newest 1000) — file-based so posting works without the server. Routes:
  `GET/POST /api/agent/notices`. New CLI `agent_notify.py "title" ["body"|-] --kind --source`
  (stdin piping supported) for terminal AIs. **End-of-run alarms**: the engine `/train` socket
  (app.py) and both job hubs (tree_service — done/sweep_done/stopped — and diffuse_service —
  done) auto-post a `run` notice with run_id + headline metrics when any training job ends;
  hooks are best-effort (can never break training). UI `static/agentpanel.js` (all 6 pages):
  polls every 8 s, unread badge, drag-anywhere header (position/minimized state persist in
  localStorage), minimizes to a pulsing 🤖 pill, "mark read"; a notice arriving while a page is
  open POPS THE PANEL OPEN (the alarm). Docs: `documentation/AGENT_PANEL.md` (visible on /docs)
  incl. instructions for tying a CLI AI (e.g. Claude in the terminal dock) into the feed.
  Verified: board post/list/since/kind-coercion/id-continuity + run-event formatting all pass;
  CLI posted the first 3 live notices. **Needs a Flask restart** (routes + hub hooks) — the
  panel UI loads now but stays empty until then; same restart as the other 2026-07-04 changes.

- **[2026-07-04] (Claude)** — Runs dashboard: **filter / label / favorite / group / tags** —
  organization for tabs with ~200 runs. Per-run user metadata lives in a new `meta.json` in the
  run dir (own file so it never races the trainer's config/summary writes): `label` (≤80 chars,
  shown as the run's list title with the date after it), `favorite` (bool), `group` (≤80),
  `tags` (≤12 × ≤24 chars). `runstore.set_meta()` merges partial patches with length caps;
  `list_runs`/`get_run` carry the fields; new route `POST /api/runs/<rid>/meta`. UI (runs.html/
  runs.js/style.css): a filter toolbar under the env tabs — live search box (matches label, tags,
  group, id, date, device, constraints, status), status dropdown, sort (newest first — the new
  default — / oldest / best score), and a ★ favorites-only toggle; the run list groups into
  collapsible sections by `group` (named groups start collapsed, collapse state persists in
  localStorage per env+group; no groups → flat list as before), favorites pin to the top of
  their section, each row gets a ☆/★ toggle and tag chips; the detail panel gains a ★ button
  plus label/group/tags editor (group input autocompletes existing groups; Enter saves). Header
  shows "N/M shown" while filtering. Labels/tags render via textContent (no HTML injection).
  Verified: 228-run store round-trip (set/partial-patch/caps/unknown-id) all pass; JS/PY syntax
  clean. **Needs a Flask restart** (runstore + app.py route), same restart as the UTF-8 fix.

- **[2026-07-04] (Claude)** — Tree LM: **UTF-8-constrained sampling** — fixes the "� boxes"
  in generated text. Generation is per-byte but the corpus is UTF-8, so the sampler could emit
  stray 0x80–0xFF bytes that never form a valid sequence and decode to U+FFFD. `_generate`
  (Generate box + /runs replay) now masks each leaf pick to bytes that keep the stream valid
  UTF-8 (`_utf8_state`/`_utf8_allowed`/`_pick_byte` in tree_service.py): continuation bytes
  only inside an open multi-byte sequence (with the lead-specific second-byte ranges),
  otherwise ASCII + valid lead bytes (C2–F4); if a leaf can't continue an open sequence the
  unfinishable partial is popped (never into the prompt) and the pick retried; a leaf with no
  valid byte at all falls back to the old unmasked pick. Model, routing, training, and
  temperature semantics unchanged; the trace inspector (`trace_generate`) intentionally left
  raw. Verified: 27 unit checks incl. a 3000-step random-leaf generation decoding as STRICT
  valid UTF-8 with zero U+FFFD. **Needs a Flask restart** (Python module change).

- **[2026-07-04] (Claude)** — **Terminals on every project page + per-page tab memory.**
  The daemon-backed terminal dock (same PTYs as the build page — one shared `/ws` bridge, so
  a shell keeps running wherever you view it) is now on **DiffEvo (/diff), Tree LM (/tree),
  Runs (/runs), and Docs (/docs)** too, not just Build (/) and I2. New `static/termdock.js`
  injects the identical dock markup (tabs, + New Tab, Changelog/Claude/Clear/Restart/Stop,
  changelog modal, drag-resizer with per-page height persistence) into any page that doesn't
  hard-code it; the four templates now load xterm + termdock + app.js. And **each page
  remembers its own active terminal tab** (`localStorage` keyed by pathname, in app.js): pick
  a tab on /diff and it's still the selected one when you come back, independent of the tab
  chosen on /tree or /. Also guarded app.js's connection dot/text so pages whose topbar lacks
  them (the indicator lives in the injected dock header there) can't break the terminal
  script. No Flask restart needed — reload the browser tabs.

- **[2026-07-03] (Claude)** — DiffEvo: added **unrolled training** (diffuse mode, default on).
  Each champion now trains on the ACTUAL output of the champions above it — the real image the
  reverse walk hands it at inference — instead of a freshly-noised image. Requires training
  noisiest-first (K→1) so upstream champions exist; each champion's input is built by running the
  partial walk down to its level (`_walk_state` + batched `_apply_champion_batch`). This removes
  the train/inference mismatch that made the walk's low-noise tail drift back up. Measured (8
  levels): the walk is now **monotonic to the end** and final L1 dropped 0.23→0.12 — matching
  single-shot denoise quality — with held-out average improvement ~25%→**~47–49%** and consistent
  across seeds/level counts. Also fixed the live preview to pick a **median-structure** test image
  instead of `test[0]`: the old fixed pick could be a near-flat outlier (98.7% of images improve,
  but a flat image can't beat its noise floor so its trace looked like divergence) — the held-out
  average is reported separately on the tile. UI: "Unrolled training" checkbox (diffuse only).

- **[2026-07-03] (Claude)** — DiffEvo: added **"diffuse" process mode** (now the default) —
  the actual diffusion reverse process: start from noise and denoise step by step toward the
  image. The mechanism is a change to *what each genome learns*: in diffuse mode champion_k is
  trained on one **incremental** step (input `x0 + σ_k·z` → target `x0 + σ_{k-1}·z`, same noise
  realization), i.e. remove only a slice of noise, not jump to clean. No single genome can finish,
  so the reverse walk (`reverse_walk`, composes champions K→1 from the noised start, no re-noising)
  genuinely moves closer each step and stays on-distribution. Kept the prior behavior as
  **"denoise" mode** (jump-to-clean champions + single-shot sampler). Measured (8 levels): diffuse
  walk L1-to-target 0.304 (noise) → 0.281 → 0.263 → 0.238 → 0.213 → 0.199 then a mild tail rise to
  0.216 (low-noise steps compound small errors); denoise single-shot gets a sharper final 0.117 but
  with no step-by-step process. UI: Process-mode selector (diffuse default; sampler field hidden
  unless denoise), reconstruction card relabeled "start · noise → final output," per-step L1 on the
  filmstrip (green=improved, red=diluted). Next lever for a sharper walk: cosine σ-schedule / more
  low-noise levels, or unrolled training (train each step on the actual walk distribution).

- **[2026-07-03] (Claude)** — DiffEvo: (1) **runs now persist to the /runs dashboard.**
  Each training run writes `runs/diffevo/<id>/{config.json,history.jsonl,summary.json}`
  in the same layout runstore uses, so a "diffevo" tab appears with a live L1-vs-generation
  sparkline (fitness.best/mean per gen, descending = improving) — written directly, not via
  runstore, to keep DiffEvo free of the engine checkpoint format. (2) **Default sampler is now
  `single`-shot.** Confirmed the "gets worse after step 1" behavior is fundamental, not a bug:
  denoising a known noise level, only the matched first step sees the *true* observation; every
  later champion sees only a processed estimate, has no new information, and can only smooth/
  contract → error rises each step. Single-shot (apply the champion matched to the input level,
  once) is therefore optimal here and can't degrade. DDIM/ancestral kept as selectable multi-step
  modes for the harder regime where stacking would pay (generation from pure noise, or steps
  conditioned on the original observation). UI copy + default updated.

- **[2026-07-03] (Claude)** — DiffEvo: fixed the reverse chain collapsing a good
  first-step reconstruction into a grey blob. Root cause: the sampler predicted x0
  then **re-noised it with fresh random Gaussian** each step; a chain of local-averaging
  denoisers over-smooths that new noise and contracts contrast → regression to the mean.
  Added a `sampler` option: **`ddim`** (new default) is deterministic — it carries the
  residual already in the image, rescaled to the next level (`x <- x0 + (s_next/s_k)(x-x0)`),
  so the chain converges onto x0 instead of walking away; **`single`** applies only the
  strongest (top-level) denoiser; **`ancestral`** is the old stochastic behavior (kept for
  contrast). Measured on a 6-level run: per-frame L1 for ancestral diluted 0.185→0.277
  (+5% held-out), ddim held ~0.16–0.19 (+32%), single-shot 0.160 (+38%). Notable finding:
  **at this data/noise scale stacking doesn't help — one tiny ~90-param genome already
  captures the signal**; extra local passes only bleed contrast. UI: sampler dropdown +
  per-frame L1 labels on the reverse-chain filmstrip (green if a step improved, red if it
  diluted) so the collapse-or-not is visible.

- **[2026-07-03] (Claude)** — DiffEvo: switched the selection objective from L2/MSE
  to **L1 (mean absolute pixel error)** per request. Squared error let a few large-miss
  pixels dominate selection (which a ~90-param genome can't chase); L1 rewards getting
  the bulk of pixels close — the "average job" objective. Renamed all reported metrics
  (`best_l1`/`mean_l1`, `in_l1`/`out_l1`, `test_in_l1`/`test_out_l1`, `l1_by_level`) and
  the UI labels to match. Smoke test now shows a cleaner monotonic per-level error floor
  rising with σ (0.077→0.139) and ~13% held-out L1 improvement.

- **[2026-07-03] (Claude)** — New project **DiffEvo** (`/diff` page, WS `/diffuse`),
  a fourth program alongside GENREG/Runs, Tree LM, and I2. Denoising diffusion by
  neuroevolution: decomposes image reconstruction into a stack of individually-easy
  reverse-diffusion steps, evolving one shared population of tiny convolutional
  per-pixel denoisers (patch → tanh(H) → clean centre pixel, ~90 params) per noise
  level. Fitness is denoising MSE over a *fresh minibatch of samples each generation*
  (stochastic averaging → generalizes instead of memorizing one image), with plateau
  early-stop per level. Per-level champions stack into a reverse chain (predict x0,
  renoise to next level). Files: `genreg_train/diffuse_service.py` (self-contained
  numpy: procedural dataset, ES, JobHub survives WS disconnect like tree_service),
  `templates/diff.html`, `static/diff.js` (canvas reconstructions + hand-rolled SVG
  fitness/level charts). Wired into `app.py`; nav links added to Build + Tree pages.
  Smoke test (4 levels, pop 24): held-out test-set denoising MSE improved ~18%,
  per-level MSE floor rises with σ as expected, plateau detection fires.

## 2026-07-03 (Claude Code / Fable 5)
- **Media stored as genome-CODED latents, never raw files (copyright /
  no base64 tax; v1.3.0, pushed live + migrated).** User: video must be
  stored as latents, not a playable file. Measured first: the text
  arithmetic coder is the WRONG tool for media — on 1MB of video-like
  data it took 27s to encode / 25s to decode and EXPANDED it 41% (dead
  end). And the old base64-in-JSON storage inflated every file 33% (the
  live 3.79MB video was 5.06MB stored/served).
  - Fix: media is now `media_code()` = **AES-CTR keyed from the genome +
    content id**. Stored/served bytes are ciphertext — a coded bitstream,
    unplayable as-is, decodable only by applying the genome-derived key
    (I2's copyright posture, same as pages). Storage split into metadata
    `<id>.json` + coded `<id>.bin`; `/api/i2/{image,video}/<id>/blob`
    serves the latent as octet-stream; the browser fetches + AES-decodes
    it (WebCrypto) into a blob URL right before playback; the child
    decodes to VERIFY (hash+sig) then caches/serves the coded latent.
    `migrate_media()` converts legacy base64 records to coded .bin at
    startup. Removes the 33% base64 tax (0% overhead now) as a bonus.
  - HONEST scope: this is coding/gating (decoder-keyed), NOT compression
    and NOT secret from genome-holders. TRUE compressive video latents =
    a neural media codec (the "media genome") = a real future build.
  - Verified: Python↔JS AES-CTR parity (JS decodes Python's latent
    exactly, size preserved); upload stores coded .bin at original size,
    no raw data in metadata; /blob is coded and decodes to the right
    hash; child fetch→decode→verify→cache→serve coded; legacy migration.
    Deployed + live 3.79MB video migrated (5.06MB→3.79MB, coded). Restart
    the child for the new media path; reload the browser.

## 2026-07-03 (Claude Code / Fable 5)
- **Wipes trickle down: primary purge → children clear matching cache.**
  A child serves cached media offline (never re-checks the primary), so
  after a primary wipe it would keep serving deleted "junk". Fixed: the
  primary records per-category wipe timestamps (`i2_store/purges.json`,
  set in `maint_wipe`) and publishes them in `/api/i2/info` (children
  already poll it ~10s). The child compares to its applied set
  (`<cache>/purges_applied.json`) and, for any category the primary wiped
  more recently, deletes that local cache dir (pages/images/videos only —
  the rest it proxies live, never caches). Selective: a videos-only wipe
  clears the child's video cache and leaves pages. Verified end-to-end:
  child cached a page + video, primary `wipe --only videos`, child polled
  and dropped videos (0) while keeping pages (1). Same `purges` field is
  the mechanism future SECONDARIES will honor too. Pushed live
  (info.purges confirmed).

## 2026-07-03 (Claude Code / Fable 5)
- **Maintenance TAB in the primary GUI console + child Cache tab shows
  media.** Follow-ups to the two requests below.
  - **Primary control panel** (`i2_node.py` GUI): new "Maintenance" tab —
    category checkboxes with live item counts/sizes, "Back up ticked",
    "Wipe ticked" (confirm dialog; auto-backup first), a one-click "Reset
    — keep only pages", and a Backups list with "Restore selected". Runs
    the maint_* functions IN-PROCESS (local operator, no signing needed;
    the signed CLI remains for remote use). Default selection is
    everything-except-pages.
  - **Child Cache tab** (`i2_child.py` GUI): it only listed cached PAGES,
    so videos/images you watched on Latent Stream never appeared even
    though they WERE cached. Now the tree shows type (page/video/image),
    name, size, detail, status for pages + cached videos + cached images;
    the overview line reports "N pages · N videos · N images". (The child
    must be on the media-caching build — restart it if watched videos
    still don't show.)

## 2026-07-03 (Claude Code / Fable 5)
- **Customizable store backups / wipes / restores from the primary
  (v1.3.0, pushed live).** Selective reset by category — e.g. drop your
  user, all posts/videos/likes/tokens but KEEP the sites.
  - Categories: pages, images, videos, identities, domains, likes,
    ledger, uptime. Genome + node keys are NOT categories (infrastructure,
    never touchable). Functions in i2_node: `maint_backup` (snapshot to
    `i2_store/_backups/<ts-label>/`), `maint_wipe` (ALWAYS auto-backs-up
    first → reversible), `maint_restore`, `maint_list_backups`,
    `maint_sizes`.
  - Admin endpoint `POST /api/i2/admin/maintenance` — gated by the same
    pinned admin key as push (Ed25519 sig over the canonical doc + fresh
    ts). New CLI `i2_admin.py`: `status`, `backup [--only ..] [--label]`,
    `wipe --keep pages --yes` (reset but keep sites) / `wipe --only
    videos,likes --yes`, `restore <backup> [--only ..]`. Default target
    10.0.0.15:8800; wipe needs `--yes`.
  - Verified end-to-end: seeded identity+video+like, `wipe --keep pages`
    kept all 7 pages while dropping the rest, `restore` brought them back,
    `wipe --only videos` dropped just videos and kept the identity;
    unsigned and wrong-key wipes both 403 with pages untouched. Pushed
    live; `status` reports the real store.

## 2026-07-03 (Claude Code / Fable 5)
- **Templates/theming + Woven (social feed) + likes (v1.3.0, pushed
  live).** Cosmetic overhaul so the browser is skinnable, not one fixed
  "AI-made" look.
  - **Theme engine**: design tokens (colors, font, radius, layout,
    feature flags) live in `THEME`, mirrored to CSS variables (`--i2-*`)
    for DOM and read directly by the canvas. 6 built-in TEMPLATES
    (Midnight, Daylight, and Woven skins Gram/Reels/Tumble/Vine). Applied
    GLOBALLY or PER-SITE, persisted in localStorage; `navigate()` picks
    the site's template. Canvas chrome + pages refactored to read THEME
    so a template re-skins the whole browser.
  - **Appearance page** (`i2://appearance` / `themes`): swatch grid of
    templates, apply Everywhere or to one site, live accent color picker.
  - **Woven** (`i2://woven`): a social feed over the network's existing
    images + videos as posts, with likes. Templates swap the LAYOUT —
    Gram (grid), Reels (vertical full-bleed, videos autoplay/loop/muted),
    Tumble/Feed (cards) — and toggle features: the Vine skin HIDES the
    like count while the network still keeps it (the requested trick).
  - **Likes backend** (`i2_service`, `i2_store/likes/`): a signed thumbs-
    up on any content id, one per identity, verified + counted; routes on
    primary + child (`/api/i2/like`, `/api/i2/likes/<id>`).
  - **Also fixed a real regression the harness caught**: the address bar
    blurred the field (which re-syncs it to the current URL) BEFORE
    reading the typed value, so Enter navigated to the current page.
    Now reads the value first — address-bar navigation works again.
  - Verified: likes backend 7/7; theme UI shim (global re-skin, per-site
    override wins on Woven, Woven renders posts, Vine hides counts,
    Appearance lists templates, accent persists); prior UI harnesses
    (input/search/LS) green. Home quick links now include Woven +
    Appearance. Pushed live (restarted, likes route confirmed). Restart
    the child for the like proxy routes.

## 2026-07-03 (Claude Code / Fable 5)
- **Welcome page shows the genome's current wire size.** Added
  `GENOME_WIRE_BYTES`/`GENOME_SIZE_STR` (computed from the actual genome)
  and a `{size}` placeholder on the home page: "the genome is 633 KB on
  the wire — downloaded once, then every page decodes locally." Computed
  dynamically so it stays accurate if the genome changes. Republished
  live to the primary (verified: decodes, signature valid, shows 633 KB);
  source synced (--no-restart) + dist.

## 2026-07-03 (Claude Code / Fable 5)
- **Search TLC: typeable home box + directory-grouped results page
  (browser-only, pushed live).** Two fixes to the reported dead search:
  - The home search box wasn't typeable — it was a canvas drawing that
    just redirected focus to the address bar (felt broken). Replaced with
    a REAL `<input>` (`homeSearchInput`) laid over the hero, visible only
    on home, positioned each frame via `homeSearchRect`. Click and type
    directly; Enter → results page.
  - Results now render as a landing page GROUPED BY DIRECTORY: sections
    for ▶ Latent Stream, 🖼 Gallery, People, Domains, each `domain/`, and
    Pages (commons); section headers are clickable (jump to that
    directory) and show counts; groups ordered by best hit. Header reads
    "N results across M directories". Replaces the old flat list.
  - Verified in the DOM shim: real home input exists, is visible on home
    / hidden elsewhere, typing+Enter searches; results group into the
    correct directory sections with a directory count.

## 2026-07-03 (Claude Code / Fable 5)
- **Welcome page (`i2://home`) rewritten + documents offline caching.**
  The home page was pre-identity/stream/gallery. Now it: keeps the
  compression pitch, adds an "On the network" section linking Latent
  Stream / Gallery / DMV, and features a **"Works offline"** section
  documenting that a node keeps verified copies of what you open (pages,
  images, whole videos), replays instantly, survives losing the primary,
  and re-verifies every cached copy — with the honest note that serving
  copies is a separate opt-in and bulk redundancy is the (undeployed)
  secondary tier. Updated the canonical `_SEED_PAGES["home"]` in
  i2_service.py AND republished the page LIVE to the primary via
  `/api/i2/publish` (seed only auto-applies on a fresh store). Verified:
  decodes cleanly, signature verifies, contains the new sections. Source
  synced to the primary (--no-restart) and dist.

## 2026-07-03 (Claude Code / Fable 5)
- **Child caches images + videos (offline resilience) — child v1.3.0.**
  Children cached PAGES but proxied images/videos straight through, so a
  dropped connection meant no media and every read hit the primary. Fixed:
  `ingest_media(kind, id, verify)` is cache-first — a cached hit is instant
  and works offline; a miss fetches from the primary, verifies (hash +
  uploader signature), writes it under `<cache>/images|videos/<id>.json`,
  then serves it. The list routes (`/api/i2/images`, `/api/i2/videos`)
  fall back to the locally cached list when the primary is unreachable.
  Stats gained `cached_images`/`cached_videos`. This is the CONSUMER role
  (a child keeps copies of what IT reads); actually SERVING those copies
  to peers is a separate OPT-IN (consumer → contributor), and bulk
  high-usage replication is the SECONDARY tier's job — neither built yet.
  - Verified end-to-end: fetch via child caches it, second fetch is a
    cache hit, and with the PRIMARY KILLED the child still serves the full
    cached video AND lists cached videos offline.
  - Child-only change (caching is client-side) — no primary push. Restart
    the child on this machine to pick it up.

## 2026-07-03 (Claude Code / Fable 5)
- **Latent Stream — the network's YouTube (v1.3.0, pushed live).** Video
  site at `i2://latent-stream` (aliases: `ls`, `latent stream`, and the
  earlier `latent space` all resolve).
  - **Backend** (`i2_service.py`, `i2_store/videos/`): videos are signed
    blobs credited to the uploader (sign over sha256(bytes)+title+mime),
    same trust model as images; NOT genome-compressed. mp4/webm/ogg,
    32MB cap; title+description indexed by search. Routes on primary +
    child (child verifies hash+sig before serving). Flask
    MAX_CONTENT_LENGTH raised to 64MB for the base64 payloads.
  - **UI**: a YouTube-style app view — sticky header (▶ Latent Stream
    logo, its own search bar, ↑ Upload, ← Exit), a responsive video
    grid, a watch page (player + title/uploader/description + "Up next"),
    and an upload form (hash+sign in-browser, chunked base64). Video
    search results (address bar or LS search) open the watch page.
    Guarded against an async render race (grid load vs. watch) with a
    generation token.
  - **Search fix** (the "search is inoperable" report): the address bar
    treated a bare word like `sunset` as a page address → 404. Now
    `navigateSmart` sends anything that isn't a path / @handle / known
    page name to SEARCH; paths and known names still navigate directly.
    Multi-word always searches.
  - Verified: video upload accepted+credited, bad-sig rejected, listed
    (no data blob), fetched with data, and search finds videos by title,
    description, and uploader — direct AND through the child proxy.
    UI shim: LS opens via all aliases, grid lists videos, watch plays +
    shows "verified ✓", bare-word address-bar input searches. Pushed
    v1.3.0 live (restarted, videos route confirmed).
  - **Restart the child** on this machine for the video proxy routes.

## 2026-07-03 (Claude Code / Fable 5)
- **Fix dead address/search bar + gallery lightbox (browser-only, pushed
  live).** User: typing an address or a search did nothing. Cause: the
  URL/search bars were floating `<input>`s created on click and laid over
  the canvas — fragile focus/blur behavior in a real browser. Replaced
  with a PERSISTENT real DOM `<input>` (`addressBar`) always mounted over
  the nav area: type an address (e.g. `directory/sites`) to go, or a
  multi-word query to search (auto-detected on whitespace). Kept in sync
  with the current URL; red border on error. Removed the floating
  `openUrlInput`/`openSearchInput`; the home hero search bar now focuses
  the address bar. Verified via the DOM/canvas shim: address navigates,
  multi-word searches (hits `/api/i2/search`), bar reflects current page.
- **Gallery lightbox**: clicking any thumbnail opens the image large in a
  full-screen overlay (caption + click-to-close). Thumbnails show a
  zoom-in cursor.

## 2026-07-03 (Claude Code / Fable 5)
- **Browser UX: chrome pages instead of modals + a real search page
  (browser-only, pushed live).** Per user: stop piling icons on the
  toolbar and make each feature a navigable PAGE.
  - **Removed the ✎/👤/🔍/🖼 toolbar icons.** Toolbar is now just
    ← → ⟳ + the address bar.
  - **DMV, Gallery, Composer, and Search are now navigable pages**, not
    modal overlays: reachable via `i2://dmv`, `i2://gallery`,
    `i2://compose`, or search. New router in `navigate()` treats these as
    built-in "chrome pages" — local interactive DOM views mounted below
    the chrome (`APP_ROOTS`), fully in the history so ← → work; each has a
    "← Back" too. The old `toggle*` overlays (dark backdrop, click-out)
    are gone.
  - **Search results are a proper page** (Google-style): a results view
    with its own search box, `N result(s) for "q"`, and ranked cards
    (address · type, title, snippet) that navigate on click. Reached from
    the home hero search bar, from `i2://search`, or by typing a
    multi-word query in the address bar (address vs. search is auto-
    detected on whitespace). Replaces the old synthetic-canvas results.
  - **Home hero** gained quick links (Directory · Gallery · New page ·
    Your ID) under the search bar, replacing the removed icons for
    discoverability.
  - Verified with a Node DOM/canvas shim loading the real i2.js: clean
    load, each app view shows as exactly ONE full page (not stacked
    modals), search renders the count + ranked results, URL bar reflects
    the active page. Pushed static/i2.js live (no restart). Reload the
    browser to get it.

## 2026-07-03 (Claude Code / Fable 5)
- **Fix "uploader is not a registered identity" on image upload
  (browser-only, pushed live).** Root cause: the live primary had ZERO
  registered identities — the browser's localStorage identity was
  registered against another primary / restored from a backup / lost to
  a store reset, so its DID was unknown to the current primary and every
  signed op ("not a registered identity") failed. Fix: `ensureRegistered()`
  in i2.js re-asserts the self-signed identity doc to the current primary
  (idempotent for the same DID; only errors on a real handle-vs-different-
  DID conflict). Called before image upload, on restore-from-backup, and
  silently when the DMV opens (so claim/renew/transfer/earn self-heal
  too). Verified: reproduced the exact 400, then re-register → upload
  succeeds. Pushed static/i2.js live (no restart; static serves fresh).
  Users just reload the browser.

## 2026-07-03 (Claude Code / Fable 5)
- **Search + Gallery (v1.2.0, pushed live to 10.0.0.15).** "Google for
  the network," plus signed image uploads to search over.
  - **Search** (`i2_service.search`, `GET /api/i2/search?q=`): the
    primary DECODES every page with the genome and keyword-ranks over
    page title+body, image captions, handles, and domains; returns
    ranked results with snippets. Linear scan (fine at this scale);
    honest keyword search, NOT semantic embeddings yet (that is the
    primary's future intelligence layer). Title hits weighted 8×, image
    caption hits +4.
  - **Gallery / images** (`i2_store/images/`): images are SIGNED BLOBS
    credited to the uploader's identity — the uploader signs
    (sha256(bytes), title, mime) with their DID; the primary verifies
    before storing and rejects tampered uploads. png/jpeg/gif/webp, 5MB
    cap. NOT genome-compressed (text codec only; a media-genome family is
    future work, stated openly). Routes: `/api/i2/image/upload`,
    `/api/i2/images`, `/api/i2/image/<id>`. The CHILD verifies every
    fetched image locally (hash + signature) before serving it, same
    trust model as pages.
  - **Browser**: new 🔍 (search) and 🖼 (gallery) chrome buttons; the
    home hero search bar is now live (was decorative); results render as
    a synthetic in-canvas page with clickable hits (page/domain →
    navigate, image → open gallery). Gallery overlay uploads (hash+sign
    in-browser, chunked base64 for MB files) and shows a verified
    thumbnail grid credited "by @handle". Typing `gallery` in the URL bar
    opens it.
  - **Verified 9/9** (exact browser/child path): signed upload accepted +
    credited, tampered upload rejected, image listed + full record
    fetchable, search finds the image by caption, finds a text page by
    decoded content, finds the identity, empty on no-match, ranked; and
    through the CHILD proxy: search, image list, and hash+sig-verified
    image fetch all work. Pushed v1.2.0 live (restarted, search/images
    routes confirmed).
  - **Restart the child** on this machine to get the search/gallery proxy
    routes (browser buttons appear immediately since static served fresh,
    but their API calls need the child's new routes).

## 2026-07-03 (Claude Code / Fable 5)
- **Node persistence + uptime correctness fixes (v1.1.1, pushed live to
  10.0.0.15).** Fixes "re-create the node every restart" and "uptime not
  accruing," and makes the child auto-reconnect without re-scanning.
  - **Root-cause bug**: a plain GUI launch passed no dir hint, and
    `run_gui` only loaded config when given a hint — so it re-ran the
    SETUP WIZARD every launch, losing the primary URL *and* the linked
    identity (which silently stopped uptime earning). Fixed in BOTH
    nodes: a stable per-user home pointer (`~/.i2/{child,primary}_home`,
    written by `save_config`) plus a load order of explicit hint →
    remembered home → default dir. A plain launch now reopens the last
    node straight to the dashboard; the wizard only appears on a true
    first run. Headless `--server`/`--status` use the same recall.
    Verified: after save, a fresh no-hint load reopened the node with
    primary_url AND identity link both intact.
  - **Auto-reconnect**: since config now persists, the child reopens with
    its primary_url and its monitor loop reconnects on its own (and keeps
    retrying if the primary is briefly down) — no rescan needed.
  - **Uptime calculated correctly across two machines**: heartbeat
    freshness window widened 120s → 300s. It only ever guarded against
    CLOCK SKEW between child and primary (the real anti-fast-forward
    guard is the server-measured 30s/beat credit cap), so a narrow window
    was silently rejecting every heartbeat as "stale" whenever the two
    machines' clocks differed by >2 min → zero uptime. Child now logs the
    primary's rejection reason so skew is diagnosable.
  - Child GUI overview shows the linked identity + earned tokens/hours
    (or a prompt to link). Both NODE_VERSIONs → 1.1.1.
  - **Deploy**: primary pushed live (restarted 1.1.1, 7 pages intact).
    The CHILD on this machine must be restarted once to pick up the new
    code; after that it stops re-running the wizard. If the child used a
    CUSTOM cache dir before, it shows the wizard one last time (no
    pointer yet), then remembers it forever.

## 2026-07-03 (Claude Code / Fable 5)
- **Uptime token minting — the real faucet (v1.1.0, pushed live to
  10.0.0.15).** Tokens are now EARNED by keeping a node online, replacing
  the sybil-farmable instant grant as the main income.
  - **Signed heartbeats**: a child sends `POST /api/i2/node/heartbeat`
    {did, ts, sig} each poll (~10s); the primary verifies the sig against
    the DID's registered key, requires a fresh ts (±120s, anti-replay),
    rejects out-of-order beats, and credits SERVER-measured elapsed time
    since the last beat, capped at 30s/beat so a client can't claim time
    it wasn't online. Mints 1 whole token per accrued hour
    (`i2_service.py`, `i2_store/uptime.json`; `_lock` → RLock so minting
    can call grant_tokens while held).
  - **Starter grant cut 30 → 20** (exactly one domain): first homestead
    free, every domain/renewal after is earned by uptime — implements the
    "one domain per fresh DID, more earned" anti-squat design directly.
  - **Child**: `/api/i2/child/link-identity` lets the browser hand the
    local node the user's identity (doc + pkcs8 priv, same-origin, own
    machine) so it earns to their wallet; unlink endpoint too; heartbeat
    sender in the poll loop; identity + mint status in child stats.
  - **Browser DMV land office**: "Earn on this node" button (links the
    local child; on a primary it explains primaries mint, don't earn),
    wallet line now shows hours-online-earned; ledger endpoint returns
    uptime.
  - **Verified** (exact browser/child code path): starter grant = 20,
    first beat credits 0, out-of-order/stale/bad-sig/unregistered beats
    all rejected, deterministic mint (seeded pending → one beat mints
    exactly 1 token, credit capped ≤30s, balance +1), and a REAL linked
    child accrued 10s of verified uptime over live heartbeats. Then
    pushed v1.1.0 to the live primary via push_to_primary.py — it
    restarted on 1.1.0 with the heartbeat route live and all 7 pages
    intact. Honest caveat: still farmable by running many real nodes for
    real hours (time is the cost now, not a free grant); reputation
    weighting / proof-of-storage are the next hardening.

## 2026-07-03 (Claude Code / Fable 5)
- **Signed remote push — deploy to the primary without hand-copying.**
  New admin-authenticated `POST /api/i2/admin/push` on the primary
  (`i2_node.py`): accepts an Ed25519-signed file bundle from ONE pinned
  admin key (`i2_admin.pub`, ships with the primary), verifies the
  manifest signature + per-file sha256 + a strict path whitelist
  (i2 source/static/templates only), backs up replaced files to
  `_backup/push-<ts>/`, writes them, and (default) restarts. No key file
  → endpoint disabled. `push_to_primary.py` on the dev machine holds the
  PRIVATE key (`i2_admin_key.json`, gitignored) and does the packaging/
  signing/POST — `python push_to_primary.py [--primary URL] [files…]
  [--no-restart]`, default target 10.0.0.15:8800.
  - **Fixed a real restart bug**: `os.execv` on Windows leaves the old
    listening socket as a stale handle (port shows LISTEN, nothing
    serves). Both nodes' `relaunch()` now spawn a DETACHED child (no
    inherited fds) + `os._exit(0)`; the bind-retry loop covers the
    overlap. Verified: pushed to a live primary, it restarted on a NEW
    pid still serving, admin key intact.
  - **Security verified**: unsigned push → 403, wrong-key signature →
    403, `../` path traversal (even correctly signed) → 403. NODE_VERSION
    bumped 1.0.0 → 1.0.1; `/api/i2/admin/version` reports it.
  - Dogfooded: pushed the full source set to the local primary, it
    restarted running the new code, then the transfer suite passed
    against it.
- **Domain transfers** (completes the economy): signed
  `domain-transfer` op (`i2domain\0` family) moves a deed to another
  registered identity, owner-initiated only — the registrar never moves
  it. Route on primary + child proxy + per-domain "Transfer" button in
  the DMV land office. Old owner's pages stop resolving under the domain
  (new owner controls the namespace) but live on at their @handle alias,
  same as a reclaim. Verified 6/6: non-holder rejected, unregistered
  recipient rejected, owner transfer accepted, replay rejected, old
  owner can't renew after, deed shows recipient.
- **Push note for deployment**: the CURRENT fileshare primary predates
  the /admin/push endpoint, so this ONE update is still manual (copy
  dist/i2_primary — now incl. i2_admin.pub — to the share + restart).
  Every push after that: `python push_to_primary.py`.

## 2026-07-03 (Claude Code / Fable 5)
- **I2 DOMAINS + TOKEN ECONOMY + AUTO-DIRECTORY** (the navigation design
  session, built): two-layer namespace — permanent free @handle aliases
  underneath, token-fed domains on top. All 20 e2e checks pass.
  - **Token ledger** (`i2_service.py`, `i2_store/ledger.json`): mint =
    30-token starter grant per new identity (the bootstrap faucet;
    uptime-minting later — sybil-farmable until then, STATED). Signed
    transfers between DIDs (`i2token\0` canonical docs, replay-protected
    via applied-sig hashes). Sink = domains: claim 20, renew 5/30 days.
  - **Domain registry** (`i2_store/domains/`): deeds claimed/renewed by
    signed ops (`i2domain\0`); states active → grace (14d) → lapsed →
    back to the pool. Reserved: home/dmv/publishing/manifesto/network/
    directory/index/commons/search + Windows device names.
  - **Pages under domains are AUTHOR-SIGNED**: multi-segment names
    (`arcade/games`, ≤4 segments, nested storage) require the deed
    holder's own Ed25519 signature over the standard page message; the
    node cannot publish into your domain. Single-segment names stay the
    open commons (primary-signed, unchanged). `resolve_page()`:
    `@handle/name` alias serves any page authored by that handle's DID
    (content outlives addresses); a reclaimed domain 404s the previous
    owner's pages under the domain address (alias keeps working); grace/
    lapsed states surface in the page footer with a renew warning.
  - **Auto-directory**: primary regenerates `directory`,
    `directory/sites`, `/people`, `/new` (ordinary signed pages) on
    start and after every publish/claim/registration. Curated `index`
    reserved for hand-curation (content to come).
  - **Browser**: typeable URL bar (click bar → input → Enter),
    path+alias link targets, composer accepts `domain/page` and
    author-signs automatically when you hold the deed, DMV card gains
    wallet + land office (balance, claim, renew, send tokens to a
    handle). New routes on primary; child proxies ledger/domain calls
    and caches nested page names safely.
  - **Verified** (exact browser code path, Node webcrypto): starter
    grant, claim + burn, reserved/held-domain/insufficient-funds
    rejections, author-signed publish, unsigned + wrong-author
    rejections, domain fetch + @alias fetch + wrong-alias 404, transfer
    + replay rejection, renew, deed aged → grace → lapsed (page still
    served, flagged) → reclaimed by new DID (old page 404s under domain,
    lives at alias), directory generation, child nested fetch/verify/
    cache + proxies. Test artifacts cleaned; dist/ refreshed. Restart
    both nodes to serve all of it. NOT built yet (designed): uptime
    minting, domain transfer, subdomain delegation, curated index
    content, visit-based feeding.

## 2026-07-03 (Claude Code / Fable 5)
- **Restart button on both node GUIs** (`i2_node.py`, `i2_child.py`):
  top-bar "Restart" with confirm → existing `relaunch()` (os.execv), so a
  node picks up new files on disk without touching a terminal. Server
  startup in both scripts now retries binding (20 × 0.5s) because after
  execv the new process can race the old one for the port — without this
  a restart could come back dead. Verified headless: instance B started
  while A held :8800, A killed, B was serving within ~3s. GUI button
  itself needs a display (not exercised headless). dist/ refreshed.

## 2026-07-03 (Claude Code / Fable 5)
- **Browser: readable error when the node runs older software.** User hit
  "unexpected token '<'" submitting the DMV form — the running child
  (8810) and fileshare primary both predate the identity routes, so the
  POST got an HTML 404 and JSON.parse choked. New `jsonOrExplain()` in
  i2.js wraps all POST/fetch response parsing (DMV register, directory,
  composer publish): a non-JSON reply now reads "node returned 404
  without JSON — its software is older than this page; restart/update
  the node". dist/ refreshed. The actual fix remains restarting both
  nodes with current files.

## 2026-07-03 (Claude Code / Fable 5)
- **DMV handles: broader charset** (user request; length was already 64).
  Handles get their own rule, decoupled from page names:
  `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` — capitals, dots, underscores now
  allowed. Uniqueness is CASE-INSENSITIVE (registry file is the
  lowercased handle; typed casing preserved on the card) so re-casing
  someone's handle can't impersonate them — takeover test passes.
  Windows-reserved basenames (con, nul, com1…) rejected. Updated the DMV
  form validation/placeholder in i2.js. Verified 7/7 against a local
  primary (mixed-case register, case-variant takeover rejected, reserved
  names rejected, space/leading-dot rejected, case-insensitive lookup
  preserves casing). dist/ refreshed; same restart notes as below apply.

## 2026-07-03 (Claude Code / Fable 5)
- **I2 DMV — self-sovereign digital identity ("the DMV of the
  internet").** New 👤 button in the browser chrome (green once
  registered) opens the DMV overlay: generate an Ed25519 keypair IN THE
  BROWSER (WebCrypto), DID = the public key (`did:i2:<b64url>`), and a
  self-signed identity document — handle (`[a-z0-9-]`, page-name rules),
  display name, emails (reserved names; mail comes later), and service
  aliases/IGNs (`service: name`). Private key stays in localStorage;
  Download-backup / restore-from-backup / forget-on-this-device / a
  network directory view, and update-details re-signs with the same key.
  - `i2_service.py`: identity registry (`i2_store/identities/`) —
    `register_identity` verifies the doc against its OWN DID (canonical
    signing bytes: `"i2id\0"` + sorted-key compact utf-8 JSON minus sig,
    byte-identical to `stableStringify()` in i2.js — parity tested),
    enforces first-come-first-served handles (update requires same DID),
    caps field sizes, rejects unknown fields. The registrar CANNOT
    recover/reassign/revoke — stated in the UI ("no reset desk").
  - Routes: primary `POST /api/i2/identity/register`,
    `GET /api/i2/identity/<handle>`, `GET /api/i2/identities`; child
    proxies all three to its primary (shared `_forward_json` helper).
  - Verified end-to-end with the exact browser code path (Node webcrypto
    + identical stableStringify) against a local primary AND through a
    child proxy — 7/7 both ways: register, takeover-by-different-key
    rejected, same-key update accepted, tampered doc rejected, bad
    handle rejected, lookup, directory. Canonical-JSON parity confirmed
    byte-identical incl. non-ASCII. Test identities/nodes cleaned up.
  - Published `i2://dmv` (explainer page) to the live fileshare primary;
    dist/ refreshed. NOTE: primary + child processes need a restart to
    serve the identity routes (also still pending from the composer:
    the user's child on 8810 and the fileshare primary's static/i2.js).
  - Groundwork stated honestly: pages are still signed by the primary at
    publish; wiring identity keys into page authorship ("published-by
    <handle>") is the next roadmap item, now unblocked.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 Composer — the browser can now make pages** ("a page that makes
  pages"). New ✎ button in the canvas browser chrome (`static/i2.js`)
  opens a DOM overlay editor: page name (`[a-z0-9-]{1,64}`), title, body
  in the existing page markup (`# ` heading, `* ` bullet,
  `[[page|Label]]` link), an "Edit current page" prefill (decoded text
  is now kept on `BROWSER.page.raw`; republish = overwrite), and
  Publish + sign → POST `/api/i2/publish` on the same-origin node, then
  the browser navigates to the new page and decode-verifies it like any
  other. New child endpoint (`i2_child.py`): `/api/i2/publish` forwards
  the draft to the primary (children don't sign yet — per-author keys
  remain roadmap; the primary encodes + signs, stated in the UI), passes
  primary rejections through, and busts the child's cached copy of a
  republished name. Verified live against the fileshare primary
  (10.0.0.15): a test child forwarded a publish (a real "How to publish
  on I2" page, now at i2://publishing), read it back
  fetched+verify_ok=1, and both the primary and the user's running
  child list it. dist/ copies refreshed. NOTE: the user's running child
  needs a restart to gain the forwarding endpoint (browser JS is live
  already); the primary on the fileshare needs its static/i2.js updated
  (self-update or recopy) to show the ✎ button.

## 2026-07-02 (Claude Code / Fable 5)
- **MILESTONE: first two-machine I2 cell verified live.** Primary running
  on the user's fileshare machine (10.0.0.15:8800), child on this machine
  (127.0.0.1:8810). Initial blocker was Windows Firewall on the primary
  host (TCP 8800 filtered; ping fine) — fixed by the user via an inbound
  rule/policy; bind stays 0.0.0.0 (correct; no per-IP bind needed).
  Verified over the network: child connected (37 ms), genome g3-71955590
  synced, uncached page fetched from primary → decoded → sha256 →
  Ed25519 verified → cached (verify_ok=1, verify_fail=0), and a repeat
  read served from local cache without touching the primary. No code
  changes; deployment/verification only.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 Child Node (`i2_child.py`) — consumer client + primary discovery.**
  (Script was written just before a session crash; this entry records it
  plus the new discovery feature.) A child connects to a PRIMARY,
  fetches/auto-updates the genome, and serves the I2 browser locally
  (default port 8810) as a verifying cache: every page fetched from the
  primary is verified locally (decode → sha256 → Ed25519 signature via
  `i2_service`) before caching; failed pages are REJECTED and never
  cached. Tracks self-measured connection reputation (hours online,
  uptime ratio, disconnects — honest placeholder until the network
  aggregates it). Tk GUI (setup wizard → Overview / Connection-Reputation
  / Cache / Update / Log tabs), `--server` headless, `--status` JSON.
  Same self-update mechanism as the primary. Genome auto-refetches when
  the primary announces a new version.
- **NEW: LAN primary discovery.** `discover_primaries()` sweeps localhost
  + every local /24 subnet on port 8800 (64 threads, 0.4s connect probe,
  then confirms via `/api/i2/info` role/version fields), sorted primaries
  first by latency. Wired into (a) a "Scan network for primaries" button
  in the setup wizard — results listed, click to fill the URL field;
  manual URL entry still works — and (b) a `--scan` CLI flag. Verified
  live: a headless primary on this machine was discovered on all
  interfaces (10.0.0.21, 172.29.144.1, 127.0.0.1) in seconds; scan with
  no primary running correctly reports none.
- `package_i2.py` child README documents scan; `dist/i2_child/` rebuilt
  (in place — a process held the folder lock so rmtree was skipped).

## 2026-07-02 (Claude Code / Fable 5)
- **I2 Primary Node app (`i2_node.py`) — deployable node + operator
  console.** Single script to drop on a file share and run; Tk GUI with
  first-run setup wizard (node name, data dir, port, update source),
  then a live dashboard: Overview, Genome (version/mixer/row counts),
  Storage/Pages (per-page size, ratio, origin, sig), Content Map (PCA-2D
  of per-page byte histograms on a Tk canvas — the "watch the latent
  space evolve" view; honest label since gen-2 content isn't
  latent-vector-based), Network (bind, peers=0/first cell, request
  counters, recent activity), Publish (encode+sign through the GUI), and
  Update. Runs an instrumented Flask node in a background thread (waitress
  if installed, else werkzeug), serving `/`, the genome, pages, publish,
  and `/api/i2/node/stats`. Serves a self-contained browser via new
  `templates/i2_node_browser.html` (canvas-only, no terminal deps).
  **Self-update**: reads `manifest.json` at a configurable update source,
  semver-compares to `NODE_VERSION`, backs up replaced files to
  `_backup/<ts>/`, copies new ones, offers restart (`os.execv`).
  `i2_service.py` now honors `I2_DATA_DIR`/`I2_CORPUS` env vars so storage
  points at the deployment share (defaults unchanged for in-repo use).
  Modes: GUI (default), `--server` (headless), `--status` (JSON dump).
  Verified headless end-to-end under `simple`: server binds, live HTTP
  genome/page/publish, content-map PCA, node-stats endpoint, and
  update check+apply all pass. **Roles (primary/secondary/child) and
  primary authority (LLM/observation, chain integrity) are DECLARED in
  config but NOT yet enforced — first primary runs unrestricted, stated
  openly.** GUI itself needs a display (not exercised headless).
- **Findings PDF updated to experiments 1–9** (9 pages): new Section 11
  (zstd-dictionary validation = the decisive result; mixer-evolution
  negative; gen-2 ship + signature/parity guarantees) and Section 12
  (node tiers + federation trust model, stated transparently: primaries
  hold real bootstrap authority, hierarchy is a reputation growth path).
  Executive summary now lists gen-2 as shipped; roadmap marked a–e done.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 GEN-2 SHIPPED + Ed25519 origin signatures** (user roadmap items
  1–4 executed in validation-first order). **Requires a Flask restart.**
  - **exp8 (mandatory validation, run BEFORE shipping claims): zstd-19
    with a 512KB dictionary trained on the same corpus = 2.86× on 1KB
    pages — a DEAD HEAT with gen-1 (2.85×) — and 2.291 bits/byte on 1MB.**
    Gen-1 could not have claimed state-of-the-art; gen-2 was required.
    (brotli python binding has no custom-dictionary support; noted.)
  - **exp9 (mixer evolution): per-confidence-bucket K evolution gave NO
    held-out gain** (2.223 evolved vs 2.216 seed) — the flat Witten-Bell
    chain (K3=16, K2=2, K1=0.5) is already near-optimal; reported
    honestly. But the exp9 pipeline itself (MLE + full backoff chain on
    u16-scaled tables) reached **2.216 continuous / ~2.24 through the
    real quantized coder path (3.57×) — beats zstd+dict in both regimes**
    (whole-buffer margin is real but narrow: 2.24 vs 2.291).
  - **Gen-2 codec** (`i2_service.py` + `static/i2_codec.js`): genome
    `g3-<hash8>`, format `mix3-ac` — order-1/2/3 u16-row-scaled count
    tables (648KB zlib wire) + flat-K backoff mixer, 3-byte rolling
    context, per-context mixed distribution quantized to integer
    frequencies (floor(p·16384+0.5), min 1, total < 2^17 → exact in JS
    doubles; flat K's chosen partly because bucketed K needs log2, a
    cross-platform float-parity risk). Same WNC coder. g0/g2 pages
    auto-migrate (legacy decoders kept only for migration).
  - **Ed25519 signatures (Day-1 requirement per user):** node keypair in
    `i2_store/keys/` (auto-generated); every page signed at publish over
    `"i2page\0"+name+"\0"+genome+"\0"+sha256(content)`; `origin` is now
    a real public key; server verifies (`verify_page`), and the browser
    verifies INDEPENDENTLY via WebCrypto Ed25519 after local decode —
    footer shows ✓ origin (verified) / ✗ (INVALID) / unverifiable.
    Tamper test passes (modified doc correctly rejected). `cryptography`
    pip-installed into `simple`.
  - Verified: Python round-trip incl. 100KB torture (2.33 bits/byte,
    3.44×); **node parity: JS decodes all pages sha-exact (100KB in
    88ms) and all signatures verify with the browser's exact message
    format**; tamper rejection confirmed.

## 2026-07-02 (Claude Code / Fable 5)
- **New report: `documentation/I2_COMPRESSION_GENOME_FINDINGS.pdf`** (user
  request: "highly detailed, but honest") — 7 pages covering experiments
  1–7: methodology + held-out protocol, contamination check, real-coder
  verification, the falsified continuous-latent codec (with garbage-decode
  sample), full classical baseline tables (whole-buffer AND small-page
  regimes), out-of-domain degradation, the tree-LM codec failure and its
  accuracy-vs-calibration diagnosis, the bits/byte evolution runs (incl.
  gradient ceiling + gate-collapse negative results), the order-3 gen-2
  proposal, what actually shipped in gen-1, and an explicit
  threats-to-validity section (untested zstd-dictionary rival, domain
  boundedness, tree corpus-overlap disclosure). Generated with reportlab
  (pip-installed into `simple` along with pypdf for verification). Visible
  on the /docs page immediately, no restart needed.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 gen-2 experiments: evolved against bits/byte (exp5/6/7, scratchpad;
  no product code changed).** Verdict: log-loss fitness WORKS as a
  mechanism, but the neural organism family loses to counting; the real
  gen-2 candidate is order-3 + backoff.
  - exp5 (softmax head on frozen best encoder 538711, ridge-seeded,
    GA pop 64 on log-loss): 5.27 → **3.85 bits/byte held-out** in 7s of
    GPU evolution; top-1 acc only 0.334 vs 0.360 — evolution happily
    trades accuracy for calibration, exactly as intended. But the
    gradient-trained CEILING of that readout class is 3.65: the
    accuracy-bred encoder features don't carry distributional info, so
    even a perfect head can't approach order-2 counts (2.78).
  - exp6 joint organism (encoder+head, 272k params, 500 gens): 4.27 —
    worse than head-only (mutation dilution across the big genome). An
    evolved CONTEXTUAL gate mixing model vs order-2 collapsed to
    g≈0.001 ("always counts"), held 2.777 ≈ order-2 alone: the neural
    model adds nothing anywhere, not even in pockets.
  - exp7 count-family ladder: **order-3 + confidence backoff
    (lam=n/(n+16), fitted on train) = 2.27 bits/byte = 3.52× held-out —
    beats brotli-11 (3.45×) even on whole 1MB buffers**, still count-based
    (same codec contract as gen-1, trivially parity-able), genome ≈0.8MB
    raw sparse (~231k distinct 4-grams / 45k order-3 contexts in corpus).
    Proposed gen-2: ship order-3+backoff; let evolution breed the MIXER
    (per-context gates over the count ladder, PAQ-style) where it can
    genuinely add value.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 gen-1 genome SHIPPED: order-2 model + arithmetic coding** (user:
  "ship the order-2 genome as gen-1"). **Requires a Flask restart.**
  - `i2_service.py` rewritten: genome = quantized trigram frequency table
    built from the full EEC corpus (row totals capped < 2^17 so all coder
    arithmetic is exact in JS doubles; every symbol freq ≥ 1; unseen
    contexts uniform), serialized sparse — **100KB on the wire** (zlib),
    version `g2-<hash8>`, cached in `i2_store/genome/order2_sparse.bin`.
    The dense table is always reconstructed FROM the sparse wire bytes so
    server and browser provably use the same table. Pages are now
    Witten-Neal-Cleary arithmetic-coded bitstreams (`payload` field +
    `ratio`); publish verifies round-trip before storing. Gen-0 pages
    auto-migrate on import (legacy orthonormal decode kept only for that).
  - New `static/i2_codec.js`: DOM-free JS mirror of the coder
    (genome parse + AC decode), loaded before i2.js; `static/i2.js`
    decode path swapped (zlib inflate via DecompressionStream); page
    footer now also shows wire bits/byte.
  - Verified: Python round-trip incl. 100KB corpus torture (2.84
    bits/byte, matches exp3 prediction); **node parity test: JS decoder
    reproduces Python bytes sha-exact on all pages** (100KB decodes in
    15ms). Genome build from the 49MB corpus takes ~2s once, then cached.
  - Gen-2 path (from exp4 diagnosis): evolve the tree-LM against
    bits/byte (log-loss) instead of top-1 accuracy, then swap the genome.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 exp3+exp4: triple-checked the compression claim; wired the tree-LM
  to a codec and measured it** (scratchpad i2_exp3_verify / i2_exp4_treelm;
  results JSONs alongside; brotli+zstandard pip-installed into `simple`).
  - Verification of "order-2 beats zlib": **contamination 0/60** held-out
    probes found in train; a REAL arithmetic coder (Witten-Neal-Cleary,
    implemented + round-trip verified byte-exact) produced actual
    compressed bytes at **2.780 bits/byte (2.88×)** vs its 2.80 theoretical
    — the number is real, zlib-9 = 2.901.
  - BUT the full suite on 1MB buffers beats the order-2 model: bz2 3.84×,
    brotli-11 3.45×, lzma 3.41×, zstd-19 3.38× vs model 2.88×. "Beats
    zlib" ≠ "beats modern compression" at large-buffer scale.
  - **Small-page regime (1KB, I2's actual workload): the model wins
    decisively** — model 2.85× vs brotli 2.19×, zlib 1.75×, zstd 1.72×,
    lzma 1.45× — because the model's "dictionary" ships once in the
    genome instead of being rebuilt per page. (Strongest untested rival:
    zstd with a trained dictionary.)
  - Out-of-domain honesty: Gutenberg-trained genome gets only ~1.7× on
    GENREG markdown (loses to zlib) — the genome must evolve toward the
    network's real content distribution.
  - **Tree-LM as codec (the requested number): 4.31 bits/byte (1.86×)** —
    best model (0.3603 acc, c6df68) with full path-probability
    distribution (router+leaf softmax, calibration scalars tr=8 tl=16 +
    1% uniform floor fitted on train only). Far worse than order-2 counts
    (2.78), and a fitted linear blend chose λ=0 (tree adds nothing on top
    of order-2). Diagnosis: evolved for top-1 accuracy, not log-loss —
    codec-grade evolution needs bits/byte (perplexity) as the fitness
    function. That is the gen-2 experiment.

## 2026-07-02 (Claude Code / Fable 5)
- **Autonomous experiment session ended on user request** — E18 (pop-1000 /
  gens-800 encoder) was killed mid-encoder-evolution; its encoder-store
  entry (created after 19:25, if any) will show status "running" forever
  and has no encoder.npz — safe to delete. Flask server stopped (user
  starts it themselves from here on); terminal daemon left running. Final
  session state: **record +6.33 pts over bigram (acc 0.3603), seed-mean
  +5.37**, best encoder `20260702-184303-encoder-538711` (ridge 0.3754),
  leaf calibration in place, all mechanisms in the /tree UI with the
  winning recipe as defaults.

## 2026-07-02 (Claude Code / Fable 5)
- **E17: +6.33 pts — new record on every axis** (acc 0.3603 vs 0.2970, run
  20260702-192113-tree-c6df68): pop-600 encoder reached held-out ridge
  **0.3754** (encoder `20260702-184303-encoder-538711` — the new
  recommended pick in the sidebar Encoder select); trees on it
  +6.33/+5.30/+4.47, **mean +5.37** across seeds, ~15s each, and they're
  post-calibration (sampling works). Encoder budget curve still climbing →
  raised the `encoder_generations` clamp 500 → 2000 (parse_config; server
  restarted). E18 running: pop-1000 / gens-800 encoder + 3-seed trees.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 gen-1 genome experiments (scratchpad i2_exp1/i2_exp2): continuous
  latent codec FALSIFIED for text; predictive-model entropy coding is the
  viable gen-1.** No product code changed — results only.
  - Exp1 (linear codec, PCA closed-form seed + GA refine, pop 64 × 200
    gens, GPU, EEC corpus, 64-byte blocks, int8 latents): byte-EXACT
    reconstruction is ~2.5% at k=48 (1.33× lossy), ~1.4% at k=16 — errors
    are dense (62–63 of 64 bytes wrong), so latent+corrections is
    EXPANSION (0.37–0.45×), and evolution cannot rescue it (GA plateaued
    at the PCA optimum; float32 PCA is equally bad, so quantization isn't
    the problem). Text's zero error tolerance makes continuous lossy
    latents a cliff, not a trade-off.
  - Exp2 (genome = predictive byte model + arithmetic coding — lossless by
    construction, decode requires the exact model, so genome-gating comes
    free): held-out 1MB cross-entropy — order-0 4.55 bits/byte (1.76×),
    order-1 3.47 (2.31×), **order-2 2.80 bits/byte (2.86×) — already beats
    zlib-9 (2.90 bits/byte, 2.76×)**. Genome size is the design lever:
    order-1 ≈ 64KB (browser-shippable), order-2 ≈ tens of MB; the evolved
    tree-LM (0.354 acc > bigram 0.297) slots in as a stronger, compact
    genome once calibrated probabilities are wired to a coder.
  (user vision: P2P network where content travels as latent vectors and
  every client's decoder auto-updates). **Requires a Flask restart.**
  - New `i2_service.py`: generation-0 genome = deterministic orthonormal
    linear codec (block dim 64, seeded, version `g0001`; lossless byte
    round-trip verified — future generations get to be evolved VAEs).
    Pages are stored ONLY as latent documents in `i2_store/pages/*.json`
    (latents + genome version + sha256 + origin placeholder). Three seed
    pages self-publish on import: home, manifesto, network.
  - New node API in `app.py`: `GET /api/i2/genome` (weights included —
    this IS the client auto-update), `GET /api/i2/pages`,
    `GET /api/i2/page/<name>`, `POST /api/i2/publish`.
  - `static/i2.js` rewritten from mock to client: fetches the genome,
    decodes latents in-browser (Wᵀv, TextDecoder), renders a tiny page
    markup (`# ` heading, `* ` bullet, `[[name|Label]]` links), with
    working link clicks, back/forward/reload, wheel scroll, genome-version
    badge in the URL bar, and a genome-mismatch "decoder out of date"
    error page instead of garbled output. Home keeps the logo + search
    bar hero (search still decorative). Footer shows origin + sha256 +
    decoding genome as proof-of-local-decode.
  - Verified end-to-end under the server's conda env (`simple`): seed
    pages publish, server and client-style decodes agree byte-for-byte,
    publish/list/get round-trip passes.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 canvas now draws a mock browser** (`static/i2.js`, display-only per
  user: "nothing has to work"): full-canvas browser chrome — tab strip
  (one tab + new-tab button), nav toolbar (back/forward/reload, padlock +
  URL bar showing `https://i2.local/home`), and a page body with an "I2"
  logo, a centered search bar, and filler content (heading/paragraph
  skeleton bars + three cards). All geometry scales with the canvas and
  degrades gracefully when the terminal dock is dragged tall. No
  hit-testing yet; state lives in the `BROWSER` object for later wiring.
  Static JS only — no Flask restart needed.

## 2026-07-02 (Claude Code / Fable 5)
- **Fixed garbage text generation: per-leaf score calibration** (user
  report: output was random bytes despite +5 pts accuracy). Diagnosis:
  ridge-seeded leaf scores live in ~[0,1], so temperature-0.8 softmax was
  near-uniform — sampling ignored everything the model knew (temp 0.05 was
  already word-shaped, proving the model itself was fine). Fix in
  `_train_node`: after each leaf freezes, a scalar α is found by bisection
  (monotone 1-D root find, no gradients) so the mean top-1 softmax
  probability equals the leaf's measured accuracy, then **α is folded into
  the frozen weights** — argmax/accuracy identical (verified: same 0.3543,
  +5.73), persists in model.npz, and temperature sampling now behaves:
  temp 0.5 gives word-shaped text with real words; temp 0.2 collapses to
  "the the the" (the true modal distribution — expected at 35% byte
  accuracy). Models trained before this fix need low temp (~0.05–0.1) or a
  retrain (~20s with a saved encoder).

## 2026-07-02 (Claude Code / Fable 5)
- **E16: window 32 falsified** — a full hero encoder at window 32 (same
  budget as the 0.3724 w16 winner) reached only ridge 0.3479 and its trees
  +3.12/+2.82 (vs w16's mean +4.94): doubling the mixer genome needs more
  than double the budget before longer context pays. **Window 16 stays.**
  E17 (running): pop-600 encoder with the best-known recipe + 3-seed trees.

## 2026-07-02 (Claude Code / Fable 5)
- **E14–E15: best encoder yet (held-out ridge 0.3724) + best seed-averaged
  trees (mean +4.94)**:
  - E14 standalone encoders (pop 400, 500 gens, depth 2, ridge+rotate):
    **sa_mutation h256 → ridge 0.3724** (encoder
    `20260702-164927-encoder-92859e`, 986s) — self-adaptive mutation is a
    clear win for the encoder GA. h512 + novelty → 0.3055 (worse: capacity
    without proportional budget, novelty diluted selection).
  - Trees on the 0.3724 encoder (~20s each): **+5.73 / +4.90 / +4.18 across
    seeds 23/11/7, mean +4.94** — best seed-averaged config of the session
    (L1 bf4, cluster, ridge seeds, node_resample). Trees (0.3543) now sit
    slightly BELOW the encoder's in-distribution readout — tree/data side
    became the constraint again.
  - E15 more tree data on that encoder: 64k +4.70/+5.26, 96k +5.40, flat
    64k +4.63 — delta holds but no breakthrough (bigram strengthens with
    data too); ~0.34–0.354 is this encoder's true held-out ceiling.
  - E16 (running): window-32 hero encoder (context length is the
    information bound now that encoding quality is high) + trees on it.
  - **Practical recipe for the user**: pick encoder
    `20260702-164927-encoder-92859e` in the /tree sidebar Encoder select →
    20-second tree runs at ≈ +5 pts over bigram; sidebar defaults already
    match the winning tree shape.

## 2026-07-02 (Claude Code / Fable 5)
- **E11–E13: reusable hero encoder + fixed-encoder tree sweeps**:
  - Trained a standalone **hero encoder** (600→clamped 500 gens, pop 300,
    depth 2, ridge+rotate; `runs/encoder/20260702-162154-encoder-5360d7`,
    held-out ridge 0.3417, 745s). Tree runs reusing it via encoder_id take
    **~18 seconds** (vs 5–10 min) at +4.1–4.4 pts — encoder-as-artifact
    works exactly as intended. (exp11 crashed after training on a
    double-stdout-wrap bug in the experiment script itself — encoder was
    saved; no product code affected.)
  - E12 tree-shape sweep on the fixed encoder (~17s/config): L1 bf4 best
    (+4.43); flat +4.30; deeper/wider worse (L2 +3.20, L3 +2.97,
    bf8 +1.62); **no cluster split −2.8 pts** (+1.65) — clustering
    confirmed essential. With encoder variance removed the node mechanisms
    finally show: **node_resample +5.63**, sa_mutation +5.28,
    both +5.65; node gens 300 adds nothing (node side saturated).
  - E13 seed spread with resample: +5.63/+4.05/+2.70 — s23 flatters all
    configs; gains vs plain are within noise across seeds. The **encoder
    remains the ceiling** (fixed-encoder trees top out ≈ its ridge acc +
    routing bonus). E14 (running): two bigger hero encoders (pop 400,
    sa_mutation; one h512+novelty) + trees on the better one.
  - Session record so far: **+5.83** (E10-encg400). Note encoder_generations
    is clamped at 500 by parse_config.

## 2026-07-02 (Claude Code / Fable 5)
- **E10: +5.83 pts above bigram — new record** (acc 0.3553 vs 0.2970, run
  20260702-160411-tree-54f5d9, encoder gens 400): the encoder budget curve
  is still climbing (gens 200 → 400: +5.63 → +5.83); hero seed-repeat +5.30
  (s11; hero mean ≈ +5.5 across seeds); 64k samples +5.23 (no gain from
  more data — encoder quality, not data volume, is the lever). E11 running:
  a 600-gen standalone **hero encoder saved to the encoder store**, then
  tree runs reusing it via encoder_id (reusable artifact — future tree runs
  skip the ~10-min encoder evolution entirely).
- Fixed `templates/tree.html` UTF-8 corruption (a PowerShell Set-Content
  default-encoding mistake while editing sidebar defaults — repaired by an
  exact byte-level round-trip; page verified serving clean UTF-8).

## 2026-07-02 (Claude Code / Fable 5)
- **New I2 page** (`/i2`, linked "I2 ↗" from the main topbar): shell for a
  new program sharing this Flask server — a full-width canvas
  (`static/i2.js`, placeholder grid, exposed as `window.I2`) above the same
  daemon-backed terminal dock as the main page (reuses `static/app.js`
  verbatim — same element IDs, same `/ws` bridge, so tabs are shared with
  the main page and survive Flask restarts). Dock height persists in its
  own `i2_layout` cookie. **Requires a Flask restart** (new route).

## 2026-07-02 (Claude Code / Fable 5)
- **NEW RECORD: +5.63 pts above bigram** (acc 0.3533 vs 0.2970, run
  20260702-153839-tree-4bd710) — E8/E9 results:
  - E8 seed-averaged (3 seeds): split-rotate mean **+3.58** vs champion
    +3.13 (rotation confirmed); deep+rotate +3.44 (equal mean, tighter
    spread).
  - E9 "hero" encoder budget (gens 200, pop 300, fitness samples 16k):
    **deep encoder +5.63** (767s), shallow +5.28 (339s) — the encoder was
    the binding constraint, budget scaling pays, and depth 2 wins at high
    budget (its nonlinearity needs generations to be found).
  - Recommended recipe (now the /tree defaults): context_dim 256, embed 32,
    window 16, 1 routing layer bf 4, cluster split, seeded embeddings,
    ridge node seeds, ridge encoder fitness + rotating split, encoder
    depth 2, GPU. E10 (running): hero seed-repeat, encoder gens 400, 64k
    samples.
- `/tree` sidebar defaults updated to the winning recipe (dims, routing,
  cluster split on, compute GPU); new checkboxes for `sa_mutation`,
  `node_resample`, and (encoder modal) `encoder_split_rotate` (default on).

## 2026-07-02 (Claude Code / Fable 5)
- **E5–E7 experiment results** (all deltas vs their run's own bigram):
  - E5: d384+routing +3.38, 2 routing layers +3.20, bf16 +2.60 — none beat
    the d256/L1bf4 champion (+4.38); "d512" was silently clamped to 384 by
    parse_config (same config hash exposed it).
  - E6: champion seed-repeats +1.90 (s7) / +3.10 (s23) / +4.38 (s11) —
    **seed noise ≈ ±1.2 pts**, ranking now uses repeats. Deep encoder
    (depth 2, h=256) **+4.35** on its first run; h=512 +4.05; doubling
    encoder gens+samples +3.70; pop 200/gens 250 +2.45 (bigger node GA
    budget does NOT help — nodes are already at their ridge optimum, the
    encoder is the binding constraint: champion router 96.4%, rare-byte
    leaves ~99.6%, tree acc ≈ encoder held-out readout 0.347).
  - E7: **encoder_split_rotate +4.58 (new best single run**, acc 0.3365,
    run 20260702-152509-tree-70c937); sa_mutation +2.90 (neutral/negative);
    both together +3.28.
- **Two more evolution mechanisms** (config + sweep, UI follows):
  - `sa_mutation`: ES-style self-adaptive mutation scale — each individual
    inherits a parent's step size × log-normal perturbation (clip
    0.005–0.5); champion still tracked on raw fitness.
  - `encoder_split_rotate`: the ridge encoder fitness re-shuffles its
    fit/val split every generation (deterministic per seed), so selection
    can't creep-overfit a fixed validation half.

## 2026-07-02 (Claude Code / Fable 5)
- **E4 scale probes on the winning recipe** (all vs their own bigram
  baseline): context_dim 384 → **+3.10**; 64k samples → +2.28; window 32 →
  +2.02 (no gain over window 16); seed 7 → +2.60 (robust across seeds); and
  the headline: **routing now HELPS** — 1 layer bf 4 → **+4.38 pts**
  (acc 0.3345, run 20260702-145406-tree-3e185e), the first time a routed
  tree beats the flat model. The aligned encoder (ridge fitness) + ridge
  seeds + clustered split turned routing from a −7pt cost into a +1.7pt
  gain over flat.
- **Two new evolution mechanisms** (for the ongoing experiment loop):
  - **`node_resample`** (sidebar-less for now, config/sweep only): EEC
    non-stationarity for node GAs — the batched fitness closure rotates a
    common half-sample every generation (deterministic per run seed), so
    nodes can't win by memorizing one fitness sample. GPU path only.
  - **`encoder_depth: 2` + `encoder_hidden`** (encoder-modal select "Encoder
    depth"): service-side `DeepContextEncoder` — flat → tanh(W1) → tanh(W2)
    two-stage evolved mixer (position interactions a single linear mix
    can't encode). Stage-2 seeded near-identity (×1.2 vs the double tanh)
    so depth 2 STARTS equal to depth 1. Batched GPU fitness handles both
    layouts; save/load moved to service-side `_save_model`/`_load_model`
    (blueprint mirror + encoder_kind/hidden metadata; old files load as
    kind 1); encoder store + embedding cloud + replay all handle deep
    encoders (smoke-tested end-to-end). Evolve-dims is disabled with
    depth 2 (ragged layout).

## 2026-07-02 (Claude Code / Fable 5)
- **BEAT THE BIGRAM BASELINE: +2.65 pts** (autonomous experiment session;
  user goal "we have to do better than −1.4"). Two new mechanisms, both in
  the project's closed-form-seed + evolve-to-refine pattern (no gradients):
  - **`ridge_seed`**: router/leaf GA seed populations now optionally include
    the closed-form ridge/least-squares solution to one-hot targets
    (`_nc_seed` — one linear solve, same moment-statistics family as the NC
    seed, but it sits at the node's actual linear optimum). Effect at dim
    256 flat: **−11.05 → −0.08 pts** vs bigram.
  - **`encoder_fitness: "ridge"`**: the encoder can now be evolved against
    the **held-out accuracy of a closed-form ridge readout** (fit on half
    the fitness sample, scored on the other half — batched on GPU via
    torch.linalg.solve) instead of the in-sample nearest-centroid proxy.
    This aligns encoder evolution with the ridge-seeded linear leaves that
    consume it AND adds generalization pressure (centroid overfit can't
    win). Effect: **−0.08 → +2.65 pts** (acc 0.3172 vs bigram 0.2908; 32k
    samples, dim 256, window 16, emb 32 seeded, cluster split, flat,
    pop 100/150 gens, seed 11, run 20260702-144002-tree-bb9d13).
  - Also added **`node_fitness: "prob"`** (geometric-mean target probability
    = inverse perplexity, smooth landscape) — currently NOT usable: ridge
    scores aren't calibrated logits, so softmax-based fitness collapses to
    1–12 unique predictions (measured twice). Kept behind the flag for
    future calibrated variants; default "acc" unchanged.
  - Experiment log: E1 scale reference (dim 256 flat −11.05, +1 routing
    layer −18.65) → E2 ridge seed (−0.08; prob-fitness arithmetic-mean
    variant collapsed to uniq=1, replaced by geometric mean) → E3 (+2.65).
    Results in scratchpad results.jsonl; all runs persisted in the runs
    store under notes "exp E…".

## 2026-07-02 (Claude Code / Fable 5)
- **Co-occurrence-seeded embeddings** (diagnosis from the new 3D plot: the
  byte-embedding table stays isotropic/random-looking even when the encoder
  scores 0.625 NC fitness, and even at embed_dim 6 — because NC fitness only
  needs *distinct* codes, not organized ones, and any embedding re-coding can
  be compensated by the mixer: the table is a **neutral plateau** evolution
  never shapes. User: "the embedding on the tree is horrible"):
  - New encoder-modal checkbox **"Seed embeddings from co-occurrence"**
    (`encoder_seed_embeddings`, default off, sweepable "seed embeddings
    (0/1)"): `_cooc_embedding_seed` PCA-projects each byte's preceding-
    context profile (`_byte_context_features`, same features as the
    clustered token split) down to embed_dim and uses that as the heuristic
    seed's embedding table (mean row norm ≈ 1) — bytes used in similar
    contexts START near each other, and evolution refines instead of
    wandering the plateau. Recorded in the runs-panel encoder card
    ("seeded embeddings: co-occurrence PCA / random").
  - Measured (25 gens, pop 40, dim 64/emb 32, GPU, seed 44): embedding
    structure **survives evolution** — PC1 5.6% → **31.4%**, effective rank
    30.0 → **15.8** of 32; NC fitness unchanged (0.290 vs 0.284), as
    predicted (fitness doesn't require embedding geometry — the experiment
    is whether the downstream tree benefits from linguistic context
    vectors). Needs the user's Flask restart.

## 2026-07-02 (Claude Code / Fable 5)
- **Population novelty bonus for the encoder GA** (user: "genomes that are
  more different should get a fitness boost, not a penalty" — the existing
  "Head diversity" is *within-genome* dim decorrelation; this is the
  *population-level* mechanism they described):
  - New encoder-modal checkbox **"Population novelty bonus"**
    (`encoder_novelty`, default off) + **"Novelty strength"**
    (`encoder_novelty_strength`, default 0.5): each generation, a genome's
    novelty = mean L2 distance to the rest of the population (gene-subsampled
    to ≤512 dims for speed), normalized 0–1; **selection** scores become
    fitness × (1 + strength × novelty). Fitness sharing: different genomes
    breed more. The **champion is still tracked on raw fitness** (novelty
    shapes who breeds, not what gets returned), and the fitness chart shows
    raw values. `_evolve` gained a `novelty` param; elites now select on the
    boosted scores.
  - Applies to the encoder accuracy phase (fixed-dim); persisted in configs,
    shown in the runs-panel encoder card ("novelty bonus"), sweepable
    ("encoder novelty bonus (0/1)" and "novelty strength").
  - Verified headless (GPU, 12 gens): runs complete, flags persist,
    acc 0.218 → 0.220 at strength 0.5 (neutral at this tiny scale; the
    mechanism matters on longer runs where convergence stalls progress).

## 2026-07-02 (Claude Code / Fable 5)
- **3D embedding-space plot on the runs page** (user: "I have a feeling it
  might be collapsing"):
  - `tree_service.embedding_cloud`: PCA (SVD) of the encoder's byte-embedding
    table (vocab × embed_dim) → 256 unit-scaled 3D points + collapse
    diagnostics: per-PC explained variance, **effective rank**
    (exp of spectral entropy — ≈embed_dim when isotropic, →1 on collapse),
    mean vector norm. Works for tree runs (model.npz) and standalone encoder
    runs (encoder.npz). `runstore.embedding(rid)` +
    `GET /api/runs/<rid>/embedding` (needs the user's Flask restart).
  - `/runs` detail (tree + encoder runs with a checkpoint): new **"Embedding
    space (PCA → 3D, drag to rotate)"** card — hand-rolled canvas scatter
    (no 3D lib), drag-rotates (yaw/pitch), depth-sorted with size/alpha
    cues, points colored by byte class (␣ / a–z / A–Z / 0–9 / punctuation /
    control+extended), common letters (␣ e t a o n i) labelled, legend, and
    a diagnostics line with an explicit ⚠ verdict when PC1 > 90% or the
    effective rank is low.
  - Fixed alongside: `_persist_finalize` now writes the **final**
    context_dim back into config.json, so dim-evolved models can actually be
    loaded again (replay + embedding previously would fail to unpack).
  - Encoder runs on the dashboard hide the inference card (nothing playable).
  - First measurement on today's small test runs: effective rank ~15.4 of 16,
    PC1 ≈ 10% — NOT collapsed, but near-isotropic, i.e. barely more
    structured than random init (evolution has mostly not shaped the
    embedding table yet). Note: high rank ≈ random-like is as informative as
    low rank ≈ collapse.

## 2026-07-02 (Claude Code / Fable 5)
- **Encoder training is now its own modal + saved/reusable encoders** (user:
  train an encoder once, then use/replace it across tree runs instead of
  re-genning both every run):
  - `/tree` sidebar: the encoder fields moved out of the sidebar into an
    **"Encoder trainer" modal** (button "Train encoder…"); the sidebar now has
    an **Encoder select** — "evolve fresh each run" (old behavior) or any
    saved encoder. New "Fitness samples" field (`encoder_samples`) is finally
    exposed in the modal.
  - `tree_service.EncoderTrainer` (op `train_encoder`, runs in the job hub
    like tree runs — survives page nav, one job at a time, Stop works):
    evolves just the encoder (all existing knobs: gens, pop, dims evolution,
    time/diversity constraints, speed phase, GPU batching) and saves
    `runs/encoder/<id>/` — config/history/summary in runstore layout (so it
    appears under a new **"encoder" tab on the runs dashboard**, sparkline +
    Encoder evolution card work) plus `encoder.npz` (genome + dims).
  - `list_encoders()` / `load_encoder()`; socket op `encoders` returns the
    saved list (requested on connect; refreshed after each save); the modal
    shows the saved-encoder table with per-row **use** buttons.
  - Tree runs with `encoder_id` set **load the saved encoder, skip encoder
    evolution entirely, and adopt its dims** (context_dim/embed_dim/window
    override the sidebar — logged in a status line; falls back to fresh
    evolution if the encoder can't load). `encoder_id` persists in run
    configs.
  - Verified headless: standalone train (GPU) → saved + listed; tree run with
    deliberately different sidebar dims adopted the encoder's dims (48/16/8),
    did not re-evolve, trained + evaluated + persisted normally.
  - app.py gained the two socket ops — needs the user's Flask restart to
    take effect (not performed here).

## 2026-07-02 (Claude Code / Fable 5)
- **Repository published to GitHub** (user request): added `LICENSE` (GNU AGPL-3.0
  — strict copyleft covering network use) and committed the full working tree for
  publication as the public repo `GENREG-Builder` under the `A1CST` account.

## 2026-07-02 (Claude Code / Fable 5)
- **GPU compute option for Tree-of-Models** (user: "these longer trainings are
  killing me"): new **Compute** selector (CPU / GPU (CUDA, batched)) in the
  `/tree` sidebar, `device` in the config (persisted in run configs).
  - GPU mode evaluates the **whole population per generation in one batched
    tensor op** on CUDA instead of a Python loop of numpy matmuls:
    `_batch_linear_acc` (every router/leaf: genomes → (P,dim,k) einsum →
    accuracy) and `_batch_encoder_fitness` (replicates
    ContextEncoder.encode — embed gather + positions → bmm mix → tanh — plus
    batched nearest-centroid accuracy, and batched time/diversity penalties
    when those constraints are on). Population chunked to bound GPU memory.
    `_evolve` gained a `batch_fitness` hook; the GA logic is unchanged.
  - Measured (pop 100, 16k samples, 30 gens, layers 2, RTX 4080):
    **79.8s → 7.4s ≈ 10.8×**; small configs ~1.8×. Encoder accuracy matches
    CPU to ~1e-3 (float32 vs float64 — champions can differ slightly, same
    caveat as the main engine's GPU mode).
  - Falls back to CPU with a status message if CUDA/torch is unavailable.
    "Evolve context dim" runs keep the CPU loop (ragged genomes); the
    dim-evolution GA is exempt from batching.

## 2026-07-02 (Claude Code / Fable 5)
- **Clustered token split** (user idea: the root's 0–63/64–127/… split is an
  arbitrary token-ID artifact, not linguistic — let the tree be built over
  context clusters instead):
  - New `/tree` checkbox **"Cluster token split"** (`cluster_tokens`, default
    off, sweepable "cluster token split (0/1)"): bytes are described by their
    **preceding-context profiles** (Hellinger-scaled distributions of the
    previous 1–2 bytes, `_byte_context_features`) and recursively partitioned
    by **capacity-balanced k-means** per node (`_balanced_kmeans` — balanced
    so leaf sizes still match the routing_layers math; deterministic per run
    seed so the dim-evolution rebuild reproduces the same partition;
    `_build_root` replaces direct `build_tree` calls). tree_lm blueprint
    untouched — its TreeNode/inference/save-load were already set-based.
  - `_train_node` is now fully set-based (membership mask, leaf local-index
    lookup, router child-of lookup) instead of assuming contiguous ranges —
    identical results for sequential trees.
  - Icicle t0/t1 are now DFS *positions* (== byte values for sequential
    trees, a contiguous layout axis for clustered ones); manifest + trace
    nodes carry a `sample` token preview ("␣ e t a …") which the icicle
    tooltip and the routing-inspector/trace views show instead of
    "bytes N–M" when present.
  - Measured (6k samples, dim 48, bf 4, layers 2, seed 11, small run):
    clustering is plainly linguistic (uppercase / common-lowercase+space /
    unused-high-bytes branches) and held-out accuracy **0.129 → 0.168**
    (+3.9 pts) vs the sequential split on identical config. Note: ␣/e/t
    landed *together* (they share preceding contexts — co-occurrence groups
    similar-context bytes, making ROUTING easy and leaves hard); it still
    beat byte-ID ranges. Clustered models replay fine from the runs store.

## 2026-07-02 (Claude Code / Fable 5)
- **Separate encoder population** (`encoder_pop_size`, `/tree` sidebar field
  "Encoder population", default 0 = inherit the shared Population): the
  encoder GA previously always used the same pop_size as every router/leaf;
  since the encoder genome is orders of magnitude larger, it can now have its
  own population. Applies to all three encoder GAs (fixed-dim `_evolve`, the
  variable-dim GA, and the phase-2 speed phase; seed-count scales with it) —
  `_evolve`/`_evolve_encoder_dims` gained a `pop_size` override parameter
  (node training is untouched; seed_pop is now also truncated to pop_size).
  Persisted in run configs, shown in the runs-panel Encoder evolution card
  (effective size), and sweepable ("encoder population (0 = shared)").
  Verified headless: 0 inherits (pop 10), 24 applies to fixed/speed/dims
  paths, full runs complete.

## 2026-07-02 (Claude Code / Fable 5)
- **Encoder dimension evolution** (answering "are we evolving the dims in the
  encoder?" — previously no, context_dim was always fixed by the sidebar):
  - New `/tree` sidebar checkbox **"Evolve context dim"**
    (`encoder_evolve_dims`, default off = fixed number, the current
    behavior). When on, `tree_service._evolve_encoder_dims` runs a
    **variable-dimension GA**: each individual carries its own context_dim;
    structural mutation (p=0.25) grows (append a small random mixer column)
    or shrinks (drop the lowest-|W|+|b| column) within **[4, 4×start]**
    (start = the sidebar Context dim); crossover mixes parents in their
    shared column subspace (child keeps parent 1's dim). Works with the
    time/diversity constrained fitness (all fitness closures now take an
    optional dim; scratch probe encoders are cached per dim via
    dataclasses.replace). Per EEC, pure accuracy fitness tends to grow dims —
    the hint says to pair with time/diversity for shrink pressure.
  - Downstream: router/leaf genome sizes depend on context_dim, so when the
    evolved dim differs the trainer **rebuilds the routing tree at the new
    width** and re-emits the tree manifest (icicle + params tiles update);
    cfg.context_dim is updated so persistence/replay (`save_tree`/`load_tree`)
    and phase 2 all use the evolved width. node_gen events for the encoder
    gain `dim`/`mean_dim`; node_done gains `context_dim`. Sweepable
    ("evolve context dim (0/1)").
  - Verified headless (start dim 32, 6 gens): fixed mode unchanged; evolve
    mode reached dim 33/34, tree rebuilt at the evolved width, full run +
    model save completes.
- **Encoder training is now fully recorded in the runs dashboard** (it
  previously only left one history line):
  - `summary.json` gains an **`encoder` block**: NC accuracy, fitness
    samples, generations, start→final context dim, per-constraint settings
    and outcomes (time budget + active-weight fraction, diversity budget +
    head redundancy), speed-phase results, plus **per-generation best/mean
    fitness curves** for "encoder" and "encoder-speed" (captured from the GA
    loops).
  - `/runs` detail pages for tree runs show a new **"Encoder evolution"
    card**: metrics table + fitness-curve sparkline(s) (labelled
    "(constrained)" when a constraint shaped the curve). Old runs without
    the block simply don't show the card. The ⤓ Export JSON download
    includes it automatically (it lives in summary).
  - Loads when the Flask server is next started (no restart performed).

## 2026-07-02 (Claude Code / Fable 5)
- **Encoder head-diversity constraint** (user experiment: time pressure didn't
  break the plateau — try pushing the encoder's output dims to be different):
  - New `/tree` sidebar checkbox **"Head diversity"** (`encoder_diversity`,
    default off) + **"Diversity budget"** field
    (`encoder_diversity_budget`, default 0.5, lower = harsher): when on, the
    encoder's whole accuracy phase is scored as
    fitness ÷ (1 + redundancy/budget), where **redundancy = mean
    |off-diagonal correlation| between the encoded context dims** on the
    fitness sample (0 = every head carries a different signal). Dead
    (constant) dims count as fully redundant so collapsing dims can't game
    the penalty.
  - Composes with the time constraint: both checkboxes on → both penalties
    multiply into one constrained fitness (`fitness_constrained` now folds in
    whichever constraints are enabled). The phase-2 speed phase got its own
    pure-time closure back so it is unaffected by the diversity flag.
  - node_done for "encoder" now carries `redundancy` when the mode is on;
    the status line reports acc + per-constraint numbers. Both new fields
    persist in run configs and are **sweepable** ("encoder head diversity
    (0/1)" and "diversity budget"), so diversity × time × budgets can be
    A/B'd in one sweep. Reported accuracy stays raw for comparability.
  - Verified headless (seed 5, 4 encoder gens): baseline acc 0.448;
    diversity on (budget 0.3) → redundancy 0.072 at acc 0.420;
    time+diversity together → 44% active weights, redundancy 0.078,
    acc 0.408; flags persist; full runs complete.
  - Loads when the Flask server is next started (no restart performed).

## 2026-07-02 (Claude Code / Fable 5)
- **Time-constrained encoder evolution from scratch** (user experiment: evolve
  the encoder with time & its original fitness *combined*, instead of only as
  a phase-2 addition):
  - New `/tree` sidebar checkbox **"Time-constrained from scratch"**
    (`encoder_time_constrained`, default off): when on, the encoder's whole
    accuracy phase (all `encoder_generations`) uses
    fitness ÷ (1 + active-weights/budget) — the time/Occam constraint folded
    into the original nearest-centroid fitness from generation 0, sharing the
    existing "Speed time budget" field. The seed population additionally gets
    magnitude-pruned variants (30/50/70%) of the heuristic genome so the
    sparse region is reachable immediately. Reported accuracy stays raw
    (unconstrained) for comparability; node_done for "encoder" now carries
    `active_fraction` and a status line reports acc + active-weight % when
    the mode is on. The flag persists in run configs and is **sweepable**
    ("time-constrained encoder (0/1)", suggest 0,1), so it can be A/B'd
    against the two-phase form in one sweep.
  - The independent phase-2 "Encoder speed phase" is unchanged (hint text now
    clarifies it runs AFTER the accuracy phase); its fitness now reuses the
    same constrained-fitness closure instead of a duplicate definition.
  - Verified headless: constrained-from-scratch run (budget 0.3) → encoder at
    44% active weights, flag persisted, full train/eval completes; two-phase
    regression run unchanged (dense phase 1 → 30% active after speed phase).
  - Note: `tree_service.py` changes load when the (currently stopped) Flask
    server is next started — no restart was performed.
- **Runs can now be exported to JSON**: every run detail page on `/runs` has a
  **"⤓ Export JSON"** button (next to the checkpoint badge) that downloads
  `<run-id>.json` containing the full run record — config (launch params +
  metadata + status), summary (final result / eval block / sweep results),
  complete per-generation history, and, for tree runs, all saved routing
  traces. Entirely client-side (`runs.js exportRun`, Blob download from the
  existing `/api/runs/<id>` + `/traces` endpoints) — no new server routes.

## 2026-07-02 (Claude Code / Fable 5)
- Added a **Documentation browser** (`/docs`, "Docs ↗" links in the build /
  runs / tree topbars) styled like the runs dashboard:
  - Copied the user's `Desktop\documentation` folder (91 files: .md, .pdf,
    .json, .docx, …) into `GENREG/documentation/`.
  - `app.py`: `GET /docs` (page), `GET /api/docs` (recursive file list with
    ext/size/mtime), `GET /api/docs/file/<path>` (md/txt/json served inline as
    utf-8 text, PDFs as application/pdf for the embedded viewer, everything
    else as a download; `send_from_directory` blocks path traversal —
    verified).
  - `templates/docs.html` + `static/docs.js` + CSS (`doc-*`, `md-body`):
    left sidebar with type tabs (All / Markdown / PDF / JSON / Other), a
    name filter, and run-node-styled file entries; detail pane renders
    **Markdown inline** via a small built-in renderer (headings, fenced code,
    tables, lists with hanging-indent continuation, blockquotes, inline
    formatting — HTML-escaped first, no external libs), **PDFs in a native
    `<embed>` viewer**, JSON pretty-printed, and a ⤓ Open/download button on
    every file. Deep-linkable via `/docs#<path>`.
  - Verified end-to-end after a Flask restart: list API, md/pdf serving,
    traversal 404, and the renderer against the real doc set (tables/code
    blocks balanced, no unescaped HTML).

## 2026-07-02 (Claude Code / Fable 5)
- **Tree training now survives leaving the page.** Previously the `/treelm`
  socket handler owned the trainer and stopped it on disconnect, so clicking
  "Runs ↗" (same-tab nav) killed the run/sweep. Now:
  - `tree_service.JobHub` (module singleton `HUB`): the trainer/sweeper runs
    in the hub, decoupled from any socket. Sockets are viewers — they
    subscribe on connect and detach on close. The hub keeps a replayable
    journal (all events except per-generation ticks, which are kept only for
    the node currently evolving; latest status only; capped at 50k), so a
    reconnecting page replays the snapshot and rebuilds mid-run state
    (icicle, fitness chart, sweep table, tiles), then gets a
    `{"type":"job","running":…}` event to sync the buttons.
  - Stopping is explicit only: the Stop button or starting a new job.
    Multiple tabs can watch the same run simultaneously.
  - Per-connection send lock (hub thread + handler thread both write).
  - Verified: killed the socket mid-run — training completed server-side and
    a fresh connection replayed tree/node/eval/done correctly.
- Added a **configurable config sweep** to the `/tree` page:
  - New "Config sweep" card: each sweepable parameter (routing layers,
    branching, dims, window, gens/pop, encoder gens, speed gens, speed budget,
    …) is either **locked to the sidebar value** (unchecked) or given a
    comma-separated **list of values to test**; every combination trains
    (Cartesian product, capped at 24 candidates, values capped at 8/param),
    live-ranked in a results table, all candidates on a shared seed (1234)
    for comparability.
  - `tree_service.TreeSweeper` (`/treelm` op "sweep"): runs candidates
    sequentially as normal TreeLMTrainer runs (each persists to the runs
    store, tagged `sweep <id> · <candidate>` in its notes) and persists the
    sweep itself as a run (`runs/tree/<stamp>-sweep-<hash>/`) whose
    summary.json carries `sweep_results` (ranked) and whose history.jsonl
    holds per-candidate accuracy (sparkline works).
  - `/runs` dashboard: sweep entries show a **"Sweep results (ranked)"**
    table in the detail view with per-candidate links that jump to the
    candidate run's detail; `/runs#<run-id>` now deep-links to a run (the
    sweep table on /tree links out this way); run details now show the
    `notes` row (so candidates display their sweep membership).
  - Sidebar: encoder speed phase is now behind an **enable checkbox**
    (default off — measurement showed it compresses but slightly hurts
    downstream accuracy) and the **time budget is exposed** as a field, so
    the constraint's strength can be tuned rather than only toggled.
- Added a **two-phase encoder evolution with a speed/time constraint**
  (user experiment: does Occam pressure break the ~25% encoder plateau?):
  - After the accuracy-only phase (encoder_generations), a second phase runs
    for `encoder_speed_generations` (default 40, 0 = off, UI field "Encoder
    speed gens") with fitness × 1/(1 + active/budget) — the project's Time/
    Occam constraint form, where "time" = fraction of genome weights with
    |w| > 0.01 (effective multiplies; for a fixed-shape dense encoder,
    sparsity is the evolvable notion of speed). `encoder_time_budget`
    (default 0.5) sets the pressure. Phase 2 is seeded with the phase-1
    champion plus magnitude-pruned variants (30/50/70/85%).
  - Streams as node "encoder-speed" (live fitness chart, run history); emits
    the accuracy delta and final active-weight fraction.
  - First measurement (dim 64, layers 0, seed 11): phase 2 compressed the
    genome to **32% active weights at −0.009 fitness** (the dense genome was
    ~3× redundant) but did not break the plateau; downstream tree accuracy
    was noisier/slightly worse. The plateau itself tracks the linear NC
    ceiling at context_dim and the 2000-sample centroid noise — raising
    `encoder_samples` and context_dim moves it more than generations do.
- Added a direct **Routing layers** lever (0–8) to the `/tree` config,
  replacing the min-leaf-tokens field: leaf size is derived as
  256 / branch^layers, and **0 layers builds no routers at all** — the root is
  a single flat evolved specialist over all 256 bytes.
  `routing_layers` is accepted by `parse_config` (explicit `min_leaf_tokens`
  still works and can now go to 256) and persisted in run configs.
  Measured at dim 64 / bf 4 (seed 11): layers 0 → 0.260, 1 → 0.205,
  2 → 0.227, 3 → 0.211 (bigram 0.336) — the flat model is currently the most
  accurate, i.e. each routing level still costs accuracy; the icicle +
  inspector show where.
- **Implemented encoder evolution (blueprint step 1) + nearest-centroid GA
  seeding** — attacks the verified below-bigram collapse:
  - `tree_service._evolve_encoder`: evolves the full ContextEncoder genome
    (via blueprint pack/unpack) before tree training. Fitness = nearest-
    centroid next-byte accuracy of the encoded context (closed-form,
    gradient-free proxy for linear-router separability). The GA population is
    seeded with a structured candidate whose mixer has identity blocks for the
    most recent byte positions (last-byte/bigram information survives the
    projection from gen 0) plus recency-decayed random projections for older
    positions. New config: `encoder_generations` (default 40, 0 = skip, UI
    field added), `encoder_samples` (default 2000). Encoder stage streams
    node_start/node_gen/node_done with id "encoder" (live fitness chart) and
    is recorded in run history.
  - `_nc_seed`: every router/leaf GA population is now seeded with the
    closed-form nearest-centroid classifier for its local data (weights =
    class centroids, bias = −½‖μ‖², absent classes −1e6) — evolution refines a
    working solution instead of random noise.
  - Measured (8k samples, pop 40, gens 40, seed 11): held-out accuracy
    0.074 → **0.228** (bigram 0.336); unique predictions 3 → 25; temp-0.8
    output goes from 'ooo…' loops to word-shaped text. Remaining gap to bigram
    is representational (a 64-dim tanh context vector can't hold a full
    256-way last-byte table); larger context_dim/embed_dim narrows it
    (dim 256 / emb 64 / bf 16: −0.075 vs bigram) at ~3× train time.
  - UI default `min_leaf_tokens` 1 → 16: single-token leaves make temperature
    sampling a no-op (nothing to sample over), which presented as "generates
    the same character regardless of temp".
  - `infer_run` now loads model.npz through an in-memory buffer — np.load
    inside the blueprint's `load_tree` otherwise holds the file handle and
    locks the file on Windows.
  - Note: run `20260702-053927-tree-211b4b` had config/summary/history
    accidentally deleted during test-run cleanup while its model.npz was
    locked by a replay; config.json was reconstructed from the model file
    (tree shape exact, GA params assumed defaults), history and eval metrics
    are lost. model.npz is intact and replays.
- **Routing traces now persist with their run** and are browsable from `/runs`:
  - `tree_service.trace_generate` appends every trace (prompt, text, temp,
    full per-step decision data, timestamp, run id) to
    `runs/tree/<id>/traces.jsonl` of the model's run; `runstore.get_traces` +
    `GET /api/runs/<rid>/traces` serve them newest-first.
  - Trace rendering extracted to shared `static/trace_view.js`
    (`TraceView.mount`), used by both the `/tree` inspector and a new
    **Saved traces** card on tree-run detail pages in `/runs` (list of traces →
    click to replay the full inspector).
  - Every mounted trace has a **⤓ JSON export** button (downloads the complete
    trace as a formatted .json file).
  - Diagnosis note (verified by measurement, not a code bug): generation IS
    autoregressive — the context vector changes every step — but the model is
    context-insensitive because the ContextEncoder is a *random, unevolved*
    projection (the blueprint's "pre-evolve encoder" step has no implementation
    yet), so routers collapse to majority routes (~everything → e/o/n leaves)
    and score below the bigram baseline. Biggest tuning lever: evolve the
    encoder (GENREG hook exists in `tree_lm.train_tree_bottom_up` step 1).
- Added a **routing inspector** to the `/tree` page for debugging generation:
  - `tree_service.trace_generate(prompt, length, temperature)` generates like
    `generate_text` but records every decision — per step: the context window,
    each router's full score vector + chosen child (with children's byte
    ranges), and the leaf's top-8 candidates with softmax probabilities.
    `/treelm` gained a `trace` op returning it as one `trace` event
    (length capped at 200 steps).
  - UI: a "Routing inspector" card — trace a prompt, get a clickable strip of
    the generated bytes (control bytes shown escaped); selecting a byte shows
    its full decision path top-down: one block per router with score bars per
    child (chosen highlighted, decision **margin** shown, ⚠ near-tie flag when
    margin < 0.05), then the leaf's candidate distribution (prob bars, chosen
    ✓). Makes it visible where routing collapses (e.g. every context funneling
    into the same child) vs. where leaves are just under-trained — the two
    things to tune differently.
- **Tree-of-Models runs now persist in the runs store** and appear on the `/runs`
  dashboard under a "tree" environment tab.
  - `tree_service.py`: every training run writes `runs/tree/<id>/` in the
    runstore layout — `config.json` (full tree config + status), `history.jsonl`
    (one line per frozen node: accuracy as best, running mean), `summary.json`
    (held-out accuracy as best.score, bigram baseline as best.base, full eval
    block), and `model.npz` (the frozen tree, blueprint `save_tree` format).
    Stopped runs persist too (partial model, status "stopped"); errored runs get
    status "error". Replaces the previous flat `runs/tree/tree_*.npz` dump
    (pre-existing flat files were left in place but have no run metadata).
  - `runstore.infer`: tree branch — "replay" for a tree run loads `model.npz`
    (via `tree_service.infer_run` / blueprint `load_tree`) and generates a
    ~400-byte text sample at temperature 0.8.
  - `runs.js`: tree runs render in the tech tree (accuracy as score, node
    history sparkline works as-is); the inference panel shows "▶ Generate
    sample" and displays the generated text instead of a game board.
  - `/treelm` socket now emits a `run` event with the run id; done/stopped
    events include `run_id`.

- Added a **Tree-of-Models text prediction page** (`/tree`, "Tree LM ↗" links in
  the build-interface and runs topbars) — a separate program from the build
  interface, implementing the GENREG tree-of-models blueprint: byte-level
  tokenizer (vocab 256), evolved context encoder, hierarchical routing tree with
  evolved leaf specialists, bottom-up freeze-and-stack training, no gradients.
  - `genreg_train/tree_lm.py`: the blueprint skeleton, kept verbatim
    (TreeConfig / ContextEncoder / TreeNode / build_tree / inference /
    save-load / evaluate / `_default_evolve` GENREG hook).
  - `genreg_train/tree_service.py`: web wrapper — re-implements only the
    orchestration (bottom-up walk + GA, mirroring `_default_evolve`) so it can
    stream per-node/per-generation events and honor stop; trains on windows
    sampled from the **49 MB / 50-book Gutenberg corpus**
    (`project/EEC-main/engine/corpus.txt`) instead of the toy demo string, with
    a held-out split, vectorized bigram baseline, temperature sampling for
    generation, and model persistence to `runs/tree/*.npz`.
  - Flask: `GET /tree` (page) + `/treelm` WebSocket
    (`start` / `stop` / `generate` ops → corpus / tree / node_* / eval / done /
    generated events), same thread layout as `/train`.
  - `templates/tree.html` + `static/treelm.js` + CSS (`tlm-*`): config sidebar,
    stat tiles (held-out accuracy, bigram baseline, Δ, tokens/sec, params,
    frozen-node progress), an **icicle map of the routing tree** (x = byte range
    0–255, y = depth; nodes colored by accuracy on a blue sequential ramp, amber
    pulse while evolving, tooltips per node), a **live best/mean fitness line
    chart** for the node currently evolving (crosshair + tooltip), **mean
    accuracy by depth** grouped bars (routers vs leaves, plus table view), and a
    text-generation panel. Chart palette validated for CVD/contrast on the dark
    surface.

## 2026-07-01 (Claude Code / Opus 4.8)
- Added a **Runs & Checkpoints dashboard** — a separate page (`/runs`, opens in a
  new tab via the "Runs ↗" link in the build-interface topbar) for a second monitor.
  - `runstore.py`: every Start Training persists `runs/<env>/<id>/` with
    `config.json` (launch config + metadata), `history.jsonl` (per-gen metrics),
    `summary.json` (final result), and `checkpoint.pkl` (champion genome, engine
    format). Wired into Flask `/train` via an emit wrapper; both trainers expose
    `champion()`. Interrupted runs still save the last completed generation's
    checkpoint.
  - Flask: `GET /runs` (page), `GET /api/runs` (list grouped by env),
    `GET /api/runs/<id>` (config + metrics + history),
    `GET /api/runs/<id>/replay` (loads the checkpoint and plays one fresh episode
    → frames for the inference viewer).
  - `templates/runs.html` + `static/runs.js` + CSS: per-environment **tech-tree**
    (env tabs to switch; runs as connected nodes with status/checkpoint badges),
    a run **details** panel (full config, result metrics, best/mean fitness
    sparkline), and an **inference/verify** viewer that animates the saved
    checkpoint playing snake/2048 on a board canvas.
  - Verified end-to-end: runs persist (CPU + GPU, snake + 2048), API lists/details
    work, and checkpoint inference replays.

## 2026-07-01 (Claude Code / Opus 4.8)
- Added a **GPU-capable vectorized training engine** (PyTorch), device-selectable.
  - Benchmarked first: naive per-step GPU is *slower* for these tiny sequential
    nets (0.4–0.7× at P≤2048), but the fully **batched** vectorized engine crosses
    over ~P=4k and reaches **1.92× faster than CPU at P=16,384** on the RTX 4080,
    and keeps scaling — the right architecture for large populations / heavier
    future envs.
  - `vector_engine.py`: batched recurrent forward, batched relative self-adapting
    mutation, batched selection/elitism, and `VSnake` (grid-based batched snake
    matching the CPU rules). Verified it learns (best fitness 1.1 → 11 in 30 gens).
  - `vector_trainer.py`: `VectorTrainer` emits the **same events** as the CPU
    trainer; each generation the champion is pulled off the device into a CPU
    genome so the Microscope layers + board replay reuse the existing path.
  - `trainer.create_trainer()` factory dispatches CPU-engine vs vectorized by a new
    `device` setting (`cpu` | `gpu` | `auto`). GPU mode currently covers **snake
    with no constraints, fixed H**; anything else (2048, any constraint,
    evolve-hidden) falls back to the full CPU engine. `auto` uses GPU only when
    cuda is present and population ≥ 2000. Flask `/train` now uses the factory.
  - Control Panel: **Compute** selector (CPU engine / GPU vectorized / Auto).
  - Installed PyTorch (CUDA 12.4 build) into the app's conda env.
  - **Vectorized 2048** (`V2048`): batched slide/merge (all 4 directions computed
    then gathered per-board), valid-move masking, 16-feature obs. Learns (best
    ~3.8k → 6.3k in 12 gens, matching the CPU engine).
  - **Vectorized constraints** (`build_vconstraints` + batched application in the
    evaluators): energy (env-specific restore — snake apples, 2048 merges),
    mortality, time (post-score), noise + occlusion (obs), entropy (state decay),
    efficiency (batched weight cost). `VEC_SUPPORTED` gates them; the factory
    routes supported combos to GPU and unsupported ones (memory-rent, scarcity,
    reproduction-cost, non-stationarity, perception-cost) to the CPU engine.
    Champion replay uses the matching CPU constraints so the board display agrees.
    Verified: energy shortens snake fitness; GPU+energy+time runs end-to-end.
- Energy constraint is now environment-specific (restore method differs per game):
  - **2048**: energy is restored on merges — each merge of two v-tiles restores v
    (a move's gain = Σ2v, so restore = gain/2 × `merge_energy`; 2+2→+2, 4+4→+4).
    New "Merge energy ×" param (default 1.0), shown only for 2048.
  - **snake**: unchanged flat "Food energy (per apple)" restore, shown only for snake.
  - Both keep Energy budget + Energy/step. Param rows are gated by a `data-env`
    attribute; `training.js` shows only the rows matching the selected environment
    and re-evaluates on environment change. (`constraints_map.Survival`,
    `templates/index.html`, `static/training.js`.) Verified per-step energy deltas.
- Control Panel settings now **persist across refresh** — every input (environment,
  population, generations, constraints, all params, evolution controls, snake dims)
  is saved to localStorage on change and restored on load, with change/input events
  re-fired so the board, param visibility, PO rings, and grey-out re-sync.
  (`static/training.js`.)
- Control Panel: checking "Evolve hidden width (H)" now greys out (disables) the
  Net width input and shows an "evolved" tag next to it. (`templates/index.html`,
  `static/training.js`, `static/style.css`.)
- Investigated "microscope always shows 24 dims": confirmed NOT a bug — the
  Microscope payload's layer-row count exactly tracks the champion's H every
  generation (verified 24→26→25→23 with evolve_hidden; 24→17 with memory-rent).
  Without a size cost, dimension evolution is neutral (per EEC "size only changes
  under a cost"), so the champion keeps its starting H — reproduced H=24 for 8
  gens identically on the live server and headless. Guidance: add memory-rent
  (Cost strength ~0.03–0.05) to drive H down and watch the Microscope shrink.

## 2026-07-01 (Claude Code / Opus 4.8)
- Microscope now clearly indicates when it is showing **live training data** vs the
  illustrative demo: the badge reads green `● live · gen N` while a real genome is
  streamed in (`setExternal`/`setGenome`), grey `demo · gen N` otherwise. Confirmed
  the data path is real — during snake training it renders the actual 2-layer
  champion (`11→24→3`) with live weights/saturations, not the 4-layer demo. The
  wiring was already correct; this just makes it unambiguous. (`static/microscope.js`.)
- Added a floating **training metrics HUD** over the game canvas (draggable by its
  header, hideable via ×). Shows gen k/N, best/mean/median fitness, champion score,
  peak score (best-ever), steps, and H·leak·bits, plus a live **sparkline** of
  best (accent) and mean (grey) fitness across generations. Appears on training
  start; driven by the generation events in `training.js`. Frontend-only
  (`templates/index.html`, `static/style.css`, `static/training.js`).
- Energy constraint: the per-step energy drain is now a configurable **`step_cost`**
  parameter (default 0.01, range 0.01–5.0) instead of a hardcoded 1.0, applied in
  both snake and 2048. New "Energy / step" input appears under Parameters when
  Energy is checked; wired through `training.js → constraints_map.Survival`.
  Verified: energy_budget/step_cost = lifespan (100/5.0 → 20 steps, 100/1.0 → 100).
- **Stop is now responsive**: `_fitness` short-circuits to a sentinel once stopped,
  so the in-flight generation ends within ~one episode instead of grinding through
  every genome; the run breaks before emitting a half-evaluated generation and the
  `done` event reports the last fully-evaluated champion (no extra replay episode).
  Verified: stop() → done in ~0.0s.

## 2026-07-01 (Claude Code / Opus 4.8)
- Added an **Evolution** section to the Control Panel with controls wired through
  to the Evolver: "Evolve hidden width (H)" (toggle dimension mutation vs fixed H),
  "Mutation only (no crossover)" (asexual `sexual=False`), "Elite (keep best N)",
  and "Breed from top %". User knobs are the base; scarcity/reproduction-cost
  constraints still override them (laws win). `training.js` sends the new fields;
  `trainer.py` parses/merges them. Verified headless (mutation-only runs, scarcity
  override, H actually evolves).
- Better training status when the socket closes before any `started` event now
  reads "couldn't reach /train — restart the Flask server", since that means the
  running server predates the `/train` route (the usual cause of "disconnected").

## 2026-07-01 (Claude Code / Opus 4.8)
- **Wired the genreg-engine to train Snake and 2048 for real.** New Python package
  `genreg_train/` bridges the neuroevolution substrate (`project/genreg-engine-main`)
  to the GUI. Pieces:
  - `envs.py` — `SnakeEnv` (W×H, 11-feature obs, 3 relative actions) and
    `Game2048Env` (4×4, 16-feature obs, valid-move masking). World-consequence
    fitness (apples / merged tile score), JSON `render_state()` for the canvas.
  - `agent.py` — recurrent rollout (`fresh_state`/`rstep`) with per-step constraint
    hooks; downsampled replay frames.
  - `constraints_map.py` — maps all 12 constraints to real effects (energy/mortality
    survival budgets, time/perception post-score, efficiency/memory-rent engine
    costs, entropy state-leak, occlusion/noise obs perturbation, scarcity/
    reproduction-cost turnover, non-stationarity world drift). Same-axis pairs
    handled "swap, don't stack".
  - `trainer.py` — common-seed episodic fitness (fair per-generation evaluation),
    Evolver with recurrence + precision (+ dimension when a size cost is present),
    streams per-gen events (telemetry, champion genome, fresh-episode replay, PO).
  - `engine_api.py` (bridge), `run_headless.py` (CLI verifier), `SKELETON.md`
    (design spec), `REVISIONS.md` (5-pass hardening log).
  - Installed `numpy` (engine dependency) into the app's Python.
- **Flask `/train` WebSocket** (`app.py`) runs a Trainer in a background thread and
  streams events; `start`/`stop` protocol; terminal daemon untouched.
- **Frontend**: `static/training.js` (owns `/train`, assembles config, fans events
  to board/microscope/PO/HUD); Control Panel gains a Training section
  (Generations, Start/Stop, status) and constraint-parameter inputs that reveal per
  checked constraint; `ui.js` board renderers now draw live training frames
  (snake body/food, 2048 tiles) via `GENREG.board`; `microscope.js` accepts real
  streamed genomes (`GENREG.scope.setGenome`) and pauses its demo while training.
- **Verified**: snake learns (random ≈0 apples → champion ~19–29 apples, best
  fitness monotonic); 2048 beats random 3–6×; `/train` end-to-end smoke passes
  (started→generation with genome+replay→done, plus stop); 10/10 edge configs;
  memory-rent shrinks H. See `genreg_train/REVISIONS.md`.

## 2026-06-30 (Claude Code / Opus 4.8)
- Layout persistence: the two sidebar widths and the terminal dock height now
  save to a cookie (`genreg_layout`) on drag-end and restore on page load
  (clamped to the existing min/max), so panel sizes survive a refresh.
  (`static/ui.js`.)

## 2026-06-30 (Claude Code / Opus 4.8)
- Added a **PO Metrics** tab beside Microscope in the right sidebar. It shows a
  hand-rolled 3D plot (canvas-2D projection, no 3D lib) of the EEC "constraint
  cone": with no constraints it's a sphere (infinite possibility); each checked
  Constraint adds a ring marching along the +x axis (leading right) and morphs
  the sphere toward a cone (tip = the few probable survivors). Rings are driven
  entirely by the Constraints checkboxes (add/remove on toggle); the readout
  lists which constraints map to which rings. Fixed 3/4 view (static — no
  auto-rotate); renders on demand (tab shown, resize, toggle). No further PO logic yet, as
  specified. New file `static/po_metrics.js` (also owns the right-sidebar tab
  switching); touched `templates/index.html` and `static/style.css`.

## 2026-06-30 (Claude Code / Opus 4.8)
- Condensed the Constraints section (CSS only, nothing removed): each axis group
  now lays its checkboxes out in two columns with tighter padding/gaps and a
  smaller font; lone constraints span the full width so long labels don't wrap.
  (`static/style.css`.)

## 2026-06-30 (Claude Code / Opus 4.8)
- Microscope: click a neuron (any row of a weight matrix, or its saturation bar)
  to add it to a **Tracked neurons** watch list that updates live each frame
  (per-neuron saturation + mean incoming |w|). Tracked neurons are marked on the
  canvas (accent tick + row outline); each list entry has an × to stop tracking.
  Watch list resets when a new genome is loaded via `setGenome`. Touched
  `static/microscope.js`, `templates/index.html`, `static/style.css`.

## 2026-06-30 (Claude Code / Opus 4.8)
- Added a **Microscope** view (right sidebar, replacing the Inspector
  placeholder): a magnified "lab" canvas that renders a genome's weight matrices
  as a heatmap (value → diverging blue/warm color), per-neuron saturation bars,
  and a hover readout showing an individual weight's value + that neuron's
  saturation. A live mutation loop (relative, self-adapting `w += N(0,1)·ms·
  (|w|+ε)`, ~7 steps/sec) animates it so you can watch the genome change;
  Pause/Play + Step controls and a generation badge. Currently ILLUSTRATIVE (synthetic
  16→24→12→4 genome); exposes `GENREG.scope.setGenome(layers)` to feed a real
  genome once training is wired. New file `static/microscope.js`; touched
  `templates/index.html` (markup + script include) and `static/style.css`.
- Wired the Environment selector to render a game board on the canvas. Selecting
  **Snake** draws an empty W×H playfield grid; **2048**
  draws the classic 4×4 tile board (static sample position). Other environments
  (CartPole / Humanoid-v5 / Language) show a "no game board" placeholder. Boards
  are illustrative only — not yet driven by the engine.
- Added a **Game Controls** section to the Control Panel (below Constraints).
  Snake exposes Board Width / Board Height inputs (5–60 cells) that resize the
  board live; 2048 shows a "fixed 4×4, no controls" note; other environments
  show an empty-state note. The visible sub-panel follows the selected
  Environment. Touched files: `templates/index.html`, `static/ui.js`
  (board renderers + control wiring), `static/style.css`.

## 2026-06-30 (Claude Code / Opus 4.8)
- Added the EEC and genreg-engine projects into `project/` (`project/EEC-main`,
  `project/genreg-engine-main`) — kept as two separate projects.
- Added a Control Panel to the left sidebar GUI (static UI, not wired to the
  engine yet). Fields map to EEC's model: **Environment** (2048 / Snake /
  CartPole / Humanoid-v5 / Language), **Population** (int 1–1000), and
  **Constraints** checkboxes. Touched files: `templates/index.html` (form
  markup, side-head renamed to "Control Panel") and `static/style.css`
  (control-panel styles).
- Expanded the Constraints checkboxes to the full EEC catalogue (`docs/
  CONSTRAINTS.md`) and grouped them by capability axis: Survival (Energy /
  Mortality), Parsimony (Time / Memory-Rent), Active maintenance (Entropy),
  Persistence/memory (Occlusion / Noise), Selection (Reproduction-Cost),
  Diversity (Scarcity), Attention (Perception-Cost), Plasticity
  (Non-Stationarity). Same-axis swap pairs (Survival, Observation) are marked
  with a "swap" tag. Removed the Efficiency checkbox (not a catalogue axis;
  overlapped Parsimony). Each box has a hover tooltip describing its mechanism.
  (`templates/index.html`, `static/style.css`.)

## 2026-06-30 (Claude Code / Opus 4.8)
- Fixed the terminal getting stuck on "connecting…" with no shell loading. The
  changelog-button wiring in `static/app.js` used unguarded
  `getElementById(...).addEventListener`, which threw when the loaded template
  did not contain the button (a stale cached HTML shell), aborting the whole
  script before the WebSocket `connect()` ran. Added null-guards so optional UI
  can never block the terminal from connecting.
- Added `Cache-Control: no-store` to the `/` route in `app.py` so the browser
  cannot serve a stale HTML shell against freshly-served JavaScript.
- Restarted the Flask web server to load the current template. The terminal
  daemon and all running terminals were left untouched (they survive Flask
  restarts by design).

## 2026-06-30 (Claude Code / Opus 4.8)
- Added a "Changelog" button to the terminal dock action bar (GUI). It opens a
  modal that fetches and displays this CHANGELOG.md. Touched files:
  `templates/index.html` (button + modal markup), `static/app.js` (open/close
  wiring + `/changelog` fetch), `static/style.css` (modal styles), and
  `app.py` (new `/changelog` route serving CHANGELOG.md as plain text).
- Added this CHANGELOG.md to track changes in the shared GENREG workspace.
