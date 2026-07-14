# Radial Space — What We Have Discovered

**Date:** 2026-07-13 · **Branch:** `radial-cifar-nogradient` · **Code:** `radial_map.py`, `radial_baseline.py` · **PDF:** `RADIAL_SPACE_FINDINGS.pdf`

The radial space v2 line, rebuilt from the activation-map insight: characterize
activation functions by how they **transform data** (behavior, not formula),
give every function a **deterministic integer address**, and the space
self-organizes into a navigable map. Everything below is measured, no
gradients anywhere, all classifiers closed-form.

---

## 1. The space itself

- **A lens is an address.** Lens *i*'s program (1–3 composed primitives
  `prim(a·x+b)` from a 14-op catalog) is generated from the integer *i*. Same
  index, same lens, forever — determinism error is exactly **0.0** in every
  domain tested. The entire infinite space is ~40 lines of generator code
  (~2 KB); a lens bank is a list of integers.
- **The map is behavioral.** Each lens is fingerprinted by its response to a
  data stream (response curve over the data's own quantile grid + 7 behavior
  stats); classical MDS of signature distances gives the 3D map, centred on
  the identity lens (origin, the red dot).

## 2. The map's structure (the "galaxy" holds)

Measured on the 2500-lens loops map:

- **Linear core.** Near-linear lenses (nl < 0.1) sit at mean radius 2.9 vs
  population 7.3. The innermost quartile shell has mean nonlinearity 0.055.
- **Families occupy distinct regions**, ordered by how gently they bend data:
  `id / logp / soft / tanh` (radius 4.3–5.2) → `sign / step / relu` (5.3–5.7)
  → `sin` (7.2) → `sq / abs / cos / gauss / sqrt / sinc` (9.2–10.3, the
  sign-destroying folds). Oscillators lean into their own lobe (centroid
  clearly displaced, radius ~9.6).
- **CORRECTION to the naive reading: radius = behavioral strangeness, not
  nonlinearity.** Nonlinearity peaks in the third shell (0.88) and *drops* in
  the outermost (0.43) — the far dots are the weirdest lenses (near-constant,
  extreme folds), not the most nonlinear. Radius–nonlinearity correlation is
  ~0.5, not ~1.
- **It is a disc, not a sphere.** Loops axis spread 5.1 / 3.8 / 2.0. Every
  domain reshapes it: MNIST collapses it to a flat cigar (6.8 / 1.7 / 1.4);
  CIFAR 4.8 / 3.7 / 2.1; text 4.8 / 4.2 / 2.2; audio 5.1 / 3.5 / 2.0. The
  map's shape is a property of the data domain — a per-domain baseline map is
  meaningful, exactly as conjectured.

## 3. Lens diversity makes hard targets linear (the founding hypothesis)

A plain closed-form ridge on the lens bank vs on raw input (loops stream):
square, abs, sin3x, step, ripple — raw linear R² ≈ 0 on the nonlinear ones;
**lens bank R² = 1.000 on all five** (heldout, also on noise). Verified
bit-identical on re-run; rotation sweeps mutate no state.

**Task ladder (auto-advance at R² ≥ 0.998):** square → abs → sin3x → ripple →
sin8x → sin16x all pass; **frontier = sin32x (0.990)** on loops, sin16x
(0.949) on noise. The current wall is *frequency resolution* (smooth lens
curves run out of wiggle), and it is data-dependent. Temporal targets are
explicitly out of scope for now.

## 4. Rotation: the map's geometry is real information

Embed the map in 3D, spin it about the Y axis 1°/step, and per angle give the
linear head only the slice of lenses in the viewing plane:

- **Loops (~23-lens slices):** best angle 81° → R² 0.998; worst 5° → 0.169;
  random same-size subsets → 0.996. Bad angles are far *below* chance:
  **co-planar lenses are behaviorally similar, i.e. redundant views.** A
  meaningless layout would make every slice look random — this one doesn't.
- **The good angle is scale-stable** (~81–82° at both 800 and 2500 lenses):
  a reusable property of the map, which is what makes it a *map*.
- **Pre-baseline fixes, both answered:** spin axis does not matter (x/y/z all
  spread ~0.83 and hit the *same* worst-case floor 0.169 — the redundant clump
  lies on every great circle; free rotation, no anchor needed), and whitening
  the flat axis changes nothing (dead zones are behavioral redundancy, not
  geometry).

## 5. The five domain baselines (pointwise bank + linear head, no genomes)

| Domain | Raw linear | Lens bank | Bank buys | Best rotation slice |
|---|---|---|---|---|
| Loops (R²) | ~0 | 1.000 | everything | 0.998 |
| MNIST | 0.8005 | 0.8395 | +3.9 pts | 0.8385 |
| CIFAR-10 | 0.3820 | 0.3815 | **0.0** | **0.4015 — beats bank AND raw** |
| Text next-char | 0.1855 | 0.2651 | +8.0 pts | 0.242 |
| Audio tones | 0.4105 | 0.3725 | **−3.8 pts** | 0.4025 — beats bank |

- **MNIST:** saturates by L≈8–32 → a compounding problem, not coverage. The
  useful lenses are gauss/abs folds (ink detectors). Hardest classes: 5, 2.
- **CIFAR:** pointwise nonlinearity buys literally nothing (interaction-bound
  domain), but a 24-lens slice at 105° beats the whole 400-lens bank — extra
  lenses are pure noise dimensions here.
- **Text:** the bank lands **within 0.001 of the measured bigram-table ceiling
  (0.2661) with no table** — pointwise lens diversity spans arbitrary
  functions of the current character, and saturates exactly AT that ceiling.
  Zero sequential structure is reachable pointwise.
- **Audio:** the bank *hurts*. Random phase makes every sample position's
  value distribution class-identical; no pointwise transform can be
  phase-invariant. **Temporal rotation is confirmed mandatory.** Audio also
  shows the largest angular structure (spread 0.16) — when the bank is mostly
  dead weight, WHICH slice you take matters most.

**Cross-domain law:** the less pointwise diversity buys (MNIST +3.9 → CIFAR
0.0 → audio −3.8), the more the *slice* beats the *bank*.

## 6. Model size: the space is free

The lens space is generated, never stored (~2 KB of code + integer indices).
Deployable model size is entirely the linear head:

- Loops regressor: **~1.6 KB** (R² 1.000)
- Text next-char at the bigram ceiling: **~26 KB at L=128** (the bigram table
  it replaces is 10 KB — we pay ~2.5× to have *no table*)
- MNIST: **~1 MB at L=32** (0.833) / 12.5 MB at L=400 (0.8395)
- CIFAR's rational model is the 24-lens slice: ~3 MB, of which 24 *bytes* are
  the lens identities

## 7. Where genomes must begin (the gaps, now quantified)

1. **Pixel interactions** (images): pointwise ceiling measured at +3.9 pts
   MNIST / +0.0 CIFAR. A genome must *compose* lenses across positions.
2. **Context beyond bigram** (text): everything above 0.2661 requires reading
   more than the current character — windows or temporal rotation.
3. **Phase-invariant temporal structure** (audio): frequency lives *between*
   samples; the roadmap's 4D (space + time) radial array starts here.
4. **Slice selection is already a model-design lever:** on hard domains the
   right 24 lenses beat 400 — an evolved slice-picker is a tiny genome with
   measured headroom (+2.0 pts CIFAR over the full bank).

## 8. The four open questions — ANSWERED (2026-07-13, round 2)

**Q1 — Why is the map a disc?** Measured: axis 1 correlates **+0.99 with
linearity** — the dominant axis literally *is* the linear↔nonlinear
direction (monotonicity +0.60 rides along). Axis 2 is response-curve *shape*
(no scalar stat explains it; the curve part of the signature carries 79% of
its variance in 2 PCs). Axis 3 is a weak oscillation/zero-cross mix. The
disc exists because lens behavior has ~2 dominant degrees of freedom —
how linear, and what shape of bend — and everything else is minor.

**Q2 — What is the dead-zone clump?** **It is the linear core itself.** The
worst slice = 23 lenses, all outer-op `id`, nonlinearity 0.000, intra-clump
signature distance 0.34 vs 9.10 for random sets — 23 copies of `x`. That is
why the floor (0.1694) exactly equals raw-linear performance, and why every
spin axis hit the *same* floor: every great circle passes through the origin.
**And it is removable:** excluding the 126 near-linear lenses (nl < 0.05)
raises the rotation floor from 0.169 to **0.822** and collapses the spread
from 0.83 to 0.18. Practical rule: never build a slice out of the core.

**Q3 — Does temporal rotation work?** **Yes — confirmed.** Rotating a lens's
view through time (features = agreement between the lens's view at t and at
t+δ, δ = 1..16 — lens-space autocorrelation; deterministic, closed-form,
phase-invariant by construction) takes the audio tone task from 0.41 (raw)
/ 0.37 (pointwise bank) to **1.0000 with 16 features**. Honest caveat: the
identity lens alone already saturates this task, so the *mechanism* is
proven but lens *diversity* in the temporal regime needs a harder task to
show its value.

**Q4 — The genome bridge.** First radial genome built and run
(`radial_slice_ga.py`, batched GPU fitness: one (POP,N,N) LU per generation,
305 s total): a 24-lens slice picker evolved on CIFAR (soft log-softmax
fitness on a held-back validation split, tournament + elitism + energy
homeostasis; the classifier stays closed-form — evolution only picks the
views). **Result: 0.376 — random-subset level (0.3774), well below the
map-geometry slice (0.4015),** with the classic flat-fitness signature
(soft −2.074 → −2.058 over 60 gens, starved 0). The honest reading: the
index-space landscape is flat by nature (random 24-subsets differ by ±0.002),
so naive evolution over raw lens indices cannot see the structure that the
map's geometry exposes for free. **The genome's search space should be the
MAP (plane/angle/region parameters), not raw indices** — which is the radial
thesis restated as an evolution recipe, and the design brief for genome #2.

## 9. Response to the independent validation report (2026-07-14)

`documentation/validation_report.txt` audits the `/radial/demo/cousins`
sub-page (a separate workstream: a small periodic 140-lens generator on a
grid, probed with 128 samples) and invalidates 5 of its 10 claims — mainly
overparameterized least squares (features > samples ⇒ R²=1 by memorization)
and modulo-duplicate lenses masquerading as spatial structure. Those
critiques are correct **for that demo**. We ran the same attacks against the
main line (`radial_map.py`, 748-lens bank, 2400-sample stream):

- **Overparameterization: does not apply.** Our ridge fits 1440 rows against
  ≤749 features and scores held-out; with the bank cut to **100 features**
  (7% of sample count) heldout R² is still 1.000 on all five tasks. The
  correct framing stands as written in §3: pointwise 1-D targets are easy
  function interpolation for a diverse bank — real, but not deep abstraction.
- **Periodicity/duplicates: does not apply.** Our generator is rng(i)-seeded,
  no modulo cycle; behavioral duplicate pairs (|corr|>0.99) are 1.17%, and
  after deduplicating at 0.95 the 171 survivors still hit R² 1.000 on
  everything.
- **Rotation-is-a-red-herring: already resolved by Q2.** Our worst-slice
  effect is driven by behavioral redundancy (the linear core), which we
  identified and removed ourselves — the below-chance floor is explained, not
  mystical.
- **Effective dimensionality: the critique LANDS, and sharpens §1.** At the
  1% singular-value threshold the 748-lens bank spans only **~46 effective
  dimensions** on the loops stream. The "infinite" lens space collapses to a
  few dozen usable directions per domain — consistent with Q1's two dominant
  DOF, with the early saturation of every accuracy-vs-L curve, and with the
  sin32x frontier (high-frequency targets need directions the bank doesn't
  span). The map is best understood as ~46 real axes plus dense redundancy —
  which is exactly why slice CHOICE beats slice COUNT.

## 10. CORRECTION (2026-07-14): the "slice beats bank" headline was inflated

Self-audit triggered by Phase A of the evolution campaign: the baseline
rotation probes **selected the best angle on the test set** (max of 360
test measurements, each with ±0.011 binomial noise on a ~0.377 field —
winner's-curse inflation ≈ +0.03). Re-run under the honest protocol
(select the angle on a held-back validation split, measure test once):

- **CIFAR: val-selected slice = 0.3845** (18 lenses) — still ≈ the 400-lens
  bank (0.3815) with 6% of the lenses, and still above random subsets
  (0.3774), but the published 0.4015 does **not** replicate; treat every
  test-selected "best slice" number (CIFAR 0.4015, audio 0.4025) as upper
  bounds, not estimates.
- **What survives untouched:** the *below*-random dead zones (structural —
  the linear core, §Q2), the loops rotation spread (0.83, floor explained
  mechanistically), and the redundancy analysis. The angular structure on
  CIFAR is real but thin: val range 0.263–0.326, all of the width on the
  DOWN side.
- **Consequences for the genome program:** slice selection on CIFAR is worth
  ~+0.007 over random, not +0.024 — index-GA (0.376), map-region GA (0.370)
  and honest slices (0.3845) all live inside a ±0.008 window. The pointwise
  ceiling is even harder than reported; interactions (Phase B) are the only
  real headroom.

## 11. Evolution campaign log (autonomous, CIFAR)

- **Phase A — region genome (evolve ON the map):** genome = (plane angles,
  offset, size) over the 3D map; batched GPU fitness; random-genome control
  at equal budget. Result: evolved 0.3695 test ≈ random control (val 0.298 vs
  0.300) — after the §10 correction this is the expected outcome: there is
  almost nothing to find. Engineering note: all earlier "CUDA wedge"
  slowdowns were a single zombie GA process saturating the GPU for hours;
  once killed, Phase A runs in **42 s**. (`radial_data/evo_region_cifar.json`)

- **Phase B — interaction feature genomes: THE CEILING BREAKS.** Environment
  = label-free patch-PCA component maps (40 comps of 6×6×3 patches, built
  once — "the features are the environment"). A feature genome = two
  components, two evolved lens bends, a combine op (mult/|diff|/min), a pool
  (mean/max) → ONE scalar per image. Evolution (pop 64, soft residual-gain
  fitness on val, tournament + elitism + energy) breeds features; each round
  the top ≤8 decorrelated survivors are FROZEN and composed; the head stays
  closed-form ridge. Results on 8k train / 2k test, test measured ONCE:
  - 24 features (4 s): **0.4030** — already past the pointwise ceiling 0.3845
  - 317 features (81 s): **0.5565** — past the hand-crafted Coates-Ng bar
    (0.493 at this data size) by +6.4 pts
  - 647 features (231 s, converged naturally — a round froze nothing):
    **0.5840**, within 0.006 of the v1 FULL-50k Coates-Ng milestone (0.5904)
    on 6× less data. Model size ≈ **52 KB** (647 genomes × ~10 numbers +
    the ridge head). No gradients anywhere.
  (`radial_data/evo_interact_cifar.json`)

- **Phase B-full — transfer to full CIFAR: THE MILESTONE FALLS.** The 647
  genomes evolved on 8k, recomputed on 50k/10k with the head refit (35 s):
  **test 0.6198 — beats the v1 hand-crafted Coates-Ng milestone (0.5904,
  full 50k, 2048-dim features) by +2.9 pts** with 647 evolved scalar
  features. The branch's original goal line now reads: raw 0.324 → PCA
  0.360 → hand-crafted patches 0.5904 → **evolved genomes 0.6198**, all
  gradient-free. Genomes transfer across data scale without re-evolution.
  (`radial_data/evo_full_cifar.json`)

- **Phase D — stacked radial spaces (the plateau safety mechanism) + the
  downstream energy economy.** When stage-1 hard-converged, its outputs were
  handed to a NEW radial space as data. The plateau broke: 8k stacked test
  0.5945 (from 0.5840), full-50k **0.6353**; stage-2 alone re-expresses
  0.5285 of stage-1 from outputs only. Then the user's energy economy for
  all downstream spaces (existing costs, outputting costs, small restore for
  any output, real energy only from above-median contribution to the right
  answer through the composition; steady-state population so the existence
  clock actually ticks — 7.5% starved/gen, in the 3–15% band): **96% of the
  stacking lift with 50 genomes instead of 411** (full-50k 0.6283) — energy
  is a compression pressure, buying efficiency rather than peak. Two
  falsified designs on the way: absolute contribution never starves anyone
  (adding any column helps ridge a little — the bar must be RELATIVE), and a
  comma-style GA never lets the existence clock tick (the population must
  PERSIST). Exports `evo_stack_cifar.json`, `evo_stack_full_cifar.json`.

- **Phase C — the genome map.** The 647 evolved genomes fingerprinted by
  behavior (feature values over 512 probe images), MDS to 3D
  (`radial_data/evo_genome_map.json`):
  - **Effective dimensionality 405 of 647** — vs the enumerated lens bank's
    46 of 748. This is WHY Phase B works: freeze-decorrelate pressure
    manufactures genuinely new behavioral directions ~9× more efficiently
    than enumeration. The population is near-SPHERICAL (8.3/7.4/6.6), not a
    disc — evolved features fill behavior space isotropically.
  - Honest negatives: discovery does NOT expand radially over time
    (corr −0.15 — the activation-galaxy expansion pattern does not reproduce
    in the residual-boosting regime), and op families barely cluster
    (intra 13.76 vs inter 14.03) — behavior is set by the component pair,
    not the op label.

All numbers reproducible from `radial_data/baseline_*.json`,
`prebaseline_fixes.json`, and the CLI (`python radial_map.py map|probe|rotate|
ladder`, `python radial_baseline.py all`). Roadmap status lives in
`RADIAL_BASELINES.md`; the archived v1 line in `archive/radial_v1/`.
