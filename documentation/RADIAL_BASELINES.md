# Radial Space — Baseline Map Roadmap

**Date:** 2026-07-14
**Status:** In Progress
**Goal:** Establish baseline performance ceilings for the radial space + linear output head (no genomes) across all major data domains. These baselines define exactly where genome evolution needs to begin.

---

## Completed

### 1. Numerical Loops
- **Lenses:** 2,500 mapped (758 unique after dedup)
- **Tasks cleared:** square, abs, sin3x, ripple, sin8x, sin16x (all R² = 1.000)
- **Frontier:** sin32x (R² = 0.990, threshold ≥ 0.998)
- **Rotation probe:** best angle 126° → 0.9937, worst 31° → 0.1257
- **Shape:** thick disc / ellipsoid (axis spreads 5.1 / 3.8 / 2.0), not a sphere
- **Key finding:** radius = behavioral strangeness, not nonlinearity. Nonlinearity peaks in shell 3 (0.88), drops in outermost shell (0.43)
- **Key finding:** slice orientation through the disc determines lens diversity — flat-face slices = high R², thin-axis slices = dead zones

### 2. Image Baseline (MNIST) — COMPLETED 2026-07-13
- **Setup:** radial-owned copy (`radial_data/mnist_radial.npz`, 8k train / 2k
  test), flat 784 vector, POINTWISE lens application, closed-form kernel-ridge
  linear head (summed per-lens linear kernels), CUDA. `radial_baseline.py`.
- **Raw-pixel linear head:** 0.8005
- **Lens bank:** L=8 → 0.8270, L=32 → 0.8330, L=128 → 0.8385, **L=400 → 0.8395**
- **ANSWER — it is a COMPOUNDING problem, not coverage.** 8 lenses already buy
  +2.7 pts; the next 392 add only +1.2 more. Pointwise lenses + linear head is
  a GAM over pixels — no pixel INTERACTIONS, which is what digits need. More
  lenses will not move this; genomes (or structured/windowed input) will.
- **Per-class:** 0/1 easy (0.966/0.983); 5 (0.715) and 2 (0.749) hardest.
- **Top-aligned lenses:** gauss/abs/sq folds (ink detectors) — even-symmetric
  transforms dominate, e.g. `gauss(2.60x-0.66) . abs(2.16x-0.63) . abs(1.59x+0.98)`.
- **Rotation probe (360°, slice ~24):** best 0.8385 @ 178°, worst 0.8105 @ 43°,
  spread only 0.028; random-subset baseline 0.8373. A 24-lens slice already
  hits the full-bank ceiling — rotation neither helps nor hurts on MNIST
  because the GAM ceiling, not lens diversity, is binding.
- **Map shape (mnist stream):** axis std 6.82 / 1.67 / 1.41 — a much flatter
  cigar than loops (5.1/3.8/2.0). The domain visibly reshapes the map: spiky
  mostly-zero pixel data collapses many lens behaviors together.
- **Export:** `radial_data/baseline_mnist.json` (probe + rotation_probe + shape).

### 3. Image Baseline (CIFAR-10) — COMPLETED 2026-07-13
- **Raw-pixel linear head 0.3820; lens bank L=400 0.3815** — the pointwise bank
  adds NOTHING on CIFAR (slightly hurts at small L). Colour/texture/shape is
  entirely interaction-bound; per-value nonlinearity buys zero.
- **Per-channel vs flattened: mathematically identical here** — pointwise
  lenses transform each value independently, so how the vector is split is
  invisible to the model. The question only becomes real with windowed input.
- **Rotation probe:** best 0.4015 @ 105° — **a 24-lens slice BEATS both the
  full 400-lens bank (0.3815) and raw linear (0.3820)**; worst 0.3685, random
  0.3774. First domain where slicing outperforms the whole bank (extra lenses
  are noise dimensions here).
- **Map shape (cifar stream):** 4.83 / 3.71 / 2.09 — disc, like loops.
- **Ceiling vs MNIST:** the GAM gains +3.9 pts on MNIST, ±0.0 on CIFAR — the
  harder the interactions, the less pointwise diversity buys.
- **Export:** `radial_data/baseline_cifar.json`.

### 4. Text / Byte-Level Baseline — COMPLETED 2026-07-13
- **Setup:** wiki corpus (read-only), 50-char vocab, next-char from the CURRENT
  char id only (scalar, pointwise), primal ridge, 60k/15k random split.
- **Majority 0.1633 · raw-linear 0.1855 · lens bank L=128+ 0.2651 ·
  measured bigram-table ceiling 0.2661** — the lens bank + linear head lands
  **within 0.001 of the bigram table with no table**: the bank spans arbitrary
  functions of the current char, and evolution-free closed-form recovers it.
- **ANSWER: pointwise lens diversity captures NO sequential structure beyond
  bigram — it saturates exactly AT the bigram ceiling.** Context (temporal
  rotation / windows) is where anything above 0.266 must come from.
- **Minimum lens count:** 8 lenses already 0.2245; 128 saturate the ceiling.
- **Map shape (text stream):** 4.84 / 4.18 / 2.16. Rotation spread 0.0587.
- **Export:** `radial_data/baseline_text.json`.

### 5. Audio / Temporal Signal Baseline — COMPLETED 2026-07-13
- **Setup:** synthetic tone detection (10 frequencies, random phase/amplitude,
  noise), raw 256-sample waveform, pointwise bank + kernel head.
- **Raw linear 0.4105; lens bank 0.3725 — the bank HURTS on temporal data.**
  With random phase every sample position's value distribution is identical
  across classes; the linear head's above-chance score is a variance exploit,
  and pointwise lenses only dilute it.
- **ANSWER: yes — this is where temporal rotation becomes mandatory.** No
  pointwise transform can be phase-invariant; frequency lives BETWEEN samples.
- **Rotation probe:** the largest angular structure of any domain (spread
  0.1625; best slice 0.4025 @ 17° — again beats the full bank), because when
  the bank is mostly dead weight, WHICH lenses you take matters most.
- **Map shape (audio stream):** 5.12 / 3.54 / 1.95.
- **Export:** `radial_data/baseline_audio.json`.

### Pre-Baseline Fixes — ANSWERED 2026-07-13
- **Z-axis expansion (whitened MDS axes): does NOT help.** Whitened-y rotation
  spread 0.8282 vs plain 0.8285, same worst-case floor 0.1694. The dead zone
  is a property of which lenses are behaviorally redundant, not of axis scale.
- **Rotation axis lock: axis choice does not matter.** X/Y/Z spins all give
  spread ~0.82-0.83 and the SAME worst-case floor (0.1694) — the redundant
  clump is hit by every great circle. Free rotation is fine; no anchor needed
  at this stage. Export: `radial_data/prebaseline_fixes.json`.

---

## Needed

### 2. Image Baseline (MNIST) — done, see Completed above
- **Input format:** 28×28 grayscale pixels, flattened to 784-element vector (or structured — TBD)
- **Task:** 10-class digit classification
- **Method:** pass raw pixel data through full radial lens bank, train linear output head on lens-transformed features
- **Measure:** classification accuracy + per-class breakdown
- **Questions to answer:**
  - Does the current lens bank produce separable features for digit classes?
  - Which lenses contribute most to class separation?
  - Where does the ceiling hit — is it a coverage problem (more lenses needed) or a compounding problem (genomes needed)?
  - Does rotation improve or degrade image classification vs. loops?

### 3. Image Baseline (CIFAR-10)
- **Input format:** 32×32×3 RGB, flattened to 3,072-element vector (or per-channel — TBD)
- **Task:** 10-class object classification
- **Method:** same as MNIST — full lens bank + linear head
- **Measure:** classification accuracy
- **Questions to answer:**
  - How much harder is color + texture + shape vs. grayscale digits?
  - Does per-channel lens application (R, G, B separately) outperform flattened?
  - Where is the ceiling relative to MNIST?

### 4. Text / Byte-Level Baseline
- **Input format:** raw byte sequences or character-level encoding
- **Task:** next-byte or next-character prediction (or simple classification)
- **Method:** lens bank applied to byte/char vectors, linear output head
- **Measure:** accuracy or perplexity
- **Questions to answer:**
  - Can lens diversity capture sequential structure at all without temporal rotation?
  - Is this where temporal rotation of the radial axis becomes mandatory?
  - What's the minimum lens count for non-trivial text performance?

### 5. Audio / Temporal Signal Baseline
- **Input format:** raw waveform samples or spectrogram slices
- **Task:** simple classification (speech command, tone detection) or signal reconstruction
- **Method:** lens bank on temporal samples, linear output head
- **Measure:** classification accuracy or reconstruction R²
- **Questions to answer:**
  - Does temporal data naturally benefit from radial rotation (ground truth is already changing)?
  - Do new branches appear in the radial map when audio activations are added (as seen in the original activation map post)?
  - Is this where the 4D (space + time) radial array becomes necessary?

---

## Pre-Baseline Fixes

### Z-Axis Expansion
- Current Z spread is only 2.0 vs X at 5.1 — the disc is flat
- Investigate whether expanding Z diversity reduces rotation probe dead zones
- May push past sin32x without genomes

### Input Format Decision
- Verify: vectorized (one flat array) vs. structured (per-channel, per-position)
- Current loops baseline uses pointwise application on flat 1D array
- Images and text may need per-channel or per-position lens application
- **This decision affects all downstream baselines**

### Rotation Axis Lock Question
- **Open question:** does rotation need one axis locked to ground truth, or can all three spatial axes rotate freely?
- Does time as a 4th dimension serve as the anchor instead?
- Test single-axis rotation first, then multi-axis

---

## Success Criteria

Each baseline is complete when:
1. Full lens bank R² or accuracy is measured on the task
2. Rotation probe is run (360° sweep, per-task breakdown)
3. The ceiling / frontier task is identified (where performance drops below threshold)
4. The radial map shape is documented for that data domain
5. The result is logged as a JSON export with probe + rotation_probe data

Once all baselines are locked, genome evolution targets are defined by the gaps.
