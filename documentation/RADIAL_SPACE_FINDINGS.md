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

All numbers reproducible from `radial_data/baseline_*.json`,
`prebaseline_fixes.json`, and the CLI (`python radial_map.py map|probe|rotate|
ladder`, `python radial_baseline.py all`). Roadmap status lives in
`RADIAL_BASELINES.md`; the archived v1 line in `archive/radial_v1/`.
