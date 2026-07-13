# Radial v1 — ARCHIVED 2026-07-13

The first radial-space line, archived intact (nothing deleted) when the project
was rebuilt much simpler as the **activation-behavior map** (`radial_map.py`,
the new `/radial` page). Full history in the top-level `CHANGELOG.md`
(2026-07-12/13 entries).

## What lives here

| File | What it was | Verdict |
|------|-------------|---------|
| `radial_space.py` | Rotation-lattice signal codec from `radial_space_theory.pdf` (address a signal to a dot, 3-float compression) | Worked as described, but compression is dominated by standard methods |
| `radial_dict.py` | RS-Gabor matching pursuit vs top-K Fourier head-to-head | Fair fight built; advantage vanishes on stationary signals |
| `radial_memory.py` | v2 §5 memory + §6 computation suites, traversal paths | Paper's mapping fails proximity; a proximity-preserving mapping fixes it |
| `radial_mfunc.py` | v3 Test Suite 10 mapping-function characterization | Built + passed |
| `radial_screen.py` | v3 §11.1 real screen-capture fingerprinting (record/train/classify) | Worked as a demo |
| `radial_lens.py` | The lens-flip idea: deterministic activation-algebra lenses over feature axes | MNIST negative (PCA 0.92 unbeatable); CIFAR positive (+2.9 to +5.3 pts over equal-dim PCA, label-free) |
| `radial_lensmap.py` | Infinite/open-ended live lens-map explorer with residual layer stacking (the `/radial` "Lens Map" mode) | Depth-1 is the heavy lift (+0.071 over PCA), plateaus by depth-3 |
| `radial_cifar.py` | No-gradient CIFAR-10 push: Coates-Ng patch features + closed-form ridge, lens expansion on top | **0.5904 on full 50k/10k** (patch D=512 pool2 + ridge); milestones 1+2 committed on branch `radial-cifar-nogradient` |
| `radial_compress_bench.py` | Compression benchmark harness | See radial_dict verdict |
| `radial.html`, `radial.js` | The old six-mode `/radial` page (Codec / vs Fourier / Memory v2 / Mapping M / Screen / Lens Map) | Replaced by the v2 map page |

`radial_data/` (the radial-owned CIFAR copies + patch-feature caches) was left
in place at the repo root — the v2 line reuses it when it graduates from
numbers to images.

## Why archived

The line had grown six modes and five theory modules deep. The v2 rebuild goes
back to the core idea with the user's activation-map research (galaxy of
evolved activation functions, characterized by BEHAVIOR on real data, not
formula): deterministic index-addressed lens programs, behavioral signatures
on simple numerical loop data first (a baseline map per data type), and a
plain linear model on top to test whether lens diversity alone does the heavy
lifting. Images/text baselines come after the numeric baseline is understood.

These modules still run if imported from this folder (they cross-import each
other by name), but they are not wired to the app anymore.
