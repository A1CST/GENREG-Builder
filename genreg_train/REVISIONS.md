# Revision log — training integration

Five hardening passes after the feature worked end-to-end. Each pass = findings +
fixes + the test that verified it.

## Pass 1 — correctness & evaluation fairness
- **Finding:** genomes were scored on *independent* random episodes (a shared rng
  advanced per call), so selection compared genomes on different challenges —
  noisy, and best-fitness was non-monotonic.
- **Fix:** rewrote `trainer.py` to evaluate every genome in a generation on the
  **same** episode seeds (median over 4). Champion replay uses a *fresh* unseen
  seed so the board shows genuine generalization, not a memorized game.
- **Verified:** headless snake — best fitness now climbs monotonically 1.2 → 19.9
  over 40 gens (pop 120); champion eats up to ~19 apples on fresh episodes.
  Random baseline ≈ 0 apples.

## Pass 2 — robustness / edge cases
- **Checked:** tiny pop (1 → clamped to 4), 5×5 board, all 12 constraints at once,
  mortality, occlusion+noise, 2048+memory-rent, generations=1, invalid environment
  (falls back to snake), per-genome eval wrapped in try/except (a broken genome
  scores −1e9 instead of crashing the run), NaN/inf guard on fitness, WS-disconnect
  stops the thread, double-start cancels the previous run, stop before gen 1.
- **Verified:** `edge_test.py` — 10/10 configs complete (started→generation→done,
  zero errors); same-axis swap notes emitted for the all-constraints case.

## Pass 3 — EEC faithfulness
- **Checked:** constraints are costs/laws, not reward shaping (base fitness is
  apples + a small survival term = world consequence; no distance-to-food
  gradient). Same-axis pairs handled ("swap, don't stack": energy>mortality,
  occlusion/noise flagged redundant).
- **Verified:** `bite_test.py` — memory-rent (size cost + dimension mutation)
  shrinks mean H 40 → 37.9 (champ 38); with no cost H stays exactly 40. The world
  shapes the model.

## Pass 4 — performance / bandwidth
- **Finding:** 2048 was the slow path — `_apply` over 4 directions each step, plus
  `valid_actions()` re-checks; ~9.5 s/gen at pop 120.
- **Fix:** memoized `Game2048Env._slide_row` on row contents (returns a fresh list
  so cached rows can't be mutated through the grid). Genome weights rounded to 4dp;
  replay frames downsampled to ≤240.
- **Verified:** 2048 ~3.1 s/gen at pop 60 (≈1.5× faster) and still learns
  (best 2354 → 4860 in 8 gens). Snake ≈1.2 s/gen at pop 120.

## Pass 5 — UX / polish
- Live board clears on manual environment change (no stale replay sticking).
- Microscope pauses its self-mutation demo while training streams real genomes
  (`setExternal`), keeps the watch list across generations when shape is unchanged.
- Training controls: Start disabled while running, Stop enabled; status HUD shows
  `gen k/N · best · score · base`. Constraint-parameter rows reveal only for the
  checked constraints.
- Docs: CHANGELOG updated; this file; SKELETON.md is the design spec.

## Known limitations (honest notes)
- 2048 at large pop×gens is slow (CPU neuroevolution; the engine README says so).
- Constrained snake needs more generations to reach high apple counts — expected,
  constraints are costs. Energy-only reaches ~7 apples in 40 gens.
- cartpole / humanoid / language remain non-trained placeholders (out of scope).
- One training job at a time per server (a new Start cancels the previous).
