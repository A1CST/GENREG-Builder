"""
Piece 10: telemetry.

What to watch while a run evolves. The single most important number is NOT the best genome's fitness — a lucky
best can spike and mislead. The honest "is the population actually improving" signal is the MEAN and MEDIAN
fitness across the whole population. `snapshot()` captures those plus aggregate gene statistics (hidden width,
precision, leak), and the evolver appends one per generation to `ev.history`.

    ev = Evolver(..., telemetry=snapshot)
    ev.run(200)
    series(ev.history, "mean")     # the population-mean trajectory
"""
import numpy as np
from precision import mean_bits


def snapshot(ranked):
    """ranked: the evaluated population, best first (each genome has .fit). Returns a stats dict."""
    fits = np.array([g.fit for g in ranked], dtype=float)
    leaks = [float(g.leak) for g in ranked if hasattr(g, "leak")]
    return dict(
        best=float(fits[0]),
        mean=float(fits.mean()),                       # the real progress signal
        median=float(np.median(fits)),
        worst=float(fits.min()),
        mean_H=float(np.mean([g.H for g in ranked])),
        mean_bits=float(np.mean([mean_bits(g) for g in ranked])),
        mean_leak=(float(np.mean(leaks)) if leaks else None),
    )


def series(history, key):
    """Pull one telemetry channel out as a numpy array (e.g. the mean-fitness curve)."""
    return np.array([h[key] for h in history], dtype=float)


# --- self-test: run `python3 telemetry.py` ---
if __name__ == "__main__":
    from genome import Genome
    from evolver import Evolver
    from precision import mutate_precision

    rng = np.random.default_rng(0)
    X = rng.normal(size=(48, 2)).astype(np.float32)
    T = (0.5 * X[:, 0:1] - 0.3 * X[:, 1:2]).astype(np.float32)
    def task(g): return -float(((g.forward(X) - T) ** 2).mean())

    # 1. history is recorded, one snapshot per generation, with the expected channels
    ev = Evolver(2, 1, task, pop=50, H0=8, seed=0, telemetry=snapshot)
    ev.run(80)
    assert len(ev.history) == 80 and ev.history[-1]["gen"] == 79
    for k in ("best", "mean", "median", "worst", "mean_H", "mean_bits"):
        assert k in ev.history[0], k

    # 2. the population MEAN climbs (and stays below best) — the honest progress signal
    mean_curve = series(ev.history, "mean")
    assert mean_curve[-1] > mean_curve[0] + 0.05, "population mean should improve"
    assert series(ev.history, "best")[-1] >= mean_curve[-1] - 1e-9, "best is at least the mean"

    # 3. gene channels track real state: with a precision hook, mean_bits is recorded and < full
    ev2 = Evolver(2, 1, task, pop=50, H0=8, seed=0, telemetry=snapshot, mutate_hook=mutate_precision)
    ev2.run(120)
    assert series(ev2.history, "mean_bits")[-1] < 8.0, "precision telemetry should show bits below full"

    print(f"telemetry: {len(ev.history)} snapshots; mean fitness {mean_curve[0]:.3f} -> {mean_curve[-1]:.3f} "
          f"(best {series(ev.history,'best')[-1]:.3f}); precision-run mean bits {series(ev2.history,'mean_bits')[-1]:.1f}")
