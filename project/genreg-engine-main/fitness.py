"""
Piece 5: robust fitness aggregation.

In a noisy world a genome's score varies run to run. If you select on a SINGLE evaluation, the population fills
with genomes that got one lucky run, not genomes that are reliably good — the "high score" stops meaning
anything (this is the lesson from the snake demo, where lucky single games inflated the leaderboard far above
what the genomes could repeat).

The fix is to evaluate each genome several times and aggregate with the MEDIAN, not the mean. The median ignores
rare lucky spikes; the mean is pulled up by them. `robust()` wraps any noisy `fitness_fn` into one the evolver
can use directly.
"""
import numpy as np


def robust(fitness_fn, n=5, agg="median"):
    """Return a fitness function that scores a genome as the median (default) of `n` independent evaluations.

    `fitness_fn(genome) -> float` must produce independent draws across calls (its own internal randomness, e.g.
    a freshly sampled environment). Use agg='mean' only if you specifically want the average.
    """
    reduce = {"median": np.median, "mean": np.mean}.get(agg)
    if reduce is None:
        raise ValueError("agg must be 'median' or 'mean'")

    def wrapped(genome):
        return float(reduce([fitness_fn(genome) for _ in range(n)]))
    return wrapped


# --- self-test: run `python3 fitness.py` ---
if __name__ == "__main__":
    from genome import Genome
    from evolver import Evolver

    noise = np.random.default_rng(1)

    # 1. variance reduction: median over 9 tracks the true value with far less noise than a single eval
    X = np.random.default_rng(0).normal(size=(32, 2)).astype(np.float32)
    T = (0.5 * X[:, :1]).astype(np.float32)
    g = Genome(2, 1, 8, np.random.default_rng(2))
    true = -float(((g.forward(X) - T) ** 2).mean())
    def noisy(gen): return -float(((gen.forward(X) - T) ** 2).mean()) + float(noise.normal(0, 0.3))
    single = np.array([noisy(g) for _ in range(500)])
    med9 = np.array([robust(noisy, 9)(g) for _ in range(500)])
    assert med9.std() < 0.6 * single.std(), (single.std(), med9.std())
    assert abs(med9.mean() - true) < 0.1, "median should track the true value"

    # 2. THE point: the median ignores lucky spikes that fool a single evaluation
    #    A: reliably good (true -0.10, light noise).   B: usually worse (-0.50) but spikes to +2 one time in ten.
    def fA(_): return -0.10 + float(noise.normal(0, 0.05))
    def fB(_): return -0.50 + (2.0 if noise.random() < 0.10 else 0.0)
    rA, rB = robust(fA, 9), robust(fB, 9)
    assert all(rA(g) > rB(g) for _ in range(100)), "median must rank the reliably-good genome above the spiky one"
    lucky_B = max(fB(g) for _ in range(100))
    assert lucky_B > fA(g), "a single lucky run of B beats A — exactly what the median protects against"

    # 3. plugs straight into the evolver: evolve under a noisy fitness
    ev = Evolver(2, 1, robust(noisy, 5), pop=50, H0=8, seed=0)
    best = ev.run(120)
    assert best.fit > true - 0.15, f"should reach near the true optimum under noise, got {best.fit:.3f}"

    print(f"fitness aggregation: median-of-9 std {med9.std():.3f} vs single {single.std():.3f}; "
          f"ranks reliable>spiky (B spikes to {lucky_B:.2f}); evolves under noise OK")
