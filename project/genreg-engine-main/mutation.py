"""
Piece 2: relative mutation.

The variation operator. Two ideas, both load-bearing:

  RELATIVE STEP   each weight is perturbed by  N(0,1) * scale * (|w| + EPS), applied to a fraction `rate` of the
                  weights. The step is proportional to the weight's OWN magnitude — big weights take big steps,
                  small weights take small steps. There is no single global step size. (The EPS floor lets a
                  weight sitting at exactly 0, e.g. a fresh bias, still move.)

  SELF-ADAPTATION the rate and scale are not constants — they are genes carried on the genome (one pair PER
                  LAYER) and they mutate too: `rate *= exp(TAU * N(0,1))`. So the search rate itself evolves;
                  lineages that happen to mutate at a useful rate are selected for.

The strategy genes (`mr`, `ms`) are attached lazily, so genome.py (piece 1) stays exactly as it was.
"""
import numpy as np

TAU = 0.20                       # self-adaptation strength for the rate/scale genes
EPS = 1e-3                       # floor so zero-valued weights can still move
RATE_BOUNDS = (0.02, 0.70)       # a layer mutates between 2% and 70% of its weights
SCALE_BOUNDS = (0.02, 0.80)


def _layers(g):
    """Group the genome's parameter arrays into layers of (weight, bias). One rate/scale gene per layer."""
    return [g.PARAMS[i:i + 2] for i in range(0, len(g.PARAMS), 2)]


def init_strategy(g, rng):
    """Give a genome its self-adaptive mutation genes (per-layer rate `mr` and scale `ms`)."""
    n = len(_layers(g))
    g.mr = rng.uniform(0.10, 0.40, n).astype(np.float32)
    g.ms = rng.uniform(0.10, 0.40, n).astype(np.float32)


def relative_step(W, rate, scale, rng):
    """The relative mutation rule, isolated so it can be tested on its own."""
    mask = rng.random(W.shape) < rate
    return (mask * rng.normal(0, 1, W.shape) * scale * (np.abs(W) + EPS)).astype(np.float32)


def mutate(g, rng):
    """Mutate a genome IN PLACE and return it. (Reproduction — making the copy first — is piece 3.)"""
    if not hasattr(g, "ms"):
        init_strategy(g, rng)
    # 1. self-adapt the strategy genes first
    g.mr = np.clip(g.mr * np.exp(TAU * rng.normal(size=g.mr.shape)), *RATE_BOUNDS).astype(np.float32)
    g.ms = np.clip(g.ms * np.exp(TAU * rng.normal(size=g.ms.shape)), *SCALE_BOUNDS).astype(np.float32)
    # 2. apply the relative rule per layer (weight and bias share the layer's rate/scale)
    for i, names in enumerate(_layers(g)):
        for name in names:
            getattr(g, name)[...] += relative_step(getattr(g, name), g.mr[i], g.ms[i], rng)
    return g


# --- self-test: run `python3 mutation.py` ---
if __name__ == "__main__":
    from genome import Genome

    # 1. THE relative property: step magnitude is proportional to |weight|
    W = np.linspace(-5, 5, 2000).reshape(-1, 1).astype(np.float32)
    r = np.random.default_rng(1); acc = np.zeros_like(W)
    for _ in range(400):
        acc += np.abs(relative_step(W, 1.0, 0.3, r))
    corr = float(np.corrcoef(np.abs(W).ravel(), (acc / 400).ravel())[0, 1])
    assert corr > 0.95, f"step should scale with |w| (corr={corr:.3f})"

    # 2. mutation changes the genome's function
    g = Genome(4, 2, 8, np.random.default_rng(2)); X = np.random.default_rng(3).normal(size=(6, 4)).astype(np.float32)
    y0 = g.forward(X).copy(); mutate(g, np.random.default_rng(4))
    assert not np.allclose(y0, g.forward(X)), "mutation must change the output"

    # 3. the strategy genes self-adapt
    ms0 = g.ms.copy(); mutate(g, np.random.default_rng(5))
    assert not np.array_equal(ms0, g.ms), "ms must self-adapt"

    # 4. reproducible: same genome + same seed -> identical mutation
    ga = Genome(4, 2, 8, np.random.default_rng(2)); gb = Genome(4, 2, 8, np.random.default_rng(2))
    mutate(ga, np.random.default_rng(9)); mutate(gb, np.random.default_rng(9))
    assert np.array_equal(ga.W1, gb.W1) and np.array_equal(ga.W2, gb.W2), "mutation must be reproducible"

    print(f"relative mutation: step proportional to |w| (corr {corr:.3f}); "
          f"changes output OK; self-adapts OK; reproducible OK")
