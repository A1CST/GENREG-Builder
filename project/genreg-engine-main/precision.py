"""
Piece 7: evolvable precision.

Each hidden neuron carries a BIT-DEPTH gene (1..8 bits). Its weights are quantized to 2^bits levels before the
forward pass. Gradient descent cannot optimize this — bit-depth has no derivative — but evolution searches over
it directly, and a precision cost (piece 6 machinery) makes the genome spend bits only where they earn their
keep.

Design choices that keep the earlier pieces intact:
  * 8 bits is an exact no-op: a full-precision genome computes identically to the bare genome (piece 1).
  * gene arrays ride along through copy() automatically (reproduction.GENE_ATTRS includes "prec").
  * the bit-depth genes are mutated by the evolver's per-child hook (`mutate_hook=mutate_precision`), so the
    weight-mutation operator (piece 2) is untouched.
  * the quantized forward is `qforward(g, X)`; a task that wants precision simply calls it instead of g.forward.
"""
import numpy as np

PMIN, PMAX = 1, 8


def quantize(W, bits, axis):
    """Quantize each neuron's weights to its bit-depth (2^bits symmetric levels, per-neuron scale).
    axis=1: neurons are columns (incoming weights);  axis=0: neurons are rows (outgoing weights)."""
    if (bits >= PMAX).all():
        return W                                            # all full precision -> exact no-op
    scale = np.abs(W).max(axis=1 - axis, keepdims=True) + 1e-9
    levels = (2.0 ** bits).reshape((-1, 1) if axis == 0 else (1, -1))
    q = np.round((W / scale + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
    return (q * scale).astype(np.float32)


def init_precision(g, rng=None):
    g.prec = np.full(g.H, PMAX, np.int16)                   # start at full precision (a no-op until a cost bites)


def mutate_precision(g, rng, rate=0.20):
    """Nudge ~`rate` of the neurons' bit-depth by +-1. Lazily attaches the gene on first call."""
    if not hasattr(g, "prec"):
        init_precision(g)
    m = rng.random(g.H) < rate
    if m.any():
        g.prec = g.prec.copy()
        g.prec[m] = np.clip(g.prec[m] + rng.integers(-1, 2, int(m.sum())), PMIN, PMAX).astype(np.int16)


def qforward(g, X):
    """Forward pass with each neuron's weights quantized to its bit-depth (falls back to bare forward if no gene)."""
    if not hasattr(g, "prec"):
        return g.forward(X)
    X = np.asarray(X, np.float32)
    if X.ndim == 1:
        X = X[None, :]
    h = np.tanh(X @ quantize(g.W1, g.prec, 1) + g.b1)
    return h @ quantize(g.W2, g.prec, 0) + g.b2


def mean_bits(g):
    return float(g.prec.mean()) if hasattr(g, "prec") else float(PMAX)


def precision_cost(k):
    """Cost proportional to total bits held (plugs into constraints.shaped)."""
    return lambda g: k * (int(g.prec.sum()) if hasattr(g, "prec") else g.H * PMAX)


# --- self-test: run `python3 precision.py` ---
if __name__ == "__main__":
    from genome import Genome
    from reproduction import copy, crossover
    from evolver import Evolver
    from constraints import shaped

    g = Genome(3, 1, 10, np.random.default_rng(0)); X = np.random.default_rng(1).normal(size=(8, 3)).astype(np.float32)

    # 1. 8 bits is an exact no-op
    init_precision(g)
    assert np.array_equal(qforward(g, X), g.forward(X)), "full precision must equal the bare forward exactly"

    # 2. low precision really quantizes (changes the output)
    g.prec[:] = 2
    assert not np.allclose(qforward(g, X), g.forward(X)), "2-bit weights must change the output"

    # 3. precision genes ride through copy and crossover
    c = copy(g); assert hasattr(c, "prec") and c.prec is not g.prec and np.array_equal(c.prec, g.prec)
    h2 = Genome(3, 1, 10, np.random.default_rng(2)); init_precision(h2)
    assert hasattr(crossover(g, h2, np.random.default_rng(3)), "prec")

    # 4. under a precision cost the network sheds bits while still solving the task
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(64, 3)).astype(np.float32); Tt = (0.4 * Xt[:, 0:1] - 0.2 * Xt[:, 1:2]).astype(np.float32)
    def task(gen): return -float(((qforward(gen, Xt) - Tt) ** 2).mean())
    free = Evolver(3, 1, task, pop=60, H0=12, seed=0, mutate_hook=mutate_precision).run(160)
    taxed = Evolver(3, 1, shaped(task, precision_cost(0.01)), pop=60, H0=12, seed=0, mutate_hook=mutate_precision).run(160)
    assert task(taxed) > -0.03, f"taxed network must still solve the task (raw {task(taxed):.3f})"
    assert mean_bits(taxed) < mean_bits(free), f"precision cost should reduce bits ({mean_bits(taxed):.1f} vs {mean_bits(free):.1f})"

    print(f"precision: 8-bit no-op exact; low-bit quantizes; genes inherited; "
          f"cost -> mean bits {mean_bits(free):.1f} -> {mean_bits(taxed):.1f} at task fit {task(taxed):.3f}")
