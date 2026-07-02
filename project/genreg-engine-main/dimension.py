"""
Piece 8: dimension evolution.

The hidden width H is not a hyperparameter you set — it is evolved. A genome can grow or shrink its hidden layer,
and every H-shaped array resizes together: W1 columns, b1, W2 rows, and the per-neuron precision genes.

Growth is FUNCTION-PRESERVING: a new neuron starts with random incoming weights but ZERO outgoing weight, so it
contributes nothing to the output until evolution decides to use it. Adding capacity therefore never disrupts a
working network — it only opens room. Shrinking does remove a neuron's contribution (it cannot be lossless), so
it is the riskier move, taken only when the saved cost outweighs the loss.

With H now variable, the `size_cost` from piece 6 finally bites: under a size cost the population is driven toward
the smallest network that still does the task.
"""
import numpy as np
from precision import PMAX

MIN_H, MAX_H = 2, 128            # MAX_H is a runaway-safety bound, far above what tasks need — not a design ceiling


def resize_H(g, new_H, rng):
    """Resize the hidden layer to new_H, resizing W1/b1/W2 (and prec, if present) consistently."""
    new_H = int(np.clip(new_H, MIN_H, MAX_H))
    if new_H == g.H:
        return
    old = g.H
    if new_H > old:
        add = new_H - old
        g.W1 = np.concatenate([g.W1, rng.normal(0, 1 / np.sqrt(g.n_in), (g.n_in, add)).astype(np.float32)], axis=1)
        g.b1 = np.concatenate([g.b1, np.zeros(add, np.float32)])
        g.W2 = np.concatenate([g.W2, np.zeros((add, g.n_out), np.float32)], axis=0)   # silent new neurons
        if hasattr(g, "prec"):
            g.prec = np.concatenate([g.prec, np.full(add, PMAX, np.int16)])
    else:
        g.W1 = g.W1[:, :new_H].copy(); g.b1 = g.b1[:new_H].copy(); g.W2 = g.W2[:new_H].copy()
        if hasattr(g, "prec"):
            g.prec = g.prec[:new_H].copy()
    if hasattr(g, "W_rec"):                               # recurrent matrix is H x H — keep the overlap, zero-pad new
        R = np.zeros((new_H, new_H), np.float32); k = min(old, new_H)
        R[:k, :k] = g.W_rec[:k, :k]; g.W_rec = R
    g.H = new_H


def mutate_dimensions(g, rng, prob=0.25, step=2):
    """Occasionally grow or shrink the hidden layer by up to `step` neurons (a per-child mutation hook)."""
    if rng.random() < prob:
        resize_H(g, g.H + int(rng.integers(-step, step + 1)), rng)


# --- self-test: run `python3 dimension.py` ---
if __name__ == "__main__":
    from genome import Genome
    from evolver import Evolver
    from constraints import shaped, size_cost
    from precision import init_precision

    g = Genome(3, 2, 8, np.random.default_rng(0)); init_precision(g)
    X = np.random.default_rng(1).normal(size=(6, 3)).astype(np.float32)
    y_before = g.forward(X).copy()

    # 1. growth is function-preserving (new neurons are silent) and all arrays stay consistent
    resize_H(g, 13, np.random.default_rng(2))
    assert g.H == 13 and g.W1.shape == (3, 13) and g.b1.shape == (13,) and g.W2.shape == (13, 2)
    assert g.prec.shape == (13,), "precision genes must resize with H"
    assert np.allclose(g.forward(X), y_before), "growth must preserve the function"

    # 2. shrink resizes consistently and still runs
    resize_H(g, 5, np.random.default_rng(3))
    assert g.H == 5 and g.W1.shape == (3, 5) and g.W2.shape == (5, 2) and g.prec.shape == (5,)
    assert g.forward(X).shape == (6, 2)

    # 3. bounds are respected
    resize_H(g, 9999, np.random.default_rng(4)); assert g.H == MAX_H
    resize_H(g, -5, np.random.default_rng(5)); assert g.H == MIN_H

    # 4. size cost now shapes the network: a size cost evolves a SMALLER hidden layer at equal task quality
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(64, 3)).astype(np.float32); Tt = (0.4 * Xt[:, 0:1] - 0.2 * Xt[:, 1:2]).astype(np.float32)
    def task(gen): return -float(((gen.forward(Xt) - Tt) ** 2).mean())
    free = Evolver(3, 1, task, pop=60, H0=20, seed=0, mutate_hook=mutate_dimensions).run(200)
    taxed = Evolver(3, 1, shaped(task, size_cost(0.01)), pop=60, H0=20, seed=0, mutate_hook=mutate_dimensions).run(200)
    assert task(taxed) > -0.03, f"taxed network must still solve the task (raw {task(taxed):.3f})"
    assert taxed.H < free.H, f"size cost should shrink the hidden layer ({taxed.H} vs {free.H})"

    print(f"dimension: growth function-preserving; arrays + precision resize; bounds OK; "
          f"size cost -> H {free.H} -> {taxed.H} at task fit {task(taxed):.3f}")
