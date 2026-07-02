"""
Piece 11: checkpoints.

Save a genome to disk and load it back, exactly — weights and every gene (mutation strategy, precision,
recurrence). Use it to keep a champion, resume a run, or run inference later.

The on-disk form is a plain dict (weights + whichever genes are present), written atomically (temp file then
rename) so a reader never sees a half-written file.
"""
import os
import pickle
from genome import Genome
from reproduction import GENE_ATTRS


def state(g):
    """Everything needed to rebuild this exact genome."""
    d = dict(n_in=g.n_in, n_out=g.n_out, H=g.H)
    for p in g.PARAMS:
        d[p] = getattr(g, p)
    for a in GENE_ATTRS:
        if hasattr(g, a):
            d[a] = getattr(g, a)
    return d


def from_state(d):
    g = Genome.__new__(Genome)
    g.n_in, g.n_out, g.H = d["n_in"], d["n_out"], d["H"]
    for p in Genome.PARAMS:
        setattr(g, p, d[p])
    for a in GENE_ATTRS:
        if a in d:
            setattr(g, a, d[a])
    return g


def save(g, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state(g), f)
    os.replace(tmp, path)                                   # atomic


def load(path):
    with open(path, "rb") as f:
        return from_state(pickle.load(f))


# --- self-test: run `python3 checkpoint.py` ---
if __name__ == "__main__":
    import numpy as np
    from precision import init_precision, qforward, mutate_precision
    from recurrence import enable, fresh_state, rstep
    from evolver import Evolver

    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ckpt_test.pkl")

    # 1. plain genome round-trips exactly
    g = Genome(3, 2, 9, np.random.default_rng(0))
    X = np.random.default_rng(1).normal(size=(7, 3)).astype(np.float32)
    save(g, p); g2 = load(p)
    assert np.array_equal(g.forward(X), g2.forward(X)), "forward must match after load"

    # 2. all genes survive: precision + recurrence
    enable(g, np.random.default_rng(2)); init_precision(g); g.prec[:4] = 3
    save(g, p); g3 = load(p)
    assert np.array_equal(g.prec, g3.prec) and np.array_equal(g.W_rec, g3.W_rec) and g.leak == g3.leak
    assert np.array_equal(qforward(g, X), qforward(g3, X)), "quantized forward must match"
    s = fresh_state(g3); o, _ = rstep(g3, s, np.array([0.1, 0.2, 0.3], np.float32))
    assert o.shape == (2,), "recurrent step must work on the loaded genome"

    # 3. save the evolved best, reload, identical fitness
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(48, 2)).astype(np.float32); Tt = (0.5 * Xt[:, 0:1]).astype(np.float32)
    def task(gen): return -float(((gen.forward(Xt) - Tt) ** 2).mean())
    best = Evolver(2, 1, task, pop=40, H0=8, seed=0, mutate_hook=mutate_precision).run(80)
    save(best, p)
    assert abs(task(load(p)) - task(best)) < 1e-9, "reloaded best must score identically"

    assert not os.path.exists(p + ".tmp"), "temp file must be cleaned up (atomic write)"
    os.remove(p)
    print("checkpoint: plain round-trip exact; precision + recurrence genes preserved; evolved best reloads identically; atomic")
