"""
Piece 3: reproduction.

How a parent becomes a child.

  copy(g)              an independent deep copy — its own arrays, including the self-adaptive strategy genes
                       (mr, ms) if the genome has them. Mutating the copy must never touch the original.

  crossover(hi, lo)    GENTLE sexual reproduction. The child is the fitter parent `hi` with only a SMALL
                       fraction of weights taken from `lo`, and only when the two parents are the SAME shape.
                       This restraint is deliberate: a network's hidden units are not aligned between genomes
                       (unit 3 in one means nothing in another), so mixing weights heavily scrambles a learned
                       function. A low rate adds a little recombination without destroying the parent.
"""
import numpy as np
from genome import Genome

XOVER_RATE = 0.10                # fraction of weights drawn from the second parent
GENE_ATTRS = ("mr", "ms", "prec", "W_rec", "leak")   # heritable genes beyond the weights (strategy, precision, recurrence)


def copy(g):
    """Independent deep copy (does not re-randomize; preserves the weights and any genes present)."""
    c = Genome.__new__(Genome)
    c.n_in, c.n_out, c.H = g.n_in, g.n_out, g.H
    for p in g.PARAMS:
        setattr(c, p, getattr(g, p).copy())
    for a in GENE_ATTRS:
        if hasattr(g, a):
            v = getattr(g, a)
            setattr(c, a, v.copy() if hasattr(v, "copy") else v)   # arrays copy; scalars (leak) assign by value
    return c


def same_shape(a, b):
    return a.W1.shape == b.W1.shape and a.W2.shape == b.W2.shape


def crossover(hi, lo, rng, rate=XOVER_RATE):
    """Child = fitter parent `hi`, with ~`rate` of its weights replaced by `lo`'s (like-shaped parents only)."""
    c = copy(hi)
    if same_shape(hi, lo):
        for p in Genome.PARAMS:
            A, B = getattr(c, p), getattr(lo, p)
            m = rng.random(A.shape) < rate            # a small random subset of this array...
            A[m] = B[m]                               # ...taken from the other parent
    return c


# --- self-test: run `python3 reproduction.py` ---
if __name__ == "__main__":
    from mutation import mutate

    # 1. copy is an independent deep copy, strategy genes included
    g = Genome(4, 2, 10, np.random.default_rng(1)); mutate(g, np.random.default_rng(2))   # attaches mr/ms
    before = g.W1.copy()
    c = copy(g)
    assert np.array_equal(c.W1, g.W1) and c.W1 is not g.W1, "copy must be independent"
    assert hasattr(c, "mr") and c.mr is not g.mr and np.array_equal(c.mr, g.mr), "strategy genes must copy"
    mutate(c, np.random.default_rng(3))
    assert np.array_equal(g.W1, before), "mutating the copy must NOT change the original"
    assert not np.array_equal(c.W1, before), "the copy did mutate"

    # 2. gentle crossover: child is mostly hi, ~rate from lo
    hi = Genome(4, 2, 40, np.random.default_rng(4)); lo = Genome(4, 2, 40, np.random.default_rng(5))
    ch = crossover(hi, lo, np.random.default_rng(6), rate=0.10)
    from_lo = float(((ch.W1 == lo.W1) & (lo.W1 != hi.W1)).mean())
    from_hi = float((ch.W1 == hi.W1).mean())
    assert 0.04 < from_lo < 0.17, f"expected ~10% from lo, got {from_lo:.3f}"
    assert from_hi > 0.80, f"child should be mostly hi, got {from_hi:.3f}"
    assert ch.forward(np.zeros((3, 4))).shape == (3, 2)

    # 3. mismatched shapes -> plain copy of hi (no mixing, no crash)
    lo2 = Genome(4, 2, 12, np.random.default_rng(7))
    ch2 = crossover(hi, lo2, np.random.default_rng(8))
    assert np.array_equal(ch2.W1, hi.W1), "different-shaped parents must not be mixed"

    # 4. reproducible
    a = crossover(hi, lo, np.random.default_rng(6)); b = crossover(hi, lo, np.random.default_rng(6))
    assert np.array_equal(a.W1, b.W1), "crossover must be reproducible"

    print(f"reproduction: deep-copy independent (+strategy genes); gentle crossover ~{100*from_lo:.0f}% from lo, "
          f"{100*from_hi:.0f}% from hi; shape-mismatch safe; reproducible OK")
