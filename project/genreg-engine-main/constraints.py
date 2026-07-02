"""
Piece 6: constraints / world-shaping.

The core of the paradigm: a genome is not judged on the task alone, but on the task MINUS the cost of the
resources it consumes. The surviving network's shape is therefore decided by the constraints of the world, not
chosen by hand. "Add a cost for X" is how you tell evolution that X is expensive, and it responds by using
less of it wherever it can without hurting the task.

A world is a task fitness plus a set of cost functions:

    world = shaped(task_fn, weight_cost(0.02), size_cost(0.001), ...)

Each cost is `coefficient x measure(genome)`. New resources (precision in piece 7, memory/energy later) are just
more cost functions composed the same way.
"""
import numpy as np


def shaped(task_fn, *costs):
    """Compose a world: fitness = task_fn(genome) - sum(cost(genome) for cost in costs)."""
    def fitness(g):
        return task_fn(g) - sum(c(g) for c in costs)
    return fitness


def _mean_abs_weight(g):
    return float(np.mean([np.abs(getattr(g, p)).mean() for p in g.PARAMS]))


def weight_cost(k):
    """Parsimony pressure on weight magnitude (works on a fixed architecture)."""
    return lambda g: k * _mean_abs_weight(g)


def size_cost(k):
    """Pressure on hidden width. Has no effect until H can evolve (piece 8); then it favors smaller networks."""
    return lambda g: k * g.H


def param_cost(k):
    """Pressure on total parameter count."""
    return lambda g: k * g.n_params()


# --- self-test: run `python3 constraints.py` ---
if __name__ == "__main__":
    from genome import Genome
    from evolver import Evolver

    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 2)).astype(np.float32)
    T = (0.5 * X[:, 0:1] - 0.3 * X[:, 1:2] + 0.2).astype(np.float32)
    def task(g): return -float(((g.forward(X) - T) ** 2).mean())

    g = Genome(2, 1, 8, np.random.default_rng(1))

    # 1. composition is exactly task minus the costs
    assert abs(shaped(task, size_cost(0.01))(g) - (task(g) - 0.01 * g.H)) < 1e-9
    assert abs(shaped(task, size_cost(0.01), weight_cost(0.02))(g)
               - (task(g) - 0.01 * g.H - 0.02 * _mean_abs_weight(g))) < 1e-9

    # 2. world-shaping: a weight cost drives the population to a LEANER network at the same task quality
    free = Evolver(2, 1, task, pop=60, H0=10, seed=0).run(200)
    taxed = Evolver(2, 1, shaped(task, weight_cost(0.05)), pop=60, H0=10, seed=0).run(200)
    raw_free, raw_taxed = task(free), task(taxed)            # compare on the RAW task, not the shaped score
    w_free, w_taxed = _mean_abs_weight(free), _mean_abs_weight(taxed)
    assert raw_taxed > -0.02, f"taxed network must still solve the task (raw {raw_taxed:.3f})"
    assert w_taxed < 0.7 * w_free, f"the cost should yield smaller weights ({w_taxed:.3f} vs {w_free:.3f})"

    print(f"world-shaping: composition exact; weight cost -> mean|w| {w_free:.3f} -> {w_taxed:.3f} "
          f"at equal task fit ({raw_free:.3f} vs {raw_taxed:.3f})")
