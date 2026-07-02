"""
End-to-end: the whole engine on one task. Uses every piece at once —
  genome (1) + relative mutation (2) + reproduction (3) + evolver (4) + median fitness (5) +
  constraints (6) + precision (7) + dimension evolution (8) + recurrence (9) + telemetry (10) +
  checkpoint (11) + world interface (12).

The world is delayed recall: the controller sees a value at step 0, then zeros; at the end it must reproduce
the value. This is impossible without memory, so it needs recurrence. We add a gentle PRECISION cost (the world
trims bits the task doesn't need) but deliberately NOT a size cost: memory needs the neurons, and a size cost
would strangle the capability before it forms — a constraint has to fit the world it shapes.  Run: python3 example.py
"""
import os
import numpy as np
from genome import Genome
from evolver import Evolver, chain
from fitness import robust
from constraints import shaped
from precision import init_precision, mutate_precision, mean_bits, precision_cost
from dimension import mutate_dimensions
from recurrence import enable, fresh_state, rstep, mutate_recurrence
from telemetry import snapshot, series
from checkpoint import save, load
from worlds import episodic

L = 4   # show the value, then L-1 steps of zeros, then recall


def noisy_episode(g):
    """A fresh random value each episode (so robust()'s median earns its keep)."""
    v = float(np.random.default_rng().uniform(-1, 1))
    s = fresh_state(g)
    o, s = rstep(g, s, np.array([v], np.float32))
    for _ in range(L - 1):
        o, s = rstep(g, s, np.zeros(1, np.float32))
    return -abs(float(o[0]) - v)


def eval_recall(g, vs=np.linspace(-0.8, 0.8, 9).astype(np.float32)):
    """Deterministic recall error over a fixed set of values (for clean reporting)."""
    err = 0.0
    for v in vs:
        s = fresh_state(g)
        o, s = rstep(g, s, np.array([v], np.float32))
        for _ in range(L - 1):
            o, s = rstep(g, s, np.zeros(1, np.float32))
        err += abs(float(o[0]) - v)
    return -err / len(vs)


def make_genome(rng):
    g = Genome(1, 1, 14, rng); enable(g, rng); init_precision(g)   # recurrent + precision-capable
    return g


if __name__ == "__main__":
    world = episodic(noisy_episode)
    fitness = robust(shaped(world, precision_cost(0.0001)), n=7)   # median over 7, minus a gentle precision cost
    hook = chain(mutate_recurrence, mutate_precision, mutate_dimensions)   # mutate recurrence, precision, and size
    ev = Evolver(1, 1, fitness, pop=80, H0=14, seed=0, make_genome=make_genome, mutate_hook=hook, telemetry=snapshot)

    best = ev.run(450)

    print("delayed recall, full engine:")
    print(f"  population-mean fitness  {series(ev.history, 'mean')[0]:.3f} -> {series(ev.history, 'mean')[-1]:.3f}")
    print(f"  best recall error        {-eval_recall(best):.3f}  (0 = perfect; memoryless floor ~0.45)")
    print(f"  evolved genome           H={best.H}  mean bits={mean_bits(best):.1f}  leak={best.leak:.2f}")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pkl")
    save(best, path)
    reloaded = load(path)
    print(f"  checkpoint               saved best.pkl; reloaded scores identically: "
          f"{abs(eval_recall(reloaded) - eval_recall(best)) < 1e-9}")
