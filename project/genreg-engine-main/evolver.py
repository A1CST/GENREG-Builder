"""
Piece 4: the evolver loop.

The generational engine that ties the three operators together. No gradients: a genome is judged only by a
pluggable `fitness_fn(genome) -> score` (higher is better), and the next generation is built from the ones that
scored well.

One generation:
  1. evaluate every genome with fitness_fn
  2. rank; the top `parent_frac` become the parent pool; the top `elite` are carried forward unchanged
  3. fill the rest: pick parents from the pool, recombine (gentle crossover) or copy, then mutate

Everything flows through one seeded rng, so a run is fully reproducible. Robust fitness (median over several
evaluations, for noisy worlds) is piece 5 — kept separate so this loop stays simple.
"""
import numpy as np
from genome import Genome
from mutation import mutate
from reproduction import copy, crossover


def chain(*hooks):
    """Combine several per-child mutation hooks into one (e.g. precision genes + dimension resizing)."""
    def hook(g, rng):
        for h in hooks:
            h(g, rng)
    return hook


class Evolver:
    def __init__(self, n_in, n_out, fitness_fn, pop=60, H0=8, seed=0,
                 sexual=True, elite=2, parent_frac=0.25, mutate_hook=None, make_genome=None, telemetry=None):
        self.fitness_fn = fitness_fn
        self.sexual = sexual          # LIVE toggle: True = crossover (default), False = asexual. Set ev.sexual any time.
        self.elite, self.parent_frac = elite, parent_frac
        self.mutate_hook = mutate_hook   # optional: called on each child after weight mutation (e.g. precision genes)
        self.telemetry = telemetry       # optional: telemetry(ranked_pop) -> dict, appended to self.history each gen
        self.rng = np.random.default_rng(seed)
        factory = make_genome or (lambda rng: Genome(n_in, n_out, H0, rng))   # e.g. one that enables recurrence
        self.pop = [factory(self.rng) for _ in range(pop)]
        self.gen = 0
        self.best = None
        self.history = []

    def _evaluate(self):
        for g in self.pop:
            g.fit = float(self.fitness_fn(g))

    def step(self):
        self._evaluate()
        order = sorted(self.pop, key=lambda g: -g.fit)
        self.best = order[0]
        if self.telemetry is not None:
            rec = self.telemetry(order); rec["gen"] = self.gen; self.history.append(rec)
        n = len(self.pop)
        n_par = max(2, int(n * self.parent_frac))
        parents = order[:n_par]
        nxt = [copy(g) for g in order[:self.elite]]          # elites carried forward unchanged
        while len(nxt) < n:
            a = parents[int(self.rng.integers(n_par))]
            if self.sexual and n_par > 1:
                b = parents[int(self.rng.integers(n_par))]
                hi, lo = (a, b) if a.fit >= b.fit else (b, a)
                child = crossover(hi, lo, self.rng)
            else:
                child = copy(a)
            mutate(child, self.rng)
            if self.mutate_hook is not None:
                self.mutate_hook(child, self.rng)
            nxt.append(child)
        self.pop = nxt
        self.gen += 1
        return self.best

    def run(self, gens):
        for _ in range(gens):
            self.step()
        self._evaluate()
        return max(self.pop, key=lambda g: g.fit)


# --- self-test: run `python3 evolver.py` ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 2)).astype(np.float32)
    T = (0.5 * X[:, 0:1] - 0.3 * X[:, 1:2] + 0.2).astype(np.float32)   # a target the network must fit
    def fitness(g): return -float(((g.forward(X) - T) ** 2).mean())     # higher (closer to 0) is better

    ev = Evolver(2, 1, fitness, pop=60, H0=8, seed=0)        # default: sexual (crossover)
    f_start = max(fitness(g) for g in ev.pop)
    bests = [ev.step().fit for _ in range(150)]
    f_end = ev.run(0).fit

    # 1. fitness improves and nearly solves the task
    assert f_end > f_start and f_end > -0.01, f"expected improvement to ~0, got {f_start:.3f} -> {f_end:.3f}"
    # 2. elitism: best fitness never goes backwards
    assert np.all(np.diff(bests) >= -1e-9), "elitism should make best-fitness monotonic non-decreasing"
    # 3. reproducible
    e2 = Evolver(2, 1, fitness, pop=60, H0=8, seed=0); b2 = e2.run(150)
    assert abs(b2.fit - f_end) < 1e-9, "same seed must give the same result"
    # 4. default is sexual; the asexual toggle also improves
    assert ev.sexual is True, "default must be sexual (crossover)"
    e3 = Evolver(2, 1, fitness, pop=60, H0=8, seed=0, sexual=False); b3 = e3.run(150)
    assert b3.fit > -0.02, "asexual run should also improve"
    # 5. the toggle is LIVE: flip it mid-run with no break
    e4 = Evolver(2, 1, fitness, pop=60, H0=8, seed=0); e4.run(40)
    e4.sexual = False; e4.run(40); e4.sexual = True; b4 = e4.run(40)
    assert b4.fit > -0.02, "flipping sexual<->asexual mid-run must keep working"

    print(f"evolver: best fitness {f_start:.3f} -> {f_end:.3f} over 150 gens; elitism monotonic OK; "
          f"reproducible OK; default=sexual; asexual {b3.fit:.3f} OK; live toggle OK")
