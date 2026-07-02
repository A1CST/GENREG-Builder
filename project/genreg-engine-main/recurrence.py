"""
Piece 9: recurrence / memory.

Opt-in memory for sequential / control / agent worlds. A recurrent genome keeps a hidden STATE that carries
across timesteps, so its output can depend on what it saw earlier, not just the current input.

Added as genes (so they inherit, resize, and never touch the feedforward path):
  * W_rec  (H x H)  recurrent weights — how the previous state feeds the next
  * leak   (scalar) how much of the old state is retained each step (a gene in [0, 1))

One step (leaky-integrator RNN):
    candidate = tanh(x @ W1 + state @ W_rec + b1)
    state'    = leak * state + (1 - leak) * candidate
    output    = state' @ W2 + b2

`W_rec` is mutated by the recurrence hook (not by piece 2, which only touches the feedforward weights); `leak`
mutates with it. Both ride through copy() and resize with H, set up by the earlier pieces.
"""
import numpy as np
from mutation import relative_step


def enable(g, rng, leak0=0.5):
    """Turn a feedforward genome into a recurrent one by attaching the recurrent genes (small initial W_rec)."""
    g.W_rec = (0.1 * rng.normal(0, 1 / np.sqrt(g.H), (g.H, g.H))).astype(np.float32)
    g.leak = float(leak0)


def fresh_state(g):
    """A zero state to start an episode."""
    return np.zeros(g.H, np.float32)


def rstep(g, state, x):
    """One recurrent step. x:(n_in,) or (B,n_in); state:(H,) or (B,H). Returns (output, new_state)."""
    x = np.asarray(x, np.float32)
    candidate = np.tanh(x @ g.W1 + state @ g.W_rec + g.b1)
    state = g.leak * state + (1.0 - g.leak) * candidate
    return state @ g.W2 + g.b2, state


def mutate_recurrence(g, rng, rate=0.20, scale=0.20):
    """Per-child hook: mutate the recurrent weights (relative rule) and the leak gene."""
    if not hasattr(g, "W_rec"):
        return
    g.W_rec += relative_step(g.W_rec, rate, scale, rng)
    g.leak = float(np.clip(g.leak + rng.normal(0, 0.10), 0.0, 0.99))


# --- self-test: run `python3 recurrence.py` ---
if __name__ == "__main__":
    from genome import Genome
    from reproduction import copy
    from evolver import Evolver

    # mechanics
    g = Genome(1, 1, 8, np.random.default_rng(0)); enable(g, np.random.default_rng(1))
    s = fresh_state(g); out, s2 = rstep(g, s, np.array([0.7], np.float32))
    assert out.shape == (1,) and s2.shape == (8,)
    c = copy(g)
    assert hasattr(c, "W_rec") and c.W_rec is not g.W_rec and hasattr(c, "leak"), "recurrent genes must inherit"

    # DELAYED RECALL: see a value at step 0, then zeros; the FINAL output must reproduce that value.
    # A memoryless network sees only the final 0 input, so the best it can do is a constant -> high error.
    L = 5
    vs = np.linspace(-0.8, 0.8, 6).astype(np.float32)
    def recall_fitness(gen):
        err = 0.0
        for v in vs:
            st = fresh_state(gen)
            o, st = rstep(gen, st, np.array([v], np.float32))         # step 0: show v
            for _ in range(L - 1):
                o, st = rstep(gen, st, np.zeros(1, np.float32))       # then show 0
            err += abs(float(o[0]) - v)                              # final output should equal v
        return -err / len(vs)

    memoryless_ceiling = -float(np.mean(np.abs(vs - np.median(vs))))  # best constant output
    def make_rec(rng): h = Genome(1, 1, 10, rng); enable(h, rng); return h
    ev = Evolver(1, 1, recall_fitness, pop=60, H0=10, seed=0,
                 make_genome=make_rec, mutate_hook=mutate_recurrence)
    best = ev.run(300)
    assert recall_fitness(best) > 0.6 * memoryless_ceiling + 0.0 and recall_fitness(best) > -0.10, \
        f"recurrent net should recall the value (fit {recall_fitness(best):.3f}, memoryless ceiling {memoryless_ceiling:.3f})"

    print(f"recurrence: rstep + state OK; genes inherit; delayed-recall fit {recall_fitness(best):.3f} "
          f"vs memoryless ceiling {memoryless_ceiling:.3f} (best leak {best.leak:.2f})")
