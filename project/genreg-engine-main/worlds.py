"""
Piece 12: the task / world interface.

A "world" is nothing more than a function `fitness(genome) -> score` (higher is better). Everything else in the
engine composes around that one type, so supervised, control, and agent worlds all plug in the same way:

    fit = robust(shaped(world, size_cost(0.001)), n=5)      # median over 5, minus a size cost
    Evolver(n_in, n_out, fit, ...).run(gens)

This module provides the few conventions worth standardizing.

  supervised(X, T)        regression: score = -mean squared error. Pass `forward=` to use a quantized or
                          recurrent forward instead of the plain one.
  classification(X, y)    score = exact accuracy (non-differentiable — exactly the kind of objective evolution
                          handles directly and gradient descent must replace with a surrogate).
  episodic(rollout)       a control/agent world is just a rollout that returns a scalar; wrap with robust() to
                          average noisy episodes.
"""
import numpy as np

_plain = lambda g, X: g.forward(X)


def supervised(X, T, forward=_plain):
    X = np.asarray(X, np.float32); T = np.asarray(T, np.float32)
    def fitness(g):
        return -float(((forward(g, X) - T) ** 2).mean())
    return fitness


def classification(X, y, forward=_plain):
    X = np.asarray(X, np.float32); y = np.asarray(y)
    def fitness(g):
        out = forward(g, X)
        pred = out.argmax(1) if out.shape[1] > 1 else np.sign(out[:, 0])
        target = y if out.shape[1] > 1 else np.sign(y)
        return float((pred == target).mean())                # exact accuracy, no gradient
    return fitness


def episodic(rollout):
    """rollout(genome) -> scalar return (higher is better). Wrap the result with robust() for noisy episodes."""
    def fitness(g):
        return float(rollout(g))
    return fitness


# --- self-test: run `python3 worlds.py` ---
if __name__ == "__main__":
    from evolver import Evolver
    from fitness import robust

    rng = np.random.default_rng(0)

    # 1. supervised regression
    X = rng.normal(size=(64, 2)).astype(np.float32); T = (0.5 * X[:, 0:1] - 0.3 * X[:, 1:2]).astype(np.float32)
    reg = Evolver(2, 1, supervised(X, T), pop=50, H0=8, seed=0).run(150)
    assert supervised(X, T)(reg) > -0.01, "regression world should solve"

    # 2. classification by exact accuracy (a linearly separable problem)
    Xc = rng.normal(size=(200, 2)).astype(np.float32); yc = np.sign(Xc[:, 0] - Xc[:, 1]).astype(np.float32)
    clf = Evolver(2, 1, classification(Xc, yc), pop=60, H0=8, seed=0).run(150)
    assert classification(Xc, yc)(clf) > 0.95, "classification world should reach high accuracy"

    # 3. episodic world: a noisy rollout, averaged by the median
    def rollout(g):
        x = float(np.random.default_rng().random())          # noisy target each episode
        return -abs(float(g.forward(np.array([[x]], np.float32))[0, 0]) - x)
    ep = Evolver(1, 1, robust(episodic(rollout), 7), pop=50, H0=8, seed=0).run(120)
    assert robust(episodic(rollout), 21)(ep) > -0.15, "episodic world should learn the mapping under noise"

    print(f"worlds: supervised {supervised(X,T)(reg):.3f}; classification {classification(Xc,yc)(clf):.2f} acc; "
          f"episodic {robust(episodic(rollout),21)(ep):.3f} -- one fitness type, three world kinds")
