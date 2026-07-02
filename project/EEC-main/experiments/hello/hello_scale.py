"""WORLD-COMPLEXITY SWEEP: scale the meaning-world and find where the communication
protocol stops being learnable -- the INNER Goldilocks boundary for communication.

The proxy grammar has G exchange types (G distinct prompts, each with ONE accepted
response -- a permutation phrasebook over V=G words). A reflex (constant reply) gets
only 1/G accepted, so it can't fake it; the organism must learn the whole mapping to
survive. We hold resources FIXED (gens, conversation length) and sweep G. Mastery
holds while G is small; past some G the per-prompt selection signal is too sparse to
learn -> mastery collapses to chance. That knee is the empirical ceiling -- how much
world this substrate can support before overshooting into unlearnable.
"""
# --- EEC path bootstrap ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "150"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "4"))))
E0, TMAX = 15.0, 60
GRID = [5, 10, 20, 40, 80, 160]


class Speaker:
    def __init__(self, V, rng): self.W = rng.normal(0, 1.0, (V, V)).astype(np.float32)
    def copy(self):
        g = Speaker.__new__(Speaker); g.W = self.W.copy(); return g
    def mutate(self, rng):
        m = rng.random(self.W.shape) < MUT_RATE
        self.W += m*rng.normal(0, 1, self.W.shape).astype(np.float32)*(MUT_SCALE*(np.abs(self.W)+0.2))
    def replies(self, prompts): return self.W[prompts].argmax(1)


def converse(org, perm, rng):
    V = len(perm); e = E0; turns = 0; ok = 0; prompt = 0
    while e > 0 and turns < TMAX:
        r = int(org.W[prompt].argmax())
        if r == perm[prompt]: e += 1.0; ok += 1
        else: e -= 2.0
        turns += 1; prompt = int(rng.integers(0, V))
    return turns, ok


def cell(task):
    G, seed = task
    rng = np.random.default_rng(seed)
    perm = rng.permutation(G)                      # prompt i -> accepted response perm[i]
    pop = [Speaker(G, rng) for _ in range(POP_SIZE)]
    for g in range(GENS):
        fits = np.array([converse(m, perm, np.random.default_rng(seed*7919+g))[0] for m in pop])
        pop = reproduce(pop, fits, rng)
    best = max(pop, key=lambda m: converse(m, perm, np.random.default_rng(seed+1))[0])
    mastery = float(np.mean(best.W.argmax(1) == perm))
    surv, _ = converse(best, perm, np.random.default_rng(seed+9))
    return (G, dict(mastery=mastery, surv=surv, chance=1.0/G))


def main():
    tasks = [(G, s) for G in GRID for s in SEEDS]
    print(f"world-complexity sweep: G(exchanges) in {GRID} x {len(SEEDS)} seeds "
          f"(gens={GENS}, conversation<= {TMAX} turns)", flush=True)
    with Pool(min(18, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for G, d in res: agg.setdefault(G, []).append(d)
    mast = [np.mean([d["mastery"] for d in agg[G]]) for G in GRID]
    surv = [np.mean([d["surv"] for d in agg[G]]) for G in GRID]
    print("\n   G   mastery   survival(/%d)   chance" % TMAX)
    for i, G in enumerate(GRID):
        print(f"  {G:>3}   {mast[i]*100:5.1f}%   {surv[i]:>6.1f}        {1.0/G:.3f}")
    knee = next((G for i, G in enumerate(GRID) if mast[i] < 0.5), GRID[-1])
    print(f"\nmastery falls below 50% at G={knee}  -> inner Goldilocks boundary for this"
          f" substrate at fixed resources (gens={GENS}, {TMAX} turns/convo).")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(GRID, [m*100 for m in mast], "o-", color="#1b5e9e", lw=2)
    ax[0].axhline(50, ls="--", color="#888"); ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("world complexity G (exchange types)"); ax[0].set_ylabel("protocol mastery %")
    ax[0].set_title("Where does the protocol stop being learnable?"); ax[0].grid(alpha=.3)
    ax[1].plot(GRID, surv, "s-", color="#2e8b57", lw=2); ax[1].axhline(TMAX, ls=":", color="#aaa")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("world complexity G"); ax[1].set_ylabel("conversation length")
    ax[1].set_title("Survival vs world size"); ax[1].grid(alpha=.3)
    fig.suptitle("Communication world-complexity sweep: the inner Goldilocks boundary", weight="bold")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hello_scale.png")
    plt.savefig(out, dpi=115); print("saved", out)


if __name__ == "__main__":
    main()
