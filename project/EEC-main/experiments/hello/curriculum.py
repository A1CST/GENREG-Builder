"""CURRICULUM RAMP -- prove evolutionary scaling: a RISING world with organisms that
ride the Goldilocks band upward, vs cold-start that overshoots and dies.

Nested grammar: word i's accepted response target[i] in [0..i] is FIXED, so growing
the world from G to G+1 ADDS one mapping and never disturbs the learned ones. We
ramp the world in small steps (+8 words/stage), carrying the population forward, and
compare against COLD-START at each size given the SAME cumulative generation budget
(no compute advantage). Claim: the ramp stays near-mastery all the way up, while
cold-start collapses to chance past the inner Goldilocks boundary (~G 12-20, measured).
If ramp(128) succeeds where cold(128) dies, scaling here is a rising world, not a
bigger organism or more compute.
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

GMAX = 128
STAGES = list(range(8, GMAX + 1, 8))        # 8,16,...,128  (+8 per stage)
STAGE_GENS = int(os.environ.get("EEC_STAGEGENS", "30"))
CHECKPOINTS = [8, 32, 64, 128]
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "4"))))
E0, TMAX = 15.0, 60
WORLD_SEED = 123
TARGET = None                                # fixed nested grammar, shared by all runs


def make_target():
    r = np.random.default_rng(WORLD_SEED)
    t = np.zeros(GMAX, np.int64)
    for i in range(1, GMAX):
        t[i] = int(r.integers(0, i + 1))       # nested: response in [0,i] -> always in active vocab
    return t


class Speaker:
    def __init__(self, rng): self.W = rng.normal(0, 1.0, (GMAX, GMAX)).astype(np.float32)
    def copy(self):
        g = Speaker.__new__(Speaker); g.W = self.W.copy(); return g
    def mutate(self, rng):
        m = rng.random(self.W.shape) < MUT_RATE
        self.W += m * rng.normal(0, 1, self.W.shape).astype(np.float32) * (MUT_SCALE * (np.abs(self.W) + 0.2))


def converse(org, G, rng, focus=None):
    """focus=(lo,hi): newly-introduced words drilled at 50% (curriculum emphasis);
    else prompts uniform over the whole world [0,G)."""
    e = E0; turns = 0; prompt = 0
    while e > 0 and turns < TMAX:
        r = int(org.W[prompt, :G].argmax())
        if r == TARGET[prompt]: e += 1.0
        else: e -= 2.0
        turns += 1
        if focus is not None and rng.random() < 0.5:
            prompt = int(rng.integers(focus[0], focus[1]))
        else:
            prompt = int(rng.integers(0, G))
    return turns


def mastery(org, G):
    return float(np.mean([org.W[p, :G].argmax() == TARGET[p] for p in range(G)]))


def evolve(pop, G, gens, rng, focus=None):
    for g in range(gens):
        fits = np.array([converse(m, G, np.random.default_rng(g * 6907 + id(m) % 9973), focus) for m in pop])
        pop = reproduce(pop, fits, rng)
    return pop


def ramp_cell(seed):
    rng = np.random.default_rng(seed)
    pop = [Speaker(rng) for _ in range(POP_SIZE)]
    out = {}; prev = 0
    for G in STAGES:
        pop = evolve(pop, G, STAGE_GENS, rng, focus=(prev, G))   # drill the NEW words this stage
        prev = G
        if G in CHECKPOINTS:
            best = max(pop, key=lambda m: converse(m, G, np.random.default_rng(seed + 1)))
            out[G] = mastery(best, G)
    return ("ramp", seed, out)


def cold_cell(task):
    G, seed = task
    cum = (STAGES.index(G) + 1) * STAGE_GENS              # SAME budget the ramp used to reach G
    rng = np.random.default_rng(seed + 1000)
    pop = [Speaker(rng) for _ in range(POP_SIZE)]
    pop = evolve(pop, G, cum, rng, focus=None)           # cold-start: everything at once, uniform
    best = max(pop, key=lambda m: converse(m, G, np.random.default_rng(seed + 2)))
    return ("cold", seed, {G: mastery(best, G)})


def dispatch(t):
    return ramp_cell(t[1]) if t[0] == "ramp" else cold_cell(t[1])


def main():
    global TARGET
    TARGET = make_target()
    print(f"CURRICULUM RAMP: stages {STAGES[0]}..{STAGES[-1]} (+8), {STAGE_GENS} gens/stage, "
          f"checkpoints {CHECKPOINTS}, {len(SEEDS)} seeds", flush=True)
    tasks = [("ramp", s) for s in SEEDS] + [("cold", (G, s)) for G in CHECKPOINTS for s in SEEDS]
    with Pool(min(18, len(tasks))) as p:
        res = p.map(dispatch, tasks, chunksize=1)
    ramp = {G: [] for G in CHECKPOINTS}; cold = {G: [] for G in CHECKPOINTS}
    for mode, seed, out in res:
        for G, m in out.items():
            (ramp if mode == "ramp" else cold)[G].append(m)
    print(f"\n   G    RAMP mastery     COLD-start mastery   (cold budget = ramp's cumulative gens)")
    for G in CHECKPOINTS:
        rm = np.mean(ramp[G]); cm = np.mean(cold[G])
        print(f"  {G:>3}    {rm*100:5.1f}% +/-{np.std(ramp[G])*100:4.1f}   {cm*100:5.1f}% +/-{np.std(cold[G])*100:4.1f}   "
              f"chance {100.0/G:4.1f}%   {'<<< RAMP BREAKS THROUGH' if rm > cm + 0.25 else ''}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(CHECKPOINTS, [np.mean(ramp[G])*100 for G in CHECKPOINTS], "o-", color="#1b5e9e", lw=2.5, label="curriculum ramp (rising world)")
    ax.plot(CHECKPOINTS, [np.mean(cold[G])*100 for G in CHECKPOINTS], "s--", color="#c44", lw=2.5, label="cold-start (same gen budget)")
    ax.plot(CHECKPOINTS, [100.0/G for G in CHECKPOINTS], ":", color="#aaa", label="chance")
    ax.set_xscale("log", base=2); ax.set_xlabel("world complexity G (words)"); ax.set_ylabel("protocol mastery %")
    ax.set_title("Curriculum ramp vs cold-start: scaling by a rising world", weight="bold")
    ax.legend(); ax.grid(alpha=.3); ax.set_ylim(0, 100)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curriculum.png")
    plt.savefig(out, dpi=120); print("saved", out)


if __name__ == "__main__":
    main()
