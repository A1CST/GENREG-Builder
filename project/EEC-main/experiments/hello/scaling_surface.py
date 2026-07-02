"""THE GENREG SCALING LAW: vocabulary mastered as a joint function of EXPERIENCE
(generations) and WORLD RICHNESS (K). The evolutionary analogue of a neural scaling
law -- but the axes are world-size and lived experience, not parameters and tokens.

Fixed real-corpus Zipfian bigram world at each K; sweep experience; measure the
EFFECTIVE VOCABULARY = number of words the organism reliably answers (and the
freq-weighted mastery = how much of actual conversation it can sustain).
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
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE, build_corpus
from mind import reproduce

TOPN, E0, TMAX = 4, 15.0, 80
KS = [40, 80, 160, 240]
GENSS = [150, 450, 1000]
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "3"))))
IDS = None
STR = {}     # K -> (accept, pp)


def build_world(ids, K):
    a, b = ids[:-1], ids[1:]
    sel = (a >= 1) & (a <= K) & (b >= 1) & (b <= K); a, b = a[sel]-1, b[sel]-1
    BG = np.zeros((K, K)); np.add.at(BG, (a, b), 1.0)
    accept = []
    for w in range(K):
        succ = [int(s) for s in np.argsort(BG[w])[::-1][:TOPN] if BG[w, s] > 1]
        accept.append(set(succ) if succ else {w})
    pp = BG.sum(1); pp = pp/pp.sum() if pp.sum() else np.ones(K)/K
    return accept, pp


class Speaker:
    def __init__(self, K, rng): self.W = rng.normal(0, 1.0, (K, K)).astype(np.float32)
    def copy(self):
        g = Speaker.__new__(Speaker); g.W = self.W.copy(); return g
    def mutate(self, rng):
        m = rng.random(self.W.shape) < MUT_RATE
        self.W += m*rng.normal(0, 1, self.W.shape).astype(np.float32)*(MUT_SCALE*(np.abs(self.W)+0.2))


def converse(org, accept, pp, rng):
    K = len(accept); e = E0; turns = 0; prompt = int(rng.choice(K, p=pp))
    while e > 0 and turns < TMAX:
        r = int(org.W[prompt].argmax())
        if r in accept[prompt]: e += 1.0
        else: e -= 2.0
        turns += 1; prompt = int(rng.choice(K, p=pp))
    return turns


def cell(task):
    K, gens, seed = task
    accept, pp = STR[K]; rng = np.random.default_rng(seed)
    pop = [Speaker(K, rng) for _ in range(POP_SIZE)]
    for g in range(gens):
        fits = np.array([converse(m, accept, pp, np.random.default_rng(seed*7919+g)) for m in pop])
        pop = reproduce(pop, fits, rng)
    best = max(pop, key=lambda m: converse(m, accept, pp, np.random.default_rng(seed+1)))
    correct = np.array([best.W[w].argmax() in accept[w] for w in range(K)])
    vocab = int(correct.sum())                                  # absolute vocabulary mastered
    fw = float(sum(pp[w]*correct[w] for w in range(K)))         # freq-weighted (usable conversation)
    return (K, gens, dict(vocab=vocab, fw=fw))


def main():
    global IDS, STR
    print("loading corpus...", flush=True)
    IDS, _, _ = build_corpus()
    for K in KS: STR[K] = build_world(IDS, K)
    tasks = [(K, g, s) for K in KS for g in GENSS for s in SEEDS]
    print(f"GENREG scaling surface: K(world) {KS} x experience(gens) {GENSS} x {len(SEEDS)} seeds", flush=True)
    with Pool(min(18, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for K, g, d in res: agg.setdefault((K, g), []).append(d)
    print("\n  effective VOCABULARY mastered (#words answered correctly):")
    print("   K \\ gens " + "".join(f"{g:>8}" for g in GENSS))
    grid = np.zeros((len(KS), len(GENSS)))
    for i, K in enumerate(KS):
        row = []
        for j, g in enumerate(GENSS):
            v = np.mean([d["vocab"] for d in agg[(K, g)]]); grid[i, j] = v; row.append(v)
        print(f"  {K:>4}     " + "".join(f"{v:>8.1f}" for v in row))
    print("\n  freq-weighted mastery (usable conversation %):")
    for i, K in enumerate(KS):
        print(f"  {K:>4}     " + "".join(f"{np.mean([d['fw'] for d in agg[(K,g)]])*100:>7.1f}%" for g in GENSS))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, K in enumerate(KS):
        ax[0].plot(GENSS, grid[i], "o-", lw=2, label=f"K={K}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("experience (generations)"); ax[0].set_ylabel("vocabulary mastered (#words)")
    ax[0].set_title("Vocabulary grows with experience (per world-size)"); ax[0].legend(); ax[0].grid(alpha=.3)
    im = ax[1].imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax[1].set_xticks(range(len(GENSS))); ax[1].set_xticklabels(GENSS)
    ax[1].set_yticks(range(len(KS))); ax[1].set_yticklabels(KS)
    ax[1].set_xlabel("experience (generations)"); ax[1].set_ylabel("world size K")
    ax[1].set_title("scaling surface: vocabulary(experience, world)")
    for i in range(len(KS)):
        for j in range(len(GENSS)):
            ax[1].text(j, i, f"{grid[i,j]:.0f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax[1], label="words mastered")
    fig.suptitle("THE GENREG SCALING LAW: vocabulary = f(experience, world-size)", weight="bold")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaling_surface.png")
    plt.savefig(out, dpi=115); print("saved", out)


if __name__ == "__main__":
    main()
