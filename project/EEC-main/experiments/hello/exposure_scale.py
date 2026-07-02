"""EXPERIENCE, not curriculum: in a frequency-structured (Zipfian) world, scaling is
EXPOSURE. The environment's own statistics order the learning -- common words are
over-sampled and normalize first; rare words lag until enough total experience
accumulates that they too cross the threshold. No imposed schedule. Re-occurrence
does it. (Works in the lookup substrate -- it's per-word sample count, not transfer.)

Fixed real-corpus world (bigram grammar, prompts drawn by true frequency). Sweep
total EXPOSURE (generations) and measure mastery split by word-frequency BAND
(common / mid / rare). Prediction: common band high from the start; the RARE band
rises with exposure -- vocabulary expanding as rare words normalize.
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

K = int(os.environ.get("EEC_K", "80"))
TOPN, E0, TMAX = 4, 15.0, 80
EXPOSURES = [50, 200, 600, 1500]           # generations = total experience
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "3"))))
ACCEPT = None; PP = None; FREQ = None; VOCAB = None; BANDS = None


def build_world():
    ids, vocab, _ = build_corpus()
    a, b = ids[:-1], ids[1:]
    sel = (a >= 1) & (a <= K) & (b >= 1) & (b <= K)
    a, b = a[sel]-1, b[sel]-1
    BG = np.zeros((K, K)); np.add.at(BG, (a, b), 1.0)
    accept = []
    for w in range(K):
        succ = [int(s) for s in np.argsort(BG[w])[::-1][:TOPN] if BG[w, s] > 1]
        accept.append(set(succ) if succ else {w})
    pp = BG.sum(1); freq = pp.copy(); pp = pp/pp.sum() if pp.sum() else np.ones(K)/K
    # frequency bands (terciles by unigram frequency)
    order = np.argsort(freq)[::-1]
    bands = {"common": order[:K//3], "mid": order[K//3:2*K//3], "rare": order[2*K//3:]}
    return accept, pp, freq, vocab, bands


class Speaker:
    def __init__(self, rng): self.W = rng.normal(0, 1.0, (K, K)).astype(np.float32)
    def copy(self):
        g = Speaker.__new__(Speaker); g.W = self.W.copy(); return g
    def mutate(self, rng):
        m = rng.random(self.W.shape) < MUT_RATE
        self.W += m*rng.normal(0, 1, self.W.shape).astype(np.float32)*(MUT_SCALE*(np.abs(self.W)+0.2))


def converse(org, rng):
    e = E0; turns = 0; prompt = int(rng.choice(K, p=PP))
    while e > 0 and turns < TMAX:
        r = int(org.W[prompt].argmax())
        if r in ACCEPT[prompt]: e += 1.0
        else: e -= 2.0
        turns += 1; prompt = int(rng.choice(K, p=PP))
    return turns


def band_mastery(org):
    correct = np.array([org.W[w].argmax() in ACCEPT[w] for w in range(K)])
    return {b: float(correct[idx].mean()) for b, idx in BANDS.items()}


def cell(task):
    gens, seed = task
    rng = np.random.default_rng(seed)
    pop = [Speaker(rng) for _ in range(POP_SIZE)]
    for g in range(gens):
        fits = np.array([converse(m, np.random.default_rng(seed*7919+g)) for m in pop])
        pop = reproduce(pop, fits, rng)
    best = max(pop, key=lambda m: converse(m, np.random.default_rng(seed+1)))
    return (gens, band_mastery(best))


def main():
    global ACCEPT, PP, FREQ, VOCAB, BANDS
    print("loading corpus + building Zipfian bigram world...", flush=True)
    ACCEPT, PP, FREQ, VOCAB, BANDS = build_world()
    print(f"experience sweep: K={K} world, exposures(gens)={EXPOSURES}, {len(SEEDS)} seeds  "
          f"(bands by frequency tercile)", flush=True)
    tasks = [(g, s) for g in EXPOSURES for s in SEEDS]
    with Pool(min(16, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for g, bm in res: agg.setdefault(g, []).append(bm)
    print(f"\n  exposure   common    mid     rare    (mastery by word-frequency band)")
    curves = {b: [] for b in ["common", "mid", "rare"]}
    for g in EXPOSURES:
        rows = agg[g]; m = {b: np.mean([r[b] for r in rows]) for b in curves}
        for b in curves: curves[b].append(m[b])
        print(f"  {g:>6}    {m['common']*100:5.1f}%  {m['mid']*100:5.1f}%  {m['rare']*100:5.1f}%")
    print(f"\nrare-band mastery: {curves['rare'][0]*100:.0f}% -> {curves['rare'][-1]*100:.0f}% "
          f"as exposure grows {EXPOSURES[0]}->{EXPOSURES[-1]} gens "
          f"({'RARE WORDS NORMALIZING with experience' if curves['rare'][-1] > curves['rare'][0]+0.05 else 'flat'})")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for b, c in [("common", "#1b5e9e"), ("mid", "#c9a227"), ("rare", "#c44")]:
        ax.plot(EXPOSURES, [x*100 for x in curves[b]], "o-", color=c, lw=2.5, label=f"{b} words")
    ax.set_xscale("log"); ax.set_xlabel("exposure (generations = total experience)")
    ax.set_ylabel("mastery %"); ax.set_ylim(0, 100); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Scaling by EXPERIENCE: rare words normalize as exposure grows", weight="bold")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exposure_scale.png")
    plt.savefig(out, dpi=120); print("saved", out)


if __name__ == "__main__":
    main()
