"""DATASET-AS-ENVIRONMENT: the proxy's grammar IS a real corpus's structure.

The organisms don't learn FROM the data, they live IN it. We build the proxy from
the corpus's BIGRAM structure: each word's ACCEPTED responses = its top real
successors (how English actually flows). The proxy emits a word (by frequency); the
organism must answer with a plausible CONTINUATION; plausible -> conversation lives
(+energy), implausible -> proxy disengages. To survive the organism must internalize
the corpus's transition structure -- the regularities are the physics of its world.

Sweep vocabulary size K (world richness) and find where it stops being learnable --
the same inner-Goldilocks ceiling as the synthetic sweep, now on REAL data. Also
show the learned continuations (does it produce plausible English bigrams?).
"""
# --- EEC path bootstrap ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
from multiprocessing import Pool
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE, build_corpus
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "150"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "4"))))
E0, TMAX, TOPN = 15.0, 60, 4
GRID = [20, 50, 100, 200]
VOCAB = None; STRUCT = {}      # K -> (accept_sets, prompt_p)


def build_struct(ids, K):
    """top-K words (ids 1..K), bigram successors restricted to top-K; accept = top-N."""
    a, b = ids[:-1], ids[1:]
    sel = (a >= 1) & (a <= K) & (b >= 1) & (b <= K)
    a, b = a[sel] - 1, b[sel] - 1                       # shift to 0..K-1
    BG = np.zeros((K, K), np.float64); np.add.at(BG, (a, b), 1.0)
    accept = []
    for w in range(K):
        succ = np.argsort(BG[w])[::-1]
        succ = [int(s) for s in succ[:TOPN] if BG[w, s] > 1]
        accept.append(set(succ) if succ else {w})       # fallback: self
    prompt_p = BG.sum(1); prompt_p = prompt_p/prompt_p.sum() if prompt_p.sum() > 0 else np.ones(K)/K
    chance = float(np.mean([len(s) for s in accept]))/K
    return accept, prompt_p, chance


class Speaker:
    def __init__(self, K, rng): self.W = rng.normal(0, 1.0, (K, K)).astype(np.float32)
    def copy(self):
        g = Speaker.__new__(Speaker); g.W = self.W.copy(); return g
    def mutate(self, rng):
        m = rng.random(self.W.shape) < MUT_RATE
        self.W += m*rng.normal(0, 1, self.W.shape).astype(np.float32)*(MUT_SCALE*(np.abs(self.W)+0.2))


def converse(org, accept, pp, rng):
    K = len(accept); e = E0; turns = 0; ok = 0
    prompt = int(rng.choice(K, p=pp))
    while e > 0 and turns < TMAX:
        r = int(org.W[prompt].argmax())
        if r in accept[prompt]: e += 1.0; ok += 1
        else: e -= 2.0
        turns += 1; prompt = int(rng.choice(K, p=pp))
    return turns, ok


def cell(task):
    K, seed = task
    accept, pp, chance = STRUCT[K]
    rng = np.random.default_rng(seed)
    pop = [Speaker(K, rng) for _ in range(POP_SIZE)]
    for g in range(GENS):
        fits = np.array([converse(m, accept, pp, np.random.default_rng(seed*7919+g))[0] for m in pop])
        pop = reproduce(pop, fits, rng)
    best = max(pop, key=lambda m: converse(m, accept, pp, np.random.default_rng(seed+1))[0])
    acc = float(np.mean([best.W[w].argmax() in accept[w] for w in range(K)]))   # weighted-uniform mastery
    accw = float(sum(pp[w]*(best.W[w].argmax() in accept[w]) for w in range(K)))  # freq-weighted
    return (K, dict(mastery=acc, mastery_w=accw, chance=chance, best=best if K == 100 else None))


def main():
    global VOCAB, STRUCT
    print("loading corpus + building bigram worlds...", flush=True)
    ids, VOCAB, _ = build_corpus()
    for K in GRID:
        STRUCT[K] = build_struct(ids, K)
    tasks = [(K, s) for K in GRID for s in SEEDS]
    print(f"dataset-as-environment: K(vocab) in {GRID} x {len(SEEDS)} seeds, "
          f"accept=top-{TOPN} real successors (gens={GENS})", flush=True)
    with Pool(min(16, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}; best100 = None
    for K, d in res:
        agg.setdefault(K, []).append(d)
        if d["best"] is not None and best100 is None: best100 = d["best"]
    print("\n   K   mastery   freq-weighted   chance")
    for K in GRID:
        r = agg[K]
        print(f"  {K:>3}   {np.mean([d['mastery'] for d in r])*100:5.1f}%   "
              f"{np.mean([d['mastery_w'] for d in r])*100:5.1f}%        {r[0]['chance']:.3f}")
    knee = next((K for K in GRID if np.mean([d['mastery'] for d in agg[K]]) < 0.5), GRID[-1])
    print(f"\nmastery falls below 50% by K={knee} -> the dataset-world ceiling at fixed resources.")

    if best100 is not None:
        accept, pp, _ = STRUCT[100]
        print("\n===== learned continuations (K=100): does it produce plausible English bigrams? =====")
        order = np.argsort(pp)[::-1][:12]
        for w in order:
            r = int(best100.W[w].argmax()); good = r in accept[w]
            opts = ", ".join(VOCAB[s+1] for s in list(accept[w])[:4])
            print(f"   '{VOCAB[w+1]:<8}' -> '{VOCAB[r+1]:<10}' {'[plausible]' if good else '[off]'}   "
                  f"(real successors: {opts})")


if __name__ == "__main__":
    main()
