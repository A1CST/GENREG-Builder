"""TALK MORE: autoregressive generation as survival. The organism's own output
becomes its next prompt; it survives only by stepping along PLAUSIBLE bigram edges
of the real corpus. A surviving organism walks a long coherent word-chain -- it
generates. Survival length = how many words it strings together before stepping off
the manifold and dying. Sweep EXPERIENCE: longer life = it talks more, and more
plausibly. Show the actual utterances it produces.
"""
# --- EEC path bootstrap ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
from multiprocessing import Pool
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE, build_corpus
from mind import reproduce

K = int(os.environ.get("EEC_K", "60"))
TOPN, E0, TMAX, TEMP = 5, 20.0, 120, 1.0
GENSS = [100, 400, 1000]
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "3"))))
ACCEPT = None; PP = None; VOCAB = None


def build_world():
    ids, vocab, _ = build_corpus()
    a, b = ids[:-1], ids[1:]
    sel = (a >= 1) & (a <= K) & (b >= 1) & (b <= K); a, b = a[sel]-1, b[sel]-1
    BG = np.zeros((K, K)); np.add.at(BG, (a, b), 1.0)
    accept = []
    for w in range(K):
        succ = [int(s) for s in np.argsort(BG[w])[::-1][:TOPN] if BG[w, s] > 1]
        accept.append(set(succ) if succ else {w})
    pp = BG.sum(1); pp = pp/pp.sum() if pp.sum() else np.ones(K)/K
    return accept, pp, vocab


class Speaker:
    def __init__(self, rng): self.W = rng.normal(0, 1.0, (K, K)).astype(np.float32)
    def copy(self):
        g = Speaker.__new__(Speaker); g.W = self.W.copy(); return g
    def mutate(self, rng):
        m = rng.random(self.W.shape) < MUT_RATE
        self.W += m*rng.normal(0, 1, self.W.shape).astype(np.float32)*(MUT_SCALE*(np.abs(self.W)+0.2))


def walk(org, start, rng):
    """autoregressive + STOCHASTIC: next ~ softmax(W[word]/TEMP). A plausible step
    (real corpus bigram edge) feeds energy; an implausible one drains it. To talk
    long the organism must keep EVERY sampled step on the manifold -- not just loop
    one cycle. Returns utterance length and the word sequence."""
    e = E0; word = start; seq = [word]; t = 0
    while e > 0 and t < TMAX:
        z = org.W[word] / TEMP; z -= z.max(); p = np.exp(z); p /= p.sum()
        nxt = int(rng.choice(K, p=p))
        e += 1.0 if nxt in ACCEPT[word] else -2.0
        seq.append(nxt); word = nxt; t += 1
    return t, seq


def fitness(org, rng):
    return np.mean([walk(org, int(rng.choice(K, p=PP)), rng)[0] for _ in range(3)])


def cell(task):
    gens, seed = task
    rng = np.random.default_rng(seed)
    pop = [Speaker(rng) for _ in range(POP_SIZE)]
    for g in range(gens):
        fits = np.array([fitness(m, np.random.default_rng(seed*7919+g)) for m in pop])
        pop = reproduce(pop, fits, rng)
    best = max(pop, key=lambda m: fitness(m, np.random.default_rng(seed+1)))
    lengths = [walk(best, int(np.random.default_rng(seed+10+i).choice(K, p=PP)),
                    np.random.default_rng(seed+50+i))[0] for i in range(20)]
    return (gens, dict(meanlen=float(np.mean(lengths)), maxlen=int(np.max(lengths)),
                       best=best if gens == GENSS[-1] else None))


def main():
    global ACCEPT, PP, VOCAB
    print("loading corpus + bigram world...", flush=True)
    ACCEPT, PP, VOCAB = build_world()
    tasks = [(g, s) for g in GENSS for s in SEEDS]
    print(f"generation-as-survival: K={K}, experience(gens)={GENSS}, {len(SEEDS)} seeds "
          f"(utterance length = words said before stepping off the manifold)", flush=True)
    with Pool(min(12, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}; best = None
    for g, d in res:
        agg.setdefault(g, []).append(d)
        if d["best"] is not None and best is None: best = d["best"]
    print("\n  experience   mean utterance   longest utterance   (words, max %d)" % TMAX)
    for g in GENSS:
        r = agg[g]
        print(f"  {g:>8}     {np.mean([d['meanlen'] for d in r]):>8.1f}        "
              f"{np.max([d['maxlen'] for d in r]):>6}")

    if best is not None:
        print("\n===== what the organism SAYS (autoregressive walks from common seeds) =====")
        for seedw in ["the", "i", "he", "and", "."]:
            if seedw in [w.lower() for w in VOCAB[1:K+1]]:
                wi = [w.lower() for w in VOCAB[1:K+1]].index(seedw)
                _, seq = walk(best, wi, np.random.default_rng(99 + wi))
                words = " ".join(VOCAB[i+1] for i in seq[:24])
                print(f"   [{seedw}] -> {words}")


if __name__ == "__main__":
    main()
