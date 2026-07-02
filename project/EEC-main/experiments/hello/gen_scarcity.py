"""TALK MORE, VARIEDLY -- the GENREG way: NOT a novelty penalty (that grades the
output), but the SCARCITY law (a validated law of existence; the shared-food ecosystem
result) applied to the communication channel.

Each word is FOOD: a finite resource that is consumed when the organism says it and
regenerates slowly. Saying a word eats its available food (energy in); metabolism
(rent) drains energy every step. A LOOP overgrazes its two words -> their food is gone
-> the organism eats ~nothing while rent drains it -> it STARVES. Not punished for
repeating -- starved by its own overgrazing. To keep eating it must FORAGE across the
vocabulary, producing a varied plausible walk. Variety EMERGES from scarcity; the word
"repetition" appears nowhere in the fitness. (Plausibility still gates eating: an
off-manifold step finds no food.)

Compare PLAIN (flat energy for any plausible step -> loops) vs SCARCITY.
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
RENT = 0.5
REGEN = float(os.environ.get("EEC_REGEN","0.12"))   # food regrowth = inverse scarcity harshness
GENS = int(os.environ.get("EEC_MGENS", "800"))
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


def walk(org, start, rng, scarcity):
    e = E0; word = start; seq = [word]; t = 0
    food = np.ones(K, np.float32)                 # the world's finite word-food
    while e > 0 and t < TMAX:
        z = org.W[word]/TEMP; z -= z.max(); p = np.exp(z); p /= p.sum()
        nxt = int(rng.choice(K, p=p))
        plaus = nxt in ACCEPT[word]
        if scarcity:
            if plaus:
                e += float(food[nxt]); food[nxt] = 0.0     # eat what's there; it's now grazed bare
            e -= RENT                                       # metabolism drains every step
            food = np.minimum(1.0, food + REGEN)            # food regrows slowly
        else:
            e += 1.0 if plaus else -2.0
        seq.append(nxt); word = nxt; t += 1
    return t, seq


def fitness(org, rng, scarcity):
    return np.mean([walk(org, int(rng.choice(K, p=PP)), rng, scarcity)[0] for _ in range(3)])


def cell(task):
    scarcity, seed = task
    rng = np.random.default_rng(seed)
    pop = [Speaker(rng) for _ in range(POP_SIZE)]
    for g in range(GENS):
        fits = np.array([fitness(m, np.random.default_rng(seed*7919+g), scarcity) for m in pop])
        pop = reproduce(pop, fits, rng)
    best = max(pop, key=lambda m: fitness(m, np.random.default_rng(seed+1), scarcity))
    varr, plr, lens = [], [], []
    for i in range(20):
        t, seq = walk(best, int(np.random.default_rng(seed+10+i).choice(K, p=PP)),
                      np.random.default_rng(seed+30+i), scarcity)
        lens.append(t); body = seq[1:]
        if body:
            varr.append(len(set(body))/len(body))
            plr.append(np.mean([seq[j+1] in ACCEPT[seq[j]] for j in range(len(seq)-1)]))
    return (scarcity, dict(varr=float(np.mean(varr)), plr=float(np.mean(plr)),
                           length=float(np.mean(lens)), best=best if seed == 0 else None))


def main():
    global ACCEPT, PP, VOCAB
    print("loading corpus...", flush=True)
    ACCEPT, PP, VOCAB = build_world()
    tasks = [(sc, s) for sc in [False, True] for s in SEEDS]
    print(f"scarcity-law generation: K={K}, gens={GENS}, {len(SEEDS)} seeds "
          f"(plain vs SCARCITY: word-food + metabolism, no repetition grade)", flush=True)
    with Pool(min(12, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}; bests = {}
    for sc, d in res:
        agg.setdefault(sc, []).append(d)
        if d["best"] is not None: bests[sc] = d["best"]
    print(f"\n  mode       utterance_len   distinct-word ratio   plausibility")
    for sc in [False, True]:
        r = agg[sc]
        print(f"  {'SCARCITY' if sc else 'plain   '}      {np.mean([d['length'] for d in r]):>6.1f}          "
              f"{np.mean([d['varr'] for d in r]):.3f}              {np.mean([d['plr'] for d in r]):.3f}")
    for sc in [False, True]:
        if sc in bests:
            print(f"\n===== utterances ({'SCARCITY law' if sc else 'plain'}) =====")
            names = [w.lower() for w in VOCAB[1:K+1]]
            for sw in ["the", "i", "he", "."]:
                if sw in names:
                    wi = names.index(sw)
                    _, seq = walk(bests[sc], wi, np.random.default_rng(7+wi), sc)
                    print(f"   [{sw}] -> " + " ".join(VOCAB[i+1] for i in seq[:24]))


if __name__ == "__main__":
    main()
