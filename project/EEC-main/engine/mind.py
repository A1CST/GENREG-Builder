"""mind.py -- Step B: a persistent-state organism with evolvable memory.

The organism carries a recurrent internal state across the token stream. Its
memory size M is an evolvable trait, and HOLDING memory costs energy every step
(memory rent proportional to M) -- capacity evolves against the energy law, the
way width did. It eats by anticipating its world (prediction = metabolism, how
energy enters a sequential environment); we never optimize toward accuracy.
Selection is pure survival -- LIFESPAN, how long it stays alive in the stream.

We read out the emergent state dynamics, not the token output.

EEC_MAXM = max memory, EEC_RENT = energy per held unit per step,
EEC_VOCAB inherited (set small for speed of the recurrent loop).
"""
import os
import pickle
import numpy as np

os.environ.setdefault("EEC_VOCAB", "2000")
from evolve import (EMBED, MUT_RATE, MUT_SCALE, POP_SIZE, TOURNAMENT_K,
                    START_ENERGY, HARD_CAP, build_corpus)

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_DIR = os.path.join(HERE, "best")
SEED = 0
MAX_M = int(os.environ.get("EEC_MAXM", "48"))     # ceiling on memory size
RENT = float(os.environ.get("EEC_RENT", "0.010")) # energy per held unit per step
SEG = int(os.environ.get("EEC_SEG", "2000"))      # max stream length per life
MIN_M = 2
# ENTROPY law: the carried state leaks each step unless actively refreshed.
# DECAY=1.0 -> no entropy (control). DECAY<1 -> order degrades; holding info
# becomes work. A world-constant, not an evolved trait.
DECAY = float(os.environ.get("EEC_DECAY", "1.0"))
GENS = int(os.environ.get("EEC_GENS", str(HARD_CAP)))
TAG = os.environ.get("EEC_TAG", "best")

# ---- population dynamics (steady-state, overlapping generations) ----------
# top 20% = elites (unchanged) + middle 60% (unchanged) are carried; bottom 20%
# is culled and refilled with mutated clones of the top 20%. Diversity persists
# by default instead of collapsing to clones-of-the-best every generation.
N_ELITE = round(0.20 * POP_SIZE)      # top 20% are the breeders
N_BOTTOM = round(0.20 * POP_SIZE)     # bottom 20% are culled/replaced


def reproduce(pop, lives, rng):
    order = np.argsort(lives)[::-1]                 # best -> worst
    carried = [pop[i] for i in order[:POP_SIZE - N_BOTTOM]]   # top 80% unchanged
    top = order[:N_ELITE]
    new = list(carried)
    while len(new) < POP_SIZE:
        parent = pop[int(top[rng.integers(0, len(top))])]
        child = parent.copy(); child.mutate(rng)
        new.append(child)
    return new


class Mind:
    def __init__(self, vocab_size, rng):
        self.E = rng.normal(0, 0.1, (vocab_size, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1/np.sqrt(EMBED), (EMBED, MAX_M)).astype(np.float32)
        self.W_rec = rng.normal(0, 1/np.sqrt(MAX_M), (MAX_M, MAX_M)).astype(np.float32)
        self.b = np.zeros(MAX_M, dtype=np.float32)
        self.W_out = rng.normal(0, 1/np.sqrt(MAX_M), (MAX_M, vocab_size)).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)
        self.M = int(rng.integers(MIN_M, MAX_M + 1))   # evolvable memory size

    def run_states(self, seg_emb):
        """Sequential recurrence over a segment. seg_emb: (T, EMBED).
        Returns S: (T, M) internal-state trajectory (only active M units)."""
        M = self.M
        Win = self.W_in[:, :M]; Wrec = self.W_rec[:M, :M]; b = self.b[:M]
        drive = seg_emb @ Win                          # (T, M) input drive, batched
        T = seg_emb.shape[0]
        S = np.empty((T, M), dtype=np.float32)
        s = np.zeros(M, dtype=np.float32)
        for t in range(T):
            s = np.tanh(drive[t] + (DECAY * s) @ Wrec + b)   # state leaks (entropy)
            S[t] = s
        return S

    def params(self):
        return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out]

    def copy(self):
        g = Mind.__new__(Mind)
        g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out = [p.copy() for p in self.params()]
        g.M = self.M
        return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            step = rng.normal(0, 1, p.shape).astype(np.float32)
            p += mask * step * (MUT_SCALE * (np.abs(p) + 1e-3))
        if rng.random() < 0.3:                          # relative memory-size mutation
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE * self.M)),
                                 MIN_M, MAX_M))


def live(mind, seg_ids, seg_emb):
    """One life in the stream. Returns (lifespan, correct, M)."""
    S = mind.run_states(seg_emb)                        # (T, M)
    logits = S @ mind.W_out[:mind.M, :] + mind.b_out    # (T, V) batched output
    preds = logits.argmax(1)
    nxt = seg_ids[1:]                                   # predict next token
    hits = preds[:-1] == nxt
    rent = RENT * mind.M
    # energy: rent every step + 1 per miss; death when budget exhausted
    cost = rent + (~hits)
    cum = np.cumsum(cost)
    if cum[-1] < START_ENERGY:
        life = len(hits); correct = int(hits.sum())
    else:
        life = int(np.searchsorted(cum, START_ENERGY)) + 1
        correct = int(hits[:life].sum())
    return life, correct, mind.M


def main():
    rng = np.random.default_rng(SEED)
    print(f"Loading corpus (V={os.environ['EEC_VOCAB']}, MAX_M={MAX_M}, "
          f"rent={RENT}/unit/step, DECAY={DECAY}, tag={TAG})...")
    ids, vocab, word2id = build_corpus()
    vocab_size = len(vocab)
    print(f"vocab {vocab_size} | stream {len(ids):,} tokens | "
          f"full-memory rent = {RENT*MAX_M:.2f}/step")

    population = [Mind(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(1, GENS + 1):
        start = int(rng.integers(0, len(ids) - SEG - 1))
        seg = ids[start:start + SEG]
        seg_emb_full = None  # each mind has its own E, so emb per genome
        res = []
        for m in population:
            emb = m.E[seg]
            res.append(live(m, seg, emb))
        lives = np.array([r[0] for r in res])
        corrs = np.array([r[1] for r in res])
        Ms = np.array([r[2] for r in res])
        bi = int(np.argmax(lives))                      # selection = survival
        print(f"gen {gen:>4} | lifespan {lives[bi]:>5}/{SEG} | M {Ms[bi]:>3} | "
              f"avg_life {lives.mean():>7.1f} | avg_M {Ms.mean():>5.1f} | "
              f"corr {corrs[bi]:>4}")

        with open(os.path.join(BEST_DIR, f"mind_{TAG}.pkl"), "wb") as f:
            pickle.dump({"genome": population[bi].params(), "M": int(Ms[bi]),
                         "gen": gen, "lifespan": int(lives[bi]), "decay": DECAY,
                         "vocab": vocab}, f)

        population = reproduce(population, lives, rng)
    print("\nDone.", os.path.join(BEST_DIR, "mind_best.pkl"))


if __name__ == "__main__":
    main()
