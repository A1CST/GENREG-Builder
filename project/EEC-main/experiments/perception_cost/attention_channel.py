"""SEARCH-WALL CONSTRUCTIVE TEST: does a DIRECTLY-SEARCHABLE attention channel
break the wall that the emergent state-gate hit?

Finding (9): an emergent state-dependent gate a_t=sigmoid(s_{t-1}.w_look+b_look)
NEVER became selective -- evolution found a global constant, never "look-when-
surprised" (conditional behavior is search-walled). The lineage fix for search
walls is a FROZEN CHANNEL + EVOLVED SCALAR MIX (not emergent routing).

Here the gate is driven by a directly-available, cheap signal -- the organism's
own PREVIOUS-STEP MISS (the energy system already 'feels' misses) -- with just two
evolved SCALARS: a_t = sigmoid(alpha * miss_{t-1} + beta). alpha is a directly-
searchable knob on a provided channel. If selectivity (reactive looking after a
surprise) NOW emerges (alpha evolves > 0) where the emergent gate failed, that
CONSTRUCTIVELY validates the search-wall reading: conditional capability needs a
channel, not pressure.

Long-range, tightened energy (unsaturated -> real pressure), look-cost on.
"""
# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
import run_matrix as RM
from evolve import EMBED, MUT_RATE, MUT_SCALE, POP_SIZE
from mind import RENT, MAX_M, MIN_M, reproduce

GENS = int(os.environ.get("EEC_MGENS", "150"))
SEG = int(os.environ.get("EEC_SEGLR", "600"))
LOOK_COST = float(os.environ.get("EEC_LOOKCOST", "0.30"))


class ChannelMind:
    """Attention gate driven by previous-step miss via two evolved scalars."""
    def __init__(self, V, rng):
        self.E = rng.normal(0, 0.1, (V, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1 / np.sqrt(EMBED), (EMBED, MAX_M)).astype(np.float32)
        self.W_rec = rng.normal(0, 1 / np.sqrt(MAX_M), (MAX_M, MAX_M)).astype(np.float32)
        self.b = np.zeros(MAX_M, dtype=np.float32)
        self.W_out = rng.normal(0, 1 / np.sqrt(MAX_M), (MAX_M, V)).astype(np.float32)
        self.b_out = np.zeros(V, dtype=np.float32)
        self.alpha = np.float32(rng.normal(0, 1.0))    # evolved knob on the miss channel
        self.beta = np.float32(rng.normal(0, 1.0))     # evolved baseline look
        self.M = int(rng.integers(MIN_M, MAX_M + 1))

    def params(self):
        return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out]

    def copy(self):
        g = ChannelMind.__new__(ChannelMind)
        (g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out) = [p.copy() for p in self.params()]
        g.alpha = self.alpha; g.beta = self.beta; g.M = self.M
        return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            p += mask * rng.normal(0, 1, p.shape).astype(np.float32) * (MUT_SCALE * (np.abs(p) + 1e-3))
        if rng.random() < MUT_RATE:
            self.alpha += np.float32(rng.normal(0, MUT_SCALE) * (abs(self.alpha) + 0.1))
        if rng.random() < MUT_RATE:
            self.beta += np.float32(rng.normal(0, MUT_SCALE) * (abs(self.beta) + 0.1))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE * self.M)), MIN_M, MAX_M))

    def run(self, seg):
        """Gated recurrence; gate from previous-step miss. Returns S,(T,M) A,(T,)."""
        M = self.M
        Win = self.W_in[:, :M]; Wrec = self.W_rec[:M, :M]; b = self.b[:M]
        Wo = self.W_out[:M, :]; bo = self.b_out
        emb = self.E[seg]; drive_full = emb @ Win
        T = len(seg)
        S = np.empty((T, M), dtype=np.float32); A = np.empty(T, dtype=np.float32)
        s = np.zeros(M, dtype=np.float32); prev_miss = 0.0
        for t in range(T):
            a = 1.0 / (1.0 + np.exp(-(self.alpha * prev_miss + self.beta)))
            s = np.tanh(a * drive_full[t] + (s @ Wrec) + b)
            S[t] = s; A[t] = a
            if t + 1 < T:                                  # did THIS step's pred miss?
                pred = int(np.argmax(s @ Wo + bo))
                prev_miss = 0.0 if pred == seg[t + 1] else 1.0
        return S, A


def eval_lives(pop, seg, look_cost, START):
    lives, looks = [], []
    nxt = seg[1:]
    for m in pop:
        S, A = m.run(seg)
        preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
        hits = preds[:-1] == nxt
        cum = np.cumsum(RENT * m.M + (~hits) + look_cost * A[:-1])
        life = len(hits) if cum[-1] < START else int(np.searchsorted(cum, START)) + 1
        lives.append(life); looks.append(float(A[:life].mean()))
    return np.array(lives), np.array(looks)


def evolve(world_ids, V, seed, look_cost, START):
    rng = np.random.default_rng(seed)
    pop = [ChannelMind(V, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        start = int(rng.integers(0, len(world_ids) - SEG - 1))
        seg = world_ids[start:start + SEG]
        lives, _ = eval_lives(pop, seg, look_cost, START)
        pop = reproduce(pop, lives, rng)
    return pop


def measure(pop, world_ids, look_cost, START, rng):
    start = int(rng.integers(0, len(world_ids) - SEG - 1))
    seg = world_ids[start:start + SEG]
    lives, _ = eval_lives(pop, seg, look_cost, START)
    bi = int(np.argmax(lives)); m = pop[bi]
    S, A = m.run(seg)
    rng2 = np.random.default_rng(int(rng.integers(1 << 30)))
    base, after, vmax = [], [], int(seg.max()) + 1
    for t0 in rng2.integers(40, SEG - 10, 24):
        s2 = seg.copy(); s2[t0] = int((s2[t0] + 3) % vmax)
        _, A2 = m.run(s2)
        base.append(A2[t0 - 5:t0].mean()); after.append(A2[t0 + 1:t0 + 4].mean())
    return dict(mean_look=float(A.mean()), life=int(lives[bi]), alpha=float(m.alpha),
                beta=float(m.beta), base=float(np.mean(base)), after=float(np.mean(after)))


def main():
    lr_ids, lrV, _ = RM.world_longrange(np.random.default_rng(999))
    print(f"gens={GENS} look_cost={LOOK_COST} | long-range V={lrV}")
    print("DIRECT MISS-CHANNEL gate  a_t = sigmoid(alpha*miss_{t-1} + beta)")
    for START in [40, 80]:
        mls, lifes, als, bas, afs = [], [], [], [], []
        for seed in [0, 1]:
            pop = evolve(lr_ids, lrV, seed, LOOK_COST, START)
            mm = measure(pop, lr_ids, LOOK_COST, START, np.random.default_rng(seed + 800))
            mls.append(mm["mean_look"]); lifes.append(mm["life"]); als.append(mm["alpha"])
            bas.append(mm["base"]); afs.append(mm["after"])
        print(f"  E0={START:>3}: mean_look={np.mean(mls):.3f} life={np.mean(lifes):.0f}/{SEG} "
              f"alpha={np.mean(als):+.3f} | reactive base={np.mean(bas):.3f} "
              f"after_surprise={np.mean(afs):.3f} UPLIFT={np.mean(afs) - np.mean(bas):+.3f}")
    print("\nCompare to emergent-gate (finding 9): uplift was +0.000 everywhere.")
    print("If UPLIFT>0 and alpha>0 here -> a searchable channel breaks the wall.")


if __name__ == "__main__":
    main()
