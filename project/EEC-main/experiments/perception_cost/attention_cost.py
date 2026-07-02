"""ACTIVE PERCEPTION COST -- a new law on a new capability axis: SELECTIVE
ATTENTION / sampling policy.

Every prior law degrades observation EXTERNALLY (occlusion/noise) or taxes the
body (rent/energy). None lets the organism CONTROL its own sensing. Here looking
costs energy: each step the organism sets an attention gate a_t = sigmoid(s_{t-1}
. w_look + b_look) in [0,1]; the input drive is scaled by a_t (attend less -> run
blind on recurrence), and metabolism drains LOOK_COST * a_t that step. So there is
pressure to look SELECTIVELY -- pay to look only when memory can't already predict.

Long-range world (periodic block + 5% noise): once phase-locked you can predict
WITHOUT looking, so the metabolically optimal policy is to GLANCE intermittently
to stay locked, then coast. PREDICTION: under look-cost, mean attention drops well
below 1 while survival holds (selective sampling emerges); without cost, gates
stay open (looking is free). We read the emergent POLICY (mean gate, reactive
looking after a surprise), never accuracy. Selection = pure lifespan.

Gates seeded with spread (b_look ~ N(0,1.5)) so attention is evolvable from gen 1
(avoids the old perc.py freeze where zero-init relative mutation calcified gates).
"""
# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_matrix as RM
from evolve import EMBED, MUT_RATE, MUT_SCALE, POP_SIZE, START_ENERGY
from mind import RENT, MAX_M, MIN_M, reproduce

HERE = _o.path.dirname(_o.path.abspath(__file__))
GENS = int(os.environ.get("EEC_MGENS", "150"))
SEG_LR = int(os.environ.get("EEC_SEGLR", "600"))
SEG_TEXT = int(os.environ.get("EEC_SEGTEXT", "500"))
LOOK_COST = float(os.environ.get("EEC_LOOKCOST", "0.30"))   # energy per unit attention/step


class AttnMind:
    """Recurrent organism with a state-dependent attention gate over perception."""
    def __init__(self, V, rng):
        self.E = rng.normal(0, 0.1, (V, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1 / np.sqrt(EMBED), (EMBED, MAX_M)).astype(np.float32)
        self.W_rec = rng.normal(0, 1 / np.sqrt(MAX_M), (MAX_M, MAX_M)).astype(np.float32)
        self.b = np.zeros(MAX_M, dtype=np.float32)
        self.W_out = rng.normal(0, 1 / np.sqrt(MAX_M), (MAX_M, V)).astype(np.float32)
        self.b_out = np.zeros(V, dtype=np.float32)
        self.w_look = rng.normal(0, 1 / np.sqrt(MAX_M), MAX_M).astype(np.float32)
        self.b_look = np.float32(rng.normal(0, 1.5))         # seeded spread -> evolvable
        self.M = int(rng.integers(MIN_M, MAX_M + 1))

    def params(self):
        return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out, self.w_look]

    def copy(self):
        g = AttnMind.__new__(AttnMind)
        (g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out, g.w_look) = [p.copy() for p in self.params()]
        g.b_look = self.b_look; g.M = self.M
        return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            p += mask * rng.normal(0, 1, p.shape).astype(np.float32) * (MUT_SCALE * (np.abs(p) + 1e-3))
        self.b_look += np.float32(rng.normal(0, MUT_SCALE) * (abs(self.b_look) + 0.1) * (rng.random() < MUT_RATE))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE * self.M)), MIN_M, MAX_M))

    def run(self, seg, decay=1.0):
        """Gated recurrence. Returns states S (T,M) and attention trace a (T,)."""
        M = self.M
        Win = self.W_in[:, :M]; Wrec = self.W_rec[:M, :M]; b = self.b[:M]
        wl = self.w_look[:M]; bl = self.b_look
        emb = self.E[seg]
        drive_full = emb @ Win                       # (T,M) full-look drive
        T = len(seg)
        S = np.empty((T, M), dtype=np.float32); A = np.empty(T, dtype=np.float32)
        s = np.zeros(M, dtype=np.float32)
        for t in range(T):
            a = 1.0 / (1.0 + np.exp(-(s @ wl + bl)))  # attention from PREVIOUS state
            s = np.tanh(a * drive_full[t] + decay * (s @ Wrec) + b)
            S[t] = s; A[t] = a
        return S, A


def eval_lives(pop, seg, look_cost):
    lives, looks = [], []
    nxt = seg[1:]
    for m in pop:
        S, A = m.run(seg)
        preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
        hits = preds[:-1] == nxt
        cost = RENT * m.M + (~hits) + look_cost * A[:-1]
        cum = np.cumsum(cost)
        life = len(hits) if cum[-1] < START_ENERGY \
            else int(np.searchsorted(cum, START_ENERGY)) + 1
        lives.append(life); looks.append(float(A[:life].mean()))
    return np.array(lives), np.array(looks)


def evolve(world_ids, V, seed, seg_len, look_cost):
    rng = np.random.default_rng(seed)
    pop = [AttnMind(V, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        start = int(rng.integers(0, len(world_ids) - seg_len - 1))
        seg = world_ids[start:start + seg_len]
        lives, _ = eval_lives(pop, seg, look_cost)
        pop = reproduce(pop, lives, rng)
    return pop


def measure(pop, world_ids, seg_len, look_cost, rng):
    start = int(rng.integers(0, len(world_ids) - seg_len - 1))
    seg = world_ids[start:start + seg_len]
    lives, looks = eval_lives(pop, seg, look_cost)
    bi = int(np.argmax(lives))
    m = pop[bi]
    S, A = m.run(seg)
    # reactive looking: does attention RISE in the 3 steps after an injected surprise?
    rng2 = np.random.default_rng(int(rng.integers(1 << 30)))
    base, after = [], []
    vmax = int(seg.max()) + 1
    for t0 in rng2.integers(40, seg_len - 10, 24):
        s2 = seg.copy(); s2[t0] = int((s2[t0] + 3) % vmax)
        _, A2 = m.run(s2)
        base.append(A2[t0 - 5:t0].mean()); after.append(A2[t0 + 1:t0 + 4].mean())
    return dict(mean_look=float(A.mean()), best_life=int(lives[bi]),
                pop_look=float(looks.mean()), M=int(m.M),
                look_base=float(np.mean(base)), look_after=float(np.mean(after)))


def main():
    text_ids, textV, _ = RM.world_text()
    lr_ids, lrV, _ = RM.world_longrange(np.random.default_rng(999))
    worlds = [("text", text_ids, textV, SEG_TEXT), ("longrange", lr_ids, lrV, SEG_LR)]
    print(f"gens={GENS} look_cost={LOOK_COST} | text V={textV} lr V={lrV}")
    res = {}
    for wname, ids, V, seg in worlds:
        for cost in [0.0, LOOK_COST]:
            mls, lifes, bas, afs = [], [], [], []
            for seed in [0, 1]:
                pop = evolve(ids, V, seed, seg, cost)
                mm = measure(pop, ids, seg, cost, np.random.default_rng(seed + 800))
                mls.append(mm["mean_look"]); lifes.append(mm["best_life"])
                bas.append(mm["look_base"]); afs.append(mm["look_after"])
            res[(wname, cost)] = dict(look=np.mean(mls), life=np.mean(lifes),
                                      base=np.mean(bas), after=np.mean(afs))
            r = res[(wname, cost)]
            tag = "FREE " if cost == 0 else "COST "
            print(f"[{wname:9} {tag}] mean_look={r['look']:.3f} best_life={r['life']:.0f} "
                  f"| reactive: base={r['base']:.3f} after_surprise={r['after']:.3f}")

    print("\n===== does metabolic look-cost induce SELECTIVE attention? =====")
    for w in ["text", "longrange"]:
        f, c = res[(w, 0.0)], res[(w, LOOK_COST)]
        print(f"  {w:9}: mean_look FREE {f['look']:.3f} -> COST {c['look']:.3f}   "
              f"(reactive uplift under cost: {c['after'] - c['base']:+.3f})")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(2); w = 0.35
    for k, (key, title) in enumerate([("look", "mean attention (1=always look)"),
                                      ("life", "best lifespan (survival)")]):
        free = [res[(wn, 0.0)][key] for wn in ["text", "longrange"]]
        cost = [res[(wn, LOOK_COST)][key] for wn in ["text", "longrange"]]
        ax[k].bar(x - w/2, free, w, label="look is free", color="#999")
        ax[k].bar(x + w/2, cost, w, label=f"look costs {LOOK_COST}", color="#1b9e77")
        ax[k].set_xticks(x); ax[k].set_xticklabels(["text", "long-range"])
        ax[k].set_title(title); ax[k].legend()
    fig.suptitle("ACTIVE PERCEPTION COST: does paying to look induce selective attention?")
    plt.tight_layout()
    out = _o.path.join(HERE, "attention_cost.png")
    plt.savefig(out, dpi=115); print("\nsaved", out)


if __name__ == "__main__":
    main()
