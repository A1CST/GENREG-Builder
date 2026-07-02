"""REMOVE THE MEMORY CAP -- is M cap-limited or rent-limited?

M (memory size) is evolvable but capped at MAX_M=48. In every converged run it
settled at 2-6, far below the cap -> the suspicion is that the cap NEVER binds;
RENT (energy per memory unit per step) sets the equilibrium, against the energy
budget. The cap is a hand-imposed ceiling; the LAW should set the structure.

2x2 in a memory-hungry world (long-range + occlusion rho=0.3):
  CAP   in {48, 160}      -- does raising the ceiling move converged M?
  RENT  in {normal, /10}  -- the real lever: if memory is cheaper, does M grow,
                             and does that BUY competence or just BLOAT?

Read converged mean M (and max), competence (clean hit-rate), recurrent gain.
Prediction: cap is non-binding (M flat across cap at fixed rent); cheaper rent
raises M -- competence rises only if memory was starved, else it is Occam bloat.
"""
# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os, itertools
import numpy as np
from multiprocessing import Pool
import run_matrix as RM
from evolve import EMBED, MUT_RATE, MUT_SCALE, POP_SIZE
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "120"))
SEG = int(os.environ.get("EEC_SEGLR", "400"))
START = float(os.environ.get("EEC_E0", "200"))
RHO = 0.3
BASE_RENT = 0.01
WORLD = None
MIN_M = 2


class Org:
    def __init__(self, V, max_m, rng):
        self.max_m = max_m
        self.E = rng.normal(0, 0.1, (V, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1 / np.sqrt(EMBED), (EMBED, max_m)).astype(np.float32)
        self.W_rec = rng.normal(0, 1 / np.sqrt(max_m), (max_m, max_m)).astype(np.float32)
        self.b = np.zeros(max_m, dtype=np.float32)
        self.W_out = rng.normal(0, 1 / np.sqrt(max_m), (max_m, V)).astype(np.float32)
        self.b_out = np.zeros(V, dtype=np.float32)
        self.M = int(rng.integers(MIN_M, max_m + 1))

    def params(self):
        return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out]

    def copy(self):
        g = Org.__new__(Org); g.max_m = self.max_m
        g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out = [p.copy() for p in self.params()]
        g.M = self.M; return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            p += mask * rng.normal(0, 1, p.shape).astype(np.float32) * (MUT_SCALE * (np.abs(p) + 1e-3))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE * self.M)), MIN_M, self.max_m))

    def run_states(self, emb):
        M = self.M
        Win = self.W_in[:, :M]; Wrec = self.W_rec[:M, :M]; b = self.b[:M]
        drive = emb @ Win; T = emb.shape[0]
        S = np.empty((T, M), dtype=np.float32); s = np.zeros(M, dtype=np.float32)
        for t in range(T):
            s = np.tanh(drive[t] + s @ Wrec + b); S[t] = s
        return S


def hits_of(m, seg, mask):
    e = m.E[seg].copy()
    if mask is not None:
        e[mask] = 0.0
    S = m.run_states(e)
    return (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == seg[1:]


def life_of(m, seg, mask, rent):
    cum = np.cumsum(rent * m.M + (~hits_of(m, seg, mask)))
    return len(seg) - 1 if cum[-1] < START else int(np.searchsorted(cum, START)) + 1


def gain_of(m):
    Wr = m.W_rec[:m.M, :m.M]
    return float(np.max(np.abs(np.linalg.eigvals(Wr)))) if m.M else 0.0


def cell(task):
    cap, rentmult, seed = task
    rent = BASE_RENT * rentmult
    rng = np.random.default_rng(seed)
    pop = [Org(6, cap, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        st = int(rng.integers(0, len(WORLD) - SEG - 1)); seg = WORLD[st:st + SEG]
        mask = rng.random(SEG) < RHO
        lives = np.array([life_of(m, seg, mask, rent) for m in pop])
        pop = reproduce(pop, lives, rng)
    # measure on clean held-out
    st = int(rng.integers(0, len(WORLD) - SEG - 1)); seg = WORLD[st:st + SEG]
    lives = np.array([life_of(m, seg, None, rent) for m in pop])
    bi = int(np.argmax(lives)); best = pop[bi]
    return dict(cap=cap, rentmult=rentmult, meanM=float(np.mean([o.M for o in pop])),
                maxM=int(max(o.M for o in pop)), bestM=int(best.M),
                comp=float(hits_of(best, seg, None).mean()), gain=gain_of(best))


def main():
    global WORLD
    WORLD = RM.world_longrange(np.random.default_rng(999))[0]
    seeds = list(range(int(os.environ.get("EEC_SEEDS", "3"))))
    tasks = [(cap, rm, sd) for cap, rm in itertools.product([48, 160], [1.0, 0.1]) for sd in seeds]
    print(f"uncap test: cap x rent x {len(seeds)} seeds, long-range+occ{RHO}, E0={START}, "
          f"seg={SEG}, gens={GENS}", flush=True)
    with Pool(min(12, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for r in res:
        agg.setdefault((r["cap"], r["rentmult"]), []).append(r)
    print("\n cap   rent      meanM   maxM   bestM   competence   gain")
    for cap in [48, 160]:
        for rm in [1.0, 0.1]:
            rows = agg[(cap, rm)]
            mm = {k: np.mean([d[k] for d in rows]) for k in ("meanM", "maxM", "bestM", "comp", "gain")}
            tag = "normal" if rm == 1.0 else "/10   "
            print(f"  {cap:>3}  {tag}    {mm['meanM']:>5.1f}  {mm['maxM']:>4.0f}   {mm['bestM']:>4.0f}    "
                  f"{mm['comp']:.3f}        {mm['gain']:.3f}")

    print("\n===== READINGS =====")
    for rm in [1.0, 0.1]:
        m48 = np.mean([d["meanM"] for d in agg[(48, rm)]])
        m160 = np.mean([d["meanM"] for d in agg[(160, rm)]])
        tag = "normal" if rm == 1.0 else "/10"
        print(f"  rent {tag:6}: meanM cap48={m48:.1f} vs cap160={m160:.1f}  "
              f"({'cap BINDS' if m160 > 1.4 * m48 + 1 else 'cap NON-binding -> rent-limited'})")
    cn = np.mean([d["comp"] for d in agg[(160, 1.0)]]); cl = np.mean([d["comp"] for d in agg[(160, 0.1)]])
    Mn = np.mean([d["meanM"] for d in agg[(160, 1.0)]]); Ml = np.mean([d["meanM"] for d in agg[(160, 0.1)]])
    print(f"  cheaper rent (cap160): M {Mn:.1f}->{Ml:.1f}, competence {cn:.3f}->{cl:.3f}  "
          f"({'BUYS competence' if cl > cn + 0.02 else 'BLOAT (more M, ~same competence)'})")


if __name__ == "__main__":
    main()
