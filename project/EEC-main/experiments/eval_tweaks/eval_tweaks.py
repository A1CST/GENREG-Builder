"""Three cheap eval-tweak axes (on/off, long-range, read internals not accuracy):

SELF-MONITORING (faults): each step a random fraction of state units is disabled.
  Capability forced: REDUNDANCY (a fault-tolerant distributed representation).
  Test: organism evolved WITH faults should predict-under-fault better than a
  control evolved without faults but tested under fault.

ASYMMETRY (catastrophic threshold): when energy runs low (< E_crit), every miss
  costs 3x. A metabolic nonlinearity (law, not reward shaping).
  Capability: risk-aversion / competence pushed up to stay out of the danger zone.

COMMITMENT (state-change cost): each step costs lambda*mean|s_t - s_{t-1}| energy.
  Capability: SMOOTH/stable internal dynamics (momentum). Test: does it reduce the
  state-change magnitude while holding competence?
"""
# --- EEC path bootstrap ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
from multiprocessing import Pool
import run_matrix as RM
from evolve import EMBED, MUT_RATE, MUT_SCALE, POP_SIZE
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "130"))
SEG = int(os.environ.get("EEC_SEGLR", "400"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "3"))))
MAXM, MINM, E0 = 48, 2, 200.0
FAULT_F = 0.3; ECRIT = 60.0; CATA = 3.0; COMMIT_L = 0.5
LR = None


class Org:
    def __init__(self, V, rng):
        self.E = rng.normal(0, 0.1, (V, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1/np.sqrt(EMBED), (EMBED, MAXM)).astype(np.float32)
        self.W_rec = rng.normal(0, 1/np.sqrt(MAXM), (MAXM, MAXM)).astype(np.float32)
        self.b = np.zeros(MAXM, dtype=np.float32)
        self.W_out = rng.normal(0, 1/np.sqrt(MAXM), (MAXM, V)).astype(np.float32)
        self.b_out = np.zeros(V, dtype=np.float32)
        self.M = int(rng.integers(MINM, MAXM+1))

    def params(self): return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out]

    def copy(self):
        g = Org.__new__(Org)
        g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out = [p.copy() for p in self.params()]
        g.M = self.M; return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            p += mask*rng.normal(0, 1, p.shape).astype(np.float32)*(MUT_SCALE*(np.abs(p)+1e-3))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE*self.M)), MINM, MAXM))

    def states(self, emb, fault=None):
        M = self.M; Win = self.W_in[:, :M]; Wr = self.W_rec[:M, :M]; b = self.b[:M]
        drive = emb @ Win; T = emb.shape[0]
        S = np.empty((T, M), np.float32); s = np.zeros(M, np.float32)
        for t in range(T):
            s = np.tanh(drive[t] + s @ Wr + b)
            if fault is not None:
                s = s * fault[t, :M]
            S[t] = s
        return S


def preds_hits(m, seg, fault=None):
    S = m.states(m.E[seg], fault)
    pr = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
    return S, pr[:-1] == seg[1:]


def life_plain(h, M):
    cum = np.cumsum(0.01*M + (~h))
    return len(h) if cum[-1] < E0 else int(np.searchsorted(cum, E0)) + 1


def evolve(seed, axis, on):
    rng = np.random.default_rng(seed)
    pop = [Org(6, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
        fault = (rng.random((SEG, MAXM)) > FAULT_F).astype(np.float32) if (axis == "fault" and on) else None
        fits = []
        for m in pop:
            S, h = preds_hits(m, seg, fault)
            if axis == "asymmetry" and on:
                base = 0.01*m.M + (~h)
                cum = np.cumsum(base); danger = cum > (E0 - ECRIT)
                cost = 0.01*m.M + (~h)*np.where(danger, CATA, 1.0)
                cc = np.cumsum(cost); life = len(h) if cc[-1] < E0 else int(np.searchsorted(cc, E0))+1
            elif axis == "commitment" and on:
                ds = np.concatenate([[0.0], np.abs(np.diff(S, axis=0)).mean(1)])
                cost = 0.01*m.M + (~h) + COMMIT_L*ds[:-1]
                cc = np.cumsum(cost); life = len(h) if cc[-1] < E0 else int(np.searchsorted(cc, E0))+1
            else:
                life = life_plain(h, m.M)
            fits.append(life)
        pop = reproduce(pop, np.array(fits), rng)
    return pop, rng


def cell(task):
    axis, on, seed = task
    pop, rng = evolve(seed, axis, on)
    st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
    # pick best by clean competence
    best = max(pop, key=lambda m: preds_hits(m, seg)[1].mean())
    S, h = preds_hits(best, seg)
    comp = float(h.mean())
    fault = (rng.random((SEG, MAXM)) > FAULT_F).astype(np.float32)
    comp_fault = float(preds_hits(best, seg, fault)[1].mean())
    dsmag = float(np.abs(np.diff(S, axis=0)).mean())
    return (axis, on, dict(comp=comp, comp_fault=comp_fault, dsmag=dsmag, M=best.M))


def main():
    global LR
    LR = RM.world_longrange(np.random.default_rng(999))[0]
    axes = ["fault", "asymmetry", "commitment"]
    tasks = [(a, on, s) for a in axes for on in [False, True] for s in SEEDS]
    print(f"eval-tweaks: 3 axes x on/off x {len(SEEDS)} seeds, long-range (gens={GENS})", flush=True)
    with Pool(min(18, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for a, on, d in res:
        agg.setdefault((a, on), []).append(d)
    def mean(a, on, k): return float(np.mean([d[k] for d in agg[(a, on)]]))

    print("\n===== SELF-MONITORING (faults -> redundancy) =====")
    print(f"  comp UNDER fault:  control(evolved no-fault) {mean('fault',False,'comp_fault'):.3f}  "
          f"-> fault-evolved {mean('fault',True,'comp_fault'):.3f}  "
          f"({'REDUNDANCY emerged' if mean('fault',True,'comp_fault') > mean('fault',False,'comp_fault')+0.03 else 'no robustness gain'})")
    print(f"  (clean comp: control {mean('fault',False,'comp'):.3f} / fault-evolved {mean('fault',True,'comp'):.3f})")

    print("\n===== ASYMMETRY (catastrophic threshold -> risk aversion) =====")
    print(f"  competence: off {mean('asymmetry',False,'comp'):.3f} -> on {mean('asymmetry',True,'comp'):.3f}  "
          f"({'pushed UP (risk-averse)' if mean('asymmetry',True,'comp') > mean('asymmetry',False,'comp')+0.02 else 'no clear effect'})")

    print("\n===== COMMITMENT (state-change cost -> smooth dynamics) =====")
    print(f"  mean|delta s|: off {mean('commitment',False,'dsmag'):.3f} -> on {mean('commitment',True,'dsmag'):.3f}  "
          f"({'SMOOTHER' if mean('commitment',True,'dsmag') < mean('commitment',False,'dsmag')-0.02 else 'no smoothing'})")
    print(f"  competence:   off {mean('commitment',False,'comp'):.3f} -> on {mean('commitment',True,'comp'):.3f}")


if __name__ == "__main__":
    main()
