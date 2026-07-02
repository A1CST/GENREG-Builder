"""PARETO FRONTIER across EVERY constraint: sweep each constraint's STRENGTH knob
and find where more pressure stops buying capability (peak / knee / collapse).

Each constraint taxes a channel and grows a capability; turn the dial up and the
capability rises, plateaus, then often collapses or bloats. The frontier is the
strength that maximizes capability per unit cost.

  constraint        knob              world      capability        cost / failure mode
  MEMORY-RENT       rent coef         LR+occ     competence        M (bloat if too cheap)
  OCCLUSION         rho               LR         recurrent gain    unlearnable past peak
  NOISE             sigma             LR         recurrent gain    (weak driver)
  ENTROPY           1-decay           LR         recurrent gain    can't hold if too high
  MORTALITY         hazard lam        LR         competence        no selection if lam=0
  REPRODUCTION      child cost        LR         fertility/comp    repro failure if too dear
  SCARCITY          carrying cap mult TEXT       diversity         monoculture / collapse
  PERCEPTION        look cost         LR(tight)  economy(look down) blind if too dear

Read internals on clean segments; selection is pure survival/fertility.
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
import run_matrix as RM
from evolve import EMBED, MUT_RATE, MUT_SCALE, POP_SIZE
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "120"))
SEG = int(os.environ.get("EEC_SEGLR", "400"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "2"))))
MAXM, MINM = 64, 2
LR = TXT = None; TXTV = 2000

LEVELS = {
    "rent":       [0.04, 0.02, 0.01, 0.005, 0.002, 0.001],
    "occlusion":  [0.0, 0.2, 0.4, 0.6, 0.8],
    "noise":      [0.0, 0.5, 1.0, 2.0, 3.0],
    "entropy":    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],   # = 1 - decay
    "mortality":  [0.0, 3.0, 8.0, 15.0, 30.0],
    "repro":      [2.0, 5.0, 10.0, 20.0, 40.0],
    "scarcity":   [1.0, 2.0, 4.0, 8.0, 16.0],
    "perception": [0.0, 0.1, 0.3, 0.6, 1.0],
}


class Org:
    def __init__(self, V, rng):
        self.E = rng.normal(0, 0.1, (V, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1/np.sqrt(EMBED), (EMBED, MAXM)).astype(np.float32)
        self.W_rec = rng.normal(0, 1/np.sqrt(MAXM), (MAXM, MAXM)).astype(np.float32)
        self.b = np.zeros(MAXM, dtype=np.float32)
        self.W_out = rng.normal(0, 1/np.sqrt(MAXM), (MAXM, V)).astype(np.float32)
        self.b_out = np.zeros(V, dtype=np.float32)
        self.wl = rng.normal(0, 1/np.sqrt(MAXM), MAXM).astype(np.float32)
        self.bl = np.float32(rng.normal(0, 1.5))
        self.M = int(rng.integers(MINM, MAXM + 1))

    def params(self):
        return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out, self.wl]

    def copy(self):
        g = Org.__new__(Org)
        (g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out, g.wl) = [p.copy() for p in self.params()]
        g.bl = self.bl; g.M = self.M; return g

    def mutate(self, rng):
        for p in self.params():
            mask = rng.random(p.shape) < MUT_RATE
            p += mask * rng.normal(0, 1, p.shape).astype(np.float32) * (MUT_SCALE*(np.abs(p)+1e-3))
        if rng.random() < MUT_RATE:
            self.bl += np.float32(rng.normal(0, MUT_SCALE)*(abs(self.bl)+0.1))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE*self.M)), MINM, MAXM))

    def states(self, emb, decay=1.0):
        M = self.M; Win = self.W_in[:, :M]; Wr = self.W_rec[:M, :M]; b = self.b[:M]
        drive = emb @ Win; T = emb.shape[0]
        S = np.empty((T, M), np.float32); s = np.zeros(M, np.float32)
        for t in range(T):
            s = np.tanh(drive[t] + decay*(s @ Wr) + b); S[t] = s
        return S

    def states_gated(self, emb):
        M = self.M; Win = self.W_in[:, :M]; Wr = self.W_rec[:M, :M]; b = self.b[:M]
        wl = self.wl[:M]; bl = self.bl; drive = emb @ Win; T = emb.shape[0]
        S = np.empty((T, M), np.float32); A = np.empty(T, np.float32); s = np.zeros(M, np.float32)
        for t in range(T):
            a = 1.0/(1.0+np.exp(-(s @ wl + bl)))
            s = np.tanh(a*drive[t] + (s @ Wr) + b); S[t] = s; A[t] = a
        return S, A


def _emb(m, seg, mask=None, sigma=0.0, pat=None):
    e = m.E[seg].copy()
    if mask is not None:
        e[mask] = 0.0
    if sigma > 0 and pat is not None:
        e = e + pat*(sigma*float(e.std()))
    return e


def hits(m, seg, mask=None, sigma=0.0, pat=None, decay=1.0):
    S = m.states(_emb(m, seg, mask, sigma, pat), decay)
    return (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == seg[1:]


def life(h, M, rent, E0=200.0):
    cum = np.cumsum(rent*M + (~h))
    return len(h) if cum[-1] < E0 else int(np.searchsorted(cum, E0)) + 1


def gain(m):
    Wr = m.W_rec[:m.M, :m.M]
    return float(np.max(np.abs(np.linalg.eigvals(Wr)))) if m.M else 0.0


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return 0.0 if n == 0 or x.sum() == 0 else float((2*np.arange(1, n+1)-n-1).dot(x)/(n*x.sum()))


# ---------- per-constraint evolve+measure ----------
def evolve_generic(seed, fitfn, V, world):
    rng = np.random.default_rng(seed)
    pop = [Org(V, rng) for _ in range(POP_SIZE)]
    extra = None
    for _ in range(GENS):
        st = int(rng.integers(0, len(world)-SEG-1)); seg = world[st:st+SEG]
        fits, extra = fitfn(pop, seg, rng)
        pop = reproduce(pop, fits, rng)
    return pop, rng


def cell(task):
    con, lvl, seed = task
    if con == "rent":
        def f(pop, seg, rng):
            mask = rng.random(SEG) < 0.3
            return np.array([life(hits(m, seg, mask), m.M, lvl) for m in pop]), None
        pop, rng = evolve_generic(seed, f, 6, LR)
        st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
        best = pop[int(np.argmax([life(hits(m, seg), m.M, lvl) for m in pop]))]
        return (con, lvl, dict(cap=float(hits(best, seg).mean()), cost=float(np.mean([o.M for o in pop]))))
    if con in ("occlusion", "noise", "entropy"):
        def f(pop, seg, rng):
            mask = (rng.random(SEG) < lvl) if con == "occlusion" else None
            sig = lvl if con == "noise" else 0.0
            pat = rng.normal(0, 1, (SEG, EMBED)).astype(np.float32) if sig else None
            dec = 1.0 - lvl if con == "entropy" else 1.0
            return np.array([life(hits(m, seg, mask, sig, pat, dec), m.M, 0.01) for m in pop]), None
        pop, rng = evolve_generic(seed, f, 6, LR)
        st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
        best = pop[int(np.argmax([life(hits(m, seg), m.M, 0.01) for m in pop]))]
        return (con, lvl, dict(cap=gain(best), cost=float(best.M)))
    if con == "mortality":
        H0 = 0.0015
        def f(pop, seg, rng):
            fits = []
            for m in pop:
                hz = H0*(1.0 + lvl*(~hits(m, seg)))
                sb = np.concatenate([[1.0], np.cumprod(1.0-hz)[:-1]])
                fits.append(sb.sum())
            return np.array(fits), None
        pop, rng = evolve_generic(seed, f, 6, LR)
        st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
        comps = [float(hits(m, seg).mean()) for m in pop]
        return (con, lvl, dict(cap=float(np.max(comps)), cost=lvl))
    if con == "repro":
        E0f, FOOD = 5.0, 1.0
        def f(pop, seg, rng):
            sv, sp = [], []
            for m in pop:
                bank = E0f + np.cumsum(hits(m, seg)*FOOD - 0.01*m.M)
                ok = bank.min() > 0; sv.append(ok); sp.append(float(bank[-1]-E0f) if ok else 0.0)
            off = np.where(sv, np.floor(np.array(sp)/lvl).astype(int), 0)
            pool = [pop[i] for i in range(len(pop)) for _ in range(int(off[i]))]
            if not pool:
                new = [pop[int(np.argmax(sp))].copy() for _ in range(POP_SIZE)]
            else:
                idx = (rng.choice(len(pool), POP_SIZE, False) if len(pool) >= POP_SIZE
                       else np.concatenate([np.arange(len(pool)), rng.choice(len(pool), POP_SIZE-len(pool), True)]))
                new = [pool[i].copy() for i in idx]
            for c in new:
                c.mutate(rng)
            return np.zeros(POP_SIZE), (new, off)
        # custom loop to carry the new population from fitfn extra
        rng = np.random.default_rng(seed); pop = [Org(6, rng) for _ in range(POP_SIZE)]; off = np.zeros(POP_SIZE)
        for _ in range(GENS):
            st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
            _, (pop, off) = f(pop, seg, rng)
        st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
        comps = [float(hits(m, seg).mean()) for m in pop]
        return (con, lvl, dict(cap=gini(off), cost=float(np.max(comps))))
    if con == "scarcity":
        def f(pop, seg, rng):
            preds_list, hits_list = [], []
            for m in pop:
                S = m.states(m.E[seg]); pr = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
                hits_list.append(pr[:-1] == seg[1:]); preds_list.append(pr[:-1])
            nxt = seg[1:]; natural = np.bincount(nxt, minlength=TXTV).astype(float)
            demand = np.zeros(TXTV)
            for h, p in zip(hits_list, preds_list):
                np.add.at(demand, p[h], 1.0)
            cap = lvl*np.maximum(natural, 1.0); scale = np.minimum(1.0, cap/np.maximum(demand, 1.0))
            fits = []
            for h, p, m in zip(hits_list, preds_list, pop):
                cost = 0.01*m.M + (~h) + (1.0-scale[p])*h
                cum = np.cumsum(cost)
                fits.append(len(h) if cum[-1] < 200 else int(np.searchsorted(cum, 200))+1)
            return np.array(fits), None
        pop, rng = evolve_generic(seed, f, TXTV, TXT)
        st = int(rng.integers(0, len(TXT)-SEG-1)); seg = TXT[st:st+SEG]
        dom = [int(np.bincount((m.states(m.E[seg]) @ m.W_out[:m.M, :] + m.b_out).argmax(1)).argmax()) for m in pop]
        return (con, lvl, dict(cap=float(len(set(dom))), cost=lvl))
    if con == "perception":
        E0p = 60.0
        def f(pop, seg, rng):
            fits = []
            for m in pop:
                S, A = m.states_gated(m.E[seg])
                h = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == seg[1:]
                cum = np.cumsum(0.01*m.M + (~h) + lvl*A[:-1])
                fits.append(len(h) if cum[-1] < E0p else int(np.searchsorted(cum, E0p))+1)
            return np.array(fits), None
        pop, rng = evolve_generic(seed, f, 6, LR)
        st = int(rng.integers(0, len(LR)-SEG-1)); seg = LR[st:st+SEG]
        looks, lives = [], []
        for m in pop:
            S, A = m.states_gated(m.E[seg])
            h = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == seg[1:]
            cum = np.cumsum(0.01*m.M + (~h) + lvl*A[:-1])
            lv = len(h) if cum[-1] < E0p else int(np.searchsorted(cum, E0p))+1
            looks.append(float(A[:lv].mean())); lives.append(lv)
        bi = int(np.argmax(lives))
        return (con, lvl, dict(cap=looks[bi], cost=float(lives[bi])))


def main():
    global LR, TXT
    rng0 = np.random.default_rng(999)
    LR = RM.world_longrange(rng0)[0]
    TXT = RM.world_text()[0]
    tasks = [(c, lvl, s) for c, lvls in LEVELS.items() for lvl in lvls for s in SEEDS]
    print(f"Pareto sweep: {len(LEVELS)} constraints, {len(tasks)} cells (gens={GENS} seg={SEG})", flush=True)
    with Pool(min(18, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for con, lvl, d in res:
        agg.setdefault(con, {}).setdefault(lvl, []).append(d)
    summary = {}
    META = {  # (capability name, cost/note, higher_cap_better)
        "rent": ("competence", "mean M", True), "occlusion": ("recurrent gain", "M", True),
        "noise": ("recurrent gain", "M", True), "entropy": ("recurrent gain", "M", True),
        "mortality": ("competence", "lam", True), "repro": ("fertility Gini", "competence", True),
        "scarcity": ("diversity", "cap mult", True), "perception": ("mean look (down=econ)", "lifespan", False)}
    fig, ax = plt.subplots(2, 4, figsize=(18, 9)); fig.subplots_adjust(hspace=0.35, wspace=0.32, top=0.9)
    fig.suptitle("Pareto frontiers: capability vs constraint strength (peak / knee = optimal price)",
                 fontsize=15, weight="bold")
    for k, con in enumerate(LEVELS):
        a = ax[k//4, k%4]
        lvls = LEVELS[con]
        cap = [np.mean([d["cap"] for d in agg[con][lv]]) for lv in lvls]
        cost = [np.mean([d["cost"] for d in agg[con][lv]]) for lv in lvls]
        capn, note, hib = META[con]
        a.plot(lvls, cap, "o-", color="#1b5e9e", lw=2)
        opt = int(np.argmax(cap)) if hib else int(np.argmin(cap))
        a.plot(lvls[opt], cap[opt], "r*", ms=16)
        a.set_title(f"{con.upper()}", fontsize=11, weight="bold")
        a.set_xlabel(f"strength ({con})"); a.set_ylabel(capn); a.grid(alpha=.3)
        a.text(0.5, -0.32, f"opt @ {lvls[opt]} | {note}={cost[opt]:.2f}", transform=a.transAxes,
               ha="center", fontsize=8, color="#555")
        summary[con] = (lvls[opt], cap[opt], cost[opt], capn, note)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pareto_fronts.png")
    plt.savefig(out, dpi=110); print("saved", out)

    print("\n===== PARETO SUMMARY (optimal strength per constraint) =====")
    for con, (lv, cp, ct, capn, note) in summary.items():
        print(f"  {con:11}: optimal strength {lv:<6} -> {capn} {cp:.3f}  ({note} {ct:.2f})")


if __name__ == "__main__":
    main()
