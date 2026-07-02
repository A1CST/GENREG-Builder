"""ABLATION BATTERY -- triple-check every headline finding. For each constraint we
run it ON vs OFF (ablated), PAIRED per seed, over N seeds, and report:
  ON mean+/-std, OFF mean+/-std, effect, CONSISTENCY (#seeds with expected sign),
  a SANITY metric (competence/survival -- is the effect a real capability or a
  selection-collapse artifact?), and a VERDICT.
VERDICT = ROBUST if consistency >= ceil(0.8 N) and |effect| > combined std; else
NOISY (sign flips) or NULL (effect ~ 0). Sanity flags if the 'capability' coincides
with a competence collapse (the neutral-drift trap) or survival saturation.
One self-contained harness (one trusted Org), so every condition is identical.
"""
# --- EEC path bootstrap ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os, math
import numpy as np
from multiprocessing import Pool
import run_matrix as RM
import mind as MIND
from evolve import EMBED, MUT_RATE, MUT_SCALE, POP_SIZE
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "120"))
SEG = int(os.environ.get("EEC_SEGLR", "400"))
SEGT = int(os.environ.get("EEC_SEGTEXT", "400"))
NSEED = int(os.environ.get("EEC_SEEDS", "6"))
MAXM, MINM = 48, 2
LR = TXT = LAT = None; TXTV = 2000; VLAT = 4


# ================= base organism =================
class Org:
    def __init__(self, V, rng):
        self.E = rng.normal(0, 0.1, (V, EMBED)).astype(np.float32)
        self.W_in = rng.normal(0, 1/np.sqrt(EMBED), (EMBED, MAXM)).astype(np.float32)
        self.W_rec = rng.normal(0, 1/np.sqrt(MAXM), (MAXM, MAXM)).astype(np.float32)
        self.b = np.zeros(MAXM, np.float32)
        self.W_out = rng.normal(0, 1/np.sqrt(MAXM), (MAXM, V)).astype(np.float32)
        self.b_out = np.zeros(V, np.float32)
        self.wl = rng.normal(0, 1/np.sqrt(MAXM), MAXM).astype(np.float32)
        self.bl = np.float32(rng.normal(0, 1.5))
        self.M = int(rng.integers(MINM, MAXM+1))

    def params(self): return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out, self.wl]

    def copy(self):
        g = Org.__new__(Org)
        (g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out, g.wl) = [p.copy() for p in self.params()]
        g.bl = self.bl; g.M = self.M; return g

    def mutate(self, rng):
        for p in self.params():
            m = rng.random(p.shape) < MUT_RATE
            p += m*rng.normal(0, 1, p.shape).astype(np.float32)*(MUT_SCALE*(np.abs(p)+1e-3))
        if rng.random() < MUT_RATE:
            self.bl += np.float32(rng.normal(0, MUT_SCALE)*(abs(self.bl)+0.1))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE*self.M)), MINM, MAXM))

    def states(self, emb, decay=1.0):
        M = self.M; Win = self.W_in[:, :M]; Wr = self.W_rec[:M, :M]; b = self.b[:M]
        drive = emb @ Win; T = emb.shape[0]; S = np.empty((T, M), np.float32); s = np.zeros(M, np.float32)
        for t in range(T):
            s = np.tanh(drive[t] + decay*(s @ Wr) + b); S[t] = s
        return S

    def states_gated(self, emb):
        M = self.M; Win = self.W_in[:, :M]; Wr = self.W_rec[:M, :M]; b = self.b[:M]; wl = self.wl[:M]
        drive = emb @ Win; T = emb.shape[0]; S = np.empty((T, M), np.float32); A = np.empty(T, np.float32)
        s = np.zeros(M, np.float32)
        for t in range(T):
            a = 1.0/(1.0+np.exp(-(s @ wl + self.bl))); s = np.tanh(a*drive[t] + (s @ Wr) + b); S[t] = s; A[t] = a
        return S, A


def emb_of(m, seg, mask=None, sigma=0.0, pat=None):
    e = m.E[seg].copy()
    if mask is not None: e[mask] = 0.0
    if sigma > 0 and pat is not None: e = e + pat*(sigma*float(e.std()))
    return e


def hits(m, seg, mask=None, sigma=0.0, pat=None, decay=1.0):
    S = m.states(emb_of(m, seg, mask, sigma, pat), decay)
    return S, (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == seg[1:]


def life(h, M, rent=0.01, E0=200.0):
    cum = np.cumsum(rent*M + (~h))
    return len(h) if cum[-1] < E0 else int(np.searchsorted(cum, E0)) + 1


def gain(m):
    Wr = m.W_rec[:m.M, :m.M]
    return float(np.max(np.abs(np.linalg.eigvals(Wr)))) if m.M else 0.0


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return 0.0 if n == 0 or x.sum() == 0 else float((2*np.arange(1, n+1)-n-1).dot(x)/(n*x.sum()))


# ================= shared evolve + measure for the recurrent-gain family =================
def evolve_gain(world, V, seed, rho, sigma, decay, seg=None):
    seg = seg or SEG; rng = np.random.default_rng(seed); pop = [Org(V, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        st = int(rng.integers(0, len(world)-seg-1)); s = world[st:st+seg]
        mask = (rng.random(seg) < rho) if rho > 0 else None
        pat = rng.normal(0, 1, (seg, EMBED)).astype(np.float32) if sigma else None
        lives = np.array([life(hits(m, s, mask, sigma, pat, decay)[1], m.M) for m in pop])
        pop = reproduce(pop, lives, rng)
    st = int(rng.integers(0, len(world)-seg-1)); s = world[st:st+seg]
    best = pop[int(np.argmax([life(hits(m, s)[1], m.M) for m in pop]))]
    return gain(best), float(hits(best, s)[1].mean())   # (capability=gain, sanity=competence)


# ================= ablation specs (each returns dict with on/off/sanity) =================
def ab_entropy(seed):
    on, son = evolve_gain(LR, 6, seed, 0, 0, 0.6); off, soff = evolve_gain(LR, 6, seed, 0, 0, 1.0)
    return dict(on=on, off=off, son=son, soff=soff)

def ab_occlusion(seed):
    on, son = evolve_gain(LR, 6, seed, 0.4, 0, 1.0); off, soff = evolve_gain(LR, 6, seed, 0, 0, 1.0)
    return dict(on=on, off=off, son=son, soff=soff)

def ab_occlusion_text(seed):   # null-control: occlusion should NOT grow gain in thin-signal text
    on, son = evolve_gain(TXT, TXTV, seed, 0.4, 0, 1.0, seg=SEGT)
    off, soff = evolve_gain(TXT, TXTV, seed, 0, 0, 1.0, seg=SEGT)
    return dict(on=on, off=off, son=son, soff=soff)

def ab_noise(seed):            # expect weak/null
    on, son = evolve_gain(LR, 6, seed, 0, 1.0, 1.0); off, soff = evolve_gain(LR, 6, seed, 0, 0, 1.0)
    return dict(on=on, off=off, son=son, soff=soff)

def _fertility(world, seed, repro):
    rng = np.random.default_rng(seed); pop = [Org(6, rng) for _ in range(POP_SIZE)]; off = np.zeros(POP_SIZE)
    E0f, FOOD, CHILD = 5.0, 1.0, 10.0
    for _ in range(GENS):
        st = int(rng.integers(0, len(world)-SEG-1)); s = world[st:st+SEG]
        if repro:
            sv, sp = [], []
            for m in pop:
                bank = E0f + np.cumsum(hits(m, s)[1]*FOOD - 0.01*m.M); ok = bank.min() > 0
                sv.append(ok); sp.append(float(bank[-1]-E0f) if ok else 0.0)
            o = np.where(sv, np.floor(np.array(sp)/CHILD).astype(int), 0)
            pool = [pop[i] for i in range(len(pop)) for _ in range(int(o[i]))]
            if not pool: new = [pop[int(np.argmax(sp))].copy() for _ in range(POP_SIZE)]
            else:
                idx = (rng.choice(len(pool), POP_SIZE, False) if len(pool) >= POP_SIZE
                       else np.concatenate([np.arange(len(pool)), rng.choice(len(pool), POP_SIZE-len(pool), True)]))
                new = [pool[i].copy() for i in idx]
            for c in new: c.mutate(rng)
            pop = new; off = o
        else:
            pop = reproduce(pop, np.array([life(hits(m, s)[1], m.M) for m in pop]), rng)
    st = int(rng.integers(0, len(world)-SEG-1)); s = world[st:st+SEG]
    comp = float(np.max([hits(m, s)[1].mean() for m in pop]))
    return (gini(off) if repro else 0.0), comp

def ab_repro(seed):
    on, son = _fertility(LR, seed, True); off, soff = _fertility(LR, seed, False)
    return dict(on=on, off=off, son=son, soff=soff)

def _scarcity(world, seed, scar):
    rng = np.random.default_rng(seed); pop = [Org(TXTV, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        st = int(rng.integers(0, len(world)-SEGT-1)); s = world[st:st+SEGT]
        prl, hl = [], []
        for m in pop:
            S = m.states(m.E[s]); pr = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
            hl.append(pr[:-1] == s[1:]); prl.append(pr[:-1])
        nxt = s[1:]; natural = np.bincount(nxt, minlength=TXTV).astype(float)
        if scar:
            demand = np.zeros(TXTV)
            for h, p in zip(hl, prl): np.add.at(demand, p[h], 1.0)
            scale = np.minimum(1.0, 3.0*np.maximum(natural, 1.0)/np.maximum(demand, 1.0))
        fits = []
        for h, p, m in zip(hl, prl, pop):
            cost = 0.01*m.M + (~h) + ((1.0-scale[p])*h if scar else 0.0)
            cum = np.cumsum(cost); fits.append(len(h) if cum[-1] < 200 else int(np.searchsorted(cum, 200))+1)
        pop = reproduce(pop, np.array(fits), rng)
    st = int(rng.integers(0, len(world)-SEGT-1)); s = world[st:st+SEGT]
    dom = [int(np.bincount((m.states(m.E[s]) @ m.W_out[:m.M, :] + m.b_out).argmax(1)).argmax()) for m in pop]
    comp = float(np.max([((m.states(m.E[s]) @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == s[1:]).mean() for m in pop]))
    return float(len(set(dom))), comp

def ab_scarcity(seed):
    on, son = _scarcity(TXT, seed, True); off, soff = _scarcity(TXT, seed, False)
    return dict(on=on, off=off, son=son, soff=soff)

def _rent(world, seed, rentmult):
    rng = np.random.default_rng(seed); pop = [Org(6, rng) for _ in range(POP_SIZE)]; rent = 0.01*rentmult
    for _ in range(GENS):
        st = int(rng.integers(0, len(world)-SEG-1)); s = world[st:st+SEG]; mask = rng.random(SEG) < 0.3
        lives = np.array([life(hits(m, s, mask)[1], m.M, rent) for m in pop])
        pop = reproduce(pop, lives, rng)
    st = int(rng.integers(0, len(world)-SEG-1)); s = world[st:st+SEG]
    best = pop[int(np.argmax([life(hits(m, s)[1], m.M, rent) for m in pop]))]
    return float(np.mean([o.M for o in pop])), float(hits(best, s)[1].mean())

def ab_rent(seed):    # capability we track = competence; sanity = M (does cheaper memory buy competence?)
    onM, on = _rent(LR, seed, 0.1); offM, off = _rent(LR, seed, 1.0)
    return dict(on=on, off=off, son=onM, soff=offM)   # son/soff carry mean M

def _perception(world, seed, lookcost):
    rng = np.random.default_rng(seed); pop = [Org(6, rng) for _ in range(POP_SIZE)]; E0p = 60.0
    for _ in range(GENS):
        st = int(rng.integers(0, len(world)-SEG-1)); s = world[st:st+SEG]; fits = []
        for m in pop:
            S, A = m.states_gated(m.E[s]); h = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == s[1:]
            cum = np.cumsum(0.01*m.M + (~h) + lookcost*A[:-1]); fits.append(len(h) if cum[-1] < E0p else int(np.searchsorted(cum, E0p))+1)
        pop = reproduce(pop, np.array(fits), rng)
    st = int(rng.integers(0, len(world)-SEG-1)); s = world[st:st+SEG]; looks, lives = [], []
    for m in pop:
        S, A = m.states_gated(m.E[s]); h = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == s[1:]
        cum = np.cumsum(0.01*m.M + (~h) + lookcost*A[:-1]); lv = len(h) if cum[-1] < E0p else int(np.searchsorted(cum, E0p))+1
        looks.append(float(A[:lv].mean())); lives.append(lv)
    bi = int(np.argmax(lives)); return looks[bi], float(lives[bi])

def ab_perception(seed):   # capability = mean look DROPS under cost; sanity = survival held
    on, son = _perception(LR, seed, 0.3); off, soff = _perception(LR, seed, 0.0)
    return dict(on=on, off=off, son=son, soff=soff)

def _nonstat(seed, shifting):
    rng = np.random.default_rng(seed); ALPH, BLK, NOISE = 6, 24, 0.05; START = 60.0
    def mk(r, n=60000):
        blk = r.integers(0, ALPH, BLK); st = np.tile(blk, n//BLK+1)[:n].copy()
        ml = r.random(n) < NOISE; st[ml] = r.integers(0, ALPH, ml.sum()); return st.astype(np.int32)
    world = mk(np.random.default_rng(1000+seed)); pop = [Org(ALPH, rng) for _ in range(POP_SIZE)]
    seg = 300; hist, shifts = [], []
    for gen in range(180):
        if shifting and gen > 0 and gen % 40 == 0:
            world = mk(np.random.default_rng(7000+seed*99+gen)); shifts.append(gen)
        st = int(rng.integers(0, len(world)-seg-1)); s = world[st:st+seg]
        lives = np.array([life(hits(m, s)[1], m.M, 0.01, START) for m in pop]); hist.append(int(lives.max()))
        pop = reproduce(pop, lives, rng)
    st = int(rng.integers(0, len(world)-seg-1)); s = world[st:st+seg]
    dom = [int(np.bincount((m.states(m.E[s]) @ m.W_out[:m.M, :] + m.b_out).argmax(1)).argmax()) for m in pop]
    rec = 1.0
    if shifting and shifts:
        rs = [np.mean(hist[g+1:g+6])/max(1e-9, np.mean(hist[g-3:g])) for g in shifts if g+6 < len(hist)]
        rec = float(np.mean(rs)) if rs else 1.0
    return float(len(set(dom))), rec

def ab_nonstat(seed):   # capability = diversity higher under shifting; sanity = recovery ratio
    on, son = _nonstat(seed, True); off, soff = _nonstat(seed, False)
    return dict(on=on, off=off, son=son, soff=soff)

def _comm(seed, sever):
    rng = np.random.default_rng(seed); V = VLAT
    class P:
        def __init__(s): s.Ws = rng.normal(0, 1, (V, V)).astype(np.float32); s.Wl = rng.normal(0, 1, (V, V)).astype(np.float32)
        def copy(s):
            g = P.__new__(P); g.Ws = s.Ws.copy(); g.Wl = s.Wl.copy(); return g
        def mutate(s, r):
            for p in (s.Ws, s.Wl):
                m = r.random(p.shape) < MUT_RATE; p += m*r.normal(0, 1, p.shape).astype(np.float32)*(MUT_SCALE*(np.abs(p)+0.2))
    pop = [P() for _ in range(POP_SIZE)]
    def lifep(pp, c, sv):
        sig = pp.Ws[c].argmax(1)
        if sv is not None: sig = sv.integers(0, V, len(c))
        h = pp.Wl[sig].argmax(1) == c; cum = np.cumsum((~h).astype(float))
        return (len(c) if cum[-1] < 120 else int(np.searchsorted(cum, 120))+1), h
    for _ in range(GENS):
        st = int(rng.integers(0, len(LAT)-SEG-1)); c = LAT[st:st+SEG]
        lives = np.array([lifep(m, c, None)[0] for m in pop]); pop = reproduce(pop, lives, rng)
    st = int(rng.integers(0, len(LAT)-SEG-1)); c = LAT[st:st+SEG]
    best = max(pop, key=lambda m: lifep(m, c, None)[0])
    sv = np.random.default_rng(seed+5) if sever else None
    return float(lifep(best, c, sv)[1].mean()), 0.0

def ab_comm(seed):   # capability = accuracy with intact channel >> severed; off = severed
    on, _ = _comm(seed, False); off, _ = _comm(seed, True)
    return dict(on=on, off=off, son=on, soff=off)


SPECS = {  # name: (fn, capability-label, higher_on_is_effect, sanity-label, expectation)
    "ENTROPY->gain":        (ab_entropy, "recurrent gain", True, "competence", "ON>OFF"),
    "OCCLUSION->gain(LR)":  (ab_occlusion, "recurrent gain", True, "competence", "ON>OFF"),
    "OCCLUSION->gain(TEXT)":(ab_occlusion_text, "recurrent gain", True, "competence", "NULL (world ctrl)"),
    "NOISE->gain":          (ab_noise, "recurrent gain", True, "competence", "NULL/weak"),
    "REPRO-COST->ecosystem":(ab_repro, "fertility Gini", True, "competence", "ON>OFF"),
    "SCARCITY->diversity":  (ab_scarcity, "uniq-dominant", True, "competence", "ON>OFF"),
    "RENT(cheap)->compet":  (ab_rent, "competence", True, "mean M", "ON>OFF"),
    "PERCEPTION->economy":  (ab_perception, "mean look", False, "survival", "ON<OFF (look down)"),
    "NON-STAT->diversity":  (ab_nonstat, "uniq-dominant", True, "recovery", "ON>OFF"),
    "COMMUNICATION->acc":   (ab_comm, "listener acc", True, "(severed=off)", "ON>>OFF"),
}


def cell(task):
    name, seed = task
    return (name, seed, SPECS[name][0](seed))


def main():
    global LR, TXT, LAT
    rng = np.random.default_rng(999)
    LR = RM.world_longrange(rng)[0]; TXT = RM.world_text()[0]
    # sticky latent world for communication
    n = 100000; LAT = np.empty(n, np.int32); sw = rng.random(n) < 0.05; LAT[0] = rng.integers(0, VLAT)
    for t in range(1, n): LAT[t] = rng.integers(0, VLAT) if sw[t] else LAT[t-1]
    tasks = [(nm, s) for nm in SPECS for s in range(NSEED)]
    print(f"ABLATION BATTERY: {len(SPECS)} constraints x {NSEED} seeds = {len(tasks)} cells "
          f"(gens={GENS}, paired ON/OFF)", flush=True)
    with Pool(min(18, len(tasks))) as p:
        res = p.map(cell, tasks, chunksize=1)
    agg = {}
    for name, seed, d in res:
        agg.setdefault(name, []).append(d)

    print(f"\n{'constraint':24} {'ON':>14} {'OFF':>14} {'effect':>9} {'consist':>8}  verdict   sanity")
    print("-"*98)
    for name, (fn, cap, hib, san, expect) in SPECS.items():
        rows = agg[name]
        on = np.array([r["on"] for r in rows]); off = np.array([r["off"] for r in rows])
        eff = on - off if hib else off - on
        consist = int(np.sum(eff > 0)); n = len(rows)
        sd = math.sqrt(on.std()**2 + off.std()**2) + 1e-9
        is_null = "NULL" in expect or "weak" in expect
        robust = consist >= math.ceil(0.8*n) and abs(eff.mean()) > sd
        if is_null:
            verdict = "NULL-OK" if abs(eff.mean()) < sd or consist <= n-math.ceil(0.8*n)+1 else "UNEXPECTED"
        else:
            verdict = "ROBUST" if robust else ("WEAK" if consist >= n/2 else "NOISE")
        son = np.mean([r["son"] for r in rows]); soff = np.mean([r["soff"] for r in rows])
        flag = ""
        if not is_null and verdict == "ROBUST" and son < 0.3 and "competence" in san:
            flag = " <-CHECK collapse?"
        print(f"{name:24} {on.mean():>6.3f}+/-{on.std():<5.3f} {off.mean():>6.3f}+/-{off.std():<5.3f} "
              f"{eff.mean():>+8.3f} {consist:>4}/{n}  {verdict:8} {san}={son:.2f}/{soff:.2f}{flag}")
    print("\nROBUST = consistent sign (>=80% seeds) AND effect > combined std.  "
          "NULL-OK = expected null confirmed.  sanity shows ON/OFF of the control metric.")


if __name__ == "__main__":
    main()
