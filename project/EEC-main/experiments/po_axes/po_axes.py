"""PO AS AXIS-COUNT: measure the minimal set of AXES (one constraint each) that
forces a target organism. PO = how many axes you must cover to collapse the space
to the behaviour you want.

Lattice over 3 ORTHOGONAL axes (chosen non-interfering; survival/ENERGY always on
as the base axis):
  OBSERVATION  = occlusion rho=0.4      -> trait: memory (recurrent gain)
  SELECTION    = reproduction-cost      -> trait: fertility ecosystem (offspring Gini)
  PARSIMONY    = rent x3                 -> trait: small memory M (efficiency)

For each of the 2^3 = 8 axis subsets we evolve and read the TRAIT VECTOR
(competence, gain, Gini, M). A target organism is a trait bundle; its PO = the
size of the minimal axis subset (plus the survival base) that produces every trait.
Ablating any axis in the minimal set must lose its trait -> the set is minimal.
"""
# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os, itertools
import numpy as np
from multiprocessing import Pool
import run_matrix as RM
import mind as MIND
from mind import Mind, RENT, START_ENERGY, reproduce
from evolve import POP_SIZE

GENS = int(os.environ.get("EEC_MGENS", "150"))
SEG = int(os.environ.get("EEC_SEGLR", "600"))
E0, FOOD, CHILD = 5.0, 1.0, 10.0
RHO_ON, RENT_MULT_ON = 0.4, 3.0
WORLD = None


def occ_emb(m, seg, mask):
    e = m.E[seg].copy()
    if mask is not None:
        e[mask] = 0.0
    return e


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return 0.0 if n == 0 or x.sum() == 0 else float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def hits_of(m, seg, mask):
    S = m.run_states(occ_emb(m, seg, mask))
    return (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)[:-1] == seg[1:]


def life_of(m, seg, mask, rent):
    cum = np.cumsum(rent * m.M + (~hits_of(m, seg, mask)))
    return len(seg) - 1 if cum[-1] < START_ENERGY else int(np.searchsorted(cum, START_ENERGY)) + 1


def surplus_of(m, seg, mask, rent):
    bank = E0 + np.cumsum(hits_of(m, seg, mask) * FOOD - rent * m.M)
    return (False, 0.0) if bank.min() <= 0 else (True, float(bank[-1] - E0))


def fertility_reproduce(pop, surplus, survived, rng):
    off = np.where(survived, np.floor(surplus / CHILD).astype(int), 0)
    pool = [pop[i] for i in range(len(pop)) for _ in range(int(off[i]))]
    if not pool:
        new = [pop[int(np.argmax(surplus))].copy() for _ in range(POP_SIZE)]
        for c in new:
            c.mutate(rng)
        return new, off
    idx = (rng.choice(len(pool), POP_SIZE, replace=False) if len(pool) >= POP_SIZE
           else np.concatenate([np.arange(len(pool)), rng.choice(len(pool), POP_SIZE - len(pool), True)]))
    new = [pool[i].copy() for i in idx]
    for c in new:
        c.mutate(rng)
    return new, off


def evolve(seed, occ, sel, parsi):
    MIND.DECAY = 1.0
    rng = np.random.default_rng(seed)
    rent = RENT * (RENT_MULT_ON if parsi else 1.0)
    rho = RHO_ON if occ else 0.0
    pop = [Mind(6, rng) for _ in range(POP_SIZE)]
    last_off = np.zeros(POP_SIZE)
    for _ in range(GENS):
        st = int(rng.integers(0, len(WORLD) - SEG - 1)); seg = WORLD[st:st + SEG]
        mask = (rng.random(SEG) < rho) if rho > 0 else None
        if sel:
            sv, sp = zip(*[surplus_of(m, seg, mask, rent) for m in pop])
            pop, last_off = fertility_reproduce(pop, np.array(sp), np.array(sv), rng)
        else:
            lives = np.array([life_of(m, seg, mask, rent) for m in pop])
            pop = reproduce(pop, lives, rng)
    return pop, last_off


def cell(task):
    occ, sel, parsi, seed = task
    global WORLD
    pop, off = evolve(seed, occ, sel, parsi)
    rng = np.random.default_rng(seed + 1234)
    st = int(rng.integers(0, len(WORLD) - SEG - 1)); seg = WORLD[st:st + SEG]
    rent = RENT * (RENT_MULT_ON if parsi else 1.0)
    fits = np.array([life_of(m, seg, None, rent) for m in pop])
    bi = int(np.argmax(fits)); m = pop[bi]
    gain = RM.measure_internals(m, WORLD, 6, 1.0, np.random.default_rng(seed + 7))["gain"]
    comp = float(hits_of(m, seg, None).mean())
    Mmean = float(np.mean([g.M for g in pop]))
    return (occ, sel, parsi, dict(comp=comp, gain=gain, gini=gini(off) if sel else 0.0, M=Mmean))


def main():
    global WORLD
    WORLD = RM.world_longrange(np.random.default_rng(999))[0]
    seeds = list(range(int(os.environ.get("EEC_SEEDS", "2"))))
    tasks = [(o, s, p, sd) for o, s, p in itertools.product([0, 1], repeat=3) for sd in seeds]
    nproc = min(18, len(tasks))
    print(f"PO lattice: 8 axis-subsets x {len(seeds)} seeds on {nproc} workers (gens={GENS})", flush=True)
    with Pool(nproc) as pool:
        res = pool.map(cell, tasks, chunksize=1)

    agg = {}
    for o, s, p, tv in res:
        agg.setdefault((o, s, p), []).append(tv)
    T = {k: {m: float(np.mean([d[m] for d in v])) for m in ("comp", "gain", "gini", "M")}
         for k, v in agg.items()}

    def name(o, s, p):
        ax = [n for n, on in [("obs", o), ("sel", s), ("parsi", p)] if on]
        return "survival+" + "+".join(ax) if ax else "survival only"

    print("\naxis subset            | competence  gain   gini    M     (traits: gain=memory, gini=ecosystem, M=parsimony[low])")
    for o, s, p in sorted(T, key=lambda k: sum(k)):
        t = T[(o, s, p)]
        print(f"  {name(o,s,p):22}| {t['comp']:.3f}      {t['gain']:.3f}  {t['gini']:.3f}  {t['M']:.2f}")

    # target organism: memory AND ecosystem AND parsimony
    base = T[(0, 0, 0)]
    g_thr = 0.5 * (T[(1, 0, 0)]["gain"] + base["gain"])      # halfway to occ-driven gain
    print(f"\nTARGET = {{gain>{g_thr:.2f} (memory), gini>0 (ecosystem), M<{base['M']:.1f} (parsimony)}}")
    full = T[(1, 1, 1)]
    have = (full["gain"] > g_thr, full["gini"] > 0, full["M"] < base["M"])
    print(f"  full set survival+obs+sel+parsi: gain={full['gain']:.3f} gini={full['gini']:.3f} M={full['M']:.2f}"
          f" -> traits met: {have}")
    # ablations from the full set
    print("  ablations (drop one axis from the full set -> which trait is lost?):")
    for drop, (o, s, p) in [("obs", (0, 1, 1)), ("sel", (1, 0, 1)), ("parsi", (1, 1, 0))]:
        t = T[(o, s, p)]
        print(f"    -{drop:6}: gain={t['gain']:.3f} gini={t['gini']:.3f} M={t['M']:.2f}")
    print("\nPO = size of the minimal axis set (incl. survival base) whose every trait survives ablation.")


if __name__ == "__main__":
    main()
