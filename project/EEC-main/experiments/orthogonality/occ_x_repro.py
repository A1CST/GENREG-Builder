"""ORTHOGONALITY POSITIVE CONTROL: do two constraints on DIFFERENT capability
axes STACK CLEANLY (both effects preserved), unlike same-budget pairs that
interfere (occ x entropy collapsed)?

Pair: OCCLUSION (drives MEMORY machinery -> recurrent gain) x REPRODUCTION-COST
(prediction->surplus->fertility; restores selection under survival-saturation ->
fertility stratification Gini). Both demonstrably fire in the LONG-RANGE world
(occ finding 6, repro finding 5), on different axes. Design-principle PREDICTION:
they share NO degradation budget, so occ still grows gain WITH/WITHOUT repro-cost,
and repro-cost still stratifies fertility WITH/WITHOUT occlusion -- no collapse.

2x2: occlusion {0, 0.4} x reproduction {lifespan-selection, fertility-cost}.
Read gain (memory, from occ) and offspring Gini (fertility, from repro-cost).
"""
# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
import run_matrix as RM
import mind as MIND
from mind import Mind, RENT, START_ENERGY, reproduce
from evolve import POP_SIZE

GENS = int(os.environ.get("EEC_MGENS", "150"))
SEG = int(os.environ.get("EEC_SEGLR", "600"))
E0, FOOD, CHILD_COST = 5.0, 1.0, 10.0


def occ_emb(m, seg, mask):
    e = m.E[seg].copy()
    if mask is not None:
        e[mask] = 0.0
    return e


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return 0.0 if n == 0 or x.sum() == 0 else float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def surplus_of(m, seg, mask):
    S = m.run_states(occ_emb(m, seg, mask))
    preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
    hits = preds[:-1] == seg[1:]
    bank = E0 + np.cumsum(hits * FOOD - RENT * m.M)
    return (False, 0.0) if bank.min() <= 0 else (True, float(bank[-1] - E0))


def life_of(m, seg, mask):
    S = m.run_states(occ_emb(m, seg, mask))
    preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
    hits = preds[:-1] == seg[1:]
    cum = np.cumsum(RENT * m.M + (~hits))
    return len(hits) if cum[-1] < START_ENERGY else int(np.searchsorted(cum, START_ENERGY)) + 1


def fertility_reproduce(pop, surplus, survived, rng):
    off = np.where(survived, np.floor(surplus / CHILD_COST).astype(int), 0)
    pool = [pop[i] for i in range(len(pop)) for _ in range(int(off[i]))]
    if not pool:
        best = int(np.argmax(surplus))
        new = [pop[best].copy() for _ in range(POP_SIZE)]
        for c in new:
            c.mutate(rng)
        return new, off
    idx = (rng.choice(len(pool), POP_SIZE, replace=False) if len(pool) >= POP_SIZE
           else np.concatenate([np.arange(len(pool)), rng.choice(len(pool), POP_SIZE - len(pool), True)]))
    new = []
    for i in idx:
        c = pool[i].copy(); c.mutate(rng); new.append(c)
    return new, off


def evolve(ids, V, seed, rho, repro_cost):
    MIND.DECAY = 1.0
    rng = np.random.default_rng(seed)
    pop = [Mind(V, rng) for _ in range(POP_SIZE)]
    last_off = np.zeros(POP_SIZE)
    for _ in range(GENS):
        start = int(rng.integers(0, len(ids) - SEG - 1))
        seg = ids[start:start + SEG]
        mask = (rng.random(SEG) < rho) if rho > 0 else None
        if repro_cost:
            sv, sp = zip(*[surplus_of(m, seg, mask) for m in pop])
            pop, last_off = fertility_reproduce(pop, np.array(sp), np.array(sv), rng)
        else:
            lives = np.array([life_of(m, seg, mask) for m in pop])
            pop = reproduce(pop, lives, rng)
    return pop, last_off


def best_gain(pop, ids, rho, rng):
    """recurrent gain of the best organism, measured on a CLEAN segment."""
    start = int(rng.integers(0, len(ids) - SEG - 1)); seg = ids[start:start + SEG]
    lives = np.array([life_of(m, seg, None) for m in pop])
    bi = int(np.argmax(lives))
    return RM.measure_internals(pop[bi], ids, 6, 1.0, np.random.default_rng(int(rng.integers(1 << 30))))["gain"]


def main():
    ids, V, _ = RM.world_longrange(np.random.default_rng(999))
    print(f"gens={GENS} seg={SEG} long-range V={V}")
    res = {}
    for rho in [0.0, 0.4]:
        for rc in [False, True]:
            gains, ginis = [], []
            for seed in [0, 1]:
                pop, off = evolve(ids, V, seed, rho, rc)
                gains.append(best_gain(pop, ids, rho, np.random.default_rng(seed + 900)))
                ginis.append(gini(off) if rc else 0.0)
            res[(rho, rc)] = (np.mean(gains), np.mean(ginis))
            print(f"  occ_rho={rho} repro_cost={str(rc):5}: gain={np.mean(gains):.3f} "
                  f"fert_gini={np.mean(ginis):.3f}")

    print("\n===== ORTHOGONALITY CHECK =====")
    g_occ_effect_noRC = res[(0.4, False)][0] - res[(0.0, False)][0]
    g_occ_effect_RC = res[(0.4, True)][0] - res[(0.0, True)][0]
    print(f"  occlusion's GAIN uplift:  without repro-cost {g_occ_effect_noRC:+.3f} | "
          f"with repro-cost {g_occ_effect_RC:+.3f}  (both >0 -> occ survives)")
    print(f"  repro-cost's FERTILITY Gini:  no-occ {res[(0.0, True)][1]:.3f} | "
          f"occ {res[(0.4, True)][1]:.3f}  (both >0 -> fertility survives)")
    print("  -> if both effects persist when combined, the constraints are ORTHOGONAL "
          "(stack cleanly), unlike same-budget occ x entropy which collapsed.")


if __name__ == "__main__":
    main()
