"""AXIS-SWAP TEST: are constraints interchangeable INSTANTIATIONS of a capability
axis? Specifically the SURVIVAL axis -- is mortality a valid ALTERNATIVE to energy
(not dead, just redundant WITH it)?

Mortality was originally judged degenerate, but only ever tested BOLTED ON TOP of
energy (same axis -> redundant). Here we SWAP it in as the SOLE survival law and
ask whether it covers the axis on its own.

Two mechanistically DIFFERENT survival laws, same axis (convert prediction ->
differential lifespan):
  ENERGY    -- additive budget: death when cumulative (miss+rent) cost >= E0.
  MORTALITY -- multiplicative hazard: each step a death-risk h0*(1+lam*miss); NO
               energy budget. Expected lifespan = sum_t prod_{s<t}(1-hazard_s).
  BOTH      -- die when EITHER fires (energy AND hazard active).

Crossed with OCCLUSION (a DIFFERENT axis -> memory). Long-range, tightened energy.

PREDICTIONS of the axis theory:
  (1) SWAP: mortality-alone competence ~ energy-alone (survival axis covered either way).
  (2) REDUNDANCY: BOTH ~ either alone (same axis -> stacking adds nothing).
  (3) AXIS-INDEPENDENCE: occlusion's gain uplift holds under BOTH survival laws
      (capability depends on the observation axis, not the survival instantiation).

Read law-independent COMPETENCE = hit-rate on a clean held-out segment, and
MEMORY = recurrent gain. We never select on these.
"""
# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
import run_matrix as RM
import mind as MIND
from mind import Mind, RENT, reproduce
from evolve import POP_SIZE

GENS = int(os.environ.get("EEC_MGENS", "150"))
SEG = int(os.environ.get("EEC_SEGLR", "600"))
E0 = float(os.environ.get("EEC_E0", "60"))          # energy budget (tight -> unsaturated)
H0 = float(os.environ.get("EEC_H0", "0.0015"))      # base per-step hazard
LAM = float(os.environ.get("EEC_LAM", "12.0"))      # miss multiplier on hazard


def hits_of(m, seg, mask):
    e = m.E[seg].copy()
    if mask is not None:
        e[mask] = 0.0
    S = m.run_states(e)
    preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
    return preds[:-1] == seg[1:]


def fit_energy(hits, M):
    cum = np.cumsum(RENT * M + (~hits))
    return float(len(hits) if cum[-1] < E0 else np.searchsorted(cum, E0) + 1)


def fit_mortality(hits, M):
    hz = H0 * (1.0 + LAM * (~hits))                 # miss raises instantaneous death risk
    surv_before = np.concatenate([[1.0], np.cumprod(1.0 - hz)[:-1]])
    return float(surv_before.sum())                 # expected lifespan under hazard


def fit_both(hits, M):
    cum = np.cumsum(RENT * M + (~hits))
    energy_alive = np.concatenate([[True], (cum[:-1] < E0)])   # alive (energy) at start of step
    hz = H0 * (1.0 + LAM * (~hits))
    surv_before = np.concatenate([[1.0], np.cumprod(1.0 - hz)[:-1]])
    return float((surv_before * energy_alive).sum())


FIT = {"energy": fit_energy, "mortality": fit_mortality, "both": fit_both}


def evolve(ids, V, seed, law, rho):
    MIND.DECAY = 1.0
    rng = np.random.default_rng(seed)
    pop = [Mind(V, rng) for _ in range(POP_SIZE)]
    fit = FIT[law]
    for _ in range(GENS):
        start = int(rng.integers(0, len(ids) - SEG - 1)); seg = ids[start:start + SEG]
        mask = (rng.random(SEG) < rho) if rho > 0 else None
        fits = np.array([fit(hits_of(m, seg, mask), m.M) for m in pop])
        pop = reproduce(pop, fits, rng)
    return pop


def measure(pop, ids, law, rng):
    """law-independent competence (clean hit-rate of best) + memory gain."""
    start = int(rng.integers(0, len(ids) - SEG - 1)); seg = ids[start:start + SEG]
    fit = FIT[law]
    fits = np.array([fit(hits_of(m, seg, None), m.M) for m in pop])
    bi = int(np.argmax(fits)); m = pop[bi]
    hr = float(hits_of(m, seg, None).mean())
    gain = RM.measure_internals(m, ids, 6, 1.0, np.random.default_rng(int(rng.integers(1 << 30))))["gain"]
    return hr, gain


def main():
    ids, V, _ = RM.world_longrange(np.random.default_rng(999))
    base = 1.0 / V                                   # random-guess hit rate (V=6 -> 0.167)
    print(f"gens={GENS} seg={SEG} V={V} | E0={E0} H0={H0} LAM={LAM} | random hit-rate={base:.3f}")
    res = {}
    for law in ["energy", "mortality", "both"]:
        for rho in [0.0, 0.4]:
            hrs, gains = [], []
            for seed in [0, 1, 2]:
                pop = evolve(ids, V, seed, law, rho)
                hr, g = measure(pop, ids, law, np.random.default_rng(seed + 1000))
                hrs.append(hr); gains.append(g)
            res[(law, rho)] = (np.mean(hrs), np.mean(gains))
            print(f"  [{law:9} occ={rho}] hit_rate={np.mean(hrs):.3f} gain={np.mean(gains):.3f}")

    print("\n===== AXIS THEORY CHECKS (occ=0 for competence) =====")
    e, mo, bo = res[("energy", 0.0)][0], res[("mortality", 0.0)][0], res[("both", 0.0)][0]
    print(f"(1) SWAP        competence: energy {e:.3f} vs mortality {mo:.3f} "
          f"(both >> random {base:.3f} -> survival axis covered either way)")
    print(f"(2) REDUNDANCY  energy {e:.3f} | mortality {mo:.3f} | BOTH {bo:.3f} "
          f"(BOTH ~ max(either) -> stacking same axis adds nothing)")
    ge = res[("energy", 0.4)][1] - res[("energy", 0.0)][1]
    gm = res[("mortality", 0.4)][1] - res[("mortality", 0.0)][1]
    print(f"(3) AXIS-INDEP  occlusion gain-uplift under energy {ge:+.3f} | under mortality {gm:+.3f} "
          f"(both >0 -> memory axis independent of survival law)")


if __name__ == "__main__":
    main()
