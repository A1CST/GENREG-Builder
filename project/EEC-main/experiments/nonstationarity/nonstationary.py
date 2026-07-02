"""FRESH LAW on an untouched axis: NON-STATIONARITY -> ADAPTABILITY/plasticity.

Every prior law is STATIC: the world's rules hold for all of evolution, so the
population can converge hard. Here the world's structure SHIFTS periodically (the
long-range periodic block is regenerated every SHIFT_EVERY gens). A population that
over-converges to one regime is stranded when it changes; survival now rewards
ADAPTABILITY -- standing diversity (bet-hedging insurance) and fast re-learning.
This targets a capability orthogonal to memory/perception: evolvability itself.

Compare STATIC vs SHIFTING. Read emergent population structure + recovery dynamics
(read the state, not accuracy):
  - standing diversity (distinct dominant behaviours) -- does churn keep it high?
  - post-shift DROP and RECOVERY of best survival -- can it re-adapt at all?
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
import mind as MIND
from mind import Mind, RENT, START_ENERGY, reproduce
from evolve import POP_SIZE

GENS = int(os.environ.get("EEC_MGENS", "240"))
SEG = int(os.environ.get("EEC_SEGLR", "400"))
SHIFT_EVERY = int(os.environ.get("EEC_SHIFT", "40"))
# tighten energy so survival DEPENDS on exploiting the current regime (else
# survival saturates at seg and a regime shift is invisible)
START = float(os.environ.get("EEC_E0", str(START_ENERGY)))
ALPH, BLK, NOISE = 6, 24, 0.05
HERE = _o.path.dirname(_o.path.abspath(__file__))


def make_world(rng, n=120000):
    block = rng.integers(0, ALPH, BLK)
    stream = np.tile(block, n // BLK + 1)[:n].copy()
    mloc = rng.random(n) < NOISE
    stream[mloc] = rng.integers(0, ALPH, mloc.sum())
    return stream.astype(np.int32)


def life_of(m, seg):
    S = m.run_states(m.E[seg])
    preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
    hits = preds[:-1] == seg[1:]
    cum = np.cumsum(RENT * m.M + (~hits))
    return len(hits) if cum[-1] < START else int(np.searchsorted(cum, START)) + 1


def diversity(pop, world, rng):
    start = int(rng.integers(0, len(world) - SEG - 1)); seg = world[start:start + SEG]
    dom = [int(np.bincount((m.run_states(m.E[seg]) @ m.W_out[:m.M, :] + m.b_out).argmax(1)).argmax())
           for m in pop]
    return len(set(dom))


def run(shifting, seed):
    MIND.DECAY = 1.0
    rng = np.random.default_rng(seed)
    world = make_world(np.random.default_rng(1000 + seed))
    pop = [Mind(ALPH, rng) for _ in range(POP_SIZE)]
    best_hist, shift_gens = [], []
    for gen in range(GENS):
        if shifting and gen > 0 and gen % SHIFT_EVERY == 0:
            world = make_world(np.random.default_rng(7000 + seed * 99 + gen))   # regime change
            shift_gens.append(gen)
        start = int(rng.integers(0, len(world) - SEG - 1)); seg = world[start:start + SEG]
        lives = np.array([life_of(m, seg) for m in pop])
        best_hist.append(int(lives.max()))
        pop = reproduce(pop, lives, rng)
    div = diversity(pop, world, np.random.default_rng(seed + 50))
    return np.array(best_hist), shift_gens, div


def recovery_stats(hist, shifts):
    """For each shift: pre = mean of 3 gens before; drop = gen right after; rec = mean 5 gens later."""
    drops, recs = [], []
    for g in shifts:
        if g + 6 >= len(hist):
            continue
        pre = hist[g - 3:g].mean()
        drop = hist[g]                       # first gen on the new regime
        rec = hist[g + 1:g + 6].mean()
        if pre > 0:
            drops.append((pre - drop) / pre)
            recs.append(rec / pre)
    return float(np.mean(drops)) if drops else 0.0, float(np.mean(recs)) if recs else 0.0


def main():
    print(f"gens={GENS} seg={SEG} shift_every={SHIFT_EVERY} | long-range V={ALPH}")
    out = {}
    for mode in ["static", "shifting"]:
        divs, drops, recs, hists = [], [], [], []
        for seed in [0, 1, 2]:
            hist, shifts, div = run(mode == "shifting", seed)
            divs.append(div)
            if mode == "shifting":
                d, r = recovery_stats(hist, shifts); drops.append(d); recs.append(r)
            hists.append(hist)
        out[mode] = dict(div=np.mean(divs), drop=np.mean(drops) if drops else 0.0,
                         rec=np.mean(recs) if recs else 0.0, hist=np.mean(hists, axis=0),
                         shifts=shifts if mode == "shifting" else [])
        if mode == "shifting":
            print(f"  {mode:8}: standing_diversity={out[mode]['div']:.1f} | "
                  f"per-shift DROP={out[mode]['drop']:.2f} RECOVERY_ratio={out[mode]['rec']:.2f} "
                  f"(1.0=full recovery to pre-shift survival)")
        else:
            print(f"  {mode:8}: standing_diversity={out[mode]['div']:.1f}")

    print("\n===== ADAPTABILITY =====")
    print(f"  standing diversity:  static {out['static']['div']:.1f} -> shifting {out['shifting']['div']:.1f}")
    print(f"  shifting recovers to {out['shifting']['rec']*100:.0f}% of pre-shift survival after a regime change "
          f"(drop {out['shifting']['drop']*100:.0f}%).")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(out["static"]["hist"], color="#999", label="static world")
    ax.plot(out["shifting"]["hist"], color="#d95f02", label="shifting world")
    for g in out["shifting"]["shifts"]:
        ax.axvline(g, color="#d95f02", ls=":", alpha=0.4)
    ax.set_xlabel("generation"); ax.set_ylabel("best lifespan (survival)")
    ax.set_title("NON-STATIONARITY: regime shifts (dotted) -> drop + re-adaptation\n"
                 "does the population recover, and stay more diverse?")
    ax.legend()
    plt.tight_layout()
    p = _o.path.join(HERE, "nonstationary.png")
    plt.savefig(p, dpi=115); print("\nsaved", p)


if __name__ == "__main__":
    main()
