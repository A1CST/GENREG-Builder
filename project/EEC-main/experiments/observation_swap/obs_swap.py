"""OBSERVATION-AXIS SWAP: are OCCLUSION and NOISE interchangeable instantiations
of the observation axis -- with WHICH one fires set by the world, not by one being
intrinsically stronger?

In long-range (concentrated, phase-recoverable signal) OCCLUSION won (gain
0.06->0.37) and noise was weak. The axis theory predicts a world where NOISE wins.

Mechanism: occlusion forces memory to bridge DISCRETE gaps (good when signal is
concentrated/recoverable). Noise forces TEMPORAL AVERAGING of redundant unreliable
views (good when the signal is spread over time). So a STICKY world -- a near-
constant latent symbol that persists for long runs -- should REVERSE it: averaging
noisy recent views recovers the latent (noise -> strong memory pressure), while
dropping a few of the redundant views barely hurts (occlusion -> weak).

Run the occlusion-sweep and the noise-sweep on the sticky world; compare which
degradation drives more recurrent gain. Reuses batch_board.internals.
"""
# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
from multiprocessing import Pool
import batch_board as B
import run_matrix as RM

V = 6
SEG = int(os.environ.get("EEC_SEGLR", "600"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "3"))))
WORLDS = {}


def world_sticky(rng, n=200000, p_switch=0.03):
    """Piecewise-constant latent: persists, rarely switches (avg run ~33).
    Predict-next is trivially 'copy current' WHEN you can see it cleanly -- so the
    only value of memory is to integrate across DEGRADED views of the stable latent."""
    s = np.empty(n, dtype=np.int32)
    sw = rng.random(n) < p_switch
    s[0] = rng.integers(0, V)
    for t in range(1, n):
        s[t] = rng.integers(0, V) if sw[t] else s[t - 1]
    return s


def cell(task):
    world, kind, level, seed = task
    ids = WORLDS[world]
    rho, sig = (level, 0.0) if kind == "occ" else (0.0, level)
    return (world, kind, level, B.internals(ids, V, seed, SEG, rho, sig, 1.0))


def main():
    global WORLDS
    rng = np.random.default_rng(999)
    WORLDS = {"sticky": world_sticky(rng), "longrange": RM.world_longrange(rng)[0]}
    rhos = [0.0, 0.2, 0.4, 0.6]
    sigs = [0.0, 0.5, 1.0, 2.0]
    tasks = []
    for world in WORLDS:
        for s in SEEDS:
            tasks += [(world, "occ", r, s) for r in rhos]
            tasks += [(world, "noise", g, s) for g in sigs]
    nproc = min(18, len(tasks))
    print(f"obs-axis swap: {len(tasks)} cells on {nproc} workers (gens={B.GENS} seg={SEG})", flush=True)
    with Pool(nproc) as p:
        res = p.map(cell, tasks, chunksize=1)

    agg = {}
    for world, kind, level, mi in res:
        agg.setdefault((world, kind, level), []).append(mi["gain"])
    G = {k: float(np.mean(v)) for k, v in agg.items()}

    def gline(world, kind, levels):
        return "  ".join(f"{lv}:{G[(world, kind, lv)]:.3f}" for lv in levels)

    for world in WORLDS:
        print(f"\n[{world}] recurrent gain")
        print(f"  OCCLUSION rho  -> {gline(world, 'occ', rhos)}")
        print(f"  NOISE     sig  -> {gline(world, 'noise', sigs)}")
        occ_uplift = max(G[(world, 'occ', r)] for r in rhos) - G[(world, 'occ', 0.0)]
        noi_uplift = max(G[(world, 'noise', s)] for s in sigs) - G[(world, 'noise', 0.0)]
        winner = "OCCLUSION" if occ_uplift > noi_uplift else "NOISE"
        print(f"  -> occ uplift {occ_uplift:+.3f} | noise uplift {noi_uplift:+.3f}  WINNER: {winner}")

    print("\n===== OBSERVATION-AXIS SWAP =====")
    print("If long-range -> OCCLUSION wins but sticky -> NOISE wins, the two are")
    print("interchangeable instantiations of one axis; which fires depends on the world.")


if __name__ == "__main__":
    main()
