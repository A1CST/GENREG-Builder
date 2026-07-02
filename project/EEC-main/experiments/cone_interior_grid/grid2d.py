"""2D grid: OCCLUSION (rho, observation budget) x ENTROPY (1-decay, maintenance
budget) on the long-range world, measured by emergent recurrent gain (active-
maintenance machinery) on CLEAN segments.

This maps the INTERIOR of the cone: not the idealized smooth asymptote but the
actual habitable topology -- the Goldilocks RIDGE where memory machinery peaks
and the COLLAPSE VALLEY where stacked degradation makes the world unlearnable and
emergence drifts back to baseline. Two constraints on the SAME capability axis
share a degradation budget; the ridge is the contour where their sum is optimal,
not the corner where both are maxed.

Reuses batch_board.{worlds,internals}. Long-range only (text is the null axis).
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os, json
import numpy as np
from multiprocessing import Pool

import batch_board as B

HERE = os.path.dirname(os.path.abspath(__file__))
RHOS = [0.0, 0.13, 0.27, 0.40, 0.53, 0.67, 0.80]      # occlusion (input hidden)
DECAYS = [1.0, 0.93, 0.87, 0.80, 0.73, 0.67, 0.60]    # entropy = 1 - decay
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "3"))))
LR = None


def gcell(task):
    rho, dec, seed = task
    ids, V, seg = LR
    mi = B.internals(ids, V, seed, seg, rho, 0.0, dec)
    return (rho, dec, seed, mi)


def main():
    global LR
    W = B.worlds()
    LR = W["longrange"]
    tasks = [(r, d, s) for r in RHOS for d in DECAYS for s in SEEDS]
    nproc = min(18, len(tasks))
    print(f"2D grid: {len(RHOS)}x{len(DECAYS)} x {len(SEEDS)} seeds = {len(tasks)} cells "
          f"on {nproc} workers (gens={B.GENS} seg={LR[2]})", flush=True)
    with Pool(nproc) as p:
        res = p.map(gcell, tasks, chunksize=1)

    # aggregate: mean over seeds -> grids[metric][i_rho, j_decay]
    grids = {m: np.zeros((len(RHOS), len(DECAYS))) for m in ("gain", "horizon", "M")}
    counts = np.zeros((len(RHOS), len(DECAYS)))
    acc = {m: np.zeros((len(RHOS), len(DECAYS))) for m in grids}
    for rho, dec, seed, mi in res:
        i, j = RHOS.index(rho), DECAYS.index(dec)
        counts[i, j] += 1
        for m in grids:
            acc[m][i, j] += mi[m]
    for m in grids:
        grids[m] = acc[m] / np.maximum(counts, 1)

    json.dump({m: grids[m].tolist() for m in grids} |
              {"rhos": RHOS, "decays": DECAYS}, open(os.path.join(HERE, "grid2d.json"), "w"), indent=1)
    _report(grids)
    _plot(grids)


def _report(grids):
    g = grids["gain"]
    ent = [round(1 - d, 2) for d in DECAYS]
    print("\nrecurrent gain  (rows=occlusion rho, cols=entropy strength 1-decay):")
    print("        " + "".join(f"{e:>7}" for e in ent))
    for i, r in enumerate(RHOS):
        print(f"  rho{r:<4} " + "".join(f"{g[i, j]:>7.3f}" for j in range(len(DECAYS))))
    fi, fj = np.unravel_index(np.argmax(g), g.shape)
    print(f"\nRIDGE peak gain {g[fi, fj]:.3f} at rho={RHOS[fi]}, entropy={ent[fj]} "
          f"(decay={DECAYS[fj]})")
    print(f"corner (both maxed) rho={RHOS[-1]} entropy={ent[-1]}: gain {g[-1, -1]:.3f} "
          f"-- collapse valley" if g[-1, -1] < g[fi, fj] * 0.6 else "")


def _plot(grids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ent = [round(1 - d, 2) for d in DECAYS]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for k, metric, title in [(0, "gain", "recurrent gain (active maintenance)"),
                             (1, "horizon", "memory horizon")]:
        G = grids[metric]
        # interpolated filled contour over the grid (rho on x, entropy on y)
        X, Y = np.meshgrid(RHOS, ent)
        cf = ax[k].contourf(X, Y, G.T, levels=14, cmap="viridis")
        ax[k].contour(X, Y, G.T, levels=7, colors="white", linewidths=0.5, alpha=0.5)
        fig.colorbar(cf, ax=ax[k])
        # mark ridge (per-column argmax) and global peak
        for j in range(len(ent)):
            i = int(np.argmax(G[:, j]))
            ax[k].plot(RHOS[i], ent[j], "w.", ms=6)
        fi, fj = np.unravel_index(np.argmax(G), G.shape)
        ax[k].plot(RHOS[fi], ent[fj], "r*", ms=20, label=f"peak {G[fi, fj]:.2f}")
        ax[k].set_xlabel("occlusion  rho  (observation hidden)")
        ax[k].set_ylabel("entropy  1-decay  (state leak / maintenance load)")
        ax[k].set_title(title); ax[k].legend(loc="upper right")
    fig.suptitle("Interior of the cone: occlusion x entropy -> memory machinery (long-range world)\n"
                 "white dots = per-entropy ridge of max gain; red star = global peak; "
                 "top-right corner = collapse valley (both maxed -> unlearnable)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(HERE, "grid2d_cone.png")
    plt.savefig(out, dpi=120); print("saved", out)


if __name__ == "__main__":
    main()
