"""Animate the cone ACTUALLY collapsing -- the population's behaviour embedding shrinking to the
survivor, generation by generation, as a GIF. Plus the PO axis: more constraints -> tighter survivor set."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
HERE = os.path.dirname(os.path.abspath(__file__))


def evolve_record(S=3, V=8, N=140, GENS=220, seed=1, mut=0.22):
    rng = np.random.default_rng(seed); g = np.zeros((S, V), int)
    for s in range(S):
        while True:
            p = rng.permutation(V)
            if np.all(p != np.arange(V)): g[s] = p; break
    R = rng.normal(0, 0.3, (N, S, V, V)); behav = []
    for t in range(GENS):
        resp = R.argmax(3); behav.append(resp.reshape(N, -1).copy())
        en = (resp == g[None]).reshape(N, -1).sum(1).astype(float)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        mr = mut * (1 - 0.7 * t / GENS)                       # anneal mutation -> sharper tip
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            m = rng.random(R[pa].shape) < 0.5
            R[w] = np.where(m, R[pa], R[pb]) + rng.normal(0, mr * 0.6, R[pa].shape)
    return np.array(behav)


def main():
    B = evolve_record(); G, n, d = B.shape
    flat = B.reshape(G * n, d).astype(float); mu = flat.mean(0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    proj = ((flat - mu) @ Vt[:2].T).reshape(G, n, 2)
    lim = np.abs(proj).max() * 1.05
    radius = np.array([np.linalg.norm(proj[t] - proj[t].mean(0), axis=1).mean() for t in range(G)])

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(proj[0, :, 0], proj[0, :, 1], s=18, alpha=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
    ttl = ax.set_title("")
    ax.set_xlabel("organism behaviour embedding (PC1, PC2)")

    def frame(t):
        sc.set_offsets(proj[t]); sc.set_color(plt.cm.viridis(t / G))
        ttl.set_text(f"gen {t:3d}   cone radius {radius[t]:.2f}   "
                     f"distinct survivors {len(np.unique(B[t],axis=0)):3d}/{n}")
        return sc, ttl
    anim = FuncAnimation(fig, frame, frames=range(0, G, 2), interval=80, blit=False)
    anim.save(os.path.join(HERE, "cone_collapse.gif"), writer=PillowWriter(fps=14))
    plt.close()
    print("saved cone_collapse.gif")

    # ---- PO axis: a cloud of every-possible-organism; each constraint (law) slices the survivor set ----
    rng = np.random.default_rng(7); M = 20000; F = rng.normal(0, 1, (M, 8))   # 20k possible organisms
    keep = np.ones(M, bool); survivors = [int(keep.sum())]
    for c in range(8):                                       # each law eliminates an infinite set of strategies
        keep &= F[:, c] > rng.uniform(-0.3, 0.6); survivors.append(int(keep.sum()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(survivors)), survivors, "o-", color="#1b5e9e", lw=2.5)
    ax.set_yscale("log"); ax.set_xlabel("PO  =  number of constraints (laws of existence) imposed")
    ax.set_ylabel("surviving organisms (log)")
    ax.set_title("The PO cone along the constraint axis: each law slices the survivor set smaller", weight="bold")
    ax.grid(alpha=.3, which="both"); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "cone_po_axis.png"), dpi=130); plt.close()
    print(f"PO axis survivors by #constraints: {survivors}; saved cone_po_axis.png")


if __name__ == "__main__":
    main()
