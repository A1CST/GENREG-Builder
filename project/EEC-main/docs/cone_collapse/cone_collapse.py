"""THE CONE, ACTUALLY COLLAPSING -- from real evolution data, not a diagram.

Run a real EEC selection process (a population learning the conversational reply function r). Record
every organism's BEHAVIOUR (its reply on each input) at every generation. Project all behaviours to 2D
(PCA), then stack generations on a vertical axis: the cross-section is a wide cloud of possible
organisms at gen 0 and narrows to the surviving tip -- the PO cone, measured. Constraints (the world's
selection) collapse infinity to the survivor, on the actual data.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
S, V, N, GENS = 3, 8, 120, 240


def evolve_record(seed=1):
    rng = np.random.default_rng(seed)
    g = np.zeros((S, V), int)
    for s in range(S):                                     # the world's reply function r(v) != v
        while True:
            p = rng.permutation(V)
            if np.all(p != np.arange(V)): g[s] = p; break
    R = rng.normal(0, 0.3, (N, S, V, V)); behav = []
    for t in range(GENS):
        resp = R.argmax(3)                                 # (N,S,V) each organism's reply behaviour
        behav.append(resp.reshape(N, -1).copy())
        en = (resp == g[None]).reshape(N, -1).sum(1).astype(float)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            m = rng.random(R[pa].shape) < 0.5
            R[w] = np.where(m, R[pa], R[pb]) + rng.normal(0, 0.22 * 0.6, R[pa].shape)
    return np.array(behav)                                  # (GENS, N, S*V)


def main():
    B = evolve_record()
    G, n, d = B.shape
    flat = B.reshape(G * n, d).astype(float)
    mu = flat.mean(0); X = flat - mu
    U, Sg, Vt = np.linalg.svd(X, full_matrices=False)
    proj = (X @ Vt[:3].T).reshape(G, n, 3)                  # 3 PCA components per organism per gen
    gens = np.arange(G)

    # cone "radius": mean distance of the population from its centroid each generation
    radius = np.array([np.linalg.norm(proj[t] - proj[t].mean(0), axis=1).mean() for t in range(G)])

    # ---- chart 1: the 3D cone (generation = depth; cross-section = population behaviour cloud) ----
    fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(111, projection="3d")
    col = plt.cm.viridis(gens / G)
    for t in range(0, G, 3):
        ax.scatter(proj[t, :, 0], proj[t, :, 1], np.full(n, t), s=6, color=col[t], alpha=0.5)
    ax.set_xlabel("behaviour PC1"); ax.set_ylabel("behaviour PC2"); ax.set_zlabel("generation (constraint pressure ->)")
    ax.set_title("The PO cone, measured: population behaviour collapsing to the survivor", weight="bold")
    ax.view_init(elev=12, azim=-60); ax.invert_zaxis()
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "cone_collapse_3d.png"), dpi=130); plt.close()

    # ---- chart 2: cross-sections (the cloud shrinking) ----
    picks = [0, G // 8, G // 4, G // 2, 3 * G // 4, G - 1]
    fig, axs = plt.subplots(1, len(picks), figsize=(18, 3.2), sharex=True, sharey=True)
    lim = np.abs(proj[:, :, :2]).max() * 1.05
    for ax, t in zip(axs, picks):
        ax.scatter(proj[t, :, 0], proj[t, :, 1], s=10, color=col[t], alpha=0.7)
        ax.set_title(f"gen {t}"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xticks([]); ax.set_yticks([])
    axs[0].set_ylabel("behaviour PC2"); fig.suptitle("Cross-sections of the cone: every organism -> the survivor", weight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "cone_collapse_sections.png"), dpi=130); plt.close()

    # ---- chart 3: the collapse curve (cone radius + behavioural diversity) ----
    uniq = np.array([len(np.unique(B[t], axis=0)) for t in range(G)])
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(gens, radius, color="#1b5e9e", lw=2.5, label="cone radius (population spread)")
    ax1.set_xlabel("generation (constraints accumulating ->)"); ax1.set_ylabel("cone radius", color="#1b5e9e")
    ax2 = ax1.twinx(); ax2.plot(gens, uniq, color="#c44", lw=2, label="distinct behaviours")
    ax2.set_ylabel("# distinct surviving behaviours", color="#c44")
    ax1.set_title("The cone collapses: infinite possible organisms -> one survivor", weight="bold"); ax1.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "cone_collapse_radius.png"), dpi=130); plt.close()

    print(f"behaviours {B.shape}; radius gen0={radius[0]:.2f} -> genLast={radius[-1]:.2f} "
          f"({radius[-1]/radius[0]*100:.0f}% of start); distinct behaviours {uniq[0]} -> {uniq[-1]}")
    print("saved cone_collapse_3d.png, cone_collapse_sections.png, cone_collapse_radius.png")


if __name__ == "__main__":
    main()
