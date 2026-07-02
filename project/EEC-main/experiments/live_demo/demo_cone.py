"""LIVE DEMONSTRATION of the GENREG principle: organisms in a 2D world, evolving in
real time, visibly changing behaviour as each LAW OF EXISTENCE drops on them -- with
the CONE shrinking beside them (measured PO + collapsing strategy-spread).

Nothing about the behaviours is coded. Each organism carries evolvable sensorimotor
weights (attraction to food / memory / separation / signal) + traits (vision, speed),
all starting RANDOM. Selection is pure survival (run out of energy -> die, replaced by
a mutated survivor). With NO laws active there is no death -> no selection -> random
walk. Each law added makes a different latent capability pay, so it EVOLVES on, and the
behaviour shifts in front of you. PO = laws covered; the cone cross-section = the
population's surviving-strategy spread, measured each frame.

Output: demo_cone.mp4
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
L = 100.0                                  # world size
N = 30                                     # organisms
NF = 48                                    # food patches
GENES = ["w_food", "w_mem", "w_sep", "w_sig", "vision", "speed"]
NG = len(GENES)

# ---- the law sequence: (key, label, what you should SEE) ----
LAWS = [
    ("none",   "no laws",        "random walk - no purpose"),
    ("energy", "+ ENERGY",       "seek food - purpose appears"),
    ("time",   "+ TIME",         "efficient paths - less wandering"),
    ("percep", "+ PERCEPTION",   "vision shrinks to what's worth seeing"),
    ("entropy","+ ENTROPY",      "revisit remembered food as memory decays"),
    ("scarce", "+ SCARCITY",     "spread out - territories, uncoded"),
    ("comm",   "+ COMMUNICATION","signal food - others respond"),
]
STAGE_FRAMES = 52
STEPS_PER_FRAME = 3

# ---- state ----
food_xy = rng.uniform(8, L-8, (NF, 2))
food_amt = rng.uniform(0.5, 1.0, NF)
pos = rng.uniform(0, L, (N, 2))
head = rng.uniform(-np.pi, np.pi, N)
energy = np.full(N, 9.0)
mem = pos.copy()                            # remembered food location per organism
memc = np.zeros(N)                          # memory confidence
signal = np.zeros((NF, 0))                  # (unused placeholder)
sig_on = np.zeros(N, bool)                  # who is signalling (on food) this step


def rand_genome(n):
    g = rng.normal(0, 1.0, (n, NG)).astype(float)
    g[:, 4] = rng.uniform(10, 30, n)        # vision
    g[:, 5] = rng.uniform(0.6, 1.8, n)      # speed
    return g


geno = rand_genome(N)
ACT = {k: False for k, _, _ in LAWS}


def norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True); return v / np.where(n < 1e-6, 1, n)


def step():
    global pos, head, energy, geno, mem, memc, sig_on
    sig_on = np.zeros(N, bool)
    # --- sense + decide (per organism) ---
    desired = np.zeros((N, 2))
    for i in range(N):
        vis = np.clip(geno[i, 4], 4, 40)
        d = food_xy - pos[i]; dist = np.linalg.norm(d, axis=1)
        seen = dist < vis
        # FOOD attraction
        if seen.any():
            j = np.where(seen)[0][np.argmin(dist[seen])]
            desired[i] += geno[i, 0] * (d[j] / max(dist[j], 1e-6))
            if food_amt[j] > 0.05 and dist[j] < 4:
                mem[i] = food_xy[j]; memc[i] = 1.0   # remember good food
        # MEMORY attraction (revisit)
        if memc[i] > 0.05:
            dm = mem[i] - pos[i]; dd = np.linalg.norm(dm)
            desired[i] += geno[i, 1] * memc[i] * (dm / max(dd, 1e-6))
        # SEPARATION from neighbours
        if ACT["scarce"]:
            dn = pos[i] - pos; dnn = np.linalg.norm(dn, axis=1)
            near = (dnn > 1e-6) & (dnn < 18)
            if near.any():
                desired[i] += geno[i, 2] * (norm(dn[near]).sum(0))
        # SIGNAL attraction (others on food)
        if ACT["comm"] and sig_src.size:
            ds = sig_src - pos[i]; dss = np.linalg.norm(ds, axis=1)
            k = np.argmin(dss)
            if dss[k] < 45:
                desired[i] += geno[i, 3] * (ds[k] / max(dss[k], 1e-6))
        desired[i] += 0.4 * rng.normal(0, 1, 2)        # exploration noise
    # --- move ---
    dirv = norm(desired)
    spd = np.clip(geno[:, 5], 0.2, 2.2) if ACT["time"] else 1.0
    pos = np.clip(pos + dirv * (spd[:, None] if np.ndim(spd) else spd), 0, L)
    # --- eat (SCARCITY = shared finite food: a patch splits among all on it -> crowding dilutes) ---
    nearest = np.array([int(np.argmin(np.linalg.norm(food_xy - pos[i], axis=1))) for i in range(N)])
    ndist = np.array([np.linalg.norm(food_xy[nearest[i]] - pos[i]) for i in range(N)])
    onfood = (ndist < 4) & (food_amt[nearest] > 0.02)
    for i in range(N):
        if onfood[i]:
            j = nearest[i]
            share = int((onfood & (nearest == j)).sum()) if ACT["scarce"] else 1
            bite = min(0.5, food_amt[j]) / share
            energy[i] += bite * 2.5; sig_on[i] = True
            if ACT["scarce"]:
                food_amt[j] -= bite
    # --- costs (laws) ---
    if ACT["energy"]:
        energy -= 0.5                                   # metabolism: must eat to live
    if ACT["time"]:
        energy -= 0.12 * np.clip(geno[:, 5], 0.2, 2.2)  # movement costs
    if ACT["percep"]:
        energy -= 0.018 * np.clip(geno[:, 4], 4, 40)**2 / 10.0   # looking costs
    # --- memory decay (entropy) ---
    memc *= (0.90 if ACT["entropy"] else 0.995)
    # --- food regen ---
    food_amt[:] = np.minimum(1.0, food_amt + (0.004 if ACT["scarce"] else 0.0))
    # --- death + steady-state reproduction (only once energy law makes death possible) ---
    if ACT["energy"]:
        dead = np.where(energy <= 0)[0]
        alive = np.where(energy > 0)[0]
        if len(alive) and len(dead):
            for d in dead:
                p = alive[rng.integers(len(alive))]
                geno[d] = geno[p] + rng.normal(0, 0.25, NG) * (np.abs(geno[p]) + 0.3)
                geno[d, 4] = np.clip(geno[d, 4], 4, 40); geno[d, 5] = np.clip(geno[d, 5], 0.2, 2.2)
                pos[d] = pos[p] + rng.normal(0, 3, 2); energy[d] = 7.0; memc[d] = 0
    energy[:] = np.clip(energy, -1, 22)


sig_src = np.zeros((0, 2))


def strategy_spread():
    """measured cone cross-section: how spread the surviving strategies are (0..1)."""
    g = geno.copy()
    rel = g.std(0) / (np.abs(g.mean(0)) + 0.6)     # relative dispersion per gene (magnitude-robust)
    return float(np.tanh(rel).mean())


# ================= rendering =================
fig, (axw, axc) = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw={"width_ratios": [1, 0.85]})
fig.patch.set_facecolor("#0d0d12")
spread0 = strategy_spread()


def draw(frame):
    global sig_src
    stage = min(frame // STAGE_FRAMES, len(LAWS) - 1)
    key, label, desc = LAWS[stage]
    for k, _, _ in LAWS[:stage + 1]:
        ACT[k] = (k != "none")
    for _ in range(STEPS_PER_FRAME):
        sig_src = food_xy[(food_amt > 0.05)][:0] if not ACT["comm"] else pos[sig_on]
        step()
    po = sum(ACT[k] for k in ACT)
    # ---- world ----
    axw.clear(); axw.set_facecolor("#10131a"); axw.set_xlim(0, L); axw.set_ylim(0, L)
    axw.set_xticks([]); axw.set_yticks([])
    axw.scatter(food_xy[:, 0], food_xy[:, 1], s=60*food_amt+8, c="#2ecc71", alpha=0.5, marker="s")
    if ACT["percep"] or stage == 0:
        for i in range(0, N, 2):
            axw.add_patch(Circle(pos[i], np.clip(geno[i, 4], 4, 40), fill=False,
                                 ec="#3a6ea5", lw=0.5, alpha=0.25))
    if ACT["comm"]:
        for i in np.where(sig_on)[0]:
            axw.add_patch(Circle(pos[i], 6, fill=False, ec="#f1c40f", lw=1.4, alpha=0.8))
    col = np.clip(energy/18, 0.1, 1) if ACT["energy"] else np.full(N, 0.6)
    axw.scatter(pos[:, 0], pos[:, 1], s=60, c=col, cmap="autumn", edgecolor="white", linewidth=0.4, zorder=5)
    axw.set_title(f"{label}   —   {desc}", color="white", fontsize=13, weight="bold", pad=8)
    # ---- cone ----
    axc.clear(); axc.set_facecolor("#0d0d12"); axc.set_xlim(-1.1, 1.1); axc.set_ylim(-0.05, 1.05)
    axc.set_xticks([]); axc.set_yticks([])
    axc.add_patch(Polygon([(-1, 1), (1, 1), (0, 0)], closed=True, fill=False, ec="#888", lw=1.5))
    # rings: one per active law, descending
    for d in range(1, po + 1):
        y = 1 - d/(len(LAWS))
        w = (1 - (1 - y)) * 1.0   # half-width at this depth = y
        axc.plot([-y, y], [y, y], color="#e74c3c", lw=2, alpha=0.85)
        axc.text(y + 0.04, y, LAWS[d][1], color="#e74c3c", fontsize=8, va="center")
    # habitable cross-section (measured spread) at current depth
    yb = 1 - po/(len(LAWS))
    halfw = max(0.02, yb * (strategy_spread()/spread0))
    axc.add_patch(Polygon([(-halfw, yb-0.012), (halfw, yb-0.012), (halfw, yb+0.012), (-halfw, yb+0.012)],
                          closed=True, color="#2ecc71", alpha=0.8))
    axc.text(0, 1.02, "INFINITY  (every organism that could exist)", color="#888", ha="center", fontsize=8)
    axc.text(0, -0.03, "tip (PO→∞)", color="#888", ha="center", fontsize=8, va="top")
    axc.text(-0.66, 0.34, f"PO = {po}", color="white", ha="center", fontsize=24, weight="bold")
    axc.text(-0.66, 0.25, f"strategies left:\n{100*strategy_spread()/spread0:.0f}% of start",
             color="#2ecc71", ha="center", fontsize=9)
    fig.suptitle("GENREG: a model built by constraint over infinity  —  watch behaviour emerge as each law drops",
                 color="white", fontsize=13)
    return []


def main():
    total = STAGE_FRAMES * len(LAWS)
    ani = FuncAnimation(fig, draw, frames=total, interval=80, blit=False)
    out = os.path.join(HERE, "demo_cone.mp4")
    ani.save(out, writer=FFMpegWriter(fps=12, bitrate=2200))
    print("saved", out)
    fig.savefig(os.path.join(HERE, "demo_cone_frame.png"), dpi=110, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
