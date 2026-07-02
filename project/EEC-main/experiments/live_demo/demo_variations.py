"""GENREG live demo -- VARIATIONS. Same world, same starting population (same seed),
different LAW SCHEDULES, each making a point about the paradigm. Run one variation:
  python3 demo_variations.py <run_name>
Renders run_<name>.mp4 (title card + the live evolution). A separate concat step
stitches them into demo_full.mp4.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
L, N, NF = 100.0, 30, 48
NG = 6                                   # w_food,w_mem,w_sep,w_sig,vision,speed
CANON = ["energy", "time", "percep", "entropy", "scarce", "comm"]
LABEL = {"energy": "ENERGY", "time": "TIME", "percep": "PERCEPTION",
         "entropy": "ENTROPY", "scarce": "SCARCITY", "comm": "COMMUNICATION"}
TITLE_FRAMES = 34
TRAIL = 10                               # motion-trail length (so paths are legible)

# ---- mutable state (reset per run) ----
S = {}


def reset(seed=7):
    r = np.random.default_rng(seed)
    g = r.normal(0, 1.0, (N, NG))
    g[:, 4] = r.uniform(10, 30, N); g[:, 5] = r.uniform(0.6, 1.8, N)
    S.update(rng=r, food_xy=r.uniform(8, L-8, (NF, 2)), food_amt=r.uniform(0.4, 0.9, NF),
             pos=r.uniform(0, L, (N, 2)), energy=np.full(N, 9.0), geno=g,
             mem=r.uniform(0, L, (N, 2)), memc=np.zeros(N), sig_on=np.zeros(N, bool),
             sig_src=np.zeros((0, 2)), t=0, births=0)
    S["trail"] = np.tile(S["pos"], (TRAIL, 1, 1))
    S["spread0"] = strategy_spread()


def norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True); return v / np.where(n < 1e-6, 1, n)


def strategy_spread():
    g = S["geno"]; rel = g.std(0) / (np.abs(g.mean(0)) + 0.6)
    return float(np.tanh(rel).mean())


def step(ACT):
    S["t"] += 1
    r = S["rng"]; pos = S["pos"]; geno = S["geno"]; food_xy = S["food_xy"]; food_amt = S["food_amt"]
    energy = S["energy"]; mem = S["mem"]; memc = S["memc"]
    S["sig_on"] = np.zeros(N, bool); sig_on = S["sig_on"]; sig_src = S["sig_src"]
    desired = np.zeros((N, 2))
    for i in range(N):
        vis = np.clip(geno[i, 4], 4, 40)
        d = food_xy - pos[i]; dist = np.linalg.norm(d, axis=1); seen = dist < vis
        if seen.any():
            j = np.where(seen)[0][np.argmin(dist[seen])]
            desired[i] += geno[i, 0] * (d[j] / max(dist[j], 1e-6))
            if food_amt[j] > 0.05 and dist[j] < 4:
                mem[i] = food_xy[j]; memc[i] = 1.0
        if memc[i] > 0.05:
            dm = mem[i] - pos[i]; dd = np.linalg.norm(dm)
            desired[i] += geno[i, 1] * memc[i] * (dm / max(dd, 1e-6))
        if ACT.get("scarce"):
            dn = pos[i] - pos; dnn = np.linalg.norm(dn, axis=1); near = (dnn > 1e-6) & (dnn < 18)
            if near.any():
                desired[i] += geno[i, 2] * norm(dn[near]).sum(0)
        if ACT.get("comm") and sig_src.size:
            ds = sig_src - pos[i]; dss = np.linalg.norm(ds, axis=1); k = np.argmin(dss)
            if dss[k] < 45:
                desired[i] += geno[i, 3] * (ds[k] / max(dss[k], 1e-6))
        desired[i] += 0.4 * r.normal(0, 1, 2)
    spd = np.clip(geno[:, 5], 0.2, 2.2) if ACT.get("time") else 1.0
    S["pos"] = np.clip(pos + norm(desired) * (spd[:, None] if np.ndim(spd) else spd), 0, L)
    pos = S["pos"]
    nearest = np.array([int(np.argmin(np.linalg.norm(food_xy - pos[i], axis=1))) for i in range(N)])
    ndist = np.array([np.linalg.norm(food_xy[nearest[i]] - pos[i]) for i in range(N)])
    onfood = (ndist < 4) & (food_amt[nearest] > 0.02)
    for i in range(N):
        if onfood[i]:
            j = nearest[i]
            share = int((onfood & (nearest == j)).sum()) if ACT.get("scarce") else 1
            bite = min(0.5, food_amt[j]) / share
            energy[i] += bite * 2.5; sig_on[i] = True
            if ACT.get("scarce"):
                food_amt[j] -= bite
    if ACT.get("energy"):
        energy -= 0.6
    if ACT.get("time"):
        energy -= 0.12 * np.clip(geno[:, 5], 0.2, 2.2)
    if ACT.get("percep"):
        energy -= 0.006 * np.clip(geno[:, 4], 4, 40)**2 / 10.0
    memc *= (0.90 if ACT.get("entropy") else 0.995)
    food_amt[:] = np.minimum(1.0, food_amt + (0.0016 if ACT.get("scarce") else 0.0))
    geno += r.normal(0, 0.012, (N, NG))      # NEUTRAL DRIFT: mutation is always present
    geno[:, :4] = np.clip(geno[:, :4], -6, 6)
    geno[:, 4] = np.clip(geno[:, 4], 4, 40); geno[:, 5] = np.clip(geno[:, 5], 0.2, 2.2)
    if ACT.get("energy"):                    # SURVIVAL: worst foragers continually outcompeted (steady-state)
        order = np.argsort(energy); K = 2
        worst = order[:K]; top = order[N - N//2:]
        S["births"] += K; med = float(np.median(energy))
        for dd in worst:
            p = int(top[r.integers(len(top))])
            geno[dd] = geno[p] + r.normal(0, 0.11, NG) * (np.abs(geno[p]) + 0.3)
            geno[dd, :4] = np.clip(geno[dd, :4], -6, 6)
            geno[dd, 4] = np.clip(geno[dd, 4], 4, 40); geno[dd, 5] = np.clip(geno[dd, 5], 0.2, 2.2)
            pos[dd] = pos[p] + r.normal(0, 3, 2); energy[dd] = med; memc[dd] = 0
    S["energy"][:] = np.clip(energy, -1, 22)


# ============ runs: (subtitle, [(frames, active_set, stage_caption)...]) ============
def aset(*names): return {n: True for n in names}


def make_runs():
    full = aset(*CANON)
    return {
        "1_canonical": ("ONE LAW AT A TIME, in order  —  structure precipitates ring by ring", [
            (80, {}, "no laws  —  aimless wandering, no purpose"),
            (165, aset("energy"), "+ ENERGY  —  seek food (purpose appears; widest space, slowest to find)"),
            (115, aset("energy", "time"), "+ TIME  —  efficient, direct paths"),
            (140, aset("energy", "time", "percep"), "+ PERCEPTION  —  vision shrinks to what's worth seeing"),
            (100, aset("energy", "time", "percep", "entropy"), "+ ENTROPY  —  revisit remembered food"),
            (110, aset("energy", "time", "percep", "entropy", "scarce"), "+ SCARCITY  —  spread out / territory"),
            (110, full, "+ COMMUNICATION  —  signal food, others respond")]),
        "2_energy_substrate": ("ENERGY IS THE SUBSTRATE  —  laws are INERT without selection", [
            (85, {}, "no laws  —  random walk"),
            (300, aset("scarce", "percep", "entropy"), "SCARCITY + PERCEPTION + ENTROPY active... but NO energy -> no death -> STILL random"),
            (330, aset("scarce", "percep", "entropy", "energy"), "+ ENERGY  ->  selection switches ON  ->  everything the other laws demanded appears")]),
        "3_order_path": ("ORDER changes the PATH, not the DESTINATION  —  scrambled order, same final laws", [
            (75, {}, "random"),
            (160, aset("energy"), "+ ENERGY"),
            (110, aset("energy", "comm"), "+ COMMUNICATION (early this time)"),
            (105, aset("energy", "comm", "scarce"), "+ SCARCITY"),
            (105, aset("energy", "comm", "scarce", "percep"), "+ PERCEPTION"),
            (135, full, "+ TIME + ENTROPY  ->  SAME final organism as run 1")]),
        "4_all_at_once": ("ALL LAWS AT ONCE  —  cold start under full pressure", [
            (75, {}, "random"),
            (640, full, "all 6 laws dropped together  —  must satisfy every law from scratch (slow, messy convergence)")]),
        "5_lift_energy": ("LIFT A LAW  —  the cone RE-OPENS  —  constraint is not permanent", [
            (75, {}, "random"),
            (330, full, "build up to PO = 6 (cone closes)"),
            (330, aset("time", "percep", "entropy", "scarce", "comm"),
             "REMOVE ENERGY  ->  selection stops  ->  drift takes over  ->  strategies re-diversify, cone re-opens")]),
    }


def main(run):
    runs = make_runs()
    subtitle, schedule = runs[run]
    reset(7)
    cum = []
    f = TITLE_FRAMES
    for nf, act, cap in schedule:
        cum.append((f, f + nf, act, cap)); f += nf
    total = f

    fig, (axw, axc) = plt.subplots(1, 2, figsize=(13, 6.6), gridspec_kw={"width_ratios": [1, 0.85]})
    fig.patch.set_facecolor("#0d0d12")

    def draw(frame):
        if frame < TITLE_FRAMES:                          # ---- title card ----
            for ax in (axw, axc):
                ax.clear(); ax.set_facecolor("#0d0d12"); ax.axis("off")
            fig.suptitle("")
            axw.text(1.05, 0.62, f"RUN {run[0]}", transform=axw.transAxes, ha="center",
                     color="#2ecc71", fontsize=30, weight="bold")
            axw.text(1.05, 0.46, subtitle, transform=axw.transAxes, ha="center",
                     color="white", fontsize=14, wrap=True)
            return []
        act, cap = {}, ""
        for a, b, ac, cp in cum:
            if a <= frame < b:
                act, cap = ac, cp; break
        else:
            act, cap = cum[-1][2], cum[-1][3]
        for _ in range(3):
            S["sig_src"] = S["pos"][S["sig_on"]] if act.get("comm") else S["pos"][:0]
            step(act)
        S["trail"] = np.roll(S["trail"], -1, axis=0); S["trail"][-1] = S["pos"].copy()
        po = sum(bool(v) for v in act.values())
        # ---- world ----
        axw.clear(); axw.set_facecolor("#10131a"); axw.set_xlim(0, L); axw.set_ylim(0, L)
        axw.axis("on"); axw.set_xticks([]); axw.set_yticks([])
        fa = S["food_amt"]; fx = S["food_xy"]; pos = S["pos"]; geno = S["geno"]; tr = S["trail"]
        axw.scatter(fx[:, 0], fx[:, 1], s=60*fa+8, c="#2ecc71", alpha=0.5, marker="s")
        for i in range(N):                                   # motion trails -> paths become legible
            seg = tr[:, i, :]
            if np.linalg.norm(seg[-1] - seg[0]) < 60:        # skip rare boundary jumps
                axw.plot(seg[:, 0], seg[:, 1], color="#e08a3a", alpha=0.30, lw=1.1, zorder=2)
        if act.get("percep") or po == 0:
            for i in range(0, N, 2):
                axw.add_patch(Circle(pos[i], np.clip(geno[i, 4], 4, 40), fill=False, ec="#3a6ea5", lw=0.5, alpha=0.22))
        if act.get("comm"):
            for i in np.where(S["sig_on"])[0]:
                axw.add_patch(Circle(pos[i], 6, fill=False, ec="#f1c40f", lw=1.4, alpha=0.8))
        col = np.clip(S["energy"]/18, 0.1, 1) if act.get("energy") else np.full(N, 0.6)
        axw.scatter(pos[:, 0], pos[:, 1], s=60, c=col, cmap="autumn", edgecolor="white", linewidth=0.4, zorder=5)
        axw.set_title(cap, color="white", fontsize=12.5, weight="bold", pad=8)
        gen = S["births"] // N
        axw.text(2.5, 2.5, f"world-step {S['t']}    ·    selection events {S['births']}    ·    ≈ generation {gen}",
                 color="#aab", fontsize=10.5, va="bottom", zorder=10)
        # ---- cone ----
        axc.clear(); axc.set_facecolor("#0d0d12"); axc.set_xlim(-1.15, 1.15); axc.set_ylim(-0.05, 1.08)
        axc.axis("on"); axc.set_xticks([]); axc.set_yticks([])
        axc.add_patch(Polygon([(-1, 1), (1, 1), (0, 0)], closed=True, fill=False, ec="#888", lw=1.5))
        active_depths = [CANON.index(k)+1 for k in CANON if act.get(k)]
        for di in active_depths:
            y = 1 - di/(len(CANON)+1)
            axc.plot([-y, y], [y, y], color="#e74c3c", lw=2, alpha=0.85)
            axc.text(y+0.04, y, "+ "+LABEL[CANON[di-1]], color="#e74c3c", fontsize=8, va="center")
        depth = max(active_depths) if active_depths else 0
        yb = 1 - depth/(len(CANON)+1) if depth else 0.98
        halfw = max(0.02, yb * (strategy_spread()/S["spread0"]))
        axc.add_patch(Polygon([(-halfw, yb-0.012), (halfw, yb-0.012), (halfw, yb+0.012), (-halfw, yb+0.012)],
                              closed=True, color="#2ecc71", alpha=0.85))
        axc.text(0, 1.04, "INFINITY", color="#888", ha="center", fontsize=8)
        axc.text(0, -0.04, "tip", color="#888", ha="center", fontsize=8, va="top")
        axc.text(-0.72, 0.32, f"PO = {po}", color="white", ha="center", fontsize=22, weight="bold")
        axc.text(-0.72, 0.23, f"strategies left:\n{100*strategy_spread()/S['spread0']:.0f}%",
                 color="#2ecc71", ha="center", fontsize=9)
        fig.suptitle("GENREG: a model built by constraint over infinity", color="white", fontsize=13)
        return []

    ani = FuncAnimation(fig, draw, frames=total, interval=80, blit=False)
    out = os.path.join(HERE, f"run_{run}.mp4")
    ani.save(out, writer=FFMpegWriter(fps=12, bitrate=2200))
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
