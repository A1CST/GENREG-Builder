"""The PO metric, rendered: constraints collapse the cone of infinite possibility.

X axis = number of constraints (the PO metric). Cross-section = remaining
possibility space (every organism still viable). Each law slices it smaller.
Radius decays exponentially -> PO->0 is an asymptote, never a point.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "po_cone.png")

# constraints along the axis: active (solid) then pocket (faded)
LAWS = ["∞\n(no laws)", "energy", "time", "perception", "memory",
        "scarcity", "entropy", "…"]
ACTIVE = 5                      # first ACTIVE rings are imposed laws (incl. x=0)
R0 = 1.0
K = 0.45                        # decay rate -> asymptote toward the tip

fig = plt.figure(figsize=(15, 7))

# --- 3D cone ---------------------------------------------------------------
ax = fig.add_subplot(1, 2, 1, projection="3d")
xs = np.linspace(0, len(LAWS) - 1, 160)
rs = R0 * np.exp(-K * xs)
th = np.linspace(0, 2 * np.pi, 80)
Xg, Tg = np.meshgrid(xs, th)
Rg = R0 * np.exp(-K * Xg)
Yg, Zg = Rg * np.cos(Tg), Rg * np.sin(Tg)
ax.plot_surface(Xg, Yg, Zg, alpha=0.18, color="#377eb8", linewidth=0)
for i, name in enumerate(LAWS):
    r = R0 * np.exp(-K * i)
    c = th
    solid = i < ACTIVE
    ax.plot(np.full_like(c, i), r*np.cos(c), r*np.sin(c),
            color="#111" if solid else "#999",
            lw=2.2 if solid else 1.0, ls="-" if solid else "--")
    ax.text(i, 0, r + 0.06, name, ha="center", fontsize=8,
            color="#111" if solid else "#999")
ax.text(0, 0, R0 + 0.25, "every possible organism", ha="center", fontsize=9, color="#377eb8")
ax.text(len(LAWS)-1, 0, 0.18, "the survivor\n(PO → 0, asymptote)", ha="center",
        fontsize=9, color="#e41a1c")
ax.set_xlabel("constraints  (the PO metric →)")
ax.set_yticks([]); ax.set_zticks([])
ax.set_title("The model is the intersection of its laws")
ax.view_init(elev=14, azim=-72)

# --- possibility-space curve (the score) -----------------------------------
ax2 = fig.add_subplot(1, 2, 2)
area = np.pi * rs ** 2
ax2.fill_between(xs, area, color="#377eb8", alpha=0.15)
ax2.plot(xs, area, color="#377eb8", lw=2)
for i, name in enumerate(LAWS):
    a = np.pi * (R0 * np.exp(-K * i)) ** 2
    solid = i < ACTIVE
    ax2.scatter([i], [a], color="#111" if solid else "#999", zorder=3,
                s=30 if solid else 18)
    ax2.annotate(name.replace("\n", " "), (i, a), textcoords="offset points",
                 xytext=(4, 8), fontsize=8, color="#111" if solid else "#999")
ax2.set_xlabel("number of laws imposed  (PO)")
ax2.set_ylabel("possibility space still viable")
ax2.set_title("Each law collapses the survivable space (asymptote → 0)")
ax2.set_ylim(0, np.pi * R0**2 * 1.15)

plt.tight_layout()
plt.savefig(OUT, dpi=115)
print("saved", OUT)
