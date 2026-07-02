"""Charts for the English-grounded communication findings. Real data, recomputed
(2 seeds, shorter runs for speed -- same qualitative curves as the full battery)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import english_comm as E

HERE = os.path.dirname(os.path.abspath(__file__))
INK = "#16181d"; ACC = "#1b5e9e"; ACC2 = "#c0392b"; GOOD = "#2e8b57"
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.color": "#eee"})


def mean_eng(seeds=2, **kw):
    return np.mean([E.evolve(**kw, seed=s)[1][-1][2] for s in range(seeds)])


print("computing chart data...", flush=True)
# (1) anchor sweep
anchors = [0, 0.5, 1, 1.5, 2, 3]
a_eng = [mean_eng(anchor=a, repro="clone", gens=250) for a in anchors]
# (2) stability under noise: clone vs sexual
noises = [0.0, 0.1, 0.2, 0.3, 0.4]
n_clone = [mean_eng(anchor=2.0, repro="clone", gens=250, noise=n) for n in noises]
n_sex = [mean_eng(anchor=2.0, repro="sexual", gens=250, noise=n) for n in noises]
# (3) exposure dose-response (acquisition from scratch, anchor=0)
fr = [0, 4, 8, 12, 16, 20]
d_sex = [mean_eng(anchor=0.0, repro="sexual", gens=400, n_native=n) for n in fr]
d_clone = [mean_eng(anchor=0.0, repro="clone", gens=400, n_native=n) for n in fr]

fig = plt.figure(figsize=(13, 4.2)); fig.patch.set_facecolor("white")
ax1 = fig.add_axes([0.06, 0.16, 0.26, 0.70])
ax1.plot(anchors, a_eng, "-o", color=ACC, lw=2)
ax1.axhline(1/E.V, color="#999", ls="--", lw=1); ax1.text(0, 1/E.V + 0.03, "chance", fontsize=7, color="#999")
ax1.set_xlabel("English-prior strength (anchor)"); ax1.set_ylabel("English usage"); ax1.set_ylim(0, 1.05)
ax1.set_title("A.  Frozen grounding -> English", fontsize=10)

ax2 = fig.add_axes([0.39, 0.16, 0.26, 0.70])
ax2.plot(noises, n_clone, "-o", color=ACC2, lw=2, label="clone")
ax2.plot(noises, n_sex, "-o", color=ACC, lw=2, label="sexual")
ax2.set_xlabel("channel noise"); ax2.set_ylabel("English usage"); ax2.set_ylim(0, 1.05); ax2.legend()
ax2.set_title("B.  Sex protects English under noise", fontsize=10)

ax3 = fig.add_axes([0.72, 0.16, 0.26, 0.70])
pct = [100 * n / 40 for n in fr]
ax3.plot(pct, d_sex, "-o", color=ACC, lw=2, label="sexual")
ax3.plot(pct, d_clone, "-o", color=ACC2, lw=2, label="clone")
ax3.axhline(1/E.V, color="#999", ls="--", lw=1)
ax3.set_xlabel("% native English speakers"); ax3.set_ylabel("English usage (acquired)"); ax3.set_ylim(0, 1.05); ax3.legend()
ax3.set_title("C.  English acquired from exposure", fontsize=10)

fig.suptitle("English-grounded communication: grounding, stability, and acquisition",
             fontsize=12, weight="bold", y=0.99)
out = os.path.join(HERE, "english_findings.png")
plt.savefig(out, dpi=120, facecolor="white"); print("saved", out)
