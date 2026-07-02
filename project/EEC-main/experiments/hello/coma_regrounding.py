"""COMA / RE-GROUNDING — the continuity hypothesis on a substrate with a STRONG capability.

Hypothesis (Payton): a capability is the surviving behaviour under an ongoing world-constraint.
Remove the constraint (a coma: no input, no consequence) and the substrate keeps drifting but
nothing pays the capability's rent, so it erodes -- NOT because the hardware breaks, but because the
capability stops keeping the organism alive. Waking = RE-grounding, which must obey the same
reachability law as first learning.

Substrate: embodied_forage.py -- a forager on a line of cells, LOCAL vision, food that regrows in
place. Fitness = food eaten (a pure world-consequence, no designed gradient). The strong, ablation-
proven capability is EXTERNAL MEMORY: an organism that marks food patches and walks back eats ~2x
what the same organism eats with marks disabled. We use that gap as the capability that can erode.

Phases:
  GROW   evolve foragers in the persistent world -> they eat well and USE memory (marks-gap large).
  COMA   for C generations: every genome DRIFTS (small gaussian noise), NO selection. The world
         imposes no foraging constraint; nothing maintains the capability. (Not flat-fitness through
         reproduce(): that freezes elites = a preserved brain. Drift is honest 'use it or lose it'.)
  WAKE   put the comatose population back in the world; evolve normally; watch food recover.

Readouts:
  - erosion       food eaten & marks-gap just after coma vs before, across coma length C
  - WALLIS test   does a coma'd organism re-acquire FASTER than a NAIVE (fresh) one at matched gens?
                  faster => residual structure survived under the surface; equal => nothing preserved.
  - dose-response longer coma -> worse recovery, and a point of no return (capability leaves the cone).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "english_comm"))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import embodied_forage as EF

PARAMS     = ("W1", "b1", "wg", "Wv", "Wm")
GROW_GENS  = 350
RECOVER    = 120
SAMPLE     = 12                       # eval the recovery trajectory every SAMPLE gens
DRIFT_SIG  = 0.05                     # per-gen gaussian drift during coma (un-grounded random walk)
COMAS      = [0, 40, 120, 300, 700]
SEEDS      = [0, 1, 2]
N          = 44

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "coma_regrounding_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.2f}+/-{a.std():.2f}"


def step_world(g, rng, mut=0.22, cull=0.25):
    """One generation of selection in the foraging world (EF's own operator, on an existing pop)."""
    n = len(g["W1"]); en = EF.fitness(g, rng)
    Kc = max(1, int(cull * n)); order = np.argsort(en)
    worst = order[:Kc]; top = order[n - max(2, n // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        for key in PARAMS:
            m = rng.random(g[key][pa].shape) < 0.5
            g[key][w] = np.where(m, g[key][pa], g[key][pb]) + rng.normal(0, mut * 0.6, g[key][pa].shape)
    return g


def drift(g, rng, sigma, gens):
    """Coma: every genome random-walks; no selection. Nothing maintains the capability."""
    g = {k: g[k].copy() for k in PARAMS}
    for _ in range(gens):
        for key in PARAMS:
            g[key] += rng.normal(0, sigma, g[key].shape)
    return g


def capability(g, rng, n=200):
    """(food eaten per life, memory-specific marks-ablation gap)."""
    on = EF.evaluate(g, rng, n=n)
    off = EF.evaluate(g, rng, n=n, no_marks=True)
    return on, on - off


def recover(g, rng, gens, track_rng=None):
    traj = []
    for t in range(gens):
        step_world(g, rng)
        if track_rng is not None and (t % SAMPLE == 0 or t == gens - 1):
            traj.append((t, EF.evaluate(g, track_rng, n=100)))
    return g, traj


if __name__ == "__main__":
    out(f"COMA / RE-GROUNDING on the foraging world (food fitness, memory = marks). "
        f"grow={GROW_GENS}, recover={RECOVER} gens, drift_sigma={DRIFT_SIG}")
    out("=" * 80)

    grown_food, grown_gap = [], []
    dose = {C: {"wfood": [], "wgap": [], "ffood": [], "fgap": []} for C in COMAS}
    recover_curves = {C: [] for C in COMAS}; naive_curves = []

    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        base = EF.evolve(gens=GROW_GENS, N=N, seed=seed)        # GROW: continuity intact
        f0, gp0 = capability(base, rng); grown_food.append(f0); grown_gap.append(gp0)

        naive = EF.new_pop(N, np.random.default_rng(7000 + seed))   # NAIVE re-ground from scratch
        _, nt = recover(naive, np.random.default_rng(2000 + seed), RECOVER, track_rng=np.random.default_rng(50 + seed))
        naive_curves.append(nt)

        for C in COMAS:
            comatose = drift(base, np.random.default_rng(3000 + seed * 17 + C), DRIFT_SIG, C)
            wf, wg = capability(comatose, rng)                  # eroded capability at wake
            woke, rt = recover(comatose, np.random.default_rng(4000 + seed * 17 + C), RECOVER,
                               track_rng=np.random.default_rng(60 + seed))
            ff, fg = capability(woke, rng)                      # recovered capability
            dose[C]["wfood"].append(wf); dose[C]["wgap"].append(wg)
            dose[C]["ffood"].append(ff); dose[C]["fgap"].append(fg)
            recover_curves[C].append(rt)

    naive_food = np.mean([c[-1][1] for c in naive_curves])
    out(f"GROWN (continuity intact): food={ms(grown_food)}  memory-gap={ms(grown_gap)}")
    out(f"NAIVE re-ground from scratch ({RECOVER} gens): food={naive_food:.2f}")
    out("-" * 80)
    out(f"{'coma C':>7} | {'wake food':>12} {'wake mem-gap':>13} | {'recovered food':>15} {'recovered gap':>14}")
    for C in COMAS:
        out(f"{C:>7} | {ms(dose[C]['wfood']):>12} {ms(dose[C]['wgap']):>13} | "
            f"{ms(dose[C]['ffood']):>15} {ms(dose[C]['fgap']):>14}")
    out("-" * 80)
    out("EROSION: does the capability decay with coma length? (use-it-or-lose-it)")
    out("  wake food: " + "  ".join(f"C{C}={np.mean(dose[C]['wfood']):.2f}" for C in COMAS))
    out("  wake gap : " + "  ".join(f"C{C}={np.mean(dose[C]['wgap']):.2f}" for C in COMAS))
    out("WALLIS TEST: coma'd recovered food vs naive (same recovery budget)")
    for C in COMAS:
        ff = np.mean(dose[C]["ffood"])
        v = "FASTER than naive (residual structure)" if ff > naive_food + 0.3 \
            else ("~ naive (point of no return)" if ff < naive_food + 0.1 else "marginal")
        out(f"  C={C:>3}: recovered {ff:.2f} vs naive {naive_food:.2f}  -> {v}")
    out("done"); LOG.close()

    # ---- charts ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(COMAS)))
    for C, col in zip(COMAS, cmap):
        xs = [p[0] for p in recover_curves[C][0]]
        ys = np.mean([[p[1] for p in seedcurve] for seedcurve in recover_curves[C]], axis=0)
        ax1.plot(xs, ys, color=col, lw=2, label=f"coma={C}")
    nxs = [p[0] for p in naive_curves[0]]
    nys = np.mean([[p[1] for p in c] for c in naive_curves], axis=0)
    ax1.plot(nxs, nys, "k--", lw=2, label="naive (from scratch)")
    ax1.axhline(np.mean(grown_food), color="#888", ls=":", label="pre-coma food")
    ax1.set_xlabel("recovery generation"); ax1.set_ylabel("food eaten per life")
    ax1.set_title("Waking up: re-grounding vs coma length", weight="bold")
    ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    wf = [np.mean(dose[C]["wfood"]) for C in COMAS]; ff = [np.mean(dose[C]["ffood"]) for C in COMAS]
    wg = [np.mean(dose[C]["wgap"]) for C in COMAS]; fg = [np.mean(dose[C]["fgap"]) for C in COMAS]
    ax2.plot(COMAS, ff, "o-", color="#1b5e9e", lw=2.5, label="recovered food")
    ax2.plot(COMAS, wf, "s--", color="#c44", lw=2, label="food at wake (eroded)")
    ax2.plot(COMAS, fg, "^-", color="#2a8", lw=2, label="recovered memory-gap")
    ax2.axhline(naive_food, color="k", ls="--", lw=1.5, label="naive ceiling")
    ax2.set_xlabel("coma length (gens of un-grounded drift)"); ax2.set_ylabel("food / gap")
    ax2.set_title("Dose-response: longer coma, worse re-grounding", weight="bold")
    ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    plt.tight_layout()
    p = os.path.join(HERE, "coma_regrounding.png")
    plt.savefig(p, dpi=120); print("saved", p)
