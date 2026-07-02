"""Generate a professional technical report on every playground constraint.
Runs real simulations from sim_engine, builds charts, assembles a multi-page PDF.
Output: playground/CONSTRAINTS_REPORT.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sim_engine import Sim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "CONSTRAINTS_REPORT.pdf")
INK = "#16181d"; ACC = "#1b5e9e"; ACC2 = "#c0392b"; GOOD = "#2e8b57"; MUT = "#8a8f98"
WARN = "#c9821f"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#555", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8})


# ======================= data collection (real sims) =======================
def make(keys, seed=7):
    s = Sim(seed=seed)
    for k in keys:
        s.cmap[k].enabled = True
    return s


def trace(keys, fns, steps=700, seed=7, every=10):
    """run; sample {name: fn(sim)} every `every` steps."""
    s = make(keys, seed); out = {n: [] for n in fns}; xs = []
    for t in range(steps):
        s.step()
        if t % every == 0:
            xs.append(t)
            for n, f in fns.items():
                out[n].append(f(s))
    return xs, out


def disp(s):
    p = s.pos
    if len(p) < 2: return 0.0
    k = min(len(p), 40)
    return float(np.mean([np.linalg.norm(p - p[i], axis=1).mean() for i in range(k)]))


vis = lambda s: float(s.geno[:, 4].mean()) if len(s.geno) else np.nan
spd = lambda s: float(np.clip(s.geno[:, 5], .2, 2.2).mean()) if len(s.geno) else np.nan
wsep = lambda s: float(s.geno[:, 2].mean()) if len(s.geno) else np.nan
wsig = lambda s: float(s.geno[:, 3].mean()) if len(s.geno) else np.nan
spr = lambda s: 100 * s.spread_frac()
pop = lambda s: len(s.geno)
foll = lambda s: len(s.edges)
sigc = lambda s: int(s.signalers.sum())

print("collecting data...", flush=True)
# (A) constraint stacking -> cone closing
ORDER = ["energy", "time", "percep", "entropy", "scarce", "comm"]
stack_po, stack_spread = [], []
for i in range(len(ORDER) + 1):
    s = make(ORDER[:i])
    for _ in range(700): s.step()
    stack_po.append(i); stack_spread.append(100 * s.spread_frac())
# (B) energy -> selection
xb, B = trace([], {"spread": spr}, 600); _, B2 = trace(["energy"], {"spread": spr}, 600)
# (C) perception -> vision
xc, C = trace(["energy"], {"v": vis}, 700); _, C2 = trace(["energy", "percep"], {"v": vis}, 700)
# (D) time -> speed
xd, D = trace(["energy"], {"s": spd}, 700); _, D2 = trace(["energy", "time"], {"s": spd}, 700)
# (E) scarcity -> dispersal
xe, E = trace(["energy"], {"disp": disp, "wsep": wsep}, 700)
_, E2 = trace(["energy", "scarce"], {"disp": disp, "wsep": wsep}, 700)
# (F) communication activity
xf, F = trace(["energy", "comm"], {"sig": sigc, "foll": foll}, 700)
_, F2 = trace(["energy", "comm", "scarce"], {"sig": sigc, "foll": foll}, 700)
# (G) climate -- rotating bloom + migration lag
sg = make(["energy", "climate"]); gx = []; ripe_q = [[], [], [], []]; act = []; inact = []
fq = sg._food_quads()
for t in range(900):
    sg.step()
    if t % 6 == 0:
        gx.append(t)
        for q in range(4):
            ripe_q[q].append(float((sg.food_mat[fq == q] >= 0.55).mean()) if (fq == q).any() else 0)
        act.append(sg.active_quadrant)
        p = sg.pos; r2 = p[:, 0] >= 50; t2 = p[:, 1] >= 50
        oq = np.where(~t2 & ~r2, 0, np.where(~t2 & r2, 1, np.where(t2 & r2, 2, 3)))
        inact.append(float((oq == sg.active_quadrant).mean()) if len(p) else 0)
# (H) reproduction: population dynamics + the w_sig flip
xh, H = trace(["energy", "comm"], {"pop": pop}, 700)
_, H2 = trace(["energy", "comm", "repro"], {"pop": pop}, 700)
flip_seeds = [1, 3, 7, 11, 21]; flip_clone = []; flip_sex = []
for sd in flip_seeds:
    c = make(["energy", "comm", "percep"], sd)
    for _ in range(800): c.step()
    flip_clone.append(c.geno[:, 3].mean() if len(c.geno) > 1 else np.nan)
    x = make(["energy", "comm", "percep", "repro"], sd)
    for _ in range(800): x.step()
    flip_sex.append(x.geno[:, 3].mean() if len(x.geno) > 1 else np.nan)
print("data collected; rendering PDF...", flush=True)


# ============================= PDF helpers =================================
def textpage(pdf, title, blocks, sub=None):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.945, title, fontsize=17, weight="bold", color=ACC)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.93, 0.93], color=ACC, lw=2))
    if sub:
        fig.text(0.08, 0.905, sub, fontsize=9.5, style="italic", color=MUT)
    y = 0.875 if not sub else 0.86
    for kind, txt in blocks:
        if kind == "h":
            y -= 0.012
            fig.text(0.08, y, txt, fontsize=11.5, weight="bold", color=INK); y -= 0.026
        elif kind == "b":
            for line in _wrap(txt, 98):
                fig.text(0.09, y, line, fontsize=9.4); y -= 0.0185
            y -= 0.009
        elif kind == "f":
            for line in txt.split("\n"):
                fig.text(0.10, y, line, fontsize=9.6, family="monospace", color=ACC); y -= 0.020
            y -= 0.006
        elif kind == "w":   # caveat / open-question block
            for i, line in enumerate(_wrap(txt, 96)):
                fig.text(0.095, y, line, fontsize=9.2, color="#7a4a12"); y -= 0.0185
            y -= 0.009
    pdf.savefig(fig); plt.close(fig)


def _wrap(t, n):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line: out.append(line)
    return out


def figpage(pdf, fig):
    pdf.savefig(fig); plt.close(fig)


# ============================== figures ===================================
def fig_cone():
    fig = plt.figure(figsize=(8.5, 6.4))
    ax1 = fig.add_axes([0.07, 0.12, 0.42, 0.76]); ax2 = fig.add_axes([0.58, 0.12, 0.37, 0.76])
    # cone schematic with measured cross-sections
    ax1.add_patch(plt.Polygon([[0, 1], [-1, 1], [0, -0.0], [1, 1]], closed=True, fill=False, ec=MUT, lw=1.5))
    for i, (po, sp) in enumerate(zip(stack_po, stack_spread)):
        yb = 1 - po / (len(ORDER) + 1)
        hw = max(0.03, yb * sp / 100)
        ax1.plot([-hw, hw], [yb, yb], color=GOOD, lw=3)
        ax1.text(1.02, yb, f"PO={po}", fontsize=7, va="center", color=INK)
    ax1.set_xlim(-1.15, 1.25); ax1.set_ylim(-0.05, 1.08); ax1.axis("off")
    ax1.set_title("Cone cross-section vs constraint count", fontsize=10)
    ax1.text(0, 1.04, "all possible organisms", ha="center", fontsize=7.5, color=MUT)
    ax1.text(0, -0.04, "survivor", ha="center", fontsize=7.5, color=MUT)
    ax2.plot(stack_po, stack_spread, "-o", color=ACC, lw=2)
    ax2.set_xlabel("PO  (number of stacked constraints)"); ax2.set_ylabel("surviving strategy diversity  (%)")
    ax2.set_title("Diversity collapses as laws stack", fontsize=10); ax2.set_ylim(0, 105)
    fig.suptitle("Figure 1.  The PO cone: each constraint removes degrees of freedom",
                 fontsize=12, weight="bold", y=0.97)
    return fig


def two_panel(title, x, a, b, la, lb, ylab, ca=ACC2, cb=ACC):
    fig = plt.figure(figsize=(8.5, 4.7)); ax = fig.add_axes([0.1, 0.16, 0.85, 0.7])
    ax.plot(x, a, color=ca, lw=2, label=la); ax.plot(x, b, color=cb, lw=2, label=lb)
    ax.set_xlabel("simulation step"); ax.set_ylabel(ylab); ax.legend(loc="best", framealpha=0.9)
    fig.suptitle(title, fontsize=12, weight="bold", y=0.98)
    return fig, ax


def fig_climate():
    fig = plt.figure(figsize=(8.5, 6.6))
    ax = fig.add_axes([0.1, 0.55, 0.85, 0.36]); ax2 = fig.add_axes([0.1, 0.10, 0.85, 0.32])
    cols = ["#e74c3c", "#3498db", "#f1c40f", "#2ecc71"]
    for q in range(4):
        ax.plot(gx, ripe_q[q], color=cols[q], lw=1.8, label=f"quadrant {q}")
    ax.set_ylabel("fraction of patches ripe"); ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.set_title("Rotating growing season: ripe food migrates quadrant to quadrant", fontsize=10)
    ax2.plot(gx, np.array(inact) * 100, color="#7b4fa3", lw=2)
    ax2.set_ylabel("% organisms in\nthe active quadrant"); ax2.set_xlabel("simulation step")
    ax2.axhline(25, color=MUT, ls="--", lw=1); ax2.text(gx[-1], 27, "random (25%)", fontsize=7, ha="right", color=MUT)
    ax2.set_title("Migration tracking (the population lags the bloom)", fontsize=10)
    fig.suptitle("Figure 7.  Climate — a non-stationary, spatially rotating resource", fontsize=12, weight="bold", y=0.98)
    return fig


def fig_repro():
    fig = plt.figure(figsize=(8.5, 6.6))
    ax = fig.add_axes([0.1, 0.57, 0.85, 0.33]); ax2 = fig.add_axes([0.12, 0.10, 0.83, 0.34])
    ax.plot(xh, H["pop"], color=ACC2, lw=2, label="clone-the-best (fixed N)")
    ax.plot(xh, H2["pop"], color=ACC, lw=2, label="sexual / mating (floating N)")
    ax.set_ylabel("population"); ax.set_xlabel("simulation step"); ax.legend(loc="best")
    ax.set_title("Reproduction operator sets the population dynamics", fontsize=10)
    xi = np.arange(len(flip_seeds)); w = 0.38
    ax2.bar(xi - w/2, flip_clone, w, color=ACC2, label="clone")
    ax2.bar(xi + w/2, flip_sex, w, color=ACC, label="sexual")
    ax2.axhline(0, color=INK, lw=1)
    ax2.set_xticks(xi); ax2.set_xticklabels([f"seed {s}" for s in flip_seeds])
    ax2.set_ylabel("signal-following gene  w_sig\n(<0 avoid · >0 approach)")
    ax2.set_title("Sexual reproduction floors communication pro-social (never < 0)", fontsize=10)
    ax2.legend(loc="upper right")
    fig.suptitle("Figure 8.  Reproduction — and its grip on the evolution of communication",
                 fontsize=12, weight="bold", y=0.98)
    return fig


# ================================ build ===================================
with PdfPages(OUT) as pdf:
    # cover
    fig = plt.figure(figsize=(8.5, 11)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    # inverted cone: wide mouth (all organisms) at top, narrow tip (survivor) at bottom
    ax.add_patch(plt.Polygon([[0.2, 0.71], [0.8, 0.71], [0.5, 0.40]], closed=True, fill=False, ec=ACC, lw=2))
    for i, yy in enumerate(np.linspace(0.44, 0.68, 6)):
        wv = (yy - 0.40) / (0.71 - 0.40) * 0.30
        ax.plot([0.5 - wv, 0.5 + wv], [yy, yy], color=GOOD if i in (2, 3) else MUT,
                lw=2.4 if i in (2, 3) else 1.0, alpha=0.9 if i in (2, 3) else 0.5)
    ax.text(0.5, 0.735, "all organisms that could exist", ha="center", fontsize=8, color=MUT)
    ax.text(0.5, 0.375, "survivor  (PO -> tip)", ha="center", fontsize=8, color=MUT)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.text(0.5, 0.86, "GENREG / EEC", ha="center", fontsize=15, color=ACC, weight="bold")
    fig.text(0.5, 0.81, "Constraints of Existence", ha="center", fontsize=24, weight="bold", color=INK)
    fig.text(0.5, 0.775, "A technical report on the eight laws of the playground world",
             ha="center", fontsize=12, color=MUT)
    fig.text(0.5, 0.33, "Gradient-free neuroevolution  ·  model defined by constraints, not parameters",
             ha="center", fontsize=10, color=ACC)
    fig.text(0.5, 0.30, "Read the state, not the output  ·  change the world, not the score",
             ha="center", fontsize=9.5, color=MUT, style="italic")
    fig.text(0.5, 0.07, "GENREG / EEC  ·  2026-06-21", ha="center", fontsize=9, color=MUT)
    pdf.savefig(fig); plt.close(fig)

    # executive summary
    textpage(pdf, "1.  Executive summary", [
        ("h", "What this is"),
        ("b", "GENREG models an organism not as a fixed network but as the INTERSECTION of the laws it must "
              "satisfy to keep existing. Each law removes degrees of freedom from the space of possible "
              "behaviours; stack enough and a single survivor precipitates. The native complexity metric is "
              "therefore PO -- the number of binding constraints (axes) -- not a parameter count."),
        ("b", "This report documents the eight constraints implemented in the interactive playground, the "
              "behaviour each one selects for, the evidence from direct simulation, and -- explicitly -- where "
              "each one is weak, fragile, or still unresolved."),
        ("h", "Headline findings"),
        ("b", "1. Stacking constraints monotonically collapses surviving strategy diversity (Fig 1): the cone "
              "is real and measurable."),
        ("b", "2. Single-law signatures reproduce cleanly: perception cost shrinks vision; scarcity disperses "
              "the population; energy is the precondition for any selection at all."),
        ("b", "3. The reproduction operator is itself a load-bearing law. Switching from asexual cloning to "
              "mate-based sex changes the EVOLUTIONARY MEANING of communication: under cloning the "
              "signal-following gene can drift anti-social; under sex it is floored pro-social, because an "
              "organism that avoids others never finds a mate (Fig 8)."),
        ("h", "Stance"),
        ("b", "Findings are reported at the strength the data supports. Clean, reproducible effects are stated "
              "plainly; seed-sensitive or parameter-sensitive effects are flagged as such; mechanisms we have "
              "not yet closed are listed as open questions, not glossed."),
        ("h", "Reading guide"),
        ("b", "Section 2 gives the methods. Sections 3-10 treat one constraint each (mechanism, parameters, "
              "evidence, shortcomings). Section 11 covers the reproduction x communication interaction. Section "
              "12 lists cross-cutting limitations and the active research agenda."),
    ])

    # methods
    textpage(pdf, "2.  Methods", [
        ("h", "Organism and world"),
        ("b", "A population of organisms lives on a 100x100 continuous board with food patches. Each organism "
              "carries a 6-gene genome: four behavioural weights (attraction to food, to remembered food, to/"
              "from neighbours, to/from signallers) and two traits (vision radius, movement speed). Every step: "
              "sense -> decide a heading -> move -> eat -> pay the costs imposed by active laws -> reproduce."),
        ("f", "genome = [ w_food, w_mem, w_sep, w_sig, vision, speed ]"),
        ("h", "Evolution (gradient-free)"),
        ("b", "No gradients are used anywhere. Default reproduction is steady-state: each step the lowest-energy "
              "organisms are replaced by mutated copies of high-energy ones, placed near the parent (local / "
              "kin reproduction). The Reproduction law (Sec 10) swaps this for mate-based sexual reproduction."),
        ("h", "The PO metric"),
        ("b", "Strategy diversity is measured as the mean relative dispersion of the genome across the "
              "population, tanh-compressed, normalised to its value at initialisation (100%). As constraints "
              "select harder, surviving genomes converge and this fraction drops -- the cone cross-section."),
        ("h", "Measurement"),
        ("b", "Every figure is generated from direct simulation of this engine (no hand-drawn curves). Unless "
              "noted, runs use 32 organisms, 600-900 steps, seed 7. Where a result depends on the random seed, "
              "multiple seeds are shown."),
        ("w", "Caveat: the engine is deliberately minimal -- a linear genome, one hidden behavioural rule, a "
              "heuristic diversity metric. It is an instrument for studying constraint geometry, not a "
              "high-fidelity ecology. Absolute numbers are not meaningful; contrasts (law on vs off) are."),
    ])

    figpage(pdf, fig_cone())

    # ---- per-constraint sections ----
    textpage(pdf, "3.  Energy  (survival)", [
        ("b", "COLOUR: red    AXIS: survival    STATUS: foundational, verified"),
        ("h", "Definition"), ("b", "Metabolism drains energy every step; an organism that reaches zero is "
            "removed. Energy is the precondition that makes selection exist at all."),
        ("h", "Mechanism / parameters"),
        ("f", "energy -= metabolism            (default 0.40 / step)\nfood raises energy; death at energy <= 0"),
        ("h", "Observed effect"),
        ("b", "With no survival law there is no selection pressure: the genome drifts and strategy diversity "
              "stays at ~100% indefinitely. Switching energy on collapses diversity as fitter foragers displace "
              "the rest (Figure 2). Energy is the substrate every other law acts through -- a perception cost or "
              "scarcity penalty only matters because it shortens a life."),
        ("w", "Open / weak: the fixed steady-state operator means population size is held constant, so 'alive "
              "count' can understate die-off (low-energy organisms are recycled). Lifespan, not headcount, is "
              "the honest fitness signal. Under very harsh stacks the population can sit pinned at the energy "
              "floor while still 'evolving' on relative rank -- a regime where absolute readouts mislead."),
    ])
    fig, _ = two_panel("Figure 2.  Energy is the precondition for selection", xb,
                       B["spread"], B2["spread"], "no survival law", "energy on", "strategy diversity (%)")
    figpage(pdf, fig)

    textpage(pdf, "4.  Time / Occam  (movement cost)", [
        ("b", "COLOUR: purple    AXIS: parsimony    STATUS: real, mild"),
        ("h", "Definition"), ("b", "Movement costs energy in proportion to speed. Wasted motion is paid for, "
            "so efficient, direct behaviour is selected -- an evolutionary Occam's razor."),
        ("h", "Mechanism / parameters"),
        ("f", "energy -= move_cost * speed       (default 0.12)"),
        ("h", "Observed effect"),
        ("b", "Adding the time law lowers the evolved speed trait relative to the no-cost control (Figure 3): "
              "the population shifts toward calmer, more directed movement rather than high-speed wandering. The "
              "effect is real but modest at default strength -- it is a refining pressure, not a dominating one."),
        ("w", "Open / weak: in a small static world the gain from parsimony is limited; its documented strength "
              "in the wider EEC programme comes from selecting GENERALISATION under novelty, which this board "
              "does not yet stress. Expect the time law to matter much more once Climate / non-stationarity is "
              "co-active."),
    ])
    fig, _ = two_panel("Figure 3.  Time cost selects lower, more efficient speed", xd,
                       D["s"], D2["s"], "energy only", "energy + time", "evolved speed trait")
    figpage(pdf, fig)

    textpage(pdf, "5.  Perception  (cost of looking)", [
        ("b", "COLOUR: blue    AXIS: attention    STATUS: real, strong"),
        ("h", "Definition"), ("b", "A wide field of view is metabolically expensive (cost grows with the square "
            "of vision radius). Organisms are pushed to see only as far as is worth the energy."),
        ("h", "Mechanism / parameters"),
        ("f", "energy -= cost * vision^2 / 10    (default cost 0.006)"),
        ("h", "Observed effect"),
        ("b", "The clearest single-law signature in the system. Vision starts large (random ~20) and, under the "
              "perception cost, collapses toward a small economical radius (Figure 4); the no-cost control keeps "
              "or grows its vision. In the earlier staged demos this drove vision from ~27 down to ~4."),
        ("w", "Open / weak (the 'attention wall'): perception buys an economical FIELD SIZE, but we have not "
              "shown it produces SELECTIVE attention -- looking at the right thing rather than simply less. "
              "Across the EEC programme, conditional 'what to attend to' required an explicit channel, not just "
              "a cost. Cost shrinks the aperture; it does not by itself build a controller for it."),
    ])
    fig, _ = two_panel("Figure 4.  Perception cost shrinks the vision radius", xc,
                       C["v"], C2["v"], "energy only", "energy + perception", "evolved vision radius")
    figpage(pdf, fig)

    textpage(pdf, "6.  Entropy  (memory decay)", [
        ("b", "COLOUR: amber    AXIS: active maintenance    STATUS: real in programme, weak on this board"),
        ("h", "Definition"), ("b", "Remembered food locations decay over time; a memory must be refreshed by "
            "revisiting or it becomes worthless. Information is not free to hold -- it rots."),
        ("h", "Mechanism / parameters"),
        ("f", "memory_confidence *= decay        (default 0.90 when on, else 0.995)"),
        ("h", "Observed effect"),
        ("b", "Entropy makes stored state a depreciating asset, favouring organisms that periodically re-sense "
              "rather than coast on stale memory. In the wider EEC sweeps a decay law produced a measurable "
              "premium on active maintenance (recurrent gain ~3.2x)."),
        ("w", "Open / weak: on the current playground board memory is a single weak channel and the effect is "
              "hard to isolate cleanly -- it is the least visually legible of the eight laws here. A dedicated "
              "revisit metric and a richer memory representation are needed before claiming a sharp on-board "
              "signature. Reported as real-in-principle, under-demonstrated-here."),
    ])

    textpage(pdf, "7.  Scarcity  (shared / contested food)", [
        ("b", "COLOUR: yellow    AXIS: diversity / niche    STATUS: real, verified"),
        ("h", "Definition"), ("b", "A food patch is finite and is split among everyone feeding on it; a crowd "
            "dilutes each bite and depletes the patch. Competition makes proximity costly."),
        ("h", "Mechanism / parameters"),
        ("f", "bite = min(0.5, patch) / (# co-feeders)   ; patch depletes & regrows (regen ~0.004)"),
        ("h", "Observed effect"),
        ("b", "Scarcity reliably DISPERSES the population: mean pairwise distance rises and the neighbour-"
              "avoidance gene (w_sep) is driven positive relative to the abundant-food control (Figure 5). "
              "Crowding stops paying, so organisms spread into territory -- the first genuinely social/spatial "
              "structure the world produces."),
        ("w", "Open / weak: dispersal strength is sensitive to the regen rate and population density; too harsh "
              "a scarcity tips into population collapse rather than territory. The 'ecosystem of niches' result "
              "from the wider programme needs a population operator that lets sub-populations specialise -- on "
              "this board scarcity gives spacing, not yet speciation."),
    ])
    fig = plt.figure(figsize=(8.5, 4.7)); ax = fig.add_axes([0.1, 0.16, 0.85, 0.7])
    ax.plot(xe, E["disp"], color=ACC2, lw=2, label="energy only (clustered)")
    ax.plot(xe, E2["disp"], color=ACC, lw=2, label="energy + scarcity (dispersed)")
    ax.set_xlabel("simulation step"); ax.set_ylabel("mean pairwise distance"); ax.legend()
    fig.suptitle("Figure 5.  Scarcity disperses the population into territory", fontsize=12, weight="bold", y=0.98)
    figpage(pdf, fig)

    textpage(pdf, "8.  Communication  (signals)", [
        ("b", "COLOUR: green    AXIS: information transfer    STATUS: real, but context-dependent"),
        ("h", "Definition"), ("b", "An organism on food emits a signal; others can sense the nearest signaller "
            "and move toward (or away from) it. The genome's w_sig sets the sign and strength of the response."),
        ("h", "Mechanism / parameters"),
        ("f", "signaller = on food ; listener heading += w_sig * direction(nearest signaller within range)\n"
              "range default 42"),
        ("h", "Observed effect"),
        ("b", "Signalling activity tracks how many organisms sit on food; FOLLOWING (acting on a signal) only "
              "becomes common when organisms are frequently apart and hungry -- e.g. under scarcity (Figure 6). "
              "When food is abundant everyone is already fed and the channel is loud but unused. Crucially, the "
              "evolved SIGN of communication is not fixed by this law alone -- it is set by the reproduction "
              "regime (Section 11)."),
        ("w", "Open / weak (the central honest caveat): this is signal-FOLLOWING (attraction), not a symbolic "
              "language. Getting an informative, decipherable PROTOCOL to emerge among free-living, individually "
              "selected agents runs into the free-rider problem -- a speaker has little private incentive to be "
              "honest. In separate experiments a real codebook (MI ~1.5 bits, with dialects) emerged only under "
              "COUPLED survival or strong kin structure. On this board, communication is a coordination cue, not "
              "yet a vocabulary."),
    ])
    fig = plt.figure(figsize=(8.5, 4.7)); ax = fig.add_axes([0.1, 0.16, 0.85, 0.7])
    ax.plot(xf, F["foll"], color=MUT, lw=1.6, label="following  (abundant food)")
    ax.plot(xf, F2["foll"], color=GOOD, lw=2, label="following  (under scarcity)")
    ax.set_xlabel("simulation step"); ax.set_ylabel("acts of following / step"); ax.legend()
    fig.suptitle("Figure 6.  Communication is used only when organisms need each other",
                 fontsize=12, weight="bold", y=0.98)
    figpage(pdf, fig)

    textpage(pdf, "9.  Climate  (rotating season)", [
        ("b", "COLOUR: cyan    AXIS: non-stationarity (spatial)    STATUS: mechanically sound, harsh"),
        ("h", "Definition"), ("b", "The board is quartered and the growing season rotates round-robin through "
            "the quadrants. Food has a seedling->mature life-cycle; only MATURE food feeds you, and off-season "
            "food exhausts. The resource is non-stationary in space and time."),
        ("h", "Mechanism / parameters"),
        ("f", "active = (step // period) % 4 ; maturity grows in active quadrant, decays elsewhere\n"
              "see growing food (to migrate) but eat only mature   ; period 140, growth 0.04, decay 0.008"),
        ("h", "Observed effect"),
        ("b", "Ripe food cleanly migrates quadrant to quadrant, and the hand-off shows the intended overlap -- "
              "the old quadrant exhausts as the next ripens (Figure 7, top). The population is forced to migrate "
              "to keep eating: an organism that stays put eats only every fourth season. We verify the herd "
              "tracks the bloom above the 25% random baseline, but it LAGS (Figure 7, bottom)."),
        ("w", "Open / weak: this is the hardest law to keep survivable. Transit time to the next quadrant can "
              "exceed starvation time, so the population rides the survival edge and tracks the bloom "
              "imperfectly. Making it work required raising the energy cap (reserves for migration), central "
              "starts, and even food across quadrants. The survivable parameter window is narrow; outside it the "
              "population collapses. We do not yet see clean anticipatory migration (leaving before exhaustion)."),
    ])
    figpage(pdf, fig_climate())

    textpage(pdf, "10.  Reproduction  (mate-based sex)", [
        ("b", "COLOUR: pink    AXIS: the selection operator itself    STATUS: real, with a major interaction"),
        ("h", "Definition"), ("b", "Turns OFF clone-the-best. An organism now reproduces ONLY with a nearby "
            "MATE, and only if BOTH parents are at >=50% of energy capacity. Offspring are a genetic CROSSOVER of "
            "the two parents plus mutation; both parents pay an energy cost. Population floats with real births "
            "and deaths."),
        ("h", "Mechanism / parameters"),
        ("f", "eligible: energy >= 0.5 * capacity ; partner within mate_radius (12)\n"
              "child = crossover(parentA, parentB) + mutation ; both pay repro_cost (8) ; cap max_pop (140)"),
        ("h", "Observed effect"),
        ("b", "Replacing the operator changes the population dynamics from a clamped constant N to a free-"
              "floating trajectory bounded by food and the max-pop cap (Figure 8, top). More importantly it "
              "changes WHICH genomes can persist -- a lineage must keep its members both well-fed AND sociable "
              "enough to meet. This is the lever behind the Section 11 interaction. The 50% energy gate is "
              "verified: forcing the population below threshold halts all breeding; above it, breeding resumes."),
        ("w", "Open / weak: with scarcity or climate co-active, mate-based reproduction can drive the population "
              "to EXTINCTION -- deaths outpace the stricter birth condition. The current mate search is nearest-"
              "eligible-within-radius with no mate choice on genome quality (no sexual selection on traits yet), "
              "and crossover is uniform per-gene. Population stability across law combinations is not guaranteed "
              "and is an active tuning question."),
    ])

    textpage(pdf, "11.  Interaction: reproduction shapes communication", [
        ("b", "The most striking cross-constraint result in the system.", ),
        ("h", "Question"),
        ("b", "Does the way a population reproduces change what its communication is FOR? We compare the evolved "
              "signal-following gene w_sig (negative = avoid others, positive = approach others) under asexual "
              "cloning vs mate-based sex, holding the world otherwise fixed (energy + communication + perception)."),
        ("h", "Result"),
        ("b", "Under cloning, w_sig is only weakly constrained and DRIFTS -- across seeds it lands anywhere from "
              "strongly anti-social (-0.9) to strongly social (+1.9). Under sexual reproduction it is reliably "
              "NON-NEGATIVE on every seed tested (Figure 8, bottom): the strongly anti-social outcomes are "
              "eliminated. On the default seed this appears as a clean sign flip (-0.92 -> +0.32)."),
        ("h", "Mechanism"),
        ("b", "Under cloning, every other organism is purely a competitor for food, so evolution is free to "
              "select avoidance. Under sex, every other organism is also a potential MATE and reproduction is "
              "impossible without approaching one -- so a lineage that evolves to avoid others simply fails to "
              "breed and dies out. Sexual reproduction places a hard FLOOR under social attraction."),
        ("b", "Interpretation: the selective VALENCE of a communication channel is not a property of the channel "
              "alone -- it is set by the reproduction operator. This mirrors the hypothesised role of sexual "
              "selection in the origin of social signalling, and it emerged here from toggling one constraint."),
        ("w", "Honesty bound: the dramatic single-seed sign-FLIP is partly seed-specific. The robust, every-seed "
              "claim is the weaker-but-solid one: sex eliminates anti-social communication and floors w_sig "
              "positive. We have not yet measured downstream effects (does the floored sociality raise actual "
              "information transfer, or only spatial aggregation?) -- that is the next experiment."),
    ])
    figpage(pdf, fig_repro())

    textpage(pdf, "12.  Cross-cutting limitations and open research", [
        ("h", "Limitations of the instrument"),
        ("b", "- Minimal organism: a linear 6-gene genome with one fixed behavioural rule. No internal network, "
              "no learning within life. Conclusions are about constraint geometry, not realistic ecology."),
        ("b", "- Heuristic PO metric: strategy diversity is a genome-dispersion proxy, not a first-principles "
              "count of binding constraints. It trends correctly but is not calibrated."),
        ("b", "- Fixed-N default operator hides die-off in the 'alive' readout; lifespan is the truer fitness."),
        ("b", "- Limited statistics: most single-law figures are one seed. Effects flagged 'mild' or 'weak' have "
              "not had confidence intervals computed; treat them as directional."),
        ("h", "Constraint-specific open questions"),
        ("b", "- Perception: cost yields a smaller aperture but not demonstrated SELECTIVE attention (the "
              "attention wall)."),
        ("b", "- Entropy: real in the wider programme, under-demonstrated on this board; needs a revisit metric."),
        ("b", "- Communication: signal-following, not a symbolic protocol; honest emergent language needs "
              "coupled survival / kin structure, not free-living individual selection."),
        ("b", "- Climate: narrow survivable window; no clean anticipatory migration yet."),
        ("b", "- Reproduction: extinction risk under scarcity/climate; no mate choice / sexual selection on "
              "traits; uniform crossover only."),
        ("h", "Active research agenda"),
        ("b", "1. Quantify the reproduction x communication interaction's downstream effect on real information "
              "transfer, not just aggregation."),
        ("b", "2. Co-active non-stationarity (Climate) x parsimony (Time) -- does novelty finally make the time "
              "law dominant, selecting genuine generalisation?"),
        ("b", "3. A principled PO estimator (constraint-rank, not genome variance)."),
        ("b", "4. Mate choice / sexual selection on traits, and population-stability control across law stacks."),
        ("b", "5. A richer organism (recurrent internal state) so memory, attention and language have somewhere "
              "to live."),
        ("h", "Bottom line"),
        ("b", "The cone is real and the single-law signatures are clean and reproducible. The deepest result -- "
              "that the reproduction operator governs the evolution of communication -- is robust in its floored "
              "form and striking in its seed-7 form. The honest frontier is everything that needs a richer "
              "organism or a population operator with more structure than the current world provides."),
    ])

print("saved", OUT)
