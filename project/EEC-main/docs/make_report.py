"""Generate the EEC findings report as a multi-page PDF (matplotlib PdfPages).
Reads the on-disk sweep JSONs; other run results are tabulated inline.
Output: docs/EEC_report.pdf
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OBS = os.path.join(ROOT, "experiments", "observability_board")
GRID = os.path.join(ROOT, "experiments", "cone_interior_grid")

INK = "#1a1a1a"; ACC = "#1b5e9e"; ACC2 = "#c44"; GOOD = "#2e8b57"; MUT = "#888"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#444", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

# ----- on-disk data -----
occ = json.load(open(os.path.join(OBS, "board_occ.json")))
noi = json.load(open(os.path.join(OBS, "board_noise.json")))
stk = json.load(open(os.path.join(OBS, "board_stack.json")))
grid = json.load(open(os.path.join(GRID, "grid2d.json")))

# ----- inline run results -----
SWAP = {"energy": (0.539, 0.258), "mortality": (0.650, 0.683), "both": (0.586, 0.355)}  # (competence, gain) occ=0
ORTHO = {"base": (0.061, 0.0), "repro": (0.257, 0.111), "occ": (0.367, 0.0), "both": (0.478, 0.067)}  # (gain, gini)
PO_LAT = [  # (label, n_axes, comp, gain, gini, M)
    ("survival", 1, 0.280, 0.138, 0.000, 4.12),
    ("+parsimony", 2, 0.254, 0.146, 0.000, 2.21),
    ("+selection", 2, 0.315, 0.302, 0.104, 5.06),
    ("+observation", 2, 0.279, 0.266, 0.000, 2.98),
    ("+sel+parsi", 3, 0.420, 0.167, 0.086, 2.17),
    ("+obs+parsi", 3, 0.280, 0.313, 0.000, 2.19),
    ("+obs+sel", 3, 0.372, 0.361, 0.080, 2.50),
    ("+obs+sel+parsi", 4, 0.368, 0.129, 0.063, 2.08)]
CATALOG = [  # (law, axis, verdict, key)
    ("ENERGY", "survival", "REAL", "base law; lifespan = fitness"),
    ("TIME / Occam", "parsimony", "REAL", "selects generalization"),
    ("MEMORY-RENT", "parsimony", "REAL", "M evolves against energy"),
    ("ENTROPY (decay)", "maintenance", "REAL", "recurrent gain x3.2"),
    ("OCCLUSION", "observation", "REAL*", "gain 0.06->0.37, Goldilocks peak rho~0.4"),
    ("NOISE", "observation", "WEAK", "gain 0.06->0.14; missing >> corrupted"),
    ("SCARCITY", "diversity", "GATED", "text niches 1.5->7.5; inert long-range"),
    ("REPRODUCTION-COST", "selection", "GATED", "surplus +41%, Gini 0->0.11; text starves"),
    ("MORTALITY", "survival (alt)", "SWAP", "viable alone (0.65>0.54); redundant on energy"),
    ("PERCEPTION-COST", "attention", "WALL", "economy yes (5x); selective attn no"),
    ("NON-STATIONARITY", "adaptability", "REAL", "diversity 1.3->2.0; recovers 84%")]
VC = {"REAL": GOOD, "REAL*": GOOD, "WEAK": "#c9a227", "GATED": ACC, "SWAP": "#7b4fa3", "WALL": ACC2}


def textpage(pdf, title, blocks):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.93, title, fontsize=18, weight="bold", color=ACC)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.915, 0.915], color=ACC, lw=2))
    y = 0.875
    for kind, txt in blocks:
        if kind == "h":
            y -= 0.012
            fig.text(0.08, y, txt, fontsize=12, weight="bold", color=INK); y -= 0.028
        elif kind == "b":
            for line in _wrap(txt, 96):
                fig.text(0.09, y, line, fontsize=9.6); y -= 0.020
            y -= 0.010
        elif kind == "f":  # formula / monospace box
            for line in txt.split("\n"):
                fig.text(0.11, y, line, fontsize=10.5, family="monospace", color=ACC); y -= 0.022
            y -= 0.008
    pdf.savefig(fig); plt.close(fig)


def _wrap(t, n):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def page_cover(pdf):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    # cone sketch
    ax.add_patch(plt.Polygon([[0.5, 0.74], [0.18, 0.40], [0.82, 0.40]], closed=True,
                             fill=False, ec=ACC, lw=2))
    for i, yy in enumerate(np.linspace(0.43, 0.71, 6)):
        w = (yy - 0.40) / (0.74 - 0.40) * 0.32
        ax.plot([0.5 - w, 0.5 + w], [yy, yy], color=GOOD if i in (2, 3) else MUT,
                lw=2.4 if i in (2, 3) else 1.0, alpha=0.9 if i in (2, 3) else 0.5)
    ax.annotate("habitable band\n(Goldilocks)", (0.5, 0.57), (0.86, 0.60), color=GOOD,
                fontsize=9, ha="left", arrowprops=dict(arrowstyle="->", color=GOOD))
    ax.text(0.5, 0.385, "tip = the single surviving behaviour (PO -> 0)", ha="center", fontsize=8, color=MUT)
    ax.text(0.5, 0.755, "all organisms that could exist", ha="center", fontsize=8, color=MUT)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.text(0.5, 0.90, "EEC — Constructing Organisms by Constraint", ha="center",
             fontsize=20, weight="bold", color=INK)
    fig.text(0.5, 0.86, "Findings report: laws of existence, the degradation budget,\n"
                        "and PO as the native complexity metric", ha="center", fontsize=11, color=MUT)
    fig.text(0.5, 0.30, "Gradient-free neuroevolution · recurrent organism · read the STATE, not the output",
             ha="center", fontsize=9.5, color=ACC)
    fig.text(0.5, 0.06, "GENREG / EEC  ·  2026-06-20", ha="center", fontsize=9, color=MUT)
    pdf.savefig(fig); plt.close(fig)


def page_formula(pdf):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, "The PO / Fitness coordinate", fontsize=18, weight="bold", color=ACC)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=ACC, lw=2))
    blocks = (
        "PO (Practical Optimization) is the native complexity unit of an evolutionary model: "
        "not parameters, but CONSTRAINTS — how many laws of existence are required to collapse the "
        "infinite cone of possible organisms down to the behaviour we need. PO counts AXES covered, "
        "not constraints stacked (two constraints on one axis are redundant).\n\n"
        "Fitness is the organism's own view: how well it survives the laws imposed. PO measures top-down "
        "(how much of infinity eliminated); fitness measures bottom-up (how well it lives under what remains). "
        "They are inverse twins of the same convergence, approaching the same unreachable tip from opposite sides.")
    y = 0.885
    for line in _wrap(blocks.replace("\n\n", " \n "), 98):
        if line == "":
            y -= 0.012; continue
        fig.text(0.09, y, line, fontsize=10); y -= 0.022
    # formula box
    y -= 0.01
    box = FancyBboxPatch((0.12, y - 0.115), 0.76, 0.115, boxstyle="round,pad=0.01",
                         fc="#f2f6fb", ec=ACC, lw=1.2, transform=fig.transFigure)
    fig.add_artist(box)
    for i, ln in enumerate([
        "organism identity  =  ( PO ,  fitness )",
        "PO  -> 0      as more axes are covered   (space collapses to the survivor)",
        "fitness -> 100  as every imposed law is solved",
        "( PO = 0 )  <=>  ( fitness = 100 )   — asymptotes, never reached"]):
        fig.text(0.15, y - 0.022 - i * 0.024, ln, fontsize=10.5,
                 family="monospace", color=ACC if i else INK, weight="bold" if i == 0 else "normal")
    y -= 0.16
    # two charts: conceptual inverse twins + empirical competence vs axes
    axn = np.arange(0, 7)
    ax1 = fig.add_axes([0.11, 0.10, 0.36, 0.30])
    po = np.exp(-0.6 * axn); fit = 100 * (1 - np.exp(-0.6 * axn))
    ax1.plot(axn, po * 100, "o-", color=ACC2, label="PO (remaining space)")
    ax1.plot(axn, fit, "s-", color=GOOD, label="fitness")
    ax1.axhline(0, color=ACC2, ls=":", lw=0.8); ax1.axhline(100, color=GOOD, ls=":", lw=0.8)
    ax1.set_xlabel("axes covered"); ax1.set_ylabel("PO (%) / fitness")
    ax1.set_title("inverse twins -> same tip", fontsize=10); ax1.legend(fontsize=7.5)
    # empirical: mean competence by axis count from the PO lattice
    by_n = {}
    for _, n, comp, *_ in PO_LAT:
        by_n.setdefault(n, []).append(comp)
    ns = sorted(by_n); comps = [np.mean(by_n[n]) for n in ns]
    ax2 = fig.add_axes([0.57, 0.10, 0.36, 0.30])
    ax2.plot(ns, comps, "D-", color=GOOD)
    ax2.set_xlabel("axes covered (PO lattice)"); ax2.set_ylabel("competence (fitness proxy)")
    ax2.set_title("empirical: covering axes raises fitness", fontsize=10)
    ax2.set_xticks(ns)
    fig.text(0.5, 0.045, "Left: the law, idealized. Right: measured — mean competence rises as more "
             "(needed) axes are covered, until conflicting axes stall it.", ha="center", fontsize=8, color=MUT)
    pdf.savefig(fig); plt.close(fig)


def page_catalog(pdf):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, "Constraint catalog (laws of existence)", fontsize=18, weight="bold", color=ACC)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.925, 0.925], color=ACC, lw=2))
    ax = fig.add_axes([0.06, 0.30, 0.88, 0.58]); ax.axis("off")
    rows = [["Law", "Axis", "Verdict", "Key result"]] + [[a, b, c, d] for a, b, c, d in CATALOG]
    tbl = ax.table(cellText=rows, loc="center", cellLoc="left",
                   colWidths=[0.20, 0.17, 0.12, 0.51])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.6); tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd")
        if r == 0:
            cell.set_facecolor(ACC); cell.set_text_props(color="white", weight="bold")
        else:
            v = CATALOG[r - 1][2]
            if c == 2:
                cell.set_facecolor(VC.get(v, "#eee")); cell.set_text_props(color="white", weight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f6f8fa")
    fig.text(0.08, 0.25, "Verdicts:", fontsize=10, weight="bold")
    leg = ("REAL = real & productive   GATED = fires only where the world pays   WEAK = present but small\n"
           "WALL = search-walled (needs a channel, not pressure)   SWAP = valid alternative on the survival axis\n"
           "* OCCLUSION shows a Goldilocks band (peaks then collapses).")
    for i, ln in enumerate(leg.split("\n")):
        fig.text(0.09, 0.225 - i * 0.022, ln, fontsize=8.6, color=MUT)
    fig.text(0.08, 0.13, "Interactions:", fontsize=10, weight="bold")
    inter = ("occ x entropy  (same axis)      -> INTERFERE  (combined 0.21 < either 0.37 / 0.54)\n"
             "occ x repro-cost (diff axes)    -> COMPOUND   (combined 0.48 > both)\n"
             "parsimony x memory (conflict)   -> SUBTRACT   (gain 0.36 -> 0.13 when stacked)")
    for i, ln in enumerate(inter.split("\n")):
        fig.text(0.09, 0.105 - i * 0.024, ln, fontsize=9, family="monospace", color=INK)
    pdf.savefig(fig); plt.close(fig)


def page_observability(pdf):
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 11)); fig.subplots_adjust(top=0.88, hspace=0.32, wspace=0.3)
    fig.suptitle("Observation axis: occlusion vs noise (memory machinery)", fontsize=15,
                 weight="bold", color=ACC, y=0.95)
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]; sigs = [0.0, 0.25, 0.5, 1.0, 2.0]
    for col, (d, xs, xl, ttl) in enumerate([(occ, rhos, "occlusion rho", "OCCLUSION (missing)"),
                                            (noi, sigs, "noise sigma", "NOISE (corrupted)")]):
        for row, met in enumerate(["gain", "horizon"]):
            a = ax[row, col]
            for wn, col2 in [("text", MUT), ("longrange", ACC)]:
                a.plot(xs, [d[f"{wn}|{x}"][met] for x in xs], "o-", color=col2, label=wn, lw=2)
            a.set_xlabel(xl); a.set_ylabel("recurrent gain" if met == "gain" else "memory horizon")
            a.set_title(f"{ttl} -> {met}", fontsize=10); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.text(0.5, 0.06, "Occlusion drives memory in long-range (Goldilocks peak ~0.4); noise is far weaker; "
             "text (thin signal) is inert. 'Missing' forces memory; 'corrupted' only nudges it.",
             ha="center", fontsize=8.5, color=MUT)
    pdf.savefig(fig); plt.close(fig)


def page_budget(pdf):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.suptitle("Degradation budget: interference & the cone interior", fontsize=15,
                 weight="bold", color=ACC, y=0.95)
    # occ x entropy bars
    ax1 = fig.add_axes([0.10, 0.55, 0.80, 0.32])
    conds = ["none", "occ", "entropy", "occ+entropy"]; cols = ["#bbb", ACC, ACC2, "#7b4fa3"]
    for i, c in enumerate(conds):
        vals = [stk[f"{w}|{c}"]["gain"] for w in ["text", "longrange"]]
        ax1.bar(np.arange(2) + (i - 1.5) * 0.2, vals, 0.2, label=c, color=cols[i])
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["text", "long-range"])
    ax1.set_ylabel("recurrent gain"); ax1.legend(fontsize=8, ncol=4, loc="upper left")
    ax1.set_title("Same-axis pair INTERFERES: occ+entropy (0.21) < either alone (0.37 / 0.54)", fontsize=10)
    # cone grid heatmap
    ax2 = fig.add_axes([0.13, 0.10, 0.62, 0.34])
    G = np.array(grid["gain"]); rs = grid["rhos"]; ent = [round(1 - d, 2) for d in grid["decays"]]
    im = ax2.contourf(rs, ent, G.T, levels=14, cmap="viridis")
    fi, fj = np.unravel_index(np.argmax(G), G.shape)
    ax2.plot(rs[fi], ent[fj], "r*", ms=18)
    ax2.set_xlabel("occlusion rho"); ax2.set_ylabel("entropy (1 - decay)")
    ax2.set_title("cone interior: gain over occ x entropy", fontsize=10)
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.text(0.5, 0.052, "No synergistic interior ridge: the optimum is single-axis (best edge > best interior).",
             ha="center", fontsize=8.3, color=MUT)
    fig.text(0.5, 0.034, "Co-maxing both -> collapse valley (top-right). Constraints share a finite coping budget.",
             ha="center", fontsize=8.3, color=MUT)
    pdf.savefig(fig); plt.close(fig)


def page_axes(pdf):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.suptitle("Axes are swappable; some compound, some conflict", fontsize=15,
                 weight="bold", color=ACC, y=0.95)
    # survival swap
    ax1 = fig.add_axes([0.10, 0.58, 0.37, 0.30])
    laws = ["energy", "mortality", "both"]
    ax1.bar(np.arange(3) - 0.2, [SWAP[l][0] for l in laws], 0.4, label="competence", color=GOOD)
    ax1.bar(np.arange(3) + 0.2, [SWAP[l][1] for l in laws], 0.4, label="gain", color=ACC)
    ax1.axhline(0.167, color=ACC2, ls="--", lw=1); ax1.text(0, 0.18, "random", fontsize=7, color=ACC2)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(laws, fontsize=8.5)
    ax1.set_title("SURVIVAL swap (energy <-> mortality)", fontsize=9.5); ax1.legend(fontsize=7.5)
    # orthogonality
    ax2 = fig.add_axes([0.57, 0.58, 0.37, 0.30])
    keys = ["base", "repro", "occ", "both"]
    ax2.bar(np.arange(4) - 0.2, [ORTHO[k][0] for k in keys], 0.4, label="gain", color=ACC)
    ax2.bar(np.arange(4) + 0.2, [ORTHO[k][1] for k in keys], 0.4, label="fert. Gini", color="#7b4fa3")
    ax2.set_xticks(range(4)); ax2.set_xticklabels(["base", "repro", "occ", "both"], fontsize=8)
    ax2.set_title("ORTHOGONAL stack (occ x repro)", fontsize=9.5); ax2.legend(fontsize=7.5)
    # PO lattice heatmap
    ax3 = fig.add_axes([0.12, 0.10, 0.80, 0.34])
    mets = ["comp", "gain", "gini", "M"]
    M = np.array([[r[2], r[3], r[4], r[5]] for r in PO_LAT])
    Mn = (M - M.min(0)) / (np.ptp(M, axis=0) + 1e-9)
    im = ax3.imshow(Mn.T, aspect="auto", cmap="magma")
    ax3.set_yticks(range(4)); ax3.set_yticklabels(["competence", "gain (memory)", "Gini (ecosystem)", "M (size)"])
    ax3.set_xticks(range(len(PO_LAT))); ax3.set_xticklabels([r[0] for r in PO_LAT], rotation=35, ha="right", fontsize=7.5)
    for i in range(len(PO_LAT)):
        for j in range(4):
            ax3.text(i, j, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if Mn[i, j] < 0.5 else "black")
    ax3.set_title("PO axis lattice: trait vector per axis subset (normalized colour)", fontsize=10)
    fig.text(0.5, 0.05, "Ecosystem (Gini): owned solely by selection.  Memory: redundantly covered (obs OR sel).",
             ha="center", fontsize=8.3, color=MUT)
    fig.text(0.5, 0.032, "Parsimony CONFLICTS with memory (+obs+sel+parsi drops gain 0.36 -> 0.13).",
             ha="center", fontsize=8.3, color=MUT)
    pdf.savefig(fig); plt.close(fig)


def page_more(pdf):
    fig, ax = plt.subplots(2, 2, figsize=(8.5, 11)); fig.subplots_adjust(top=0.88, hspace=0.42, wspace=0.32)
    fig.suptitle("Further laws: reproduction, adaptability, perception, search-wall",
                 fontsize=14, weight="bold", color=ACC, y=0.95)
    # reproduction-cost
    a = ax[0, 0]
    a.bar([0, 1], [152.5, 214.8], color=["#bbb", GOOD]); a.set_xticks([0, 1])
    a.set_xticklabels(["lifespan", "repro-cost"]); a.set_title("Reproduction-cost: surplus +41%\n(fertility Gini 0->0.11)", fontsize=9)
    a.set_ylabel("median surplus")
    # non-stationarity
    a = ax[0, 1]
    a.bar([0, 1], [1.3, 2.0], color=["#bbb", "#d95f02"]); a.set_xticks([0, 1])
    a.set_xticklabels(["static", "shifting"]); a.set_title("Non-stationarity: diversity up\n(recovers 84% after 19% drop)", fontsize=9)
    a.set_ylabel("standing diversity")
    # perception cost
    a = ax[1, 0]
    a.bar([0, 1], [0.342, 0.068], color=["#bbb", GOOD]); a.set_xticks([0, 1])
    a.set_xticklabels(["free", "look-costs"]); a.set_title("Perception-cost: 5x less looking,\nsame survival (economy, not attention)", fontsize=9)
    a.set_ylabel("mean attention")
    # search-wall channel
    a = ax[1, 1]
    a.bar([0, 1], [0.0, 0.025], color=[ACC2, GOOD]); a.set_xticks([0, 1])
    a.set_xticklabels(["emergent gate", "direct channel"])
    a.set_title("Search-wall: conditional policy needs\na channel (|reactive| 0.00 -> 0.025)", fontsize=9)
    a.set_ylabel("|conditional response|")
    fig.text(0.5, 0.06, "Each law fires only where the world makes its capability pay AND reachable; conditional "
             "policies need a directly-searchable channel, not more pressure.", ha="center", fontsize=8.5, color=MUT)
    pdf.savefig(fig); plt.close(fig)


def page_synthesis(pdf):
    blocks = [
        ("h", "Meta-principles (the laws about laws)"),
        ("b", "P1  Reachability: a capability emerges only where the world makes it pay AND the organism "
              "can reach the alternative by mutation. Pressure shapes how well a reachable solution is found; "
              "it does not make unreachable solutions reachable."),
        ("b", "P2  Degradation budget / wall thickness: same-axis constraints share a finite coping budget. "
              "An outer boundary (too little pressure) and an inner boundary (too much -> unlearnable) bound a "
              "habitable Goldilocks band. The cone has thickness; the organism lives in the band, not at the tip."),
        ("b", "P3  Two ropes: same capability label, different internal mechanics (entropy drives gain; occlusion "
              "drives horizon). Read multiple internal facets, never one metric."),
        ("b", "P4  Orthogonal axes stack cleanly (even synergize); same-axis pairs interfere; cost-vs-capability "
              "pairs can outright conflict (parsimony subtracts memory)."),
        ("b", "P5  Search-wall: conditional/policy capability is unreachable by mutation when a constant suffices "
              "- it needs a directly-searchable channel, not more pressure."),
        ("b", "P6  The reproduction operator is itself a law: turnover dynamics are load-bearing."),
        ("b", "P7  Axes, not constraints. Cover each needed axis once, with the instantiation the world favors, "
              "avoiding axes that conflict with desired traits. PO = axes covered. (PO, fitness) is the full coordinate."),
        ("h", "Method"),
        ("b", "Read the STATE, not the output - headline numbers were repeatedly selection-collapse artifacts. "
              "Beware survival-saturation (tighten energy so survival depends on the capability under test). "
              "Verify before celebrating: two independent lines agreeing is what makes a finding trustworthy."),
        ("h", "Open frontier"),
        ("b", "Not every axis has a symmetric swap (observation is occlusion-dominant across worlds tested). "
              "A genuine noise-favoring world would require a distributed-computation task (target = function of "
              "many noisy inputs, nothing worth holding) - deliberately not built, to avoid Goodhart-ing the test."),
    ]
    textpage(pdf, "Synthesis", blocks)


def main():
    out = os.path.join(HERE, "EEC_report.pdf")
    with PdfPages(out) as pdf:
        page_cover(pdf)
        page_formula(pdf)
        page_catalog(pdf)
        page_observability(pdf)
        page_budget(pdf)
        page_axes(pdf)
        page_more(pdf)
        page_synthesis(pdf)
    print("saved", out)


if __name__ == "__main__":
    main()
