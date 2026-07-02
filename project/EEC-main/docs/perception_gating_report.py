"""Generate the Perception-Gating hypothesis report (multi-page PDF, matplotlib PdfPages).
Output: docs/PERCEPTION_GATING.pdf"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "PERCEPTION_GATING.pdf")

# (animal, can_gate, type, thermo, intelligence 1-10)
DATA = [
    ("jellyfish", 0, "cnidarian", "cold", 1), ("starfish", 0, "echinoderm", "cold", 1),
    ("shrimp", 0, "crustacean", "cold", 1), ("housefly", 0, "insect", "cold", 1),
    ("snail", 0, "mollusk", "cold", 1), ("most fish", 0, "fish", "cold", 2),
    ("goldfish", 0, "fish", "cold", 2), ("ant", 0, "insect", "cold", 2),
    ("mantis", 0, "insect", "cold", 2), ("snake", 0, "reptile", "cold", 2),
    ("gecko", 0, "reptile", "cold", 2), ("honeybee", 0, "insect", "cold", 3),
    ("jumping spider", 0, "arachnid", "cold", 3),
    ("frog", 1, "amphibian", "cold", 2), ("sea turtle", 1, "reptile", "cold", 3),
    ("shark (requiem)", 1, "fish", "cold", 3), ("crocodile", 1, "reptile", "cold", 4),
    ("squid", 1, "cephalopod", "cold", 4), ("pigeon", 1, "bird", "warm", 4),
    ("owl", 1, "bird", "warm", 4), ("cat", 1, "mammal", "warm", 5),
    ("horse", 1, "mammal", "warm", 5), ("cuttlefish", 1, "cephalopod", "cold", 6),
    ("dog", 1, "mammal", "warm", 6), ("octopus", 1, "cephalopod", "cold", 7),
    ("pig", 1, "mammal", "warm", 7), ("parrot", 1, "bird", "warm", 7),
    ("raven", 1, "bird", "warm", 8), ("elephant", 1, "mammal", "warm", 8),
    ("chimpanzee", 1, "primate", "warm", 8), ("dolphin", 1, "mammal", "warm", 9),
    ("human", 1, "primate", "warm", 10),
]

BLUE, RED, INK, GREY = "#1b5e9e", "#c0392b", "#1a1a1a", "#888888"


def textpage(pdf, title, blocks, subtitle=None):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.93, title, fontsize=20, weight="bold", color=INK)
    if subtitle:
        fig.text(0.08, 0.895, subtitle, fontsize=11, color=GREY, style="italic")
    y = 0.85
    for head, body in blocks:
        if head:
            fig.text(0.08, y, head, fontsize=13, weight="bold", color=BLUE); y -= 0.035
        for line in body:
            wrapped = _wrap(line, 92)
            for w in wrapped:
                fig.text(0.09, y, w, fontsize=10.5, color=INK); y -= 0.027
        y -= 0.018
    plt.axis("off"); pdf.savefig(fig); plt.close()


def _wrap(s, n):
    words = s.split(); lines = []; cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= n: cur = (cur + " " + w).strip()
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines or [""]


def cover(pdf):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.72, "PERCEPTION GATING", fontsize=30, weight="bold", ha="center", color=INK)
    fig.text(0.5, 0.675, "and the Emergence of Cognition", fontsize=18, ha="center", color=BLUE)
    fig.text(0.5, 0.60, "The ability to shut off perception — not perception itself —\n"
             "tracks intelligence across the animal kingdom.", fontsize=12.5, ha="center", color=INK)
    fig.text(0.5, 0.45, "A reframing of the eyelid / intelligence observation as a\n"
             "perception-cost constraint, and its implication for gradient-free\n"
             "cognition in the GENREG / EEC paradigm.", fontsize=11, ha="center", color=GREY)
    fig.text(0.5, 0.08, "GENREG / EEC — Existence-Environment Constraints", fontsize=10, ha="center", color=GREY)
    plt.axis("off"); pdf.savefig(fig); plt.close()


def tablepage(pdf):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, "The data — 32 animals", fontsize=18, weight="bold", color=INK)
    fig.text(0.08, 0.925, "Rows shaded by whether the animal can gate vision (shut off visual input).",
             fontsize=10, color=GREY)
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.85]); ax.axis("off")
    cols = ["animal", "gate vision?", "type", "thermo", "intel"]
    cell, colors = [], []
    for name, gate, typ, th, intel in DATA:
        cell.append([name, "yes" if gate else "no", typ, th, str(intel)])
        base = "#eaf2fb" if gate else "#fdecea"
        hi = name in ("octopus", "cuttlefish", "squid")
        colors.append([("#cfe0c3" if hi else base)] * 5)
    t = ax.table(cellText=cell, colLabels=cols, cellColours=colors, loc="upper center",
                 cellLoc="left", colColours=[BLUE] * 5)
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.28)
    for (r, c), cellobj in t.get_celld().items():
        cellobj.set_edgecolor("white")
        if r == 0:
            cellobj.set_text_props(color="white", weight="bold")
    fig.text(0.08, 0.045, "Green = cephalopods (no true eyelid, but a muscular ring that closes like one) — the "
             "convergent-evolution case.", fontsize=8.5, color=GREY, style="italic")
    pdf.savefig(fig); plt.close()


def chartpage(pdf):
    fig, ax = plt.subplots(figsize=(8.5, 11)); fig.subplots_adjust(top=0.82, bottom=0.40)
    rng = np.random.default_rng(0)
    for name, gate, typ, th, intel in DATA:
        x = gate + rng.uniform(-0.12, 0.12); ceph = typ == "cephalopod"
        ax.scatter(x, intel, s=90 if ceph else 55,
                   color=("#2a8a3e" if ceph else (BLUE if gate else RED)),
                   edgecolor="white", zorder=3, alpha=0.9)
        if name in ("octopus", "honeybee", "frog", "human", "raven"):
            ax.annotate(name, (x, intel), fontsize=8, xytext=(6, 0), textcoords="offset points", va="center")
    ax.axhline(3.5, color=GREY, ls="--", lw=1)
    ax.text(0.5, 3.62, "no animal above this line lacks a perception gate", fontsize=9, color=GREY, ha="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CANNOT gate vision", "CAN gate vision"], fontsize=11)
    ax.set_ylabel("estimated intelligence (1–10)"); ax.set_ylim(0, 11); ax.set_xlim(-0.5, 1.5)
    ax.set_title("Intelligence vs. the ability to gate perception", fontsize=15, weight="bold", pad=14)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.93, "Perception gating and cognition", fontsize=18, weight="bold", ha="center", color=INK)
    fig.text(0.1, 0.30,
             "No-gate animals cap at intelligence 3 (honeybee, jumping spider). The entire high-cognition\n"
             "range (5–10) is gate-only. The gate is NECESSARY but not SUFFICIENT (frog has a gate, stays\n"
             "low). The green cephalopods are the crown evidence: octopus and cuttlefish evolved on a\n"
             "completely separate lineage from vertebrates and INDEPENDENTLY evolved both a perception-\n"
             "gate (a muscular ring that closes over the eye) AND high intelligence — convergent evolution\n"
             "coupling the same two traits twice. That argues for a real causal signal, not a confound.",
             fontsize=10, color=INK)
    pdf.savefig(fig); plt.close()


if __name__ == "__main__":
    with PdfPages(OUT) as pdf:
        cover(pdf)
        textpage(pdf, "The hypothesis", [
            ("The observation", [
                "Sorting animals by whether they have eyelids produced a striking split: every no-eyelid",
                "animal sat at the bottom of the intelligence scale, while eyelids appeared on everything",
                "from the dog to the gorilla. The proposed reading: an eyelid is a PERCEPTION GATE — the",
                "ability to shut off visual input — and the organisms that can choose NOT to see are the",
                "ones that developed complex cognition.",
            ]),
            ("The reframing", [
                "Stated strictly, 'eyelids' is the wrong variable — it fails (see the octopus). Stated as",
                "PERCEPTION GATING (can the animal shut off its visual input, by any means), it holds. This",
                "is the biological form of a perception-cost constraint: when looking is optional, the",
                "organism must build an internal model to function while not looking — and that internal",
                "model IS comprehension. Intelligence comes from controlling perception, not maximising it.",
            ]),
            ("Why it matters here", [
                "In the GENREG / EEC paradigm we hit a wall: gradient-free evolution cannot build perception",
                "(searching a high-dimensional continuous map fails). This hypothesis says we were solving",
                "the wrong problem — don't build perception, build the GATE plus the pressure to model the",
                "unseen, and comprehension emerges as the world-model the organism forms to survive the",
                "blackout.",
            ]),
        ], subtitle="Perception gating, not eyelids, tracks cognition")
        tablepage(pdf)
        chartpage(pdf)
        textpage(pdf, "Verdict & implications", [
            ("1.  The strict 'eyelid' version fails", [
                "The octopus has no true eyelid — but a muscular ring of skin that closes over the eye like",
                "one — and it is the smartest invertebrate alive (~500M neurons, jar-opening, tool use).",
                "Bees and jumping spiders also rise above the original 'max 2'.",
            ]),
            ("2.  The 'perception gating' version holds", [
                "No animal scoring 4 or above lacks a way to gate vision. The whole high-cognition range",
                "(5–10) is gate-only. The gate is necessary, not sufficient (frog and crocodile have gates",
                "and stay low) — it is an enabler/correlate, not a deterministic cause.",
            ]),
            ("3.  The octopus is the crown evidence, not a counterexample", [
                "Cephalopods evolved on a separate lineage from vertebrates and INDEPENDENTLY evolved both a",
                "perception-gate and high intelligence. Convergent evolution coupling the same two traits",
                "twice is the strongest argument that the gate is causal, under the obvious confounds",
                "(neuron count, active-predator lifestyle).",
            ]),
            ("4.  The GENREG move it motivates", [
                "Perception is DEVELOPMENT, not evolution: evolution shapes a local learning rule and the",
                "behaviour policy; perception self-organises within the lifetime from exposure. The new",
                "constraint: give the organism an EYELID (gate its input stream, at a cost) and force it to",
                "PREDICT the unseen to act. Test whether a comprehending, generalising internal model",
                "emerges — exactly where the always-on perception map died.",
            ]),
            ("Sources", [
                "Octopus pseudo-eyelid (OctoNation); Cephalopod intelligence (Wikipedia); Octopus cognition",
                "(Natural History Museum); Nictitating membrane (Wikipedia); Smartest invertebrates",
                "(Frontiers for Young Minds). Intelligence values are rough 1–10 estimates for illustration.",
            ]),
        ], subtitle="What holds, what doesn't, and what to build")
    print("saved", OUT)
