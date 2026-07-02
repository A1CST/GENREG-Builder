"""Consolidated technical report on the language/cognition arc:
grounding -> real corpus -> syntax -> memory (invented) -> organizing space.
All numbers are from this session's direct experiments. Output: LANGUAGE_REPORT.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "LANGUAGE_REPORT.pdf")
INK = "#16181d"; ACC = "#1b5e9e"; ACC2 = "#c0392b"; GOOD = "#2e8b57"; MUT = "#8a8f98"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#555", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.grid": True, "grid.color": "#ececec"})


def _wrap(t, n):
    out, line = [], ""
    for w in t.split():
        if len(line) + len(w) + 1 > n: out.append(line); line = w
        else: line = (line + " " + w).strip()
    if line: out.append(line)
    return out


def textpage(pdf, title, blocks, sub=None):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.945, title, fontsize=17, weight="bold", color=ACC)
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.93, 0.93], color=ACC, lw=2))
    if sub: fig.text(0.08, 0.905, sub, fontsize=9.5, style="italic", color=MUT)
    y = 0.875 if not sub else 0.86
    for kind, txt in blocks:
        if kind == "h":
            y -= 0.012; fig.text(0.08, y, txt, fontsize=11.5, weight="bold", color=INK); y -= 0.026
        elif kind == "b":
            for ln in _wrap(txt, 98): fig.text(0.09, y, ln, fontsize=9.4); y -= 0.0185
            y -= 0.009
        elif kind == "f":
            for ln in txt.split("\n"): fig.text(0.10, y, ln, fontsize=9.6, family="monospace", color=ACC); y -= 0.020
            y -= 0.006
        elif kind == "w":
            for ln in _wrap(txt, 96): fig.text(0.095, y, ln, fontsize=9.2, color="#7a4a12"); y -= 0.0185
            y -= 0.009
    pdf.savefig(fig); plt.close(fig)


def embed(pdf, png, title, caption):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    fig.text(0.08, 0.93, title, fontsize=13, weight="bold", color=INK)
    try:
        img = plt.imread(os.path.join(HERE, png)); h, w = img.shape[:2]; aspect = h / w
        aw = 0.84; ah = aw * aspect * (8.5 / 11)
        ax = fig.add_axes([0.08, 0.52 - ah / 2, aw, ah]); ax.imshow(img); ax.axis("off")
    except Exception as e:
        fig.text(0.1, 0.5, f"[figure {png} not found: {e}]", fontsize=9, color=ACC2)
    y = 0.30
    for ln in _wrap(caption, 96): fig.text(0.09, y, ln, fontsize=9.4); y -= 0.019
    pdf.savefig(fig); plt.close(fig)


def fig_syntax(pdf):
    fig = plt.figure(figsize=(8.5, 5.2)); fig.patch.set_facecolor("white")
    ax1 = fig.add_axes([0.09, 0.16, 0.38, 0.66]); ax2 = fig.add_axes([0.58, 0.16, 0.38, 0.66])
    # scramble test (order is load-bearing)
    ax1.bar([0, 1], [0.29, 0.07], color=[GOOD, ACC2], width=0.6)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["message\nintact", "order\nscrambled"])
    ax1.set_ylabel("full-event accuracy"); ax1.set_title("Word order is load-bearing", fontsize=10)
    ax1.axhline(0.017, color=MUT, ls="--", lw=1); ax1.text(1, 0.03, "chance", fontsize=7, color=MUT)
    for x, v in [(0, 0.29), (1, 0.07)]: ax1.text(x, v + 0.008, f"{v:.2f}", ha="center", fontsize=9)
    # the world requires order: sequential vs bag (binding world)
    ax2.bar([0, 1], [0.28, 0.19], color=[ACC, MUT], width=0.6)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["sequential\nchannel", "order-free\nbag"])
    ax2.set_ylabel("attribute-binding accuracy"); ax2.set_title("The world requires structure", fontsize=10)
    for x, v in [(0, 0.28), (1, 0.19)]: ax2.text(x, v + 0.006, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle("Figure 3.  Syntax emerges from a relational world (nothing wired in)",
                 fontsize=12, weight="bold", y=0.97)
    pdf.savefig(fig); plt.close(fig)


def fig_memory(pdf):
    fig = plt.figure(figsize=(8.5, 5.2)); fig.patch.set_facecolor("white")
    ax1 = fig.add_axes([0.07, 0.16, 0.26, 0.66]); ax2 = fig.add_axes([0.40, 0.16, 0.26, 0.66])
    ax3 = fig.add_axes([0.73, 0.16, 0.24, 0.66])
    ax1.bar([0, 1], [0.99, 0.16], color=[GOOD, ACC2], width=0.6)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["world\nslot", "slot\ndisabled"])
    ax1.set_ylabel("recall"); ax1.set_title("Invented memory\n(uses the world)", fontsize=9.5)
    ax2.bar([0, 1, 2], [0.42, 0.96, 0.96], color=[MUT, GOOD, GOOD], width=0.6)
    ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(["W=1", "W=3", "W=5"])
    ax2.set_ylabel("sequence reproduced"); ax2.set_title("Organizing space\n(tape width)", fontsize=9.5)
    ax3.bar([0, 1], [0.95, 0.38], color=[GOOD, ACC2], width=0.6)
    ax3.set_xticks([0, 1]); ax3.set_xticklabels(["movable\nhead", "head\nfrozen"])
    ax3.set_ylabel("reproduced"); ax3.set_title("Space is\nload-bearing", fontsize=9.5)
    fig.suptitle("Figure 4.  Memory and spatial organization -- invented by a feedforward organism",
                 fontsize=12, weight="bold", y=0.97)
    pdf.savefig(fig); plt.close(fig)


with PdfPages(OUT) as pdf:
    # cover
    fig = plt.figure(figsize=(8.5, 11)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    rungs = ["single concepts", "grounded English", "real-corpus structure", "word order (syntax)",
             "attribute binding", "invented memory", "organized space"]
    for i, r in enumerate(rungs):
        yy = 0.30 + i * 0.055
        ax.plot([0.30, 0.40], [yy, yy], color=ACC, lw=2)
        ax.text(0.43, yy, r, fontsize=10, va="center", color=INK if i else MUT)
    ax.annotate("", xy=(0.35, 0.30 + 6 * 0.055 + 0.03), xytext=(0.35, 0.28),
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.text(0.5, 0.86, "GENREG / EEC", ha="center", fontsize=15, color=ACC, weight="bold")
    fig.text(0.5, 0.81, "Language and Cognition by Constraint", ha="center", fontsize=22, weight="bold", color=INK)
    fig.text(0.5, 0.775, "How grounding, syntax, memory, and spatial organization emerge\n"
                         "from worlds that demand them -- never from wiring them in",
             ha="center", fontsize=11, color=MUT)
    fig.text(0.5, 0.20, "Gradient-free evolution  ·  change the world, not the score  ·  read the state, not the output",
             ha="center", fontsize=9.5, color=ACC, style="italic")
    fig.text(0.5, 0.07, "GENREG / EEC  ·  2026-06-21", ha="center", fontsize=9, color=MUT)
    pdf.savefig(fig); plt.close(fig)

    textpage(pdf, "1.  Executive summary", [
        ("h", "Thesis"),
        ("b", "A single principle runs through every result here: a cognitive capability appears only "
              "when the WORLD makes it the cheapest way to survive -- and it must be discovered, not "
              "installed. Repeatedly, the instinct to hand the organism the answer (a grammar channel, "
              "a memory register, better embeddings) was the wrong move; the right move was to build a "
              "world with the relevant structure and let evolution invent the mechanism."),
        ("h", "The ladder (each rung is a separate experiment)"),
        ("b", "A. Grounding makes a code ENGLISH: pressure alone yields an arbitrary code; a frozen "
              "English prior makes it English (usage 0.03 -> 0.95+)."),
        ("b", "B-E. Sexual reproduction stabilises a shared language under noise; English can be ACQUIRED "
              "from exposure (a pidgin, dose-responsive); grounded vocabularies are COMPOSITIONAL "
              "(zero-shot on held-out combinations); real-corpus grounding reproduces real semantic "
              "confusion structure and frequency effects."),
        ("b", "F-G. SYNTAX (word order) emerges from a world of RELATIONS where 'A acts B' != 'B acts A'; "
              "richer worlds (composite entities) drive constituent BINDING."),
        ("b", "H-I. MEMORY: rather than being given a recurrent register, a strictly feedforward organism "
              "INVENTS external memory -- it discovers it can write to a persistent world and read back, "
              "including its own encoding."),
        ("b", "J. It then learns to ORGANIZE that memory in SPACE -- laying a sequence across a tape and "
              "navigating to reproduce it."),
        ("h", "The recurring lever (P1 / reachability)"),
        ("b", "Every emergence required the world's reward to make the FIRST steps survivable -- graded, "
              "not all-or-nothing. The same fix (partial credit for partial success) unlocked syntax, "
              "memory, and spatial organization; its absence produced clean reachability walls. This is "
              "the single most load-bearing design choice in the whole arc."),
        ("h", "Honesty"),
        ("b", "Effects are reported at the strength the data supports. Absolute accuracies are often modest "
              "(this is a minimal gradient-free substrate); the claims are about MECHANISM and CONTRAST "
              "(structure on vs off, world-demands-it vs not), which are clean and reproducible."),
    ])

    textpage(pdf, "2.  Methods", [
        ("h", "Substrate"),
        ("b", "Populations of simple organisms evolve gradient-free (mutation + crossover; no backprop). "
              "Tasks are referential/communication or memory games. Genomes are small linear maps; for the "
              "memory work the organism is strictly FEEDFORWARD with no internal state."),
        ("h", "Grounding"),
        ("b", "An English/semantic prior enters as a FROZEN channel added to an evolved residual (a skip "
              "connection): the structure is preserved while evolution adapts use. Real-corpus grounding "
              "uses a 60-word vocabulary, frequencies, and PPMI-SVD embeddings computed from ~2M tokens of "
              "real prose (engine/corpus.txt)."),
        ("h", "Worlds that demand structure"),
        ("b", "Relations: events (agent, action, target) over a shared entity pool, sequential channel -> "
              "order can mark roles. Composite entities (attribute+type) -> binding. Memory: stimulus "
              "spread over time with a persistent world the organism can write/read. Space: a tape of "
              "cells with a movable head."),
        ("h", "Decisive controls"),
        ("b", "Scramble (destroy message order), bag (destroy positions), ablation (sever the world slot / "
              "freeze the head), and capacity sweeps. These separate 'genuinely using structure' from "
              "'partial-credit artefact'."),
        ("w", "Caveat: a minimal substrate. Absolute numbers are not the point; on/off contrasts are. Most "
              "single conditions are a few seeds; effects flagged 'modest' are directional."),
    ])

    embed(pdf, "english_findings.png", "Figure 1.  Grounding, stability, and acquisition of English",
          "A: a frozen English prior turns an otherwise-arbitrary code into English (usage rises from "
          "chance to ~1.0 with prior strength). B: under channel noise, sexual reproduction protects the "
          "shared English code (~0.75) where cloning lets it erode (~0.35). C: with no innate prior, "
          "English is ACQUIRED from exposure to native speakers -- a dose-response that saturates around "
          "half the vocabulary (a stable pidgin), because resident-to-resident talk never requires English.")

    embed(pdf, "corpus_findings.png", "Figure 2.  Real-corpus grounding carries real English structure",
          "A: when the organism errs, the confused word is far more SEMANTICALLY similar to the target than "
          "chance (+0.24 vs +0.11) -- e.g. right->left, time->same. The system confuses RELATED words, the "
          "same failure signature real language models show. B: a frequency (Zipf) effect -- more frequent "
          "words are communicated more reliably. Grounding here is real word statistics, not a toy lexicon.")

    fig_syntax(pdf)
    textpage(pdf, "3.  Syntax from a relational world", [
        ("b", "Single-word communication suffices for single concepts, so no syntax ever pays for itself. "
              "Word order becomes load-bearing only when the world has RELATIONS: events (agent, action, "
              "target) where agent and target are interchangeable, so 'A chases B' and 'B chases A' are the "
              "same symbols with opposite meaning. Survival depends on conveying who-did-what-to-whom."),
        ("h", "Result (Figure 3)"),
        ("b", "Order-based role-marking emerges with nothing wired in (full-event accuracy ~0.29 vs 0.017 "
              "chance). It is genuinely USING order: scrambling the message collapses accuracy to 0.07. And "
              "the world REQUIRES order -- an order-free 'bag' listener cannot tell 'A acts B' from 'B acts "
              "A' (binding 0.19 vs the sequential channel's 0.28). Richer worlds (composite entities) push "
              "this to constituent binding."),
        ("w", "Honest limit: absolute accuracy is modest and entities are still confused; generalisation to "
              "held-out events is partial. The MECHANISM (order emerges, is load-bearing, is required) is "
              "clean; high-accuracy grammar is not. The bottleneck became the organism: a flat genome has "
              "nowhere to HOLD a constituent -- which motivated the memory work."),
    ])

    fig_memory(pdf)
    textpage(pdf, "4.  Memory and space -- invented, not installed", [
        ("h", "The distinction that matters"),
        ("b", "The easy move is to give the organism a recurrent register and a state-update rule; then it "
              "'has memory' and merely tunes it. The paradigm-pure question is whether the genome can BUILD "
              "a memory mechanism from primitives never designated as memory. So: a strictly FEEDFORWARD "
              "organism (no internal state), and the only persistence is in the WORLD."),
        ("h", "Invented memory (Figure 4, left)"),
        ("b", "Given one persistent world slot and a task needing the past, the genome invents writing: it "
              "writes a cue, preserves it, and reads it back -- and it invents its OWN ENCODING (storing cue "
              "0 as symbol 4, etc.). Ablation: 0.99 with the slot, 0.16 (chance) with it disabled. The "
              "organism is genuinely memoryless; the memory is external and self-built."),
        ("h", "Organizing space (Figure 4, middle/right)"),
        ("b", "Given a tape and a movable head, it learns to lay a sequence ACROSS cells and scan back to "
              "reproduce it: W=1 caps at 0.42 (one item), W>=3 reaches 0.96. Freezing the head collapses it "
              "to the one-cell ceiling (0.95 -> 0.38). It invents both the spatial layout and a per-cell "
              "cipher. This is the seed of a map -- structured knowledge held in the world."),
        ("w", "Honest path: the spatial result required a GRADED task (reproduce a sequence; partial credit "
              "for each item). The all-or-nothing index-query version hit a reachability wall and produced "
              "no organization. The world's reward structure -- not any wired layout -- was the lever."),
    ])

    textpage(pdf, "5.  The reachability thread, and open frontiers", [
        ("h", "P1, proven repeatedly"),
        ("b", "Across syntax, memory, and spatial organization, the same pattern held: the capability the "
              "world demanded was unreachable from random genomes UNLESS partial progress was rewarded. "
              "All-or-nothing survival gave flat chance; graded survival (more of the event conveyed, more "
              "items recalled, more of the sequence reproduced = more energy) let the search climb. This is "
              "ecologically honest -- conveying or remembering more genuinely helps -- and it is not wiring "
              "in the answer; it is shaping the world so the first rung is survivable."),
        ("h", "What is solid vs preliminary"),
        ("b", "Solid: grounding -> English; sex stabilises it; real-corpus confusion + frequency structure; "
              "order emerges and is load-bearing and required; memory is invented (ablation-proven); space "
              "is organized (ablation-proven)."),
        ("b", "Preliminary/modest: absolute syntax accuracy; compositional binding scale; memory is a "
              "low-capacity single grip (given-register case); generalisation is partial."),
        ("h", "Open frontiers"),
        ("b", "1. CONTENT-ADDRESSED retrieval -- find by WHAT, not where -- the step that turns a tape into "
              "a memory. 2. Scaling: longer sequences, larger vocabularies, deeper relations. 3. Fusing the "
              "recurrent/world memory into the syntax worlds so binding has somewhere to live. 4. A "
              "principled measure of when a world is 'reachable' for a given capability."),
        ("h", "Bottom line"),
        ("b", "From arbitrary signals to grounded English to word order to invented, spatially-organized "
              "memory -- each capability emerged from a world that made it the cheapest way to live, and "
              "each was the organism's own discovery. The frontier is everything that needs a richer world "
              "or a more reachable path, not a richer hand-built organism."),
    ])

print("saved", OUT)
