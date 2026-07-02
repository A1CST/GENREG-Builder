"""Audit & re-test report: what survived scrutiny, what was an artifact, what failed.
All numbers from this session's high-seed re-tests. -> AUDIT_REPORT.pdf
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "AUDIT_REPORT.pdf")
INK = "#16181d"; ACC = "#1b5e9e"; GOOD = "#2e8b57"; BAD = "#c0392b"; WARN = "#c9821f"; MUT = "#8a8f98"
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
    fig.text(0.07, 0.95, title, fontsize=16, weight="bold", color=ACC)
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.935, 0.935], color=ACC, lw=2))
    if sub: fig.text(0.07, 0.912, sub, fontsize=9.3, style="italic", color=MUT)
    y = 0.885 if not sub else 0.872
    for kind, txt in blocks:
        if kind == "h":
            y -= 0.010; fig.text(0.07, y, txt, fontsize=11, weight="bold", color=INK); y -= 0.024
        elif kind == "b":
            for ln in _wrap(txt, 100): fig.text(0.08, y, ln, fontsize=9.2, color=INK); y -= 0.0172
            y -= 0.008
        elif kind == "g":   # good/green
            for ln in _wrap(txt, 98): fig.text(0.085, y, ln, fontsize=9.2, color="#1d6b3f"); y -= 0.0172
            y -= 0.008
        elif kind == "r":   # red/failure
            for ln in _wrap(txt, 98): fig.text(0.085, y, ln, fontsize=9.2, color="#9c2d22"); y -= 0.0172
            y -= 0.008
        elif kind == "w":   # warn/amber
            for ln in _wrap(txt, 98): fig.text(0.085, y, ln, fontsize=9.2, color="#7a4a12"); y -= 0.0172
            y -= 0.008
        elif kind == "f":
            for ln in txt.split("\n"): fig.text(0.09, y, ln, fontsize=9.3, family="monospace", color=ACC); y -= 0.0185
            y -= 0.006
    pdf.savefig(fig); plt.close(fig)


def bars(ax, labels, vals, errs=None, colors=None, ylab="", title="", chance=None, rot=0):
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=colors or ACC, width=0.62, capsize=3, error_kw=dict(lw=1))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=rot)
    ax.set_ylabel(ylab, fontsize=9); ax.set_title(title, fontsize=9.5)
    if chance is not None:
        ax.axhline(chance, color=MUT, ls="--", lw=1); ax.text(len(labels) - 1, chance + 0.01, "chance", fontsize=6.5, color=MUT, ha="right")
    for xi, v in zip(x, vals): ax.text(xi, v + (max(vals) * 0.02), f"{v:.2f}", ha="center", fontsize=7.5)


with PdfPages(OUT) as pdf:
    # ---------- cover ----------
    fig = plt.figure(figsize=(8.5, 11)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    fig.text(0.5, 0.72, "GENREG / EEC", ha="center", fontsize=14, color=ACC, weight="bold")
    fig.text(0.5, 0.66, "Audit & Re-test of the", ha="center", fontsize=22, weight="bold", color=INK)
    fig.text(0.5, 0.61, "Cognition Findings", ha="center", fontsize=22, weight="bold", color=INK)
    fig.text(0.5, 0.55, "Hunting holes in the logic; re-testing every load-bearing claim\n"
                        "with high seed counts; separating what emerged from what was shaped.",
             ha="center", fontsize=11, color=MUT)
    fig.text(0.5, 0.40, "Central correction: \"no gradients ever\" bans designed reward-gradients,\n"
                        "not just gradient optimisers. Several headline results leaned on shaped rewards.",
             ha="center", fontsize=10, color=BAD)
    fig.text(0.5, 0.07, "GENREG / EEC  ·  2026-06-22  ·  fitness = world-consequence, never a designed slope",
             ha="center", fontsize=9, color=MUT)
    pdf.savefig(fig); plt.close(fig)

    # ---------- executive summary ----------
    textpage(pdf, "1.  Executive summary", [
        ("h", "What this audit did"),
        ("b", "Re-read every claim in the cognition arc (FINDINGS.md A-K), listed the holes (AUDIT.md), "
              "then re-tested the load-bearing ones with 5-8 seeds and proper controls. The trigger was "
              "the rule 'no gradients ever' -- which, taken fully, forbids DESIGNED reward-gradients "
              "(reward shaping), not only gradient optimisers. That reframes the whole 'graded survival' "
              "story."),
        ("h", "Headline corrections"),
        ("g", "HELD UP: external memory is invented and pays in real currency -- ablation 0.67 vs 0.16 "
              "(n=8), and in a pure-food foraging world 4.01 vs 1.93 on the same organism. Sexual "
              "reproduction robustly protects a shared language under noise (0.79 vs 0.36 at high noise, "
              "n=8). Word order is load-bearing (scramble collapses it, 0.22->0.055, n=8)."),
        ("w", "REPAIRED: 'sequential vastly beats bag' was a TRAINING ARTIFACT -- the bag was undertrained; "
              "at fair training the gap shrinks to a modest 0.37 vs 0.25. 'Memory never forgets' is "
              "refined: a one-step drop then a flat plateau to delay 24."),
        ("r", "FAILED / SHAPED: content-addressing is a hard wall (fails three ways). The binding and "
              "spatial-layout results lean on DESIGNED graded rewards and are downgraded to "
              "reward-dependent. Compositionality is largely tautological (per-word grounding). The "
              "foraging 'relocating' control was confounded by abundance. One probe used an explicit "
              "designed gradient and is disavowed."),
        ("h", "The refined thesis"),
        ("b", "Capabilities emerge from worlds whose SURVIVAL PHYSICS make them the cheapest way to live -- "
              "external memory most cleanly. Word order emerges too, but only when partial communication "
              "carries partial survival value (a realistic condition), and robustly across reward shapes. "
              "The lever is NOT 'add a gradient to the reward'; where we did that, the result is weaker."),
    ])

    # ---------- the reward-shaping reckoning ----------
    textpage(pdf, "2.  The reward-shaping reckoning  (the deepest hole)", [
        ("h", "The hole"),
        ("b", "The session's 'reachability lever' was: grade survival so partial progress pays. In three "
              "results that grading was something I DESIGNED, not a world-consequence: syntax (+1 per "
              "decoded role), binding (I switched the reward to per-entity until binding appeared), and "
              "spatial layout (per-position credit). Designing a slope toward the target is reward shaping "
              "-- a gradient in disguise -- even with a gradient-free optimiser."),
        ("h", "The test"),
        ("b", "Re-ran the relational (syntax) world under five fitness SHAPES, 5 seeds each, asking whether "
              "order still emerges and stays load-bearing (Figure 1, left):"),
        ("f", "all-or-nothing (whole message or death) : full 0.011  -> does NOT emerge (chance)\n"
              "any (>=1 role right)                     : full 0.062  -> does NOT emerge\n"
              "threshold2 (>=2 right, coarse)           : full 0.146  -> EMERGES, load-bearing\n"
              "per_role (linear count)                  : full 0.225  -> EMERGES, load-bearing\n"
              "square (convex)                          : full 0.228  -> EMERGES, load-bearing"),
        ("h", "What it means"),
        ("g", "Order emerges across THREE different graded shapes but NOT under all-or-nothing. Robustness "
              "across shapes is the signature that the WORLD STRUCTURE (relations) drives it, not a tuned "
              "slope. So the result survives -- with a stated condition."),
        ("w", "The condition: emergence requires partial communication to carry partial survival value. "
              "That is realistic (understanding 'predator!' helps even if you miss 'from the north'); "
              "all-or-nothing is the unnatural case. But it is NOT free -- in a world where only perfect "
              "comprehension keeps a partner alive, order is unreachable. 'Graded reward is a free lever' "
              "was the wrong lesson; 'partial success must carry partial survival value, as it does in real "
              "worlds' is the right one."),
        ("h", "Reclassification"),
        ("b", "NATURAL grading (legitimate): food eaten (foraging), correct-recall count (memory), "
              "partial-comprehension value (syntax, robust to shape). DESIGNED grading (shaped, downgrade): "
              "the per-entity switch for binding; the killed proximity probe; 'make guessing worthless to "
              "open a gradient' in content-addressing."),
    ])

    # ---------- Figure 1: reward shapes + sex protects ----------
    fig = plt.figure(figsize=(8.5, 5.4)); fig.patch.set_facecolor("white")
    ax1 = fig.add_axes([0.08, 0.17, 0.40, 0.66]); ax2 = fig.add_axes([0.58, 0.17, 0.38, 0.66])
    shapes = ["all-or-\nnothing", "any", "thresh2", "per_role", "square"]
    sfull = [0.011, 0.062, 0.146, 0.225, 0.228]; sscr = [0.015, 0.030, 0.048, 0.067, 0.070]
    x = np.arange(len(shapes)); w = 0.38
    ax1.bar(x - w/2, sfull, w, color=ACC, label="intact")
    ax1.bar(x + w/2, sscr, w, color=MUT, label="scrambled")
    ax1.axhline(0.017, color=BAD, ls="--", lw=1); ax1.text(0, 0.025, "chance", fontsize=6.5, color=BAD)
    ax1.set_xticks(x); ax1.set_xticklabels(shapes, fontsize=7.5); ax1.set_ylabel("full-event accuracy", fontsize=9)
    ax1.set_title("Order emerges only with partial-value reward\n(and is robust to its shape)", fontsize=9.5)
    ax1.legend(fontsize=7.5)
    noises = [0.0, 0.15, 0.3]
    cl = [0.950, 0.659, 0.359]; cle = [0.019, 0.152, 0.157]
    sx = [1.000, 0.906, 0.789]; sxe = [0.000, 0.069, 0.095]
    xn = np.arange(3)
    ax2.errorbar(xn, cl, yerr=cle, marker="o", color=BAD, lw=2, capsize=3, label="clone")
    ax2.errorbar(xn, sx, yerr=sxe, marker="o", color=ACC, lw=2, capsize=3, label="sexual")
    ax2.set_xticks(xn); ax2.set_xticklabels(noises); ax2.set_xlabel("channel noise", fontsize=9)
    ax2.set_ylabel("English usage", fontsize=9); ax2.set_ylim(0, 1.05); ax2.legend(fontsize=8)
    ax2.set_title("Sexual reproduction protects English\n(n=8; clone has higher variance)", fontsize=9.5)
    fig.suptitle("Figure 1.  Reward-shape dependence (left) and a result that held up (right)",
                 fontsize=12, weight="bold", y=0.97)
    pdf.savefig(fig); plt.close(fig)

    # ---------- what held up ----------
    textpage(pdf, "3.  What held up under scrutiny", [
        ("h", "External memory is invented and load-bearing -- the strongest result"),
        ("g", "Invented (feedforward organism + persistent world slot): recall 0.67 +/- 0.19 with the slot "
              "vs 0.16 +/- 0.01 with it disabled (n=8). The high variance is seed-dependent REACHABILITY "
              "(some seeds never find the trick), but where it is found, severing the world collapses it to "
              "chance -- the memory is genuinely external."),
        ("g", "And it pays in REAL CURRENCY: in a pure-food foraging world (fitness = food eaten, no shaping "
              "of any kind), the same evolved organism eats 4.01 with its marks vs 1.93 when marks are "
              "disabled -- a 2x effect. For a local-vision feedforward forager, marks can ONLY serve as "
              "external memory, so this is memory paying its way in survival. (Figure 2.)"),
        ("h", "Sexual reproduction stabilises a shared language"),
        ("g", "Under channel noise, cloning lets English erode (0.36 +/- 0.16 at noise 0.3) while sexual "
              "reproduction holds it (0.79 +/- 0.10). Crucially cloning also has ~1.6x the variance: it "
              "propagates whatever each lineage drifted to, while crossover regresses to the shared code. "
              "Mechanism and magnitude both robust at n=8."),
        ("h", "Word order is genuinely used (not just present)"),
        ("g", "Scrambling the message slots before decoding collapses full-event accuracy from 0.22 to 0.055 "
              "(n=8). Meaning lives in the ORDER; this is the cleanest, least-shaped evidence of syntax and "
              "it is robust."),
        ("h", "Spatial layout uses the space (with a caveat)"),
        ("w", "Freezing the head collapses the tape organism from 0.83 to 0.41 (= the one-cell ceiling), n=6 "
              "-- it really uses the geometry. CAVEAT: this rests on a DESIGNED per-position reward; under "
              "pure world-consequence fitness it was unreachable. Report as reward-dependent."),
    ])

    # ---------- Figure 2: ablations + delay ----------
    fig = plt.figure(figsize=(8.5, 5.6)); fig.patch.set_facecolor("white")
    a1 = fig.add_axes([0.06, 0.55, 0.26, 0.34]); a2 = fig.add_axes([0.40, 0.55, 0.26, 0.34])
    a3 = fig.add_axes([0.73, 0.55, 0.24, 0.34]); a4 = fig.add_axes([0.10, 0.10, 0.82, 0.30])
    bars(a1, ["slot\non", "slot\noff"], [0.668, 0.161], [0.185, 0.012], [GOOD, BAD], "recall",
         "Invented memory\n(n=8)", chance=0.167)
    bars(a2, ["marks\non", "marks\noff"], [4.01, 1.93], None, [GOOD, BAD], "food eaten",
         "Foraging, same organism\n(pure-food fitness)")
    bars(a3, ["W=3\nmove", "W=3\nfrozen"], [0.831, 0.406], [0.221, 0.043], [GOOD, BAD], "reproduced",
         "Spatial: freeze\nablation (n=6)")
    Ls = [2, 4, 8, 12, 16, 24]; rec = [0.848, 0.631, 0.608, 0.616, 0.648, 0.612]
    rece = [0.103, 0.152, 0.058, 0.117, 0.133, 0.029]
    a4.errorbar(Ls, rec, yerr=rece, marker="o", color=ACC, lw=2, capsize=3)
    a4.axhline(0.167, color=MUT, ls="--", lw=1); a4.text(24, 0.18, "chance", fontsize=7, color=MUT, ha="right")
    a4.set_xlabel("delay L (steps the cue must survive)", fontsize=9); a4.set_ylabel("recall", fontsize=9)
    a4.set_ylim(0, 1.0)
    a4.set_title("Invented memory vs delay (n=5): one-step drop, then FLAT to L=24 -- external storage does not decay over time",
                 fontsize=9)
    fig.suptitle("Figure 2.  Ablations that held up, and the (refined) delay-invariance",
                 fontsize=12, weight="bold", y=0.97)
    pdf.savefig(fig); plt.close(fig)

    # ---------- what was repaired ----------
    textpage(pdf, "4.  What was repaired (artifacts, not lies, but wrong as stated)", [
        ("h", "'Sequential vastly beats the bag' -- a TRAINING ARTIFACT"),
        ("b", "Claim: a sequential channel (0.24) beat an order-free bag (0.14), proving the world REQUIRES "
              "order. Hole: both were trained only 600 generations, and the bag converges slower. Its "
              "analytic ceiling (convey action + entity-set, then GUESS which entity is the agent = x0.5) "
              "is ~0.4-0.5, nowhere near 0.14."),
        ("f", "bag full-event @  600 gens : 0.166      (undertrained -- the original number)\n"
              "bag full-event @ 1500 gens : 0.252      (its real plateau, = action+entities, guess roles)\n"
              "FAIR (both @1500): sequential full 0.369+/-0.035  vs  bag 0.247+/-0.028\n"
              "                   sequential swap 0.345+/-0.035  vs  bag 0.252+/-0.038"),
        ("w", "Repaired verdict: the order advantage is REAL and statistically separated (~3 sigma) but "
              "MODEST (~0.12), not the dramatic gap first reported. The bag does about as well as a "
              "role-guesser should. The strong, clean evidence for order remains the SCRAMBLE test, not the "
              "bag comparison."),
        ("h", "'External memory never forgets' -- REFINED"),
        ("b", "2-seed sweep gave a noisy 0.48-0.76 'no decay' story. 5-seed sweep (Figure 2, bottom) shows "
              "the true shape: a single drop (L=2 0.85 -> L=4 0.63) then a DEAD-FLAT plateau to L=24 "
              "(0.61-0.65). The plateau over 20 extra delay steps with zero decay genuinely is the external-"
              "storage signature -- a decaying internal memory would keep falling. So the claim is upheld "
              "for L>=4, but at ~0.6 (not 0.75) and with an honest initial-difficulty drop."),
    ])

    # ---------- failures in painful detail ----------
    textpage(pdf, "5.  Failures, in painful detail", [
        ("h", "Content-addressing -- a genuine wall (find by WHAT, not where)"),
        ("r", "A world that DEMANDS associative retrieval was built: key->value pairs in random arrival "
              "order, so position cannot be the address. Three escalating attempts, all failed:"),
        ("f", "linear policy        : ~0.44  -- degenerate fixed guess; a linear map over separate one-hots\n"
              "                              cannot even compute 'move to the cell indexed by the key'\n"
              "+ nonlinear hidden   : ~0.31  -- now expressible, still converges to a fixed output per\n"
              "                              query, ignoring storage entirely\n"
              "+ large value space  : ~0.10  -- reaches 'store ONE item, sometimes retrieve it', never\n"
              "                              two items keyed by content. Ablation: no use of the head."),
        ("r", "Honest: associative / content-addressed memory is beyond this gradient-free substrate. The "
              "capability hierarchy is real -- external storage (reached) < spatial layout (reached, small "
              "N) < content-addressing (NOT reached). And the third attempt ('large V to open a gradient') "
              "was itself a reward-shaping move, now disavowed. We did NOT keep adding architecture, because "
              "'fix it with a bigger brain' is the gradient-thinking trap."),
        ("h", "The foraging 'relocating' control was CONFOUNDED"),
        ("r", "Intended as 'memory useless -> marks should not help'. But relocating food reappears fresh at "
              "new cells, so it is MORE abundant (5.33) than persistent food (4.22). The control changed "
              "abundance, not memory's usefulness -- invalid. The valid evidence is the same-organism marks "
              "ablation (4.01 vs 1.93) within the persistent world."),
        ("h", "Compositionality (D) is near-tautological"),
        ("r", "Zero-shot held-out generalisation (0.88-1.00) appears ONLY with the anchor, which grounds "
              "each word INDEPENDENTLY. Independent per-word grounding IS compositional structure, so "
              "recombining into novel phrases is largely guaranteed by construction -- not evidence that "
              "compositionality EVOLVED. Honest claim: grounding ENABLES compositional reuse."),
        ("h", "Binding (G) leaned on a switched reward"),
        ("r", "The binding result only appeared after I changed the reward from per-component to per-entity "
              "-- i.e. tuned the reward until the result showed. Downgraded to suggestive / reward-dependent, "
              "not established. Absolute accuracy was ~0.03 regardless."),
    ])

    # ---------- hierarchy + thesis ----------
    textpage(pdf, "6.  Capability hierarchy by evidence quality, and the refined thesis", [
        ("h", "Graded by how clean the evidence is (not how exciting the claim)"),
        ("f", "external memory (marks/world)  STRONG   natural fitness; ablation 0.67/0.16 & food 4.0/1.9\n"
              "sex stabilises language        STRONG   n=8; 0.79 vs 0.36; variance mechanism matches\n"
              "word order is used             MODERATE scramble 0.22->0.055 (n=8); bag gap modest (fair)\n"
              "spatial layout                 MODERATE freeze ablation clean BUT designed per-pos reward\n"
              "constituent binding            WEAK     low absolute; reward was switched-to (shaped)\n"
              "content-addressing             WALL     fails three ways; not reached"),
        ("h", "The refined thesis"),
        ("b", "1. Capabilities emerge from worlds whose survival PHYSICS make them the cheapest way to live. "
              "External memory is the clean demonstration: it pays in food, ablation-proven, with zero "
              "reward shaping."),
        ("b", "2. The reachability 'lever' is NOT a free gradient. Emergence requires that partial progress "
              "carries partial survival value -- which real worlds provide (partial communication partially "
              "helps; more food is more life) but which I must not COUNTERFEIT with a designed slope. Where "
              "I counterfeited it (binding, the proximity probe, large-V), the results are weak or void."),
        ("b", "3. There is a real capability ceiling for this gradient-free substrate, and it sits at "
              "ASSOCIATIVE (content-addressed) memory. External and spatial memory are reachable; "
              "key->value association is not."),
        ("h", "What honestly remains to do"),
        ("b", "- Re-derive binding and spatial layout from a WORLD-CONSEQUENCE fitness (e.g. embodied), and "
              "see if they survive without the designed grading."),
        ("b", "- A genuinely reachable world for associative memory (embodied, where returning to a "
              "remembered place pays in food) -- WITHOUT a proximity slope. Hard, open."),
        ("b", "- Confidence intervals on everything still at n<=3 (the binding/content numbers)."),
        ("h", "Bottom line"),
        ("b", "The spine of the arc survives: worlds that demand a capability can evolve it, gradient-free, "
              "and external memory shows it cleanly in survival currency. But roughly a third of the "
              "headline drama was either a training artifact (the bag) or rested on a reward I shaped "
              "(binding, spatial). Stated honestly, the result is narrower and more trustworthy: build the "
              "world's physics, never the reward's slope."),
    ])

print("saved", OUT)
