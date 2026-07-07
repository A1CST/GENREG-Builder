# Model card: `pure_layered_v1` — per-layer (localized) constraints

Per GENREG_RULES §II. The idea (user, 2026-07-06): **tissue differentiation
without changing the evolutionary machinery.** PURE's node graph already wires
a constraint to a specific layer; the only new mechanic is evaluating each
constraint against its **wired layer's activations**, not the whole network's.
Same tournament, same energy, same whole-organism reproduction — the fitness
landscape just becomes *locally uneven*, so different layers face different
survival conditions and specialize like tissues.

## Fitness (unchanged machinery, localized penalties)
`fitness = base_soft × Π_c penalty_c(activations of c's wired layer)`
- base_soft = mean log-softmax[target] (soft, multiplicative — §IV.1).
- Global (control): every penalty reads the whole-network activation stat.
- Per-layer (test): penalty reads only its wired layer.
Selection/energy/tournament/mutation identical to genreg_lm.

## Constraints tested (from the PURE catalog)
- **Energy** (power cost) on Layer 1: `penalty = 1/(1 + mean|a_L1|/budget)`.
  A power-hog L1 drags the whole genome down even if output is perfect.
- **Consequential Drive** on Layer 2: neurons must MATTER. Per neuron j,
  consequence = std(a_j) · mean|W_out row j| (how much it moves the output).
  `penalty = 1/(1 + dead_frac/budget)`, dead = consequence below a floor.
  A dense-but-useless L2 gets culled; a lean-but-load-bearing L2 thrives.

## Architecture
Next-char task (real, connects to the LM line). Input = last K chars embedded
→ **L1** (dense + per-neuron 8-catalog activation) → **L2** (dense + activation)
→ readout → V logits. Everything evolved; no gradients.

## Success criteria (the decisive comparison)
Per-layer vs global, matched compute:
1. **Differentiation**: under per-layer, L1 mean-activation-power drops sharply
   while L2's does NOT (Energy is local to L1); L2 dead-neuron fraction drops
   while L1's does NOT (Consequential Drive is local to L2). Under global, both
   layers move together — no clean differentiation.
2. **Cooperation preserved**: held-out next-char accuracy under per-layer ≥
   the unconstrained control minus a small margin (the organism still works).
3. Honest null: if per-layer looks the same as global, the mechanic adds
   nothing here — report it.

## Failure modes
Constraint budgets too harsh (genocidal — starves the task) or too soft
(decorative). Log per-layer power + dead-frac every gen for both layers.
Energy homeostasis starved% in 3–15% band as always.

## Artifacts
`genreg_train/pure_engine.py` (reusable — the backend PURE's graph feeds),
`runs/pure/<id>/…`, findings appended here.

## 2026-07-06 — RESULT: tissue differentiation confirmed, controlled by the wire

4 matched conditions (pop 400, 3000 gens, next-char, Energy budget 0.3 on one
layer, Consequential Drive on the other):

| condition | top1 | L1 pow | L2 pow | L1 dead | L2 dead |
|---|---|---|---|---|---|
| control (no constraints) | 20.6% | 0.316 | 0.503 | 0% | 0% |
| global (whole-net eval) | 20.8% | 0.074 | 0.187 | 0% | 0% |
| **per-layer** E→L1, Cq→L2 | 20.8% | **0.039** | 0.340 | 1% | 0% |
| **swapped** E→L2, Cq→L1 | 20.8% | 0.217 | **0.103** | 0% | 18.8% |

**The mechanic works and is causal.** Energy wired to L1 crushes L1's power to
0.039 (8× below control) while L2 keeps its natural ~0.34 — a *local* pressure.
Swap the wire (Energy→L2) and the collapse MOVES to L2 (0.103) while L1 stays
high (0.217). Which layer becomes the low-power "tissue" is determined by which
layer the constraint is wired to — exactly the prediction. Global eval can't do
this: it compresses both layers together (0.074 / 0.187), no differentiation.

**Cooperation preserved.** Held-out top-1 is 20.8% in every constrained
condition — equal to or above the unconstrained control (20.6%). The layers
specialized without breaking the task; same tournament, same energy, same
whole-organism reproduction.

**Interaction observed (swapped row):** with Energy crushing L2 and no
Consequential Drive on L2, L2 grew 18.8% dead neurons — energy starvation
without a "must-matter" counter-pressure lets a layer atrophy. Consequential
Drive wired to a layer holds its dead-fraction at 0%. So the two constraints
compose as expected: Energy trims power, Consequential Drive prevents collapse.

**Verdict:** per-layer constraint evaluation is a real, controllable landscape
lever — genuinely new vs global constraints. Next: (a) wire the PURE node graph
(`PureGraph.getGraph()`) into this engine so constraints-to-layers is authored
in the browser; (b) apply it to the LM — Consequential Drive on the readout is
a candidate cure for the word-level function-word Goodhart.
