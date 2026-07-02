# EEC — Catalog of Constraints (Laws of Existence)

Companion to `PARADIGM.md`. This is the running record of every constraint we have
imposed on the recurrent organism and what EMERGED under it. Per the paradigm we
grade by what the organism BUILDS (internal state — recurrent gain, memory horizon,
evolved memory M, population diversity, fertility), **never** by accuracy/loss.

Worlds used throughout:
- **text** — WikiText, V=2000, thin near-unigram signal (memory rarely pays).
- **long-range** — periodic block (len 24, alphabet 6) + 5% noise; phase-memory
  predicts, so structure is recoverable and exploitable.

Verdict legend: REAL = real & productive; WORLD-GATED = fires only where it pays;
WEAK = present but small; WALL = search-walled; SURVIVAL-AXIS ALT = a valid
instantiation of the survival axis (redundant if stacked on energy, viable if
swapped in).

---

## 1. The capability-axis view

A constraint is not "pressure" in the abstract. Each one taxes a specific channel
and, where the world makes the alternative pay AND reachable, grows a specific
capability. The axes we have touched:

| axis | what it is | driven by |
|------|-----------|-----------|
| survival/metabolism | staying alive in the stream | ENERGY, MEMORY-RENT |
| parsimony | doing it with less | TIME/Occam, RENT |
| active maintenance | holding state against loss | ENTROPY |
| persistence/memory | bridging gaps in observation | OCCLUSION (·NOISE) |
| selection pressure | who gets to reproduce | REPRODUCTION-COST |
| diversity/niches | an ecosystem vs a monoculture | SCARCITY |
| attention/sampling | when to pay to observe | PERCEPTION-COST |
| adaptability/plasticity | coping with a changing world | NON-STATIONARITY |

**Constraints are interchangeable INSTANTIATIONS of an axis — pick ONE per axis to
match the world.** The catalog is not a fixed linear sequence; it is a GRAPH of
axes, each offering multiple constraint implementations. The survival axis can be
instantiated by ENERGY (additive budget) *or* MORTALITY (multiplicative hazard) —
swap, don't stack (stacking the same axis wastes budget and adds nothing). The
observation axis has OCCLUSION *and* NOISE; which one "fires" depends on the world,
not on one being intrinsically strong. Different instantiation → different organism
at the tip (same axis covered, different internal structure: see the swap result,
mortality builds far more recurrent gain than energy at equal competence). So PO
counts **axes covered, not constraints stacked**, and an organism's full coordinate
is **(PO, fitness)** — how deep into the cone, and how well it survives there.

---

## 2. The catalog

| # | Constraint | Mechanism | Capability axis | Verdict | Key result |
|---|-----------|-----------|-----------------|---------|-----------|
| 1 | **ENERGY** | prediction=metabolism; miss burns energy, death at 0 | survival | REAL | the base law; lifespan IS fitness |
| 2 | **TIME / Occam** | fitness × 1/(1+time/budget) | parsimony | REAL | selects generalization; coupling, not schedule |
| 3 | **MEMORY-RENT** | hold cost ∝ M energy/step | parsimony of memory | REAL | M evolves against energy; vestigial unless world pays |
| 4 | **ENTROPY (decay)** | carried state leaks unless refreshed | active maintenance | REAL | recurrent **gain ×3.2** (text), up to **0.54** (long-range) |
| 5 | **OCCLUSION** | hide fraction ρ of input (sensory blackout) | persistence/memory | REAL (regime-dep) | gain rises **~2× (0.13→0.24, 4/6 seeds, triple-checked)**; orig 2-seed 0.06→0.37 overstated; masked at tight energy; text null |
| 6 | **NOISE** | corrupt input (additive, rel-SNR) | persistence/memory | WEAK | gain only 0.06→0.14, M at floor; **missing ≫ corrupted** |
| 7 | **SCARCITY** | shared finite food, carrying capacity | diversity/niches | WORLD-GATED | text niches **1.5→7.5**; inert in long-range (no reachable niches) |
| 8 | **REPRODUCTION-COST** | prediction→surplus→offspring (cost CHILD) | selection pressure | WORLD-GATED | long-range surplus **+41%**, fertility Gini **0→0.11**; text starves (no surplus) |
| 9 | **MORTALITY** | survival law: hard age-cap, OR hazard (death-risk rises with misses) | survival (alternative to ENERGY) | SURVIVAL-AXIS ALT | dead only when STACKED on energy (redundant); as the SOLE law, hazard-mortality gives competence 0.65 > energy 0.54 > random 0.17. Same axis: swap, don't stack. |
| 10 | **PERCEPTION-COST** | state-gated attention; looking drains energy | attention/sampling | WALL | **economy yes** (5× less looking, no survival loss); **selective attention no** (constant gate) |
| 11 | **NON-STATIONARITY** | world's structure shifts every K gens | adaptability/plasticity | REAL (modest) | diversity **1.3→2.0**; per-shift drop 19%, **recovers 84%** in 5 gens |

Interaction & mechanism experiments:

| Constraint pair / variant | Finding |
|---|---|
| **OCCLUSION × ENTROPY** (same axis) | **INTERFERE** — combined gain 0.21 < either alone (0.37, 0.54). Share a degradation budget. |
| **OCCLUSION × REPRODUCTION-COST** (different axes) | **STACK / COMPOUND** — combined gain 0.478 > both; each signature preserved (occ→gain, repro→Gini). |
| **PERCEPTION via searchable channel** | gate on previous-miss with evolved scalar α: α→−1.6/−1.9, a **conditional policy evolves** where the emergent gate produced **zero**. A channel breaks the wall. |
| **SURVIVAL-AXIS SWAP** (energy vs mortality vs both) | **Mortality alone is viable** (competence 0.65 > energy 0.54 > random 0.17); **BOTH (0.59) adds nothing** over the better single → same axis. Different law → **different organism** (mortality gain 0.68 vs energy 0.26 at similar competence). |

---

## 2c. Verification — 6-seed ablation triple-check

Every headline finding was re-run ON vs OFF, paired, 6 seeds, with a sanity control
(`experiments/ablation/ablation.py`). Findings sorted by what survived:

- **ROCK-SOLID (6/6 seeds):** COMMUNICATION→protocol (acc 0.75 vs severed 0.25);
  REPRODUCTION-COST→ecosystem (Gini only with repro-cost, competence held);
  PERCEPTION→economy (look 0.30→0.05).
- **REAL but smaller/noisier than first reported:** OCCLUSION→gain (~2× not 6×, 4/6,
  regime-dependent — the orig 0.06→0.37 was 2 lucky seeds); ENTROPY→gain (5/6, high
  variance); RENT→M robust, competence payoff modest (4/6).
- **NULL CONTROLS confirmed (correctly flat):** occlusion in TEXT; NOISE→gain.
- **FLAGGED:** SCARCITY→diversity consistent (6/6) but at near-chance competence (0.04)
  — diversity is real, "functional ecosystem" not established by the count alone.
- **NOT ROBUST:** NON-STATIONARITY→diversity (3/6; orig 1.3→2.0 sits inside the noise;
  only the recovery aspect holds).

Methodology rule baked in: report effects with their **regime and seed-count attached**,
never as universal; never trust a 2-seed magnitude. No fabricated effects were found —
every claim has the correct sign on a majority of seeds — but several were regime-
sensitive or lived in low-competence regimes.

## 3. Meta-principles (the laws *about* laws)

These are the durable, transferable findings — they predict outcomes before we run.

**P1 — Reachability / the world must pay.** A constraint creates pressure, but a
capability emerges only where (a) the world makes it pay and (b) the organism can
*reach* the alternative by mutation. Occlusion grows memory only in long-range
(recoverable), never text. Reproduction-cost ignites only where surplus is earnable.
Mortality grows nothing — it removes no survival strategy energy doesn't already.

**P2 — Degradation budget / the cone has wall thickness.** Constraints aimed at the
**same capability axis** share a finite coping budget. There is an OUTER boundary
(too little pressure → capability not forced) and an INNER boundary (too much →
world unlearnable → selection collapses → drift to baseline). The organism lives in
the **Goldilocks band** between. Proven: occlusion peaks at ρ≈0.4 then collapses;
occ+entropy co-maxed collapses below either alone. The 2D occ×entropy grid is the
first map of this interior (`experiments/cone_interior_grid/`): no synergistic
interior ridge — the optimum is always single-axis (best edge > best interior on
gain, horizon, and M).

**P3 — Two ropes, not one wall.** Same capability *label*, different internal
*mechanics*. Entropy drives **gain** (pump against leak); occlusion drives
**horizon** (persist across gaps). Because they pull different ropes from one
budget, stacking them strands the organism — hence P2's interference. Always read
multiple internal facets; a single metric (gain) would have mislabeled this.

**P4 — Orthogonal axes stack cleanly.** Constraints on **different** axes share no
budget and compose (even synergize). occ (memory) × repro-cost (selection) →
combined gain *above* either, both signatures intact. **Design rule:** before
stacking two constraints, ask whether they share a degradation budget (same axis →
find the ridge, expect interference) or are orthogonal (different axis → stack).

**P5 — The search-wall: conditional capability needs a channel, not pressure.**
State-conditional / policy behaviors ("look when surprised", "route this feature")
are unreachable by random mutation when a *constant* suffices — even under strong
selection that rewards the policy. The emergent attention gate never became
selective (constant only); a directly-searchable channel (evolved scalar on a
provided miss signal) immediately produced a conditional policy. This is the same
wall that defines the LM lineage (frozen n-gram channel + evolved mix beats emergent
routing). Pressure shapes *how well* a reachable solution is found; it does not make
unreachable solutions reachable.

**P6 — The reproduction operator is itself a law.** Turnover dynamics are
load-bearing. Full generational replacement out-converges any ecological pressure
(masked scarcity entirely); steady-state overlap (top 20% breed, bottom 20% culled,
middle carried) lets diversity persist and reveals scarcity/repro effects.

**P7 — Axes, not constraints; cover each axis once; (PO, fitness) is the coordinate.**
A constraint is one INSTANTIATION of a capability axis; you pick one per axis to fit
the world (ENERGY *or* MORTALITY for survival; OCCLUSION *or* NOISE for observation).
Stacking two on the *same* axis is redundant (the survival-swap test: BOTH adds
nothing over the better single) — this is the deep reason mortality once looked
"degenerate" (it was only ever doubled with energy). Covering a *new* axis is what
compresses the cone. So **PO = axes covered, not constraints counted**, and the
metric is the inverse twin of fitness: PO measures top-down (how much of infinity
eliminated), fitness bottom-up (how well the organism survives what's imposed); both
asymptote to the same unreachable tip from opposite sides. An organism's full
identity is **(PO, fitness)** — fitness alone is meaningless without knowing how many
axes produced it. And the *choice* of instantiation matters: same axis, different
constraint → different organism at the tip (mortality-selection builds more recurrent
gain than energy-selection at equal competence).

*Refinements (the axis graph is not a stack of independent rings):*
- **Instantiations can be ASYMMETRIC.** Survival's energy↔mortality are genuinely
  swappable (comparable competence). But the observation axis has a **dominant**
  instantiation: occlusion robustly beats noise in every world tested (missing forces
  memory harder than corrupting — world-independent). Not every axis offers a free swap.
- **REDUNDANCY.** A trait can be covered by more than one axis (memory rises under
  observation *or* selection). Covering it twice is wasted PO — pick one.
- **CONFLICT.** Some axes fight over a shared resource. Parsimony (rent, wants small
  M) directly degrades occlusion/selection-driven memory (gain 0.36→0.13 when added):
  a cost axis can SUBTRACT a capability. More axes ≠ more capability.
- **CLEAN OWNERSHIP.** Some traits have exactly one source (the fertility ecosystem
  comes only from selection, ablation-verified). There, PO contribution = that one axis.

So minimizing PO is: cover each *needed* trait once, with the instantiation the world
favors, avoiding axes that conflict with traits you want.

---

## 4. Methodology notes (hard-won)

- **Read the STATE, not the output.** Headline numbers (e.g. "diversity 37", "M=16")
  were repeatedly **selection-collapse artifacts** (neutral drift once selection was
  erased), caught only by reading internals and checking competence.
- **Survival-saturation pitfall.** In long-range with default energy (500), everyone
  survives the full segment → selection saturates → many laws read as a flat null
  (mortality, attention, non-stationarity all hit this). **Fix: tighten START_ENERGY
  (~40–80) so survival depends on the capability under test**, then re-measure.
- **Verify before celebrating.** Always check the baseline and whether an effect
  could be drift/saturation. Two independent lines agreeing (e.g. ρ-sweep collapse +
  stack antagonism) is what makes a finding trustworthy.
- **Convergence vs cost.** Internal structure needs ~150 gens at seg≈600 to prune to
  baseline; shorter segments/fewer gens leave init noise. Don't trade convergence for
  speed — parallelize instead.

---

## 5. Where to find each experiment

All scripts import the shared engine via a path bootstrap; run as
`python3 experiments/<name>/<script>.py`. See `project_eec_folder_layout` (memory)
or `engine/` for the substrate.

| experiment | folder |
|-----------|--------|
| reward-shaping era (superseded) | `experiments/01_reward_shaping/` |
| perception (early) | `experiments/perception/` |
| entropy / long-range / memory | `experiments/entropy_memory/`, `engine/run_matrix.py` |
| scarcity | `experiments/scarcity/` |
| mortality | `experiments/mortality/` |
| reproduction-cost | `experiments/reproduction_cost/` |
| occlusion | `experiments/occlusion/` |
| observability board (occ/noise/stack) | `experiments/observability_board/`, `engine/batch_board.py` |
| cone-interior 2D grid | `experiments/cone_interior_grid/` |
| perception-cost + searchable channel | `experiments/perception_cost/` |
| orthogonality control | `experiments/orthogonality/` |
| non-stationarity | `experiments/nonstationarity/` |
| survival-axis swap (energy / mortality / both) | `experiments/axis_swap/` |
| observation-axis swap (occlusion vs noise) | `experiments/observation_swap/` |
| PO axis-covering lattice | `experiments/po_axes/` |

_Last updated 2026-06-20._
