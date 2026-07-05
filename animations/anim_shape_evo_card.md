# Model card — `anim_shape_evo` (GENREG §II)

**Purpose.** Gradient-free evolved classifier that names the moving shape from a
single 64×64 frame. Trained by presenting ONE clip at a time (shuffled order);
**energy homeostasis integrates performance across the shuffled clip stream**, so
a genome that only nails the current clip drains energy when the next clip is
shown — only generalists that stay above the population median across all shapes
keep energy and survive. Energy, not per-clip fitness rank, is the memory that
defeats catastrophic interference. No gradients, no crossover.

**Interface.** in: frame (4096,) f32 [0,1]. out: 10 shape logits. Stateless.

**Evolved params / genome.** W1(4096×12)+b1(12)+act1_ids(12), W2(12×24)+b2(24)+
act2_ids(24), W3(24×10)+b3(10). ~49.7k. Per-neuron activation id ∈ 8-func catalog
[identity, tanh, relu, sigmoid, sin, gaussian, abs, softsign] (§VI signature
primitive). Weights Xavier init; biases 0; act ids random.

**Fitness (§IV.1-2, soft + multiplicative).** On the presented clip's TRAIN
frames, `f = exp( mean_frames log softmax(logits)[correct] )` — geometric mean of
the correct-class probability ∈ (0,1]. Not accuracy (stair-step, no gradient).

**Energy (§III, mandatory).** `E ← clip(E·DECAY + GAIN·(f − median_f), 0, E_MAX)`.
DECAY 0.9, GAIN 2.0, FLOOR 0.2 (cull), E_MAX 1.5, init 0.7. Target 3–15% starved/gen.

**Selection/mutation (§V).** POP 300. Dead (E<FLOOR) slots replaced by offspring
of MATURE (age≥1) survivors, tournament-weighted by fitness. Self-adaptive
mut_rate [0.005,0.2] start .05 and mut_scale floor .02 start .05; anneal ×0.5 after
80% gens. Mutation only.

**Success (§VII).** Held-out (per-clip 80/20 frame split) top-1 ≫ 10% majority
baseline; ideally all 10 shapes recognized; train→heldout drop <10%. Synthetic
deterministic task → 80%+ achievable (§VIII).

**Failure modes.** mode-collapse (all one shape), energy decorative (starved 0) or
genocidal (>50%), train/heldout gap, position-overfit within a clip.
