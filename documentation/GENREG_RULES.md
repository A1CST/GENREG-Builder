# GENREG Core Rules

Derived from all documents in this folder. These are the invariants every
GENREG model must respect — the hard prohibitions, the mandatory properties,
the validated primitives, and the failure modes to predict in advance.

---

## I. Foundational Prohibitions (Hard Rules)

1. **No gradients, no backprop, no hybrid.** Pure gradient-free evolution only.
2. **No scaling shortcut.** Don't scale `hidden_dim` or architecture to fix a
   plateau — apply landscape pressure instead.
3. **No long runs.** Cap at 2–8k generations per component sweep (5k max per
   iteration). Make changes fast, validate fast.
4. **No metric-inflation tricks.** Never let OOV→id-0 collapse or similar
   inflate a single-class marginal.
5. **No cascade-from-random-init** and no lookup-based n-gram channels
   (superseded).
6. **Never celebrate** before passing all three verification checks (§VII).

---

## II. Mandatory Properties — Every GENREG Model Must Define

Draft these **before** writing code (per `LLM__components__MODEL_TEMPLATE.md`):

1. **Name** — `<model_id>`
2. **Purpose** — one paragraph; what breaks without it
3. **Interface** — input/output shape, dtype, range; runtime state or "stateless"
4. **Evolved parameters** — for each tensor: what it is, shape, init scheme,
   init rationale, total count
5. **Fitness equation** — full formula; per-term meaning, range, weight
   rationale
6. **Energy equation** — `ENERGY_DECAY`, `ENERGY_GAIN`, `ENERGY_FLOOR`,
   `E_MAX`, delta signal
7. **Selection** — `POP_SIZE`, `SURVIVAL_PCT`, maturation gate flag,
   reproduction method
8. **Mutation** — `mut_rate` and `mut_scale` (both self-adapted), per-tensor
   scaling, anneal policy
9. **Hyperparameters** — `N_GENERATIONS`, `BATCH_SIZE`/`PROBE_SIZE`,
   `LOG_EVERY`, `ANNEAL_AFTER`
10. **Success criteria** — local bar AND downstream bar (both required)
11. **Failure modes to watch** — enumerated in advance with the log pattern
    that reveals each
12. **Baselines to beat** — random-init + closed-form readout, hand-crafted
    solution, prior model generation
13. **Artifacts** — `<model>_best.pkl`, `<model>_findings.md`, `run_<model>.log`

---

## III. Energy Homeostasis (Non-Negotiable)

- Energy is **mandatory** on every model; it is **homeostatic, not a reward
  signal** — it determines who survives, independent of fitness rank.
- `energy_next = energy_current * DECAY + delta`, clamped to `[0, E_MAX]`.
- Typical ranges:
  - `ENERGY_DECAY` 0.85–0.95 (too high ⇒ inert; too low ⇒ dies before proving)
  - `ENERGY_GAIN` 1.5–3.0
  - `ENERGY_FLOOR` 0.15–0.25 (hard cull threshold)
  - `E_MAX` ~1.5 (ceiling — prevents lucky early immortals)
- Delta signal is usually `fitness - median_fitness` (relative performance),
  optionally plus explicit costs/bonuses.
- **Target starved-per-gen: 3–15% of population.**
  - `starved == 0` → energy is decorative, selection is pure tournament
  - `starved > 50%` → energy is genocidal; good genomes die before proving

---

## IV. Fitness Landscape Rules

1. **Soft fitness only.** Use `mean log_prob[target]` (negative cross-entropy).
   `argmax == target` is a discrete step function with no climbing gradient.
   A_4/A_6/A_7 stuck ~10% with hard fitness; A_8 broke to 24% the moment soft
   fitness was introduced.
2. **Multiplicative > additive.** Use geometric mean across per-target scores;
   summed fitness collapses to the mean-seeker (multi-number V1 went to 0% at
   N=5; V2 with geometric mean → 94% at N=100).
3. **Energy as gradient, not filter.** Never make it a separate survival
   threshold that hides the gradient.
4. **Every scalar proxy gets reward-hacked.** Four consecutive Goodharts on
   this project (n-gram rollout, vbias, top-K recall, cosine-emb α=0). Anchor
   fitness to position-varying ground truth.
5. **The landscape is designed, not given.** Basin engineering is the central
   problem — make the target the only stable attractor.
6. Single-target landscapes can be funnels (Find-42's 7-layer funnel:
   direction → magnitude → neighborhood → proximity → precision → kill-zone).
7. Noise-driven culling destroys ratchets — add EMA smoothing when fitness is
   noisy gen-to-gen (CIFAR V1 12% → V2 27% from this alone).

---

## V. Selection & Mutation

- **Tournament selection with maturation gate** (offspring cannot reproduce
  until surviving one full generation). Known-good.
- `SURVIVAL_PCT` ≈ 20% (lower = harsher, batch-noise risk; higher = slower
  convergence).
- `POP_SIZE` 300–500 on 15GB GPU; 1000+ with more VRAM.
- `mut_rate` self-adapts per genome, bounds `[0.005, 0.2]`, start 0.03–0.08.
- `mut_scale` self-adapts per genome, start 0.02–0.08.
- **Keep a mutation floor (~0.02)** — V3→V4 CIFAR showed mutation collapse
  (0.028) starved exploration.
- Per-tensor scaling: 3D→`s.view(-1,1,1)`, 2D→`s.view(-1,1)`, scalars simple;
  activation IDs have bespoke rules.
- **Anneal late.** Halve `mut_scale` after ~80% of gens for coarse → fine.
- Reproduction: tournament-weighted sampling from the elite pool (fitness →
  probability).

---

## VI. Architectural Primitives (The Working Ones)

- **Per-neuron evolved activation from the 8-function catalog.** The GENREG
  signature primitive — don't skip it. Each genome literally "sees" through a
  different mathematical lens.
- **Low-rank trigram interaction**:
  `logits = bigram[a] + (E1[a] * E2[b]) @ O` where `E1, E2: (V, H)` and
  `O: (H, V)`. Multiplicative — captures real interaction structure additive
  K-gram cannot.
- **Two-phase training** to break local minima: (1) mutate bigram only,
  (2) freeze bigram, mutate trigram interaction, (3) both for fine-tuning.
  A_24 confirmed this beats joint-from-random.
- **Protein cascade** (stacked decay-rate layers = local + phrase-level
  memory) **must be bootstrapped** — init as no-op (`decay→1`, `momentum→0`)
  then evolve. Cannot be evolved from random init (A_30 regression).
- **Hash output trick** replaces argmax over wide vocab with hash projection
  of the full hidden state. +269% on tokenizer. Removes the ~50-bin argmax
  bottleneck.
- **SVD-of-bigram-matrix** is the fallback embedding init when no evolved
  embedding exists.
- **Rerank generation** — n-gram proposes candidates, attention scores them
  by `cosine(attn_out, candidate_emb)`. Bypasses the flat ridge softmax
  (std ≈ 0.003) entirely and is the current best generation method for
  multi-clause coherent English.

---

## VII. Verification Checklist (Before Claiming Any Win)

Every claimed result must pass all three:

1. **Majority-class baseline.** Real top-1 must be meaningfully above "always
   predict the most common chunk/word/char" on the same stream. "Nx random"
   is misleading.
2. **Held-out split test.** Random per-pair (or per-window) 80/20 split.
   Train→heldout drop < 10% relative. Contiguous splits on real text are NOT
   fair — different sections have different statistics.
3. **Inspect actual predictions** on a real prefix. Continuous fitness hides
   nonsense (experiment O looked great but was pure byte averaging).

---

## VIII. Corpus / Data Rules

- Real-text bigram ceilings are hard-capped low:
  - WikiText words (vocab 128–256): ~22% top-1
  - WikiText words (vocab 64): ~24% top-1
  - **WikiText chars (vocab 32): ~27% top-1, trigram ~37%** ← best
- **Character-level beats word-level** on natural text — character
  conditionals are sharper (`q`→`u`, `th`→`e`).
- **More data does not raise the ceiling** on natural text (A_28 at 200K words
  same as A_25 at 80K). The wall is intrinsic.
- Smaller vocab → higher utilization but caps trigram ceiling too.
- Synthetic deterministic corpora can reach 80%+ but they're toy.
- **Generalization requires structure in the data.** Random images → random
  numbers: test correlation −0.05 (pure memorization). Structured (visual
  encoding of value) images → numbers: 0.9881 on 200 unseen pairs.

---

## IX. Operational Rules

- Always log `soft_score` (mean log-prob), `top-1`, AND `top-5`. `soft_score`
  shows landscape progress when `top-1` plateaus.
- Mini-batch when OOM. Final eval uses chunked full-stream.
- **Bootstrap when possible** — two-phase training or load a prior checkpoint.
  Don't re-learn from scratch.
- **Save checkpoints with ALL learned state**, not just the primary tensor.
  A_13 lost its trigram interaction by inheriting A_8's save structure.
- Keep `LOG_EVERY` reasonable — 100 gens of noise is hard to read; too
  frequent and the log is a wall of numbers.

---

## X. Component-First Discipline

- Each sub-component is its own GENREG model with an isolated fitness task.
- A component must clear **both** its local bar AND its downstream bar before
  being frozen and composed.
- Freeze-and-compose only — never re-train a frozen component mid-pipeline.
- Canonical build order: embedding → attention → readout → optimizer
  (meta-GENREG). Assembly target: match or beat A_101's 34% heldout top-1.

---

## XI. Known Failure Modes (Predict in Advance)

- Training fitness climbs but heldout is flat → train/inference distribution
  mismatch.
- All genomes converge to identical output → mode collapse; mutation can't
  escape.
- **Reward hacking** — fitness satisfied without doing the intended task.
  Seen repeatedly: n-gram rollout (rare-token self-loops), vbias (`and` soup),
  top-K recall (`to/the/of` soup), cosine-emb (α=0 trivial).
- `starved == 0` for 500+ gens → energy system is decorative.
- `starved > 50%` of pop → energy is genocidal.
- Flat softmax with tiny std (~0.003) over wide vocab → sampling
  uniform-over-top-K (ridge broadcast bug).
- Head fit on bidirectional attention features fails at 8% t1 on causal
  features at inference (`predhead_wiki_causal` mismatch bug).

---

## XII. The GENREG Thesis (Underlies Everything)

- **The fitness landscape is the only lever that matters.**
- Architecture is downstream of landscape. Optimizer choice is downstream of
  landscape.
- **Don't design the solution — design conditions where the only stable
  attractor is the solution.**
- Some problems have no exploitable basin regardless of landscape design
  (SHA-256 by construction; XOR-fold is learnable at +2.3 bits).
- Gradient descent and evolution are traversal strategies on designable
  landscapes — the terrain determines which is optimal. Step functions,
  integer ops, conditional branches, and compositions thereof have zero
  gradient everywhere; evolution climbs those staircases, gradients cannot.
