# LM_RULES.md — gradient-free text predictor work

Rules to follow during the autonomous A_5+ iteration. Built from the
A_1..A_29 sprint and the prior GENREG documentation.

## VERIFY BEFORE CELEBRATING (the prime rule)

Every claimed result must pass three checks before I call it a win:
1. **Majority-class baseline check.** Compute "always predict the most common
   chunk/word/char" accuracy on the SAME stream. Real top-1 must be
   meaningfully above this floor. The "Nx random" metric is misleading.
2. **Held-out split test.** Random per-pair (or per-window) 80/20 split. The
   train→heldout drop must be small (<10% relative). Contiguous splits on
   real text are NOT fair — different sections have different statistics.
3. **Inspect actual predictions.** Print what the model outputs on a real
   prefix. Generous fitness with continuous scoring is deceptive (e.g.
   experiment O looked great but was just byte averaging).

## FITNESS LANDSCAPE (the wall I keep hitting)

- **Soft fitness only.** `argmax == target` is a discrete step function with
  no climbing gradient. Use `mean log_prob[target]` (negative cross-entropy).
  Confirmed: A_4/A_6/A_7 stuck at ~10% with hard fitness; A_8 broke through
  to 24% the moment soft fitness was introduced.
- **Energy as gradient, not filter.** Random baseline is energy-NEUTRAL,
  improvements ADD, collapse DRAINS. Don't use energy as a separate survival
  threshold that hides the gradient.
- **No metric inflation tricks.** OOV→id-0 collapse made "always predict the"
  win. Drop OOV from the stream entirely OR give it a dedicated id. Don't
  inflate the marginal of any single class.
- **Multiplicative fitness > additive.** Additive fitness lets degenerate
  strategies score by hitting one term. Multiplicative makes them score ~0.

## ARCHITECTURE (what works on this substrate)

- **Pure bigram (V×V table) cannot memorize trigram patterns** — by
  construction. Use it as the generalization gold standard.
- **Low-rank trigram interaction**: `logits = bigram[a] + (E1[a] * E2[b]) @ O`
  where E1, E2: (V, H), O: (H, V). Multiplicative — captures real
  interaction structure additive K-gram cannot.
- **Two-phase training** to break local minima: phase 1 only mutates bigram,
  phase 2 freezes bigram and only mutates trigram interaction. Phase 3 does
  both for fine-tuning. A_24 confirmed this beats joint training from
  random init.
- **Protein cascade is the documented context primitive.** Stacked layers
  with different decay rates = local + phrase-level memory. A_9 failed only
  because of argmax fitness — try cascade + soft fitness next.
- **Hash output trick (+269% on tokenizer).** Replace argmax over wide vocab
  with hash projection of full hidden state. Each output bit reads the
  entire hidden vector. Removes the ~50-bin argmax bottleneck.
- **Evolved per-neuron activation from the 8-function catalog.** Each
  genome literally "sees" via a different mathematical lens. This is the
  GENREG signature primitive — don't skip it.
- **Don't scale hidden_dim to fix plateaus.** Tanh saturation already gives
  2^k partitions. Bigger ≠ better.

## CORPUS RULES (entropy is the wall on real text)

- **Real text bigram ceilings are HARD-CAPPED low**:
  - WikiText words (vocab 128-256): ~22% top-1
  - WikiText words (vocab 64): ~24% top-1
  - **WikiText characters (vocab 32): ~27% top-1, trigram 37%** ← best
- **Character-level beats word-level on natural text** because character
  conditional distributions are sharper ("q" → "u", "th" → "e").
- **Synthetic deterministic corpora** can reach 80%+ but they're toy.
- **Conversational/dialog corpora** would help (formulaic structure) but
  HuggingFace daily_dialog is deprecated. Need to find an alternative.
- **More data ≠ higher ceiling on natural text** — A_28 with 200K words
  gave the same bigram ceiling as A_25 with 80K. The wall is intrinsic.
- **Smaller vocab → higher utilization but caps trigram ceiling too.**
  Tradeoff: A_27 vocab=64 hit 91% of bigram ceiling but ceiling was lower.

## OPERATIONAL RULES

- **No long runs.** Cap N_GENERATIONS at 5000 for iteration. Make changes
  fast, validate fast.
- **Mini-batch when OOM.** Sample BATCH_SIZE triples per gen if full eval
  doesn't fit. Final eval uses chunked full-stream.
- **Anneal late.** After ~80% of gens, halve mutation scale to fine-tune.
- **Bootstrap when possible.** Two-phase or load prior checkpoint to avoid
  re-learning from scratch.
- **Always log soft_score (mean log-prob) AND top-1 AND top-5.** soft_score
  shows fitness landscape progress even when top-1 is plateau'd.
- **Save checkpoints with ALL learned state**, not just bigram. A_13's
  trigram interaction was lost because save_checkpoint inherited A_8's list.

## A_5..A_29 SCOREBOARD (for reference)

- A_5: bigram ceiling probe → WikiText ceiling = 31.76%, trigram = 63.14%
- A_8: pure bigram + soft fitness → 24% top-1 (76% of ceiling) — substrate proven
- A_12: redundant corpus → 52% top-1 (92% of 57% ceiling)
- A_13: trigram on synthetic → 72% top-1 BUT memorized (A_14 caught it: 37% drop)
- A_16: random pair split → first real generalization test
- **A_18-20: synthetic deterministic peak — 77% top-1 with 0.1% drop**
- A_21-22: real WikiText baseline established (12-15% top-1, fair split)
- A_27: vocab=64 → 22% top-1, 91% of bigram ceiling
- A_28: full 200K words → 20% top-1
- **A_29: character-level → 32% top-1, 68% top-5, 0.4% drop ← current best on real text**

## NEXT MOVES TO TRY (in priority order)

1. **A_30**: cascade + soft fitness on character WikiText. Should break the
   trigram ceiling because cascade gives effective context >>3.
2. **A_31**: hash output mechanism on top of A_30's substrate.
3. **A_32**: per-neuron evolved activations.
4. **A_33+**: combine the wins, push for top-5 above 80%.

## STOPPING CRITERION

A "working text predictor" means:
- Real top-1 above 40% on natural text characters, OR
- Real top-1 above 30% on natural text words (vocab 128+), OR
- Top-5 above 75% on either
- AND held-out drop < 5% (real generalization)

When ANY of those is hit, document the breakthrough and continue pushing.
Don't stop until hitting at least one. Current best (A_29): 32% top-1 / 68%
top-5 character-level — close but not yet at 40% bar.

## A_30 finding: cascade-from-random-init does NOT work

A_30 added the documented protein cascade (delta/momentum/integral) on top
of soft fitness. Result: 27.5% top-1, BELOW A_29's 32.4%. The recurrent
state is impossible to evolve from random init — the dependency between
gen-0 weights and the trajectory is non-monotonic and very noisy.

**Lesson:** stateful recurrent architectures need bootstrapping. Either
load weights from a non-stateful trained model first, or initialize the
cascade transparently (decay→1, momentum→0) so it starts as a no-op and
then evolves to use state.

**Decision:** abandon cascade-from-scratch. Stay with multiplicative n-gram
factorization (A_29's pattern), push to higher n.
