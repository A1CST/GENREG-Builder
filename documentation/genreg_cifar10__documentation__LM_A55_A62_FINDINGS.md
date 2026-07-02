# LM A_55 to A_62: Word-Level GENREG LM Findings

## Summary

Experiments A_55 through A_62 pushed the GENREG LM to word-level prediction
on WikiText-103 (9.2M words, V=2000). The full architecture is working:
evolved encoder → protein cascade → trust modulation → controller. All
gradient-free.

## Key Results

| Run | D | Top-1 (heldout) | Neural trust | Override rate | Notes |
|---|---|---|---|---|---|
| A_55 | 48 | 17.59% | 10% | n/a | Baseline, bigram-dominated |
| A_58 | 512 | 17.53% | 35% | n/a | Bigger embedding, controller more trusted |
| A_59 | 512 | 17.53% | 8.5% | n/a | Added context window, made things worse |
| A_60 | 512 | 17.53% | 16% | 0.0% | Energy rule, controller never overrides |
| A_61 | 512 | 17.54% | 6% | 0.1% | Adaptive mutation, minimal overrides |
| A_62 | 512 | 14.43% | 32% | 52% | Balanced energy, overrides too much |

Bigram ceiling: 18.48%. All models plateau at or below this.

## What Works

1. **Cascade trust modulation**: The cascade reliably learns to weight
   candidates. Trust distributions are meaningful and stable.
2. **Evolved encoder activations**: The population consistently selects
   `resonance` (3) or `quadratic_relu` (6) — both create rich nonlinearities.
3. **SVD predictive embeddings**: D=512 from transition matrix SVD gives the
   controller 35% trust (vs 10% with D=48 co-occurrence embeddings).
4. **Adaptive mutation**: Fitness-reactive rates (bad genomes explore hard)
   keep the population diverse.
5. **Soft probability fitness**: Mean log-prob remains the essential gradient
   channel for evolution.

## What Doesn't Work (Yet)

1. **Controller beating bigram at specific positions**: With 66K params and
   5-10K gens of evolution, the controller can't predict specific next words
   better than bigram lookup. It's trusted (35%) but produces similar-quality
   predictions.
2. **Override energy rules**: The energy rule correctly identified that
   overrides have negative expected value. With +20/-0.1 balanced energy,
   the model overrides 52% of the time but only 4.6% are correct — not
   enough to beat bigram.
3. **Context window concatenation (A_59)**: Adding raw recent word embeddings
   to the controller input made the search space harder, reducing neural
   trust from 35% to 8.5%.

## The Plateau: Why 18.5%?

The bigram ceiling at V=2000 is 18.48%. The model sits at 17.5% because:
- Bigram handles ~85% of trust and achieves close to its ceiling
- The remaining 15-35% trust on the neural controller produces similar-quality
  predictions (not better)
- Override attempts succeed only 4.6% of the time (below bigram's 18.5%)

To break through, the controller needs to achieve >18.5% accuracy on
override positions. This requires learning trigram+ patterns, which the
66K-param controller hasn't managed in 10K gens.

## Next Directions to Try

1. **Oracle override analysis**: Compute what % a trigram model gets right
   on positions where bigram fails. This tells us the achievable ceiling.
2. **Remove exploration bonus**: Keep +20/-0.1 energy but no +0.3 bonus.
   The model should learn to override selectively, not everywhere.
3. **Bootstrap from A_58**: Start with the controller that has 35% trust
   and 17.5% accuracy, then apply energy rule to push it toward overrides.
4. **Longer evolution on smaller vocab**: V=500 where bigram ceiling is
   higher and trigram information is denser.
5. **Structured controller**: Instead of linear H→D, use an MLP with
   nonlinearity (tanh hidden layer). More expressiveness per param.

## A_63-A_64: BREAKTHROUGH — Broke Bigram Ceiling at Word Level

### A_63 (reranking): same 17.53% — reranking bigram's top-20 didn't help.

### A_64 (SPARSE TRIGRAM): **19.40% heldout — ABOVE bigram ceiling (18.48%)**

The sparse trigram approach worked:
- Precompute 62,721 (prev2, prev1) → distribution entries (0.7 MB)
- Available for 82.5% of positions
- Cascade trust: 39% bigram, 34% trigram, 16% neural
- Perfect generalization: 0.4% drop

This is the A_52 pattern (cascade gates trigram reliability) applied to V=2000.
The cascade learned WHEN to trust the trigram (when the pair is well-attested)
and when to fall back (when it's not).

Key: the trigram oracle showed 48.8% was theoretically achievable with perfect
trigram overrides. A_64 at 19.4% is just the beginning — the cascade is
conservatively trusting trigram at 34%. With longer training or larger
population, the trigram trust could increase toward its ceiling.

Oracle analysis:
- Bigram ceiling: 18.48%
- Trigram ceiling: 29.54% (on full stream)
- Trigram override ceiling: 37.2% (on bigram-failure positions)
- Maximum with perfect override: 48.8%
- A_64 achieved: 19.40% (first step past bigram)

## A_65-A_66: Pushing the sparse n-gram approach

### A_65 (longer training, 10K gens): 20.04% heldout, 40.51% top-5
- Trust: bg=26%, trigram=37%, neural=14%
- Trigram became dominant trust source

### A_66 (+ sparse 4-gram): 19.93% heldout, 40.15% top-5
- Trust: bg=36%, trigram=21%, 4-gram=25%, neural=12%
- 4-gram IS being used but model hasn't fully learned when
- The cascade now distributes trust across 5 levels

## Best word-level results (pure gradient-free, WikiText-103, V=2000)

| Model | Heldout top-1 | Top-5 | Above bigram? |
|---|---|---|---|
| A_55 (bigram) | 17.59% | 34.73% | no |
| A_64 (+ trigram) | 19.40% | 39.14% | YES +0.9pp |
| **A_65 (longer)** | **20.04%** | **40.51%** | **YES +1.6pp** |
| A_66 (+ 4-gram) | 19.93% | 40.15% | YES +1.5pp |

The trigram provides the clearest lift. 4-gram adds coverage but training
hasn't converged to use it optimally yet.

## Architecture proven at word level (V=2000)

The full GENREG LM pipeline works:
1. SVD predictive embeddings (D=512, shared)
2. Frozen bigram table (V², shared)
3. Sparse trigram + 4-gram tables (shared, ~120K entries total)
4. Evolved encoder with resonance/abs_gate activation
5. Protein cascade (delta/momentum/integral)
6. Cascade → 4-5 dim trust signal → dynamic n-gram order selection
7. Soft probability fitness driving evolution

All evolved, zero gradients. The cascade learned to trust higher-order
n-grams when they're available and reliable.
