# Model Gallery

All models are V=32 char-level, 7.7K evolved params, trained on WikiText-103.

| Model | top-1 | top-5 | Drop | Architecture differences |
|---|---|---|---|---|
| `A_77_best.pkl` | 54.96% | 87.07% | 0.18% | Base: bigram, trigram, 4-gram, 5-gram, residual, hash (7 channels) |
| `A_78_best.pkl` | 54.95% | 87.07% | 0.18% | + slow-integral cascade state (decay ~0.99) |
| `A_79_best.pkl` | **55.15%** | **87.17%** | 0.16% | + ultra-slow cascade (decay ~0.999) + SEQ_LEN=256 |
| `A_80_best.pkl` | 54.98% | 87.06% | 0.15% | Residual basis from 5-gram gaps |

A_79 is marginally best but all four are within ±0.2pp. The multi-scale
cascades and alternative residual bases didn't materially improve quality —
the 5-gram channel dominates trust in every variant.

## When to use which

- **A_77**: default. Smallest, fastest, same quality as others.
- **A_78**: if your inputs benefit from ~100-token topic memory
  (untested; cascade slow-state was usually ignored by evolution).
- **A_79**: best top-1 by a hair. Marginally higher quality for long
  generations.
- **A_80**: similar to A_77 with different neural channel internals.
- **ensemble**: `python ensemble.py --models A_77_best.pkl A_79_best.pkl A_80_best.pkl`
  Averages log-probs. Doesn't obviously improve quality.

## Why we stopped

Architecture changes (A_78/A_79 + residual-basis experiments) plateaued
at 55% top-1. The model is saturated by the n-gram lookup channels.
Further gains likely require larger vocab (V=64 with caps/digits) or
fundamentally different state representations.

**Coherence wins came from inference-time sampling** (nucleus + repetition
penalty), not architecture. See generate.py options.
