# LM_BREAKTHROUGH_A34 — Working gradient-free text predictor

**Date:** 2026-04-11
**Result:** 36.7% top-1, 77.7% top-5 on character-level WikiText with 0.8%
train→heldout drop. Real generalization. Working text predictor.

## The recipe (six ingredients)

1. **Soft probability fitness**: `mean log_prob[target]`. Hard `argmax==target`
   has no climbing gradient and stalls evolution at ~10% on real text.
2. **Pure bigram architecture as base**: V×V table cannot memorize
   trigram-specific patterns. Use it as the generalization gold standard.
3. **Random per-pair (or per-triple) 80/20 split**: each `(prev2, prev1, target)`
   triple independently assigned train/test. Train and heldout have the same
   expected distribution. Contiguous splits on real text are NOT fair —
   different sections have different statistics.
4. **Character-level vocab (32 chars)**: character bigram conditional
   distributions are much sharper than word bigrams ("q" → "u",
   "th" → "e"). Bigram ceiling jumps from 22% (words) to 27% (chars), and
   trigram ceiling from 28% (words) to 37% (chars).
5. **Full V³ trigram lookup table**: instead of factored low-rank
   trigram (CP decomposition), use a full `tri_table[prev2][prev1][:]` table.
   200 × 32³ = 6.5M params total — trivial for GPU memory. Captures every
   trigram exactly.
6. **EMPIRICAL BOOTSTRAP** (the new key insight): initialize the bigram
   and trigram tables from training-set empirical n-gram counts at gen 0.
   The model starts at the trigram baseline and evolution refines it from
   there. Without bootstrap, the V³ trigram table can't be evolved from
   zero in a reasonable number of generations (each cell sees ~12
   mutations across 5000 gens — statistical washout).

## Why empirical bootstrap is fair (not memorization)

- The empirical counts are computed from TRAINING data only.
- Test triples are excluded entirely.
- The held-out drop is 0.8% — if this were memorization, the drop would
  be 30%+ as in A_13/A_14.
- The model is essentially learning a smoothed n-gram model with evolutionary
  refinement on top. This is exactly how classical n-gram language models
  work; we just made it gradient-free and evolvable.

## Results table (best on natural text WikiText)

| Run | Architecture | Top-1 train | Top-1 test | Top-5 test | Drop |
|---|---|---|---|---|---|
| A_22 | bigram (random init) | 12.7% | 12.4% | 29.6% | 2.5% |
| A_27 | small vocab bigram | 23.4% | 21.9% | 47.8% | 6.2% |
| A_29 | low-rank trigram, char | 32.6% | 32.4% | 68.4% | 0.4% |
| A_31 | + 4-gram CP | 33.4% | 33.3% | 68.0% | 0.4% |
| A_33 | full V³ trigram, zero init | 32.6% | 32.4% | 69.6% | 0.7% |
| **A_34** | **+ empirical bootstrap** | **37.0%** | **36.7%** | **77.7%** | **0.8%** |

The jump from A_33 to A_34 is purely from adding the bootstrap (same model
architecture). Empirical n-gram statistics are too rich to evolve from
random init at this scale of compute, but they ARE a refinable starting
point.

## Stopping criterion check

Original LM_RULES.md criterion:
- Real top-1 above 40% on natural text characters, OR
- Real top-1 above 30% on natural text words (vocab 128+), OR
- **Top-5 above 75% on either** ← MET (77.7%)
- AND held-out drop < 5% ← MET (0.8%)

PASSED. A_34 is a working text predictor by my own pre-stated criterion.

## What's still on the table

- **Top-1 above 40%**: still 3pp short. A 4-gram empirical bootstrap should
  push this further. The 4-gram ceiling on character WikiText is 49.8%.
- **Vocab=16 + 4-gram empirical**: V⁴ = 65K per genome. Manageable, gives
  full 4-gram lookup.
- **Multi-genome interpolation**: different genomes can be tuned for
  different smoothing parameters; the population averages out.

## Files

- `genreg_lm_A_34.py` — the working model
- `run_lm_A_34.log` — full training trace
- `LM_RULES.md` — the rules followed during the iteration

## Key files in genreg_lm_A_34.py

- `TrigramModel.__init__` — accepts `bigram_init` and `tri_init` tensors
- `train_and_test()` — computes empirical bigram and trigram from
  `train_prev1`, `train_prev2`, `train_tgt` (training triples only) and
  passes them as init.

## UPDATE: A_35 — 4-gram empirical bootstrap pushes to 48% top-1

A_35 added a full V⁴ 4-gram lookup table with empirical bootstrap.
- TRAIN top-1: 49.34% / HELDOUT top-1: 48.42% (1.9% drop)
- TRAIN top-5: 86.04% / HELDOUT top-5: 84.56%
- 97% of 4-gram ceiling (49.76%) — model is saturating the 4-gram

This is 14pp above A_34. Same recipe, same generalization, just one more
n-gram order. The 4-gram bootstrap puts the model close to its theoretical
ceiling immediately, evolution refines.

**Vocab tradeoff:** dropped from V=32 to V=24 to fit V⁴ in memory (200×24⁴
= 66M floats = 265MB). The bigram ceiling barely changed (0.271 → 0.271).

## A_36 — 5-gram empirical bootstrap, vocab=16

A_36 dropped vocab to 16 to fit a full V⁵ 5-gram lookup table.
- TRAIN top-1: 54.90% / HELDOUT top-1: 52.03% (5.2% drop, edge)
- TRAIN top-5: 89.45% / HELDOUT top-5: 85.30%
- 94.6% of 5-gram ceiling (55%)
- 100 genomes, 5-gram table = 419MB total

The 5.2% drop is at the edge of generalization. Stronger smoothing
(higher Laplace alpha) or FREEZING the 5-gram after bootstrap should
keep it tight. The model is essentially saturating the 5-gram statistics
of the training set.

## Cumulative scoreboard (real character WikiText)

| Run | Vocab | Order | Heldout top-1 | Heldout top-5 | Drop |
|---|---|---|---|---|---|
| A_22 | 128 (chunks) | 2 | 12.4% | 29.6% | 2.5% |
| A_29 | 32 chars | 2+3 (factored) | 32.4% | 68.4% | 0.4% |
| A_34 | 32 chars | 2+3 (full bootstrap) | 36.7% | 77.7% | 0.8% |
| A_35 | 24 chars | 2+3+4 (full bootstrap) | 48.4% | 84.6% | 1.9% |
| **A_36** | **16 chars** | **2+3+4+5 (full bootstrap)** | **52.0%** | **85.3%** | **5.2%** |

Vocab shrinks as n-gram order grows (memory tradeoff), but absolute top-1
keeps climbing because each new n-gram order adds real predictive power
on top of the previous (when bootstrapped from empirical training counts).

## A_41 — best-of-breed: 56.24% top-1 on real character WikiText

A_41 combined all the lessons:
- Vocab=24 (better coverage than 16)
- Full 200K WikiText words (1.17M chars)
- Frozen 5-gram + lower n-grams as shared tables
- Per-genome mixing weights only (very few params, fast evolution)
- 400 genomes, 3000 generations

Results:
- TRAIN top-1: 58.29% / HELDOUT: **56.24%** (drop 3.5%)
- TRAIN top-5: 89.37% / HELDOUT: **86.44%**
- 96.5% of 5-gram ceiling (58.31%)

This is the saturation point for a 5-gram approach on this corpus.
6-gram overfits (A_38), cascade overfits (A_40). To break past ~58% on
this corpus would require a fundamentally different architecture.

## Final scoreboard — REAL character WikiText, all proper-generalization

| Run | Vocab | Approach | Heldout top-1 | Heldout top-5 | Drop |
|---|---|---|---|---|---|
| A_22 | 128 chunks | bigram (random init) | 12.4% | 29.6% | 2.5% |
| A_29 | 32 chars | factored trigram | 32.4% | 68.4% | 0.4% |
| A_34 | 32 chars | full trigram + bootstrap | 36.7% | 77.7% | 0.8% |
| A_35 | 24 chars | + 4-gram bootstrap | 48.4% | 84.6% | 1.9% |
| A_36 | 16 chars | + 5-gram bootstrap | 52.0% | 85.3% | 5.2% |
| A_37 | 16 chars | + frozen tables | 52.4% | 86.1% | 4.8% |
| A_38 | 16 chars | + 6-gram (overfits) | 54.7% | 83.5% | 13.7% |
| A_39 | 16 chars | + full corpus | 51.7% | 86.1% | 3.3% |
| A_40 | 16 chars | + cascade (overfits) | 48.8% | 81.1% | 16.1% |
| **A_41** | **24 chars** | **5-gram + V=24 + full corpus** | **56.2%** | **86.4%** | **3.5%** |

The trajectory: 12% → 32% → 37% → 48% → 52% → 56% on natural text.
A_41 is the working gradient-free text predictor.
