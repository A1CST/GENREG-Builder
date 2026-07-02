# Overnight log — pure-neural GENREG LM

Starting state before sleep:
- Breakthrough: A_89_long hit **30.51% heldout top-1** (bigram ceiling = 27.66%)
- Architecture: pure-neural char LM, recurrent encoder [char_emb, tanh(prev_hidden)] → enc, 4-scale cascade, energy homeostasis, maturation-gate-off
- Current run: A_90 (H=128, 8000 gens) — testing if capacity is the bottleneck

Scoreboard so far:
| Model | Heldout top-1 | top-5 | vs bigram ceiling | note |
|---|---|---|---|---|
| A_83 pure-neural H=64 | 27.02% | 67.52% | -0.64pp | bigram ceiling |
| A_84 stratified | 27.01% | 67.32% | -0.65pp | noise |
| A_85 residual-only | 21.80% | 65.38% | -5.9pp | collapsed |
| A_86 curriculum | 25.19% | 66.26% | -2.5pp | worse |
| A_87 two-layer head | 27.08% | 67.43% | -0.58pp | noise |
| A_88 maturation gate | 26.88% | 67.06% | -0.78pp | noise |
| A_89 recurrent enc (2500 gen) | 28.11% | 63.08% | +0.44pp | first to cross |
| **A_89_long recurrent (8000 gen)** | **30.51%** | **67.08%** | **+2.84pp** | structural breakthrough |

Plan for overnight (autonomous):
- Monitor A_90 (H=128 + recurrent, 8000 gens) to completion
- If A_90 improves: scale up further (H=192)
- If A_90 plateaus: try A_91 = recurrent + maturation gate combined (A_88 × A_89)
- After A_91: stacked recurrent layers (2-deep enc) OR longer SEQ_LEN
- Keep going. Each iteration appended below.

---

## A_90 done (H=128 + recurrent)
- Heldout: **29.10% top-1 / 64.52% top-5** / drop 0.3% / +1.43pp vs bigram
- **Worse** than A_89_long (30.51%). Doubling width didn't help — too many params for evolution to tune in 8000 gens, or neurons-per-activation-id spread too thin.
- Conclusion: H=64 was not the capacity bottleneck.

## A_91 done (recurrent + maturation)
- Heldout: **30.45% top-1 / 66.42% top-5** / drop 0.4% / +2.79pp vs bigram
- Statistically identical to A_89_long (30.51%). Maturation gate adds nothing on top of recurrent encoder.

## A_92 done (stacked 2-layer recurrent)
- Heldout: **28.44% top-1 / 63.69% top-5** / drop -0.4% / +0.78pp vs bigram
- Worse than A_89_long. Doubled params under same gen budget = under-tuned.
- Pattern now clear: more capacity (H=128 or 2-layer) under 8000 gens HURTS vs single-layer H=64.

## A_93 done (SEQ_LEN=256)
- Heldout: **29.83% top-1 / 66.37% top-5** / drop 0.3% / +2.17pp vs bigram
- Worse than A_89_long. Window length isn't the lever. Cascades saturated at 128.

## Locked pattern
**A_89_long (30.51%) wins against every complexification in same gen budget:**
- H=128: 29.10% (worse)
- 2-layer stacked: 28.44% (worse)
- maturation gate: 30.45% (noise)
- SEQ_LEN=256: 29.83% (worse)
Single-layer recurrent H=64 + 8000 gens is the sweet spot so far.

## A_94 done (16k gens) — NEW BEST
- Heldout: **31.28% top-1 / 67.92% top-5** / drop 0.3% / **+3.62pp vs bigram**
- Beat A_89_long's 30.51% by +0.77pp. More gens still help; architecture hadn't saturated.
- Training peaks 31.5% at gen 15500.

## A_95 done (POP=1000, 16k gens)
- Heldout: **31.30% top-1 / 68.01% top-5** — +0.02pp over A_94 (noise)
- POP=1000 didn't unlock a better basin. Explores same plateau wider.
- Training peaks 32.7% (vs A_94's 31.5%) but heldout ~same → exploration without transfer.

## Gen-scaling pattern
- 2500: 28.11%
- 8000: 30.51% (+2.40pp)
- 16000: 31.28% (+0.77pp)
- 16000 POP=1000: 31.30% (noise)
Diminishing returns but still positive. Architecture isn't saturated.

## A_96 done (32k gens) — NEW BEST
- Heldout: **31.73% top-1 / 68.40% top-5** / drop 0.5% / **+4.07pp vs bigram**
- Gen scaling: 8k→30.51, 16k→31.28, 32k→31.73. Diminishing but still positive.
- Training peaks hit 33.71% at gen 31000.

## A_97 done (skip connection)
- Heldout: **30.84% top-1 / 67.06% top-5** — worse than A_94 (31.28%)
- Extra W_skip params under-tuned at 16k gens. Same capacity lesson.

## A_98 done (prev_char concat, 16k gens) — NEW BEST
- Heldout: **31.97% top-1 / 67.52% top-5** / drop 0.7% / **+4.31pp vs bigram**
- Beat A_96 (31.73% at 32k gens) at HALF the gen budget.
- First architecture change since recurrent encoder (A_89) to actually improve over pure gen-scaling.
- Key insight: giving enc direct structural access to previous char > letting cascade smear it into hidden state.

## A_99 done (prev_char × 32k gens) — NEW BEST BY WIDE MARGIN
- Heldout: **33.38% top-1 / 68.29% top-5** / drop 0.2% / **+5.72pp vs bigram**
- Jump of +1.41pp over A_98 and +1.65pp over A_96. Levers composed strongly.
- Neural channel trust dominated: 0.66 (vs typical 0.50-0.55).
- Tight generalization: 0.2% train-held drop.

## A_100 done (3-char concat × 32k)
- Heldout: **33.11% top-1 / 68.50% top-5** / drop 0.2% / +5.45pp vs bigram
- Slightly worse than A_99 (33.38%). Added input dim under-tuned at same 32k budget.
- Trust redistributed: neural 0.49 (down from 0.66), hash 0.30 (up from 0.12).
- Training still climbing at gen 31000 (34.18% peak).

## A_101 done (64k gens on A_99 base) — NEW BEST
- Heldout: **34.00% top-1 / 69.70% top-5** / drop 0.6% / **+6.34pp vs bigram**
- Top-5 broke 70% for first time.
- +0.62pp from doubling gens. Scaling still alive, not saturated.
- Trust shift: hash climbed to 0.39, neural dropped to 0.48 (similar pattern to A_100).

## A_102 done (3-char × 64k gens)
- Heldout: **34.12% top-1 / 69.53% top-5** / drop 0.6% / +6.46pp vs bigram
- Essentially tied with A_101 (34.00%, 69.70%). 3-char caught up with enough gens but no real win.
- Training peaks 35.88% at gen 62000 — still climbing.

## A_103 done (evolved ctx w=8, 16k gens)
- Heldout: **30.45% top-1 / 67.38% top-5** / drop 0.7%
- Worse than A_98 (31.97% prev_char @ 16k). Evolved ctx needs more gens to tune.
- Neural trust locked on at 0.62.
- Note: originally started at 64k; killed for fast iteration and restarted at 16k per user request.

## Queue running (16k gens each, sequential)
- A_104 CTX_WIN=16
- A_105 per-dim ctx weights
- A_106 CTX_WIN=32

## Progression summary (pure-neural, zero lookups, gradient-free)
| Model | Arch | Gens | Heldout top-1 | top-5 |
|---|---|---|---|---|
| A_83 | baseline | 2.5k | 27.02% | 67.52% |
| A_89 | +recurrent | 2.5k | 28.11% | 63.08% |
| A_89_long | +recurrent | 8k | 30.51% | 67.08% |
| A_94 | +recurrent | 16k | 31.28% | 67.92% |
| A_96 | +recurrent | 32k | 31.73% | 68.40% |
| A_98 | +prev_char | 16k | 31.97% | 67.52% |
| A_99 | +prev_char | 32k | 33.38% | 68.29% |
| A_100 | +3-char | 32k | 33.11% | 68.50% |
| **A_101** | **+prev_char** | **64k** | **34.00%** | **69.70%** |

## Final scoreboard so far
| Model | Heldout top-1 | top-5 | vs bigram |
|---|---|---|---|
| A_83 baseline (2500 gen) | 27.02% | 67.52% | -0.64 |
| A_89_long recurrent (8k) | 30.51% | 67.08% | +2.84 |
| A_94 recurrent (16k) | 31.28% | 67.92% | +3.62 |
| A_96 recurrent (32k) | 31.73% | 68.40% | +4.07 |
| A_98 +prev_char (16k) | 31.97% | 67.52% | +4.31 |
| **A_99 +prev_char (32k)** | **33.38%** | **68.29%** | **+5.72** |
