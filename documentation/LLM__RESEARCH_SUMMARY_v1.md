# GENREG Char-Level LM — Research Summary (monolithic phase)

Pure gradient-free, zero-lookup-table, char-level language model
trained on WikiText-103. Everything evolved via tournament selection
with mandatory energy homeostasis. No gradients, no backprop, no
hybrid approaches.

## Headline result

**A_101: 34.00% heldout top-1 / 69.70% top-5 / drop 0.6%**

- +6.34pp above bigram ceiling (27.66%)
- +16.95pp above majority baseline (17.06%)
- 64000 generations, 500 genomes, H=64, D=24
- Pure neural — cascade + recurrent encoder + hash + uniform channels only
- Tight generalization (0.6% train-heldout gap)

## Final scoreboard (monolithic phase)

| Model | Arch | Gens | Heldout top-1 | top-5 | note |
|---|---|---|---|---|---|
| A_83 | baseline cascade | 2500 | 27.02% | 67.52% | bigram ceiling |
| A_89 | +recurrent enc | 2500 | 28.11% | 63.08% | first above ceiling |
| A_89_long | +recurrent | 8000 | 30.51% | 67.08% | breakthrough |
| A_94 | +recurrent | 16000 | 31.28% | 67.92% | |
| A_96 | +recurrent | 32000 | 31.73% | 68.40% | |
| A_98 | +prev_char concat | 16000 | 31.97% | 67.52% | structural win |
| A_99 | +prev_char | 32000 | 33.38% | 68.29% | big jump |
| **A_101** | **+prev_char** | **64000** | **34.00%** | **69.70%** | **best** |
| A_102 | +3-char concat | 64000 | 34.12% | 69.53% | tied A_101 |

## What worked (and why)

**1. Recurrent encoder (A_89): +3pp**
Feeding `[char_t_emb, tanh(prev_hidden)]` to enc instead of just
`char_t_emb`. Gave the encoder true recurrence — current output
depends on previous step's encoded state, not just cascade smearing.
This was THE structural unlock. Prior cascades could only express
bigram-level statistics because enc's view at step t was just f(char_t).

**2. Gen scaling: +0.5-0.7pp per doubling (diminishing)**
| Gens | Δ top-1 |
|---|---|
| 2500 → 8000 | +2.4pp |
| 8000 → 16000 | +0.8pp |
| 16000 → 32000 | +0.4pp |
| 32000 → 64000 | +0.6pp |

Still positive at 64k — not saturated. But wall-clock cost is brutal
(~3h per 64k run).

**3. Prev_char concat (A_98): +1pp at 16k, +1.6pp at 32k**
Adding `[char_t_emb, char_{t-1}_emb, tanh(last)]` gave enc direct
bigram-structure access at input. Beat same-budget runs that relied
on cascade to preserve that info. First architectural lever after
A_89 to actually help.

## What didn't work (and why)

Every complexification — H=128, stacked 2-layer enc, skip connection,
SEQ_LEN=256, POP=1000, maturation gate, bigram-residual fitness,
curriculum, 3-char concat, evolved context-embedding sum with variant
window sizes and per-dim weights — was either **neutral noise** or
**hurt performance** at equivalent compute budgets.

Root cause: **evolutionary selection pressure gets diffused when you
add parameters**. New weights need gens to tune; existing weights
degrade while tuning; interaction effects between modules can't be
resolved by pure tournament selection within budget.

## Observations worth flagging

- **Training-heldout gap stayed 0.2-0.7% across all runs.** Not
  overfitting. We're hitting the ceiling of what tournament selection
  + cascade+recurrent can express, not memorizing training data.

- **Top-5 barely scaled with top-1.** Top-5 went 67.5% → 69.7%
  (+2.2pp) while top-1 went 27.0% → 34.0% (+7pp). Distribution
  quality stagnant — model sharpens top guess without sharpening
  runner-up distribution.

- **Trust channel routing is diagnostic.** When architecture is
  working well, neural trust dominates (0.60+). When struggling,
  hash trust rises (0.30+). Hash acts as a fallback.

- **Energy homeostasis is working.** Starved counts 10-25 per gen
  consistently across all successful runs. Genomes below population
  median lose energy faster, fall below ENERGY_FLOOR=0.15, get
  culled regardless of tournament rank.

## What this means

Monolithic architecture is saturated at ~34% with this evolution
scheme. Gains came from structural unlocks (recurrent, prev_char
concat); further structural additions conflict with selection dynamics.

The entire LM was evolved as one blob — tokenizer, embedding, encoder,
cascade, output head, trust mixer all coupled. Fitness signal diffuses
across too many sub-systems for clean optimization.

## Pivot: component-first

Going forward:
1. Each LLM sub-component is its own GENREG model with its own
   isolated fitness task.
2. Embedding — word2vec-style, predict-context objective.
3. Attention — Q·K retrieval on copy/needle task.
4. Optimizer — meta-learned mutation operator on toy functions.
5. Output head / readout — classification on frozen features.
6. Stack components into full LM only after each hits its own
   performance bar in isolation.

See new research branch starting from /embeddings.
