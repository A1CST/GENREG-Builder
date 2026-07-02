# Tokenizer Hash Breakthrough — April 8-9, 2026

## Summary

Replaced argmax-over-output-layer token generation with a **hash of the hidden layer sign pattern**. This single change produced **+269% improvement** in tokenizer quality, establishing a new architecture baseline for the GENREG tokenizer.

## The Experiment Matrix

| Exp | Token Method | Hidden | Words/Stream | Unique/106 | Collision | Score |
|-----|-------------|--------|:---:|:---:|:---:|:---:|
| A | argmax(512-out) | 256 | 87 | 36 (34%) | 66% | 0.116 |
| C | argmax(512-out) | 128 | 175 | 34 (32%) | 68% | 0.104 |
| D | argmax^3(512-out) | 256 | 87 | 23 (22%) | 78% | 0.048 |
| E | argmax(512-out) | 128 | 263 | 24 (23%) | 77% | 0.052 |
| **F** | **hash(hidden)** | **256** | **87** | **63 (59%)** | **41%** | **0.357** |
| **G** | **hash(hidden)** | **256** | **175** | **69 (65%)** | **35%** | **0.428** |

Score = (unique/total)^2 * consistency, tested on 106 unseen words.

## What Changed

### Old approach (Experiments A-E):
```
context -> hidden(256, tanh) -> output(512, linear) -> argmax -> token_id
```
The 512-neuron output layer has 131K parameters. Evolution must discover weight patterns where different inputs activate different output neurons. Only ~36-42 distinct output patterns emerged regardless of training pressure.

### New approach (Experiments F-G):
```
context -> hidden(256, tanh) -> hash_proj(9 random projections) -> sign bits -> binary-to-int -> token_id
```
9 evolved projection vectors are dotted with the 256-dim hidden state. Each dot product's sign gives one bit. 9 bits = 512 possible token IDs. Only 2,304 parameters (9x256) replace the 131K output layer.

### Why it works:
1. **Hidden layer has massive diversity** — 256 tanh neurons produce binary sign patterns with 2^256 theoretical states
2. **Argmax is a bottleneck** — it selects ONE output neuron as winner. If 50 neurons have similar weight patterns, only 50 bins get used
3. **Hash projections read ALL hidden neurons** — each token bit depends on the full 256-dim activation, exploiting the hidden layer's natural diversity
4. **7x fewer parameters** — 21K vs 150K per genome, enabling faster evolution

## Key Findings

### 1. Pressure alone doesn't break the argmax ceiling
Experiments C (2x words), D (sharper fitness curve), E (3x words) all performed EQUAL OR WORSE than A. More words with argmax just dilutes the discrimination rate because the architecture can only produce ~50 distinct output patterns.

### 2. The output layer is the representational bottleneck
F (hash, 87 words) beats A (argmax, 87 words) by +208%. Same hidden layer, same evolution, same fitness function — only the token generation changed. The bottleneck was never the hidden layer or the landscape. It was the 512-output argmax.

### 3. Pressure DOES help once the bottleneck is removed
G (hash, 175 words) beats F (hash, 87 words) by +20%. With the argmax bottleneck gone, more words = more discrimination pressure = better tokens. This validates the original pressure hypothesis, just in the right order: fix the bottleneck first, THEN add pressure.

### 4. Experiment B's extinction was a pure energy scaling bug
Collision cost scales quadratically with words-per-bin. Doubling the stream without scaling energy creates a ~4x deficit. Fix: STARTING_ENERGY scales linearly with stream, COLLISION_COST scales with (base_words/actual_words)^2.

### 5. Consistency follows discrimination
Early in training, hash genomes show "spurious diversity" (tok > n_words). By gen 2000+, consistency improves as the model settles on stable hidden representations. The multiplicative fitness naturally selects for genomes that are BOTH diverse AND consistent.

## Experiment B Post-Mortem

Experiment B (174 words, argmax, original energy) died because:
- 175 unique words in ~3 output bins at init
- Collision cost: 0.3 * sum(0..57) * 3 bins = ~1,488 energy
- Starting energy: 500 + 307 regen = 807 total income
- Deficit: -681 per generation -> instant extinction

Fix applied in C: STARTING_ENERGY=1000, COLLISION_COST=0.075. Formula: cost * (87/175)^2 keeps total collision budget constant across stream sizes.

## Architecture Comparison

| | A (argmax) | F/G (hash) |
|---|---|---|
| Parameters/genome | 150,728 | 21,448 |
| Output layer | 256x512 linear | 256x9 projection |
| Token selection | argmax (1 winner) | binary sign pattern |
| Unique tokens @ gen 500 | ~30/87 | ~80/87 |
| Unique tokens @ inference | 36/106 | 63-69/106 |
| Training time | ~50 min | ~33 min |

## Next Steps

1. **Push G further** — Run with 256+ words and scaled energy
2. **Test 1024 output bins** — Use 10 hash bits instead of 9 for 1024-token vocabulary
3. **Build sequence predictor** — Use the tokenizer as a perception layer for next-token prediction
4. **Evolved hash projections** — The hash_proj weights evolve alongside the hidden layer. Test whether fixing them to random (non-evolving) still works — if so, the hidden layer alone carries all the discrimination
5. **Cross-validate on real text** — Test consistency on full paragraphs, not just single words

## Connection to GENREG Thesis

This result both validates and nuances the thesis:

**Validated:** The landscape (multiplicative fitness, energy system, collision cost) is the primary lever — the same landscape works with both argmax and hash architectures.

**Nuanced:** When a **representational bottleneck** exists in the output mechanism, no amount of landscape pressure can push past it. The correct order is: remove bottleneck first, then design landscape pressure. This is consistent with the theory — the landscape must have a basin that leads to the solution, and argmax's limited output diversity prevents that basin from existing above ~50 tokens.
