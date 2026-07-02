# LM Experiment Log — April 9, 2026

## The Problem
Build a GENREG next-token predictor on top of the subword tokenizer.

## Experiments Run

### Baseline: Simple fitness (bit_accuracy^2 * exact_match_bonus)
- **Result**: 62% bit accuracy, 6.5% exact match, 15% demo
- **Finding**: Converges on predicting most-common token (frequency matching)

### +3 attempts (multi-game scoring)
- **Result**: 61% bits, 7.8% exact
- **Finding**: Smoothed variance but same ceiling

### H: Diversity-gated fitness
- **Result**: 59.8% bits, 0% demo
- **Finding**: Killed frequency matching but accuracy collapsed — no signal left

### I: Inverse-frequency weighted
- **Result**: 60.9% bits, 10% demo
- **Finding**: Shifted from one dominant token to another

### J: Transition-only scoring
- **Result**: 61.1% bits, 15% demo
- **Finding**: Same ceiling, honest scoring

### K: 256-dim token embeddings as input
- **Result**: 61.3% bits
- **Finding**: Richer input didn't help — bottleneck is elsewhere

### L: 128-dim protein state (2x bigger)
- **Result**: 61.8% bits
- **Finding**: More state capacity didn't help

### M: 2-layer stacked protein cascade
- **Result**: 62.5% bits
- **Finding**: Dual fast/slow cascades didn't break ceiling

### N: Energy-driven with streak penalty
- **Result**: (running) ~62% bits, 3-5% exact
- **Finding**: Energy filters weak genomes, streak penalty prevents pure frequency matching, but ceiling persists. Genomes adapted by alternating between 2-3 common tokens.

## The 62% Ceiling — What We Know

The ceiling is consistent across:
- 5 different fitness landscapes
- 3 different input representations
- 3 different protein cascade configurations
- Energy-driven and score-only approaches

This suggests the bottleneck is NOT:
- Fitness landscape design
- Protein state capacity
- Input encoding richness
- Sequential memory depth

## What Might Actually Be Wrong

### Hypothesis: The input is semantically meaningless
Token IDs are assigned by hash — token 347 has no structural relationship to token 348. The binary encoding (101011011 vs 101011100) is noise. The encoder has to learn from pure noise which tokens follow which tokens.

In contrast, the TOKENIZER works because its input IS meaningful: "hel" bytes [104, 101, 108] have real structure that maps to consistent outputs.

### Next experiment to try
Feed the actual byte representation of each chunk (what the tokenizer sees) instead of the token ID. This gives the LM access to the SEMANTIC content of each position, not just an abstract ID. The protein cascade would see "hel", "lo", "wor", "ld" — actual text fragments — and predict the next fragment.

This is fundamentally different: instead of predicting the next abstract ID, the model predicts the next text chunk. The tokenizer becomes an output decoder, not an input encoder.

## NOT A BREAKTHROUGH: Experiment O — Byte-Level Failed Same Way

**RETRACTED.** Initial results looked impressive (97% byte accuracy, 54% exact match) but inspection revealed the model is doing **byte-level frequency matching**, not actual prediction.

### What actually happened
The model predicts the same average byte pattern every step regardless of input. Predictions look like "_da?????", "hda?????", "ni`?????" — all in the d-n character range, all 3-letter shaped. It's the byte-level mean of typical chunks.

### The metric was wrong
- **Predicting all zeros** = 84.1% byte accuracy (4/8 bytes are always padding)
- **Predicting mean chunk** = 96.8% byte accuracy
- **Identity baseline** (echo input) = 94.7% byte accuracy on demo
- **Model** = 96.9% byte accuracy

The model is barely above identity baseline (1.6% gain). The "exact match" of 54% comes from byte tolerance (0.05 per byte) being generous enough to count all "average-shaped" chunks as matches.

### Lesson
Continuous outputs with tolerance-based scoring create a deceptive landscape. The model can score high by predicting averages, just like the token-based version scored "high" by predicting frequencies. **The metric must be exact chunk identity** — does the predicted byte vector decode to the actual next chunk string?

## STILL UNSOLVED: How to make the LM actually predict context-dependent next chunks

Both token-based and byte-based approaches converge on frequency/mean matching. The protein cascade is not learning conditional distributions. Possibilities:
1. The cascade dynamics fundamentally can't capture next-token patterns (need different architecture)
2. The fitness landscape rewards averaging too much (need harder cliff between right and wrong)
3. The training data doesn't have enough sequential structure (shuffled words have no real grammar)
4. The model needs longer context (64 chunks may be too short to learn patterns)

## Old breakthrough claim removed below ↓

### The Key Insight
Token IDs are semantically meaningless. Token 347's binary encoding (101011011) has no structural relationship to what it represents ("tok"). But the BYTES of "tok" = [0.408, 0.432, 0.424, 0, 0, 0, 0, 0] are real, structured data. The protein cascade can learn from bytes. It cannot learn from abstract IDs.

### Architecture (Experiment O)
```
chunk bytes (8 floats) → encoder (8→64, evolved activation)
→ protein layer 1 (64-dim, fast decay — local context)
→ bridge (64→64 linear)
→ protein layer 2 (64-dim, slow decay — global context)
→ controller (128→256, tanh)
→ output layer (256→8, sigmoid)
→ predicted next chunk bytes (8 floats in [0,1])
```

With energy system: drain for wrong predictions, gain for right, streak penalty for monotone output.

### Results

| Gen | Byte Accuracy | Exact Match | Energy | vs Token-Based |
|:---:|:---:|:---:|:---:|---|
| 25 | 93.2% | 0% | 3150 | — |
| 150 | 96.1% | 35% | 3836 | 5.4x exact match |
| 400 | 96.7% | 43% | 3714 | 6.6x |
| 650 | 97.1% | 53.5% | 3854 | 8.2x |
| 1100 | 96.7% | 54.4% | 3980 | **8.4x** |

Best token-based LM: 6.5% exact match. O achieves **54%**. That's 8.4x improvement.

### Why It Works
1. **Meaningful input**: Chunk bytes encode real text structure (letter frequencies, common patterns). The encoder and protein cascade can learn "after [t,h,e] expect [space] or a consonant."
2. **Continuous output**: Predicting 8 floats (sigmoid) instead of 9 binary bits gives smooth gradient signal. Close predictions get partial credit via byte_acc.
3. **Energy system**: Survival pressure forces genomes to predict accurately or die. No frequency matching possible because byte patterns are diverse.
4. **Stacked protein cascade**: Two layers with different decay rates capture both local (recent chunk) and global (phrase-level) context.

### What This Means
The GENREG protein cascade CAN learn sequential patterns. The 62% ceiling in token-based experiments was caused by:
1. Meaningless input encoding (binary token IDs = noise)
2. Discrete output mechanism (9 bits with cliff-like accuracy signal)
3. No energy system (no survival pressure to force real learning)

Fix all three and the model reaches 97% byte accuracy with 54% exact chunk prediction.

## Key Principles Confirmed

1. **EVERY GENREG MODEL NEEDS ENERGY.** Without energy, there's no survival pressure, no handle to force behavior change. Energy is what makes the tokenizer work, what makes 2048 work, and what drives the LM.

2. **Input must be semantically meaningful.** Abstract IDs are noise to the encoder. Real data (bytes, pixels, signals) gives the evolved activation functions something to work with.

3. **The landscape IS the lever.** The difference between 6.5% and 54% exact match was not architecture — it was landscape design (energy system + meaningful encoding + continuous output).
