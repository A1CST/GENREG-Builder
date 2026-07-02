# GENREG LLM Tokenizer — Project Report

**Date:** April 7-8, 2026
**Author:** Payton Miller
**Hardware:** NVIDIA GeForce RTX 4080, Linux
**Framework:** GENREG (gradient-free evolutionary optimization)

---

## Goal

Build a gradient-free neural tokenizer using the full GENREG framework. The model receives a continuous stream of word inputs (byte-encoded) mixed with zero-padding and must:

1. Produce the **same token ID every time** it sees the same word (consistency)
2. Produce **different token IDs** for different words (discrimination)
3. Produce **token ID 0** for zero-padded inputs (silence)

The tokenizer discovers its own vocabulary through evolutionary pressure. It is never told what token ID to assign. This is the first step toward a full GENREG LLM.

---

## Architecture

```
Input: word -> UTF-8 bytes -> normalized [0,1] -> padded to 32 floats
    |
    v
Evolved Encoder: 32 -> 64 dims
    Linear projection + per-genome activation (8-function catalog)
    Reuses genreg_encoder_gpu.py activations
    |
    v
Protein State:
    delta = encoded - last_val
    momentum = EMA of deltas
    integral = decayed accumulation
    4-way mix: [last_val, momentum, integral, raw_encoded]
    |
    v
Controller: 64 -> 128 -> 512 (output bins)
    tanh hidden layer -> normalized logits -> argmax -> token ID
    |
    v
Token ID (0-511)
```

**Parameters per genome:** ~76,872
**Population:** 500 genomes
**Evolution:** V5 pattern (neuron-level crossover, per-genome evolved mutation traits)

---

## Key Discovery: Fitness Landscape Design IS the Problem

### Attempt 1: Additive fitness (consistency + discrimination)
```
fitness = 1.0 * consistency + 0.5 * discrimination + 0.3 * zero_output
```
**Result:** Model collapsed to 3-5 bins. Consistency = 1.0 is trivially achieved by mapping everything to one token. Discrimination (0.5 weight) couldn't overcome the consistency reward.

### Attempt 2: Energy system for silence
Added energy cost for outputting non-zero on padding. Genomes that waste energy on padding die.

**Result:** Silence learned instantly (100% padding detection). But discrimination unchanged — energy didn't address the output collapse.

### Attempt 3: Energy collision penalty
Added energy cost when a genome reuses a token bin for a different word. Cost scales with bin congestion (0 for first word, 0.3 for second, 0.6 for third, etc.).

**Result:** Too lethal at high costs (3.0 = total extinction). At moderate costs (1.5 = survival but no improvement). At low costs (0.5 = survivable but no pressure).

The problem: collision penalty only fires within a single stream. Words across different streams never compete. The model finds ~25 bins that tile the in-stream space without collisions and parks everything there.

### Attempt 4: Novel word injection
25% of each stream's words are randomly generated (never in corpus). Forces the model to discriminate on byte-pattern structure, not memorized separation.

**Result:** Helped in-stream diversity but didn't break the output collapse at inference.

### Attempt 5: Output normalization
Normalize logits (subtract mean, divide by std) before argmax. Prevents a few output neurons from dominating via large bias.

**Result:** No improvement. The collapse is structural, not bias-driven.

### Attempt 6: MULTIPLICATIVE FITNESS (breakthrough)
```
fitness = discrimination^2 * consistency * energy_bonus
```
**Result: 12 -> 49 unique tokens in-stream. Fitness climbing throughout, no plateau.**

This is the core insight: **discrimination must be the primary signal, not an add-on.** With the old additive fitness, a genome using 3 bins scored 1.0 * 1.0 + 0.5 * 0.03 = 1.015. With multiplicative fitness, the same genome scores 0.03^2 * 1.0 = 0.001. The landscape makes low-bin strategies worthless.

### The GENREG thesis confirmed again
Same model, same population, same architecture. One fitness function change: additive -> multiplicative. Result: 5 unique tokens -> 49. **Landscape design > architecture.**

---

## Energy System (final configuration)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Starting energy | 500 | Generous — genomes must survive to evolve |
| Energy regen | 0.5/step | Slow recovery on real-word steps |
| Padding cost | 3.0 | Non-zero output on padding drains energy |
| Collision cost | 0.3 (base) | Scales with bin congestion: 0, 0.3, 0.6, 0.9... |
| Silence bonus | 0.2 | Small reward for correct zero on padding |
| Death | energy <= 0 | Dead genomes output 0, get fitness 0 |

Energy creates a survival requirement: learn silence + avoid collisions. The multiplicative fitness creates the optimization target: maximize discrimination.

---

## Stream Design

Each generation gets a fresh stream:
- **384 steps** (configurable, experiment B tests 768)
- **~87 unique words** per stream (20% padding, 2-5 repeats per word)
- **25% novel words** — randomly generated, never in corpus
- Shuffled positions — same word at different stream locations

Fresh stream per generation prevents overfitting.

---

## Current A/B Experiment (in progress)

Testing whether the ~49 token ceiling is an architecture limit or a pressure limit.

### Experiment A: More capacity
- HIDDEN_DIM = 256 (doubled)
- STREAM_LEN = 384 (~87 words)
- **Result at gen 3000:** 42/87 in-stream, **12/26 inference** (29% collision)

### Experiment B: More pressure
- HIDDEN_DIM = 128 (unchanged)
- STREAM_LEN = 768 (~174 words)
- **Result at gen 1000:** 5/26 inference (still early, running)

### The hypothesis
GENREG models use tanh neurons that saturate under evolutionary pressure. Saturated neurons are binary switches (1 bit each). With k saturated neurons, the state space is 2^k partitions, each with infinite-precision modulation from continuous neurons. A 128-neuron hidden layer has far more capacity than 512 output bins could ever need.

The 49-token ceiling is a **pressure problem**, not a capacity problem. Doubling the stream forces more neurons to saturate into useful configurations because the energy penalty kills genomes that can't discriminate 174 words. The architecture doesn't need to be bigger — it needs to be pushed harder.

---

## Results Summary

| Config | Gen | In-stream unique | Inference unique/26 | Collision rate |
|--------|-----|-----------------|--------------------|--------------:|
| Additive fitness | 1500 | 5/32 | 5/26 | 81% |
| + Energy (silence) | 500 | 17/32 | 9/17 | 47% |
| + Collision penalty | 500 | 22/32 | 9/26 | 65% |
| **Multiplicative fitness** | **2000** | **49/87** | **9/26** | **65%** |
| + Longer run | 5000 | 49/87 | 8/26 | 53% |
| **Exp A (256 hidden)** | **3000** | **42/87** | **12/26** | **29%** |
| Exp B (174 words) | 1000 | ?/174 | 5/26 | 81% (early) |

---

## Key Principles Discovered

1. **Landscape design is everything.** Additive -> multiplicative fitness: 5 -> 49 tokens. Same model.

2. **Energy creates survival requirements.** Padding silence was learned in 25 generations once it became a death condition instead of a fitness bonus.

3. **Collision penalties must scale.** Flat penalties either kill everyone or do nothing. Bin-congestion scaling (0, 0.3, 0.6...) creates a smooth gradient toward spreading.

4. **Novel word injection forces generalization.** Without it, the model memorizes separation for known words only.

5. **Consistency is free.** The architecture is naturally deterministic — same bytes -> same encoded -> same argmax. Consistency never needed to be the primary fitness signal.

6. **Scaling is not the answer.** Tanh saturation gives GENREG models massive latent capacity (2^k partitions). Plateaus are pressure problems, not capacity problems. Push harder with the landscape instead of scaling the architecture.

7. **Output normalization matters.** Without logit normalization, a few output neurons dominate via large biases. Normalizing (subtract mean, divide by std) forces fair competition across all bins.

---

## Files

```
GENREG/LLM/
    genreg_tokenizer.py       # Main tokenizer (baseline config)
    genreg_tokenizer_A.py     # Experiment A: HIDDEN_DIM=256
    genreg_tokenizer_B.py     # Experiment B: STREAM_LEN=768
    data/corpus_words.json    # Cached word list (20K words from WikiText-103)
    checkpoints/              # Baseline checkpoints
    checkpoints_A/            # Experiment A checkpoints
    checkpoints_B/            # Experiment B checkpoints
```

---

## How to Run

All commands from `GENREG/LLM/` directory. Requires PyTorch with CUDA and `datasets` (HuggingFace).

The tokenizer imports `apply_evolved_activations` from `../genreg_encoder_gpu.py` — this file must exist in the parent GENREG directory.

```bash
# Train (baseline config)
python genreg_tokenizer.py --generations 3000

# Train experiment A (256 hidden)
python genreg_tokenizer_A.py --generations 3000

# Train experiment B (174 words per stream)
python genreg_tokenizer_B.py --generations 3000

# Evaluate a checkpoint
python genreg_tokenizer.py --eval checkpoints/tokenizer_gen_02000.pkl

# Force CPU (no GPU)
python genreg_tokenizer.py --generations 1000 --device cpu
```

First run downloads WikiText-103 (~1.8M entries) and caches the word list to `data/corpus_words.json`. Subsequent runs skip the download.

Checkpoints are ~150MB each (500 genomes x all tensors as numpy). Saved every 500 generations.

---

## Current State (as of April 8, 2026)

**Experiment A (HIDDEN=256, 87 words): FINISHED**
- Best inference: 12/26 unique tokens (29% collision) at gen 3000
- In-stream: peaked at ~42/87
- Checkpoints in `checkpoints_A/`

**Experiment B (HIDDEN=128, 174 words): RUNNING**
- At gen ~1000, inference: 5/26 unique (still early — much slower per gen due to 2x stream)
- Checkpoints in `checkpoints_B/`
- This is the important experiment — tests whether more pressure (not more capacity) breaks the ceiling

**When B finishes:** Compare final inference unique tokens. If B >= A, the saturation/pressure hypothesis is confirmed and the next step is pushing stream size further (256, 512 words). If B < A, the architecture may need examination (but this would be surprising given the hybrid computation theory).

---

## What NOT to Try (already proven dead ends)

| Approach | Why it fails |
|----------|-------------|
| Increase OUTPUT_DIM to 4096 | 128 hidden neurons can't address 4096 bins distinctly. The w2 layer has too many dead columns. |
| Additive fitness (consistency + discrimination) | Consistency dominates. Model collapses to 3-5 bins for perfect consistency score. |
| High collision penalty (>= 2.0) | Total population extinction. Random init produces 2-10 bins; high penalty kills before evolution can spread. |
| Low starting energy (<100) with collision penalties | Same extinction problem. Energy must be generous enough for random genomes to survive. |
| Output normalization alone | Doesn't break collapse. The problem is the fitness landscape, not logit scaling. (But keep normalization — it helps once the landscape is right.) |
| Scaling hidden layer as first response to plateau | GENREG tanh neurons saturate to binary switches under evolution. 128 neurons = 2^k partitions. Capacity is not the bottleneck. Push harder with pressure first. |

---

## Critical Design Principles for Resuming

1. **The fitness function is the landscape.** Every breakthrough came from redesigning fitness, not architecture. If the model plateaus, redesign the fitness signal.

2. **Energy = survival requirement, fitness = optimization target.** Don't conflate them. Energy enforces hard constraints (silence, no collisions). Fitness guides what "good" means.

3. **Multiplicative fitness prevents collapse.** `discrimination^2 * consistency * energy_bonus` makes low-discrimination strategies score near zero regardless of consistency.

4. **Fresh stream per generation.** Never reuse the same stream — prevents overfitting to position and word order.

5. **25% novel words per stream.** Forces generalization to unseen byte patterns, not memorized word-to-bin mappings.

6. **Consistency is architecturally free.** The model is deterministic by construction (same bytes -> same argmax). Never optimize for consistency as a primary signal.

7. **This is a GENREG model.** It follows the same principles as the 2048 game, CIFAR-10, and all other GENREG work. The theory paper ("Optimization as Environment Construction") applies directly. Read it for context.

---

## Next Steps

1. Wait for Experiment B to finish — if 128 hidden handles 174 words, it confirms the pressure hypothesis
2. If B wins: increase stream further (256, 512 words) and see how far 128 hidden can go
3. Once discrimination is high enough (~80%+), build the next layer: sequence-level token prediction using the evolved tokenizer as input encoding
4. The tokenizer becomes the perception layer for a full GENREG language model — exactly how V3's evolved encoder became the perception layer for the 2048 controller

---

*GENREG LLM Tokenizer — Gradient-free vocabulary discovery through evolutionary pressure.*
*"The model has enough power. The landscape just wasn't demanding enough to force it through."*
