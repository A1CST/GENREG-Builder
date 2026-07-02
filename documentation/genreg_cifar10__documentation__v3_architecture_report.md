# GENREG V3: Evolved Encoder Architecture

## Overview

V3 adds an **evolved perception layer** between the raw game signals and the neural controller. Instead of the controller seeing noisy, raw board data directly, it sees a filtered representation that evolution itself has shaped. Each genome evolves not just *what to do* with information, but *how to see it*.

This is the key insight: in biological systems, perception and action are co-evolved. An eagle's retina doesn't pass raw photon counts to its brain — it has evolved edge detectors, motion sensors, and contrast filters that pre-process visual data before any decision-making happens. V3 brings this principle to GENREG.

## Architecture

```
                    ┌─────────────────────────┐
                    │     RAW BOARD STATE      │
                    │  16 cells (log2/11) +    │
                    │  6 meta signals = 22 dim │
                    └───────────┬──────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │   EVOLVED ENCODER (new)   │
                    │                           │
                    │  Linear: 22 → 32          │
                    │  736 weights + 32 bias    │
                    │                           │
                    │  Activation: EVOLVED       │
                    │  Selected from catalog     │
                    │  of 8 functions, each      │
                    │  with tunable parameters   │
                    │                           │
                    │  Each genome sees the      │
                    │  board through a different  │
                    │  mathematical lens          │
                    └───────────┬──────────────┘
                                │
                         32 filtered features
                                │
                    ┌───────────▼──────────────┐
                    │   CONTROLLER (existing)   │
                    │                           │
                    │  Linear: 32 → 32 (tanh)   │
                    │  Linear: 32 → 4           │
                    │  argmax → action (0-3)    │
                    └───────────┬──────────────┘
                                │
                         UP / DOWN / LEFT / RIGHT
```

## Parameter Count

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Encoder weights | 736 | 22×32 weight matrix + 32 bias |
| Activation params | 5 | 4 tunable floats + 1 catalog index |
| Controller layer 1 | 1,056 | 32×32 weights + 32 bias |
| Controller layer 2 | 132 | 32×4 weights + 4 bias |
| **Total per genome** | **1,929** | |

For comparison:
- V1 GENREG (no encoder): **868 params**
- V3 GENREG (with encoder): **1,929 params** (2.2× V1)
- DQN V6 (best gradient model): **938,885 params** (487× V3)

## The Evolved Activation Catalog

The encoder's activation function is not fixed — it is **selected by evolution** from a catalog of 8 diverse nonlinearities. Each genome picks the one that best filters signal from noise for its particular strategy. The activation's parameters are also evolved, so two genomes using the same function can tune it differently.

### Catalog

| ID | Name | Formula | What it does |
|----|------|---------|-------------|
| 0 | **tanh_scaled** | `α · tanh(β·x + γ)` | Classic bounded nonlinearity with evolved scale and shift. General-purpose filtering. |
| 1 | **gated_linear** | `scale · x · σ(gate·x)` | Passes strong signals through, suppresses weak ones. Like a noise gate in audio. |
| 2 | **soft_threshold** | `scale · σ(sharpness·(x - threshold))` | Binary detector: outputs ~0 below threshold, ~1 above. Ideal for "is there a high tile here?" type features. |
| 3 | **resonance** | `amp · sin(freq·x + phase)` | Periodic response. Creates multi-modal sensitivity — can detect patterns at specific intervals. |
| 4 | **dual_path** | `w₁·tanh(s₁·x) + w₂·x·σ(s₂·x)` | Two parallel paths blended: bounded nonlinear + gated linear. Most expressive — can approximate any of the others. |
| 5 | **abs_gate** | `scale · (1 - e^(-rate·|x|))` | Magnitude detector. Responds to how far from zero, regardless of sign. "Is anything here?" detector. |
| 6 | **quadratic_relu** | `scale · max(0, x - threshold)²` | Amplifies strong signals quadratically, ignores everything below threshold. Attention-like mechanism. |
| 7 | **identity_plus** | `x + nudge · tanh(bend·x)` | Near pass-through with a learnable nonlinear nudge. Conservative — lets raw signal through but adds structure. |

### Why Evolved Activations Matter

Fixed activation functions (tanh, ReLU) impose a single view on all features. But different board features need different treatment:

- **Empty cells**: Binary — you either have space or you don't → `soft_threshold`
- **Tile values**: Relative ordering matters → `tanh_scaled`
- **Score deltas**: Magnitude matters more than sign → `abs_gate`
- **Corner patterns**: Complex spatial relationships → `dual_path`

By evolving the activation, each genome discovers which mathematical lens makes the board most actionable for its strategy. One genome might evolve a sharp threshold to detect "any tile ≥ 256" while another evolves a smooth resonance to track periodic merge patterns.

## How It Evolves

### What mutates:
1. **Encoder weights** — which spatial patterns to extract from the 22 raw signals
2. **Activation parameters** — how to tune the selected activation (4 floats, bounded)
3. **Activation selection** — which of the 8 functions to use (rare mutation, ~0.5% chance per generation)
4. **Controller weights** — how to act on the encoded features (same as V1)
5. **Protein parameters** — trust modifiers for fitness shaping (same as V1)

### Selection dynamics:
- **Top 20%** of genomes (by trust) survive as elite — kept as-is, including their activation choice
- **Middle 60%** are preserved untouched — their activation and encoder may be suboptimal but they get another chance
- **Bottom 20%** are culled and replaced with mutated clones from the elite — inheriting the elite's activation function but potentially mutating parameters

### Activation convergence:
Over many generations, the population tends to converge toward 1-2 activation functions that work best for the game. However, the mutation mechanism preserves diversity by occasionally switching a genome's activation entirely, allowing the population to re-explore if the current consensus is suboptimal.

## V3 vs V1 — What Changes

| Aspect | V1 (Standard) | V3 (Evolved Encoder) |
|--------|--------------|---------------------|
| Input to controller | 22 raw signals | 32 filtered features |
| Signal processing | None | Learned encoder + evolved activation |
| Parameters | 868 | 1,929 |
| What evolves | Weights + proteins | Weights + proteins + encoder + activation |
| Perception | Fixed, same for all genomes | Evolved, different per genome |
| Spatial awareness | None (flat MLP) | Encoder can learn spatial patterns |
| Noise handling | Controller must learn to ignore noise | Encoder filters noise before controller sees it |

## Why V3 Breaks Plateaus

The V1 architecture hits a ceiling because the controller must simultaneously:
1. Extract useful features from raw signals
2. Filter out noise and irrelevant information
3. Make good action decisions

With only 868 parameters and a single hidden layer, there isn't enough capacity to do all three well. The controller gets stuck doing mediocre feature extraction and mediocre decision-making.

V3 separates these concerns:
- **Encoder** handles feature extraction and noise filtering (736 params)
- **Controller** handles decision-making on clean features (1,188 params)

More importantly, the evolved activation allows each genome to develop a *specialized perception* tuned to its strategy. A genome that plays a corner strategy might evolve a `soft_threshold` activation that outputs "corner tile is big" as a strong binary signal. A genome that plays a snake pattern might evolve `resonance` to detect the alternating tile pattern along a row.

## GPU Implementation

The entire V3 pipeline runs on GPU:
1. All 8 activation functions are implemented as differentiable tensor operations
2. Per-genome activation selection uses `torch.gather` — no branching, no CPU sync
3. Encoder weights, controller weights, and activation parameters are all stored as GPU tensors
4. Evolution (selection, cloning, mutation) operates directly on GPU tensors

The overhead vs V1 is ~30% more compute per generation (one extra matrix multiply + activation), but the encoder's 22→32 transform is small relative to the game simulation cost.

## Running V3

```bash
# CLI
python genreg_2048_cli.py --v3

# App (headless with GPU)
python genreg_2048_app.py --headless_v3

# All existing commands work identically:
#   start, stop, status, save, load, set, extend
```

---
*GENREG V3 — Evolved Perception for Gradient-Free Intelligence*
