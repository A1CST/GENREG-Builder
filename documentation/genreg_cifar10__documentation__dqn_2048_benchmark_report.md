# DQN 2048 Benchmark Report

Generated: 2026-03-29 19:38  
Hardware: NVIDIA GeForce RTX 4080  
PyTorch: 2.10.0+cu128  
Seed: 42

## Fairness Statement

The DQN agent was given **every available advantage** over GENREG:

| Advantage | GENREG | DQN (best) | Who benefits |
|-----------|--------|------------|--------------|
| Game Environment | Game2048Env | Game2048Env (SAME CODE) | Equal |
| State Encoding | 22-dim flat signals | 4×4×16 one-hot grid (262-dim) | **DQN** |
| Spatial Awareness | None (flat MLP) | CNN with 2D convolutions | **DQN** |
| Action Masking | None (wastes energy on invalid) | Full valid-move masking | **DQN** |
| Reward Signal | Generic trust (no game knowledge) | Hand-crafted 2048 heuristics | **DQN** |
| Network Params | ~868 per genome | 200K-940K | **DQN (up to 1,080×)** |
| Learning Method | Gradient-free evolution | Full backpropagation | **DQN** |
| Parallel Games | 1000 batched on GPU | 32-64 vectorized | Equal-ish |
| Wall-Clock Time | Same budget | Same budget | Equal |

## How to Read the Game Counts

**GENREG** uses evolutionary training: each generation, all 1,000 genomes in the
population each play one complete game. No learning happens *during* a game —
genomes are evaluated, ranked by trust (fitness), and the best are selected to
reproduce. So "237 generations" means 237 × 1,000 = 237,000 total game episodes
were played, but only 237 selection events (learning steps) occurred. This is
analogous to 237 gradient updates, not 237,000.

**DQN** learns continuously: every 4 environment steps it performs a gradient update
on a mini-batch sampled from its replay buffer. Each training game generates many
gradient updates. So DQN's game count and gradient count are both meaningful.

## Results Summary

| Version | Params | Games (episodes) | Learning Steps | Best Tile | Avg Tile | Avg Score | Time |
|---------|--------|------------------|----------------|-----------|----------|-----------|------|
| **GENREG Evolutionary** | 868 | 240,000 (240 gen × 1,000 pop) | 240 (selection events) | 256 | 42 | 398 | 180s |
| **Vanilla DQN (MLP baseline)** | 47,300 | 2,155 | 56,326 (grad updates) | 128 | 48 | 429 | 180s |
| **CNN + Grid Input** | 873,092 | 982 | 37,962 (grad updates) | 128 | 70 | 637 | 180s |
| **CNN + Masking + Reward Shaping** | 873,092 | 1,116 | 34,469 (grad updates) | 256 | 110 | 1155 | 180s |
| **Double DQN + Dueling CNN** | 938,885 | 825 | 25,268 (grad updates) | 256 | 122 | 1321 | 180s |
| **Vectorized Envs (32 parallel)** | 938,885 | 2,465 | 14,329 (grad updates) | 512 | 261 | 3219 | 180s |
| **Full Optimization** | 938,885 | 4,340 | 4,502 (grad updates) | 1024 | 294 | 3636 | 180s |

## Optimization Progression

### GENREG: GENREG Evolutionary

Trust-based evolutionary model.  868 params per genome, gradient-free, no action masking, no spatial convolutions, no domain reward shaping.  Generic regulatory genome only.

- **Parameters**: 868 per genome
- **Population**: 1,000 genomes
- **Generations**: 240
- **Total game episodes**: 240,000 (240 gen × 1,000 pop — but only 240 evolutionary selection events)
- **Gradient updates**: 0 (gradient-free)
- **Wall clock**: 180.1s
- **Eval best tile**: 256
- **Eval avg tile**: 42
- **Eval avg score**: 398
- **Tile distribution**: 256: 5, 128: 149, 64: 132, 32: 242, 16: 203, 8: 205, 4: 64

### V1: Vanilla DQN (MLP baseline)

Baseline DQN with 22-dim signal input (same as GENREG), 3-layer MLP, uniform replay, hard target updates. Same information as GENREG but with 47K params and gradient learning.

- **Parameters**: 47,300
- **Training games**: 2,155
- **Env steps**: 230,303
- **Gradient updates**: 56,326
- **Wall clock**: 180.1s
- **Eval best tile**: 128
- **Eval avg tile**: 48
- **Eval avg score**: 429
- **Tile distribution**: 128: 6, 64: 31, 32: 62, 16: 1

### V2: CNN + Grid Input

CNN on 4×4×16 one-hot board grid.  Spatial convolutions can learn tile patterns (corners, monotonicity, merge opportunities) that flat MLPs cannot.  ~200K params.  Still no action masking.

- **Parameters**: 873,092
- **Training games**: 982
- **Env steps**: 156,844
- **Gradient updates**: 37,962
- **Wall clock**: 180.3s
- **Eval best tile**: 128
- **Eval avg tile**: 70
- **Eval avg score**: 637
- **Tile distribution**: 128: 24, 64: 52, 32: 17, 16: 6, 4: 1
- **Delta vs previous**: score +208, avg tile +23

### V3: CNN + Masking + Reward Shaping

Action masking: DQN checks which moves are valid BEFORE acting, so it never wastes energy on invalid moves.  GENREG has no such advantage.  Domain reward shaping adds corner bonus, monotonicity, smoothness, and empty-cell incentives — hand-crafted 2048 knowledge that GENREG's generic trust system does not have.

- **Parameters**: 873,092
- **Training games**: 1,116
- **Env steps**: 142,875
- **Gradient updates**: 34,469
- **Wall clock**: 180.1s
- **Eval best tile**: 256
- **Eval avg tile**: 110
- **Eval avg score**: 1155
- **Tile distribution**: 256: 7, 128: 54, 64: 33, 32: 6
- **Delta vs previous**: score +517, avg tile +40

### V4: Double DQN + Dueling CNN

Double DQN reduces Q-value overestimation.  Dueling architecture separates state-value from action-advantage, improving learning when many actions have similar outcomes (common in 2048).  ~940K params (1,080× GENREG).

- **Parameters**: 938,885
- **Training games**: 825
- **Env steps**: 106,069
- **Gradient updates**: 25,268
- **Wall clock**: 180.2s
- **Eval best tile**: 256
- **Eval avg tile**: 122
- **Eval avg score**: 1321
- **Tile distribution**: 256: 12, 128: 56, 64: 30, 32: 2
- **Delta vs previous**: score +166, avg tile +12

### V5: Vectorized Envs (32 parallel)

32 parallel game instances with batched GPU inference.  Fills the replay buffer 32× faster, dramatically increasing the number of training games within the time budget.  This directly addresses the throughput gap vs GENREG's 1000-game batched GPU evolution.

- **Parameters**: 938,885
- **Training games**: 2,465
- **Env steps**: 463,520
- **Gradient updates**: 14,329
- **Wall clock**: 180.0s
- **Eval best tile**: 512
- **Eval avg tile**: 261
- **Eval avg score**: 3219
- **Tile distribution**: 512: 20, 256: 48, 128: 24, 64: 8
- **Delta vs previous**: score +1898, avg tile +139

### V6: Full Optimization

Everything: CNN, dueling, double, vectorized (64 envs), action masking, domain rewards, 3-step returns, Huber loss, soft target updates (tau=0.005), cosine LR schedule, 512 batch.  This is the MAXIMUM EFFORT DQN with every known advantage.

- **Parameters**: 938,885
- **Training games**: 4,340
- **Env steps**: 782,808
- **Gradient updates**: 4,502
- **Wall clock**: 180.0s
- **Eval best tile**: 1024
- **Eval avg tile**: 294
- **Eval avg score**: 3636
- **Tile distribution**: 1024: 1, 512: 26, 256: 48, 128: 19, 64: 6
- **Delta vs previous**: score +418, avg tile +33


## Deployment Metrics: Model Size & Inference Speed

These metrics matter for real-world deployment — a model that requires
a GPU and 4 MB of weights has very different deployment constraints than
one that fits in 4 KB and runs on a microcontroller.

| Version | Model Size | Params | Inference (games/sec) |
|---------|-----------|--------|----------------------|
| **GENREG Evolutionary** | 145.3 KB | 868 | 243.7 |
| **Vanilla DQN (MLP baseline)** | 188.0 KB | 47,300 | 25.5 |
| **CNN + Grid Input** | 3.3 MB | 873,092 | 13.8 |
| **CNN + Masking + Reward Shaping** | 3.3 MB | 873,092 | 16.2 |
| **Double DQN + Dueling CNN** | 3.6 MB | 938,885 | 11.8 |
| **Vectorized Envs (32 parallel)** | 3.6 MB | 938,885 | 6.0 |
| **Full Optimization** | 3.6 MB | 938,885 | 5.2 |


## Hyperparameter Appendix

```
gamma              = 0.99
replay_capacity    = 200,000
optimizer          = Adam
gradient_clip      = 10.0 (L2 norm)
weight_init        = Kaiming (CNN) / Xavier (MLP)
```

Per-version settings:

| Version | Batch | LR | Eps Decay | N-step | Envs | Masking | Rewards | Soft τ |
|---------|-------|----|-----------|--------|------|---------|---------|--------|
| GENREG | — | — | — | — | 1000 | No | Generic trust | — |
| V1 | 256 | 0.0001 | 50,000 | 1 | 1 | No | Base | None |
| V2 | 256 | 0.0001 | 50,000 | 1 | 1 | No | Base | None |
| V3 | 256 | 0.0001 | 50,000 | 1 | 1 | Yes | Shaped | None |
| V4 | 256 | 0.0001 | 50,000 | 1 | 1 | Yes | Shaped | None |
| V5 | 512 | 0.0003 | 100,000 | 1 | 32 | Yes | Shaped | 0.005 |
| V6 | 512 | 0.0003 | 150,000 | 3 | 64 | Yes | Shaped | 0.005 |

---
*Report generated by dqn_2048_benchmark.py*