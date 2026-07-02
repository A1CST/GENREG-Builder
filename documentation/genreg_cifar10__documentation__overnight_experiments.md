# Overnight Experiment Results — Unified Cascade

## ALL-TIME BEST: 47.65% kNN (Maha-Only Pool, L1, Augmented)

### Top Results
1. **47.65%** — D6: pop 60, 1024n, augment, maha-only pool (L1, 10D)
2. **46.92%** — D2: pop 100, 1024n, maha-only pool (L1, 10D)
3. **46.68%** — D4: pop 60, 1024n, seed=123, maha-only pool (L1, 10D)
4. **39.34%** — C6: pop 60, 1024n, all-features pool (L3, ~3100D)
5. **38.92%** — C1: pop 100, 1024n, all-features pool (L4, ~4136D)

Baseline kNN(k=5): 21.51% | Baseline NCC: 40.86%

## Critical Finding: Maha-Only Pool >> All-Features Pool

The Mahalanobis distance features (10D per layer) dramatically outperform the combined projection+maha features (1034D per layer) for kNN classification:

- **Maha-only pool, L1**: 46-48% kNN on 10D
- **All-features pool, best**: 38-39% kNN on 3000-4000D
- **Improvement: ~9%** just by dropping projections from pool

### Why Maha-Only Works
1. kNN suffers from curse of dimensionality — 10D >> 1034D for kNN
2. Maha features are class-distance representations (one per class)
3. Maha features are normalized and comparable across layers
4. Projection features (1024D) are raw neuron outputs — noisy and high-dimensional

### Why Maha-Only Still Degrades
1. L2+ Maha features are based on distorted inputs (previous layer output, not raw data)
2. Even 10D per layer adds up: L5 = 50D, L10 = 100D — kNN degrades
3. L2+ Maha features are partially redundant with L1's
4. Training fitness diverges from test performance (80% train vs 47% test)

## Batch 1: Baseline Tests (pop 20)

| Exp | Config | Best | Peak |
|-----|--------|:---:|:---:|
| 1 | 512n, eval=500/2000 | 27.98% | L4 |
| 2 | 1024n, eval=500/2000 | 28.07% | L5 |
| 3 | 1024n, augment | 30.72% | L5 |
| 4 | Neuron search, eval=1000 | 26.77% | L2 |
| 5 | NCC fitness, 1024n | 31.40% | L4 |
| 6 | Pop 60, 1024n | 34.05% | L5 |
| 7 | Pop 60, 512n, eval=2000 | 35.16% | L4 |
| 8 | Pop 60, 1024n, 10L | 35.12% | L3 |

## Batch 2: Lever Combinations

| Exp | Config | Best | Peak |
|-----|--------|:---:|:---:|
| B1 | Pop 60, 1024n, augment | 34.22% | L3 |
| B2 | Pop 60, NCC fitness | 31.24% | L5 |
| **B3** | **Pop 60, eval=4K/6K** | **38.73%** | **L4** |
| B4 | Pop 60, pool-pca=256 | 30.87% | L3 |
| B5 | Pop 60, pool-pca=512 | 30.32% | L3 |
| B6 | Pop 60, aug+NCC | 32.86% | L5 |
| B7 | Pop 100, eval=2K/4K | 35.77% | L3 |
| B8 | Pop 60, aug+pca=512 | 36.86% | L4 |

## Batch 3: Pushing Best Combos

| Exp | Config | Best | Peak |
|-----|--------|:---:|:---:|
| C1 | Pop 100, eval=4K/6K | 38.92% | L4 |
| C2 | Pop 60, eval=6K/6K | 38.24% | L4 |
| C3 | Pop 60, eval=4K/6K, 8L | 38.73% | L4 |
| C4 | Pop 60, 512n, eval=4K/6K, 8L | 36.73% | L5 |
| C5 | Pop 100, 512n, eval=4K/6K, 8L | 35.57% | L3 |
| **C6** | **Pop 60, seed=123** | **39.34%** | **L3** |

## Batch 4: Maha-Only Focus

| Exp | Config | Best | Peak |
|-----|--------|:---:|:---:|
| D1 | Maha-only, 2048n | 37.47% | L1 |
| **D2** | **Maha-only, pop 100** | **46.92%** | **L1** |
| D3 | Maha-only, 256n, 20L | 42.96% | L1 |
| D4 | Maha-only, seed=123 | 46.68% | L1 |
| D5 | Proj-only | 38.45% | L4 |
| **D6** | **Maha-only, augment** | **47.65%** | **L1** |

### Batch 4 Findings
- **2048 neurons hurt Maha** — inv_cov estimation unstable at 2048D (37% vs 47%)
- **256 neurons work but lower** — 42.96% (less projection capacity)
- **1024 neurons is optimal** for Maha quality
- **Augmentation helps slightly** — 47.65% vs 46.92%
- **Pop 100 ≈ Pop 60** for maha-only (46.92% vs 46.43-46.68%)
- **Proj-only ≈ All-features** — 38.45% vs 38.73% (Maha negligible in high-D pool)
- **Maha features ARE the pool** — 10D Maha >> 1024D projections for kNN

## Key Findings Summary

### Lever Rankings (for all-features pool)
1. **Eval quality** — eval=4K/6K gives ~10% over eval=500/2K
2. **Pop size** — pop 60 ~6% over pop 20, pop 100 ~1% over pop 60
3. **Neuron count** — 1024 > 512 by ~2-3%
4. **Pool PCA** — HURTS (removes signal)
5. **NCC fitness** — worse than kNN fitness
6. **Augmentation** — helps at pop 20, neutral at pop 60+

### Architecture Insights
1. **Curse of dimensionality is THE bottleneck** — 10D maha pool >> 1034D all-features pool
2. **Cascade doesn't help maha-only pool** — all peaks at L1, degrades monotonically
3. **Cascade helps all-features pool** — peaks at L3-L5 (adds dimensions that help kNN)
4. **Massive generalization gap** — 60-83% train fitness vs 35-47% test kNN
5. **NCC and kNN diverge** — pool NCC degrades while pool kNN improves (and vice versa)

## Open Questions / Next Steps
1. Can we make L2+ Maha features orthogonal to L1's? (diverse projections)
2. What about Maha features computed on original data, not previous layer output?
3. Can we use weighted kNN to downweight redundant pool dimensions?
4. What about ensemble: best Maha (L1) + all-features pool (L3)?
5. Can we evolve the fold specifically to maximize Maha quality?
