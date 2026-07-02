# GENREG Model Design Doc — Wiki Positional Encoder

## 1. Name
`posenc_wiki_v1`

## 2. Purpose
Evolve a positional encoding that adds order-awareness to the frozen
`embed_wiki_v1` embeddings, WITHOUT destroying their semantic structure.
Same skip-philosophy as embedding stage: preserve the frozen substrate,
let evolution add useful structure on top.

## 3. Interface

**Input:** (position: int in [0, 512)), plus the bare embedding of the
token at that position.

**Output:** 768-dim float vector = `emb + global_gain × per_dim_gain ⊙ activation(P[pos])`

Stateless per call. Same (token, position) → same vector always.

## 4. Evolved parameters per genome

| name | shape | init | role |
|---|---|---|---|
| `P` | `(max_len=512, D=768)` | randn × 0.01 | raw position table |
| `dim_gain` | `(D,)` | zeros + small noise | per-dim control of position contribution |
| `global_gain` | `()` scalar | 0.1 | overall magnitude of position signal |
| `act_ids` | `(D,)` | randint(0, NUM_ACTIVATIONS) | per-dim activation on position signal |
| `act_p1..p4` | `(D,)` each | DEFAULT_PARAMS | activation params |

Per-genome total: 512×768 + 768 + 1 + 5×768 ≈ **397 K params**.

**Why per-dim gain + per-dim activation:** The frozen embedding has
768 dims, each encoding specific semantic features. If position
encoding steps on those dims linearly, it destroys semantic structure.
Per-dim gain lets evolution discover that certain dims should carry
position info (high gain) while others should be left alone (near-zero
gain). Per-dim activation lets the organism shape the position signal
non-linearly — sinusoidal-like curves, sharp transitions, etc. — giving
it real expressive power beyond a flat lookup.

This is the GENREG move that a vanilla "learned position table" lacks.

## 5. Fitness equation

The key anti-Goodhart signal is **probe uplift over bare embeddings**.
Position info must IMPROVE next-token prediction, not just exist.

```
fitness = preservation_gate × (
    w1 * probe_uplift
  + w2 * discriminability
  + w3 * smoothness
)
```

**`probe_uplift`** (primary signal, range [0, 1]):
1. Sample N random (start_pos, window=4) context windows.
2. Position-encode: `out[t] = emb[t] + gain × dim_gain ⊙ act(P[abs_pos])`.
3. Mean-pool context window → (D,) feature vector per sample.
4. Fit closed-form ridge regression: feature → token[start+4].
5. Measure top-1 accuracy on held-out split.
6. Baseline: same probe on bare embeddings (no position info).
7. **Uplift = probe_acc_with_pos − probe_acc_bare**.
8. Rescale: `uplift_score = clamp(uplift / 0.05, 0, 1)` (5pp gain = perfect).

Mean-pooling destroys order. Only if position encoding adds
order-preserving structure can a pooled context predict what's next.

**`discriminability`** (range [0, 1]):
Same token at different positions must produce measurably different
outputs. Score = `1 − mean(|cos(out[tok, pos1], out[tok, pos2])|)`
sampled over random (tok, pos1, pos2) triples with pos1 ≠ pos2.
If position has no effect: score ≈ 0 (outputs identical to bare emb).
If position dominates and kills semantics: score close to 1 but
preservation gate will kill this regime.

**`smoothness`** (range [0, 1]):
Adjacent positions should have similar-but-not-identical signatures.
`smoothness = mean(cos(P[t], P[t+1]))` over sampled adjacent pairs.
Clamp to [0, 1]. Low smoothness = chaotic, unusable. Too high (≈1.0)
means no position differentiation. Target ~0.7–0.9.

**`preservation_gate`** (multiplicative, range [0, 1]):
Position encoding must preserve semantic content of the frozen embedding.
`preservation = mean(cos(bare_emb[tok], out[tok, pos]))` over sampled pairs.
Gate: `gate = clamp((preservation − 0.70) / 0.20, 0, 1)`.
- preservation ≥ 0.90: full credit (gate = 1.0)
- preservation ≤ 0.70: full kill (gate = 0.0)
- between: linear ramp

Weights: `w1=0.70 probe_uplift, w2=0.20 discrim, w3=0.10 smoothness`.
Probe uplift dominates because it's the ONLY honest measure of whether
the position info is actually useful.

## 6. Energy equation

Standard GENREG:
```
energy_next = energy × ENERGY_DECAY + (fitness − median_fitness) × ENERGY_GAIN
```
- ENERGY_DECAY = 0.90, ENERGY_GAIN = 2.5, ENERGY_FLOOR = 0.15, E_MAX = 1.5

## 7. Selection

- POP_SIZE = 96
- ELITE_PCT = 10 (hard cull, 90% replaced each gen)
- Maturation gate: yes (offspring cannot parent until proving fitness)
- Reproduction: fitness-weighted sampling from elite pool

## 8. Mutation schedule

- INIT_MUT_RATE = 0.03
- MIN_MUT_RATE = 0.005
- Table P mutated at full rate (Gaussian σ=0.01)
- dim_gain at rate r (σ=0.02), clamped to [-1, 1]
- global_gain at rate r (σ=0.02), clamped to [0.01, 2.0]
- act_ids flipped at 1% rate per dim
- act_p1..p4 at rate r (σ=0.02)

## 9. Hyperparameters

| param | value | reason |
|---|---|---|
| MAX_LEN | 512 | per design spec |
| D | 768 | matches embed_wiki_v1 |
| POP | 96 | VRAM constraint |
| N_GENERATIONS | 1000 | |
| PROBE_WINDOW | 4 | bag-of-context size |
| PROBE_SAMPLES | 512 | (ctx, next_token) samples per gen |
| PROBE_TOPK | 500 | frequent-token subset for probe targets |
| DISCRIM_SAMPLES | 256 | (tok, pos1, pos2) triples |
| PRESERVATION_SAMPLES | 256 | (tok, pos) pairs for cosine check |
| LOG_EVERY | 25 | |

## 10. Success criteria

**Local bar:** probe_uplift ≥ 0.02 (2pp improvement over bare embeddings
on bag-of-context next-token prediction). Below that, position info
isn't earning its keep.

**Preservation bar:** mean cosine(bare_emb, out) ≥ 0.90 — position
encoding must not destroy semantic structure.

**Qualitative bar:** for a repeated token at positions 0, 100, 300,
the three outputs must differ meaningfully (>0.05 cosine distance)
while all still being recognizably near the bare token embedding.

**Downstream bar:** when this stage is frozen and plugged into attention,
the LM must train faster than with bare embeddings. Measured in the next
stage.

## 11. Failure modes to watch

- **Semantic destruction:** global_gain grows too large, position swamps
  semantics. Caught by preservation gate.
- **Dead position:** dim_gain evolves to all-zero, position info = 0
  everywhere, uplift = 0. Caught by discrim signal.
- **Decorative position:** discriminability high but uplift = 0 —
  position info exists but doesn't help prediction. Caught by probe
  uplift being THE primary signal.
- **Bag-of-context probe ceiling:** if bare embedding probe is already
  at 99% (e.g., because mean-pooling happens to carry enough info),
  there's no room for uplift. Watched at gen 0; if bare baseline too
  high, lengthen context window or use harder probe.

## 12. Baselines to beat

- Bare embeddings (no position info) — must beat this on probe
- Sinusoidal positional encoding (fixed, no learning)
- Zero-initialized table (starts with no effect, measures if evolution
  even moves)

## 13. Output artifacts

- `posenc_wiki_v1_best.pkl` — best genome's table + gain + activations
- `posenc_wiki_v1_findings.md` — what worked/didn't
- `run_posenc_wiki_v1.log` — per-gen trace
