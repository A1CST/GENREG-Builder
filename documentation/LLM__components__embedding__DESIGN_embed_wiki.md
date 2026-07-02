# GENREG Model Design Doc — Wiki Embedding Organism

## 1. Name
`embed_wiki_v1`

## 2. Purpose
Evolve a full-scale embedding model (51,641 × 768) that GENERATES
embeddings through internal structure, not a flat lookup table. The
organism earns its embedding space through evolutionary pressure.
When good enough, freeze it as the input layer for attention/LM.

## 3. Interface

**Input:** token_id (int in [0, 51641))

**Output:** 768-dimensional float vector

**Runtime state:** Stateless per call. Same token → same vector always.

## 4. Evolved parameters per genome

The organism is a GENREG neural encoder, not a matrix:

| name | shape | init | role |
|---|---|---|---|
| `hash_in` | `(V, K)` | sparse random | K-bit token signature (K=64) |
| `W_enc` | `(K, H)` | randn/√K | first projection: signature → hidden |
| `enc_b` | `(H,)` | zeros | encoder bias |
| `act_ids` | `(H,)` | randint(0,8) | per-neuron activation function |
| `act_p1..p4` | `(H,)` | defaults | activation params |
| `W_out` | `(H, D)` | randn/√H | hidden → embedding dim |
| `out_b` | `(D,)` | zeros | output bias |

Where K=64, H=256, D=768.

Forward: `token_id → hash_in[token_id] → W_enc → activate → W_out → embedding`

Total per genome: ~64×256 + 256 + 256×768 + 768 ≈ **213K params**.
Population of 256 genomes: ~54M params total in GPU memory.

Why not a flat (V×D) table: 51641×768 = 39.7M params per genome.
256 genomes = 10B params. Won't fit. The encoder compresses V tokens
through a K-dim hash, making it tractable.

## 5. Fitness equation

```
fitness = w1 * cooccurrence_coherence
        + w2 * separation_pressure
        + w3 * downstream_probe
        + w4 * analogy_score    (periodic, every N gens)
```

**`cooccurrence_coherence`** (range [0, 1]):
Sample P positive pairs (tokens that co-occur within window W in
corpus). Sample P negative pairs (random). Compute cosine similarity
for each. Score = mean(cos_pos) - mean(cos_neg). Higher = better
structure. Measures: words in similar contexts → similar vectors.

**`separation_pressure`** (range [0, 1]):
Compute std of pairwise distances among a random sample of N
embeddings. Low std = everything collapsed together (bad). High std =
space is well-utilized. Normalize to [0,1] by dividing by theoretical
max std for D dimensions.

**`downstream_probe`** (range [0, 1]):
Freeze embeddings. For each position t in a corpus sample, take
embedding[t], predict token[t+1] via a FIXED (not evolved) linear
probe: `softmax(W_probe @ emb[t])`. Top-1 accuracy on a batch.
W_probe is fitted via closed-form least-squares each eval (same
trick as embed_03). Measures: do embeddings carry enough info for
prediction?

**`analogy_score`** (range [0, 1], periodic):
Every 50 gens, evaluate on a set of ~200 analogy tuples
(a:b :: c:d). For each, find nearest neighbor to `emb[b]-emb[a]+emb[c]`.
Score = fraction where nearest = d. Expensive, so periodic.

Weights: `w1=0.3, w2=0.2, w3=0.4, w4=0.1`.
Downstream probe gets highest weight because it's the most honest
signal — decorative geometry doesn't help prediction.

## 6. Energy equation

```
energy_next = energy * ENERGY_DECAY + (fitness - median_fitness) * ENERGY_GAIN
```

- ENERGY_DECAY = 0.90
- ENERGY_GAIN = 2.5
- ENERGY_FLOOR = 0.15
- E_MAX = 1.5

Expected: 5-15% of population culled per gen. Organisms that achieve
fitness through brute memorization (high downstream but low coherence/
separation) get pulled down by the multi-objective fitness. Organisms
that collapse embeddings to a blob get killed by separation pressure.

## 7. Selection

- POP_SIZE = 256
- ELITE_PCT = 10 (hard cull, 90% replaced each gen)
- Maturation gate: yes
- Reproduction: fitness-weighted sampling from elite pool

## 8. Mutation schedule

- INIT_MUT_RATE = 0.03 (per-element probability)
- MIN_MUT_RATE = 0.003
- Anneals as fitness approaches goal
- Per-tensor: W_enc/W_out get Gaussian perturbation; act_ids get
  categorical flips at 1% rate; hash_in stays frozen after init
  (it's a fixed random projection, not learned)

## 9. Hyperparameters

| param | value | reason |
|---|---|---|
| D (embed dim) | 768 | match GPT-2 |
| H (hidden) | 256 | compression bottleneck |
| K (hash bits) | 64 | token signature width |
| N_GENERATIONS | 2000 | |
| BATCH_SIZE | 4096 | co-occurrence pairs per eval |
| WINDOW | 5 | co-occurrence window half-width |
| LOG_EVERY | 25 | |

## 10. Success criteria

**Local bar:** downstream probe accuracy > bigram baseline on the
same corpus. If the embeddings can't beat "always predict the most
common next word," they carry no information.

**Downstream bar:** when frozen and plugged into the next LM stage,
the model trains faster and reaches higher accuracy than SVD or
random embeddings. Measured in the next component's training.

## 11. Failure modes to watch

- **Embedding collapse:** all tokens map to similar vectors. Caught
  by separation pressure going to 0.
- **Decorative geometry:** high coherence + separation but low
  downstream probe = pretty clusters that don't help prediction.
- **Hash collision damage:** K=64 hash may map some tokens to
  identical signatures → identical embeddings. Monitor unique-vector
  count. If <90% of vocab has unique embeddings, bump K.
- **Fitness dominated by one term:** if w3 (probe) overwhelms
  w1+w2, organism just optimizes for next-token at the expense of
  structure. Monitor per-term breakdown.

## 12. Baselines to beat

- Random embeddings + optimal probe
- SVD of co-occurrence matrix (current monolithic approach)
- word2vec-style PMI factorization (if precomputed)

## 13. Output artifacts

- `embed_wiki_v1_best.pkl` — best genome's weights + config + scores
- `embed_wiki_v1_findings.md` — what worked/didn't
- `run_embed_wiki_v1.log` — per-gen trace
