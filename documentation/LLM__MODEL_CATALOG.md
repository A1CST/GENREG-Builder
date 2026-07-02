# GENREG Wiki LLM — Complete Model Catalog

**Date:** 2026-04-16
**Status:** End-to-end pipeline working. Generates English words with topical coherence. Pursuing grammatical coherence via prediction head optimization.

## Pipeline Overview

```
token_ids → [Tokenizer] → [Embedding] → [PosEnc] → [Attn L0] → [Attn L1] → [PredHead] → logits → text
              frozen         frozen        frozen     frozen       frozen      ACTIVE
```

Total frozen model size: **49.1 MB** (excluding prediction head)
Total with ridge head: **208 MB** | with evolved head: **51.4 MB**

---

## 1. Tokenizer

| field | value |
|-------|-------|
| **checkpoint** | `components/tokenizer/checkpoints_wiki/wikitok_gen_00300_final.pkl` (271 bytes) |
| **token stream** | `components/tokenizer/wiki_token_stream.pkl` (75 MB) |
| **vocabulary** | 51,641 tokens (incl. 4 special + 71 chars + 51,566 words, min_count=5) |
| **corpus** | WikiText-103, 19,370,666 tokens, 9,198,862 words |
| **method** | GENREG evolved morphological decomposition (prefixes/suffixes) |
| **coverage** | known 34.7%, decomposed 7.1%, char-fallback 58.2% |

**Interface:** text → list of token IDs (whitespace split, lowercase)

## 2. Embedding (`embed_wiki_v1`)

| field | value |
|-------|-------|
| **checkpoint** | `components/embedding/checkpoints_embed_wiki/embed_wiki_gen_00600_final.pkl` (27 MB) |
| **architecture** | PPMI-SVD hash (V=51641, K=128) → evolved encoder (K→H=256→D=768) + skip connection |
| **output dim** | 768 |
| **params** | shared frozen hash: 6.6M, evolved encoder: 231K, skip_gain: 0.837 |
| **fitness** | co=0.83, sep=0.83, probe=0.90, nn_recall=0.49, analogy=0.62 |
| **quality** | king→reign/prince/throne, france→spain/italy, king:queen::man:woman(0.80) |

**Interface:** token_id (int) → 768-dim float vector

**Key parameters:**
- `hash_in`: (51641, 128) — PPMI-SVD of co-occurrence matrix, frozen
- `W_skip`: (768, 128) — random orthogonal skip projection, frozen
- `skip_gain`: 0.837 — evolved scalar
- `W_enc`: (256, 128) — evolved encoder weights
- `enc_b`: (256,) — evolved bias
- `act_ids`: (256,) — per-neuron activation IDs (8 types available)
- `act_p1..p4`: (256,) each — activation params
- `W_out`: (768, 256) — evolved output projection
- `out_b`: (768,) — output bias

**Forward:** `emb = skip_gain * (hash @ W_skip^T) + W_out @ act(W_enc @ hash + enc_b) + out_b`

## 3. Positional Encoding (`posenc_wiki_v1`)

| field | value |
|-------|-------|
| **checkpoint** | `components/posenc/checkpoints_posenc_wiki/posenc_wiki_gen_00500_final.pkl` (1.6 MB) |
| **architecture** | sinusoidal init table (512, 768) + per-dim evolved gain + per-dim activations |
| **max sequence** | 512 positions |
| **params** | P: 393K, dim_gain: 768, global_gain: 0.814, activations: 768×5 |
| **quality** | preservation=0.989, posrec=100% (32 buckets), magnitude=0.15 |

**Interface:** position (int) → 768-dim additive signal

**Key parameters:**
- `P`: (512, 768) — evolved position table (sinusoidal init)
- `dim_gain`: (768,) — per-dim scaling [0.001, 0.373], mean=0.200
- `global_gain`: 0.814 — overall magnitude scalar
- `act_ids`: (768,) — per-dim activation IDs (mostly identity_plus=7)
- `act_p1..p4`: (768,) each

**Forward:** `pos_signal = global_gain * dim_gain * act(P[position])`
**Combined:** `pos_aware = embedding + pos_signal`

## 4. Attention Stack (2 layers, auto-stacked)

### Layer 0 (`attn_wiki L0`)

| field | value |
|-------|-------|
| **checkpoint** | `checkpoints_attn_wiki/attn_wiki_gen_00248_L0_final.pkl` (9.1 MB) |
| **architecture** | 6 heads × 128 dim, evolved Q/K/V/O + per-head logit activations |
| **params** | W_Q/K/V/O: 4×768×768 = 2.36M, head_gain: (6,), logit activations: 6×5 |
| **cloze** | 0.831 (plateau at gen 98, frozen at gen 248) |

### Layer 1 (`attn_wiki L1`)

| field | value |
|-------|-------|
| **checkpoint** | `checkpoints_attn_wiki/attn_wiki_gen_00150_L1_final.pkl` (9.1 MB) |
| **architecture** | same as L0 (independently evolved) |
| **cloze** | 0.929 (+9.8pp over L0 alone, plateau at gen 0, frozen at gen 150) |

**Shared architecture per layer:**
- `W_Q, W_K, W_V, W_O`: (768, 768) each — evolved projections
- `head_gain`: (6,) — per-head output scaling
- `logit_act_ids`: (6,) — per-head activation on QK logits
- `logit_act_p1..p4`: (6,) each
- Local attention bias: `-0.1 × |pos_i - pos_j|` (fixed, not evolved)
- Skip connection: `output = input + attention_output`

**Forward per layer:**
```
Q, K, V = input @ W_Q/K/V  →  reshape to (6, seq, 128)
logits = Q @ K^T / sqrt(128) + local_bias
logits = per_head_activation(logits)
attn = softmax(logits)
out = concat(attn @ V * head_gain) @ W_O
output = input + out
```

**Auto-stacking:** Layer 2 was evolved but discarded (cloze 0.904 < stack's 0.929).

## 5. Prediction Head

### Ridge (baseline)
- Closed-form `W = (X^TX + λI)^{-1} X^T Y`, fitted on 65K corpus samples
- `W_head`: (768, 51641) = 158 MB
- Test accuracy: 55.3% top-1, 68.9% top-5
- Produces: frequent tokens ("the", "of", " "), topically related but no grammar

### Evolved (current, in development)
- `W_proj`: (768, 768) + per-neuron activations + `out_scale`
- Weight-tied: logits = activated(W_proj @ attn_output) @ emb_table^T
- 590K params per genome (vs 39.6M for full ridge)
- Fitness: surprise-weighted accuracy with frequency cost
- Status: gen 100, surp=0.080, raw=0.480. Energy cost mechanism just added.

## 6. Generation Script

`generate_wiki.py` — supports `--prompt`, `--interactive`, `--auto`, `--metrics`
- Temperature, top-k, repetition penalty controls
- Hot-reload of attention checkpoints (`/reload`)
- JSONL logging to `generation_log_wiki.jsonl`
- Auto-monitor mode checks for new checkpoints every N seconds

## Total Parameter Counts

| component | evolved params | shared/frozen substrate | file size |
|-----------|---------------|------------------------|-----------|
| Tokenizer | ~30 rules | — | 271 B |
| Embedding | 231K | 6.6M (PPMI hash) + 590K (W_skip) | 27 MB |
| PosEnc | 397K | — | 1.6 MB |
| Attn L0 | 2.36M | — | 9.1 MB |
| Attn L1 | 2.36M | — | 9.1 MB |
| PredHead (evolved) | 590K | emb_table (39.7M, shared) | 2.3 MB |
| **TOTAL** | **5.9M evolved** | **47M shared** | **49.1 MB** |

For comparison: GPT-2 small = 124M parameters, 500 MB.

## Tunable Knobs (for optimization)

### Inference-time
- `temperature`: controls sampling randomness (default 0.8)
- `top_k`: vocabulary cutoff for sampling (default 50)
- `rep_penalty`: penalizes recently generated tokens (default 1.5)
- Generation uses autoregressive decoding with full 512-token context window

### Prediction Head
- Ridge λ (regularization strength)
- Ridge training data volume (N windows)
- Evolved head: W_proj init, mutation rate/sigma, out_scale, activation types
- Weight-tying vs independent W_head
- Frequency cost scaling in evolved fitness

### Attention
- Local attention bias strength (-0.1 × distance, currently fixed)
- head_gain values per layer
- Number of heads (currently 6)
- Head dimension (currently 128)

### Embedding
- skip_gain (frozen at 0.837, could refit)
- Hash dimensionality K=128

### Positional Encoding
- global_gain (frozen at 0.814)
- dim_gain distribution
