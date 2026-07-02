# GENREG v6_words — Evolved Language Model

## Overview

A next-token predictor built entirely through evolution. No gradients, no transformers, no optimizers. Three layers, each built on top of the last:

1. **Tokenizer** — dictionary-based with evolved decomposition rules (DONE)
2. **N-grams** — bigram/trigram statistics from tokenized corpus (DONE)
3. **Semantic Embeddings** — evolved to capture PMI co-occurrence patterns (TRAINING)
4. **Combined Predictor** — blends n-gram + embeddings for final prediction (BUILT)

## Architecture

```
Raw Text
    │
    ▼
┌──────────────┐
│  TOKENIZER   │  259,625 vocab (NLTK dictionary)
│  351 bytes   │  Evolved prefix/suffix rules for unknown words
│  100% consistent │
└──────┬───────┘
       │ token IDs
       ▼
┌──────────────┐
│   N-GRAMS    │  160M tokens from 1,000 books
│   160 MB     │  10,001 bigram contexts
│              │  3,220,335 trigram contexts
└──────┬───────┘
       │ co-occurrence statistics
       ▼
┌──────────────┐
│  SEMANTIC    │  1,001 active vocab × 32D embeddings
│  EMBEDDINGS  │  32,032 evolved parameters
│  ~700 KB     │  PMI-weighted fitness function
└──────┬───────┘
       │ similarity scores
       ▼
┌──────────────┐
│  COMBINED    │  N-gram scores × 0.7 + Embedding scores × 0.3
│  PREDICTOR   │  Trigram → Bigram → Embedding fallback chain
└──────────────┘
```

## Component Details

### 1. Tokenizer (`tokenizer.py`)

**Status: COMPLETE**

- **Base vocabulary**: 259,625 tokens from NLTK English dictionary
- **Special tokens**: `<pad>`, `<unk>`, `<bos>`, `<eos>` + all ASCII characters
- **Unknown word handling**: Evolved prefix/suffix decomposition rules
  - 15 suffix rules: `es, en, ed, er, ing, ity, 'll, al, 've, ship, s, est, 'd, ness, ling`
  - 11 prefix rules: `dis, il, micro, re, over, ex, de, en, in, sub, em`
- **Key property**: 100% deterministic — same input always produces same token IDs
- **Unknown word examples**:
  - `retweeted` → `['re', 'tweet', 'ed']`
  - `microservices` → `['micro', 'service', 's']`
  - `overfitting` → `['over', 'fitting']`
  - `ChatGPT` → `['C', 'h', 'a', 't', 'G', 'P', 'T']` (character fallback)
- **File**: `tokenizer.json` (351 bytes — just the rules, vocab rebuilds from NLTK)

### 2. N-gram Statistics (`ngrams.py`)

**Status: COMPLETE**

- **Corpus**: 1,000 Project Gutenberg books, 487.8 MB raw text
- **Tokens processed**: 160,362,589
- **Unique tokens used**: 114,525 (of 259,625 vocab)
- **Active vocab**: Top 10,000 tokens (covers 99%+ of corpus)
- **Bigram contexts**: 10,001 (every active token has continuation stats)
- **Trigram contexts**: 3,220,335
- **File**: `ngrams.pkl` (160 MB)

**Baseline accuracy on held-out test set (50 books, 24,900 predictions):**

| Model | Top-1 | Top-5 | Top-10 |
|-------|-------|-------|--------|
| Bigram | 14.9% | 35.1% | 46.8% |
| Trigram | 25.1% | 49.5% | 59.8% |

### 3. Semantic Embeddings (`semantic.py`)

**Status: TRAINING (pop=500 run in progress)**

- **Approach**: Evolve token embeddings so embedding similarity correlates with PMI
- **Active vocab**: 1,001 tokens (top 1,000 + UNK)
- **Embedding dimension**: 32
- **Total evolved parameters**: 32,032 (1,001 × 32)
- **Fitness function**: PMI-weighted correlation
  - For each bigram context, compute PMI for all possible next tokens
  - Score = correlation between embedding similarity scores and PMI values
  - Only positive PMI counts (genuine associations, not chance co-occurrence)
- **PMI training contexts**: ~10,000 contexts with positive PMI associations
- **Evolution config**: pop=500, mutation_rate=0.08, mutation_scale=0.02, elite=3

**Training progress:**

| Config | Pop | PMI Correlation | Status |
|--------|-----|-----------------|--------|
| 10K vocab, 64D (640K params) | 50 | 0.004 | Failed — too many params |
| 10K vocab, 8D (80K params) | 50 | 0.002 | Failed — still too many |
| **1K vocab, 32D (32K params)** | **50** | **0.146** | **Works — clear learning** |
| **1K vocab, 32D (32K params)** | **500** | **0.041 @ gen 280** | **Running — projected 0.18-0.22** |

**PMI correlation targets:**
- 0.20-0.30 — embeddings capture real semantic neighborhoods
- 0.40-0.50 — comparable to early Word2Vec quality
- 0.60+ — strong, competitive with gradient-trained embeddings

### 4. Combined Predictor (`predictor.py`)

**Status: BUILT, awaiting better embeddings**

- **Strategy**: Weighted blend of n-gram and embedding scores
- **Fallback chain**: Trigram → Bigram → Embedding
- **Default weights**: 70% n-gram, 30% embedding
- **Evaluation framework**: Tests all three models (ngram, embed, combined) on held-out data
- **Text generation**: Weighted random sampling from combined predictions

**Current results (with old 10K vocab embeddings — not meaningful yet):**

| Model | Top-1 | Top-5 | Top-10 |
|-------|-------|-------|--------|
| N-gram | 25.1% | 49.5% | 59.8% |
| Embed (old) | 1.9% | 4.7% | 5.1% |
| Combined | 13.2% | 36.3% | 49.7% |

The old embeddings hurt the combined score. The 1K vocab embeddings (once trained) should help.

## Files

```
v6_words/
  __init__.py
  tokenizer.py        — Evolved tokenizer with NLTK base + prefix/suffix rules
  tokenizer.json      — Saved rules (351 bytes)
  ngrams.py           — N-gram statistics builder
  ngrams.pkl          — Bigram/trigram counts (160 MB)
  semantic.py         — Evolved semantic embeddings with PMI fitness
  semantic.npz        — Saved embeddings (~700 KB)
  predictor.py        — Combined n-gram + embedding predictor
  download_books.py   — Gutenberg corpus downloader
  data/               — 1,000 downloaded books (487.8 MB)
  ARCHITECTURE.md     — This file
```

## Key Design Decisions

### Why NLTK dictionary instead of BPE?
BPE learns subword units through frequency — that's a form of optimization. Using the full English dictionary as base vocab means every known word is a single token. The evolution only handles unknown word decomposition (prefix/suffix rules), which is a much smaller search space.

### Why 1K active vocab for embeddings?
The full 259K vocab has 114K tokens used in the corpus, but only ~3,000 cover 95%. Evolving 259K × 32D = 8.3M parameters is impossible with mutation-only evolution. 1K × 32D = 32K parameters is tractable with pop=500. The top 1,000 tokens cover the most important words; rare tokens map to UNK.

### Why PMI as fitness instead of raw prediction accuracy?
PMI (Pointwise Mutual Information) measures genuine association between tokens, not just frequency. A token that always appears (like "the") would dominate raw accuracy but provides no semantic signal. PMI rewards embeddings that capture *surprising* co-occurrences — the real semantic relationships.

### Why n-gram + embedding blend instead of pure evolution?
N-grams are unbeatable for seen contexts — they're exact statistics from 160M tokens. Embeddings add value for *unseen* contexts by generalizing from similar contexts. The blend lets each component do what it's best at.

## Training Pipeline

```bash
# 1. Download corpus
python v6_words/download_books.py

# 2. Evolve tokenizer (fast — 50 gens, ~2 min)
python v6_words/tokenizer.py

# 3. Tokenize corpus and build n-grams (~5 min)
# (tokenize inline, then run ngrams)
python v6_words/ngrams.py

# 4. Evolve semantic embeddings (slow — 2000 gens, hours)
python v6_words/semantic.py

# 5. Evaluate combined predictor
python v6_words/predictor.py
```

## Migration Notes

When moving to a new machine:
1. Copy the entire `v6_words/` directory
2. The `data/` folder (487 MB) contains the books — can re-download if needed
3. `ngrams.pkl` (160 MB) is the critical file — expensive to rebuild
4. `tokenizer.json` (351 bytes) and `semantic.npz` (~700 KB) are small
5. The `v6/` directory is needed for `GPUEvolutionState` imports (or copy the evolution code)
6. Dependencies: `torch`, `numpy`, `nltk`, `requests`
7. Install NLTK data: `python -c "import nltk; nltk.download('words')"`

## What's Next

1. **Finish pop=500 semantic training** — projected 0.18-0.22 PMI correlation
2. **Evaluate combined model** — does embedding improve over n-gram baseline?
3. **If yes**: Push training harder (more gens, bigger pop, wider embed dim)
4. **If marginal**: Try different fitness (direct prediction accuracy instead of PMI)
5. **Text generation quality** — human-readable output from the combined model
6. **Scale test** — can the architecture handle larger active vocab (5K, 10K)?
