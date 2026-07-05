# Word-Level Tree LM — implementation + first test campaign (2026-07-04)

## What was built

The Tree-of-Models project now has a **Token level** switch (tree page sidebar):
**byte** (256 symbols, the original) or **word** (top-K word/punctuation
vocabulary). Word mode gives the model real semantic units — the embedding
table IS a word table, generation emits whole words, and the runs-page
embedding clouds show word geometry directly.

Mechanics:
- Tokenizer: corpus decoded UTF-8, typographic punctuation folded to ASCII;
  tokens = words (apostrophes kept), numbers, single punctuation. Top-K
  vocabulary, id 0 = `<unk>`. 10.4M corpus tokens; coverage 85.6% @ 2048,
  90.4% @ 4096.
- `<unk>` is **never a training target and never emitted** in generation
  (it remains context). Without this the model just predicts `<unk>` forever.
- Word-mode models persist their vocabulary inside `model.npz`, so /runs
  replay and generation detokenize correctly.
- Accuracy is over **real words only** — a harder, honest metric.

## Test campaign (RTX 4080, 10 configs)

All: vocab 2048 (except C: 4096), ctx_dim 256, ridge-fitness encoder
(rotate + co-occurrence seed, depth 2), ridge-seeded nodes, cluster split,
32k samples (A–C: 24k), encoder 200 gens / pop 150 / 4k samples (A–C: 100/-/2k).

| run | window | embed | layers×bf | acc | bigram | Δ pts |
|-----|--------|-------|-----------|-----|--------|-------|
| A | 16 | 24 | 4×4 | 7.7% | 15.5% | −7.9 |
| B | 16 | 24 | 5×4 | 7.8% | 15.5% | −7.7 |
| C (v4096) | 24 | 24 | 4×4 | 8.3% | 13.5% | −5.2 |
| D | 16 | 64 | 4×4 | 10.6% | 16.7% | −6.1 |
| E (flat) | 16 | 64 | 0 | 1.7% | 16.7% | −14.9 |
| F | 16 | 96 | 3×4 | 11.2% | 16.7% | −5.4 |
| G (no speed) | 16 | 96 | 3×4 | 11.7% | 16.7% | −5.0 |
| H (no speed) | 16 | 128 | 3×4 | 12.2% | 16.7% | −4.5 |
| I (no speed) | 4 | 128 | 3×4 | 11.5% | 15.7% | −4.2 |
| **J (no speed)** | **8** | **128** | **3×4** | **12.7%** | **14.9%** | **−2.15** |

Generation (J, temp 0.5): *"in the world of the right of the world, and she
went not, and the poor of the gentle, … and the table of the fire of the
history"* — grammatical phrase structure, zero junk tokens. Byte-level at the
same effort produced letter salad.

## Findings

1. **Embed dim is the binding constraint at word level.** 24 dims cannot
   carry which-of-2048-words identity (the entire bigram signal); every
   doubling helped: 7.7 → 10.6 → 11.2 → 12.2 pts.
2. **Trap: `encoder_speed_generations` defaults to 40 in the trainer.** The
   speed/Occam phase degraded the evolved encoder 0.148 → ~0.10 ridge acc
   before tree building. UI runs are safe (checkbox off sends 0); scripts
   and sweeps that omit the key silently eat a big regression.
3. **Window 8 beats both 4 and 16** at this capacity — enough context to use,
   not so much that the mixer dilutes the recent words.
4. **The remaining −2 pt gap is structural**, same lesson as the I2
   compression experiments: an exact count table (bigram) is hard to match
   with a compressed continuous encoder + linear readout. The encoder's own
   ridge ceiling (~15%) sits just below bigram.
5. **Layers 0 (flat 2048-way specialist) collapses** (1.7%): GA over a 526k-
   parameter leaf genome regresses away from its ridge seed. Use routed
   trees at word level; don't use layers 0.

## Rounds 5–6: evolved embedding dimension (2026-07-05)

| run | mechanism | evolved dim | acc | bigram | Δ pts |
|-----|-----------|-------------|-----|--------|-------|
| K (evolve-embed, drift) | grow/shrink 1 column | 64 → 66 | 12.0% | 14.9% | −2.9 |
| **L (mixed-dim seeding)** | seeds at [32,64,128,256] | 64 → **256** | **13.2%** | 14.9% | **−1.67** |

One-column-at-a-time growth has no fitness gradient (a fresh random column
helps only after it learns something) — the population drifted 64→66.
**Mixed-dim seeding** builds a real heuristic genome at each capacity level
so selection compares levels head-to-head: the whole population converged to
256 within 10 generations and produced the campaign's best result. The
sidebar embed clamp was raised to 256 accordingly (best run:
`20260705-000924-tree-1ce73a`).

## Next levers (in expected-value order)

- **Bigram/count backoff at the leaves** — blend leaf scores with the count
  table (counts where context-1 suffices, tree where longer context pays).
  This is how word level should *pass* the baseline, not just approach it.
- Longer encoder training (300+ gens, 8k fitness samples, pop 200+) — the
  measured byte-level recipe; the encoder is still the ceiling.
- Word-level `context_dim` > 256 (needs the clamp raised) and/or evolved
  dims.
- Larger corpus — the tokenizer caches per vocab size and scales linearly.

Best config to reproduce (tree page): Token level=word, vocab 2048,
window 8, embed 128, ctx 256, layers 3, bf 4, pop 60, gens 60, samples 32000,
encoder: ridge fitness + rotate + seed, 200 gens, pop 150, 4000 samples,
speed phase OFF, ridge seed + cluster split, GPU.
