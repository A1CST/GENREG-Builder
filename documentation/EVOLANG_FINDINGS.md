# EvoLang — Stage 1 findings (char-level, Gutenberg corpus)

**Date:** 2026-07-06 (autonomous run). **Setup:** evolution-native char LM —
context K chars → evolved embedding → tanh(H) → 37-way char logits, bred by
tournament + elitism + energy homeostasis, soft log-prob fitness, **no
gradients**. Corpus: `project/EEC-main/engine/corpus.txt` (~48.6M chars).
Every number is **held-out** `val_ppl` (train on first 90%, measured on a fixed
4096-window sample from the reserved tail). Population 200, minibatch 256.

A char-level perplexity of 37 = uniform random over the charset; lower = real
structure. These tiny 1-layer genomes reach ~12, so they capture genuine
sub-word statistics but are nowhere near word-level fluency (expected — one tiny
net, one part of a language model).

## 1. Capacity sweep (1200 generations each)

| context K | hidden H | **val_ppl** | train_ppl | note |
|---|---|---|---|---|
| 4 | 32 | 14.64 | 11.33 | |
| **4** | **48** | **13.84** | 11.26 | **winner** |
| 6 | 32 | 14.15 | 11.30 | |
| 6 | 48 | 16.07 | 12.57 | widest train/val gap → overfit |
| 8 | 32 | 14.31 | 11.73 | |
| 8 | 48 | 14.94 | 12.70 | |

**More context does not help this genome.** K6/K8 are no better than K4 and the
larger (K6/K8, H48) combos overfit — the tiny single-layer readout can't
actually exploit a longer window, it just gains parameters to memorise the train
region. The useful lever is **hidden width at short context** (K4: H32→H48 helps).
Takeaway: for a genome this small, spend capacity on width, not context length.

## 2. Novelty constraint A/B (winner K4H48, 1800 generations)

| novelty weight | **val_ppl** | train_ppl | note |
|---|---|---|---|
| **off** | **13.19** | 10.61 | best |
| 0.3 | 13.67 | 11.12 | |
| 0.6 | 14.21 | 11.40 | |
| 1.0 | 18.11 | 15.63 | collapsed — plateaued at 769 gens |

**Novelty monotonically hurts perplexity, and at weight 1.0 it wrecks the
model.** This is the honest, expected result: novelty rewards *token variety*,
which trades directly against *next-char accuracy* — a textbook Goodhart lever.
It is **not** a perplexity improver and shouldn't be sold as one.

Where it earns its keep is the failure mode it was built for: repetitive collapse
("the the and and"). On the tiny toy corpus that collapse was severe and novelty
visibly fought it; on 48.6M chars of real English the base model already inherits
the corpus's variety, so novelty's benefit is small and its ppl cost dominates.
**Keep it off by default; reach for it only when a run is visibly collapsing into
a few tokens, and keep the weight low (≤0.3).**

## 3. Long run — ceiling probe (winner K4H48, 6000 generations)

**val_ppl 12.25** (train 8.99). Sample (temp 0.6):
`" me  waeve in  thend, fe nd the  we s th ande a  "ar  the    whas  so"`

More generations keep helping (val 13.84 @1200 → 13.19 @1800 → **12.25 @6000**),
but the train/val gap widens (8.99 vs 12.25) — the extra generations increasingly
memorise the train region. Real words surface at the char level ("the", "in",
"and", "a", "we") with word-like fragments between them, but there is no
word-level grammar. **~val_ppl 12 is roughly the char-level ceiling for a single
tiny evolved genome here.**

## Verdict & next levers

A single tiny evolved char net recovers real sub-word statistics (37 → ~12 ppl,
gradient-free) but plateaus well short of fluency — consistent with the whole-lab
thesis that one genome does *one part* of a language model. The bottleneck is
**model class**, not training budget: at 6000 gens we're memorising, not
generalising better.

Honest next directions (none of them "add gradients" or "add an n-gram table"):
- **Composition, not scale** — the lab's recurring answer. Several specialised
  genomes (per-context-type, per-position, or a routing organism) rather than one
  net asked to do everything. This is the EvoLang analogue of DiffEvo's
  one-easy-denoiser-per-level decomposition.
- **A recurrent/stateful genome** (carry an evolved hidden state across chars)
  so context isn't capped at a fixed K window — the sweep shows widening the
  *window* is a dead end, but evolved *state* is a different mechanism.
- **Landscape shaping over raw log-prob** — the Intelligence-Engine bet: pressures
  that make generalisation (not train memorisation) the only stable attractor,
  since more generations currently buy memorisation.

## Reproduce
`scratchpad/evolang_experiments.py` (battery), `scratchpad/results.json` (raw).
Defaults updated to the winner (K4, H48, E12) on `/evolang`.
