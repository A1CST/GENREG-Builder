# Tokenizer component findings

## Two independent tries, two different failure modes

### tok_v2 (per-word recurrent, argmax output)

- Recurrent byte encoder, walks each word byte-by-byte
- Output head H → V_out=4096, argmax selects token_id
- Fitness = uniqueness on 2048-word probe

**Final: 54% uniqueness, ~2200 active token IDs / 4096 slots**

Plateau driver: H=48 hidden clusters similar words. argmax concentrates
output into the most popular buckets. Still climbing slowly when
restarted — may reach 60% with more gens, likely capped under 70%.

### tokenizer G (stream recurrent, hash-bit output, proven design)

- Continuous stream inference: 384-768 word stream with 20% padding
- Proteins (decay + momentum + integral + raw) maintained across stream
- Output: 9 sign-binarized hash projections → bit-packed int [0, 512)
- Fitness = discrimination² × consistency × energy_bonus, penalizes
  non-silence on padding

**Training peaked: 76% uniqueness on 175 unique words per stream
(gen 3000, fit=0.2256)**

**Fresh-stream inference: 8.7% uniqueness on 2000 unique probe words.**

The training fitness does not generalize to corpus inference. Two
reasons:
1. Training streams repeat each unique word 2-5× (MIN_REPEATS /
   MAX_REPEATS); cascade state gets pulled toward attractors that
   stabilize within the short repeated-word stream. A fresh stream
   of 2000 distinct novel words has no repetition.
2. `silent: 0/154` the entire run — genomes never learned to output
   0 on padding. Energy system should have punished this (cost=3.0
   per padding slot) but everyone stayed alive at energy 800+. The
   padding-silence signal never became meaningfully pressure.

## Honest diagnosis

Both designs are producing ~100-200 effective token IDs when used to
tokenize a real corpus of 148k unique words. That is 700-1500×
collision rate. Unusable for a word-level LM downstream.

The G design's hash-bit output is the right idea in principle — each
bit is an independent decision, so the output distribution is
naturally spread. But the training distribution (stream of 175
repeated words + padding) and the inference distribution (9M words
of real text) are far enough apart that the learned cascade state
produces garbage at inference.

## Options from here

**A. Train a proper per-word deterministic tokenizer.**
Combine tok_v2's per-word recurrent byte encoder with G's hash-bit
output. No stream, no cascade state. Each word hashes independently.
Fitness = discrimination across a large word batch. Longer training
(5000-10000 gens). Might hit 70-80% uniqueness on diverse vocabulary.

**B. Pivot back to char-level.**
We have a working char-level LM (A_101 @ 34% heldout) and a working
char-level embedding component (embed_03 beat SVD by +0.92pp). The
component-first paradigm can be validated on char-level before
taking on the tokenizer problem.

**C. Accept 50-70% uniqueness tokenizer as-is.**
Tokenize corpus with whatever we have, build word-level LM on top.
Collision rate will blow up the prediction task — rare words collapse
into noisy signal. LM quality will likely be poor but we'd have an
assembled pipeline end-to-end.

**D. Redesign tokenizer from first principles.**
Something like: learn a byte-pair-encoding-analog where byte pairs
are merged into subword tokens based on frequency + separability.
Fundamentally different GENREG setup. Big commitment.

## Recommendation

Option B for now. Option A as parallel work if resources allow.
Option C is waste without a better tokenizer; option D is a research
track of its own.

Neither G nor tok_v2 is ready to feed downstream. The word-level
track requires a tokenizer that clears >70% uniqueness on corpus-scale
inference before downstream components have a signal to work with.
