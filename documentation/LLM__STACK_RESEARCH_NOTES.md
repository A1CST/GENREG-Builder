# Alternating-Fitness Stack — Research Notes

## Established so far

- **FFN with CE fitness works against full ridge head.** Trained FFN
  shifts log-p by +2.0 nats (correct token ~7.4x more likely) and
  boosts top-5 from 9.8% to 18.3% on held-out. Alpha evolved from 0.05
  to 1.35 naturally, so the layer is genuinely contributing.
- **Alternating fitness works in principle but pairing matters.** Stage 2
  attention with top-1 fitness on top of the CE FFN improved top-1 by
  +0.95pp but dragged top-5 down by -3.2pp. Each layer achieves its own
  target and actively fights the layer below when the targets disagree.
- **SVD-64 head alone outperforms full ridge alone.** This is the
  unexpected result. SVD-64 gives 9.7% top-1 / 21.1% top-5, full ridge
  gives 5.3% / 9.8%. Low-rank approximation is acting as a regularizer.
- **CE FFN (trained against full ridge) is incompatible with SVD-64.**
  FFN writes into directions the SVD-64 reconstruction doesn't preserve,
  so the combined prediction is worse than either alone.

## Plan (in progress)

### (2) FFN trained directly against SVD-64 — RUNNING
Same architecture and CE fitness as the stage-1 FFN, but the target
head during training is the SVD-64 reconstruction (the one that ships
in `github_repo/checkpoints/predhead.pkl`).

Expectation: alpha should grow to a meaningful value and the combined
(FFN + SVD-64) should outperform SVD-64 alone on held-out log-p/top-5,
because the FFN is free to move output into directions the SVD-64 head
actually uses.

If this works, the FFN + SVD-64 pair becomes a valid addition to the
shipped repo.

### (X) KL-divergence attention from n-gram teacher — DONE, NEGATIVE FOR GENERATION

Re-trained attention with KL fitness against 2/3/4-gram cascade as soft
teacher (hot-started from CE attention). Teacher built only at
word-target positions with word-only context (n-grams were built on a
space-stripped stream).

Metrics moved the right direction across the board:
| | KL-fit | CE | top-1 | top-5 | alpha |
|---|---|---|---|---|---|
| gen 0 (CE hot-start) | -8.585 | -8.969 | 9.9% | 21.7% | 1.00 |
| gen 400 (final) | -8.324 | -8.819 | 10.4% | 24.0% | 1.10 |

+0.26 KL nats closer to teacher, +0.15 CE nats, +2.3pp top-5. Evolution
raised alpha slightly.

**BUT generation is still function-word soup.** Compared to CE attention,
KL attention spreads mass across more function words (of/was/in/are/had)
instead of piling onto "and", but still produces no content words and no
spaces. Example (prompt "of valkyria chronicles ii valkyria chronicles iii was"):
- CE:  "...andandandandandandandandandand..."
- KL:  "...ofinofofandwasandofofandwasofandandwasandandandwaswaswas..."

More diverse but equally useless. Even with chars allowed (no ban), 66
(space) never gets sampled — the ridge head just doesn't produce it in
the top-K.

**Conclusion: the ridge head is the generation bottleneck, not the
attention.** This confirms the rank-sweep finding (SVD-1 ≈ full-rank on
top-1). Better attention features don't help if the head can't express
them. Path forward is a genuinely better head with capacity to produce
context-sensitive distributions (evolved predhead didn't clear the bar
either at +0.21 nats).

Checkpoint: `components/attention/checkpoints_attn_kl/attn_kl_gen_00400_final.pkl`

### (1) Better stage-2 fitness — NEXT
Two-stage stack where both stages get CE (pure stacking) instead of
CE + top-1. Removes the fight. Open question: does a second CE stage
add anything or does the first FFN already saturate the CE target?

Alternative stage-2 fitnesses to test:
- top-K recall (K=20 say) instead of top-1 argmax
- perplexity on a held-out slice
- selective accuracy: top-1 only where stage 1 was wrong
  (boosting-style: make the next layer fix what's left)

### Option 2 — FFN trained against SVD-64 — DONE, NEGATIVE
log-p climbed +1.86 nats but top-1 dropped -2.5pp and top-5 dropped -2.95pp.
Same pathology: CE fitness alone concentrates probability mass on a
small set of tokens that includes the correct one but pushes argmax
toward a competing wrong token. CE is necessary but not sufficient.
Possible fix: hybrid fitness like `log P(correct) - margin_penalty *
top1_distance`, or just `log P(correct) + lambda * top1_correct`.

### Option 1 — Pure CE stacking — DONE, NEGATIVE
Stage 2 attention with CE fitness (same as stage 1 FFN).
Evolution drove alpha to 0.01 (floor) within 10 gens. Log-p barely
moved (+0.013 nats). The stage-1 FFN already saturated what CE can
extract from the frozen attention features, so a second CE layer has
zero slack to improve.

### Option 4a — Skip from raw embedding — DONE, NEGATIVE
Stage 2 attention with input = ffn_out + skip_gain * raw_embed.
Both alpha and skip_gain evolved. Result:
- alpha pinned to 0.01 (attention layer off)
- skip_gain climbed to 1.50 (clamp ceiling, wants more raw-embed)
- log-p +0.025 nats (noise)
- top-1 -0.6pp, top-5 +0.2pp (noise)

Evolution is using skip_gain as a "just pass through raw embeddings
harder" knob. The attention layer adds nothing.

### Generation test (the actual goal)
Integrated CE FFN into github_repo and ran A/B with vs without on the
same seed across 3 prompts and 4 ngram_weight values:

- ngram=0 (attention path only): garbage in BOTH conditions
  ("is that and is that and is that ...")
- ngram=1 (pure n-gram, FFN unused): coherent English
  ("house where lody as she slept in the th quarter to ...")

The FFN improves CE/log-p metrics but does not improve generation
quality. Removed from repo again.

The reason: at sampling temperature, what matters is the SHAPE of
the distribution across many tokens. Our attention+head produces a
distribution where the top several tokens are all "the/of/and/is",
regardless of context. The FFN sharpens that distribution toward
the correct token slightly but the alternatives are still common
function words. Sampling under temperature picks one of those
common words ~95% of the time, producing function-word soup.

Fix would require either:
- A fundamentally different head that produces context-sensitive
  distributions (not just "what word is most likely overall")
- Or attention features that vary more with context

The current frozen 2-layer attention was trained on cloze, which
rewards "guess what word fills this gap" — and "the" is a great
guess in many contexts. So the features it produces are biased
toward predicting common words. That bias propagates to everything
downstream.

### Option 1 (re-evolve attention from scratch with CE) — DONE, BIG WIN

Replaced the cloze-trained 2-layer attention with a single CE-trained
layer. Same eval data, same ridge head, same char-ban.

| pipeline | log-p | top-1 | top-5 |
|---|---|---|---|
| no attention (embed+posenc only) | -10.83 | 9.02% | 18.27% |
| cloze 2-layer (current ship) | -10.86 | 5.22% | 10.22% |
| **CE 1-layer (new)** | **-8.93** | **9.02%** | **20.32%** |

**The cloze-trained attention was actively HARMFUL.** It dragged top-1
from 9% to 5% and top-5 from 18% to 10%. Replacing it with one CE-
trained layer recovers the no-attention baseline AND adds +1.9 nats
log-p and +2pp top-5.

This explains why every "stack on top" experiment failed: the cloze
attention features were so biased toward common-word predictions that
nothing built on top could escape that bias. Once the substrate is
trained for the right task, the picture changes.

#### Stacking a second CE layer
Trained a second attention layer with same CE fitness on top of the
first. Result on a different sample (windows differ between runs):
- L1 alone: -9.33
- L1 + L2: -9.19 (+0.14 nats, alpha=0.074)

L2 adds modest improvement (~7% of L1's gain). Alpha stayed small,
suggesting most of the gain was already captured. Two CE layers is
the sweet spot; deeper probably does not pay off without architectural
changes.

#### What this means for the shipped repo
The FFN-on-top-of-cloze-attention combo we shipped was compensating
for bad attention features. With CE attention, that compensation is
unnecessary. The new shipped pipeline could be:
  embed -> posenc -> CE attn L1 -> CE attn L2 -> ridge head
Smaller (one fewer component), better (-8.93 log-p instead of -8.96
from cloze 2L + CE FFN), and conceptually cleaner.

### KEY TAKEAWAY: the frozen stack is mined out
Across four stage-2 attempts (top-1, CE, pure-CE-stacking, skip-from-
raw-embed), every new layer above the CE FFN either contributes nothing
(alpha -> 0) or actively hurts some metric when forced to contribute.

This is not a property of attention specifically; it is a property of
sequentially freezing layers and expecting the next one to pick up
what the last one missed. Once a layer fully optimizes its fitness on
the available features, there is no residual signal for the next layer
to find. The "boosting intuition" (stacking layers covers residuals)
doesn't hold when the lower layer already extracted all its CE objective
allows.

Paths forward that don't involve "stack another layer":
- Re-evolve the 2-layer attention WITH CE fitness directly (instead of
  cloze + our post-hoc CE FFN). The attention layer itself should be
  CE-optimal, not the FFN on top of it.
- Scale the existing ridge head / embedding. Small, low-capacity
  components saturating is exactly why stacking can't help.
- Curriculum on the existing layers (train them with progressively
  harder/different targets).

### (4) Skip / cross-layer connections — USER REQUEST
- Sparse connections between FFN and attention layers
- Downstream attention layers see outputs of earlier attention layers
  (DenseNet-style residual cross-skips)
Both worth trying; the second is the more impactful change since
attention currently only sees the immediate previous block.
Simplest version: stage 2 attention input = ffn_out + alpha * raw_embed
(skip from the very bottom). Lets deep attention see token identity
directly.

### (3) Why SVD-64 beats full ridge — ANSWERED, REGULARIZATION
SVD rank sweep results (top-1 / top-5 on held-out, char-banned):
- rank 1:  8.88% / 13.77% (best top-1)
- rank 2:  8.74% / 13.59%
- rank 4:  8.20% /  9.47%
- rank 8:  5.07% / 10.46%
- rank 64: 4.71% /  9.51% (currently shipping)
- full:    4.71% /  9.15%

The dominant direction of the ridge predicts the majority class, which
gets ~9% top-1 baseline. Higher ranks add noise that hurts argmax.
**The ridge head is barely above the majority baseline.** The head is
the bottleneck, not the attention or the FFN.

Implication: SVD-64 isn't actually doing anything clever; it's
predicting "the" most of the time. We need a genuinely better head.

### (3) Why SVD-64 beats full ridge — ANSWERED ABOVE; legacy note follows
Full ridge gets 5.3% / 9.8% top-1/top-5. SVD-64 gets 9.7% / 21.1%.
That should not happen with a faithful low-rank approximation.

Hypotheses:
- **Regularization via truncation.** The full ridge overfit to its
  training procedure; dropping the bottom 700+ singular directions
  removes overfit noise and the remaining subspace is cleaner.
- **Head-attention mismatch.** The full ridge was fit on features from
  a different attention distribution than the one evaluated here. The
  top-64 singular directions are more robust to that shift than the
  tail.
- **Char-token interference.** The ridge has learned capacity dedicated
  to char tokens (ids < 96). We ban those at inference. The full ridge
  may still leak probability into neighboring word directions through
  its discarded rank; SVD-64 truncates that leak.

Diagnostic to run:
1. Try SVD-128, SVD-256, SVD-512 to see where the curve peaks
2. Measure top-1 / top-5 as a function of SVD rank
3. If SVD-256 or SVD-512 is even better, the story isn't "SVD is
   regularizing" but "the tail of the ridge was catastrophically bad"
4. If the peak is at low rank (16 or 32 say), regularization is real

This is worth understanding before training more components on top,
because we might be able to IMPROVE the head by explicit regularization
during the ridge fit, without needing any evolution.
