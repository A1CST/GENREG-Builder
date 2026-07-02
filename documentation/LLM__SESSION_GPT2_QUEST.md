# Session: Pushing Toward GPT-2 Level Sentences

Goal: end-to-end gradient-free LLM producing GPT-2-quality text.

## Summary

Tested three "fix attention stacking" approaches, discovered fundamental
architectural issues in generation pipeline, fixed n-gram tables,
produced first coherent fragments.

**Current best generation quality:** short coherent phrases. Far from
GPT-2 multi-sentence coherence.

## Attention-stacking tests (all failed to break past 1-2 layers)

| approach | file | result |
|---|---|---|
| v2 residual cloze | `components/attention/genreg_attn_wiki_v2_residual.py` | Layer 1 degraded (-2.3pp). Attention is global; fixing 13% of residuals requires changing features at all 87% of positions. |
| v3 gated residual | `components/attention/genreg_attn_wiki_v3_gated.py` | Evolution pushed alpha to max clamp (2.0). Gate was an option, not a constraint. |
| v4 curriculum (rare/long/wide) | `components/attention/genreg_attn_wiki_v4_curriculum.py` | Rare-word gave +0.0098 once, -0.0282 once — noise-sensitive. Long-range/wide-mask collapsed v1's multiplicative gates (fit=0). |
| v5 multi-objective | `components/attention/genreg_attn_wiki_v5_multiobj.py` | Catastrophic collapse (0.833 → 0.119) — training/measurement mismatch bug. |

**Conclusion:** sequential frozen attention has a hard ceiling at 1-2
layers with cloze-trained fitness. None of residual/gating/curriculum
reliably produced a third useful layer.

## FFN component — built but buggy

`components/ffn/genreg_ffn_wiki_v1.py` — per-token FFN (D=768 → H=2048 → D)
with per-neuron evolved activations and residual gate.

**Bug:** FFN evaluation reports cloze 0.85 even though masked-position
features are provably CONSTANT (cos=1.000 across all masked positions,
verified by debug script `/tmp/debug_ffn.py`). The ridge probe somehow
extracts >random accuracy from constants. Not diagnosed; results
unreliable.

## Generation pipeline — FIXED n-gram bug

**Root cause found:** corpus tokenizer inserts SPACE tokens (id 66)
between words. Runtime tokenizer (`.split()`-based) does NOT. So n-gram
table keys `(word1, space)` never matched runtime lookups `(word1, word2)`.

**Fix:** `components/predhead/rebuild_ngrams_nospace.py` rebuilds tables
from the space-filtered stream. New stats:
- Bigram: 51,567 contexts (same count as before; each now has 1000x
  more real continuations)
- Trigram: 2,327,139 contexts (was 114,797 — 20x more)
- "the" bigram: 28,003 continuations (was 10)
- ("on","the") trigram: 4,894 continuations (was not in table)

## Current best generation

Config: `temp=0.3, top_k=10, ngram-weight=1.0, freq-penalty=0`
(pure n-gram from corpus, no attention/head signal)

Sample outputs:
```
the president of the th division to create their map to create and produce
  a new and different lyrics were inspired by new line of scrimmage and
  was the most significant of

during this time she became executive officer in his final in and the two
  she and the second the u boat flotilla later she would only need one

he played for the first game was later ported over years later he later
  he hit two game and ported from to from

the film was directed by john f l i n s e e d the first game as
  s i s t e the two countries have the right wing
```

See `CURRENT_GENERATION_SAMPLES.txt`.

**Quality honest assessment:** 2-3 word coherent fragments. Noun
phrases and simple verb-phrases work. Multi-sentence coherence
doesn't. When trigram has no strong continuation, output COLLAPSES
INTO CHAR MODE (single letters like "s i s t e") — this is a critical
failure mode.

## What's blocking GPT-2 quality

1. **Attention trained for cloze, not autoregressive.** Cloze sees
   future context; generation doesn't. Causal checkpoints were trained
   with cloze fitness on causally-masked inputs — still not
   autoregressive next-token training. Need to re-evolve with
   `AUTOREG_FITNESS=True` using the ridge head as the next-token target.

2. **Predhead bad for sampling.** Both ridge (51%) and v7 evolved (48%)
   produce fine top-1 accuracy but terrible sampling distributions.
   When sampled with temperature, they return random rare words. This
   is a KL-divergence issue: the head was scored on argmax accuracy,
   not full-distribution quality.

3. **Char-fallback degeneracy.** Character tokens (ids 4-70) are in
   vocab. When trigram fails, bigram/ridge fall back to these. Need
   to penalize chars at inference OR remove them from the sampling
   pool unless the previous token is also a char.

4. **No FFN in stack.** Missing the per-token nonlinear capacity GPT-2
   uses between attention blocks. v1 has bugs; need proper debug.

5. **Only 2 attention layers.** GPT-2 small has 12. Our stacking hits a
   ceiling with cloze fitness; need different approach to go deeper.

6. **Tokenizer word/char mix.** The vocab includes both words and
   single chars. Sampling mixes them chaotically. A clean word-only
   vocab OR proper BPE would help.

## Ordered fix list (next session)

1. **Fix char-fallback** at inference time: set logits for char tokens
   (ids 4-70) to -inf unless prev token is also char. 1-line fix,
   immediate improvement.
2. **Retrain attention with AUTOREG_FITNESS.** Use ridge head as
   next-token target. 1 layer, 500+ gens.
3. **Debug FFN v1.** Either the "cloze 0.85 on constants" is real
   (and probe exploits something subtle) or my forward has a bug.
4. **Add 5-gram fallback** for when trigram misses. Have
   `ngram_tables_5.pkl` already.
5. **Sampling head** — evolve a head with temperature-aware fitness
   (score full distribution KL, not just top-1). The current heads
   are optimized for wrong metric.

## Files produced this session

- `components/attention/genreg_attn_wiki_v2_residual.py` ... `_v5_multiobj.py`
- `components/ffn/genreg_ffn_wiki_v1.py`
- `components/predhead/rebuild_ngrams.py`, `rebuild_ngrams_nospace.py`
- `test_e2e_generation.py`
- `CURRENT_GENERATION_SAMPLES.txt`
- This file.

## Fix applied: char-fallback suppression

`generate_wiki.py`: after n-gram blend and freq penalty, hard-set
logits[:75] = -1e9 to ban char tokens (ids 0-74 are specials + single
chars). Prevents the char cascade when word-level paths have no strong
continuation.

## New finding: attention/ridge HURTS generation

Direct ablation on same prompts, temp=0.4, top_k=20:

| ngram_weight | "the king sat on the ..." | "during the second world war ..." |
|---|---|---|
| 0.0 (pure attn+ridge) | "two second under where city new york not when show no storm" | "other not she city where new she show game she two new no" |
| 0.3 | "national new been game or second city other show not" | "his when when his his other you she game two" |
| 0.7 | "game other road not game city game th show jordan" | "national under to when also king th when not new" |
| **1.0 (pure n-gram)** | **"billboard music was catchy tracks in keeping plot device is designed"** | **"rooms his debut against lord nelson rockefeller began as one of her work"** |

The attention/ridge path produces common-word soup under sampling.
The ridge head was optimized for TOP-1 cloze accuracy (51%), not
full-distribution quality. At temp 0.4 with top-k 20, the non-top
positions matter a lot — and there the ridge produces noise.

**Blending n-gram at anything less than 1.0 CORRUPTS the n-gram's
real statistical structure.** Pure n-gram is the best current path.

## Current best samples (pure n-gram, temp=0.4, top_k=20, char-banned)

```
the king sat on the south side rowing association between sight by this
  led to him after several batches several of its original american nbc
  television network between january

during the second world war began again to be built by henry alfred died
  nearly three year of junior lecturer responsible to an underground
  british forces

the film was directed by the city and from september while heading
  northwest after their departure for america after being hit or in part
  for him for their september october

it was the first to attempt an estimated the damages caused problems
  charles edward and stede to their problems caused it with and against
  australian war between british royal

he played for the film as belonging the th brigade were deployed and
  with her family did it is her best work possible values family is
  decided in late
```

Quality: ~80% grammatical, fragment-level coherent. Zero long-range
semantic coherence. "a" random walk through trigram transitions with
occasional repeat loops.

## Updated bottleneck analysis

The attention stack is fundamentally miscalibrated for generation:
- Trained on cloze (reconstruct masked positions)
- Uses top-1 accuracy as fitness signal
- Produces features that don't sample well

**Two options:**

A. **Re-evolve attention with DISTRIBUTIONAL fitness.** Not top-1
   accuracy but perplexity / KL divergence on a held-out set. This
   rewards genomes that produce sharp-but-spread distributions where
   the correct answer has high probability AND non-correct distribution
   has mass on plausible alternatives.

B. **Replace the LM head with something that produces good samples.**
   Current ridge is a least-squares fit — it's optimized for MSE on
   one-hot targets, not for sampling. A cross-entropy / KL head would
   be much better but requires gradient-free optimization of something
   harder than MSE.

Option A is probably more tractable in a gradient-free framework.

## Status

Char-ban fix shipped. Pure n-gram now produces consistent English-ish
fragments. Attention/head path is a dead-end for generation as
currently trained. Next: re-evolve attention with distributional
fitness, test again.

---

## Further fixes applied

### 4-gram + 5-gram tables built

`components/predhead/rebuild_ngrams_5_nospace.py` — built from
space-filtered stream.
- 4-gram: 879,125 contexts (min_count=2 prune)
- 5-gram: 585,014 contexts (min_count=2 prune)
- File size: 108.7 MB

### Fallback cascade in predict_next

Modified `generate_wiki.py` to try 5→4→3→2-gram in order. Each tier
only fills slots the previous didn't. Real long-context coherence.

### Char cutoff extended to 96

After blend, `logits[:96] = -1e9` bans all 4 specials + 62 chars
+ ~30 punctuation. Words start at id 96.

## Current best generation

Config: temp=0.4, top_k=20, freq=0, ngram=1.0, char-banned, 5-gram cascade

```
the king sat on the floor fasting aaliyah and his second son qedar
  jerome in its opening weekend monsters to become chief by her
  second and its associated publicity from until and united

the film was directed by anthony freud aa mounts these events were
  incorporated elements incorporated his wife anne hathaway who later
  recalled the words is short the user and takes its title one review

the battle of waterloo was less important agent for an agent the game
  the lowest level the aircraft is in lowest walks allowed at intervals
  over for military band over the game and it became

she was born in on an american on to lose interest is often confused
  the number and severity it often is for and won six world in and
  his complaint from number eight being

the album was released from jail in in an interview to his son cecil
  sharp curve onto to its first time to develop three years and as
  in and is buried as she put

during the second world war but we can we know and that was not
  observed again not in in and the show ended up playing style he
  displayed the hinomaru on to the united he
```

**Honest quality:** word-level coherent English fragments. Some
real facts ("anne hathaway", "cecil sharp curve", "waterloo was less
important agent"). Proper names, verb phrases, prepositional phrases
all work. BUT:
- No sentence-level coherence: "the film was directed by anthony freud"
  is plausible, but "these events were incorporated elements incorporated
  his wife anne hathaway" jumps topics
- Repetition ("incorporated ... incorporated", "often is ... often
  is")
- Semantic drift over 20+ tokens

## Gap to GPT-2

- GPT-2 produces paragraph-level coherent text. We produce phrase-level.
- GPT-2 maintains topics across 200+ tokens. We lose topic by 10.
- GPT-2 has rarely-repeated words. We loop common words.

Gradient-free path to close this gap requires:
1. **Distributional attention fitness** — not top-1, but full KL-divergence
   against corpus next-token distributions.
2. **More attention layers** trained sequentially (or simultaneously via
   different scheme).
3. **An FFN** between attention and head (v1 has bug).
4. **Topic/state tracking** across sentences. Current n-gram has zero
   memory beyond 4 tokens. Need a mechanism that carries semantic
   state longer.

Topic-bias via prompt-embedding mean (tested) did not help — common
function words dominate similarity. Need learned topic signal, not
raw embedding similarity.

## Session files

- `SESSION_GPT2_QUEST.md` — this file
- `GEN_BEST_FINAL.txt` — final best samples
- `CURRENT_GENERATION_SAMPLES.txt` — intermediate
- `GEN_BLEND.txt` — attention-blend ablation
- `components/predhead/rebuild_ngrams_nospace.py` — 2/3-gram fix
- `components/predhead/rebuild_ngrams_5_nospace.py` — 4/5-gram build
- Modified: `generate_wiki.py` (topic_weight, char cutoff, 5-gram cascade)
