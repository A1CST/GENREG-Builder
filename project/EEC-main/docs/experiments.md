# EEC ceiling-breaking experiments

Goal: get **raw** next-word accuracy past the `<unk>` unigram ceiling
(smoothed ~27.5, raw peak ~31 at vocab=8000) using only gradient-free
environment construction (energy / time / depth / novel dynamics). No gradients.

Floor reference: always-predict-`<unk>` ≈ 5.1% ≈ raw 26/512. Layer-1 frozen
substrate solo ≈ raw 21-22 (varies with shuffle).

Verdict legend: ✅ broke ceiling · ➖ matched floor · ❌ worse / inert · 🔬 inconclusive

| Label | Gambit | Result | Verdict |
|-------|--------|--------|---------|
| STACK_A | Residual depth-stack: frozen L1 + evolvable L2, near-zero skip init | avg==best always; L2 inert (logits2~0.008 vs logits1~1) | ❌ inert |
| STACK_B | Same, but L2 output init at full magnitude (selectable) | still all-`<unk>`; L1 winning logit ~1900, gap to 2nd ~1745 — residual drowned | ❌ saturated |
| STACK_C | No-skip: L2 makes final logits from frozen [emb; h1] (h1 std 0.32, context-rich) | slow cold-start, climbing toward same ~26 ceiling; stopped @gen296 (control) | ➖ confirms reframe |
| RAMP_A | Ramped rarity gate, plateau-triggered | noise kept resetting plateau → never ramped (T=0 @gen266) | ❌ trigger too soft |
| RAMP_B | Same, fixed-cadence ramp + starvation guard + diversity readout | stripped ',' @80, '<unk>' @130 → POP COLLAPSE: monocultured pop all hit credit 0 at once, no gradient, never recovered (uniq 24-29 but all wrong) | ❌ extinction cliff |
| SURP_A | Surprisal-weighted fitness (−log2 p per correct), pow=1; energy normal | no collapse, but converged to bits/hit 4.3 = pure `<unk>`; 4× diff too weak vs energy pull | ➖ matched |
| SURP_B | Surprisal-weighted fitness, pow=3 (rare word ~62× a `<unk>` hit) | bits/hit→215 (predicts ~6-bit word) but raw collapsed 26→7; just moved the constant up the freq axis | ❌ relocates |
| NICHE_A | Fitness sharing: reward for a token instance split among all genomes that get it | pop still covers same ~21 `<unk>` windows; sharing flattened fitness to equal+tiny → weak selection, no bridge | ❌ flattens |
| PMI_A | Conditional-info fitness: Σ p(word\|prev)·surprisal(word) over hits; constant scores ~0 | raw median 26 (ceiling), uniq collapsed 56→23 back to comma; raw-41 spikes were shuffle noise; flickers context but can't hold it | 🔬 search wall |
| VOCAB_A | Small vocab (V=1000) + drop OOV-target windows → tractable output space, no `<unk>` crutch | converged to comma constant (uniq 4-5), acc 8.26% ≈ ceiling 8.45%, never exceeded | ❌ search wall holds |
| NGRAM_A | Feed bigram-predicted next-token as extra INPUT token | stuck ~1% @gen144 — evolution can't learn to route/copy the feature through the MLP | ❌ unusable as input |
| NGRAM_B | Bigram prediction as DIRECT additive logit channel + evolved scalar mix | **19% acc, +12pp over unigram ceiling**, exceeds raw bigram-copy; mix latched ~3.7 instantly, uniq ~45 | ✅ **CEILING BROKEN** |

## NEW CONSTRAINT CLASS — physics of existence (NOT reward shaping, NOT architecture)
Direction correction from researcher: energy & time work because they are
task-agnostic LAWS OF EXISTENCE (cost/resource on living), under which a
capability EMERGES as instrumental to survival — never rewarded, never wired in.
Build MORE such constraints. Pocket: perception cost (active), then memory-rent,
resource-depletion, metabolic-upkeep.

| Label | Constraint | Result | Verdict |
|-------|-----------|--------|---------|
| PERC_A | Perception cost: gate p_i=σ(g_i)/pos, looking drains κ·Σp energy; L=8 | gates FROZEN at 0.5 — g zero-init + relative mutation = calcification (couldn't move). Bug | ❌ bug |
| PERC_B | Same, gates seeded with spread (g~N(0,1.5)) so they're evolvable | _running_ | _pending_ |

## CONCLUSION (reward-shaping / architecture era — superseded as the approach)

13 experiments. The `<unk>`/unigram ceiling is a SEARCH wall, not a reward-shape
or architecture-depth problem:

1. **Depth can't break it** (STACK_A/B/C) — the ceiling is a fitness-landscape
   attractor; stacking re-derives it. Also: can't residual-correct a saturated
   frozen output (winner logit ~1900).
2. **Reward reshaping can't break it** (RAMP/SURP/NICHE/PMI) — every reward is
   maximized by SOME reachable constant. Rarity reweighting just relocates the
   constant; hard cliffs cause population extinction; fitness sharing flattens
   selection; even an ungameable conditional (PMI) reward only FLICKERS context
   then drifts back. Shrinking vocab (VOCAB_A) doesn't help either.
3. **The wall is reachability**: random mutation through embedding→MLP→softmax
   cannot discover/hold context-conditional outputs in a large output space.
   Even handing the answer as an INPUT feature fails (NGRAM_A) — the MLP can't
   learn to route it.
4. **What works**: inject context as a DIRECT additive logit channel with one
   evolved scalar mix (NGRAM_B) → +12pp over ceiling instantly. The mixing must
   be a directly-searchable parameter, not an emergent routing.

This exactly validates and explains the GENREG LM lineage: frozen n-gram
channels + evolved mixing. Energy/time/depth shape HOW WELL a reachable solution
is found and how parsimonious it is — they do NOT make unreachable solutions
reachable. For that you must supply the channel.

**NGRAM_A lesson:** having context info as an INPUT isn't enough — the MLP can't
evolve to route it. The mixing must be a directly-searchable parameter (one
scalar added to logits), exactly the frozen-channel + evolved-mix architecture
of the working LM lineage. → NGRAM_B.

**PMI_A lesson (the big one):** even a conditional reward a constant can't game
only FLICKERS context — the population keeps drifting back to the reachable
constant. The bottleneck is SEARCH: mutation can't find/hold context-specific
outputs in an 8000-way space. Fix = make context reachable (shrink vocab, kill
OOV crutch) → VOCAB_*.

**NICHE_A lesson:** sharing devalues crowding but creates no PATH to context —
no genome breaks away because breaking away requires context-use that mutation
can't reach from the constant. Confirms the wall is REPRESENTATIONAL/SEARCH, not
just reward shape. Conditional reward (PMI) is the one a constant can't game.

**META-FINDING (SURP_A/B):** token-rarity reweighting (count or surprisal, any
exponent) only RELOCATES the constant attractor along the frequency spectrum —
it never induces context, because any single-token objective is maximized by
some constant. The missing ingredient across all runs is DIVERSITY (populations
monoculture, then reward tweaks just move the monoculture). → NICHE.

**RAMP_B lesson:** hard-stripping a crutch from a CONVERGED population = mass
extinction (all lose strategy simultaneously → fitness 0 for all → no gradient).
Cliffs kill. Need continuous pressure (SURP) and/or diversity preservation (NICHE).

## Key reframe (after STACK_A/B/C)
Depth alone CANNOT break the ceiling: predicting a constant most-frequent token
is a fitness-landscape attractor, independent of architecture. STACK_A/B failed
because L1's output saturated (winner logit ~1900, gap ~1745) — unfixable by
residual. STACK_C reuses L1's context-rich hidden but must relearn output and
will re-plateau at the same ~26 ceiling (landscape unchanged). The lever is the
ENERGY/REWARD landscape (RAMP_*), not depth.

## Log

### STACK_A — residual depth-stack (near-zero skip init)
Hypothesis: freezing the unigram layer and evolving a residual L2 on
`[emb; h1]` forces context use; skip-floor init starts at L1 accuracy.
Watch: does `avg` separate from `best`? If `avg==best` forever, L2 is inert
(near-zero W2 never flips an argmax → no selection gradient).
