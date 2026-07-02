# Constraints for English communication — the design

What laws of existence must a world impose so that **communicating in real English is the
surviving behaviour** — not a behaviour we graded toward? "English communication" is not one
capability; it is a stack, each layer demanding its own constraint. This doc outlines the stack,
the minimal sets, the guardrails, and the first test.

Paradigm-correct target:
> An organism that communicates in real English because, in its world, English is the cheapest way
> to stay alive. English usage is **measured, never rewarded**. We read the STATE, not the token.

## The decomposition — each capability axis and the law that forces it

| # | Capability | Law of existence | World physics that imposes it | Status |
|---|---|---|---|---|
| 1 | signal at all | **coupled survival** + ENERGY | speaker & listener share fate; a helpful signal feeds both | validated (0.75 vs 0.25 severed) |
| 2 | grounding (symbol↔world) | **perception** over a referential world | meanings are perceivable world-states the listener must act on; right action = food | validated (corpus grounding 0.77) |
| 3 | English not arbitrary code | **cultural anchor / conformity** | born into a world that already speaks English and won't accommodate you — align or starve | validated at vocab level (acquisition) |
| 4 | whole lexicon not a pidgin | **Zipfian scarcity / criticality** | rare referents are survival-critical (rare = deadly), so every word earns its keep | **the open wall — tested here** |
| 5 | grammar / word order | **channel-capacity limit + relational world** | more distinct events than atomic symbols; events have roles → order must carry the role; a bag is provably insufficient | partial (order emerges; binding weak) |
| 6 | reference across time | **entropy + occlusion** | linguistic state leaks / is hidden across turns, so holding context pays | memory laws validated, not yet wired to language |
| 7 | learned not innate | **mortality + reproduction + iterated transmission + continuity** | each generation re-acquires English from the last under a curriculum ramp; continuity is load-bearing | continuity shown (coma test); iterated learning not yet built |

## Minimal sets (the PO answer — cover each axis once)

- **English vocabulary** = coupled-survival + energy + perception/grounding + cultural-anchor. (≈ done)
- **+ grammar** = add channel-capacity + relational world.
- **+ full fluency** = add Zipfian scarcity. (kills the pidgin ceiling — highest-leverage new law)
- **+ acquired** = add mortality + iterated transmission + continuity ramp.

## Guardrails (what decides whether it is real)

1. **No designed gradient.** Comprehension must pay through the world (partner survives, food
   eaten), never a similarity-to-English score or per-word accuracy reward. Robustness across
   reward *shapes* is the signature of emergence; fragility to a hand-picked grading is the trap
   (see `THEORY_REFINED.md`).
2. **The search-wall (P5) is the real risk for grammar.** Conditional/compositional behaviour
   tends to be unreachable by pressure alone — it needs a *channel* (frozen grounded lexical
   scaffold + evolved composition), exactly the LM-lineage lesson. Plan for a scaffold; don't
   expect grammar to emerge from selection alone.
3. **Energy mandatory; cover each axis once** (don't stack two survival laws).

## Test 1 — Zipfian scarcity / criticality (the pidgin ceiling)

**Claim.** The pidgin ceiling is P1 at the per-word level: a word English-ifies only where *that
word* pays its rent, and `rent = frequency × stake`. Under flat stakes rent ∝ frequency, so only
frequent words clear threshold (pidgin). Make rare referents survival-critical (rare = deadly) and
rent flattens across the lexicon, so the rare tail English-ifies too — escaping the pidgin without
any designed push toward English.

**World.** Real-corpus grounding (K=60 words, real freq + PPMI-SVD embeddings). Residents start with
no innate English (anchor 0) and must align with fixed **native English speakers** to eat
(cultural anchor). Meanings are sampled by frequency (Zipf — frequent things come up more). Each
correct decode pays `stake[word]` energy.

**Conditions (world physics, not rewards):**
- `flat` — every referent worth the same (stake = 1). Predicts pidgin: usage ∝ frequency.
- `criticality` — rare referents are deadly (stake ∝ 1/freq, clipped). Predicts rent ≈ constant →
  full-lexicon English, flat usage-vs-frequency.
- `random` — stake independent of frequency (sanity: high-stake rare words still get learned).

**Readouts (state, not reward):**
- overall english_usage (fraction of meanings a resident *speaks* with the correct English word);
- corr(log frequency, per-word usage) — positive = pidgin tail, ≈0 = full lexicon;
- usage on the RARE half of the vocab — the tail that pidgin abandons;
- per-word usage as a threshold function of `rent = freq × stake`, collapsing all conditions onto
  one curve (the real law).

Implementation: `zipf_scarcity.py` (regime found by `probe_bootstrap.py`: real PPMI-SVD embeddings,
synthetic Zipf frequency spread ~33× over K=24 referents, anchor 0 so English is acquired, not innate).

### Outcome (5 seeds) — the rent law holds; scarcity is necessary but not sufficient

The pidgin reproduces cleanly under flat stakes: usage 0.20, rare-half **0.042**, `corr(logFreq,usage)
= +0.57` (frequent words English-ify, the tail doesn't).

**Criticality (rare = deadly) is a real lever:** overall English usage **0.199 → 0.306 (+54%)** and
rare-tail usage **0.042 → 0.118 (~3×)**. And the deeper claim is confirmed — pooling all three worlds,
per-word usage is governed by **rent = frequency × stake**, not frequency alone (pooled
`corr(log rent, usage) = +0.62`; usage rises monotonically across rent quartiles 0.07 → 0.20 → 0.24
→ 0.46). The pidgin is P1 at the per-word level, as predicted.

**But criticality did NOT flatten the lexicon** — `corr(logFreq,usage)` stayed +0.6. Reason: rent sets
the *prize* for a word, but frequency independently sets *exposure* — how often selection can act on
that word. Rare words become worth learning under criticality, but they are still seen rarely, so they
are learned only partially. **Two separable constraints:** value (rent) and learnability (exposure).

**Implication for the stack:** scarcity/criticality is necessary to make the rare lexicon *pay*, but
full fluency also needs a second axis that raises rare-word **exposure** — i.e. constraint #7
(iterated transmission / a teaching or curriculum pressure that drills the rare words). Scarcity alone
lifts the tail ~3× but cannot finish the job. Charts: `zipf_scarcity.png`.

Later constraints (#5–#7) build on this: #7 is now motivated by data, not just theory.

## Test 2 — #7 exposure / urgency-gated communication: BREAKS the pidgin

`exposure_teaching.py`. The exposure fix, paradigm-correctly: you communicate what *matters*, not what
is merely frequent (alarm calls are about the rare predator, not the grass). The channel is a mixture
`(1-u)·frequency-chatter + u·stake-driven-urgency`; `u` is the new **exposure axis**. Energy is still
`stake[m]`; usage measured, never rewarded.

Result (5 seeds), on top of criticality:

| u | usage | freq-half | rare-half | corr(logFreq,usage) |
|---|---|---|---|---|
| flat (pidgin ref) | 0.199 | 0.356 | 0.042 | **+0.57** |
| 0.0 | 0.306 | 0.495 | 0.118 | +0.60 |
| **0.25** | **0.341** | 0.320 | 0.363 | +0.15 |
| 0.5 | 0.258 | 0.193 | 0.323 | +0.05 |
| 0.75 | 0.277 | 0.141 | 0.414 | −0.29 |
| 1.0 | 0.315 | 0.140 | 0.490 | −0.39 |

The pidgin signature (corr +0.57) **collapses and inverts** (−0.39): rare words end up better-learned
than frequent. Rare-tail usage quadruples (0.042 → 0.49). And a **P2 Goldilocks** appears exactly:
overall usage peaks at u≈0.25, then urgency starves the frequent core (freq-half 0.50 → 0.14).
**Exposure is a finite channel; urgency reallocates it.** #7 is the missing second axis: scarcity makes
the rare lexicon *pay*, exposure makes it *learnable*. Together they break the pidgin. Charts:
`exposure_teaching.png`.

## Test 3 — constraint ORDER (scheduling): not load-bearing for a fixed lexicon

`constraint_ordering.py`. Does the ORDER of applying the exposure constraint matter (curriculum ramp,
continuity)? Matched urgency budget (mean u=0.25), only the schedule differs; `freq_first` and
`urgent_first` are exact time-reverses. 10 seeds:

| schedule | usage | freq-half | rare-half |
|---|---|---|---|
| constant | **0.333** | 0.359 | 0.306 |
| staged | 0.304 | 0.355 | 0.252 |
| freq_first | 0.289 | 0.292 | 0.286 |
| urgent_first | 0.285 | 0.325 | 0.244 |

**No schedule beat the best constant; all within ~0.5 std; the two time-reverses are identical.** Order
is NOT load-bearing here.

**Why — and the refinement of the ordering principle.** Order matters when a constraint changes the
*reachable complexity* of the task — a **growing world** (curriculum.py: G 8→128 is unreachable cold;
the coma needs graduated re-grounding). Here the world is a **fixed 24-word lexicon**; urgency only
*reweights airtime* within a fixed task. Reordering a reweighting of a fixed-size problem has no
reachability ramp to climb, so it washes out (likely also because early-bootstrapped words *erode* once
their airtime drops — the use-it-or-lose-it / coma law cancels any head-start). **So: constraint order
is load-bearing for reachability ramps (growing worlds), not for reweighting a fixed task.** To see
order bite in the English line, the right test is a **growing lexicon** (start few words, add words —
curriculum.py's structure on the acquisition world), not an urgency schedule. Flagged for when the
additional constraints land.

## Test 4 — GROWING lexicon: order IS load-bearing, ramp advantage appears past the reachability cliff

`growing_lexicon.py`. The lexicon GROWS in stages (carry the population), introducing words most-
frequent-first / rarest-first / random, vs `cold` (all words from gen 0). Tracks **two-way
comprehension** (the honest metric). Swept vocab size K (the inspector showed the absolute level is a
partial pidgin — these short runs isolate the *laws*, not fluency):

| K | cold | freq_first | rare_first | random |
|---|---|---|---|---|
| 24 | **0.368** | 0.362 | 0.156 | 0.182 |
| 40 | 0.226 | 0.165 | 0.095 | 0.142 |
| 56 | 0.096 | **0.145 (+51%)** | 0.035 | 0.116 |

Two robust findings:
1. **Word-introduction order is always load-bearing** — freq-first beats rare-first at every size
   (2.3× / 1.7× / 4.1×). Build the frequent core first; introducing rare words first wrecks alignment
   (no scaffold to extend from, and rare words are semantically confusable).
2. **The ramp ADVANTAGE over cold appears only past the reachability cliff.** At K=24 cold reaches the
   target, so the curriculum only ties (order then only *protects* against a bad sequence). At K=56
   cold collapses (0.096) and the frequency-curriculum rescues it (+51%) — curriculum.py's result
   reproduced in the English world. (K=40 is the noisy transition.)

Fully-scoped ordering principle: **order among sub-tasks is load-bearing whenever the world is large
enough that cold-start cannot reach it; below that cliff a good order merely ties cold and a bad order
still hurts.** For scaling English to a large vocab: grow the lexicon frequent-first, do not cold-start
the full vocab. (Absolute comprehension here is a partial pidgin by design — short runs, tiny pop; the
value is the law, which tells us how to spend real training.)

## Fluency push — comprehension is training-limited on the correct landscape

`fluency_push.py`. Running the known-correct landscape (frequent-first grow + criticality + urgency)
with real exposure, tracking two-way comprehension per stage:

| training | comprehension |
|---|---|
| fixed champion (480 gen, pop 40) | 0.30 |
| probe grow (480 gen) | 0.362 |
| push (1400 gen, pop 60) | 0.568 |
| push (2600 gen, pop 72) | **0.612** (~14.6x chance) |

The per-stage curve rises monotonically (not plateaued) -> fluency is TRAINING-LIMITED, not
landscape-limited: scaling training pays (comprehension doubled), now into diminishing returns.
Diagnosis: when the speaker says English the listener gets it ~80% (`comp|saidEN`), so the cap is
COVERAGE of the rare/mid tail (speaker usage ~0.48, listener-uniform ~0.46), not alignment. Next
lever, by the push-the-landscape principle: a targeted constraint for the listener / rare tail
(nearest-embedding decode confuses semantic neighbours) is more efficient than brute training.

## Listener fix — SHARED LEXICON (tie speak & hear): the highest-leverage move

`listener_fix.py`. Diagnosis: the listener capped fluency because it was an INDEPENDENT matrix
decoding via nearest-embedding (semantic-neighbour collisions) and had to align with the speaker by
luck. Fix ("understanding before expression"): ONE shared association matrix A (K x K) used both to
speak (row argmax) and hear (column argmax); learning a word in either direction fixes both, no
embedding-cluster confusion. Head-to-head on the same landscape (freq-first grow + criticality +
urgency, pop 60, 1400 gens):

| | two-matrix | SHARED lexicon |
|---|---|---|
| final two-way comprehension | 0.507 | **0.831 (1.64×)** |

The landscape/architecture fix BEAT brute training (training alone reached 0.61 at 2600 gens; the
shared lexicon hits 0.83 at 1400). Sample conversation is clean English (13/14 understood:
`man→man`, `don→don`, `pierre→pierre`). Per-word fluency is now **bimodal**: ~14/24 words fully
fluent (95–100%), ~10 dead (0%) — a word either owns its English slot or **collides** and loses it
to a neighbour (`life→men`). So the remaining fluency gap is COLLISIONS in the shared association,
not a gradual rare tail. Next lever: resolve collisions (more capacity/exposure, or urgency targeting
the collided 0% words) rather than brute training. Principle confirmed: **push the landscape, not the
optimizer** — the shared-lexicon constraint bought more fluency than 2× the generations did.

## Scaling curve — honest result: flat vocabulary does NOT show the curriculum-scaling advantage

`scaling_curve.py`. Swept vocab K = 24..240 (shared lexicon, fixed 1800-gen budget), cold vs
frequent-first curriculum, measuring two-way comprehension both frequency-weighted AND uniform
(every word equal — the right lens for vocab scaling).

| K | chance | cold (freq / uniform) | curriculum (freq / uniform) | uniform advantage |
|---|---|---|---|---|
| 24 | 0.042 | 0.763 / 0.748 | 0.804 / 0.734 | −0.014 |
| 48 | 0.021 | 0.659 / 0.580 | 0.721 / 0.615 | +0.035 |
| 96 | 0.010 | 0.629 / 0.522 | 0.649 / 0.499 | −0.023 |
| 160 | 0.006 | 0.595 / 0.447 | 0.644 / 0.501 | +0.055 |
| 240 | 0.004 | 0.591 / 0.442 | 0.560 / 0.450 | +0.008 |

Findings (stated straight):
1. **The uniform metric reveals a real rare tail** — uniform < freq-weighted, gap grows with K
   (K=240: 0.591 vs 0.442). The frequency-weighted metric was hiding tail weakness.
2. **Cold-start does NOT collapse** — even on uniform, K=240 is 0.442 (~110× chance). Graceful
   degradation (0.75→0.44 over a 10× vocab), no cliff.
3. **The curriculum advantage is noise** (−0.014..+0.055, mean ~+0.01, sign-flipping, 3 seeds). No
   widening gap on either metric.

**Verdict: the curriculum-scaling claim is NOT supported here — and the test was fair.** The flaw is
the TASK: a shared-lexicon vocabulary is permutation learning (an identity map over K *independent*
symbols), intrinsically easy and gracefully scalable, with no hard structure for cold-start to choke
on — so no cliff for a curriculum to rescue. The cliff and curriculum advantage live where meanings
**combine** (grammar/composition, #5), where expressible meanings grow combinatorially. That is the
legitimate arena for the scaling claim, not bigger flat vocabulary.

### Scorecard (this session)
- REAL: pidgin broken (scarcity rent-law + exposure); fluency 0.30→0.83 via shared-lexicon listener
  fix; substrate scales *gracefully* to 10× vocab (no collapse).
- NOT PROVEN (tested fairly): the curriculum *scaling advantage* — flat vocabulary is the wrong arena.
- OPEN: tail comprehension degrades with K at fixed budget (collision/coverage); is fluency-at-scale
  compute-limited (scales with gens) or architecture-limited (K×K lookup wall)? -> `scaling_diagnostic.py`.

## Figuring out scaling — the honest synthesis (architecture + compute sweeps)

Three experiments (`scaling_curve.py`, `scaling_diagnostic.py`, `compute_scaling.py`) pin down why
fluent English degrades with vocabulary, and what scaling actually requires.

**1. Decomposability is the gradient-free scaling axis (counterintuitive).** Three shared-lexicon reps
at K=24→128 (uniform comprehension):

| rep | params | K24 → K128 | retains |
|---|---|---|---|
| lookup (K×K) | K² | 0.733 → 0.486 | 66% |
| codes (K×D, grounded) | K·D | 0.638 → 0.377 | 59% |
| structured (D×D shared map) | D² | 0.615 → 0.143 | 23% |

The *most decomposable* rep (lookup: every meaning×signal cell independent) scales BEST; the compact
*entangled* rep (one shared D×D map) scales WORST — its mutations perturb all words at once, so
evolution can't make a local correction. **Gradient descent loves weight-sharing (co-adapts via the
gradient); gradient-free evolution needs DECOMPOSABILITY (mutations must stay local).** Grounded
per-word codes (decomposable + linear params) beat the entangled map handily, confirming the axis.

**2. Fluency-at-scale is compute-limited, not a wall — but the law is ~quadratic.** Lookup uniform
comprehension keeps rising with generations at every K (no plateau except small K), so more compute
buys fluency at any vocab. But gens-to-reach-0.70: K=32 → 2700; K=64,128 → >4500. Compute-to-fluency
grows super-linearly (the K² parameters), a steep cost.

**3. The resolution = compact AND decomposable, i.e. COMPOSITION.** The tension is fundamental:
decomposable flat reps (lookup) learn well but cost K²; compact reps (shared maps) are O(1) params but
evolution can't learn them. The only way to be *both* compact and decomposable is **compositional
sub-word structure** — a small shared alphabet with words as independently-mutable sequences (exactly
how natural language builds an unbounded lexicon from ~40 phonemes). That is **constraint #5**
(channel-capacity + composition) arriving from the scaling direction: **scaling fluent English to a
large vocabulary requires compositional structure, not a bigger flat lexicon.** The scaling fix and the
grammar constraint are the same fix.

### Updated scorecard
- REAL: pidgin broken; fluency 0.30→0.83 (shared-lexicon listener fix); decomposability is the
  gradient-free scaling axis; fluency-at-scale is compute-limited (no wall).
- HONEST NEGATIVE: curriculum scaling-advantage is noise on flat vocab; a compact *entangled* rep
  scales worse than a dumb decomposable table; flat-lexicon compute cost is ~K².
- NEXT (now motivated from two directions): compositional sub-word representation (#5) — the only rep
  that is both compact and decomposable, the actual path to large-vocab fluency.

## Test 5 — COMPOSITION: vocabulary for free (the scaling fix, confirmed)

`compositional_scaling.py`. Meanings are composed: (a,b) with a,b∈[0,V) → M=V² meanings. A
compositional organism learns PER-SLOT machinery (two V×V associations shared across all
combinations); a flat organism treats each (a,b) as atomic (M×M lookup). Decisive test = ZERO-SHOT:
train on 70% of combinations, measure comprehension on HELD-OUT combinations never seen.

| V | M=V² | chance | compositional held-out | flat held-out |
|---|---|---|---|---|
| 4 | 16 | 0.062 | **0.934** | 0.215 |
| 6 | 36 | 0.028 | **0.956** | 0.297 |
| 8 | 64 | 0.016 | **0.818** | 0.295 |
| 10 | 100 | 0.010 | **0.731** | 0.374 |

Compositional held-out comprehension is **0.73–0.96** — it understands meanings it never trained on,
because each slot value was seen in *some* combination, so a new pairing composes. Compositional
held-out ≈ its seen accuracy (small gap = genuine generalisation); flat collapses from seen to
held-out and stays far below compositional. **Composition gives zero-shot generalisation to unseen
meanings — vocabulary for free — with O(V) per-slot params instead of O(V²).** This is the
compact-AND-decomposable representation the scaling synthesis predicted: the scaling fix and the
grammar fix are one and the same, now demonstrated.

## Pushing fluency × vocab up with composition (autonomous campaign)

`multislot_composition.py`, `english_compositional.py`. A meaning is an S-tuple (V values/slot) →
effective vocabulary M = V^S; the organism learns per-slot machinery (S associations of V×V, shared),
params O(S·V²) independent of M. Composition turns one huge-vocab problem into S small ones.

**Per-slot accuracy is the fluency lever (full-meaning = per-slot^S).** Iterating: (1) base, (2) more
budget+pop, (3) a **full-meaning bonus** (partner acts only on the whole meaning — a world-consequence)
that lifts the weakest slot and makes per-slot accuracy flat across S (~0.95). Result (V=8):

| S | vocabulary M=V^S | per-slot acc | full-meaning comprehension |
|---|---|---|---|
| 3 | 512 | 0.97 | 0.90 |
| 4 | 4,096 | 0.96 | 0.85 |
| 5 | 32,768 | 0.94 | 0.74 |
| 6 | 262,144 | 0.95 | 0.73 |
| 8 | 16,777,216 | 0.88 | 0.36 (undertrained) |

**0.73 comprehension over a 262k vocabulary, 0.85 over 4k** — where the flat lexicon sagged to 0.49
over *128* words and couldn't represent more. Vocabulary scales ~for free; per-slot accuracy (a tiny
fixed problem) is the only thing to train.

**Real English, zero-shot (`english_compositional.py`).** Give each real word a compositional code by
product-quantising its embedding (split 64-d into S chunks, k-means V centroids each → S-tuple). Train
on 70% of 300 real words; test code-comprehension on the **held-out 30% never seen**:

| metric | value |
|---|---|
| seen-word code-comprehension | 0.836 |
| **HELD-OUT (zero-shot) real words** | **0.828** (chance 2.4e-4) |
| code uniqueness (1 = no collisions) | 0.888 |

Held-out ≈ seen: the organism comprehends real English words it **never trained on**, via shared
per-slot machinery — real-English vocabulary scales by composition, zero-shot. (Caveat: ~11% of words
collide in code at S=4,V=8; more slots/values reduce collisions — a word-identification ceiling, not a
comprehension one.) This is the scaling synthesis delivered end-to-end: fluency AND vocabulary up,
together, by composition.

## Scaling REAL English vocabulary 300 → 2000 at constant compute

`english_comp_scale.py` (+ `grounding_xl.npz`, 2000 real words, 96-d). The organism's genome is per-slot
machinery S×V×V — **independent of vocabulary K**. Each real word → a product-quantised S=6,V=8 code
(97% unique). Train on 70% of words, test HELD-OUT (zero-shot).

| K (real words) | held-out code-comp | word-recovery | uniqueness | budget |
|---|---|---|---|---|
| 300 | 0.523 | 0.512 | 0.971 | 4000 |
| 1,000 | 0.569 | 0.546 | 0.964 | 4000 |
| 2,000 | 0.459 | 0.437 | 0.957 | 4000 |
| 2,000 | **0.556** | **0.528** | 0.956 | 8000 |

**Held-out comprehension stays ~flat as the real vocabulary scales 300→2000 — at constant genome
size and compute.** A flat lexicon needs O(K²) params and scores *zero* on held-out words; composition
generalises to unseen real vocabulary for free. The absolute level (~0.53 word-recovery on 2000 real
words, zero-shot) is per-slot-accuracy-limited (full = per-slot^S): more budget at K=2000 lifted it
0.459→0.556 — training-limited, still climbing, not walled. Trade-off: more slots S → higher uniqueness
(word-ID) but lower full-code comprehension; the lever for both is driving per-slot accuracy up.

## Utterance length: BREATH × TIME (toward multi-word responses)

Toward responses longer than one word: how does an organism know how long to speak? Not by planning
length (that would be designed) — length must emerge. `breath_test.py`, `breath_time.py`.

**Breath alone is NOT a length calibrator (`breath_test.py`, honest falsification).** Three terminators
on a sequential comm substrate (meanings of varying complexity C): a learned STOP, a finite BREATH
budget, and a designed cap=C (oracle). The learned **stop calibrated length to meaning almost perfectly**
(corr(C,len)=0.99, full-meaning 0.75 ≈ oracle 0.80); a fixed **breath budget could NOT calibrate**
(corr=0.006, flat length, full-meaning 0.59). So per-meaning length comes from "stop when the meaning is
discharged," not from breath. (Caveat: the stop got a clean "is-this-slot-active" signal — the search-
wall lives where "am I done" needs *memory*, untested here.)

**Breath × TIME is the URGENCY mechanism (`breath_time.py`, confirmed, 4 seeds).** Weight breath by
production SPEED σ (a fast 'fat' word burns more air): words/breath = B/σ, utterance time = L/σ. One knob
σ trades breath against time. Sweeping time pressure τ:

| time pressure | full-meaning | mean length | evolved σ | breath cap | breath-truncated |
|---|---|---|---|---|---|
| strong (τ=1.5) | 0.32 | 1.88 | 2.76 | 2.01 | 66% |
| medium (τ=6) | 0.66 | 3.05 | 1.29 | 4.33 | 28% |
| weak (τ=24) | 0.79 | 3.42 | 0.89 | 5.63 | 6% |

Monotone: urgency → organism evolves **faster speech** (σ↑) to save time → **fewer words per breath**
(cap↓) → **breath binds** (truncation↑) → **messages shorten** (len↓), comm degrades. Urgency → fast
speech → breath-limited → clipped speech, emergent from energy+time+breath, nothing designed.

**Synthesis (complementary, both real):** the STOP sets length to meaning *content*; BREATH×TIME
modulates it by *urgency* via speaking speed. An organism says what the meaning needs, capped by the air
rushed urgent speech leaves it. (σ here is evolved per-organism to the world's pressure; per-utterance
dynamic speed — state/arousal coupling — is the deferred next rung.)

## Breath: the honest conclusion (role test)

`breath_role.py`. The speaker always knows its own meaning, so "have I said it all?" is always
perceivable -> a learned stop should calibrate length in ANY regime, and breath (a fixed budget) never.
Confirmed (4 seeds, contiguous AND scattered active slots):

| regime | terminator | full-meaning | mean len | corr(span,len) | len−span |
|---|---|---|---|---|---|
| contiguous | stop | 0.74 | 4.49 | **1.00** | 0.00 |
| contiguous | breath | 0.80 | 7.99 | — (constant) | +3.50 |
| scattered | stop | 0.77 | 6.95 | **1.00** | 0.00 |
| scattered | breath | 0.72 | 8.00 | — (constant) | +1.05 |

**The stop calibrates length perfectly (len = span exactly) in both regimes; breath maxes out to a
constant (always say the most, wasteful-but-safe).** So breath is NOT the length mechanism — the STOP
is, because the speaker knows its own content and "stop when discharged" is always reachable. Breath's
genuine roles, established across the four breath scripts: (1) prosodic CHUNKING into breath groups
(`breath_turn.py` transcripts), and (2) the urgency -> clipped-speech coupling via speed<->time
(`breath_time.py`). Breath is the delivery/prosody layer, not the length controller.

## From PARROTING to COMMUNICATION (transform, don't copy)

`conversation.py`. All prior comm tasks were transmission (hear a meaning, say the SAME meaning back —
a telephone). Real communication is: hear X, produce the appropriate DIFFERENT reply Y. The world makes
copying fatal: the reply function r(v) != v, and survival = the conversation CONTINUES, which happens
only when the organism answers with r(prompt). Parrot the prompt → conversation dies. Per-slot reply
machinery → generalises (a vocabulary of replies for free).

Result (S=2, V=8, 64 meanings, 4 seeds): **response accuracy 0.93** (chance 0.125) — the organism
learned to TRANSFORM input into the correct different reply; **parrot baseline conversation length 0.00**
(copying always dies). Transcript shows genuine non-echo replies: `(3,4)→(1,2)`, `(6,7)→(0,6)`.

Bootstrap note (paradigm-honest): conversation-length fitness alone is too sparse (need all slots right
on turn 1 = 1/64) and stalls at chance; an exact per-reply-rule count (or partial credit = partner
partially understands) bootstraps it. The transformation itself is then learned cleanly. This is the
foundation for an LLM-as-environment: a partner that stays engaged only when the reply is appropriate.

## LLM as the environment (the conversational world is a real model)

`llm_world.py`. The conversational partner is a running LLM (llama3.2:3b), not a hand-coded proxy. The
LLM defines the world: for each prompt it gives the appropriate reply (the move that keeps a chat going)
— hello->hi, thanks->appreciate, danger->alert, food->restaurant. The organism evolves to produce those
replies; survival = the conversation continues = the reply is one the LLM accepts. The LLM never sees a
score — it just talks, and the organism becomes the thing that keeps it talking.

Result: organism learned the LLM's conversational replies at **accuracy 0.88**, while the **parrot
baseline scores 0.00** (every LLM reply differs from its prompt, so copying always stalls the chat).
It then held a LIVE multi-turn conversation with the running model (hello->hi, how->what, what->question),
producing contextually-appropriate, non-echo replies that kept the LLM engaged.

This realises the design: communication = transform-don't-copy, with another *mind* as the environment.
Honest limits: 16-word fixed vocab (off-vocab LLM messages fall back); cached reply table for cheap
evolution (a live per-eval LLM fitness is far too slow at neuroevolution scale). The clear next step is
to make it COMPOSITIONAL — scale the conversational vocabulary and generalise zero-shot to prompts the
organism never trained on (the off-vocab case), combining `compositional_scaling` with `conversation`.

## Zero-shot conversation (generalise the chat, don't memorise it)

`conversation_scale.py`. Compose transform-don't-copy with composition: prompt (a,b), reply per-slot
(g0(a),g1(b)). Train on 70% of V^2 prompt combinations, test reply accuracy on HELD-OUT prompts never
seen (4 seeds):

| V | prompts V^2 | seen reply acc | HELD-OUT reply acc | held conv length |
|---|---|---|---|---|
| 6 | 36 | 0.957 | 0.958 | 42.5 |
| 10 | 100 | 0.848 | 0.762 | 4.6 |
| 16 | 256 | 0.660 | 0.601 | 1.9 |

**Held-out ≈ seen** — the organism replies correctly to prompt combinations it never trained on:
zero-shot conversation, the direct fix for the off-vocab case. Accuracy falls with V at fixed budget
(per-slot reply rules undertrained — the full-meaning bonus / more budget would lift it, as in
multislot composition). Conversation length is high when accuracy is high (V=6: 42 turns) and collapses
when it isn't (each turn needs all slots right). Combines `conversation` (transform) + composition
(generalise) — the path to a model that converses about things it was never explicitly taught.

## Generalising sentence-level conversation (compositional, no lookup)

`conversation_compose.py`. Prompt (a,b) compositional; reply = a multi-word phrase per component
(phrase(a) ++ phrase(b)). Per-component integer genome -> the organism produces the correct multi-word
reply to UNSEEN prompt combinations. Train 70% of V^2, held-out reply-correct (4 seeds):

| V | prompts | seen reply-correct | HELD-OUT reply-correct |
|---|---|---|---|
| 6 | 36 | 0.950 | 0.951 |
| 10 | 100 | 0.951 | 0.951 |
| 16 | 256 | 0.943 | 0.943 |

Held-out == seen, flat to 256 prompts -> generalising sentence-level conversation: it answers prompt
combinations it never trained on, killing the fixed-prompt lookup. (Integer per-component genome holds
~0.95 where the argmax-matrix sagged to 0.60 — the representation lesson again.)

## Full pipeline: hear -> understand -> reply (series not parallel)

`conversation_pipeline.py`. The organism now COMPREHENDS the incoming message itself (no oracle prompt):
stacked comprehension C (signal->understood value) + response R (value->reply phrase). Joint vs series
training (S=2, V=10, 4 seeds):

| training | seen full-reply | held-out full-reply |
|---|---|---|
| joint | 0.637 | 0.538 |
| series | **0.929** | **0.931** |

The full pipeline reaches 0.93 and generalises to unseen prompts (held-out == seen) — but only when
trained in SERIES (comprehension first, freeze, then response). Joint training collapses, confirming
the parallel-component caution. So an organism that comprehends real input AND replies appropriately to
prompts it never saw is reachable; it just has to be staged.

## Communicate, don't mimic — and the LLM oracle is noisy

`conversation_accept.py`. Instead of copying the LLM's exact reply, the world accepts a SET of valid
replies (sampled from the LLM); the organism discovers its OWN reply that lands in the set. Result:
acceptable-rate 1.00 against the sampled set, and **64% of its replies differ from the LLM's modal
reply** — it found its own valid responses, not mimicry. But an independent LLM JUDGE accepted only
**8/14**, and it rejected obviously-sensible replies (`"how are you"->"fine"` judged "no").

Honest finding: the organism does discover its own valid responses, but **LLM-as-oracle is noisy** —
temperature-sampling yields a too-loose acceptable set (marginal replies like `hungry->"time"`), and
the judge is harsh/inconsistent. So "the LLM is the environment" works as a direction but needs a robust
acceptance signal (majority-vote judging, tighter sampling) before the acceptance number means much.

## Generalising English conversation (readable) — answer situations never trained on

`conversation_real.py`. A situation is (intent, topic); the reply composes opener(intent) ++
content(topic). The organism learns per-component production and answers UNSEEN intent+topic
combinations in real English. 6 intents x 6 topics, train 60%, held-out reply-correct 0.77 (seen 0.81).
HELD-OUT transcript (situations never trained on):

    [want + you ] -> "i need my friend"        [ask + name]  -> "tell me your name"
    [want + food] -> "i need some food"        [thank + help]-> "thanks for your help"
    [greet+ home] -> "hello there going home"  [warn + water]-> "watch out the water"

The organism never saw "i need my friend"; it composed it from want->"i need" + you->"my friend".
This is the lookup killed end-to-end: generalising, readable, sentence-level English conversation.

## Conversation needs MEMORY (context-dependent replies)

`conversation_memory.py`. Real conversation: the right reply to a follow-up depends on what was said
earlier. Turn 1 sets context c, turn 2 gives prompt p, correct reply h(c,p) depends on BOTH. A stateless
organism (replies only to the current prompt) cannot; a stateful one WRITES c to a memory register at
turn 1 and READS it at turn 2. Result (5 seeds):

| organism | context-dependent reply accuracy |
|---|---|
| stateless | 0.339 |
| stateful | **0.956** |

The stateful organism evolves the write/read to carry conversational context (0.96 vs 0.34). Conversation
is a world that demands memory, and the organism invents using it -- connecting the dialogue work to the
EEC memory laws. Together with the rest of the conversation suite (transform-not-copy, comprehend+respond
in series, compositional generalisation, breath-chunked delivery), the pieces of real dialogue are here.

## The comprehension wall: open input needs domain-matched embeddings

`conversation_understand.py`. To remove the LLM-matching crutch, the organism comprehends a message as
the mean of its word embeddings and classifies the situation (nearest centroid), tested on HELD-OUT
LLM-generated paraphrases. Result: **0.38** (chance 0.10) — weak. The mechanisms (transform, generalise,
memory) all work; the wall is comprehension of *varied real input*, gated by embedding domain: our
embeddings are from a 19th-century literary corpus while the LLM's paraphrases are modern idiomatic chat
("running on fumes", "later gator"), so the mean-embedding doesn't cluster by meaning. The live
conversation collapses to a couple of situations without good comprehension. Honest boundary: open
conversation is comprehension-limited, and comprehension is only as good as the perceptual grounding.

## Comprehension wall broken: proper embeddings -> open conversation, no crutch

`conversation_understand2.py`. Swapping the 19th-century word-average embeddings for a real embedding
model (nomic-embed-text) lifts held-out paraphrase comprehension **0.38 -> 0.85** (chance 0.10). The
organism now comprehends the running LLM's OWN messages (no oracle matching) and replies coherently:

    "hey there how is it going"   -> [greeting] -> "hello there friend"
    "i could really use a nap"    -> [tired]    -> "you should rest"
    "i'm feeling just fine thanks" -> [happy]   -> "that is great"

"i could really use a nap" -> tired is genuine comprehension of an unseen phrasing. So the conversation
loop closes end-to-end: the organism PERCEIVES real English (grounded in a good embedding), COMPREHENDS
the situation (0.85 on held-out paraphrases), and REPLIES appropriately, conversing with a live LLM
without any matching crutch. The earlier wall was perceptual grounding quality, not the mechanisms.

## Integrated live conversation (perceive->comprehend->reply), judged

`conversation_live.py`. The full loop: nomic-embedding perception + situation comprehension (16
situations) + per-situation reply, over a 16-turn live conversation with the LLM, each reply JUDGED by
the LLM (majority of 3). Coherence: **10/16 = 0.62**. Honest failure mode: it LOOPS on "weather" for
~10 turns -- fixed per-situation replies + no memory create a feedback loop (the organism keeps replying
"the weather is nice today", the LLM keeps talking weather). Good isolated turns (greeting, compliment,
agree) but repetition drags coherence. The fix is exactly what conversation_memory.py showed is
learnable: VARIED replies + conversational MEMORY (don't repeat, move the topic). The integrated
perceive->comprehend->reply loop is real; sustained coherent dialogue needs reply variation + memory.

## Sustained flowing conversation: variation + memory + initiative

`conversation_live2.py`. Fixing v1's loop: each situation has several replies, the organism remembers
recent situations and (a) doesn't repeat a reply, (b) PIVOTS to a new topic with a question when a
situation recurs (conversational initiative). Same nomic perception + comprehension; LLM-judged. Result:
coherence **0.62 -> 0.75** and **15/16 distinct replies** (v1 looped on ~2). The 16-turn conversation now
flows through greeting -> weather -> food -> fun -> tiredness, the organism steering with pivots
("by the way are you getting hungry") instead of repeating. Sustained, varied, coherent dialogue with a
live LLM -- the organism perceiving, comprehending, remembering, and taking initiative. This caps the
conversation arc: parrot -> transform -> generalise -> comprehend real input -> sustained flowing chat.

## Emergent dialogue: the back-channel (feedback) emerges between two organisms

`emergent_dialogue.py`. Pure two-organism test (no LLM, no designed reply map). A target has two
attributes; the listener already knows one and needs the other; the speaker knows the full target. Does
a back-channel emerge -- the listener telling the speaker what it needs? Result (6 seeds):

| protocol | task success |
|---|---|
| one-way (speaker blind) | 0.500 |
| two-way (back-channel) | **1.000** |

The speaker cannot do it blind (capped at 0.5 -- must guess which attribute the listener lacks). With a
back-channel the pair co-evolves a request->response convention (listener encodes 'what i need', speaker
reads it and sends the right attribute), reaching 1.0. Feedback -- the foundation of dialogue -- emerges
from coupled survival, gradient-free, with no designed mapping. This adds the two-way dimension to the
conversation work: not just transform-a-prompt, but a genuine exchange where each side shapes the other.

## Emergent multi-turn dialogue: directed clarification (feedback + memory)

`emergent_dialogue_multiturn.py`. Extending the back-channel to genuine turn-taking: a target has K=6
attributes, the listener knows 3 and needs 3 within a tight T=3 turn budget. With FEEDBACK the listener
requests a still-missing attribute each turn (tracking received = memory); BLIND the speaker sends a
fixed sequence ignoring the listener. Result (6 seeds):

| protocol | task success |
|---|---|
| blind | 0.135 |
| feedback | **1.000** |

The listener directs the dialogue (request the missing, remember the received) and the speaker answers,
covering exactly the missing set; the blind speaker wastes the budget (caps 0.135 vs ~0.05 chance). A
multi-turn directed clarification protocol -- real turn-taking, where the listener controls the flow --
emerges gradient-free with no designed script. Together with `emergent_dialogue.py` (single back-channel),
this shows the structural core of dialogue (feedback, memory, turn-taking) is reachable by coupled
survival alone.

## How comprehension is acquired: experience-averaging, not evolving weights

`conversation_evolved_comp.py`. Is the conversational comprehension a GENREG-evolved artifact? Held-out
paraphrase comprehension: nearest-centroid (analytic) **0.86** vs gradient-free EVOLVED prototypes from
random **0.39**. Evolving 10x768 prototypes from scratch fails -- gradient-free mutation cannot search
high-dimensional continuous prototype space efficiently. But the centroid is just the MEAN of experienced
examples = **prototype formation from exposure**, a non-gradient MEMORY mechanism (not backprop). So in
this paradigm comprehension should be ACQUIRED BY EXPERIENCE-AVERAGING (a prototype/memory of the
situation), not by evolving weights. Honest boundary: gradient-free evolution owns the discrete/decomposable
parts (the reply policy, the protocol); the high-d perceptual prototype is best formed by averaging
exposure -- which is still gradient-free, just a different mechanism, and connects to the EEC memory laws.

## Few-shot conversational acquisition (learn a new situation from exposure)

`conversation_fewshot.py`. Following the experience-averaging finding: can the organism LEARN a new
conversational situation from a few exposures (prototype = mean of k examples, non-gradient memory)?
Leave-one-situation-out, comprehend the novel situation's held-out paraphrases vs the known situations:

| exposures | novel-situation comprehension (chance 0.08) |
|---|---|
| 1 | 0.65 |
| 2 | 0.73 |
| 3 | 0.79 |
| 8 | 0.79 |

One-shot already gives 0.65 (8x chance); it plateaus ~0.79 by 3 exposures. The organism acquires a new
conversational concept from a handful of examples by experience-averaging -- gradient-free few-shot
learning to understand, the acquisition counterpart to "understanding before expression". So the
conversational organism not only comprehends and replies, it can LEARN new situations from exposure.
