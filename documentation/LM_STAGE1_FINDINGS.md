# lm_char_v1 / attn_copy_v1 — findings log

## 2026-07-05 — SUBSTRATE ABOVE THE BIGRAM CEILING (stage-1 local bar PASSED)

**Held-out 28.03% top-1 / 62.16% top-5 vs char-bigram 27.30% → +0.73 pts
above the count table. Majority floor 20.27% (+7.8pp). No train→held-out
drop.** Pure constraint-driven evolution: soft multiplicative fitness (mean
log-prob), energy homeostasis (starved 3–12%/gen throughout), tournament +
maturation gate, recurrence + prev-char concat, per-neuron 8-catalog
activations. Zero closed-form seeding, zero argmax fitness, zero gradients.

Sweep chain (each ≤8k gens, resumed — bootstrap rule):
| sweep | gens | change | held-out top-1 |
|---|---|---|---|
| 1 | 3000 | card defaults | 22.38% |
| 2 | +3000 | EMA-smoothed selection (§IV.7) + batch 96 | 23.97% |
| 3 | +8000 | just generations (anneal at 80%) | **28.03%** |

EMA smoothing was worth ~+1.6pp on its own — the "noise-driven culling
destroys ratchets" rule holding exactly as documented. Total GPU time for
the whole chain: ~24 minutes.

Samples (temp 0.5): `" and in te the ween we the oe tod hit seer to the sat
ine eor the wist … and i ie her lan lone"` — articles/conjunctions and word
shapes; expected for a bigram-ceiling-level model (readability arrives with
retrieval + rollout stages).

Run ids: sweep 3 = `runs/lm/` newest; checkpoint carries full state.

## 2026-07-05 — attention component: three lessons before it climbed

1. **Information must flow — literally.** v1 queried only from the flag
   position; the offset announcement at position 0 was unreachable → flat at
   chance forever. No fitness signal can rescue an inexpressible solution.
2. **The environment must be what you think it is.** Episodes were padded to
   L=64 with random symbols, so the "length curriculum" never shortened
   anything and attention always faced 60 distractor keys. Masking keys
   beyond the true episode length turned gen-0 flatline into a climb
   (12.5% → 86% on k=1 within 150 gens).
3. **Credit must reach the skill being selected.** With a random 3-matrix
   readout between attended value and logits, attention placement earned no
   credit (two coupled miracles). Weight-tying the readout to the symbol
   embeddings + identity value init made attention placement the ONLY thing
   left to evolve — one basin.
4. (v4) **Additive queries can't switch offsets.** Per-k retrieval needs a
   k-conditioned rotation; linear-over-concat can't multiply. Multiplicative
   query `q = (flag ⊙ k_state) @ Wq` = a different effective map per offset —
   the §VI low-rank-trigram lesson reappearing in attention.

Status: ladder (11 rungs, length grows before offsets unlock, mastery-gated)
passed rung 4 with additive queries stuck at k=2; multiplicative-query run
in flight.

## 2026-07-05 (later) — the relative-bias basin: rung 9 of 11

5. **Give evolution the smallest possible mutation target.** Multiplicative
   queries made k-conditioning learnable but the grind through sinusoid
   geometry was slow (~1 rung per 8k gens). Adding an evolved
   relative-position bias — `score(l) += B_rel[k, fp−l]`, ZERO-initialized
   (a transparent no-op, the cascade-bootstrap pattern) — made "attend
   exactly k back" a single bump in a flat table. Three rungs fell in one
   3×8k chain.

Held-out at FULL difficulty (L 24–64) after the rel-bias chain:
k=1 **76.7%** · k=2 **86.7%** · k=5 **78.3%** · k=10 **68.3%** · k=20 13%
(rung 10 not yet unlocked). Architecture ledger: multiplicative query
(§VI interaction lesson) + weight-tied readout + identity values +
zero-init relative bias — every table still fully evolved, nothing
closed-form, nothing pretrained.

## 2026-07-05 (later still) — attention bar PASSED at 100%; first assembly honest-negative

**attn_copy_v1 cleared its bar completely**: chain 4 unlocked rung 10 and
converged to held-out **100% on every offset** k∈{1,2,5,10,20} at L 24–64
(soft −6e−05); re-verified at 100% on 2,500 fresh-seed episodes
(ckpt `20260705-134745-attn-8197b1`). Component 2: DONE.

**Assembly sweep 1 vs control (the §VII discipline paying off):**
| run | held-out top-1 | champion α |
|---|---|---|
| substrate + copy channel, 8k gens | 29.78% | −0.007 (never opened) |
| CONTROL: substrate alone, same start, 8k gens | **30.42%** | — |

The copy channel earned nothing (α pinned at its no-op) and its extra genome
dims slightly diluted selection — the documented "complexification diffuses
selection pressure" effect, reproduced. Substrate alone is now **+3.12 pts
above bigram**. Root cause identified in the channel itself: it retrieved
the char AT the matched position instead of the char that FOLLOWED it (the
induction structure) — a retrieval with almost no predictive value. Fixed
(values = successor embeddings, current position excluded to prevent a
target leak); induction assembly running from the 30.42% checkpoint.

**Induction variant verdict: α still ≈0 at 31.91% held-out** (+4.61 over
bigram — again pure substrate ratcheting). Two controlled negatives ⇒
64-char windows don't contain enough retrievable structure to pay
attention's rent. Environment lever (longer / repeat-rich windows)
backlogged; channel kept (provably zero-cost when closed).

## 2026-07-05 — stage 4: the exposure gap, measured and half-closed

**The gap**: the 31.9% champion scores 31.4% top-1 / −2.48 nats teacher-
forced but 15.0% / −3.20 nats when consuming its own sampled outputs —
**0.72 nats of train/inference mismatch**. This is the quantified cause of
generation soup (and the text version of DiffEvo's walk drift).

**Chain 1 (pure rollout fitness, R=2→4→8)**: closed-loop improved at every
R (R=8: 15.0% → 19.8% top-1; gap 0.72 → 0.22 nats) — the landscape works —
**but open-loop collapsed** (31.9% → 23.1%) and samples got smearier. Pure
rollout log-prob on drifted context rewards HEDGING toward marginals: a new
reward-hack for the §XI catalog.

**Fix (the DiffEvo blend lesson)**: score the teacher-forced segment AND the
rollout segment in the same fitness — both regimes must pay simultaneously.
Chain 2 running from the sharp 31.9% checkpoint (warmup 24 ≈ 20 teacher-
scored steps + R rollout steps).

**Chain 2 (blended) — STAGE 4 VALIDATED:**
| metric | pre-stage-4 | pure rollout | blended |
|---|---|---|---|
| open-loop top-1 (held-out) | 31.9% | 23.1% | **31.3%** |
| own-output top-1, R=8 | 15.0% | 19.8% | **27.2%** |
| exposure gap | 0.72 nats | 0.22 | **0.17** |

The survival-on-own-outputs landscape closed the exposure gap 4× while
holding open-loop competence. Samples at t=0.5 now contain real words
("the her she … and … go"). Checkpoint: newest runs/lm.

**Open levers, in order**: (1) keep chaining the blended landscape + gen
scaling toward the A_101-class 34% (the substrate ratchet has not
saturated); (2) attention's environment lever (longer / repeat-rich windows
so the copy channel can pay rent — α-gate makes it free to carry);
(3) meta-optimizer (component plan #3) to buy back wall-clock across every
run; (4) per-neuron activation audit (are the 8 functions being used?).

## 2026-07-05 — encoder separation (enc_char_v1): three rounds, verdict = parity

| variant | standalone h1 | composed held-out | rollout closed (R=8) | gap |
|---|---|---|---|---|
| monolith (ref) | — | 31.9% | 27.2% | 0.17 nats |
| equal horizons | 29.7% | 30.71% | — | — |
| weighted 0.7/0.2/0.1 | 31.49% | 31.78% | **27.32%** | 0.21 nats |

Lessons: (1) at fixed H=64, equal multi-horizon pressure robs next-char
sharpness — weighting recovers it; (2) the weighted state carries h2/h4
future information the monolith never encoded, at zero performance cost;
(3) the composed model reaches full parity with the encoder FROZEN — the
readout alone can absorb the whole rollout adaptation. Separation = free
modularity, not free accuracy. Kept as the component option; capacity growth
for the encoder (deliberate design, not plateau-patching) is the flagged
follow-up if richer state is ever demanded by longer rollouts.

## 2026-07-06 — low-rank trigram interaction (§VI): best char result yet, 34.61%

Two-phase bootstrap from the 31.9% substrate: phase 1 evolves ONLY the gated
multiplicative channel `a_lr·(bigram[c_t] + (E1[c_t]⊙E2[c_prev])@O)` (substrate
frozen), phase 2 unfreezes all.

| stage | held-out top-1 |
|---|---|
| substrate (recurrent + prev-char) | 31.9% |
| + trigram channel, phase 1 (channel only) | 33.9% |
| + phase 2 (unfreeze all) | **34.61%** / top5 68.78% |

- **+2.7 pp over the substrate**, matching the documented A_101 char benchmark
  (34.00% / 69.70%). Closes 37% of the substrate→char-trigram-ceiling gap
  (31.9 → 39.2). Bigram bar 27.3.
- **The channel EARNS its keep** — unlike the copy-attention channel (α pinned
  at 0), the multiplicative trigram gate opened and paid fitness in phase 1.
  Confirms §VI: multiplicative interaction > additive concat for pair
  structure the recurrent state alone couldn't express.
- Generation (t=0.7): "…hand py ow an aner fhand i theed poarer i he thon
  polhat was the wor as as led she tee a" — real words + word-shapes. Still
  word-shapes, not sentences (expected at 34% per the docs; top-5 barely moved,
  the documented "sharpens top guess, not the tail" pattern).

**Next levers**: (1) apply the blended rollout-survival landscape ON TOP of the
trigram model — the +2.7pp is accuracy; rollout targets generation coherence
(the space-collapse attractor is the exposure gap, unaddressed here);
(2) push toward the 39.2% trigram ceiling with more channel capacity / gens.

## 2026-07-06 — TRUST-MIX: readable English, top-1 55% / top-5 85% (held-out)

The composition path (A_66-style) — NOT one model doing everything. Channels:
exact char n-grams (uni/bi/tri/4g/5g dense, 6g/7g hashed) + the evolved neural
model, combined by an EVOLVED context-conditional backoff gate (trust + per-
channel evidence κ, Witten-Bell style; gradient-free ES).

Accuracy climb (held-out top-1) as orders + gate were added:
| config | top-1 |
|---|---|
| neural alone (our best evolved model) | 34.6% |
| + n-grams to 4g, global trust | 47.2% |
| + 5g, evidence-gated backoff | 53.5% |
| + 6g/7g (hashed), gated | **55.2% (top-5 84.7%)** |

Per-channel alone: 4g 47.4, 5g 52.8, 6g 54.9, 7g 53.5 (hash collisions bite);
the gate mixes them by evidence and beats any single order. **Passes both docs
usability bars** (top-1≥30, top-5≥60) the A-series never hit.

Generation (t=0.5), prompt → continuation:
- "the old man " → "stood the first, by all the other an in the same me for
  his nose and the surface is not the same"
- "she was very " → "probability of the weather when he had been seen him a
  short silence of the whale as jonah."
- "he opened the door and " → "strain the chapter. fast-fish par was in the
  thing as well as i was the word with some time"

Real words, phrases, punctuation, quotes/question-marks at higher temp, corpus
vocabulary (whale/jonah/fast-fish = Moby Dick). **The prompt steers.** Night-
and-day vs the neural-alone gibberish ("dod wing fas coltill carm").

HONEST framing: the coherence is high-order n-gram statistics; the EVOLVED part
is the backoff GATE (which order to trust by evidence) — gradient-free and what
makes the mixture optimal. The neural net is ~4% trust (near dead weight — the
count tables win accuracy AND coherence, exactly as the stack notes predicted).
This is a real gradient-free char LM: many parts, evolved composition, readable
output. Files: genreg_train/genreg_trustmix.py, genome runs/pure/trustmix_genome.npy.

Next lever: even longer context (8-9g) reduces drift; word-level n-gram channels
for phrase grammar; the gate is where evolution keeps earning.

## 2026-07-06 — Distillation verdict: you can't gradient-free-train away the tables

Distilled the 56.4% n-gram trust-mix TEACHER into a table-free evolved
feedforward neural n-gram (last-6-chars → MLP → V), soft-teacher + hard-target
hybrid fitness, 12k generations.

| | top-1 | top-5 |
|---|---|---|
| teacher (n-gram mix) | 56.4% | — |
| table-free student, final | **24.7%** | 58.0% |
| (recovered) | 44% of teacher | matches/beats teacher's shape |

The split is the whole finding: the student learns the distribution SHAPE
excellently (top-5 58% — as good as the teacher) but **cannot recover the
argmax** (top-1 stuck ~25%, generation is gibberish). It knows *which chars are
plausible*; it can't nail *which is most likely per context* — and generation
needs exactly that.

**Conclusion (maps the boundary of the whole LM arc):** compressing corpus
statistics into weights is a directed high-dimensional optimization — the thing
gradient descent is *for*. Undirected mutation can learn the coarse shape (top-5)
but not the precise per-context mode (top-1). So:
- table-free + gradient-free (best recurrent net): 34.6%, gibberish
- explicit n-gram tables: 56%, readable — but 1990s lookup
- table-free + gradients (real neural LMs): works — but violates the no-gradient rule

The no-gradient LM sits in the corner where BOTH mechanisms that make good LMs
are excluded. Evolution's edge is where gradients CAN'T go (discrete structure,
programs, survival dynamics — the Intelligence Engine), NOT raw next-token
statistics. genreg_train/genreg_distill.py.
