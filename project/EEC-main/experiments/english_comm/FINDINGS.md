# English-grounded communication — findings

Goal: get evolved agents to communicate in actual **English**, not just an arbitrary
emergent code. Built on three earlier lessons: arbitrary codes drift; a frozen grounded
channel + evolved residual preserves structure; coupled survival (and sexual reproduction)
make communication evolve.

Setup: a referential game over 10 meanings with real English names, 16 possible signal
words (10 English + 6 distractors). Agents carry an evolved speak/listen residual on top
of an optional **frozen English prior** (anchor). Coupled-survival selection, gradient-free.
`english_usage` = fraction of meanings spoken with the correct English word (chance ≈ 0.06).

See `english_findings.png` (panels A/B/C below).

## Results

**A. A frozen English prior produces English.** With no prior (anchor 0) agents converge
to an arbitrary code (english_usage ≈ 0.03–0.05). Adding the prior scales usage up: anchor 1
→ 0.71, anchor 2 → 0.95, anchor 3 → 1.00. Grounding, not pressure, is what makes the code
*English*. Transcripts are human-readable: `food→food, run→run, danger→danger`.

**B. Sexual reproduction protects English under noise.** Under per-generation channel noise,
clone reproduction lets English erode (0.95 → 0.35 as noise rises to 0.3); sexual
reproduction holds it (≈ 0.75). Crossover of two English-speakers stays English; cloning
propagates each individual's drift. This is this session's reproduction finding applied to
language — sex stabilises a shared code.

**C. English can be ACQUIRED from exposure, with no innate prior.** Install fixed native
English speakers that residents must understand to eat. Starting from anchor 0, residents
climb well above chance — a dose-response in the fraction of natives, saturating around
0.5–0.54 at 40–50 % native exposure. At 50 % natives the transcript is almost entirely
readable English (`sleep→sleep, red→red, big→big, cold→cold`).

## Honest limitations

- **Acquisition saturates at a PIDGIN (~half English), not full fluency.** Resident-to-resident
  talk never *requires* English, so only the native-relevant words get English-ified; the rest
  stay arbitrary. This is a realistic creolisation equilibrium, not a failure — but it is not
  full English from exposure alone.
- **Making English the sole food source backfired** (english_usage ≈ 0.17): native encounters
  are too rare, so the selection signal is sparse and noisy. Pure pressure without frequency
  is weak.
- **The "English" here is a 10-word identity lexicon** — no grammar, no compositionality, no
  real corpus statistics. It demonstrates *grounding + stability + acquisition* of a vocabulary,
  not syntax. Real English structure (bigrams, composition) is the next step.
- Acquisition curves are noisy at low seed counts; treat panel C as directional.

## D. Compositional English (zero-shot generalisation)

`compositional.py`: two-word messages [attribute, object], 4x5 = 20 compound meanings,
6 held out (never trained). Generalising to held-out combos = compositionality.

- No grounding (anchor 0): train acc ~0.30, held-out ~0.14 (≈ chance). Arbitrary codes do
  NOT generalise — each combination would need its own symbol.
- Grounded (anchor 2): train 0.93–1.00, **held-out 0.88–1.00**. Sexual reproduction gives a
  clean 1.00 / 1.00. Agents emit correct novel phrases zero-shot: `cold water`, `red food`,
  `big friend` — combinations never seen in training.

Grounding turns the vocabulary compositional: a consistent per-word meaning lets agents
recombine words into novel correct phrases. This is the step from words toward grammar,
and it only appears with grounding.

## E. Real-corpus grounding (the toy lexicon replaced with real English)

`build_corpus_grounding.py` + `corpus_comm.py`. Grounding now comes from the actual corpus
(`engine/corpus.txt`, ~2M tokens of real prose): a 60-word content vocabulary, real
frequencies, and PPMI-SVD embeddings (real semantic geometry — `house ~ door`, `right ~ left`).
The listener decodes a signal into the real EMBEDDING space and picks the nearest word, so
mistakes are forced through real semantics. See `corpus_findings.png`.

- **Real-English naming:** grounding → english_usage 0.77 over 60 real words (vs 0.02 arbitrary;
  chance 0.017). Transcripts read as real English: `pierre→pierre, room→room`.
- **Errors follow real semantic similarity (key result):** confused words are far more similar
  to the target than chance (+0.24 vs +0.11). Actual confusions are linguistically real:
  `right→left` (sim 0.89), `time→same` (0.71), `man→like`, `asked→looking`. The system confuses
  *related* words, not random ones — the same failure signature real language models show.
- **Zipf effect:** frequent words are communicated more reliably (corr(log-freq, accuracy)
  ≈ +0.1 to +0.2; modest, and bounded by the narrow frequency band of a top-60 vocabulary).

This is genuine real-corpus structure emerging in evolved agent communication: real words, real
semantic confusion geometry, real frequency effects — not a hand-built toy.

## F. Emergent SYNTAX from a relational world (no grammar wired in)

`relational_syntax.py`. Paradigm-correct response to "where does word order come from": not a
frozen bigram channel, but a WORLD that demands it. Events are (agent, action, target) with
agent and target from the same entity pool, so "A chases B" and "B chases A" are the same
symbols in different roles. The channel is sequential (3 symbol slots); survival is graded by
how many roles the listener recovers. The per-slot speak maps and the role-readers are all free
and random — nothing says "slot 0 = agent".

- **Order-based role-marking emerges.** Full-event accuracy reaches ~0.29 (chance 0.017, ~15x).
  Action is conveyed reliably (~0.90); the entities less so (~0.5–0.65).
- **Word order is load-bearing (key result).** Scrambling the message slots before decoding
  collapses accuracy from 0.29 to 0.07. Meaning lives in the ORDER, not just the symbols —
  the signature of syntax.
- **The world requires order.** A control listener given only the BAG of symbols (positions
  destroyed) is capped at full ~0.14 vs ~0.24 for the sequential channel: it cannot distinguish
  agent from target, exactly because "A acts B" and "B acts A" have identical bags. Order earns
  its keep by disambiguating who-does-what-to-whom.
- **Partial compositional generalisation:** held-out events (never trained) reach ~0.07 vs 0.017
  chance, and remain order-dependent (scrambled → ~0.02).

Honest limits: absolute accuracy is modest (~0.24–0.29 full event); the protocol confuses
entities often and generalises only partially. This is a proof that relational structure DRIVES
the emergence of word order under selection — not a high-performing grammar. Scaling accuracy
(bigger populations, longer evolution, curriculum on swap-pairs) and richer syntax (recursion,
multi-clause) are open. But the mechanism is the paradigm-correct one: order emerged because the
world had relationships single symbols could not express.

## G. Richer world -> constituent structure (binding)

`richer_syntax.py`. Next rung: entities are COMPOSITE -- (attribute, type) -- and several things
share a type, so "big wolf chases small deer" vs "small wolf chases big deer" are the same
symbols, different meaning. Survival now needs BINDING (which attribute goes with which noun).
Credit is per-ENTITY: an entity counts only if attribute AND type are both right and bound to
the correct role -- so wrong binding earns nothing. Nothing wired in.

- **Structure beats the bag on binding:** sequential channel binding 0.28 vs order-free bag 0.19.
  When binding is load-bearing, the sequential channel develops grouping the bag cannot.
- **Order is load-bearing:** scrambling collapses the sequential channel (0.034 -> 0.004); the
  bag is unaffected (it never used order). Same syntax signature, one level up.
- Honest limit: absolute accuracy is LOW (full 5-component event ~0.03, ~14x chance but small).
  The composite-binding task is at/near this evolutionary substrate's ceiling -- per-component
  recovery is ~0.5 but getting a whole bound event right is rare. The MECHANISM (richer relations
  -> grouping emerges, order load-bearing) is shown; high-accuracy compositional binding is not.

The progression across E-G is the real claim: single concepts need no syntax; relations make
word ORDER pay; composite entities make GROUPING pay. Each rung of structure appears only when a
richer world makes it the cheapest way to survive -- never by wiring it in. Scaling the accuracy
(curriculum, larger populations, a recurrent organism with real working memory) is the open front.

## H. A constraint that makes MEMORY evolve

`memory_evolve.py`. The syntax work stalled because the flat organism had nowhere to HOLD a
constituent. The fix is two-part, and the split is the whole paradigm:
  - CAPABILITY (substrate, not answer): a recurrent state register h (dim H); its read/write
    weights are free and evolved -- nothing says what to store. A body part, like a vision gene.
  - CONSTRAINT (the world law): perception distributed over TIME -- a stimulus arrives one symbol
    per step, each gone on the next, and survival needs the WHOLE history (reproduce the sequence).
    No single moment suffices, so internal state is the only bridge from past to response.

Result (recall of a 4-symbol sequence from the final state; chance per position 0.12):
- **No memory** (register severed / H=0): recall `[0.12 0.12 0.13 0.94]` -- only the PRESENT
  (last) symbol is recoverable; the past is at chance.
- **Memory evolved**: recall `[0.45 ...]` -- it carries a symbol from 3 steps ago forward through
  every update (0.45 vs 0.12), even sacrificing the free present symbol to hold the past.
- **Ablation proves storage in the register**: oldest-symbol recall 0.47 intact -> 0.14 severed.
- **Capacity scales it**: H=0 -> chance (0.13); H>=4 -> above chance (0.26).

Honest limit: gradient-free evolution finds an IMPERFECT, low-capacity memory -- it latches one
salient item (often the oldest) rather than cleanly buffering all four (mean recall ~0.25). Real
memory (ablation-confirmed), but a single-slot grip, not a working-memory buffer. High-capacity
sequential memory is its own hard search.

This is the missing organ for the syntax work (G): a world demanding temporal integration, plus a
register, makes memory evolve. Wiring the two together -- a recurrent listener under the relational/
binding worlds -- is the route to higher-accuracy syntax. The world demands the structure; the
register gives it somewhere to live.

## I. The genome INVENTS its own memory (not a register we hand it)

`memory_invent.py`. Section H still wired in the answer: we gave a recurrent register AND its
update rule h=tanh(Wx x + Wh h + b), and evolution only tuned the dials. The paradigm-pure question
is whether the genome can BUILD a memory mechanism from primitives we never designated as memory.

Substrate: the organism is strictly FEEDFORWARD -- action_t = f(perception_t), no internal state,
no update rule. The only persistence is in the WORLD: one slot that simply keeps whatever was last
written (like a patch of marked ground). The feedforward policy emits each step (response, write?,
what-to-write). Nothing labels the slot "memory". Task: output a CUE seen L-1 steps earlier (gone
from view) -- bridgeable only by storing it in the world and reading it back.

- **Slot disabled** (no persistence): recall 0.17 = chance. A feedforward organism cannot reach the past.
- **Slot writable**: recall 0.73. The genome invented a store-and-recall policy.
- **What it invented** (trace): it writes ONLY at the cue step (conditioned on the slot being empty),
  preserves through the distractors, and reads at the end -- AND it invented its OWN ENCODING:
  cue 0->store 4, 1->store 2, 2->store 1 (an arbitrary internal code, not the literal cue), decoded
  on read-out. We prescribed none of write-timing, encoding, preservation, or read.
- **Ablation**: same organism 0.99 with slot, 0.16 (chance) with slot disabled -> the memory is
  genuinely external; the organism is genuinely memoryless; it built the protocol itself.

This is the honest answer to "make memory EVOLVE, not give it memory": a feedforward body + a world
that merely persists + a task that needs the past = the genome invents memory, including its own
representation of what to store. Memory was not provided architecture; it was an evolved use of the
world's persistence. (Open: a single world slot is a 1-item store; multi-slot / spatial-trail
substrates and the capacity/competition between them are the next question.)

## J. From one mark to ORGANIZING SPACE (external spatial memory)

`spatial_memory.py` (the wall) and `spatial_seq.py` (the breakthrough). Next jump after I: not
one slot but a TAPE of cells with a movable head -- decide not just whether to write but WHERE,
and navigate back to read. Still strictly feedforward, no internal state; the world's persistence
now includes the tape contents AND the head position.

- The index-query version (recall an arbitrarily-asked item) HIT A REACHABILITY WALL: spreading
  items pays nothing unless navigation also works, so no gradient. The organism wrote everything to
  cell 0 and guessed; W=4 (0.42) barely beat W=1 (0.38); freeze-head ablation showed no reliance on
  space (0.46 vs 0.45). Honest negative.
- The fix (graded world, no answer wired in): STORE a sequence and REPRODUCE it -- storing one more
  item correctly is one more point, so partial spatial use is rewarded. Then it emerged cleanly:
  W=1 reproduced 0.42 (one-cell ceiling) vs W>=3 = 0.96 (whole sequence).
- What it invented: lays the sequence ACROSS cells by moving the head, and -- as in I -- its OWN
  per-cell encoding (e.g. stores cue 0 as 2, decodes on read). Spatial layout AND cipher, both
  unprescribed. Ablation airtight: movable head 0.95 -> frozen head 0.38 (back to one-cell ceiling).

This is the leap from internal mark to organized external store -- the seed of a map / externalised
knowledge system, held in the world's geometry, not the head. The lever was again P1: the world's
reward had to make the first steps survivable (graded), not any wiring of layout or navigation.
Open: longer sequences, larger spaces, and CONTENT-addressed retrieval (find by what, not where).

## K. Scaling the invented faculties, and the content-addressing frontier

`scaling_harness.py` -> `scaling_results.txt`. We make only the WORLD harder (longer delays,
longer sequences, more pairs); no reward changes, nothing wired in. Honest reading: a low score
at large size can be reachability (more generations/curriculum needed), not a hard ceiling.

- **Invented memory is DELAY-ROBUST.** Single-world-slot recall stays 0.48-0.76 across delays
  L=2..12 (chance 0.17), with no systematic decay. Because the store is in a world that PERSISTS,
  time does not erode it -- unlike the given recurrent register (H), which degraded with distance.
  External storage genuinely does not forget.
- **Spatial sequence memory: ~3-4 item capacity.** Reproduction peaks at length 3 (0.96) and falls
  off (length 5: 0.58, length 6: 0.35). Organising longer sequences across the tape exceeds this
  substrate's budget.

### Content-addressing is the wall (find by WHAT, not where)

`content_address.py` (linear) and `content_address_v2.py` (nonlinear). A world that DEMANDS content-
addressing was built: key->value pairs in RANDOM arrival order, so position cannot be the address;
only the key can. Retrieval stays at/just above the guessing floor across three principled tries:
- linear policy: ~0.44 (degenerate fixed guess; a single linear layer cannot even compute
  'move toward the cell indexed by the key' -- a key x position JOIN -- an expressivity wall);
- + nonlinear hidden layer (now expressible): still ~0.31 (= 1/K floor) -- converges to a fixed
  output per query, ignoring storage;
- + large value space (V=20, so guessing is worthless, opening a gradient) and distinct values:
  retrieval 0.10 vs 0.05 floor -- it reaches 'store ONE item and sometimes retrieve it', but never
  two items addressed by key. Ablation shows no real use of the head/space throughout.

Honest conclusion: associative / content-addressed memory is a genuine frontier for this gradient-
free substrate. There is a clear capability hierarchy -- external storage (reachable, robust) <
spatial/sequential organisation (reachable, small N) < content-addressing (NOT reached). The last
needs a symmetric key->location function applied identically at write AND read, which mutation+
crossover does not find here. We deliberately did NOT keep throwing architecture/compute at it
(that is the gradient-thinking trap). The paradigm-pure open question is whether a more REACHABLE
world -- e.g. an embodied foraging world where an organism must return to where it found a
resource (spatial-associative, with a smooth incremental payoff) -- lets associative memory emerge
the way graded worlds unlocked syntax, memory, and spatial layout. That is the next experiment, not
a bigger brain.

## What this shows (and doesn't)

Shows: to communicate in *English* specifically you need a grounded shared prior (innate or
acquired from exposure); communicative pressure then *maintains* it, and sexual reproduction
*stabilises* it. Three independent levers, each measured.

Does not yet show: compositional/grammatical English, or full fluency acquired from exposure
alone. Those need a richer lexicon (real corpus grounding) and a compositional channel.
