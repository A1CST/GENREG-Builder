# SKELETON — full mechanics of the current model

Snapshot of the latest line: **the consequence-world generative organism** and its **fluency
extension**. Everything below is what actually runs in `consequence_world.py` (working coherence core)
and `fluent_world.py` / `curriculum_fluent.py` (grammar/fluency attempt). Read this top-to-bottom; it is
the whole machine.

---

## 0. LINEAGE / WHAT THIS IS

A population of tiny recurrent organisms that **generate a token stream from their own state** and are
selected purely on **how long they survive** while generating. There is **no n-gram channel** in the
generator — memory IS the generator. The defining law is **CONSEQUENCE**: an organism's output at step
T becomes its input at step T+1, so survival depends on the coherence of what it *generates*, not on
predicting a passive stream. Result already achieved: memory evolves (M→44) and generates topic-coherent
text (holds a topic across neutral gaps, 1.00). Open: grammatical fluency.

Two files:
- `consequence_world.py` — core. Coherence law only. **Works** (coherent generation).
- `fluent_world.py` — adds a grammar judge (fluency). `curriculum_fluent.py` grows that grammar. Grammar
  not yet learned (blocker: gradient-free can't evolve a generative grammar map).

---

## 1. THE ORGANISM (genome)

Each organism is a single-hidden-layer **recurrent net** plus six scalar control genes. Arrays are
allocated at a fixed MAX size; genes choose how much of each array is actually used (like memory M in the
engine). **Everything an organism uses is a gene** — no hardcoded organism constants.

### 1a. Scalar genes (all evolvable, all mutated)
| gene  | meaning                                   | init range      | mutation                         | clamp        |
|-------|-------------------------------------------|-----------------|----------------------------------|--------------|
| `ED`  | embedding dimension actually used         | int [4, 64]     | round(+N(0, ms·ED))              | [4, MAX_ED]  |
| `M`   | recurrent memory size actually used       | int [2, 48]     | round(+N(0, ms·M))               | [2, MAX_M]   |
| `decay`| per-organism state leak (entropy)        | U(0.5, 0.98)    | +N(0, 0.05)                      | [0.3, 0.995] |
| `mr`  | mutation RATE (frac of weights perturbed) | U(0.05, 0.30)   | ×exp(TAU·N(0,1)) self-adaptive   | [0.01, 0.6]  |
| `ms`  | mutation SCALE (relative step size)       | U(0.10, 0.40)   | ×exp(TAU·N(0,1)) self-adaptive   | [0.02, 0.8]  |
| `temp`| generation sampling temperature           | U(0.4, 1.1)     | +N(0, 0.1)                       | [0.2, 1.5]   |

### 1b. Weight matrices (all evolved by mutation)
Allocation caps: `MAX_ED = 64`, `MAX_M = 48`, `V` = vocabulary size (50 in consequence_world, 45 in
fluent_world). Only the `[:ED]` / `[:M]` slices are used in the forward pass.

| weight   | shape          | role                                  |
|----------|----------------|---------------------------------------|
| `E`      | (V, MAX_ED)    | token → embedding (uses `E[x, :ED]`)  |
| `W_in`   | (MAX_ED, MAX_M)| embedding → memory drive              |
| `W_rec`  | (MAX_M, MAX_M) | recurrent state → state (the memory)  |
| `b`      | (MAX_M,)       | state bias                            |
| `W_out`  | (MAX_M, V)     | state → token logits (the GENERATOR)  |
| `b_out`  | (V,)           | output bias                           |

There is **no embedding-input occlusion and no n-gram channel** in the consequence/fluent generator
(occlusion belongs to the earlier perception line; it is NOT in the current generative model).

---

## 2. FORWARD DYNAMICS (one step)

```
ed, M = org.ED, org.M
s_t = tanh( E[x_t, :ed] @ W_in[:ed, :M]        # input drive from current token
            + decay * ( s_{t-1}[:M] @ W_rec[:M,:M] )   # leaky recurrence  (entropy = decay<1)
            + b[:M] )                            # bias
logits = s_t @ W_out[:M, :] + b_out             # (V,)  -- generated from STATE ALONE
p = softmax(logits / temp)                       # sample with the organism's own temperature
y_t = sample(p)                                  # the emitted token
x_{t+1} = y_t                                    # *** CONSEQUENCE: output becomes next input ***
```

State carries across the whole life. Generation is autoregressive and **self-driven** — that is what
makes memory load-bearing (and what made earlier predict-trained memories collapse).

---

## 3. THE WORLD / ENVIRONMENT

### 3a. Vocabulary (`consequence_world.py`)
- `NEUTRAL` (14 words): `i you the to a it is we do so today really just and` — topic-ambiguous filler.
- 6 `TOPICS` × 6 content words (FOOD/WEATHER/WORK/SLEEP/SPORT/MUSIC). V = 14 + 36 = **50**.
- `NEU_IDS` = neutral token ids; `TOPIC_OF[id]` = which topic a content word belongs to.

### 3b. Vocabulary (`fluent_world.py`, fluency version)
- `FUNC` (15 words): `i am do you want to need a the is nice let us now today`.
- 5 `TOPICS`, each = (2 adjectives, 2 nouns, 2 verbs). V = 15 + 30 = **45**.
- `TEMPLATES` (the grammar), 5 frames:
  `i am ADJ today` · `do you want to VERB` · `i need a NOUN` · `the NOUN is nice` · `let us VERB now`.

### 3c. The consequence loop (a "life")
1. Prime the state with a couple of seed tokens (a neutral + a random content word).
2. For up to `L` steps: the organism generates (§2); its output feeds back as next input.
3. Each emitted token changes `energy` by the **coherence law** (§4). Lifespan = steps before energy ≤ 0.

---

## 4. THE LAWS / CONSTRAINTS (the world's energy economy — NOT genes)

These constants define the *environment*, not the organism. They are the levers I tune; the organism
evolves everything else against them.

### 4a. Consequence-world coherence law (`consequence_world.py`) — WORKS
Per emitted token `y` (running = the topic the organism is currently generating):
```
cost = BASE                                  # BASE = 0.5  (metabolism every step; silence starves)
if y is NEUTRAL:        prev_neu = True       # pacing word, just pays BASE
else (content word):
    ty = TOPIC_OF[y]
    if running is None: running = ty
    if ty != running:   cost += SHIFT; running = ty     # SHIFT = 0.3  (deliberate topic change, mild)
    elif prev_neu and y not in recent(3):  cost += REWARD  # REWARD = -2.5  (INCOME: held own topic
                                                            #   across a neutral gap, varied → memory pays)
    prev_neu = False
energy -= cost
```
- `START_E = 30`, `L = 300`.
- **Why this needs memory:** income only comes from continuing your OWN topic *after a neutral word*.
  A neutral word makes the input topic-ambiguous, so the organism must RECALL its thread → memory.
- **Why it can't be gamed:** silence (all-neutral) only pays BASE → starves (~60 steps); repetition is
  blocked by the `not in recent(3)` variety clause; off-topic chatter earns no income.

### 4b. Fluency extension (`fluent_world.py`) — grammar added as a JUDGE
A grammar bigram `glp[prev, y] = log P(y | prev)` is built once from a templated corpus
(`gen_corpus` → `grammar_logp`). It JUDGES the organism's transitions; it is **never fed to the
organism's generator**.
```
cost  = GW * ( -glp[prev, y] )               # GW = 0.5   grammar surprise: ungrammatical = expensive
if y in recent(4):           cost += REP     # REP = 1.0  variety / anti-collapse
if y is content and ty == running and y not in recent:
                             cost -= TOPIC_BONUS   # TOPIC_BONUS = 1.4  on-topic income (memory)
```
- `START_E = 26`, `L = 220`.
- Survival = low grammar surprise (fluent) AND on-topic varied content (memory).

### 4c. Curriculum (`curriculum_fluent.py`) — GROW the grammar (dissolve the no-foothold blocker)
- `LEVELS = [(1,1),(2,1),(2,2),(3,2),(3,3),(4,4),(5,5)]` = (#templates, #topics) the world grows through.
- Start at level 0; when best life > `MASTER = 0.85·L` for `HOLD = 3` gens, advance a level and rebuild
  `glp` from the bigger corpus. **The genome carries forward** (the foothold for the next addition).

---

## 5. ENERGY / SURVIVAL / FITNESS

- Energy starts at `START_E`, drains/gains by the coherence (and grammar) law per step, death at ≤ 0.
- **Lifespan** (steps survived) is the only fitness signal — never accuracy, never a target match.
- `fitness(org) = mean lifespan over 3 independent lives` (averages out generation-sampling noise).

---

## 6. POPULATION CONTROLS (world-level, not organism genes)

- `POP_SIZE = 64` (imported from `engine/evolve.py` — the world's carrying capacity).
- **Steady-state / overlapping generations** (`reproduce`):
  - Rank by fitness. **Top 80%** carried UNCHANGED into the next generation.
  - **Bottom 20%** (`N_BOTTOM = 0.2·POP ≈ 13`) culled.
  - Refill the culled slots with **mutated copies of the top 20%** (`N_ELITE = 0.2·POP ≈ 13` breeders,
    chosen uniformly among the elite each birth).
  - No crossover. Mutation only.
- Net effect: diversity persists (the middle 60% are never overwritten), elites breed, the weakest die.

---

## 7. MUTATION (self-adaptive)

`TAU = 0.25`. On reproduction a child:
1. Self-adapts its mutation params first: `mr *= exp(TAU·N(0,1))`, `ms *= exp(TAU·N(0,1))` (clamped).
2. Perturbs every weight RELATIVELY: for each weight `w`, with prob `mr`, `w += N(0,1)·ms·(|w|+1e-3)`
   (step scales with the weight's own magnitude — big weights never calcify, zeros can still move).
3. Mutates the structural/control genes: `ED`, `M` (relative steps ∝ ms), `decay`, `temp`.

So mutation rate and scale are themselves under selection — the organism evolves how it evolves.

---

## 8. GENE vs WORLD-LAW BOUNDARY  (the "everything evolvable" rule)

**Genes (organism evolves them):** `ED`, `M`, `decay`, `mr`, `ms`, `temp`, and all six weight arrays.
**World laws (the environment, I set them):** the vocabulary + topic structure, the templates/grammar,
the energy-economy constants (`REWARD/BASE/SHIFT/REP/GW/TOPIC_BONUS`), `START_E`, `L`, the curriculum
schedule, `POP_SIZE`, the steady-state carry/cull fractions, `TAU`.
Allocation caps `MAX_ED=64`, `MAX_M=48` are array sizes, not behavioral constants — the genes choose
within them (and in practice ED has run to its cap, so a real run should raise the cap, not hardcode it).

---

## 9. MEASUREMENT / METRICS (read the STATE, not accuracy)

- `held-own-topic-through-gaps` — fraction of paced content words whose topic matches the running topic
  (memory holding the thread). Chance = 1/#topics.
- `gram-surprise` — mean `-glp[prev,y]` over the generated stream (lower = more grammatical/fluent).
- `life` — lifespan / L.
- Plus the evolved genes themselves (M, ED, decay, temp) read off the best organism each checkpoint.
- A transcript of the best organism's actual generated tokens is printed for direct inspection.

---

## 10. CURRENT STATE / OPEN BLOCKER

- **Coherence: SOLVED.** consequence_world → M→44, held-topic 0.23→1.00, life→300/300; transcript holds
  one topic (e.g. SLEEP: sleep/rest/bed) across every neutral gap, never jumping. Memory IS the generator.
- **Fluency: BLOCKED (not a wall).** Even at the trivial 1-template level the organism cannot *evolve*
  the grammar-generation map (it never learns `i→am`). Identified cause: gradient-free mutation cannot
  search the weights for a generative bigram-quality map (same limit proven cold earlier). Proposed
  environment dissolution (not yet built): grammar is *structure*, and structure is **acquired from
  experience** (distributional learning, proven to hit 1.0) rather than evolved — give the organism
  within-lifetime exposure to grammatical text so the grammar becomes its OWN acquired structure (not a
  frozen channel that masks memory). Two-timescale generation: experience acquires grammar, evolved
  memory holds topic.

---

## 11. FILE MAP

| file                     | what it is                                                          |
|--------------------------|--------------------------------------------------------------------|
| `consequence_world.py`   | the working generative core (coherence law, full evolvable genome) |
| `fluent_world.py`        | adds the grammar judge + variety law (fluency attempt)             |
| `curriculum_fluent.py`   | grows the grammar level-by-level (genome carries forward)          |
| `consequence_results.txt`| the coherence-success run (M→44, held-topic→1.0)                    |
| engine `mind.py`/`evolve.py` | the original recurrent organism + energy-gated evolution this descends from |
```
