# How the Intelligence Engine Works

A practical and theoretical guide to the self-replicating genome system.

---

## 1. The Core Idea

Most ML systems train a network to perform a task, then measure success by a score function. The Intelligence Engine inverts this. There is **no task and no score**. There are only two rules:

1. **Survive** — never let your energy hit zero
2. **Reproduce** — produce a mutated copy of yourself that can also survive and reproduce

A genome is "fit" only if its lineage continues. Fitness is recursive: your offspring must themselves be fit, and theirs must be fit, all the way down. There is no fixed objective. Whatever survives, survives.

This is the simplest possible expression of the GENREG thesis: **don't design the solution, design the conditions where the only stable attractor is the solution.** Intelligence is what falls out when survival pressure runs over enough generations.

---

## 2. The Six Signs of Life

The system instantiates all six biological signs of life rather than simulating them. The hypothesis: intelligence is what these six look like from the inside when running simultaneously under pressure.

| Sign | How it's instantiated |
|---|---|
| **Organization** | Each genome has structured neural architecture: encoder + protein cascade + controller + output heads |
| **Metabolism** | Energy is drained every step (passive cost) and restored only by standing on food patches |
| **Growth** | Structural mutations can change activation functions, swap neurons, shift protein dynamics |
| **Response to stimuli** | Protein cascade reads environment signals every step (food gradient, neighbors, own state) |
| **Reproduction** | When energy crosses a threshold and the filter passes, a mutated child is created |
| **Adaptation** | Lineages with fertile descendants persist; bad lineages die off |

Removing any one of these collapses a dimension intelligence requires:

- No metabolism → no cost to decisions, no pressure to be efficient
- No response → closed loop, can't model the environment
- No reproduction → adaptations die with the individual
- No adaptation → reactions without memory
- No organization → complexity without structure
- No growth → can react and reproduce but never improve

---

## 3. Architecture Per Genome

Each genome is a small GENREG-style neural network. Per genome:

```
Inputs (32 dims)
  ├── Environment signals (16): own energy, food gradients, neighbors, etc.
  └── Self-perception (16): a window of own encoder weights — lets the genome
                            "feel" its own structure
        ↓
Encoder (32 → 64) with evolved activation function (one of 8 catalog functions,
                                                     per-genome parameters)
        ↓
Protein Layer 1 (64-dim, fast decay)
  Maintains short-term context — what just happened
        ↓
Bridge (64 → 64 linear)
        ↓
Protein Layer 2 (64-dim, slow decay)
  Maintains long-term context — patterns over many steps
        ↓
Concatenated context (128 dims = L1 + L2)
        ↓
Controller (128 → 128 hidden, tanh)
        ↓
Output heads:
  ├── Mutation Head (128 → 16): directed mutation deltas to apply to self
  │                              when reproducing (shapes its own offspring)
  └── Action Head (128 → 5): logits for the 5 possible actions
        ↓
Action chosen by argmax: UP / DOWN / LEFT / RIGHT / REST
```

**Why two protein layers?** Layer 1 (fast) tracks "I just ate" or "I just got hit." Layer 2 (slow) tracks "this region has lots of food" or "I've been in this neighborhood for a while." The controller sees both perspectives concatenated and decides what to do.

**Why the self-perception input?** A self-replicating organism has to know something about itself to copy itself. A fixed window of the encoder weights gives the genome a stable "view" of part of its own structure. This is what makes the network capable of producing meaningful mutation deltas instead of random noise.

**Total parameters per genome**: ~25,000. Small enough for hundreds of genomes on a single GPU, large enough to learn nontrivial behavior.

---

## 4. The Environment

A 64×64 toroidal grid (wraps around at the edges) populated with **food patches** — energy sources that genomes must navigate to and exploit.

### Food patches

- 180 patches scattered randomly at startup
- Each holds energy (starts at 1.0)
- When a genome stands on or near a patch (within 1 cell), it automatically consumes 0.05 energy from the patch and gains 4.0 energy
- Patches slowly regenerate (0.5% chance per step that a depleted patch grows back)
- A patch becoming empty turns dark; a fresh patch is bright green

### Genomes

- Each genome occupies one grid cell (multiple genomes can share a cell)
- Color in the GUI represents energy: red = starving, yellow = surviving, green = thriving
- Position changes only when the genome chooses a movement action

---

## 5. The Action Space (5 deterministic options)

| ID | Action | Effect | Cost |
|---|---|---|---|
| 0 | UP    | y -= 1 | 0.5 energy |
| 1 | DOWN  | y += 1 | 0.5 energy |
| 2 | LEFT  | x -= 1 | 0.5 energy |
| 3 | RIGHT | x += 1 | 0.5 energy |
| 4 | REST  | stay still | 0 energy |

Plus a passive drain of **0.3 energy per step** for everyone, just for existing.

The genome's brain controls one thing: where to move. Everything else is automatic.

---

## 6. Two Things That Are NOT Decisions

Eating and reproducing are **biological consequences**, not decisions. A cell doesn't decide to absorb a nutrient — it absorbs because it's in contact with one. A cell doesn't decide to divide — it divides when its internal state triggers division.

### Auto-consume

If a genome is standing on or adjacent to a food patch with energy > 0:
- Patch loses 0.05 energy
- Genome gains 4.0 energy
- Happens automatically every step

This means the learning task is **navigation**, not strategy. Get to the food, stay near the food, and you eat for free.

### Auto-reproduce

When a genome's energy crosses **280** (started at 200) — or stays above 330 — the system flags it as wanting to reproduce. If the **reproduction filter** passes (see next section), reproduction happens:

- Parent pays 80 energy
- A dead slot in the population is found (or one is created by displacement)
- The parent's weights are copied to the slot with mutations applied
- The child starts with 100 energy and the parent's position
- Lineage tracker records the birth

If no filter is set or no dead slots are available, reproduction fails silently.

---

## 7. The Reproduction Filter — The Lever

This is the most important control mechanism in the system. A genome only reproduces if it passes a configurable filter consisting of multiple criteria.

### Available criteria

| Criterion | Meaning |
|---|---|
| **energy_positive** | Energy delta over recent window > 0 (you're getting better, not worse) |
| **age** | Survived at least N steps (not a baby) |
| **energy_threshold** | Current energy above a threshold (you're well-fed) |
| **not_frozen** | Action distribution has nontrivial entropy (you actually do things) |
| **lineage_fertile** | Has at least one offspring already (you're a proven reproducer) |
| **structural_novelty** | Differs from population mean encoder weights (you're not a clone) |

### How the filter is composed

Set `required_count` to control tightness:

- **1 of 6** — very loose, almost any old genome reproduces (early generations)
- **3 of 6** — moderate, must be doing well in multiple ways
- **6 of 6** — extreme, only the best lineages get to reproduce

### Auto-tightening

The filter ramps up automatically as the population matures. Every N generations, the required count increases by 1, up to a maximum. This mimics biology: early ecosystems accept anything that breathes, mature ecosystems demand specialization.

### Why this matters

The filter is the **landscape lever** that GENREG theory says is the only thing that matters. Loose filter = wild diverse exploration. Tight filter = slow careful evolution. The shape of intelligence that emerges depends entirely on what the filter selects for.

---

## 8. Mutation — How Children Differ From Parents

Two kinds of mutation happen during reproduction:

### Value mutation (every birth)

Every weight tensor in the child is the parent's tensor plus gaussian noise. The noise scale and probability are **per-genome evolved traits** — each genome has its own `mut_rate` and `mut_scale` that themselves mutate slowly over generations. Populations self-organize their reproductive strategies.

Additionally, the parent's **mutation head** outputs 16 directed mutation deltas during the step, and these are added to a stride window of the child's encoder weights. This is the genome shaping its own offspring — not random drift but intentional change.

### Structural mutation (rare)

About 2% of births trigger a structural mutation. One of:

- **flip_activation**: change which of the 8 evolved activation functions the genome uses (and reset its parameters)
- **swap_neuron**: replace one encoder neuron with random new weights
- **reset_protein**: dramatically reshuffle one protein cascade layer's decay/momentum
- **shift_decay**: large random shift to a protein decay rate

Structural mutations drive open-ended evolution by occasionally breaking symmetries that value mutation can't reach.

---

## 9. The Main Loop — Step by Step

Every step, the engine runs this sequence:

1. **Get environment signals** — for each genome, build a 16-dim vector containing its energy, age, food gradient in 4 directions, neighbor density, etc.

2. **Forward pass** — every genome runs its full network: encoder → protein layers → controller → action head + mutation head. Output: action and mutation deltas.

3. **Apply actions** — movement is performed. Energy is drained for moves and passive existence.

4. **Auto-consume** — every alive genome that's on or near a food patch automatically gains energy.

5. **Mark deaths** — any genome whose energy hit zero is removed from the alive set. Lineage tracker records the death.

6. **Update history** — recent energies and actions are appended for the filter to use.

7. **Filter check** — the reproduction filter evaluates all alive genomes. Returns a boolean mask of who is allowed to reproduce.

8. **Reproduce** — for every genome that wants to reproduce AND passes the filter:
   - Find a dead slot (slots from genomes that died this step or earlier)
   - Pay the reproduction cost
   - Apply value + structural mutations to copy parent weights into the slot
   - Reset the child's environment state (energy, position, age)
   - Record the birth in the lineage tracker

9. **Snapshot stats** — population size, generation max, mean energy, criteria pass rates, etc. The GUI reads from this snapshot.

10. **Increment step counter** — repeat.

There is no training, no gradient, no backprop. The population evolves by selection alone.

---

## 10. Lineage Tracking

Every birth and every death is recorded. The tracker maintains:

- A dict of all genomes ever born, indexed by ID
- Parent-child relationships (so we can walk the family tree)
- Per-generation lists of births
- Death timestamps and lifespans
- Snapshots of population stats over time

This data powers the GUI panels: top lineages by descendant count, lineage depth, the 6 signs of life evaluation per genome, and the historical plots.

---

## 11. Reading the GUI

### Top bar
- **Initialize**: build the engine with current Pop / Required settings
- **Start / Stop**: run the background loop
- **Step 1**: advance exactly one step (good for debugging)
- **Reset**: tear down and rebuild with current settings

### Population Grid
The 64×64 spatial view of the world. Dark green dots are food patches (brighter = more energy). Colored dots are genomes (red = dying, yellow = okay, green = thriving). You can click a dot to inspect that genome.

### Six Signs of Life
For a sample of alive genomes, shows how many demonstrate each sign:
- **Green dot** = >70% of sample passes this sign
- **Orange dot** = 30-70%
- **Gray dot** = <30%

Watch this panel to see which dimensions of life the population is achieving.

### Reproduction Filter
Shows the current required count (e.g., "1 of 6") and a bar chart of how often each criterion is currently passing. If a criterion is at 0.92, almost everyone passes it. If it's at 0.06, almost no one does.

### Stats Over Time
A small line plot showing:
- **Green** = number alive
- **Yellow** = mean energy
- **Blue** = max generation reached

### Lineage / Top Descendants
Lists the lineages with the most living descendants. A lineage with many descendants has been "successful" by the only metric that matters: it propagated.

### Genome Inspector
Click a dot to see that genome's ID, parent, generation, energy, action history, mutation traits, and which signs of life it currently demonstrates.

### Event Log
Initialization, start/stop, errors, extinctions.

---

## 12. What You Should See If It's Working

A successful run, in rough order:

1. **First few steps**: random scatter, mostly red dots, energy plummeting
2. **By step ~50**: most starting genomes are dead. The few that wandered onto food patches are still alive (yellow/green)
3. **By step ~200**: those survivors have reproduced. Green clusters appear around food patches. Lineage panel shows 2-5 dominant lineages
4. **By step ~500**: max generation crosses 10. Filter starts auto-tightening (required goes from 1 to 2)
5. **By step ~1000**: stable equilibrium emerges. A few hundred genomes alive, mostly green, distributed across food patches. Some food patches are "owned" by descendants of the same root genome
6. **By step ~3000**: structural mutations have produced specialist sub-lineages. Different protein activation patterns dominate different patches

If instead you see immediate extinction, the food/energy balance is wrong (try Reset with more food or fewer genomes). If you see an unchanging blob of red dots, evolution hasn't kicked in yet — give it time.

---

## 13. Connection to the GENREG Thesis

The GENREG framework rests on one principle: **the fitness landscape is the only lever that matters.** Architecture is downstream of landscape. Optimizer choice is downstream of landscape. Whatever survives is whatever the landscape allows.

The Intelligence Engine is the most extreme expression of this principle:

- The "fitness function" is just **don't hit zero energy**
- The selection mechanism is just **dead slots in the population**
- The reproductive bar is the **filter criteria**
- Everything else is consequence

There is no objective function. There is no loss. There is no reward. There is only a metabolic substrate, a movement primitive, a reproduction primitive, and a filter that gates which genomes get to copy themselves.

If intelligence emerges from this — if genomes learn to navigate, find food, cluster around resources, develop specialized behaviors, build lineages with characteristic traits — it will not be because we trained them to. It will be because survival pressure on a designed substrate has only one solution, and that solution looks like intelligence from the inside.

---

## 14. What's Not Yet Implemented

Honest list of things this engine does NOT do (yet):

- **Combat / predation** — genomes don't eat each other
- **Communication** — no signals between genomes
- **Multi-cell organisms** — every genome is independent
- **Semantic memory** — protein cascade is the only memory mechanism, no long-term storage of "where I've been"
- **Energy gradient** — all moves cost the same regardless of terrain
- **Reproductive sex** — currently asexual cloning + mutation; the crossover trait exists but is unused
- **Evaluation harness** — no benchmark suite to compare runs

These are deliberate omissions to keep the system minimal. The thesis is that intelligence emerges from the minimum viable substrate. If this engine produces intelligent behavior, none of these features are necessary. If it doesn't, we add them one at a time and see which one tips the balance.

---

## 15. Tuning Cheat Sheet

If genomes go extinct immediately:
- Reduce `Pop` (fewer mouths to feed)
- Increase `N_FOOD_PATCHES` in `energy_environment.py`
- Increase `PATCH_REGEN_RATE`
- Decrease `PASSIVE_DRAIN`

If genomes never die and never evolve:
- Decrease `N_FOOD_PATCHES` (more competition)
- Increase `Required` count in the filter
- Increase `COST_REPRODUCE`

If lineages die off after a few generations:
- Lower `REPRO_ENERGY_THRESHOLD` (easier to reproduce)
- Make the filter looser (fewer required criteria)

If the population is stable but boring:
- Enable more structural mutations (raise `base_rate` in `code_mutator.structural_mutation_check`)
- Enable auto-tighten on the filter so pressure ramps up over time

---

## 16. Files

```
INTELLIGENCE_ENGINE/
├── README.md                       Project overview
├── HOW_IT_WORKS.md                 This file
├── core/                           Copied GENREG infrastructure
│   ├── genreg_gpu_v5.py            Latest evolver
│   ├── genreg_genome.py            Base genome class
│   ├── genreg_encoder_gpu.py       Evolved activation catalog
│   ├── genreg_proteins.py          Protein cascade
│   └── ...
├── self_replicating_genome.py      Population class — networks that output mutated copies
├── code_mutator.py                 Value + structural mutation operators
├── reproduction_filter.py          X-criteria gate (auto-tightens)
├── energy_environment.py           Spatial metabolic environment
├── lineage_tracker.py              Family tree + 6 signs of life
├── intelligence_engine.py          Main loop (CLI runner)
└── gui.py                          Tkinter dashboard
```

---

## 17. Running It

```bash
cd INTELLIGENCE_ENGINE

# Headless training (CLI)
python intelligence_engine.py --pop 200 --required 1 --steps 5000

# Interactive GUI
python gui.py
```

In the GUI:
1. Set Pop and Required values at the top
2. Click **Initialize**
3. Click **Start**
4. Watch the population evolve, or click dots to inspect individual genomes
5. Click **Reset** to rebuild with new settings

---

*Built as part of the GENREG research program. The Intelligence Engine is an experiment in instantiating life rather than simulating it. Whether it produces intelligence is an open question — that's the point of running it.*
