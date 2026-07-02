# GENREG Model Design Template

Fill this out BEFORE writing code. If you can't fill a section in plain
English without copying code, the design isn't clear enough yet.

---

## 1. Name
`<model_id>`. Examples: `tok_v4`, `embed_04`, `attn_v1`.

## 2. Purpose
One paragraph. What problem this model solves and why it exists in the
pipeline. What gets stuck or broken without it?

## 3. Interface

**Input** — what comes in. Shape, dtype, value range, example meaning.
Example: "A batch of words, byte-encoded. Shape `(N, 32)`, float32
values in [0, 1] representing each byte / 255. N is the number of words
being tokenized in a call."

**Output** — what goes out. Same format.
Example: "A token ID per input word. Shape `(N,)`, int64 in
`[0, V_TOK)`. Downstream models use this as a categorical feature."

**Runtime state** — does anything persist between calls?
Example: "Stateless — each call is independent." OR "Maintains a
cascade state of shape `(H,)` across a stream of calls; reset between
unrelated streams."

## 4. Evolved parameters

For each learned tensor, explain:
- **What it is** (embedding? weight? bias? bit projection?)
- **Shape** and which dim indexes what
- **How it's initialized** (random Gaussian, SVD of something, zeros)
- **Why that init** (prevents dead neurons, warm-start from baseline, etc.)

Then total it up: "~N params per genome."

## 5. Fitness equation

Write the full equation, then describe each term.

```
fitness = <whatever the equation is>
```

For **every term**:
- **What it measures** in plain English
- **Range** (e.g., 0..1 fraction, can be negative, etc.)
- **Why this signal** — what behavior it rewards/punishes
- **Why this weight** — what happens if you double or halve it

A "good" fitness should be explicable: "A genome scoring X means it's
doing Y. A perfect score of Z is achievable only if it does W."

## 6. Energy equation

GENREG energy is homeostatic, not a reward signal. It determines who
**survives** independent of rank. Describe:

```
energy_next = energy_current * DECAY + delta
delta = (signal)                              # what drives gain/loss
energy_next clamped to [0, E_MAX]
```

- **`ENERGY_DECAY`** — the "metabolism" rate. Lower = harsher drain.
  Typical 0.85-0.95. Too high (e.g. 0.99) and energy is inert
  (nothing starves); too low (e.g. 0.5) and genomes die before they
  can prove themselves.

- **`ENERGY_GAIN`** — multiplier on the signal feeding delta.
  Typical 1.5-3.0. Bigger gain = energy responds more sharply to
  relative performance. Set so the sign of delta actually matters
  (above-median gain meaningfully; below-median lose meaningfully).

- **`ENERGY_FLOOR`** — below this, the genome is starved and
  excluded from selection regardless of fitness rank. Typical
  0.15-0.25. This is the "hard cull" threshold.

- **`E_MAX`** — ceiling to prevent runaway accumulation.
  Typical 1.5. Without it, lucky early genomes stockpile energy
  and become immortal even if they later fail.

- **What signal drives delta?** Usually `fitness - median_fitness`
  (relative performance). Sometimes includes explicit costs (e.g.,
  per-step cost for outputting non-zero on padding) and bonuses
  (e.g., regen for correct outputs).

Explain what starved count you expect per gen. `starved = 0` means
energy is asleep. `starved > POP/2` means it's genocidal.
Target: 3-15% of population starved per gen.

## 7. Selection

- **`POP_SIZE`** — how many genomes per generation. More = more
  diversity but slower per-gen. 300-500 typical on a 15GB GPU;
  1000+ if you have VRAM.

- **`SURVIVAL_PCT`** — top fraction kept as "elite" between gens.
  Typical 20. Lower = harsher selection, may lose good genomes to
  batch noise. Higher = softer, slower convergence.

- **Maturation gate** (yes/no) — if yes, new offspring can't
  reproduce until they survive one full generation. Filters
  lucky-batch noise from the reproduction pool.

- **Reproduction method** — usually tournament-weighted sampling
  from the elite pool. Fitness weights become probabilities.

## 8. Mutation

- **`mut_rate` (per-genome, self-adapted)** — probability each
  parameter element gets mutated per evolve step. Typical starting
  value 0.03-0.08. Self-adapts over generations; bounds `[0.005, 0.2]`
  keep it from collapsing or exploding.

- **`mut_scale` (per-genome, self-adapted)** — size of the
  Gaussian perturbation applied when a parameter is mutated. Typical
  starting value 0.02-0.08.

- **Per-tensor scaling** — different tensor ranks mutate at
  different rates. 3D weights get `s.view(-1,1,1)`, 2D get
  `s.view(-1,1)`, scalars get simple scaling. Some tensors
  (activation IDs) have bespoke mutation rules.

- **Anneal policy** — does mutation scale shrink late in training?
  Example: multiply `mut_scale` by 0.4 after `ANNEAL_AFTER` gens.
  Gives a "coarse search → fine refinement" curriculum.

## 9. Hyperparameters

For each setting: value, what it controls, how you'd change it if
things go wrong.

- **`N_GENERATIONS`** — total evolution length. Set long enough that
  training is still improving at the end; if it flatlines for 1000+
  gens straight, you've over-committed.

- **`BATCH_SIZE` / `PROBE_SIZE`** — data seen per fitness eval.
  Bigger = less noisy fitness signal but higher memory. If
  population fitness variance is noisy gen-to-gen, batch is too small.

- **`LOG_EVERY`** — how often to print. 100 gens of noise is hard to
  read; too frequent and the log is a wall of numbers.

- **`ANNEAL_AFTER`** — gen at which mutation scale shrinks.

## 10. Success criteria

Two bars, both required.

- **Local bar** — standalone task performance that makes this
  component usable. Example: "uniqueness > 0.70 on a 2048-word
  heldout probe." If you don't clear this, the component is broken.

- **Downstream bar** — the metric that justifies the component
  existing. Example: "swapped into A_101's input slot, lifts heldout
  top-1 by at least +0.5pp vs the prior component." If you clear the
  local bar but not the downstream bar, the component is
  theoretically fine but practically useless.

## 11. Failure modes to watch

Enumerate specific ways this model could produce misleading numbers
or degenerate behavior. Examples:

- Training fitness climbs but held-probe fitness is flat →
  distribution mismatch between training samples and inference use.
- All genomes converge to identical output → mode collapse; mutation
  can't escape.
- Fitness reward-hacks: the model satisfies the fitness formula
  without doing the intended task (e.g., outputs a constant if the
  "consistency" term dominates).
- Starved count stays at 0 for 500+ gens → energy is inert, selection
  is pure tournament, energy system is decorative.
- Starved count is >50% of pop → energy is genocidal; good genomes
  die before proving themselves.

The point is to predict these in advance and define what log pattern
would reveal each one.

## 12. Baselines to beat

Name every baseline the final model will be compared against. Each
must be evaluated on the same task, same heldout data, same readout
procedure. Example:

- Random initialization + optimal closed-form readout
- Fixed hand-crafted solution (SVD of bigram matrix, hash of bytes)
- Prior model generation (tok_v2's 54% uniqueness)

Report all three in the final eval.

## 13. Artifacts

On completion, leave these behind:

- **`<model>_best.pkl`** — evolved params of the best genome, plus
  config and final scores dict. Loadable by downstream code.
- **`<model>_findings.md`** — what worked, what didn't, honest
  assessment of whether the model cleared its bars.
- **`run_<model>.log`** — per-gen training trace.

---

## Workflow

1. Draft this doc in full.
2. Sanity-check the design with whoever's around before writing code.
3. Implement matching the doc.
4. Train. If doc and code diverge mid-run, update the doc.
5. On completion, fill in the findings + bar-clearing status.
