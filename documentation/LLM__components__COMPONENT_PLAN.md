# GENREG Component-First Research Plan

The monolithic phase ended at 34% char top-1 (A_101). Gains came from
structural unlocks (recurrent encoder, prev_char concat) and gen
scaling. Further complexifications diffused selection pressure.

New phase: each sub-component of an LLM is its own GENREG model with
an isolated fitness task. Once a component hits its standalone bar,
it's frozen and composes with the next component.

Hard rules (carried over):
- No gradients, no backprop, no hybrid approaches.
- Energy homeostasis mandatory (starved > 0 each gen).
- Tournament selection with maturation gate.
- Iterations kept tight — typically 2-8k gens per component sweep
  unless a component genuinely benefits from more.

## Components

### 1. Embedding
**Task:** predict-surrounding-chars. Given a center char, output a
distribution over its neighbors (char_{t-1} and char_{t+1}). Evolved
embedding matrix E (V, D) + small readout head maps center char to
context distribution.

**Interface:** `E[char_id]` → D-dim vector.

**Fitness:** sum of log-prob of true neighbors across a batch of
positions, or top-1 accuracy on neighbor prediction.

**Bar to clear:** beat SVD-init-of-bigram-matrix embedding on this
task. Then test downstream: freeze E, plug into A_101 architecture,
measure heldout top-1 lift.

**Why first:** every downstream component uses an embedding. A
better embedding improves everything for free. Also the cleanest
isolated task — single-char in, neighbor distribution out, no
sequence modeling required.

### 2. Attention
**Task:** copy-with-offset. Given a sequence with a "flag" token
somewhere, the model must output what was k positions before the
flag. Forces evolved Q·K weights to learn selective retrieval.

**Interface:** attention(H_seq) → weighted sum of H_seq.

**Fitness:** top-1 on copy task with varying offsets and sequence
lengths.

**Bar to clear:** solve offset-k copy for k ∈ {1, 2, 5, 10, 20}
with sequence length up to 64. If attention can't do this reliably,
it can't support long-range dependencies in an LM.

**Why second:** the clearest missing capability in monolithic runs —
cascades smooth context but can't selectively retrieve. Attention is
the canonical answer, and it composes naturally on top of embedding.

### 3. Optimizer (meta-GENREG)
**Task:** minimize test functions (Rastrigin, Rosenbrock, etc.) using
an evolved mutation operator. The mutation operator IS the genome.

**Interface:** `mutate(parent_genome, fitness_history) → child_genome`.

**Fitness:** final objective value after N mutation steps on a batch
of test functions.

**Bar to clear:** beat a fixed Gaussian-scaled mutation on at least 3
benchmark functions.

**Why third:** if evolved optimizer beats the fixed one, it can be
plugged into the *other* GENREG runs to accelerate them. Genuinely
self-bootstrapping. Decouples selection-dynamics innovations from
model-architecture innovations.

### 4. Readout / output head
**Task:** classification on frozen features. Input: D-dim features
from a pretrained embedding + attention stack (or just random
features). Output: char distribution over V=32.

**Fitness:** cross-entropy / top-1 on frozen-feature classification.

**Bar to clear:** match or beat the current A_101 output head on the
same features.

**Why fourth:** lowest-value component to evolve separately (just a
linear + softmax); mostly a sanity-check harness that validates the
pipeline end-to-end.

## Build order (proposed)

1. **components/embedding** — build standalone, prove it beats SVD
   embedding on context-prediction, freeze, test downstream.
2. **components/attention** — build standalone on copy task, prove
   it scales to offset-k≥10, freeze.
3. **components/readout** — smoke test integration pipeline.
4. **components/optimizer** — parallel track, can be developed
   alongside once the first component is stable.
5. **Assembly** — embedding + attention + readout → new full LM.
   Compare heldout top-1 against A_101's 34%.

## What we keep from monolithic phase

- Per-neuron activation catalog (8 evolved functions) — this
  primitive worked everywhere.
- Energy homeostasis design — mandatory, validated.
- Tournament selection with maturation gate — known-good.
- SVD-of-bigram embedding as FALLBACK init when evolved embedding
  doesn't exist yet.
- Hash channel — stays as a runtime-fast fallback, not a lookup table.

## What we're leaving behind (for now)

- Monolithic A_* lineage: `genreg_lm_A_*.py` files are frozen as
  historical record. No more variants in that line.
- Cascade-only models (no recurrence): superseded by A_89 architecture.
- Lookup-based n-gram channels: not coming back.

## Success metric for this phase

At minimum, assembled component stack must match A_101's 34% heldout
top-1 within the same compute budget. The real win is if it **beats**
A_101 meaningfully — that would justify the component-first paradigm.
If it merely matches, we've at least built reusable modules.
