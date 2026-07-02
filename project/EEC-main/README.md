# EEC — Existence-Environment Constraints

Gradient-free neuroevolution where **capabilities are specified as constraints, not optimised toward.**
You do not train a model to approximate an answer; you impose **laws of existence** on a world until
the behaviour you want is the cheapest way for an organism to survive in it. The model is the
*intersection of its constraints*, not the sum of its weights.

> **No gradients, ever** — no backprop, no designed reward-gradient. Fitness is a *world-consequence*
> (food eaten, partner survives, offspring made), never a score pointing at a target. We **read the
> internal state, not the token output**, and we **never grade by accuracy**.

See [`docs/PARADIGM.md`](docs/PARADIGM.md) for the philosophy and [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md)
for the catalogue of laws and what emerged under each.

## The core idea (PO — the real metric)

The native unit of an evolutionary model is **constraints**, not parameters.

> **PO** = how many *laws of existence* (axes) it takes to collapse the infinite space of possible
> organisms down to the behaviour we need.

Three laws (energy, time, perception) yield one class of survivor; add memory-rent and the viable
space shrinks again. The minimum constraint set that makes the target behaviour *inevitable* is the
model's true complexity. An organism's full identity is **(PO, fitness)** — how many axes produced it,
and how well it survives them. The geometry is a **cone**, not a cube: each constraint is a ring
slicing infinity smaller; the tip is the single surviving behaviour.

## Repository layout

| path | what |
|------|------|
| `engine/` | the shared recurrent substrate, evolution loop, corpus, run matrix |
| `experiments/` | one folder per law / capability (see below) |
| `docs/` | `PARADIGM.md`, `CONSTRAINTS.md` (catalogue + meta-principles), report generators |
| `playground/` | PyQt6 real-time GUI: board + PO cone + comm chart |

Every script bootstraps the engine via a path insert; run from the repo root, e.g.
`python3 experiments/<name>/<script>.py`.

## Laws of existence (the catalogue)

Each constraint taxes one channel and, where the world makes the alternative *pay* and *reachable*,
grows one capability. Established axes: **energy** (survival), **time/Occam** (parsimony),
**memory-rent**, **entropy/decay** (active maintenance), **occlusion/noise** (persistence),
**reproduction-cost** (selection), **scarcity** (niches), **mortality** (survival, alternative to
energy), **perception-cost** (attention), **non-stationarity** (plasticity). Full results, verdicts,
and the six-seed ablation are in [`docs/CONSTRAINTS.md`](docs/CONSTRAINTS.md).

**Meta-principles** (laws *about* laws, they predict before we run):
- **P1 — the world must pay.** A capability emerges only where the world rewards it *and* mutation can
  reach it.
- **P2 — degradation budget / Goldilocks.** Constraints on the same axis share a coping budget; there
  is a sweet spot between too little pressure and an unlearnable world.
- **P5 — the search-wall.** State-conditional / compositional behaviour is unreachable by pressure
  alone; it needs a *channel* (frozen scaffold + evolved mixing), not more selection.
- **P7 — axes, not constraints.** Cover each capability axis once with the instantiation the world
  favours; stacking the same axis is wasted PO.

## Highlighted results

- **Constraint sweep & ablation** (`experiments/ablation`, `entropy_memory`, `scarcity`, …): entropy
  drives recurrent gain, memory doubles in long-range worlds, scarcity forms niches, communication
  yields a protocol — each verified ON/OFF across six seeds.
- **Emergent multi-agent communication** (`experiments/communication`): a c→sig→c protocol under
  coupled survival, gradient-free — a capability the single-agent substrate cannot produce.
- **English-grounded communication & scaling** (`experiments/english_comm`,
  [`ENGLISH_CONSTRAINTS.md`](experiments/english_comm/ENGLISH_CONSTRAINTS.md)): evolved agents acquire
  **real English** from native speakers and hold conversations (two-way comprehension 0.30 → **0.83**
  via a shared-lexicon "understanding-before-expression" organism). The scaling investigation found
  that **decomposability**, not weight-sharing, is the gradient-free scaling axis, that flat-lexicon
  fluency costs ~K² compute, and that the fix is **composition**: a compositional organism comprehends
  **held-out meanings it never trained on** (zero-shot 0.73–0.96) — vocabulary for free. The scaling
  fix and the grammar fix turn out to be the same thing.
- **Utterance length & breath** (`experiments/english_comm`, `breath_*.py`): a learned **stop**
  calibrates reply length to the meaning (the speaker knows its own content); **breath** is not the
  length controller but the *prosody/urgency* layer — speed-weighted breath × time produces
  urgency→clipped speech, and breath-group chunking of multi-word replies.
- **From parroting to conversation** (`experiments/english_comm`, `conversation*.py`, `llm_*.py`):
  the move from transmission (echo the meaning) to **communication** — hear X, produce the appropriate
  *different* reply, where copying is fatal (parrot conversation-length 0.00 vs 0.93 accuracy). It
  generalises (zero-shot replies to unseen prompts), comprehends real input itself via embeddings
  (held-out paraphrase comprehension **0.85** with a proper embedder), needs **memory** for
  context-dependent replies (0.96 vs 0.34), and a **back-channel** (feedback) emerges between two
  organisms (1.0 vs 0.5 blind). With a local LLM as the conversational *environment*, the organism
  holds a sustained, varied, LLM-judged conversation (coherence 0.75) — perceiving, comprehending,
  remembering, and taking initiative, with no designed score.

## Running it

Requires Python 3.11+, `numpy`, `matplotlib` (and `PyQt6` for the playground). No GPU; everything is
CPU neuroevolution. Examples:

```bash
python3 experiments/ablation/ablation.py                      # 6-seed ON/OFF constraint ablation
python3 experiments/english_comm/exposure_teaching.py         # urgency breaks the pidgin (P2 Goldilocks)
python3 experiments/english_comm/compositional_scaling.py     # composition = vocabulary for free
python3 playground/...                                         # real-time GUI (PyQt6)
```

## Stance

This is **not** gradient training in an evolutionary costume. If a result depends on a designed
reward-slope, it is disavowed and re-derived from world-consequence fitness (see the english_comm
audit). Findings are reported with their **regime and seed-count attached**, never as universal, and
negative results are kept honestly alongside the wins.
