# The EvoLang pivot — away from n-gram tables, back to evolution

**Date:** 2026-07-06 · **Decision by:** the user, after the distillation verdict.

## What happened

The whole Language line — the LM campaign (`genreg_lm` / `genreg_attn` /
`genreg_enc` / `genreg_trustmix` / `genreg_distill`) and the Tree-of-Models
(`tree_service` / `tree_lm`) — kept converging on the **same underlying object: an
n-gram lookup table.** The "breakthrough" trust-mix run that hit 56% readable
English was, correctly diagnosed, a bigram/trigram count table with a neural
coat. N-gram models have existed since the 1990s. That is not what this lab is
for.

We then proved the boundary honestly (see `LM_STAGE1_FINDINGS.md` and the
2026-07-06 CHANGELOG entry): you **cannot gradient-free-train away the tables.**
Distilling the table-teacher into a table-free evolved net recovered the
distribution *shape* (top-5 58%) but not the *argmax* (top-1 24.7%, generation
gibberish). Compressing corpus statistics into weights at per-context precision
is directed high-dimensional optimization — that is the job gradients do.
Undirected mutation can't reach that precision, so the table-free + gradient-free
model sits exactly where **both** mechanisms that make a good LM are excluded.

## The decision

Stop trying to out-compete n-gram statistics. That framing was the trap: it made
"a better next-token predictor" the goal, and the only things that win at that
game are tables (1990s) or gradients (rule #1 forbids them).

We are **not** building an attention mechanism. We are **not** building an
optimizer. We are **not** distilling a table. We are building an *entirely new
type of language model* — an evolved organism whose only tool is selection on a
fitness landscape, whose behavior is meta (the genome shapes its own offspring,
its own memory, its own activations), and which will therefore look, function,
and be judged differently from a gradient-trained LM.

The path forward is the lab's actual thesis (`GENREG_RULES.md`, the Intelligence
Engine): *design conditions where the only stable attractor is the solution.*
Evolution's edge is where gradients can't go — not raw next-token stats.

## What was archived (nothing deleted)

| Moved to | Contents |
|---|---|
| `archive/lm_and_tree/` | `genreg_lm.py`, `genreg_attn.py`, `genreg_enc.py`, `genreg_trustmix.py`, `genreg_distill.py`, `lm_sample.py`, `genreg_rerank.py`, `pure_engine.py`, `tree_service.py`, `tree_lm.py` (+ a README explaining why) |
| `runs/_archive/` | the `lm`, `attn`, `enc`, `encoder`, `tree`, `distill`, `pure` run directories |

The findings docs (`LM_STAGE1_*`, `LM_ENCODER_COMPONENT.md`,
`PURE_PER_LAYER_CONSTRAINTS.md`) are **kept** — they map the boundary and remain
the honest record of why this pivot was necessary.

## What replaced it — EvoLang (`/evolang`)

Deliberately minimal (the instruction was "a very basic GA and a small corpus of
text — do nothing else," because we had over-engineered our way off course):

- **`genreg_train/evolang.py`** — one small fixed English corpus, one tiny neural
  next-char predictor per genome (context → evolved embedding → tanh → V logits),
  evolved by **tournament + elitism + mandatory energy homeostasis** with
  self-adaptive mutation. Fitness is the **soft** mean log-prob of the true next
  character — never argmax, never a count table. No gradient touches a weight.
- **`templates/evolang.html` + `static/evolang.js`** — a page with the shared
  terminal dock / Agent panel / Run-Config panel: config sidebar, live fitness
  chart, the emerging sample (re-sampled every 25 generations), a generate box,
  and the full corpus on display.
- Runs land in `runs/evolang/<id>/` and show on the `/runs` dashboard.

This is a starting point, not a destination — the smallest honest evolution-native
language experiment, to build the *new* type from, instead of chasing the table.
