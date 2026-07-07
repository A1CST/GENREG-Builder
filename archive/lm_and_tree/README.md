# Archived: Tree LM + LM campaign (2026-07-06)

Archived on the pivot away from n-gram-dependent and gradient-era approaches.
These are kept for reference only — NOT imported by the live app.

- `tree_lm.py`, `tree_service.py` — the Tree-of-Models (byte/word routing tree).
- `genreg_lm.py` — recurrent char/word substrate (best table-free: 34.6% top-1).
- `genreg_attn.py`, `genreg_enc.py` — attention + encoder components.
- `genreg_trustmix.py` — n-gram trust-mix (55% top-1, readable — but a 1990s
  n-gram lookup with an evolved backoff gate; NOT an evolved model).
- `genreg_distill.py` — distilling the n-gram teacher into a table-free model.
  VERDICT: recovered only top-1 24.7% (top-5 58%) — you cannot gradient-free-
  train away the tables. Compressing corpus statistics into weights is what
  gradients are FOR; evolution can't. See documentation/LM_STAGE1_FINDINGS.md.
- `genreg_rerank.py`, `pure_engine.py`, `lm_sample.py` — supporting experiments.

**Why archived:** these leaned on mechanisms borrowed from the gradient era
(n-gram statistics, attention, distillation). The new direction is an entirely
evolution-native language model — no n-grams, no attention, no optimizers.
See documentation/EVOLANG_PIVOT.md.
