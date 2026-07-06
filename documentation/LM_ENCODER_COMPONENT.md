# Model card: `enc_char_v1` — the recurrent encoder as its own model

Per GENREG_RULES §II / §X (component-first) and the user's call (2026-07-05):
"separate the encoder so it's its own model." The Tree LM precedent measured
the encoder as the binding constraint; this gives the LM's encoder its own
landscape instead of a fitness signal diffused through the whole organism.

## 1. Name
`enc_char_v1` (runs env: `enc`)

## 2. Purpose
Breed the STATE, not the prediction. The substrate's encoder only ever felt
next-char pressure, so its hidden state carries exactly one step of future.
Here the encoder is scored on how much of the future (horizons 1, 2, 4) can
be decoded from its state by evolved heads — a state rich enough for h=4
carries phrase-level information the composed LM (and later the rollout
landscape) can spend. The heads are scaffolding: discarded at composition.

## 3. Interface
- Identical recurrence to lm_char_v1: `h = act([E[c_t], E[c_{t-1}], tanh(h_prev)] @ W_in + b_h)`
  — same shapes (D=24, H=64, V=38) so freeze-and-compose is a tensor copy.
- Frozen deliverable: `E, W_in, b_h, act` only.

## 4. Evolved parameters (per genome)
Encoder: E (V,D), W_in (2D+H,H), b_h (H), act_id (H) — as lm_char_v1.
Scaffold heads (one per horizon h ∈ {1,2,4}): W_out_h (H,V) + b_out_h (V).
Self-adaptive mut_rate/mut_scale as usual. ≈ 11.5k + 2·2.5k params.

## 5. Fitness equation
`fitness = (1/3) · Σ_{h∈{1,2,4}} mean log softmax(h_t @ W_out_h + b_out_h)[c_{t+h}]`
Equal weight per horizon (the attention lesson: a degenerate h=1 specialist
must sink). Soft everywhere; fresh windows every generation; warmup 8.

## 6–8. Energy / selection / mutation
Identical to lm_char_v1 (homeostat, EMA-smoothed selection, tournament k=3 +
maturation gate, self-adaptive mutation, floor 0.02, anneal at 80%).

## 9. Hyperparameters
POP 400 · batch 96 · SEQ 64 · 2–8k gens/sweep · warm-startable from any
lm_char_v1 checkpoint (bootstrap rule — E/W_in/b_h/act transfer; heads init
from the substrate's W_out for h=1, random for h=2,4).

## 10. Success criteria
- Local: horizon-1 decode ≥ the substrate's own top-1 (it must not LOSE
  next-char sharpness) AND horizon-2 decode above the skip-bigram baseline
  P(c_{t+2}|c_t) — proof the state carries beyond-next-char future.
- Downstream: composed (encoder FROZEN, readout evolved) held-out top-1
  > 31.9% — the current best substrate. Freeze-and-compose only; the frozen
  encoder is never re-trained mid-pipeline (§X).

## 11. Failure modes
- Horizon-1 regression (state spread too thin) — logged per-horizon.
- Heads doing the work instead of the state (audit: linear probe of frozen
  state must reproduce head accuracy — evolved probe, not ridge).
- Usual: mode collapse, starved out of band, mutation floor pin.

## 12. Baselines
Majority; char bigram (h=1, 27.30%); **skip-bigram** tables P(c_{t+h}|c_t)
for h=2 and h=4 (computed before training).

## 13. Artifacts
`runs/enc/<id>/…` + checkpoint (ALL state incl. heads); composition via
`genreg_lm.run(cfg={"encoder_ckpt": …})` which loads + freezes encoder
tensors; findings appended to LM_STAGE1_FINDINGS.md.
