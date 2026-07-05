# Model card: `attn_copy_v1` — evolved selective retrieval (component 2)

Per GENREG_RULES §II and the component plan ("Attention: copy-with-offset").
Charter: no gradients, no gradient-like mechanics, constraints shape the
environment, information must flow.

## 1. Name
`attn_copy_v1` (runs env: `attn`)

## 2. Purpose
The capability the monolithic phase proved missing: cascades smooth context
but cannot *selectively retrieve*. Without retrieval there are no long-range
dependencies, and no real completion beyond a phrase. This component evolves
one Q·K attention head until it solves offset-k copy — the documented bar
before it may be composed into the LM.

## 3. Interface
- Episode: sequence of L symbol ids (L up to 64) containing one FLAG token;
  the episode's offset k is announced by a k-token at position 0
  (the environment states the question; the model must learn to read it).
- Output: distribution over the S=16 symbol alphabet — the symbol k positions
  before the FLAG.

## 4. Evolved parameters (per genome)
| tensor | shape | init |
|---|---|---|
| E (symbol+special embeddings) | (V≈23, D=32) | N(0, 0.1) |
| P (positional embeddings) | (64, D) | N(0, 0.1) — evolved, not sinusoid |
| Wq, Wk | (D, A=32) each | N(0, 1/√D) |
| Wv | (D, A) | N(0, 1/√D) |
| W_h | (A, H=32) + b_h + act_id(H) | N(0, 1/√A) — per-neuron 8-catalog FFN |
| W_out | (H, S) + b_out | N(0, 1/√H) |
| mut_rate, mut_scale | scalars | self-adaptive, floor 0.02 |

Query = (E+P)[flag_pos]·Wq; keys/values over the whole sequence; softmax(q·K/√A)
is the head's normalization (architecture, not training mechanics).

## 5. Fitness equation
Per generation, fresh episodes for every k ∈ {1, 2, 5, 10, 20} (B per k):
`fitness = mean over k of ( mean log softmax(logits)[target] within k )`
— equal weight per offset, so solving only the easy offsets cannot win
(multiplicative-rule intent: a degenerate k=1 specialist scores ~log(1/16)
on the others and sinks). Soft everywhere; argmax only for reporting.

## 6. Energy equation
Same homeostat as lm_char_v1: `e ← clamp(e·0.90 + 8.0·(fit_ema − median), 0, 1.5)`,
floor 0.20, spawn 1.0, EMA-smoothed fitness (α=0.75), starved target 3–15%.

## 7–8. Selection & mutation
Tournament k=3 among mature survivors (maturation gate), energy culls
independently; self-adaptive mut_rate [0.005, 0.2] / mut_scale floor 0.02,
per-tensor noise, act-id reassignment at mut_rate/4, anneal at 80%.

## 9. Hyperparameters
POP 400 · B 12 episodes/k (60/gen) · L sampled 24–64 · 3000 gens/sweep ·
LOG_EVERY 25.

## 10. Success criteria
- **Local bar (from the component plan):** top-1 ≥ 95% on held-out episodes
  for EVERY k ∈ {1,2,5,10,20} at L up to 64.
- Downstream bar: composed with the frozen lm substrate, held-out top-1 must
  exceed the substrate alone (assembly phase).

## 11. Failure modes to watch
- Solves small k only (per-k accuracy table logged — the equal-weight
  fitness is the counter-pressure).
- Position-memorization instead of flag-relative retrieval (counter: flag
  position sampled uniformly; L varies).
- Mode collapse / starved out of band — same gauges as lm_char_v1.

## 12. Baselines
- Chance = 1/16 = 6.25%.
- "Most frequent symbol" ≈ chance (symbols uniform).
- Positional heuristic (predict symbol at fixed position) — computed and
  logged; the model must clear it decisively.

## 13. Artifacts
`runs/attn/<id>/…` (runstore layout, full-state checkpoint) +
findings appended to `documentation/LM_STAGE1_FINDINGS.md`.
