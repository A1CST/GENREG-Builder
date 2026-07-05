# Model card: `lm_char_v1` — recurrent char substrate (stage 1 of the autoregressive path)

Per GENREG_RULES §II — drafted before code. Operating charter (user, 2026-07-05):
no gradients (incl. no gradient-pretrained bases), no gradient-like bolt-ons,
constraints manipulate the environment, information must flow — learn by
experience.

## 1. Name
`lm_char_v1` (runs env: `lm`)

## 2. Purpose
The evolved substrate every later component (attention, rollout fitness) sits
on. Reproduces the documented A_89/A_98 structural wins — recurrence +
prev-char input — inside this repo, on the Gutenberg corpus, with mandatory
energy homeostasis and soft multiplicative fitness. Without it there is no
organism to apply landscape pressure to; stage 4's "survive generating your
own future" needs a substrate that already predicts above the bigram ceiling.

## 3. Interface
- Input: window of char ids `(SEQ_LEN,)`, vocab V≈40 (lowercase + space +
  basic punctuation; everything else folded to a rare bucket).
- Runtime state: hidden `h ∈ R^H` carried across the window (recurrent —
  information flows).
- Output per step: logits over V for the NEXT char.

## 4. Evolved parameters (per genome)
| tensor | shape | init | why |
|---|---|---|---|
| E (embeddings) | (V, D=24) | N(0, 0.1) | evolved lookup — no SVD/ridge init |
| W_in | (2D+H, H=64) | N(0, 1/√(2D+H)) | mixes [emb_t, emb_{t−1}, tanh(h_prev)] |
| b_h | (H,) | 0 | |
| act_id | (H,) ints | uniform over 8-catalog | per-neuron evolved activation — the signature primitive |
| W_out | (H, V) | N(0, 1/√H) | readout to logits |
| b_out | (V,) | 0 | |
| mut_rate | scalar | 0.05 | self-adaptive, bounds [0.005, 0.2] |
| mut_scale | scalar | 0.05 | self-adaptive, floor 0.02 |

Total ≈ V·D + (2D+H)·H + H + H·V + V ≈ 11.5k params. Deliberately small —
plateaus get landscape pressure, not parameters (§I.2).

Activation catalog (8): tanh, sigmoid, relu, sin, gaussian `exp(−x²)`,
identity, softsign, abs-clip.

## 5. Fitness equation
`fitness = mean over (batch windows × steps after warmup) of log softmax(logits)[target]`
— soft (mean log-prob, §IV.1), and multiplicative across positions by
construction (sum of logs). No argmax anywhere in selection. Warmup: first 8
steps of each window excluded (hidden state filling). Fresh window sample
each generation (non-stationarity — nothing can memorize a fixed batch).

## 6. Energy equation (homeostatic, not reward)
`energy ← clamp(energy · DECAY + GAIN · (fitness − median_fitness), 0, E_MAX)`
- DECAY 0.90, GAIN tuned so starved/gen lands in 3–15% (start 8.0 given
  fitness deltas ~0.01–0.1 nats), E_FLOOR 0.20 (cull below), E_MAX 1.5,
  spawn energy 1.0.
- Culling by energy is independent of tournament rank. `starved == 0` for
  many gens → raise GAIN; `starved > 50%` → lower it. Logged every gen.

## 7. Selection
- POP_SIZE 400 (RTX 4080, batched torch, inference-mode only — no autograd).
- Energy cull first, then tournament (k=3) among survivors fills the gap.
- **Maturation gate**: offspring must survive one full generation before
  reproducing (age ≥ 1).
- SURVIVAL isn't a fixed % — energy decides who dies; tournament decides who
  breeds.

## 8. Mutation
- Per-genome self-adaptive `mut_rate`/`mut_scale`, log-normal perturbation
  (×exp(0.2·N)) then clamped; **floor mut_scale 0.02** (V3→V4 CIFAR lesson).
- Per-tensor scaling of mutation noise; `act_id` entries reassigned with
  prob mut_rate/4.
- Anneal late: halve mut_scale after 80% of generations.

## 9. Hyperparameters
`SEQ_LEN 64 · BATCH 48 windows/gen · N_GENERATIONS 3000/sweep (§I.3: 2–8k,
never long runs) · LOG_EVERY 25 · ANNEAL_AFTER 0.8 · warmup 8`

## 10. Success criteria
- **Local bar**: held-out top-1 above the corpus char-BIGRAM ceiling
  (measured before training; A-series precedent: recurrence is worth +3pp
  over it).
- **Downstream bar**: substrate checkpoint loads + continues improving under
  stage-4 rollout fitness without collapse.
- Verification per §VII before any claim: majority baseline, random
  per-window 80/20 held-out (drop < 10% relative), inspect actual sampled
  generations.

## 11. Failure modes to watch (log pattern in parentheses)
- Energy decorative (`starved=0` for 500+ gens) or genocidal (`starved>50%`).
- Mode collapse (all genomes identical top-1; population fitness std → 0).
- Function-word/char soup — e.g. everything predicts space/'e'
  (top-1 ≈ majority baseline while soft_score climbs).
- Train↑ heldout-flat → train/inference mismatch.
- Mutation collapse (mut_scale pinned at floor across population).

## 12. Baselines to beat
1. Majority class (most common char).
2. Char bigram ceiling (count table on the same stream) — the real bar.
3. Char trigram ceiling — the stretch reference.
(No closed-form readout baseline is used as an INIT — baselines are bars,
not seeds.)

## 13. Artifacts
`runs/lm/<id>/{config.json, history.jsonl, summary.json, checkpoint.npz}`
(checkpoint = ALL state: tensors, act ids, energy, mut params, rng, gen) +
`documentation/LM_STAGE1_FINDINGS.md` after each sweep.
