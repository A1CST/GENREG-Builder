# Changelog — LM (autoregressive path)

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.
Seeded 2026-07-05 from the main changelog (keyword split, best effort).

---

- **[2026-07-06] (Claude)** — **Word-level recurrent LM** added to genreg_lm (push toward
  sentence/grammar structure). The synthesis: real word tokens (Tree LM tokenizer) + the
  recurrent sequential substrate + prev-token (genreg_lm) + the blended rollout-survival
  landscape at WORD horizons — so a genome must keep producing grammatically-plausible next
  words to survive its own generation (grammar = what long-horizon rollout rewards). Engine
  changes: token_mode "char"|"word"; per-run vocab (V dynamic); <unk> never a scored target or
  emitted (mask in evaluate/rollout/generate); word bigram/trigram baselines (dict-based
  trigram — a dense table is TB-scale at word vocab); word-mode generation detokenizes via the
  persisted vocab. Two rules-endorsed fixes made word-vocab evolvable (naive V=2048 stalled at
  0% — the giant W_out genome diffuses selection): **weight-tied readout** (collapse the ~V·H
  W_out into a tiny H→D projection scored against the shared embedding table) and **bigram-SVD
  embedding seed** (§VI's endorsed init — words in similar contexts start near each other).
  Smoke: 0.9%→8.4% held-out in 200 gens. Bars: word-bigram 17.6%, word-trigram 22.7% (the
  grammar reference). Chain running (2 substrate sweeps + blended rollout R=6 words).

- **[2026-07-05] (Claude)** — **Encoder separation: final verdict = parity, kept as
  optionality.** The frozen-encoder composed model under the blended rollout landscape ties
  the monolith on every metric (open 31.45% vs 31.3 · closed R=8 27.32% vs 27.2 · gap 0.21 vs
  0.17 nats) — notably reaching parity with the encoder FROZEN (only the readout evolved).
  Per the pre-committed decision rule: separation is a validated non-regressing component —
  same performance, measurably richer state (h2/h4 future decodable), cleaner modularity
  (reusable frozen encoder) — not a breakthrough. enc_char_v1 stays available; the monolith
  remains co-champion. Full 3-round trail in LM_STAGE1_FINDINGS.md.

- **[2026-07-05] (Claude)** — Encoder component rounds 1–2 + the decisive rollout test
  (running). Equal-weight horizons: h1 fell to 29.7%, composed 30.71% — under the bar (state
  budget robbed next-char sharpness). **Weighted horizons (0.7/0.2/0.1): h1 recovered to
  31.49%, composed 31.78% — parity with the monolith within eval noise (bar 31.9% not
  passed), with future-decodability (h2/h4) the monolith never had.** The value hypothesis is
  now testable exactly where richer state should pay: the blended rollout-survival landscape —
  frozen-encoder composed model vs monolith benchmarks (open 31.3 / closed 27.2 / gap 0.17
  nats). `horizon_weights` config added to genreg_enc.py.

- **[2026-07-05] (Claude)** — **Encoder separated into its own model: `enc_char_v1`**
  (`genreg_train/genreg_enc.py`, card `documentation/LM_ENCODER_COMPONENT.md`) — per user +
  §X component-first. Fitness = evolved-head decodability at horizons {1,2,4} from the hidden
  state (equal weight — h=1 specialists sink), breeding the STATE rather than the prediction;
  heads are scaffolding, the frozen deliverable is (E, W_in, b_h, act). Skip-gram baselines
  measured first (skip2 22.5%, skip4 20.1%). genreg_lm gained composed mode
  (`encoder_ckpt` → tensors copied + FROZEN; mutation excludes encoder incl. act ids — §X
  freeze-and-compose, never retrained). Warm-start smoke: h1 preserved at 31.3%, h4 at its
  bar in 30 gens. Encoder sweep → composition pipeline running; composed bar > 31.9%.

- **[2026-07-05] (Claude)** — **Stage 4 VALIDATED — blended rollout-survival landscape.** Pure
  rollout fitness bred hedging (open-loop 31.9%→23.1%; new §XI reward-hack: flatten toward
  marginals to score safely on drifted context). Blended fitness (teacher + own-output
  segments scored together, the DiffEvo unrolled lesson): open-loop HELD at 31.3% while
  own-output top-1 at R=8 went 15.0%→27.2% — **exposure gap 0.72→0.17 nats**, the quantified
  cause of generation soup cut 4×. Samples now contain real words at t=0.5. evaluate_rollout
  blended scoring in genreg_lm.py; complete campaign trail + ranked open levers in
  documentation/LM_STAGE1_FINDINGS.md.

- **[2026-07-05] (Claude)** — LM campaign, stages 3–4: **assembly honest-negatives + rollout
  landscape launched.** Copy-attention channel (α-gated, zero-init) stayed pinned at α≈0 through
  TWO variants — v1 retrieved the matched char (control beat it 30.42% vs 29.78%: complexity
  diluted selection, as documented); v2 fixed to induction values (retrieve the successor,
  target-leak excluded) still unopened at 31.91% held-out (+4.61 over bigram — pure substrate
  ratcheting). Verdict: 64-char windows don't pay attention rent; environment lever (longer/
  repeat-rich windows) backlogged. **Exposure gap measured: champion 31.4% teacher-forced vs
  15.0% consuming its own outputs (0.72 nats)** — the quantified train/inference mismatch
  behind generation soup. Stage 4 running: rollout-survival fitness (score TRUE continuation
  while eating own samples; anchored per §IV.4), curriculum R=2→4→8. evaluate_rollout +
  rollout summary fields added to genreg_lm.py; full trail in LM_STAGE1_FINDINGS.md.

- **[2026-07-05] (Claude)** — **attn_copy_v1 PASSED its bar at 100%** (bar: ≥95% held-out on
  every offset k∈{1,2,5,10,20}, L≤64). Verified on 2,500 fresh-seed episodes: 100% on all
  five offsets, soft −6e-05. The recipe that made retrieval evolvable: multiplicative query
  (per-k conditional maps — additive provably could not switch offsets), weight-tied readout +
  identity values (credit reaches attention placement directly), zero-init relative-position
  bias (transparent no-op; evolution grows one bump per offset), 11-rung mastery-gated ladder
  (length grows before offsets unlock). Full lesson ledger in
  documentation/LM_STAGE1_FINDINGS.md. Both stage-1 components (substrate +0.73 pts over
  bigram; attention 100%) now cleared — assembly next.
- **[2026-07-05] (Claude)** — **lm_char_v1 PASSED its stage-1 bar: held-out 28.03% top-1 vs the
  27.30% char-bigram ceiling (+0.73 pts), majority floor +7.8pp, no train→held-out drop** —
  pure constraint-driven evolution per the model card (soft multiplicative fitness, energy
  homeostasis in-band all run, tournament+maturation, recurrence+prev-char, per-neuron
  activations; EMA-smoothed selection worth +1.6pp exactly as §IV.7 documents). Three resumed
  sweeps, ~24 GPU-min. attn_copy_v1: three landscape lessons logged (query couldn't SEE the
  offset; padded episodes nullified the length curriculum — masking fixed it, k=1 went
  12.5%→86% in 150 gens; weight-tied readout + identity values collapsed two coupled miracles
  into one basin) + v4 multiplicative query (additive can't switch offsets — the §VI lesson)
  running the 11-rung ladder. Findings: documentation/LM_STAGE1_FINDINGS.md.

- **[2026-07-05] (Claude)** — **New project: `lm_char_v1`** (`genreg_train/genreg_lm.py`) — stage 1
  of the true-autoregressive path, built per the pre-drafted model card
  `documentation/LM_STAGE1_SUBSTRATE.md` and GENREG_RULES: recurrent char substrate (emb_t +
  emb_t−1 + tanh(h_prev) → per-neuron 8-catalog activations → logits), soft multiplicative
  fitness (mean log-prob, no argmax anywhere), mandatory energy homeostasis (starved landed
  6–16%/gen in the target band), tournament selection + maturation gate, self-adaptive
  mutation with 0.02 floor + late anneal, fresh windows each generation, full-state
  checkpoints, runs/lm/ persistence (dashboard tab "lm"). Torch inference-mode only — no
  autograd, no pretrained weights. Corpus §VII bars measured first: majority 20.3%, char
  bigram 27.3%, trigram 39.2%. Sweep 1 (3000 gens, pop 400) running.
