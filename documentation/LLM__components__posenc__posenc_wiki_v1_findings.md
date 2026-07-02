# posenc_wiki_v1 findings

## Final state (gen 500)
- fit = 0.698
- global_gain = 0.814
- dim_gain range: [0.001, 0.373], mean 0.200

## Metrics on final checkpoint
| signal | value | notes |
|---|---|---|
| preservation cos(bare, out) | 0.989 | semantics fully intact |
| magnitude ratio sig/bare | 0.150 | target sweet spot (5-15%) |
| position recovery (32 classes) | 1.000 (bare 0.031) | +97pp uplift |
| relative position (65 classes) | 0.076 (chance 0.015) | +6pp uplift, weak |
| smoothness cos(P[t], P[t+1]) | 0.97 | sinusoidal structure preserved |

## Comparison: evolved vs raw sinusoidal (same magnitude target, no evolution)

| signal | sinusoidal | evolved | delta |
|---|---|---|---|
| preservation | 0.989 | 0.989 | ±0 |
| posrec | 1.000 | 1.000 | ±0 |
| relpos | 0.060 | 0.076 | +0.016 |

**Evolution reached parity with raw sinusoidal, not superiority.** Sinusoidal
encoding is already near-optimal for isolated-stage position encoding;
GENREG's real contribution was tuning magnitude and per-dim selection
to hit the preservation/utility tradeoff. Unlike the embedding stage
(where PPMI-SVD substrate + evolved skip-residual produced
qualitatively better neighbors than raw SVD), position encoding doesn't
have the same headroom.

## Why relpos stayed at ~7%
Ridge regression on 65-way classification with position-signal differences
is a hard test. The sinusoidal substrate enables this in principle
(rotation by consistent angle per delta step), but:
- Per-dim evolved activations add non-linearity that partially scrambles
  the clean rotation structure (even with identity_plus init + low flip rate,
  some dims deviate from linear)
- Ridge regression argmax doesn't naturally handle the cyclic structure
  of sinusoidal encoding

For attention downstream, this matters less: W_Q and W_K have their
own evolved structure and can read position from the output's
consistent directional encoding more flexibly than a ridge probe.

## Architectural decisions that worked

1. **Sinusoidal init + small per-genome perturbation**
   Random init gave zero fitness gradient (all genomes produced near-zero
   signal, no differentiation). Sinusoidal init + noise gave immediate
   posrec=1.0 at gen 0 with preservation=0.99.

2. **Low activation-flip rate (0.001/gen/dim)**
   Default 1%/gen randomization over 500 gens flips ~99% of dims at least
   once, destroying the identity-init structure. 0.1%/gen keeps most dims
   near-identity throughout training.

3. **Multi-signal fitness with preservation gate (multiplicative)**
   - 0.40 magnitude (primary evolvable signal — tunable sweet spot)
   - 0.20 relpos uplift
   - 0.15 posrec uplift (saturates but anchors the signal)
   - 0.15 next-token uplift (noisy ceiling but kept honest)
   - 0.10 smoothness
   - × preservation gate (0.70 full kill → 0.90 full credit on cos)

4. **Additive structure (no interference with frozen embedding)**
   `out = bare_emb + per_dim_activated(P[pos]) × dim_gain × global_gain`
   Same skip-philosophy as embedding stage. Cannot destroy bare, only add.

## Architectural decisions that didn't work

1. **Cosine-based discriminability** (1 - |cos(out1, out2)| for same token)
   Position signal is 15% of bare norm, so cosine stays ~1.0 even with
   meaningful position info. Metric was insensitive in this regime.
   Replaced with position-magnitude and relpos probes.

2. **Next-token prediction uplift as primary signal**
   Bare-embedding bag-of-context probe achieves 85%+ top-1 on top-200
   next-token. Ceiling is too tight for reliable uplift signal; position
   info's actual contribution is swamped by ridge variance (~2pp std).

3. **Per-dim random activation init**
   Random activations over 768 dims destroy sinusoidal directional
   structure. With 1% flip rate per gen, evolution can't recover it
   before the rest of the fitness landscape converges on a basin.
   Fixed by initializing all to identity_plus (id=7).

## Artifact
`components/posenc/checkpoints_posenc_wiki/posenc_wiki_gen_00500_final.pkl`

Includes: P (512, 768), dim_gain (768,), global_gain, act_ids, act_p1..p4
plus config.
