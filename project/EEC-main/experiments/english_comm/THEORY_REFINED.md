# Refined theory — after the audit and re-tests

The session's working theory was: *capabilities emerge from worlds that demand them; the lever
that makes them reachable is graded survival.* The audit (AUDIT.md) + high-seed re-tests force
three corrections. The theory is stronger for them, and narrower.

## Correction 1 — "graded survival" must be split: NATURAL vs DESIGNED

"No gradients ever" applies to the reward too. A fitness that GRADES how close an organism is to a
target is reward-shaping — a designed gradient — even with a gradient-free optimiser. The lever is
legitimate ONLY when the grading is a **world-consequence**:

- LEGITIMATE (world-consequence): food eaten (foraging), number of correct recalls (invented
  memory), partial communication that partially helps a partner survive.
- ILLEGITIMATE (designed slope): a proximity/distance-to-target score; switching the reward until
  the result appears; "make guessing worthless to open a gradient." Several mid-session moves were
  this and are disavowed: the binding per-entity switch (G), the spatial index->sequence switch was
  borderline, and the killed proximity probe.

**Evidence this matters (reward-shape sweep, 5 seeds each):** in the relational world, order-marking
- does NOT emerge under ALL-OR-NOTHING survival (full=0.011 ~chance) or near-zero gradient ("any",
  0.062);
- DOES emerge under every "partial-understanding-partially-helps" shape: threshold2 (0.146),
  per_role (0.225), square (0.228), all order-load-bearing (scramble collapses them).

Reading: emergence requires partial success to carry partial survival value — but it is ROBUST to
the exact shape (a coarse "mostly understood" works as well as a fine per-role count). Robustness
across shapes is the signature that the **world structure** (relations) is doing the work, not a
hand-tuned slope. The all-or-nothing failure is the honest boundary: in a degenerate world where
only perfect comprehension keeps your partner alive, order is unreachable. Real worlds are not
all-or-nothing, so the result stands — but stated WITH this condition, not as "graded reward is a
free lever."

## Correction 2 — the capability hierarchy, graded by EVIDENCE QUALITY

| capability | emerges? | fitness type | evidence | strength |
|---|---|---|---|---|
| external memory (marks/world slot) | yes | NATURAL (recall count; food eaten) | ablation 0.67 vs 0.16 (n=8); foraging 4.01 vs 1.93 same-organism | **strong** |
| word order (syntax) | yes, if partial-value | borderline (per-role grading; but robust to shape) | scramble 0.22->0.055 (n=8); bag gap fair 0.37 vs 0.25 | **moderate** |
| spatial sequence layout | yes, small N | DESIGNED (per-position) | freeze ablation; W-sweep | moderate, reward-dependent |
| constituent binding | weak | DESIGNED (per-entity, switched-to) | bag gap 0.28 vs 0.19, low absolute | **weak / shaped** |
| content-addressing | N/A — REJECTED | — | not a biological operation | out of paradigm |

The clean anchors are external memory (natural fitness, ablation-proven, twice) and — to a lesser
degree — word order (robust to reward shape). Binding and spatial layout lean on designed grading
and should be reported as *suggestive, reward-dependent*, not established.

## Correction 3 — specific claims overturned or downgraded

- **"Sequential vastly beats bag" (F) was a TRAINING ARTIFACT.** At 600 gens the bag was
  undertrained (0.14). At fair 1500-gen plateau the bag reaches 0.25 (its analytic "convey
  action+entities, guess roles" ceiling) and sequential 0.37. The order advantage is real and
  statistically separated (~3 sigma) but MODEST (~0.12), not the dramatic gap first reported. The
  load-bearing scramble test (0.22->0.055) remains the strongest, clean evidence order is used.
- **"External memory never forgets / delay-robust" (K) — REFINED, mostly upheld.** 5-seed re-test
  L=2..24: recall drops once (L=2 0.85 -> L=4 0.63) then PLATEAUS dead flat to L=24 (0.61, 0.62,
  0.65, 0.61). The plateau over 20 extra delay steps with no decay IS the signature of external
  storage (a decaying internal memory would keep dropping). So "doesn't forget over time" holds for
  L>=4 -- but at ~0.6, not the noisy ~0.75 first reported, and with an initial-difficulty drop. The
  original 2-seed "0.48-0.76" was noise; the real shape is drop-then-flat.
- **Compositionality (D) is near-tautological.** Held-out generalisation appears only WITH the
  anchor, which grounds each word INDEPENDENTLY; independent per-word grounding IS compositional
  structure, so zero-shot reuse is largely guaranteed by construction. Honest claim: grounding
  ENABLES compositional reuse; it is not evidence compositionality EVOLVED under pressure.
- **Foraging relocating control is CONFOUNDED.** Relocating food reappears fresh and is therefore
  MORE abundant (5.33 > persistent 4.22), so that control changed abundance, not memory's
  usefulness. The valid memory evidence is the same-organism marks-ablation (4.01 vs 1.93) in the
  persistent world, where a local-vision feedforward organism can only use marks as memory.
- **"Invented cipher" (I)** is decorative; the ablation is the evidence. Stated cautiously.
- **Content-addressing — REJECTED as out-of-paradigm, not an open wall.** Organic evolution does
  not content-address: biological memory is associative / reconstructive (cue → pattern completion),
  never random-access key→value lookup. So the "fails three ways" result is the substrate correctly
  refusing a non-biological operation, not a capability we failed to reach. The earlier framing of
  this as the frontier to break was a category error; the probes (incl. the large-V reward-shaping
  attempt) are disavowed and the line is closed.

## The refined thesis

Capabilities emerge from worlds whose **survival physics** make them the cheapest way to live —
external memory most cleanly (it pays in food, ablation-proven). Word order emerges too, conditional
on the realistic property that partial communication partially helps, and robustly across reward
shapes. But the lever is NOT "add a gradient to the reward": that is the trap. Where we leaned on
designed grading (binding, spatial layout) or hand-tuned rewards (the disavowed probes), the results
are weaker and should be re-derived from world-consequence fitness. The honest frontier is
**re-grounding the reward-shaped results (binding, spatial layout) in natural survival** — not
content-addressing, which is rejected as out-of-paradigm (organisms don't do key→value lookup;
memory is associative/reconstructive).
