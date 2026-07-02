# Audit of the scaling/cognition findings — holes in the logic

Self-critical pass over FINDINGS.md (A–K). Goal: find where the evidence is weaker than the
claim, where confounds lurk, and — most importantly under "no gradients ever" — where a result
secretly depends on a **designed reward-gradient** rather than a world-consequence.

Severity: ★★★ load-bearing / could overturn a headline claim · ★★ real caveat · ★ minor.

---

## H1 ★★★ Designed reward-gradients masquerading as "graded survival"

The recurring "reachability lever" — *grade survival so partial progress pays* — is, in three
headline results, a reward I **designed**, not a world-consequence:

- **F (syntax):** survival = +1 per correctly-decoded role. All-or-nothing survival gave chance
  (0.01); order only emerged with the per-role grading. So the syntax result is **conditional on a
  designed grading**.
- **G (binding):** I explicitly *switched* the reward from per-component to per-ENTITY to make
  binding appear. Changing the reward until the result shows up is the definition of shaping.
- **J (spatial sequence):** the index-query task (no designed grading) hit a wall; the result only
  appeared once I rewarded per-position reproduction. Again conditional on designed grading.

This does **not** automatically invalidate them — partial communication plausibly yields partial
survival in a real world — but the claim "the world demanded it" is **unproven** while the grading
is mine. TEST: do these emerge under a *world-consequence* fitness (an embodied listener that eats
only when it acts correctly), and are they robust across reward shapes, or fragile to the exact one
I picked? If only the hand-picked grading works → the result is shaped, not emergent.

Clean by contrast (naturally graded by the world, no designed slope):
- **I (invented memory):** fitness = count of correct recalls; binary per trial.
- **Foraging:** fitness = food eaten. These are the paradigm-honest anchors.

## H2 ★★★ The reachability THEORY contradicts "no gradients ever"

FINDINGS repeatedly elevates "graded survival is the universal lever." That *is* reward-shaping as
a stated principle. The theory must be split: **NATURAL grading** (food count, recall count, lifespan
— a world-consequence) is legitimate; **DESIGNED grading** (a slope I add toward the target) is not.
Every result must be reclassified on this line, and the "lever" reworded accordingly.

## H3 ★★ Statistics: 2–3 seeds, noisy, some non-monotonic

- Scaling sweeps bounce: invented-memory delay = 0.72, 0.56, 0.76, 0.48, 0.56, 0.57 (claimed
  "robust"); spatial length = 0.78, 0.96, 0.73, 0.58, 0.35, 0.53 (non-monotonic). With 2 seeds these
  could be largely noise. TEST: 6–8 seeds, report mean ± std; check which contrasts survive.
- Headline contrasts (B sex 0.35 vs 0.75; F bag 0.14 vs seq 0.24; K) all need CIs.

## H4 ★★ "Delay-robust / does not forget" (K) may be overclaimed

The data trends DOWN (0.72 → ~0.55) and is noisy. "External storage genuinely does not forget" is a
strong claim from 2 seeds with a visible downward drift. TEST: more seeds, longer delays (L=16,24);
is there real decay (reachability of the read policy) even though the slot itself persists?

## H5 ★★ The bag control (F/G) may be undertrained, not at its ceiling

Bag full-event 0.14 vs sequential 0.24 — both low, small gap. The bag's *theoretical* ceiling
(convey action + entity-set, then guess agent/target = ×0.5) is ~0.4–0.5, not 0.14. If the bag is
just undertrained, the "sequential beats bag" contrast is weak. TEST: train the bag harder / verify
it approaches its analytic ceiling; recompute the gap with CIs.

## H6 ★★ Compositionality (D) may be near-tautological

Held-out generalisation (0.88–1.00) appears only WITH the anchor, which grounds each word
INDEPENDENTLY. Independent per-word grounding *is* compositional structure, so generalising to new
combinations is almost guaranteed by construction — not an emergent surprise. Honest restatement:
grounding *enables* compositional reuse; it is not evidence that compositionality *evolved*.

## H7 ★ Acquisition (C) confound + noise

Dose-response is noisy (the 30% dip) and 2–4 seeds. Also: residents get energy from BOTH native and
resident exchanges; the "acquired from exposure" share is not isolated. TEST: measure english_usage
gain attributable specifically to native-pairing success.

## H8 ★ Invented "cipher" (I) — is it real structure or argmax noise?

The cute "stores cue 0 as 4" cipher could be incidental tie-breaking. The ablation (0.99→0.16) is
the real evidence; the cipher narrative is decorative and should be stated cautiously.

## H9 ★ Content-addressing "expressivity wall" reasoning

The claim "a linear layer cannot compute move-toward-cell-indexed-by-key" is plausible but asserted,
not proven; and the v2 (hidden layer) failure is then attributed to reachability — two different
walls invoked without cleanly separating them. Honest: we showed it FAILS three ways; we did not
prove WHICH wall. (And note: V=20 "opening a gradient" was itself a reward-shaping move — H1.)

---

## What the audit demands (run order)
1. Re-test F/G/J under a WORLD-CONSEQUENCE fitness (embodied), and across reward shapes → resolve H1/H2.
2. High-seed re-runs of every headline contrast with mean±std → H3, H4, H5.
3. Bag analytic ceiling check → H5.
4. Reclassify all results NATURAL vs DESIGNED grading; reword the theory → H2.
5. Foraging (pure food fitness) as the clean anchor that memory emerges without any designed slope.
