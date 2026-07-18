# Changelog — VISION_DEMO

Per-project log for the `/vision_demo` showcase. New entries go at the top, and
also in the master `CHANGELOG.md`.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-18] (Claude)** — **Transfer-efficiency A/B + observation-first rewrite
  (reviewer feedback).** From-scratch control (`vision_continue.run(warm=False)`) +
  efficiency block: to reach 98.67%, **grow 328 new genomes vs scratch 527** (~38%
  fewer); scratch edges higher (0.9885 @ 639) so reuse buys efficiency, not ceiling —
  table shows both. New efficiency table + grow-vs-scratch curve. Copy: "staple" →
  "composition strategies (Merge / Grow)"; cross-modal transfer promoted to an
  observation-first callout; added why-union-doesn't-interfere + why-no-forgetting
  explanations; new "What this suggests" (composability / closure-under-composition)
  section. **Flask restart** pending for routes.

- **[2026-07-18] (Claude)** — **Animated inference panels + downloadable checkpoints
  & inference script.** `mm/vision_samples.py` renders fresh samples, runs each frozen
  checkpoint, exports predictions as PNG data-URIs (shape 1.0 / letter 0.917 / union
  1.0 / continued 0.981) -> four **animated panels** cycling live samples with ✓/✗.
  New inference CLI `mm/vision_infer.py` (`--model shape|letter|union|continued`,
  renders random samples, prints predictions + accuracy, `--save` PNGs) and download
  routes `/api/vision_demo/download/<name|bundle>` (bundle zip = 4 checkpoints +
  script + README). `/api/vision_demo/samples` endpoint; `vision_samples` wired into
  the orchestrator. Shape panel refits its 10-class head on the centered-crop basis
  (saved head was tracker-basis-tied). **Flask restart required** for new routes.

- **[2026-07-18] (Claude)** — **NEW project /vision_demo — two gradient-free staples
  on vision-grounded models.** Showcases (1) **UNION**: fuse the frozen SHAPE
  recognizer (634 genomes, 10 classes) + frozen LETTER recognizer (597 genomes, 26
  classes) into one 36-class head — shape-bank 0.9444 / letter-bank 0.9795 / **FUSED
  0.9946** (`mm/mm_merge.py`, reused); and (2) **CONTINUED TRAINING**: warm-start
  from the 634 shape genomes and evolve NEW genomes on the 36-class task until the
  shapes model reads letters too, with NO separate letter model — **letters 0.9231
  → 0.9816 (+5.85%), overall 0.9444 → 0.9867**, 400 new genomes, one ridge head
  (37,260 params), 85s on the 4080 (`mm/vision_continue.py`, the `anim/dot_shape.py`
  evolve loop seeded with the frozen shape base). Orchestrator `mm/vision_demo.py`
  runs both and writes `radial_data/vision_demo.json`; run recorded to
  `runs/vision_demo/` (five-file set) with completion alert. Page: route
  `/vision_demo` + `/api/vision_demo/data` (no-store), `templates/vision_demo.html`
  + `static/vision_demo.js` (hand-rolled inline SVG — union grouped bars, before→after
  bars, the letters-climb curve, union-vs-continued explainer). Nav entry in the
  Vision group; changelog-modal mapping added. Honest nuance surfaced on the page:
  shape features already transfer to letters cross-modally (~0.92), so continued
  training closes the last gap rather than starting from chance. **Flask restart
  required** to serve the new routes.
