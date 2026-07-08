# RESUME — GENREG semantic-relation genomes (2026-07-07, round 2 complete)

Quick-start context for picking this back up. See also `genomes.txt`
(full roadmap + all battery results + honest failure analysis) and `CHANGELOG.md`
(top entries) for the full writeup. This file is a superseded-and-updated version of
the round-1 RESUME — the "in flight at restart" jobs from round 1 have all finished.

## The thesis (why we're doing this)
Memory: `ga-abstraction-thesis`. **Don't evolve the embedding space — BUILD it from corpus
statistics; evolution learns ONE tiny relationship inside the pre-built space.** "The features
are the environment; evolution is the organism navigating it."

## Corpus (DONE, durable on disk)
`corpora/wikipedia/wiki_corpus.txt` (302MB, 106K articles, 51M words) +
`corpora/wikipedia/wiki_feats.npz` (30K-vocab, 128-dim distributional SVD: `vocab`,`feat`,`freq`).
NN sanity check strong: king→queen/prince, france→germany/spain, hot→cold (top).

## Battery round 2 — FINAL VERDICTS (this session)
Full logs: `corpora/wikipedia/build/{hypernym,synant,synant_unified,batch2,register}.log`.
Detailed reasoning for every cut/ship lives in `genomes.txt` under "Battery note".

- **Hypernym — VALIDATED.** Directional heldout-acc 0.86, probes 10/10 clean. Ships as-is
  (not wired to the live pipeline yet — that's a separate wiring decision).
- **Synonym/Antonym — UNIFIED, PARTIAL.** Training them as two separate "related vs
  unrelated" detectors FAILED the decisive probe test both ways (synonym genome ranked
  an unrelated control above real synonyms; antonym genome failed its own flagship
  hot/cold example). Reframed per the user's suggestion into ONE genome that only asks
  "given a pair already known to be related, same or opposite meaning?" — real
  improvement (hot/cold now correct, 11/14 probes), but a residual, specific failure on
  size-adjective synonym pairs (big/large, small/little) that a coordination-context
  contamination explains. Not shipped; logged as a real partial result.
- **Meronym — VALIDATED-weaker.** 9/10 probes, modest heldout-acc (0.35-0.41).
- **Sentiment (monolithic) — CUT.** Inverted probes (war > joy). Bad mining, not a
  marginal miss.
- **Polysemy — CUT.** Strong val_acc (0.88) was a red herring; probes failed because the
  NN-spread proxy doesn't survive Wikipedia's dominant-sense skew.
- **Register — CUT.** Weak, incoherent (val_acc 0.61).
- **Analogy — CUT AS DESIGNED, not a dead end.** Chance-level through 600+ gens. User's
  diagnosis: analogy is a LAYER-3 relation (relationship BETWEEN relationships) and can't
  be trained on raw layer-2 distributional offset vectors — it needs the layer-2 relation
  genomes' OUTPUTS as its input features. Correct future recipe logged in `genomes.txt`:
  score = agreement between `hypernym_genome(a1,b1)` and `hypernym_genome(a2,b2)`.

## Next proposed step (not yet built — pending go-ahead)
**Decompose sentiment** the same way the rest of the pipeline was decomposed: instead of
one "positive or negative?" genome (which failed), several tighter binary genomes with
cleaner corpus signals — **Good**, **Bad**, **Intensity**, **Emotion**. Each is a smaller,
more mineable question than the monolith. Don't start training without checking in first
(these runs take ~15-30 min each and there may be a smarter batching now that a second
compute node exists — see below).

## Also shipped this session: I2 secondary node + job dispatch
Unrelated to the genome work but shipped in this session (see `documentation/changelogs/
CHANGELOG_I2.md`): `i2_node.py` now supports `--role secondary --primary <url>` (compute-
only, no content plumbing) with signed job submit/status/log/cancel, reusing the existing
Ed25519 admin-key trust model. `run_job.py` dispatches from this machine. Job history
persists across restarts; a Jobs tab in the primary's Tk console shows/streams them.
Pushed live to the real primary (10.0.0.15, now v1.4.0). **The Dell PowerEdge secondary
still needs to be brought up by hand** (`python i2_node.py --role secondary --primary
http://10.0.0.15:8800 --port 8800` on that machine, which needs this repo + a copy of
`i2_admin_key.json`) — once that's done, training jobs can run in parallel across both
machines via `run_job.py --node <url> <script> --watch`.

## Open questions for the user
1. Wire hypernym / unified-synant-partial / meronym into the live generation pipeline
   (semantic coherence re-rank, contrast/variation), or keep them as a standalone
   knowledge layer for now? (Same question round 1 raised, still open.)
2. Build the sentiment decomposition (Good/Bad/Intensity/Emotion) next?
3. Bring the PowerEdge secondary online to parallelize that work once scoped?
