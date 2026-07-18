# Changelog — LM (autoregressive path)

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.
Seeded 2026-07-05 from the main changelog (keyword split, best effort).

---

- **[2026-07-18] (Claude)** — **Wiki model live locally: pack val 0.2685
  (=pod), 13.7/1.84 tok/s plain/polish, demo trace generated, module-40
  export enriched. One restart delivers everything. Detail in main
  CHANGELOG.**

- **[2026-07-18] (Claude)** — **/lm_demo page: the real computation,
  animated from a recorded trace (bank -> decomposed head math -> genomes
  -> specialist votes -> sample). Route + nav + modal mapping; trace via
  lm/lm_demo_trace.py. FLASK RESTART for the route. Detail in main
  CHANGELOG.**

- **[2026-07-18] (Claude)** — **Local pack builder (32GB-safe, GPU-chunked
  gram) + inference bench; wiki polished samples + perf pending the chain.
  Rerunnable: build_pack_local.py then bench_infer.py. Module 40 already
  on main. Detail in main CHANGELOG.**

- **[2026-07-18] (Claude)** — **Prose foundation (module 40): wiki corpus
  15x, tables 9x (1.08GB), first prose-shaped samples. 0.2998/0.5032
  (+42% over own trigram; not comparable to dialogue 0.5601). Blind slice
  0.0146 needs a wiki probe; pack + specialist re-measure await the next
  pod. Run runs/lm/20260718-063008-lm-wiki-9ded75. Detail in main
  CHANGELOG.**

- **[2026-07-18] (Claude)** — **Grammar specialist (module 39): 0.6642 on
  pure order (anchor below chance); per-step union lifts hold 0.69->0.81
  free; three-specialist decode deployed. Run
  runs/lm/20260718-044436-lm-grammar-eb736f. Detail in main CHANGELOG.**

- **[2026-07-18] (Claude)** — **Fluency frontier (module 38): merit-pool
  decode deployed - judge -8.17 (+1.4 nats) at hold 0.69, ~5s polished.
  Frontier mapped across 4 rounds; moving it further is training-side.
  Run runs/lm/20260718-040030-lm-fluency-b6da1a. Detail in main
  CHANGELOG.**

- **[2026-07-18] (Claude)** — **LM scripts -> lm/ package (21 files, repo
  cleanup on the user's call); radial_lm.py stays root (shared). Pod
  mirrored. Commit 23ef447. Detail in main CHANGELOG.**

- **[2026-07-18] (Claude)** — **Fluency honesty pass: single-sample judge
  flat (-9.43 vs -9.20) - the crank bought accuracy + hold cost + POLISH
  fluency (-8.60), not raw sample fluency. Defaults retuned to the
  measured best point: lam=1.5, polish on (2.1-2.6s local). Detail in
  main CHANGELOG.**

- **[2026-07-18] (Claude)** — **Post-crank deploy (module 37): hold 16/16
  at -6.5pts (lam=2 default), polish temp0.9/top5 best-of-8, inference
  pack fixes the local hour-long build. Run runs/lm/20260718-023911-lm-postcrank-91f425.
  Detail in main CHANGELOG.**

- **[2026-07-18] (Claude)** — **CRANK LANDS: 0.2949 -> 0.5601 top-1
  (0.7753 top-5); blind slice 0.3123, has-target 0.7262 - the probe's
  prediction held. Run runs/lm/20260718-021722-lm-crank-e4438a. Post-crank sweeps running;
  module 37 on their landing. Detail in main CHANGELOG.**

- **[2026-07-18] (Claude)** — **Crank probe + retrain launch:** 40.1% of
  test blind, new quad/skip tables answer 62.4% of it (answerable 59.9% ->
  84.9%). Bank 17.3k -> 27.7k cols, checkpoint bank="skip5k", infer
  replays by flag + CPU fallback for the 4080. Retrain on pod. Detail in
  main CHANGELOG same date.

- **[2026-07-18] (Claude)** — **Coherence grid (module 36): temp 0.7/top-3
  new live default (judge -8.98, hold 0.875); best-of-8 polish checkbox
  (-8.418, 2.2x, ~8x latency). Run runs/lm/20260718-014439-lm-coherence-1ce075. Detail in main
  CHANGELOG.**

- **[2026-07-18] (Claude)** — **EVIDENCE FLOOR (module 35): 16/16 hold at
  lam=2; live default lam=1.5 (hold 0.875, sanity 0.185).** Steering
  restricted to words with >=3 occurrences in the topic TRAIN articles
  (2860/5000). Better hold at every lambda, ~half the coherence cost vs
  module 34. Run runs/lm/20260718-013250-lm-steer-ev-beeaa8. Detail in main CHANGELOG.

- **[2026-07-18] (Claude)** — **Live autocomplete holds topic:** steering
  (auto, confidence-gated at 0.30, evidence-floored, lam 1.0) wired into
  lm_word_infer.complete + route + page checkbox; graceful fallback to
  plain autocomplete on any steering failure. Same pending Flask restart.
  Detail in main CHANGELOG same date.

- **[2026-07-18] (Claude)** — **/lm LIVE module: status flicker fixed +
  build ETA.** Poll retries no longer re-stamp "completing…" (one stable
  building line with elapsed); lm_word_infer stamps stage fractions and
  serves elapsed/eta, shown as "~N min left". Template fix live on
  refresh; backend ETA rides the same pending Flask restart. Detail in
  main CHANGELOG same date.

- **[2026-07-18] (Claude)** — **V=5000 GENERATOR + steering (module 34):
  vocabulary bottleneck FIXED; bottleneck moved to tail noise.** Generator
  test 0.2949/0.5401 (trigram baseline 0.175) - V=2000 accuracy held on a
  2.5x harder question. Steering: hold 0.125 -> 0.75 (lam 2), sanity falls
  faster than V=2000. Chemistry - module 33's dead topic - now emits
  acid/oxygen/reaction/element and holds 2/2 at lam=1; the cost is noisy
  proper-name attractors from the 5000-word tail (aubrey/kathryn/freddy).
  Three pod OOM/numerics fixes landed in core (radial_evo2 scorer chunk +
  jitter fallback, radial_lm_word lazy attend substrate). V5K checkpoint
  deployed with V2000 backups kept. lm_word_v5k.py; export kid_steer5k.json;
  run runs/lm/20260718-010506-lm-v5k-0426e0. Full detail in main CHANGELOG
  same date.

- **[2026-07-18] (Claude)** — **TOPIC-STEERED GENERATION (module 33):
  topic-hold 0.125 -> 0.81 across lambda 0->2; FREE at lambda=0.5 (hold
  0.438 at next-word 0.325 vs 0.3225 baseline).** The module-32 persistence
  topic model steers the live word generator: linear topic head => fixed
  per-topic score per target word; the prompt's accumulated state picks the
  topic (16/16 correct); lambda*score added to logits before top-5 so
  topical words enter the pool. Judge = held-out-article corpus counts,
  independent of the steering model. Decode fixes: no bonus for emitted
  words + lambda-scaled repetition penalty (kills the wine-wine-wine
  degeneracy caught in the local smoke). Bottleneck measured: the V=2000
  dialogue-heavy target vocab (chemistry worst - no chemistry words to
  emit); next lever is a topical target vocabulary. `topic_steer.py` on the
  fresh pod; `kid_plang_model.json` saved (persistence_lang.py now persists
  the ACCUM model; re-run reproduced 0.6109 exactly); export
  `kid_steer.json`; run `runs/lm/20260718-001011-lm-steer-9579fc`; shadow
  copies in runpod_shadow/genreg-lm/. Same pending Flask restart covers
  module 33. Full detail in main CHANGELOG same date.

- **[2026-07-17] (Claude)** — **PERSISTENCE ON LANGUAGE (module 32): the
  operator transfers. Topic identity from a 12-word window: single word
  0.2247 -> raw-space accumulate 0.4871 -> feature-space accumulate 0.6109
  (chance 0.125) - the exact letters ordering, and evolution EARNS over every
  anchor (+13 over the strongest, mean-vec ridge 0.4815; +16 over the fat
  concat 0.4463).** The recurring signal is the TOPIC: every word of a window
  is one noisy view of it. `build_topic_stream.py` (8 wiki topics x 8
  articles via local zetifile, embed_rs vectors, article-disjoint test) +
  `persistence_lang.py` (SINGLE / VECMEAN / ACCUM, same detectors and
  budget). Position hurts (concat anchor < mean anchor) = the order-invariant
  regime persistence owns. Caveats: 8 well-separated hand-picked topics
  (measures the operator, not hard topic ID); 3 near-empty articles; law
  topic lighter (1,110 train windows). Export kid_plang.json, module 32,
  run `runs/lm/20260717-232734-lm-plang-6fe949`. Covered by the SAME pending
  Flask restart as modules 17-31 (kid_* whitelist); no new restart needed.
  Full detail in main CHANGELOG same date.

- **[2026-07-17] (Claude)** — **TEMPORAL COMPOSITION arc: next-word is dead,
  but temporal composition is real - as PERSISTENCE, not transitions. /lm
  modules 30-31.** Full day exploring whether temporal (stream) composition
  helps the language line, after the head-bug work.
  **Transition operator (order-sensitive, `a[t]` vs `b[t-o]`):** built
  `radial_temporal.py` (temporal space 0 -> static space 1, genome-only head)
  + the char/word stream builders. NEXT-WORD is a NULL 3 ways - word (temporal
  0.0744 vs static 0.0715, split-metric noise), char (0.0685 vs 0.0719, loss),
  and recency-pooling (no help). Next-word is recency/position/content-
  dominated and n-gram-shaped: the wrong task. STRUCTURE tasks: real-vs-shuffled
  full-scramble looked like a win (0.996) but was a position artifact (linear
  anchor 0.986); the LOCAL-shuffle control (adjacent swaps preserve position,
  break local transitions) collapsed the anchor and isolated the real signal -
  temporal beats static 0.7365 vs 0.6909 at 4 swaps (~14 SE) where position is
  useless. GRAMMAR (word real-vs-shuffled on embed_rs vectors): the cleanest
  dissociation - static froze ZERO genomes (stuck at chance), temporal the only
  earner, but weak (0.62) because embed_rs is SEMANTIC not syntactic.
  **Persistence operator (order-INVARIANT accumulation) - the payoff:**
  `persistence_test.py`. Persistence = a repeating signal, accumulated; a
  per-frame detector summed over the stream. Corrupted-viewpoint letters (K=8,
  noise+occlusion+jitter): single view 0.3835 -> pixel-space accumulate 0.5465
  -> FEATURE-space accumulate **0.93**. The pixel-vs-feature gap proves
  accumulation must be over detector RESPONSES not pixels (views aren't
  aligned; the detector fires wherever the letter lands). This is the temporal
  space's real primitive: coherent accumulator of recurring responses =
  persistence/attention; static = classifier of what survived. Saved as memory
  temporal-persistence-operator. /lm modules 30 (transition arc) + 31
  (persistence); exports kid_temporal.json / kid_persistence.json; whitelist
  broadened to kid_* (needs the pending Flask restart). All runs on a fresh pod
  (old one died mid-session; data regenerated, eye-anchor reproduced prior DNE
  0.0614 exactly). Next-word stays off the table; the temporal line continues
  as persistence.

- **[2026-07-16] (Claude)** — **/lm PAGE: modules 24-29 (the full head-bug
  arc) + the export route was BROKEN for the entire kid curriculum.
  FLASK RESTART NEEDED.**
  **Bug found while adding the modules:** `/api/lm/export/<name>` (app.py
  ~1617) whitelists `(lm_radial|lm_probe|embed_report)[A-Za-z0-9_]*\.json`,
  but every kid module points at `kid_stage*.json`. Verified live:
  `curl /api/lm/export/kid_stageC5.json -> HTTP 404 {"error":"not an lm
  export"}`. **Modules 17-23 (stages A, B, C, C2, C3, C4, C5) have been
  rendering "export not available" since the curriculum started** - the files
  were on disk the whole time, the route just refused the name. Fixed by
  adding `kid_stage` to the pattern (still fullmatch-anchored; `[A-Za-z0-9_]*`
  admits no dots or slashes, so no traversal).
  **Restart needed and why:** `lm_modules.json` is read from disk per request,
  so the server already lists all 29 modules - but the whitelist is code.
  Until the user restarts Flask, 24-29 will appear and their exports will 404
  exactly as 17-23 do now. Agents never restart the server.
  **Modules added** (append-only; no template edits, no existing module or
  entry altered, per the /lm contract):
  - **24 kid-stage-c6** - 100k scaling, 0.1538/0.3455, 2 genomes. Carries the
    correction that "the data lever is spent" was the head, and that scaling
    made the head bug WORSE.
  - **25 kid-stage-d1** - autoregression's first attempt: 0.1678 with ZERO
    genomes = a ridge head, 98% of 936,481 params. The head bug, stated.
  - **26 kid-stage-d2** - genome-only head: 0 -> 153 genomes, params -> 97,568,
    and still honestly losing to ridge (0.1403 vs 0.1678).
  - **27 kid-stage-c7** - the C line's headlines were the head; honest evolved
    cloze is 0.0889, not 0.1538.
  - **28 kid-stage-d-decomp** - DLEN beats ridge (first ever); the COMPOSE is
    a null (val down, test up = noise).
  - **29 kid-stage-d-ablate** - the ears suppress evolution; the eye is real
    (0.0753, 37x chance, pixels only); the dial is TABLE SHARE not question
    size; the size law retracted.
  Every export sets `fat_anchor_test` = that run's linear reference so the
  existing generic renderer draws it as the "anchor (no genomes)" bar - the
  metric this whole line now turns on - plus a `breakdown` table for the
  cross-run grids. **Caveat stated in each desc: the anchor bar is VAL-side
  while the model bars are TEST**; there is no test-side anchor for the
  genome-only runs (D1/C6 ARE the test-side head numbers, which is why they
  appear in 26/27's breakdown).
  **Data-loss note:** D1's export no longer existed - `stage_d` derives its
  filename from (stage letter + ears tag), so D2 wrote `kid_stageD3.json` over
  it. Module 25 is reconstructed verbatim from the json read during the D1 run
  (test 0.1678/0.3563, 0 spaces, 812 genomes, 918,000 head params, 1253s), and
  says so in its `note`. The filename collision is a real trap for any future
  A/B on the same stage - worth a run tag at the source.

- **[2026-07-16] (Claude)** — **EAR ABLATION: the ears SUPPRESS evolution.
  Take them away and genomes BEAT ridge by 22% on pixels alone - the first
  clean case in this line where the evolved model wins because the signal is
  non-linear.**
  Same kid_next.npz pixels, same split, cached word features, genome-only
  head. The ONLY variable is ears on/off:
  ```
  bank        linear anchor   genomes(val)   test     verdict
  eye + ears     0.1703          0.1514      0.1403   ridge wins by 12%
  eye only       0.0614          0.0752      0.0753   EVOLUTION wins by 22%
  [DNE] spaces [56, 1] | TEST 0.0753 / 0.2378 | 37x chance (0.002)
  ```
  **WHY:** the ears (embed_rs_next / _prev / _sim) are dense, type-level
  LOOKUP TABLES. Ridge reads a table perfectly; there is no structure left for
  composition to add. They carry **0.109 of the 0.1703 anchor** - i.e. most of
  every language number this project has reported. So they did not merely
  inflate the score, they ACTIVELY STARVED evolution by handing the head a
  linearly-sufficient answer.
  **The collapse had TWO stacked causes, not one:** (1) the head reading the
  raw environment (found and fixed earlier today), and (2) a ridge-friendly
  table sitting in the environment. Fixing only (1) still left evolution
  losing; fixing (2) is what flips the verdict. My earlier note that "the
  `next` ear is doing the work" was right about the ANCHOR but missed the
  suppression - the more important half.
  **THE POSITIVE RESULT (the real one):** the kid's EYE genuinely carries
  language signal. Four rendered context words predict the next word at
  **0.0753, 37x chance, from pixels alone** - and composition is what extracts
  it (0.0752 vs ridge's 0.0614). Every prior "the eye contributes nothing to
  language" reading in this file is wrong; the eye was simply never measured
  without a table drowning it.
  **THE "QUESTION-SIZE LAW" IS FALSIFIED - my own claim, one entry below,
  is WRONG.** I wrote that % of anchor scales monotonically with question size
  (8 -> 101%, 26 -> 95%, 500 -> 86%). The no-ears arm reverses the ordering:
  ```
  question        classes  ears   % of anchor
  DLEN length        8      on      100.7%
  DFL first letter   26     on       94.6%
  D2 next word       500    on       88.9%
  DFLNE first letter 26     OFF     106.4%   (val 0.1687 vs anchor 0.1586)
  DNE next word      500    OFF     122.5%   (val 0.0752 vs anchor 0.0614)
  ```
  WITH ears % falls with class count; WITHOUT ears it RISES. A real law of
  question size would point the same way in both. It does not - so question
  size was never the driver. **The surviving explanation is EAR SHARE of the
  anchor:**
  ```
  question       ear share of anchor          evolution % of anchor
  first letter   0.0697 / 0.2283 =  31%              94.6%
  next word      0.1089 / 0.1703 =  64%              88.9%
  either, ears off             0%                 106-122%
  ```
  The more of the anchor a LOOKUP TABLE supplies, the worse evolution does
  against ridge - monotone across every cell measured. Same suppression law,
  now on a dial.
  **PREDICTION MADE AND CONFIRMED (DLENNE):** ear share predicted that length's
  anchor would barely move without ears, since length is not what a
  continuation table encodes. Measured: length 0.3186 -> **0.2618 (-18%)**,
  where next-word collapsed 0.1703 -> 0.0614 (-64%). Grid complete and
  monotone:
  ```
  question       with-ears anchor  no-ears anchor  ear share  evo % of anchor
  length              0.3186          0.2618         17.8%        100.7%
  first letter        0.2283          0.1586         30.5%         94.6%
  next word           0.1703          0.0614         63.9%         88.9%
  ```
  Lowest ear share = the ONLY question evolution wins with ears on; highest ear
  share = the biggest shortfall. This is materially stronger than the retracted
  size law: that was 3 points fitted post-hoc and died at first ablation; this
  made a falsifiable call in advance and survived it. It also explains DLEN's
  win with no special pleading - 8 classes was never the reason.
  (DLENNE: val 0.2716 vs anchor 0.2618 = 103.7%, TEST 0.2741 / 0.9445,
  spaces [48, 2].)
  **Observed, NOT a law:** with ears OFF, evolution's margin rises with class
  count (length 103.7% -> first letter 106.4% -> next word 122.5%), plausibly
  because the linear head degrades faster than composition as the problem
  hardens. 3 points, no prediction tested - i.e. exactly the shape of the claim
  that just got retracted. Do not promote it without an independent test.
  Saved as a cross-project memory (evolution-suppression-law) with the
  diagnostic: on any "evolution earns nothing", check genomes-frozen, re-run
  with head_mode=genomes, and ablate table-like channels BEFORE concluding the
  question is hard. Honest metric = % of the linear anchor; genome_share is
  confounded (head = (features+1) x n_classes, so few classes inflate it).

- **[2026-07-16] (Claude)** — **DECOMPOSED D (the user's call): evolution
  BEATS the ridge head for the first time on the smallest question; % of
  anchor scales monotonically with question size; but the sub-answers DO NOT
  compose back up.**
  Setup: three stages over the SAME kid_next.npz pixels and the SAME split.
  DFL/DLEN labels are DERIVED from the same next-word answers (no new data, no
  new pixels, test untouched). Head genome-only throughout, so every stage
  earns its own model.
  ```
  stage  classes  val_final  ref_anchor  % of anchor  test
  DLEN     8       0.3207     0.3186      100.7%      0.3266
  DFL      26      0.2159     0.2283       94.6%      0.2184
  DC       500     0.1461     0.1703       85.8%      0.1464
  D2       500     0.1514     0.1703       88.9%      0.1403   (undecomposed)
  ```
  **1. DLEN BEATS ITS ANCHOR (val 0.3207 > 0.3186)** - the first genome-only
  model anywhere in this line to beat a ridge head on the same environment.
  **2. ~~THE SCALE LAW is clean and monotone: 8 -> 101%, 26 -> 95%,
  500 -> 86%~~ - RETRACTED, see the EAR ABLATION entry above.** I claimed the
  gap to a linear head grows with the SIZE of the question. The ear-ablation
  arm reverses the ordering (no ears: 26 -> 106%, 500 -> 122%), which a real
  size law could not do. The driver is EAR SHARE of the anchor, not class
  count. Left here, struck through rather than deleted, because the numbers in
  this row are real and only the interpretation was wrong.
  **3. DEPTH returns when the question is answerable:** DFL stacked 5 spaces
  (108/43/12/18/7), val 0.1923 -> 0.2022 -> 0.2079 -> 0.2156 -> 0.2159 - a real
  composition ladder. D2 was one wide space (140) + two vestigial; depth was
  doing nothing there.
  **4. COMPOSE IS A NULL - DC IS NOT A WIN.** Handing the 500-way question 308
  pre-solved genomes (188 DFL + 120 DLEN) as ENVIRONMENT channels (bank 2724 ->
  3032) bought nothing: DC **LOST** to D2 on val (0.1461 vs 0.1514) and won on
  test (0.1464 vs 0.1403). The two metrics move in OPPOSITE directions =>
  noise, not signal. DC even earned fewer genomes (147 vs 153). Knowing the
  next word's first letter and its length AS FEATURES does not get you to
  which of 500 words it is.
  **VERDICT: the decomposition thesis HALF-confirms.** Decomposing makes each
  piece answerable (evolution matches or beats ridge on it) - but the
  REASSEMBLY is where it dies. My earlier "the question is too big, decompose
  it" framing was only half right; the open problem is not finding answerable
  sub-questions, it is that their answers do not carry the 500-way question.
  Worth noting the sub-questions chosen may simply be too WEAK a summary of a
  word identity: DFL is only 0.2184 accurate and DLEN 0.3266, so "first letter
  + length" is not merely a coarse summary - it is a coarse summary the model
  cannot even read reliably. Composing two weak, noisy predictors cannot
  exceed what they encode.
  **CORRECTION (same entry, my error):** I first justified DFL as "reusing
  stage B's spelling, which D throws away". That is WRONG and worth recording
  so nobody builds on it. The next word is NEVER RENDERED - no spelling
  competence applies to it, and predicting its first letter is not a
  perceptual question at all. Stage B's competence is used to READ THE CONTEXT
  WORDS via the frozen A->B eye, which every D stage already does; D throws
  nothing away. The same error kills the "per-letter-position spelling" probe
  I proposed: predicting letter i of an unseen word is exactly as unanswerable
  as predicting the word. Do not run it.
  The honest open question instead: the linear anchor 0.1703 may BE roughly
  the information a 4-word context carries about the next word in this
  environment, in which case DC's 0.1461 (86% of it) is near the real limit
  and no decomposition helps. Two cheap probes that would settle it - (a) the
  symbolic next-word TABLE ceiling (the C4 treatment, which for cloze gave
  0.2494/0.4728) to say what is knowable at all, and (b) an EAR ABLATION to
  say how much of 0.1703 is the `next` ear rather than the eye.
  **Infra win:** the word-feature cache did exactly its job - DLEN and DC
  skipped the 21-min eye sweep and each completed in **8 seconds**. Question
  design is now essentially free to iterate on cached features; only a new
  data set or a changed eye costs a sweep.
  All runs detached + watchdogged, `exit=0`, no OOM, sentinel clean.

- **[2026-07-16] (Claude)** — **C7: THE C LINE'S HEADLINE NUMBERS WERE THE
  RIDGE HEAD, NOT THE KID. Genome-only cloze earns 56 genomes (C6 froze 2)
  and reaches TEST 0.0889 where its head reported 0.1538.**
  Setup: stage C re-run with `head_mode="genomes"` on C6's EXACT 100k
  kid_cloze.npz - the head rule is the ONLY variable. Validated by the
  environment reproducing to 4dp: C7's reference linear anchor is **val
  0.1538**, precisely C6's word-bag baseline.
  Raw:
  ```
  [C] head = GENOMES ONLY. Reference linear anchor (bag+ears straight to a
      ridge head, NOT model input): val 0.1538
    [C space 0] opening - bank 2724 channels, base 0 cols
      round 0  +27 (space 27)  val 0.0795
      round 2  +16 (space 48)  val 0.0895
    [C space 0] FULL: 49 genomes, val 0.0905 (+0.0905) (2264s)
    [C space 1] FULL: 7 genomes, val 0.0924 (+0.0019)
  [kid C] DONE: TEST top1 0.0889 top5 0.2505 (2265s)
  C7 RESULT: {"test_acc": 0.0889, "test_top5": 0.2505, "space_caps": [49, 7],
              "ref_anchor": 0.1538, "genome_share": 0.4046,
              "total_params": 47864}
  ```
  **(1) The head bug masked earning in cloze too:** 2 -> 56 genomes the moment
  the head stopped reading the raw environment. C's composition was never
  dead; it was outbid.
  **(2) But the evolved model is far weaker than the head it replaced:**
  0.0889 vs 0.1538 = **58% of its own anchor**. Therefore **C6's 0.1538, C5's
  0.1516 and C3's 0.1418 were RIDGE-HEAD readouts of the bag+ears** - those
  runs froze 2/5/10 genomes, so the accuracy was overwhelmingly the linear
  head. The "61%/69%/73% of the table ceiling" framing was measuring the head,
  not the curriculum. The honest evolved cloze number is **0.0889**.
  **Re-reads the D comparison:** next-word genomes reach **84%** of their
  anchor (0.1403/0.1678); cloze genomes reach **58%**. The supposedly HARDER
  question is where evolution does relatively better - consistent with the
  `next` ear being aligned to next-word prediction, and a hint that the ear
  choice, not the question's difficulty, sets what evolution can reach.
  **NOT re-tested - do not assume:** C6's "data scaling is spent" was measured
  in the head-fed regime. There is no genome-only C at 50k, so whether scaling
  helps when genomes carry the weight is an OPEN question. The scaling entry
  above stands only for the head-fed setup.
  Hygiene: C6's artifacts backed up to `kid_stageC3_c6.json` /
  `kid_modelC3_c6.json` and restored as canonical (C7 is an experiment until
  it earns the stage-C slot on the numbers); C7 archived as `*_c7.json`.
  Decomposed D auto-started behind it on GPU release (2010s wait).

- **[2026-07-16] (Claude)** — **STAGE D + THE HEAD BUG: the semantic-stage
  collapse was the READOUT, not the question. 0 genomes -> 153.**
  Stage D = autoregression: P_C=4 context words as tile strips, name the word
  that FOLLOWS them, target never rendered (strictly harder than cloze - no
  right-hand context). Built by generalizing stage_c with
  data/stage/warm/head_mode whose DEFAULTS keep C byte-identical
  (`head_mode="all"`, `stage="C"`, `warm=None`); `make_next_d` + `stage_d`
  wrapper. No 250-line copy.
  **D1 (legacy head): TEST 0.1678/0.3563 with ZERO genomes frozen** - not a
  model, just a ridge head (98% of 936,481 params). It even beat C6's
  0.1538 on the harder question, which was the tell.
  **THE BUG (the user's catch, and the important result of the day):** the
  head was fed the RAW environment - `all_tr` = 297 bag columns + **1536 raw
  ear channels** + warm = 1835 columns wired straight into the readout. So a
  genome could only earn by beating a FULL LINEAR READOUT OF THE WHOLE
  ENVIRONMENT. That single fact explains the entire curriculum collapse
  (C3 10 -> C5 5 -> C6 2 -> D 0) AND why data scaling made it worse: more
  data enriches the linear answer, RAISING the bar evolution must clear. The
  earlier "cloze is semantic, pixels contain no answer" reading was WRONG -
  the genomes could always find the signal; the anchor was eating it first.
  Stage A was the control sitting in plain sight: empty head -> 515 genomes.
  **FIX:** `head_mode="genomes"` - head sees ONLY frozen genome outputs; bag
  and ears become ENVIRONMENT the genomes read from, never head inputs.
  **D2 (same data, only the head rule changed): 153 genomes [140, 6, 7],
  val 0.0 -> 0.1514, TEST 0.1403 top-1 / 0.3294 top-5, total params 936,481
  -> 97,568, genome_share 2% -> 21%.** Space 0 alone froze 140 genomes from
  an empty head (+0.1441) - the first real earning on a semantic stage in
  this curriculum.
  **HONEST - do not read D2 as a win:** the evolved model LOSES to the linear
  head it replaced (0.1403 vs 0.1678 test; val 0.1514 vs the 0.1703 reference
  anchor). Evolution now carries the model but is ~2.75 points short of ridge
  on the same environment. The head is also still 79% of params (77,000 vs
  20,568) - not the old bug (it only reads genomes now), just the cost of a
  500-class readout. The run self-stopped BY RULE while still climbing (space
  2 gain 0.0028 < MIN_SPACE_GAIN 0.003), and the shape is one wide space (140)
  + two thin (6, 7), so depth is contributing ~nothing.
  This is the user's predicted DECOMPOSITION signal: 140 genomes of signal
  found instantly, then a stall short of ridge = the question is too big to
  answer in one leap, not signal-free. Next: decompose next-word into
  answerable questions (next word's LENGTH, its FIRST LETTER - which reuses
  stage B's earned spelling, currently thrown away at D - function-vs-content)
  and compose. Awaiting the user's call.
  Infra: `warm` defaults None for D (C's 2 genomes measured NEGATIVE as a warm
  base: 0.1703 -> 0.1699); `ref_anchor` + `genome_share` now recorded on every
  run; a genome-only head that earns nothing prints `NO MODEL (0 genomes
  earned)` rather than reporting a head-only number as a result. Cosmetic:
  the inner replay log tag now follows the stage (was hardcoded `[C] slot`).

- **[2026-07-16] (Claude)** — **C6 RESULT: 100k phrases - the data-scaling
  lever is SPENT. TEST 0.1538 top-1 / 0.3455 top-5 (62%/73% of the table
  ceiling), and evolution earned nothing again.** Doubling 50k -> 100k bought
  **+0.0022 top-1** (0.1516 -> 0.1538) and +0.018 top-5 (0.3273 -> 0.3455).
  C5 gained ~4 points of ceiling-share from the previous doubling; C6 gained
  ~1 on top-1. The curve is flat: the last per-slot-GPU-bank scale step
  returned almost nothing, so more phrases is no longer the lever.
  THE REAL SIGNAL: the whole gain is the LINEAR bag baseline rising on data
  (val 0.1538). Composition earned **2 genomes at val 0.1529 - BELOW its own
  0.1538 baseline** - then space 1 produced nothing and stopped. Genomes
  earned: C3 10 -> C5 5 -> C6 2, monotonically dying as data grows. The bag
  keeps eating the content the composition space would have had to earn:
  a richer linear answer raises the bar evolution must clear, and the cloze
  question stays unanswerable by composition on top of it (fitness-as-
  answerable-question: the question is not getting more findable with N).
  Cost note: 2357 of 2360s was feature building (the 4x slot-outer eye
  sweeps); evolution itself ran in ~3s. The RAM fix worked - run completed
  at ~108GB peak where 211GB died. Sentinel: `exit=0 oom_kill=1
  peak_bytes=250999996416` - the oom_kill counter did NOT increment across
  this successful run, which retroactively supports the pre-existing
  oom_kill=1 having been the original C6 (the local same-minute crash is
  still unexplained and remains un-investigated per the user's call).
  No page module added: a +0.0022 null needs the user's call on how to frame
  it.
  CORRECTION (same session): I first wrote here that stage_c has no
  `_record_run` wiring. That was WRONG - stage_c:737 calls it and C6 recorded
  fine. The actual rule-4 gap is narrower and worth knowing:
  `_record_run` hardcodes `_RUNS = <here>/runs/radial_stack` and IGNORES the
  `cfg["env"]` label it is handed, so every curriculum stage files under
  runs/radial_stack/ rather than runs/kid-stageC/ (C6 =
  runs/radial_stack/20260716-162608-radial_stack-251ee1). It also writes on
  the POD, so the local /runs page cannot see it until shadowed back. Its
  whole body is wrapped in `try/except` that only prints "(non-fatal)", so a
  recording failure never fails a run - worth remembering when a run seems
  to have vanished from /runs.

- **[2026-07-16] (Claude)** — **C6 relaunched: stage-C `word_feats` rewritten
  SLOT-OUTER + row-chunked; peak RAM 211GB -> 53GB.** C6 (100k phrases) died
  at 10:18 EDT mid `[C] replaying A(515) -> B(297)`. Cause: `word_feats`
  parked ALL slot banks in system RAM at once - P_C(4) x 515 x N x L_MAX(8) x
  G(8) x G(8) fp16 = **211GB at 100k** - and this container is capped at
  **251GB**, not the 1.5TB the line-530 comment assumed. C5's park-in-RAM fix
  bought exactly one doubling (105GB at 50k) and C6 was the step that ran out.
  Fix: one slot bank parked at a time (52.7GB), B chain in `ROW_CH=10000`
  phrase chunks (5.3GB GPU). Feature values are UNCHANGED - `feature_r0` is
  per-row with baked gate stats, so reordering the sweep cannot move a number.
  A per-chunk `Env` was REJECTED: `Env` derives its patch-PCA basis from
  `Xtr[:2000]`, so each chunk would have invented its own basis and silently
  made features a function of the chunk - the exact
  features-are-the-environment violation the C-line already fixed once. Price:
  P_C eye sweeps instead of one.
  **Diagnosis honesty:** the cgroup `oom_kill` counter is cumulative and
  untimestamped, so it does NOT alone prove C6 was the OOM victim. The local
  Flask + terminals died in the same minute with no local OOM, no reboot and
  no app-error event, while the window carries DNS/domain-controller errors -
  so a network blip SIGHUPing an un-nohup'd run is equally consistent. Both
  causes are now closed: `run_c6.sh` launches under `setsid` (verified PPID 1,
  own session) and always writes `run_c6.exit` (code + oom_kill + peak bytes).
  **Rule-3 gap:** the crash raised no alert. A cgroup OOM SIGKILLs python so
  no in-process completion path can report it, and pod-side `agent_notify`
  writes to the POD's `notices.jsonl` where the panel never looks - so
  `watchdog_pod_run.py` now runs LOCALLY and polls the sentinel. C5's results
  and modules are intact; nothing was lost. No Flask restart needed.

- **[2026-07-16] (Claude)** — **C5: cloze data scaling works - 0.1516 top-1 /
  0.3273 top-5 at 50k phrases (61%/69% of the table ceiling, up from
  57%/62%).** The bag baseline rose most (linear ear content feeds on
  data); composition thin (5 genomes). Infra: word_feats slot banks now
  park in system RAM (4x26GB on-GPU OOMed at 50k). Module 23 on /lm.
  C6 at 100k phrases launching - the last per-slot-GPU-bank scale step.

- **[2026-07-16] (Claude)** — **C4: ear-capacity NULL + the cloze ceiling.**
  128-dim directional ears: 0.1384/0.2850, slightly below 64-dim
  (0.1418/0.2952) - capacity ruled out as the bottleneck. Then the missing
  reference: symbolic tables (tri/bi/uni, vocab-restricted, left context)
  score **0.2494 top-1 / 0.4728 top-5** on the same cloze set. REFRAME:
  the fully evolution-made kid runs at 57-62% of the table ceiling from
  pixels + bred experience alone; the absolute gate bar never existed.
  Note: the kid sees RIGHT context too (tables cannot), so ceiling-plus is
  possible in principle. Module 22 on /lm. Decision for the user: call
  60%-of-ceiling a C-gate pass and open Stage D, or close the gap first.

- **[2026-07-16] (Claude)** — **Stage C3: SEQUENCE EARS - evolution earns on
  the semantic stage for the first time. Cloze 0.1418 top-1 / 0.2952 top-5.**
  Built two directional RS layers on pod #3 (pod #2 died overnight;
  everything restored from shadows, zero loss, new pod staged in 4 min):
  embed_rs_next (words separated by what FOLLOWS: three->five two seven
  six eight four) and embed_rs_prev (by what PRECEDES - grammatical:
  said->killed found separated caught kept). Ear ladder: deaf 0.0668 ->
  sim 0.1212 -> +seq 0.1418; composition genomes froze (10) after three
  zero-earn attempts. Intrinsic gates below SVD but the 30k lesson holds
  (task A/B is the gate). Module 21 on /lm. Next lever: scale directional
  ears (64->128+ dims) or pair-context continuation ears.

- **[2026-07-15] (Claude)** — **Multimodal merge: the kid LETTER recognizer
  (kid_modelA.json, Stage A, 0.996 solo) fused with the vision SHAPE recognizer
  into ONE 36-class classifier.** Late fusion (each frozen bank in its own basis
  at its native scale, concat, one closed-form head): FUSED test 0.9946 (shapes
  1.000, letters 0.9925); fusion beats either bank alone. Cross-modal: the shape
  bank alone reads letters ~0.92 and the letter bank reads shapes 1.0 - evolved
  radial features transfer across modalities. mm_merge.py; merged model at
  multimodal/mm_model.json. No genome retrained; all models backed up first.
- **[2026-07-15] (Claude)** — **Stage C2: EARS attached (the RS-evolved
  semantic layer as listening experience) - cloze DOUBLES to 0.1212 top-1
  / 0.2684 top-5 (baseline 0.0608 -> 0.1085) but the gate still fails and
  evolution earned zero.** The curriculum stays evolution-made (eyes from
  pixels, ears from evolved co-occurrence separation; blank word gets no
  vector). Sharpened diagnosis: RS ears carry SIMILARITY; cloze needs
  SEQUENCE experience - heard word-order, not meaning clusters. Candidate
  next ear: an RS-style EVOLVED CONTINUATION layer (genomes that separate
  contexts by what follows them - the evolutionary counterpart of the
  continuation tables). Module 20 on /lm; Stage D remains gated.

- **[2026-07-15] (Claude)** — **CURRICULUM Stage C: CLOZE - GATE FAILED
  honestly, and the failure is the curriculum's most important lesson.**
  Word-bag baseline 0.0608; evolution earned NOTHING (first space zero
  genomes); final 0.0668 / 0.2150 top-5 (33x chance, no composition).
  Contrast: A (letters) passed in 7s, B (spelling) in 30s - deterministic
  answers in pixels. C is SEMANTIC: 20k phrases of pixels contain no
  answer to "which of 500 words fills this blank" - a kid at this stage
  has YEARS of heard language. Design decision recorded for the user:
  (a) give Stage C listening experience - corpus co-occurrence statistics
  as environment channels (the earlier line's tables, now justified as
  curriculum experience), (b) scale data by orders of magnitude, or both.
  OOM fix en route: 640k-row envs need max_cached=1 + scale-sorted eye
  sweep. Module 19 on /lm. Stage D blocked on the C gate per the
  curriculum's own rule.

- **[2026-07-15] (Claude)** — **CURRICULUM Stage B: WORDS - the model learns
  to SPELL. TEST 0.7066 top-1 / 0.9358 top-5 (V=500, pixels only, 30s).**
  Words as 8-letter tile strips; Stage A's 515 letter genomes frozen as
  the eye. THE MEASURED STORY: orderless letter-bag baseline (eye only)
  0.1910 -> first composition space +45 points by ordering letters ->
  0.7066. The (letter-genome x slot) hand-off carried spelling exactly as
  it carried motion and word order before. 297 new genomes on the frozen
  eye; kid_modelB.json is Stage C's base. Module 18 on /lm. Stage C
  (cloze: name the blank in a short phrase) is next.

- **[2026-07-15] (Claude)** — **THE ENGLISH CURRICULUM (user's design) begins:
  Stage A - LETTERS - gate PASSED at TEST 0.9960 in 7 seconds.** The plan:
  learn language like a kid - A: identify letters; B: identify words
  (word = strip of letter tiles, stage-A genomes frozen as the eye);
  C: cloze (name the blank in a short phrase); D: autoregression - each
  stage its own model warm-started on the previous, no stage advances
  before its gate. Stage A (radial_kid.py): 25k letter tiles (14-26px,
  jitter +-4, noise), 515 R0 genomes, self-stopped at 2 spaces
  [515, 82] exactly like the animation shape twin. Retroactive diagnosis
  of the sentence-strip failure: single-glyph perception earns instantly -
  the vision line needed a childhood, not bigger strips. Module 17 on
  /lm; checkpoint kid_modelA.json is Stage B's perceptual base.

- **[2026-07-15] (Claude)** — **MODULE 17 (vision LM) FIRST ATTEMPT:
  language-grounded-in-vision launched (user's pivot - the glyph line
  resurrected at word level: 12-word strips rendered as 32x32 tiles, NO
  embeddings/tables/one-hots, pixels are the whole environment). Result:
  R0 froze ZERO genomes (round 0 gain +0.0000 vs cap) - the same machinery
  that instantly earns on ID-based words gets nothing from word-tile
  perception. Diagnosis queue: window-MEAN over 12 tiles likely washes out
  per-word variance (try window-max / per-slot fitness), tiles may be too
  small for whole words at 32x32 (try 48px, fewer words), check candidate
  fitness distribution. Also: empty-stack guard needed (crashed at
  torch.stack on zero features). Strip data (36k strips) rendered + cached
  on pod (lm_vis.npz) so iterations are cheap. The thesis is untested, not
  disproven - perception must earn before the language question can be
  asked. radial_lm_vision.py committed; not yet a page module (no result
  to show honestly).

- **[2026-07-15] (Claude)** — **GRAMMAR CONTRACT FIX (user's call: fix the
  environment, don't strap on monitors).** The generation bug's true root:
  feature_vec's GATE normalized its stream with RUNTIME BATCH statistics -
  gated genomes were functions of (row, current batch), violating
  "features are the environment" (and silently making test features depend
  on test-batch stats). The reference-batch inference workaround is
  REJECTED. Fix: gate normalization constants are now ENVIRONMENT-PROVIDED
  - `bake_gate_stats()` freezes each gated genome's gate-stream mean/sd
  over the training environment into its genes at freeze time
  (gt["stats"]); feature_vec uses baked stats when present (evolution-time
  fallback: population batch stats, which is legitimately the population's
  environment during selection). Genomes are pure per-row functions after
  freezing; batch == single-row by construction. TO APPLY: replay
  pipelines must call bake_gate_stats(genomes, train_bank) per space
  before computing features; existing checkpoints get baked on next
  replay. Generation/parity remain PARKED until then per user.

- **[2026-07-15] (Claude)** — **GENERATION SALAD ROOT-CAUSED: the model is
  fine; the single-row step pipeline is buggy.** Parity check (30 held-out
  windows, step argmax vs verified batch preds): 15/30 MISMATCH, and every
  step-side prediction is the same attractor set (your/how/that) while the
  batch side predicts sensible words. All reported metrics (0.5663/0.5671
  etc.) came from the verified batch path and stand. The word-salad samples
  in Module 16's export are the BUG's output, not the model's. Known wart:
  the uni-backoff cont branch builds zeros where training used the unigram
  distribution; a block-diff diagnostic (debug hook in
  lm_generate3._step_logits + gen_diag.py on the pod) is bisecting the
  first divergent feature block on a mismatched row. Fix, re-verify parity
  to 30/30, THEN regenerate. Lesson recorded: any reimplementation of a
  feature pipeline gets a parity test BEFORE its output is trusted.

- **[2026-07-15] (Claude)** — **MODULE 16 - third crank: TEST 0.5663 top-1 /
  0.7280 top-5 - the top-5 usefulness bar (60-70%) is CROSSED.** True
  4-gram table (last three words - never tabulated before) + far skips
  (w-6,w-1)/(w-5,w-2); headroom 84.5% of remaining errors answerable (quad
  alone 68%). Continue-train on the frozen Module-15 base: +80 genomes,
  32s. Blind class 0.2051 -> 0.2938 top-1 / 0.4832 top-5; has-target
  0.736. Cumulative crank 0.2974 -> 0.3992 -> 0.4832 -> 0.5663 at ~1.2M
  params, gradient-free. OOM fix: chained environments must FREE
  intermediate banks (bank3 contains bank2 contains the lean bank - 89GB
  resident before the fix). Module 16 on /lm.

- **[2026-07-15] (Claude)** — **MODULE 15 - second crank: far skip-grams
  (w-4,w-1)/(w-5,w-1) take the lean model to TEST 0.4832 top-1 / 0.6643
  top-5 - the user's "simple neural LM territory" band, gradient-free at
  ~1.2M params.** Loop repeated on the Module-14 model: cont2 rebuild
  verified (0.3987 vs 0.3992), headroom 64.9% of remaining errors
  answerable by the far tables, continue-train +178 genomes in 64s. Blind
  class 0.1106 -> 0.2051, bigram-backoff 0.327 -> 0.476, every slice up,
  no forgetting. Cumulative crank: 0.2974 -> 0.3992 -> 0.4832. Module 15
  on /lm.

- **[2026-07-15] (Claude)** — **MODULE 14 - THE LOOP CLOSES: skip-gram
  environment channels + continue-train take the lean model to TEST 0.3992
  top-1 / 0.5981 top-5 - the best model of the entire LM line, beating
  every fat head (best 0.3296) at ~1.1M params.** Module 13's finding
  applied: the miss class needed NEW environment statistics. Added skipA
  (w-3,w-1) and skipB (w-3,w-2) continuation tables from the independent
  slice (+4,256 channels, cached lm_skip_tables.pkl); HEADROOM diagnostic
  before evolving: 45.9% of blind windows answerable by the new tables.
  Continue-train on the same frozen 285-genome base: +65 genomes in 3
  spaces, 45s pod. Blind class 0.0040 -> 0.1106 top-1 (28x), 0.2521 top-5;
  unigram-backoff 0.058 -> 0.214; has-target 0.480 -> 0.579 - improvement
  EVERYWHERE, zero forgetting. The probe -> environment-gap ->
  continue-train method is validated on language. Module 14 on /lm.

- **[2026-07-15] (Claude)** — **LM modules 12+13: PROBE + CONTINUE-TRAIN (the
  animation Module-3 method ported to language).** PROBE (read-only on the
  RS-30k lean checkpoint): the model NEVER overrides its tables - when the
  continuation table's top-5 contains the target (62% of test) top-1 is
  0.4801; when it doesn't (38%) top-1 is 0.0040, chance. Worst-missed words
  are the commonest (i/you/the/it) - wider-context cases. CONTINUE-TRAIN
  (lm_continue.py): frozen 285-genome warm base, new spaces on a 3x
  table-miss-weighted mix, head refit on the true distribution. THE METHOD
  WORKS (no forgetting, overall 0.2974->0.2993) BUT THE REPAIR DOESN'T:
  miss-class 0.0040->0.0056, evolution dry after 10 genomes. FINDING:
  continue-training cannot conjure signal the environment doesn't carry -
  the miss class needs NEW environment statistics (skip-gram continuation
  tables, syntax-position profiles), not more search. Both modules on /lm
  (renderer now shows breakdown tables + example predictions); registry
  encoding mojibake fixed; two OOM fixes (mix subsampling - the attend
  substrate is ~10GB and cannot be duplicated full-size even on 96GB).

- **[2026-07-15] (Claude)** — NOTE: the full radial-LM arc of 2026-07-15
  (char isolation, word pivot, lean/hybrid shapes, RS embeddings, vocab
  scaling, /lm iteration-log rebuild) is in the MAIN CHANGELOG; this
  project file resumes per-project logging with this entry per AGENTS.md.

- **[2026-07-12] (Claude)** — Round 3: sem_next + grammar_real in a corpus-built
  PPMI/eig feature space; no lookup tables in the model (full-vocab genome scoring
  replaces follower pools at inference). Full details in main CHANGELOG.md same
  date. Key results: grammar_real 56.3% balanced (validated); sem_next beat the
  majority-frequency baseline once (30.1% vs 25.3%) but as a degenerate echo
  genome; with echo negatives evolution truly climbs (0.128->0.233, loops gone)
  yet the static per-word bias reward-hacked into word soup — SS XI's vbias
  Goodhart reproduced. GA repairs shipped: relative per-tensor mutation, fixed
  probe batch, lineage fitness EMA, SS VI bootstrap inits, logfreq-as-environment.
  Energy starved band still unreached (known deviation). Runs:
  `runs/lm/20260712-202708-lm-*`. Next levers: context-gated bias, two-phase
  freeze, transition-only ablation. **Flask restart** needed for /lm.

- **[2026-07-11] (Claude)** — GENREG_RULES compliance pass + hard negatives + rerank
  generation, retrained (crash-recovery: rewrite was complete on disk, run never
  launched; watcher crash root-caused to a cp1252 UnicodeEncodeError in `run_job.py`
  poll(), fixed with stdout errors="replace"). Soft log-prob fitness (§IV.1) in all
  trainers; energy homeostasis (§III) in ga_step (starved=0 observed all run — band
  not reached, tune next round); HARD negatives for `next_word` (same-preceding-word
  followers) + majority-frequency baseline; generate() switched to §VI rerank over
  top-200 follower pools (mined into the artifact). Results: **next_word 21.16% —
  beats 16.67% chance, does NOT beat the 26.92% majority-frequency baseline (honest
  negative)**; fill_word 21.92%; punctuation/opener/length ~unchanged
  (opener_question 0.7015 still best). Qualitative win: no more repetition loops —
  varied, near-grammatical short sentences with correct intent-driven end marks.
  Full details in the main CHANGELOG.md entry of the same date. 10 runs recorded at
  `runs/lm/20260711-233847-lm-*/`. Live /lm needs a **Flask restart**.

- **[2026-07-09] (Claude)** — New group **`next`** (`next_word`) — intent-conditioned,
  autoregressive next-word prediction, reviving the archived pipeline's Selection genome
  concept. **Honest negative result**: 24.25% holdout accuracy, essentially tied with
  `fill_word`'s 24.08% — fixing the train/inference mismatch and adding intent-conditioning
  didn't move the needle. Live check confirms generation is still repetitive. Points to
  negative-sampling strategy (deliberately kept simple/random this round) as the real lever
  for next time. Full details in the main CHANGELOG.md entry of the same date. `generate()`
  switched to use `next_word`. 10 genomes recorded at `runs/lm/20260709-213338-lm-*/`.

- **[2026-07-09] (Claude)** — First generation mechanism (hangman-style, variable-length,
  intent-driven): two new groups, **`length`** (`length_continue`, 61.1% balanced vs 50%
  chance) and **`fill`** (`fill_word`, a contrastive discriminator, 24.1% vs 16.7% chance).
  Generation mechanism itself works (real variable length, correct intent wiring); word choice
  is honestly weak/repetitive — `fill_word`'s accuracy is only modest. Full details, including a
  real design-tension scope reduction from the original plan, in the main CHANGELOG.md entry of
  the same date. New `/api/lm/generate` endpoint + "Generate" section on `/lm`. All 9 genomes
  recorded at `runs/lm/20260709-161129-lm-*/`.

- **[2026-07-09] (Claude)** — New group **"opener"**: `opener_question`/`opener_exclaim` read
  ONLY a sentence's first word to confirm its eventual intent — mirrors the punctuation group
  but forward-looking. **`opener_question` (70.6% balanced) is the strongest genome trained
  yet**, beating `punct_question` (67.2%); `opener_exclaim` (61.4%) also beats `punct_exclaim`
  (58.5%). Confirms the user's intuition that the first word is a stronger, cheaper signal than
  the words before the mark. Also fixed a bug in `record_lm_run.py` that mislabeled all opener
  runs under group "punctuation" on first write. Full details in the main CHANGELOG.md entry of
  the same date. 7 runs (5 punctuation + 2 opener) recorded at
  `runs/lm/20260709-124755-lm-*/`.

- **[2026-07-09] (Claude)** — Split into 5 binary genomes under group **"punctuation"**
  (`punct_end`, `punct_question`, `punct_exclaim`, `punct_semicolon`, `punct_colon`) — fixes
  the collapse from the entry below. All 5 beat 50% chance and none collapsed: exclaim recall
  went 0.15%→62.7%, colon 2.5%→47.9%. Live check: "please bring the following items" now
  correctly gets `:` at 80.8%. Full details in the main CHANGELOG.md entry of the same date.
  5 runs recorded at `runs/lm/20260709-111525-lm-punct_*/`, tagged `group: "punctuation"`.

- **[2026-07-09] (Claude)** — Genome #1 retrained with the train/eval split fixed: **balanced
  accuracy 26.5% (chance 16.7%)**, but collapsed to mostly predicting `?`/`;` (recall on those
  two: 63.7%/67.5%; on `!`/`:`: 0.15%/2.5%). Root cause of the entry below: champion selection
  used raw accuracy on an imbalanced holdout while training batches were class-balanced, so
  training was rewarded for getting WORSE at rare marks. Fixed by making balanced accuracy
  (mean per-class recall) drive both selection and the headline number. Full details in the
  main CHANGELOG.md entry of the same date. Run recorded at
  `runs/lm/20260709-110038-lm-intent02/` (original kept at `.../20260709-024735-lm-intent01/`
  for comparison).

- **[2026-07-09] (Claude)** — Genome #1 (intent recognition) trained: **17.8% held-out, barely
  above 16.7% chance** — an honest first result, not a working recognizer yet. Full details
  (the mining-bug catch/fix, confusion matrix, per-class recall, open question about
  balanced-vs-natural eval) in the main CHANGELOG.md entry of the same date. Run recorded at
  `runs/lm/20260709-024735-lm-intent01/`.

- **[2026-07-09] (Claude)** — **LM name revived, third incarnation.** History: this name
  originally belonged to the char/word-level autoregressive campaign below (archived
  2026-07-06, pivoted to /evolang); /evolang was then archived 2026-07-09
  (`archive/evolang_v1/`, see `documentation/WORDPIPE_FIELD_NOTES.pdf`) after its fluency
  ceiling never moved across every architecture variant tried. `/lm` now names the fresh
  gradient-free genome-pipeline rebuild, starting from nothing but the kept datasets. See the
  main CHANGELOG.md entry for what genome #1 (intent recognition) actually is — unrelated in
  approach to both prior LM lines below (no n-gram tables, no char/word autoregression, no
  reused code).

- **[2026-07-06] (Claude)** — **ARCHIVED — line retired by the EvoLang pivot.** `genreg_lm.py`,
  `genreg_attn.py`, `genreg_enc.py`, `genreg_trustmix.py`, `genreg_distill.py`, `lm_sample.py`,
  `genreg_rerank.py`, `pure_engine.py` moved to `archive/lm_and_tree/`; `lm/attn/enc/encoder/distill`
  run dirs to `runs/_archive/`. The campaign mapped a real boundary (distillation verdict: can't
  gradient-free-train away n-gram tables) and that boundary is exactly why we pivoted. Findings docs
  (`LM_STAGE1_*`, `LM_ENCODER_COMPONENT.md`) kept. Successor: `/evolang`. See
  `documentation/EVOLANG_PIVOT.md`. Code preserved, not deleted.

- **[2026-07-06] (Claude)** — LM: **distillation verdict — you can't gradient-free-train away
  the n-gram tables** (honest negative, maps the boundary). Distilled the 56.4% trust-mix
  teacher into a table-free evolved feedforward neural n-gram (soft-teacher + hard-target hybrid
  fitness, 12k gens). Result: student top-5 **58.0%** (matches/beats the teacher's distribution
  SHAPE) but top-1 only **24.7%** (recovered 44% of the teacher; generation is gibberish). The
  split IS the finding — evolution learns *which chars are plausible* (top-5) but not *which is
  most likely per context* (top-1), and generation needs the latter. Compressing corpus
  statistics into weights is directed high-dim optimization = gradient's job; undirected
  mutation can't do it at precision. Boundary: table-free+gradient-free 34.6% gibberish · n-gram
  tables 56% readable (but 1990s lookup) · table-free+gradients = real LMs (violates rule #1).
  The no-gradient LM sits where BOTH good-LM mechanisms are excluded; evolution's edge is where
  gradients can't go (Intelligence Engine), not raw next-token stats. genreg_train/genreg_distill.py.

- **[2026-07-06] (Claude)** — **TRUST-MIX breakthrough: readable English, held-out top-1 55.2%
  / top-5 84.7%** — passes both docs usability bars (top-1≥30, top-5≥60) the A-series never
  reached. The composition path (not one model doing everything): exact char n-gram channels
  (uni..5g dense + 6g/7g hashed) + the evolved neural model, combined by an EVOLVED
  context-conditional backoff GATE (trust + per-channel evidence κ, Witten-Bell style,
  gradient-free ES). Accuracy climbed with orders+gate: neural-alone 34.6 → 4g-mix 47.2 →
  5g-gated 53.5 → 7g-gated **55.2**. Generation is real English with phrases/punctuation and
  the PROMPT STEERS ("the old man "→"stood the first, by all the other … for his nose and the
  surface is not the same"; picks up Moby Dick vocab whale/jonah/fast-fish) — vs the
  neural-alone gibberish ("dod wing fas coltill carm"). Honest: coherence = high-order n-gram
  statistics; the EVOLVED part is the backoff gate (the neural is ~4% trust, near dead weight —
  count tables win accuracy AND coherence, exactly as the stack notes predicted). A real
  gradient-free char LM. genreg_train/genreg_trustmix.py; full progression in
  LM_STAGE1_FINDINGS.md.

- **[2026-07-06] (Claude)** — LM: **trigram interaction channel = best char result yet,
  34.61% top-1 / 68.78% top-5** (two-phase bootstrap from the 31.9% substrate). +2.7pp over the
  substrate, matching the documented A_101 benchmark (34.00/69.70), closing 37% of the
  substrate→char-trigram-ceiling gap (31.9→39.2, bigram 27.3). The §VI multiplicative gate
  OPENED and paid fitness where the additive copy-attention channel never did — confirming
  multiplicative>additive for pair interaction the recurrent state can't express. Generation
  (t=0.7) shows real words + word-shapes but not sentences (expected at 34%; top-5 barely
  moved). Full table in LM_STAGE1_FINDINGS.md. Next: rollout landscape on top (accuracy is up;
  generation coherence / space-collapse is the exposure gap, still open).

- **[2026-07-06] (Claude)** — LM: **low-rank trigram interaction channel (§VI)** added to
  genreg_lm — the documented word/char-pair primitive, targeting the big untapped gap (char
  substrate 31.9% sits far below the char-TRIGRAM count ceiling 39.2%). Gated multiplicative
  channel `logits += a_lr·(bigram[c_t] + (E1[c_t] ⊙ E2[c_prev]) @ O)` — captures the pair
  interaction the additive prev-char concat structurally cannot (§VI). Wired into
  evaluate/rollout/generate; `trigram_only` flag freezes the substrate for two-phase bootstrap
  (evolve the channel alone, then unfreeze). Gate `a_lr` init 0 (transparent — verified gen-0
  identical to substrate). Two-phase chain running from the 31.9% checkpoint: phase-1 channel
  already opening and climbing (31.8%→32.3% by gen 800, soft improving), vs the copy-attention
  channel which never opened — multiplicative interaction pays rent where additive copy didn't.

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
