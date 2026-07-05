# Changelog — GENREG

Tracks changes made by AI assistants working in this directory.
Multiple AIs share this workspace, so every change should be logged here with
date, author, and a short description. Append new entries at the top of the
log below; don't rewrite existing entries.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-05] (Claude)** — **Published to GitHub: A1CST/GENREG-LAB (private).** Full program
  snapshot pushed — engine, Tree LM, DiffEvo, Animation, LM campaign, panels, docs — with the
  I2 program excluded per user decision (also caught + fixed a real hazard: the old .gitignore
  had a trailing comment ON the pattern line, so `i2_admin_key.json` didn't actually match and
  the PRIVATE KEY was staged — pattern fixed, key never pushed, verified absent from the
  remote tree). Also excluded: runs/ (577MB runtime artifacts, untracked in a follow-up
  commit), dist/, i2_store/, agent_store/, corpora/. Local remote name `genreg-lab`; origin
  (GENREG-Builder) untouched. Final remote tree: 557 files, zero i2/key/runs leaks (verified
  via API).

- **[2026-07-05] (Claude)** — **Per-project changelogs.** Each project page's Changelog button
  now defaults to a project-scoped log with a "show all projects" toggle (app.js modal, all
  pages incl. Animation). Files live in `documentation/changelogs/CHANGELOG_{BUILD,TREE,
  DIFFEVO,ANIMATION,I2,LM}.md` — served live via the existing docs API (no Flask restart) and
  visible on /docs. Seeded from the main changelog by keyword split (best effort — a few
  cross-referencing entries appear in multiple projects). **Convention from now on: every
  change is logged in the MAIN CHANGELOG.md AND appended to the matching project file(s).**
  Verified: files serve 200 live; DOM test confirms project-default + toggle behavior.

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
- **[2026-07-05] (Claude)** — I2: **ZetiFile — the latent encyclopedia** (per user: local Wikipedia,
  new page, always first in search). New `zetifile.py`: serves articles from a LOCAL Wikipedia
  **multistream** dump (`corpora/wikipedia/`, gitignored, NEVER pushed to the network — code ships
  everywhere, corpus stays on this machine). Random access: seek to the block offset -> decompress
  ONE bz2 stream -> extract the page; ~260ms per article over HTTP, no extraction of the 22GB dump
  ever. Title index = SQLite (19,103,098 titles incl. redirects; exact + prefix + FTS5 word search;
  built in ~3 min from the index file). Wikitext stripped to clean blocks (headings/paras/lists;
  templates, refs, tables, file links removed; redirects resolved). Routes `/api/i2/zetifile/{info,
  search,article}` on primary + child; the child serves from its local corpus and **injects ZetiFile
  hits ahead of primary results in /api/i2/search** — ZetiFile is always the first search result
  group (score 100000+). Browser: new `i2://zetifile` app page (serif wordmark + home search,
  result list, wiki-style article view with sections; no emojis), deep links `i2://zetifile/<title>`,
  search-results integration. Articles beyond the still-downloading portion of the dump explain
  themselves ("sits at byte X, only Y downloaded"). Corpus download (enwiki multistream ~22GB +
  index) running in background via resumable curl. E2E tested (info/search/article/search-first).
  **Deployed**: whitelist widened for `zetifile.py`, pushed to primary (reports available:false
  there — by design, no corpus), dist/i2_child refreshed (child on this machine serves it).
  **Follow-ups**: corpus download completed (26.4GB; deep-offset articles verified). ZetiFile
  added to the network LANDING PAGE (per user): republished the primary's `home` with a
  ZetiFile bullet + `[[zetifile]]` link; flushed the child's page cache via the new
  flush-cache API and verified the updated home decodes through the child.
  **REDESIGN (per user — the corpus must feed the LATENT SPACE, not be a serving path)**:
  ZetiFile v2 = on-demand latentization. The dump is feedstock only. On first read of
  `i2://zetifile/<slug>` the corpus-holding child extracts the article, renders it as page
  markup, genome-encodes it, signs it with its node key, caches it as an ORDINARY page doc
  (`Child._zetifile_latentize`), and best-effort publishes it to the primary (`zetifile/` is
  an open ingest namespace in `publish()`) — after that the article IS a network latent,
  served/verified/cached by the standard page pipeline everywhere, offline included. Articles
  render in the canvas page renderer (app deep-links removed); search targets are latent page
  names `zetifile/<slug>` (slug = readable prefix + title-hash6; 19.1M-row slug table backfilled).
  E2E: offline latentize 86KB -> ratio 0.368 in 2.95s verified; push to primary verified;
  corpus-less child received + verified the latent from the network; search-first intact.
  Child v1.6.0. Deployed: primary pushed (incl. zetifile.py), dist/i2_child refreshed —
  restart the child to activate.

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
- **[2026-07-05] (Claude)** — I2: **YouTube-style social layer across the media pages** (per user:
  comments, likes/downvotes, thumbnails, details, up-next). Backend (`i2_service.py`): likes
  extended to **signed up/down votes** (value 1/-1/0, one vote per DID, up/down mutually
  exclusive; legacy like/unlike wire format + `count` field unchanged); new **signed comments**
  (`i2_store/comments/<target>.json`, sig covers `sha256(text)`, ≤1000 chars, ≤500/target,
  newest-first, works on any content id); **video thumbnails** (optional poster frame at upload,
  jpeg/png/webp ≤200KB, stored genome-coded under key id `<vid>:thumb`, served via
  `/api/i2/video/<id>/thumb`); **view counter** (`/viewed` POST, unsigned/node-local, stated
  openly). Routes added to primary (`i2_node.py`) + child proxy (`i2_child.py`, v1.5.0).
  Browser (`static/i2.js`): shared `voteRow`/`commentSection` components (theme-var driven);
  Latent Stream grid gets real thumbs + views + age; watch page gets channel row, 👍/👎, view
  count, description panel, full comment section, view-ranked Up-next with thumbs; upload
  auto-captures a 480px poster frame (canvas, preview shown); Woven posts get votes + 💬
  (inline thread for images, jump-to-watch for videos); Gallery cells get votes + age.
  Integration-tested against an ephemeral primary (votes up/down/switch/clear, comments incl.
  forged-sig 400, views, thumb coded-at-rest round-trip + 404) — all pass. **Deployed**: pushed
  to primary 10.0.0.15 (restarted, endpoints verified live; backup `_backup/push-20260705-094110`),
  `dist/i2_child` refreshed in place to v1.5.0 (running child needs a restart).
  **Follow-up (per user, hard rule): NO emojis anywhere in the UI** — vote buttons are now
  monochrome ▲/▼, the comments chip is plain text ("N comments"), image tiles/Gallery headers
  use ▦, search placeholder and DMV headers de-emojied; typographic marks (✓ ✗ ⚠ ▶) stay.
  Re-pushed to primary + dist child refreshed.

- **[2026-07-05] (Claude)** — Animation Evo: **per-clip presentation regime** (user: the genome
  must NOT see all 240 frames at once). Now ONE clip per generation — the population is scored
  only on a single clip's 24 frames, presented in temporal order (shape moving along its path);
  clip order is shuffled and reshuffled each epoch. Fitness = accuracy on just that clip
  (stochastic minibatch-of-one-clip, same idea as DiffEvo's fresh-minibatch fitness). No
  240-frame batch anywhere in the training loop; the final champion accuracy + confusion /
  per-clip / encoder-PCA report is computed clip-by-clip (ten 24-frame passes) at the end only.
  Training metrics: noisy per-clip fitness + rolling epoch average. Verified baseline (same
  basic model): plateaus fast — champion 0.179 across all clips at gen 107 (~7s); rolling avg
  oscillates ~0.25 because the population chases whichever shape the current clip shows, so no
  generalist emerges under pure per-clip selection. Honest expected result, left as-is. GUI:
  tiles (this-clip / rolling / pop-mean / champion-all-clips), current-clip highlight on the
  grid, chart plots per-clip fitness + rolling avg. Report: `animations/evo_report.png`.

- **[2026-07-05] (Claude)** — Animation GUI: grid captions now label each clip by its **shape
  name** (circle, plus, crescent, …) — bold, primary — with the motion path as secondary text
  ("· wave path"); previously only the path name showed. `/api/animations` now returns each
  clip's `shape`. Live after the user's next Flask restart.

- **[2026-07-05] (Claude)** — Animation: **every clip now has a UNIQUE shape** (user: no
  duplicate shapes across animations). Five new rasterizers in `animation_data.py` (plus,
  xcross, hexagon, crescent, frame/hollow-square); mapping: line→circle, diagonal→square,
  swoop→triangle, loop→ring, figure8→diamond, zigzag→plus, wave→xcross, spiral→hexagon,
  bounce→crescent, scurve→frame. `animations/` regenerated + verified visually. Animation Evo
  becomes a balanced 10-class task automatically; palettes extended to 10 and the report PNG
  re-laid-out for a 10×10 confusion matrix. New baseline (same basic model, accuracy fitness,
  mutation only): 0.358 at max_gens 500 (~3.4 min), still creeping — not yet a hard plateau;
  best clips line 0.92 / swoop 0.67 / bounce 0.63, worst figure8 0.0. Report:
  `animations/evo_report.png`.

- **[2026-07-05] (Claude)** — I2 child: **flush-cache control** (child v1.4.0). New
  `Child.flush_cache(categories)` — wipes cached pages + sealed media blobs (all or per
  category), keeps genome/identity/config/purge state; returns files + bytes freed. Exposed
  as a confirm-guarded "Flush cache" button on the GUI Cache tab and as same-origin
  `POST /api/i2/child/flush-cache` (optional `{"categories": [...]}`) for headless children.
  Tested: full + selective flush, route incl. bad-category 400; genome untouched.
  Deployed to `dist/i2_child` by copying the code files in place (NOT via `package_i2.py`,
  whose rebuild would have wiped the live child's `i2_cache/` inside that folder);
  `__pycache__` cleared. The running child picks it up on its next restart.

- **[2026-07-05] (Claude)** — Animation: **Animation Evo — deliberately basic mutation-only shape
  classifier** (`genreg_train/animation_evo.py`). Genome: frame(4096) → tanh(enc 12) →
  tanh(hid 24) → shape logits (5 classes); ~49.6k params. Evolution is mutation ONLY (no
  crossover): elites kept, children = single elite parent + fixed-step Gaussian noise. Fitness
  is exactly the stated goal — fraction of frames whose shape is named correctly, despite the
  movement. Kept intentionally basic per the user (an earlier balanced-CE + self-adaptive
  variant that reached 100% was REVERTED — the user wants the honest baseline; it plateaus
  around 0.62, mostly the majority class). Report: `animations/evo_report.png`. Reused JobHub
  from diffuse_service (now parameterised by program name); runs persist to `runs/animevo/`.
  New WS `/animevo` + full /animation page UI (config sidebar, accuracy chart, live per-clip
  predictions on the playing dataset grid, final confusion/PCA/per-clip report) — live after
  the user's next Flask restart. Headless anytime: `python -m genreg_train.animation_evo`.

- **[2026-07-05] (Claude)** — Tree LM evolve-embed: **mixed-dimension seeding — NEW BEST
  word-level result, −1.67 pts vs bigram.** The drift problem (64→66 in 100 gens) is fixed by
  seeding the population with REAL heuristic genomes at several capacity levels ([start/2,
  start, 2×start, 4×start] — built per level with the co-occurrence embedding seed), so
  selection compares whole capacity levels head-to-head from generation 0 instead of growing
  one random column at a time. Measured (run `20260705-000924-tree-1ce73a`, config J + evolve-
  embed from 64, ladder [32,64,128,256]): the whole population converged to embed **256**
  within 10 generations; encoder ridge acc 0.148 (drift version: 0.129); final **13.20% vs
  word-bigram 14.88%**, beating hand-set embed 128 (−2.15). Campaign delta progression:
  −7.9 → −6.1 → −4.5 → −2.15 → **−1.67**. Since evolution chose a dim above the manual clamp,
  the sidebar **Embed dim max is raised 128 → 256** (parse_config + tree.html, with a word-mode
  hint). `_evolve_encoder_embed` now takes (genome, dim) seed pairs.

- **[2026-07-05] (Claude)** — I2: **cache keys are now network-issued and RAM-only (v2, per user:
  "never at rest, changes each time")**. New primary endpoint `POST /api/i2/node/cachekey`
  (`i2_node.py`) + `i2_service.issue_cache_master()`: child sends a DID-signed, freshness-checked
  request (same trust bar as heartbeats); primary derives `HMAC(node-key, did)` — deterministic,
  nothing stored — and records the issuance (never the key) on the ledger. Child (`i2_child.py`):
  master fetched at connect, held ONLY in RAM; every cache write seals with a one-off key
  `HKDF(master, fresh 16B salt, kind|mid)` → AES-256-GCM (`I2EC\x02` header carries the salt, so
  decryption is stateless). No key file exists at rest anymore: v1 blobs are re-sealed and
  `cache_key.json` deleted on first issuance; identity-less children keep the v1 local-key
  fallback. If no key is obtainable (offline restart), sealed files are kept (KeyError ≠ tamper)
  and nothing is ever written plaintext. `i2_bench.py cache` extended (10 checks, all pass);
  `transfer` now runs a child end-to-end: key issuance 13ms, cold fetch+verify+seal 77.7 MB/s,
  warm sealed-cache hit 322 MB/s, restart+re-key 272 MB/s (8MB video, loopback). **Deployed**:
  pushed v1.3.0 to primary 10.0.0.15 via signed admin push (self-restarted; endpoint verified
  live; backup `_backup/push-20260704-232331`). Children need the updated `i2_child.py` to use it.
  (1) `i2_child.py`: cached media blobs (`videos/*.bin`, `images/*.bin`) are now sealed with
  AES-256-GCM under a per-child random 256-bit key (`<cache>/keys/cache_key.json`) — the
  genome-CTR coding alone gave no at-rest confidentiality since its key derives from the
  *public* genome hash + content id. Legacy plaintext cache files still serve and are migrated
  to sealed form on first hit; tampered/foreign-key files are dropped and refetched (GCM auth,
  AAD binds kind+id). (2) New `i2_bench.py`: `verify` (codec round-trip + measured bits/byte —
  **2.271 b/B, 3.52×** on corpus samples, beats zlib-9 at 4.14 b/B; claim confirmed), `cache`
  (seal/unseal/tamper/wrong-key self-test, all pass), `transfer` (spawns ephemeral primary,
  measures genome/page/blob transfer: video blob ~114 MB/s wire + 1.5 GB/s AES decode over
  loopback; text limited by the pure-Python coder at ~126 KB/s decode → compression wins on
  links slower than ~90 KB/s, decoder is the bottleneck above that). (`encoder_evolve_embed`,
  per user) + one word-level test. New variable-embed GA (`_evolve_encoder_embed`): each
  individual carries its own embed_dim; growing appends an embedding column AND its mixer row
  in every position block, shrinking drops the lowest-magnitude column, crossover in the shared
  embed subspace; bounds [8, max(4×start, 256)]; CPU (ragged genomes), shallow encoder only,
  mutually exclusive with evolve-dims; evolved dim recorded in the encoder card
  (`embed_dim`/`start_embed_dim`/`evolve_embed`). Also fixed `ctx_best` scoring to use the
  final encoder (correct for any evolved dims). **Test result (run 20260704-233101-tree-b6a79f,
  config J + evolve-embed from 64, 100 gens):** evolution drifted 64 → 66 only and plateaued at
  0.129 ridge acc by gen 20; final 11.95% vs bigram 14.88% (−2.92 pts) — WORSE than hand-set
  embed 128 (−2.15). Honest read: one-column-at-a-time growth has no fitness gradient (a fresh
  random column helps only after it learns something), so pure-accuracy evolution can't climb
  to 128+ in 100 gens — matches the EEC finding that size only changes direction under
  pressure. Promising follow-up: seed the population at MIXED dims (64/128/192/256) so
  selection picks the level directly instead of drifting one column at a time.

- **[2026-07-04] (Claude)** — Tree LM word-level: **first test campaign (10 configs, RTX 4080)
  — best −2.15 pts vs word-bigram; full findings in
  `documentation/WORD_LEVEL_TREE_LM.md`.** Delta progression −7.9 → −2.15 across rounds by:
  embed dim 24→128 (the binding constraint at word level — 24 dims can't carry which-of-2048-
  words identity), killing the encoder speed phase (TRAP: `encoder_speed_generations` defaults
  to 40 in the trainer and DEGRADES the evolved encoder 0.148→0.10 ridge acc before tree
  building — UI runs safe, scripted/sweep runs that omit the key are not), and window 8 (beats
  4 and 16 at this capacity). Best run `20260704-231826-tree-ee861e`: 12.7% real-word accuracy
  vs bigram 14.9%; generation is grammatical phrase structure ("in the world of the right of
  the world, and she went not…"). Honest read: the remaining gap is structural (exact count
  table vs compressed continuous encoder — the I2 lesson again); the lever that should PASS
  bigram is count-backoff blending at the leaves (documented as next work). Known issue:
  routing_layers 0 (flat 2048-way leaf) collapses to 1.7% — GA regresses off its ridge seed on
  a 526k-param genome; use routed trees at word level.

- **[2026-07-04] (Claude)** — Animation: **10-clip procedural dataset** (`genreg_train/animation_data.py`).
  Each clip is 24 frames of 64×64 grayscale, black background, one white shape moving on a
  unique path: line, diagonal, swoop, loop, figure-eight, zigzag, wave, spiral, bounce, S-curve
  (shapes vary: circle/square/diamond/ring/triangle; anti-aliased sub-pixel motion).
  `python -m genreg_train.animation_data` renders `animations/<name>.npy` (uint8) + preview GIFs
  — already run, files are in `animations/`. Added `/api/animations` (base64-packed frames) and a
  playing preview grid on the /animation page (`static/animation.js`). Route + API need the
  user's next Flask restart; the .npy/.gif files are usable now.

- **[2026-07-04] (Claude)** — New **Animation** tab scaffold: `/animation` route + `templates/animation.html`.
  Page is intentionally mostly blank — just the topbar with cross-links and the shared
  terminal dock / agent panel / config panel scripts (same as the other program pages).
  Added "Animation ↗" nav links to the build, tree, diff, and runs pages.
  Flask server NOT restarted — a restart is needed for the new `/animation` route to be live.

- **[2026-07-04] (Claude)** — Tree LM: **word/token-level prediction** (user: byte level can't
  get sentence structure — "let the model develop actual semantic embeddings"). Implemented as
  a **Token level** mode on the existing tree page (the engine was already vocab-parameterized;
  a v2 page would have duplicated ~2000 lines for two fields), byte mode untouched/default.
  - **Tokenizer** (`WordVocab`/`load_word_tokens`): corpus decoded UTF-8 with typographic
    punctuation folded to ASCII (curly quotes had shattered into junk vocab entries), tokens =
    words (apostrophes kept: "don't"), numbers, single punctuation; top-K vocabulary, id 0 =
    `<unk>` (2048 → 85.6% coverage of the 10.4M-token corpus); detokenizer with spacing rules.
    Cached per vocab size (~5 s once).
  - **`<unk>` policy**: never a training/eval TARGET (samples with unk targets dropped,
    oversampled to compensate — keeps accuracy honest on real words; without this the model
    just predicted unk forever) and never EMITTED at generation (masked in the leaf); it
    remains valid context. Applies to tree + encoder trainers.
  - **Config**: `token_mode` byte|word + `vocab_size` (256–8192); routing-layers leaf math now
    uses the real vocab; word-vocab reuse guard on saved encoders (byte encoders can't be
    reused in word runs). Bigram baseline now (V,V) int32. Co-occurrence embedding seed
    switches from full SVD to a Gaussian random projection above V=1024 (SVD would take
    minutes at 2048+). Saved models persist their vocabulary (`vocab_blob` in model.npz) so
    /runs replay + generation detokenize correctly; `_load_model` returns (root, encoder,
    vocab). Generation, trace inspector (word labels in samples/top-k), letters cloud (labels
    words when the run is word-mode), and word cloud (words ARE tokens — encoded by id) all
    mode-aware. UI: "Token level" selector + vocabulary-size field on /tree.
  - Verified: clean vocab top-15, tokenize/detokenize round-trip, end-to-end CPU smoke run
    (train → eval → generate readable words, zero <unk> in output → saved-model replay via
    /runs works with the persisted vocab). GPU word-level test campaign running; results to
    follow in the Agent panel. **Needs a Flask restart.**

- **[2026-07-04] (Claude)** — Run-Config panel: **config history + copy buttons.** New
  "Config history — all projects" section at the bottom of the panel: the newest 3/5/10 runs
  (selector persists in localStorage) ACROSS every project — tree, encoder, diffevo, engine
  envs mixed by recency. Each entry: status dot, project, time, and a program-aware one-line
  result (tree → `acc 35.2%`, encoder → `nc %`, diffevo → `L1 0.12`, engine → `score N`);
  click to expand the full flattened config + results inline (run id in the expansion deep-
  links to /runs#id); every entry has a **copy** button that puts the run's config on the
  clipboard as pretty JSON (navigator.clipboard with execCommand fallback, "copied ✓"
  feedback). The tracked active run also gained a **copy** button in the panel header.
  Backend: `runstore.recent_runs(limit≤50)` + `GET /api/run-history?limit=` (7–12 ms), polled
  every 12 s on ALL pages (history is cross-project, unlike the scoped active-run view).
  Verified: real-data cross-project history (tree+2048 interleaved) + 8/8 DOM-harness checks
  (render, headlines, expand/collapse, clipboard JSON round-trip, feedback, limit
  persistence). **Needs a Flask restart** (new route).

- **[2026-07-04] (Claude)** — Tree LM: **fixed "cannot reshape array of size N into shape
  (…)" when generating from a run on /runs.** Root cause: a run that REUSES a saved encoder
  (encoder_id) adopts the encoder's dims in memory (tree_service ~1759) AFTER config.json was
  written with the sidebar dims, so replay/embedding rebuilt the encoder with the wrong shape
  and `unpack_genome`'s reshape blew up. Two-part fix: (1) `_save_model` now records the
  encoder's REAL dims (`encoder_dims`) in model.npz and `_load_model` prefers them — new runs
  are self-describing; (2) legacy files: `_adopt_encoder_dims()` copies dims from the run's
  encoder_id (same adoption the trainer did) before loading — wired into infer_run,
  embedding_cloud, and word_cloud. Verified on a real reused-encoder run: reproduced the exact
  reshape error via the old path, then generated 400 clean chars (no U+FFFD — the UTF-8
  sampler visibly working) and both embedding clouds render. **Needs a Flask restart.**

- **[2026-07-04] (Claude)** — **Run-Config panel** — a second floating window (sibling of the
  Agent panel: draggable header, minimizes to a "Config" pill, position/state persist) that
  tracks the run belonging to THE PAGE YOU'RE ON, showing two sections: the run's FULL
  configuration (every setting, nested keys flattened e.g. `params.hidden`) and its RESULTS
  (live last-generation metrics while running; the full summary — accuracy/best/eval/etc. —
  once finished). Adoption rules exactly per spec: navigating to a page CLEARS the panel (it
  never shows a run that finished before you arrived); it adopts a run you start on that page
  (created ≥ page-entry, compared in local-time ISO to match runstore stamps); it adopts a run
  already mid-flight when you arrive and fills in results when it completes; a newer run
  replaces the tracked one. Status chip (running/finished/stopped/error) + "open run" button
  deep-linking to /runs#id. Scope per page: / → engine envs, /tree → tree+encoder, /diff →
  diffevo; other pages show an explanatory idle note. Backend: `runstore.latest_run(envs)` —
  newest run dir by id timestamp, reads only that run's config/summary + history tail
  (last_metric), measured 7.6 ms/call — served by `GET /api/active-run?scope=`, polled every
  4 s (works regardless of which client started the run, since it reads the run files on
  disk). Files: `static/configpanel.js` (all 6 pages), styles in style.css. Verified: 10/10
  jsdom state-machine checks (old-run cleared, adopt-on-start, config+live metrics, results
  fill-in on finish, chip transitions, mid-flight adoption, non-project message, newer-run
  replacement) + real-data latest_run for all three scopes. **Needs a Flask restart** (route).

- **[2026-07-04] (Claude)** — Runs dashboard: **word-level embedding cloud** alongside the
  letter one (user: "i need both word and letter"). The byte cloud can only ever show letters —
  the tree LM is byte-level, so 256 byte vectors is the whole embedding TABLE; words exist only
  as encoder computations. New `tree_service.word_cloud()`: takes the corpus's ~180 most
  frequent words (cached count over the 49 MB corpus; single letters dropped except a/i),
  encodes each as `' word'` through the run's trained context encoder (batched, left-padded
  like encode()'s own padding), PCAs the context vectors to 3D with the same variance/effective-
  rank diagnostics. `runstore.word_embedding()` + route `GET /api/runs/<rid>/words`. Run detail
  page now shows TWO cards for tree/encoder runs: "Embedding space — letters" (unchanged) and
  "Embedding space — words": drag-to-rotate scatter, dot size = word frequency, top-30 words
  labeled, hover any dot for its word + corpus count. Verified on a real tree run: 180 words,
  2.8 s cold / 0.07 s cached, effective rank 21 (real structure), coords unit-scaled.
  **Needs a Flask restart** (new route + service function).

- **[2026-07-04] (Claude)** — Agent board: **fixed duplicate notice ids across processes.**
  Flask and CLI posts each cached an in-memory id counter, so two writers handed out the same
  id (two id-5 notices existed in the live feed — which double-counted the unread badge).
  `post()` now re-scans the file's max id under the lock on every write; the existing feed was
  renumbered sequentially. Verified: simulated two fresh processes → sequential ids.
  Applies live for CLI posts; Flask picks it up on its next restart.

- **[2026-07-04] (Claude)** — **Config panels no longer reset when you leave the page.**
  Diagnosed the "configuration panel resets" report: the Build page's Control Panel already
  persisted (training.js `genreg_controls` — verified working with a jsdom repro of two page
  loads over the real markup), but the Tree LM and DiffEvo **Configuration sidebars had no
  persistence at all**. New `static/cfgpersist.js` (loaded on /tree and /diff after the page
  scripts): saves every input/select/textarea with an id (skipping the terminal dock / Agent
  panel / changelog modal) to localStorage per page on any edit, restores on load, then fires
  input+change so dependent UI re-syncs (e.g. the DiffEvo sampler field shown in denoise mode);
  selects whose options arrive async over the socket (the tree Encoder dropdown) re-apply their
  saved value via a MutationObserver once the option exists; a `restoring` flag prevents saving
  half-restored state. Verified end-to-end in jsdom: tree fields, diff fields, dependent
  visibility, and the async encoder select all restore across page loads. Static-only — browser
  reload picks it up, no Flask restart.

- **[2026-07-04] (Claude)** — Agent panel: **run notices are clickable** — any notice carrying a
  run_id (the auto run-alarms) navigates to `/runs#<run_id>`, landing on that exact run.
  runs.js deep-linking upgraded: hash handling is now a shared `openFromHash()` that also fires
  on `hashchange` (works when already on /runs) and reloads the run list first if the id isn't
  known yet (a just-finished run). Linked rows show a pointer cursor + hover highlight.

- **[2026-07-04] (Claude)** — Agent panel: **crashed runs now alarm too.** The run-alarm hooks
  only fired on done/sweep_done/stopped, so a run that died mid-training ended silently. Added
  `error` to the hooked event types in both job hubs and the engine `/train` socket;
  `post_run_event` turns it into a red `alert` notice ("<program> run FAILED" + the exception
  message, run_id when one exists). Verified: error event → alert posted.

- **[2026-07-04] (Claude)** — Agent panel: removed the robot emoji per user request — the header
  is just "Agent" and the minimized pill is a small uppercase "Agent" chip (mono font, same
  badge/pulse behavior). Docs and CLI docstring updated to match.

- **[2026-07-04] (Claude)** — **Agent panel** — a floating, draggable, minimizable notice feed
  on EVERY page (topmost, above content, below modals), the shared channel where AIs and
  automated jobs post updates/test results/alarms. New `agent_board.py` stores notices as JSON
  lines in `agent_store/notices.jsonl` (id/ts/kind/source/title/body/run_id; kinds info · test ·
  run · alert; trimmed to newest 1000) — file-based so posting works without the server. Routes:
  `GET/POST /api/agent/notices`. New CLI `agent_notify.py "title" ["body"|-] --kind --source`
  (stdin piping supported) for terminal AIs. **End-of-run alarms**: the engine `/train` socket
  (app.py) and both job hubs (tree_service — done/sweep_done/stopped — and diffuse_service —
  done) auto-post a `run` notice with run_id + headline metrics when any training job ends;
  hooks are best-effort (can never break training). UI `static/agentpanel.js` (all 6 pages):
  polls every 8 s, unread badge, drag-anywhere header (position/minimized state persist in
  localStorage), minimizes to a pulsing 🤖 pill, "mark read"; a notice arriving while a page is
  open POPS THE PANEL OPEN (the alarm). Docs: `documentation/AGENT_PANEL.md` (visible on /docs)
  incl. instructions for tying a CLI AI (e.g. Claude in the terminal dock) into the feed.
  Verified: board post/list/since/kind-coercion/id-continuity + run-event formatting all pass;
  CLI posted the first 3 live notices. **Needs a Flask restart** (routes + hub hooks) — the
  panel UI loads now but stays empty until then; same restart as the other 2026-07-04 changes.

- **[2026-07-04] (Claude)** — Runs dashboard: **filter / label / favorite / group / tags** —
  organization for tabs with ~200 runs. Per-run user metadata lives in a new `meta.json` in the
  run dir (own file so it never races the trainer's config/summary writes): `label` (≤80 chars,
  shown as the run's list title with the date after it), `favorite` (bool), `group` (≤80),
  `tags` (≤12 × ≤24 chars). `runstore.set_meta()` merges partial patches with length caps;
  `list_runs`/`get_run` carry the fields; new route `POST /api/runs/<rid>/meta`. UI (runs.html/
  runs.js/style.css): a filter toolbar under the env tabs — live search box (matches label, tags,
  group, id, date, device, constraints, status), status dropdown, sort (newest first — the new
  default — / oldest / best score), and a ★ favorites-only toggle; the run list groups into
  collapsible sections by `group` (named groups start collapsed, collapse state persists in
  localStorage per env+group; no groups → flat list as before), favorites pin to the top of
  their section, each row gets a ☆/★ toggle and tag chips; the detail panel gains a ★ button
  plus label/group/tags editor (group input autocompletes existing groups; Enter saves). Header
  shows "N/M shown" while filtering. Labels/tags render via textContent (no HTML injection).
  Verified: 228-run store round-trip (set/partial-patch/caps/unknown-id) all pass; JS/PY syntax
  clean. **Needs a Flask restart** (runstore + app.py route), same restart as the UTF-8 fix.

- **[2026-07-04] (Claude)** — Tree LM: **UTF-8-constrained sampling** — fixes the "� boxes"
  in generated text. Generation is per-byte but the corpus is UTF-8, so the sampler could emit
  stray 0x80–0xFF bytes that never form a valid sequence and decode to U+FFFD. `_generate`
  (Generate box + /runs replay) now masks each leaf pick to bytes that keep the stream valid
  UTF-8 (`_utf8_state`/`_utf8_allowed`/`_pick_byte` in tree_service.py): continuation bytes
  only inside an open multi-byte sequence (with the lead-specific second-byte ranges),
  otherwise ASCII + valid lead bytes (C2–F4); if a leaf can't continue an open sequence the
  unfinishable partial is popped (never into the prompt) and the pick retried; a leaf with no
  valid byte at all falls back to the old unmasked pick. Model, routing, training, and
  temperature semantics unchanged; the trace inspector (`trace_generate`) intentionally left
  raw. Verified: 27 unit checks incl. a 3000-step random-leaf generation decoding as STRICT
  valid UTF-8 with zero U+FFFD. **Needs a Flask restart** (Python module change).

- **[2026-07-04] (Claude)** — **Terminals on every project page + per-page tab memory.**
  The daemon-backed terminal dock (same PTYs as the build page — one shared `/ws` bridge, so
  a shell keeps running wherever you view it) is now on **DiffEvo (/diff), Tree LM (/tree),
  Runs (/runs), and Docs (/docs)** too, not just Build (/) and I2. New `static/termdock.js`
  injects the identical dock markup (tabs, + New Tab, Changelog/Claude/Clear/Restart/Stop,
  changelog modal, drag-resizer with per-page height persistence) into any page that doesn't
  hard-code it; the four templates now load xterm + termdock + app.js. And **each page
  remembers its own active terminal tab** (`localStorage` keyed by pathname, in app.js): pick
  a tab on /diff and it's still the selected one when you come back, independent of the tab
  chosen on /tree or /. Also guarded app.js's connection dot/text so pages whose topbar lacks
  them (the indicator lives in the injected dock header there) can't break the terminal
  script. No Flask restart needed — reload the browser tabs.

- **[2026-07-03] (Claude)** — DiffEvo: added **unrolled training** (diffuse mode, default on).
  Each champion now trains on the ACTUAL output of the champions above it — the real image the
  reverse walk hands it at inference — instead of a freshly-noised image. Requires training
  noisiest-first (K→1) so upstream champions exist; each champion's input is built by running the
  partial walk down to its level (`_walk_state` + batched `_apply_champion_batch`). This removes
  the train/inference mismatch that made the walk's low-noise tail drift back up. Measured (8
  levels): the walk is now **monotonic to the end** and final L1 dropped 0.23→0.12 — matching
  single-shot denoise quality — with held-out average improvement ~25%→**~47–49%** and consistent
  across seeds/level counts. Also fixed the live preview to pick a **median-structure** test image
  instead of `test[0]`: the old fixed pick could be a near-flat outlier (98.7% of images improve,
  but a flat image can't beat its noise floor so its trace looked like divergence) — the held-out
  average is reported separately on the tile. UI: "Unrolled training" checkbox (diffuse only).

- **[2026-07-03] (Claude)** — DiffEvo: added **"diffuse" process mode** (now the default) —
  the actual diffusion reverse process: start from noise and denoise step by step toward the
  image. The mechanism is a change to *what each genome learns*: in diffuse mode champion_k is
  trained on one **incremental** step (input `x0 + σ_k·z` → target `x0 + σ_{k-1}·z`, same noise
  realization), i.e. remove only a slice of noise, not jump to clean. No single genome can finish,
  so the reverse walk (`reverse_walk`, composes champions K→1 from the noised start, no re-noising)
  genuinely moves closer each step and stays on-distribution. Kept the prior behavior as
  **"denoise" mode** (jump-to-clean champions + single-shot sampler). Measured (8 levels): diffuse
  walk L1-to-target 0.304 (noise) → 0.281 → 0.263 → 0.238 → 0.213 → 0.199 then a mild tail rise to
  0.216 (low-noise steps compound small errors); denoise single-shot gets a sharper final 0.117 but
  with no step-by-step process. UI: Process-mode selector (diffuse default; sampler field hidden
  unless denoise), reconstruction card relabeled "start · noise → final output," per-step L1 on the
  filmstrip (green=improved, red=diluted). Next lever for a sharper walk: cosine σ-schedule / more
  low-noise levels, or unrolled training (train each step on the actual walk distribution).

- **[2026-07-03] (Claude)** — DiffEvo: (1) **runs now persist to the /runs dashboard.**
  Each training run writes `runs/diffevo/<id>/{config.json,history.jsonl,summary.json}`
  in the same layout runstore uses, so a "diffevo" tab appears with a live L1-vs-generation
  sparkline (fitness.best/mean per gen, descending = improving) — written directly, not via
  runstore, to keep DiffEvo free of the engine checkpoint format. (2) **Default sampler is now
  `single`-shot.** Confirmed the "gets worse after step 1" behavior is fundamental, not a bug:
  denoising a known noise level, only the matched first step sees the *true* observation; every
  later champion sees only a processed estimate, has no new information, and can only smooth/
  contract → error rises each step. Single-shot (apply the champion matched to the input level,
  once) is therefore optimal here and can't degrade. DDIM/ancestral kept as selectable multi-step
  modes for the harder regime where stacking would pay (generation from pure noise, or steps
  conditioned on the original observation). UI copy + default updated.

- **[2026-07-03] (Claude)** — DiffEvo: fixed the reverse chain collapsing a good
  first-step reconstruction into a grey blob. Root cause: the sampler predicted x0
  then **re-noised it with fresh random Gaussian** each step; a chain of local-averaging
  denoisers over-smooths that new noise and contracts contrast → regression to the mean.
  Added a `sampler` option: **`ddim`** (new default) is deterministic — it carries the
  residual already in the image, rescaled to the next level (`x <- x0 + (s_next/s_k)(x-x0)`),
  so the chain converges onto x0 instead of walking away; **`single`** applies only the
  strongest (top-level) denoiser; **`ancestral`** is the old stochastic behavior (kept for
  contrast). Measured on a 6-level run: per-frame L1 for ancestral diluted 0.185→0.277
  (+5% held-out), ddim held ~0.16–0.19 (+32%), single-shot 0.160 (+38%). Notable finding:
  **at this data/noise scale stacking doesn't help — one tiny ~90-param genome already
  captures the signal**; extra local passes only bleed contrast. UI: sampler dropdown +
  per-frame L1 labels on the reverse-chain filmstrip (green if a step improved, red if it
  diluted) so the collapse-or-not is visible.

- **[2026-07-03] (Claude)** — DiffEvo: switched the selection objective from L2/MSE
  to **L1 (mean absolute pixel error)** per request. Squared error let a few large-miss
  pixels dominate selection (which a ~90-param genome can't chase); L1 rewards getting
  the bulk of pixels close — the "average job" objective. Renamed all reported metrics
  (`best_l1`/`mean_l1`, `in_l1`/`out_l1`, `test_in_l1`/`test_out_l1`, `l1_by_level`) and
  the UI labels to match. Smoke test now shows a cleaner monotonic per-level error floor
  rising with σ (0.077→0.139) and ~13% held-out L1 improvement.

- **[2026-07-03] (Claude)** — New project **DiffEvo** (`/diff` page, WS `/diffuse`),
  a fourth program alongside GENREG/Runs, Tree LM, and I2. Denoising diffusion by
  neuroevolution: decomposes image reconstruction into a stack of individually-easy
  reverse-diffusion steps, evolving one shared population of tiny convolutional
  per-pixel denoisers (patch → tanh(H) → clean centre pixel, ~90 params) per noise
  level. Fitness is denoising MSE over a *fresh minibatch of samples each generation*
  (stochastic averaging → generalizes instead of memorizing one image), with plateau
  early-stop per level. Per-level champions stack into a reverse chain (predict x0,
  renoise to next level). Files: `genreg_train/diffuse_service.py` (self-contained
  numpy: procedural dataset, ES, JobHub survives WS disconnect like tree_service),
  `templates/diff.html`, `static/diff.js` (canvas reconstructions + hand-rolled SVG
  fitness/level charts). Wired into `app.py`; nav links added to Build + Tree pages.
  Smoke test (4 levels, pop 24): held-out test-set denoising MSE improved ~18%,
  per-level MSE floor rises with σ as expected, plateau detection fires.

## 2026-07-03 (Claude Code / Fable 5)
- **Media stored as genome-CODED latents, never raw files (copyright /
  no base64 tax; v1.3.0, pushed live + migrated).** User: video must be
  stored as latents, not a playable file. Measured first: the text
  arithmetic coder is the WRONG tool for media — on 1MB of video-like
  data it took 27s to encode / 25s to decode and EXPANDED it 41% (dead
  end). And the old base64-in-JSON storage inflated every file 33% (the
  live 3.79MB video was 5.06MB stored/served).
  - Fix: media is now `media_code()` = **AES-CTR keyed from the genome +
    content id**. Stored/served bytes are ciphertext — a coded bitstream,
    unplayable as-is, decodable only by applying the genome-derived key
    (I2's copyright posture, same as pages). Storage split into metadata
    `<id>.json` + coded `<id>.bin`; `/api/i2/{image,video}/<id>/blob`
    serves the latent as octet-stream; the browser fetches + AES-decodes
    it (WebCrypto) into a blob URL right before playback; the child
    decodes to VERIFY (hash+sig) then caches/serves the coded latent.
    `migrate_media()` converts legacy base64 records to coded .bin at
    startup. Removes the 33% base64 tax (0% overhead now) as a bonus.
  - HONEST scope: this is coding/gating (decoder-keyed), NOT compression
    and NOT secret from genome-holders. TRUE compressive video latents =
    a neural media codec (the "media genome") = a real future build.
  - Verified: Python↔JS AES-CTR parity (JS decodes Python's latent
    exactly, size preserved); upload stores coded .bin at original size,
    no raw data in metadata; /blob is coded and decodes to the right
    hash; child fetch→decode→verify→cache→serve coded; legacy migration.
    Deployed + live 3.79MB video migrated (5.06MB→3.79MB, coded). Restart
    the child for the new media path; reload the browser.

## 2026-07-03 (Claude Code / Fable 5)
- **Wipes trickle down: primary purge → children clear matching cache.**
  A child serves cached media offline (never re-checks the primary), so
  after a primary wipe it would keep serving deleted "junk". Fixed: the
  primary records per-category wipe timestamps (`i2_store/purges.json`,
  set in `maint_wipe`) and publishes them in `/api/i2/info` (children
  already poll it ~10s). The child compares to its applied set
  (`<cache>/purges_applied.json`) and, for any category the primary wiped
  more recently, deletes that local cache dir (pages/images/videos only —
  the rest it proxies live, never caches). Selective: a videos-only wipe
  clears the child's video cache and leaves pages. Verified end-to-end:
  child cached a page + video, primary `wipe --only videos`, child polled
  and dropped videos (0) while keeping pages (1). Same `purges` field is
  the mechanism future SECONDARIES will honor too. Pushed live
  (info.purges confirmed).

## 2026-07-03 (Claude Code / Fable 5)
- **Maintenance TAB in the primary GUI console + child Cache tab shows
  media.** Follow-ups to the two requests below.
  - **Primary control panel** (`i2_node.py` GUI): new "Maintenance" tab —
    category checkboxes with live item counts/sizes, "Back up ticked",
    "Wipe ticked" (confirm dialog; auto-backup first), a one-click "Reset
    — keep only pages", and a Backups list with "Restore selected". Runs
    the maint_* functions IN-PROCESS (local operator, no signing needed;
    the signed CLI remains for remote use). Default selection is
    everything-except-pages.
  - **Child Cache tab** (`i2_child.py` GUI): it only listed cached PAGES,
    so videos/images you watched on Latent Stream never appeared even
    though they WERE cached. Now the tree shows type (page/video/image),
    name, size, detail, status for pages + cached videos + cached images;
    the overview line reports "N pages · N videos · N images". (The child
    must be on the media-caching build — restart it if watched videos
    still don't show.)

## 2026-07-03 (Claude Code / Fable 5)
- **Customizable store backups / wipes / restores from the primary
  (v1.3.0, pushed live).** Selective reset by category — e.g. drop your
  user, all posts/videos/likes/tokens but KEEP the sites.
  - Categories: pages, images, videos, identities, domains, likes,
    ledger, uptime. Genome + node keys are NOT categories (infrastructure,
    never touchable). Functions in i2_node: `maint_backup` (snapshot to
    `i2_store/_backups/<ts-label>/`), `maint_wipe` (ALWAYS auto-backs-up
    first → reversible), `maint_restore`, `maint_list_backups`,
    `maint_sizes`.
  - Admin endpoint `POST /api/i2/admin/maintenance` — gated by the same
    pinned admin key as push (Ed25519 sig over the canonical doc + fresh
    ts). New CLI `i2_admin.py`: `status`, `backup [--only ..] [--label]`,
    `wipe --keep pages --yes` (reset but keep sites) / `wipe --only
    videos,likes --yes`, `restore <backup> [--only ..]`. Default target
    10.0.0.15:8800; wipe needs `--yes`.
  - Verified end-to-end: seeded identity+video+like, `wipe --keep pages`
    kept all 7 pages while dropping the rest, `restore` brought them back,
    `wipe --only videos` dropped just videos and kept the identity;
    unsigned and wrong-key wipes both 403 with pages untouched. Pushed
    live; `status` reports the real store.

## 2026-07-03 (Claude Code / Fable 5)
- **Templates/theming + Woven (social feed) + likes (v1.3.0, pushed
  live).** Cosmetic overhaul so the browser is skinnable, not one fixed
  "AI-made" look.
  - **Theme engine**: design tokens (colors, font, radius, layout,
    feature flags) live in `THEME`, mirrored to CSS variables (`--i2-*`)
    for DOM and read directly by the canvas. 6 built-in TEMPLATES
    (Midnight, Daylight, and Woven skins Gram/Reels/Tumble/Vine). Applied
    GLOBALLY or PER-SITE, persisted in localStorage; `navigate()` picks
    the site's template. Canvas chrome + pages refactored to read THEME
    so a template re-skins the whole browser.
  - **Appearance page** (`i2://appearance` / `themes`): swatch grid of
    templates, apply Everywhere or to one site, live accent color picker.
  - **Woven** (`i2://woven`): a social feed over the network's existing
    images + videos as posts, with likes. Templates swap the LAYOUT —
    Gram (grid), Reels (vertical full-bleed, videos autoplay/loop/muted),
    Tumble/Feed (cards) — and toggle features: the Vine skin HIDES the
    like count while the network still keeps it (the requested trick).
  - **Likes backend** (`i2_service`, `i2_store/likes/`): a signed thumbs-
    up on any content id, one per identity, verified + counted; routes on
    primary + child (`/api/i2/like`, `/api/i2/likes/<id>`).
  - **Also fixed a real regression the harness caught**: the address bar
    blurred the field (which re-syncs it to the current URL) BEFORE
    reading the typed value, so Enter navigated to the current page.
    Now reads the value first — address-bar navigation works again.
  - Verified: likes backend 7/7; theme UI shim (global re-skin, per-site
    override wins on Woven, Woven renders posts, Vine hides counts,
    Appearance lists templates, accent persists); prior UI harnesses
    (input/search/LS) green. Home quick links now include Woven +
    Appearance. Pushed live (restarted, likes route confirmed). Restart
    the child for the like proxy routes.

## 2026-07-03 (Claude Code / Fable 5)
- **Welcome page shows the genome's current wire size.** Added
  `GENOME_WIRE_BYTES`/`GENOME_SIZE_STR` (computed from the actual genome)
  and a `{size}` placeholder on the home page: "the genome is 633 KB on
  the wire — downloaded once, then every page decodes locally." Computed
  dynamically so it stays accurate if the genome changes. Republished
  live to the primary (verified: decodes, signature valid, shows 633 KB);
  source synced (--no-restart) + dist.

## 2026-07-03 (Claude Code / Fable 5)
- **Search TLC: typeable home box + directory-grouped results page
  (browser-only, pushed live).** Two fixes to the reported dead search:
  - The home search box wasn't typeable — it was a canvas drawing that
    just redirected focus to the address bar (felt broken). Replaced with
    a REAL `<input>` (`homeSearchInput`) laid over the hero, visible only
    on home, positioned each frame via `homeSearchRect`. Click and type
    directly; Enter → results page.
  - Results now render as a landing page GROUPED BY DIRECTORY: sections
    for ▶ Latent Stream, 🖼 Gallery, People, Domains, each `domain/`, and
    Pages (commons); section headers are clickable (jump to that
    directory) and show counts; groups ordered by best hit. Header reads
    "N results across M directories". Replaces the old flat list.
  - Verified in the DOM shim: real home input exists, is visible on home
    / hidden elsewhere, typing+Enter searches; results group into the
    correct directory sections with a directory count.

## 2026-07-03 (Claude Code / Fable 5)
- **Welcome page (`i2://home`) rewritten + documents offline caching.**
  The home page was pre-identity/stream/gallery. Now it: keeps the
  compression pitch, adds an "On the network" section linking Latent
  Stream / Gallery / DMV, and features a **"Works offline"** section
  documenting that a node keeps verified copies of what you open (pages,
  images, whole videos), replays instantly, survives losing the primary,
  and re-verifies every cached copy — with the honest note that serving
  copies is a separate opt-in and bulk redundancy is the (undeployed)
  secondary tier. Updated the canonical `_SEED_PAGES["home"]` in
  i2_service.py AND republished the page LIVE to the primary via
  `/api/i2/publish` (seed only auto-applies on a fresh store). Verified:
  decodes cleanly, signature verifies, contains the new sections. Source
  synced to the primary (--no-restart) and dist.

## 2026-07-03 (Claude Code / Fable 5)
- **Child caches images + videos (offline resilience) — child v1.3.0.**
  Children cached PAGES but proxied images/videos straight through, so a
  dropped connection meant no media and every read hit the primary. Fixed:
  `ingest_media(kind, id, verify)` is cache-first — a cached hit is instant
  and works offline; a miss fetches from the primary, verifies (hash +
  uploader signature), writes it under `<cache>/images|videos/<id>.json`,
  then serves it. The list routes (`/api/i2/images`, `/api/i2/videos`)
  fall back to the locally cached list when the primary is unreachable.
  Stats gained `cached_images`/`cached_videos`. This is the CONSUMER role
  (a child keeps copies of what IT reads); actually SERVING those copies
  to peers is a separate OPT-IN (consumer → contributor), and bulk
  high-usage replication is the SECONDARY tier's job — neither built yet.
  - Verified end-to-end: fetch via child caches it, second fetch is a
    cache hit, and with the PRIMARY KILLED the child still serves the full
    cached video AND lists cached videos offline.
  - Child-only change (caching is client-side) — no primary push. Restart
    the child on this machine to pick it up.

## 2026-07-03 (Claude Code / Fable 5)
- **Latent Stream — the network's YouTube (v1.3.0, pushed live).** Video
  site at `i2://latent-stream` (aliases: `ls`, `latent stream`, and the
  earlier `latent space` all resolve).
  - **Backend** (`i2_service.py`, `i2_store/videos/`): videos are signed
    blobs credited to the uploader (sign over sha256(bytes)+title+mime),
    same trust model as images; NOT genome-compressed. mp4/webm/ogg,
    32MB cap; title+description indexed by search. Routes on primary +
    child (child verifies hash+sig before serving). Flask
    MAX_CONTENT_LENGTH raised to 64MB for the base64 payloads.
  - **UI**: a YouTube-style app view — sticky header (▶ Latent Stream
    logo, its own search bar, ↑ Upload, ← Exit), a responsive video
    grid, a watch page (player + title/uploader/description + "Up next"),
    and an upload form (hash+sign in-browser, chunked base64). Video
    search results (address bar or LS search) open the watch page.
    Guarded against an async render race (grid load vs. watch) with a
    generation token.
  - **Search fix** (the "search is inoperable" report): the address bar
    treated a bare word like `sunset` as a page address → 404. Now
    `navigateSmart` sends anything that isn't a path / @handle / known
    page name to SEARCH; paths and known names still navigate directly.
    Multi-word always searches.
  - Verified: video upload accepted+credited, bad-sig rejected, listed
    (no data blob), fetched with data, and search finds videos by title,
    description, and uploader — direct AND through the child proxy.
    UI shim: LS opens via all aliases, grid lists videos, watch plays +
    shows "verified ✓", bare-word address-bar input searches. Pushed
    v1.3.0 live (restarted, videos route confirmed).
  - **Restart the child** on this machine for the video proxy routes.

## 2026-07-03 (Claude Code / Fable 5)
- **Fix dead address/search bar + gallery lightbox (browser-only, pushed
  live).** User: typing an address or a search did nothing. Cause: the
  URL/search bars were floating `<input>`s created on click and laid over
  the canvas — fragile focus/blur behavior in a real browser. Replaced
  with a PERSISTENT real DOM `<input>` (`addressBar`) always mounted over
  the nav area: type an address (e.g. `directory/sites`) to go, or a
  multi-word query to search (auto-detected on whitespace). Kept in sync
  with the current URL; red border on error. Removed the floating
  `openUrlInput`/`openSearchInput`; the home hero search bar now focuses
  the address bar. Verified via the DOM/canvas shim: address navigates,
  multi-word searches (hits `/api/i2/search`), bar reflects current page.
- **Gallery lightbox**: clicking any thumbnail opens the image large in a
  full-screen overlay (caption + click-to-close). Thumbnails show a
  zoom-in cursor.

## 2026-07-03 (Claude Code / Fable 5)
- **Browser UX: chrome pages instead of modals + a real search page
  (browser-only, pushed live).** Per user: stop piling icons on the
  toolbar and make each feature a navigable PAGE.
  - **Removed the ✎/👤/🔍/🖼 toolbar icons.** Toolbar is now just
    ← → ⟳ + the address bar.
  - **DMV, Gallery, Composer, and Search are now navigable pages**, not
    modal overlays: reachable via `i2://dmv`, `i2://gallery`,
    `i2://compose`, or search. New router in `navigate()` treats these as
    built-in "chrome pages" — local interactive DOM views mounted below
    the chrome (`APP_ROOTS`), fully in the history so ← → work; each has a
    "← Back" too. The old `toggle*` overlays (dark backdrop, click-out)
    are gone.
  - **Search results are a proper page** (Google-style): a results view
    with its own search box, `N result(s) for "q"`, and ranked cards
    (address · type, title, snippet) that navigate on click. Reached from
    the home hero search bar, from `i2://search`, or by typing a
    multi-word query in the address bar (address vs. search is auto-
    detected on whitespace). Replaces the old synthetic-canvas results.
  - **Home hero** gained quick links (Directory · Gallery · New page ·
    Your ID) under the search bar, replacing the removed icons for
    discoverability.
  - Verified with a Node DOM/canvas shim loading the real i2.js: clean
    load, each app view shows as exactly ONE full page (not stacked
    modals), search renders the count + ranked results, URL bar reflects
    the active page. Pushed static/i2.js live (no restart). Reload the
    browser to get it.

## 2026-07-03 (Claude Code / Fable 5)
- **Fix "uploader is not a registered identity" on image upload
  (browser-only, pushed live).** Root cause: the live primary had ZERO
  registered identities — the browser's localStorage identity was
  registered against another primary / restored from a backup / lost to
  a store reset, so its DID was unknown to the current primary and every
  signed op ("not a registered identity") failed. Fix: `ensureRegistered()`
  in i2.js re-asserts the self-signed identity doc to the current primary
  (idempotent for the same DID; only errors on a real handle-vs-different-
  DID conflict). Called before image upload, on restore-from-backup, and
  silently when the DMV opens (so claim/renew/transfer/earn self-heal
  too). Verified: reproduced the exact 400, then re-register → upload
  succeeds. Pushed static/i2.js live (no restart; static serves fresh).
  Users just reload the browser.

## 2026-07-03 (Claude Code / Fable 5)
- **Search + Gallery (v1.2.0, pushed live to 10.0.0.15).** "Google for
  the network," plus signed image uploads to search over.
  - **Search** (`i2_service.search`, `GET /api/i2/search?q=`): the
    primary DECODES every page with the genome and keyword-ranks over
    page title+body, image captions, handles, and domains; returns
    ranked results with snippets. Linear scan (fine at this scale);
    honest keyword search, NOT semantic embeddings yet (that is the
    primary's future intelligence layer). Title hits weighted 8×, image
    caption hits +4.
  - **Gallery / images** (`i2_store/images/`): images are SIGNED BLOBS
    credited to the uploader's identity — the uploader signs
    (sha256(bytes), title, mime) with their DID; the primary verifies
    before storing and rejects tampered uploads. png/jpeg/gif/webp, 5MB
    cap. NOT genome-compressed (text codec only; a media-genome family is
    future work, stated openly). Routes: `/api/i2/image/upload`,
    `/api/i2/images`, `/api/i2/image/<id>`. The CHILD verifies every
    fetched image locally (hash + signature) before serving it, same
    trust model as pages.
  - **Browser**: new 🔍 (search) and 🖼 (gallery) chrome buttons; the
    home hero search bar is now live (was decorative); results render as
    a synthetic in-canvas page with clickable hits (page/domain →
    navigate, image → open gallery). Gallery overlay uploads (hash+sign
    in-browser, chunked base64 for MB files) and shows a verified
    thumbnail grid credited "by @handle". Typing `gallery` in the URL bar
    opens it.
  - **Verified 9/9** (exact browser/child path): signed upload accepted +
    credited, tampered upload rejected, image listed + full record
    fetchable, search finds the image by caption, finds a text page by
    decoded content, finds the identity, empty on no-match, ranked; and
    through the CHILD proxy: search, image list, and hash+sig-verified
    image fetch all work. Pushed v1.2.0 live (restarted, search/images
    routes confirmed).
  - **Restart the child** on this machine to get the search/gallery proxy
    routes (browser buttons appear immediately since static served fresh,
    but their API calls need the child's new routes).

## 2026-07-03 (Claude Code / Fable 5)
- **Node persistence + uptime correctness fixes (v1.1.1, pushed live to
  10.0.0.15).** Fixes "re-create the node every restart" and "uptime not
  accruing," and makes the child auto-reconnect without re-scanning.
  - **Root-cause bug**: a plain GUI launch passed no dir hint, and
    `run_gui` only loaded config when given a hint — so it re-ran the
    SETUP WIZARD every launch, losing the primary URL *and* the linked
    identity (which silently stopped uptime earning). Fixed in BOTH
    nodes: a stable per-user home pointer (`~/.i2/{child,primary}_home`,
    written by `save_config`) plus a load order of explicit hint →
    remembered home → default dir. A plain launch now reopens the last
    node straight to the dashboard; the wizard only appears on a true
    first run. Headless `--server`/`--status` use the same recall.
    Verified: after save, a fresh no-hint load reopened the node with
    primary_url AND identity link both intact.
  - **Auto-reconnect**: since config now persists, the child reopens with
    its primary_url and its monitor loop reconnects on its own (and keeps
    retrying if the primary is briefly down) — no rescan needed.
  - **Uptime calculated correctly across two machines**: heartbeat
    freshness window widened 120s → 300s. It only ever guarded against
    CLOCK SKEW between child and primary (the real anti-fast-forward
    guard is the server-measured 30s/beat credit cap), so a narrow window
    was silently rejecting every heartbeat as "stale" whenever the two
    machines' clocks differed by >2 min → zero uptime. Child now logs the
    primary's rejection reason so skew is diagnosable.
  - Child GUI overview shows the linked identity + earned tokens/hours
    (or a prompt to link). Both NODE_VERSIONs → 1.1.1.
  - **Deploy**: primary pushed live (restarted 1.1.1, 7 pages intact).
    The CHILD on this machine must be restarted once to pick up the new
    code; after that it stops re-running the wizard. If the child used a
    CUSTOM cache dir before, it shows the wizard one last time (no
    pointer yet), then remembers it forever.

## 2026-07-03 (Claude Code / Fable 5)
- **Uptime token minting — the real faucet (v1.1.0, pushed live to
  10.0.0.15).** Tokens are now EARNED by keeping a node online, replacing
  the sybil-farmable instant grant as the main income.
  - **Signed heartbeats**: a child sends `POST /api/i2/node/heartbeat`
    {did, ts, sig} each poll (~10s); the primary verifies the sig against
    the DID's registered key, requires a fresh ts (±120s, anti-replay),
    rejects out-of-order beats, and credits SERVER-measured elapsed time
    since the last beat, capped at 30s/beat so a client can't claim time
    it wasn't online. Mints 1 whole token per accrued hour
    (`i2_service.py`, `i2_store/uptime.json`; `_lock` → RLock so minting
    can call grant_tokens while held).
  - **Starter grant cut 30 → 20** (exactly one domain): first homestead
    free, every domain/renewal after is earned by uptime — implements the
    "one domain per fresh DID, more earned" anti-squat design directly.
  - **Child**: `/api/i2/child/link-identity` lets the browser hand the
    local node the user's identity (doc + pkcs8 priv, same-origin, own
    machine) so it earns to their wallet; unlink endpoint too; heartbeat
    sender in the poll loop; identity + mint status in child stats.
  - **Browser DMV land office**: "Earn on this node" button (links the
    local child; on a primary it explains primaries mint, don't earn),
    wallet line now shows hours-online-earned; ledger endpoint returns
    uptime.
  - **Verified** (exact browser/child code path): starter grant = 20,
    first beat credits 0, out-of-order/stale/bad-sig/unregistered beats
    all rejected, deterministic mint (seeded pending → one beat mints
    exactly 1 token, credit capped ≤30s, balance +1), and a REAL linked
    child accrued 10s of verified uptime over live heartbeats. Then
    pushed v1.1.0 to the live primary via push_to_primary.py — it
    restarted on 1.1.0 with the heartbeat route live and all 7 pages
    intact. Honest caveat: still farmable by running many real nodes for
    real hours (time is the cost now, not a free grant); reputation
    weighting / proof-of-storage are the next hardening.

## 2026-07-03 (Claude Code / Fable 5)
- **Signed remote push — deploy to the primary without hand-copying.**
  New admin-authenticated `POST /api/i2/admin/push` on the primary
  (`i2_node.py`): accepts an Ed25519-signed file bundle from ONE pinned
  admin key (`i2_admin.pub`, ships with the primary), verifies the
  manifest signature + per-file sha256 + a strict path whitelist
  (i2 source/static/templates only), backs up replaced files to
  `_backup/push-<ts>/`, writes them, and (default) restarts. No key file
  → endpoint disabled. `push_to_primary.py` on the dev machine holds the
  PRIVATE key (`i2_admin_key.json`, gitignored) and does the packaging/
  signing/POST — `python push_to_primary.py [--primary URL] [files…]
  [--no-restart]`, default target 10.0.0.15:8800.
  - **Fixed a real restart bug**: `os.execv` on Windows leaves the old
    listening socket as a stale handle (port shows LISTEN, nothing
    serves). Both nodes' `relaunch()` now spawn a DETACHED child (no
    inherited fds) + `os._exit(0)`; the bind-retry loop covers the
    overlap. Verified: pushed to a live primary, it restarted on a NEW
    pid still serving, admin key intact.
  - **Security verified**: unsigned push → 403, wrong-key signature →
    403, `../` path traversal (even correctly signed) → 403. NODE_VERSION
    bumped 1.0.0 → 1.0.1; `/api/i2/admin/version` reports it.
  - Dogfooded: pushed the full source set to the local primary, it
    restarted running the new code, then the transfer suite passed
    against it.
- **Domain transfers** (completes the economy): signed
  `domain-transfer` op (`i2domain\0` family) moves a deed to another
  registered identity, owner-initiated only — the registrar never moves
  it. Route on primary + child proxy + per-domain "Transfer" button in
  the DMV land office. Old owner's pages stop resolving under the domain
  (new owner controls the namespace) but live on at their @handle alias,
  same as a reclaim. Verified 6/6: non-holder rejected, unregistered
  recipient rejected, owner transfer accepted, replay rejected, old
  owner can't renew after, deed shows recipient.
- **Push note for deployment**: the CURRENT fileshare primary predates
  the /admin/push endpoint, so this ONE update is still manual (copy
  dist/i2_primary — now incl. i2_admin.pub — to the share + restart).
  Every push after that: `python push_to_primary.py`.

## 2026-07-03 (Claude Code / Fable 5)
- **I2 DOMAINS + TOKEN ECONOMY + AUTO-DIRECTORY** (the navigation design
  session, built): two-layer namespace — permanent free @handle aliases
  underneath, token-fed domains on top. All 20 e2e checks pass.
  - **Token ledger** (`i2_service.py`, `i2_store/ledger.json`): mint =
    30-token starter grant per new identity (the bootstrap faucet;
    uptime-minting later — sybil-farmable until then, STATED). Signed
    transfers between DIDs (`i2token\0` canonical docs, replay-protected
    via applied-sig hashes). Sink = domains: claim 20, renew 5/30 days.
  - **Domain registry** (`i2_store/domains/`): deeds claimed/renewed by
    signed ops (`i2domain\0`); states active → grace (14d) → lapsed →
    back to the pool. Reserved: home/dmv/publishing/manifesto/network/
    directory/index/commons/search + Windows device names.
  - **Pages under domains are AUTHOR-SIGNED**: multi-segment names
    (`arcade/games`, ≤4 segments, nested storage) require the deed
    holder's own Ed25519 signature over the standard page message; the
    node cannot publish into your domain. Single-segment names stay the
    open commons (primary-signed, unchanged). `resolve_page()`:
    `@handle/name` alias serves any page authored by that handle's DID
    (content outlives addresses); a reclaimed domain 404s the previous
    owner's pages under the domain address (alias keeps working); grace/
    lapsed states surface in the page footer with a renew warning.
  - **Auto-directory**: primary regenerates `directory`,
    `directory/sites`, `/people`, `/new` (ordinary signed pages) on
    start and after every publish/claim/registration. Curated `index`
    reserved for hand-curation (content to come).
  - **Browser**: typeable URL bar (click bar → input → Enter),
    path+alias link targets, composer accepts `domain/page` and
    author-signs automatically when you hold the deed, DMV card gains
    wallet + land office (balance, claim, renew, send tokens to a
    handle). New routes on primary; child proxies ledger/domain calls
    and caches nested page names safely.
  - **Verified** (exact browser code path, Node webcrypto): starter
    grant, claim + burn, reserved/held-domain/insufficient-funds
    rejections, author-signed publish, unsigned + wrong-author
    rejections, domain fetch + @alias fetch + wrong-alias 404, transfer
    + replay rejection, renew, deed aged → grace → lapsed (page still
    served, flagged) → reclaimed by new DID (old page 404s under domain,
    lives at alias), directory generation, child nested fetch/verify/
    cache + proxies. Test artifacts cleaned; dist/ refreshed. Restart
    both nodes to serve all of it. NOT built yet (designed): uptime
    minting, domain transfer, subdomain delegation, curated index
    content, visit-based feeding.

## 2026-07-03 (Claude Code / Fable 5)
- **Restart button on both node GUIs** (`i2_node.py`, `i2_child.py`):
  top-bar "Restart" with confirm → existing `relaunch()` (os.execv), so a
  node picks up new files on disk without touching a terminal. Server
  startup in both scripts now retries binding (20 × 0.5s) because after
  execv the new process can race the old one for the port — without this
  a restart could come back dead. Verified headless: instance B started
  while A held :8800, A killed, B was serving within ~3s. GUI button
  itself needs a display (not exercised headless). dist/ refreshed.

## 2026-07-03 (Claude Code / Fable 5)
- **Browser: readable error when the node runs older software.** User hit
  "unexpected token '<'" submitting the DMV form — the running child
  (8810) and fileshare primary both predate the identity routes, so the
  POST got an HTML 404 and JSON.parse choked. New `jsonOrExplain()` in
  i2.js wraps all POST/fetch response parsing (DMV register, directory,
  composer publish): a non-JSON reply now reads "node returned 404
  without JSON — its software is older than this page; restart/update
  the node". dist/ refreshed. The actual fix remains restarting both
  nodes with current files.

## 2026-07-03 (Claude Code / Fable 5)
- **DMV handles: broader charset** (user request; length was already 64).
  Handles get their own rule, decoupled from page names:
  `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` — capitals, dots, underscores now
  allowed. Uniqueness is CASE-INSENSITIVE (registry file is the
  lowercased handle; typed casing preserved on the card) so re-casing
  someone's handle can't impersonate them — takeover test passes.
  Windows-reserved basenames (con, nul, com1…) rejected. Updated the DMV
  form validation/placeholder in i2.js. Verified 7/7 against a local
  primary (mixed-case register, case-variant takeover rejected, reserved
  names rejected, space/leading-dot rejected, case-insensitive lookup
  preserves casing). dist/ refreshed; same restart notes as below apply.

## 2026-07-03 (Claude Code / Fable 5)
- **I2 DMV — self-sovereign digital identity ("the DMV of the
  internet").** New 👤 button in the browser chrome (green once
  registered) opens the DMV overlay: generate an Ed25519 keypair IN THE
  BROWSER (WebCrypto), DID = the public key (`did:i2:<b64url>`), and a
  self-signed identity document — handle (`[a-z0-9-]`, page-name rules),
  display name, emails (reserved names; mail comes later), and service
  aliases/IGNs (`service: name`). Private key stays in localStorage;
  Download-backup / restore-from-backup / forget-on-this-device / a
  network directory view, and update-details re-signs with the same key.
  - `i2_service.py`: identity registry (`i2_store/identities/`) —
    `register_identity` verifies the doc against its OWN DID (canonical
    signing bytes: `"i2id\0"` + sorted-key compact utf-8 JSON minus sig,
    byte-identical to `stableStringify()` in i2.js — parity tested),
    enforces first-come-first-served handles (update requires same DID),
    caps field sizes, rejects unknown fields. The registrar CANNOT
    recover/reassign/revoke — stated in the UI ("no reset desk").
  - Routes: primary `POST /api/i2/identity/register`,
    `GET /api/i2/identity/<handle>`, `GET /api/i2/identities`; child
    proxies all three to its primary (shared `_forward_json` helper).
  - Verified end-to-end with the exact browser code path (Node webcrypto
    + identical stableStringify) against a local primary AND through a
    child proxy — 7/7 both ways: register, takeover-by-different-key
    rejected, same-key update accepted, tampered doc rejected, bad
    handle rejected, lookup, directory. Canonical-JSON parity confirmed
    byte-identical incl. non-ASCII. Test identities/nodes cleaned up.
  - Published `i2://dmv` (explainer page) to the live fileshare primary;
    dist/ refreshed. NOTE: primary + child processes need a restart to
    serve the identity routes (also still pending from the composer:
    the user's child on 8810 and the fileshare primary's static/i2.js).
  - Groundwork stated honestly: pages are still signed by the primary at
    publish; wiring identity keys into page authorship ("published-by
    <handle>") is the next roadmap item, now unblocked.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 Composer — the browser can now make pages** ("a page that makes
  pages"). New ✎ button in the canvas browser chrome (`static/i2.js`)
  opens a DOM overlay editor: page name (`[a-z0-9-]{1,64}`), title, body
  in the existing page markup (`# ` heading, `* ` bullet,
  `[[page|Label]]` link), an "Edit current page" prefill (decoded text
  is now kept on `BROWSER.page.raw`; republish = overwrite), and
  Publish + sign → POST `/api/i2/publish` on the same-origin node, then
  the browser navigates to the new page and decode-verifies it like any
  other. New child endpoint (`i2_child.py`): `/api/i2/publish` forwards
  the draft to the primary (children don't sign yet — per-author keys
  remain roadmap; the primary encodes + signs, stated in the UI), passes
  primary rejections through, and busts the child's cached copy of a
  republished name. Verified live against the fileshare primary
  (10.0.0.15): a test child forwarded a publish (a real "How to publish
  on I2" page, now at i2://publishing), read it back
  fetched+verify_ok=1, and both the primary and the user's running
  child list it. dist/ copies refreshed. NOTE: the user's running child
  needs a restart to gain the forwarding endpoint (browser JS is live
  already); the primary on the fileshare needs its static/i2.js updated
  (self-update or recopy) to show the ✎ button.

## 2026-07-02 (Claude Code / Fable 5)
- **MILESTONE: first two-machine I2 cell verified live.** Primary running
  on the user's fileshare machine (10.0.0.15:8800), child on this machine
  (127.0.0.1:8810). Initial blocker was Windows Firewall on the primary
  host (TCP 8800 filtered; ping fine) — fixed by the user via an inbound
  rule/policy; bind stays 0.0.0.0 (correct; no per-IP bind needed).
  Verified over the network: child connected (37 ms), genome g3-71955590
  synced, uncached page fetched from primary → decoded → sha256 →
  Ed25519 verified → cached (verify_ok=1, verify_fail=0), and a repeat
  read served from local cache without touching the primary. No code
  changes; deployment/verification only.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 Child Node (`i2_child.py`) — consumer client + primary discovery.**
  (Script was written just before a session crash; this entry records it
  plus the new discovery feature.) A child connects to a PRIMARY,
  fetches/auto-updates the genome, and serves the I2 browser locally
  (default port 8810) as a verifying cache: every page fetched from the
  primary is verified locally (decode → sha256 → Ed25519 signature via
  `i2_service`) before caching; failed pages are REJECTED and never
  cached. Tracks self-measured connection reputation (hours online,
  uptime ratio, disconnects — honest placeholder until the network
  aggregates it). Tk GUI (setup wizard → Overview / Connection-Reputation
  / Cache / Update / Log tabs), `--server` headless, `--status` JSON.
  Same self-update mechanism as the primary. Genome auto-refetches when
  the primary announces a new version.
- **NEW: LAN primary discovery.** `discover_primaries()` sweeps localhost
  + every local /24 subnet on port 8800 (64 threads, 0.4s connect probe,
  then confirms via `/api/i2/info` role/version fields), sorted primaries
  first by latency. Wired into (a) a "Scan network for primaries" button
  in the setup wizard — results listed, click to fill the URL field;
  manual URL entry still works — and (b) a `--scan` CLI flag. Verified
  live: a headless primary on this machine was discovered on all
  interfaces (10.0.0.21, 172.29.144.1, 127.0.0.1) in seconds; scan with
  no primary running correctly reports none.
- `package_i2.py` child README documents scan; `dist/i2_child/` rebuilt
  (in place — a process held the folder lock so rmtree was skipped).

## 2026-07-02 (Claude Code / Fable 5)
- **I2 Primary Node app (`i2_node.py`) — deployable node + operator
  console.** Single script to drop on a file share and run; Tk GUI with
  first-run setup wizard (node name, data dir, port, update source),
  then a live dashboard: Overview, Genome (version/mixer/row counts),
  Storage/Pages (per-page size, ratio, origin, sig), Content Map (PCA-2D
  of per-page byte histograms on a Tk canvas — the "watch the latent
  space evolve" view; honest label since gen-2 content isn't
  latent-vector-based), Network (bind, peers=0/first cell, request
  counters, recent activity), Publish (encode+sign through the GUI), and
  Update. Runs an instrumented Flask node in a background thread (waitress
  if installed, else werkzeug), serving `/`, the genome, pages, publish,
  and `/api/i2/node/stats`. Serves a self-contained browser via new
  `templates/i2_node_browser.html` (canvas-only, no terminal deps).
  **Self-update**: reads `manifest.json` at a configurable update source,
  semver-compares to `NODE_VERSION`, backs up replaced files to
  `_backup/<ts>/`, copies new ones, offers restart (`os.execv`).
  `i2_service.py` now honors `I2_DATA_DIR`/`I2_CORPUS` env vars so storage
  points at the deployment share (defaults unchanged for in-repo use).
  Modes: GUI (default), `--server` (headless), `--status` (JSON dump).
  Verified headless end-to-end under `simple`: server binds, live HTTP
  genome/page/publish, content-map PCA, node-stats endpoint, and
  update check+apply all pass. **Roles (primary/secondary/child) and
  primary authority (LLM/observation, chain integrity) are DECLARED in
  config but NOT yet enforced — first primary runs unrestricted, stated
  openly.** GUI itself needs a display (not exercised headless).
- **Findings PDF updated to experiments 1–9** (9 pages): new Section 11
  (zstd-dictionary validation = the decisive result; mixer-evolution
  negative; gen-2 ship + signature/parity guarantees) and Section 12
  (node tiers + federation trust model, stated transparently: primaries
  hold real bootstrap authority, hierarchy is a reputation growth path).
  Executive summary now lists gen-2 as shipped; roadmap marked a–e done.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 GEN-2 SHIPPED + Ed25519 origin signatures** (user roadmap items
  1–4 executed in validation-first order). **Requires a Flask restart.**
  - **exp8 (mandatory validation, run BEFORE shipping claims): zstd-19
    with a 512KB dictionary trained on the same corpus = 2.86× on 1KB
    pages — a DEAD HEAT with gen-1 (2.85×) — and 2.291 bits/byte on 1MB.**
    Gen-1 could not have claimed state-of-the-art; gen-2 was required.
    (brotli python binding has no custom-dictionary support; noted.)
  - **exp9 (mixer evolution): per-confidence-bucket K evolution gave NO
    held-out gain** (2.223 evolved vs 2.216 seed) — the flat Witten-Bell
    chain (K3=16, K2=2, K1=0.5) is already near-optimal; reported
    honestly. But the exp9 pipeline itself (MLE + full backoff chain on
    u16-scaled tables) reached **2.216 continuous / ~2.24 through the
    real quantized coder path (3.57×) — beats zstd+dict in both regimes**
    (whole-buffer margin is real but narrow: 2.24 vs 2.291).
  - **Gen-2 codec** (`i2_service.py` + `static/i2_codec.js`): genome
    `g3-<hash8>`, format `mix3-ac` — order-1/2/3 u16-row-scaled count
    tables (648KB zlib wire) + flat-K backoff mixer, 3-byte rolling
    context, per-context mixed distribution quantized to integer
    frequencies (floor(p·16384+0.5), min 1, total < 2^17 → exact in JS
    doubles; flat K's chosen partly because bucketed K needs log2, a
    cross-platform float-parity risk). Same WNC coder. g0/g2 pages
    auto-migrate (legacy decoders kept only for migration).
  - **Ed25519 signatures (Day-1 requirement per user):** node keypair in
    `i2_store/keys/` (auto-generated); every page signed at publish over
    `"i2page\0"+name+"\0"+genome+"\0"+sha256(content)`; `origin` is now
    a real public key; server verifies (`verify_page`), and the browser
    verifies INDEPENDENTLY via WebCrypto Ed25519 after local decode —
    footer shows ✓ origin (verified) / ✗ (INVALID) / unverifiable.
    Tamper test passes (modified doc correctly rejected). `cryptography`
    pip-installed into `simple`.
  - Verified: Python round-trip incl. 100KB torture (2.33 bits/byte,
    3.44×); **node parity: JS decodes all pages sha-exact (100KB in
    88ms) and all signatures verify with the browser's exact message
    format**; tamper rejection confirmed.

## 2026-07-02 (Claude Code / Fable 5)
- **New report: `documentation/I2_COMPRESSION_GENOME_FINDINGS.pdf`** (user
  request: "highly detailed, but honest") — 7 pages covering experiments
  1–7: methodology + held-out protocol, contamination check, real-coder
  verification, the falsified continuous-latent codec (with garbage-decode
  sample), full classical baseline tables (whole-buffer AND small-page
  regimes), out-of-domain degradation, the tree-LM codec failure and its
  accuracy-vs-calibration diagnosis, the bits/byte evolution runs (incl.
  gradient ceiling + gate-collapse negative results), the order-3 gen-2
  proposal, what actually shipped in gen-1, and an explicit
  threats-to-validity section (untested zstd-dictionary rival, domain
  boundedness, tree corpus-overlap disclosure). Generated with reportlab
  (pip-installed into `simple` along with pypdf for verification). Visible
  on the /docs page immediately, no restart needed.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 gen-2 experiments: evolved against bits/byte (exp5/6/7, scratchpad;
  no product code changed).** Verdict: log-loss fitness WORKS as a
  mechanism, but the neural organism family loses to counting; the real
  gen-2 candidate is order-3 + backoff.
  - exp5 (softmax head on frozen best encoder 538711, ridge-seeded,
    GA pop 64 on log-loss): 5.27 → **3.85 bits/byte held-out** in 7s of
    GPU evolution; top-1 acc only 0.334 vs 0.360 — evolution happily
    trades accuracy for calibration, exactly as intended. But the
    gradient-trained CEILING of that readout class is 3.65: the
    accuracy-bred encoder features don't carry distributional info, so
    even a perfect head can't approach order-2 counts (2.78).
  - exp6 joint organism (encoder+head, 272k params, 500 gens): 4.27 —
    worse than head-only (mutation dilution across the big genome). An
    evolved CONTEXTUAL gate mixing model vs order-2 collapsed to
    g≈0.001 ("always counts"), held 2.777 ≈ order-2 alone: the neural
    model adds nothing anywhere, not even in pockets.
  - exp7 count-family ladder: **order-3 + confidence backoff
    (lam=n/(n+16), fitted on train) = 2.27 bits/byte = 3.52× held-out —
    beats brotli-11 (3.45×) even on whole 1MB buffers**, still count-based
    (same codec contract as gen-1, trivially parity-able), genome ≈0.8MB
    raw sparse (~231k distinct 4-grams / 45k order-3 contexts in corpus).
    Proposed gen-2: ship order-3+backoff; let evolution breed the MIXER
    (per-context gates over the count ladder, PAQ-style) where it can
    genuinely add value.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 gen-1 genome SHIPPED: order-2 model + arithmetic coding** (user:
  "ship the order-2 genome as gen-1"). **Requires a Flask restart.**
  - `i2_service.py` rewritten: genome = quantized trigram frequency table
    built from the full EEC corpus (row totals capped < 2^17 so all coder
    arithmetic is exact in JS doubles; every symbol freq ≥ 1; unseen
    contexts uniform), serialized sparse — **100KB on the wire** (zlib),
    version `g2-<hash8>`, cached in `i2_store/genome/order2_sparse.bin`.
    The dense table is always reconstructed FROM the sparse wire bytes so
    server and browser provably use the same table. Pages are now
    Witten-Neal-Cleary arithmetic-coded bitstreams (`payload` field +
    `ratio`); publish verifies round-trip before storing. Gen-0 pages
    auto-migrate on import (legacy orthonormal decode kept only for that).
  - New `static/i2_codec.js`: DOM-free JS mirror of the coder
    (genome parse + AC decode), loaded before i2.js; `static/i2.js`
    decode path swapped (zlib inflate via DecompressionStream); page
    footer now also shows wire bits/byte.
  - Verified: Python round-trip incl. 100KB corpus torture (2.84
    bits/byte, matches exp3 prediction); **node parity test: JS decoder
    reproduces Python bytes sha-exact on all pages** (100KB decodes in
    15ms). Genome build from the 49MB corpus takes ~2s once, then cached.
  - Gen-2 path (from exp4 diagnosis): evolve the tree-LM against
    bits/byte (log-loss) instead of top-1 accuracy, then swap the genome.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 exp3+exp4: triple-checked the compression claim; wired the tree-LM
  to a codec and measured it** (scratchpad i2_exp3_verify / i2_exp4_treelm;
  results JSONs alongside; brotli+zstandard pip-installed into `simple`).
  - Verification of "order-2 beats zlib": **contamination 0/60** held-out
    probes found in train; a REAL arithmetic coder (Witten-Neal-Cleary,
    implemented + round-trip verified byte-exact) produced actual
    compressed bytes at **2.780 bits/byte (2.88×)** vs its 2.80 theoretical
    — the number is real, zlib-9 = 2.901.
  - BUT the full suite on 1MB buffers beats the order-2 model: bz2 3.84×,
    brotli-11 3.45×, lzma 3.41×, zstd-19 3.38× vs model 2.88×. "Beats
    zlib" ≠ "beats modern compression" at large-buffer scale.
  - **Small-page regime (1KB, I2's actual workload): the model wins
    decisively** — model 2.85× vs brotli 2.19×, zlib 1.75×, zstd 1.72×,
    lzma 1.45× — because the model's "dictionary" ships once in the
    genome instead of being rebuilt per page. (Strongest untested rival:
    zstd with a trained dictionary.)
  - Out-of-domain honesty: Gutenberg-trained genome gets only ~1.7× on
    GENREG markdown (loses to zlib) — the genome must evolve toward the
    network's real content distribution.
  - **Tree-LM as codec (the requested number): 4.31 bits/byte (1.86×)** —
    best model (0.3603 acc, c6df68) with full path-probability
    distribution (router+leaf softmax, calibration scalars tr=8 tl=16 +
    1% uniform floor fitted on train only). Far worse than order-2 counts
    (2.78), and a fitted linear blend chose λ=0 (tree adds nothing on top
    of order-2). Diagnosis: evolved for top-1 accuracy, not log-loss —
    codec-grade evolution needs bits/byte (perplexity) as the fitness
    function. That is the gen-2 experiment.

## 2026-07-02 (Claude Code / Fable 5)
- **Autonomous experiment session ended on user request** — E18 (pop-1000 /
  gens-800 encoder) was killed mid-encoder-evolution; its encoder-store
  entry (created after 19:25, if any) will show status "running" forever
  and has no encoder.npz — safe to delete. Flask server stopped (user
  starts it themselves from here on); terminal daemon left running. Final
  session state: **record +6.33 pts over bigram (acc 0.3603), seed-mean
  +5.37**, best encoder `20260702-184303-encoder-538711` (ridge 0.3754),
  leaf calibration in place, all mechanisms in the /tree UI with the
  winning recipe as defaults.

## 2026-07-02 (Claude Code / Fable 5)
- **E17: +6.33 pts — new record on every axis** (acc 0.3603 vs 0.2970, run
  20260702-192113-tree-c6df68): pop-600 encoder reached held-out ridge
  **0.3754** (encoder `20260702-184303-encoder-538711` — the new
  recommended pick in the sidebar Encoder select); trees on it
  +6.33/+5.30/+4.47, **mean +5.37** across seeds, ~15s each, and they're
  post-calibration (sampling works). Encoder budget curve still climbing →
  raised the `encoder_generations` clamp 500 → 2000 (parse_config; server
  restarted). E18 running: pop-1000 / gens-800 encoder + 3-seed trees.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 gen-1 genome experiments (scratchpad i2_exp1/i2_exp2): continuous
  latent codec FALSIFIED for text; predictive-model entropy coding is the
  viable gen-1.** No product code changed — results only.
  - Exp1 (linear codec, PCA closed-form seed + GA refine, pop 64 × 200
    gens, GPU, EEC corpus, 64-byte blocks, int8 latents): byte-EXACT
    reconstruction is ~2.5% at k=48 (1.33× lossy), ~1.4% at k=16 — errors
    are dense (62–63 of 64 bytes wrong), so latent+corrections is
    EXPANSION (0.37–0.45×), and evolution cannot rescue it (GA plateaued
    at the PCA optimum; float32 PCA is equally bad, so quantization isn't
    the problem). Text's zero error tolerance makes continuous lossy
    latents a cliff, not a trade-off.
  - Exp2 (genome = predictive byte model + arithmetic coding — lossless by
    construction, decode requires the exact model, so genome-gating comes
    free): held-out 1MB cross-entropy — order-0 4.55 bits/byte (1.76×),
    order-1 3.47 (2.31×), **order-2 2.80 bits/byte (2.86×) — already beats
    zlib-9 (2.90 bits/byte, 2.76×)**. Genome size is the design lever:
    order-1 ≈ 64KB (browser-shippable), order-2 ≈ tens of MB; the evolved
    tree-LM (0.354 acc > bigram 0.297) slots in as a stronger, compact
    genome once calibrated probabilities are wired to a coder.
  (user vision: P2P network where content travels as latent vectors and
  every client's decoder auto-updates). **Requires a Flask restart.**
  - New `i2_service.py`: generation-0 genome = deterministic orthonormal
    linear codec (block dim 64, seeded, version `g0001`; lossless byte
    round-trip verified — future generations get to be evolved VAEs).
    Pages are stored ONLY as latent documents in `i2_store/pages/*.json`
    (latents + genome version + sha256 + origin placeholder). Three seed
    pages self-publish on import: home, manifesto, network.
  - New node API in `app.py`: `GET /api/i2/genome` (weights included —
    this IS the client auto-update), `GET /api/i2/pages`,
    `GET /api/i2/page/<name>`, `POST /api/i2/publish`.
  - `static/i2.js` rewritten from mock to client: fetches the genome,
    decodes latents in-browser (Wᵀv, TextDecoder), renders a tiny page
    markup (`# ` heading, `* ` bullet, `[[name|Label]]` links), with
    working link clicks, back/forward/reload, wheel scroll, genome-version
    badge in the URL bar, and a genome-mismatch "decoder out of date"
    error page instead of garbled output. Home keeps the logo + search
    bar hero (search still decorative). Footer shows origin + sha256 +
    decoding genome as proof-of-local-decode.
  - Verified end-to-end under the server's conda env (`simple`): seed
    pages publish, server and client-style decodes agree byte-for-byte,
    publish/list/get round-trip passes.

## 2026-07-02 (Claude Code / Fable 5)
- **I2 canvas now draws a mock browser** (`static/i2.js`, display-only per
  user: "nothing has to work"): full-canvas browser chrome — tab strip
  (one tab + new-tab button), nav toolbar (back/forward/reload, padlock +
  URL bar showing `https://i2.local/home`), and a page body with an "I2"
  logo, a centered search bar, and filler content (heading/paragraph
  skeleton bars + three cards). All geometry scales with the canvas and
  degrades gracefully when the terminal dock is dragged tall. No
  hit-testing yet; state lives in the `BROWSER` object for later wiring.
  Static JS only — no Flask restart needed.

## 2026-07-02 (Claude Code / Fable 5)
- **Fixed garbage text generation: per-leaf score calibration** (user
  report: output was random bytes despite +5 pts accuracy). Diagnosis:
  ridge-seeded leaf scores live in ~[0,1], so temperature-0.8 softmax was
  near-uniform — sampling ignored everything the model knew (temp 0.05 was
  already word-shaped, proving the model itself was fine). Fix in
  `_train_node`: after each leaf freezes, a scalar α is found by bisection
  (monotone 1-D root find, no gradients) so the mean top-1 softmax
  probability equals the leaf's measured accuracy, then **α is folded into
  the frozen weights** — argmax/accuracy identical (verified: same 0.3543,
  +5.73), persists in model.npz, and temperature sampling now behaves:
  temp 0.5 gives word-shaped text with real words; temp 0.2 collapses to
  "the the the" (the true modal distribution — expected at 35% byte
  accuracy). Models trained before this fix need low temp (~0.05–0.1) or a
  retrain (~20s with a saved encoder).

## 2026-07-02 (Claude Code / Fable 5)
- **E16: window 32 falsified** — a full hero encoder at window 32 (same
  budget as the 0.3724 w16 winner) reached only ridge 0.3479 and its trees
  +3.12/+2.82 (vs w16's mean +4.94): doubling the mixer genome needs more
  than double the budget before longer context pays. **Window 16 stays.**
  E17 (running): pop-600 encoder with the best-known recipe + 3-seed trees.

## 2026-07-02 (Claude Code / Fable 5)
- **E14–E15: best encoder yet (held-out ridge 0.3724) + best seed-averaged
  trees (mean +4.94)**:
  - E14 standalone encoders (pop 400, 500 gens, depth 2, ridge+rotate):
    **sa_mutation h256 → ridge 0.3724** (encoder
    `20260702-164927-encoder-92859e`, 986s) — self-adaptive mutation is a
    clear win for the encoder GA. h512 + novelty → 0.3055 (worse: capacity
    without proportional budget, novelty diluted selection).
  - Trees on the 0.3724 encoder (~20s each): **+5.73 / +4.90 / +4.18 across
    seeds 23/11/7, mean +4.94** — best seed-averaged config of the session
    (L1 bf4, cluster, ridge seeds, node_resample). Trees (0.3543) now sit
    slightly BELOW the encoder's in-distribution readout — tree/data side
    became the constraint again.
  - E15 more tree data on that encoder: 64k +4.70/+5.26, 96k +5.40, flat
    64k +4.63 — delta holds but no breakthrough (bigram strengthens with
    data too); ~0.34–0.354 is this encoder's true held-out ceiling.
  - E16 (running): window-32 hero encoder (context length is the
    information bound now that encoding quality is high) + trees on it.
  - **Practical recipe for the user**: pick encoder
    `20260702-164927-encoder-92859e` in the /tree sidebar Encoder select →
    20-second tree runs at ≈ +5 pts over bigram; sidebar defaults already
    match the winning tree shape.

## 2026-07-02 (Claude Code / Fable 5)
- **E11–E13: reusable hero encoder + fixed-encoder tree sweeps**:
  - Trained a standalone **hero encoder** (600→clamped 500 gens, pop 300,
    depth 2, ridge+rotate; `runs/encoder/20260702-162154-encoder-5360d7`,
    held-out ridge 0.3417, 745s). Tree runs reusing it via encoder_id take
    **~18 seconds** (vs 5–10 min) at +4.1–4.4 pts — encoder-as-artifact
    works exactly as intended. (exp11 crashed after training on a
    double-stdout-wrap bug in the experiment script itself — encoder was
    saved; no product code affected.)
  - E12 tree-shape sweep on the fixed encoder (~17s/config): L1 bf4 best
    (+4.43); flat +4.30; deeper/wider worse (L2 +3.20, L3 +2.97,
    bf8 +1.62); **no cluster split −2.8 pts** (+1.65) — clustering
    confirmed essential. With encoder variance removed the node mechanisms
    finally show: **node_resample +5.63**, sa_mutation +5.28,
    both +5.65; node gens 300 adds nothing (node side saturated).
  - E13 seed spread with resample: +5.63/+4.05/+2.70 — s23 flatters all
    configs; gains vs plain are within noise across seeds. The **encoder
    remains the ceiling** (fixed-encoder trees top out ≈ its ridge acc +
    routing bonus). E14 (running): two bigger hero encoders (pop 400,
    sa_mutation; one h512+novelty) + trees on the better one.
  - Session record so far: **+5.83** (E10-encg400). Note encoder_generations
    is clamped at 500 by parse_config.

## 2026-07-02 (Claude Code / Fable 5)
- **E10: +5.83 pts above bigram — new record** (acc 0.3553 vs 0.2970, run
  20260702-160411-tree-54f5d9, encoder gens 400): the encoder budget curve
  is still climbing (gens 200 → 400: +5.63 → +5.83); hero seed-repeat +5.30
  (s11; hero mean ≈ +5.5 across seeds); 64k samples +5.23 (no gain from
  more data — encoder quality, not data volume, is the lever). E11 running:
  a 600-gen standalone **hero encoder saved to the encoder store**, then
  tree runs reusing it via encoder_id (reusable artifact — future tree runs
  skip the ~10-min encoder evolution entirely).
- Fixed `templates/tree.html` UTF-8 corruption (a PowerShell Set-Content
  default-encoding mistake while editing sidebar defaults — repaired by an
  exact byte-level round-trip; page verified serving clean UTF-8).

## 2026-07-02 (Claude Code / Fable 5)
- **New I2 page** (`/i2`, linked "I2 ↗" from the main topbar): shell for a
  new program sharing this Flask server — a full-width canvas
  (`static/i2.js`, placeholder grid, exposed as `window.I2`) above the same
  daemon-backed terminal dock as the main page (reuses `static/app.js`
  verbatim — same element IDs, same `/ws` bridge, so tabs are shared with
  the main page and survive Flask restarts). Dock height persists in its
  own `i2_layout` cookie. **Requires a Flask restart** (new route).

## 2026-07-02 (Claude Code / Fable 5)
- **NEW RECORD: +5.63 pts above bigram** (acc 0.3533 vs 0.2970, run
  20260702-153839-tree-4bd710) — E8/E9 results:
  - E8 seed-averaged (3 seeds): split-rotate mean **+3.58** vs champion
    +3.13 (rotation confirmed); deep+rotate +3.44 (equal mean, tighter
    spread).
  - E9 "hero" encoder budget (gens 200, pop 300, fitness samples 16k):
    **deep encoder +5.63** (767s), shallow +5.28 (339s) — the encoder was
    the binding constraint, budget scaling pays, and depth 2 wins at high
    budget (its nonlinearity needs generations to be found).
  - Recommended recipe (now the /tree defaults): context_dim 256, embed 32,
    window 16, 1 routing layer bf 4, cluster split, seeded embeddings,
    ridge node seeds, ridge encoder fitness + rotating split, encoder
    depth 2, GPU. E10 (running): hero seed-repeat, encoder gens 400, 64k
    samples.
- `/tree` sidebar defaults updated to the winning recipe (dims, routing,
  cluster split on, compute GPU); new checkboxes for `sa_mutation`,
  `node_resample`, and (encoder modal) `encoder_split_rotate` (default on).

## 2026-07-02 (Claude Code / Fable 5)
- **E5–E7 experiment results** (all deltas vs their run's own bigram):
  - E5: d384+routing +3.38, 2 routing layers +3.20, bf16 +2.60 — none beat
    the d256/L1bf4 champion (+4.38); "d512" was silently clamped to 384 by
    parse_config (same config hash exposed it).
  - E6: champion seed-repeats +1.90 (s7) / +3.10 (s23) / +4.38 (s11) —
    **seed noise ≈ ±1.2 pts**, ranking now uses repeats. Deep encoder
    (depth 2, h=256) **+4.35** on its first run; h=512 +4.05; doubling
    encoder gens+samples +3.70; pop 200/gens 250 +2.45 (bigger node GA
    budget does NOT help — nodes are already at their ridge optimum, the
    encoder is the binding constraint: champion router 96.4%, rare-byte
    leaves ~99.6%, tree acc ≈ encoder held-out readout 0.347).
  - E7: **encoder_split_rotate +4.58 (new best single run**, acc 0.3365,
    run 20260702-152509-tree-70c937); sa_mutation +2.90 (neutral/negative);
    both together +3.28.
- **Two more evolution mechanisms** (config + sweep, UI follows):
  - `sa_mutation`: ES-style self-adaptive mutation scale — each individual
    inherits a parent's step size × log-normal perturbation (clip
    0.005–0.5); champion still tracked on raw fitness.
  - `encoder_split_rotate`: the ridge encoder fitness re-shuffles its
    fit/val split every generation (deterministic per seed), so selection
    can't creep-overfit a fixed validation half.

## 2026-07-02 (Claude Code / Fable 5)
- **E4 scale probes on the winning recipe** (all vs their own bigram
  baseline): context_dim 384 → **+3.10**; 64k samples → +2.28; window 32 →
  +2.02 (no gain over window 16); seed 7 → +2.60 (robust across seeds); and
  the headline: **routing now HELPS** — 1 layer bf 4 → **+4.38 pts**
  (acc 0.3345, run 20260702-145406-tree-3e185e), the first time a routed
  tree beats the flat model. The aligned encoder (ridge fitness) + ridge
  seeds + clustered split turned routing from a −7pt cost into a +1.7pt
  gain over flat.
- **Two new evolution mechanisms** (for the ongoing experiment loop):
  - **`node_resample`** (sidebar-less for now, config/sweep only): EEC
    non-stationarity for node GAs — the batched fitness closure rotates a
    common half-sample every generation (deterministic per run seed), so
    nodes can't win by memorizing one fitness sample. GPU path only.
  - **`encoder_depth: 2` + `encoder_hidden`** (encoder-modal select "Encoder
    depth"): service-side `DeepContextEncoder` — flat → tanh(W1) → tanh(W2)
    two-stage evolved mixer (position interactions a single linear mix
    can't encode). Stage-2 seeded near-identity (×1.2 vs the double tanh)
    so depth 2 STARTS equal to depth 1. Batched GPU fitness handles both
    layouts; save/load moved to service-side `_save_model`/`_load_model`
    (blueprint mirror + encoder_kind/hidden metadata; old files load as
    kind 1); encoder store + embedding cloud + replay all handle deep
    encoders (smoke-tested end-to-end). Evolve-dims is disabled with
    depth 2 (ragged layout).

## 2026-07-02 (Claude Code / Fable 5)
- **BEAT THE BIGRAM BASELINE: +2.65 pts** (autonomous experiment session;
  user goal "we have to do better than −1.4"). Two new mechanisms, both in
  the project's closed-form-seed + evolve-to-refine pattern (no gradients):
  - **`ridge_seed`**: router/leaf GA seed populations now optionally include
    the closed-form ridge/least-squares solution to one-hot targets
    (`_nc_seed` — one linear solve, same moment-statistics family as the NC
    seed, but it sits at the node's actual linear optimum). Effect at dim
    256 flat: **−11.05 → −0.08 pts** vs bigram.
  - **`encoder_fitness: "ridge"`**: the encoder can now be evolved against
    the **held-out accuracy of a closed-form ridge readout** (fit on half
    the fitness sample, scored on the other half — batched on GPU via
    torch.linalg.solve) instead of the in-sample nearest-centroid proxy.
    This aligns encoder evolution with the ridge-seeded linear leaves that
    consume it AND adds generalization pressure (centroid overfit can't
    win). Effect: **−0.08 → +2.65 pts** (acc 0.3172 vs bigram 0.2908; 32k
    samples, dim 256, window 16, emb 32 seeded, cluster split, flat,
    pop 100/150 gens, seed 11, run 20260702-144002-tree-bb9d13).
  - Also added **`node_fitness: "prob"`** (geometric-mean target probability
    = inverse perplexity, smooth landscape) — currently NOT usable: ridge
    scores aren't calibrated logits, so softmax-based fitness collapses to
    1–12 unique predictions (measured twice). Kept behind the flag for
    future calibrated variants; default "acc" unchanged.
  - Experiment log: E1 scale reference (dim 256 flat −11.05, +1 routing
    layer −18.65) → E2 ridge seed (−0.08; prob-fitness arithmetic-mean
    variant collapsed to uniq=1, replaced by geometric mean) → E3 (+2.65).
    Results in scratchpad results.jsonl; all runs persisted in the runs
    store under notes "exp E…".

## 2026-07-02 (Claude Code / Fable 5)
- **Co-occurrence-seeded embeddings** (diagnosis from the new 3D plot: the
  byte-embedding table stays isotropic/random-looking even when the encoder
  scores 0.625 NC fitness, and even at embed_dim 6 — because NC fitness only
  needs *distinct* codes, not organized ones, and any embedding re-coding can
  be compensated by the mixer: the table is a **neutral plateau** evolution
  never shapes. User: "the embedding on the tree is horrible"):
  - New encoder-modal checkbox **"Seed embeddings from co-occurrence"**
    (`encoder_seed_embeddings`, default off, sweepable "seed embeddings
    (0/1)"): `_cooc_embedding_seed` PCA-projects each byte's preceding-
    context profile (`_byte_context_features`, same features as the
    clustered token split) down to embed_dim and uses that as the heuristic
    seed's embedding table (mean row norm ≈ 1) — bytes used in similar
    contexts START near each other, and evolution refines instead of
    wandering the plateau. Recorded in the runs-panel encoder card
    ("seeded embeddings: co-occurrence PCA / random").
  - Measured (25 gens, pop 40, dim 64/emb 32, GPU, seed 44): embedding
    structure **survives evolution** — PC1 5.6% → **31.4%**, effective rank
    30.0 → **15.8** of 32; NC fitness unchanged (0.290 vs 0.284), as
    predicted (fitness doesn't require embedding geometry — the experiment
    is whether the downstream tree benefits from linguistic context
    vectors). Needs the user's Flask restart.

## 2026-07-02 (Claude Code / Fable 5)
- **Population novelty bonus for the encoder GA** (user: "genomes that are
  more different should get a fitness boost, not a penalty" — the existing
  "Head diversity" is *within-genome* dim decorrelation; this is the
  *population-level* mechanism they described):
  - New encoder-modal checkbox **"Population novelty bonus"**
    (`encoder_novelty`, default off) + **"Novelty strength"**
    (`encoder_novelty_strength`, default 0.5): each generation, a genome's
    novelty = mean L2 distance to the rest of the population (gene-subsampled
    to ≤512 dims for speed), normalized 0–1; **selection** scores become
    fitness × (1 + strength × novelty). Fitness sharing: different genomes
    breed more. The **champion is still tracked on raw fitness** (novelty
    shapes who breeds, not what gets returned), and the fitness chart shows
    raw values. `_evolve` gained a `novelty` param; elites now select on the
    boosted scores.
  - Applies to the encoder accuracy phase (fixed-dim); persisted in configs,
    shown in the runs-panel encoder card ("novelty bonus"), sweepable
    ("encoder novelty bonus (0/1)" and "novelty strength").
  - Verified headless (GPU, 12 gens): runs complete, flags persist,
    acc 0.218 → 0.220 at strength 0.5 (neutral at this tiny scale; the
    mechanism matters on longer runs where convergence stalls progress).

## 2026-07-02 (Claude Code / Fable 5)
- **3D embedding-space plot on the runs page** (user: "I have a feeling it
  might be collapsing"):
  - `tree_service.embedding_cloud`: PCA (SVD) of the encoder's byte-embedding
    table (vocab × embed_dim) → 256 unit-scaled 3D points + collapse
    diagnostics: per-PC explained variance, **effective rank**
    (exp of spectral entropy — ≈embed_dim when isotropic, →1 on collapse),
    mean vector norm. Works for tree runs (model.npz) and standalone encoder
    runs (encoder.npz). `runstore.embedding(rid)` +
    `GET /api/runs/<rid>/embedding` (needs the user's Flask restart).
  - `/runs` detail (tree + encoder runs with a checkpoint): new **"Embedding
    space (PCA → 3D, drag to rotate)"** card — hand-rolled canvas scatter
    (no 3D lib), drag-rotates (yaw/pitch), depth-sorted with size/alpha
    cues, points colored by byte class (␣ / a–z / A–Z / 0–9 / punctuation /
    control+extended), common letters (␣ e t a o n i) labelled, legend, and
    a diagnostics line with an explicit ⚠ verdict when PC1 > 90% or the
    effective rank is low.
  - Fixed alongside: `_persist_finalize` now writes the **final**
    context_dim back into config.json, so dim-evolved models can actually be
    loaded again (replay + embedding previously would fail to unpack).
  - Encoder runs on the dashboard hide the inference card (nothing playable).
  - First measurement on today's small test runs: effective rank ~15.4 of 16,
    PC1 ≈ 10% — NOT collapsed, but near-isotropic, i.e. barely more
    structured than random init (evolution has mostly not shaped the
    embedding table yet). Note: high rank ≈ random-like is as informative as
    low rank ≈ collapse.

## 2026-07-02 (Claude Code / Fable 5)
- **Encoder training is now its own modal + saved/reusable encoders** (user:
  train an encoder once, then use/replace it across tree runs instead of
  re-genning both every run):
  - `/tree` sidebar: the encoder fields moved out of the sidebar into an
    **"Encoder trainer" modal** (button "Train encoder…"); the sidebar now has
    an **Encoder select** — "evolve fresh each run" (old behavior) or any
    saved encoder. New "Fitness samples" field (`encoder_samples`) is finally
    exposed in the modal.
  - `tree_service.EncoderTrainer` (op `train_encoder`, runs in the job hub
    like tree runs — survives page nav, one job at a time, Stop works):
    evolves just the encoder (all existing knobs: gens, pop, dims evolution,
    time/diversity constraints, speed phase, GPU batching) and saves
    `runs/encoder/<id>/` — config/history/summary in runstore layout (so it
    appears under a new **"encoder" tab on the runs dashboard**, sparkline +
    Encoder evolution card work) plus `encoder.npz` (genome + dims).
  - `list_encoders()` / `load_encoder()`; socket op `encoders` returns the
    saved list (requested on connect; refreshed after each save); the modal
    shows the saved-encoder table with per-row **use** buttons.
  - Tree runs with `encoder_id` set **load the saved encoder, skip encoder
    evolution entirely, and adopt its dims** (context_dim/embed_dim/window
    override the sidebar — logged in a status line; falls back to fresh
    evolution if the encoder can't load). `encoder_id` persists in run
    configs.
  - Verified headless: standalone train (GPU) → saved + listed; tree run with
    deliberately different sidebar dims adopted the encoder's dims (48/16/8),
    did not re-evolve, trained + evaluated + persisted normally.
  - app.py gained the two socket ops — needs the user's Flask restart to
    take effect (not performed here).

## 2026-07-02 (Claude Code / Fable 5)
- **Repository published to GitHub** (user request): added `LICENSE` (GNU AGPL-3.0
  — strict copyleft covering network use) and committed the full working tree for
  publication as the public repo `GENREG-Builder` under the `A1CST` account.

## 2026-07-02 (Claude Code / Fable 5)
- **GPU compute option for Tree-of-Models** (user: "these longer trainings are
  killing me"): new **Compute** selector (CPU / GPU (CUDA, batched)) in the
  `/tree` sidebar, `device` in the config (persisted in run configs).
  - GPU mode evaluates the **whole population per generation in one batched
    tensor op** on CUDA instead of a Python loop of numpy matmuls:
    `_batch_linear_acc` (every router/leaf: genomes → (P,dim,k) einsum →
    accuracy) and `_batch_encoder_fitness` (replicates
    ContextEncoder.encode — embed gather + positions → bmm mix → tanh — plus
    batched nearest-centroid accuracy, and batched time/diversity penalties
    when those constraints are on). Population chunked to bound GPU memory.
    `_evolve` gained a `batch_fitness` hook; the GA logic is unchanged.
  - Measured (pop 100, 16k samples, 30 gens, layers 2, RTX 4080):
    **79.8s → 7.4s ≈ 10.8×**; small configs ~1.8×. Encoder accuracy matches
    CPU to ~1e-3 (float32 vs float64 — champions can differ slightly, same
    caveat as the main engine's GPU mode).
  - Falls back to CPU with a status message if CUDA/torch is unavailable.
    "Evolve context dim" runs keep the CPU loop (ragged genomes); the
    dim-evolution GA is exempt from batching.

## 2026-07-02 (Claude Code / Fable 5)
- **Clustered token split** (user idea: the root's 0–63/64–127/… split is an
  arbitrary token-ID artifact, not linguistic — let the tree be built over
  context clusters instead):
  - New `/tree` checkbox **"Cluster token split"** (`cluster_tokens`, default
    off, sweepable "cluster token split (0/1)"): bytes are described by their
    **preceding-context profiles** (Hellinger-scaled distributions of the
    previous 1–2 bytes, `_byte_context_features`) and recursively partitioned
    by **capacity-balanced k-means** per node (`_balanced_kmeans` — balanced
    so leaf sizes still match the routing_layers math; deterministic per run
    seed so the dim-evolution rebuild reproduces the same partition;
    `_build_root` replaces direct `build_tree` calls). tree_lm blueprint
    untouched — its TreeNode/inference/save-load were already set-based.
  - `_train_node` is now fully set-based (membership mask, leaf local-index
    lookup, router child-of lookup) instead of assuming contiguous ranges —
    identical results for sequential trees.
  - Icicle t0/t1 are now DFS *positions* (== byte values for sequential
    trees, a contiguous layout axis for clustered ones); manifest + trace
    nodes carry a `sample` token preview ("␣ e t a …") which the icicle
    tooltip and the routing-inspector/trace views show instead of
    "bytes N–M" when present.
  - Measured (6k samples, dim 48, bf 4, layers 2, seed 11, small run):
    clustering is plainly linguistic (uppercase / common-lowercase+space /
    unused-high-bytes branches) and held-out accuracy **0.129 → 0.168**
    (+3.9 pts) vs the sequential split on identical config. Note: ␣/e/t
    landed *together* (they share preceding contexts — co-occurrence groups
    similar-context bytes, making ROUTING easy and leaves hard); it still
    beat byte-ID ranges. Clustered models replay fine from the runs store.

## 2026-07-02 (Claude Code / Fable 5)
- **Separate encoder population** (`encoder_pop_size`, `/tree` sidebar field
  "Encoder population", default 0 = inherit the shared Population): the
  encoder GA previously always used the same pop_size as every router/leaf;
  since the encoder genome is orders of magnitude larger, it can now have its
  own population. Applies to all three encoder GAs (fixed-dim `_evolve`, the
  variable-dim GA, and the phase-2 speed phase; seed-count scales with it) —
  `_evolve`/`_evolve_encoder_dims` gained a `pop_size` override parameter
  (node training is untouched; seed_pop is now also truncated to pop_size).
  Persisted in run configs, shown in the runs-panel Encoder evolution card
  (effective size), and sweepable ("encoder population (0 = shared)").
  Verified headless: 0 inherits (pop 10), 24 applies to fixed/speed/dims
  paths, full runs complete.

## 2026-07-02 (Claude Code / Fable 5)
- **Encoder dimension evolution** (answering "are we evolving the dims in the
  encoder?" — previously no, context_dim was always fixed by the sidebar):
  - New `/tree` sidebar checkbox **"Evolve context dim"**
    (`encoder_evolve_dims`, default off = fixed number, the current
    behavior). When on, `tree_service._evolve_encoder_dims` runs a
    **variable-dimension GA**: each individual carries its own context_dim;
    structural mutation (p=0.25) grows (append a small random mixer column)
    or shrinks (drop the lowest-|W|+|b| column) within **[4, 4×start]**
    (start = the sidebar Context dim); crossover mixes parents in their
    shared column subspace (child keeps parent 1's dim). Works with the
    time/diversity constrained fitness (all fitness closures now take an
    optional dim; scratch probe encoders are cached per dim via
    dataclasses.replace). Per EEC, pure accuracy fitness tends to grow dims —
    the hint says to pair with time/diversity for shrink pressure.
  - Downstream: router/leaf genome sizes depend on context_dim, so when the
    evolved dim differs the trainer **rebuilds the routing tree at the new
    width** and re-emits the tree manifest (icicle + params tiles update);
    cfg.context_dim is updated so persistence/replay (`save_tree`/`load_tree`)
    and phase 2 all use the evolved width. node_gen events for the encoder
    gain `dim`/`mean_dim`; node_done gains `context_dim`. Sweepable
    ("evolve context dim (0/1)").
  - Verified headless (start dim 32, 6 gens): fixed mode unchanged; evolve
    mode reached dim 33/34, tree rebuilt at the evolved width, full run +
    model save completes.
- **Encoder training is now fully recorded in the runs dashboard** (it
  previously only left one history line):
  - `summary.json` gains an **`encoder` block**: NC accuracy, fitness
    samples, generations, start→final context dim, per-constraint settings
    and outcomes (time budget + active-weight fraction, diversity budget +
    head redundancy), speed-phase results, plus **per-generation best/mean
    fitness curves** for "encoder" and "encoder-speed" (captured from the GA
    loops).
  - `/runs` detail pages for tree runs show a new **"Encoder evolution"
    card**: metrics table + fitness-curve sparkline(s) (labelled
    "(constrained)" when a constraint shaped the curve). Old runs without
    the block simply don't show the card. The ⤓ Export JSON download
    includes it automatically (it lives in summary).
  - Loads when the Flask server is next started (no restart performed).

## 2026-07-02 (Claude Code / Fable 5)
- **Encoder head-diversity constraint** (user experiment: time pressure didn't
  break the plateau — try pushing the encoder's output dims to be different):
  - New `/tree` sidebar checkbox **"Head diversity"** (`encoder_diversity`,
    default off) + **"Diversity budget"** field
    (`encoder_diversity_budget`, default 0.5, lower = harsher): when on, the
    encoder's whole accuracy phase is scored as
    fitness ÷ (1 + redundancy/budget), where **redundancy = mean
    |off-diagonal correlation| between the encoded context dims** on the
    fitness sample (0 = every head carries a different signal). Dead
    (constant) dims count as fully redundant so collapsing dims can't game
    the penalty.
  - Composes with the time constraint: both checkboxes on → both penalties
    multiply into one constrained fitness (`fitness_constrained` now folds in
    whichever constraints are enabled). The phase-2 speed phase got its own
    pure-time closure back so it is unaffected by the diversity flag.
  - node_done for "encoder" now carries `redundancy` when the mode is on;
    the status line reports acc + per-constraint numbers. Both new fields
    persist in run configs and are **sweepable** ("encoder head diversity
    (0/1)" and "diversity budget"), so diversity × time × budgets can be
    A/B'd in one sweep. Reported accuracy stays raw for comparability.
  - Verified headless (seed 5, 4 encoder gens): baseline acc 0.448;
    diversity on (budget 0.3) → redundancy 0.072 at acc 0.420;
    time+diversity together → 44% active weights, redundancy 0.078,
    acc 0.408; flags persist; full runs complete.
  - Loads when the Flask server is next started (no restart performed).

## 2026-07-02 (Claude Code / Fable 5)
- **Time-constrained encoder evolution from scratch** (user experiment: evolve
  the encoder with time & its original fitness *combined*, instead of only as
  a phase-2 addition):
  - New `/tree` sidebar checkbox **"Time-constrained from scratch"**
    (`encoder_time_constrained`, default off): when on, the encoder's whole
    accuracy phase (all `encoder_generations`) uses
    fitness ÷ (1 + active-weights/budget) — the time/Occam constraint folded
    into the original nearest-centroid fitness from generation 0, sharing the
    existing "Speed time budget" field. The seed population additionally gets
    magnitude-pruned variants (30/50/70%) of the heuristic genome so the
    sparse region is reachable immediately. Reported accuracy stays raw
    (unconstrained) for comparability; node_done for "encoder" now carries
    `active_fraction` and a status line reports acc + active-weight % when
    the mode is on. The flag persists in run configs and is **sweepable**
    ("time-constrained encoder (0/1)", suggest 0,1), so it can be A/B'd
    against the two-phase form in one sweep.
  - The independent phase-2 "Encoder speed phase" is unchanged (hint text now
    clarifies it runs AFTER the accuracy phase); its fitness now reuses the
    same constrained-fitness closure instead of a duplicate definition.
  - Verified headless: constrained-from-scratch run (budget 0.3) → encoder at
    44% active weights, flag persisted, full train/eval completes; two-phase
    regression run unchanged (dense phase 1 → 30% active after speed phase).
  - Note: `tree_service.py` changes load when the (currently stopped) Flask
    server is next started — no restart was performed.
- **Runs can now be exported to JSON**: every run detail page on `/runs` has a
  **"⤓ Export JSON"** button (next to the checkpoint badge) that downloads
  `<run-id>.json` containing the full run record — config (launch params +
  metadata + status), summary (final result / eval block / sweep results),
  complete per-generation history, and, for tree runs, all saved routing
  traces. Entirely client-side (`runs.js exportRun`, Blob download from the
  existing `/api/runs/<id>` + `/traces` endpoints) — no new server routes.

## 2026-07-02 (Claude Code / Fable 5)
- Added a **Documentation browser** (`/docs`, "Docs ↗" links in the build /
  runs / tree topbars) styled like the runs dashboard:
  - Copied the user's `Desktop\documentation` folder (91 files: .md, .pdf,
    .json, .docx, …) into `GENREG/documentation/`.
  - `app.py`: `GET /docs` (page), `GET /api/docs` (recursive file list with
    ext/size/mtime), `GET /api/docs/file/<path>` (md/txt/json served inline as
    utf-8 text, PDFs as application/pdf for the embedded viewer, everything
    else as a download; `send_from_directory` blocks path traversal —
    verified).
  - `templates/docs.html` + `static/docs.js` + CSS (`doc-*`, `md-body`):
    left sidebar with type tabs (All / Markdown / PDF / JSON / Other), a
    name filter, and run-node-styled file entries; detail pane renders
    **Markdown inline** via a small built-in renderer (headings, fenced code,
    tables, lists with hanging-indent continuation, blockquotes, inline
    formatting — HTML-escaped first, no external libs), **PDFs in a native
    `<embed>` viewer**, JSON pretty-printed, and a ⤓ Open/download button on
    every file. Deep-linkable via `/docs#<path>`.
  - Verified end-to-end after a Flask restart: list API, md/pdf serving,
    traversal 404, and the renderer against the real doc set (tables/code
    blocks balanced, no unescaped HTML).

## 2026-07-02 (Claude Code / Fable 5)
- **Tree training now survives leaving the page.** Previously the `/treelm`
  socket handler owned the trainer and stopped it on disconnect, so clicking
  "Runs ↗" (same-tab nav) killed the run/sweep. Now:
  - `tree_service.JobHub` (module singleton `HUB`): the trainer/sweeper runs
    in the hub, decoupled from any socket. Sockets are viewers — they
    subscribe on connect and detach on close. The hub keeps a replayable
    journal (all events except per-generation ticks, which are kept only for
    the node currently evolving; latest status only; capped at 50k), so a
    reconnecting page replays the snapshot and rebuilds mid-run state
    (icicle, fitness chart, sweep table, tiles), then gets a
    `{"type":"job","running":…}` event to sync the buttons.
  - Stopping is explicit only: the Stop button or starting a new job.
    Multiple tabs can watch the same run simultaneously.
  - Per-connection send lock (hub thread + handler thread both write).
  - Verified: killed the socket mid-run — training completed server-side and
    a fresh connection replayed tree/node/eval/done correctly.
- Added a **configurable config sweep** to the `/tree` page:
  - New "Config sweep" card: each sweepable parameter (routing layers,
    branching, dims, window, gens/pop, encoder gens, speed gens, speed budget,
    …) is either **locked to the sidebar value** (unchecked) or given a
    comma-separated **list of values to test**; every combination trains
    (Cartesian product, capped at 24 candidates, values capped at 8/param),
    live-ranked in a results table, all candidates on a shared seed (1234)
    for comparability.
  - `tree_service.TreeSweeper` (`/treelm` op "sweep"): runs candidates
    sequentially as normal TreeLMTrainer runs (each persists to the runs
    store, tagged `sweep <id> · <candidate>` in its notes) and persists the
    sweep itself as a run (`runs/tree/<stamp>-sweep-<hash>/`) whose
    summary.json carries `sweep_results` (ranked) and whose history.jsonl
    holds per-candidate accuracy (sparkline works).
  - `/runs` dashboard: sweep entries show a **"Sweep results (ranked)"**
    table in the detail view with per-candidate links that jump to the
    candidate run's detail; `/runs#<run-id>` now deep-links to a run (the
    sweep table on /tree links out this way); run details now show the
    `notes` row (so candidates display their sweep membership).
  - Sidebar: encoder speed phase is now behind an **enable checkbox**
    (default off — measurement showed it compresses but slightly hurts
    downstream accuracy) and the **time budget is exposed** as a field, so
    the constraint's strength can be tuned rather than only toggled.
- Added a **two-phase encoder evolution with a speed/time constraint**
  (user experiment: does Occam pressure break the ~25% encoder plateau?):
  - After the accuracy-only phase (encoder_generations), a second phase runs
    for `encoder_speed_generations` (default 40, 0 = off, UI field "Encoder
    speed gens") with fitness × 1/(1 + active/budget) — the project's Time/
    Occam constraint form, where "time" = fraction of genome weights with
    |w| > 0.01 (effective multiplies; for a fixed-shape dense encoder,
    sparsity is the evolvable notion of speed). `encoder_time_budget`
    (default 0.5) sets the pressure. Phase 2 is seeded with the phase-1
    champion plus magnitude-pruned variants (30/50/70/85%).
  - Streams as node "encoder-speed" (live fitness chart, run history); emits
    the accuracy delta and final active-weight fraction.
  - First measurement (dim 64, layers 0, seed 11): phase 2 compressed the
    genome to **32% active weights at −0.009 fitness** (the dense genome was
    ~3× redundant) but did not break the plateau; downstream tree accuracy
    was noisier/slightly worse. The plateau itself tracks the linear NC
    ceiling at context_dim and the 2000-sample centroid noise — raising
    `encoder_samples` and context_dim moves it more than generations do.
- Added a direct **Routing layers** lever (0–8) to the `/tree` config,
  replacing the min-leaf-tokens field: leaf size is derived as
  256 / branch^layers, and **0 layers builds no routers at all** — the root is
  a single flat evolved specialist over all 256 bytes.
  `routing_layers` is accepted by `parse_config` (explicit `min_leaf_tokens`
  still works and can now go to 256) and persisted in run configs.
  Measured at dim 64 / bf 4 (seed 11): layers 0 → 0.260, 1 → 0.205,
  2 → 0.227, 3 → 0.211 (bigram 0.336) — the flat model is currently the most
  accurate, i.e. each routing level still costs accuracy; the icicle +
  inspector show where.
- **Implemented encoder evolution (blueprint step 1) + nearest-centroid GA
  seeding** — attacks the verified below-bigram collapse:
  - `tree_service._evolve_encoder`: evolves the full ContextEncoder genome
    (via blueprint pack/unpack) before tree training. Fitness = nearest-
    centroid next-byte accuracy of the encoded context (closed-form,
    gradient-free proxy for linear-router separability). The GA population is
    seeded with a structured candidate whose mixer has identity blocks for the
    most recent byte positions (last-byte/bigram information survives the
    projection from gen 0) plus recency-decayed random projections for older
    positions. New config: `encoder_generations` (default 40, 0 = skip, UI
    field added), `encoder_samples` (default 2000). Encoder stage streams
    node_start/node_gen/node_done with id "encoder" (live fitness chart) and
    is recorded in run history.
  - `_nc_seed`: every router/leaf GA population is now seeded with the
    closed-form nearest-centroid classifier for its local data (weights =
    class centroids, bias = −½‖μ‖², absent classes −1e6) — evolution refines a
    working solution instead of random noise.
  - Measured (8k samples, pop 40, gens 40, seed 11): held-out accuracy
    0.074 → **0.228** (bigram 0.336); unique predictions 3 → 25; temp-0.8
    output goes from 'ooo…' loops to word-shaped text. Remaining gap to bigram
    is representational (a 64-dim tanh context vector can't hold a full
    256-way last-byte table); larger context_dim/embed_dim narrows it
    (dim 256 / emb 64 / bf 16: −0.075 vs bigram) at ~3× train time.
  - UI default `min_leaf_tokens` 1 → 16: single-token leaves make temperature
    sampling a no-op (nothing to sample over), which presented as "generates
    the same character regardless of temp".
  - `infer_run` now loads model.npz through an in-memory buffer — np.load
    inside the blueprint's `load_tree` otherwise holds the file handle and
    locks the file on Windows.
  - Note: run `20260702-053927-tree-211b4b` had config/summary/history
    accidentally deleted during test-run cleanup while its model.npz was
    locked by a replay; config.json was reconstructed from the model file
    (tree shape exact, GA params assumed defaults), history and eval metrics
    are lost. model.npz is intact and replays.
- **Routing traces now persist with their run** and are browsable from `/runs`:
  - `tree_service.trace_generate` appends every trace (prompt, text, temp,
    full per-step decision data, timestamp, run id) to
    `runs/tree/<id>/traces.jsonl` of the model's run; `runstore.get_traces` +
    `GET /api/runs/<rid>/traces` serve them newest-first.
  - Trace rendering extracted to shared `static/trace_view.js`
    (`TraceView.mount`), used by both the `/tree` inspector and a new
    **Saved traces** card on tree-run detail pages in `/runs` (list of traces →
    click to replay the full inspector).
  - Every mounted trace has a **⤓ JSON export** button (downloads the complete
    trace as a formatted .json file).
  - Diagnosis note (verified by measurement, not a code bug): generation IS
    autoregressive — the context vector changes every step — but the model is
    context-insensitive because the ContextEncoder is a *random, unevolved*
    projection (the blueprint's "pre-evolve encoder" step has no implementation
    yet), so routers collapse to majority routes (~everything → e/o/n leaves)
    and score below the bigram baseline. Biggest tuning lever: evolve the
    encoder (GENREG hook exists in `tree_lm.train_tree_bottom_up` step 1).
- Added a **routing inspector** to the `/tree` page for debugging generation:
  - `tree_service.trace_generate(prompt, length, temperature)` generates like
    `generate_text` but records every decision — per step: the context window,
    each router's full score vector + chosen child (with children's byte
    ranges), and the leaf's top-8 candidates with softmax probabilities.
    `/treelm` gained a `trace` op returning it as one `trace` event
    (length capped at 200 steps).
  - UI: a "Routing inspector" card — trace a prompt, get a clickable strip of
    the generated bytes (control bytes shown escaped); selecting a byte shows
    its full decision path top-down: one block per router with score bars per
    child (chosen highlighted, decision **margin** shown, ⚠ near-tie flag when
    margin < 0.05), then the leaf's candidate distribution (prob bars, chosen
    ✓). Makes it visible where routing collapses (e.g. every context funneling
    into the same child) vs. where leaves are just under-trained — the two
    things to tune differently.
- **Tree-of-Models runs now persist in the runs store** and appear on the `/runs`
  dashboard under a "tree" environment tab.
  - `tree_service.py`: every training run writes `runs/tree/<id>/` in the
    runstore layout — `config.json` (full tree config + status), `history.jsonl`
    (one line per frozen node: accuracy as best, running mean), `summary.json`
    (held-out accuracy as best.score, bigram baseline as best.base, full eval
    block), and `model.npz` (the frozen tree, blueprint `save_tree` format).
    Stopped runs persist too (partial model, status "stopped"); errored runs get
    status "error". Replaces the previous flat `runs/tree/tree_*.npz` dump
    (pre-existing flat files were left in place but have no run metadata).
  - `runstore.infer`: tree branch — "replay" for a tree run loads `model.npz`
    (via `tree_service.infer_run` / blueprint `load_tree`) and generates a
    ~400-byte text sample at temperature 0.8.
  - `runs.js`: tree runs render in the tech tree (accuracy as score, node
    history sparkline works as-is); the inference panel shows "▶ Generate
    sample" and displays the generated text instead of a game board.
  - `/treelm` socket now emits a `run` event with the run id; done/stopped
    events include `run_id`.

- Added a **Tree-of-Models text prediction page** (`/tree`, "Tree LM ↗" links in
  the build-interface and runs topbars) — a separate program from the build
  interface, implementing the GENREG tree-of-models blueprint: byte-level
  tokenizer (vocab 256), evolved context encoder, hierarchical routing tree with
  evolved leaf specialists, bottom-up freeze-and-stack training, no gradients.
  - `genreg_train/tree_lm.py`: the blueprint skeleton, kept verbatim
    (TreeConfig / ContextEncoder / TreeNode / build_tree / inference /
    save-load / evaluate / `_default_evolve` GENREG hook).
  - `genreg_train/tree_service.py`: web wrapper — re-implements only the
    orchestration (bottom-up walk + GA, mirroring `_default_evolve`) so it can
    stream per-node/per-generation events and honor stop; trains on windows
    sampled from the **49 MB / 50-book Gutenberg corpus**
    (`project/EEC-main/engine/corpus.txt`) instead of the toy demo string, with
    a held-out split, vectorized bigram baseline, temperature sampling for
    generation, and model persistence to `runs/tree/*.npz`.
  - Flask: `GET /tree` (page) + `/treelm` WebSocket
    (`start` / `stop` / `generate` ops → corpus / tree / node_* / eval / done /
    generated events), same thread layout as `/train`.
  - `templates/tree.html` + `static/treelm.js` + CSS (`tlm-*`): config sidebar,
    stat tiles (held-out accuracy, bigram baseline, Δ, tokens/sec, params,
    frozen-node progress), an **icicle map of the routing tree** (x = byte range
    0–255, y = depth; nodes colored by accuracy on a blue sequential ramp, amber
    pulse while evolving, tooltips per node), a **live best/mean fitness line
    chart** for the node currently evolving (crosshair + tooltip), **mean
    accuracy by depth** grouped bars (routers vs leaves, plus table view), and a
    text-generation panel. Chart palette validated for CVD/contrast on the dark
    surface.

## 2026-07-01 (Claude Code / Opus 4.8)
- Added a **Runs & Checkpoints dashboard** — a separate page (`/runs`, opens in a
  new tab via the "Runs ↗" link in the build-interface topbar) for a second monitor.
  - `runstore.py`: every Start Training persists `runs/<env>/<id>/` with
    `config.json` (launch config + metadata), `history.jsonl` (per-gen metrics),
    `summary.json` (final result), and `checkpoint.pkl` (champion genome, engine
    format). Wired into Flask `/train` via an emit wrapper; both trainers expose
    `champion()`. Interrupted runs still save the last completed generation's
    checkpoint.
  - Flask: `GET /runs` (page), `GET /api/runs` (list grouped by env),
    `GET /api/runs/<id>` (config + metrics + history),
    `GET /api/runs/<id>/replay` (loads the checkpoint and plays one fresh episode
    → frames for the inference viewer).
  - `templates/runs.html` + `static/runs.js` + CSS: per-environment **tech-tree**
    (env tabs to switch; runs as connected nodes with status/checkpoint badges),
    a run **details** panel (full config, result metrics, best/mean fitness
    sparkline), and an **inference/verify** viewer that animates the saved
    checkpoint playing snake/2048 on a board canvas.
  - Verified end-to-end: runs persist (CPU + GPU, snake + 2048), API lists/details
    work, and checkpoint inference replays.

## 2026-07-01 (Claude Code / Opus 4.8)
- Added a **GPU-capable vectorized training engine** (PyTorch), device-selectable.
  - Benchmarked first: naive per-step GPU is *slower* for these tiny sequential
    nets (0.4–0.7× at P≤2048), but the fully **batched** vectorized engine crosses
    over ~P=4k and reaches **1.92× faster than CPU at P=16,384** on the RTX 4080,
    and keeps scaling — the right architecture for large populations / heavier
    future envs.
  - `vector_engine.py`: batched recurrent forward, batched relative self-adapting
    mutation, batched selection/elitism, and `VSnake` (grid-based batched snake
    matching the CPU rules). Verified it learns (best fitness 1.1 → 11 in 30 gens).
  - `vector_trainer.py`: `VectorTrainer` emits the **same events** as the CPU
    trainer; each generation the champion is pulled off the device into a CPU
    genome so the Microscope layers + board replay reuse the existing path.
  - `trainer.create_trainer()` factory dispatches CPU-engine vs vectorized by a new
    `device` setting (`cpu` | `gpu` | `auto`). GPU mode currently covers **snake
    with no constraints, fixed H**; anything else (2048, any constraint,
    evolve-hidden) falls back to the full CPU engine. `auto` uses GPU only when
    cuda is present and population ≥ 2000. Flask `/train` now uses the factory.
  - Control Panel: **Compute** selector (CPU engine / GPU vectorized / Auto).
  - Installed PyTorch (CUDA 12.4 build) into the app's conda env.
  - **Vectorized 2048** (`V2048`): batched slide/merge (all 4 directions computed
    then gathered per-board), valid-move masking, 16-feature obs. Learns (best
    ~3.8k → 6.3k in 12 gens, matching the CPU engine).
  - **Vectorized constraints** (`build_vconstraints` + batched application in the
    evaluators): energy (env-specific restore — snake apples, 2048 merges),
    mortality, time (post-score), noise + occlusion (obs), entropy (state decay),
    efficiency (batched weight cost). `VEC_SUPPORTED` gates them; the factory
    routes supported combos to GPU and unsupported ones (memory-rent, scarcity,
    reproduction-cost, non-stationarity, perception-cost) to the CPU engine.
    Champion replay uses the matching CPU constraints so the board display agrees.
    Verified: energy shortens snake fitness; GPU+energy+time runs end-to-end.
- Energy constraint is now environment-specific (restore method differs per game):
  - **2048**: energy is restored on merges — each merge of two v-tiles restores v
    (a move's gain = Σ2v, so restore = gain/2 × `merge_energy`; 2+2→+2, 4+4→+4).
    New "Merge energy ×" param (default 1.0), shown only for 2048.
  - **snake**: unchanged flat "Food energy (per apple)" restore, shown only for snake.
  - Both keep Energy budget + Energy/step. Param rows are gated by a `data-env`
    attribute; `training.js` shows only the rows matching the selected environment
    and re-evaluates on environment change. (`constraints_map.Survival`,
    `templates/index.html`, `static/training.js`.) Verified per-step energy deltas.
- Control Panel settings now **persist across refresh** — every input (environment,
  population, generations, constraints, all params, evolution controls, snake dims)
  is saved to localStorage on change and restored on load, with change/input events
  re-fired so the board, param visibility, PO rings, and grey-out re-sync.
  (`static/training.js`.)
- Control Panel: checking "Evolve hidden width (H)" now greys out (disables) the
  Net width input and shows an "evolved" tag next to it. (`templates/index.html`,
  `static/training.js`, `static/style.css`.)
- Investigated "microscope always shows 24 dims": confirmed NOT a bug — the
  Microscope payload's layer-row count exactly tracks the champion's H every
  generation (verified 24→26→25→23 with evolve_hidden; 24→17 with memory-rent).
  Without a size cost, dimension evolution is neutral (per EEC "size only changes
  under a cost"), so the champion keeps its starting H — reproduced H=24 for 8
  gens identically on the live server and headless. Guidance: add memory-rent
  (Cost strength ~0.03–0.05) to drive H down and watch the Microscope shrink.

## 2026-07-01 (Claude Code / Opus 4.8)
- Microscope now clearly indicates when it is showing **live training data** vs the
  illustrative demo: the badge reads green `● live · gen N` while a real genome is
  streamed in (`setExternal`/`setGenome`), grey `demo · gen N` otherwise. Confirmed
  the data path is real — during snake training it renders the actual 2-layer
  champion (`11→24→3`) with live weights/saturations, not the 4-layer demo. The
  wiring was already correct; this just makes it unambiguous. (`static/microscope.js`.)
- Added a floating **training metrics HUD** over the game canvas (draggable by its
  header, hideable via ×). Shows gen k/N, best/mean/median fitness, champion score,
  peak score (best-ever), steps, and H·leak·bits, plus a live **sparkline** of
  best (accent) and mean (grey) fitness across generations. Appears on training
  start; driven by the generation events in `training.js`. Frontend-only
  (`templates/index.html`, `static/style.css`, `static/training.js`).
- Energy constraint: the per-step energy drain is now a configurable **`step_cost`**
  parameter (default 0.01, range 0.01–5.0) instead of a hardcoded 1.0, applied in
  both snake and 2048. New "Energy / step" input appears under Parameters when
  Energy is checked; wired through `training.js → constraints_map.Survival`.
  Verified: energy_budget/step_cost = lifespan (100/5.0 → 20 steps, 100/1.0 → 100).
- **Stop is now responsive**: `_fitness` short-circuits to a sentinel once stopped,
  so the in-flight generation ends within ~one episode instead of grinding through
  every genome; the run breaks before emitting a half-evaluated generation and the
  `done` event reports the last fully-evaluated champion (no extra replay episode).
  Verified: stop() → done in ~0.0s.

## 2026-07-01 (Claude Code / Opus 4.8)
- Added an **Evolution** section to the Control Panel with controls wired through
  to the Evolver: "Evolve hidden width (H)" (toggle dimension mutation vs fixed H),
  "Mutation only (no crossover)" (asexual `sexual=False`), "Elite (keep best N)",
  and "Breed from top %". User knobs are the base; scarcity/reproduction-cost
  constraints still override them (laws win). `training.js` sends the new fields;
  `trainer.py` parses/merges them. Verified headless (mutation-only runs, scarcity
  override, H actually evolves).
- Better training status when the socket closes before any `started` event now
  reads "couldn't reach /train — restart the Flask server", since that means the
  running server predates the `/train` route (the usual cause of "disconnected").

## 2026-07-01 (Claude Code / Opus 4.8)
- **Wired the genreg-engine to train Snake and 2048 for real.** New Python package
  `genreg_train/` bridges the neuroevolution substrate (`project/genreg-engine-main`)
  to the GUI. Pieces:
  - `envs.py` — `SnakeEnv` (W×H, 11-feature obs, 3 relative actions) and
    `Game2048Env` (4×4, 16-feature obs, valid-move masking). World-consequence
    fitness (apples / merged tile score), JSON `render_state()` for the canvas.
  - `agent.py` — recurrent rollout (`fresh_state`/`rstep`) with per-step constraint
    hooks; downsampled replay frames.
  - `constraints_map.py` — maps all 12 constraints to real effects (energy/mortality
    survival budgets, time/perception post-score, efficiency/memory-rent engine
    costs, entropy state-leak, occlusion/noise obs perturbation, scarcity/
    reproduction-cost turnover, non-stationarity world drift). Same-axis pairs
    handled "swap, don't stack".
  - `trainer.py` — common-seed episodic fitness (fair per-generation evaluation),
    Evolver with recurrence + precision (+ dimension when a size cost is present),
    streams per-gen events (telemetry, champion genome, fresh-episode replay, PO).
  - `engine_api.py` (bridge), `run_headless.py` (CLI verifier), `SKELETON.md`
    (design spec), `REVISIONS.md` (5-pass hardening log).
  - Installed `numpy` (engine dependency) into the app's Python.
- **Flask `/train` WebSocket** (`app.py`) runs a Trainer in a background thread and
  streams events; `start`/`stop` protocol; terminal daemon untouched.
- **Frontend**: `static/training.js` (owns `/train`, assembles config, fans events
  to board/microscope/PO/HUD); Control Panel gains a Training section
  (Generations, Start/Stop, status) and constraint-parameter inputs that reveal per
  checked constraint; `ui.js` board renderers now draw live training frames
  (snake body/food, 2048 tiles) via `GENREG.board`; `microscope.js` accepts real
  streamed genomes (`GENREG.scope.setGenome`) and pauses its demo while training.
- **Verified**: snake learns (random ≈0 apples → champion ~19–29 apples, best
  fitness monotonic); 2048 beats random 3–6×; `/train` end-to-end smoke passes
  (started→generation with genome+replay→done, plus stop); 10/10 edge configs;
  memory-rent shrinks H. See `genreg_train/REVISIONS.md`.

## 2026-06-30 (Claude Code / Opus 4.8)
- Layout persistence: the two sidebar widths and the terminal dock height now
  save to a cookie (`genreg_layout`) on drag-end and restore on page load
  (clamped to the existing min/max), so panel sizes survive a refresh.
  (`static/ui.js`.)

## 2026-06-30 (Claude Code / Opus 4.8)
- Added a **PO Metrics** tab beside Microscope in the right sidebar. It shows a
  hand-rolled 3D plot (canvas-2D projection, no 3D lib) of the EEC "constraint
  cone": with no constraints it's a sphere (infinite possibility); each checked
  Constraint adds a ring marching along the +x axis (leading right) and morphs
  the sphere toward a cone (tip = the few probable survivors). Rings are driven
  entirely by the Constraints checkboxes (add/remove on toggle); the readout
  lists which constraints map to which rings. Fixed 3/4 view (static — no
  auto-rotate); renders on demand (tab shown, resize, toggle). No further PO logic yet, as
  specified. New file `static/po_metrics.js` (also owns the right-sidebar tab
  switching); touched `templates/index.html` and `static/style.css`.

## 2026-06-30 (Claude Code / Opus 4.8)
- Condensed the Constraints section (CSS only, nothing removed): each axis group
  now lays its checkboxes out in two columns with tighter padding/gaps and a
  smaller font; lone constraints span the full width so long labels don't wrap.
  (`static/style.css`.)

## 2026-06-30 (Claude Code / Opus 4.8)
- Microscope: click a neuron (any row of a weight matrix, or its saturation bar)
  to add it to a **Tracked neurons** watch list that updates live each frame
  (per-neuron saturation + mean incoming |w|). Tracked neurons are marked on the
  canvas (accent tick + row outline); each list entry has an × to stop tracking.
  Watch list resets when a new genome is loaded via `setGenome`. Touched
  `static/microscope.js`, `templates/index.html`, `static/style.css`.

## 2026-06-30 (Claude Code / Opus 4.8)
- Added a **Microscope** view (right sidebar, replacing the Inspector
  placeholder): a magnified "lab" canvas that renders a genome's weight matrices
  as a heatmap (value → diverging blue/warm color), per-neuron saturation bars,
  and a hover readout showing an individual weight's value + that neuron's
  saturation. A live mutation loop (relative, self-adapting `w += N(0,1)·ms·
  (|w|+ε)`, ~7 steps/sec) animates it so you can watch the genome change;
  Pause/Play + Step controls and a generation badge. Currently ILLUSTRATIVE (synthetic
  16→24→12→4 genome); exposes `GENREG.scope.setGenome(layers)` to feed a real
  genome once training is wired. New file `static/microscope.js`; touched
  `templates/index.html` (markup + script include) and `static/style.css`.
- Wired the Environment selector to render a game board on the canvas. Selecting
  **Snake** draws an empty W×H playfield grid; **2048**
  draws the classic 4×4 tile board (static sample position). Other environments
  (CartPole / Humanoid-v5 / Language) show a "no game board" placeholder. Boards
  are illustrative only — not yet driven by the engine.
- Added a **Game Controls** section to the Control Panel (below Constraints).
  Snake exposes Board Width / Board Height inputs (5–60 cells) that resize the
  board live; 2048 shows a "fixed 4×4, no controls" note; other environments
  show an empty-state note. The visible sub-panel follows the selected
  Environment. Touched files: `templates/index.html`, `static/ui.js`
  (board renderers + control wiring), `static/style.css`.

## 2026-06-30 (Claude Code / Opus 4.8)
- Added the EEC and genreg-engine projects into `project/` (`project/EEC-main`,
  `project/genreg-engine-main`) — kept as two separate projects.
- Added a Control Panel to the left sidebar GUI (static UI, not wired to the
  engine yet). Fields map to EEC's model: **Environment** (2048 / Snake /
  CartPole / Humanoid-v5 / Language), **Population** (int 1–1000), and
  **Constraints** checkboxes. Touched files: `templates/index.html` (form
  markup, side-head renamed to "Control Panel") and `static/style.css`
  (control-panel styles).
- Expanded the Constraints checkboxes to the full EEC catalogue (`docs/
  CONSTRAINTS.md`) and grouped them by capability axis: Survival (Energy /
  Mortality), Parsimony (Time / Memory-Rent), Active maintenance (Entropy),
  Persistence/memory (Occlusion / Noise), Selection (Reproduction-Cost),
  Diversity (Scarcity), Attention (Perception-Cost), Plasticity
  (Non-Stationarity). Same-axis swap pairs (Survival, Observation) are marked
  with a "swap" tag. Removed the Efficiency checkbox (not a catalogue axis;
  overlapped Parsimony). Each box has a hover tooltip describing its mechanism.
  (`templates/index.html`, `static/style.css`.)

## 2026-06-30 (Claude Code / Opus 4.8)
- Fixed the terminal getting stuck on "connecting…" with no shell loading. The
  changelog-button wiring in `static/app.js` used unguarded
  `getElementById(...).addEventListener`, which threw when the loaded template
  did not contain the button (a stale cached HTML shell), aborting the whole
  script before the WebSocket `connect()` ran. Added null-guards so optional UI
  can never block the terminal from connecting.
- Added `Cache-Control: no-store` to the `/` route in `app.py` so the browser
  cannot serve a stale HTML shell against freshly-served JavaScript.
- Restarted the Flask web server to load the current template. The terminal
  daemon and all running terminals were left untouched (they survive Flask
  restarts by design).

## 2026-06-30 (Claude Code / Opus 4.8)
- Added a "Changelog" button to the terminal dock action bar (GUI). It opens a
  modal that fetches and displays this CHANGELOG.md. Touched files:
  `templates/index.html` (button + modal markup), `static/app.js` (open/close
  wiring + `/changelog` fetch), `static/style.css` (modal styles), and
  `app.py` (new `/changelog` route serving CHANGELOG.md as plain text).
- Added this CHANGELOG.md to track changes in the shared GENREG workspace.
