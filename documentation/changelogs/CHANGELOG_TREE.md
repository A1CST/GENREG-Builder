# Changelog — Tree LM

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.
Seeded 2026-07-05 from the main changelog (keyword split, best effort).

---

- **[2026-07-06] (Claude)** — Tree LM page: **copy refresh to reflect byte AND word/token level**
  (user: "reflect what we are actually doing now"). The page was live and functional but its
  copy was byte-only and outdated. Fixed: brand subtitle "byte-level" → "byte & word-level";
  "Context window (bytes)" → "(bytes / tokens)" with a word-mode hint; routing-layers hint
  generalized (hardcoded "256 bytes" → "vocab / branch^layers", + the word-level layers-0
  collapse warning); cluster-split, generate, routing-inspector, and icicle card copy updated
  to say token/vocabulary not byte. treelm.js: the icicle x-axis was hardcoded to /256 (word-
  mode trees rendered off the right edge) — now derives the vocab span from the widest node
  and labels the axis "byte 0..255" or "token 0..N" by mode; encoder status "next-byte" →
  "next-token (ridge / nearest-centroid)"; tooltip "bytes" → "tokens". No functional/training
  change; the page still does live GPU training, generation, sweeps, and the routing inspector.
  Static+template only — reload the browser.

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

