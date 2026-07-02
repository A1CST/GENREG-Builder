# Changelog — GENREG

Tracks changes made by AI assistants working in this directory.
Multiple AIs share this workspace, so every change should be logged here with
date, author, and a short description. Append new entries at the top of the
log below; don't rewrite existing entries.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

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
