# Changelog — DiffEvo

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.
Seeded 2026-07-05 from the main changelog (keyword split, best effort).

---

- **[2026-07-05] (Claude)** — **Stage 4 VALIDATED — blended rollout-survival landscape.** Pure
  rollout fitness bred hedging (open-loop 31.9%→23.1%; new §XI reward-hack: flatten toward
  marginals to score safely on drifted context). Blended fitness (teacher + own-output
  segments scored together, the DiffEvo unrolled lesson): open-loop HELD at 31.3% while
  own-output top-1 at R=8 went 15.0%→27.2% — **exposure gap 0.72→0.17 nats**, the quantified
  cause of generation soup cut 4×. Samples now contain real words at t=0.5. evaluate_rollout
  blended scoring in genreg_lm.py; complete campaign trail + ranked open levers in
  documentation/LM_STAGE1_FINDINGS.md.

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

