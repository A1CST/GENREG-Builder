# Changelog — Build interface & shared infrastructure

Project-scoped view. Convention: every change is logged in the MAIN
CHANGELOG.md (repo root) AND appended here when it touches this project.
Seeded 2026-07-05 from the main changelog (keyword split, best effort).

---

- **[2026-07-06] (Claude)** — **PER-LAYER ("tissue") constraints — new engine + validated.**
  User's idea: evaluate each constraint against its WIRED LAYER'S activations instead of the
  whole network, so layers face different survival conditions like tissues — same tournament /
  energy / whole-organism reproduction, landscape just becomes locally uneven. Built
  `genreg_train/pure_engine.py` (multi-layer evolved MLP, per-neuron activations, energy
  homeostasis; the evolving backend PURE's node graph will feed). Decisive 4-condition A/B
  (next-char, matched compute): Energy→L1 crushes L1 power to 0.039 (8x below control) while L2
  stays 0.34; SWAP the wire (Energy→L2) and the collapse follows to L2 (0.103) while L1 stays
  high — differentiation is CAUSAL, controlled by the wiring. Global eval compresses both
  layers uniformly (no differentiation). Held-out top-1 preserved at 20.8% across all
  constrained runs (cooperation intact). Consequential Drive holds a wired layer's dead-neuron
  fraction at 0%; its absence + energy starvation grows 18.8% dead (constraints compose as
  predicted). Card + full table: documentation/PURE_PER_LAYER_CONSTRAINTS.md.

- **[2026-07-05] (Claude)** — **Fixed the Claude terminal button** — it typed
  `[?1;2cclaude --dangerously-skip-permissions` (PowerShell "Missing type name after '['"
  parse error). Cause: the button blind-fired on a 600 ms timer, racing the terminal's
  device-attributes handshake — the shell queries ESC[c during startup, xterm.js auto-replies
  ESC[?1;2c, and that reply landed on the not-yet-ready command line as literal text, then our
  command concatenated onto it. Fix (app.js): the command now waits for the actual PowerShell
  PROMPT in the terminal output (ANSI-stripped tail match, 5 s blind fallback), sends Escape
  first (PSReadLine RevertLine clears any stray reply chars), then types the command. Verified
  in a DOM harness: no input before the prompt, escape-then-command after it, fallback fires.
  Static-only — reload the browser.

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
- **Restart button on both node GUIs** (`i2_node.py`, `i2_child.py`):
  top-bar "Restart" with confirm → existing `relaunch()` (os.execv), so a
  node picks up new files on disk without touching a terminal. Server
  startup in both scripts now retries binding (20 × 0.5s) because after
  execv the new process can race the old one for the port — without this
  a restart could come back dead. Verified headless: instance B started
  while A held :8800, A killed, B was serving within ~3s. GUI button
  itself needs a display (not exercised headless). dist/ refreshed.

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
- **New I2 page** (`/i2`, linked "I2 ↗" from the main topbar): shell for a
  new program sharing this Flask server — a full-width canvas
  (`static/i2.js`, placeholder grid, exposed as `window.I2`) above the same
  daemon-backed terminal dock as the main page (reuses `static/app.js`
  verbatim — same element IDs, same `/ws` bridge, so tabs are shared with
  the main page and survive Flask restarts). Dock height persists in its
  own `i2_layout` cookie. **Requires a Flask restart** (new route).

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
## 2026-07-03 (Claude Code / Fable 5)
- **Welcome page shows the genome's current wire size.** Added
  `GENOME_WIRE_BYTES`/`GENOME_SIZE_STR` (computed from the actual genome)
  and a `{size}` placeholder on the home page: "the genome is 633 KB on
  the wire — downloaded once, then every page decodes locally." Computed
  dynamically so it stays accurate if the genome changes. Republished
  live to the primary (verified: decodes, signature valid, shows 633 KB);
  source synced (--no-restart) + dist.

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
- **Repository published to GitHub** (user request): added `LICENSE` (GNU AGPL-3.0
  — strict copyleft covering network use) and committed the full working tree for
  publication as the public repo `GENREG-Builder` under the `A1CST` account.

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

