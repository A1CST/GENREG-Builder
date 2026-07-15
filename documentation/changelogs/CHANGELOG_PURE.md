# Changelog — PURE

Per-project log for the PURE line. Seeded 2026-07-14 from the master
CHANGELOG.md (all entries mentioning this project); new PURE entries go at
the top of the log below, and also in the master CHANGELOG.md.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-08] (Claude)** — CIFAR **round 2: 58.84% test with a PURE GENOME STACK** (user
  directive after the MNIST warm-start critique: no statistics anywhere in the classifier).
  Environment v5: ZCA patch whitening + NICHED bank evolution (55 niches = 45 class pairs +
  10 one-vs-rest, each with its own Fisher survival condition, round-robin decorrelated
  selection) -> 51 diverse detectors (vs 15 in round 1); regularised linear ceiling 58.83 ->
  **~67.0** test; unit-scale centroid floor 65.71. KEY DISCOVERY (cut + fixed with evidence):
  unit-variance standardisation of deep PCA tails makes the binary-genome landscape
  UNCLIMBABLE (chance at 2048 dims, 85% at top-256 — noise dims amplified to signal
  magnitude drown mutation-based search); fix = eigenvalue-proportional scaling
  (`build_features(pca_scale='eig')`), full ceiling kept, detector learnability restored
  (76% at 600 gens). Pure stack (`--pure --eig`): 10 detectors evolved from random (70-79%
  balanced) -> mixer genome -> algebraic fold (53.2 val) -> joint evolution +5.4 -> referees
  +0.9 test (margin 0.5 — they finally contribute, exactly as gate theory predicts for a
  below-ceiling head) = **58.84 test** (round 1 centroid-start: 58.69). Honest gap to
  ceiling ~8 points = the cost of purity at 8k gens. `_train_binary`/`train_detectors`/
  `train_pairwise` gained mag_scale; run_battery gained warm='fold'/pca_scale; champions
  `demo/cifar_genomes.pkl` (+ `_v5_pure.pkl` backup), bank `demo/cifar_detbank_v5.pkl`.
- **[2026-07-08] (Claude)** — NEW `/evolang/layers`: a genome layer/flow map for the WordPipe
  pipeline. Three horizontal bands (Structural / Semantic / Abstraction — per this session's
  "layer" framework: form vs. built-meaning-space vs. relations composed on that space) each
  positioned by which real generation-pipeline stage they wire into (Skeleton / Fill /
  Boundary), with a backbone showing the actual flow (Skeleton → Fill → Boundary → Output) and
  connector lines from each genome to its stage. Includes the 3 wired-but-experimental relation
  genomes AND the cut ones (sentiment/polysemy/register/analogy/open-obligation) as
  dashed/faded nodes with their one-line cut reason in the hover tooltip — same "report the cut
  list too" discipline as the rest of this project. NEW `static/evolang_layers.js` (static
  hand-authored data array + a lightweight hand-rolled SVG renderer, same pattern as
  diff.js/animation.js — NOT the PURE node-graph editor, this is read-only) and
  `templates/evolang_layers.html`; new route in `app.py` (`evolang_layers_page`); linked from
  the main `/evolang` page next to the WORDPIPE_FINDINGS docs link; two new CSS tokens
  (`--layer-semantic`, `--layer-abstraction`) in `style.css`. Verified end-to-end with a
  temporary local server on a spare port + headless Chrome screenshot (both bands/columns and
  the legend render correctly) — the live site's Flask process was never touched. **Flask
  restart needed** to see this on the live site (new route in `app.py`).
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
- **[2026-07-06] (Claude)** — **PURE: server-side video decode — ANY format (mkv/avi/mov/…).** The
  browser `<video>` element only decodes mp4/webm/ogg, so the user's MKV failed with "zero
  dimensions". Added a Flask endpoint `POST /api/pure/frames` that decodes server-side via
  **imageio + bundled ffmpeg** (installed `imageio`, `imageio-ffmpeg`; added to requirements.txt) —
  handles essentially any format. Extracts frames per the Data node's size/start/skip/max/gray and
  returns 0-255 pixel arrays. Client `acquireFrames` tries the server first, falls back to browser
  decode if the route is 404 (server not restarted), and caches results per File+params so a big
  video isn't re-uploaded each Run. Broadened the Data node's file picker to accept any video.
  Verified end-to-end: an MKV → `/api/pure/frames` → 6×16×16 grayscale frames with real content
  (not black). **Requires a Flask restart** (new route) — until then MP4 still works via the
  browser fallback.
- **[2026-07-06] (Claude)** — **PURE: fix black-frame video decode + Data-node frame preview.**
  User reported every target frame decoded black. Root cause: `drawImage` ran before the video
  painted the seeked frame (and a `t=0` seek never fires `seeked`). Rewrote `grabFrame` to wait for
  the frame to actually present (`requestVideoFrameCallback`, else double-rAF), nudge the seek so an
  event always fires, append the video (hidden) to the DOM and use a `willReadFrequently` context,
  and decode on `loadeddata`. Critically, if every decoded frame is uniform/black the decode now
  **errors clearly** ("frames decoded but every one is uniform/black…") instead of silently feeding
  black frames. Added a **Preview frame** button in the Data node Properties that decodes + shows
  the first frame so you can verify the video feeds correctly without running training. Browser-
  only path (not headless-testable); module load + synthetic path verified intact. Ready on next
  Flask restart.
- **[2026-07-06] (Claude)** — **PURE: per-frame Reconstruction panel (all frames, not just one).**
  New bottom panel (resizable, `data-resize="recon"`) that shows the best prediction for EVERY
  frame: targets on the top row, the model's reconstruction on the bottom row, aligned as N×N
  images — so there are as many output frames as input frames. `sampleModel` now computes the
  best genome's output for every dataset frame (cap 48) when the model's output is a square frame
  (the decoder; a latent-output encoder shows nothing meaningful). `drawRecon` renders the grid
  in a horizontally-scrollable canvas; cleared on Reset. Headless-verified: 2-model autoencoder on
  32 image frames → the decoder's grid drew all 32 frame reconstructions. Ready on next Flask
  restart.
- **[2026-07-06] (Claude)** — **PURE: video Data node now feeds REAL frames (was mock).** Fixed the
  trust gap the user flagged — a Data (video) node previously did nothing; the engine trained on a
  synthetic sine fallback regardless. Now `extractVideoFrames` actually decodes the selected video
  (seek → draw into an N×N canvas → grayscale/RGB pixels in [-1,1]) and those frames become the
  training dataset. `Engine.start` is async: if the source is a Data node it decodes first
  (status "decoding video frames…"), and errors clearly ("no video file selected" / "could not
  decode") instead of silently falling back to mock. The run status now says **"training on REAL
  video"** vs **"training on synthetic data"** so it's never ambiguous. Output plot upgraded: when
  the output is a square frame it renders **target vs reconstruction as side-by-side images** so
  you can watch the model actually rebuild the frame. Verified headlessly: synthetic path intact +
  labeled; video-without-file gives a clear error, no silent mock. NOTE: the actual browser video
  decode (video/canvas APIs) can't be headless-tested — that path is verified in-browser only.
  STILL TODO (deferred, acknowledged): genome visual showing ALL models in a chain; saving PURE
  runs to the Runs page with playback.
- **[2026-07-06] (Claude)** — **PURE: Autoencoder template + template picker.** The toolbar
  Template button is now a picker (select) with two options: **Basic model** (the single
  Synthetic→Input→Layer→Output→Fitness + Energy example) and **Autoencoder** — a wired
  encoder→decoder chain (16→12→**4** latent, reconstruct → **4**→12→16, reconstruct_source) so the
  decoder rebuilds the original through the bottleneck. `templateGraph(kind)` +
  `window.PureGraph.template(kind)`. Verified both load and parse (autoencoder → 2 models with the
  right per-model objectives). Ready on next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: `reconstruct_source` fitness objective (real autoencoder
  chains).** Added a third Fitness objective, `reconstruct_source`, plus carrying the ORIGINAL
  pre-encoder input (`x0`) through the model chain unchanged. So a decoder wired after an encoder
  can be scored against the source input rather than the latent it receives — making encoder(D→L)
  → decoder(L→D) a true autoencoder. `makeTarget()` centralizes target selection; every sample now
  carries `x0` (= x at model 0, preserved through pass-throughs). Headless-verified: encoder
  8→6→3 (reconstruct) then decoder 3→6→8 (reconstruct_source) reconstructs the original 8-dim
  input through the 3-dim bottleneck to 0.026 error. Ready on next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: Data (video) source node + multi-model chains via Fitness
  pass-through.** (1) New **Data** node (Data group): pick an mp4/webm/ogg file in Properties, set
  Assumed FPS / Start frame / Skip (stride) / Max frames / Frame size / Grayscale, and it shows
  the computed total frames live (source ≈ duration×fps → after stride → after max cap). File kept
  in a runtime map (not persisted); duration read from video metadata. (2) **Fitness node now has
  a pass-through `data` output** (was `fit`) that forwards the model's Output onward, so you can
  chain multiple models — each Input→…→Output→Fitness segment is its own model with its OWN
  fitness objective/metric. Engine rewritten for this: `parseModels` walks the chain into ordered
  segments; models train **sequentially**, each with its own population/genes/data — model k>0's
  inputs are the previous model's trained outputs (freeze-and-compose, GENREG-style). Readout shows
  `model i/N`; the genome visual fires the active segment (engine supplies its columns). Headless-
  verified: 2-model chain trained both (model 1 mse→0.069, model 2 mae→0.164, different fitness
  each), single-model unchanged, frame math correct. NOTE: the engine still generates training data
  from the Synthetic node / a default; feeding actual decoded video frames from the Data node is
  the next step (async decode). Ready on next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: Template button (correctly-wired example).** Toolbar gains a
  **Template** button that loads the canonical graph so the correct wiring is visible: Synthetic →
  Input → Layer → Output → **Fitness** (Output feeds Fitness so its objective can score the model),
  with an **Energy** constraint wired into a Fitness input port to show the constraint→fitness
  pattern. Laid out spatially (Fitness below Output, constraint below the layer) so the wires read
  clearly. `templateGraph()` + `window.PureGraph.template()`; `seedDefault` now delegates to it.
  Confirm-guarded (replaces current graph). Headless-verified: all 5 edges present, fitness
  dynamic ports dense, buildSpec accepts (arch 16→24→10, constraint reported). Ready on next Flask
  restart.
- **[2026-07-06] (Claude)** — **PURE fix: unwired layers no longer appear connected / get trained.**
  Both the genome visual (`orderedStructure`) and the engine (`buildSpec`) were collecting ALL
  input/layer/output nodes by type and sorting by x, so a Layer dropped on the canvas but never
  wired still showed as a column (with connections) and would be built into the network. New
  shared `wiredPath()` follows the actual edges Input → hidden layers → Output and returns only
  the connected chain in true data-flow order; both consumers now build from it. Engine also
  errors clearly ("wire Input → … → Output") when the chain doesn't reach the Output instead of
  training a disconnected graph. Bonus: layer order now follows wiring, not canvas x-position.
  Headless-verified: floating layer excluded from visual + arch; wiring order 6→8→5→6 honored over
  x-order; broken chain errors. Ready on next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: neurons colored by saturation.** During a run the genome
  neurons now color yellow (barely used, |activation|≈0) → red (near full saturation, |activation|≈1)
  instead of the teal firing ramp. Uses the ABSOLUTE activation (clamped to 1), so "saturated"
  means genuinely near ±1 rather than merely high relative to its column (`satColor` replaces
  `fireColor`; dropped the per-column max normalization for fill). Panel caption updated. Verified
  ramp (t=0 yellow → t=1 red). Ready on next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: genome visual shows evolved layer width live.** The engine
  now reports each layer's effective active-neuron count for the best genome (`dims` in the tick
  sample); `renderGenome` renders an evolve-units column at that effective width, so the column
  physically shrinks/grows during a run and the caption shows `N / max` (accent) when below max.
  Connections only touch the active units. Also improved the GA: integer genes (width, k) now seed
  the initial population across [1, units] instead of all pinning at max, so the width/k dimension
  is explored from generation 0. Headless-verified: layer column varied 3→6→4 / 12 across a run,
  back to 12 idle. Ready on next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: evolvable hidden-neuron count on Layer nodes.** Layer
  Properties gain an **Evolve unit count** checkbox; the Units field becomes the max (labeled
  "Units (max if evolving)"). Implemented as a width gene: weights are allocated up to the max, and
  a per-genome gene picks how many units are active — the rest are masked (output 0, contribute
  nothing to the next layer). Generalized the gene machinery so a layer can carry a width gene
  and/or a k gene (both mutate in ~unit steps, clamp in the forward pass); `layerOut` centralizes
  the per-layer path (activation → width mask → k-sparsity). Badge shows `≤N` when evolving.
  Headless-verified: fixed units 95% lower error, evolve-units 93%, evolve-units+evolve-k 76%,
  evolve-units+fixed-k 91% — all learn, genes compose. Ready on next Flask restart. (Note: genome
  visual still shows the max-width column; masked units read 0 → yellow.)
- **[2026-07-06] (Claude)** — **PURE: k-sparsity on Layer nodes (fixed or evolvable).** Layer node
  Properties gain a **k-sparsity** checkbox; when on, a **k (active units)** number field appears,
  or check **Evolve k** to make k a per-genome evolved gene instead of fixed (the k field hides
  when evolving). Implemented in the engine: after a layer's activation, `topK` keeps the k
  largest-magnitude units and zeros the rest. Fixed k reads the node value; evolve-k appends one
  gene per evolving layer past the weights, initialized to the set k, mutated in ~unit steps
  (`mutScale`), clamped to [1, units] per forward pass. Also generalized the property `when`
  mechanism to accept an array of conditions (all must hold) and made checkbox toggles re-render
  dependent fields. Layer badge shows `k4` / `k*` (evolving). Headless-verified: fixed k=3 learns
  to a higher error floor than unconstrained (6.2e-2 vs 3.0e-2 — the constraint genuinely bites),
  evolve-k adds 1 gene and still converges (84% lower). Not deployed live (user has a run in
  progress); ready on next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: genome neurons now show their connections.** The genome
  visual draws the edges between consecutive layers' neurons (fully-connected). Idle = faint
  structural skeleton (all connections, grey). During a run the engine exposes the best genome's
  weight matrices per layer (`wLayers` in the tick sample), and each connection is colored by
  weight sign (blue positive / red negative), opacity by |weight| strength, and brightened when
  its source neuron is firing — so you watch the actual evolved wiring change live. Connections
  drawn behind the neurons; capped columns connect their shown units by real unit index.
  Headless-verified: 120 faint lines idle → 120 weight-colored (81 blue / 39 red) mid-run → faint
  again after Reset. Live after the next Flask restart (not urgent — user has a run in progress).
- **[2026-07-06] (Claude)** — **PURE: Reset button + real-time genome firing.** Run panel gains a
  **Reset** (stops the engine, clears charts/metric, returns the genome visual to static, keeps the
  graph). The genome dot-columns now **fire in real time during a run**: the engine exposes the
  best genome's per-layer activations each generation (`activate()`, added to the tick sample),
  and `renderGenome(activations)` colors each dot by |activation| (dark→bright teal via
  `fireColor`), mapped to the correct unit even in capped/gap columns (`colLayoutG` now returns
  per-dot unit indices). `orderedStructure` reordered to data-flow order (input → layers-by-x →
  output) so columns align with the activation arrays. Headless-verified: dots color with 22
  distinct intensities mid-run; Reset returns them static. Live after the next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: engine wired up + Fitness node (the baseline GA runs).**
  Run/Stop now drive a real in-browser plain GA over the assembled graph: Synthetic data feeds
  the Input, flows through the hidden Layer(s) to the Output, and the population evolves toward
  the Fitness node's objective. New **Fitness node** (new "Objective" toolbar group) with a
  DYNAMIC input port set — it always keeps one open port, growing as you wire constraints in
  (wire the model Output into it plus any constraints; the engine finds nodes by type so exact
  placement is free). Implemented general dynamic-port support (`nodeInputs`/`normalizeDynamic` —
  incoming edges densify, spare "＋" port at the end) + selection-preserving `renderAll`.
  Engine (`Engine` in pure.js, exposed as `window.PureEngine`): seeded RNG, builds an MLP from
  input dims → layer units/activations → output classes, generates a dataset from the Synthetic
  node (waveforms windowed; images flattened), fitness = −MSE/MAE for objective reconstruct or
  predict_next, plain (μ,elite) GA with fixed mutation — NO energy/self-adaptation/evolved-acts
  yet (those are the constraints, added later and measured against this). Constraints wired into
  Fitness are collected and reported but do NOT yet affect the fitness math (baseline first).
  Left panel gains a live training readout: gen/best/mean-error, a best-fitness sparkline, and a
  best-output-vs-target plot. Seed graph now ships the full runnable flow
  (Synthetic→Input→Layer→Output→Fitness). Headless-verified: **error 1.29→0.10 (92% lower) over
  150 gens**; buildSpec + constraint collection correct. Live after the next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: Synthetic data-source node.** New "Synthetic" node (new
  "Data" toolbar group) that produces synthetic data to feed the model. Configured in Properties:
  a `kind` selector (sine · square · ramp · noise · image) whose parameters swap per kind —
  waveforms get frequency/amplitude/phase/samples, image gets pattern
  (gradient/checker/circle/stripes)/size/loop. Added a conditional-field mechanism (`when` on a
  prop) so only the relevant params show, and a **live preview** in the Properties panel (waveform
  sparkline or rendered image pattern) that updates as you edit. The Input node gained a `data`
  input port so Synthetic.out wires into it (Synthetic → Input). Toolbar generalized to render
  groups (Data/Structure/Constraints). Config + preview only — no engine consuming it yet.
  Headless-tested (node/ports, per-kind fields, groups). Live after the next Flask restart.
- **[2026-07-06] (Claude)** — **Standardized top navigation across the whole Flask app** (per
  user — nav buttons changed places / went missing per page). New shared Jinja partial
  `templates/_nav.html` renders ONE canonical set + order on every page:
  Build · Tree LM · DiffEvo · Animation · PURE · I2 · Runs · Docs. Each page includes it with
  `{% set nav_active = '<key>' %}` so the current page's link is marked active (highlighted,
  `aria-current`). Replaced the divergent hand-written nav in index/tree/diff/animation/pure/i2
  and added it to runs/docs (which had no project nav, only a Refresh button — kept). Dropped the
  inconsistent `target="_blank"` (index/i2 opened new tabs, others didn't) so switching is
  in-place everywhere; removed the ↗ arrows. Added `.runs-link.active` CSS. Verified: all 8
  templates render the identical 8-link nav in identical order with exactly one active. Live after
  the user's next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: resizable panels.** Drag-resizers on `/pure` — a column
  resizer on each side (Run panel width, Properties panel width) and a row resizer on the
  graph/genome split (drag to trade space between the node canvas and the genome visual). The
  graph canvas grows/shrinks as its neighbors move; the genome SVG now scales to fill its panel
  (flex column). Sizes clamp (sidebars 150-620px; genome min 60px, max = main height - 130) and
  persist to localStorage (`pure.size.left/right/genome`), reapplied on load. Handles highlight
  accent on hover/drag. `initResizers()` in `pure.js`, `.pg-resizer-col/.pg-resizer-row` CSS.
  Headless-tested (saved-size apply, drag delta, persistence). Live after the next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: genome visual re-added, now derived from the graph.** Split
  the `/pure` main area — node graph on top, a Genome dot-column panel below it (`renderGenome`
  in `pure.js`, `.pg-genome-panel` CSS). One column per dimensional structure node
  (Input/Layer/Output), left-to-right by position, node-dot style, colored to match the nodes;
  oversized layers cap with an ellipsis gap-marker while the caption shows the true unit count.
  Re-renders live on every node add/remove/resize (hooked into `save()`), so editing a node's
  units/dims updates the organism immediately. Headless-tested (seed columns, live layer add,
  ordering). Live after the user's next Flask restart.
- **[2026-07-06] (Claude)** — **PURE pivot: node-graph model assembler.** Rebuilt `/pure` as a
  drag-and-drop node editor (`static/pure.js` rewritten, `pure.html` restructured, graph CSS in
  style.css). You assemble a model by dropping nodes and wiring ports: left panel reduced to
  Run/Stop + global run settings (population/generations/seed); centre is the graph canvas
  (pan by dragging background, drag node headers to move, drag output→input port to wire, click a
  wire to delete); right panel is the Properties of the selected node. Dims/units/activation and
  every constraint's settings live INSIDE their nodes. Node catalog is data-driven: structure
  nodes (Input·Layer·Genome·Output) + the 12 constraint nodes from the user's canonical list —
  Energy, Temporal Budget, Consequential Drive, Capacity Cost, Observation Cost, Prediction Error,
  Information Gradient, Stimulus Stagnation, Predictive Variance, Homeostatic Proximity,
  Consolidation Threshold, Hazard Signal — each with typed, editable properties (e.g. Energy:
  cost-per-action / recover-on-consequence / decay / floor / ceiling). Constraints are wireable
  but NOT yet backed by any engine (per user: don't wire them up yet). Graph persists to
  localStorage; `window.PureGraph.getGraph()` serializes it for the eventual GA. Supersedes the
  previous non-wired genome viz + checkbox constraints panel (those were placeholders). Headless
  tested. Live after the user's next Flask restart.
- **[2026-07-06] (Claude)** — ~~PURE: slim right-hand constraints panel~~ (SUPERSEDED same day by
  the node-graph pivot; the checkbox list was replaced by constraint nodes with the correct list). New right
  sidebar on `/pure` listing the GENREG mechanisms that separate the framework from a plain GA,
  each an activatable checkbox, grouped: Fitness landscape (soft fitness, multiplicative,
  EMA smoothing, position-varying ground truth), Metabolism & selection (energy homeostasis,
  tournament+maturation gate, self-adaptive mutation, mutation floor, late anneal), Architecture
  primitives (per-neuron evolved activation, structural mutation, bootstrap/two-phase), Landscape
  design (basin/constraint deformation, novelty/fitness-sharing). 14 toggles, all OFF by default
  (PURE = the naked baseline). Data-driven render in `pure.js`, per-checkbox hint tooltips,
  state persisted to localStorage, header badge shows "baseline"/"N active", and a
  `pure:constraints` DOM event + `window.PureConstraints` API (`get/isActive/active/list`) expose
  the active set for the eventual wiring into the GA. CSS added for the slim panel (214px).
  Headless-tested (render, toggle, persist, count, API). Live after the user's next Flask restart.
- **[2026-07-06] (Claude)** — **PURE: genome visualizer, stage 1 (non-wired FFN skeleton)**
  (per user — wants a visual of the best genome when building models). New `static/pure.js` +
  `templates/pure.html` main card: renders the genome as three columns of circle "neurons" —
  Encoder · Hidden · Output, one column per layer — in the classic node-dot style, NO edges yet
  (wiring + per-genome weight coloring are the next stages; node geometry is retained on
  `host._nodes` for that). SVG, theme-var colored (encoder=tlm-s1, hidden=accent, output=tlm-s2),
  scales to container. Layer sizes are live-editable from three sidebar inputs
  (`PureGenome.setShape({encoder,hidden,output})`); oversized layers cap at ~21 visible dots with
  a centered ellipsis gap-marker while the caption still reports the true unit count (a 4096-input
  layer draws ~20 dots, not 4096). Headless render-tested (node counts, gap-marker, single-x
  columns, ascending y, captions). Live after the user's next Flask restart.
- **[2026-07-05] (Claude)** — **PURE: new program page scaffold** (per user — pivot to a
  baseline-first campaign). `/pure` route in `app.py` + `templates/pure.html`: blank page in the
  house style (topbar + nav, config sidebar placeholder, empty main card) with the shared
  daemon-backed terminal dock (xterm/termdock/app.js/agentpanel). PURE will hold the very first
  baseline model — a textbook GA with nothing added — that every GENREG bell and whistle gets
  measured against, added one at a time. No model/WS backend yet, deliberately (user: blank page
  then stop). PURE link added to the build page nav. Live after the user's next Flask restart.
