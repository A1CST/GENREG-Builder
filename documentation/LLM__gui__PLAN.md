# GENREG GUI — Design Plan

PyQt6, modular tab-based observability and control surface for any
GENREG model. Text-only data for now. No code in this doc; this is the
contract before implementation.

---

## 0. Goals (in priority order)

1. **Model-agnostic.** The GUI must work for tokenizer, embedding,
   attention, LM, optimizer — anything we build in `components/`.
   Adding a new model type is one new file in `adapters/`.
2. **Full config control.** Every mutatable parameter (POP_SIZE,
   ENERGY_DECAY, mut_rate, layer dims, fitness weights, etc.)
   editable before/between training runs.
3. **Inference observability.** Watch the model think on a single
   input — internal activations, cascade state, hash bits, energy
   distribution, fitness trajectory. Plateau identification at a
   glance.
4. **Run labeling and persistence.** Tag every run as
   training/test/inference, name it, save it, compare it to others.
5. **Modular code.** Tabs, adapters, plots, persistence are all
   independent. No tab can break another tab.
6. **Never blocks the UI.** Training runs in worker threads; GUI
   stays responsive.

Non-goals for v1: no image/audio data, no multi-model comparison
view inside a single tab, no remote control / network. Add later if
needed.

---

## 1. Architecture overview

Four clean layers:

```
┌───────────────────────────────────────────────┐
│  PyQt6 GUI shell (main window + tabs)         │
└────────────────────┬──────────────────────────┘
                     │ signals/slots
┌────────────────────▼──────────────────────────┐
│  RunState (single source of truth)            │
│  - active model + adapter                     │
│  - current population                         │
│  - training history (metrics over time)       │
│  - config object                              │
└────────────────────┬──────────────────────────┘
                     │
┌────────────────────▼──────────────────────────┐
│  Adapter layer (one file per model type)      │
│  - config_schema()                            │
│  - dataset_options()                          │
│  - load / save / step                         │
│  - infer(input, genome_idx)                   │
│  - introspection hooks                        │
└────────────────────┬──────────────────────────┘
                     │
┌────────────────────▼──────────────────────────┐
│  Model code (existing components/*)           │
│  - unchanged; adapters wrap them              │
└───────────────────────────────────────────────┘
```

The GUI never imports a model directly. It always talks to an
adapter. Adapters expose a fixed protocol (described in section 4).

---

## 2. File / folder layout

```
gui/
├── PLAN.md                       ← this doc
├── main.py                       ← entry point, builds main window
├── shell/
│   ├── __init__.py
│   ├── main_window.py            ← QMainWindow + tab container
│   ├── run_state.py              ← shared state, signals
│   ├── worker.py                 ← QThread base for training/inference
│   └── theme.py                  ← single light/dark style
├── tabs/
│   ├── __init__.py
│   ├── base_tab.py               ← QWidget base with common helpers
│   ├── setup_tab.py              ← model picker, checkpoint loader, run label
│   ├── config_tab.py             ← auto-generated form from adapter schema
│   ├── inference_tab.py          ← single-input observability
│   ├── population_tab.py         ← view all genomes
│   ├── training_tab.py           ← start/pause/stop training, live metrics
│   ├── charts_tab.py             ← all plots (fitness, energy, mutation, etc.)
│   ├── logs_tab.py               ← raw stdout/stderr stream
│   └── runs_tab.py               ← list, load, compare past runs
├── adapters/
│   ├── __init__.py
│   ├── base.py                   ← AdapterProtocol abstract class
│   ├── tokenizer_g.py            ← wraps tokenizer G
│   ├── embedding.py              ← wraps embed_03 / future
│   ├── lm_a101.py                ← wraps A_101 monolithic
│   └── README.md                 ← how to add a new adapter
├── widgets/
│   ├── __init__.py
│   ├── config_form.py            ← schema → form widget builder
│   ├── plot_panel.py             ← matplotlib/pyqtgraph plot widget
│   ├── tensor_view.py            ← heatmap / bar chart for tensor inspection
│   ├── token_stream.py           ← display token IDs colored by frequency
│   └── log_view.py               ← scrolling log with filters
├── persistence/
│   ├── __init__.py
│   ├── run_io.py                 ← save/load run bundles
│   └── runs/                     ← actual saved runs live here
└── tests/
    ├── test_run_state.py
    └── test_adapters.py
```

All tabs are siblings under a `QTabWidget`. Adding a tab is one
import + one line in `main_window.py`. Adding an adapter is one file
in `adapters/` registered in a small registry dict.

---

## 3. Tabs (the user-facing surface)

Header tabs across the top of the window. Each tab is independent —
no tab knows about the others' UI; they talk only via `RunState`
signals. Order is the typical workflow.

### 3.1 Setup tab

Where you define what you're working on.

- **Model type dropdown** — populated from registered adapters
  (tokenizer_g, embedding, lm_a101, ...).
- **Checkpoint path** — file picker + "load" button. Loads the
  population into `RunState`.
- **Dataset selector** — dropdown of datasets the adapter declares
  it supports (e.g., wikitext-103-words, custom-stream-from-file).
- **Dataset input mode** — radio buttons: "stream", "batch", "single
  example" — only the modes the adapter supports are enabled.
- **Run mode** — radio: training / test / inference. Determines
  which other tabs are enabled.
- **Run label** — text field. Required for training/test runs;
  optional for inference. Becomes the saved-run filename.
- **Run notes** — multiline text area for free-form notes saved
  with the run.
- **"Begin run"** button — only enabled once all required fields
  are populated.

This tab is the gatekeeper. Until a model + checkpoint + mode are
chosen, all the other tabs are disabled.

### 3.2 Config tab

Every mutatable parameter, auto-generated from the adapter's
config schema. Sections grouped by concern:

- **Population & selection** — POP_SIZE, SURVIVAL_PCT, maturation
  gate on/off.
- **Energy** — DECAY, GAIN, FLOOR, E_MAX, regen, costs/bonuses
  specific to that model.
- **Fitness weights** — every term in the fitness equation gets a
  numeric input (with reset-to-default).
- **Mutation** — initial mut_rate / mut_scale, anneal schedule.
- **Architecture** — dimensions (H, D, V_TOK, etc.) — only editable
  before training; readonly once a checkpoint is loaded.
- **Training schedule** — N_GENERATIONS, BATCH_SIZE, LOG_EVERY,
  ANNEAL_AFTER.

Each field has:
- Label (param name)
- Numeric/text/dropdown input
- "?" tooltip with the description from the schema
- "Reset" button → adapter default
- "Save preset" / "Load preset" at the bottom

The schema lives in the adapter, so adding a parameter to the model
just means adding it to the schema — the form rebuilds itself.

### 3.3 Inference tab

The "watch it think" view. Most important tab for plateau detection.

Layout (top to bottom):
- **Input box** — multi-line text input. Below it, an adapter-
  declared button: "Tokenize" / "Encode" / "Predict next".
- **Output box** — the model's output (token IDs, predicted text,
  embedding vector preview, etc.). Format depends on adapter.
- **Internal-state panels** (collapsible, populated by adapter
  introspection hooks):
  - Cascade state: bar chart of `last`, `momentum`, `slow`,
    `ultra`, `epoch` per neuron
  - Activation outputs: heatmap of (neuron × time)
  - Hash bits (if hash output): 9-bit display per word
  - Trust mixer weights (if multi-channel): bar chart
  - Per-genome fitness on this input: histogram
- **Genome selector** — dropdown to switch which genome you're
  watching. Default = best by current fitness metric.
- **"Step" / "Run continuously"** — for stream models, step one
  word at a time and see state evolve.

Plateau identification: if you step through a stream and the cascade
state stops changing, you see it visually. If output IDs collapse to
a few values, the token-stream widget shows it.

### 3.4 Population tab

View all N genomes simultaneously.

- **Sort by** dropdown: fitness, energy, age (gens since birth),
  mut_rate, mut_scale, custom metric.
- **Filter** input: "fitness > 0.3", "alive only", "parents only".
- **Table** — rows = genomes, columns = key scalars.
- **Selecting a row** → sets that genome as active in Inference tab.
- **Bulk action** buttons: kill all, reset energy, freeze top-k.

This is for diagnosing population health: if 90% of genomes have
identical fitness, mode collapse. If energy distribution is bimodal,
selection pressure is split.

### 3.5 Training tab

Start/pause/stop a training run + live metric strip.

- **"Start" / "Pause" / "Stop" / "Save checkpoint"** buttons.
- **Generation counter + ETA**.
- **Live metric strip** at top: best fitness, mean fitness, starved
  count, energy mean — updated every gen.
- **Per-gen log line stream** (mini, scrollable), the same lines
  that go to the run's log file.
- Status indicator: training / paused / done / errored.

The training runs in a QThread. Stopping is graceful — it finishes
the current gen, then saves before exiting.

### 3.6 Charts tab

Single tab containing all the plots. Plots are configurable; user
picks which metrics to display from a sidebar.

- **Sidebar** (left) — checklist of available metrics:
  - best/median/min fitness
  - per-fitness-term breakdown (e.g., discrimination, consistency)
  - energy mean / std / starved count
  - mut_rate distribution
  - per-channel trust weights (if multi-channel)
  - any custom metric the adapter exposes
- **Plot area** (right) — N stacked time-series plots, one per
  selected metric. Each plot has its own y-axis. X-axis is shared
  (generation number).
- **Window controls**: zoom, pan, log/linear toggle, export PNG.
- **Compare runs**: overlay one or more saved runs as ghosted lines.

Library: `pyqtgraph` for fast updates during training (matplotlib
becomes laggy after ~10k points).

### 3.7 Logs tab

Raw log stream + filters.

- **Log view** — scrolling text widget showing the run's log file
  in real time.
- **Filter dropdown**: ALL / GEN_LOGS / WARNINGS / ERRORS /
  CHECKPOINTS.
- **Search box** — find text in log.
- **"Open in editor"** button — open the log file externally.

### 3.8 Runs tab

List, load, compare past runs.

- **Run list table** — columns: label, type (training/test/
  inference), model, dataset, started, gens completed, best score,
  status.
- **Filters**: by model, by date, by score range, by label substring.
- **Selecting a run** → "Load checkpoint" / "View log" / "Compare
  in charts" buttons become active.
- **Multi-select** → "Compare in charts" overlays the runs in the
  Charts tab.
- **Delete** with confirm.

---

## 4. Adapter protocol

Every model adapter implements one Python class with this contract.
The GUI requires nothing more, nothing less.

```python
class AdapterProtocol:
    NAME: str          # e.g., "Tokenizer G"
    KIND: str          # "tokenizer" / "embedding" / "lm" / etc.

    # --- Configuration ---
    def config_schema(self) -> list[ConfigField]:
        """Return list of ConfigField describing every mutatable param."""

    def default_config(self) -> dict:
        """Default values — used to seed the Config tab."""

    # --- Datasets ---
    def dataset_options(self) -> list[DatasetSpec]:
        """Datasets this model can be fed. Each spec defines name,
        loader function, supported input modes (stream/batch/single)."""

    # --- Lifecycle ---
    def init_population(self, config: dict, device: str) -> Population:
        """Construct a fresh population from a config dict."""

    def load_checkpoint(self, path: str, device: str) -> Population:
        """Load a saved population from disk."""

    def save_checkpoint(self, pop: Population, path: str) -> None: ...

    # --- Training ---
    def train_step(self, pop: Population, batch) -> StepResult:
        """One generation of training. Returns metrics dict."""

    # --- Inference ---
    def infer(self, pop: Population, input_data, genome_idx: int = 0)
            -> InferenceResult:
        """Single-input inference. Returns output + introspection
        bundle for the Inference tab."""

    # --- Introspection ---
    def metric_names(self) -> list[MetricSpec]:
        """List of metrics the model produces. Charts tab uses this
        to populate the sidebar."""

    def state_panels(self) -> list[PanelSpec]:
        """Internal-state panels for the Inference tab. Each panel
        spec declares: name, kind (heatmap / bar / histogram /
        token_stream), data accessor function."""
```

`ConfigField`, `DatasetSpec`, `StepResult`, `InferenceResult`,
`MetricSpec`, `PanelSpec` are simple dataclasses defined in
`adapters/base.py`. They're documented inline.

A new model: write one file in `adapters/`, register its class in a
small dict in `adapters/__init__.py`. The GUI auto-discovers it.

---

## 5. RunState (the shared state object)

A single `RunState` instance lives in the main window and is passed
to every tab. Tabs read/write through it; tabs subscribe to its
signals.

```python
class RunState(QObject):
    # --- Data ---
    adapter: AdapterProtocol | None
    population: Population | None
    config: dict
    history: list[dict]           # one entry per gen
    log_lines: list[str]
    run_label: str
    run_mode: str                 # "training" | "test" | "inference"
    dataset: DatasetSpec | None

    # --- Signals (for cross-tab communication) ---
    adapter_changed = pyqtSignal()
    population_loaded = pyqtSignal()
    config_changed = pyqtSignal()
    gen_completed = pyqtSignal(int)         # gen number
    metric_updated = pyqtSignal(str, float) # metric name, value
    log_appended = pyqtSignal(str)
    inference_done = pyqtSignal(object)     # InferenceResult
```

Tabs subscribe to relevant signals in their `__init__`. The shell
constructs the `RunState` and injects it into every tab. Tabs never
hold references to each other — only to the state.

---

## 6. Threading model

- **Main thread**: only the Qt event loop and UI updates.
- **TrainerThread**: runs training generations. Emits `gen_completed`
  signals (Qt-safe queued connections) after each gen.
- **InferenceThread**: handles one-off inference calls so a 5-second
  forward doesn't freeze the UI.
- **PersistenceThread**: writes checkpoints and logs in the
  background.

All worker threads inherit from `shell/worker.py`'s `BaseWorker`,
which provides start/pause/stop semantics and graceful shutdown
on window close.

---

## 7. Persistence format

Every run is a folder under `gui/persistence/runs/<label>_<timestamp>/`:

```
runs/embed_03_2026-04-14_19-12/
├── meta.json           # label, type, model, dataset, start time
├── config.json         # full config used
├── checkpoint.pkl      # final population (and intermediates)
├── history.json        # per-gen metrics
├── run.log             # stdout/stderr
└── notes.md            # user notes from setup tab
```

Loading a run = read meta, restore config, optionally load
checkpoint + history. Comparing runs in charts = reading their
history.json files.

---

## 8. Build phases

Implementation in stages, each leaves a working partial product.

**Phase 1 — Skeleton** (no model integration, just UI):
- Main window + empty tabs (with placeholder labels)
- RunState class + signal plumbing
- Basic theme

**Phase 2 — First adapter** (tokenizer G as the canary):
- Build adapter for tokenizer G
- Setup tab fully wired
- Config tab auto-generates from G's schema
- Inference tab loads a checkpoint and runs single-word inference
- Population tab reads pop and shows fitness

**Phase 3 — Training loop**:
- Training tab + TrainerThread
- Live metric updates via signals
- Logs tab streaming
- Save/load checkpoint working end-to-end

**Phase 4 — Charts**:
- Charts tab with pyqtgraph
- Metric registration via adapter
- Live updates during training
- Compare-runs feature

**Phase 5 — Runs management**:
- Runs tab with list / load / delete
- Persistence layer save format finalized
- Comparison overlay in Charts tab

**Phase 6 — Polish**:
- Inference tab introspection panels (heatmap, bars, token stream)
- Population tab filters and bulk actions
- Theme refinement
- Error dialogs with stack traces

**Phase 7 — Second adapter** (proves modularity):
- Embedding adapter (embed_03)
- Verify nothing in the GUI needed to change

---

## 9. What the user sees on day one (after Phase 6)

Open the app → Setup tab. Pick "Tokenizer G" from model dropdown,
load `tokenizer_gen_03000.pkl`, label this run "G_inspect_v1",
choose run mode "inference", click Begin.

Inference tab activates. Type "hello world test" in input. Click
"Tokenize". Output box shows `[421, 437, 412]`. Below: cascade
state bars updating left-to-right as the stream plays. Hash bits
panel shows the 9 binary decisions per word. Histogram of the 500
genomes' outputs on this exact input shows clusters.

Switch to Charts tab. Sidebar lists 14 metrics from G's adapter.
Select "fit", "discrimination", "consistency", "starved", "best
genome bin usage". Five plots stack. Load a previous training run
from Runs tab → its history overlays as ghosted curves.

Switch to Config tab. Bump POP_SIZE from 500 to 1000. Switch run
mode to "training". Click Start in Training tab. Watch live
metrics. Plateau detection: best-fit line flattens for 500 gens →
visible immediately, decide to stop.

---

## 10. Risks / open questions

- **PyQt6 + GPU torch**: must ensure CUDA tensors are detached and
  copied to CPU before any QImage / matplotlib call. Easy to crash.
- **Live plot performance**: pyqtgraph handles 100k points fine but
  needs care if we plot per-genome history (300 lines × 10k gens).
  Plan: aggregate to mean + min/max bands when over a threshold.
- **Adapter schema enforcement**: schema is duck-typed Python. Need
  validation that a new adapter actually implements all required
  methods. Add `assert isinstance(...)` in registry.
- **Training pause is hard**: can't yank a generation mid-step.
  Pause = "stop after current gen finishes" only.
- **Long inference**: full corpus tokenization (the 9M word job)
  shouldn't run via the GUI; it's a script. GUI stays for
  interactive observability.

---

## 11. Out of scope for v1

- Image / audio data
- Multi-model side-by-side single-tab comparison (use Runs tab
  + chart overlays instead)
- Remote / distributed training control
- Genetic crossover visualizations (later)
- Real-time mutation editing during training (later)

---

## 12. Open call

Things I want sign-off on before writing code:

1. Confirm the adapter approach is the right modularity boundary.
   Alternative: tabs themselves are model-specific plugins. Adapter
   is cleaner IMO; tabs stay one shape.
2. Confirm pyqtgraph for plots vs matplotlib. pyqtgraph is faster
   but uglier; matplotlib is prettier but slower.
3. Confirm phase order. I'd start with Phase 1 + 2 (skeleton + one
   adapter) and ship that as the first usable version.
4. Anything that should be in v1 that I have in "out of scope"?
