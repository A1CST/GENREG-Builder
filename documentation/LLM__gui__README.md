# GENREG GUI

PyQt6 observability + config surface for any GENREG model.

## Run it

From the LLM directory:

```bash
python -m gui.main
```

(or `python gui/main.py` from the LLM directory).

On startup, a self-check (`gui/init_check.py`) runs 9 sanity tests:
PyQt6, pyqtgraph, torch, RunState, MainWindow, adapter registry,
log parser, persistence, corpus data. If any fail, a dialog appears
with a summary + "Show details" containing every check result and
tracebacks for the failures. Failures are also written to
`gui/crashlog.txt`. The GUI continues launching either way so you
can investigate.

You can also run the self-check standalone:

```bash
python -m gui.init_check
```

Exits 1 if any check fails, 0 otherwise.

## What's here (Phases 1 + 2 — current)

Working:
- **All 8 tabs build and switch.**
- **Setup tab:** pick adapter, browse + load checkpoint, choose dataset
  + input mode, set run mode and label.
- **Config tab:** auto-generated form from the adapter's schema.
  Every field has a tooltip with description, type-correct widget
  (spinbox / checkbox / dropdown), reset-to-defaults.
- **Inference tab:** type text, pick genome, run, see token IDs +
  per-step internal state panels (cascade, hash bits, mix weights).
- **Population tab:** sortable table of per-genome scalars.
- **Logs tab:** live filterable log stream from RunState.
- **Charts tab:** pyqtgraph plots, metric checklist sidebar
  (no live data yet — needs Phase 3 training loop).
- **Training tab:** controls scaffolded, worker not wired (Phase 3).
- **Runs tab:** browses `gui/persistence/runs/` (load/delete in
  Phase 5).

Adapters available:
- **Tokenizer G** — wraps `components/tokenizer/genreg_tokenizer_G.py`.
  Loads any `tokenizer_gen_*.pkl` containing `hash_proj`. Inference
  is stream-mode with cascade-state introspection.

## Quick try

1. Run `python -m gui.main`.
2. Setup tab: pick **Tokenizer G**, browse to
   `components/tokenizer/checkpoints_G/tokenizer_gen_03000.pkl`,
   click Load. Pick run mode = inference. Click Begin (label is
   optional in inference mode).
3. Inference tab: type any sentence, click Run. See token IDs +
   internal-state panels.
4. Population tab: sortable table of all 500 genomes.
5. Config tab: every G hyperparameter is editable with descriptions.

## Architecture

See `PLAN.md` for the full design. Quick map:
- `shell/` — main window, RunState, theme, worker base.
- `tabs/` — each tab is a `BaseTab` subclass. No tab knows about
  any other; all coupling is via RunState signals.
- `adapters/` — one file per model type. `base.py` defines the
  contract; `__init__.py` is the registry. Adding a model = one
  new adapter file + one line in the registry dict.
- `widgets/` — reusable widgets (config form, plots, etc.) — most
  built into the tabs in v1 for simplicity; will be extracted in
  Phase 6 if reuse demands it.
- `persistence/` — runs/ folder format (Phase 5).

## What's coming (Phases 3 – 5) — DONE

- **Phase 3:** ✅ TrainerThread (subprocess), Training tab launches
  the standalone script, log lines stream to Logs tab + Charts,
  metrics parsed via `adapter.parse_log_line()`.
- **Phase 4:** ✅ Charts overlay — select runs in Runs tab → "Compare
  in Charts" → ghosted dashed lines per metric. Clear via sidebar btn.
- **Phase 5:** ✅ Run persistence: every training run creates
  `gui/persistence/runs/<label>_<ts>/` with meta.json, history.json,
  run.log, copied checkpoint.pkl. Runs tab lists, refreshes, deletes,
  and feeds Charts overlay.

## Training in the GUI

Set up a training run:
1. Setup tab: pick adapter, set run mode to **training**, give it
   a label, click Begin.
2. Config tab: tweak any hyperparameter (note: G's standalone script
   doesn't yet read GUI overrides — Phase 3.5 will inject config).
3. Training tab: click Start. The adapter's standalone script spawns
   as a subprocess. Live metrics flow into the strip and Charts tab.
4. When done, the final checkpoint copies into the run folder.
5. Runs tab: see your run with gens, best score, etc. Select 1+ and
   click "Compare in Charts" to overlay.

Stop: cooperative — sends SIGTERM. Pause is not supported for
external-process training (intentional; would need OS signal hacks).

## Adding a new adapter

1. Copy `adapters/tokenizer_g.py` → `adapters/your_model.py`.
2. Implement the methods documented in `adapters/base.py`:
   `config_schema`, `dataset_options`, `load_checkpoint`,
   `infer`, `metric_names`, `state_panels`,
   `population_size`, `per_genome_scalars`.
3. Register in `adapters/__init__.py`:
   ```python
   REGISTRY["My Model"] = _lazy("gui.adapters.your_model", "YourModelAdapter")
   ```
4. Restart the GUI — your model appears in the Setup tab dropdown.

No GUI code changes needed.
