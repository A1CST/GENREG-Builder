# GENREG LAB

A **gradient-free neuroevolution research lab**. Every model here is evolved —
no gradients, no backprop, no pretrained weights. The guiding thesis: *the
fitness landscape is the only lever that matters — don't design the solution,
design conditions where the only stable attractor is the solution.* Selection,
energy homeostasis, and constraints do the work that gradient descent does
elsewhere.

The lab is a single Flask app that hosts several independent projects, all
sharing one browser GUI with **real interactive terminals** (you can run the
`claude` TUI right in the page) plus a floating **Agent panel** (shared notice
feed) and **Run-Config panel** (live view of the current run + history).

> The full rules and thesis live in [`documentation/GENREG_RULES.md`](documentation/GENREG_RULES.md)
> and [`documentation/LLM__LM_RULES.md`](documentation/LLM__LM_RULES.md).
> Every change is logged in [`CHANGELOG.md`](CHANGELOG.md) and a per-project
> file under [`documentation/changelogs/`](documentation/changelogs/).

## Projects (pages)

| Page          | What it is |
|---------------|------------|
| `/`  Build    | The neuroevolution **engine** — evolve tiny nets on Snake, 2048, CartPole, Humanoid, Language, under composable EEC constraints (energy, mortality, occlusion, scarcity, non-stationarity…). Live game board, microscope (watch a genome mutate), PO-metrics. |
| `/tree`  Tree LM | **Tree-of-Models** text prediction — a routing tree of tiny evolved genomes over **byte or word/token** vocabularies. Live icicle map of the tree, per-node fitness, generation, config sweeps, routing inspector. |
| `/diff`  DiffEvo | **Denoising diffusion by neuroevolution** — one shared population of ~90-param per-pixel denoisers per noise level; the reverse walk composes them. |
| `/animation`  | Procedural animation / shape-evolution experiments. |
| `/pure`       | **PURE** — assemble a model from a node graph (drag nodes onto the canvas and wire them). |
| `/runs`       | **Runs dashboard** — every training run (filter, label, favorite, group, tag), metrics, embedding clouds, and per-run generation/replay. |
| `/docs`       | Browsable project documentation (model cards, findings, changelogs). |

## The LM autoregressive campaign (`genreg_train/genreg_lm.py` + `genreg_attn.py` + `genreg_enc.py`)

A staged, component-first push toward true autoregressive text — script-driven
(results land in `runs/lm`, `runs/attn`, `runs/enc`; browse them on `/runs`):

1. **Substrate** (`lm_char_v1`) — recurrent char model (embed + prev-char +
   `tanh(h_prev)` → per-neuron evolved activations → readout), soft
   multiplicative fitness, mandatory energy homeostasis, tournament + maturation
   gate. Evolved **above the char-bigram ceiling** (held-out 31.9% vs 27.3%).
2. **Attention** (`attn_copy_v1`) — evolved selective retrieval; **100%** on
   the offset-k copy bar via a mastery-gated curriculum.
3. **Encoder** (`enc_char_v1`) — the recurrent encoder as its own component,
   bred for multi-horizon state; freeze-and-composes to parity.
4. **Rollout survival** — the model is scored while consuming *its own* sampled
   output (learn by experience); the blended landscape cut the train/inference
   "exposure gap" from 0.72 → 0.17 nats.

Sample from any checkpoint:

```powershell
python -m genreg_train.lm_sample --list
python -m genreg_train.lm_sample --prompt "the old man " --temp 0.7 --len 200
```

Findings: [`documentation/LM_STAGE1_FINDINGS.md`](documentation/LM_STAGE1_FINDINGS.md),
[`documentation/LM_STAGE1_SUBSTRATE.md`](documentation/LM_STAGE1_SUBSTRATE.md),
[`documentation/LM_ENCODER_COMPONENT.md`](documentation/LM_ENCODER_COMPONENT.md).

## Browser terminals (the GUI infrastructure)

```
Browser  <-- WebSocket -->  Flask (app.py)  <-- TCP -->  Daemon  <-- ConPTY -->  PowerShell(s)
        (xterm.js)                         (terminal_daemon.py)      (pywinpty)
```

- **Real ConPTY pseudo-terminals** (via `pywinpty`) — arrow keys, colors, live
  redraw, the `claude` TUI, REPLs all work. The dock is on every page.
- They live in a **separate long-lived process** (`terminal_daemon.py`), so you
  can restart Flask (e.g. while editing `app.py`) and your terminals and any
  running programs keep going; Flask reconnects and replays the recent screen.
- Controls: **+ New Tab**, **Claude** (launches the TUI cleanly), **Clear**,
  **Restart**, **Stop**; resizing reflows the active terminal.

## Run

```powershell
cd $HOME\Documents\GENREG
pip install -r requirements.txt      # first time only
python app.py
```

Then open <http://127.0.0.1:5000>. Training uses CUDA when available (the LM
campaign was run on an RTX 4080) and falls back to CPU.

## Layout

| Path | Purpose |
|------|---------|
| `app.py` | Flask server, routes, WebSocket relays; auto-starts the daemon. |
| `terminal_daemon.py` | Owns the ConPTY shells + scrollback; survives Flask restarts. |
| `genreg_train/` | Training engines: `trainer.py` (engine), `tree_service.py` (Tree LM), `diffuse_service.py` (DiffEvo), `genreg_lm.py`/`genreg_attn.py`/`genreg_enc.py` (LM campaign), `runstore.py` (run persistence). |
| `agent_board.py`, `agent_notify.py` | The Agent-panel notice feed (post from Python or the CLI). |
| `templates/`, `static/` | Pages + front-end (vendored xterm.js, no CDN at runtime). |
| `documentation/` | Rules, model cards, findings, per-project changelogs. |
| `project/` | The Gutenberg training corpus + upstream engine sources. |
| `runs/` | Training-run artifacts (gitignored — regenerated locally). |

## Notes

- Binds to `127.0.0.1` only (Flask 5000, daemon 5001) — not exposed to the
  network. Requires Windows (ConPTY via `pywinpty`).
- Stop the **daemon** (not Flask) to end all terminals:
  ```powershell
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'terminal_daemon\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  ```
- Licensed AGPL-3.0 (see `LICENSE`).
