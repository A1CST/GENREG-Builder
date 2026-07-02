# Assistant Automation Platform — Build & Delivery Specification

**Document type:** Engineering specification mapped to a standard SDLC pipeline
**Audience:** Development team (vetting, estimating, and re-building the platform)
**Source of truth:** Working prototype in `genreg_gui/panels/` (v0.27.1) + `CHANGELOG.md`
**Status:** Prototype complete and functional; this document re-frames it as a
production build so it can be reviewed, decomposed into tickets, tested, and
released through the normal pipeline.

---

## 0. How to read this document

The platform already exists as a working single-developer prototype. This spec
walks it through the eight standard pipeline phases —
**Discovery → Design → Breakdown → Development → Testing → Staging → Release →
Maintenance** — so a team can validate the architecture, split it into scoped
work, and rebuild it to production standards without re-deriving intent.

Each phase section states (a) what the phase produces, (b) the concrete
artifacts for *this* platform, and (c) where the prototype already satisfies it.

A glossary of domain terms is at the end (§9). When this doc says "the
prototype," it means the code under `genreg_gui/panels/`.

---

## 1. Discovery

**Purpose:** establish what is being built and why; capture requirements.
**Output:** Product Requirements Document (PRD), personas, scope boundaries.

### 1.1 Problem statement

Repetitive desktop workflows (interacting with GUI applications: clicking
buttons, filling fields, reading on-screen values, reacting to which screen/app
is currently in focus) are time-consuming and brittle to script with
coordinate-based macros, because UI state changes and pixel positions drift.

We need a **visual automation platform** that can:

1. **Perceive** the current screen state — which application/page is showing,
   and whether specific UI regions ("anchors") are present/active.
2. **Decide** what to do based on that perception, including using a Large
   Language Model (LLM) for judgment and structured output, and remembering
   prior decisions.
3. **Act** on the real desktop — click, type, send keystrokes, read text via
   OCR — driven by the perception, not by fixed coordinates.

### 1.2 Vision statement

> A node-graph "flow" editor where a non-programmer builds perception→decision→
> action automations visually. The graph runs continuously, reacting to live
> screen state, calling a local or remote LLM for decisions, and driving the
> desktop — with full visibility (live signal animation, colored logging,
> memory inspection) and a hard-stop panic key.

### 1.3 Stakeholders & personas

| Role | Description |
|------|-------------|
| **Operator / Author (primary)** | Builds and runs flows. Technical-savvy but not necessarily a programmer. Owns the screen being automated. (Today: the project owner.) |
| **Reviewer (secondary)** | A developer/QA validating that a flow does what it claims before it runs against real apps. |
| **Maintainer** | Keeps the platform working as target apps and the host OS change. |

Single-user, single-machine deployment is the baseline. Multi-user/server
deployment is **out of scope** (see §1.6).

### 1.4 Functional requirements (FR)

> Numbered for ticket traceability. Each maps to one or more nodes/components in
> §2 and epics in §3.

**Perception**
- FR-P1: Capture a live image of a selected monitor at an adjustable frame rate.
- FR-P2: Define **anchors** — rectangular screen regions — by drawing a marquee
  over a frozen frame; store them in full-resolution capture coordinates.
- FR-P3: **Stable anchors** lock onto a region's hash and report match/mismatch
  (present/absent) each frame.
- FR-P4: **Unstable anchors** measure volatile regions; gated by stable
  "parent" anchors; report `gated / changing / solidified / offscreen` based on
  how long the region has held still (`solidify_secs`).
- FR-P5: Anchors carry a **role** (`field` / `display` / `button`) and an
  optional **group tie** (the app/site they belong to).
- FR-P6: Anchors support **option sub-boxes** (a strip of N sub-regions inside
  the boundary: count, gap, orientation row/column, axis offset).
- FR-P7: **Pages** — named snapshots of a 20×20 sampling grid — detect "which
  view of which app am I on" by re-hashing identifying (stable) cells; a page is
  detected at ≥ `PAGE_DETECT_THRESHOLD` (85%) cell match.
- FR-P8: Pages are organized into **groups** (an app with many views); a group
  is "active" if any member page is detected.
- FR-P9: **Auto-discovery** — a learning pass classifies grid cells as
  `stable / settling / unstable` by temporal hysteresis (a region that changes
  can recover to stable).
- FR-P10: Edit saved anchors (everything but the drawn boundary) and re-baseline
  a stable anchor's reference hash.
- FR-P11: Expose live perception to the decision layer:
  `current_active_group()`, `group_active(group)`, `anchor_active(name)`,
  `anchor_action_info(name)`.

**Decision / Flow engine**
- FR-D1: A **node graph** with typed input/output ports and bezier edges,
  zoom/pan canvas, multi-select, copy/paste, drag-to-connect, and a
  drop-in-empty-space connection menu.
- FR-D2: **Signal-driven execution** — a `Start` node emits a signal that rides
  *with the data* along edges; nodes fire when a signal reaches them.
- FR-D3: Support **loops** (cyclic edges, `GoTo` jumps) running indefinitely
  until stopped, with a runaway guard for un-paced tight loops.
- FR-D4: Run off the GUI thread (no UI freeze during slow LLM/Wait nodes), with
  **Run**, **Live** (re-fire ~2×/s), and **Stop** controls, plus a **global END
  hotkey** that kills a run even when the editor is unfocused.
- FR-D5: Per-node **activity bars** showing where the signal is (progress for
  timed nodes, flash-and-fade for instant nodes).
- FR-D6: Nodes have a unique **id**, an optional **custom name**, and an
  **enabled/disabled** toggle (disabled = skipped, signal passes through).
- FR-D7: The full **node catalogue** (§2.7): Start, Command, Anchor, If/Else,
  Proceed, GoTo, Wait, LLM, Action, Type, Keystrokes, LTM, STM, Logging.
- FR-D8: Persist a flow to JSON; autosave on close; New/Open/Save/Save As.

**LLM**
- FR-L1: An **LLM node** sends its text input (+ an optional hidden system
  prompt) to a model and outputs the reply.
- FR-L2: **Hidden prompt library** — reusable system prompts stored by name,
  selected on an LLM node, folded in front of the input, never shown on canvas.
- FR-L3: **Local or remote** model, switchable by a single **Preferences
  toggle** that reroutes every LLM node at once.
- FR-L4: Local model served by **Ollama** (default `phi4`); manage it
  (start/restart/stop = load/reload/unload from VRAM) and test prompts from a
  dedicated window.
- FR-L5: **Max output tokens** control per LLM node.
- FR-L6: Strip a reasoning model's `<think>…</think>` scratchpad from output.

**Memory**
- FR-M1: **LTM** (long-term, unbounded RAM) and **STM** (short-term, capped at
  20, FIFO) nodes that **store** named variables and **recall** them by key.
- FR-M2: Blank key → auto-derive a key from the value content (repeats reuse one
  slot — pattern dedup for cached decisions).
- FR-M3: A **memory viewer** showing both caches as JSON with per-scope Clear
  and Reset-all, auto-refreshing live.

**Action**
- FR-A1: An **Action node** drives a saved anchor by its role: `display` → OCR
  the region and output the text; `field`/`button` → left-click the region
  centre.
- FR-A2: A **Type node** types text (the param or the incoming data) as real
  keystrokes, paced per character, with newline handling (default Shift+Enter so
  it doesn't submit chat boxes).
- FR-A3: A **Keystrokes node** presses key combos (`ctrl+a`, `ctrl+shift+t`,
  `enter`, …).
- FR-A4: Map an anchor's capture-coordinate rect to **absolute screen pixels**
  via its monitor offset for clicking/OCR.

**Observability / UX**
- FR-O1: A **console** that streams logging output; **Logging nodes** with a
  label and a selectable **console color**.
- FR-O2: A **crash reporter** capturing uncaught exceptions to disk and to an
  in-app dialog.
- FR-O3: Tab-gated simulations / capture lifecycle (don't burn resources on
  hidden tabs).

### 1.5 Non-functional requirements (NFR)

- NFR-1 **Responsiveness:** the GUI never blocks on capture, LLM calls, OCR,
  model load, or paced typing — all run off the GUI thread.
- NFR-2 **Resilience:** optional heavy dependencies (EasyOCR, pynput) degrade to
  a console note, not a crash, when absent. Uncaught errors are reported, not
  silent.
- NFR-3 **Safety:** a global **END** panic key stops any run instantly; a
  runaway-loop guard prevents a tight loop from pegging a core.
- NFR-4 **Secrets hygiene:** API host/key live in a config file in the OS app-
  config dir, never in source.
- NFR-5 **Coordinate stability:** anchors are stored in full-resolution capture
  coordinates, independent of window scaling/letterboxing.
- NFR-6 **Determinism of structure:** node identity is a stable integer id; user
  labels never affect wiring or saved edges.
- NFR-7 **Performance targets:** capture 1–60 fps (default 10); perception
  re-check throttled (~5 Hz for anchors/pages); flow step budget guarded
  (runaway = 5000 steps in < 2 s with no pacing).

### 1.6 Out of scope (explicit)

- Multi-user / hosted / server deployment; RBAC; audit trails.
- Cross-platform input on **Wayland** (the action backend targets X11; Windows/
  macOS would need a different backend — see §6).
- Recording/replay of macros; image-template matching beyond region hashing.
- Training/fine-tuning models (the platform *consumes* an LLM only).

### 1.7 Assumptions & constraints

- Single desktop, X11 session, Python 3.11, PyQt6.
- A CUDA GPU is available for the local model (prototype: RTX 4080, 16 GB) but
  the platform also supports a **remote** LLM endpoint, so a GPU is not strictly
  required.
- Ollama is installed and on `PATH` for local-model mode.

---

## 2. Design

**Purpose:** plan *how* to build the requirements.
**Output:** architecture, component contracts, data models, data-flow, UI
wireframes, tech choices — enough that a developer builds without guessing.

### 2.1 Architecture overview (three layers + supporting services)

```
            ┌──────────────────────────────────────────────────────────┐
            │                     PERCEPTION LAYER                       │
            │  Assistant tab (assistant_tab.py)                          │
            │  • CaptureWorker (QThread, mss)  → live frames             │
            │  • CaptureView (marquee draw, coord mapping)               │
            │  • Anchors (stable/unstable) + Pages + Auto-discovery      │
            │  • Detection loop: anchor match / page detect / group act  │
            │  exposes: current_active_group, group_active,              │
            │           anchor_active, anchor_action_info                │
            └───────────────┬──────────────────────────────────────────┘
                            │ live perception (read on the run thread)
                            ▼
            ┌──────────────────────────────────────────────────────────┐
            │                  DECISION / FLOW ENGINE                    │
            │  Flow Editor (flow_editor.py)                              │
            │  • Node graph (FlowNode, Port, Edge, FlowView, Inspector)  │
            │  • Signal-driven executor: FlowRunner (QThread) + _fire()  │
            │  • RunContext: perception accessors, llm(), memory, log    │
            │  • Controls: Run / Live / Stop / global END / activity bars│
            └───────┬───────────────────────┬──────────────────┬────────┘
                    │ ctx.llm()             │ memory store     │ actions
                    ▼                       ▼                  ▼
        ┌────────────────────┐  ┌────────────────────┐  ┌─────────────────────┐
        │   LLM SERVICE      │  │   MEMORY SERVICE   │  │   ACTION BACKEND    │
        │ llm_client.py      │  │ flow_memory.py     │  │ anchor_actions.py   │
        │ • remote /chat     │  │ • MemoryStore      │  │ • OCR (easyocr)     │
        │ • local Ollama     │  │   LTM (∞) STM(≤20)  │  │ • click/type/combo  │
        │ local_llm.py:      │  │ • store / recall   │  │   (pynput)          │
        │ • model lifecycle  │  │ MemoryViewer (JSON) │  │ • coord → screen px │
        │ • LocalLLMWindow   │  └────────────────────┘  └─────────────────────┘
        └────────────────────┘

  SUPPORTING: preferences.py (QSettings) · crash_reporter.py · app.py (tabs)
  PERSISTENCE: assistant_anchors.json · assistant_pages.json · assistant_flow.json
               · prompt_library.json · llm_config.json · QSettings (prefs/UI)
```

**Design principle (carried from the prototype): self-contained modules.** Each
file owns one concern and does not reach into others except through the small,
explicit interfaces below. The Flow Editor talks to the Assistant only through
the four perception accessors; to the LLM only through `llm_client.chat()`; to
memory only through `flow_memory.STORE`; to the desktop only through
`anchor_actions`. This is what makes the system testable in isolation (§5).

### 2.2 Component responsibilities

| Module | Responsibility | Key classes / fns |
|--------|----------------|-------------------|
| `assistant_tab.py` | Screen capture, anchors, pages, auto-discovery, detection, perception accessors, opens Flow Editor & Local LLM windows | `AssistantTab`, `CaptureWorker`, `CaptureView`, `Anchor`, `Page`, `GridCell` |
| `flow_editor.py` | Node graph UI + signal-driven executor + run controls + memory/prompt windows | `FlowEditorWindow`, `FlowView`, `FlowNode`, `Port`, `Edge`, `Inspector`, `Palette`, `RunContext`, `FlowRunner`, `_fire()`, `MemoryViewer`, `PromptLibraryWindow`, `_EndHotkey` |
| `llm_client.py` | Model I/O: remote `/chat` or local Ollama `/api/chat`; mode switch; token cap; think-strip; config | `chat`, `health`, `test`, `load_config`, `save_config`, `set_mode` |
| `local_llm.py` | Local model lifecycle (load/unload/restart) + control & test window | `LocalLLMWindow`, `start/stop/restart`, `status_text`, `list_models` |
| `anchor_actions.py` | Desktop actuation: OCR a region, click, type text, press combos; coord mapping | `do_action`, `read_display`, `left_click`, `type_text`, `press_combo`, `center_abs` |
| `flow_memory.py` | In-RAM keyed store, LTM unbounded / STM capped, thread-safe | `MemoryStore`, `STORE`, `auto_key` |
| `preferences.py` / `preferences_dialog.py` | Persisted user prefs incl. LLM source toggle | `Preferences`, `PreferencesDialog` |
| `crash_reporter.py` | Capture uncaught exceptions (all threads) to disk + dialog | `install`, `set_ui` |
| `app.py` | Host window, tab bar, tab-gated lifecycle | `MainWindow` |

### 2.3 Data models

> All persisted in JSON under the OS app-config dir unless noted. Runtime-only
> fields are marked `(transient)`.

**Anchor** (`assistant_anchors.json`)
```
id: int                      # stable identity
name: str
kind: "stable" | "unstable"
rect: {x, y, w, h}           # full-res capture coordinates
monitor: int                 # capture monitor index (-1 = default)
digest: str                  # stable: locked SHA-1 of the region; unstable: unused
role: "field" | "display" | "button"
group: str                   # app/site tie ("" = Free, always-on)
parents: [int]               # unstable: stable anchor ids that gate it
solidify_secs: float         # unstable: stillness threshold (default 30)
options: bool                # split boundary into sub-boxes
opt_count, opt_spacing: int
opt_vertical: bool           # row (false) / column (true)
opt_offset: int              # shift sub-boxes along their axis
# (transient) matches: bool|None   state: "gated|changing|solidified|offscreen"
```

**Page** (`assistant_pages.json`)
```
name: str                    # auto-numbered "Page N" within group
group: str
dims: (w, h)                 # grid dims when learned
cells: [ {rect, tier, hash} ]   # 20×20 sampled cells; stable cells are identifying
# (transient) detected: bool|None    match_frac: float
```

**GridCell (auto-discovery, transient):** `(row, col, rect)` + temporal state
(`stable / settling / unstable`) by hysteresis (`AUTO_STABLE_HOLD=5s`,
`AUTO_UNSTABLE_HOLD=15s`).

**FlowNode** (`assistant_flow.json` → `nodes[]`)
```
id: int                      # unique identity (drives edges)
type: str                    # one of the catalogue (§2.7)
name: str                    # optional user label (display only)
enabled: bool                # disabled → skipped, signal passes through
x, y: float                  # canvas position
params: { ... }              # per node type (§2.7)
```

**Edge** (`assistant_flow.json` → `edges[]`)
```
src: int, src_port: str, dst: int, dst_port: str
```

**MemoryStore (RAM only):** `ltm: dict`, `stm: OrderedDict (cap 20)`; values are
arbitrary (strings, LLM replies, action results).

**Configs:** `llm_config.json` = `{host, api_key, verify, mode,
local_host, local_model}`; `prompt_library.json` = `{prompts: {name: text}}`;
`QSettings` = prefs (auto-resume, restore-*, LLM source) + UI state
(auto-collapse) + session snapshot.

### 2.4 Execution model (the heart of the engine)

**Signal-driven BFS.** `FlowRunner.run()` builds a queue seeded from every
`Start` node and processes `(node, data)` items:

1. Pop a node; emit `node_enter(id)` (activity bar pulse).
2. Call the **pure** `_fire(node_snapshot, data, ctx)` → returns *routes*:
   `("port", out_name, out_data)` or `("goto", target_id, data)`.
3. For a `port` route, enqueue every node connected to that output port; for a
   `goto`, enqueue the target node directly.
4. Emit `node_exit(id)`.

**Why `_fire` is pure and operates on a snapshot dict:** so the executor runs on
a background `QThread` with **no references to QGraphics objects** (thread-safe).
The GUI thread only renders; the run thread only computes.

**Loops & the runaway guard.** Cyclic edges and `GoTo`-back are allowed and run
**forever until Stop**. A guard trips only on a *tight* loop: 5000 steps in
< 2 s with no pacing (which would peg a CPU core and flood the console). A loop
paced by a `Wait`/`Proceed`/`LLM` resets the window and continues indefinitely.

**Cancellation.** `requestInterruption()` is checked every step *and* inside the
blocking loops of `Wait`/`Proceed`/`Type`, so Stop / END / window-close
interrupt promptly — even mid-10-second wait. On any unwind the run reports
`[run] stopped by user.`

**Live mode.** Re-fires the graph ~2×/s against live perception; overlapping
runs are ignored while one is in flight, so it can't pile up. The `LLM` node
fires **only on a manual Run**, never on a Live tick (so it never hammers the
endpoint).

**Threads & signals (concurrency contract):**
```
GUI thread        : all QWidget/QGraphics; renders activity bars; appends console
CaptureWorker     : QThread; grabs frames via mss; emits QImage to GUI thread
FlowRunner        : QThread; runs _fire loop; emits line / line_colored /
                    node_enter / node_progress / node_exit (queued → GUI thread)
_EndHotkey        : pynput listener thread; emits `fired` (queued → GUI thread)
LocalLLM workers  : QThread per blocking op (load/test); emit (ok, text)
```
Cross-thread perception reads (e.g. `current_active_group()`) touch only plain
Python attributes set by the capture/detection loop — GIL-safe for the read-only
access pattern used.

### 2.5 External API contracts

**Remote LLM** (`mode = remote`)
```
POST {host}/chat        body {"message": "<system+user folded>", "max_tokens"?: int}
                        headers Authorization: Bearer <key>, X-API-Key: <key>
                        → JSON; reply extracted from response/reply/content/
                          choices[0].message.content/message.content/...
GET  {host}/health      → 200 = alive
verify: false (self-signed; bearer is the real auth) | path to .pem (strict)
```

**Local LLM** (`mode = local`, Ollama)
```
POST {local_host}/api/chat   body {"model", "messages":[{role,content}...],
                                   "stream": false, "options": {"num_predict"?}}
                             → {"message": {"content": ...}}  (think-stripped)
GET  {local_host}/             → 200 "Ollama is running"  (health)
GET  {local_host}/api/tags     → installed models
GET  {local_host}/api/ps       → loaded (resident) models
POST {local_host}/api/generate {"model","keep_alive": -1|0}  load / unload
```

**Internal — Assistant → Flow Editor** (the only coupling):
```
current_active_group() -> str            # best-matching detected page's group
group_active(group) -> bool              # page detected OR tied anchor matching
anchor_active(name) -> bool              # stable match / unstable solidified
anchor_action_info(name) -> {name, role, monitor, rect(x,y,w,h)} | None
```

### 2.6 Coordinate mapping (perception → action)

Anchors are stored relative to the captured monitor's top-left (mss grabs
`monitors[idx]`). To click/OCR, resolve to absolute screen pixels:
```
monitor = mss.monitors[idx]   (idx<0 → 1 if multi-head else 0)
abs_x = monitor.left + rect.x (+ w/2 for centre)
abs_y = monitor.top  + rect.y (+ h/2 for centre)
```
OCR grabs `{left, top, width, height}` of that absolute region; the click moves
the pointer to the centre and presses left.

### 2.7 Node catalogue (the executor's instruction set)

> Ports are `in`/`out` unless noted. Node-level fields `id`, `name`, `enabled`
> apply to all. A **disabled** node emits its input on all outputs unchanged.

| Node | Category | Ports | Params | Behaviour |
|------|----------|-------|--------|-----------|
| **Start** | flow | — → out | — | Emits the initial signal (`data=None`). Run seeds from every Start. |
| **Command** | io | in → out | `text` | Source: emits the configured text as data. |
| **Anchor** | perception | in → out | `mode` (active/passive), `group` | active → emits the **live** active group; passive → emits `group` only while that group is active. Reads perception fresh each fire. |
| **If / Else** | logic | in → true/false | `check` (contains / equals / not empty / anchor active / group active), `anchor`, `group`, `condition` | Routes by the selected check; relevant field shown per check. |
| **Proceed** | logic | in → proceed/timeout | `watch` (anchor/group/any active), `anchor`, `group`, `timeout`, `timeout_output` | **Timed fork:** polls live perception up to `timeout` s; proceed if the watched signal goes active (emits the matched name), else timeout (emits `timeout_output` or the data). Cancellable. |
| **GoTo** | flow | in → — | `target` (node id) | Dispatches the signal to any node by id (jumps / subroutines / loops). |
| **Wait** | flow | in → out | `amount`, `unit` (s/ms) | Pauses then passes through. Paces loops; cancellable; reports progress to its activity bar. |
| **LLM** | ai | in (**text**) → out | `prompt` (hidden prompt name), `max_tokens` | Sends text (+ hidden system prompt) to the model, outputs reply. **Manual Run only.** |
| **Action** | action | in → out | `anchor` | display → OCR region → output text; field/button → left-click centre (pass data through). |
| **Type** | action | in → out | `text`, `delay_ms`, `newline` (shift+enter/enter/space/strip) | Types `text` or the incoming data as paced keystrokes; cancellable; settle pause after. |
| **Keystrokes** | action | in → out | `combo` | Presses a chord (`ctrl+a`, `ctrl+shift+t`, `enter`…). |
| **LTM** | memory | in → out | `mode` (store/recall), `key` | Long-term (unbounded) store/recall; blank key = content hash. |
| **STM** | memory | in → out | `mode` (store/recall), `key` | Short-term (cap 20, FIFO) store/recall. |
| **Logging** | io | in → out | `label`, `color` | Prints `label: value` to the console in the chosen colour (dedup in Live, always in Run). Pass-through. |

### 2.8 UI wireframes (text)

**Assistant tab** (resizable splitters):
```
┌─ Control ─────────────┐┌──────────────────────────────────────┐
│ Monitor ▾  FPS ▭───── ││                                      │
│ [▶ Start] [■ Stop]    ││         LIVE CAPTURE VIEW            │
│ [⧉ Open Flow Editor]  ││   (marquee draw; anchors drawn as    │
│ [🧠 Local LLM]        ││    green/red/amber overlays)          │
│ Status: …             ││                                      │
├─ Anchors ─────────────┤│                                      │
│ ✚ Draw  (Stable ▾)    ││                                      │
│ • Login   btn  ● match ││                                      │
│ • Score   dsp  ● diff  │└──────────────────────────────────────┘
│ ◉ Learn layout  Clear │
├─ Pages ───── [⊟ Auto-collapse] ┐
│ ▸ YouTube (3) ●               │
│ ▸ Gmail (1)   ●               │
└───────────────────────────────┘
```

**Flow Editor** (separate window):
```
[New][Open][Save][Save As] | [Run][Stop][Live][Clear] | [Prompts][Test LLM][Memory] | [Fit]
┌ Nodes ─┐┌──────────── Canvas (zoom/pan grid) ─────────────┐┌ Inspector ─┐
│ Start  ││  ┌Start┐   ┌Anchor┐   ┌LLM "Greeter"┐           ││ Name  ____ │
│ Command││  └──●─┘──▶─●─out──●─▶─text         out─▶ …       ││ ☑ Enabled  │
│ Anchor ││            (loading bars fill/flash as it runs)  ││ Hidden ▾   │
│ …      ││                                                  ││ Max tokens │
└────────┘└──────────────────────────────────────────────────┘└────────────┘
┌ Console ──────────────────────────────────────────────────────────────────┐
│ where: YouTube      (colored per Logging node)                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**Local LLM window:** status line (`● ready — phi4 loaded`), model picker,
[▶ Start][⟳ Restart][■ Stop][↻ Refresh], prompt box + [Send ▸] + reply box.

**Memory window:** two JSON panes (LTM / STM) with counts, [Clear LTM]
[Clear STM] [⟲ Reset all] [↻ Refresh], auto-refreshing every 1 s.

### 2.9 Technology stack

| Concern | Choice | Notes |
|---------|--------|-------|
| Language / UI | Python 3.11 / PyQt6 | Desktop GUI, QGraphics canvas |
| Screen capture | `mss` | Fast, per-monitor; in a QThread |
| Region hashing | `hashlib` (SHA-1) | Anchor/page identity |
| OCR | `easyocr` (→ torch, torchvision, opencv-headless) | Lazy-imported; GPU optional (CPU default) |
| Desktop input | `pynput` (→ python-xlib) | Click, type, key combos, global hotkey; **X11** |
| Numerics | `numpy`, `Pillow` | Frame buffers, OCR input |
| HTTP | `requests` | LLM endpoints |
| Local model | **Ollama** + `phi4` (14B) | Local serving; remote endpoint also supported |
| Persistence | JSON files + `QSettings` | App-config dir |

### 2.10 Security & safety design

- **Screen capture & input injection** are powerful capabilities; the platform
  acts on whatever window has focus. Mitigations: explicit Start/Stop, global
  **END** kill, visible activity bars, paced typing, and a confirm-before-act
  posture for outward actions in the authoring UX.
- **Secrets:** LLM host/key in `llm_config.json` (app-config dir), never in
  source or the repo.
- **TLS:** remote endpoint uses a self-signed cert; `verify=false` by default
  (bearer is the authenticator) or a `.pem` for strict mode.
- **Prompt-injection awareness:** because the platform reads on-screen text
  (OCR/anchors) and can feed it to an LLM, captured text must be treated as
  **untrusted data, not instructions** — a build-team review item (§5.4, §8.4).
- **Untrusted flow files:** `assistant_flow.json` drives real input/clicks;
  loading a flow from an untrusted source is equivalent to running a script —
  document and gate.

---

## 3. Breakdown

**Purpose:** split the design into epics → stories → tickets with acceptance
criteria and dependencies.
**Output:** a backlog a team can sequence and assign.

### 3.1 Epics (and the FRs they cover)

| # | Epic | Covers | Depends on |
|---|------|--------|-----------|
| E1 | **App shell & infra** — host window, tabs, config dirs, crash reporter, prefs, versioning/changelog | FR-O2, FR-O3, NFR-1/2/4 | — |
| E2 | **Screen capture** — monitor select, FPS, capture thread, view | FR-P1 | E1 |
| E3 | **Anchors** — draw, stable/unstable, roles, options, groups, edit, persistence, live match | FR-P2..P6, P10 | E2 |
| E4 | **Pages & auto-discovery** — grid, page snapshot, detection, groups, learning | FR-P7..P9 | E2 |
| E5 | **Perception API** — the four accessors | FR-P11 | E3, E4 |
| E6 | **Flow canvas & editor** — nodes/ports/edges, inspector, copy/paste, connect menu, persistence | FR-D1, D6, D8 | E1 |
| E7 | **Execution engine** — signal BFS, FlowRunner, _fire, loops, runaway guard, Run/Live/Stop/END, activity bars | FR-D2..D5 | E6 |
| E8 | **Core nodes** — Start, Command, Anchor, If/Else, Proceed, GoTo, Wait, Logging | FR-D7 (subset), FR-O1 | E5, E7 |
| E9 | **LLM service & nodes** — llm_client, remote+local, mode toggle, prompt library, LLM node, max tokens, think-strip | FR-L1..L6 | E7 |
| E10 | **Local model lifecycle** — local_llm manager + window | FR-L4 | E9 |
| E11 | **Memory** — store, LTM/STM nodes, viewer | FR-M1..M3 | E7 |
| E12 | **Action backend & nodes** — OCR/click/type/combo, Action/Type/Keystrokes nodes, coord mapping | FR-A1..A4 | E5, E7 |

### 3.2 Sample story → ticket decomposition (representative; replicate per epic)

**E7 — Execution engine**

- *Story E7.1:* As an operator I can Run a flow off the GUI thread without
  freezing the UI.
  - T-E7.1a Extract pure `_fire(node_snapshot, data, ctx)`; unit-test each node
    type in isolation. **AC:** every node type returns the documented routes;
    no QGraphics references in `_fire`.
  - T-E7.1b `FlowRunner(QThread)` snapshots the graph and runs the BFS; streams
    console output via queued signals. **AC:** a flow with a 120 s LLM node keeps
    the window responsive; overlapping runs ignored.
- *Story E7.2:* As an operator I can loop a flow until I stop it.
  - T-E7.2a Remove the hard step cap; add the rate-based runaway guard. **AC:** a
    Wait-paced loop runs > 1 window without tripping; a tight loop trips at 5000
    steps in < 2 s with a helpful message.
  - T-E7.2b Stop button + global END hotkey; interruption checked every step and
    inside Wait/Proceed/Type. **AC:** Stop/END halt a run within ~one poll, even
    mid-wait; END works when the editor is unfocused.
- *Story E7.3:* As an operator I can see where the signal is.
  - T-E7.3a `node_enter/progress/exit` signals + per-node activity bars + 33 ms
    decay animation. **AC:** timed nodes show real progress; instant nodes flash
    and fade; animation stops when idle.

**E9 — LLM service** (representative ACs)

- T-E9.1 `llm_client.chat(text, system, max_tokens)` dispatches on `mode`;
  remote `/chat` + local Ollama `/api/chat`. **AC:** both return clean text; a
  bad endpoint yields a graceful error string, not a crash.
- T-E9.2 Preferences LLM-source toggle persists and reroutes all LLM nodes.
  **AC:** flipping the toggle changes the endpoint every node uses, no per-node
  edits.
- T-E9.3 Hidden prompt library CRUD + folding. **AC:** an LLM node referencing a
  prompt by name sends `system\n\n input`; the prompt never appears in node data.
- T-E9.4 `max_tokens` cap + `<think>` strip. **AC:** `max_tokens=16` measurably
  shortens output; a `<think>…</think>` (closed or unclosed) never reaches the
  output port.

### 3.3 Sequencing (critical path)

```
E1 → E2 → {E3, E4} → E5 → ...
E1 → E6 → E7 → E8
E7 → {E9 → E10, E11, E12}
E5 feeds E8 (Anchor/Proceed/If-Else) and E12 (Action)
```
Recommended delivery order: **E1, E2, E3/E4, E5, E6, E7, E8, E9, E11, E12,
E10** — perception and the engine before the nodes that depend on them; the
local-model window (E10) last since remote mode (E9) unblocks LLM nodes earlier.

### 3.4 Estimation note

The prototype is ~5,000 LOC across 9 modules (Assistant 2,022; Flow Editor
1,878; the rest small). Use it as a reference implementation for sizing, **not**
as production code to ship — see §4.1.

---

## 4. Development

**Purpose:** write the code + tests, via reviewed PRs.
**Output:** merged, reviewed, tested increments.

### 4.1 Treat the prototype as an executable spec, not production code

It was built single-developer, feature-by-feature, with the entire Flow Editor
in one 1,878-line file. For production: keep the **module boundaries and
interfaces** (§2.2/§2.5) but **split large files** (e.g. `flow_editor.py` →
`graph_view/`, `executor/`, `nodes/`, `windows/`), add type hints throughout,
and add the test suite (§5) that the prototype only has ad-hoc.

### 4.2 Coding standards & conventions (carried from the prototype, to formalize)

- **Self-contained modules**, narrow interfaces (§2.2). A node's behaviour lives
  in `_fire`; its UI lives in the spec + Inspector — keep them separable.
- **Lazy-import optional heavy deps** (EasyOCR, pynput); degrade to a console
  note, never crash the app, when absent.
- **No secrets in source.** Config in the app-config dir.
- **Pure executor.** Anything that runs on the FlowRunner thread must operate on
  plain snapshots, not Qt objects.
- **Stable identities.** Node `id` and anchor `id` are the identity; names are
  cosmetic and must never affect wiring or saved edges.
- **Backward-compatible persistence.** New node params/fields must load old JSON
  with sensible defaults (the prototype does this for every node change).

### 4.3 Change discipline (already practiced; make it policy)

- **Every change** bumps `__version__` (SemVer) and adds a `CHANGELOG.md` entry
  describing *what changed and why*. This existing log is the de-facto release
  history and should remain the PR-merge gate.
- One scoped change per PR; PR description references the ticket and the FR.

### 4.4 Per-area development notes / gotchas (lessons already paid for)

- **Qt combo rebuild crash:** rebuilding an Inspector form *inside* a combo's own
  `currentTextChanged` deletes the live widget → native crash. Defer with
  `QTimer.singleShot(0, …)`.
- **Live perception in loops:** nodes must read perception *fresh each fire*
  (`live_active_group()`), not a start-of-run snapshot, or they freeze inside a
  loop.
- **Group ≠ anchor:** "active group" is *page* detection; "anchor active" is the
  green-dot hash match. `group_active()` unifies them (page OR tied anchor).
- **Typing too fast drops keys** and returns before the app processes them →
  per-char pacing + settle pause; newline default **Shift+Enter** so chat boxes
  don't submit early.
- **Reasoning models** emit `<think>` (sometimes unclosed) — strip robustly.
- **Action backend is X11-only** (pynput/xlib) — Windows/macOS need a different
  backend behind the same `anchor_actions` interface.

---

## 5. Testing

**Purpose:** verify functions, components, and the whole system.
**Output:** passing unit/integration/QA suites; bugs filed as tickets.

The prototype was validated with **headless PyQt** harnesses (instantiate
widgets under a `QApplication`, drive logic directly, assert on results) — adopt
this as the standard test style.

### 5.1 Unit tests (pure logic — highest value, no display needed)

- **`_fire` per node type** — feed a fake `RunContext` and assert the returned
  routes for every branch: Command text, Anchor active/passive, If/Else all five
  checks (+ backward compat blank-condition truthiness), Proceed
  anchor/group/any × hit/timeout/cancel, Wait timing + cancel, Type pacing +
  newline modes + cancel, Keystrokes combo parsing, LTM/STM store/recall +
  auto-key, Logging color routing, disabled-node passthrough.
- **`MemoryStore`** — STM FIFO cap at 20, LTM unbounded, `auto_key` stability/
  content-addressing.
- **`llm_client`** — `_strip_think` (closed/unclosed/none), `save_config` merge
  (no key loss), mode dispatch, `max_tokens` wiring.
- **`anchor_actions`** — combo→key resolution, paced-type sequence (fake
  Controller), newline→Shift+Enter chord, coordinate centre math.
- **Perception** — `group_active` (page-detected vs anchor-matching vs neither),
  `anchor_active` (stable match vs unstable solidified), `current_active_group`
  (best match).

### 5.2 Integration tests (components together)

- **Engine + perception:** a looping flow whose live group changes mid-run; an
  Anchor(active)→Logging chain tracks the change (and only logs on change).
- **Engine + LLM (local):** Command→LLM→Logging end-to-end against a running
  Ollama; assert a clean, think-stripped reply and that `max_tokens` shortens it.
- **Engine + memory:** store then recall across nodes; STM eviction under load.
- **Runner lifecycle:** Run/Stop/END interrupt a mid-Wait run; runaway guard
  trips a tight loop and resets for a paced one.
- **Persistence round-trips:** save→load a flow preserves node `name`,
  `enabled`, params, and edges; anchors/pages reload identically.

### 5.3 QA / system tests (user perspective, real desktop)

- Author an anchor over a real button → Action(button) clicks it.
- Author a display anchor over a readout → Action(display) OCRs the right text.
- Proceed(watch=group) proceeds when you switch into that app, else takes the
  timeout branch.
- A full perceive→decide→act loop drives a real app unattended; END stops it.

### 5.4 Security / safety tests

- OCR'd/anchor text fed to an LLM cannot redirect the flow (treat as data).
- Global END always stops a run; a tight loop never hard-locks the UI.
- No secret ever appears in logs, the canvas, saved flows, or crash reports.

### 5.5 Definition of done (per ticket)

Code + unit tests + (where applicable) an integration test + changelog entry +
version bump + PR review approved + manual QA for any node that touches the real
desktop.

---

## 6. Staging

**Purpose:** validate in an environment that mirrors production before release.
**Output:** a deployable build proven on a clean machine.

### 6.1 Environment / provisioning checklist

- OS: Linux **X11** session (action backend requirement — see §6.3 for other
  OSes). Python 3.11.
- Python deps: PyQt6, mss, numpy, Pillow, requests, easyocr (+ torch,
  torchvision, opencv-python-headless, scikit-image, etc.), pynput
  (+ python-xlib).
- **Local model:** Ollama installed + on PATH; `ollama pull phi4` (~9 GB, Q4).
  GPU recommended (prototype: RTX 4080 16 GB) — verify with a `/api/chat`
  round-trip.
- **Remote model (if used):** reachable `/chat` + `/health`; key in
  `llm_config.json`.
- App-config dir writable (anchors/pages/flow/prompt/llm configs land there).

### 6.2 Staging smoke tests

- App launches; crash reporter installed; all tabs load.
- Capture starts on the selected monitor at the target FPS.
- Draw → save → match an anchor; save → detect a page.
- Open Flow Editor; Run a trivial Start→Command→Logging flow; see console output.
- Local LLM: Start (model loads), Send a test prompt, get a reply; toggle
  Preferences → LLM source and confirm an LLM node uses it.
- Save a flow, restart the app, reload it intact.

### 6.3 Platform portability gate

The action backend (`anchor_actions`, `_EndHotkey`) is **X11/pynput**. A
Wayland/Windows/macOS release must implement the same interface
(`type_text`, `press_combo`, `left_click`, `center_abs`, global hotkey) on a
native backend; everything above that interface is portable. **Block release on
unsupported platforms until the backend exists.**

---

## 7. Release

**Purpose:** ship to users with documented changes.
**Output:** versioned release + release notes.

- **Versioning:** SemVer in `genreg_gui/__init__.py`; the **CHANGELOG** is the
  release notes (already maintained per change). Tag the release at the version.
- **Packaging:** define the run target (today: `python main.py`). For a real
  release, produce a launchable artifact (venv/installer/AppImage) bundling the
  Python deps; the model is provisioned separately (Ollama pull) and documented
  in the release notes.
- **Release notes must call out:** new nodes/params, any persistence format
  changes (with the backward-compat behaviour), the platform requirement (X11),
  and the model requirement (Ollama/phi4 or a remote endpoint).
- **Rollback:** releases are config-compatible backward; keep the prior version
  available. JSON config files are versionless but additive — older app versions
  ignore unknown fields.

---

## 8. Maintenance

**Purpose:** keep it working; absorb bugs and requests; iterate.
**Output:** a triaged backlog feeding the loop again.

### 8.1 Bug intake

The in-app **crash reporter** writes every uncaught exception (all threads) to
`crashes/<utc>.txt` with version + platform + traceback, and surfaces a
non-blocking dialog. Make these the canonical bug artifact; a triage step files
them as tickets.

### 8.2 Known limitations / tech debt (carry into the backlog)

- **X11-only** action backend (portability — §6.3).
- **Memory is RAM-only** (LTM resets on restart); a disk-backed LTM is a
  candidate feature.
- **`flow_editor.py` is monolithic** (§4.1) — split for maintainability.
- **Anchor identity is a region hash** — robust to scaling but not to theme
  changes/large content shifts; re-capture is the manual remedy.
- **Single fan-in port** on memory/most nodes (fan-in is "over time," not N
  labelled ports) — a multi-input node would need the executor to track the
  destination port.
- **Live mode + LLM:** LLM nodes deliberately don't fire on Live ticks; document
  this so authors don't expect Live to drive LLM decisions.

### 8.3 Feature-request pipeline

Requests re-enter at **Discovery** (does it fit the vision?) → **Design** (which
layer? new node vs. new param?) → **Breakdown** (ticket with ACs) → the rest of
the loop. New nodes are the most common request and have a well-worn path: add a
spec entry (ports/params/category/`visible_if`/`label`), a `_fire` branch, any
`RunContext`/backend hook, persistence defaults, and tests.

### 8.4 Operational / security maintenance

- Watch for **prompt-injection** via OCR'd screen text reaching the LLM; keep
  captured text as data, never instructions.
- Rotate the remote LLM key in `llm_config.json`; never commit it.
- Re-validate the action backend after OS/desktop-environment upgrades.

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| **Anchor** | A saved screen region matched by a hash of its pixels. *Stable* = locked hash (present/absent); *unstable* = volatile region measured for stillness. |
| **Role** | An anchor's purpose: `field` (click to focus), `display` (OCR), `button` (click). |
| **Page** | A named snapshot of the 20×20 sampling grid; detected when ≥85% of its stable cells still match — "which view am I on." |
| **Group** | An app/site that owns many pages and anchors; *active* if any member page is detected or any tied anchor is matching. |
| **Signal** | The execution token that rides *with the data* along edges; a node fires when a signal reaches it. |
| **`_fire`** | The pure function implementing a node's behaviour on a plain snapshot; runs on the FlowRunner thread. |
| **RunContext** | Per-run handle exposing perception accessors, `llm()`, memory, logging, progress, and cancellation to `_fire`. |
| **LTM / STM** | Long-term (unbounded) / short-term (≤20, FIFO) in-RAM keyed variable stores. |
| **Hidden prompt** | A reusable system instruction stored by name, folded in front of an LLM node's text input, never shown on the canvas. |
| **Activity bar** | Per-node loading bar showing where the signal is (progress for timed nodes, flash for instant). |
| **Runaway guard** | Aborts a *tight* loop (5000 steps in <2 s, no pacing) while letting paced loops run forever. |

---

*Generated from the working prototype (v0.27.1) and its CHANGELOG. The prototype
is the reference implementation; this document is the contract for rebuilding it
to production standard through the standard pipeline.*
