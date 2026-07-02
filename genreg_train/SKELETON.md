# GENREG training integration — SKELETON / design spec

Wire the **genreg-engine** (gradient-free neuroevolution substrate at
`project/genreg-engine-main`) to the GENREG web GUI so a user can train a genome
to actually play **Snake** and **2048**, shaped by the **Constraints** they check,
and watch it happen live: the game board on the main canvas, the champion genome
in the Microscope, and the constraint cone in PO Metrics.

Guiding rule from EEC (`project/EEC-main`): **no gradients, no designed
reward-gradient.** Fitness is a *world consequence* (apples eaten, tiles merged,
lifespan). Constraints are *costs/laws imposed on the world*, not reward shaping.

---

## 1. Data flow

```
Browser (control panel)                     Flask (app.py)                genreg_train
  │  Start Train ─ config JSON ─► WS /train ──► Trainer thread ──► Evolver (engine)
  │                                                 │  per generation:
  │  ◄──────────── event JSON frames ──────────────┤   • telemetry (fitness curve)
  │   • board frames  → main canvas                │   • best genome → layers
  │   • genome        → Microscope                 │   • best replay → board frames
  │   • telemetry/PO  → HUD + PO tab               │   • PO (constraint count)
  │  Stop ─ {op:stop} ─────────────► sets stop flag; thread joins; final event
```

The existing terminal daemon (`terminal_daemon.py`, WS `/ws`) is **untouched**.
Training uses a **separate** WS route `/train`. Training runs *in-process* in a
Flask worker thread (CPU neuroevolution, numpy) — no daemon, no subprocess.

---

## 2. Modules & responsibilities

| file | responsibility |
|------|----------------|
| `engine_api.py` | Put engine dir on `sys.path`; re-export the pieces we use; `genome_layers(g)` → weight matrices for the Microscope; `genome_summary(g)`. Single import surface so nothing else touches the engine path. |
| `envs.py` | `SnakeEnv`, `Game2048Env`. Pure game logic. `reset()->obs`, `step(action)->(obs,reward,done,info)`, `observe()->np.float32 vector`, `render_state()->JSON dict`, class attrs `N_IN`,`N_OUT`,`NAME`. No engine imports. |
| `agent.py` | `rollout(genome, env, hooks, rng, record=False) -> RolloutResult`. Recurrent stepping via `fresh_state`/`rstep`. Action selection per env. Applies per-step constraint hooks. Returns score, frames, stats. |
| `constraints_map.py` | `build_constraints(names, params) -> Constraints`. Turn the checkbox set + params into: obs transform, per-step survival hook, post-score fn, engine cost list, evolver kwargs, env-reseed cadence. One dataclass, faithful to EEC axes. |
| `trainer.py` | `Trainer(config, emit)`. Assemble env factory + fitness (`robust(shaped(episodic(rollout), *costs))`) + `Evolver`. `run()` loops generations, calls `emit(event)` each gen, honors `stop()`. `TrainConfig` dataclass + `parse_config(dict)`. |
| `run_headless.py` | CLI to train without the browser and print the fitness curve. The primary automated verification. |
| `__init__.py` | Export `Trainer`, `TrainConfig`, `parse_config`, env registry. |

Frontend:

| file | responsibility |
|------|----------------|
| `static/training.js` | Own the `/train` WS. Read config from the control panel, send it, consume events, fan them out to board / microscope / PO / HUD. Start/Stop button state. |
| `static/ui.js` | Board renderers gain a **live-state** path: `GENREG.board.setLive(state)` / `clearLive()`. When a live state is present, draw it; else the static illustrative board. |
| `static/po_metrics.js` | Already constraint-driven; add `GENREG.po.setPO(n)` so training can push the live ring count if constraints change server-side (optional). |
| `templates/index.html` | Training section (Generations, Start/Stop, HUD) + constraint-parameter inputs. |
| `static/style.css` | Styles for the above. |

---

## 3. WebSocket contracts

### 3.1 Browser → server (`/train`)

`start`:
```json
{ "op": "start",
  "environment": "snake",            // "snake" | "2048"
  "population": 100,                  // 1..1000
  "generations": 60,                 // 1..1000
  "constraints": ["energy","time"],  // checked constraint values
  "params": { "energy_budget": 120, "time_budget": 400, "noise": 0.1,
              "cost_strength": 0.01, "hidden": 24 },
  "snake": { "w": 20, "h": 15 },     // present for snake
  "seed": 0 }
```
`stop`: `{ "op": "stop" }`

### 3.2 Server → browser (events; every message has `type`)

- `started`  `{type, environment, generations, population, n_in, n_out, constraints}`
- `generation` (once per gen):
  ```json
  { "type":"generation", "gen":12, "generations":60,
    "fitness": {"best":3.4,"mean":1.1,"median":0.9},
    "best": {"score":5, "steps":180, "H":24, "leak":0.7, "bits":8.0},
    "po": 2,
    "genome": { "layers": [ {"rows":24,"cols":11,"w":[..]}, {"rows":3,"cols":24,"w":[..]} ] },
    "replay": { "env":"snake", "frames":[ <render_state>, ... ], "meta": {...} } }
  ```
  `genome.layers[*].w` is a flat row-major `number[]` (rows*cols). Microscope
  reshapes. `replay.frames` is a list of `render_state()` dicts (see §4).
- `done` `{type, gen, reason:"finished"|"stopped"|"error", best:{...}, message?}`
- `error` `{type, message}`

Bandwidth control: `replay` only for the **best** genome, sampled to ≤ `MAX_FRAMES`
(default 240) per gen; genome weights sent every gen but Float rounded to 4 dp.

---

## 4. Environments

### 4.1 SnakeEnv  (NAME="snake", N_IN=11, N_OUT=3)

- Grid `w×h` (from GUI game controls, clamped 5..60). Origin top-left, y down.
- Directions `DIRS=[(0,-1),(1,0),(0,1),(-1,0)]` = up,right,down,left. `dir` int.
- Actions (relative): `0=turn-left`, `1=straight`, `2=turn-right`. Prevents 180° reversal by construction.
- Snake starts length 3 centered, moving right. One food at a random empty cell.
- **Observation (11)** float32 in {0,1} unless noted:
  1–3 danger straight / danger left / danger right (next cell in that abs dir is wall or body)
  4–7 direction one-hot (up,right,down,left)
  8–11 food is up / right / down / left of head (absolute; two may be 1)
- **step**: turn, advance head. Death = wall or body (excluding the tail cell it vacates when not eating). Eat food → grow, `score+1`, `hunger=0`, respawn food, `reward=+1`. Else move (pop tail), `reward=+STEP_REWARD (0.01)`. `hunger+1`; if `hunger>max_hunger` → done ("starved"). Win (board full) → done.
- **Fitness (base, world-consequence)**: `apples*APPLE(1.0) + steps*STEP_REWARD(0.01)`. `max_hunger` default `2*(w+h)` (finite episodes, implicit metabolism). Constraints (energy/time/…) modify this — see §5.
- **render_state**: `{env:"snake", w,h, snake:[[x,y],...(head first)], food:[x,y]|null, score, steps, alive}`.

### 4.2 Game2048Env  (NAME="2048", N_IN=16, N_OUT=4)

- 4×4 int grid (0 empty, else tile value). Start: two spawns (`2` w.p. .9 else `4`).
- Actions `0=up,1=right,2=down,3=left`. Slide+merge (each tile merges once per move). A move is **valid** iff it changes the grid.
- `valid_moves()` → list of valid action indices. `step(action)` applies (caller guarantees valid), then spawns a tile. Game over when `valid_moves()` empty.
- **Observation (16)**: `log2(tile)/11` per cell row-major (0 for empty; 11≈log2(2048)).
- **Fitness**: cumulative game score = sum of merged tile values (world consequence). Report `max_tile` too.
- **render_state**: `{env:"2048", grid:[[..4..],..4..], score, moves, over, max_tile}`.

---

## 5. Constraint → effect mapping (all 12)

Each constraint touches ONE of five integration points. Faithful to the EEC axis
catalogue (`docs/CONSTRAINTS.md`). Same-axis pairs are noted (swap, don't stack).

| constraint | axis | integration point | effect |
|---|---|---|---|
| **energy** | survival | per-step + env | Metabolism budget. Start `energy_budget`; −1/step; +`food_energy` on eat; death at 0. Replaces plain hunger. |
| **mortality** | survival (alt) | per-step | Hazard death: each step, `p_death = hazard * hunger_frac`; sampled death. Swap with energy. |
| **time** | parsimony | post-score | `score *= 1/(1 + steps/time_budget)` — solving in fewer steps is favored (Occam). |
| **memory-rent** | parsimony(mem) | engine cost | `size_cost(cost_strength)` on H — memory must pay rent (needs dimension mutation on). |
| **efficiency** | parsimony | engine cost | `weight_cost(cost_strength)` — leaner weights at equal task. |
| **entropy** | active maint. | per-step obs/state | Recurrent state leaks extra: after `rstep`, `state *= (1-decay)`; forces active refresh. |
| **occlusion** | persistence | obs transform | With prob `occlusion_p`, zero the observation this step (sensory blackout) → memory must bridge. |
| **noise** | persistence | obs transform | Add `N(0, noise)` to observation. Swap with occlusion. |
| **scarcity** | diversity | evolver | Steady-state turnover: lower `elite`, higher `parent_frac` cull — niches persist. |
| **reproduction-cost** | selection | evolver + score | Only surplus reproduces: raise selection pressure (`parent_frac↓`), require score>0 to be a parent-eligible. |
| **perception-cost** | attention | per-step | Observing drains energy: extra `−perception_cost` energy/step (pairs with energy; if energy off, acts as a step penalty in post-score). |
| **non-stationarity** | plasticity | trainer | Re-seed env structure every `shift_every` gens (new food RNG stream / 2048 seed) — world drifts, plasticity favored. |

Notes:
- Engine-cost constraints (`memory-rent`,`efficiency`) require the dimension/weight
  machinery; we always run recurrence on, and enable dimension mutation when
  `memory-rent` is checked so `size_cost` can bite.
- Survival axis: if both `energy` and `mortality` checked, energy wins (they're the
  same axis; we log a note rather than double-applying — "swap, don't stack").
- Observation axis: `occlusion` and `noise` may both apply but we log the redundancy.

### 5.1 Constraint parameters (defaults)

| param | default | range | used by |
|---|---|---|---|
| `energy_budget` | 120 | 20..1000 | energy |
| `food_energy` | 60 | 5..500 | energy |
| `hazard` | 0.02 | 0..0.2 | mortality |
| `time_budget` | 400 | 50..5000 | time |
| `cost_strength` | 0.01 | 0..0.2 | efficiency, memory-rent |
| `decay` | 0.15 | 0..0.9 | entropy |
| `occlusion_p` | 0.25 | 0..0.9 | occlusion |
| `noise` | 0.15 | 0..1 | noise |
| `perception_cost` | 1.0 | 0..10 | perception-cost |
| `shift_every` | 15 | 2..500 | non-stationarity |
| `hidden` (H0) | 24 | 4..128 | all (net width) |

---

## 6. Trainer flow

```
cfg = parse_config(dict)
env_factory(rng) -> SnakeEnv/Game2048Env(cfg.snake, rng)
C = build_constraints(cfg.constraints, cfg.params)          # §5
def episode(g):                                             # one noisy episode
    return rollout(g, env_factory(fresh_rng()), C.hooks, rng).score_shaped
task   = episodic(episode)
world  = shaped(task, *C.engine_costs)                      # weight/size costs
fit    = robust(world, n=EPISODES_PER_EVAL)                 # median, noise-robust
make_g = genome factory: Genome(N_IN,N_OUT,H0); enable(recurrence); init_precision
hook   = chain(mutate_recurrence, mutate_precision, [mutate_dimensions if memory-rent])
ev     = Evolver(N_IN, N_OUT, fit, pop, H0, seed, telemetry=snapshot,
                 make_genome=make_g, mutate_hook=hook, **C.evolver_kwargs)
for gen in range(cfg.generations):
    if stop: break
    ev.step()
    if C.shift_every and gen % C.shift_every == 0: reseed env structure
    best = ev.best
    emit(generation_event(ev, best, replay=rollout(best, env, record=True)))
emit(done_event)
```

`EPISODES_PER_EVAL` default 5 (median). Population & generations from GUI.
Guardrails: `pop∈[1,1000]`, `generations∈[1,1000]`, `H0∈[4,128]`, snake `w,h∈[5,60]`.

---

## 7. Frontend integration points

- Control panel: new **Training** fieldset — `Generations` (number), `Start`/`Stop`
  buttons, a status line (`idle` / `gen 12/60 · best 3.4 · score 5`).
- **Constraint params**: a compact panel that reveals the inputs relevant to the
  checked constraints (energy_budget, time_budget, noise, cost_strength, …).
- `training.js`:
  - `assembleConfig()` reads environment, population, generations, checked
    constraints, params, snake W/H.
  - On `generation`: `GENREG.board.playReplay(replay)` (animate frames),
    `GENREG.scope.setGenome(layers)`, `GENREG.po.setPO(po)`, update HUD.
  - Start disables inputs; Stop re-enables.
- `ui.js`: `GENREG.board = { setLive, clearLive, playReplay }`. `drawCanvas`
  prefers `liveState` when set. Replay animates frames on a timer (~15 fps),
  independent of the mutation demo.
- Microscope: real genomes flow through the existing `setGenome`; the illustrative
  self-mutation loop pauses while training drives it (add `GENREG.scope.setExternal(true)`).

---

## 8. Verification plan

1. **Engine self-tests** already pass under numpy 2.4.6 (done).
2. **Headless snake** (`run_headless.py snake --gens 60 --pop 120`): report base
   fitness gen0 → genN and apples-per-episode; must beat a random-policy baseline
   by a clear margin (target: mean apples/episode strictly increasing; best genome
   eats ≥ a few apples reliably).
3. **Headless snake + constraints** (energy, time): still trains; energy shortens
   episodes; time favors efficiency. No crashes; numbers reported.
4. **Headless 2048** (`run_headless.py 2048 --gens 60`): best score beats random
   baseline (random valid moves) by a clear margin.
5. **Constraint unit checks**: each constraint changes behavior in the expected
   direction (e.g., energy caps steps; noise lowers obs SNR; memory-rent shrinks H).
6. **Browser smoke**: start Flask; a python WS client sends a small `start`;
   receive `started`, ≥1 `generation` with genome+replay, then `stop`→`done`.
   Terminals still work (regression check).
7. **Manual**: open the app, Snake + Energy, Start → board plays, Microscope shows
   the champion, PO shows rings. (User does the final visual review.)

Baselines to record in `run_headless.py`: random-policy mean score over 200 eps.

---

## 9. Five-pass revision checklist (do after it works)

- [ ] **Pass 1 — correctness**: game rules (snake self/tail/wall, 2048 merge-once,
      spawn odds, valid-move masking), observation correctness, fitness signs.
- [ ] **Pass 2 — robustness/edge cases**: pop=1, generations=1, snake 5×5, full
      board win, all-constraints-on, no-constraints, WS disconnect mid-train,
      double-start, stop before first gen, NaN/inf guards.
- [ ] **Pass 3 — EEC faithfulness**: no reward-gradient leaked in; constraints are
      costs/laws not shaping; same-axis swap handled; costs actually bite.
- [ ] **Pass 4 — performance/bandwidth**: frame sampling, weight rounding, per-gen
      wall-clock at pop 100 / gen 60, main-thread jank on the client, memory.
- [ ] **Pass 5 — UX/polish**: disabled states, status/HUD clarity, param ranges,
      error surfacing, CHANGELOG, code comments matching house style.

Each pass: note findings + fixes in CHANGELOG and in `REVISIONS.md`.

---

## 10. Assumptions / decisions

- Training is in-process in Flask (threaded=True already). One active run at a
  time (a new `start` cancels the previous). Fine for a single-user lab tool.
- Snake uses 11-feature relative-action encoding (proven learnable by tiny
  evolved nets) — not raw pixels. 2048 uses log2 tile features.
- Recurrence is always on (control/agent worlds); precision always initialized;
  dimension mutation only when a size cost is present.
- "Language / cartpole / humanoidv5" remain non-trained placeholders (out of scope
  now, as the user said only snake & 2048).
```
