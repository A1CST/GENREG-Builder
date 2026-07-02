# GENREG API Backend

FastAPI HTTP service exposing the same adapters / persistence / training
machinery the GUI uses. Use it for scripted experiments, cURL testing,
or as a future foundation for a web UI.

## Run

```bash
cd LLM
python -m api.main                  # binds 127.0.0.1:7077
# or with auto-reload:
uvicorn api.main:app --host 127.0.0.1 --port 7077 --reload
```

Open `http://localhost:7077/docs` for interactive Swagger UI.
`http://localhost:7077/redoc` for ReDoc.

## Quick tour with curl

```bash
# 1. Health
curl -s http://127.0.0.1:7077/health | jq

# 2. List adapters
curl -s http://127.0.0.1:7077/adapters | jq

# 3. Get an adapter's config schema (every editable hyperparameter)
curl -s "http://127.0.0.1:7077/adapters/Tokenizer%20G%20%28hash-bit%2C%20stream%29/schema" | jq

# 4. Get adapter defaults
curl -s "http://127.0.0.1:7077/adapters/Tokenizer%20G%20%28hash-bit%2C%20stream%29/defaults" | jq

# 5. Validate a config
curl -s -X POST http://127.0.0.1:7077/configs/validate \
  -H 'Content-Type: application/json' \
  -d '{"adapter":"Tokenizer G (hash-bit, stream)","config":{"POP_SIZE":500}}'

# 6. Start a training run
curl -s -X POST http://127.0.0.1:7077/training/start \
  -H 'Content-Type: application/json' \
  -d '{"adapter":"Tokenizer G (hash-bit, stream)",
       "config":{"POP_SIZE":300,"N_GENERATIONS":100},
       "label":"api_smoketest"}'
# → {"run_id":"abc123...", "run_dir":"..."}

# 7. Poll status
curl -s http://127.0.0.1:7077/training/abc123 | jq

# 8. Stream logs (Server-Sent Events)
curl -s -N http://127.0.0.1:7077/training/abc123/stream

# 9. Stop early
curl -s -X POST http://127.0.0.1:7077/training/abc123/stop

# 10. List saved runs
curl -s http://127.0.0.1:7077/runs | jq

# 11. Get one run's full meta + history
curl -s http://127.0.0.1:7077/runs/api_smoketest_2026-04-14_22-00-00 | jq

# 12. Inference with a checkpoint
curl -s -X POST http://127.0.0.1:7077/inference \
  -H 'Content-Type: application/json' \
  -d '{"adapter":"Tokenizer G (hash-bit, stream)",
       "checkpoint_path":"components/tokenizer/checkpoints_G/tokenizer_gen_03000.pkl",
       "input":"hello world",
       "genome_idx":108}' | jq

# 13. Templates
curl -s http://127.0.0.1:7077/templates | jq
```

## Endpoint reference

### Health
- `GET /health` — `{ok, gpu_available, adapter_count, active_trainings, runs_dir}`

### Adapters
- `GET /adapters` — registered adapters
- `GET /adapters/{display_name}/schema` — config schema (per-field metadata)
- `GET /adapters/{display_name}/defaults` — default config values
- `GET /adapters/{display_name}/datasets` — supported datasets
- `GET /adapters/{display_name}/metrics` — metric specs (for charts)
- `GET /adapters/{display_name}/panels` — introspection panel specs

### Configs
- `POST /configs/validate` — body `{adapter, config}` → `{ok, issues: []}`

### Templates
- `GET /templates?folder=...` — list templates in folder (default `gui/templates/`)
- `GET /templates/{filename}?folder=...` — get one template

### Training
- `POST /training/start` — body `{adapter, config, label, notes?}` → `{run_id, run_dir}`
- `GET /training/active` — currently-running run_ids
- `GET /training/{run_id}` — `{status, gens_completed, latest_metrics, ...}`
- `POST /training/{run_id}/stop` — cooperative stop (SIGTERM)
- `GET /training/{run_id}/log` — full captured log (text/plain)
- `GET /training/{run_id}/history` — list of per-gen metric dicts
- `GET /training/{run_id}/stream` — SSE stream of `{type: log|metrics|status}` events

### Inference
- `POST /inference` — body `{adapter, checkpoint_path, input, genome_idx?, full_panels?}`

### Runs (persisted)
- `GET /runs` — list saved runs
- `GET /runs/{folder}` — meta + history
- `DELETE /runs/{folder}` — delete
- `GET /runs/{folder}/checkpoint` — download checkpoint.pkl

## How it relates to the GUI

Both share:
- `gui/adapters/` — adapter registry + protocol
- `gui/persistence/` — runs/templates I/O
- `components/*` — actual model code

API never imports from `gui/shell/` or `gui/tabs/` (no Qt dep). Training
subprocess management is `api/training_runner.py` — same logic as
`gui/shell/trainer.py` minus PyQt signals (uses `threading` + `Queue`).

A run started via the GUI shows up in `GET /runs` once it finishes
(both write to the same `gui/persistence/runs/` folder). A run started
via API can be inspected in the GUI's Runs tab.

## Concurrency

You can fire multiple training runs simultaneously via API. The server
won't stop you. Practically: one GPU can only train one efficiently;
others will stall waiting for VRAM. The server doesn't queue — that's
your responsibility.

The GUI's TrainingTab and the API's `/training/start` operate on
independent registries; they don't see each other's in-memory state.
Persisted runs (after completion) are visible to both.

## Auth / security

None. Local-only (`127.0.0.1`). Don't expose to a network without
adding a reverse-proxy + auth layer.
