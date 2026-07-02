# GENREG API Backend

FastAPI HTTP service exposing everything the GUI can do. Same adapter
registry, same persistence format — just a different client surface.
Use cases: cURL testing, scripted experiments, remote clients,
potentially a web UI later.

**Not** a replacement for the desktop GUI. Runs in parallel.

## Design constraints

- Local-only for v1. No auth, CORS open, binds 127.0.0.1.
- Reuses `gui/adapters`, `gui/persistence`, `components/*`. Zero duplication.
- Training subprocess is per-API-call; multiple concurrent trainings
  allowed (each on its own run_id).
- Inference is synchronous (fast) — no queue needed for v1.
- Long-running training exposes status + log streaming endpoints.

## Endpoints

### Health
- `GET /health` — `{ok, gpu_available, adapter_count, runs_dir}`

### Adapters
- `GET /adapters` — list registered adapters with KIND + NAME
- `GET /adapters/{display_name}/schema` — config schema
- `GET /adapters/{display_name}/datasets` — supported datasets
- `GET /adapters/{display_name}/metrics` — metric specs
- `GET /adapters/{display_name}/panels` — introspection panel specs

### Configs
- `POST /configs/validate` — `{adapter, config}` → `{ok, issues: []}`

### Templates
- `GET /templates` — list every valid JSON in `gui/templates/`
- `GET /templates/{filename}` — full template payload

### Training
- `POST /training/start` — `{adapter, config, label, notes?}` → `{run_id}`
- `GET /training/{run_id}` — status + latest metrics + pid + exit_code
- `POST /training/{run_id}/stop` — cooperative SIGTERM
- `GET /training/{run_id}/log` — full captured log (plain text)
- `GET /training/{run_id}/stream` — text/event-stream of log lines
  (Server-Sent Events; Accept: text/event-stream)
- `GET /training/{run_id}/history` — JSON list of per-gen metrics
- `GET /training/active` — list of currently-running run_ids

### Inference
- `POST /inference` — `{adapter, checkpoint_path, input, genome_idx?}`
  → `{output, output_text, panels_summary}`
  (panels_summary gives shapes + small previews; full panels returnable
  via `?full=true` query param)

### Runs (persisted)
- `GET /runs` — list all saved runs with meta
- `GET /runs/{folder}` — meta + history
- `DELETE /runs/{folder}` — delete
- `GET /runs/{folder}/checkpoint` — download checkpoint.pkl

## File layout

```
api/
├── __init__.py
├── main.py             ← FastAPI app, route registration, run-registry
├── models.py           ← Pydantic request/response schemas
├── training_runner.py  ← non-Qt subprocess manager for training
├── routes/
│   ├── health.py
│   ├── adapters.py
│   ├── training.py
│   ├── inference.py
│   ├── runs.py
│   └── templates.py
└── PLAN.md             ← this doc
```

## Running

```bash
cd LLM
python -m api.main           # binds 127.0.0.1:7077 by default
# or
uvicorn api.main:app --host 127.0.0.1 --port 7077 --reload
```

Interactive docs at `http://localhost:7077/docs` (Swagger UI, auto-generated).

## Concurrency

Each training run is a thread owning a subprocess. The app registry is
`dict[run_id, RunnerHandle]`. SSE clients poll the handle's queue.
Runners clean up their queue on exit. The app does **not** limit
concurrent trainings — that's user responsibility (GPU can only do one
well at a time).

## Auth / security (out of scope for v1)

No auth. Do not expose to the internet. For a cloud deployment:
- Reverse-proxy with TLS + auth (nginx, Caddy)
- Add bearer-token middleware
- Consider rate-limiting long-running endpoints

## What it enables

- `curl` your way through a full experiment loop without the GUI.
- Scripted experiment sweeps: fire N training runs with different configs
  and query status via a loop.
- Potentially a browser-based GUI later (shares same backend).
- External tooling (grafana) can poll `/training/{id}` for metrics.
