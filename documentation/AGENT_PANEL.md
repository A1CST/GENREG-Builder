# Agent Panel — AI notice feed for the GENREG workspace

The **Agent panel** is the floating Agent window present on every page (Build,
Runs, Tree LM, DiffEvo, I2, Docs). It is the shared channel where AI
assistants and automated jobs post updates: finished training runs, test
results, changes they made, alerts. Notices carry an unread badge; a notice
arriving while a page is open pops the panel up automatically (the
end-of-run "alarm").

- **Move it**: drag the header. Position persists across pages and reloads.
- **Minimize**: the `—` button collapses it to an "Agent" pill (top-right). The
  pill shows the unread count and pulses while anything is unread.
- **Mark read**: clears the badge; unread rows are outlined until then.

## Where notices live

`agent_store/notices.jsonl` — one JSON object per line:

```json
{"id": 42, "ts": "2026-07-04T21:05:00", "kind": "run", "source": "tree-lm",
 "title": "tree-lm run finished", "body": "run_id: ...\nseconds: 812", "run_id": "..."}
```

`kind` is one of `info` (grey), `test` (green), `run` (blue), `alert` (red).
The panel polls `GET /api/agent/notices` every ~8 s. The file is trimmed to
the newest 1000 notices automatically.

## Automatic run alarms

Every project's training jobs post a `run` notice when they end (finished or
stopped): the engine trainer (Snake / 2048 / etc. via `/train`), Tree LM
runs, encoder trainings, config sweeps, and DiffEvo runs. No setup needed —
the hooks live in the job hubs, so a run you started and walked away from
still raises its alarm.

## Posting a notice

**From the command line** (works with or without the server running):

```
python agent_notify.py "title" "body text" --kind test --source claude
pytest -q 2>&1 | python agent_notify.py "test results" - --kind test
```

**From Python**:

```python
import agent_board
agent_board.post("UTF-8 sampler verified", "27/27 checks pass", kind="test", source="claude")
```

**Over HTTP** (server must be up):

```
POST http://127.0.0.1:5000/api/agent/notices
{"title": "...", "body": "...", "kind": "info", "source": "gpt", "run_id": null}
```

## Tying a CLI AI to the workspace

Any terminal-based AI (e.g. Claude Code) opened in the workspace terminal
dock shares this machine, this repo, and these tools. To wire one in:

1. Open any page and click **Claude** in the terminal dock (or `+ New Tab`
   and launch your CLI agent by hand). The terminals are daemon-backed, so
   the session survives page switches and Flask restarts.
2. Give it the standing instructions (paste or put in its memory/config):
   - Work happens in `C:\Users\paytonm\Documents\GENREG`.
   - **Log every change** at the top of `CHANGELOG.md` (see its format header)
     **and** append the same entry to the matching project changelog(s) under
     `documentation/changelogs/` (BUILD, TREE, DIFFEVO, ANIMATION, I2, LM).
   - **Post to the Agent panel** when done with a piece of work or a test:
     `python agent_notify.py "<what happened>" "<details>" --kind test|info|alert --source <your-name>`.
   - Never restart the Flask server on port 5000; say so when a restart is
     needed.
3. That's it — the panel is file-based, so anything that can run a shell
   command in the repo can post. Multiple AIs can post side by side; the
   `source` field says who spoke.

## Files

| file | role |
| --- | --- |
| `agent_board.py` | store + `post()` / `list_notices()` / `post_run_event()` |
| `agent_notify.py` | CLI poster for terminals / AIs |
| `agent_store/notices.jsonl` | the feed itself (JSON lines) |
| `static/agentpanel.js` | the floating panel (injected on every page) |
| `/api/agent/notices` | GET feed / POST a notice |
