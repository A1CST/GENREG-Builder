# GENREG

A Flask-based web interface for building the GENREG project, with real
interactive terminals embedded in the browser — including the ability to run
the `claude` TUI right inside the GUI.

## Architecture

```
Browser  <-- WebSocket -->  Flask (app.py)  <-- TCP -->  Daemon  <-- ConPTY -->  PowerShell(s)
        (xterm.js)                         (terminal_daemon.py)      (pywinpty)
```

- The terminals are **real ConPTY pseudo-terminals** (via `pywinpty`), so
  fully interactive programs work — arrow keys, colors, live redraw, the
  `claude` TUI, REPLs, prompts, etc.
- They live in a **separate long-lived process** (`terminal_daemon.py`), not
  inside Flask. You can stop/restart the Flask server — e.g. while editing
  `app.py` — and your terminals and any running programs keep going. When
  Flask comes back it reconnects and replays each terminal's recent screen.
- A single **WebSocket** carries keystrokes/resize/control ops to the daemon
  and streams raw terminal output back.

## Files

| File                   | Purpose                                                       |
|------------------------|---------------------------------------------------------------|
| `app.py`               | Flask server + WebSocket relay. Auto-starts the daemon.       |
| `terminal_daemon.py`   | Owns the ConPTY shells + scrollback. Survives Flask restarts. |
| `templates/index.html` | The GUI shell.                                                |
| `static/app.js`        | Tabbed xterm.js front-end (WebSocket client).                 |
| `static/style.css`     | Styling.                                                      |
| `static/vendor/`       | Vendored xterm.js, xterm.css, fit addon (no CDN at runtime).  |

## Run

```powershell
cd $HOME\Documents\GENREG
pip install -r requirements.txt      # first time only
python app.py
```

Then open <http://127.0.0.1:5000>.

## Talking to Claude inside the GUI

Just type it in any terminal like you would in a normal shell:

```powershell
claude
```

Because these are real PTYs, the interactive Claude Code TUI runs directly in
the browser tab. (The earlier "must pass a prompt" error only happened because
the previous version used plain pipes, not a PTY — that's fixed.)

## Controls

- **Type anything** — it's a real terminal; keystrokes go straight to the shell.
- **+ New Tab** — open another independent terminal.
- **× on a tab** — close that terminal.
- **Clear** — clear the active terminal.
- **Restart** — kill and respawn the shell with fresh state.
- **Stop** — kill the shell process (halts a running program); Restart brings it back.
- Resizing the window reflows the active terminal (the PTY is told the new size).

## Notes

- Stopping the **daemon** (not Flask) ends all terminals. It keeps running in
  the background after you close Flask. To stop it:
  ```powershell
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'terminal_daemon\.py' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
  ```
- Binds to `127.0.0.1` only (Flask on 5000, daemon on 5001) — not exposed to
  your network.
- Requires Windows (ConPTY via `pywinpty`).
