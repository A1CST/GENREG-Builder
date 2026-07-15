# AGENTS.md - standing instructions for every AI agent in this workspace

Read this before doing anything. These rules apply to every session, every
agent, every project in GENREG. They are not suggestions.

---

## 1. Changelogs - every change, two places

1. **Main changelog** (`CHANGELOG.md`, repo root): every change gets an entry
   at the TOP of the file, newest first. Never rewrite or delete existing
   entries. Multiple agents share this file: **Read it immediately before
   editing** (another session may have pushed an entry since you last looked)
   and re-read + retry on edit conflicts.
2. **Project changelog** (`documentation/changelogs/CHANGELOG_<PROJECT>.md`):
   every change that touches a project is ALSO appended to that project's
   changelog, same entry or a project-scoped version of it. If the file does
   not exist for the project you touched, create it (copy the header
   convention from an existing one).

Do not batch days of work into one entry after the fact. Log as you land.

## 2. New project checklist

When you create a new project (a new page / pipeline / model line), ALL of
the following before you call it done:

- [ ] **Runs folder**: create `runs/<project>/` for its run records.
- [ ] **Navbar**: add the project to `templates/_nav.html` so every page can
      reach it. Verify the link renders on at least one existing page.
- [ ] **Project changelog**: create
      `documentation/changelogs/CHANGELOG_<PROJECT>.md` with the standard
      header, and add the project mapping to the changelog modal (missing
      mappings have bitten before - see the /evolang and /mnist fix).
- [ ] **Main changelog entry** announcing the project.
- [ ] **Agent notice** announcing the project (see rule 3).
- [ ] Note in your entry whether a **Flask restart** is needed for new
      routes/templates - the user restarts; agents never restart the server.

## 3. Agent alerts - everything generates a notice

- After **every task and every test**, post a notice:
  `python agent_notify.py "<title>" "<body>" --kind test|info|run|alert --source claude`
- **Every training run must raise an alert when it ends** (finished OR
  stopped/crashed). If you build a new trainer, wire the notice into its job
  hub or completion path so runs started-and-walked-away-from still alarm.
  No silent runs.

## 4. Training runs go into the RUNS project

Every project's training runs are ALSO recorded into the RUNS project
(`runs/<...>` with the standard file trio) so they appear on the **/runs
page - that page layout is the shared instrument for testing and comparing
run data across projects.**

The record is five files in `runs/<env>/<run-id>/` (see
`radial_stack._record_run` for a working reference):

- `config.json` - id, environment, created, full config, status
- `history.jsonl` - one line per round/gen: fitness, added, n
- `summary.json` - final stats (test metrics, params, counts)
- `meta.json` - label, tags, favorite, group
- `report.json` - config + stats + tail of the log lines

If your trainer does not record runs, add it before running experiments -
"the data looks great in my log file" does not count. Untracked training
runs are the reason this file exists.

## 5. How to build a page

Every GENREG page follows the same skeleton. Copy it; do not invent a new
layout language per page.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GENREG — <Project></title>
  <link rel="stylesheet" href="{{ url_for('static', filename='vendor/xterm.css') }}" />
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}" />
  <style>/* page-scoped styles, prefixed (e.g. .lm-, .an-) */</style>
</head>
<body class="tlm-body">
  <header class="topbar">
    <div class="brand">GENREG<span class="brand-sub"><project> · <one-line subtitle></span></div>
    <div class="status">{% set nav_active = '<project>' %}{% include '_nav.html' %}</div>
  </header>
  <main> ... page content ... </main>
  <!-- the shared dock, ALWAYS at the bottom, in this order -->
  <script src="{{ url_for('static', filename='vendor/xterm.js') }}"></script>
  <script src="{{ url_for('static', filename='vendor/addon-fit.js') }}"></script>
  <script src="{{ url_for('static', filename='termdock.js') }}"></script>
  <script src="{{ url_for('static', filename='app.js') }}"></script>
  <script src="{{ url_for('static', filename='agentpanel.js') }}"></script>
  <script src="{{ url_for('static', filename='configpanel.js') }}"></script>
</body>
</html>
```

Conventions:
- Dark palette: page `#0b0d10`, cards `#0d1117` with `1px #1c232c` border,
  radius 8-10px; body text `#8b95a1`, emphasis `#c7d0da`/`#dbe2ea`.
- Numbers in `ui-monospace` with `font-variant-numeric: tabular-nums`.
- **No emojis anywhere in the UI.** Monochrome glyphs only (▼ ✓ ·) or words.
- Results render from exported JSON via small API endpoints - never hardcode
  numbers into templates.
- Syntax-check any nontrivial inline JS (`node --check`) before shipping.

### Append-only iteration pages (the /lm pattern - preferred for model lines)

The page is a stack of MODULES, one per iteration, newest at the BOTTOM.
Old modules are never edited. Modules are data-driven:

1. A registry json (e.g. `radial_data/lm_modules.json`): append-only list of
   `{id, date, kind, title, desc, export}` entries.
2. Two endpoints: `/api/<proj>/modules` (serves the registry) and
   `/api/<proj>/export/<name>` (serves a module's export json, whitelisted
   filename pattern only).
3. The template renders each module generically: title/date/desc, metric
   chips (top-1, top-5, params total/evolved/head, genomes, spaces), the
   ceiling-ladder bars, per-space table, and the model's raw output samples
   verbatim. Every module MUST show output and params.
4. Adding an iteration = write the export json + append one registry entry.
   No template edits.

### Auto-scroll to newest (required on iteration pages)

After all modules render, snap to the bottom so the newest module is what
the user sees, and give them a floating button to get back there:

```js
var main = document.getElementById("page-main");
function snap() {
  main.scrollTop = main.scrollHeight;
  window.scrollTo(0, document.body.scrollHeight);
}
// call snap() after the last module renders, and after any live module
// (e.g. autocomplete) appends new output
document.getElementById("snap-btn").addEventListener("click", snap);
```

```html
<button id="snap-btn" style="position:fixed;right:26px;bottom:26px;z-index:50;
  border-radius:18px;padding:8px 15px;font-size:12px;background:#1c2733;
  color:#dbe2ea;border:1px solid #2e7d5b;cursor:pointer">newest ▼</button>
```

## 6. How to add a project, end to end

1. **Route** in `app.py`: `@app.route("/<project>")` returning
   `render_template("<project>.html")`, near the other page routes.
2. **Template** `templates/<project>.html` using the skeleton above.
3. **Navbar**: add the project to `templates/_nav.html`; `nav_active`
   value in the template must match.
4. **API endpoints** for its data (exports under `radial_data/` or the
   project's own data dir; whitelist file patterns on any by-name route).
5. **Runs folder** `runs/<project>/` + run recording wired into the trainer
   (rule 4 above).
6. **Project changelog** + changelog-modal mapping (rule 2 above).
7. **Notices** wired (rule 3 above).
8. Main changelog entry + agent notice announcing it, with the
   **Flask restart** note. Then verify every checklist item in rule 2.

## 7. House rules that keep applying

- **No gradients.** Read `documentation/GENREG_RULES.md` before training
  anything. Closed-form ridge heads are the only fitting allowed.
- **Append, don't replace.** Project pages (e.g. /lm) are iteration logs:
  new work adds a module; old modules are never rewritten.
- **Test touched once.** Fitness never sees test; report the anchor and the
  classical ceilings next to every result.
- **Raw output in reports.** When reporting a run to the user, include the
  actual output lines and samples verbatim, not just summaries.
- **Never restart the Flask server** - tell the user when a restart is
  needed.
- **Pods**: keep a shadow copy of every pod file locally
  (`runpod_shadow/...`), and check for zombie/orphaned python processes
  before diagnosing slow runs.
