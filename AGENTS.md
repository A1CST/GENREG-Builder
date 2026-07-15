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

## 5. House rules that keep applying

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
