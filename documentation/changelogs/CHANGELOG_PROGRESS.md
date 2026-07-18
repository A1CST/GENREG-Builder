# Changelog — PROGRESS

Per-project log for the PROGRESS dashboard (`/progress`). New entries go at the
top, and also in the master `CHANGELOG.md`.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-18] (Claude)** — **Fresh data on every visit.** Added `Cache-Control:
  no-store` to `/api/progress/data` + `{cache:"no-store"}` on the client fetch, so a
  visit always re-parses the current `CHANGELOG.md` (endpoint already re-reads per
  request; this stops the browser cache from serving stale JSON). Not live-polling —
  updates on load. **Flask restart required** (app.py changed).

- **[2026-07-18] (Claude)** — **Layout fix + interactive dots.** (1) `.pg-main` is now
  the `flex:1` internal-scroll region so the page no longer clips under the taskbar and
  the injected terminal dock stays docked at the bottom (present on the page). (2) Chart
  dots are interactive: hover -> viewport-aware popup (flips up near the bottom edge)
  listing that day's entry titles; click -> modal with the full changelog text (date,
  author, impact badge, body). `progress_service.py` now emits an `entries` array;
  `progress.js` builds project|date + date indexes and drives the tooltip/modal;
  `progress.html` gains modal markup + styles. **Flask restart required** (Python caches
  `progress_service.py`).

- **[2026-07-18] (Claude)** — **Created the /progress dashboard.** New Flask page
  that reads the master `CHANGELOG.md` and renders: (1) measurable **goal cards**
  per project (MNIST complete at 0.9909; CIFAR 0.7111→0.75; ResNet 0.948→0.95;
  LM 69%→100% of cloze ceiling) from an editable `progress_data/goals.json`;
  (2) a **multi-line chart** of changelog entries per project per day, with
  clickable legend toggles; (3) an **impact-weighted timeline** answering
  "activity ≠ progress" — every entry is auto-classified into an impact level
  (Discovery 5 / Refutation 4 / Validation 3 / Architecture 3 / Engineering 1 /
  Documentation 0.5 / Maintenance 0.3) and shown as stacked daily bars with a
  weighted-score line; (4) a **per-project impact composition** (how much of each
  project's volume is discovery vs. cleanup); (5) a computed read-out. Backend
  `progress_service.py` (keyword taxonomy + parser), routes `/progress` +
  `/api/progress/data`, template + `static/progress.js` (all charts hand-rolled
  inline SVG, no external libs). Nav entry added to the Workspace group;
  changelog-modal mapping added. **Flask restart required** to serve the routes.
