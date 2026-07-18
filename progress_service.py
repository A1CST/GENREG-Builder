"""progress_service — parses the master CHANGELOG.md into structured progress
data for the /progress dashboard.

Three views come out of one parse:
  1. daily-by-project counts  -> the multi-line "activity per project per day"
     chart.
  2. impact-weighted timeline -> answers "activity != progress": every entry is
     classified into an impact LEVEL (discovery / validation / refutation /
     architecture / engineering / documentation / maintenance) with a weight, so
     twenty typo fixes and one new primitive do NOT count the same.
  3. project goal cards       -> measurable completion toward each project's
     target, read from progress_data/goals.json (editable; never hardcoded in
     the template).

Pure read-side analytics — no training, no state. Everything is derived live
from CHANGELOG.md so the page is always current.
"""

import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
CHANGELOG = os.path.join(_HERE, "CHANGELOG.md")
GOALS = os.path.join(_HERE, "progress_data", "goals.json")

# ── project taxonomy ────────────────────────────────────────────────────────
# (key, label, colour, regex). First match wins. Colours mirror PROJECT_GROUPS
# in app.py so the two pages read as one system.
PROJECTS = [
    ("i2",        "I2",        "#2ea043", r'\bI2\b|zetifile|latent stream|/i2\b|\bwoven\b|canvas browser|primary 10\.0\.0'),
    ("diff",      "DiffEvo",   "#d2a8ff", r'diffevo|/diff\b|denois|diffusion'),
    ("cifar",     "CIFAR",     "#ff7b72", r'cifar|radial|seed-stack|detbank|encoder|grammar-v2'),
    ("mnist",     "MNIST",     "#e3b341", r'\bmnist\b'),
    ("lm",        "LM",        "#56d364", r'\blm\b|/lm\b|cloze|evolang|wordpipe|convers|bigram|trigram|char-level|\btoken|perplex|\bppl\b|semantic'),
    ("resnet",    "ResNet",    "#d29922", r'\bresnet\b|\bR0\b|stacked-residual'),
    ("animation", "Attention/Anim", "#ff9e64", r'animation|attention|\bcursor\b|shape model|temporal|persistence operator'),
    ("video",     "Video",     "#f0883e", r'/video|slideshow|storyboard|ffmpeg|anim_service|\brig\b|\bpose'),
    ("images",    "Images",    "#a5a5f5", r'/images|reverse tab|\bblip\b|caption'),
    ("infra",     "Infra/UI",  "#8b95a1", r'/runs\b|navbar|termdock|configpanel|\bnav\b|changelog modal'),
]

# ── impact taxonomy ─────────────────────────────────────────────────────────
# ordered highest-signal first; first regex to hit assigns the level.
IMPACT = [
    ("refutation",    "Refutation",    4.0, "#ff7b72",
     r'\bFAILS?\b|refut|invalidat|\bnull\b|regress|earns nothing|Goodhart|does not (hold|reproduce)|no gain|debunk|contradict'),
    ("discovery",     "Discovery",     5.0, "#f778ba",
     r'\bRECORD\b|first (semantic|ever|time)|broke .*ceiling|new .*(best|record)|primitive|\blaw\b|milestone|unlock|emerged|novel mechanism|new best'),
    ("validation",    "Validation",    3.0, "#56d364",
     r'validat|confirm|verified end-to-end|reproduc|holds\b|leave-one-out|\bLOO\b|proved|proof\b'),
    ("architecture",  "Architecture",  3.0, "#79c0ff",
     r'rebuild|redesign|\bpivot\b|platform|restructure|new page|architecture|mechanism|NEW:|introduc'),
    ("documentation", "Documentation", 0.5, "#a5a5f5",
     r'changelog|\bdoc(s|ument)|\bguide\b|README|RESUME|\bpage\b update|write-up|writeup'),
    ("maintenance",   "Maintenance",   0.3, "#8b95a1",
     r'cleanup|clean up|deploy|\bpush(ed)?\b|restart|refactor|gitignore|shadow copy|bump|rename|tidy'),
    ("engineering",   "Engineering",   1.0, "#d29922",  # default catch-all
     r'.'),
]

_ENTRY = re.compile(r'(?=^- \*\*\[\d{4}-\d{2}-\d{2}\])', re.M)
_HEAD = re.compile(r'- \*\*\[(\d{4}-\d{2}-\d{2})\] \(([^)]*)\)\*\* — (.*)', re.S)


def _project_of(body):
    for key, _label, _c, pat in PROJECTS:
        if re.search(pat, body, re.I):
            return key
    return "other"


def _impact_of(body):
    for key, _label, _w, _c, pat in IMPACT:
        if re.search(pat, body, re.I):
            return key
    return "engineering"


def _load_goals():
    try:
        with open(GOALS, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def parse():
    """Parse CHANGELOG.md -> the full data bundle for the dashboard."""
    try:
        txt = open(CHANGELOG, encoding="utf-8").read()
    except OSError:
        txt = ""

    rows = []
    for chunk in _ENTRY.split(txt):
        m = _HEAD.match(chunk)
        if not m:
            continue
        date, author, body = m.group(1), m.group(2), m.group(3).replace("\n", " ")
        body = re.sub(r'\s+', ' ', body).strip()
        tm = re.match(r'\*\*(.+?)\*\*', body)
        title = (tm.group(1).strip().rstrip('.:') if tm
                 else (body[:90] + ('…' if len(body) > 90 else '')))
        rows.append({
            "date": date,
            "author": author,
            "project": _project_of(body),
            "impact": _impact_of(body),
            "title": title,
            "body": body,
        })

    dates = sorted({r["date"] for r in rows})
    proj_keys = [p[0] for p in PROJECTS] + ["other"]
    impact_keys = [i[0] for i in IMPACT]
    impact_w = {i[0]: i[2] for i in IMPACT}

    # daily-by-project (line chart)
    daily = {d: {k: 0 for k in proj_keys} for d in dates}
    proj_total = {k: 0 for k in proj_keys}
    proj_days = {k: set() for k in proj_keys}
    for r in rows:
        daily[r["date"]][r["project"]] += 1
        proj_total[r["project"]] += 1
        proj_days[r["project"]].add(r["date"])

    # impact timeline (weighted)
    impact_daily = {d: {k: 0 for k in impact_keys} for d in dates}
    impact_total = {k: 0 for k in impact_keys}
    weighted_daily = {d: 0.0 for d in dates}
    # per-project impact matrix + weighted score (activity != progress, per project)
    proj_impact = {k: {ik: 0 for ik in impact_keys} for k in proj_keys}
    proj_weighted = {k: 0.0 for k in proj_keys}
    for r in rows:
        impact_daily[r["date"]][r["impact"]] += 1
        impact_total[r["impact"]] += 1
        weighted_daily[r["date"]] += impact_w[r["impact"]]
        proj_impact[r["project"]][r["impact"]] += 1
        proj_weighted[r["project"]] += impact_w[r["impact"]]

    return {
        "generated_from": os.path.basename(CHANGELOG),
        "total_entries": len(rows),
        "dates": dates,
        "projects": [
            {"key": k, "label": l, "color": c,
             "total": proj_total.get(k, 0),
             "active_days": len(proj_days.get(k, set())),
             "weighted": round(proj_weighted.get(k, 0.0), 1),
             "impact": proj_impact.get(k, {})}
            for (k, l, c, _p) in PROJECTS
        ] + [{"key": "other", "label": "Other", "color": "#3a4250",
              "total": proj_total.get("other", 0),
              "active_days": len(proj_days.get("other", set())),
              "weighted": round(proj_weighted.get("other", 0.0), 1),
              "impact": proj_impact.get("other", {})}],
        "impact_levels": [
            {"key": k, "label": l, "weight": w, "color": c, "total": impact_total.get(k, 0)}
            for (k, l, w, c, _p) in IMPACT
        ],
        "daily": [
            {"date": d, **daily[d]} for d in dates
        ],
        "impact_daily": [
            {"date": d, "weighted": round(weighted_daily[d], 1),
             "count": sum(impact_daily[d].values()), **impact_daily[d]}
            for d in dates
        ],
        "goals": _load_goals(),
        "entries": rows,
    }
