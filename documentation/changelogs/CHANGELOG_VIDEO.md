# Changelog — VIDEO

Per-project log for the VIDEO line. Seeded 2026-07-14 from the master
CHANGELOG.md (all entries mentioning this project); new VIDEO entries go at
the top of the log below, and also in the master CHANGELOG.md.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-12] (Claude)** — Built **Radial Space v3 §11.1: real screen-capture
  fingerprinting** — actual desktop frames, not simulated streams.
  `radial_screen.py` grabs frames via PIL ImageGrab (~36ms full-res, downscaled
  to 160×90), extracts per-frame features (brightness, edge density,
  colourfulness, frame-to-frame motion), maps each through the winning linear M
  into a radial traversal path, and fingerprints the clip with a 10-dim stat
  vector. Nearest-centroid classifier with leave-one-out accuracy. Endpoints
  `/api/radial/screen/{record,train,classify,status,clear}` (record/classify
  block for N seconds of live capture). New **"Screen"** page mode: pick an
  activity → Record while doing it → paths overlay by label (idle/browsing/video/
  coding) → Train → "What am I doing now?" classifies live from path shape with
  confidence bars. Verified end-to-end: real capture 10fps, idle reads motion
  ~0.06; train/classify plumbing gives LOO 100% on separable clips. The real
  separation numbers come from the user recording actual activity. **Needs Flask
  restart to serve the routes.**
- **[2026-07-12] (Claude)** — Built + tested **Radial Space v2** from
  `~/Downloads/radial_space_theory_v2.pdf` (adds §5 Memory + §6 Computation
  claims and the §9.2/§9.3 test batteries). `radial_memory.py` implements the v2
  reference code (`mapping`, `lookup`, `chain`) and runs all suites honestly.
  Results: **9.1 8/8, 9.2 7/8, 9.3 7/9.** The two failures are the load-bearing
  new claims: 9.2.3 proximity preservation FAILS (adjacent inputs land 1.31×
  *farther* than random — "proximity IS similarity" is false, because
  phi=v*2.47 mod 2π scrambles neighbours) and 9.3.5 reversibility FAILS (chain
  uses abs+cos, both many-to-one). The chain is a contraction (Lyapunov -49) that
  collapses every input to 0 — a trivial dynamical system, not a processor;
  single logic gates ARE realizable (cos nonlinearity, XOR incl.) but can't be
  wired together. **Constructive finding (the paper's own §10.3/§11.3): the
  mapping M is the whole lever.** Swapping the broken M for a proximity-
  preserving one fixes both instantly — proximity **1.3→0.02**, activity-stream
  classifier **70%→98%**. So the one real capability — a traversal path that
  fingerprints activity (idle/switching/video) — works, but only under a mapping
  the paper didn't use. New endpoints `/api/radial/v2suite` + `/api/radial/
  traversal`; page got a **"Memory v2"** mode: activity-stream picker, side-by-
  side traversal paths (paper's M scatters vs fixed M draws a clean shape), and
  the grouped suite results with the M-lever headline. **Needs Flask restart.**
- **[2026-07-10] (Claude)** — STORYBOARD mechanism + first full episode. New story
  layer: a story = ordered list of saved scenes, rendered into ONE mp4 in a single
  encoder pass (all shots rasterized at the first scene's size/fps; no stitching
  needed). Store runs/video/stories, endpoints /api/anim/stories (+delete) and
  /api/anim/render_story, Storyboard card in the Scenes view (add/reorder/remove
  shots, save, render). CONTENT: "The Glitches — Incident 001: Withdrawal" built as
  proof — rig ensemble (ep1-norm, ep1-corrector, custom ep1-atm with door-tagged cash
  tray, ep1-wall) + 8 saved shots (street title / tolerated no-clip through wall /
  ATM abuse with cash eject + LEDGER box / enforcement-log still / shadowban /
  corrector arrives / deallocation fade / thesis card) + saved story
  the-glitches-ep1 -> rendered the-glitches-ep1.mp4 (56.5s, 1280x720@24) in the
  library. Audio deliberately skipped (user records VO separately). **Flask restart
  required** (anim_service.py + app.py changed).
- **[2026-07-10] (Claude)** — Rig editor UX round: (1) UNDO/REDO — Ctrl+Z / Ctrl+Y
  (Ctrl+Shift+Z too) in all three /video views: rig edits, scene edits, editor
  timeline; snapshot stacks per document, cleared on open, native text-undo left
  alone while typing in fields. (2) Parts panel + selected-part form now FLOAT over
  the rig stage (collapsible overlay panels) instead of a card below. (3) On-stage
  transform gizmos: selected part gets a rotate handle (round, gold) and a resize
  handle (square, blue) around its pivot; new part properties `rot` (base rotation,
  applies to the subtree like verb rotations) and `scale` (uniform, shape only so
  children don't inherit) — added to BOTH anim_service.py and animrig.js renderers,
  plus rotate/size fields in the part form. Old rigs without rot/scale unchanged.
- **[2026-07-10] (Claude)** — /video is now an ANIMATION PLATFORM (SCP-Explained flat
  style): three views — **Rigs** (manual part-by-part SVG puppet editor: layered parts
  with parent/pivot/z, semantic tags arms/legs/mouth shapes, drag-to-position, live
  verb test idle/walk/talk/point; plus a procedural generator with 10 seeded archetypes:
  researcher/guard/dclass/suit/civilian + crate/table/door/terminal/containment),
  **Scenes** (actors placed by drag, verb action timeline walk/move/talk/point/fade,
  caption/title/infobox overlays, optional voiceover audio muxed from the library,
  scrub+play preview), **Editor** (the original cut/stitch/convert editor). Scene
  renders: per-frame SVG -> resvg -> ffmpeg pipe -> mp4 lands in the library for the
  timeline. Verb/pose math lives twice ON PURPOSE: anim_service.py (render) and
  static/animrig.js (preview) must stay in sync. New: anim_service.py, animrig.js,
  animstudio.js, /api/anim/* routes; library now accepts audio (mp3/wav/…) for
  voiceover tracks. resvg-py added to requirements. Verified end-to-end (generate ->
  scene -> render -> probe). **Flask restart required.**
- **[2026-07-09] (Claude)** — NEW: /video page — ffmpeg-backed video editor (plain
  utility, nothing evolved). Library (upload any format, probe via ffprobe, cached
  thumbnails, download/delete), clip timeline with per-clip in/out cut points +
  reorder + stitch-export (concat filter: mixed codecs/resolutions/framerates
  normalised, silent inputs get a silence track), per-file cut-to-file and container
  conversion (mp4/mkv/webm/mov/m4v/avi/gif), background jobs with live progress
  (`-progress pipe:1`) and cancel. Backend `video_service.py`, routes in app.py
  (`/api/video/*`), files under `runs/video/library`. ffmpeg full build extracted from
  the user's downloaded 7z into `tools/ffmpeg-2026-07-09-.../bin` (gitignored;
  resolution order: tools > PATH > imageio-ffmpeg bundle). All ops verified end-to-end
  with generated test clips. **Flask restart required** to pick up the new routes.
- **[2026-07-07] (Claude)** — `/images` reverse tab: image/video -> prompt via BLIP captioning +
  CLIP-ranked medium/style/lighting/quality tags — `genreg_train/reverse_service.py`,
  `POST /api/images/reverse` (single image or video, frames extracted with imageio/ffmpeg),
  `GET /api/images/file/<path>` to serve results. Output lands in a structured job folder
  `runs/images/reverse/<job_id>/{frames,prompts}/frame_NNNNN.{png,txt}` + `manifest.json`.
  Caption length and modifier-tags-per-category are adjustable from the sidebar.
- **[2026-07-05] (Claude)** — I2: **YouTube-style social layer across the media pages** (per user:
  comments, likes/downvotes, thumbnails, details, up-next). Backend (`i2_service.py`): likes
  extended to **signed up/down votes** (value 1/-1/0, one vote per DID, up/down mutually
  exclusive; legacy like/unlike wire format + `count` field unchanged); new **signed comments**
  (`i2_store/comments/<target>.json`, sig covers `sha256(text)`, ≤1000 chars, ≤500/target,
  newest-first, works on any content id); **video thumbnails** (optional poster frame at upload,
  jpeg/png/webp ≤200KB, stored genome-coded under key id `<vid>:thumb`, served via
  `/api/i2/video/<id>/thumb`); **view counter** (`/viewed` POST, unsigned/node-local, stated
  openly). Routes added to primary (`i2_node.py`) + child proxy (`i2_child.py`, v1.5.0).
  Browser (`static/i2.js`): shared `voteRow`/`commentSection` components (theme-var driven);
  Latent Stream grid gets real thumbs + views + age; watch page gets channel row, 👍/👎, view
  count, description panel, full comment section, view-ranked Up-next with thumbs; upload
  auto-captures a 480px poster frame (canvas, preview shown); Woven posts get votes + 💬
  (inline thread for images, jump-to-watch for videos); Gallery cells get votes + age.
  Integration-tested against an ephemeral primary (votes up/down/switch/clear, comments incl.
  forged-sig 400, views, thumb coded-at-rest round-trip + 404) — all pass. **Deployed**: pushed
  to primary 10.0.0.15 (restarted, endpoints verified live; backup `_backup/push-20260705-094110`),
  `dist/i2_child` refreshed in place to v1.5.0 (running child needs a restart).
  **Follow-up (per user, hard rule): NO emojis anywhere in the UI** — vote buttons are now
  monochrome ▲/▼, the comments chip is plain text ("N comments"), image tiles/Gallery headers
  use ▦, search placeholder and DMV headers de-emojied; typographic marks (✓ ✗ ⚠ ▶) stay.
  Re-pushed to primary + dist child refreshed.
