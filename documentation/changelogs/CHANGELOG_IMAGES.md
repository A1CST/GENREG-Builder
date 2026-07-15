# Changelog — IMAGES

Per-project log for the IMAGES line. Seeded 2026-07-14 from the master
CHANGELOG.md (all entries mentioning this project); new IMAGES entries go at
the top of the log below, and also in the master CHANGELOG.md.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-07] (Claude)** — `/images` reverse-tab polish: caption length ceiling raised 75->200
  tokens; BLIP was hitting its own early-stop well under budget regardless, so `_caption()` now
  passes `min_new_tokens` + sampling (top_p, temperature, repetition_penalty, no_repeat_ngram_size)
  to actually force the extra length instead of silently truncating. Noted in the UI that >~75-100
  tokens degrades into stock-photo-metadata noise (BLIP's real ceiling, not a bug). Gallery UX:
  "Process" now appends each run's frames under a job header instead of clearing the canvas, and
  each frame renders as an index/thumbnail/prompt row stacked in a single column (was a wrapping
  card grid).
- **[2026-07-07] (Claude)** — `/images` reverse tab: image/video -> prompt via BLIP captioning +
  CLIP-ranked medium/style/lighting/quality tags — `genreg_train/reverse_service.py`,
  `POST /api/images/reverse` (single image or video, frames extracted with imageio/ffmpeg),
  `GET /api/images/file/<path>` to serve results. Output lands in a structured job folder
  `runs/images/reverse/<job_id>/{frames,prompts}/frame_NNNNN.{png,txt}` + `manifest.json`.
  Caption length and modifier-tags-per-category are adjustable from the sidebar.
- **[2026-07-07] (Claude)** — `/images` text-to-image: wired a pretrained Stable Diffusion 1.5
  pipeline (diffusers) into the blank Images page — `genreg_train/sd_service.py` (lazy-loaded
  singleton pipeline, GPU if available), `POST /api/images/generate`, prompt/negative-prompt/
  steps/guidance/size/seed controls in the sidebar, generated PNGs saved under `runs/images/`.
  Not evolved — a plain pretrained-checkpoint generation utility, unlike the rest of GENREG.
- **[2026-07-07] (Claude)** — New project page `/images` — blank scaffold (terminals, run-config
  panel, agent-alerts panel) added to the nav, no canvas content yet.
