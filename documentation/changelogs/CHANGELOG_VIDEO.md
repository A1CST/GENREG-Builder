# Changelog — VIDEO

Per-project log for the VIDEO line. Seeded 2026-07-14 from the master
CHANGELOG.md (all entries mentioning this project); new VIDEO entries go at
the top of the log below, and also in the master CHANGELOG.md.

## Format
- **[YYYY-MM-DD] (author)** — description of change

---

- **[2026-07-19] (Claude)** — **VIDEO: TTS credit guard - a lookup json
  ties each generated line to its clip so identical narration NEVER bills
  twice (user's call).** `runs/video/slide_audio/tts_cache.json` maps
  sha256(voice | whitespace-normalized text) -> {id, dur, voice, text,
  created}. The /api/video/tts route checks it before touching the API
  (file-existence validated) and returns the existing clip with
  cached:true; new generations are recorded after probing. DELETING a
  TTS clip from a slide keeps the mp3 on disk when the cache owns it
  (kept:true) - removing a line from one slide can't destroy the reuse
  for the next deck; mic recordings (.webm) still delete for real. UI
  surfaces hits ("reused cached narration", "N from cache" on template
  builds). VERIFIED LIVE: same line twice -> one API call (1.1s) then a
  cache hit in 10ms returning the SAME clip id (with different
  whitespace, proving normalization); delete returned kept:true and the
  file survived. One tiny credit spent on the proof.

- **[2026-07-19] (Claude)** — **VIDEO: Script Studio armed - ElevenLabs
  key wired (the user's `ElevenLabs` system env var; backend checks
  ELEVENLABS_API_KEY / ElevenLabs / ELEVENLABS then the .keys file),
  default voice set to the user's (nxNsTXLZ8x7PeZNBs9Js, prefilled in
  the voice field), and a per-slide NARRATE button added to the audio
  panel - synthesizes the active slide's caption into an mp3 clip on
  demand (uses the voice field; clip flows through trim/playback/mux
  like any recording). VERIFIED LIVE: real API call with the user's key
  + voice produced a 2.2s mp3 clip (35KB) saved through the slide-clip
  path. NOTE: the running Flask predates the env var - start Flask from
  a NEW shell (fresh env) when doing the pending restart, or the route
  will not see the key.

- **[2026-07-19] (Claude)** — **VIDEO SESSION CLOSE (user switching to
  CIFAR). State of /video at stop:** the slide builder is a complete
  narrated-explainer pipeline - slide manager (visual cards,
  drag-reorder, duplicate, robust delete), per-slide mic recording
  (ordered clips, move, lock-resilient delete), seamless multi-cut trim
  editor (waveform, persistent selection, WebAudio gapless playback),
  preview that faithfully rehearses the export (audio scheduled at true
  timestamps, live video overlay synced to the deck clock with start
  offset + loop, scrub-aware), one-click library mute, three-layer
  duration floor (duration -> kept audio -> media runtime, identical
  client/renderer), first-class video embeds (stage thumbnails, frame-
  accurate export compositing, corner-grip resize), media timeline
  (draggable start, loop), and SCRIPT STUDIO (modal: narration tools +
  JSON deck templates with downloadable schema + per-slide ElevenLabs
  narration as mp3 clips). Every feature verified with real renders
  (charttest/looptest/resizetest.mp4 in the library, all exact).
  PENDING on the user: the accumulated Flask restart (slide-audio, mute,
  meta, image-upload, TTS routes) and the ElevenLabs key in
  .keys/elevenlabs.key. All work committed and merged to origin/main
  (19a51f8). The uncommitted files in the tree (app.py Replicate
  routes, radial_evo2, several static/*.js) belong to ANOTHER session's
  in-progress work and were deliberately left untouched.

- **[2026-07-19] (Claude)** — **VIDEO: SCRIPT STUDIO modal + JSON deck
  templates + ElevenLabs narration (user's design). Pose-alignment UI
  removed (drag owns placement; CC captions kept).** The script side
  panel now opens a MODAL with two tabs: SCRIPT (the existing narration
  tools, moved intact - highlight-to-slide, span, auto-split) and
  TEMPLATE: paste a JSON deck template, download the sample (schema
  documented in-file: per slide - pose + position, chart + position/
  size/start/loop, script text (doubles as the CC caption), duration,
  transition, free-form meta), and BUILD THE DECK in one click - with
  optional ELEVENLABS NARRATION: each slide's script is synthesized
  (voice id field, default Rachel), saved as an mp3 slide clip, and the
  slide auto-floors to its narration. New POST /api/video/tts (key from
  ELEVENLABS_API_KEY env or gitignored .keys/elevenlabs.key; graceful
  503 with instructions when unset - verified); clip whitelist +
  mimetypes extended to .mp3 (decode/trim/mux all work on mp3 already).
  Template upload -> entire narrated video: build deck -> TTS -> record
  over/trim as needed -> export. Routes ride the pending restart; page
  hard-refresh. NOTE for the user: drop your ElevenLabs key in
  .keys/elevenlabs.key (one line) or set ELEVENLABS_API_KEY.

- **[2026-07-19] (Claude)** — **VIDEO: media-timeline drag FIXED +
  resizable charts/videos on the stage (user's report + ask).**
  (1) The start slider was killing its own drag: every pointer move
  rebuilt the timeline DOM, destroying the captured handle mid-gesture.
  Now the drag updates the handle/span/readout inline and only re-renders
  on release - window-level listeners, smooth drag.
  (2) RESIZE: charts and videos get a blue corner grip on the stage
  (bottom-right, nwse cursor) wired into the existing drag system as a
  chart-resize mode - slide.chart_w/chart_h (min 80x60, defaults
  550x420), sanitized, copied by apply-to-all, used by the stage image,
  the LIVE VIDEO OVERLAY (so the playing video matches the resized box),
  and the RENDERER. Verified: rendered a deck with an 880x500 looping
  video at a custom position (resizetest.mp4, 2.0s, clean). Frontend
  hard-refresh; renderer hot-loaded.

- **[2026-07-19] (Claude)** — **VIDEO: media timeline responsive
  pre-restart + the preview stage actually PLAYS videos (user's report).**
  Two causes: (1) the media timeline armed off /api/video/meta, which is
  still behind the pending Flask restart - with no duration it stayed
  hidden. Now the BROWSER probes mp4/webm durations itself
  (video-element metadata, 5s timeout) as a fallback, and existing decks
  whose videos were assigned before durations worked are healed on load.
  (2) Preview showed video charts as static thumbnails by design - wrong
  for a rehearsal tool. Now a muted HTML <video> overlay is positioned
  over the stage at the chart's exact box (viewBox -> CSS mapping,
  letterbox-aware) and driven by the deck clock: hidden before the start
  offset (the SVG poster shows), seeks on scrub, plays/pauses with the
  deck, loops when loop is set, holds the last frame when not. Narration
  owns the audio (overlay always muted). Frontend only, hard refresh.

- **[2026-07-19] (Claude)** — **VIDEO: MEDIA TIMELINE under the audio
  panel (user's call) - choose when a slide's video/animation starts,
  with loop.** When the active slide embeds animated media, a MEDIA
  TIMELINE appears below the audio clips: a track scaled to the slide's
  effective duration, the media's play window drawn on it (repeating
  ghost spans when looping), and a DRAGGABLE START HANDLE (click
  anywhere on the track also sets the start); a loop checkbox; a live
  readout ("starts at 1.5s, plays 10.0s (ends 11.5s)" / "loops every
  10.0s"). SEMANTICS, mirrored exactly in the renderer: the video holds
  its poster frame until start, then plays; non-looping media FLOORS the
  slide at start + runtime (the floor order stays duration -> audio ->
  media); looping media imposes no floor - it fills whatever the slide
  gives it, repeating (frame index modulo). Frame extraction adapts:
  loop extracts the full video once (120s cap), non-loop extracts what
  fits after the start offset. slide.chart_start/chart_loop sanitized +
  copied by apply-to-all. VERIFIED: floor math both ways (start 1.5 +
  10s clip -> 11.5s; looping clip on a 4s slide -> 4.0s) and an
  end-to-end loop render (looptest.mp4, exactly 4.0s). Frontend
  hard-refresh; renderer hot-loaded.

- **[2026-07-18] (Claude)** — **VIDEO: charts/videos now actually show up
  - on the stage AND in exports (user's report). ROOT CAUSE: the library
  contained ZERO images - the "Upload Graphic / Chart" input posts to
  /api/video/upload, which REJECTED image formats, so image charts never
  existed; and video files placed as embeds hit <image href=...mp4>
  (broken icon in the browser, silently absent in resvg). Fixed the whole
  chain:** (1) uploads accept images (png/jpg/jpeg/gif/webp) into the
  library; (2) VIDEOS are first-class slide embeds - the chart dropdown
  gains [video] entries, the Videos cards gain "Use on slide", the STAGE
  shows the video's server thumbnail (no broken icon), and the EXPORT
  extracts the video's real frames once per render (ffmpeg fps-matched,
  550px-scaled) and composites the correct frame per output frame
  (slide_to_svg_group gains local_t + a chart_frames provider, threaded
  through the crossfade branches); (3) BONUS BUG: the frame compositor's
  slide ranges ignored the new duration floors (audio/media) - fixed to
  _eff_slide_dur, so visuals stay on the right slide while audio holds
  it. VERIFIED end-to-end: rendered a deck with the muted Gemini clip as
  an embed - charttest.mp4, 11.5s exactly (10s video floor + 1.5s second
  slide), frames composited. Upload route rides the pending restart;
  page is hard-refresh.

- **[2026-07-18] (Claude)** — **VIDEO: embedded-media duration joins the
  slide floor (user's call).** A slide now stays up for
  max(set duration, kept narration, EMBEDDED MEDIA runtime) - if the
  chart is animated media (gif / video), its ffprobe duration becomes a
  third floor, evaluated after audio exactly as asked. New
  /api/video/meta?name= (mtime-cached probe; stills report 0 and change
  nothing); client stores slide.chart_dur (refreshed on chart assignment
  from the select or the library, copied by apply-to-all); renderer
  mirrors via _chart_dur in _eff_slide_dur, probing server-side so the
  mp4 is authoritative. VERIFIED on the real muted Gemini clip: 10.0s
  probe; 2s slide -> floors to 10.0s; with 9s narration the 10s video
  still wins. Routes ride the pending restart; page is hard-refresh.

- **[2026-07-18] (Claude)** — **VIDEO: trim-editor selection now persists
  after release** - pointerup was clearing the selection on every
  interaction, so Delete cut disarmed the moment the mouse let go.
  Now: releasing keeps the touched cut selected (Delete stays armed);
  a bare click on empty waveform deselects (and the zero-width cut a
  click would create is discarded). Frontend only, hard refresh.

- **[2026-07-18] (Claude)** — **VIDEO: audio TRIM editor + audio-driven
  minimum slide duration (user's two fixes).**
  (1) TRIM: each clip row gains a Trim button opening a waveform editor
  (decoded peaks on canvas) - drag on the wave to create a cut, drag a
  red edge to resize it, click to select, Delete cut removes it; end
  trims are just cuts anchored at the edges; Add cut drops one mid-way
  through the largest kept span. Cuts are stored per clip
  (clip.cuts=[[s,e],...], normalized/merged), and playback across cuts
  is SEAMLESS everywhere: the preview player and the clip/trim Play
  buttons now run on WebAudio (kept segments scheduled back-to-back,
  sample-accurate - no gap where a trim exists), and the RENDERER builds
  the matching ffmpeg graph (atrim per kept segment -> concat -> adelay),
  so the mp4 matches the preview exactly. Clip rows show the EFFECTIVE
  (kept) duration with a "(trimmed)" tag.
  (2) MINIMUM SLIDE DURATION: a slide now stays on screen at least as
  long as its total kept audio - effDur(slide) = max(set duration, sum of
  effective clip durations) - adopted in every timeline computation
  (total, ranges, scrub, schedule, slide cards) AND mirrored in the
  renderer (_eff_slide_dur), with a "slide held at Ns by its audio" hint
  under the duration input. Verified: cut-complement math matches
  client/server (5s clip, cuts [0-.5]+[2-2.5] -> 4.0s kept; 2s slide held
  at 4.0s). node --check + ast clean. Frontend hard-refresh; renderer is
  hot-loaded.

- **[2026-07-18] (Claude)** — **VIDEO: preview player now PLAYS the
  recorded slide audio.** The Play button only advanced visuals; recorded
  clips were render-time-only. Now play() builds the same clip schedule
  the renderer uses (slide start + prior-clip offsets), and the tick loop
  starts/stops Audio elements as the playhead enters/leaves each clip's
  window - scrub-aware (jumping mid-clip starts playback at the correct
  offset), loop-aware (audio resets when the deck loops), and pause stops
  everything. Frontend only, hard refresh.

- **[2026-07-18] (Claude)** — **VIDEO: audio panel repositioned - it had
  landed INSIDE the stage's flex-centering wrapper (floating beside the
  SVG); now it sits properly under the stage preview + scrubber.**
  Template-only, hard refresh.

- **[2026-07-18] (Claude)** — **VIDEO: library Videos section + one-click
  MUTE (user's call: Gemini-generated animations arrive with baked-in
  audio).** Answered + built: exports save to `runs/video/library/`
  (every rendered mp4 + uploads; /api/video/library reads it). New in
  the media panel: a **Videos** list (mp4/webm/mov/mkv/gif from the
  library) with inline muted preview players and a **Mute** button per
  video - POST /api/video/mute writes `<name>_muted.<ext>` via ffmpeg
  `-c:v copy -an` (video stream untouched, lossless + fast); muted
  copies are labeled and not re-mutable. New GET /api/video/videos.
  VERIFIED live: muted the actual Gemini clip
  (3D_scene_camera_orbit_clockwise_202607182042.mp4 ->
  ..._muted.mp4, HTTP 200) through the Flask test client. One
  self-inflicted bug caught: a patch escape mangled a string literal in
  app.py (unterminated) - repaired and ast-verified before anything ran.
  Routes need the SAME pending Flask restart; page part is
  hard-refresh-only.

- **[2026-07-18] (Claude)** — **VIDEO: per-slide AUDIO panel (user's call)
  - record narration per slide through the browser mic, ordered clips,
  move/delete, and the renderer muxes every clip at its true timestamp.
  FLASK RESTART NEEDED (new routes).** Under the stage preview: SLIDE
  AUDIO panel - Record/Stop (MediaRecorder, webm/opus; the button goes
  red while recording and the take is PINNED to the slide that was
  active when recording started, so clicking around mid-take cannot
  misfile it), clip rows per active slide (index, duration, Play/Stop,
  move-to-slide dropdown, Delete), multiple clips per slide in order,
  and a green audio dot on slide-manager cards. Storage: clips POST to
  /api/video/slide_audio (runs/video/slide_audio/, whitelisted webm ids;
  GET serves, DELETE is Windows-lock-resilient with a deferred retry -
  deleting a clip while the browser streams it cannot 500). Durations
  measured client-side via decodeAudioData (webm blobs report Infinity
  through the audio element in Chromium). Slide JSON gains
  slide.clips=[{id,dur}] (NEW field - the existing slide.audio string is
  the deck-level export track and the export handler clobbers it, so
  clips deliberately avoid it; both mux together). RENDER:
  anim_service.render_slides builds ffmpeg adelay/amix graphs - each
  clip starts at its slide's start time plus the durations of prior
  clips on the same slide; deck track still supported; output trimmed to
  deck duration. Verified: endpoint roundtrip + traversal guard via the
  Flask test client; mux wiring inspected; node --check + ast clean.
  One build bug caught by the roundtrip test: the SLIDE_AUDIO_DIR
  definition was skipped by its own existence-guard (the mux patch had
  already introduced the reference).

- **[2026-07-18] (Claude)** — **VIDEO: slide manager rebuilt (user's call:
  decluttered + the broken delete) - visual cards, drag-to-reorder,
  duplicate, and a robust delete.** ROOT CAUSE of the dead delete: decks
  saved by earlier code versions store `duration` as a STRING, and the
  list renderer's `duration.toFixed(1)` throws on the first legacy slide,
  killing the whole list render - including every Del button. Works on
  fresh decks, broken on real ones. Fixes: (1) `sanitizeSlide` coerces
  every field (numbers, defaults) on load, so one malformed slide can
  never break the manager again; (2) the manager is now visual CARDS -
  pose thumbnail, slide number + duration badges, chart indicator dot,
  caption snippet, hover-reveal actions (duplicate, delete with red
  hover); (3) actions are DELEGATED on the container with indices read at
  click time (no stale closures) and each card renders inside its own
  try/catch; (4) drag-to-reorder with a drop indicator, active-slide
  tracking preserved across moves; (5) delete fixes activeIndex shifting
  when removing above the selection. Frontend only, hot-loaded
  (cache-busted) - hard refresh, no Flask restart. node --check clean.

- **[2026-07-18] (Claude)** — **Script-to-slides workflow + pose position persistence.** Reworked the slideshow builder ([static/slideshow.js](file:///C:/Users/paytonm/Documents/GENREG/static/slideshow.js), [templates/video.html](file:///C:/Users/paytonm/Documents/GENREG/templates/video.html)). (1) **Assets stay put between frames** — "Add Slide" now inherits the previous slide's pose AND chart with their exact `pose_x/pose_y`/`chart_x/chart_y` position + alignment; added **"Apply this pose/chart + position to all slides"** buttons to lock one placement across the whole deck. (2) New **Script** tab: paste the whole narration into one block; highlight a sentence and click **Highlight → new slide(s)** to append it as the next caption in order, or **Highlight → selected slide caption** to set the current slide. (3) **Span** input makes one highlighted line fill N consecutive slides. (4) **Auto-split** the whole script into slides by sentence / line / paragraph (append or replace). Script persists in localStorage. Pure frontend; renderer already honors per-slide positions. No Flask restart required.

- **[2026-07-17] (Antigravity)** — **Generated Transparent Happy Anime Pose.** Used diffusion model reference mapping to generate a new happy anime character pose. Post-processed the image using a Pillow Python script to extract background pixels to transparency, saving the result directly to the user's poses library folder at `C:\Users\paytonm\Pictures\poses\happy_pose_anime.png`.

- **[2026-07-17] (Antigravity)** — **Implemented Drag-and-Drop Positioning for Poses & Charts.** Enabled real-time PointerEvent dragging on the preview stage inside [static/slideshow.js](file:///C:/Users/paytonm/Documents/GENREG/static/slideshow.js) to dynamically reposition poses and charts. Integrated custom `pose_x`, `pose_y`, `chart_x`, and `chart_y` attributes in the client SVG generator, the onion-skin overlay positioning, and the python FFmpeg video renderer in [anim_service.py](file:///C:/Users/paytonm/Documents/GENREG/anim_service.py). No Flask restart is required since code is hot-loaded.

- **[2026-07-17] (Antigravity)** — **Added Onion Skinning & Asset Carry-over.** Configured new slides to start blank but show translucent ghost overlays (onion layers) of the previous slide's pose and chart assets in [static/slideshow.js](file:///C:/Users/paytonm/Documents/GENREG/static/slideshow.js). Overlaid interactive green plus buttons (+) centered over the ghosts; clicking a button carries the asset over to the new slide. No Flask restart is required since code is hot-loaded.

- **[2026-07-17] (Antigravity)** — **Added Hover Zoom Previews to Slide Editor.** Integrated viewport-aware floating preview overlays in [static/slideshow.js](file:///C:/Users/paytonm/Documents/GENREG/static/slideshow.js) that pop up on hover over pose or chart cards in the Media Library or over the selected slide's thumbnails in the edit form. Overwrote [templates/video.html](file:///C:/Users/paytonm/Documents/GENREG/templates/video.html) to render mini-preview thumbnails and absolute preview container markup. No Flask restart is required since code is hot-loaded.

- **[2026-07-17] (Antigravity)** — **Added Gemini Terminal Tab Launcher.** Appended a "Gemini" button to the termdock action bar in [static/termdock.js](file:///C:/Users/paytonm/Documents/GENREG/static/termdock.js). Wired [static/app.js](file:///C:/Users/paytonm/Documents/GENREG/static/app.js) to trigger a new terminal tab titled "Gemini" and type the `agy` CLI command into it once the PowerShell shell prompt appears. No Flask restart is required since code is hot-loaded.

- **[2026-07-17] (Antigravity)** — **Added Slide Transition Speed & Restored Terminal Docks.** Restored the standard `xterm` script imports and stylesheets at the bottom of [templates/video.html](file:///C:/Users/paytonm/Documents/GENREG/templates/video.html). Added a Transition Duration input field next to the slide transition type dropdown. Updated [static/slideshow.js](file:///C:/Users/paytonm/Documents/GENREG/static/slideshow.js) to dynamically hide/show the transition speed container, read/write custom transition speeds, and incorporate the custom `transition_dur` in the client-side crossfade render preview. No Flask restart is required since code is hot-loaded.

- **[2026-07-17] (Antigravity)** — **Re-configured Video Studio to a Slide Explainer Builder.** Replaced vector puppet rigs on `/video` with an image-based Slide presentation editor. Created [static/slideshow.js](file:///C:/Users/paytonm/Documents/GENREG/static/slideshow.js) to manage localstorage slide decks, preview SVG frames (combining background colors, streaming poses, uploaded embeds/charts, and captioned CC text), and drive timeline play/scrub controls. Appended `/api/poses` (serving assets from `C:\Users\paytonm\Pictures\poses`), `/api/charts` (listing embeds), and `/api/video/render_slides` routes to [app.py](file:///C:/Users/paytonm/Documents/GENREG/app.py). Appended base64 image encoders and frame-by-frame SVG compilers supporting fade transitions to [anim_service.py](file:///C:/Users/paytonm/Documents/GENREG/anim_service.py). Overwrote [templates/video.html](file:///C:/Users/paytonm/Documents/GENREG/templates/video.html) with a slide editor workspace. No Flask restart is required since code is hot-loaded.

- **[2026-07-17] (Antigravity)** — **Implemented Front-Facing Torso & Limb Geometry for Humanoids.** Upgraded [anim_service.py](file:///C:/Users/paytonm/Documents/GENREG/anim_service.py) and [static/animrig.js](file:///C:/Users/paytonm/Documents/GENREG/static/animrig.js) to support complete front-facing body rendering in `rig_svg` / `rigSVG` when `facing == "front"`. Broadens the torso capsule (1.3x width), dynamically centers and scales torso clothing layers (vest, shirt, labcoat, badge), spaces limbs symmetrically, and updates layering order to render the left arm in front of the torso. No Flask restart is required since code is hot-loaded.

- **[2026-07-17] (Antigravity)** — **Added new character variations, environmental objects, and pose verbs to Video Studio.** Modified [anim_service.py](file:///C:/Users/paytonm/Documents/GENREG/anim_service.py) to support 4 new character archetypes (`scientist` with lab coat, `professor` with vest and custom gray hair, `robot` with chassis plate and custom skin, `cyborg` with metallic mask and glowing eye) and 6 new objects (`skyline`, `street`, `house`, `skyscraper`, `lab_building`, `server_rack`). Implemented 4 new reusable animation verbs (`present`, `explain`, `think`, `code`) and separated `walk` into 4 directional walk/climb verbs (`walk_right`, `walk_left`, `walk_up_stairs_right`, `walk_up_stairs_left`) on the backend (`actor_state`) and frontend ([static/animrig.js](file:///C:/Users/paytonm/Documents/GENREG/static/animrig.js)). Added a new `face` action verb and default `facing` actor properties to dynamically render front-facing head geometry (symmetrical eyes, nose path, centered mouth, and accessory layers). Registered verb schemas and added Stage preview test buttons to [static/animstudio.js](file:///C:/Users/paytonm/Documents/GENREG/static/animstudio.js) and [templates/video.html](file:///C:/Users/paytonm/Documents/GENREG/templates/video.html). No Flask restart is required since code is hot-loaded.

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
