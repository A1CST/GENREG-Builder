(function() {
  const $ = (id) => document.getElementById(id);
  
  // Tab views switching
  const tabs = document.querySelectorAll("#vd-views button");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".control-panel").forEach((panel) => {
        panel.hidden = panel.dataset.vpanel !== btn.dataset.view;
      });
    });
  });

  // State Variables
  let slides = [];
  let activeIndex = -1;
  let isPlaying = false;
  let playStartRealTime = 0;
  let playStartScrubTime = 0;
  let playerAnimFrame = null;
  let currentTime = 0;
  let totalDuration = 0;
  let posesList = [];
  let chartsList = [];
  let tempGhosts = null;
  let ghostIndex = -1;

  // Initialize from LocalStorage
  function normCuts(cuts, dur) {
    // sorted, merged, clamped removed-intervals within [0, dur]
    if (!Array.isArray(cuts)) return [];
    const cs = cuts
      .map((c) => [Math.max(0, Number(c[0]) || 0),
                   Math.min(dur, Number(c[1]) || 0)])
      .filter((c) => c[1] - c[0] > 0.01)
      .sort((a, b) => a[0] - b[0]);
    const out = [];
    cs.forEach((c) => {
      const last = out[out.length - 1];
      if (last && c[0] <= last[1] + 0.005) last[1] = Math.max(last[1], c[1]);
      else out.push([c[0], c[1]]);
    });
    return out;
  }

  function keptSegments(clip) {
    // complement of the cuts: the audio that actually plays, in order
    const dur = Number(clip.dur) || 0;
    const cuts = normCuts(clip.cuts, dur);
    const segs = [];
    let t = 0;
    cuts.forEach((c) => {
      if (c[0] - t > 0.01) segs.push([t, c[0]]);
      t = c[1];
    });
    if (dur - t > 0.01) segs.push([t, dur]);
    return segs;
  }

  function effClipDur(clip) {
    return keptSegments(clip).reduce((a, seg) => a + (seg[1] - seg[0]), 0);
  }

  function slideAudioTotal(slide) {
    return (slide.clips || []).reduce((a, c) => a + effClipDur(c), 0);
  }

  function mediaItems(slide) {
    return Array.isArray(slide.media) ? slide.media : [];
  }

  function mediaVisible(m, t) {
    // an item is on stage from its start until its explicit vanish time
    // (end); end 0 = auto (until the slide ends; non-loop video holds
    // its last frame)
    if (t < (Number(m.start) || 0)) return false;
    const end = Number(m.end) || 0;
    return !(end > 0 && t >= end);
  }

  function mediaFadeOpacity(m, t) {
    // 0.5s ramps at the visibility-window edges when fades are on
    const F = 0.5;
    let op = 1;
    if (m.fade_in) op = Math.min(op, (t - (Number(m.start) || 0)) / F);
    if (m.fade_out && Number(m.end) > 0) op = Math.min(op, (Number(m.end) - t) / F);
    return Math.max(0, Math.min(1, op));
  }

  function mediaFloor(slide) {
    // each non-looping timed item floors the slide at start + runtime
    // (or its explicit vanish time if that is earlier); loops and still
    // images add no floor
    let f = 0;
    mediaItems(slide).forEach((m) => {
      const d = Number(m.dur) || 0;
      if (!d || m.loop) return;
      const natural = (Number(m.start) || 0) + d;
      const end = Number(m.end) || 0;
      f = Math.max(f, end > 0 ? Math.min(end, natural) : natural);
    });
    return f;
  }

  function effDur(slide) {
    // the slide floor: set duration, then narration, then embedded media
    return Math.max(Number(slide.duration) || 3.0, slideAudioTotal(slide),
                    mediaFloor(slide));
  }

  function probeDurClient(name) {
    // works before the /api/video/meta route is live (Flask restart):
    // the browser reads mp4/webm metadata directly
    return new Promise((resolve) => {
      if (!/\.(mp4|webm|mov)$/i.test(name)) { resolve(0); return; }
      const v = document.createElement("video");
      v.preload = "metadata";
      v.onloadedmetadata = () => resolve(isFinite(v.duration) ? v.duration : 0);
      v.onerror = () => resolve(0);
      v.src = `/api/video/file/${encodeURIComponent(name)}`;
      setTimeout(() => resolve(0), 5000);
    });
  }

  async function refreshMediaDur(slide, item) {
    if (!item || !item.name) return;
    if (!/\.(mp4|webm|mov|gif)$/i.test(item.name)) { item.dur = 0; return; }
    let d = 0;
    try {
      const r = await (await fetch(
        `/api/video/meta?name=${encodeURIComponent(item.name)}`)).json();
      d = Number(r.duration) || 0;
    } catch (e) { d = 0; }
    if (!d) d = await probeDurClient(item.name);
    item.dur = Math.round(d * 10) / 10;
    saveSlides(); renderSlideList(); updateScrubMax(); renderPreview();
    renderMediaTimeline();
  }

  function sanitizeSlide(s) {
    // legacy decks stored numbers as strings and lack newer fields; one
    // bad slide must never crash the manager
    s = s || {};
    return {
      pose: s.pose || "", pose_align: s.pose_align || "left",
      pose_x: Number(s.pose_x) || 0, pose_y: Number(s.pose_y) || 0,
      media: Array.isArray(s.media)
        ? s.media.filter((m) => m && m.name).map((m) => ({
            name: String(m.name),
            x: Number.isFinite(Number(m.x)) ? Number(m.x) : 650,
            y: Number.isFinite(Number(m.y)) ? Number(m.y) : 80,
            w: Math.max(80, Number(m.w) || 550),
            h: Math.max(60, Number(m.h) || 420),
            start: Math.max(0, Number(m.start) || 0),
            dur: Math.max(0, Number(m.dur) || 0),
            loop: !!m.loop,
            end: Math.max(0, Number(m.end) || 0),
            fade_in: !!m.fade_in,
            fade_out: !!m.fade_out,
          }))
        : (s.chart ? [{
            // migrate the legacy single-chart fields into media[0]
            name: String(s.chart),
            x: Number(s.chart_x) || (s.chart_align === "left" ? 80
               : s.chart_align === "center" ? 365 : 650),
            y: Number(s.chart_y) || 80,
            w: Math.max(80, Number(s.chart_w) || 550),
            h: Math.max(60, Number(s.chart_h) || 420),
            start: Math.max(0, Number(s.chart_start) || 0),
            dur: Math.max(0, Number(s.chart_dur) || 0),
            loop: !!s.chart_loop,
            end: 0,
          }] : []),
      pose_gesture: typeof s.pose_gesture === "string" ? s.pose_gesture : "",
      media_request: typeof s.media_request === "string" ? s.media_request : "",
      text: typeof s.text === "string" ? s.text : "",
      bg: s.bg || s.background || "",
      duration: Math.max(0.5, Number(s.duration) || 3.0),
      transition: s.transition || "fade",
      transition_dur: Math.max(0, Number(s.transition_dur) || 0.5),
      clips: Array.isArray(s.clips)
        ? s.clips.filter((c) => c && c.id)
            .map((c) => ({ id: String(c.id), dur: Number(c.dur) || 0,
                           cuts: normCuts(c.cuts, Number(c.dur) || 0) }))
        : [],
    };
  }
  try {
    const saved = localStorage.getItem("genreg_slides");
    if (saved) {
      slides = JSON.parse(saved).map(sanitizeSlide);
    }
  } catch (e) {
    console.error("Failed to load slides state:", e);
  }
  if (!slides || !slides.length) {
    slides = [
      {
        pose: "",
        pose_align: "left",
        media: [],
        text: "Welcome to the evolutionary gradient-free explainer!",
        duration: 3.0,
        transition: "fade",
        transition_dur: 0.5
      }
    ];
  }

  // heal decks whose media was assigned before duration probing worked
  slides.forEach((sl) => {
    mediaItems(sl).forEach((m) => {
      if (m.name && !m.dur && /\.(mp4|webm|mov|gif)$/i.test(m.name)) {
        refreshMediaDur(sl, m);
      }
    });
  });

  // Load Library Data
  async function loadLibrary() {
    try {
      const poses = await (await fetch("/api/poses")).json();
      posesList = Array.isArray(poses) ? poses : [];
      populateDropdown($("slide-pose"), posesList, "none");
      renderPosesLibrary();
      loadVideosLibrary();
    } catch (e) {
      console.error("Error loading poses:", e);
    }

    try {
      const charts = await (await fetch("/api/charts")).json();
      chartsList = Array.isArray(charts) ? charts : [];
      populateDropdown($("slide-chart"), chartsList, "none");
      renderChartsLibrary();
      // videos are selectable slide embeds too (thumbnail on stage,
      // real frames in the export)
      try {
        const vres = await (await fetch("/api/video/videos")).json();
        const vsel = $("slide-chart");
        (Array.isArray(vres) ? vres : []).forEach((v) => {
          const o = document.createElement("option");
          o.value = v;
          o.textContent = "[video] " + v;
          vsel.appendChild(o);
        });
      } catch (e) {}
    } catch (e) {
      console.error("Error loading charts:", e);
    }
    
    // Also load audio files for the mux dropdown
    try {
      const audioRes = await (await fetch("/api/video/library")).json();
      const audioList = (Array.isArray(audioRes) ? audioRes : [])
        .filter(f => f.name.endsWith(".mp3") || f.name.endsWith(".wav"));
      const audioSel = $("exp-audio");
      audioSel.innerHTML = '<option value="">none</option>';
      audioList.forEach(a => {
        const o = document.createElement("option");
        o.value = a.name;
        o.textContent = a.name;
        audioSel.appendChild(o);
      });
    } catch (e) {
      console.error("Error loading audio list:", e);
    }
  }

  function populateDropdown(select, items, fallbackLabel) {
    const prevVal = select.value;
    select.innerHTML = `<option value="">${fallbackLabel}</option>`;
    items.forEach((item) => {
      const o = document.createElement("option");
      o.value = item;
      o.textContent = item;
      select.appendChild(o);
    });
    if (items.includes(prevVal)) {
      select.value = prevVal;
    }
  }

  // Hover Preview Helper
  function bindHoverPreview(element, getUrlFn) {
    element.addEventListener("mouseenter", () => {
      const url = typeof getUrlFn === "function" ? getUrlFn() : getUrlFn;
      if (!url) return;
      
      const rect = element.getBoundingClientRect();
      const preview = $("pose-hover-preview");
      preview.querySelector("img").src = url;
      preview.style.display = "block";
      
      const popH = 480;
      const popW = 320;
      let top = rect.top - popH / 2 + rect.height / 2;
      top = Math.max(10, Math.min(window.innerHeight - popH - 10, top));
      
      let left = rect.right + 10;
      if (left + popW > window.innerWidth) {
        left = rect.left - popW - 10;
      }
      
      preview.style.top = top + "px";
      preview.style.left = left + "px";
    });
    element.addEventListener("mouseleave", () => {
      $("pose-hover-preview").style.display = "none";
    });
  }

  // Render library panel grid thumbnails
  let videosList = [];

  async function loadVideosLibrary() {
    try {
      const res = await (await fetch("/api/video/videos")).json();
      videosList = Array.isArray(res) ? res : [];
    } catch (e) { videosList = []; }
    renderVideosLibrary();
  }

  function renderVideosLibrary() {
    const box = $("videos-library");
    if (!box) return;
    box.innerHTML = "";
    if (!videosList.length) {
      box.innerHTML = '<div style="font-size:11px;color:#5a6672;">no videos in the library yet</div>';
      return;
    }
    videosList.forEach((name) => {
      const card = document.createElement("div");
      card.className = "media-card";
      card.title = name;
      const muted = /_muted\.[a-z0-9]+$/i.test(name);
      card.innerHTML =
        `<video src="/api/video/file/${encodeURIComponent(name)}" muted preload="metadata"` +
        ` style="max-width:100%; max-height:80px; object-fit:contain; margin-bottom:6px;"></video>` +
        `<span>${name}</span>` +
        `<div style="display:flex; gap:4px; margin-top:4px;">` +
        `<button class="runs-btn vd-mini" data-a="prev">Preview</button>` +
        `<button class="runs-btn vd-mini" data-a="use">Use on slide</button>` +
        (muted ? `<span style="font-size:10px;color:#7ee787;align-self:center;">muted</span>`
               : `<button class="runs-btn vd-mini" data-a="mute">Mute</button>`) +
        `</div>`;
      const vid = card.querySelector("video");
      card.querySelector('[data-a="prev"]').addEventListener("click", () => {
        if (vid.paused) { vid.currentTime = 0; vid.play(); }
        else { vid.pause(); }
      });
      card.querySelector('[data-a="use"]').addEventListener("click", () => {
        if (activeIndex < 0) { alert("Select a slide first."); return; }
        addMediaToSlide(slides[activeIndex], name);
      });
      const muteBtn = card.querySelector('[data-a="mute"]');
      if (muteBtn) {
        muteBtn.addEventListener("click", async () => {
          muteBtn.disabled = true;
          muteBtn.textContent = "Muting...";
          try {
            const r = await (await fetch("/api/video/mute", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name }),
            })).json();
            if (r.error) throw new Error(r.error);
            muteBtn.textContent = "Done";
            await loadVideosLibrary();
          } catch (e) {
            muteBtn.textContent = "Mute";
            muteBtn.disabled = false;
            alert("Mute failed: " + e.message);
          }
        });
      }
      box.appendChild(card);
    });
  }

  // Pose gesture vocabulary: a labeled pose (e.g. "Pose-gesture-right")
  // tells the layout where media belongs - the character gestures AT the
  // media, so gesture-right = media on the RIGHT, pose on the left.
  function gestureOf(s) {
    const g = (s.pose_gesture || s.pose || "").toLowerCase();
    if (g.includes("gesture-right") || g.includes("point-right")) return "right";
    if (g.includes("gesture-left") || g.includes("point-left")) return "left";
    if (g.includes("gesture-up") || g.includes("point-up")) return "up";
    return "";
  }

  function gestureMediaPos(g) {
    if (g === "left") return { x: 80, y: 80 };
    if (g === "up") return { x: 365, y: 30 };
    return { x: 650, y: 80 };            // right / default
  }

  function gesturePosePos(g) {
    if (g === "left") return { x: 750, y: 80 };   // media left -> pose right
    return { x: 80, y: 80 };
  }

  function resolvePoseLabel(label) {
    // template gave a gesture label instead of a filename: match it
    // against the (user-labeled) poses library
    if (!label) return "";
    if (posesList.includes(label)) return label;
    const norm = label.toLowerCase().replace(/^pose-/, "");
    const hit = posesList.find((p) => p.toLowerCase().includes(norm));
    return hit || "";
  }

  function addMediaToSlide(slide, name) {
    // slides can carry MULTIPLE charts/videos; each add appends an item
    // with its own position, size, start, loop, and vanish time
    slide.media = slide.media || [];
    const n = slide.media.length;
    const item = { name: name, x: Math.max(80, 650 - n * 24), y: 80 + n * 24,
                   w: 550, h: 420, start: 0, dur: 0, loop: false, end: 0 };
    slide.media.push(item);
    refreshMediaDur(slide, item);
    saveSlides(); renderPreview(); renderSlideList(); renderMediaTimeline();
  }

  function chartHref(name) {
    return /\.(mp4|webm|mov|mkv)$/i.test(name || "")
      ? `/api/video/thumb/${encodeURIComponent(name)}`
      : `/api/video/file/${encodeURIComponent(name)}`;
  }

  function renderPosesLibrary() {
    const box = $("poses-library");
    box.innerHTML = "";
    posesList.forEach((poseName) => {
      const card = document.createElement("div");
      card.className = "media-card";
      card.style.cursor = "pointer";
      card.title = poseName;
      card.innerHTML = `<img src="/api/poses/${encodeURIComponent(poseName)}" /><span>${poseName}</span>`;
      card.addEventListener("click", () => {
        if (activeIndex >= 0) {
          slides[activeIndex].pose = poseName;
          $("slide-pose").value = poseName;
          saveSlides();
          renderPreview();
          // Update mini preview
          const poseMini = $("slide-pose-preview-mini");
          const poseCont = $("slide-pose-preview-container");
          poseMini.src = `/api/poses/${encodeURIComponent(poseName)}`;
          poseCont.style.display = "block";
        }
      });
      bindHoverPreview(card, () => `/api/poses/${encodeURIComponent(poseName)}`);
      box.appendChild(card);
    });
  }

  function renderChartsLibrary() {
    const box = $("embeds-library");
    box.innerHTML = "";
    chartsList.forEach((chartName) => {
      const card = document.createElement("div");
      card.className = "media-card";
      card.style.cursor = "pointer";
      card.title = chartName;
      card.innerHTML = `<img src="/api/video/file/${encodeURIComponent(chartName)}" /><span>${chartName}</span>`;
      card.addEventListener("click", () => {
        if (activeIndex >= 0) {
          addMediaToSlide(slides[activeIndex], chartName);
          const chartMini = $("slide-chart-preview-mini");
          const chartCont = $("slide-chart-preview-container");
          chartMini.src = `/api/video/file/${encodeURIComponent(chartName)}`;
          chartCont.style.display = "block";
        }
      });
      bindHoverPreview(card, () => `/api/video/file/${encodeURIComponent(chartName)}`);
      box.appendChild(card);
    });
  }

  // Upload embed chart file
  $("chart-upload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await (await fetch("/api/video/upload", {
        method: "POST",
        body: form
      })).json();
      if (res.error) alert("Upload error: " + res.error);
      else {
        await loadLibrary();
        if (activeIndex >= 0) {
          addMediaToSlide(slides[activeIndex], res.name);
        }
      }
    } catch (e) {
      alert("Upload failed: " + e.message);
    }
  });

  // Save state
  function saveSlides() {
    localStorage.setItem("genreg_slides", JSON.stringify(slides));
    renderSlideList();
    calculateDuration();
  }

  function calculateDuration() {
    totalDuration = slides.reduce((acc, s) => acc + effDur(s), 0);
    const scrub = $("player-scrub");
    scrub.max = totalDuration;
    updateTimeLabel();
  }

  function updateScrubMax() {
    totalDuration = slides.reduce((acc, sl) => acc + effDur(sl), 0);
    $("player-scrub").max = totalDuration;
    updateTimeLabel();
  }

  function updateTimeLabel() {
    $("player-time-label").textContent = `${currentTime.toFixed(1)}s / ${totalDuration.toFixed(1)}s`;
  }

  // Slide manager: visual cards, drag-to-reorder, duplicate, delete.
  // Actions are DELEGATED on the container (indices read at click time -
  // no stale closures) and one malformed slide cannot break the list.
  let dragFrom = -1;

  function slideCard(slide, idx) {
    const card = document.createElement("div");
    card.className = "sld-card" + (idx === activeIndex ? " active" : "");
    card.draggable = true;
    card.dataset.idx = idx;
    const poseImg = slide.pose
      ? `<img class="sld-pose" src="/api/poses/${encodeURIComponent(slide.pose)}" draggable="false" />`
      : "";
    const chartDot = (slide.media && slide.media.length)
      ? `<span class="sld-dot" title="${slide.media.length} media item(s)"></span>` : "";
    const reqDot = (slide.media_request && !(slide.media || []).length)
      ? '<span class="sld-dot" style="background:#f0a35e; right: 22px;" title="media needed - see the media timeline"></span>' : "";
    const audDot = (slide.clips && slide.clips.length)
      ? `<span class="sld-dot" style="background:#7ee787; right: 12px;" title="${slide.clips.length} audio clip(s)"></span>`
      : "";
    const cap = (slide.text || "").slice(0, 60);
    card.innerHTML =
      `<div class="sld-thumb">${poseImg}` +
      `<span class="sld-num">${idx + 1}</span>` +
      `<span class="sld-dur">${effDur(slide).toFixed(1)}s</span>${chartDot}${reqDot}${audDot}</div>` +
      `<div class="sld-cap">${cap ? "" : '<span class="sld-empty">no caption</span>'}</div>` +
      `<div class="sld-acts">` +
      `<button class="sld-act" data-act="up" data-idx="${idx}" title="move up"${idx === 0 ? " disabled" : ""}>&#9650;</button>` +
      `<button class="sld-act" data-act="down" data-idx="${idx}" title="move down"${idx === slides.length - 1 ? " disabled" : ""}>&#9660;</button>` +
      `<button class="sld-act" data-act="dup" data-idx="${idx}" title="duplicate">+</button>` +
      `<button class="sld-act sld-del" data-act="del" data-idx="${idx}" title="delete">&times;</button>` +
      `</div>`;
    card.querySelector(".sld-cap").prepend(document.createTextNode(cap));
    return card;
  }

  function renderSlideList() {
    const container = $("slide-list");
    container.innerHTML = "";
    slides.forEach((slide, idx) => {
      try {
        container.appendChild(slideCard(slide, idx));
      } catch (e) {
        console.error("slide card failed", idx, e);
      }
    });
    if (!slides.length) {
      container.innerHTML = '<div class="sld-empty" style="padding:10px">no slides yet - use Add Slide or the Script tab</div>';
    }
  }

  function deleteSlide(idx) {
    if (idx < 0 || idx >= slides.length) return;
    slides.splice(idx, 1);
    if (activeIndex >= slides.length) activeIndex = slides.length - 1;
    else if (idx < activeIndex) activeIndex -= 1;
    saveSlides();
    if (activeIndex >= 0) selectSlide(activeIndex);
    else {
      $("slide-editor").style.display = "none";
      renderSlideList();
      renderPreview();
    }
  }

  function moveSlide(idx, dir) {
    const to = idx + dir;
    if (idx < 0 || idx >= slides.length || to < 0 || to >= slides.length) return;
    const activeSlide = slides[activeIndex];
    const [moved] = slides.splice(idx, 1);
    slides.splice(to, 0, moved);
    activeIndex = slides.indexOf(activeSlide);
    saveSlides();
    renderSlideList();
    renderPreview();
    updateScrubMax();
  }

  function duplicateSlide(idx) {
    if (idx < 0 || idx >= slides.length) return;
    slides.splice(idx + 1, 0, sanitizeSlide(JSON.parse(JSON.stringify(slides[idx]))));
    if (activeIndex > idx) activeIndex += 1;
    saveSlides();
    selectSlide(idx + 1);
  }

  (function bindSlideList() {
    const container = $("slide-list");
    container.addEventListener("click", (ev) => {
      const btn = ev.target.closest(".sld-act");
      if (btn) {
        ev.stopPropagation();
        const idx = Number(btn.dataset.idx);
        if (btn.dataset.act === "del") deleteSlide(idx);
        else if (btn.dataset.act === "dup") duplicateSlide(idx);
        else if (btn.dataset.act === "up") moveSlide(idx, -1);
        else if (btn.dataset.act === "down") moveSlide(idx, 1);
        return;
      }
      const card = ev.target.closest(".sld-card");
      if (card) selectSlide(Number(card.dataset.idx));
    });
    container.addEventListener("dragstart", (ev) => {
      const card = ev.target.closest(".sld-card");
      if (!card) return;
      dragFrom = Number(card.dataset.idx);
      ev.dataTransfer.effectAllowed = "move";
    });
    container.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      const card = ev.target.closest(".sld-card");
      container.querySelectorAll(".sld-card").forEach((c) =>
        c.classList.remove("drop-before"));
      if (card) card.classList.add("drop-before");
    });
    container.addEventListener("dragleave", () => {
      container.querySelectorAll(".sld-card").forEach((c) =>
        c.classList.remove("drop-before"));
    });
    container.addEventListener("drop", (ev) => {
      ev.preventDefault();
      container.querySelectorAll(".sld-card").forEach((c) =>
        c.classList.remove("drop-before"));
      const card = ev.target.closest(".sld-card");
      if (!card || dragFrom < 0) return;
      let to = Number(card.dataset.idx);
      if (to === dragFrom) { dragFrom = -1; return; }
      const activeSlide = slides[activeIndex];
      const [moved] = slides.splice(dragFrom, 1);
      slides.splice(to, 0, moved);
      activeIndex = slides.indexOf(activeSlide);
      dragFrom = -1;
      saveSlides();
      renderSlideList();
      renderPreview();
    });
  })();

  function selectSlide(idx) {
    activeIndex = idx;
    setTimeout(() => { renderAudioPanel(); renderMediaTimeline(); }, 0);
    if (idx !== ghostIndex) {
      tempGhosts = null;
      ghostIndex = -1;
    }
    renderSlideList();
    
    const editor = $("slide-editor");
    editor.style.display = "block";

    const slide = slides[idx];
    $("slide-pose").value = slide.pose || "";
    $("slide-chart").value = "";
    $("slide-text").value = slide.text || "";
    $("slide-duration").value = slide.duration;
    $("slide-transition").value = slide.transition || "none";
    $("slide-trans-dur").value = slide.transition_dur !== undefined ? slide.transition_dur : 0.5;
    $("slide-trans-dur-container").style.display = slide.transition === "fade" ? "block" : "none";

    // Update Mini Previews
    const poseMini = $("slide-pose-preview-mini");
    const poseCont = $("slide-pose-preview-container");
    if (slide.pose) {
      poseMini.src = `/api/poses/${encodeURIComponent(slide.pose)}`;
      poseCont.style.display = "block";
    } else {
      poseCont.style.display = "none";
    }

    const chartMini = $("slide-chart-preview-mini");
    const chartCont = $("slide-chart-preview-container");
    const firstMedia = (slide.media || [])[0];
    if (firstMedia) {
      chartMini.src = chartHref(firstMedia.name);
      chartCont.style.display = "block";
    } else {
      chartCont.style.display = "none";
    }

    // Jump time to start of this slide
    let start = 0;
    for (let i = 0; i < idx; i++) {
      start += effDur(slides[i]);
    }
    currentTime = start;
    $("player-scrub").value = currentTime;
    renderPreview();
  }

  // Bind editor form updates
  $("slide-pose").addEventListener("change", (e) => {
    if (activeIndex >= 0) {
      slides[activeIndex].pose = e.target.value;
      saveSlides();
      renderPreview();
      // Update mini preview
      const poseMini = $("slide-pose-preview-mini");
      const poseCont = $("slide-pose-preview-container");
      if (e.target.value) {
        poseMini.src = `/api/poses/${encodeURIComponent(e.target.value)}`;
        poseCont.style.display = "block";
      } else {
        poseCont.style.display = "none";
      }
    }
  });
  $("slide-chart").addEventListener("change", (e) => {
    // the select is now an ADD control: each pick appends another media
    // item to the slide (manage/remove them in the media timeline)
    if (activeIndex >= 0 && e.target.value) {
      addMediaToSlide(slides[activeIndex], e.target.value);
      const chartMini = $("slide-chart-preview-mini");
      const chartCont = $("slide-chart-preview-container");
      chartMini.src = chartHref(e.target.value);
      chartCont.style.display = "block";
      e.target.value = "";
    }
  });
  $("slide-chart-align").addEventListener("change", (e) => {
    // align presets nudge the most recently added media item
    if (activeIndex < 0) return;
    const ms = slides[activeIndex].media || [];
    const m = ms[ms.length - 1];
    if (!m) return;
    m.x = e.target.value === "left" ? 80
        : e.target.value === "center" ? Math.round((1280 - m.w) / 2) : 650;
    saveSlides(); renderPreview();
  });
  $("slide-text").addEventListener("input", (e) => {
    if (activeIndex >= 0) { slides[activeIndex].text = e.target.value; saveSlides(); renderPreview(); }
  });
  $("slide-duration").addEventListener("input", (e) => {
    if (activeIndex >= 0) {
      slides[activeIndex].duration = Math.max(0.5, Number(e.target.value) || 3.0);
      const at = slideAudioTotal(slides[activeIndex]);
      $("aud-state").textContent = at > slides[activeIndex].duration
        ? `slide held at ${at.toFixed(1)}s by its audio` : "";
      saveSlides(); renderPreview(); renderSlideList();
    }
  });
  $("slide-transition").addEventListener("change", (e) => {
    if (activeIndex >= 0) {
      slides[activeIndex].transition = e.target.value;
      slides[activeIndex].transition_dur = e.target.value === "fade" ? 0.5 : 0;
      $("slide-trans-dur").value = slides[activeIndex].transition_dur;
      $("slide-trans-dur-container").style.display = e.target.value === "fade" ? "block" : "none";
      saveSlides();
      renderPreview();
    }
  });
  $("slide-trans-dur").addEventListener("input", (e) => {
    if (activeIndex >= 0) {
      slides[activeIndex].transition_dur = Math.max(0.1, Number(e.target.value) || 0.5);
      saveSlides();
      renderPreview();
    }
  });

  // Build a new slide that inherits pose/chart (image + position + alignment)
  // from a reference slide, so assets stay in the same spot between frames.
  function makeInheritedSlide(ref, caption) {
    const r = ref || {};
    return {
      pose: r.pose || "",
      pose_align: r.pose_align || "left",
      pose_x: r.pose_x,
      pose_y: r.pose_y,
      media: JSON.parse(JSON.stringify(r.media || [])),
      text: caption || "",
      duration: 3.0,
      transition: "fade",
      transition_dur: 0.5
    };
  }

  $("slide-add").addEventListener("click", () => {
    const last = slides[slides.length - 1] || {};
    slides.push(makeInheritedSlide(last, ""));
    tempGhosts = null;
    ghostIndex = -1;
    saveSlides();
    selectSlide(slides.length - 1);
  });

  // Apply the current slide's pose (image, position, alignment) to every slide.
  $("apply-pose-all").addEventListener("click", () => {
    if (activeIndex < 0) return;
    const s = slides[activeIndex];
    slides.forEach((sl) => {
      sl.pose = s.pose;
      sl.pose_align = s.pose_align;
      sl.pose_x = s.pose_x;
      sl.pose_y = s.pose_y;
    });
    saveSlides();
    renderPreview();
  });

  // Apply the current slide's chart (image, position, alignment) to every slide.
  $("apply-chart-all").addEventListener("click", () => {
    if (activeIndex < 0) return;
    const s = slides[activeIndex];
    slides.forEach((sl) => {
      sl.media = JSON.parse(JSON.stringify(s.media || []));
    });
    saveSlides();
    renderPreview();
  });

  // ---- GLOBAL BACKGROUND: a library gif/video looped behind every
  // slide, replacing the flat black background ------------------------
  let deckBg = "";
  try { deckBg = localStorage.getItem("genreg_deck_bg") || ""; } catch (e) {}
  let deckBgEl = null;

  function positionDeckBg() {
    if (!deckBgEl) return;
    const stage = $("slides-stage");
    const wrap = stage.parentElement;
    const sr = stage.getBoundingClientRect();
    const wr = wrap.getBoundingClientRect();
    const scale = Math.min(sr.width / 1280, sr.height / 720);
    deckBgEl.style.left = ((sr.width - 1280 * scale) / 2 + (sr.left - wr.left)) + "px";
    deckBgEl.style.top = ((sr.height - 720 * scale) / 2 + (sr.top - wr.top)) + "px";
    deckBgEl.style.width = (1280 * scale) + "px";
    deckBgEl.style.height = (720 * scale) + "px";
  }

  function syncDeckBgEl() {
    const stage = $("slides-stage");
    const wrap = stage.parentElement;
    if (deckBgEl) { deckBgEl.remove(); deckBgEl = null; }
    const label = $("bg-name");
    if (label) label.textContent = deckBg || "none";
    if (!deckBg) { renderPreview(); return; }
    const isGif = /\.gif$/i.test(deckBg);
    deckBgEl = document.createElement(isGif ? "img" : "video");
    if (!isGif) {
      deckBgEl.muted = true;
      deckBgEl.loop = true;
      deckBgEl.autoplay = true;
      deckBgEl.playsInline = true;
    }
    deckBgEl.src = `/api/video/file/${encodeURIComponent(deckBg)}`;
    deckBgEl.style.position = "absolute";
    deckBgEl.style.objectFit = "cover";
    deckBgEl.style.pointerEvents = "none";
    deckBgEl.style.zIndex = "0";
    stage.style.position = "relative";
    stage.style.zIndex = "1";
    wrap.insertBefore(deckBgEl, stage);
    positionDeckBg();
    if (!isGif) deckBgEl.play && deckBgEl.play().catch(() => {});
    renderPreview();
  }

  function setDeckBg(name) {
    deckBg = name || "";
    try { localStorage.setItem("genreg_deck_bg", deckBg); } catch (e) {}
    syncDeckBgEl();
  }

  if ($("bg-upload-btn")) {
    $("bg-upload-btn").addEventListener("click", () => $("bg-upload").click());
    $("bg-upload").addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const form = new FormData();
      form.append("file", file);
      try {
        const res = await (await fetch("/api/video/upload", {
          method: "POST", body: form,
        })).json();
        if (res.error) throw new Error(res.error);
        await loadLibrary();
        setDeckBg(res.name);
      } catch (err) {
        alert("Background upload failed: " + err.message);
      }
      e.target.value = "";
    });
    $("bg-clear").addEventListener("click", () => setDeckBg(""));
    window.addEventListener("resize", positionDeckBg);
    setTimeout(syncDeckBgEl, 0);
  }

  $("slide-clear").addEventListener("click", () => {
    if (confirm("Clear all slides?")) {
      slides = [];
      activeIndex = -1;
      $("slide-editor").style.display = "none";
      saveSlides();
      renderPreview();
    }
  });

  // Preview Renderer (Client Side HTML/SVG)
  function renderPreview() {
    const stage = $("slides-stage");
    stage.innerHTML = "";
    
    if (!slides.length) {
      $("stage-slide-label").textContent = "no slides";
      return;
    }

    // Accumulate Ranges
    const ranges = [];
    let curr = 0;
    slides.forEach((s) => {
      const dur = effDur(s);
      const transDur = s.transition === "fade" ? (s.transition_dur !== undefined ? Number(s.transition_dur) : 0.5) : 0;
      ranges.push({
        slide: s,
        start: curr,
        end: curr + dur,
        transDur: transDur
      });
      curr += dur;
    });

    // Determine Active Slide
    let activeIdx = ranges.findIndex(r => currentTime >= r.start && currentTime <= r.end);
    if (activeIdx === -1) {
      activeIdx = ranges.length - 1;
    }
    
    $("stage-slide-label").textContent = `Slide ${activeIdx + 1} / ${slides.length}`;
    
    const bg = slides[0].bg || "#0b0d10";
    stage.style.background = deckBg ? "transparent" : bg;
    positionDeckBg();

    const r = ranges[activeIdx];
    if (!r) return;

    // Check transition overlap
    if (r.transDur > 0 && (r.end - currentTime) < r.transDur && activeIdx < ranges.length - 1) {
      const nextR = ranges[activeIdx + 1];
      const alpha = (currentTime - (r.end - r.transDur)) / r.transDur;
      
      const g1 = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g1.setAttribute("opacity", 1.0 - alpha);
      g1.innerHTML = makeSlideGroup(r.slide, currentTime - r.start);

      const g2 = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g2.setAttribute("opacity", alpha);
      g2.innerHTML = makeSlideGroup(nextR.slide, 0);

      stage.appendChild(g1);
      stage.appendChild(g2);
    } else {
      stage.innerHTML = makeSlideGroup(r.slide, currentTime - r.start);

      if (tempGhosts && activeIndex === ghostIndex) {
        if (!r.slide.pose && tempGhosts.pose) {
          stage.appendChild(makeGhostGroup(tempGhosts.pose, tempGhosts.pose_align, "keep-pose"));
        }
        if (!(r.slide.media || []).length && tempGhosts.chart) {
          stage.appendChild(makeGhostGroup(tempGhosts.chart, tempGhosts.chart_align, "keep-chart"));
        }
      }
    }

    // screen-boundary outline: marks the exact 1280x720 frame that ends
    // up in the export, so placed items are visibly on/off screen
    const border = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    border.setAttribute("x", "1.5");
    border.setAttribute("y", "1.5");
    border.setAttribute("width", "1277");
    border.setAttribute("height", "717");
    border.setAttribute("fill", "none");
    border.setAttribute("stroke", "#8b95a1");
    border.setAttribute("stroke-width", "3");
    border.setAttribute("stroke-dasharray", "14 10");
    border.setAttribute("opacity", "0.55");
    border.style.pointerEvents = "none";
    stage.appendChild(border);
  }

  function makeGhostGroup(assetName, align, actionType) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("opacity", "0.35");
    
    const w = 1280, h = 720;
    let out = [];
    
    const prevSlide = slides[activeIndex - 1] || {};
    let cx = 305, cy = 320;
    
    if (actionType === "keep-pose") {
      let px = 80, py = 80;
      const pw = 450, ph = 480;
      if (prevSlide.pose_x !== undefined && prevSlide.pose_y !== undefined) {
        px = prevSlide.pose_x;
        py = prevSlide.pose_y;
      } else {
        if (align === "right") px = 750;
        else if (align === "center") px = (w - pw) / 2;
      }
      cx = px + pw / 2;
      cy = py + ph / 2;
      out.push(`<image href="/api/poses/${encodeURIComponent(assetName)}" x="${px}" y="${py}" width="${pw}" height="${ph}" preserveAspectRatio="xMidYMid meet"/>`);
    } else {
      let chx = 650, chy = 80;
      const cw = 550, ch = 420;
      const pm = (prevSlide.media || [])[0];
      if (pm) {
        chx = pm.x;
        chy = pm.y;
      } else {
        if (align === "left") chx = 80;
        else if (align === "center") chx = (w - cw) / 2;
      }
      cx = chx + cw / 2;
      cy = chy + ch / 2;
      out.push(`<image href="${chartHref(assetName)}" x="${chx}" y="${chy}" width="${cw}" height="${ch}" preserveAspectRatio="xMidYMid meet"/>`);
    }
    
    g.innerHTML = out.join("");
    
    const plusG = document.createElementNS("http://www.w3.org/2000/svg", "g");
    plusG.setAttribute("data-action", actionType);
    plusG.setAttribute("class", "onion-plus");
    plusG.style.cursor = "pointer";
    
    plusG.innerHTML = `<circle cx="${cx}" cy="${cy}" r="24" fill="#2e7d5b" stroke="#ffffff" stroke-width="2" />`
      + `<line x1="${cx - 10}" y1="${cy}" x2="${cx + 10}" y2="${cy}" stroke="#ffffff" stroke-width="3" stroke-linecap="round" />`
      + `<line x1="${cx}" y1="${cy - 10}" x2="${cx}" y2="${cy + 10}" stroke="#ffffff" stroke-width="3" stroke-linecap="round" />`;
      
    const wrapper = document.createElementNS("http://www.w3.org/2000/svg", "g");
    wrapper.appendChild(g);
    wrapper.appendChild(plusG);
    return wrapper;
  }

  function makeSlideGroup(s, localT) {
    let out = [];
    const w = 1280, h = 720;
    const t = Number(localT) || 0;
    
    // Pose
    if (s.pose) {
      const align = s.pose_align || "left";
      let px = 80, py = 80;
      const pw = 450, ph = 480;
      if (s.pose_x !== undefined && s.pose_y !== undefined) {
        px = s.pose_x;
        py = s.pose_y;
      } else {
        if (align === "right") px = 750;
        else if (align === "center") px = (w - pw) / 2;
      }
      
      if (align !== "none" || s.pose_x !== undefined) {
        out.push(`<image href="/api/poses/${encodeURIComponent(s.pose)}" x="${px}" y="${py}" width="${pw}" height="${ph}" data-drag="pose" style="cursor: move;" preserveAspectRatio="xMidYMid meet"/>`);
      }
    }

    // Media items (multiple charts/videos per slide, each with its own
    // visibility window). While EDITING (paused) an item outside its
    // window still shows at low opacity so it stays selectable/dragable.
    (s.media || []).forEach((m, mi) => {
      if (!m.name) return;
      const vis = mediaVisible(m, t);
      if (!vis && (typeof isPlaying !== "undefined" && isPlaying)) return;
      let op = "";
      if (!vis) op = ' opacity="0.3"';
      else {
        const f = mediaFadeOpacity(m, t);
        if (f < 0.999) op = ` opacity="${f.toFixed(3)}"`;
      }
      out.push(`<g${op}>` +
        `<image href="${chartHref(m.name)}" x="${m.x}" y="${m.y}" width="${m.w}" height="${m.h}" data-drag="media" data-mi="${mi}" style="cursor: move;" preserveAspectRatio="xMidYMid meet"/>` +
        `<rect x="${m.x + m.w - 14}" y="${m.y + m.h - 14}" width="16" height="16" rx="3" fill="#4ea1ff" fill-opacity="0.85" data-drag="media-resize" data-mi="${mi}" style="cursor: nwse-resize;"/>` +
        `</g>`);
    });

    // Media-request placeholder: the template asked for a chart/video
    // that only the human can supply - show WHERE it goes (per the pose
    // gesture) and WHAT is needed, until something is uploaded
    if (s.media_request && !(s.media || []).length) {
      const gp = gestureMediaPos(gestureOf(s));
      const rEsc = (txt) => String(txt).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const words = String(s.media_request).split(/\s+/);
      const reqLines = [];
      let cur = "";
      words.forEach((wd) => {
        const cand = cur ? cur + " " + wd : wd;
        if (cand.length > 38 && cur) { reqLines.push(cur); cur = wd; }
        else cur = cand;
      });
      if (cur) reqLines.push(cur);
      const spans = reqLines.slice(0, 6).map((ln, i) =>
        `<tspan x="${gp.x + 275}" dy="${i ? 22 : 0}">${rEsc(ln)}</tspan>`).join("");
      out.push(`<g opacity="0.85">` +
        `<rect x="${gp.x}" y="${gp.y}" width="550" height="420" rx="10" fill="#10141c" fill-opacity="0.5" stroke="#f0a35e" stroke-width="2" stroke-dasharray="10 8"/>` +
        `<text x="${gp.x + 275}" y="${gp.y + 170}" font-family="Arial, Helvetica, sans-serif" font-size="20" font-weight="bold" fill="#f0a35e" text-anchor="middle">MEDIA NEEDED</text>` +
        `<text x="${gp.x + 275}" y="${gp.y + 205}" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="#c7d0da" text-anchor="middle">${spans}</text>` +
        `<text x="${gp.x + 275}" y="${gp.y + 390}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#8b95a1" text-anchor="middle">upload it from the MEDIA TIMELINE panel below</text>` +
        `</g>`);
    }

    // CC Text - word-wrapped to the box, box grows with the lines
    if (s.text) {
      const esc = (txt) => txt.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const wrapCC = (text, maxChars) => {
        const outLines = [];
        String(text).split("\n").forEach((para) => {
          let cur = "";
          para.split(/\s+/).filter(Boolean).forEach((word) => {
            const cand = cur ? cur + " " + word : word;
            if (cand.length > maxChars && cur) { outLines.push(cur); cur = word; }
            else cur = cand;
          });
          outLines.push(cur || "");
        });
        return outLines;
      };
      let ccFont = 24;
      let ccLines = wrapCC(s.text, 68);
      if (ccLines.length > 4) { ccFont = 20; ccLines = wrapCC(s.text, 82); }
      const ccLineH = ccFont * 1.35;
      const ccPad = 14;
      const ccBoxH = ccPad * 2 + ccLineH * ccLines.length;
      const ccBoxY = 692 - ccBoxH;
      const ccBase = ccBoxY + ccPad + ccFont * 0.85;
      const ts = ccLines.map((ln, i) => `<tspan x="640" dy="${i ? ccLineH : 0}">${esc(ln)}</tspan>`).join("");
      out.push(`<rect x="100" y="${ccBoxY}" width="1080" height="${ccBoxH}" rx="8" fill="#10141c" fill-opacity="0.85" stroke="#1c232c" stroke-width="1"/>`);
      out.push(`<text x="640" y="${ccBase}" font-family="Arial, Helvetica, sans-serif" font-size="${ccFont}" font-weight="bold" fill="#f0ede4" text-anchor="middle">${ts}</text>`);
    }

    return out.join("");
  }

  // Player controls
  // live video overlays: the stage SVG shows static thumbs for video
  // media; during preview a POOL of HTML <video> elements is positioned
  // over the stage (one per concurrently visible video) and kept in
  // sync with the deck clock
  let stagePool = [];

  function stageVideoEl(k) {
    while (stagePool.length <= k) {
      const wrap = $("slides-stage").parentElement;
      const v = document.createElement("video");
      v.muted = true;                 // narration owns the audio
      v.playsInline = true;
      v.style.position = "absolute";
      v.style.zIndex = "2";           // above the global background layer
      v.style.display = "none";
      v.style.objectFit = "contain";
      v.style.pointerEvents = "none";
      wrap.appendChild(v);
      stagePool.push(v);
    }
    return stagePool[k];
  }

  function hideStageVideos(from) {
    for (let i = from || 0; i < stagePool.length; i++) {
      const v = stagePool[i];
      if (!v.paused) v.pause();
      v.style.display = "none";
      v.dataset.src = "";
      v.removeAttribute("src");
    }
  }
  function hideStageVideo() { hideStageVideos(0); }

  function syncStageVideo(t) {
    // t = deck time; find the active slide and drive one overlay per
    // visible video item
    let curr = 0;
    let slide = null;
    let localT = 0;
    for (const s of slides) {
      const d = effDur(s);
      if (t < curr + d) { slide = s; localT = t - curr; break; }
      curr += d;
    }
    if (!slide) { hideStageVideos(0); return; }
    const stage = $("slides-stage");
    const wrap = stage.parentElement;
    const sr = stage.getBoundingClientRect();
    const wr = wrap.getBoundingClientRect();
    const scale = Math.min(sr.width / 1280, sr.height / 720);
    const ox = (sr.width - 1280 * scale) / 2 + (sr.left - wr.left);
    const oy = (sr.height - 720 * scale) / 2 + (sr.top - wr.top);
    let k = 0;
    (slide.media || []).forEach((m) => {
      const d = Number(m.dur) || 0;
      if (!m.name || !/\.(mp4|webm|mov)$/i.test(m.name) || !d) return;
      if (!mediaVisible(m, localT)) return;
      let mt = localT - (Number(m.start) || 0);
      if (mt < 0) return;
      if (m.loop) mt = mt % d;
      else if (mt >= d) mt = d - 0.05;               // hold last frame
      const v = stageVideoEl(k++);
      const url = `/api/video/file/${encodeURIComponent(m.name)}`;
      if (v.dataset.src !== url) {
        v.dataset.src = url;
        v.src = url;
      }
      v.loop = !!m.loop;
      v.style.left = (ox + m.x * scale) + "px";
      v.style.top = (oy + m.y * scale) + "px";
      v.style.width = (m.w * scale) + "px";
      v.style.height = (m.h * scale) + "px";
      v.style.opacity = String(mediaFadeOpacity(m, localT));
      v.style.display = "block";
      if (Math.abs((v.currentTime || 0) - mt) > 0.3) v.currentTime = mt;
      if (isPlaying && v.paused) v.play().catch(() => {});
      if (!isPlaying && !v.paused) v.pause();
    });
    hideStageVideos(k);
  }

  // preview audio: WebAudio players so multi-cut clips play SEAMLESSLY
  // (kept segments scheduled back to back, sample-accurate)
  let audioCtx = null;
  const bufferCache = {};
  let previewAudios = {};
  let playSchedule = [];

  function getCtx() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === "suspended") audioCtx.resume();
    return audioCtx;
  }

  async function clipBuffer(id) {
    if (bufferCache[id]) return bufferCache[id];
    const resp = await fetch(`/api/video/slide_audio/${encodeURIComponent(id)}`);
    const buf = await getCtx().decodeAudioData(await resp.arrayBuffer());
    bufferCache[id] = buf;
    return buf;
  }

  function playClipSeamless(clip, offEff) {
    // start playback offEff seconds into the clip's EFFECTIVE (kept)
    // timeline; returns a handle with stop()
    const handle = { sources: [], dead: false,
                     stop() { this.dead = true;
                              this.sources.forEach((src) => { try { src.stop(); } catch (e) {} });
                              this.sources = []; } };
    clipBuffer(clip.id).then((buf) => {
      if (handle.dead) return;
      const ctx = getCtx();
      let when = ctx.currentTime + 0.02;
      let skip = Math.max(0, offEff);
      keptSegments(clip).forEach((seg) => {
        const segLen = seg[1] - seg[0];
        if (skip >= segLen) { skip -= segLen; return; }
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        src.start(when, seg[0] + skip, segLen - skip);
        when += segLen - skip;
        skip = 0;
        handle.sources.push(src);
      });
    }).catch(() => {});
    return handle;
  }

  function clipSchedule() {
    const sched = [];
    let t = 0;
    slides.forEach((s) => {
      let off = 0;
      (s.clips || []).forEach((c, ci) => {
        const d = effClipDur(c);
        sched.push({ key: c.id + ":" + t + ":" + ci, clip: c,
                     start: t + off, end: t + off + d });
        off += d;
      });
      t += effDur(s);
    });
    return sched;
  }

  function syncPreviewAudio(t) {
    playSchedule.forEach((c) => {
      const a = previewAudios[c.key];
      const within = t >= c.start && t < c.end - 0.05;
      if (within && !a) {
        previewAudios[c.key] = playClipSeamless(c.clip, t - c.start);
      } else if (!within && a) {
        a.stop();
        delete previewAudios[c.key];
      }
    });
  }

  function stopPreviewAudio() {
    Object.values(previewAudios).forEach((a) => a.stop());
    previewAudios = {};
  }

  $("player-play").addEventListener("click", () => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  });

  function play() {
    if (!slides.length) return;
    isPlaying = true;
    $("player-play").textContent = "Pause";
    playStartRealTime = performance.now();
    playStartScrubTime = currentTime;
    playSchedule = clipSchedule();
    
    function tick(now) {
      if (!isPlaying) return;
      const elapsed = (now - playStartRealTime) / 1000;
      currentTime = playStartScrubTime + elapsed;
      
      if (currentTime >= totalDuration) {
        currentTime = 0;
        playStartRealTime = now;
        playStartScrubTime = 0;
        stopPreviewAudio();
      }
      
      syncPreviewAudio(currentTime);
      syncStageVideo(currentTime);
      $("player-scrub").value = currentTime;
      updateTimeLabel();
      renderPreview();
      playerAnimFrame = requestAnimationFrame(tick);
    }
    playerAnimFrame = requestAnimationFrame(tick);
  }

  function pause() {
    isPlaying = false;
    $("player-play").textContent = "Play";
    if (playerAnimFrame) cancelAnimationFrame(playerAnimFrame);
    stopPreviewAudio();
    stagePool.forEach((v) => { if (!v.paused) v.pause(); });
  }

  $("player-prev").addEventListener("click", () => {
    pause();
    // find start of active slide, jump to previous
    let idx = activeIndex;
    if (idx === -1) idx = 0;
    else if (idx > 0) idx--;
    selectSlide(idx);
  });

  $("player-next").addEventListener("click", () => {
    pause();
    let idx = activeIndex;
    if (idx === -1) idx = 0;
    else if (idx < slides.length - 1) idx++;
    selectSlide(idx);
  });

  $("player-scrub").addEventListener("input", (e) => {
    pause();
    currentTime = Number(e.target.value);
    updateTimeLabel();
    renderPreview();
    syncStageVideo(currentTime);
  });

  $("slides-stage").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn || activeIndex === -1 || !tempGhosts) return;
    
    const action = btn.dataset.action;
    if (action === "keep-pose") {
      slides[activeIndex].pose = tempGhosts.pose;
      slides[activeIndex].pose_align = tempGhosts.pose_align;
      slides[activeIndex].pose_x = tempGhosts.pose_x;
      slides[activeIndex].pose_y = tempGhosts.pose_y;
      $("slide-pose").value = tempGhosts.pose;
      
      const poseMini = $("slide-pose-preview-mini");
      const poseCont = $("slide-pose-preview-container");
      poseMini.src = `/api/poses/${encodeURIComponent(tempGhosts.pose)}`;
      poseCont.style.display = "block";
      
      tempGhosts.pose = "";
      saveSlides();
      renderPreview();
    } else if (action === "keep-chart") {
      const gsl = slides[activeIndex];
      gsl.media = gsl.media || [];
      const gItem = { name: tempGhosts.chart,
        x: Number(tempGhosts.chart_x) || 650,
        y: Number(tempGhosts.chart_y) || 80,
        w: 550, h: 420, start: 0, dur: 0, loop: false, end: 0 };
      gsl.media.push(gItem);
      refreshMediaDur(gsl, gItem);
      
      const chartMini = $("slide-chart-preview-mini");
      const chartCont = $("slide-chart-preview-container");
      chartMini.src = chartHref(tempGhosts.chart);
      chartCont.style.display = "block";
      
      tempGhosts.chart = "";
      saveSlides();
      renderPreview();
    }
  });

  let dragItem = null;
  let dragMi = -1;
  let dragStartPos = { x: 0, y: 0 };
  let itemStartPos = { x: 0, y: 0 };
  
  $("slides-stage").addEventListener("pointerdown", (e) => {
    const item = e.target.closest("[data-drag]");
    if (!item || activeIndex === -1 || isPlaying) return;
    
    dragItem = item.dataset.drag;
    e.preventDefault();
    item.setPointerCapture(e.pointerId);
    
    const stage = $("slides-stage");
    const rect = stage.getBoundingClientRect();
    const scaleX = 1280 / rect.width;
    const scaleY = 720 / rect.height;
    
    dragStartPos = {
      x: e.clientX * scaleX,
      y: e.clientY * scaleY
    };
    
    const slide = slides[activeIndex];
    if (dragItem === "pose") {
      let curX = slide.pose_x;
      let curY = slide.pose_y;
      if (curX === undefined || curY === undefined) {
        const align = slide.pose_align || "left";
        curX = 80; curY = 80;
        if (align === "right") curX = 750;
        else if (align === "center") curX = 415;
      }
      itemStartPos = { x: curX, y: curY };
    } else if (dragItem === "media" || dragItem === "media-resize") {
      dragMi = Number(item.dataset.mi);
      const m = (slide.media || [])[dragMi];
      if (!m) { dragItem = null; return; }
      itemStartPos = dragItem === "media"
        ? { x: m.x, y: m.y } : { w: m.w, h: m.h };
    }
    
    const moveHandler = (ev) => {
      if (!dragItem) return;
      const curMouseX = ev.clientX * scaleX;
      const curMouseY = ev.clientY * scaleY;
      const dx = curMouseX - dragStartPos.x;
      const dy = curMouseY - dragStartPos.y;
      
      const slideToEdit = slides[activeIndex];
      if (dragItem === "pose") {
        slideToEdit.pose_x = itemStartPos.x + dx;
        slideToEdit.pose_y = itemStartPos.y + dy;
      } else if (dragItem === "media") {
        const m = (slideToEdit.media || [])[dragMi];
        if (m) {
          m.x = Math.round(itemStartPos.x + dx);
          m.y = Math.round(itemStartPos.y + dy);
        }
      } else if (dragItem === "media-resize") {
        const m = (slideToEdit.media || [])[dragMi];
        if (m) {
          m.w = Math.max(80, Math.round(itemStartPos.w + dx));
          m.h = Math.max(60, Math.round(itemStartPos.h + dy));
        }
      }
      renderPreview();
    };
    
    const upHandler = (ev) => {
      if (dragItem) {
        item.releasePointerCapture(ev.pointerId);
        dragItem = null;
        saveSlides();
      }
      window.removeEventListener("pointermove", moveHandler);
      window.removeEventListener("pointerup", upHandler);
    };
    
    window.addEventListener("pointermove", moveHandler);
    window.addEventListener("pointerup", upHandler);
  });

  // ── Script block: paste a script, highlight -> slides ───────────────
  let scriptSel = "";

  // Restore any previously pasted script.
  try {
    const savedScript = localStorage.getItem("genreg_script");
    if (savedScript) $("script-text").value = savedScript;
  } catch (e) { /* ignore */ }

  function updateScriptSelection() {
    const ta = $("script-text");
    const raw = ta.value.substring(ta.selectionStart, ta.selectionEnd).trim();
    if (raw) scriptSel = raw;
    const info = $("script-sel-info");
    if (raw) {
      const preview = raw.length > 42 ? raw.slice(0, 42) + "…" : raw;
      info.textContent = `Selected (${raw.length} chars): "${preview}"`;
    } else {
      info.textContent = "Highlight text above, then use the buttons below.";
    }
  }

  ["mouseup", "keyup", "select"].forEach((ev) =>
    $("script-text").addEventListener(ev, updateScriptSelection)
  );
  $("script-text").addEventListener("input", () => {
    localStorage.setItem("genreg_script", $("script-text").value);
  });

  // Highlight -> append N new slides carrying it as the caption.
  $("script-add-sel").addEventListener("click", () => {
    if (!scriptSel) { alert("Highlight some text in the script first."); return; }
    const span = Math.max(1, Math.min(50, parseInt($("script-span").value, 10) || 1));
    let ref = slides[slides.length - 1];
    for (let i = 0; i < span; i++) {
      slides.push(makeInheritedSlide(ref, scriptSel));
      ref = slides[slides.length - 1];
    }
    saveSlides();
    selectSlide(slides.length - span);
  });

  // Highlight -> set the currently selected slide's caption.
  $("script-set-current").addEventListener("click", () => {
    if (activeIndex < 0) { alert("Select a slide first."); return; }
    if (!scriptSel) { alert("Highlight some text in the script first."); return; }
    slides[activeIndex].text = scriptSel;
    $("slide-text").value = scriptSel;
    saveSlides();
    renderPreview();
  });


  // ---- SLIDE AUDIO: per-slide mic recordings (ordered clips) ----------
  let mediaStream = null;
  let mediaRecorder = null;
  let recChunks = [];
  let playingAudio = null;

  function fmtDur(d) { return (Number(d) || 0).toFixed(1) + "s"; }

  async function clipDuration(blob) {
    // webm blobs report Infinity via <audio> in Chromium; decode instead
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    try {
      const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
      return buf.duration;
    } finally { ctx.close(); }
  }

  function renderAudioPanel() {
    const box = $("aud-clips");
    const label = $("aud-slide-label");
    if (!box) return;
    box.innerHTML = "";
    if (activeIndex < 0 || !slides[activeIndex]) {
      label.textContent = "select a slide";
      $("aud-record").disabled = true;
      return;
    }
    $("aud-record").disabled = false;
    label.textContent = `slide #${activeIndex + 1}`;
    const clips = slides[activeIndex].clips || [];
    if (!clips.length) {
      box.innerHTML = '<div style="font-size:11px;color:#5a6672;">no recordings for this slide</div>';
      return;
    }
    clips.forEach((clip, ci) => {
      const row = document.createElement("div");
      row.className = "aud-row";
      const moveOpts = slides.map((_, i2) =>
        `<option value="${i2}" ${i2 === activeIndex ? "disabled" : ""}>#${i2 + 1}</option>`).join("");
      row.innerHTML =
        `<span>clip ${ci + 1}</span><span style="color:#8b95a1">${fmtDur(effClipDur(clip))}${(clip.cuts && clip.cuts.length) ? " (trimmed)" : ""}</span>` +
        `<button data-a="play">Play</button>` +
        `<button data-a="trim">Trim</button>` +
        `<label style="display:flex;gap:4px;align-items:center;color:#8b95a1">move to` +
        ` <select data-a="move">${moveOpts}</select></label>` +
        `<button data-a="del" class="aud-del" style="margin-left:auto">Delete</button>`;
      row.querySelector('[data-a="play"]').addEventListener("click", (ev) => {
        const btn = ev.target;
        const wasStop = btn.textContent === "Stop";
        if (playingAudio) { playingAudio.stop(); playingAudio = null; }
        document.querySelectorAll('.aud-row [data-a="play"]').forEach(function (b) { b.textContent = "Play"; });
        if (wasStop) return;
        playingAudio = playClipSeamless(clip, 0);
        btn.textContent = "Stop";
        setTimeout(() => {
          if (playingAudio) { playingAudio.stop(); playingAudio = null; }
          btn.textContent = "Play";
        }, effClipDur(clip) * 1000 + 150);
      });
      row.querySelector('[data-a="trim"]').addEventListener("click", () => {
        const openEd = document.querySelector(".trim-ed");
        if (openEd) openEd.remove();
        if (trimOpen && trimOpen.clip === clip) { trimOpen = null; return; }
        trimOpen = { clip };
        openTrimEditor(row, clip);
      });
      row.querySelector('[data-a="del"]').addEventListener("click", async () => {
        slides[activeIndex].clips.splice(ci, 1);
        saveSlides(); renderAudioPanel(); renderSlideList();
        try { await fetch(`/api/video/slide_audio/${encodeURIComponent(clip.id)}`, { method: "DELETE" }); }
        catch (e) { console.error("clip delete", e); }
      });
      const sel = row.querySelector('[data-a="move"]');
      sel.value = String(activeIndex);
      sel.addEventListener("change", () => {
        const to = Number(sel.value);
        if (to === activeIndex || !slides[to]) return;
        slides[activeIndex].clips.splice(ci, 1);
        slides[to].clips = slides[to].clips || [];
        slides[to].clips.push(clip);
        saveSlides(); renderAudioPanel(); renderSlideList();
      });
      box.appendChild(row);
    });
  }

  // ---- TRIM EDITOR: waveform + draggable cut regions -----------------
  let trimOpen = null;                 // {slide, ci} currently open
  let trimPlayer = null;

  function drawTrim(cv, buf, clip, sel) {
    const cx2 = cv.getContext("2d");
    const W2 = cv.width, H2 = cv.height;
    cx2.clearRect(0, 0, W2, H2);
    cx2.fillStyle = "#0b0e12";
    cx2.fillRect(0, 0, W2, H2);
    if (buf) {                         // waveform peaks
      const data = buf.getChannelData(0);
      const step = Math.max(1, Math.floor(data.length / W2));
      cx2.fillStyle = "#3d5a80";
      for (let x = 0; x < W2; x++) {
        let mx = 0;
        const a0 = x * step;
        for (let i2 = a0; i2 < a0 + step && i2 < data.length; i2 += 16) {
          const v = Math.abs(data[i2]);
          if (v > mx) mx = v;
        }
        const hh = Math.max(1, mx * (H2 - 8));
        cx2.fillRect(x, (H2 - hh) / 2, 1, hh);
      }
    }
    const dur = Number(clip.dur) || 1;
    normCuts(clip.cuts, dur).forEach((c, i2) => {
      const x0 = c[0] / dur * W2, x1 = c[1] / dur * W2;
      cx2.fillStyle = i2 === sel ? "rgba(248,81,73,0.4)" : "rgba(248,81,73,0.22)";
      cx2.fillRect(x0, 0, x1 - x0, H2);
      cx2.fillStyle = "#f85149";
      cx2.fillRect(x0, 0, 2, H2);
      cx2.fillRect(x1 - 2, 0, 2, H2);
    });
  }

  function openTrimEditor(row, clip) {
    if (trimPlayer) { trimPlayer.stop(); trimPlayer = null; }
    const ed = document.createElement("div");
    ed.className = "trim-ed";
    ed.innerHTML =
      '<canvas class="trim-cv" width="560" height="64"></canvas>' +
      '<div class="trim-bar">' +
      '<button data-t="play">Play kept</button>' +
      '<button data-t="addcut">Add cut</button>' +
      '<button data-t="delcut" disabled>Delete cut</button>' +
      '<span class="trim-info"></span>' +
      '<button data-t="close" style="margin-left:auto">Close</button></div>' +
      '<div class="trim-hint">drag a red edge to resize a cut - drag on the wave to make a new cut - click a cut to select</div>';
    row.after(ed);
    const cv = ed.querySelector(".trim-cv");
    const info = ed.querySelector(".trim-info");
    const delBtn = ed.querySelector('[data-t="delcut"]');
    let buf = null;
    let sel = -1;
    let drag = null;                   // {mode:'new'|'edge', idx, edge, x0}
    const dur = Number(clip.dur) || 1;

    function refresh() {
      clip.cuts = normCuts(clip.cuts, dur);
      drawTrim(cv, buf, clip, sel);
      info.textContent = `kept ${effClipDur(clip).toFixed(1)}s of ${dur.toFixed(1)}s` +
        (clip.cuts.length ? ` - ${clip.cuts.length} cut(s)` : "");
      delBtn.disabled = sel < 0;
      saveSlides(); renderSlideList(); updateScrubMax();
    }

    clipBuffer(clip.id).then((b) => { buf = b; refresh(); });

    function evT(ev) {
      const r = cv.getBoundingClientRect();
      return Math.max(0, Math.min(dur,
        (ev.clientX - r.left) / r.width * dur));
    }

    cv.addEventListener("pointerdown", (ev) => {
      const t = evT(ev);
      const cuts = normCuts(clip.cuts, dur);
      for (let i2 = 0; i2 < cuts.length; i2++) {
        const edge0 = Math.abs(t - cuts[i2][0]) < dur * 0.02;
        const edge1 = Math.abs(t - cuts[i2][1]) < dur * 0.02;
        if (edge0 || edge1) {
          drag = { mode: "edge", idx: i2, edge: edge0 ? 0 : 1 };
          sel = i2;
          cv.setPointerCapture(ev.pointerId);
          return;
        }
        if (t > cuts[i2][0] && t < cuts[i2][1]) {
          sel = i2; refresh(); return;
        }
      }
      drag = { mode: "new", x0: t };
      clip.cuts = cuts.concat([[t, t]]);
      sel = clip.cuts.length - 1;
      cv.setPointerCapture(ev.pointerId);
    });
    cv.addEventListener("pointermove", (ev) => {
      if (!drag) return;
      const t = evT(ev);
      const cuts = clip.cuts;
      if (drag.mode === "new") {
        const c = cuts[cuts.length - 1];
        c[0] = Math.min(drag.x0, t); c[1] = Math.max(drag.x0, t);
      } else {
        cuts[drag.idx][drag.edge] = t;
        if (cuts[drag.idx][0] > cuts[drag.idx][1]) {
          const tmp = cuts[drag.idx][0];
          cuts[drag.idx][0] = cuts[drag.idx][1];
          cuts[drag.idx][1] = tmp;
          drag.edge = 1 - drag.edge;
        }
      }
      drawTrim(cv, buf, clip, sel);
    });
    cv.addEventListener("pointerup", (ev) => {
      const t = evT(ev);
      const wasNew = drag && drag.mode === "new";
      drag = null;
      clip.cuts = normCuts(clip.cuts, dur);
      // a bare click on empty wave creates a zero-width cut: drop it and
      // treat the click as deselect; otherwise keep the touched cut
      // SELECTED so Delete stays armed after release
      if (wasNew) {
        const tiny = clip.cuts.findIndex((c) => c[1] - c[0] < dur * 0.015);
        if (tiny >= 0) {
          clip.cuts.splice(tiny, 1);
          sel = -1;
          refresh();
          return;
        }
      }
      sel = clip.cuts.findIndex((c) =>
        t >= c[0] - dur * 0.02 && t <= c[1] + dur * 0.02);
      refresh();
    });

    ed.querySelector('[data-t="play"]').addEventListener("click", (ev) => {
      if (trimPlayer) { trimPlayer.stop(); trimPlayer = null;
        ev.target.textContent = "Play kept"; return; }
      trimPlayer = playClipSeamless(clip, 0);
      ev.target.textContent = "Stop";
      const total = effClipDur(clip);
      setTimeout(() => { if (trimPlayer) { trimPlayer.stop(); trimPlayer = null; }
        ev.target.textContent = "Play kept"; }, total * 1000 + 150);
    });
    ed.querySelector('[data-t="addcut"]').addEventListener("click", () => {
      const segs = keptSegments(clip);
      if (!segs.length) return;
      const seg = segs.reduce((a, b) => (b[1] - b[0] > a[1] - a[0] ? b : a));
      const mid = (seg[0] + seg[1]) / 2;
      const w = Math.min(0.5, (seg[1] - seg[0]) / 4);
      clip.cuts = normCuts((clip.cuts || []).concat([[mid - w / 2, mid + w / 2]]), dur);
      sel = -1; refresh();
    });
    delBtn.addEventListener("click", () => {
      if (sel >= 0) { clip.cuts.splice(sel, 1); sel = -1; refresh(); }
    });
    ed.querySelector('[data-t="close"]').addEventListener("click", () => {
      if (trimPlayer) { trimPlayer.stop(); trimPlayer = null; }
      ed.remove(); trimOpen = null;
    });
    refresh();
  }

  // ---- MEDIA TIMELINE: one row per media item - when it appears, if
  // it loops, and when it vanishes so another item can take its place
  function renderMediaTimeline() {
    const panel = $("media-tl");
    if (!panel) return;
    if (activeIndex < 0 || !slides[activeIndex]) { panel.style.display = "none"; return; }
    const slide = slides[activeIndex];
    const items = slide.media || [];
    if (!items.length && !slide.media_request) { panel.style.display = "none"; return; }
    panel.style.display = "block";
    const total = effDur(slide);
    const box = $("media-rows");
    box.innerHTML = "";
    if (slide.media_request) {
      const req = document.createElement("div");
      req.className = "media-req";
      const rEsc = (txt) => String(txt).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      req.innerHTML =
        `<span class="media-req-tag">MEDIA NEEDED</span>` +
        `<span class="media-req-text">${rEsc(slide.media_request)}</span>` +
        `<button class="runs-btn vd-mini media-req-upload">Upload</button>` +
        `<button class="runs-btn vd-mini media-req-dismiss" title="clear the request without uploading">Dismiss</button>`;
      box.appendChild(req);
    }
    items.forEach((m, mi) => {
      const d = Number(m.dur) || 0;
      const geom = () => {
        const s2 = Math.min(Number(m.start) || 0, Math.max(0, total - 0.1));
        const eAuto = (m.loop || !d) ? total : Math.min(total, s2 + d);
        const e2 = Number(m.end) > 0 ? Math.min(Number(m.end), total) : eAuto;
        return { s: s2, e: Math.max(e2, s2) };
      };

      const row = document.createElement("div");
      row.className = "media-row";
      const head = document.createElement("div");
      head.className = "media-row-head";
      head.innerHTML =
        `<span class="media-name" title="${m.name}">${m.name}</span>` +
        `<label><input type="checkbox" ${m.loop ? "checked" : ""} data-mi="${mi}" class="media-loop-cb" /> loop</label>` +
        `<label title="0.5s fade when it appears"><input type="checkbox" ${m.fade_in ? "checked" : ""} data-mi="${mi}" class="media-fadein-cb" /> fade in</label>` +
        `<label title="0.5s fade at its vanish time (needs a vanish handle set)"><input type="checkbox" ${m.fade_out ? "checked" : ""} data-mi="${mi}" class="media-fadeout-cb" /> fade out</label>` +
        `<span class="media-row-info"></span>` +
        `<button class="sld-act sld-del media-rm" data-mi="${mi}" title="remove from slide">&times;</button>`;
      row.appendChild(head);

      const track = document.createElement("div");
      track.className = "media-track";
      const span = document.createElement("div");
      span.className = "media-span" + (m.loop ? " media-span-loop" : "");
      track.appendChild(span);
      const hs = document.createElement("div");
      hs.className = "media-handle";
      hs.title = "drag: when this media appears";
      track.appendChild(hs);
      const he = document.createElement("div");
      he.className = "media-handle media-handle-end";
      he.title = "drag: when it vanishes (far right = until slide end)";
      track.appendChild(he);
      row.appendChild(track);
      box.appendChild(row);

      const info = head.querySelector(".media-row-info");
      const paint = () => {
        const g = geom();
        hs.style.left = (g.s / total * 100) + "%";
        he.style.left = (g.e / total * 100) + "%";
        span.style.left = (g.s / total * 100) + "%";
        span.style.width = (Math.max(0, g.e - g.s) / total * 100) + "%";
        const endTxt = Number(m.end) > 0 ? `vanishes ${Number(m.end).toFixed(1)}s`
          : (m.loop || !d ? "until slide end"
                          : `ends ${Math.min(total, g.s + d).toFixed(1)}s`);
        info.textContent = `in ${g.s.toFixed(1)}s, ${endTxt}` +
          (d ? `, runtime ${d.toFixed(1)}s` : "");
      };
      paint();

      // drag WITHOUT rebuilding the DOM (a rebuild destroys the handle
      // mid-gesture) - inline paint while moving, full re-render on release
      const dragHandle = (hd, apply) => {
        hd.addEventListener("pointerdown", (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          const move = (e2) => {
            const r = track.getBoundingClientRect();
            const t = Math.max(0, Math.min(total,
              (e2.clientX - r.left) / r.width * total));
            apply(Math.round(t * 10) / 10);
            paint();
          };
          const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            saveSlides();
            renderMediaTimeline(); renderSlideList(); updateScrubMax();
            renderPreview();
          };
          window.addEventListener("pointermove", move);
          window.addEventListener("pointerup", up);
        });
      };
      dragHandle(hs, (t) => {
        m.start = Math.min(t, Math.max(0, total - 0.1));
      });
      dragHandle(he, (t) => {
        // dragging to the far right resets to auto (until slide end)
        m.end = t >= total - 0.05 ? 0
          : Math.max(t, (Number(m.start) || 0) + 0.1);
      });
    });
  }

  $("media-rows").addEventListener("change", (ev) => {
    if (activeIndex < 0) return;
    const cls = ev.target.classList;
    const m = (slides[activeIndex].media || [])[Number(ev.target.dataset.mi)];
    if (!m) return;
    if (cls.contains("media-loop-cb")) m.loop = ev.target.checked;
    else if (cls.contains("media-fadein-cb")) m.fade_in = ev.target.checked;
    else if (cls.contains("media-fadeout-cb")) m.fade_out = ev.target.checked;
    else return;
    saveSlides(); renderMediaTimeline(); renderSlideList(); updateScrubMax();
    renderPreview();
  });
  $("media-rows").addEventListener("click", (ev) => {
    if (activeIndex < 0) return;
    if (ev.target.classList.contains("media-req-upload")) {
      $("media-req-file").click();
      return;
    }
    if (ev.target.classList.contains("media-req-dismiss")) {
      slides[activeIndex].media_request = "";
      saveSlides(); renderMediaTimeline(); renderSlideList(); renderPreview();
      return;
    }
    const btn = ev.target.closest(".media-rm");
    if (!btn) return;
    (slides[activeIndex].media || []).splice(Number(btn.dataset.mi), 1);
    saveSlides(); renderMediaTimeline(); renderSlideList(); updateScrubMax();
    renderPreview();
  });

  // fulfilling a media request: upload lands on the requesting slide at
  // the gesture-implied position and clears the prompt
  $("media-req-file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file || activeIndex < 0) return;
    const slide = slides[activeIndex];
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await (await fetch("/api/video/upload", {
        method: "POST", body: form,
      })).json();
      if (res.error) throw new Error(res.error);
      await loadLibrary();
      addMediaToSlide(slide, res.name);
      const item = slide.media[slide.media.length - 1];
      const gp = gestureMediaPos(gestureOf(slide));
      item.x = gp.x; item.y = gp.y;
      slide.media_request = "";
      saveSlides(); renderMediaTimeline(); renderSlideList(); renderPreview();
    } catch (err) {
      alert("Upload failed: " + err.message);
    }
    e.target.value = "";
  });

  $("aud-narrate").addEventListener("click", async () => {
    const state = $("aud-state");
    if (activeIndex < 0) { state.textContent = "select a slide first"; return; }
    const slide = slides[activeIndex];
    const text = (slide.text || "").trim();
    if (!text) { state.textContent = "this slide has no caption to narrate"; return; }
    const btn = $("aud-narrate");
    btn.disabled = true;
    state.textContent = "synthesizing narration...";
    try {
      const voice = ($("tmpl-voice") && $("tmpl-voice").value.trim()) || "";
      const r = await (await fetch("/api/video/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, voice: voice }),
      })).json();
      if (r.error) throw new Error(r.error);
      slide.clips = slide.clips || [];
      slide.clips.push({ id: r.id, dur: Number(r.dur) || 0, cuts: [] });
      saveSlides();
      state.textContent = (r.cached ? "reused cached narration " : "narrated ") +
        `${(Number(r.dur) || 0).toFixed(1)}s`;
      renderAudioPanel(); renderSlideList(); updateScrubMax();
    } catch (e) {
      state.textContent = "narration failed: " + e.message;
    }
    btn.disabled = false;
  });

  $("aud-record").addEventListener("click", async () => {
    const btn = $("aud-record");
    const state = $("aud-state");
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
      return;
    }
    if (activeIndex < 0) { state.textContent = "select a slide first"; return; }
    try {
      if (!mediaStream) {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      recChunks = [];
      mediaRecorder = new MediaRecorder(mediaStream, { mimeType: "audio/webm;codecs=opus" });
      const forSlide = activeIndex;        // pin: user may click around
      mediaRecorder.addEventListener("dataavailable", (ev) => {
        if (ev.data.size) recChunks.push(ev.data);
      });
      mediaRecorder.addEventListener("stop", async () => {
        btn.textContent = "Record";
        btn.classList.remove("recording");
        state.textContent = "saving...";
        try {
          const blob = new Blob(recChunks, { type: "audio/webm" });
          const dur = await clipDuration(blob);
          const res = await (await fetch("/api/video/slide_audio", {
            method: "POST", body: blob,
            headers: { "Content-Type": "application/octet-stream" },
          })).json();
          if (res.error) throw new Error(res.error);
          const tgt = slides[forSlide];
          if (tgt) {
            tgt.clips = tgt.clips || [];
            tgt.clips.push({ id: res.id, dur: Math.round(dur * 10) / 10 });
            saveSlides();
          }
          state.textContent = `saved ${fmtDur(dur)} to slide #${forSlide + 1}`;
          renderAudioPanel(); renderSlideList();
        } catch (e) {
          state.textContent = "save failed: " + e.message;
        }
      });
      mediaRecorder.start();
      btn.textContent = "Stop";
      btn.classList.add("recording");
      state.textContent = `recording for slide #${activeIndex + 1}...`;
    } catch (e) {
      state.textContent = "mic unavailable: " + e.message;
    }
  });

  // ---- AUDIO STUDIO modal --------------------------------------------
  // every clip on every slide: listen, edit the script line, regenerate
  // and replace, optionally purging the replaced recording for good
  let ttsMap = {};             // clip id -> {text, voice, dur}
  let auPlayer = null;         // shared preview player
  let auPlayingId = null;

  async function loadTtsMap() {
    try {
      const r = await (await fetch("/api/video/tts_map")).json();
      if (r && !r.error) ttsMap = r;
    } catch (e) { /* route goes live after Flask restart; captions prefill */ }
  }

  function auStopPlayback() {
    if (auPlayer) { auPlayer.pause(); auPlayer = null; }
    auPlayingId = null;
    document.querySelectorAll(".au-play").forEach((b) => { b.textContent = "Play"; });
  }

  function renderAudioStudio() {
    const body = $("austudio-body");
    const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    const rows = [];
    slides.forEach((s, si) => {
      const clips = s.clips || [];
      if (!clips.length) return;
      const cap = (s.text || "").split("\n")[0].slice(0, 70);
      rows.push(`<div class="au-slide-hd">SLIDE #${si + 1}${cap ? " &mdash; " + esc(cap) : ""}</div>`);
      clips.forEach((c, ci) => {
        const isTts = /\.mp3$/i.test(c.id || "");
        const known = ttsMap[c.id];
        const script = known ? known.text : (isTts ? (s.text || "") : "");
        const ph = isTts && !known
          ? "script lookup needs a Flask restart - caption prefilled; edit and regenerate"
          : "type the line to synthesize - it replaces this clip";
        rows.push(`<div class="au-row" data-si="${si}" data-ci="${ci}">
          <div class="au-row-top">
            <span class="au-badge${isTts ? "" : " au-badge-mic"}">${isTts ? "TTS" : "MIC"}</span>
            <span class="au-dur">${(Number(c.dur) || 0).toFixed(1)}s</span>
            <span class="au-id">${esc(c.id || "")}</span>
            <button class="runs-btn vd-mini au-play">${auPlayingId === c.id ? "Stop" : "Play"}</button>
          </div>
          <textarea class="au-script" rows="2" placeholder="${esc(ph)}">${esc(script)}</textarea>
          <div class="au-row-bot">
            <button class="runs-btn vd-mini au-regen">Regenerate &amp; replace</button>
            <label><input type="checkbox" class="au-purge" checked /> permanently delete replaced clip</label>
            <span class="au-status"></span>
          </div>
        </div>`);
      });
    });
    body.innerHTML = rows.length ? rows.join("")
      : '<div style="color:#5a6672; font-size:12px;">No audio clips on any slide yet. Record or Narrate from the panel under the stage.</div>';
  }

  $("open-audio-studio").addEventListener("click", async () => {
    $("austudio-overlay").style.display = "block";
    await loadTtsMap();
    renderAudioStudio();
  });
  $("austudio-close").addEventListener("click", () => {
    auStopPlayback();
    $("austudio-overlay").style.display = "none";
  });

  $("austudio-body").addEventListener("click", async (ev) => {
    const row = ev.target.closest(".au-row");
    if (!row) return;
    const si = Number(row.dataset.si);
    const ci = Number(row.dataset.ci);
    const slide = slides[si];
    const clip = slide && slide.clips && slide.clips[ci];
    if (!clip) return;

    if (ev.target.classList.contains("au-play")) {
      const btn = ev.target;
      if (auPlayingId === clip.id) { auStopPlayback(); return; }
      auStopPlayback();
      auPlayer = new Audio(`/api/video/slide_audio/${encodeURIComponent(clip.id)}`);
      auPlayingId = clip.id;
      btn.textContent = "Stop";
      auPlayer.addEventListener("ended", auStopPlayback);
      auPlayer.play().catch(() => { auStopPlayback(); });
      return;
    }

    if (ev.target.classList.contains("au-regen")) {
      const text = row.querySelector(".au-script").value.trim();
      const status = row.querySelector(".au-status");
      if (!text) { status.textContent = "enter a script line first"; return; }
      const btn = ev.target;
      btn.disabled = true;
      status.textContent = "synthesizing...";
      try {
        const voice = ($("tmpl-voice") && $("tmpl-voice").value.trim()) || "";
        const r = await (await fetch("/api/video/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text, voice: voice }),
        })).json();
        if (r.error) throw new Error(r.error);
        if (r.id === clip.id) {
          status.textContent = "unchanged - same script maps to this same clip";
          btn.disabled = false;
          return;
        }
        const oldId = clip.id;
        auStopPlayback();
        slide.clips[ci] = { id: r.id, dur: Number(r.dur) || 0, cuts: [] };
        saveSlides();
        ttsMap[r.id] = { text: text, dur: Number(r.dur) || 0 };
        let purged = "";
        if (row.querySelector(".au-purge").checked) {
          const stillUsed = slides.some((s2) =>
            (s2.clips || []).some((c2) => c2.id === oldId));
          if (stillUsed) {
            purged = ", old clip kept (another slide uses it)";
          } else {
            try {
              await fetch(`/api/video/slide_audio/${encodeURIComponent(oldId)}?purge=1`,
                          { method: "DELETE" });
              purged = ", old clip deleted";
            } catch (e) { purged = ", old clip delete failed"; }
          }
        }
        renderAudioStudio(); renderAudioPanel(); renderSlideList();
        updateScrubMax(); renderPreview();
        const ns = $("austudio-body").querySelector(
          `.au-row[data-si="${si}"][data-ci="${ci}"] .au-status`);
        if (ns) ns.textContent = (r.cached ? "reused cached narration" : "replaced") +
          ` ${(Number(r.dur) || 0).toFixed(1)}s` + purged;
      } catch (e) {
        status.textContent = "failed: " + e.message;
        btn.disabled = false;
      }
    }
  });

  // ---- SCRIPT STUDIO modal -------------------------------------------
  $("open-script-studio").addEventListener("click", () => {
    $("studio-overlay").style.display = "block";
  });
  $("studio-close").addEventListener("click", () => {
    $("studio-overlay").style.display = "none";
  });
  document.querySelectorAll(".studio-tab").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll(".studio-tab").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      $("studio-script").style.display = b.dataset.stab === "script" ? "block" : "none";
      $("studio-template").style.display = b.dataset.stab === "template" ? "block" : "none";
    });
  });

  const DECK_TEMPLATE = {
    _doc: {
      what: "GENREG slide-deck template - build an entire narrated video from JSON",
      how_to_fill: [
        "1. Write one entry in `slides` per scene, in playback order. Only `script` is truly required; every other field has a sane default.",
        "2. `script` is what gets spoken (ElevenLabs) AND shown as the closed caption. Keep each slide to 1-3 sentences; long scripts auto-extend the slide because narration sets a minimum duration.",
        "3. Pick a pose: use a gesture label from pose_vocabulary (e.g. Pose-gesture-right) when you do not know the library filenames. The gesture also decides which side media goes on.",
        "4. Media you KNOW exists in the library: reference it in `media` (see below). Media you CANNOT attach: describe it in `media_request` instead - the human gets an on-slide upload prompt at the gesture position.",
        "5. Slides can hold MULTIPLE images/videos/gifs at once via the `media` list. If you add more than one, GIVE EACH ITEM ITS TIMES: `start` (seconds into the slide it appears) and `end` (seconds when it vanishes; 0 = stays to the end). Stagger them so items replace each other instead of stacking, e.g. item A {start: 0, end: 4}, item B {start: 4, end: 0}.",
        "6. Per media item you can also set `loop` (true = repeat while visible, good for short gifs), `fade_in` / `fade_out` (0.5s ramps so appearing/vanishing is not abrupt - recommended whenever you set an `end`), and `x, y, w, h` for placement on the 1280x720 stage.",
        "7. `duration` is a MINIMUM - narration audio and non-looping video runtimes extend the slide automatically, so set the media times to fit the script rather than stretching duration by hand.",
        "8. Use `transition: 'fade'` with `transition_dur` ~0.5 between scenes; `meta` is your scratch space and is ignored by the builder."
      ],
      pose_vocabulary: {
        about: "poses are labeled character images; use a gesture LABEL in the pose field when you do not know the library filenames - the builder matches it against the labeled library and positions everything",
        "Pose-gesture-right": "character gestures to the right - media goes on the RIGHT half (x~650), pose on the left",
        "Pose-gesture-left": "character gestures to the left - media goes LEFT (x~80), pose moves to the right",
        "Pose-gesture-up": "character points up - media goes TOP-CENTER",
        "Pose-explain": "talking pose, no directional gesture - media anywhere or none",
        "Pose-neutral": "neutral stance - typically caption-only slides"
      },
      slide_fields: {
        pose: "pose image filename from your poses library, OR a gesture label from pose_vocabulary above (blank = none)",
        pose_x: "x position on the 1280x720 stage", pose_y: "y position",
        media_request: "can't attach a file? DESCRIBE the chart/video this slide needs (e.g. 'bar chart of accuracy per module'). The builder shows it as an on-slide upload prompt for the human, positioned by the pose gesture. Blank = no media needed.",
        chart: "library image or video filename to embed (blank = none; legacy single-item form)",
        chart_x: "x", chart_y: "y", chart_w: "width", chart_h: "height",
        chart_start: "seconds into the slide the video starts", chart_loop: "true = loop the video",
        media: "list of items - a slide can hold SEVERAL images/videos/gifs: [{name, x, y, w, h, start, end, loop, fade_in, fade_out}]. When adding multiple, set each item's times: start = seconds into the slide it appears, end = seconds it vanishes (0 = until slide end). loop repeats a video/gif while visible; fade_in/fade_out add 0.5s ramps.",
        script: "what you (or ElevenLabs) say on this slide - also becomes the CC caption",
        duration: "minimum seconds on screen (audio/media can extend it)",
        transition: "none | fade", transition_dur: "crossfade seconds",
        meta: "free-form notes, ignored by the builder"
      }
    },
    slides: [
      { pose: "Pose-neutral", pose_x: 80, pose_y: 80,
        media_request: "",
        script: "Welcome - this deck was built from a template.",
        duration: 3.0, transition: "fade", transition_dur: 0.5,
        meta: "intro" },
      { pose: "Pose-gesture-right",
        media_request: "line chart of accuracy over training generations",
        script: "As you can see on the chart, accuracy climbs steadily.",
        duration: 4.0, transition: "fade", transition_dur: 0.5,
        meta: "the human uploads the chart when loading; it lands on the right because of the gesture" },
      { pose: "", pose_x: 80, pose_y: 80,
        chart: "", chart_x: 400, chart_y: 100, chart_w: 880, chart_h: 500,
        chart_start: 1.0, chart_loop: true,
        script: "Here the animation starts one second in and loops while I talk.",
        duration: 4.0, transition: "fade", transition_dur: 0.5,
        meta: "demo slide with a known library file" },
      { pose: "Pose-gesture-right",
        media: [
          { name: "intro_chart.png", x: 650, y: 80, w: 550, h: 420,
            start: 0, end: 4.0, fade_out: true },
          { name: "results_anim.mp4", x: 650, y: 80, w: 550, h: 420,
            start: 4.0, end: 0, loop: true, fade_in: true }
        ],
        script: "First the static chart, then at four seconds the looping animation takes its place.",
        duration: 8.0, transition: "fade", transition_dur: 0.5,
        meta: "multiple media items with times - the chart fades out at 4s and the video fades in to replace it" }
    ]
  };

  $("tmpl-download").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(DECK_TEMPLATE, null, 2)],
                         { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "deck_template.json";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  function slideFromTemplate(t) {
    // pose may be a gesture LABEL rather than a filename: resolve it
    // against the labeled poses library; keep the label either way so
    // the gesture keeps steering media placement
    const rawPose = String(t.pose || "");
    const resolved = resolvePoseLabel(rawPose);
    const gesture = gestureOf({ pose: rawPose });
    let px = Number(t.pose_x) || 0;
    let py = Number(t.pose_y) || 0;
    if (!px && !py && gesture) {
      const pp = gesturePosePos(gesture);
      px = pp.x; py = pp.y;
    }
    return sanitizeSlide({
      pose: resolved, pose_gesture: rawPose,
      media_request: String(t.media_request || t.chart_request || ""),
      pose_x: px, pose_y: py,
      media: Array.isArray(t.media) ? t.media : undefined,
      chart: t.chart || "", chart_x: Number(t.chart_x) || 0, chart_y: Number(t.chart_y) || 0,
      chart_w: Number(t.chart_w) || 550, chart_h: Number(t.chart_h) || 420,
      chart_start: Number(t.chart_start) || 0, chart_loop: !!t.chart_loop,
      text: String(t.script || t.text || ""),
      duration: Number(t.duration) || 3.0,
      transition: t.transition || "fade",
      transition_dur: Number(t.transition_dur) || 0.5,
      clips: [],
    });
  }

  $("tmpl-load").addEventListener("click", async () => {
    const status = $("tmpl-status");
    let doc;
    try {
      doc = JSON.parse($("tmpl-json").value);
    } catch (e) {
      status.textContent = "invalid JSON: " + e.message;
      return;
    }
    const list = Array.isArray(doc) ? doc : doc.slides;
    if (!Array.isArray(list) || !list.length) {
      status.textContent = "template needs a 'slides' array";
      return;
    }
    const built = list.map(slideFromTemplate);
    // media filenames the library does NOT have become upload prompts
    // instead of broken items - same flow as media_request
    let prompts = 0;
    const known = new Set([].concat(chartsList, videosList));
    built.forEach((sl) => {
      const missing = (sl.media || []).filter((m) => !known.has(m.name));
      if (missing.length) {
        sl.media = (sl.media || []).filter((m) => known.has(m.name));
        const desc = missing.map((m) => m.name).join(", ");
        sl.media_request = (sl.media_request ? sl.media_request + "; " : "") +
          "upload: " + desc;
      }
      if (sl.media_request) prompts += 1;
    });
    if ($("tmpl-replace").checked) slides = built;
    else slides = slides.concat(built);
    saveSlides();
    selectSlide(slides.length - built.length);
    built.forEach((sl) => { mediaItems(sl).forEach((m) => refreshMediaDur(sl, m)); });
    const promptNote = prompts
      ? ` - ${prompts} slide(s) NEED MEDIA (amber prompts on stage/cards)` : "";
    status.textContent = `built ${built.length} slide(s)` + promptNote;

    if ($("tmpl-tts").checked) {
      const voice = $("tmpl-voice").value.trim();
      let done = 0;
      for (const sl of built) {
        if (!sl.text || !sl.text.trim()) continue;
        status.textContent = `narrating slide ${done + 1}/${built.length}...`;
        try {
          const r = await (await fetch("/api/video/tts", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: sl.text, voice: voice }),
          })).json();
          if (r.error) throw new Error(r.error);
          if (r.cached) window.__ttsCacheHits = (window.__ttsCacheHits || 0) + 1;
          sl.clips = [{ id: r.id, dur: Number(r.dur) || 0, cuts: [] }];
          saveSlides();
        } catch (e) {
          status.textContent = "narration failed: " + e.message;
          renderSlideList(); renderAudioPanel(); updateScrubMax();
          return;
        }
        done += 1;
      }
      status.textContent = `deck built - ${done} slide(s) narrated` +
        (window.__ttsCacheHits ? ` (${window.__ttsCacheHits} from cache)` : "") +
        promptNote;
      renderSlideList(); renderAudioPanel(); updateScrubMax(); renderPreview();
    }
  });

  // Auto-split the entire script into slides.
  $("script-split").addEventListener("click", () => {
    const raw = $("script-text").value;
    if (!raw.trim()) { alert("Paste your script first."); return; }
    const mode = $("script-split-mode").value;
    let chunks = [];
    if (mode === "sentence") {
      chunks = raw.replace(/\s+/g, " ").match(/[^.!?]+[.!?]*/g) || [];
    } else if (mode === "line") {
      chunks = raw.split(/\n/);
    } else {
      chunks = raw.split(/\n\s*\n/);
    }
    chunks = chunks.map((c) => c.trim()).filter(Boolean);
    if (!chunks.length) { alert("Nothing to split."); return; }

    let ref;
    if ($("script-replace").checked) {
      slides = [];
      ref = null;
    } else {
      ref = slides[slides.length - 1];
    }
    chunks.forEach((c) => {
      slides.push(makeInheritedSlide(ref, c));
      ref = slides[slides.length - 1];
    });
    saveSlides();
    selectSlide(slides.length - chunks.length);
  });

  // Render video export job
  $("exp-render").addEventListener("click", async () => {
    if (!slides.length) {
      alert("Add slides first!");
      return;
    }
    const outName = $("exp-outname").value.trim() || "slideshow";
    const res = $("exp-res").value.split("x");
    const fps = Number($("exp-fps").value) || 24;
    const audio = $("exp-audio").value;

    // Attach audio track to first slide so renderer picks it up
    const configSlides = slides.map((s, idx) => ({
      ...s,
      audio: idx === 0 ? audio : ""
    }));

    const statusEl = $("exp-status");
    statusEl.innerHTML = "Submitting render job...";

    try {
      const resData = await (await fetch("/api/video/render_slides", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slides: configSlides,
          out_name: outName,
          fps: fps,
          bg: deckBg,
          w: Number(res[0]),
          h: Number(res[1])
        })
      })).json();

      if (resData.error) {
        statusEl.innerHTML = `<span style="color: #ff3333;">Error: ${resData.error}</span>`;
      } else {
        pollJob(resData.id);
      }
    } catch (e) {
      statusEl.innerHTML = `<span style="color: #ff3333;">Submission failed: ${e.message}</span>`;
    }
  });

  // Poll background job progress
  function pollJob(jobId) {
    const statusEl = $("exp-status");
    const interval = setInterval(async () => {
      try {
        const jobs = await (await fetch("/api/video/jobs")).json();
        const job = jobs.find(j => j.id === jobId);
        if (!job) {
          clearInterval(interval);
          statusEl.innerHTML = "Job lost.";
          return;
        }

        if (job.status === "done") {
          clearInterval(interval);
          statusEl.innerHTML = `<span style="color: #2e7d5b;">Render complete! Output saved: <b>${job.name}</b></span>`;
          // Trigger browser workspace agent alert
          fetch(`/api/video/library`)
            .then(r => r.json())
            .then(() => {
              // alert done
            });
        } else if (job.status === "error") {
          clearInterval(interval);
          statusEl.innerHTML = `<span style="color: #ff3333;">Render failed: ${job.message}</span>`;
        } else {
          const pct = Math.round(job.progress * 100);
          statusEl.innerHTML = `<div style="display:flex; justify-content:space-between; margin-bottom: 5px;">`
            + `<span>Encoding (${job.status})...</span>`
            + `<span class="ui-monospace">${pct}%</span></div>`
            + `<div style="background:#10141c; height: 6px; border-radius: 3px; border:1px solid #1c232c; overflow:hidden;">`
            + `<div style="background:#2e7d5b; height:100%; width: ${pct}%;"></div></div>`;
        }
      } catch (e) {
        console.error("Error polling job:", e);
      }
    }, 1000);
  }

  // Load everything on startup
  bindHoverPreview($("slide-pose-preview-mini"), () => $("slide-pose-preview-mini").src);
  bindHoverPreview($("slide-chart-preview-mini"), () => $("slide-chart-preview-mini").src);

  loadLibrary().then(() => {
    calculateDuration();
    if (slides.length) selectSlide(0);
    else renderPreview();
  });
})();
