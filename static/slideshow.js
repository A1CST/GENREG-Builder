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

  function mediaFloor(slide) {
    // a non-looping video floors the slide at start + runtime; a looping
    // one fills whatever the slide gives it (no floor)
    const d = Number(slide.chart_dur) || 0;
    if (!d || slide.chart_loop) return 0;
    return (Number(slide.chart_start) || 0) + d;
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

  async function refreshChartDur(slide) {
    if (!slide.chart) { slide.chart_dur = 0; return; }
    let d = 0;
    try {
      const r = await (await fetch(
        `/api/video/meta?name=${encodeURIComponent(slide.chart)}`)).json();
      d = Number(r.duration) || 0;
    } catch (e) { d = 0; }
    if (!d) d = await probeDurClient(slide.chart);
    slide.chart_dur = Math.round(d * 10) / 10;
    saveSlides(); renderSlideList(); updateScrubMax(); renderPreview();
    renderMediaTimeline();
  }

  // heal existing decks whose videos were assigned before durations worked
  slides.forEach((sl) => {
    if (sl.chart && !sl.chart_dur && /\.(mp4|webm|mov|gif)$/i.test(sl.chart)) {
      refreshChartDur(sl);
    }
  });

  function sanitizeSlide(s) {
    // legacy decks stored numbers as strings and lack newer fields; one
    // bad slide must never crash the manager
    s = s || {};
    return {
      pose: s.pose || "", pose_align: s.pose_align || "left",
      pose_x: Number(s.pose_x) || 0, pose_y: Number(s.pose_y) || 0,
      chart: s.chart || "", chart_align: s.chart_align || "right",
      chart_dur: Number(s.chart_dur) || 0,
      chart_start: Math.max(0, Number(s.chart_start) || 0),
      chart_loop: !!s.chart_loop,
      chart_x: Number(s.chart_x) || 0, chart_y: Number(s.chart_y) || 0,
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
        chart: "",
        chart_align: "right",
        text: "Welcome to the evolutionary gradient-free explainer!",
        duration: 3.0,
        transition: "fade",
        transition_dur: 0.5
      }
    ];
  }

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
        slides[activeIndex].chart = name;
        $("slide-chart").value = name;
        refreshChartDur(slides[activeIndex]);
        saveSlides(); renderPreview(); renderSlideList();
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
          slides[activeIndex].chart = chartName;
          refreshChartDur(slides[activeIndex]);
          $("slide-chart").value = chartName;
          saveSlides();
          renderPreview();
          // Update mini preview
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
          slides[activeIndex].chart = res.name;
          $("slide-chart").value = res.name;
          saveSlides();
          renderPreview();
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
    const chartDot = slide.chart ? '<span class="sld-dot" title="has chart"></span>' : "";
    const audDot = (slide.clips && slide.clips.length)
      ? `<span class="sld-dot" style="background:#7ee787; right: 12px;" title="${slide.clips.length} audio clip(s)"></span>`
      : "";
    const cap = (slide.text || "").slice(0, 60);
    card.innerHTML =
      `<div class="sld-thumb">${poseImg}` +
      `<span class="sld-num">${idx + 1}</span>` +
      `<span class="sld-dur">${effDur(slide).toFixed(1)}s</span>${chartDot}${audDot}</div>` +
      `<div class="sld-cap">${cap ? "" : '<span class="sld-empty">no caption</span>'}</div>` +
      `<div class="sld-acts">` +
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
    $("slide-pose-align").value = slide.pose_align || "left";
    $("slide-chart").value = slide.chart || "";
    $("slide-chart-align").value = slide.chart_align || "right";
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
    if (slide.chart) {
      chartMini.src = chartHref(slide.chart);
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
  $("slide-pose-align").addEventListener("change", (e) => {
    if (activeIndex >= 0) { slides[activeIndex].pose_align = e.target.value; saveSlides(); renderPreview(); }
  });
  $("slide-chart").addEventListener("change", (e) => {
    if (activeIndex >= 0) {
      slides[activeIndex].chart = e.target.value;
      saveSlides();
      renderPreview();
      // Update mini preview
      const chartMini = $("slide-chart-preview-mini");
      const chartCont = $("slide-chart-preview-container");
      if (e.target.value) {
        chartMini.src = chartHref(e.target.value);
        chartCont.style.display = "block";
      } else {
        chartCont.style.display = "none";
      }
    }
  });
  $("slide-chart").addEventListener("change", () => {
    if (activeIndex >= 0) refreshChartDur(slides[activeIndex]);
  });
  $("slide-chart-align").addEventListener("change", (e) => {
    if (activeIndex >= 0) { slides[activeIndex].chart_align = e.target.value; saveSlides(); renderPreview(); }
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
      chart: r.chart || "",
      chart_align: r.chart_align || "right",
      chart_x: r.chart_x,
      chart_y: r.chart_y,
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
      sl.chart = s.chart;
      sl.chart_align = s.chart_align;
      sl.chart_x = s.chart_x;
      sl.chart_y = s.chart_y;
      sl.chart_dur = s.chart_dur;
      sl.chart_start = s.chart_start;
      sl.chart_loop = s.chart_loop;
    });
    saveSlides();
    renderPreview();
  });

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
    stage.style.background = bg;

    const r = ranges[activeIdx];
    if (!r) return;

    // Check transition overlap
    if (r.transDur > 0 && (r.end - currentTime) < r.transDur && activeIdx < ranges.length - 1) {
      const nextR = ranges[activeIdx + 1];
      const alpha = (currentTime - (r.end - r.transDur)) / r.transDur;
      
      const g1 = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g1.setAttribute("opacity", 1.0 - alpha);
      g1.innerHTML = makeSlideGroup(r.slide);
      
      const g2 = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g2.setAttribute("opacity", alpha);
      g2.innerHTML = makeSlideGroup(nextR.slide);
      
      stage.appendChild(g1);
      stage.appendChild(g2);
    } else {
      stage.innerHTML = makeSlideGroup(r.slide);
      
      if (tempGhosts && activeIndex === ghostIndex) {
        if (!r.slide.pose && tempGhosts.pose) {
          stage.appendChild(makeGhostGroup(tempGhosts.pose, tempGhosts.pose_align, "keep-pose"));
        }
        if (!r.slide.chart && tempGhosts.chart) {
          stage.appendChild(makeGhostGroup(tempGhosts.chart, tempGhosts.chart_align, "keep-chart"));
        }
      }
    }
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
      if (prevSlide.chart_x !== undefined && prevSlide.chart_y !== undefined) {
        chx = prevSlide.chart_x;
        chy = prevSlide.chart_y;
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

  function makeSlideGroup(s) {
    let out = [];
    const w = 1280, h = 720;
    
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

    // Chart
    if (s.chart) {
      const align = s.chart_align || "right";
      let cx = 650, cy = 80;
      const cw = 550, ch = 420;
      if (s.chart_x !== undefined && s.chart_y !== undefined) {
        cx = s.chart_x;
        cy = s.chart_y;
      } else {
        if (align === "left") cx = 80;
        else if (align === "center") cx = (w - cw) / 2;
      }
      
      if (align !== "none" || s.chart_x !== undefined) {
        out.push(`<image href="${chartHref(s.chart)}" x="${cx}" y="${cy}" width="${cw}" height="${ch}" data-drag="chart" style="cursor: move;" preserveAspectRatio="xMidYMid meet"/>`);
      }
    }

    // CC Text
    if (s.text) {
      const esc = (txt) => txt.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const lines = String(s.text).split("\n");
      const y0 = 630 - (lines.length - 1) * 14;
      const ts = lines.map((ln, i) => `<tspan x="640" dy="${i ? 26 * 1.35 : 0}">${esc(ln)}</tspan>`).join("");
      
      out.push(`<rect x="100" y="570" width="1080" height="110" rx="8" fill="#10141c" fill-opacity="0.85" stroke="#1c232c" stroke-width="1"/>`);
      out.push(`<text x="640" y="${y0}" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="bold" fill="#f0ede4" text-anchor="middle">${ts}</text>`);
    }

    return out.join("");
  }

  // Player controls
  // live video overlay: the stage SVG shows a static thumb for video
  // charts; during preview an HTML <video> is positioned over the stage
  // at the chart's exact box and kept in sync with the deck clock
  let stageVideo = null;

  function ensureStageVideo() {
    if (stageVideo) return stageVideo;
    const wrap = $("slides-stage").parentElement;
    stageVideo = document.createElement("video");
    stageVideo.muted = true;                 // narration owns the audio
    stageVideo.playsInline = true;
    stageVideo.style.position = "absolute";
    stageVideo.style.display = "none";
    stageVideo.style.objectFit = "contain";
    stageVideo.style.pointerEvents = "none";
    wrap.appendChild(stageVideo);
    return stageVideo;
  }

  function hideStageVideo() {
    if (stageVideo) {
      stageVideo.pause();
      stageVideo.style.display = "none";
      stageVideo.dataset.src = "";
      stageVideo.removeAttribute("src");
    }
  }

  function syncStageVideo(t) {
    // t = deck time; find the active slide and place/seek the overlay
    let curr = 0;
    let slide = null;
    let localT = 0;
    for (const s of slides) {
      const d = effDur(s);
      if (t < curr + d) { slide = s; localT = t - curr; break; }
      curr += d;
    }
    if (!slide || !slide.chart || !/\.(mp4|webm|mov)$/i.test(slide.chart)
        || !(Number(slide.chart_dur) > 0)) { hideStageVideo(); return; }
    const start = Number(slide.chart_start) || 0;
    const d = Number(slide.chart_dur);
    let mt = localT - start;
    if (mt < 0) { hideStageVideo(); return; }        // poster shows in SVG
    if (slide.chart_loop) mt = mt % d;
    else if (mt >= d) mt = d - 0.05;                  // hold last frame
    const v = ensureStageVideo();
    const url = `/api/video/file/${encodeURIComponent(slide.chart)}`;
    if (v.dataset.src !== url) {
      v.dataset.src = url;
      v.src = url;
      v.loop = !!slide.chart_loop;
    }
    // position: map viewBox (1280x720) coords to the stage's CSS box
    const stage = $("slides-stage");
    const wrap = stage.parentElement;
    const sr = stage.getBoundingClientRect();
    const wr = wrap.getBoundingClientRect();
    const scale = Math.min(sr.width / 1280, sr.height / 720);
    const ox = (sr.width - 1280 * scale) / 2 + (sr.left - wr.left);
    const oy = (sr.height - 720 * scale) / 2 + (sr.top - wr.top);
    const cw = 550, ch = 420;
    let cx = 650, cy = 80;
    if (slide.chart_x !== undefined && slide.chart_x !== null && slide.chart_x !== 0) cx = Number(slide.chart_x);
    if (slide.chart_y !== undefined && slide.chart_y !== null && slide.chart_y !== 0) cy = Number(slide.chart_y);
    else {
      const align = slide.chart_align || "right";
      if (align === "left") cx = 80;
      else if (align === "center") cx = (1280 - cw) / 2;
    }
    v.style.left = (ox + cx * scale) + "px";
    v.style.top = (oy + cy * scale) + "px";
    v.style.width = (cw * scale) + "px";
    v.style.height = (ch * scale) + "px";
    v.style.display = "block";
    if (Math.abs((v.currentTime || 0) - mt) > 0.3) v.currentTime = mt;
    if (isPlaying && v.paused) v.play().catch(() => {});
    if (!isPlaying && !v.paused) v.pause();
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
    if (stageVideo && !stageVideo.paused) stageVideo.pause();
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
      $("slide-pose-align").value = tempGhosts.pose_align;
      
      const poseMini = $("slide-pose-preview-mini");
      const poseCont = $("slide-pose-preview-container");
      poseMini.src = `/api/poses/${encodeURIComponent(tempGhosts.pose)}`;
      poseCont.style.display = "block";
      
      tempGhosts.pose = "";
      saveSlides();
      renderPreview();
    } else if (action === "keep-chart") {
      slides[activeIndex].chart = tempGhosts.chart;
      slides[activeIndex].chart_align = tempGhosts.chart_align;
      slides[activeIndex].chart_x = tempGhosts.chart_x;
      slides[activeIndex].chart_y = tempGhosts.chart_y;
      $("slide-chart").value = tempGhosts.chart;
      $("slide-chart-align").value = tempGhosts.chart_align;
      
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
    } else if (dragItem === "chart") {
      let curX = slide.chart_x;
      let curY = slide.chart_y;
      if (curX === undefined || curY === undefined) {
        const align = slide.chart_align || "right";
        curX = 650; curY = 80;
        if (align === "left") curX = 80;
        else if (align === "center") curX = 365;
      }
      itemStartPos = { x: curX, y: curY };
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
      } else if (dragItem === "chart") {
        slideToEdit.chart_x = itemStartPos.x + dx;
        slideToEdit.chart_y = itemStartPos.y + dy;
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

  // ---- MEDIA TIMELINE: when the slide's video/gif starts + loop ------
  function renderMediaTimeline() {
    const panel = $("media-tl");
    if (!panel) return;
    if (activeIndex < 0 || !slides[activeIndex]) { panel.style.display = "none"; return; }
    const slide = slides[activeIndex];
    const d = Number(slide.chart_dur) || 0;
    if (!slide.chart || !d) { panel.style.display = "none"; return; }
    panel.style.display = "block";
    const total = effDur(slide);
    const start = Math.min(Number(slide.chart_start) || 0, Math.max(0, total - 0.1));
    const track = $("media-track");
    const info = $("media-info");
    $("media-loop").checked = !!slide.chart_loop;
    track.innerHTML = "";
    // play window span(s)
    const mk = (a, b, cls) => {
      const sp = document.createElement("div");
      sp.className = cls;
      sp.style.left = (a / total * 100) + "%";
      sp.style.width = (Math.max(0, Math.min(b, total) - a) / total * 100) + "%";
      track.appendChild(sp);
    };
    if (slide.chart_loop) {
      for (let t0 = start, k = 0; t0 < total && k < 40; t0 += d, k++) {
        mk(t0, t0 + d, "media-span" + (k > 0 ? " media-span-loop" : ""));
      }
    } else {
      mk(start, start + d, "media-span");
    }
    // draggable start handle
    const hd = document.createElement("div");
    hd.className = "media-handle";
    hd.style.left = (start / total * 100) + "%";
    hd.title = "drag: when the media starts";
    track.appendChild(hd);
    info.textContent = `${slide.chart} - starts at ${start.toFixed(1)}s` +
      (slide.chart_loop ? `, loops every ${d.toFixed(1)}s`
                        : `, plays ${d.toFixed(1)}s (ends ${(start + d).toFixed(1)}s)`);
    let dragging = false;
    const move = (ev) => {
      if (!dragging) return;
      const r = track.getBoundingClientRect();
      const t = Math.max(0, Math.min(total - 0.1,
        (ev.clientX - r.left) / r.width * total));
      slide.chart_start = Math.round(t * 10) / 10;
      saveSlides();
      renderMediaTimeline();
      renderSlideList(); updateScrubMax();
    };
    hd.addEventListener("pointerdown", (ev) => {
      dragging = true;
      hd.setPointerCapture(ev.pointerId);
    });
    hd.addEventListener("pointermove", move);
    hd.addEventListener("pointerup", () => { dragging = false; });
    track.addEventListener("pointerdown", (ev) => {
      if (ev.target !== track) return;
      const r = track.getBoundingClientRect();
      slide.chart_start = Math.round(Math.max(0, Math.min(total - 0.1,
        (ev.clientX - r.left) / r.width * total)) * 10) / 10;
      saveSlides(); renderMediaTimeline(); renderSlideList(); updateScrubMax();
    });
  }

  $("media-loop").addEventListener("change", (e) => {
    if (activeIndex < 0) return;
    slides[activeIndex].chart_loop = e.target.checked;
    saveSlides(); renderMediaTimeline(); renderSlideList(); updateScrubMax();
    renderPreview();
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
