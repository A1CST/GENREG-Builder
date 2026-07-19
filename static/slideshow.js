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
  try {
    const saved = localStorage.getItem("genreg_slides");
    if (saved) {
      slides = JSON.parse(saved);
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
    } catch (e) {
      console.error("Error loading poses:", e);
    }

    try {
      const charts = await (await fetch("/api/charts")).json();
      chartsList = Array.isArray(charts) ? charts : [];
      populateDropdown($("slide-chart"), chartsList, "none");
      renderChartsLibrary();
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
    totalDuration = slides.reduce((acc, s) => acc + (Number(s.duration) || 3.0), 0);
    const scrub = $("player-scrub");
    scrub.max = totalDuration;
    updateTimeLabel();
  }

  function updateTimeLabel() {
    $("player-time-label").textContent = `${currentTime.toFixed(1)}s / ${totalDuration.toFixed(1)}s`;
  }

  // Render list of slides in sidebar
  function renderSlideList() {
    const container = $("slide-list");
    container.innerHTML = "";
    slides.forEach((slide, idx) => {
      const item = document.createElement("div");
      item.className = "slide-item" + (idx === activeIndex ? " active" : "");
      
      const label = document.createElement("div");
      label.style.fontWeight = "bold";
      label.style.fontSize = "12px";
      label.textContent = `#${idx + 1} (${slide.duration.toFixed(1)}s) — ${slide.pose ? "me" : "blank"}`;
      
      const previewText = document.createElement("div");
      previewText.style.fontSize = "11px";
      previewText.style.color = "#8b95a1";
      previewText.style.textOverflow = "ellipsis";
      previewText.style.overflow = "hidden";
      previewText.style.whiteSpace = "nowrap";
      previewText.style.width = "180px";
      previewText.textContent = slide.text || "(no text)";

      const leftCol = document.createElement("div");
      leftCol.appendChild(label);
      leftCol.appendChild(previewText);

      const delBtn = document.createElement("button");
      delBtn.className = "runs-btn vd-mini";
      delBtn.textContent = "Del";
      delBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        slides.splice(idx, 1);
        if (activeIndex >= slides.length) {
          activeIndex = slides.length - 1;
        }
        saveSlides();
        if (activeIndex >= 0) selectSlide(activeIndex);
        else {
          $("slide-editor").style.display = "none";
          renderPreview();
        }
      });

      item.appendChild(leftCol);
      item.appendChild(delBtn);
      item.addEventListener("click", () => selectSlide(idx));
      container.appendChild(item);
    });
  }

  function selectSlide(idx) {
    activeIndex = idx;
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
      chartMini.src = `/api/video/file/${encodeURIComponent(slide.chart)}`;
      chartCont.style.display = "block";
    } else {
      chartCont.style.display = "none";
    }

    // Jump time to start of this slide
    let start = 0;
    for (let i = 0; i < idx; i++) {
      start += slides[i].duration;
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
        chartMini.src = `/api/video/file/${encodeURIComponent(e.target.value)}`;
        chartCont.style.display = "block";
      } else {
        chartCont.style.display = "none";
      }
    }
  });
  $("slide-chart-align").addEventListener("change", (e) => {
    if (activeIndex >= 0) { slides[activeIndex].chart_align = e.target.value; saveSlides(); renderPreview(); }
  });
  $("slide-text").addEventListener("input", (e) => {
    if (activeIndex >= 0) { slides[activeIndex].text = e.target.value; saveSlides(); renderPreview(); }
  });
  $("slide-duration").addEventListener("input", (e) => {
    if (activeIndex >= 0) { slides[activeIndex].duration = Math.max(0.5, Number(e.target.value) || 3.0); saveSlides(); renderPreview(); }
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
      const dur = s.duration;
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
      out.push(`<image href="/api/video/file/${encodeURIComponent(assetName)}" x="${chx}" y="${chy}" width="${cw}" height="${ch}" preserveAspectRatio="xMidYMid meet"/>`);
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
        out.push(`<image href="/api/video/file/${encodeURIComponent(s.chart)}" x="${cx}" y="${cy}" width="${cw}" height="${ch}" data-drag="chart" style="cursor: move;" preserveAspectRatio="xMidYMid meet"/>`);
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
    
    function tick(now) {
      if (!isPlaying) return;
      const elapsed = (now - playStartRealTime) / 1000;
      currentTime = playStartScrubTime + elapsed;
      
      if (currentTime >= totalDuration) {
        currentTime = 0;
        playStartRealTime = now;
        playStartScrubTime = 0;
      }
      
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
      chartMini.src = `/api/video/file/${encodeURIComponent(tempGhosts.chart)}`;
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
