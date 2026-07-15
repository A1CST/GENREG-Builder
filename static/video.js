// Video — the editor half of the animation studio. Library (upload/convert/
// delete), a clip timeline (per-clip in/out cut points, reorder, stitch-
// export), and a polled job list (/api/video/jobs). The Rigs/Scenes views
// live in animstudio.js; view switching + shared hooks are exposed here on
// window.VideoEditor.
(function () {
  // ── view switching (Rigs | Scenes | Editor) ─────────────────────────
  const viewTabs = document.querySelectorAll("#vd-views .side-tab");
  const viewPanels = document.querySelectorAll("[data-vpanel]");
  const viewMains = document.querySelectorAll("[data-vmain]");
  viewTabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      viewTabs.forEach((b) => b.classList.toggle("active", b === btn));
      viewPanels.forEach((p) => { p.hidden = p.dataset.vpanel !== btn.dataset.view; });
      viewMains.forEach((m) => { m.hidden = m.dataset.vmain !== btn.dataset.view; });
    });
  });

  const uploadEl = document.getElementById("vd-upload");
  const libStatusEl = document.getElementById("vd-lib-status");
  const libraryEl = document.getElementById("vd-library");
  const formatEl = document.getElementById("vd-format");
  const crfEl = document.getElementById("vd-crf");
  const scaleEl = document.getElementById("vd-scale");
  const fpsEl = document.getElementById("vd-fps");
  const outNameEl = document.getElementById("vd-outname");
  const playerEl = document.getElementById("vd-player");
  const playerMetaEl = document.getElementById("vd-player-meta");
  const playerPlaceholderEl = document.getElementById("vd-player-placeholder");
  const timelineEl = document.getElementById("vd-timeline");
  const timelineMetaEl = document.getElementById("vd-timeline-meta");
  const jobsEl = document.getElementById("vd-jobs");
  const jobsMetaEl = document.getElementById("vd-jobs-meta");
  const exportBtn = document.getElementById("vd-export");
  const clearBtn = document.getElementById("vd-clear");

  let library = [];               // server library entries
  let timeline = [];              // {name, start, end, duration, playable}
  let loadedName = null;          // file currently in the player
  let clipSeq = 0;

  // timeline undo/redo (Ctrl+Z / Ctrl+Y in the Editor view — dispatched
  // from animstudio.js's shared key handler)
  const tlHist = { undo: [], redo: [] };
  function pushTL() {
    tlHist.undo.push(JSON.stringify(timeline));
    if (tlHist.undo.length > 200) tlHist.undo.shift();
    tlHist.redo.length = 0;
  }
  function undoTimeline(dir) {
    const from = dir === "undo" ? tlHist.undo : tlHist.redo;
    const to = dir === "undo" ? tlHist.redo : tlHist.undo;
    if (!from.length) return;
    to.push(JSON.stringify(timeline));
    timeline = JSON.parse(from.pop());
    renderTimeline();
  }

  const fmtTime = (s) => {
    if (!isFinite(s)) return "?";
    const m = Math.floor(s / 60), sec = (s - m * 60).toFixed(1);
    return `${m}:${sec.padStart(4, "0")}`;
  };
  const fmtSize = (b) =>
    b > 1 << 30 ? (b / (1 << 30)).toFixed(2) + " GB"
      : b > 1 << 20 ? (b / (1 << 20)).toFixed(1) + " MB"
        : Math.round(b / 1024) + " KB";

  // ── status / formats ────────────────────────────────────────────────
  async function init() {
    try {
      const st = await (await fetch("/api/video/status")).json();
      if (!st.ok) {
        libStatusEl.textContent = "unavailable: " + (st.err || "ffmpeg missing");
        return;
      }
      st.formats.filter((f) => f !== "gif" || true).forEach((f) => {
        const opt = document.createElement("option");
        opt.value = f;
        opt.textContent = f;
        if (f === "mp4") opt.selected = true;
        formatEl.appendChild(opt);
      });
      await refreshLibrary();
      pollJobs();
    } catch (err) {
      libStatusEl.textContent = "error: " + err.message;
    }
  }

  // ── library ─────────────────────────────────────────────────────────
  async function refreshLibrary() {
    const resp = await fetch("/api/video/library");
    const data = await resp.json();
    if (!resp.ok) {
      libStatusEl.textContent = "error: " + (data.error || resp.status);
      return;
    }
    library = data;
    libStatusEl.textContent = `${library.length} file(s)`;
    renderLibrary();
    document.dispatchEvent(new CustomEvent("vd-library-changed", { detail: library }));
  }

  function libCard(f) {
    const card = document.createElement("div");
    card.className = "vd-libcard";

    if (!f.is_audio) {
      const thumb = document.createElement("img");
      thumb.className = "vd-thumb";
      thumb.loading = "lazy";
      thumb.src = "/api/video/thumb/" + encodeURIComponent(f.name);
      thumb.alt = f.name;
      thumb.onerror = () => { thumb.style.display = "none"; };
      card.appendChild(thumb);
    }

    const name = document.createElement("div");
    name.className = "vd-libname";
    name.textContent = f.name;
    name.title = f.name;
    card.appendChild(name);

    const v = f.video || {};
    const meta = document.createElement("div");
    meta.className = "vd-libmeta";
    meta.textContent = [
      fmtTime(f.duration), v.width ? `${v.width}x${v.height}` : null,
      v.codec, f.audio ? f.audio.codec : "no audio", fmtSize(f.size),
    ].filter(Boolean).join(" · ");
    card.appendChild(meta);

    const row = document.createElement("div");
    row.className = "vd-librow";
    const mk = (label, title, fn) => {
      const b = document.createElement("button");
      b.className = "runs-btn vd-mini";
      b.textContent = label;
      b.title = title;
      b.addEventListener("click", fn);
      row.appendChild(b);
      return b;
    };
    mk("Play", "load into the preview player", () => loadPlayer(f));
    if (!f.is_audio) {
      mk("+ Timeline", "append to the timeline", () => addToTimeline(f));
      mk("Convert", "convert to the format in the Export settings", () => convertFile(f));
    }
    const dl = document.createElement("a");
    dl.className = "runs-btn vd-mini";
    dl.textContent = "Save";
    dl.title = "download this file";
    dl.href = "/api/video/file/" + encodeURIComponent(f.name) + "?download=1";
    row.appendChild(dl);
    mk("Del", "delete from the library", () => deleteFile(f));
    card.appendChild(row);
    return card;
  }

  function renderLibrary() {
    libraryEl.innerHTML = "";
    if (!library.length) {
      const ph = document.createElement("div");
      ph.className = "im-placeholder";
      ph.textContent = "library is empty — upload a video above";
      libraryEl.appendChild(ph);
      return;
    }
    library.forEach((f) => libraryEl.appendChild(libCard(f)));
  }

  uploadEl.addEventListener("change", async () => {
    const files = Array.from(uploadEl.files || []);
    if (!files.length) return;
    for (let i = 0; i < files.length; i++) {
      libStatusEl.textContent = `uploading ${i + 1}/${files.length}: ${files[i].name}…`;
      const form = new FormData();
      form.append("file", files[i]);
      try {
        const resp = await fetch("/api/video/upload", { method: "POST", body: form });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || resp.status);
      } catch (err) {
        libStatusEl.textContent = "upload failed: " + err.message;
        uploadEl.value = "";
        return;
      }
    }
    uploadEl.value = "";
    await refreshLibrary();
  });

  async function deleteFile(f) {
    if (!confirm(`Delete ${f.name} from the library?`)) return;
    const resp = await fetch("/api/video/library/" + encodeURIComponent(f.name),
      { method: "DELETE" });
    if (resp.ok) {
      if (timeline.some((c) => c.name === f.name)) pushTL();
      timeline = timeline.filter((c) => c.name !== f.name);
      renderTimeline();
      await refreshLibrary();
    }
  }

  // ── player ──────────────────────────────────────────────────────────
  function loadPlayer(f, seekTo) {
    loadedName = f.name;
    if (f.playable) {
      playerEl.src = "/api/video/file/" + encodeURIComponent(f.name);
      playerEl.hidden = false;
      playerPlaceholderEl.hidden = true;
      if (seekTo != null) {
        playerEl.addEventListener("loadedmetadata", () => { playerEl.currentTime = seekTo; },
          { once: true });
      }
      playerEl.load();
      playerMetaEl.textContent = f.name;
    } else {
      playerEl.hidden = true;
      playerEl.removeAttribute("src");
      playerPlaceholderEl.hidden = false;
      playerPlaceholderEl.textContent =
        `${f.name}: the browser can't play this container — convert it to mp4/webm to preview (editing still works)`;
      playerMetaEl.textContent = f.name + " (no browser preview)";
    }
  }

  // ── timeline ────────────────────────────────────────────────────────
  function addToTimeline(f) {
    if (!f.video) {
      jobsMetaEl.textContent = f.name + " has no video stream — timeline takes video only";
      return;
    }
    pushTL();
    timeline.push({
      id: ++clipSeq, name: f.name, start: 0,
      end: Math.round(f.duration * 10) / 10 || 0,
      duration: f.duration, playable: f.playable,
    });
    renderTimeline();
  }

  function clipCard(clip, idx) {
    const card = document.createElement("div");
    card.className = "vd-clip";

    const head = document.createElement("div");
    head.className = "vd-clip-head";
    head.textContent = `${idx + 1}. ${clip.name}`;
    head.title = "click to preview this clip from its in-point";
    head.addEventListener("click", () => {
      const f = library.find((x) => x.name === clip.name);
      if (f) loadPlayer(f, clip.start);
    });
    card.appendChild(head);

    const mkPoint = (label, key) => {
      const wrap = document.createElement("div");
      wrap.className = "vd-clip-point";
      const lab = document.createElement("span");
      lab.textContent = label;
      const inp = document.createElement("input");
      inp.type = "number";
      inp.min = "0";
      inp.step = "0.1";
      inp.value = clip[key];
      inp.addEventListener("change", () => {
        pushTL();
        clip[key] = Math.max(0, Number(inp.value) || 0);
        updateTimelineMeta();
      });
      const set = document.createElement("button");
      set.className = "runs-btn vd-mini";
      set.textContent = "@ playhead";
      set.title = "set from the player's current time (load this clip's file first)";
      set.addEventListener("click", () => {
        if (loadedName !== clip.name || playerEl.hidden) return;
        pushTL();
        clip[key] = Math.round(playerEl.currentTime * 10) / 10;
        inp.value = clip[key];
        updateTimelineMeta();
      });
      wrap.appendChild(lab);
      wrap.appendChild(inp);
      wrap.appendChild(set);
      return wrap;
    };
    const points = document.createElement("div");
    points.className = "vd-clip-points";
    points.appendChild(mkPoint("in (s)", "start"));
    points.appendChild(mkPoint("out (s)", "end"));
    card.appendChild(points);

    const row = document.createElement("div");
    row.className = "vd-librow";
    const mk = (label, title, fn, disabled) => {
      const b = document.createElement("button");
      b.className = "runs-btn vd-mini";
      b.textContent = label;
      b.title = title;
      b.disabled = !!disabled;
      b.addEventListener("click", fn);
      row.appendChild(b);
    };
    mk("▲", "move earlier", () => { moveClip(idx, -1); }, idx === 0);
    mk("▼", "move later", () => { moveClip(idx, 1); }, idx === timeline.length - 1);
    mk("Cut to file", "save just this in/out range as a new library file", () => cutClip(clip));
    mk("Remove", "remove from the timeline", () => {
      pushTL();
      timeline.splice(idx, 1);
      renderTimeline();
    });
    card.appendChild(row);
    return card;
  }

  function moveClip(idx, delta) {
    const j = idx + delta;
    if (j < 0 || j >= timeline.length) return;
    pushTL();
    [timeline[idx], timeline[j]] = [timeline[j], timeline[idx]];
    renderTimeline();
  }

  function updateTimelineMeta() {
    const total = timeline.reduce((s, c) => s + Math.max(0, c.end - c.start), 0);
    timelineMetaEl.textContent = timeline.length
      ? `${timeline.length} clip(s) · ${fmtTime(total)} total`
      : "empty";
  }

  function renderTimeline() {
    timelineEl.innerHTML = "";
    if (!timeline.length) {
      const ph = document.createElement("div");
      ph.className = "im-placeholder";
      ph.textContent = "add clips from the library — each clip gets its own in/out cut points, then export stitches them in order";
      timelineEl.appendChild(ph);
    } else {
      timeline.forEach((c, i) => timelineEl.appendChild(clipCard(c, i)));
    }
    updateTimelineMeta();
  }

  clearBtn.addEventListener("click", () => {
    if (timeline.length) pushTL();
    timeline = [];
    renderTimeline();
  });

  // ── jobs ────────────────────────────────────────────────────────────
  const exportSettings = () => ({
    format: formatEl.value || "mp4",
    crf: Number(crfEl.value) || 23,
    scale_h: scaleEl.value === "" ? null : Number(scaleEl.value),
    fps: fpsEl.value === "" ? null : Number(fpsEl.value),
  });

  async function postJob(body, okMsg) {
    try {
      const resp = await fetch("/api/video/job", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.status);
      jobsMetaEl.textContent = okMsg;
      pollJobs(true);
    } catch (err) {
      jobsMetaEl.textContent = "error: " + err.message;
    }
  }

  function convertFile(f) {
    postJob({ op: "convert", name: f.name, ...exportSettings() },
      `converting ${f.name} -> ${formatEl.value}`);
  }

  function cutClip(clip) {
    if (clip.end - clip.start <= 0.01) {
      jobsMetaEl.textContent = "set the clip's out point after its in point first";
      return;
    }
    postJob({ op: "cut", name: clip.name, start: clip.start, end: clip.end,
      precise: true, crf: Number(crfEl.value) || 23 },
    `cutting ${clip.name}`);
  }

  exportBtn.addEventListener("click", () => {
    if (!timeline.length) {
      jobsMetaEl.textContent = "timeline is empty";
      return;
    }
    const bad = timeline.find((c) => c.end - c.start <= 0.01);
    if (bad) {
      jobsMetaEl.textContent = `fix cut points on ${bad.name} (out must be after in)`;
      return;
    }
    postJob({
      op: "stitch",
      clips: timeline.map((c) => ({ name: c.name, start: c.start, end: c.end })),
      out_name: outNameEl.value.trim(),
      ...exportSettings(),
    }, "export started");
  });

  // ── job polling ─────────────────────────────────────────────────────
  let pollTimer = null;
  let knownDone = new Set();

  function jobRow(j) {
    const row = document.createElement("div");
    row.className = "vd-job vd-job-" + j.status;
    const top = document.createElement("div");
    top.className = "vd-job-top";
    const label = document.createElement("span");
    label.textContent = `${j.op} · ${j.label}`;
    top.appendChild(label);
    const state = document.createElement("span");
    state.className = "vd-job-state";
    state.textContent = j.status === "running"
      ? Math.round(j.progress * 100) + "%" : j.status;
    top.appendChild(state);
    if (j.status === "running" || j.status === "queued") {
      const cancel = document.createElement("button");
      cancel.className = "runs-btn vd-mini";
      cancel.textContent = "Cancel";
      cancel.addEventListener("click", async () => {
        await fetch("/api/video/job/" + j.id + "/cancel", { method: "POST" });
        pollJobs(true);
      });
      top.appendChild(cancel);
    }
    if (j.status === "done") {
      const dl = document.createElement("a");
      dl.className = "runs-btn vd-mini";
      dl.textContent = "Save " + j.output;
      dl.href = "/api/video/file/" + encodeURIComponent(j.output) + "?download=1";
      top.appendChild(dl);
    }
    row.appendChild(top);
    const bar = document.createElement("div");
    bar.className = "vd-job-bar";
    const fill = document.createElement("div");
    fill.className = "vd-job-fill";
    fill.style.width = Math.round(j.progress * 100) + "%";
    bar.appendChild(fill);
    row.appendChild(bar);
    if (j.status === "error" && j.message) {
      const msg = document.createElement("pre");
      msg.className = "vd-job-msg";
      msg.textContent = j.message;
      row.appendChild(msg);
    }
    return row;
  }

  async function pollJobs(force) {
    if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
    let jobs = [];
    try {
      jobs = await (await fetch("/api/video/jobs")).json();
    } catch (err) { /* server briefly away — keep polling */ }
    jobsEl.innerHTML = "";
    if (!jobs.length) {
      const ph = document.createElement("div");
      ph.className = "im-placeholder";
      ph.textContent = "no jobs yet";
      jobsEl.appendChild(ph);
    } else {
      jobs.forEach((j) => jobsEl.appendChild(jobRow(j)));
    }
    // refresh the library when a job newly finishes (output lands there)
    const doneNow = jobs.filter((j) => j.status === "done").map((j) => j.id);
    if (doneNow.some((id) => !knownDone.has(id))) {
      knownDone = new Set(doneNow);
      refreshLibrary();
    }
    const active = jobs.some((j) => j.status === "running" || j.status === "queued");
    pollTimer = setTimeout(pollJobs, active ? 1000 : 5000);
  }

  init();

  // hooks for animstudio.js (render jobs land in the shared job list; the
  // audio-track picker needs the library)
  window.VideoEditor = {
    refreshLibrary,
    getLibrary: () => library,
    pollJobs: () => pollJobs(true),
    undoTimeline,
  };
})();
