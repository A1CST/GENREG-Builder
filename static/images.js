// Images — text-to-image (/api/images/generate) and reverse image/video ->
// prompt (/api/images/reverse).
(function () {
  // ── tabs ────────────────────────────────────────────────────────────
  const tabs = document.querySelectorAll("#im-tabs .side-tab");
  const panels = document.querySelectorAll("[data-panel]");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabs.forEach((b) => b.classList.toggle("active", b === btn));
      const key = btn.dataset.tab;
      panels.forEach((p) => { p.hidden = p.dataset.panel !== key; });
    });
  });

  // ── generate ────────────────────────────────────────────────────────
  const promptEl = document.getElementById("im-prompt");
  const negEl = document.getElementById("im-neg");
  const stepsEl = document.getElementById("im-steps");
  const guidanceEl = document.getElementById("im-guidance");
  const widthEl = document.getElementById("im-width");
  const heightEl = document.getElementById("im-height");
  const seedEl = document.getElementById("im-seed");
  const goBtn = document.getElementById("im-generate");
  const statusEl = document.getElementById("im-status");
  const metaEl = document.getElementById("im-meta");
  const imgEl = document.getElementById("im-output");
  const placeholderEl = document.getElementById("im-placeholder");

  async function generate() {
    const prompt = promptEl.value.trim();
    if (!prompt) {
      statusEl.textContent = "enter a prompt first";
      return;
    }
    goBtn.disabled = true;
    statusEl.textContent = "generating… (first run loads the checkpoint, can take a while)";
    metaEl.textContent = "";
    try {
      const resp = await fetch("/api/images/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          negative_prompt: negEl.value.trim(),
          steps: Number(stepsEl.value) || 25,
          guidance: Number(guidanceEl.value) || 7.5,
          width: Number(widthEl.value) || 512,
          height: Number(heightEl.value) || 512,
          seed: seedEl.value === "" ? null : Number(seedEl.value),
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      imgEl.src = "data:image/png;base64," + data.image_b64;
      imgEl.hidden = false;
      placeholderEl.hidden = true;
      metaEl.textContent =
        `${data.width}x${data.height} · ${data.steps} steps · guidance ${data.guidance} · ` +
        `${data.elapsed.toFixed(1)}s on ${data.device} · saved ${data.path}`;
      statusEl.textContent = "done";
    } catch (err) {
      statusEl.textContent = "error: " + err.message;
    } finally {
      goBtn.disabled = false;
    }
  }

  goBtn.addEventListener("click", generate);

  // ── reverse: image/video -> prompt ─────────────────────────────────
  const fileEl = document.getElementById("rev-file");
  const strideEl = document.getElementById("rev-stride");
  const maxFramesEl = document.getElementById("rev-maxframes");
  const maxSideEl = document.getElementById("rev-maxside");
  const tokensEl = document.getElementById("rev-tokens");
  const topkEl = document.getElementById("rev-topk");
  const runBtn = document.getElementById("rev-run");
  const revStatusEl = document.getElementById("rev-status");
  const revMetaEl = document.getElementById("rev-meta");
  const revGrid = document.getElementById("rev-grid");

  function frameCard(frame) {
    const card = document.createElement("div");
    card.className = "im-rev-card";
    const img = document.createElement("img");
    img.src = "/api/images/file/" + frame.image;
    img.alt = "frame " + frame.index;
    const label = document.createElement("div");
    label.className = "im-rev-idx";
    label.textContent = "#" + frame.index;
    const prompt = document.createElement("textarea");
    prompt.className = "im-rev-prompt";
    prompt.readOnly = true;
    prompt.rows = 4;
    prompt.value = frame.prompt;
    card.appendChild(label);
    card.appendChild(img);
    card.appendChild(prompt);
    return card;
  }

  let revJobCount = 0;

  function jobHeader(data) {
    const head = document.createElement("div");
    head.className = "im-rev-jobhead";
    head.textContent = `${data.kind} · ${data.frames.length} frame(s) · job ${data.job_id}`;
    return head;
  }

  async function runReverse() {
    const file = fileEl.files[0];
    if (!file) {
      revStatusEl.textContent = "choose a file first";
      return;
    }
    runBtn.disabled = true;
    revStatusEl.textContent = "processing… (loading BLIP/CLIP on first run can take a while)";
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("stride", strideEl.value || "1");
      form.append("max_frames", maxFramesEl.value || "30");
      form.append("max_side", maxSideEl.value || "768");
      form.append("max_new_tokens", tokensEl.value || "40");
      form.append("top_k", topkEl.value || "1");
      const resp = await fetch("/api/images/reverse", { method: "POST", body: form });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
      if (revJobCount === 0) revGrid.innerHTML = "";   // clear the "no results yet" placeholder
      revJobCount++;
      revGrid.appendChild(jobHeader(data));
      data.frames.forEach((f) => revGrid.appendChild(frameCard(f)));
      revMetaEl.textContent = `${revJobCount} job(s) processed this session`;
      revStatusEl.textContent = "done — saved under runs/images/reverse/" + data.job_id;
    } catch (err) {
      revStatusEl.textContent = "error: " + err.message;
    } finally {
      runBtn.disabled = false;
    }
  }

  runBtn.addEventListener("click", runReverse);
})();
