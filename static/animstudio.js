// Animation studio — Rigs view (manual part editor + procedural generate)
// and Scenes view (actors, verb actions, overlays, live preview, render).
// Preview math comes from animrig.js, which mirrors anim_service.py exactly.
(function () {
  const $ = (id) => document.getElementById(id);
  const INK = "#20242a";

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  // labelled numeric/text/select input helpers for dynamic rows
  function mkNum(label, value, onchange, step) {
    const wrap = el("label", "vd-f");
    wrap.appendChild(el("span", null, label));
    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = step || "1";
    inp.value = value;
    inp.addEventListener("change", () => onchange(Number(inp.value) || 0));
    wrap.appendChild(inp);
    return wrap;
  }
  function mkText(label, value, onchange) {
    const wrap = el("label", "vd-f");
    wrap.appendChild(el("span", null, label));
    const inp = document.createElement("input");
    inp.type = "text";
    inp.value = value;
    inp.addEventListener("change", () => onchange(inp.value));
    wrap.appendChild(inp);
    return wrap;
  }
  function mkColor(label, value, onchange) {
    const wrap = el("label", "vd-f");
    wrap.appendChild(el("span", null, label));
    const inp = document.createElement("input");
    inp.type = "color";
    try { inp.value = value; } catch (e) { inp.value = "#888888"; }
    inp.addEventListener("input", () => onchange(inp.value));
    wrap.appendChild(inp);
    return wrap;
  }
  function mkSelect(label, options, value, onchange) {
    const wrap = el("label", "vd-f");
    wrap.appendChild(el("span", null, label));
    const sel = document.createElement("select");
    options.forEach(([v, lab]) => {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = lab;
      if (v === value) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener("change", () => onchange(sel.value));
    wrap.appendChild(sel);
    return wrap;
  }
  function mkCheck(label, value, onchange) {
    const wrap = el("label", "vd-f");
    wrap.appendChild(el("span", null, label));
    const inp = document.createElement("input");
    inp.type = "checkbox";
    inp.checked = !!value;
    inp.addEventListener("change", () => onchange(inp.checked));
    wrap.appendChild(inp);
    return wrap;
  }

  // svg pointer -> user coordinates
  function svgPoint(svg, ev) {
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX;
    pt.y = ev.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }

  let TAGS = ["body", "head", "arm_l", "arm_r", "leg_l", "leg_r",
    "mouth_closed", "mouth_half", "mouth_open", "other"];

  // ── undo / redo (Ctrl+Z / Ctrl+Y) ───────────────────────────────────
  // Snapshot stacks per document kind; every mutating callback pushes the
  // pre-change state. Opening another rig/scene clears its stack.
  const hist = { rig: { undo: [], redo: [] }, scn: { undo: [], redo: [] } };
  function pushHist(kind, snapshot) {
    const h = hist[kind];
    h.undo.push(snapshot);
    if (h.undo.length > 200) h.undo.shift();
    h.redo.length = 0;
  }
  const pushRig = () => { if (rig) pushHist("rig", JSON.stringify(rig)); };
  const pushScn = () => { if (scene) pushHist("scn", JSON.stringify(scene)); };
  const clearHist = (kind) => { hist[kind].undo.length = 0; hist[kind].redo.length = 0; };

  function undoRedo(kind, dir) {
    const h = hist[kind];
    const from = dir === "undo" ? h.undo : h.redo;
    const to = dir === "undo" ? h.redo : h.undo;
    const cur = kind === "rig" ? rig : scene;
    if (!from.length || !cur) return;
    to.push(JSON.stringify(cur));
    const snap = JSON.parse(from.pop());
    if (kind === "rig") {
      rig = snap;
      if (selPart && !rig.parts.some((p) => p.id === selPart)) selPart = null;
      $("rig-title").textContent = `${rig.name} (${rig.kind})`;
      renderRigList();
      renderPartList();
      renderPartForm();
      drawRig(0);
    } else {
      scene = snap;
      syncSceneInputs();
      renderSceneList();
      renderActors();
      renderActions();
      renderOverlays();
      drawScene();
    }
  }

  document.addEventListener("keydown", (ev) => {
    if (!(ev.ctrlKey || ev.metaKey)) return;
    const k = ev.key.toLowerCase();
    if (k !== "z" && k !== "y") return;
    const t = ev.target;
    // leave native text-editing undo alone while typing in a field
    if (t && (t.tagName === "TEXTAREA"
      || (t.tagName === "INPUT" && (t.type === "text" || t.type === "number")))) return;
    const active = document.querySelector("#vd-views .side-tab.active");
    const view = active ? active.dataset.view : "";
    const dir = (k === "y" || (k === "z" && ev.shiftKey)) ? "redo" : "undo";
    if (view === "rigs") undoRedo("rig", dir);
    else if (view === "scenes") undoRedo("scn", dir);
    else if (view === "edit" && window.VideoEditor && window.VideoEditor.undoTimeline) {
      window.VideoEditor.undoTimeline(dir);
    } else return;
    ev.preventDefault();
  });

  // ════════════════════════ RIGS ════════════════════════
  let rigs = [];                    // full rig docs from the server
  let rig = null;                   // the open (working-copy) rig
  let selPart = null;               // selected part id
  let verbTest = "";                // active test verb
  let rigAnim = null;               // rAF handle

  const rigStage = $("rig-stage");
  const rigStatus = $("rig-status");

  async function loadRigs(openName) {
    const resp = await fetch("/api/anim/rigs");
    rigs = resp.ok ? await resp.json() : [];
    rigStatus.textContent = `${rigs.length} rig(s)`;
    renderRigList();
    fillSceneRigSelect();
    if (openName) {
      const r = rigs.find((x) => x.name === openName);
      if (r) openRig(r);
    } else if (!rig && rigs.length) {
      openRig(rigs[0]);
    }
    drawScene();
  }

  function renderRigList() {
    const list = $("rig-list");
    list.innerHTML = "";
    if (!rigs.length) {
      list.appendChild(el("div", "im-placeholder", "no rigs yet — generate one"));
      return;
    }
    rigs.forEach((r) => {
      const row = el("div", "vd-docrow" + (rig && rig.name === r.name ? " sel" : ""));
      row.appendChild(el("span", "vd-docname", r.name));
      row.appendChild(el("span", "vd-docsub", `${r.kind} · ${r.parts.length} parts`));
      row.addEventListener("click", () => openRig(r));
      list.appendChild(row);
    });
  }

  function openRig(r) {
    rig = JSON.parse(JSON.stringify(r));
    delete rig._mtime;
    clearHist("rig");
    selPart = null;
    verbTest = "";
    $("rig-title").textContent = `${rig.name} (${rig.kind})`;
    renderRigList();
    renderPartList();
    renderPartForm();
    drawRig(0);
  }

  function rigViewBox() {
    const c = rig.canvas || { w: 300, h: 300 };
    const w = Math.max(120, c.w), h = Math.max(120, c.h);
    return [-w / 2, -h, w, h + 40];
  }

  // static world position of a part's pivot (edit mode: rotations ignored)
  function pivotWorld(pid) {
    const byId = {};
    rig.parts.forEach((p) => { byId[p.id] = p; });
    let p = byId[pid];
    if (!p) return null;
    let x = (p.pivot || [0, 0])[0], y = (p.pivot || [0, 0])[1];
    while (p) {
      x += (p.offset || [0, 0])[0];
      y += (p.offset || [0, 0])[1];
      p = byId[p.parent];
    }
    return [x, y];
  }

  function testPose(t) {
    if (!verbTest) return AnimRig.actorState({ id: "t", x: 0, y: 0 }, [], t,
      rig.kind === "character");
    const loop = 2.5;
    const tt = t % loop;
    const action = { actor: "t", verb: verbTest, t0: 0, t1: loop, args: {} };
    if (verbTest.startsWith("walk")) {
      action.args = { to_x: verbTest.includes("right") ? 100 : -100 };
      if (verbTest.includes("stairs")) action.args.to_y = -60;
    }
    if (verbTest === "point" || verbTest === "present") action.args = { arm: "r", angle: -45 };
    if (verbTest === "face") action.args = { front: 1 };
    return AnimRig.actorState({ id: "t", x: 0, y: 0 }, [action], tt,
      rig.kind === "character");
  }

  function drawRig(t) {
    if (!rig) { rigStage.innerHTML = ""; return; }
    const [vx, vy, vw, vh] = rigViewBox();
    rigStage.setAttribute("viewBox", `${vx} ${vy} ${vw} ${vh}`);
    const st = testPose(t || 0);
    let extra = `<line x1="${vx}" y1="0" x2="${vx + vw}" y2="0" stroke="#4a5162" stroke-width="1" stroke-dasharray="4 4"/>`
      + `<line x1="0" y1="${vy}" x2="0" y2="${vy + vh}" stroke="#3a4152" stroke-width="0.5" stroke-dasharray="2 6"/>`;
    let marker = "";
    if (selPart && !verbTest) {
      const pw = pivotWorld(selPart);
      const p = rig.parts.find((x) => x.id === selPart);
      if (pw && p) {
        const R = 46;
        const aR = ((p.rot || 0) - 90) * Math.PI / 180;   // rotate handle: "up"
        const aS = (p.rot || 0) * Math.PI / 180;          // scale handle: "right"
        const rx = pw[0] + Math.cos(aR) * R, ry = pw[1] + Math.sin(aR) * R;
        const sx = pw[0] + Math.cos(aS) * R, sy = pw[1] + Math.sin(aS) * R;
        marker = `<g stroke="#e4b13c" stroke-width="1.5" fill="none">`
          + `<circle cx="${pw[0]}" cy="${pw[1]}" r="5"/>`
          + `<line x1="${pw[0] - 9}" y1="${pw[1]}" x2="${pw[0] + 9}" y2="${pw[1]}"/>`
          + `<line x1="${pw[0]}" y1="${pw[1] - 9}" x2="${pw[0]}" y2="${pw[1] + 9}"/>`
          + `<line x1="${pw[0]}" y1="${pw[1]}" x2="${rx}" y2="${ry}" stroke-dasharray="3 3"/>`
          + `<line x1="${pw[0]}" y1="${pw[1]}" x2="${sx}" y2="${sy}" stroke-dasharray="3 3"/></g>`
          + `<circle data-handle="rotate" class="vd-handle" cx="${rx}" cy="${ry}" r="7" `
          + `fill="#e4b13c" stroke="#20242a" stroke-width="1.5"/>`
          + `<rect data-handle="scale" class="vd-handle" x="${sx - 6}" y="${sy - 6}" `
          + `width="12" height="12" rx="2" fill="#6fb3e0" stroke="#20242a" stroke-width="1.5"/>`;
      }
    }
    rigStage.innerHTML = extra
      + `<g transform="translate(0,${st.dy.toFixed(2)})">${AnimRig.rigSVG(rig, st, true)}</g>`
      + marker;
    if (selPart) {
      const g = rigStage.querySelector(`[data-pid="${CSS.escape(selPart)}"]`);
      if (g) g.classList.add("vd-part-sel");
    }
  }

  // verb test animation loop
  document.querySelectorAll("[data-verbtest]").forEach((btn) => {
    btn.addEventListener("click", () => {
      verbTest = btn.dataset.verbtest;
      if (rigAnim) cancelAnimationFrame(rigAnim);
      if (!verbTest && btn.textContent === "Stop") { drawRig(0); return; }
      const start = performance.now();
      const tick = (now) => {
        drawRig((now - start) / 1000);
        rigAnim = requestAnimationFrame(tick);
      };
      if (verbTest || btn.textContent === "Idle") rigAnim = requestAnimationFrame(tick);
    });
  });

  // part selection + move / rotate / scale drags on the stage
  let dragPart = null;
  const ptAngle = (pt, pivot) =>
    Math.atan2(pt.y - pivot[1], pt.x - pivot[0]) * 180 / Math.PI;

  rigStage.addEventListener("mousedown", (ev) => {
    if (!rig) return;
    const pt = svgPoint(rigStage, ev);
    const handle = ev.target.closest("[data-handle]");
    if (handle && selPart) {
      const p = rig.parts.find((x) => x.id === selPart);
      const pw = pivotWorld(selPart);
      if (!p || !pw) return;
      if (handle.dataset.handle === "rotate") {
        dragPart = { mode: "rotate", part: p, pivot: pw,
          startRot: p.rot || 0, startAng: ptAngle(pt, pw) };
      } else {
        const d0 = Math.hypot(pt.x - pw[0], pt.y - pw[1]);
        dragPart = { mode: "scale", part: p, pivot: pw,
          s0: p.scale !== undefined ? Number(p.scale) || 1 : 1, d0: Math.max(2, d0) };
      }
      ev.preventDefault();
      return;
    }
    const g = ev.target.closest("[data-pid]");
    if (!g) return;
    selPart = g.dataset.pid;
    renderPartList();
    renderPartForm();
    const p = rig.parts.find((x) => x.id === selPart);
    dragPart = { mode: "move", part: p, sx: pt.x, sy: pt.y,
      ox: p.offset[0], oy: p.offset[1] };
    drawRig(0);
    ev.preventDefault();
  });
  window.addEventListener("mousemove", (ev) => {
    if (!dragPart) return;
    if (!dragPart.pushed) { pushRig(); dragPart.pushed = true; }
    const pt = svgPoint(rigStage, ev);
    const d = dragPart;
    if (d.mode === "move") {
      d.part.offset = [
        Math.round((d.ox + pt.x - d.sx) * 10) / 10,
        Math.round((d.oy + pt.y - d.sy) * 10) / 10];
    } else if (d.mode === "rotate") {
      let rot = d.startRot + ptAngle(pt, d.pivot) - d.startAng;
      rot = ((rot + 180) % 360 + 360) % 360 - 180;         // keep in [-180, 180]
      d.part.rot = Math.round(rot * 2) / 2;
    } else if (d.mode === "scale") {
      const dist = Math.hypot(pt.x - d.pivot[0], pt.y - d.pivot[1]);
      d.part.scale = Math.min(20, Math.max(0.05,
        Math.round(d.s0 * (dist / d.d0) * 100) / 100));
    }
    renderPartForm();
    drawRig(0);
  });
  window.addEventListener("mouseup", () => { dragPart = null; });

  // floating panel collapse toggles
  document.querySelectorAll(".vd-float-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const box = $(btn.dataset.float);
      box.classList.toggle("collapsed");
      btn.textContent = box.classList.contains("collapsed") ? "+" : "–";
    });
  });

  // parts list + property form
  function renderPartList() {
    const list = $("part-list");
    list.innerHTML = "";
    if (!rig) return;
    $("part-count").textContent = `${rig.parts.length} part(s)`;
    rig.parts.forEach((p) => {
      const row = el("div", "vd-docrow" + (p.id === selPart ? " sel" : ""));
      row.appendChild(el("span", "vd-docname", p.id));
      row.appendChild(el("span", "vd-docsub", p.tag + (p.parent ? " < " + p.parent : "")));
      row.addEventListener("click", () => { selPart = p.id; renderPartList(); renderPartForm(); drawRig(0); });
      list.appendChild(row);
    });
  }

  function renderPartForm() {
    const form = $("part-form");
    form.innerHTML = "";
    if (!rig || !selPart) {
      form.appendChild(el("div", "im-placeholder", "select a part"));
      return;
    }
    const p = rig.parts.find((x) => x.id === selPart);
    if (!p) { form.appendChild(el("div", "im-placeholder", "select a part")); return; }
    const redraw = () => drawRig(0);
    const B = (fn) => (v) => { pushRig(); fn(v); };   // record undo state first

    form.appendChild(mkText("id", p.id, B((v) => {
      v = v.trim() || p.id;
      rig.parts.forEach((c) => { if (c.parent === p.id) c.parent = v; });
      p.id = v;
      selPart = v;
      renderPartList();
      redraw();
    })));
    form.appendChild(mkSelect("tag", TAGS.map((t) => [t, t]), p.tag || "other",
      B((v) => { p.tag = v; redraw(); })));
    const parents = [["", "(none)"]].concat(
      rig.parts.filter((x) => x.id !== p.id).map((x) => [x.id, x.id]));
    form.appendChild(mkSelect("parent", parents, p.parent || "",
      B((v) => { p.parent = v || null; redraw(); })));
    form.appendChild(mkNum("z", p.z || 0, B((v) => { p.z = v; redraw(); }), "0.1"));
    form.appendChild(mkNum("offset x", p.offset[0], B((v) => { p.offset[0] = v; redraw(); }), "0.5"));
    form.appendChild(mkNum("offset y", p.offset[1], B((v) => { p.offset[1] = v; redraw(); }), "0.5"));
    form.appendChild(mkNum("pivot x", p.pivot[0], B((v) => { p.pivot[0] = v; redraw(); }), "0.5"));
    form.appendChild(mkNum("pivot y", p.pivot[1], B((v) => { p.pivot[1] = v; redraw(); }), "0.5"));
    form.appendChild(mkNum("rotate", p.rot || 0,
      B((v) => { p.rot = v; redraw(); }), "0.5"));
    form.appendChild(mkNum("size", p.scale !== undefined ? p.scale : 1,
      B((v) => { p.scale = Math.min(20, Math.max(0.05, v || 1)); redraw(); }), "0.05"));
    form.appendChild(mkColor("fill", p.fill || "#888888", B((v) => { p.fill = v; redraw(); })));
    form.appendChild(mkColor("stroke", p.stroke === "none" ? "#000000" : (p.stroke || INK),
      B((v) => { p.stroke = v; redraw(); })));
    form.appendChild(mkNum("stroke w", p.sw !== undefined ? p.sw : 2,
      B((v) => { p.sw = v; redraw(); }), "0.5"));

    const sh = p.shape || (p.shape = { type: "rect", x: -20, y: -20, w: 40, h: 40, rx: 0 });
    form.appendChild(mkSelect("shape", [["rect", "rect"], ["ellipse", "ellipse"], ["path", "path"]],
      sh.type, B((v) => {
        if (v === sh.type) return;
        if (v === "rect") p.shape = { type: "rect", x: -20, y: -20, w: 40, h: 40, rx: 0 };
        else if (v === "ellipse") p.shape = { type: "ellipse", cx: 0, cy: 0, rx: 20, ry: 20 };
        else p.shape = { type: "path", d: "M -20 0 L 20 0" };
        renderPartForm();
        redraw();
      })));
    if (sh.type === "rect") {
      ["x", "y", "w", "h", "rx"].forEach((k) => form.appendChild(
        mkNum("rect " + k, sh[k] || 0, B((v) => { sh[k] = v; redraw(); }), "0.5")));
    } else if (sh.type === "ellipse") {
      ["cx", "cy", "rx", "ry"].forEach((k) => form.appendChild(
        mkNum("ell " + k, sh[k] || 0, B((v) => { sh[k] = v; redraw(); }), "0.5")));
    } else {
      const wrap = el("label", "vd-f vd-f-wide");
      wrap.appendChild(el("span", null, "path d"));
      const inp = document.createElement("textarea");
      inp.rows = 3;
      inp.value = sh.d || "";
      inp.addEventListener("change", () => { pushRig(); sh.d = inp.value; redraw(); });
      wrap.appendChild(inp);
      form.appendChild(wrap);
    }
  }

  // part + rig actions
  $("part-add").addEventListener("click", () => {
    if (!rig) return;
    pushRig();
    let n = rig.parts.length + 1;
    while (rig.parts.some((p) => p.id === "part-" + n)) n++;
    const maxZ = Math.max(0, ...rig.parts.map((p) => p.z || 0));
    rig.parts.push({ id: "part-" + n, tag: "other", parent: null, offset: [0, -50],
      pivot: [0, 0], z: maxZ + 1, shape: { type: "rect", x: -20, y: -20, w: 40, h: 40, rx: 0 },
      fill: "#8a8f9a", stroke: INK, sw: 2 });
    selPart = "part-" + n;
    renderPartList();
    renderPartForm();
    drawRig(0);
  });
  $("part-dup").addEventListener("click", () => {
    if (!rig || !selPart) return;
    pushRig();
    const p = rig.parts.find((x) => x.id === selPart);
    const copy = JSON.parse(JSON.stringify(p));
    let n = 2;
    while (rig.parts.some((x) => x.id === p.id + "-" + n)) n++;
    copy.id = p.id + "-" + n;
    copy.offset = [copy.offset[0] + 12, copy.offset[1]];
    rig.parts.push(copy);
    selPart = copy.id;
    renderPartList();
    renderPartForm();
    drawRig(0);
  });
  $("part-del").addEventListener("click", () => {
    if (!rig || !selPart) return;
    pushRig();
    const p = rig.parts.find((x) => x.id === selPart);
    rig.parts = rig.parts.filter((x) => x.id !== selPart);
    rig.parts.forEach((c) => { if (c.parent === selPart) c.parent = p.parent || null; });
    selPart = null;
    renderPartList();
    renderPartForm();
    drawRig(0);
  });

  async function saveRig(r) {
    const resp = await fetch("/api/anim/rigs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(r),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.status);
    return data;
  }
  $("rig-save").addEventListener("click", async () => {
    if (!rig) return;
    try {
      const saved = await saveRig(rig);
      rigStatus.textContent = "saved " + saved.name;
      await loadRigs(saved.name);
    } catch (err) { rigStatus.textContent = "error: " + err.message; }
  });
  $("rig-dup").addEventListener("click", async () => {
    if (!rig) return;
    const copy = JSON.parse(JSON.stringify(rig));
    copy.name = rig.name + "-copy";
    try {
      const saved = await saveRig(copy);
      await loadRigs(saved.name);
    } catch (err) { rigStatus.textContent = "error: " + err.message; }
  });
  $("rig-del").addEventListener("click", async () => {
    if (!rig || !confirm(`Delete rig ${rig.name}?`)) return;
    await fetch("/api/anim/rig/" + encodeURIComponent(rig.name), { method: "DELETE" });
    rig = null;
    $("rig-title").textContent = "no rig loaded";
    rigStage.innerHTML = "";
    renderPartForm();
    await loadRigs();
  });
  // export the open rig (with any unsaved edits) as a downloadable .json
  $("rig-export").addEventListener("click", () => {
    if (!rig) { rigStatus.textContent = "no rig loaded"; return; }
    const doc = JSON.parse(JSON.stringify(rig));
    delete doc._mtime;
    const blob = new Blob([JSON.stringify(doc, null, 1)],
      { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (doc.name || "rig") + ".json";
    a.click();
    URL.revokeObjectURL(a.href);
    rigStatus.textContent = "exported " + a.download;
  });

  // import rig .json file(s): saved to the library and the last one opened
  $("rig-import").addEventListener("click", () => $("rig-import-file").click());
  $("rig-import-file").addEventListener("change", async () => {
    const files = Array.from($("rig-import-file").files || []);
    $("rig-import-file").value = "";
    if (!files.length) return;
    let lastName = null;
    for (const file of files) {
      try {
        const doc = JSON.parse(await file.text());
        if (!Array.isArray(doc.parts) || !doc.parts.length) {
          throw new Error("no parts list — not a rig file");
        }
        if (!doc.name) doc.name = file.name.replace(/\.json$/i, "") || "imported";
        if (!doc.kind) doc.kind = "character";
        if (!doc.canvas) doc.canvas = { w: 300, h: 300 };
        const saved = await saveRig(doc);
        lastName = saved.name;
      } catch (err) {
        rigStatus.textContent = `import ${file.name} failed: ${err.message}`;
        return;
      }
    }
    await loadRigs(lastName);
    rigStatus.textContent = `imported ${files.length} rig(s)`;
  });

  $("rig-new").addEventListener("click", () => {
    openRig({ name: "new-rig", kind: "character", canvas: { w: 300, h: 300 },
      parts: [{ id: "torso", tag: "body", parent: null, offset: [0, 0], pivot: [0, 0],
        z: 1, shape: { type: "rect", x: -25, y: -140, w: 50, h: 80, rx: 8 },
        fill: "#8a8f9a", stroke: INK, sw: 2 }] });
    rigStatus.textContent = "new rig — edit and Save";
  });
  $("gen-go").addEventListener("click", async () => {
    rigStatus.textContent = "generating…";
    try {
      const resp = await fetch("/api/anim/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archetype: $("gen-arch").value,
          seed: $("gen-seed").value === "" ? null : Number($("gen-seed").value),
          name: $("gen-name").value.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.status);
      await loadRigs(data.name);
      rigStatus.textContent = "generated " + data.name;
    } catch (err) { rigStatus.textContent = "error: " + err.message; }
  });

  // ════════════════════════ SCENES ════════════════════════
  let scenes = [];
  let scene = null;
  let scnPlaying = false;
  let scnAnim = null;
  let scnT = 0;

  const scnStage = $("scn-stage");
  const scnStatus = $("scn-status");

  const rigsByName = () => {
    const m = {};
    rigs.forEach((r) => { m[r.name] = r; });
    return m;
  };

  function newScene() {
    return { name: "scene-1", w: 1280, h: 720, fps: 24, dur: 8,
      bg: { sky: "#242a36", floor: "#3a4150", floor_y: 0.82 },
      actors: [], actions: [], overlays: [], audio: "" };
  }

  async function loadScenes(openName) {
    const resp = await fetch("/api/anim/scenes");
    scenes = resp.ok ? await resp.json() : [];
    renderSceneList();
    fillShotSelect();
    renderShots();
    if (openName) {
      const s = scenes.find((x) => x.name === openName);
      if (s) openScene(s);
    } else if (!scene) {
      openScene(scenes.length ? scenes[0] : newScene());
    }
  }

  function renderSceneList() {
    const list = $("scn-list");
    list.innerHTML = "";
    if (!scenes.length) {
      list.appendChild(el("div", "im-placeholder", "no saved scenes"));
      return;
    }
    scenes.forEach((s) => {
      const row = el("div", "vd-docrow" + (scene && scene.name === s.name ? " sel" : ""));
      row.appendChild(el("span", "vd-docname", s.name));
      row.appendChild(el("span", "vd-docsub",
        `${s.actors.length} actor(s) · ${s.dur}s`));
      const delBtn = el("button", "runs-btn vd-mini", "x");
      delBtn.title = "delete scene";
      delBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        if (!confirm(`Delete scene ${s.name}?`)) return;
        await fetch("/api/anim/scene/" + encodeURIComponent(s.name), { method: "DELETE" });
        await loadScenes();
      });
      row.appendChild(delBtn);
      row.addEventListener("click", () => openScene(s));
      list.appendChild(row);
    });
  }

  function syncSceneInputs() {
    $("scn-name").value = scene.name;
    $("scn-w").value = scene.w;
    $("scn-h").value = scene.h;
    $("scn-fps").value = scene.fps;
    $("scn-dur").value = scene.dur;
    $("scn-sky").value = (scene.bg || {}).sky || "#242a36";
    $("scn-floor").value = (scene.bg || {}).floor || "#3a4150";
    $("scn-floory").value = (scene.bg || {}).floor_y !== undefined ? scene.bg.floor_y : 0.82;
    $("scn-time").max = scene.dur;
    fillAudioSelect();
    $("scn-title").textContent = scene.name;
  }

  function openScene(s) {
    stopPlay();
    scene = JSON.parse(JSON.stringify(s));
    delete scene._mtime;
    clearHist("scn");
    scnT = 0;
    syncSceneInputs();
    $("scn-time").value = 0;
    renderSceneList();
    renderActors();
    renderActions();
    renderOverlays();
    drawScene();
  }

  // scene property bindings
  const bindProp = (id, fn) => $(id).addEventListener("change", () => { if (scene) { pushScn(); fn(); drawScene(); } });
  bindProp("scn-name", () => { scene.name = $("scn-name").value.trim() || "scene-1"; $("scn-title").textContent = scene.name; });
  bindProp("scn-w", () => { scene.w = Number($("scn-w").value) || 1280; });
  bindProp("scn-h", () => { scene.h = Number($("scn-h").value) || 720; });
  bindProp("scn-fps", () => { scene.fps = Number($("scn-fps").value) || 24; });
  bindProp("scn-dur", () => { scene.dur = Number($("scn-dur").value) || 8; $("scn-time").max = scene.dur; });
  bindProp("scn-sky", () => { (scene.bg = scene.bg || {}).sky = $("scn-sky").value; });
  bindProp("scn-floor", () => { (scene.bg = scene.bg || {}).floor = $("scn-floor").value; });
  bindProp("scn-floory", () => { (scene.bg = scene.bg || {}).floor_y = Number($("scn-floory").value) || 0.82; });
  bindProp("scn-audio", () => { scene.audio = $("scn-audio").value; });

  function fillAudioSelect() {
    const sel = $("scn-audio");
    const lib = (window.VideoEditor ? window.VideoEditor.getLibrary() : []) || [];
    const cur = scene ? scene.audio || "" : "";
    sel.innerHTML = "";
    const none = document.createElement("option");
    none.value = "";
    none.textContent = "none";
    sel.appendChild(none);
    lib.filter((f) => f.is_audio).forEach((f) => {
      const o = document.createElement("option");
      o.value = f.name;
      o.textContent = f.name;
      if (f.name === cur) o.selected = true;
      sel.appendChild(o);
    });
  }
  document.addEventListener("vd-library-changed", fillAudioSelect);

  // a rig with no kind (hand-imported etc.) counts as a character
  const rigKind = (name) => (rigsByName()[name] || {}).kind === "object"
    ? "object" : "character";

  function fillSceneRigSelect() {
    const fill = (sel, kind) => {
      sel.innerHTML = "";
      rigs.filter((r) => (r.kind === "object" ? "object" : "character") === kind)
        .forEach((r) => {
          const o = document.createElement("option");
          o.value = r.name;
          o.textContent = r.name;
          sel.appendChild(o);
        });
    };
    fill($("scn-addrig"), "character");
    fill($("scn-addobj"), "object");
  }

  function drawScene() {
    if (!scene) { scnStage.innerHTML = ""; return; }
    scnStage.setAttribute("viewBox", `0 0 ${scene.w} ${scene.h}`);
    scnStage.innerHTML = AnimRig.sceneSVG(scene, rigsByName(), scnT, true);
    $("scn-timelabel").textContent = scnT.toFixed(1) + "s";
  }

  $("scn-time").addEventListener("input", () => {
    stopPlay();
    scnT = Number($("scn-time").value) || 0;
    drawScene();
  });

  function stopPlay() {
    scnPlaying = false;
    $("scn-play").textContent = "Play";
    if (scnAnim) cancelAnimationFrame(scnAnim);
  }
  $("scn-play").addEventListener("click", () => {
    if (!scene) return;
    if (scnPlaying) { stopPlay(); return; }
    scnPlaying = true;
    $("scn-play").textContent = "Stop";
    const t0 = performance.now() - scnT * 1000;
    const tick = (now) => {
      if (!scnPlaying) return;
      scnT = ((now - t0) / 1000) % scene.dur;
      $("scn-time").value = scnT;
      drawScene();
      scnAnim = requestAnimationFrame(tick);
    };
    scnAnim = requestAnimationFrame(tick);
  });

  // actor drag on the scene stage
  let dragActor = null;
  scnStage.addEventListener("mousedown", (ev) => {
    if (!scene) return;
    const g = ev.target.closest("[data-aid]");
    if (!g) return;
    const actor = scene.actors.find((a) => a.id === g.dataset.aid);
    if (!actor) return;
    const pt = svgPoint(scnStage, ev);
    dragActor = { actor, sx: pt.x, sy: pt.y, ox: actor.x, oy: actor.y };
    ev.preventDefault();
  });
  window.addEventListener("mousemove", (ev) => {
    if (!dragActor) return;
    if (!dragActor.pushed) { pushScn(); dragActor.pushed = true; }
    const pt = svgPoint(scnStage, ev);
    dragActor.actor.x = Math.round(dragActor.ox + pt.x - dragActor.sx);
    dragActor.actor.y = Math.round(dragActor.oy + pt.y - dragActor.sy);
    renderActors();
    drawScene();
  });
  window.addEventListener("mouseup", () => { dragActor = null; });

  // actors / objects / actions / overlays editors — actors and objects share
  // scene.actors in the JSON (the renderers don't care); the UI splits them
  // by their rig's kind
  function renderActors() {
    renderPlacements("scn-actors", "character",
      "no actors — pick a character rig above and Add actor");
    renderPlacements("scn-objects", "object",
      "no objects — pick an object rig above and Add object");
  }

  function renderPlacements(boxId, kind, emptyMsg) {
    const box = $(boxId);
    box.innerHTML = "";
    const mine = scene ? scene.actors.filter((a) => rigKind(a.rig) === kind) : [];
    if (!mine.length) {
      box.appendChild(el("div", "im-placeholder", emptyMsg));
      return;
    }
    const S = (fn) => (v) => { pushScn(); fn(v); };   // record undo state first
    const rigOpts = rigs.filter((r) => (r.kind === "object" ? "object" : "character") === kind)
      .map((r) => [r.name, r.name]);
    mine.forEach((a) => {
      const row = el("div", "vd-row");
      row.appendChild(mkText("id", a.id, S((v) => {
        v = v.trim() || a.id;
        scene.actions.forEach((ac) => { if (ac.actor === a.id) ac.actor = v; });
        a.id = v;
        renderActions();
        drawScene();
      })));
      row.appendChild(mkSelect("rig", rigOpts, a.rig,
        S((v) => { a.rig = v; drawScene(); })));
      row.appendChild(mkNum("x", a.x, S((v) => { a.x = v; drawScene(); })));
      row.appendChild(mkNum("y", a.y, S((v) => { a.y = v; drawScene(); })));
      row.appendChild(mkNum("scale", a.scale || 1, S((v) => { a.scale = v; drawScene(); }), "0.05"));
      row.appendChild(mkCheck("flip", a.flip, S((v) => { a.flip = v; drawScene(); })));
      row.appendChild(mkSelect("facing", [["profile", "profile"], ["front", "front"]], a.facing || "profile",
        S((v) => { a.facing = v; drawScene(); })));
      const rm = el("button", "runs-btn vd-mini", "Remove");
      rm.addEventListener("click", () => {
        pushScn();
        scene.actors.splice(scene.actors.indexOf(a), 1);
        scene.actions = scene.actions.filter((ac) => ac.actor !== a.id);
        renderActors();
        renderActions();
        drawScene();
      });
      row.appendChild(rm);
      box.appendChild(row);
    });
  }

  function addPlacement(selId, prefix) {
    if (!scene) return;
    const rname = $(selId).value;
    if (!rname) { scnStatus.textContent = "no rigs of that kind — make one in the Rigs view"; return; }
    pushScn();
    let n = 1;
    while (scene.actors.some((a) => a.id === prefix + n)) n++;
    scene.actors.push({ id: prefix + n, rig: rname, x: Math.round(scene.w * 0.3),
      y: Math.round(scene.h * ((scene.bg || {}).floor_y || 0.82) * 0.98 + scene.h * 0.06),
      scale: 1, flip: false });
    renderActors();
    renderActions();
    drawScene();
  }
  $("scn-addactor").addEventListener("click", () => addPlacement("scn-addrig", "a"));
  $("scn-addobject").addEventListener("click", () => addPlacement("scn-addobj", "o"));

  const VERB_ARGS = {
    walk_right: [["to_x", "to x", 1]],
    walk_left: [["to_x", "to x", 1]],
    walk_up_stairs_right: [["to_x", "to x", 1], ["to_y", "to y", 1]],
    walk_up_stairs_left: [["to_x", "to x", 1], ["to_y", "to y", 1]],
    move: [["to_x", "to x", 1], ["to_y", "to y", 1]],
    talk: [],
    point: [],
    present: [["angle", "angle °", 1]],
    explain: [],
    think: [],
    code: [],
    face: [["front", "front-facing", 1]],
    fade: [["from", "from", 0.05], ["to", "to", 0.05]],
    open: [["dx", "slide x", 1], ["dy", "slide y", 1], ["angle", "hinge °", 1]],
    close: [],
  };
  // fresh defaults when an action's verb changes
  const VERB_DEFAULTS = {
    open: () => ({ dx: -60, dy: 0, angle: -100 }),
    walk_right: () => ({ to_x: Math.round(scene.w * 0.8) }),
    walk_left: () => ({ to_x: Math.round(scene.w * 0.2) }),
    walk_up_stairs_right: () => ({ to_x: Math.round(scene.w * 0.7), to_y: Math.round(scene.h * 0.5) }),
    walk_up_stairs_left: () => ({ to_x: Math.round(scene.w * 0.3), to_y: Math.round(scene.h * 0.5) }),
    present: () => ({ arm: "r", angle: -45 }),
    face: () => ({ front: 1 }),
  };

  // Sequencing: actions execute top to bottom. Each action's start mode:
  //   after (default) — starts when the previous action ENDS (chain)
  //   with            — starts when the previous action STARTS (parallel tie,
  //                     e.g. two actors talking at once)
  //   at              — explicit t0 (what pre-sequencing scenes use)
  // t0/t1 are RESOLVED here and stored, so the scene JSON, the preview math
  // and the server renderer all keep working on plain times.
  function resolveActions() {
    if (!scene) return;
    let prevStart = 0, prevEnd = 0;
    for (const ac of scene.actions) {
      const mode = ac.mode || "at";
      const dur = ac.dur !== undefined ? Number(ac.dur) || 0
        : Math.max(0, (Number(ac.t1) || 0) - (Number(ac.t0) || 0));
      ac.dur = dur;
      if (mode === "after") ac.t0 = prevEnd;
      else if (mode === "with") ac.t0 = prevStart;
      else ac.t0 = Number(ac.t0) || 0;
      ac.t0 = Math.round(ac.t0 * 100) / 100;
      ac.t1 = Math.round((ac.t0 + dur) * 100) / 100;
      prevStart = ac.t0;
      prevEnd = ac.t1;
    }
    const end = Math.max(0, ...scene.actions.map((a) => a.t1));
    if (end > scene.dur) {              // auto-grow the scene to fit the chain
      scene.dur = Math.ceil(end * 2) / 2;
      syncSceneInputs();
    }
  }

  function renderActions() {
    const box = $("scn-actions");
    box.innerHTML = "";
    if (!scene || !scene.actions.length) {
      box.appendChild(el("div", "im-placeholder", "no actions — the scene is a still"));
      return;
    }
    resolveActions();
    const actorOpts = scene.actors.map((a) =>
      [a.id, rigKind(a.rig) === "object" ? a.id + " (obj)" : a.id]);
    // objects move/fade/open/close; characters get the acting verbs
    const verbsFor = (actorId) => {
      const a = scene.actors.find((x) => x.id === actorId);
      return a && rigKind(a.rig) === "object"
        ? ["move", "fade", "open", "close"]
        : ["walk_right", "walk_left", "walk_up_stairs_right", "walk_up_stairs_left", "move", "talk", "point", "present", "explain", "think", "code", "face", "fade", "open", "close"];
    };
    const S = (fn) => (v) => { pushScn(); fn(v); };
    const rerender = () => { renderActions(); drawScene(); };
    scene.actions.forEach((ac, i) => {
      const row = el("div", "vd-row");

      const ord = el("div", "vd-ord");
      const up = el("button", "runs-btn vd-mini", "▲");
      up.title = "run earlier";
      up.disabled = i === 0;
      up.addEventListener("click", () => {
        pushScn();
        [scene.actions[i - 1], scene.actions[i]] = [scene.actions[i], scene.actions[i - 1]];
        rerender();
      });
      const down = el("button", "runs-btn vd-mini", "▼");
      down.title = "run later";
      down.disabled = i === scene.actions.length - 1;
      down.addEventListener("click", () => {
        pushScn();
        [scene.actions[i + 1], scene.actions[i]] = [scene.actions[i], scene.actions[i + 1]];
        rerender();
      });
      ord.appendChild(el("span", "vd-ordnum", String(i + 1)));
      ord.appendChild(up);
      ord.appendChild(down);
      row.appendChild(ord);

      row.appendChild(mkSelect("actor", actorOpts, ac.actor, S((v) => {
        ac.actor = v;
        if (!verbsFor(v).includes(ac.verb)) {   // e.g. talk -> object target
          ac.verb = verbsFor(v)[0];
          ac.args = VERB_DEFAULTS[ac.verb] ? VERB_DEFAULTS[ac.verb]() : {};
        }
        rerender();
      })));
      row.appendChild(mkSelect("verb", verbsFor(ac.actor).map((v) => [v, v]), ac.verb,
        S((v) => {
          ac.verb = v;
          ac.args = VERB_DEFAULTS[v] ? VERB_DEFAULTS[v]() : {};
          rerender();
        })));
      row.appendChild(mkSelect("start", [["after", "after prev"], ["with", "with prev"], ["at", "at time"]],
        ac.mode || "at", S((v) => { ac.mode = v; rerender(); })));
      if ((ac.mode || "at") === "at") {
        row.appendChild(mkNum("t0", ac.t0, S((v) => { ac.t0 = v; rerender(); }), "0.1"));
      }
      row.appendChild(mkNum("dur (s)", ac.dur !== undefined ? ac.dur
        : Math.max(0, (ac.t1 || 0) - (ac.t0 || 0)),
      S((v) => { ac.dur = Math.max(0.1, v); rerender(); }), "0.1"));
      (VERB_ARGS[ac.verb] || []).forEach(([key, label, step]) => {
        row.appendChild(mkNum(label, (ac.args || {})[key] !== undefined ? ac.args[key] : 0,
          S((v) => { (ac.args = ac.args || {})[key] = v; drawScene(); }), String(step)));
      });
      if (ac.verb === "point" || ac.verb === "present") {
        row.appendChild(mkSelect("arm", [["r", "right"], ["l", "left"]],
          (ac.args || {}).arm || "r", S((v) => { (ac.args = ac.args || {}).arm = v; drawScene(); })));
      }
      row.appendChild(el("span", "vd-timespan", `${ac.t0.toFixed(1)} → ${ac.t1.toFixed(1)}s`));
      const rm = el("button", "runs-btn vd-mini", "Remove");
      rm.addEventListener("click", () => { pushScn(); scene.actions.splice(i, 1); rerender(); });
      row.appendChild(rm);
      box.appendChild(row);
    });
  }
  $("scn-addaction").addEventListener("click", () => {
    if (!scene) return;
    if (!scene.actors.length) { scnStatus.textContent = "add an actor first"; return; }
    pushScn();
    scene.actions.push({ actor: scene.actors[0].id, verb: "walk_right", mode: "after",
      dur: 2, t0: 0, t1: 2, args: { to_x: Math.round(scene.w * 0.8) } });
    renderActions();
    drawScene();
  });

  function renderOverlays() {
    const box = $("scn-overlays");
    box.innerHTML = "";
    if (!scene || !scene.overlays.length) {
      box.appendChild(el("div", "im-placeholder", "no overlays"));
      return;
    }
    const S = (fn) => (v) => { pushScn(); fn(v); };
    scene.overlays.forEach((ov, i) => {
      const row = el("div", "vd-row");
      row.appendChild(mkSelect("type", [["caption", "caption"], ["title", "title"], ["box", "infobox"]],
        ov.type || "caption", S((v) => { ov.type = v; renderOverlays(); drawScene(); })));
      const wrap = el("label", "vd-f vd-f-wide");
      wrap.appendChild(el("span", null, "text"));
      const ta = document.createElement("textarea");
      ta.rows = 2;
      ta.value = ov.text || "";
      ta.addEventListener("change", () => { pushScn(); ov.text = ta.value; drawScene(); });
      wrap.appendChild(ta);
      row.appendChild(wrap);
      row.appendChild(mkNum("t0", ov.t0, S((v) => { ov.t0 = v; drawScene(); }), "0.1"));
      row.appendChild(mkNum("t1", ov.t1, S((v) => { ov.t1 = v; drawScene(); }), "0.1"));
      if (ov.type === "box") {
        row.appendChild(mkNum("x", ov.x !== undefined ? ov.x : 60, S((v) => { ov.x = v; drawScene(); })));
        row.appendChild(mkNum("y", ov.y !== undefined ? ov.y : 60, S((v) => { ov.y = v; drawScene(); })));
        row.appendChild(mkNum("w", ov.w !== undefined ? ov.w : 300, S((v) => { ov.w = v; drawScene(); })));
      }
      const rm = el("button", "runs-btn vd-mini", "Remove");
      rm.addEventListener("click", () => { pushScn(); scene.overlays.splice(i, 1); renderOverlays(); drawScene(); });
      row.appendChild(rm);
      box.appendChild(row);
    });
  }
  $("scn-addoverlay").addEventListener("click", () => {
    if (!scene) return;
    pushScn();
    scene.overlays.push({ type: "caption", text: "…", t0: 0, t1: Math.min(3, scene.dur) });
    renderOverlays();
    drawScene();
  });

  // ── storyboard: ordered shots -> one rendered mp4 ───────────────────
  let story = { name: "", shots: [] };
  let stories = [];

  function fillShotSelect() {
    const sel = $("sb-scene");
    sel.innerHTML = "";
    scenes.forEach((s) => {
      const o = document.createElement("option");
      o.value = s.name;
      o.textContent = s.name;
      sel.appendChild(o);
    });
  }

  async function loadStories() {
    const resp = await fetch("/api/anim/stories");
    stories = resp.ok ? await resp.json() : [];
    renderStoryList();
  }

  function renderStoryList() {
    const list = $("sb-list");
    list.innerHTML = "";
    stories.forEach((st) => {
      const row = el("div", "vd-docrow" + (story.name === st.name ? " sel" : ""));
      row.appendChild(el("span", "vd-docname", st.name));
      row.appendChild(el("span", "vd-docsub", `${st.shots.length} shot(s)`));
      const delBtn = el("button", "runs-btn vd-mini", "x");
      delBtn.title = "delete story";
      delBtn.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        if (!confirm(`Delete story ${st.name}?`)) return;
        await fetch("/api/anim/story/" + encodeURIComponent(st.name), { method: "DELETE" });
        await loadStories();
      });
      row.appendChild(delBtn);
      row.addEventListener("click", () => {
        story = JSON.parse(JSON.stringify(st));
        delete story._mtime;
        $("sb-name").value = story.name;
        renderStoryList();
        renderShots();
      });
      list.appendChild(row);
    });
  }

  function renderShots() {
    const box = $("sb-shots");
    box.innerHTML = "";
    if (!story.shots.length) {
      box.appendChild(el("div", "im-placeholder",
        "no shots — save scenes, then add them here in story order"));
      return;
    }
    story.shots.forEach((sn, i) => {
      const row = el("div", "vd-row");
      const ord = el("div", "vd-ord");
      ord.appendChild(el("span", "vd-ordnum", String(i + 1)));
      const up = el("button", "runs-btn vd-mini", "▲");
      up.disabled = i === 0;
      up.addEventListener("click", () => {
        [story.shots[i - 1], story.shots[i]] = [story.shots[i], story.shots[i - 1]];
        renderShots();
      });
      const down = el("button", "runs-btn vd-mini", "▼");
      down.disabled = i === story.shots.length - 1;
      down.addEventListener("click", () => {
        [story.shots[i + 1], story.shots[i]] = [story.shots[i], story.shots[i + 1]];
        renderShots();
      });
      ord.appendChild(up);
      ord.appendChild(down);
      row.appendChild(ord);
      row.appendChild(mkSelect("scene", scenes.map((s) => [s.name, s.name]), sn,
        (v) => { story.shots[i] = v; }));
      const missing = !scenes.some((s) => s.name === sn);
      if (missing) row.appendChild(el("span", "vd-timespan", "missing scene"));
      const rm = el("button", "runs-btn vd-mini", "Remove");
      rm.addEventListener("click", () => { story.shots.splice(i, 1); renderShots(); });
      row.appendChild(rm);
      box.appendChild(row);
    });
  }

  $("sb-add").addEventListener("click", () => {
    const sn = $("sb-scene").value;
    if (!sn) { $("sb-status").textContent = "save a scene first"; return; }
    story.shots.push(sn);
    renderShots();
  });
  $("sb-save").addEventListener("click", async () => {
    story.name = $("sb-name").value.trim() || "story-1";
    try {
      const resp = await fetch("/api/anim/stories", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(story),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.status);
      story.name = data.name;
      $("sb-name").value = data.name;
      $("sb-status").textContent = "saved " + data.name;
      await loadStories();
    } catch (err) { $("sb-status").textContent = "error: " + err.message; }
  });
  $("sb-render").addEventListener("click", async () => {
    if (!story.shots.length) { $("sb-status").textContent = "add shots first"; return; }
    $("sb-status").textContent = "starting…";
    try {
      const resp = await fetch("/api/anim/render_story", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shots: story.shots,
          out_name: $("sb-name").value.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.status);
      $("sb-status").textContent =
        `rendering ${data.output} — progress in Editor > Jobs`;
      if (window.VideoEditor) window.VideoEditor.pollJobs();
    } catch (err) { $("sb-status").textContent = "error: " + err.message; }
  });

  // generate a template scene (props arrive as new object rigs)
  $("scn-generate").addEventListener("click", async () => {
    scnStatus.textContent = "generating…";
    try {
      const resp = await fetch("/api/anim/generate_scene", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ template: $("scn-template").value,
          seed: $("scn-genseed").value === "" ? null : Number($("scn-genseed").value),
          name: $("scn-genname").value.trim() }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.status);
      await loadRigs();                 // pick up the generated prop rigs
      await loadScenes(data.name);
      scnStatus.textContent = "generated " + data.name;
    } catch (err) { scnStatus.textContent = "error: " + err.message; }
  });

  // save / new / render
  $("scn-new").addEventListener("click", () => {
    let n = 1;
    while (scenes.some((s) => s.name === "scene-" + n)) n++;
    const s = newScene();
    s.name = "scene-" + n;
    openScene(s);
    scnStatus.textContent = "new scene — Save when ready";
  });
  $("scn-save").addEventListener("click", async () => {
    if (!scene) return;
    try {
      const resp = await fetch("/api/anim/scenes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scene),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.status);
      scnStatus.textContent = "saved " + data.name;
      await loadScenes(data.name);
    } catch (err) { scnStatus.textContent = "error: " + err.message; }
  });
  $("scn-render").addEventListener("click", async () => {
    if (!scene) return;
    $("scn-render-status").textContent = "starting render…";
    try {
      const resp = await fetch("/api/anim/render", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scene),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.status);
      $("scn-render-status").textContent =
        `rendering ${data.output} — progress in the Editor view's Jobs panel`;
      if (window.VideoEditor) window.VideoEditor.pollJobs();
    } catch (err) {
      $("scn-render-status").textContent = "error: " + err.message;
    }
  });

  // ── init ────────────────────────────────────────────────────────────
  (async function initStudio() {
    try {
      const st = await (await fetch("/api/anim/status")).json();
      if (!st.ok) {
        rigStatus.textContent = "unavailable: " + (st.err || "?");
        return;
      }
      if (st.tags) TAGS = st.tags;
      const archSel = $("gen-arch");
      (st.archetypes || []).forEach((a) => {
        const o = document.createElement("option");
        o.value = a;
        o.textContent = a;
        archSel.appendChild(o);
      });
      const tplSel = $("scn-template");
      (st.scene_templates || []).forEach((t) => {
        const o = document.createElement("option");
        o.value = t;
        o.textContent = t;
        tplSel.appendChild(o);
      });
      if (!st.raster) {
        $("scn-render-status").textContent =
          "server rasterizer missing: " + (st.raster_err || "") + " — pip install resvg-py";
      }
      await loadRigs();
      await loadScenes();
      await loadStories();
    } catch (err) {
      rigStatus.textContent = "error: " + err.message;
    }
  })();
})();
