// GENREG PO Metrics: a 3D plot of the "constraint cone". With no constraints the
// object is a sphere (infinite possibility). Each checked constraint adds a ring
// along the +x axis (leading right) and pulls the shape from a sphere toward a
// cone — each ring slices the space of possible organisms smaller, tip = the few
// probable survivors. Rings are driven purely by the Constraints checkboxes.
//
// Hand-rolled canvas-2D 3D (no vendored 3D lib): a surface of revolution about
// the x-axis, projected under a fixed tilt with a slow orbit for depth.

(() => {
  const canvas = document.getElementById("po-canvas");
  const stage = document.getElementById("po-stage");
  const readout = document.getElementById("po-readout");
  const bodyMicro = document.getElementById("tab-microscope");
  const bodyPO = document.getElementById("tab-po");
  const tabs = document.querySelectorAll(".side-tab");
  if (!canvas || !stage) return;
  const ctx = canvas.getContext("2d");

  // -- constraint -> rings -------------------------------------------------
  let rings = [];   // [{label, color}]
  function ringColor(i) { return `hsl(${(i * 47 + 205) % 360}, 70%, 62%)`; }

  function syncRings() {
    const checked = Array.from(document.querySelectorAll('input[name="constraint"]:checked'));
    rings = checked.map((cb, i) => ({
      label: (cb.parentElement.textContent || cb.value).trim(),
      color: ringColor(i),
    }));
    if (readout) {
      readout.textContent = rings.length
        ? `${rings.length} constraint${rings.length > 1 ? "s" : ""} → ${rings.length} ring${rings.length > 1 ? "s" : ""}: ${rings.map((r) => r.label).join(", ")}`
        : "No constraints — infinite possibility (sphere).";
    }
  }

  // -- geometry ------------------------------------------------------------
  // Radius profile along x in [-1, 1]: sphere when t=0, cone when t=1.
  function sphereR(x) { return Math.sqrt(Math.max(0, 1 - x * x)); }
  function coneR(x) { return Math.max(0, (1 - x) / 2); }   // base r=1 at x=-1, tip at x=+1
  function profileR(x, t) { return (1 - t) * sphereR(x) + t * coneR(x); }
  function morphT(n) { return n / (n + 3); }               // 0 -> sphere, grows toward cone

  // -- 3D transform / projection ------------------------------------------
  const PITCH = -0.42;   // fixed tilt so rings read as ellipses
  let yaw = 0.6;

  function rot(p) {
    // yaw about world Y, then pitch about world X
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    let x = p.x * cy + p.z * sy;
    let z = -p.x * sy + p.z * cy;
    let y = p.y;
    const cx = Math.cos(PITCH), sx = Math.sin(PITCH);
    const y2 = y * cx - z * sx;
    const z2 = y * sx + z * cx;
    return { x, y: y2, z: z2 };
  }

  function project(p, cx, cy, scale) {
    const camZ = 3.4, focal = 2.4;
    const denom = camZ - p.z;
    const s = (focal / denom) * scale;
    return { X: cx + p.x * s, Y: cy - p.y * s, depth: p.z };
  }

  // one revolution ring (circle about the x-axis) at position `xpos`, radius `rad`
  function ringPoints(xpos, rad, seg) {
    const pts = [];
    for (let i = 0; i <= seg; i++) {
      const a = (i / seg) * Math.PI * 2;
      pts.push({ x: xpos, y: rad * Math.cos(a), z: rad * Math.sin(a) });
    }
    return pts;
  }

  function strokePath(pts, cx, cy, scale) {
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const q = project(rot(pts[i]), cx, cy, scale);
      if (i === 0) ctx.moveTo(q.X, q.Y); else ctx.lineTo(q.X, q.Y);
    }
    ctx.stroke();
  }

  // -- render --------------------------------------------------------------
  function render() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = "#07090d";
    ctx.fillRect(0, 0, w, h);

    const cx = w / 2, cy = h / 2, scale = Math.min(w, h) * 0.34;
    const n = rings.length;
    const t = morphT(n);

    // x-axis guide (the cone's axis, leading right)
    ctx.strokeStyle = "rgba(125,135,148,0.25)";
    ctx.lineWidth = 1;
    strokePath([{ x: -1.15, y: 0, z: 0 }, { x: 1.15, y: 0, z: 0 }], cx, cy, scale);

    // surface wireframe: meridians (longitude curves) so it reads as a solid
    const XS = 40;
    const MER = 16;
    for (let m = 0; m < MER; m++) {
      const phi = (m / MER) * Math.PI * 2;
      const cphi = Math.cos(phi), sphi = Math.sin(phi);
      const pts = [];
      for (let i = 0; i <= XS; i++) {
        const x = -1 + (2 * i) / XS;
        const r = profileR(x, t);
        pts.push({ x, y: r * cphi, z: r * sphi });
      }
      // dim back, brighter front (use the meridian's mid-depth)
      const mid = rot(pts[Math.floor(pts.length / 2)]);
      const front = (mid.z + 1) / 2;                 // ~0 back .. ~1 front
      ctx.strokeStyle = `rgba(120,150,190,${0.10 + 0.22 * front})`;
      ctx.lineWidth = 1;
      strokePath(pts, cx, cy, scale);
    }

    // a couple of faint structural latitude rings (not constraints)
    ctx.strokeStyle = "rgba(120,150,190,0.10)";
    for (const gx of [-0.5, 0, 0.5]) strokePath(ringPoints(gx, profileR(gx, t), 48), cx, cy, scale);

    // constraint rings: marching from left toward the tip on +x
    for (let k = 0; k < n; k++) {
      const xpos = -0.55 + ((k + 0.5) / n) * 1.4;     // spread across the axis, leading right
      const rad = profileR(xpos, t);
      ctx.strokeStyle = rings[k].color;
      ctx.lineWidth = 2;
      strokePath(ringPoints(xpos, rad, 48), cx, cy, scale);
    }

    // tip marker (the surviving few)
    if (n > 0) {
      const tip = project(rot({ x: 1, y: 0, z: 0 }), cx, cy, scale);
      ctx.fillStyle = "#f0f3f8";
      ctx.beginPath(); ctx.arc(tip.X, tip.Y, 2.5, 0, Math.PI * 2); ctx.fill();
    }

    // caption
    ctx.fillStyle = "rgba(125,135,148,0.85)";
    ctx.font = '11px "Cascadia Code", Consolas, monospace';
    ctx.textAlign = "left"; ctx.textBaseline = "top";
    ctx.fillText(n === 0 ? "sphere · PO 0" : `cone · PO ${n}`, 10, 8);
  }

  // Static view (no auto-rotate); renders on demand — tab shown, resize, toggle.
  let visible = false;

  // -- tab switching -------------------------------------------------------
  function selectTab(name) {
    tabs.forEach((tb) => tb.classList.toggle("active", tb.dataset.tab === name));
    if (bodyMicro) bodyMicro.hidden = name !== "microscope";
    if (bodyPO) bodyPO.hidden = name !== "po";
    visible = name === "po";
    if (visible) { syncRings(); render(); }
  }
  tabs.forEach((tb) => tb.addEventListener("click", () => selectTab(tb.dataset.tab)));

  // recompute rings whenever a constraint is toggled
  for (const cb of document.querySelectorAll('input[name="constraint"]')) {
    cb.addEventListener("change", () => { syncRings(); if (visible) render(); });
  }

  new ResizeObserver(() => { if (visible) render(); }).observe(stage);
  syncRings();

  window.GENREG = window.GENREG || {};
  window.GENREG.po = { render, syncRings, selectTab };
})();
