// AnimRig — shared rig/scene math for the /video animation studio previews.
// This is a JS mirror of the SVG composition + verb math in anim_service.py;
// if you change a verb or the part transform model, change BOTH files.
(function () {
  const MOUTH_TAGS = new Set(["mouth_closed", "mouth_half", "mouth_open"]);
  const MOUTH_SEQ = ["open", "half", "open", "closed", "open", "half"];

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  const lerp = (a, b, u) => a + (b - a) * Math.max(0, Math.min(1, u));

  function shapeSVG(part) {
    const sh = part.shape || {};
    let style = `fill="${esc(part.fill || "#888")}"`;
    if (part.stroke && part.stroke !== "none" && part.sw) {
      style += ` stroke="${esc(part.stroke)}" stroke-width="${part.sw}" stroke-linejoin="round"`;
    }
    if (sh.type === "rect") {
      const rx = sh.rx ? ` rx="${sh.rx}"` : "";
      return `<rect x="${sh.x || 0}" y="${sh.y || 0}" width="${sh.w || 10}" height="${sh.h || 10}"${rx} ${style}/>`;
    }
    if (sh.type === "ellipse") {
      return `<ellipse cx="${sh.cx || 0}" cy="${sh.cy || 0}" rx="${sh.rx || 5}" ry="${sh.ry || 5}" ${style}/>`;
    }
    if (sh.type === "path") {
      return `<path d="${esc(sh.d || "")}" ${style} stroke-linecap="round"/>`;
    }
    return "";
  }

  // The rig as <g> markup in local space (origin = feet), posed.
  // withIds adds data-pid attributes so the rig editor can hit-test parts.
  function rigSVG(rig, pose, withIds) {
    pose = pose || {};
    const rot = pose.rot || {};
    const trans = pose.trans || {};
    const mouth = pose.mouth || "closed";
    const kids = {}, roots = [];
    (rig.parts || []).forEach((p) => {
      if (p.parent) (kids[p.parent] = kids[p.parent] || []).push(p);
      else roots.push(p);
    });
    const emit = (p) => {
      const tag = p.tag || "other";
      if (MOUTH_TAGS.has(tag) && tag !== "mouth_" + mouth) return "";
      const [ox, oy] = p.offset || [0, 0];
      const [px, py] = p.pivot || [0, 0];
      const [tx, ty] = trans[tag] || [0, 0];
      // verb rotation + the part's authored base rotation (both subtree-wide)
      const ang = (rot[tag] || 0) + (Number(p.rot) || 0);
      let tf = `translate(${ox + tx},${oy + ty})`;
      if (ang) tf += ` rotate(${ang},${px},${py})`;
      let shape = shapeSVG(p);
      const scl = p.scale !== undefined ? Number(p.scale) || 1 : 1;
      if (scl !== 1) shape = `<g transform="scale(${scl})">${shape}</g>`;   // shape only
      const ch = (kids[p.id] || []).slice().sort((a, b) => (a.z || 0) - (b.z || 0));
      const inner = ch.filter((c) => (c.z || 0) < 0).map(emit).join("")
        + shape
        + ch.filter((c) => (c.z || 0) >= 0).map(emit).join("");
      const idAttr = withIds ? ` data-pid="${esc(p.id)}"` : "";
      return `<g${idAttr} transform="${tf}">${inner}</g>`;
    };
    return roots.slice().sort((a, b) => (a.z || 0) - (b.z || 0)).map(emit).join("");
  }

  // Position / facing / pose of one actor at time t — verb math.
  function actorState(actor, actions, t, isChar) {
    let x = Number(actor.x) || 0, y = Number(actor.y) || 0;
    let flip = !!actor.flip, opacity = 1;
    const rot = {};
    let mouth = "closed";
    let dy = isChar ? Math.sin(2 * Math.PI * 0.5 * t) * 1.5 : 0;
    let walking = false;
    // open/close state: openness 0..1 persists between actions; amplitudes
    // come from the most recent open action ("door" slides, "hinge" rotates)
    let openness = 0, ampDx = -60, ampDy = 0, ampAng = -100;

    const mine = (actions || []).filter((a) => a.actor === actor.id)
      .slice().sort((a, b) => (Number(a.t0) || 0) - (Number(b.t0) || 0));
    for (const a of mine) {
      const t0 = Number(a.t0) || 0, t1 = Number(a.t1) || 0;
      if (t1 <= t0 || t < t0) continue;
      const u = Math.min(1, (t - t0) / (t1 - t0));
      const args = a.args || {};
      if (a.verb === "walk") {
        const toX = args.to_x !== undefined ? Number(args.to_x) : x;
        if (Math.abs(toX - x) > 0.5) flip = toX < x;
        if (t <= t1) {
          walking = true;
          const ph = 2 * Math.PI * 1.6 * (t - t0);
          const s = Math.sin(ph);
          rot.leg_l = 24 * s;
          rot.leg_r = -24 * s;
          // knee flexes while its leg swings, straight at the pass
          rot.leg_l_lower = -20 * Math.max(0, s);
          rot.leg_r_lower = -20 * Math.max(0, -s);
          rot.arm_l = -16 * s;
          rot.arm_r = 16 * s;
          // elbow keeps a soft bend opposite the upper-arm swing
          rot.arm_l_lower = -10 - 8 * Math.max(0, s);
          rot.arm_r_lower = 10 + 8 * Math.max(0, -s);
          dy = -3 * Math.abs(s);
        }
        x = lerp(x, toX, u);
      } else if (a.verb === "move") {
        x = lerp(x, args.to_x !== undefined ? Number(args.to_x) : x, u);
        y = lerp(y, args.to_y !== undefined ? Number(args.to_y) : y, u);
      } else if (a.verb === "talk" && t <= t1) {
        mouth = MOUTH_SEQ[Math.floor((t - t0) * 8) % MOUTH_SEQ.length];
      } else if (a.verb === "point" && t <= t1) {
        const arm = (args.arm || "r") === "r" ? "arm_r" : "arm_l";
        const ramp = 0.35;
        let ang;
        if (t - t0 < ramp) ang = lerp(0, -75, (t - t0) / ramp);
        else if (t1 - t < ramp) ang = lerp(0, -75, (t1 - t) / ramp);
        else ang = -75;
        rot[arm] = arm === "arm_r" ? ang : -ang;
      } else if (a.verb === "fade") {
        opacity = lerp(args.from !== undefined ? Number(args.from) : 1,
          args.to !== undefined ? Number(args.to) : 0, u);
      } else if (a.verb === "open" || a.verb === "close") {
        if (a.verb === "open") {
          if (args.dx !== undefined) ampDx = Number(args.dx) || 0;
          if (args.dy !== undefined) ampDy = Number(args.dy) || 0;
          if (args.angle !== undefined) ampAng = Number(args.angle) || 0;
        }
        openness = lerp(openness, a.verb === "open" ? 1 : 0, u);
      }
    }
    const trans = {};
    if (openness) {
      trans.door = [ampDx * openness, ampDy * openness];
      rot.hinge = (rot.hinge || 0) + ampAng * openness;
    }
    return { x, y, flip, opacity, rot, trans, mouth, dy, walking };
  }

  function overlaySVG(ov, t, w, h) {
    const t0 = Number(ov.t0) || 0, t1 = Number(ov.t1) || 0;
    if (t < t0 || t > t1) return "";
    const a = t1 - t0 > 0.5 ? Math.min(1, Math.min(t - t0, t1 - t) / 0.25) : 1;
    const lines = String(ov.text || "").split("\n");
    const font = 'font-family="Arial, Helvetica, sans-serif"';
    if (ov.type === "title") {
      const fs = Math.round(h * 0.07);
      const ts = lines.map((ln, i) =>
        `<tspan x="${w / 2}" dy="${i ? fs * 1.25 : 0}">${esc(ln)}</tspan>`).join("");
      return `<g opacity="${a.toFixed(3)}"><text x="${w / 2}" y="${h * 0.44}" ${font} font-size="${fs}" font-weight="bold" fill="#e9e6dd" text-anchor="middle">${ts}</text></g>`;
    }
    if (ov.type === "box") {
      const x = ov.x !== undefined ? Number(ov.x) : w * 0.06;
      const y = ov.y !== undefined ? Number(ov.y) : h * 0.1;
      const bw = ov.w !== undefined ? Number(ov.w) : w * 0.3;
      const fs = 15, pad = 10;
      const bh = pad * 2 + lines.length * fs * 1.35;
      const ts = lines.map((ln, i) =>
        `<tspan x="${x + pad}" dy="${i ? fs * 1.35 : fs}">${esc(ln)}</tspan>`).join("");
      return `<g opacity="${a.toFixed(3)}"><rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="6" fill="#141821" fill-opacity="0.88" stroke="#c9a13c" stroke-width="1.5"/><text x="${x + pad}" y="${y + pad}" ${font} font-size="${fs}" fill="#e9e6dd">${ts}</text></g>`;
    }
    const fs = Math.round(h * 0.037);
    const bh = fs * 1.6 * lines.length + fs * 0.8;
    const y0 = h - bh - h * 0.04;
    const ts = lines.map((ln, i) =>
      `<tspan x="${w / 2}" dy="${i ? fs * 1.6 : fs * 1.15}">${esc(ln)}</tspan>`).join("");
    return `<g opacity="${a.toFixed(3)}"><rect x="${w * 0.08}" y="${y0}" width="${w * 0.84}" height="${bh}" rx="8" fill="#10141c" fill-opacity="0.82"/><text x="${w / 2}" y="${y0}" ${font} font-size="${fs}" fill="#f0ede4" text-anchor="middle">${ts}</text></g>`;
  }

  // Inner markup of one full frame (goes inside an <svg viewBox="0 0 w h">).
  // withIds tags each actor group with data-aid for stage drag/hit-testing.
  function sceneSVG(scene, rigsByName, t, withIds) {
    const w = Number(scene.w) || 1280, h = Number(scene.h) || 720;
    const bg = scene.bg || {};
    const floorY = (bg.floor_y !== undefined ? Number(bg.floor_y) : 0.82) * h;
    const out = [
      `<rect width="${w}" height="${h}" fill="${esc(bg.sky || "#242a36")}"/>`,
      `<rect y="${floorY}" width="${w}" height="${h - floorY}" fill="${esc(bg.floor || "#3a4150")}"/>`,
    ];
    const states = [];
    for (const actor of scene.actors || []) {
      const rig = rigsByName[actor.rig];
      if (!rig) continue;
      states.push([actor, rig,
        actorState(actor, scene.actions || [], t, rig.kind === "character")]);
    }
    states.sort((p, q) => p[2].y - q[2].y);
    for (const [actor, rig, st] of states) {
      const s = Number(actor.scale) || 1;
      const sx = st.flip ? -s : s;
      const idAttr = withIds ? ` data-aid="${esc(actor.id)}"` : "";
      out.push(`<g${idAttr} transform="translate(${st.x.toFixed(2)},${(st.y + st.dy).toFixed(2)}) scale(${sx.toFixed(3)},${s.toFixed(3)})" opacity="${st.opacity.toFixed(3)}">`
        + rigSVG(rig, st) + "</g>");
    }
    for (const ov of scene.overlays || []) out.push(overlaySVG(ov, t, w, h));
    return out.join("");
  }

  window.AnimRig = { rigSVG, actorState, sceneSVG, overlaySVG };
})();
