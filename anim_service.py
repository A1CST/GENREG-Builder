"""Animation platform backend for the /video studio.

Rigs are JSON documents (layered SVG parts with pivots + semantic tags),
scenes are JSON timelines (actors + verb actions + overlays). Rendering
composes one SVG per frame, rasterizes it with resvg, and pipes the PNGs
straight into ffmpeg — the finished mp4 lands in the video library so it
can be cut/stitched with the existing editor.

The verb math here is mirrored in static/animrig.js (the browser preview);
if you change a verb, change both.

Rig format (origin = feet centre, y negative is up):
  {name, kind: character|object, canvas: {w, h},
   parts: [{id, tag, parent, offset:[x,y], pivot:[x,y], z,
            shape: {type: rect|ellipse|path, ...}, fill, stroke, sw}]}
Tags drive animation: body, head, arm_l, arm_r, leg_l, leg_r,
mouth_closed, mouth_half, mouth_open, other.

Scene format:
  {name, w, h, fps, dur, bg: {sky, floor, floor_y},
   actors: [{id, rig, x, y, scale, flip}],
   actions: [{actor, verb, t0, t1, args}],       verbs: walk, move, talk,
   overlays: [{type: caption|title|box, text, t0, t1, x, y, w}],  point, fade
   audio: "<library file>" (optional voiceover track)}
"""

import json
import math
import os
import random
import re
import subprocess
import threading
import time

import video_service

try:
    import resvg_py
    RASTER_OK, RASTER_ERR = True, None
except Exception as _e:                       # pragma: no cover
    RASTER_OK, RASTER_ERR = False, str(_e)

BASE = os.path.dirname(os.path.abspath(__file__))
RIG_DIR = os.path.join(BASE, "runs", "video", "rigs")
SCENE_DIR = os.path.join(BASE, "runs", "video", "scenes")
STORY_DIR = os.path.join(BASE, "runs", "video", "stories")
os.makedirs(RIG_DIR, exist_ok=True)
os.makedirs(SCENE_DIR, exist_ok=True)
os.makedirs(STORY_DIR, exist_ok=True)

TAGS = ["body", "head", "arm_l", "arm_r", "arm_l_lower", "arm_r_lower",
        "leg_l", "leg_r", "leg_l_lower", "leg_r_lower",
        "mouth_closed", "mouth_half", "mouth_open",
        "door", "hinge", "other"]
MOUTH_TAGS = {"mouth_closed", "mouth_half", "mouth_open"}

_slug = re.compile(r"[^a-z0-9_-]+")


def slug(name):
    s = _slug.sub("-", str(name or "").strip().lower()).strip("-")
    return s[:60] or "untitled"


# --------------------------------------------------------------------------
# Rig / scene stores
# --------------------------------------------------------------------------
def _store_list(directory):
    out = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, fn), "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            doc["_mtime"] = os.path.getmtime(os.path.join(directory, fn))
            out.append(doc)
        except (OSError, ValueError):
            continue
    out.sort(key=lambda d: -d["_mtime"])
    return out


def _store_save(directory, doc):
    doc = dict(doc)
    doc["name"] = slug(doc.get("name"))
    doc.pop("_mtime", None)
    with open(os.path.join(directory, doc["name"] + ".json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    return doc


def _store_delete(directory, name):
    path = os.path.join(directory, slug(name) + ".json")
    if os.path.isfile(path):
        os.unlink(path)
        return True
    return False


def list_rigs():
    return _store_list(RIG_DIR)


def save_rig(rig):
    if not isinstance(rig.get("parts"), list):
        raise ValueError("rig needs a parts list")
    return _store_save(RIG_DIR, rig)


def delete_rig(name):
    return _store_delete(RIG_DIR, name)


def get_rig(name):
    path = os.path.join(RIG_DIR, slug(name) + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_scenes():
    return _store_list(SCENE_DIR)


def get_scene(name):
    path = os.path.join(SCENE_DIR, slug(name) + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_stories():
    return _store_list(STORY_DIR)


def save_story(story):
    if not isinstance(story.get("shots"), list):
        raise ValueError("story needs a shots list (scene names)")
    return _store_save(STORY_DIR, story)


def delete_story(name):
    return _store_delete(STORY_DIR, name)


def save_scene(scene):
    return _store_save(SCENE_DIR, scene)


def delete_scene(name):
    return _store_delete(SCENE_DIR, name)


# --------------------------------------------------------------------------
# Procedural rig generator — "the lazy button"
# --------------------------------------------------------------------------
SKIN = ["#e8b990", "#d29b6e", "#b07a4f", "#8a5a36", "#f0cba6", "#6e4428"]
HAIR = ["#2b2118", "#4a3320", "#7a5230", "#9c9c9c", "#1a1a1a", "#5c4a68"]
SHIRT = ["#5577aa", "#7a8c69", "#a06868", "#8a7fae", "#6a9a9a", "#b08d57"]
PANTS = ["#3a4150", "#4d4438", "#2e3a2e", "#37324a", "#514646"]
INK = "#20242a"                     # outline colour for the flat style

CHAR_ARCHETYPES = ["researcher", "guard", "dclass", "suit", "civilian"]
OBJ_ARCHETYPES = ["crate", "table", "door", "terminal", "containment",
                  "tree", "bush", "rock", "desk", "plant", "whiteboard",
                  "building", "streetlight"]
ARCHETYPES = CHAR_ARCHETYPES + OBJ_ARCHETYPES
SCENE_TEMPLATES = ["basic", "office", "forest", "city"]


def _p(pid, tag, parent, offset, shape, fill, z=0, pivot=(0, 0), stroke=INK, sw=2):
    return {"id": pid, "tag": tag, "parent": parent, "offset": list(offset),
            "pivot": list(pivot), "z": z, "shape": shape, "fill": fill,
            "stroke": stroke, "sw": sw}


def _rect(x, y, w, h, rx=0):
    return {"type": "rect", "x": x, "y": y, "w": w, "h": h, "rx": rx}


def _ell(cx, cy, rx, ry):
    return {"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry}


def _path(d):
    return {"type": "path", "d": d}


def _humanoid(rng, torso_fill, pants_fill, extras):
    """Shared flat-humanoid skeleton; archetypes differ in colours/extras."""
    # SIDE PROFILE, facing +x (right); actor flip mirrors to face left.
    H = rng.uniform(175, 205)
    leg_h, torso_h = 0.42 * H, 0.40 * H
    head_r = 0.115 * H
    torso_w = rng.uniform(0.17, 0.22) * H
    leg_w, arm_w = 0.075 * H, 0.055 * H
    arm_h = torso_h * 0.88
    hip, shoulder_y = torso_w * 0.16, -(leg_h + torso_h) + 10
    skin = rng.choice(SKIN)

    # two-segment limbs: thigh/shin and upper-arm/forearm, hinged at the
    # knee/elbow (the lower segment's group origin IS the joint pivot)
    thigh_h, shin_h = leg_h * 0.53, leg_h * 0.47
    uarm_h, farm_h = arm_h * 0.52, arm_h * 0.48

    def _limb(side, sign, z_upper):
        return [
            _p(f"leg_{side}", f"leg_{side}", None, (sign * hip, -leg_h),
               _rect(-leg_w / 2, 0, leg_w, thigh_h + leg_w * 0.4, leg_w / 2),
               pants_fill, z=z_upper),
            _p(f"shin_{side}", f"leg_{side}_lower", f"leg_{side}", (0, thigh_h),
               _rect(-leg_w / 2, -leg_w * 0.4, leg_w, shin_h + leg_w * 0.4, leg_w / 2),
               pants_fill, z=1),
            _p(f"shoe_{side}", "other", f"shin_{side}", (0, shin_h - 7),
               _rect(-leg_w / 2 - 4, 0, leg_w + 8, 8, 3), "#1c1c20", z=1),
        ]

    parts = _limb("l", -1, 0) + _limb("r", 1, 1) + [
        _p("torso", "body", None, (0, 0),
           _rect(-torso_w / 2, -(leg_h + torso_h), torso_w, torso_h, 9), torso_fill, z=2),
        _p("arm_l", "arm_l", "torso", (-2, shoulder_y),
           _rect(-arm_w / 2, 0, arm_w, uarm_h + arm_w * 0.4, arm_w / 2), torso_fill, z=-1),
        _p("arm_r", "arm_r", "torso", (2, shoulder_y),
           _rect(-arm_w / 2, 0, arm_w, uarm_h + arm_w * 0.4, arm_w / 2), torso_fill, z=3),
        _p("forearm_l", "arm_l_lower", "arm_l", (0, uarm_h),
           _rect(-arm_w / 2, -arm_w * 0.4, arm_w, farm_h + arm_w * 0.4, arm_w / 2),
           torso_fill, z=1),
        _p("forearm_r", "arm_r_lower", "arm_r", (0, uarm_h),
           _rect(-arm_w / 2, -arm_w * 0.4, arm_w, farm_h + arm_w * 0.4, arm_w / 2),
           torso_fill, z=1),
        _p("hand_l", "other", "forearm_l", (0, farm_h - 4),
           _ell(0, 0, arm_w * 0.62, arm_w * 0.62), skin, z=1, sw=1.5),
        _p("hand_r", "other", "forearm_r", (0, farm_h - 4),
           _ell(0, 0, arm_w * 0.62, arm_w * 0.62), skin, z=1, sw=1.5),
        _p("head", "head", "torso", (0, -(leg_h + torso_h) - 4),
           _ell(0, -head_r, head_r * 0.92, head_r), skin, z=4),
        # profile face: one eye, a nose bump and the mouth on the leading edge
        _p("eye", "other", "head", (head_r * 0.42, -head_r * 1.05),
           _ell(0, 0, 2.7, 2.7), INK, z=1, sw=0),
        _p("nose", "other", "head", (0, 0),
           _path(f"M {head_r * 0.86} {-head_r * 0.88} l {head_r * 0.18} {head_r * 0.14} "
                 f"l {-head_r * 0.2} {head_r * 0.12}"), "none", z=1, stroke=INK, sw=1.8),
        _p("mouth_closed", "mouth_closed", "head", (head_r * 0.52, -head_r * 0.5),
           _rect(-head_r * 0.3, -1, head_r * 0.6, 2.4, 1.2), INK, z=1, sw=0),
        _p("mouth_half", "mouth_half", "head", (head_r * 0.52, -head_r * 0.5),
           _ell(0, 0, head_r * 0.24, head_r * 0.15), "#5a2f2f", z=1, sw=1),
        _p("mouth_open", "mouth_open", "head", (head_r * 0.5, -head_r * 0.48),
           _ell(0, 0, head_r * 0.27, head_r * 0.28), "#4a2424", z=1, sw=1),
    ]

    hair_kind = rng.choice(["flat", "side", "bald", "flat", "side"])
    if extras.get("hat") == "cap":
        parts.append(_p("cap", "other", "head", (-head_r * 0.1, -head_r * 1.42),
                        _rect(-head_r * 0.95, -head_r * 0.55, head_r * 1.9, head_r * 0.75, 4),
                        extras.get("hat_fill", "#2e3a55"), z=2))
        parts.append(_p("visor", "other", "head", (head_r * 0.35, -head_r * 0.88),
                        _rect(0, -3, head_r * 1.0, 6, 3),
                        extras.get("hat_fill", "#2e3a55"), z=2))
    elif hair_kind != "bald":
        hair = rng.choice(HAIR)
        # cap of hair over the top and down the back of the head
        parts.append(_p("hair", "other", "head", (0, 0),
                        _ell(-head_r * 0.18, -head_r * 1.48, head_r * 0.8, head_r * 0.52),
                        hair, z=2))
        parts.append(_p("hair_back", "other", "head", (0, 0),
                        _ell(-head_r * 0.62, -head_r * (1.0 if hair_kind == "flat" else 0.85),
                             head_r * 0.42, head_r * (0.62 if hair_kind == "flat" else 0.85)),
                        hair, z=2))
    if extras.get("glasses"):
        parts.append(_p("glasses", "other", "head", (0, -head_r * 1.05),
                        _path(f"M {-head_r * 0.45} 0 H {head_r * 0.25} "
                              f"m 0 -5 h {head_r * 0.55} v 10 h {-head_r * 0.55} Z"),
                        "none", z=2, stroke=INK, sw=1.6))
    for extra in extras.get("parts", []):
        parts.append(extra(torso_w, leg_h, torso_h, head_r))
    return parts, {"w": 2 * H, "h": H + 30}


def _character(archetype, rng):
    if archetype == "researcher":
        extras = {"glasses": rng.random() < 0.6, "parts": [
            lambda tw, lh, th, hr: _p("shirt", "other", "torso", (0, 0),
                                      _rect(-tw * 0.17, -(lh + th) + 6, tw * 0.34, th * 0.5),
                                      rng.choice(SHIRT), z=0.5),
            lambda tw, lh, th, hr: _p("badge", "other", "torso", (0, 0),
                                      _rect(tw * 0.18, -(lh + th * 0.72), 9, 12, 2),
                                      "#d8d2c2", z=0.6, sw=1)]}
        return _humanoid(rng, "#e9e6dd", rng.choice(PANTS), extras)
    if archetype == "guard":
        fill = "#33415e"
        extras = {"hat": "cap", "hat_fill": "#2b3650", "parts": [
            lambda tw, lh, th, hr: _p("vest", "other", "torso", (0, 0),
                                      _rect(-tw * 0.36, -(lh + th) + 8, tw * 0.72, th * 0.62, 6),
                                      "#242f45", z=0.5)]}
        return _humanoid(rng, fill, "#2b3245", extras)
    if archetype == "dclass":
        return _humanoid(rng, "#cd6a2e", "#cd6a2e", {"parts": [
            lambda tw, lh, th, hr: _p("dtag", "other", "torso", (0, 0),
                                      _rect(-tw * 0.2, -(lh + th * 0.66), tw * 0.4, 10, 2),
                                      "#f0e9d8", z=0.5, sw=1)]})
    if archetype == "suit":
        extras = {"glasses": rng.random() < 0.3, "parts": [
            lambda tw, lh, th, hr: _p("shirt", "other", "torso", (0, 0),
                                      _rect(-tw * 0.14, -(lh + th) + 6, tw * 0.28, th * 0.46),
                                      "#e9e6dd", z=0.5),
            lambda tw, lh, th, hr: _p("tie", "other", "torso", (0, 0),
                                      _path(f"M 0 {-(lh + th) + 8} l 5 10 l -5 {th * 0.34} "
                                            f"l -5 {-th * 0.34} Z"), "#7a2430", z=0.6, sw=1)]}
        return _humanoid(rng, "#2c2f38", rng.choice(PANTS), extras)
    # civilian
    return _humanoid(rng, rng.choice(SHIRT), rng.choice(PANTS),
                     {"glasses": rng.random() < 0.25})


def _object(archetype, rng):
    if archetype == "crate":
        w, h = rng.uniform(90, 150), rng.uniform(80, 120)
        fill = rng.choice(["#8a6f4d", "#7a6a55", "#6f7a55"])
        return [_p("box", "other", None, (0, 0), _rect(-w / 2, -h, w, h, 3), fill, z=0),
                _p("slat1", "other", "box", (0, 0), _rect(-w / 2, -h * 0.62, w, 8), fill, z=1),
                _p("slat2", "other", "box", (0, 0), _rect(-6, -h, 12, h), fill, z=1),
                ], {"w": w + 20, "h": h + 20}
    if archetype == "table":
        w, h = rng.uniform(140, 220), rng.uniform(70, 95)
        return [_p("leg1", "other", None, (-w / 2 + 12, -h), _rect(-5, 8, 10, h - 8), "#4a4438", z=0),
                _p("leg2", "other", None, (w / 2 - 12, -h), _rect(-5, 8, 10, h - 8), "#4a4438", z=0),
                _p("top", "other", None, (0, -h), _rect(-w / 2, 0, w, 10, 3), "#6e6250", z=1),
                ], {"w": w + 20, "h": h + 20}
    if archetype == "door":
        w, h = rng.uniform(70, 90), rng.uniform(170, 200)
        # panel is tagged "door": the open/close verbs slide it (dark opening
        # behind it shows through); default open slide is dx=-60
        return [_p("frame", "other", None, (0, 0), _rect(-w / 2 - 7, -h - 7, w + 14, h + 7, 2),
                   "#3a3f4a", z=0),
                _p("opening", "other", None, (0, 0), _rect(-w / 2, -h, w, h), "#14171e", z=1),
                _p("door", "door", None, (0, 0), _rect(-w / 2, -h, w, h), "#59606e", z=2),
                _p("handle", "other", "door", (0, 0), _ell(w * 0.32, -h * 0.48, 4, 4),
                   "#c9c4b4", z=1, sw=1),
                ], {"w": w + 30, "h": h + 20}
    if archetype == "terminal":
        w = rng.uniform(70, 100)
        return [_p("stand", "other", None, (0, 0), _rect(-8, -34, 16, 34), "#3a3f4a", z=0),
                _p("base", "other", None, (0, 0), _rect(-w * 0.35, -8, w * 0.7, 8, 3), "#3a3f4a", z=1),
                _p("case", "other", None, (0, -34), _rect(-w / 2, -w * 0.72, w, w * 0.72, 5),
                   "#2c313c", z=2),
                _p("screen", "other", "case", (0, 0), _rect(-w * 0.42, -w * 0.64, w * 0.84, w * 0.5),
                   "#264a38", z=1, sw=1),
                ], {"w": w + 20, "h": w + 60}
    if archetype == "containment":
        w, h = rng.uniform(220, 320), rng.uniform(200, 260)
        return [_p("cell", "other", None, (0, 0), _rect(-w / 2, -h, w, h, 4), "#4a505c", z=0),
                _p("window", "other", "cell", (0, 0),
                   _rect(-w * 0.28, -h * 0.78, w * 0.56, h * 0.4, 3), "#222834", z=1),
                _p("stripe", "other", "cell", (0, 0), _rect(-w / 2, -h * 0.22, w, 14),
                   "#c9a13c", z=1),
                ], {"w": w + 20, "h": h + 20}
    if archetype == "tree":
        th = rng.uniform(200, 300)
        trunk_h, trunk_w = th * 0.45, rng.uniform(14, 22)
        fol = rng.choice(["#3c5a3a", "#46653d", "#35513b", "#4d6b3f"])
        r0 = th * 0.30
        return [_p("trunk", "other", None, (0, 0),
                   _rect(-trunk_w / 2, -trunk_h, trunk_w, trunk_h, 3), "#5a4632", z=0),
                _p("can2", "other", None, (0, 0),
                   _ell(-r0 * 0.7, -trunk_h - r0 * 0.45, r0 * 0.72, r0 * 0.6), fol, z=1),
                _p("can3", "other", None, (0, 0),
                   _ell(r0 * 0.7, -trunk_h - r0 * 0.5, r0 * 0.75, r0 * 0.62), fol, z=1),
                _p("can1", "other", None, (0, 0),
                   _ell(0, -trunk_h - r0 * 0.85, r0, r0 * 0.9), fol, z=2),
                ], {"w": th, "h": th + 30}
    if archetype == "bush":
        bw = rng.uniform(60, 100)
        fol = rng.choice(["#3c5a3a", "#46653d", "#35513b"])
        return [_p("b1", "other", None, (0, 0), _ell(-bw * 0.25, -bw * 0.22, bw * 0.4, bw * 0.3),
                   fol, z=0),
                _p("b2", "other", None, (0, 0), _ell(bw * 0.25, -bw * 0.2, bw * 0.38, bw * 0.28),
                   fol, z=1),
                _p("b3", "other", None, (0, 0), _ell(0, -bw * 0.33, bw * 0.36, bw * 0.32), fol, z=2),
                ], {"w": bw + 20, "h": bw * 0.7 + 20}
    if archetype == "rock":
        rw = rng.uniform(40, 90)
        return [_p("rock", "other", None, (0, 0), _ell(0, -rw * 0.26, rw * 0.5, rw * 0.3),
                   "#5c6068", z=0),
                _p("hi", "other", None, (0, 0), _ell(-rw * 0.12, -rw * 0.34, rw * 0.2, rw * 0.1),
                   "#6d727c", z=1, sw=0),
                ], {"w": rw + 20, "h": rw * 0.6 + 20}
    if archetype == "desk":
        w, h = rng.uniform(160, 230), rng.uniform(72, 92)
        body = "#6e6250"
        return [_p("panel_l", "other", None, (-w / 2 + 10, -h), _rect(-8, 8, 16, h - 8), body, z=0),
                _p("drawers", "other", None, (w / 2 - w * 0.19, -h),
                   _rect(-w * 0.16, 8, w * 0.32, h - 8, 2), body, z=0),
                _p("d1", "other", "drawers", (0, 0), _rect(-w * 0.12, 16, w * 0.24, 12, 2),
                   "#7d7160", z=1, sw=1),
                _p("d2", "other", "drawers", (0, 0), _rect(-w * 0.12, 34, w * 0.24, 12, 2),
                   "#7d7160", z=1, sw=1),
                _p("top", "other", None, (0, -h), _rect(-w / 2, 0, w, 10, 3), "#7d7160", z=1),
                ], {"w": w + 20, "h": h + 20}
    if archetype == "plant":
        ph = rng.uniform(70, 110)
        fol = rng.choice(["#46653d", "#3c5a3a"])
        return [_p("pot", "other", None, (0, 0), _rect(-ph * 0.18, -ph * 0.3, ph * 0.36, ph * 0.3, 3),
                   "#8a5a3c", z=1),
                _p("l1", "other", None, (0, 0), _ell(-ph * 0.16, -ph * 0.62, ph * 0.16, ph * 0.3),
                   fol, z=0),
                _p("l2", "other", None, (0, 0), _ell(ph * 0.16, -ph * 0.6, ph * 0.16, ph * 0.28),
                   fol, z=0),
                _p("l3", "other", None, (0, 0), _ell(0, -ph * 0.72, ph * 0.14, ph * 0.34), fol, z=0),
                ], {"w": ph * 0.8 + 20, "h": ph + 20}
    if archetype == "whiteboard":
        w, h = rng.uniform(180, 260), rng.uniform(100, 130)
        parts = [_p("frame", "other", None, (0, 0), _rect(-w / 2, -h, w, h, 4), "#3a3f4a", z=0),
                 _p("board", "other", "frame", (0, 0),
                    _rect(-w / 2 + 8, -h + 8, w - 16, h - 16, 2), "#dcd8cc", z=1, sw=1)]
        y0 = -h + 26
        for i in range(3):
            lw = w * rng.uniform(0.3, 0.62)
            parts.append(_p(f"line{i}", "other", "frame", (0, 0),
                            _path(f"M {-w / 2 + 20} {y0 + i * 20} h {lw:.0f}"), "none",
                            z=2, stroke=rng.choice(["#41546e", "#6e4141", "#20242a"]), sw=2))
        return parts, {"w": w + 20, "h": h + 20}
    if archetype == "building":
        w, h = rng.uniform(180, 280), rng.uniform(300, 430)
        body = rng.choice(["#3b4252", "#464e5e", "#3f4450", "#525866"])
        win_on = "#c9b06a"
        win_off = "#2a3040"
        parts = [_p("body", "other", None, (0, 0), _rect(-w / 2, -h, w, h), body, z=0),
                 _p("roof", "other", "body", (0, 0), _rect(-w / 2 - 6, -h - 10, w + 12, 10, 2),
                    body, z=1)]
        cols, rows = rng.randint(3, 4), rng.randint(4, 6)
        ww, wh = w / (cols * 2), h / (rows * 2.6)
        for r in range(rows):
            for c in range(cols):
                x = -w / 2 + w * (c + 0.5) / cols - ww / 2
                y = -h + h * (r + 0.35) / (rows + 0.5)
                lit = rng.random() < 0.35
                parts.append(_p(f"w{r}_{c}", "other", "body", (0, 0),
                                _rect(x, y, ww, wh, 1), win_on if lit else win_off, z=1, sw=1))
        return parts, {"w": w + 20, "h": h + 30}
    # streetlight
    lh = rng.uniform(220, 280)
    return [_p("pole", "other", None, (0, 0), _rect(-4, -lh, 8, lh, 2), "#3a3f47", z=0),
            _p("base", "other", None, (0, 0), _rect(-12, -8, 24, 8, 2), "#3a3f47", z=1),
            _p("arm", "other", "pole", (0, 0), _rect(0, -lh, lh * 0.28, 7, 3), "#3a3f47", z=1),
            _p("head", "other", "pole", (lh * 0.28, -lh + 3), _ell(0, 6, 14, 8), "#e0c477",
               z=2, sw=1.5),
            ], {"w": lh * 0.7 + 30, "h": lh + 20}


def generate_rig(archetype, seed=None, name=""):
    archetype = str(archetype or "").lower()
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype: {archetype} (have {', '.join(ARCHETYPES)})")
    seed = int(seed) if seed not in (None, "") else random.randrange(1 << 30)
    rng = random.Random(seed)
    is_char = archetype in CHAR_ARCHETYPES
    parts, canvas = (_character if is_char else _object)(archetype, rng)
    rig = {"name": slug(name or f"{archetype}-{seed % 10000}"),
           "kind": "character" if is_char else "object",
           "archetype": archetype, "seed": seed, "canvas": canvas, "parts": parts}
    return save_rig(rig)


# --------------------------------------------------------------------------
# Scene templates — a bg palette + procedurally generated prop rigs placed
# as objects. Props are ordinary rigs (named <scene>-<archetype>-<n>) so the
# scene stays fully editable; regenerating with the same name overwrites them.
# --------------------------------------------------------------------------
def generate_scene(template, seed=None, name=""):
    template = str(template or "").lower()
    if template not in SCENE_TEMPLATES:
        raise ValueError(f"unknown template: {template} (have {', '.join(SCENE_TEMPLATES)})")
    seed = int(seed) if seed not in (None, "") else random.randrange(1 << 30)
    rng = random.Random(seed)
    name = slug(name or f"{template}-{seed % 10000}")
    w, h = 1280, 720
    bg = {"basic":  {"sky": "#242a36", "floor": "#3a4150", "floor_y": 0.82},
          "office": {"sky": "#3a4050", "floor": "#463f36", "floor_y": 0.80},
          "forest": {"sky": "#22302c", "floor": "#2f3d2c", "floor_y": 0.78},
          "city":   {"sky": "#262c3a", "floor": "#33363e", "floor_y": 0.80}}[template]
    fy = bg["floor_y"] * h
    actors = []

    def prop(arch, x, y, scale=1.0, flip=False):
        rig = generate_rig(arch, seed=rng.randrange(1 << 30),
                           name=f"{name}-{arch}-{len(actors) + 1}")
        actors.append({"id": f"o{len(actors) + 1}", "rig": rig["name"],
                       "x": round(x), "y": round(y), "scale": scale, "flip": flip})

    if template == "office":
        prop("whiteboard", w * rng.uniform(0.16, 0.3), fy - 60)     # hangs on the wall
        prop("desk", w * 0.72, fy + 60)
        prop("terminal", w * 0.72 + 24, fy + 52, scale=0.8)         # peeks over the desk
        prop("plant", w * rng.uniform(0.9, 0.95), fy + 55)
        prop("door", w * rng.uniform(0.42, 0.5), fy + 8)
    elif template == "forest":
        for i in range(rng.randint(3, 5)):                          # back row of trees
            prop("tree", w * (0.08 + 0.9 * i / 4 + rng.uniform(-0.04, 0.04)),
                 fy + rng.uniform(4, 26), scale=rng.uniform(0.8, 1.2))
        for _ in range(rng.randint(2, 3)):
            prop("bush", w * rng.uniform(0.1, 0.9), fy + rng.uniform(40, 90))
        prop("rock", w * rng.uniform(0.25, 0.75), fy + rng.uniform(60, 100))
    elif template == "city":
        xs = sorted(rng.uniform(0.08, 0.92) for _ in range(rng.randint(3, 4)))
        for x in xs:                                                # skyline at the horizon
            prop("building", w * x, fy + 6, scale=rng.uniform(0.85, 1.15))
        prop("streetlight", w * rng.uniform(0.12, 0.2), fy + 70)
        prop("streetlight", w * rng.uniform(0.78, 0.88), fy + 70, flip=True)
    # "basic": palette only, no props

    scene = {"name": name, "w": w, "h": h, "fps": 24, "dur": 8, "bg": bg,
             "actors": actors, "actions": [], "overlays": [], "audio": "",
             "template": template, "seed": seed}
    return save_scene(scene)


# --------------------------------------------------------------------------
# Pose / verb math — MIRRORED in static/animrig.js
# --------------------------------------------------------------------------
def _lerp(a, b, u):
    return a + (b - a) * max(0.0, min(1.0, u))


MOUTH_SEQ = ["open", "half", "open", "closed", "open", "half"]


def actor_state(actor, actions, t, is_char):
    """Position / facing / pose of one actor at time t (verb math)."""
    x, y = float(actor.get("x", 0)), float(actor.get("y", 0))
    flip = bool(actor.get("flip"))
    opacity = 1.0
    rot = {}
    mouth = "closed"
    dy = math.sin(2 * math.pi * 0.5 * t) * 1.5 if is_char else 0.0
    walking = False
    # open/close state: openness 0..1 persists between actions; amplitudes
    # come from the most recent open action ("door" parts slide, "hinge"
    # parts rotate around their pivot)
    openness, amp_dx, amp_dy, amp_ang = 0.0, -60.0, 0.0, -100.0

    for a in sorted([a for a in actions if a.get("actor") == actor.get("id")],
                    key=lambda a: float(a.get("t0", 0))):
        t0, t1 = float(a.get("t0", 0)), float(a.get("t1", 0))
        if t1 <= t0 or t < t0:
            continue
        u = min(1.0, (t - t0) / (t1 - t0))
        verb, args = a.get("verb"), a.get("args") or {}
        if verb == "walk":
            to_x = float(args.get("to_x", x))
            if abs(to_x - x) > 0.5:
                flip = to_x < x
            if t <= t1:
                walking = True
                ph = 2 * math.pi * 1.6 * (t - t0)
                s = math.sin(ph)
                rot["leg_l"] = 24 * s
                rot["leg_r"] = -24 * s
                # knee flexes while its leg swings, straight at the pass
                rot["leg_l_lower"] = -20 * max(0.0, s)
                rot["leg_r_lower"] = -20 * max(0.0, -s)
                rot["arm_l"] = -16 * s
                rot["arm_r"] = 16 * s
                # elbow keeps a soft bend opposite the upper-arm swing
                rot["arm_l_lower"] = -10 - 8 * max(0.0, s)
                rot["arm_r_lower"] = 10 + 8 * max(0.0, -s)
                dy = -3 * abs(s)
            x = _lerp(x, to_x, u)
        elif verb == "move":
            x = _lerp(x, float(args.get("to_x", x)), u)
            y = _lerp(y, float(args.get("to_y", y)), u)
        elif verb == "talk" and t <= t1:
            mouth = MOUTH_SEQ[int((t - t0) * 8) % len(MOUTH_SEQ)]
        elif verb == "point" and t <= t1:
            arm = "arm_r" if args.get("arm", "r") == "r" else "arm_l"
            ramp = 0.35
            if t - t0 < ramp:
                ang = _lerp(0, -75, (t - t0) / ramp)
            elif t1 - t < ramp:
                ang = _lerp(0, -75, (t1 - t) / ramp)
            else:
                ang = -75
            rot[arm] = ang if arm == "arm_r" else -ang
        elif verb == "fade":
            opacity = _lerp(float(args.get("from", 1)), float(args.get("to", 0)), u)
        elif verb in ("open", "close"):
            if verb == "open":
                amp_dx = float(args.get("dx", amp_dx))
                amp_dy = float(args.get("dy", amp_dy))
                amp_ang = float(args.get("angle", amp_ang))
            openness = _lerp(openness, 1.0 if verb == "open" else 0.0, u)
    trans = {}
    if openness:
        trans["door"] = (amp_dx * openness, amp_dy * openness)
        rot["hinge"] = rot.get("hinge", 0) + amp_ang * openness
    return {"x": x, "y": y, "flip": flip, "opacity": opacity, "rot": rot,
            "trans": trans, "mouth": mouth, "dy": dy, "walking": walking}


# --------------------------------------------------------------------------
# SVG composition — MIRRORED in static/animrig.js
# --------------------------------------------------------------------------
def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _shape_svg(part):
    sh = part.get("shape") or {}
    fill = part.get("fill", "#888")
    stroke = part.get("stroke", "none")
    sw = part.get("sw", 0)
    style = f'fill="{_esc(fill)}"'
    if stroke and stroke != "none" and sw:
        style += f' stroke="{_esc(stroke)}" stroke-width="{sw}" stroke-linejoin="round"'
    kind = sh.get("type")
    if kind == "rect":
        rx = f' rx="{sh.get("rx", 0)}"' if sh.get("rx") else ""
        return (f'<rect x="{sh.get("x", 0)}" y="{sh.get("y", 0)}" width="{sh.get("w", 10)}" '
                f'height="{sh.get("h", 10)}"{rx} {style}/>')
    if kind == "ellipse":
        return (f'<ellipse cx="{sh.get("cx", 0)}" cy="{sh.get("cy", 0)}" '
                f'rx="{sh.get("rx", 5)}" ry="{sh.get("ry", 5)}" {style}/>')
    if kind == "path":
        return f'<path d="{_esc(sh.get("d", ""))}" {style} stroke-linecap="round"/>'
    return ""


def rig_svg(rig, pose=None):
    """The rig as a <g> in local space (origin = feet), posed."""
    pose = pose or {}
    rot = pose.get("rot") or {}
    trans = pose.get("trans") or {}
    mouth = pose.get("mouth", "closed")
    parts = rig.get("parts") or []
    kids, roots = {}, []
    for p in parts:
        if p.get("parent"):
            kids.setdefault(p["parent"], []).append(p)
        else:
            roots.append(p)

    def emit(p):
        tag = p.get("tag", "other")
        if tag in MOUTH_TAGS and tag != f"mouth_{mouth}":
            return ""
        ox, oy = p.get("offset", [0, 0])
        px, py = p.get("pivot", [0, 0])
        tx, ty = trans.get(tag, (0, 0))
        # verb rotation + the part's authored base rotation (both subtree-wide)
        ang = rot.get(tag, 0) + float(p.get("rot", 0) or 0)
        tf = f"translate({ox + tx},{oy + ty})"
        if ang:
            tf += f" rotate({ang},{px},{py})"
        shape = _shape_svg(p)
        scl = float(p.get("scale", 1) or 1)
        if scl != 1:                       # scales the shape only, not children
            shape = f'<g transform="scale({scl})">{shape}</g>'
        children = sorted(kids.get(p.get("id"), []), key=lambda c: c.get("z", 0))
        inner = ("".join(emit(c) for c in children if c.get("z", 0) < 0)
                 + shape
                 + "".join(emit(c) for c in children if c.get("z", 0) >= 0))
        return f'<g transform="{tf}">{inner}</g>'

    return "".join(emit(p) for p in sorted(roots, key=lambda r: r.get("z", 0)))


def _overlay_svg(ov, t, w, h):
    t0, t1 = float(ov.get("t0", 0)), float(ov.get("t1", 0))
    if not (t0 <= t <= t1):
        return ""
    a = min(1.0, min(t - t0, t1 - t) / 0.25) if t1 - t0 > 0.5 else 1.0
    text = str(ov.get("text", ""))
    lines = text.split("\n")
    kind = ov.get("type", "caption")
    font = 'font-family="Arial, Helvetica, sans-serif"'
    if kind == "title":
        fs = int(h * 0.07)
        tspan = "".join(f'<tspan x="{w / 2}" dy="{fs * 1.25 if i else 0}">{_esc(ln)}</tspan>'
                        for i, ln in enumerate(lines))
        return (f'<g opacity="{a:.3f}"><text x="{w / 2}" y="{h * 0.44}" {font} font-size="{fs}" '
                f'font-weight="bold" fill="#e9e6dd" text-anchor="middle">{tspan}</text></g>')
    if kind == "box":
        x, y = float(ov.get("x", w * 0.06)), float(ov.get("y", h * 0.1))
        bw = float(ov.get("w", w * 0.3))
        fs, pad = 15, 10
        bh = pad * 2 + len(lines) * fs * 1.35
        tspan = "".join(f'<tspan x="{x + pad}" dy="{fs * 1.35 if i else fs}">{_esc(ln)}</tspan>'
                        for i, ln in enumerate(lines))
        return (f'<g opacity="{a:.3f}"><rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" '
                f'fill="#141821" fill-opacity="0.88" stroke="#c9a13c" stroke-width="1.5"/>'
                f'<text x="{x + pad}" y="{y + pad}" {font} font-size="{fs}" '
                f'fill="#e9e6dd">{tspan}</text></g>')
    # caption
    fs = int(h * 0.037)
    bh = fs * 1.6 * len(lines) + fs * 0.8
    y0 = h - bh - h * 0.04
    tspan = "".join(f'<tspan x="{w / 2}" dy="{fs * 1.6 if i else fs * 1.15}">{_esc(ln)}</tspan>'
                    for i, ln in enumerate(lines))
    return (f'<g opacity="{a:.3f}"><rect x="{w * 0.08}" y="{y0}" width="{w * 0.84}" '
            f'height="{bh}" rx="8" fill="#10141c" fill-opacity="0.82"/>'
            f'<text x="{w / 2}" y="{y0}" {font} font-size="{fs}" fill="#f0ede4" '
            f'text-anchor="middle">{tspan}</text></g>')


def scene_svg(scene, rigs_by_name, t):
    """One full frame as an SVG document string."""
    w, h = int(scene.get("w", 1280)), int(scene.get("h", 720))
    bg = scene.get("bg") or {}
    floor_y = float(bg.get("floor_y", 0.82)) * h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="{_esc(bg.get("sky", "#242a36"))}"/>',
           f'<rect y="{floor_y}" width="{w}" height="{h - floor_y}" '
           f'fill="{_esc(bg.get("floor", "#3a4150"))}"/>']
    actions = scene.get("actions") or []
    actors = list(scene.get("actors") or [])
    states = []
    for actor in actors:
        rig = rigs_by_name.get(actor.get("rig"))
        if not rig:
            continue
        st = actor_state(actor, actions, t, rig.get("kind") == "character")
        states.append((actor, rig, st))
    states.sort(key=lambda s: s[2]["y"])          # painter order: lower = nearer
    for actor, rig, st in states:
        s = float(actor.get("scale", 1))
        sx = -s if st["flip"] else s
        out.append(f'<g transform="translate({st["x"]:.2f},{st["y"] + st["dy"]:.2f}) '
                   f'scale({sx:.3f},{s:.3f})" opacity="{st["opacity"]:.3f}">'
                   + rig_svg(rig, st) + "</g>")
    for ov in scene.get("overlays") or []:
        out.append(_overlay_svg(ov, t, w, h))
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# Rendering — frames piped into ffmpeg, output lands in the video library
# --------------------------------------------------------------------------
def render_scene(scene):
    if not RASTER_OK:
        raise RuntimeError(f"resvg unavailable: {RASTER_ERR} (pip install resvg-py)")
    if not video_service.available():
        raise RuntimeError("ffmpeg unavailable")
    fps = max(1, min(60, int(scene.get("fps", 24))))
    dur = max(0.1, min(600.0, float(scene.get("dur", 5))))
    total = int(round(fps * dur))
    rigs = {}
    for actor in scene.get("actors") or []:
        rname = actor.get("rig")
        if rname not in rigs:
            rig = get_rig(rname)
            if not rig:
                raise ValueError(f"actor uses unknown rig: {rname}")
            rigs[rname] = rig

    audio = None
    if scene.get("audio"):
        audio = os.path.join(video_service.LIB_DIR,
                             video_service.safe_name(scene["audio"]))
        if not os.path.isfile(audio):
            raise ValueError(f"audio file not in library: {scene['audio']}")

    name = slug(scene.get("name") or "scene")
    out = video_service.unique_path(name + ".mp4")
    cmd = [video_service.FFMPEG, "-y",
           "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-i", audio]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [out]

    job = video_service.custom_job("animate", f"{name} · {dur:.1f}s @ {fps}fps", out)

    def run():
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    creationflags=video_service.CREATE_NO_WINDOW)
            job["proc"] = proc
            for i in range(total):
                if job.get("cancelled"):
                    break
                svg = scene_svg(scene, rigs, i / fps)
                png = bytes(resvg_py.svg_to_bytes(svg_string=svg))
                proc.stdin.write(png)
                job["progress"] = (i + 1) / total
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            proc.wait()
            job["proc"] = None
            if job.get("cancelled"):
                job.update(status="cancelled", message="cancelled")
                try:
                    os.unlink(out)
                except OSError:
                    pass
            elif proc.returncode == 0:
                job.update(status="done", progress=1.0)
                video_service.invalidate_meta(os.path.basename(out))
            else:
                job.update(status="error",
                           message="\n".join(err.strip().splitlines()[-6:]))
        except Exception as exc:
            job.update(status="error", message=str(exc))
        job["finished"] = time.time()

    threading.Thread(target=run, daemon=True, name="anim-render").start()
    return job


def render_story(shots, out_name=""):
    """Render an ordered list of saved scenes into ONE mp4 (the storyboard).

    All shots are rasterized at the first scene's size and fps and piped into
    a single encoder, so the output needs no stitching pass. Audio is not
    handled here — voiceover gets muxed later in the editor.
    """
    if not RASTER_OK:
        raise RuntimeError(f"resvg unavailable: {RASTER_ERR} (pip install resvg-py)")
    if not video_service.available():
        raise RuntimeError("ffmpeg unavailable")
    scenes = []
    for sn in shots or []:
        s = get_scene(str(sn))
        if not s:
            raise ValueError(f"no such scene: {sn}")
        scenes.append(s)
    if not scenes:
        raise ValueError("storyboard has no shots")
    w = int(scenes[0].get("w", 1280))
    h = int(scenes[0].get("h", 720))
    fps = max(1, min(60, int(scenes[0].get("fps", 24))))
    counts = [int(round(max(0.1, min(600.0, float(s.get("dur", 5)))) * fps))
              for s in scenes]
    total = sum(counts)
    rigcache = {}
    for s in scenes:
        for actor in s.get("actors") or []:
            rname = actor.get("rig")
            if rname not in rigcache:
                rig = get_rig(rname)
                if not rig:
                    raise ValueError(f"scene {s.get('name')}: unknown rig {rname}")
                rigcache[rname] = rig

    name = slug(out_name or (scenes[0].get("name", "story") + "-story"))
    out = video_service.unique_path(name + ".mp4")
    cmd = [video_service.FFMPEG, "-y",
           "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-preset", "medium", out]
    job = video_service.custom_job("story", f"{name} · {len(scenes)} shot(s)", out)

    def run():
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    creationflags=video_service.CREATE_NO_WINDOW)
            job["proc"] = proc
            done = 0
            for s, cnt in zip(scenes, counts):
                for i in range(cnt):
                    if job.get("cancelled"):
                        break
                    svg = scene_svg(s, rigcache, i / fps)
                    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h))
                    proc.stdin.write(png)
                    done += 1
                    job["progress"] = done / total
                if job.get("cancelled"):
                    break
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            proc.wait()
            job["proc"] = None
            if job.get("cancelled"):
                job.update(status="cancelled", message="cancelled")
                try:
                    os.unlink(out)
                except OSError:
                    pass
            elif proc.returncode == 0:
                job.update(status="done", progress=1.0)
                video_service.invalidate_meta(os.path.basename(out))
            else:
                job.update(status="error",
                           message="\n".join(err.strip().splitlines()[-6:]))
        except Exception as exc:
            job.update(status="error", message=str(exc))
        job["finished"] = time.time()

    threading.Thread(target=run, daemon=True, name="anim-story").start()
    return job
