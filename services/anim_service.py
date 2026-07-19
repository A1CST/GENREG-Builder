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
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401

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

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

CHAR_ARCHETYPES = ["researcher", "guard", "dclass", "suit", "civilian", "scientist", "professor", "robot", "cyborg"]
OBJ_ARCHETYPES = ["crate", "table", "door", "terminal", "containment",
                  "tree", "bush", "rock", "desk", "plant", "whiteboard",
                  "building", "streetlight", "skyline", "street", "house",
                  "skyscraper", "lab_building", "server_rack"]
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
    skin = extras.get("skin") or rng.choice(SKIN)

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

    hair_kind = extras.get("hair_kind") or rng.choice(["flat", "side", "bald", "flat", "side"])
    if extras.get("hat") == "cap":
        parts.append(_p("cap", "other", "head", (-head_r * 0.1, -head_r * 1.42),
                        _rect(-head_r * 0.95, -head_r * 0.55, head_r * 1.9, head_r * 0.75, 4),
                        extras.get("hat_fill", "#2e3a55"), z=2))
        parts.append(_p("visor", "other", "head", (head_r * 0.35, -head_r * 0.88),
                        _rect(0, -3, head_r * 1.0, 6, 3),
                        extras.get("hat_fill", "#2e3a55"), z=2))
    elif hair_kind not in ("bald", "none"):
        hair = extras.get("hair_fill") or rng.choice(HAIR)
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
    if archetype == "scientist":
        extras = {"glasses": rng.random() < 0.75, "parts": [
            lambda tw, lh, th, hr: _p("labcoat", "other", "torso", (0, 0),
                                      _rect(-tw * 0.52, -(lh + th * 0.88), tw * 1.04, th * 0.95, 4),
                                      "#ffffff", z=0.4),
            lambda tw, lh, th, hr: _p("shirt", "other", "torso", (0, 0),
                                      _rect(-tw * 0.17, -(lh + th) + 6, tw * 0.34, th * 0.5),
                                      rng.choice(SHIRT), z=0.5),
            lambda tw, lh, th, hr: _p("tie", "other", "torso", (0, 0),
                                      _path(f"M 0 {-(lh + th) + 8} l 3 6 l -3 {th * 0.2} l -3 {-th * 0.2} Z"), "#41546e", z=0.6, sw=1),
        ]}
        return _humanoid(rng, "#ffffff", rng.choice(PANTS), extras)
    if archetype == "professor":
        extras = {"glasses": True, "hair_kind": "side", "hair_fill": "#d8d3c5", "parts": [
            lambda tw, lh, th, hr: _p("vest", "other", "torso", (0, 0),
                                      _rect(-tw * 0.42, -(lh + th) + 8, tw * 0.84, th * 0.75, 4),
                                      "#614d3f", z=0.5),
            lambda tw, lh, th, hr: _p("shirt", "other", "torso", (0, 0),
                                      _rect(-tw * 0.18, -(lh + th) + 6, tw * 0.36, th * 0.45),
                                      "#ffffff", z=0.4),
        ]}
        return _humanoid(rng, "#614d3f", "#3c3836", extras)
    if archetype == "robot":
        extras = {"skin": "#5a6375", "hair_kind": "none", "parts": [
            lambda tw, lh, th, hr: _p("visor", "other", "head", (hr * 0.3, -hr * 1.05),
                                      _rect(-4, -4, 16, 8, 2), "#00ffff", z=1, sw=0),
            lambda tw, lh, th, hr: _p("chassis", "other", "torso", (0, 0),
                                      _rect(-tw * 0.3, -(lh + th * 0.8), tw * 0.6, th * 0.5, 4),
                                      "#ffd700", z=0.5, sw=1.5),
        ]}
        return _humanoid(rng, "#3b4252", "#2e3440", extras)
    if archetype == "cyborg":
        extras = {"parts": [
            lambda tw, lh, th, hr: _p("metalface", "other", "head", (hr * 0.2, -hr * 1.3),
                                      _rect(0, 0, hr * 0.8, hr * 0.8, 3), "#8892b0", z=4.5),
            lambda tw, lh, th, hr: _p("redeye", "other", "head", (hr * 0.5, -hr * 1.05),
                                      _ell(0, 0, 3, 3), "#ff0000", z=5, sw=0),
            lambda tw, lh, th, hr: _p("techarm", "other", "arm_r", (0, 0),
                                      _rect(-4, 0, 8, 30, 2), "#8892b0", z=3.5, sw=1),
        ]}
        return _humanoid(rng, rng.choice(SHIRT), "#1c2330", extras)
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
    if archetype == "skyline":
        w = rng.uniform(400, 600)
        h = rng.uniform(250, 380)
        parts = []
        colors = ["#2b303c", "#242832", "#1f222b"]
        for i in range(5):
            bw = rng.uniform(80, 140)
            bh = rng.uniform(h * 0.5, h)
            bx = -w/2 + (i * w/4) - bw/2 + rng.uniform(-20, 20)
            parts.append(_p(f"b{i}", "other", None, (bx, 0), _rect(-bw/2, -bh, bw, bh), colors[i % len(colors)], z=i))
            if rng.random() < 0.5:
                parts.append(_p(f"ant{i}", "other", f"b{i}", (0, -bh), _rect(-1.5, -30, 3, 30), colors[i % len(colors)], z=0))
        return parts, {"w": w + 50, "h": h + 50}
    if archetype == "street":
        w, h = rng.uniform(300, 500), 40
        parts = [
            _p("pavement", "other", None, (0, 0), _rect(-w/2, -h, w, h), "#3e424b", z=0),
            _p("curb", "other", None, (0, 0), _rect(-w/2, -h, w, 4), "#8e929b", z=1)
        ]
        dash_w = 20
        for x in range(int(-w/2 + 10), int(w/2 - 10), int(dash_w * 2)):
            parts.append(_p(f"dash_{x}", "other", None, (x, -h/2 - 2), _rect(0, 0, dash_w, 4), "#d1b045", z=1, sw=0))
        return parts, {"w": w + 20, "h": h + 20}
    if archetype == "house":
        w, h = rng.uniform(120, 160), rng.uniform(110, 140)
        body = rng.choice(["#8f6250", "#5c6e8f", "#7a8f6e", "#8c8f8a"])
        roof = rng.choice(["#5e3434", "#34415e", "#444a44"])
        parts = [
            _p("walls", "other", None, (0, 0), _rect(-w/2, -h, w, h), body, z=0),
            _p("roof", "other", None, (0, 0), _path(f"M {-w/2 - 10} {-h} L 0 {-h - h*0.4} L {w/2 + 10} {-h} Z"), roof, z=1),
            _p("door", "other", None, (0, 0), _rect(-15, -45, 30, 45), "#3e2723", z=1),
            _p("knob", "other", "door", (8, -22), _ell(0, 0, 2.5, 2.5), "#ffd700", z=1, sw=0),
            _p("win", "other", None, (-w*0.28, -h*0.75), _rect(-14, -14, 28, 28, 2), "#e0f7fa", z=1, sw=1),
            _p("win_pane", "other", "win", (0, 0), _path("M -14 0 H 14 M 0 -14 V 14"), "none", z=1, stroke=INK, sw=1),
        ]
        return parts, {"w": w + 30, "h": h * 1.5}
    if archetype == "skyscraper":
        w, h = rng.uniform(140, 180), rng.uniform(400, 550)
        body = rng.choice(["#1d2330", "#2c3545", "#212836"])
        win_color = "#e0f7fa"
        parts = [
            _p("body", "other", None, (0, 0), _rect(-w / 2, -h, w, h), body, z=0),
            _p("ant", "other", None, (0, -h), _rect(-2, -45, 4, 45), "#11141a", z=0),
            _p("ant_glow", "other", "ant", (0, -45), _ell(0, 0, 5, 5), "#ff3333", z=1, sw=0)
        ]
        cols, rows = rng.randint(4, 5), rng.randint(8, 12)
        ww, wh = w / (cols * 2), h / (rows * 2.2)
        for r in range(rows):
            for c in range(cols):
                x = -w/2 + w * (c + 0.5)/cols - ww/2
                y = -h + h * (r + 0.25)/(rows + 0.3)
                lit = rng.random() < 0.6
                parts.append(_p(f"w{r}_{c}", "other", "body", (0, 0),
                                _rect(x, y, ww, wh, 0.5), win_color if lit else "#11141d", z=1, sw=0.5))
        return parts, {"w": w + 20, "h": h + 60}
    if archetype == "lab_building":
        w, h = rng.uniform(220, 300), rng.uniform(260, 360)
        body = "#dcdfe4"
        parts = [
            _p("body", "other", None, (0, 0), _rect(-w / 2, -h, w, h), body, z=0),
            _p("roof", "other", "body", (0, 0), _rect(-w / 2 - 4, -h - 6, w + 8, 6), "#3b4252", z=1),
            _p("door_l", "other", None, (-24, -55), _rect(0, 0, 24, 55), "#4c566a", z=1),
            _p("door_r", "other", None, (0, -55), _rect(0, 0, 24, 55), "#4c566a", z=1),
            _p("stripe1", "other", "body", (0, 0), _rect(-w*0.44, -h + 30, 16, h - 80), "#5e81ac", z=1, sw=0),
            _p("stripe2", "other", "body", (0, 0), _rect(w*0.44 - 16, -h + 30, 16, h - 80), "#5e81ac", z=1, sw=0),
            _p("sign", "other", "body", (0, -h * 0.75), _rect(-40, -12, 80, 24, 4), "#88c0d0", z=1, sw=1),
        ]
        return parts, {"w": w + 30, "h": h + 20}
    if archetype == "server_rack":
        w, h = rng.uniform(70, 90), rng.uniform(160, 190)
        parts = [
            _p("frame", "other", None, (0, 0), _rect(-w/2, -h, w, h, 3), "#1e1e24", z=0),
        ]
        slots = 10
        slot_h = (h - 20) / slots
        for i in range(slots):
            y = -h + 10 + i * slot_h
            parts.append(_p(f"blade_{i}", "other", "frame", (0, 0), _rect(-w/2 + 6, y + 2, w - 12, slot_h - 4, 1), "#2d2d34", z=1))
            led1 = "#00ff00" if rng.random() < 0.8 else "#ff0000"
            led2 = "#00ff00" if rng.random() < 0.5 else "#888888"
            parts.append(_p(f"led1_{i}", "other", f"blade_{i}", (-w/2 + 12, y + slot_h/2 - 1.5), _ell(0, 0, 2, 2), led1, z=2, sw=0))
            parts.append(_p(f"led2_{i}", "other", f"blade_{i}", (-w/2 + 18, y + slot_h/2 - 1.5), _ell(0, 0, 2, 2), led2, z=2, sw=0))
        return parts, {"w": w + 20, "h": h + 20}
    if archetype == "streetlight":
        lh = rng.uniform(220, 280)
        return [_p("pole", "other", None, (0, 0), _rect(-4, -lh, 8, lh, 2), "#3a3f47", z=0),
                _p("base", "other", None, (0, 0), _rect(-12, -8, 24, 8, 2), "#3a3f47", z=1),
                _p("arm", "other", "pole", (0, 0), _rect(0, -lh, lh * 0.28, 7, 3), "#3a3f47", z=1),
                _p("head", "other", "pole", (lh * 0.28, -lh + 3), _ell(0, 6, 14, 8), "#e0c477",
                   z=2, sw=1.5),
                ], {"w": lh * 0.7 + 30, "h": lh + 20}
    raise ValueError(f"unknown archetype: {archetype}")


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
    facing = actor.get("facing", "profile")
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
        if verb in ("walk_right", "walk_left", "walk_up_stairs_right", "walk_up_stairs_left"):
            to_x = float(args.get("to_x", x))
            to_y = float(args.get("to_y", y))
            if verb in ("walk_right", "walk_up_stairs_right"):
                flip = False
            else:
                flip = True
            if t <= t1:
                walking = True
                ph = 2 * math.pi * 1.6 * (t - t0)
                s = math.sin(ph)
                rot["leg_l"] = 24 * s
                rot["leg_r"] = -24 * s
                lift = -35 if "stairs" in verb else -20
                rot["leg_l_lower"] = lift * max(0.0, s)
                rot["leg_r_lower"] = lift * max(0.0, -s)
                rot["arm_l"] = -16 * s
                rot["arm_r"] = 16 * s
                rot["arm_l_lower"] = -10 - 8 * max(0.0, s)
                rot["arm_r_lower"] = 10 + 8 * max(0.0, -s)
                dy = -3 * abs(s)
            x = _lerp(x, to_x, u)
            if "stairs" in verb:
                y = _lerp(y, to_y, u)
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
        elif verb == "present" and t <= t1:
            arm = "arm_r" if args.get("arm", "r") == "r" else "arm_l"
            target_ang = float(args.get("angle", -45.0))
            ramp = 0.35
            if t - t0 < ramp:
                ang = _lerp(0, target_ang, (t - t0) / ramp)
            elif t1 - t < ramp:
                ang = _lerp(0, target_ang, (t1 - t) / ramp)
            else:
                ang = target_ang
            rot[arm] = ang if arm == "arm_r" else -ang
        elif verb == "explain" and t <= t1:
            ph = 2 * math.pi * 2.5 * (t - t0)
            rot["arm_l"] = -15 + 10 * math.sin(ph)
            rot["arm_r"] = -15 + 10 * math.cos(ph)
            rot["arm_l_lower"] = -40 + 15 * math.cos(ph)
            rot["arm_r_lower"] = -40 + 15 * math.sin(ph)
        elif verb == "think" and t <= t1:
            ramp = 0.35
            if t - t0 < ramp:
                u_r = (t - t0) / ramp
                rot["arm_r"] = _lerp(0, -110, u_r)
                rot["arm_r_lower"] = _lerp(0, -100, u_r)
            elif t1 - t < ramp:
                u_r = (t1 - t) / ramp
                rot["arm_r"] = _lerp(0, -110, u_r)
                rot["arm_r_lower"] = _lerp(0, -100, u_r)
            else:
                rot["arm_r"] = -110
                rot["arm_r_lower"] = -100
            dy += math.sin(2 * math.pi * 0.8 * (t - t0)) * 0.8
        elif verb == "code" and t <= t1:
            rot["arm_l"] = -60
            rot["arm_r"] = -60
            ph = 2 * math.pi * 5.0 * (t - t0)
            rot["arm_l_lower"] = -90 + 8 * math.sin(ph)
            rot["arm_r_lower"] = -90 + 8 * math.cos(ph)
        elif verb == "face":
            dir_val = args.get("dir", "profile")
            if dir_val == "front" or int(args.get("front", 0)) == 1:
                facing = "front"
            else:
                facing = "profile"
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
            "trans": trans, "mouth": mouth, "dy": dy, "walking": walking,
            "facing": facing}


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
        facing = pose.get("facing") or rig.get("facing") or "profile"
        torso_part = next((hp for hp in parts if hp.get("id") == "torso"), None)
        torso_w, torso_h, leg_h, shoulder_y = 40.0, 80.0, 84.0, -154.0
        if torso_part and torso_part.get("shape") and torso_part["shape"].get("width"):
            torso_w = float(torso_part["shape"]["width"])
            torso_h = float(torso_part["shape"]["height"])
            ty = float(torso_part["shape"]["y"])
            leg_h = -ty - torso_h
            shoulder_y = ty + 10.0
        ox, oy = p.get("offset", [0, 0])
        if facing == "front":
            pid = p.get("id")
            if pid == "arm_l":
                ox, oy = -torso_w * 0.65, shoulder_y
            elif pid == "arm_r":
                ox, oy = torso_w * 0.65, shoulder_y
            elif pid == "leg_l":
                ox = -torso_w * 0.3
            elif pid == "leg_r":
                ox = torso_w * 0.3
        px, py = p.get("pivot", [0, 0])
        tx, ty = trans.get(tag, (0, 0))
        # verb rotation + the part's authored base rotation (both subtree-wide)
        ang = rot.get(tag, 0) + float(p.get("rot", 0) or 0)
        tf = f"translate({ox + tx},{oy + ty})"
        if ang:
            tf += f" rotate({ang},{px},{py})"
        shape = _shape_svg(p)
        if facing == "front":
            head_part = next((hp for hp in parts if hp.get("id") == "head"), None)
            head_r = 20.0
            if head_part and head_part.get("shape") and head_part["shape"].get("ry"):
                head_r = float(head_part["shape"]["ry"])
            pid = p.get("id")
            fill, stroke, sw = p.get("fill"), p.get("stroke"), p.get("sw", 2)
            if pid == "torso":
                shape = f'<rect x="{-torso_w * 0.65}" y="{torso_part["shape"]["y"]}" width="{torso_w * 1.3}" height="{torso_h}" rx="9" fill="{_esc(fill)}" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
            elif pid in ("vest", "shirt", "labcoat", "dtag"):
                shape = f'<g transform="scale(1.3, 1)">{_shape_svg(p)}</g>'
            elif pid == "head":
                shape = f'<ellipse cx="0" cy="{-head_r}" rx="{head_r}" ry="{head_r}" fill="{_esc(fill)}" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
            elif pid == "eye":
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<ellipse cx="{-head_r * 0.28}" cy="0" rx="2.7" ry="2.7" fill="{_esc(fill)}"/>'
                         f'<ellipse cx="{head_r * 0.28}" cy="0" rx="2.7" ry="2.7" fill="{_esc(fill)}"/>'
                         f'</g>')
            elif pid == "nose":
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<path d="M -4 {-head_r * 0.72} Q 0 {-head_r * 0.65} 4 {-head_r * 0.72}" fill="none" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
                         f'</g>')
            elif pid.startswith("mouth_"):
                shape = f'<g transform="translate({-ox}, 0)">{_shape_svg(p)}</g>'
            elif pid == "glasses":
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<path d="M {-head_r * 0.6} 0 H {head_r * 0.6}" fill="none" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
                         f'<rect x="{-head_r * 0.58}" y="-6" width="{head_r * 0.4}" height="12" rx="3" fill="none" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
                         f'<rect x="{head_r * 0.18}" y="-6" width="{head_r * 0.4}" height="12" rx="3" fill="none" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
                         f'</g>')
            elif pid == "hair":
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<ellipse cx="0" cy="{-head_r * 1.38}" rx="{head_r * 0.95}" ry="{head_r * 0.42}" fill="{_esc(fill)}" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
                         f'</g>')
            elif pid == "hair_back":
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<ellipse cx="{-head_r * 0.72}" cy="{-head_r * 0.8}" rx="{head_r * 0.38}" ry="{head_r * 0.7}" fill="{_esc(fill)}"/>'
                         f'<ellipse cx="{head_r * 0.72}" cy="{-head_r * 0.8}" rx="{head_r * 0.38}" ry="{head_r * 0.7}" fill="{_esc(fill)}"/>'
                         f'</g>')
            elif pid == "cap":
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<rect x="{-head_r * 0.92}" y="{-head_r * 0.55}" width="{head_r * 1.84}" height="{head_r * 0.7}" rx="4" fill="{_esc(fill)}" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
                         f'</g>')
            elif pid == "visor":
                vw = head_r * 1.0 if rig.get("archetype") == "robot" else head_r * 1.84
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<rect x="{-vw / 2}" y="-3" width="{vw}" height="6" rx="2" fill="{_esc(fill)}" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
                         f'</g>')
            elif pid == "metalface":
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<rect x="{-head_r * 0.4}" y="{-head_r * 1.3}" width="{head_r * 0.8}" height="{head_r * 0.8}" rx="2" fill="{_esc(fill)}" stroke="{_esc(stroke)}" stroke-width="{sw}"/>'
                         f'</g>')
            elif pid == "redeye":
                shape = (f'<g transform="translate({-ox}, 0)">'
                         f'<ellipse cx="0" cy="0" rx="3" ry="3" fill="{_esc(fill)}"/>'
                         f'</g>')
        scl = float(p.get("scale", 1) or 1)
        if scl != 1:                       # scales the shape only, not children
            shape = f'<g transform="scale({scl})">{shape}</g>'
        def get_z(c):
            if facing == "front" and c.get("id") == "arm_l":
                return 3
            return c.get("z", 0)
        children = sorted(kids.get(p.get("id"), []), key=get_z)
        inner = ("".join(emit(c) for c in children if get_z(c) < 0)
                 + shape
                 + "".join(emit(c) for c in children if get_z(c) >= 0))
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


import base64

def _get_base64_img(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lower().strip(".")
        if ext == "jpg":
            ext = "jpeg"
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/{ext};base64,{b64}"
    except Exception:
        return None

VIDEO_CHART_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".gif")


def slide_to_svg_group(slide, w=1280, h=720, local_t=0.0, chart_frames=None):
    """Render a single slide as an SVG group <g> string. For VIDEO charts,
    chart_frames maps chart name -> (frame_dir, fps, n_frames) and local_t
    picks the frame; without a provider the video chart falls back to its
    first frame if available."""
    out = []
    
    # Pose Image (User photos from C:/Users/paytonm/Pictures/poses)
    pose = slide.get("pose")
    if pose:
        pose_path = os.path.normpath(os.path.join("C:/Users/paytonm/Pictures/poses", pose))
        if pose_path.startswith("C:\\Users\\paytonm\\Pictures\\poses"):
            img_uri = _get_base64_img(pose_path)
            if img_uri:
                px, py, pw, ph = 80, 80, 450, 480
                if slide.get("pose_x") is not None and slide.get("pose_y") is not None:
                    px = float(slide["pose_x"])
                    py = float(slide["pose_y"])
                else:
                    align = slide.get("pose_align", "left")
                    if align == "right":
                        px = 750
                    elif align == "center":
                        px = (w - pw) // 2
                out.append(f'<image href="{img_uri}" x="{px}" y="{py}" width="{pw}" height="{ph}" preserveAspectRatio="xMidYMid meet"/>')
                
    # Chart Image / Uploaded Embed Graphic (From runs/video/library)
    chart = slide.get("chart")
    if chart:
        chart_path = os.path.normpath(os.path.join(video_service.LIB_DIR, video_service.safe_name(chart)))
        if chart_path.startswith(os.path.normpath(video_service.LIB_DIR)):
            if chart.lower().endswith(VIDEO_CHART_EXTS) and chart_frames \
                    and chart in chart_frames:
                fdir, ffps, nfr = chart_frames[chart]
                m_start = max(0.0, float(slide.get("chart_start", 0) or 0))
                mt = local_t - m_start
                if mt <= 0:
                    fi = 1                    # poster frame until start
                elif slide.get("chart_loop"):
                    fi = int(mt * ffps) % nfr + 1
                else:
                    fi = min(nfr, int(mt * ffps) + 1)
                img_uri = _get_base64_img(
                    os.path.join(fdir, f"f_{fi:05d}.png"))
            else:
                img_uri = _get_base64_img(chart_path)
            if img_uri:
                cw = max(80.0, float(slide.get("chart_w", 550) or 550))
                ch_h = max(60.0, float(slide.get("chart_h", 420) or 420))
                cx, cy = 650, 80
                if slide.get("chart_x") is not None and slide.get("chart_y") is not None:
                    cx = float(slide["chart_x"])
                    cy = float(slide["chart_y"])
                else:
                    align = slide.get("chart_align", "right")
                    if align == "left":
                        cx = 80
                    elif align == "center":
                        cx = (w - cw) // 2
                out.append(f'<image href="{img_uri}" x="{cx}" y="{cy}" width="{cw}" height="{ch_h}" preserveAspectRatio="xMidYMid meet"/>')
                
    # Caption / CC Text - word-wrapped, box grows with the lines
    # (identical math to the client preview in slideshow.js)
    text = slide.get("text")
    if text:
        def _wrap_cc(txt, max_chars):
            wrapped = []
            for para in str(txt).split("\n"):
                cur = ""
                for word in para.split():
                    cand = (cur + " " + word) if cur else word
                    if len(cand) > max_chars and cur:
                        wrapped.append(cur)
                        cur = word
                    else:
                        cur = cand
                wrapped.append(cur)
            return wrapped

        cc_font = 24
        lines = _wrap_cc(text, 68)
        if len(lines) > 4:
            cc_font = 20
            lines = _wrap_cc(text, 82)
        line_h = cc_font * 1.35
        pad = 14
        box_h = pad * 2 + line_h * len(lines)
        box_y = 692 - box_h
        base0 = box_y + pad + cc_font * 0.85
        ts = "".join(f'<tspan x="640" dy="{line_h if i > 0 else 0}">{_esc(ln)}</tspan>'
                     for i, ln in enumerate(lines))
        out.append(f'<rect x="100" y="{box_y:.1f}" width="1080" height="{box_h:.1f}" rx="8" fill="#10141c" fill-opacity="0.85" stroke="#1c232c" stroke-width="1"/>')
        out.append(f'<text x="640" y="{base0:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="{cc_font}" font-weight="bold" fill="#f0ede4" text-anchor="middle">{ts}</text>')
        
    return "".join(out)

def slideshow_svg(slides, t, w=1280, h=720, chart_frames=None):
    """Renders a complete SVG frame for time t, handling crossfades."""
    ranges = []
    curr = 0.0
    for s in slides:
        dur = _eff_slide_dur(s)
        trans_dur = float(s.get("transition_dur", 0.5)) if s.get("transition") == "fade" else 0.0
        ranges.append({
            "slide": s,
            "start": curr,
            "end": curr + dur,
            "trans_dur": trans_dur
        })
        curr += dur
        
    bg_color = slides[0].get("bg", "#0b0d10") if slides else "#0b0d10"
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" fill="{_esc(bg_color)}"/>']
           
    for idx, r in enumerate(ranges):
        if r["start"] <= t <= r["end"]:
            if r["trans_dur"] > 0.0 and (r["end"] - t) < r["trans_dur"] and idx < len(ranges) - 1:
                next_r = ranges[idx + 1]
                alpha = (t - (r["end"] - r["trans_dur"])) / r["trans_dur"]
                out.append(f'<g opacity="{1.0 - alpha:.3f}">{slide_to_svg_group(r["slide"], w, h, t - r["start"], chart_frames)}</g>')
                out.append(f'<g opacity="{alpha:.3f}">{slide_to_svg_group(next_r["slide"], w, h, 0.0, chart_frames)}</g>')
            else:
                out.append(slide_to_svg_group(r["slide"], w, h, t - r["start"], chart_frames))
            break
    else:
        if ranges:
            out.append(slide_to_svg_group(ranges[-1]["slide"], w, h, 0.0, chart_frames))
            
    out.append("</svg>")
    return "".join(out)

def _norm_cuts(cuts, dur):
    """Sorted, merged, clamped removed-intervals within [0, dur]."""
    cs = []
    for c in (cuts or []):
        try:
            a, b = max(0.0, float(c[0])), min(dur, float(c[1]))
        except (TypeError, ValueError, IndexError):
            continue
        if b - a > 0.01:
            cs.append([a, b])
    cs.sort()
    out = []
    for c in cs:
        if out and c[0] <= out[-1][1] + 0.005:
            out[-1][1] = max(out[-1][1], c[1])
        else:
            out.append(c)
    return out


def _kept_segments(clip):
    dur = float(clip.get("dur", 0.0) or 0.0)
    segs, t = [], 0.0
    for c in _norm_cuts(clip.get("cuts"), dur):
        if c[0] - t > 0.01:
            segs.append((t, c[0]))
        t = c[1]
    if dur - t > 0.01:
        segs.append((t, dur))
    return segs


def _eff_clip_dur(clip):
    return sum(b - a for a, b in _kept_segments(clip))


def _chart_dur(name):
    """Duration of an embedded chart if it is animated media (gif/video);
    0 for stills. ffprobe result is mtime-cached by video_service."""
    if not name:
        return 0.0
    try:
        safe = video_service.safe_name(str(name))
        path = os.path.normpath(os.path.join(video_service.LIB_DIR, safe))
        if not path.startswith(os.path.normpath(video_service.LIB_DIR))                 or not os.path.isfile(path):
            return 0.0
        return float(video_service._meta(safe, path).get("duration", 0) or 0)
    except Exception:
        return 0.0


def _media_floor(s):
    """Non-looping media floors the slide at start + runtime; looping
    media fills whatever the slide gives it."""
    d = _chart_dur(s.get("chart"))
    if not d or s.get("chart_loop"):
        return 0.0
    return max(0.0, float(s.get("chart_start", 0) or 0)) + d


def _eff_slide_dur(s):
    """The slide floor: set duration, then audio, then embedded media -
    a slide stays up long enough for everything it carries."""
    base = float(s.get("duration", 3.0) or 3.0)
    audio = sum(_eff_clip_dur(c) for c in (s.get("clips") or []))
    return max(base, audio, _media_floor(s))


SLIDE_AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "runs", "video", "slide_audio")
os.makedirs(SLIDE_AUDIO_DIR, exist_ok=True)


def render_slides(slides, out_name="", fps=24, w=1280, h=720):
    """Encodes a presentation slides deck into an MP4 file."""
    if not RASTER_OK:
        raise RuntimeError(f"resvg unavailable: {RASTER_ERR} (pip install resvg-py)")
    if not video_service.available():
        raise RuntimeError("ffmpeg unavailable")
        
    fps = max(1, min(60, int(fps)))
    dur = sum(_eff_slide_dur(s) for s in slides)
    if dur <= 0.01:
        raise ValueError("total duration must be greater than 0")
    total = int(round(fps * dur))
    
    audio = None
    first_slide_audio = next((s.get("audio") for s in slides if s.get("audio")), None)
    if first_slide_audio:
        audio = os.path.normpath(os.path.join(video_service.LIB_DIR, video_service.safe_name(first_slide_audio)))
        if not os.path.isfile(audio):
            audio = None

    # per-slide mic clips (slide["clips"] = [{id, dur}, ...] in order):
    # each clip starts at its slide's start time plus the durations of the
    # clips before it on the same slide; all clips (and the optional deck
    # track) are mixed with adelay + amix
    clip_inputs = []                     # (path, start_ms, kept_segments)
    t_cursor = 0.0
    for s_ in slides:
        off = 0.0
        for c_ in (s_.get("clips") or []):
            cid = re.sub(r"[^A-Za-z0-9_.-]", "", str(c_.get("id", "")))
            p_ = os.path.join(SLIDE_AUDIO_DIR, cid)
            if cid and os.path.isfile(p_):
                segs = _kept_segments(c_)
                if segs:
                    clip_inputs.append((p_, int(round((t_cursor + off) * 1000)),
                                        segs))
                    off += sum(b - a for a, b in segs)
        t_cursor += _eff_slide_dur(s_)

    # pre-extract frames for VIDEO charts (once per unique video, at the
    # deck fps, capped to the longest slide that embeds it) so the frame
    # compositor can base64 the right frame per output frame
    chart_frames = {}
    import tempfile
    for s_ in slides:
        ch_name = s_.get("chart") or ""
        if not ch_name.lower().endswith(VIDEO_CHART_EXTS):
            continue
        if ch_name in chart_frames:
            continue
        src = os.path.normpath(os.path.join(
            video_service.LIB_DIR, video_service.safe_name(ch_name)))
        if not src.startswith(os.path.normpath(video_service.LIB_DIR)) \
                or not os.path.isfile(src):
            continue
        need = 0.0
        for x in slides:
            if (x.get("chart") or "") != ch_name:
                continue
            if x.get("chart_loop"):
                need = max(need, min(_chart_dur(ch_name), 120.0))
            else:
                st_ = max(0.0, float(x.get("chart_start", 0) or 0))
                need = max(need, _eff_slide_dur(x) - st_)
        fdir = tempfile.mkdtemp(prefix="slidechart_")
        r_ = subprocess.run(
            [video_service.FFMPEG, "-y", "-i", src, "-t", f"{need:.3f}",
             "-vf", f"fps={fps},scale=550:-2",
             os.path.join(fdir, "f_%05d.png")],
            capture_output=True, timeout=600,
            creationflags=video_service.CREATE_NO_WINDOW)
        nfr = len([f for f in os.listdir(fdir) if f.endswith(".png")])
        if r_.returncode == 0 and nfr:
            chart_frames[ch_name] = (fdir, fps, nfr)

    name = slug(out_name or "slideshow")
    out = video_service.unique_path(name + ".mp4")
    cmd = [video_service.FFMPEG, "-y",
           "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-"]
    n_audio_in = 0
    if audio:
        cmd += ["-i", audio]
        n_audio_in += 1
    for p_, _ms, _segs in clip_inputs:
        cmd += ["-i", p_]
        n_audio_in += 1
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"]
    if clip_inputs:
        parts, labels = [], []
        ai = 1
        if audio:
            parts.append(f"[{ai}:a]anull[t0]")
            labels.append("[t0]")
            ai += 1
        for j, (_p, ms, segs) in enumerate(clip_inputs):
            if len(segs) == 1 and segs[0][0] < 0.01:
                # untrimmed (or head-only): single atrim keeps it simple
                a0, b0 = segs[0]
                parts.append(f"[{ai}:a]atrim=start={a0:.3f}:end={b0:.3f},"
                             f"asetpts=PTS-STARTPTS,adelay={ms}|{ms}[c{j}]")
            else:
                # multi-cut: trim each kept segment, concat SEAMLESSLY,
                # then delay to the clip's slide time
                seg_lbls = []
                for k2, (a0, b0) in enumerate(segs):
                    parts.append(f"[{ai}:a]atrim=start={a0:.3f}:end={b0:.3f},"
                                 f"asetpts=PTS-STARTPTS[s{j}_{k2}]")
                    seg_lbls.append(f"[s{j}_{k2}]")
                parts.append("".join(seg_lbls) +
                             f"concat=n={len(seg_lbls)}:v=0:a=1,"
                             f"adelay={ms}|{ms}[c{j}]")
            labels.append(f"[c{j}]")
            ai += 1
        parts.append("".join(labels) +
                     f"amix=inputs={len(labels)}:normalize=0:dropout_transition=0[aout]")
        cmd += ["-filter_complex", ";".join(parts),
                "-map", "0:v", "-map", "[aout]",
                "-c:a", "aac", "-b:a", "192k", "-t", f"{dur:.3f}"]
    elif audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [out]
    
    job = video_service.custom_job("slides", f"{name} · {dur:.1f}s @ {fps}fps", out)
    
    def run():
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     creationflags=video_service.CREATE_NO_WINDOW)
            job["proc"] = proc
            for i in range(total):
                if job.get("cancelled"):
                    break
                svg = slideshow_svg(slides, i / fps, w, h, chart_frames)
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
        
    threading.Thread(target=run, daemon=True, name="slides-render").start()
    return job

