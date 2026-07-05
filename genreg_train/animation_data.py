"""Animation dataset — procedural 64x64 shape-motion clips.

Ten short animations, each a single white shape moving over a black
background along a unique path. Every clip is FRAMES x SIZE x SIZE
float32 in [0, 1] (0 = black, 1 = white), rendered with soft (anti-
aliased) edges so sub-pixel motion is smooth between frames.

Use from Python:
    from genreg_train import animation_data
    clips = animation_data.generate_all()          # {name: (24,64,64) float32}

Or from the command line to render inspectable files:
    python -m genreg_train.animation_data [outdir]   # default: animations/
writes <name>.npy (uint8) plus an upscaled preview <name>.gif per clip.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

SIZE = 64          # frame width/height in pixels
FRAMES = 24        # frames per animation
RADIUS = 5.0       # shape "radius" in pixels
MARGIN = 8.0       # keep shape centres this far from the border

_YY, _XX = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)


# ── shape rasterizers ────────────────────────────────────────────────
# Each takes a sub-pixel centre (x, y) and returns a SIZE x SIZE float
# alpha mask with a ~1px soft edge (signed-distance clipped to [0,1]).

def _soft(d):
    """Signed distance (negative = inside) -> anti-aliased alpha."""
    return np.clip(0.5 - d, 0.0, 1.0)


def circle(x, y, r=RADIUS):
    d = np.hypot(_XX - x, _YY - y) - r
    return _soft(d)


def square(x, y, r=RADIUS - 0.5):
    d = np.maximum(np.abs(_XX - x), np.abs(_YY - y)) - r
    return _soft(d)


def diamond(x, y, r=RADIUS + 1.0):
    d = (np.abs(_XX - x) + np.abs(_YY - y)) - r
    return _soft(d)


def ring(x, y, r=RADIUS + 0.5, w=2.0):
    d = np.abs(np.hypot(_XX - x, _YY - y) - r) - w / 2.0
    return _soft(d)


def triangle(x, y, r=RADIUS + 1.5):
    # upward-pointing equilateral triangle: intersection of 3 half-planes
    dx, dy = _XX - x, _YY - y
    k = math.sqrt(3.0)
    d = np.maximum(dy - r / 2.0,                      # bottom edge
                   np.maximum((-dy - dx * k) - r,     # right edge
                              (-dy + dx * k) - r) / 2.0)
    return _soft(d)


def plus(x, y, r=RADIUS + 1.0, w=1.6):
    ax, ay = np.abs(_XX - x), np.abs(_YY - y)
    d = np.maximum(np.minimum(ax, ay) - w, np.maximum(ax, ay) - r)
    return _soft(d)


def xcross(x, y, r=RADIUS + 1.0, w=1.6):
    # plus rotated 45 degrees
    u = np.abs((_XX - x) + (_YY - y)) / math.sqrt(2.0)
    v = np.abs((_XX - x) - (_YY - y)) / math.sqrt(2.0)
    d = np.maximum(np.minimum(u, v) - w, np.maximum(u, v) - r)
    return _soft(d)


def hexagon(x, y, r=RADIUS + 0.5):
    ax, ay = np.abs(_XX - x), np.abs(_YY - y)
    k = math.sqrt(3.0) / 2.0
    d = np.maximum(ax * k + ay / 2.0, ay) - r
    return _soft(d)


def crescent(x, y, r=RADIUS + 0.5):
    # a disc with a same-size disc bite taken out of its right side
    d_outer = np.hypot(_XX - x, _YY - y) - r
    d_bite = np.hypot(_XX - (x + r * 0.9), _YY - y) - r * 0.9
    return _soft(np.maximum(d_outer, -d_bite))


def frame(x, y, r=RADIUS - 0.5, w=1.8):
    # hollow square outline
    d = np.abs(np.maximum(np.abs(_XX - x), np.abs(_YY - y)) - r) - w / 2.0
    return _soft(d)


# ── motion paths ─────────────────────────────────────────────────────
# Each takes t in [0, 1] and returns a centre (x, y) in pixel coords.

_LO, _HI = MARGIN, SIZE - MARGIN          # usable centre range
_SPAN = _HI - _LO
_MID = SIZE / 2.0


def path_line(t):
    """Straight horizontal sweep, left to right."""
    return _LO + _SPAN * t, _MID


def path_diagonal(t):
    """Straight diagonal, top-left to bottom-right."""
    return _LO + _SPAN * t, _LO + _SPAN * t


def path_swoop(t):
    """Parabolic swoop: dives from top-left, skims the bottom, exits top-right."""
    x = _LO + _SPAN * t
    y = _HI - _SPAN * (2.0 * t - 1.0) ** 2   # bottom at t=0.5, top at ends
    return x, y


def path_loop(t):
    """One full clockwise loop around the centre."""
    a = 2.0 * math.pi * t - math.pi / 2.0
    r = _SPAN / 2.0
    return _MID + r * math.cos(a), _MID + r * math.sin(a)


def path_figure8(t):
    """Figure-eight (Lissajous 1:2)."""
    a = 2.0 * math.pi * t
    return _MID + (_SPAN / 2.0) * math.sin(a), _MID + (_SPAN / 2.0) * math.sin(2.0 * a)


def path_zigzag(t):
    """Left-to-right while bouncing between top and bottom (3 zigs)."""
    x = _LO + _SPAN * t
    tri = abs((t * 3.0) % 2.0 - 1.0)          # triangle wave in [0,1]
    return x, _LO + _SPAN * tri


def path_wave(t):
    """Left-to-right riding a two-period sine wave."""
    x = _LO + _SPAN * t
    return x, _MID + (_SPAN / 2.0) * math.sin(4.0 * math.pi * t)


def path_spiral(t):
    """Spiral outward from the centre, two turns."""
    a = 4.0 * math.pi * t
    r = (_SPAN / 2.0) * t
    return _MID + r * math.cos(a), _MID + r * math.sin(a)


def path_bounce(t):
    """Moves right while bouncing off the floor, each bounce lower."""
    x = _LO + _SPAN * t
    # three parabolic hops with decaying height
    hops = [(0.0, 0.45, 1.0), (0.45, 0.75, 0.55), (0.75, 1.0, 0.3)]
    for t0, t1, h in hops:
        if t <= t1 or (t0, t1, h) == hops[-1]:
            u = (t - t0) / (t1 - t0)
            y = _HI - _SPAN * h * (1.0 - (2.0 * u - 1.0) ** 2)
            return x, y
    return x, _HI


def path_scurve(t):
    """Smooth S-curve: cubic Bezier from bottom-left to top-right."""
    p0, p1, p2, p3 = (_LO, _HI), (_HI, _HI), (_LO, _LO), (_HI, _LO)
    u = 1.0 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return x, y


# ── the ten animations: (name, path fn, shape fn) ────────────────────
# every clip gets a UNIQUE shape — no shape appears in two animations

ANIMATIONS = [
    ("line",     path_line,     circle),
    ("diagonal", path_diagonal, square),
    ("swoop",    path_swoop,    triangle),
    ("loop",     path_loop,     ring),
    ("figure8",  path_figure8,  diamond),
    ("zigzag",   path_zigzag,   plus),
    ("wave",     path_wave,     xcross),
    ("spiral",   path_spiral,   hexagon),
    ("bounce",   path_bounce,   crescent),
    ("scurve",   path_scurve,   frame),
]


def generate(name):
    """Render one named animation -> (FRAMES, SIZE, SIZE) float32 in [0,1]."""
    for nm, path, shape in ANIMATIONS:
        if nm == name:
            frames = np.zeros((FRAMES, SIZE, SIZE), dtype=np.float32)
            for i in range(FRAMES):
                t = i / (FRAMES - 1)
                x, y = path(t)
                frames[i] = shape(x, y)
            return frames
    raise KeyError(f"unknown animation {name!r}; have {[a[0] for a in ANIMATIONS]}")


def generate_all():
    """All ten animations -> {name: (FRAMES, SIZE, SIZE) float32}."""
    return {name: generate(name) for name, _, _ in ANIMATIONS}


def save_all(outdir):
    """Write <name>.npy (uint8 0..255) and an 8x-upscaled <name>.gif preview."""
    from PIL import Image

    os.makedirs(outdir, exist_ok=True)
    for name, frames in generate_all().items():
        u8 = (frames * 255.0 + 0.5).astype(np.uint8)
        np.save(os.path.join(outdir, f"{name}.npy"), u8)
        imgs = [Image.fromarray(f, mode="L").resize((SIZE * 8, SIZE * 8), Image.NEAREST)
                for f in u8]
        imgs[0].save(os.path.join(outdir, f"{name}.gif"), save_all=True,
                     append_images=imgs[1:], duration=1000 // 12, loop=0)
        print(f"  {name:9s} -> {name}.npy ({u8.shape[0]} frames) + {name}.gif")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "animations"
    print(f"rendering {len(ANIMATIONS)} animations to {out}/")
    save_all(out)
