"""radial_screen.py — Radial Space v3 §11.1: REAL screen-capture fingerprinting.

Not simulated streams this time — actual desktop frames. Each captured frame is
reduced to a few features (brightness, edge density, colourfulness, and frame-to-
frame motion), mapped through the winning linear M into radial space, and strung
into a traversal path. Different activities (idle / browsing / video) should draw
geometrically distinct paths; a nearest-centroid classifier on path statistics
tests whether they separate — the paper's >90% bar.

Workflow: record a few seconds while doing an activity (label it), repeat for each
activity, train (build centroids), then classify live. All local, all on the
machine running the Flask app.
"""
import time
import numpy as np
from PIL import ImageGrab

import radial_mfunc as rmf

# feature scales (rough full-range, for normalising into the mapping's domain)
_SCALE = {"motion": 40.0, "bright": 255.0, "edge": 60.0, "color": 70.0}

CLIPS = {}          # label -> list of stat-vectors (in-memory training set)
LAST = {"path": [], "label": None}


def _grab(size=(160, 90)):
    """One downscaled frame: (grayscale float array, rgb float array)."""
    im = ImageGrab.grab().resize(size)
    rgb = np.asarray(im, dtype=np.float32)[:, :, :3]
    gray = rgb.mean(2)
    return gray, rgb


def _frame_features(gray, rgb, prev):
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    return {
        "bright": float(gray.mean()),
        "edge": float((gx + gy) / 2),
        "color": float(rgb.std(2).mean()),
        "motion": 0.0 if prev is None else float(np.abs(gray - prev).mean()),
    }


def _capture_stream(seconds=4.0, fps=10.0):
    """Capture for `seconds`, returning per-frame features + the radial path."""
    frames, prev, t0 = [], None, time.time()
    dt = 1.0 / fps
    while time.time() - t0 < seconds:
        g, rgb = _grab()
        frames.append(_frame_features(g, rgb, prev))
        prev = g
        elapsed = (time.time() - t0) % dt
        if dt - elapsed > 0.002:
            time.sleep(min(dt, dt - elapsed))
    # radial path: r from motion (activity magnitude), phi from brightness
    path = []
    for f in frames:
        r_norm = min(1.0, f["motion"] / _SCALE["motion"])
        b = f["bright"] / _SCALE["bright"] * 255.0
        r_map, phi, _ = rmf.M_linear(b, 0.0, 255.0)
        r = 0.3 + 1.7 * r_norm                     # radius carries activity level
        path.append([r * np.cos(phi), r * np.sin(phi)])
    return frames, np.array(path)


def _stat_vector(frames, path):
    """A fixed-length fingerprint of a clip: what its path and features look like."""
    F = {k: np.array([f[k] for f in frames]) for k in ("bright", "edge", "color", "motion")}
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.array([0.0])
    r = np.linalg.norm(path, axis=1) if len(path) else np.array([0.0])
    return np.array([
        F["motion"].mean(), F["motion"].std(),
        F["bright"].mean(), F["bright"].std(),
        F["edge"].mean(), F["color"].mean(),
        seg.mean(), seg.std(),                     # path jitter / speed
        r.mean(), r.std(),                         # how far out / how spread
    ], dtype=float)


def record(label, seconds=4.0):
    frames, path = _capture_stream(seconds)
    vec = _stat_vector(frames, path)
    CLIPS.setdefault(label, []).append(vec.tolist())
    LAST["path"] = path.round(4).tolist(); LAST["label"] = label
    return {"label": label, "n_frames": len(frames), "clips": {k: len(v) for k, v in CLIPS.items()},
            "path": LAST["path"],
            "summary": {"motion": round(vec[0], 2), "bright": round(vec[2], 1), "edge": round(vec[4], 2)}}


def _training_matrix():
    X, y, labels = [], [], sorted(CLIPS)
    for li, lab in enumerate(labels):
        for v in CLIPS[lab]:
            X.append(v); y.append(li)
    return np.array(X), np.array(y), labels


def _standardise(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def train():
    if sum(len(v) for v in CLIPS.values()) < 2 or len(CLIPS) < 2:
        return {"ready": False, "msg": "record at least 2 activities (>=2 clips each helps)."}
    X, y, labels = _training_matrix()
    Xs, mu, sd = _standardise(X)
    cent = np.array([Xs[y == c].mean(0) for c in range(len(labels))])
    # leave-one-out nearest-centroid accuracy
    correct = 0
    for i in range(len(Xs)):
        mask = np.ones(len(Xs), bool); mask[i] = False
        c = np.array([Xs[mask][y[mask] == k].mean(0) if (y[mask] == k).any() else np.full(Xs.shape[1], 1e9)
                      for k in range(len(labels))])
        pred = int(np.argmin(np.linalg.norm(c - Xs[i], axis=1)))
        correct += pred == y[i]
    train_MODEL.update({"mu": mu.tolist(), "sd": sd.tolist(), "cent": cent.tolist(), "labels": labels})
    return {"ready": True, "labels": labels, "n": int(len(Xs)),
            "loo_accuracy": round(correct / len(Xs), 3),
            "clips": {k: len(v) for k, v in CLIPS.items()}}


train_MODEL = {}


def classify(seconds=3.0):
    if not train_MODEL:
        return {"error": "train first"}
    frames, path = _capture_stream(seconds)
    vec = _stat_vector(frames, path)
    mu, sd = np.array(train_MODEL["mu"]), np.array(train_MODEL["sd"])
    cent, labels = np.array(train_MODEL["cent"]), train_MODEL["labels"]
    xs = (vec - mu) / sd
    d = np.linalg.norm(cent - xs, axis=1)
    pred = int(np.argmin(d))
    conf = np.exp(-d); conf = conf / conf.sum()
    return {"predicted": labels[pred], "path": path.round(4).tolist(),
            "scores": [{"label": labels[i], "conf": round(float(conf[i]), 3)} for i in range(len(labels))],
            "summary": {"motion": round(vec[0], 2), "bright": round(vec[2], 1)}}


def status():
    return {"clips": {k: len(v) for k, v in CLIPS.items()}, "trained": bool(train_MODEL),
            "labels": train_MODEL.get("labels", [])}


def clear():
    CLIPS.clear(); train_MODEL.clear(); LAST["path"] = []
    return {"cleared": True}


if __name__ == "__main__":
    print("capturing 3s idle-ish sample...")
    r = record("test", 3.0)
    print("frames", r["n_frames"], "summary", r["summary"], "path pts", len(r["path"]))
