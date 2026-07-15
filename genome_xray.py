"""genome_xray.py — watch a real genome pull tangled data into structure.

No abstract map, no rotation gimmick. The ground truth is real MNIST digits.
Left of the animation they sit as a tangled cloud in raw feature space (classes
overlapping). A solved genome then acts on them — the same forward pass the
/mnist page uses — and every point migrates toward the corner of the digit the
genome thinks it is. Watch the blob resolve into ten clean clusters.

Colour = the TRUE digit. So a point that lands on a wrong-coloured corner is a
mistake you can see. Swap genomes or toggle their layers (detectors -> mixer ->
pairwise referees) and watch the separation get sharper or blurrier. That is the
genome working, made visible.

Feature build is ~11s; it is cached after the first call. All the compared
genomes share the same v2 features, so the STARTING tangle is identical and any
difference you see at the end is the genome, not the data.
"""
import os
import pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# feat_v2 genomes: all have the 10 detectors + mixer + pairs over the same
# features, so they are directly comparable on one shared starting cloud.
GENOME_FILES = [
    ("r5",     "demo/mnist_genomes_r5.pkl"),
    ("r6",     "demo/mnist_genomes_r6.pkl"),
    ("r2",     "demo/mnist_genomes_r2.pkl"),
    ("pre_v4", "demo/mnist_genomes_pre_v4.pkl"),
]

# ten distinct hues for the digits 0..9
DIGIT_COLOR = ["#e6484d", "#e08a3b", "#e0c53b", "#7bc043", "#2fb8a0",
               "#3a9ad9", "#5566d6", "#9b57d3", "#d357a8", "#8b8f99"]

_CACHE = {}          # shared features + starting projection (built once)
_GENOMES = {}        # loaded champion pickles by id


def _load_genome(gid):
    if gid not in _GENOMES:
        path = dict(GENOME_FILES).get(gid)
        if not path:
            return None
        with open(os.path.join(_HERE, path), "rb") as f:
            _GENOMES[gid] = pickle.load(f)
    return _GENOMES[gid]


def _ensure_ground_truth(n_per_class=50):
    """Build the shared feature set once, take a balanced sample of real test
    digits, and project the raw features to 2D — the tangled starting cloud."""
    key = ("gt", n_per_class)
    if key in _CACHE:
        return _CACHE[key]
    from genreg_train import mnist_pipe as mp
    D = mp.build_features(version=2)
    Fte, yte = D["Fte"], D["yte"]
    idx = []
    for d in range(10):
        idx += list(np.where(yte == d)[0][:n_per_class])
    idx = np.array(idx)
    F, y = Fte[idx], yte[idx]
    # 2D PCA of the raw features = the tangled ground truth
    Fc = F - F.mean(0)
    U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
    start = Fc @ Vt[:2].T
    start = start / (np.abs(start).max() + 1e-9)      # into a tidy box
    out = {"F": F, "y": y, "start": start, "n": len(y)}
    _CACHE[key] = out
    return out


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def list_genomes():
    items = []
    for gid, path in GENOME_FILES:
        full = os.path.join(_HERE, path)
        acc = None
        if os.path.exists(full):
            g = _load_genome(gid)
            acc = g.get("joint_val_acc") if g else None
        items.append({"id": gid, "label": gid, "acc": acc,
                      "exists": os.path.exists(full)})
    return items


def transform(gid="r5", use_pairs=True, use_mixer=True, n_per_class=50):
    """Apply one genome to the shared ground-truth sample. Returns each point's
    start (raw) and end (post-genome) 2D position, plus the digit anchors and
    accuracy — everything the page needs to animate the separation."""
    from genreg_train import mnist_pipe as mp
    gt = _ensure_ground_truth(n_per_class)
    F, y, start = gt["F"], gt["y"], gt["start"]
    ch = _load_genome(gid)
    if ch is None:
        return {"error": f"genome {gid} not found"}

    pred, L = mp.predict(ch, F, use_mixer=use_mixer, use_pairs=use_pairs)
    P = _softmax(L)

    # ten anchors on a circle, one per digit; end position = confidence-weighted
    # blend of anchors, so a confident correct point lands on its own colour.
    ang = np.arange(10) / 10 * 2 * np.pi - np.pi / 2
    anchors = np.stack([np.cos(ang), np.sin(ang)], 1) * 1.35
    end = P @ anchors

    acc = float((pred == y).mean())
    per_digit = [round(float((pred[y == d] == d).mean()), 3) if (y == d).any() else None
                 for d in range(10)]
    # cluster tightness: mean distance to the correct digit's anchor (lower = crisper)
    tight = float(np.mean(np.linalg.norm(end - anchors[y], axis=1)))

    points = [{"x0": round(float(start[i, 0]), 4), "y0": round(float(start[i, 1]), 4),
               "x1": round(float(end[i, 0]), 4), "y1": round(float(end[i, 1]), 4),
               "d": int(y[i]), "pred": int(pred[i])} for i in range(len(y))]
    anchor_out = [{"d": d, "x": round(float(anchors[d, 0]), 4),
                   "y": round(float(anchors[d, 1]), 4)} for d in range(10)]

    return {
        "genome": gid, "use_pairs": bool(use_pairs), "use_mixer": bool(use_mixer),
        "n": len(y), "acc": round(acc, 4), "tightness": round(tight, 3),
        "per_digit": per_digit, "points": points, "anchors": anchor_out,
        "colors": DIGIT_COLOR, "genomes": list_genomes(),
    }


if __name__ == "__main__":
    import json, time
    t = time.time()
    r = transform("r5")
    print("first (with build):", round(time.time() - t, 1), "s")
    print(json.dumps({k: v for k, v in r.items() if k not in ("points",)}, indent=2)[:900])
    t = time.time(); transform("pre_v4"); print("second:", round(time.time() - t, 2), "s")
