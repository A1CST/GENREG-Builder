"""radial_synth.py — a synthetic HIERARCHICAL task for end-to-end stack tests.

Each 32x32x3 image carries one of 5 shape motifs in the top half and one of 5
in the bottom half (position-jittered, noisy, identical gray tint so SHAPE is
the only signal). The label is

    y = (5 * top_motif + bottom_motif) mod 10

Either motif alone says almost nothing about the class; the class lives in
the PAIRING, and mod-10 over the pair is linearly inseparable even from
perfect one-hot motif detections. So: space 0 has honest work (become motif
detectors) and deeper spaces have NECESSARY work (fold two detections into a
pair feature) — the stack must function end-to-end or the task caps out.
Rules are deterministic, so ~100% is reachable: it can be trained to
completion. Same npz layout as cifar_full (Xtr/ytr/Xte/yte uint8) so the
whole radial_stack pipeline runs unmodified via its data_npz parameter.
"""
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _motifs():
    """Five 12x12 gray shapes with roughly equal ink."""
    m = np.zeros((5, 12, 12), np.float32)
    m[0, 5:7, 1:11] = 1.0; m[0, 1:11, 5:7] = 1.0          # cross
    yy, xx = np.mgrid[0:12, 0:12]
    r = np.sqrt((yy - 5.5) ** 2 + (xx - 5.5) ** 2)
    m[1] = ((r > 3.2) & (r < 5.2)).astype(np.float32)     # ring
    for k in range(-12, 12, 3):                            # diagonal stripes
        for i in range(12):
            j = i + k
            if 0 <= j < 12:
                m[2, i, j] = 1.0
    m[3] = ((yy // 3 + xx // 3) % 2).astype(np.float32) * 0.7   # checker
    m[4] = (r < 3.6).astype(np.float32)                    # blob
    return m


def make_data(n_train=12000, n_test=3000, seed=0, noise=0.08, rule="hard",
              path=os.path.join(_HERE, "radial_data", "synth_hier.npz")):
    """rule="easy": y = (5t+b) mod 10 — COLLAPSES to 5*(t mod 2)+b, linear over
    detectors (kept for harness smoke tests). rule="hard": y = (t+b) mod 5 +
    5*[t==b] — cyclic addition + equality, provably NOT additively separable
    over one-hot detections: composition is mandatory."""
    rng = np.random.default_rng(seed)
    M = _motifs()
    n = n_train + n_test
    top = rng.integers(0, 5, n)
    bot = rng.integers(0, 5, n)
    if rule == "hard":
        y = (top + bot) % 5 + 5 * (top == bot).astype(np.int64)
    else:
        y = (5 * top + bot) % 10
    X = np.zeros((n, 32, 32), np.float32)
    for i in range(n):
        oy, ox = rng.integers(0, 5), rng.integers(0, 21)   # top-half placement
        X[i, oy:oy + 12, ox:ox + 12] += M[top[i]]
        oy, ox = rng.integers(16, 21), rng.integers(0, 21)  # bottom-half placement
        X[i, oy:oy + 12, ox:ox + 12] += M[bot[i]]
    X = X * rng.uniform(0.75, 1.0, (n, 1, 1)).astype(np.float32)   # brightness
    X = X + rng.normal(0, noise, X.shape).astype(np.float32)
    X = np.clip(X, 0, 1)
    X3 = (np.repeat(X[..., None], 3, axis=3) * 255).astype(np.uint8)
    np.savez(path,
             Xtr=X3[:n_train], ytr=y[:n_train].astype(np.int64),
             Xte=X3[n_train:], yte=y[n_train:].astype(np.int64))
    print(f"synth_hier: {n_train}/{n_test} written -> {path} "
          f"(class balance {np.bincount(y, minlength=10).tolist()})", flush=True)
    return path


if __name__ == "__main__":
    make_data()
