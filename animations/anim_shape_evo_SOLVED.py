"""Two-phase (§VI): FREEZE the oriented-filter bank as a fixed translation-
invariant feature extractor, evolve ONLY the classifier head on the precomputed
features. Because filters are shared+fixed, pool features ONCE per frame -> the
evolving part is a tiny 20->24->10 MLP on 20-d inputs. ~1000x faster; directly
tests whether the invariant feature basis separates all 10 shapes.

Pure evolution: mutation only, energy homeostasis, soft geo-mean fitness,
per-neuron evolved activations, one clip per generation. Held-out per-clip 80/20.
"""
import sys, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
sys.path.insert(0, r"C:\Users\paytonm\Documents\GENREG")
import importlib.util
_p = r"C:\Users\paytonm\AppData\Local\Temp\claude\C--Users-paytonm-Documents-GENREG\7ac18d53-1944-489a-adf5-f5e8091f196b\scratchpad"
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
eg = _load("eg", _p + r"\exp_genreg.py")
cv = _load("cv", _p + r"\exp_conv.py")
act_apply, build, split, FRAMES = eg.act_apply, eg.build, eg.split, eg.FRAMES
K, STRIDE, SIDE = cv.K, cv.STRIDE, cv.SIDE


CROP = 20        # centered window; shapes span <=~14px so this contains them


def pooled_features(X):
    """CENTROID normalization (generic translation-invariance operator, no
    gradients): find each frame's white-pixel center of mass — which tracks the
    shape as it moves — and crop a fixed CROPxCROP window around it. Position is
    removed while ALL shape arrangement is preserved, so a centered shape is
    trivially classifiable. Returns (Nframes, CROP*CROP)."""
    imgs = X.reshape(-1, SIDE, SIDE)
    n = imgs.shape[0]
    ys, xs = np.mgrid[0:SIDE, 0:SIDE].astype(np.float32)
    out = np.zeros((n, CROP, CROP), np.float32)
    half = CROP // 2
    pad = np.pad(imgs, ((0, 0), (half, half), (half, half)))
    for i in range(n):
        img = imgs[i]; tot = img.sum()
        if tot < 1e-6:
            continue
        cy = int(round(float((ys * img).sum() / tot)))
        cx = int(round(float((xs * img).sum() / tot)))
        # in padded coords the centroid is at (cy+half, cx+half); crop around it
        out[i] = pad[i, cy:cy + CROP, cx:cx + CROP]
    return out.reshape(n, CROP * CROP).astype(np.float32)


class HeadPop:
    """Tiny MLP  nfilt -> tanh/act(hid) -> logits(C), evolved (mutation only)."""
    def __init__(self, P, nin, hid, C, seed, mut0=0.05):
        r = np.random.default_rng(seed)
        self.W1 = (r.standard_normal((P, nin, hid)) / np.sqrt(nin)).astype(np.float32)
        self.b1 = np.zeros((P, hid), np.float32)
        self.a1 = r.integers(0, 8, (P, hid)).astype(np.int8)
        self.W2 = (r.standard_normal((P, hid, C)) / np.sqrt(hid)).astype(np.float32)
        self.b2 = np.zeros((P, C), np.float32)
        self.mut_rate = np.full(P, mut0, np.float32)
        self.mut_scale = np.full(P, mut0, np.float32)
        self.energy = np.full(P, 0.7, np.float32)
        self.age = np.zeros(P, np.int32)
        self.fit_ema = np.full(P, 1.0 / C, np.float32)
        self.P, self.C = P, C
        self.WN = ("W1", "b1", "W2", "b2"); self.AN = ("a1",)

    def logits(self, F, idx=None):
        W1, b1, a1, W2, b2 = self.W1, self.b1, self.a1, self.W2, self.b2
        if idx is not None:
            s = slice(idx, idx + 1); W1, b1, a1, W2, b2 = W1[s], b1[s], a1[s], W2[s], b2[s]
        h = act_apply(np.matmul(F[None], W1) + b1[:, None, :], a1)
        return np.matmul(h, W2) + b2[:, None, :]

    def geo_fit(self, F, y):
        z = self.logits(F); z = z - z.max(2, keepdims=True)
        logp = z - np.log(np.exp(z).sum(2, keepdims=True))
        return np.exp(logp[:, np.arange(len(y)), y].mean(1))

    def predict(self, F, idx):
        return self.logits(F, idx=idx)[0].argmax(1)


def run(P=300, nfilt=20, hid=24, gens=6000, seed=1234, DECAY=0.93, GAIN=2.0,
        FLOOR=0.2, E_MAX=1.5, EMA=0.15, anneal_after=0.8, log_every=200):
    X, y, clip_of, shapes, names = build()
    C = len(shapes); n_clips = len(names)
    feat = pooled_features(X)                           # (240, nfeat) precomputed once
    nfeat = feat.shape[1]
    tr, te = split(clip_of, n_clips, FRAMES, seed + 7)
    rng = np.random.default_rng(seed)
    g = HeadPop(P, nfeat, hid, C, seed)
    print(f"features: {nfeat}  (max+mean pool of {nfeat//2} filters)", flush=True)
    CHANCE = 1.0 / C

    def heldout(idx):
        cor = tot = 0; per = {}
        for ci in range(n_clips):
            pr = g.predict(feat[te[ci]], idx)
            c = int((pr == y[te[ci]]).sum()); cor += c; tot += len(te[ci])
            per[names[ci]] = round(c / len(te[ci]), 2)
        return cor / tot, per
    def train_acc(idx):
        cor = tot = 0
        for ci in range(n_clips):
            pr = g.predict(feat[tr[ci]], idx)
            cor += int((pr == y[tr[ci]]).sum()); tot += len(tr[ci])
        return cor / tot

    seq = []
    def next_clip():
        if not seq: seq.extend(int(i) for i in rng.permutation(n_clips))
        return seq.pop()

    t0 = time.time(); best_ho = 0.0; hof = None; hof_score = -1.0
    for gen in range(1, gens + 1):
        ci = next_clip(); m = tr[ci]
        fit = g.geo_fit(feat[m], y[m])
        g.fit_ema = (1 - EMA) * g.fit_ema + EMA * fit.astype(np.float32)
        g.energy = np.clip(g.energy * DECAY + GAIN * (g.fit_ema - CHANCE), 0.0, E_MAX)
        g.age += 1
        dead = np.where(g.energy < FLOOR)[0]
        tgt = max(int(0.05 * P), 1)
        if len(dead) < tgt: dead = np.argsort(g.energy)[:tgt]
        mature = np.setdiff1d(np.where(g.age >= 1)[0], dead)
        if len(mature) == 0: mature = np.setdiff1d(np.arange(P), dead)
        w = np.maximum(g.energy[mature] - FLOOR, 1e-6); w = w / w.sum()
        scale = g.mut_scale * (0.5 if gen > anneal_after * gens else 1.0)
        for d in dead:
            p = mature[rng.choice(len(mature), p=w)]
            mr = float(np.clip(g.mut_rate[p] * np.exp(0.2 * rng.standard_normal()), 0.005, 0.2))
            ms = float(np.clip(scale[p] * np.exp(0.2 * rng.standard_normal()), 0.02, 0.3))
            for nm in g.WN:
                a = getattr(g, nm); a[d] = a[p] + rng.standard_normal(a[p].shape).astype(np.float32) * ms
            for nm in g.AN:
                a = getattr(g, nm); ch = a[p].copy(); fl = rng.random(ch.shape) < mr
                ch[fl] = rng.integers(0, 8, int(fl.sum())).astype(np.int8); a[d] = ch
            g.mut_rate[d] = mr; g.mut_scale[d] = ms; g.energy[d] = 0.7; g.age[d] = 0
            g.fit_ema[d] = g.fit_ema[p]
        if gen % 10 == 0 or gen == 1:
            cand = int(np.argmax(g.fit_ema)); ts = train_acc(cand)
            if ts > hof_score:
                hof_score = ts; hof = {nm: getattr(g, nm)[cand].copy() for nm in g.WN + g.AN}
        if gen % log_every == 0 or gen == 1:
            if hof:
                for nm, v in hof.items(): getattr(g, nm)[0] = v
            ho, per = heldout(0)
            if ho > best_ho: best_ho = ho
            print(f"gen {gen:5d}  HoF_train {hof_score:.3f}  heldout {ho:.3f}  best {best_ho:.3f}  "
                  f"medE {np.median(g.energy):.2f}  medEMA {np.median(g.fit_ema):.3f}  {time.time()-t0:.0f}s", flush=True)
            if gen % (log_every * 3) == 0 or ho >= 0.9:
                print(f"        per-clip {per}", flush=True)
            if ho >= 1.0:
                print(f"*** ALL SHAPES (held-out) at gen {gen} ***", flush=True); break
    if hof:
        for nm, v in hof.items(): getattr(g, nm)[0] = v
    ho, per = heldout(0)
    print(f"END HoF_train {hof_score:.4f}  heldout {ho:.4f}  per-clip {per}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    run(P=300, nfilt=20, gens=6000, log_every=200)
