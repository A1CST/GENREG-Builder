"""§VII verification of the centroid-normalized evolved shape classifier.
Multiple random 80/20 per-clip splits; report train/heldout top-1, per-clip,
and the train->heldout drop. Majority-class baseline = 1/10 = 0.10."""
import sys, time, numpy as np, importlib.util
sys.path.insert(0, r"C:\Users\paytonm\Documents\GENREG")
_p = r"C:\Users\paytonm\AppData\Local\Temp\claude\C--Users-paytonm-Documents-GENREG\7ac18d53-1944-489a-adf5-f5e8091f196b\scratchpad"
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
tp = _load("tp", _p + r"\exp_twophase.py")

X, y, clip_of, shapes, names = tp.build()
C = len(shapes); n_clips = len(names)
feat = tp.pooled_features(X)
maj = np.bincount(y).max() / len(y)
print(f"features {feat.shape[1]}-d | classes {C} | majority-class baseline {maj:.3f}", flush=True)

def evolve(Ftr, ytr, seed, gens=400):
    P, hid = 300, 24
    rng = np.random.default_rng(seed)
    g = tp.HeadPop(P, feat.shape[1], hid, C, seed)
    sig = np.full(P, 0.05, np.float32); ne = int(0.2 * P)
    for gen in range(gens):
        z = g.logits(Ftr); z = z - z.max(2, keepdims=True)
        lp = z - np.log(np.exp(z).sum(2, keepdims=True))
        fit = np.exp(lp[:, np.arange(len(ytr)), ytr].mean(1))
        order = np.argsort(-fit); elite = order[:ne]
        if float((g.predict(Ftr, int(order[0])) == ytr).mean()) >= 1.0:
            break
        par = elite[rng.integers(0, ne, P - ne)]
        cs = np.clip(sig[par] * np.exp(0.2 * rng.standard_normal(P - ne).astype(np.float32)), 0.02, 0.3)
        for nm in g.WN:
            w = getattr(g, nm); st = cs.reshape((P - ne,) + (1,) * (w.ndim - 1))
            setattr(g, nm, np.concatenate([w[elite], w[par] + rng.standard_normal(w[par].shape).astype(np.float32) * st]))
        a = g.a1; ch = a[par].copy(); fl = rng.random(ch.shape) < 0.03
        ch[fl] = rng.integers(0, 8, int(fl.sum())).astype(np.int8)
        g.a1 = np.concatenate([a[elite], ch]); sig = np.concatenate([sig[elite], cs])
    return g, gen

for seed in (11, 22, 33):
    tr, te = tp.split(clip_of, n_clips, tp.FRAMES, seed)
    tri, tei = np.concatenate(tr), np.concatenate(te)
    t0 = time.time()
    g, gens = evolve(feat[tri], y[tri], seed)
    tra = float((g.predict(feat[tri], 0) == y[tri]).mean())
    hoa = float((g.predict(feat[tei], 0) == y[tei]).mean())
    per = {names[ci]: round(float((g.predict(feat[te[ci]], 0) == y[te[ci]]).mean()), 2) for ci in range(n_clips)}
    drop = (tra - hoa) / tra if tra else 0
    print(f"seed {seed}: train {tra:.3f}  heldout {hoa:.3f}  drop {drop*100:.1f}%  "
          f"conv@gen {gens}  ({time.time()-t0:.0f}s)\n   per-clip {per}", flush=True)
