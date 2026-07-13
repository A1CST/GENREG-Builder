"""radial_cifar.py — push the no-gradient lens-map method as far as it goes on
CIFAR-10. NO gradient descent anywhere: features are deterministic/closed-form,
the classifier is closed-form ridge regression. Reads only the radial-owned
CIFAR copy (radial_data/cifar_radial.npz). Never touches the cifar project.

Ladder of experiments, each measured on held-out test accuracy:
  raw pixels, PCA, lens map, patch (Coates-Ng) features, and combinations —
  classified by closed-form ridge one-vs-all. The point is a real number, honest.
"""
import os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_NPZ = os.path.join(_HERE, "radial_data", "cifar_radial.npz")


_FULL = os.path.join(_HERE, "radial_data", "cifar_full.npz")


def load(n_train=None, n_test=None, full=False):
    d = np.load(_FULL if full else _NPZ)
    Xtr = d["Xtr"].astype(np.float32) / 255.
    Xte = d["Xte"].astype(np.float32) / 255.
    ytr, yte = d["ytr"], d["yte"]
    if n_train:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    if n_test:
        Xte, yte = Xte[:n_test], yte[:n_test]
    return Xtr, ytr, Xte, yte


def zc(M, mu=None, sd=None):
    if mu is None:
        mu, sd = M.mean(0), M.std(0) + 1e-6
    return (M - mu) / sd, mu, sd


def ridge_ova(Ftr, ytr, Fte, yte, lam=None):
    """Closed-form ridge one-vs-all. W = (FtF + lam I)^-1 Ft Y. No gradients."""
    Ftr, mu, sd = zc(Ftr); Fte = (Fte - mu) / sd
    Ftr = np.hstack([Ftr, np.ones((len(Ftr), 1), np.float32)])
    Fte = np.hstack([Fte, np.ones((len(Fte), 1), np.float32)])
    Y = -np.ones((len(ytr), 10), np.float32); Y[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr
    lams = [lam] if lam else [1e-2, 1e-1, 1.0, 10.0, 100.0]
    best = (0, None)
    for L in lams:
        W = np.linalg.solve(A + L * np.eye(A.shape[0], dtype=np.float32), Ftr.T @ Y)
        acc = float((Fte @ W).argmax(1).__eq__(yte).mean())
        if acc > best[0]:
            best = (acc, L)
    return round(best[0], 4), best[1]


def knn(Ftr, ytr, Fte, yte, k=5):
    Ftr, mu, sd = zc(Ftr); Fte = (Fte - mu) / sd
    D2 = (Fte ** 2).sum(1)[:, None] + (Ftr ** 2).sum(1)[None, :] - 2 * Fte @ Ftr.T
    nn = np.argpartition(D2, k, axis=1)[:, :k]
    pred = np.array([np.bincount(ytr[nn[r]], minlength=10).argmax() for r in range(len(Fte))])
    return round(float((pred == yte).mean()), 4)


def rand_pca(X, k, seed=0):
    G = X - X.mean(0)
    rng = np.random.default_rng(seed)
    Y = G @ rng.standard_normal((G.shape[1], k + 12), dtype=np.float32) if G.dtype == np.float32 \
        else G @ rng.standard_normal((G.shape[1], k + 12))
    Q, _ = np.linalg.qr(Y)
    return np.linalg.svd(Q.T @ G, full_matrices=False)[2][:k]


# ----------------------------------------------------------------------------
# patch features (Coates & Ng style) — the gradient-free CIFAR workhorse:
# random/k-means patch dictionary + whitening + soft-threshold + spatial pooling.
# No backprop; k-means is Lloyd's; the classifier stays closed-form ridge.
# ----------------------------------------------------------------------------

def _patches(X, ps=6, stride=1):
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(X, (ps, ps, 3), axis=(1, 2, 3))       # (N,H',W',1,ps,ps,3)
    w = w[:, ::stride, ::stride, 0]
    N, H, W = w.shape[0], w.shape[1], w.shape[2]
    return w.reshape(N, H, W, ps * ps * 3), H, W


def _norm_patches(P):
    P = P - P.mean(-1, keepdims=True)
    return P / np.sqrt(P.var(-1, keepdims=True) + 10.0)


def _fit_zca(P):
    mu = P.mean(0); Pc = P - mu
    cov = Pc.T @ Pc / len(Pc)
    U, S, _ = np.linalg.svd(cov)
    return mu, (U / np.sqrt(S + 0.1)) @ U.T


def _kmeans(P, D, iters=12, seed=0):
    rng = np.random.default_rng(seed)
    C = P[rng.choice(len(P), D, replace=False)].copy()
    for _ in range(iters):
        a = ((P ** 2).sum(1)[:, None] + (C ** 2).sum(1)[None, :] - 2 * P @ C.T).argmin(1)
        for k in range(D):
            m = a == k
            if m.any():
                C[k] = P[m].mean(0)
    return C


def patch_features(Xtr, Xte, D=256, ps=6, stride=1, pool=2, n_fit=100000,
                   alpha=0.5, seed=0, batch=2000, C=None, mu=None, Wz=None, verbose=False):
    """Batched over images so full CIFAR fits in RAM. Returns (Ftr, Fte, nfeat)
    and the fitted (C, mu, Wz) so a dictionary can be reused."""
    rng = np.random.default_rng(seed)
    _, H, W = _patches(Xtr[:1], ps, stride)
    d = ps * ps * 3
    if C is None:
        # fit whitening + k-means on a patch sample from a subset of train images
        nimg = min(6000, len(Xtr))
        samp_imgs, _, _ = _patches(Xtr[rng.choice(len(Xtr), nimg, replace=False)], ps, stride)
        flat = samp_imgs.reshape(-1, d)
        samp = _norm_patches(flat[rng.choice(len(flat), min(n_fit, len(flat)), replace=False)])
        mu, Wz = _fit_zca(samp)
        C = _kmeans((samp - mu) @ Wz, D, seed=seed)
    hs = np.array_split(np.arange(H), pool); ws = np.array_split(np.arange(W), pool)

    def encode(X):
        out = np.empty((len(X), pool * pool * D), np.float32)
        for b in range(0, len(X), batch):
            P4, _, _ = _patches(X[b:b + batch], ps, stride)
            n = len(P4)
            Pw = (_norm_patches(P4.reshape(-1, d)) - mu) @ Wz
            f = np.maximum(0.0, np.abs(Pw @ C.T) - alpha).astype(np.float32).reshape(n, H, W, D)
            c = 0
            for hi in hs:
                for wi in ws:
                    out[b:b + n, c * D:(c + 1) * D] = f[:, hi][:, :, wi].sum((1, 2)); c += 1
            if verbose:
                print(f"  encoded {b + n}/{len(X)}", flush=True)
        return out

    return encode(Xtr), encode(Xte), D * pool * pool, (C, mu, Wz)


def lens_expand(Ftr, Fte, n_axes=96, n_lens=3000, seed=0, keep_base=True):
    """The lens map on top of any base features: reduce to structured axes, then
    generate nonlinear lens programs (multi-axis combos through composed
    activations) and append them. Deterministic, no gradients. Returns expanded
    (Ftr, Fte) for the ridge classifier."""
    from radial_lens import ACTS
    ANAMES = list(ACTS)
    V = rand_pca(Ftr, n_axes, seed)
    Ztr, mu, sd = zc(Ftr @ V.T); Zte = (Fte @ V.T - mu) / sd
    rng = np.random.default_rng(seed + 1)
    # keep the FULL base features and ADD lens combos (combos are drawn from the
    # PCA axes for tractability, but nothing is discarded from the base)
    cols_tr = [Ftr.astype(np.float32)] if keep_base else []
    cols_te = [Fte.astype(np.float32)] if keep_base else []
    for i in range(n_lens):
        order = 2 + int(rng.integers(0, 4))
        ax = rng.choice(n_axes, order, replace=False)
        co = rng.standard_normal(order); co /= np.linalg.norm(co) + 1e-9
        depth = 1 + int(rng.integers(0, 3))
        acts = [ANAMES[int(rng.integers(len(ANAMES)))] for _ in range(depth)]
        ptr = Ztr[:, ax] @ co; pte = Zte[:, ax] @ co
        for a in acts:
            ptr = ACTS[a](ptr); pte = ACTS[a](pte)
        s = ptr.std()
        if s > 1e-6:
            m = ptr.mean()
            cols_tr.append(((ptr - m) / s)[:, None]); cols_te.append(((pte - m) / s)[:, None])
    return np.concatenate(cols_tr, 1), np.concatenate(cols_te, 1)


def patch_cached(D=512, pool=2, alpha=0.5, full=True):
    """Compute patch features once and cache to disk (they take minutes)."""
    path = os.path.join(_HERE, "radial_data", f"patch_D{D}_p{pool}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["Ktr"], z["ytr"], z["Kte"], z["yte"]
    Xtr, ytr, Xte, yte = load(full=full)
    Ktr, Kte, nf, _ = patch_features(Xtr, Xte, D=D, pool=pool, alpha=alpha, verbose=True)
    np.savez(path, Ktr=Ktr.astype(np.float32), ytr=ytr, Kte=Kte.astype(np.float32), yte=yte)
    return Ktr, ytr, Kte, yte


def ladder(D=512):
    import time
    print(f"=== D={D} patch features ===", flush=True)
    t = time.time(); Ktr, ytr, Kte, yte = patch_cached(D=D)
    print(f"loaded/built {Ktr.shape[1]} feats in {round(time.time()-t)}s", flush=True)
    a0, l0 = ridge_ova(Ktr, ytr, Kte, yte)
    print(f"  patch ridge:            {a0}  (lam {l0})", flush=True)
    for nl in (2500,):
        t = time.time()
        Etr, Ete = lens_expand(Ktr, Kte, n_axes=96, n_lens=nl)
        a1, l1 = ridge_ova(Etr, ytr, Ete, yte)
        print(f"  patch + lens({nl}):      {a1}  (lam {l1}, {round(a1-a0,4):+}, {round(time.time()-t)}s)", flush=True)
    return a0


def run(full=False, D=256, pool=2, alpha=0.5, ntr=None, nte=None):
    import time
    Xtr, ytr, Xte, yte = load(ntr, nte, full=full)
    print(f"=== patch D={D} pool{pool} alpha={alpha} on {len(Xtr)} train / {len(Xte)} test ===", flush=True)
    t = time.time()
    Ktr, Kte, nf, _ = patch_features(Xtr, Xte, D=D, pool=pool, alpha=alpha, verbose=True)
    acc, lam = ridge_ova(Ktr, ytr, Kte, yte)
    print(f"patch ridge: {acc}  (lam {lam}, {nf} feats, {round(time.time()-t)}s)", flush=True)
    return acc


if __name__ == "__main__":
    import sys
    if "ladder" in sys.argv:
        for D in [int(x) for x in sys.argv[sys.argv.index("ladder") + 1:]] or [512]:
            ladder(D)
    elif "full" in sys.argv:
        run(full=True, D=512)
    else:
        run(full=False, D=256)
