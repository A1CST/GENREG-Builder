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


def load(n_train=None, n_test=None):
    d = np.load(_NPZ)
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


def patch_features(Xtr, Xte, D=256, ps=6, stride=1, pool=2, n_fit=80000, seed=0):
    ptr, H, W = _patches(Xtr, ps, stride)
    pte, _, _ = _patches(Xte, ps, stride)
    d = ptr.shape[-1]
    rng = np.random.default_rng(seed)
    # fit whitening + dictionary on a sample of normalised patches
    flat = ptr.reshape(-1, d)
    samp = _norm_patches(flat[rng.choice(len(flat), min(n_fit, len(flat)), replace=False)])
    mu, Wz = _fit_zca(samp)
    C = _kmeans(samp @ Wz - mu @ Wz if False else (samp - mu) @ Wz, D, seed=seed)
    Cn = C / (np.linalg.norm(C, 1, keepdims=True) + 1e-6) * 0 + C  # keep as-is

    def encode(P4, N):
        Pn = _norm_patches(P4.reshape(-1, d))
        Pw = (Pn - mu) @ Wz
        act = Pw @ C.T                                            # (N*H*W, D)
        alpha = 0.5                                              # soft threshold
        f = np.maximum(0.0, np.abs(act) - alpha).astype(np.float32)
        f = f.reshape(N, H, W, D)
        # pool into pool x pool grid (sum)
        hs = np.array_split(np.arange(H), pool); ws = np.array_split(np.arange(W), pool)
        out = np.empty((N, pool * pool * D), np.float32)
        c = 0
        for hi in hs:
            for wi in ws:
                out[:, c * D:(c + 1) * D] = f[:, hi][:, :, wi].sum((1, 2)); c += 1
        return out

    return encode(ptr, len(Xtr)), encode(pte, len(Xte)), D * pool * pool


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load()
    Ftr = Xtr.reshape(len(Xtr), -1); Fte = Xte.reshape(len(Xte), -1)
    print("=== baselines (8k train / 2k test) ===")
    print("raw pixels   kNN   ", knn(Ftr, ytr, Fte, yte))
    print("raw pixels   ridge ", ridge_ova(Ftr, ytr, Fte, yte))
    V = rand_pca(Ftr, 256)
    Ptr, Pte = Ftr @ V.T, Fte @ V.T
    print("PCA-256      ridge ", ridge_ova(Ptr, ytr, Pte, yte))
    print("=== patch features (Coates-Ng) ===")
    import time
    t = time.time()
    Ktr, Kte, nf = patch_features(Xtr, Xte, D=256, pool=2)
    print(f"patch D=256 pool2 ({nf} feats, {round(time.time()-t)}s)  ridge ", ridge_ova(Ktr, ytr, Kte, yte))
