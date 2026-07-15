"""CIFAR-Pipe — the MNIST-Pipe program, verbatim, on CIFAR-10 (staged 2026-07-08).

SAME EXACT model and process as `mnist_pipe` — this module only adapts the
data plumbing (32x32 RGB instead of 28x28 gray) and reuses the MNIST machinery
wherever it is feature-agnostic (ga_step, binary/joint/mixer trainers, the
pipeline predict). The pipeline:

  ENVIRONMENT (built + evolved, never trained end-to-end):
    v2 — built statistics: per-channel zone means + row/col profiles,
         gradient-orientation histograms on luminance, PCA of raw pixels.
    v4 — + the EVOLVED DETECTOR BANK: 5x5x3 conv-kernel genomes with evolved
         per-neuron activations (8-function catalog), Fisher class-separability
         fitness, harvested across seeded rounds, correlation-decorrelated;
         multi-shape mean pools (3x3 / 4x2 / 2x4); everything PCA'd to 1024.

  CLASSIFIER GENOMES (all gradient-free, all from mnist_pipe):
    10 one-vs-rest detectors -> joint linear head (centroid warm start,
    full-train deterministic landscape, magnitude-scaled mutation, L2) ->
    45 one-vs-one pairwise referees -> margin gate on val -> one-shot test.

STAGED ONLY — nothing here has been run yet (per instruction: MNIST first).
Entry points when ready:
    python -m genreg_train.cifar_pipe --detbank      # evolve the conv bank
    python -m genreg_train.cifar_pipe --v4           # full battery on v4
    python -m genreg_train.cifar_pipe                # v2 battery (no bank)
"""
import os
import pickle
import time

import numpy as np

from genreg_train import mnist_pipe as mp
from genreg_train import evo_gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "corpora", "cifar10", "cifar-10-batches-py")
CACHE = os.path.join(ROOT, "demo", "cifar_genomes.pkl")
DETBANK = os.path.join(ROOT, "demo", "cifar_detbank.pkl")

N_VAL = 5000
LABELS = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse",
          "ship", "truck"]

# response map is 28x28 (32 - 5 + 1); same pool shapes as MNIST
RESP = 28
POOLS = mp.POOLS                                   # (3,3), (4,2), (2,4) -> 25 dims


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
_CIFAR = None


def _batch(name):
    with open(os.path.join(DATA_DIR, name), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,32,32,3)
    return X.astype(np.float32) / 255.0, np.array(d[b"labels"], np.int64)


def load_cifar():
    """(Xtr (45k,32,32,3), ytr, Xva (5k,..), yva, Xte (10k,..), yte)."""
    global _CIFAR
    if _CIFAR is None:
        Xs, ys = zip(*[_batch(f"data_batch_{i}") for i in range(1, 6)])
        X, y = np.concatenate(Xs), np.concatenate(ys)
        Xt, yt = _batch("test_batch")
        n = len(X) - N_VAL
        _CIFAR = (X[:n], y[:n], X[n:], y[n:], Xt, yt)
    return _CIFAR


def gray(X):
    """(N,32,32,3) -> (N,32,32) luminance."""
    return (0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2]).astype(np.float32)


# --------------------------------------------------------------------------
# Built statistics (the v2 environment) — MNIST's stat bank adapted to 32px RGB
# --------------------------------------------------------------------------
def _zones(X, g):
    """Per-channel g x g zone means -> (N, g*g*3)."""
    N = len(X)
    s = 32 // g
    return X[:, :g * s, :g * s, :].reshape(N, g, s, g, s, 3) \
        .mean(axis=(2, 4)).reshape(N, g * g * 3)


def _profiles(X):
    """Per-channel row + column mean profiles, downsampled to 16 each -> (N,96)."""
    rows = X.mean(axis=2).reshape(len(X), 16, 2, 3).mean(axis=2)
    cols = X.mean(axis=1).reshape(len(X), 16, 2, 3).mean(axis=2)
    return np.concatenate([rows.reshape(len(X), -1), cols.reshape(len(X), -1)], axis=1)


def _grad_hist(G, cells, nbin=8):
    """Gradient-orientation histograms on luminance (32px version of the MNIST
    stat): magnitude-weighted votes into nbin bins per cell -> (N, cells^2*nbin)."""
    gy = np.zeros_like(G); gx = np.zeros_like(G)
    gy[:, 1:-1, :] = G[:, 2:, :] - G[:, :-2, :]
    gx[:, :, 1:-1] = G[:, :, 2:] - G[:, :, :-2]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    b = ((ang + np.pi) / (2 * np.pi) * nbin).astype(np.int64) % nbin
    N = len(G)
    s = 32 // cells
    out = np.zeros((N, cells, cells, nbin), np.float32)
    cy = (np.arange(32) // s).clip(0, cells - 1)
    n_idx = np.repeat(np.arange(N), 32 * 32)
    cy_idx = np.tile(np.repeat(cy, 32), N)
    cx_idx = np.tile(np.tile(cy, 32), N)
    np.add.at(out, (n_idx, cy_idx, cx_idx, b.reshape(-1)), mag.reshape(-1))
    out = out.reshape(N, -1)
    return out / (out.sum(axis=1, keepdims=True) + 1e-6)


def stat_features(X, pca=None):
    G = gray(X)
    parts = [_zones(X, 4), _zones(X, 8), _profiles(X),
             _grad_hist(G, 4), _grad_hist(G, 8)]
    if pca is not None:
        mu, comps = pca
        parts.append((X.reshape(len(X), -1) - mu) @ comps.T)
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_pca(Xtr, D=128):
    """Unsupervised PCA of raw train pixels (3072 dims)."""
    F = Xtr.reshape(len(Xtr), -1)
    mu = F.mean(axis=0)
    C = np.cov((F - mu).T.astype(np.float64))
    w, v = np.linalg.eigh(C)
    comps = v[:, ::-1][:, :D].T.astype(np.float32)
    return mu.astype(np.float32), comps


# --------------------------------------------------------------------------
# Patch whitening (ZCA) — round-2 environment conditioning. Unsupervised
# statistics of 5x5x3 patches (no labels), same legal status as PCA: it
# reorganises the patch space so that distinct structure, not raw contrast,
# wins Fisher selection. Classic result: whitening is the single biggest
# lever for shallow features on CIFAR.
# --------------------------------------------------------------------------
_ZCA = None


def build_zca(n_patch=200000, eps=0.01, seed=0):
    """(mean (75,), W_zca (75,75)) from random train patches. Cached."""
    global _ZCA
    if _ZCA is None:
        Xtr = load_cifar()[0]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(Xtr), n_patch)
        ys = rng.integers(0, RESP, n_patch)
        xs = rng.integers(0, RESP, n_patch)
        P = np.stack([Xtr[i, y:y + 5, x:x + 5, :].reshape(75)
                      for i, y, x in zip(idx, ys, xs)]).astype(np.float64)
        mu = P.mean(0)
        C = np.cov((P - mu).T)
        w, v = np.linalg.eigh(C)
        Wz = v @ np.diag(1.0 / np.sqrt(w + eps)) @ v.T
        _ZCA = (mu.astype(np.float32), Wz.astype(np.float32))
    return _ZCA


def _whiten_patches(Pf):
    """(N,75) raw patches -> ZCA-whitened."""
    mu, Wz = build_zca()
    return (Pf - mu) @ Wz


# --------------------------------------------------------------------------
# Evolved detector bank (the v4 environment) — 5x5x3 kernels, same recipe
# --------------------------------------------------------------------------
def _im2col5c(X):
    """(N,32,32,3) -> (N, 28*28, 75) sliding 5x5x3 patches."""
    N = len(X)
    s0, s1, s2, s3 = X.strides
    v = np.lib.stride_tricks.as_strided(
        X, (N, RESP, RESP, 5, 5, 3), (s0, s1, s2, s1, s2, s3))
    return np.ascontiguousarray(v.reshape(N, RESP * RESP, 75))


def _pool_resp(resp, pools=None):
    """(N,28,28) activated response -> (N,PD) multi-shape mean pools."""
    pools = pools or POOLS
    N = len(resp)
    out = []
    for r, c in pools:
        hr, hc = RESP // r, RESP // c
        out.append(resp[:, :r * hr, :c * hc].reshape(N, r, hr, c, hc)
                   .mean(axis=(2, 4)).reshape(N, r * c))
    return np.concatenate(out, axis=1)


def _pool_all(resp, act, pools=None):
    """(N,28,28,P) responses + activation ids -> (P,N,PD)."""
    pools = pools or POOLS
    PD = sum(r * c for r, c in pools)
    N, _, _, P = resp.shape
    pooled = np.empty((P, N, PD), np.float32)
    for a in range(8):
        ids = np.where(act == a)[0]
        if len(ids) == 0:
            continue
        blk = mp._acts(resp[..., ids], a).transpose(3, 0, 1, 2) \
            .reshape(len(ids) * N, RESP, RESP)
        pooled[ids] = _pool_resp(blk, pools).reshape(len(ids), N, PD)
    return pooled


def evolve_detbank(n_bank=96, rounds=48, pop=48, gens=60, sub=2500, seed=7,
                   corr_cap=0.95, log=print):
    """Identical procedure to mp.evolve_detbank; kernels are 75-gene (5x5x3)."""
    Xtr, ytr = load_cifar()[0], load_cifar()[1]
    rng0 = np.random.default_rng(seed)
    harvest = []
    for rd in range(rounds):
        rs = np.random.default_rng(seed + 101 * rd)
        idx = np.concatenate([rs.choice(np.where(ytr == c)[0], sub // 10, replace=False)
                              for c in range(10)])
        Xs, ys = Xtr[idx], ytr[idx]
        patches = _im2col5c(Xs)
        Pf = patches.reshape(-1, 75)
        dbf = None
        if evo_gpu.HAS_GPU:
            dbf = evo_gpu.DetbankFitGPU(patches, ys, RESP, POOLS)
        K = (rs.standard_normal((pop, 75)) * 0.25).astype(np.float32)
        b = np.zeros(pop, np.float32)
        act = rs.integers(0, 8, pop).astype(np.float32)
        sigma = np.full(pop, 0.08, np.float32)
        for gen in range(1, gens + 1):
            ai = np.round(act).astype(np.int64) % 8
            if dbf is not None:
                fit = dbf(K, b, ai)
            else:
                resp = (Pf @ K.T + b).reshape(len(Xs), RESP, RESP, pop)
                fit = mp._fisher_all(_pool_all(resp, ai), ys)
            if gen % 20 == 0 or gen == gens:
                o = np.argsort(fit)[::-1][:8]
                for j in o:
                    harvest.append((float(fit[j]), K[j].copy(), float(b[j]),
                                    int(act[j]) % 8, None))
            pd = {"K": K, "b": b, "act": act, "sigma": sigma}
            mp.ga_step(pd, fit, rng0)
            K, b, act, sigma = pd["K"], pd["b"], pd["act"], pd["sigma"]
            act = np.round(act) % 8
        log(f"  [cifar-detbank] round {rd + 1}/{rounds}: best fisher "
            f"{max(h[0] for h in harvest):.3f}, harvested {len(harvest)}")
    rr = np.random.default_rng(seed + 9999)
    ridx = np.concatenate([rr.choice(np.where(ytr == c)[0], 200, replace=False)
                           for c in range(10)])
    Pref = _im2col5c(Xtr[ridx]).reshape(-1, 75)
    harvest.sort(key=lambda h: -h[0])
    picked, feats = [], []
    for f, K, b, a, _ in harvest:
        resp = (Pref @ K + b).reshape(len(ridx), RESP, RESP)
        v = _pool_resp(mp._acts(resp, a)).reshape(-1)
        v = (v - v.mean()) / (v.std() + 1e-8)
        if any(abs(float(v @ u) / len(v)) > corr_cap for u in feats):
            continue
        picked.append((K, b, a)); feats.append(v)
        if len(picked) >= n_bank:
            break
    bank = {"K": np.stack([p[0] for p in picked]),
            "b": np.array([p[1] for p in picked], np.float32),
            "act": np.array([p[2] for p in picked], np.int64)}
    with open(DETBANK, "wb") as f:
        pickle.dump(bank, f)
    log(f"cifar detector bank: {len(picked)} genomes (from {len(harvest)}) -> {DETBANK}")
    return bank


def bank_features(X, bank, chunk=1024):
    pools = tuple(tuple(p) for p in bank.get("pools", POOLS))
    white = bank.get("whitened", False)
    if evo_gpu.HAS_GPU:
        fn = (lambda Xc: _whiten_patches(_im2col5c(Xc).reshape(-1, 75))
              .reshape(len(Xc), RESP * RESP, 75)) if white else _im2col5c
        return evo_gpu.bank_features_gpu(fn, X, bank, RESP, pools)
    nb = len(bank["K"])
    PD = sum(r * c for r, c in pools)
    out = np.empty((len(X), nb * PD), np.float32)
    for lo in range(0, len(X), chunk):
        Xc = X[lo:lo + chunk]
        P = _im2col5c(Xc).reshape(-1, 75)
        if white:
            P = _whiten_patches(P)
        resp = (P @ bank["K"].T + bank["b"]).reshape(len(Xc), RESP, RESP, nb)
        for j in range(nb):
            out[lo:lo + len(Xc), j * PD:(j + 1) * PD] = \
                _pool_resp(mp._acts(resp[..., j], int(bank["act"][j])), pools)
    return out


# --------------------------------------------------------------------------
# v5 environment — NICHED bank evolution on whitened patches. The diversity
# collapse (15 usable detectors from one global Fisher objective) is fixed by
# decomposition: each of 55 niches (45 class pairs + 10 one-vs-rest) breeds
# its own small population whose survival condition is Fisher separability of
# ITS distinction only. The WordPipe principle applied to the feature layer:
# the constraint is the genome's reason for existing.
# --------------------------------------------------------------------------
DETBANK5 = os.path.join(ROOT, "demo", "cifar_detbank_v5.pkl")
POOLS5 = ((3, 3), (4, 2), (2, 4), (4, 4))          # +4x4 for spatial detail -> 41


def evolve_detbank_niched(n_bank=192, pop=32, gens=40, per_class=400, seed=7,
                          corr_cap=0.95, harvest_top=6, log=print, out=None):
    Xtr, ytr = load_cifar()[0], load_cifar()[1]
    rng0 = np.random.default_rng(seed)
    niches = [(a, b) for a in range(10) for b in range(a + 1, 10)] \
        + [(c, -1) for c in range(10)]             # -1 = rest
    harvest = []
    for ni, (a, b) in enumerate(niches):
        rs = np.random.default_rng(seed + 977 * ni)
        ia = rs.choice(np.where(ytr == a)[0], per_class, replace=False)
        if b >= 0:
            ib = rs.choice(np.where(ytr == b)[0], per_class, replace=False)
        else:
            ib = rs.choice(np.where(ytr != a)[0], per_class, replace=False)
        idx = np.concatenate([ia, ib])
        ys = (np.arange(len(idx)) >= per_class).astype(np.int64)   # 0=a, 1=b/rest
        patches = _whiten_patches(_im2col5c(Xtr[idx]).reshape(-1, 75)) \
            .reshape(len(idx), RESP * RESP, 75)
        dbf = evo_gpu.DetbankFitGPU(patches, ys, RESP, POOLS5, nc=2) \
            if evo_gpu.HAS_GPU else None
        K = (rs.standard_normal((pop, 75)) * 0.25).astype(np.float32)
        bb = np.zeros(pop, np.float32)
        act = rs.integers(0, 8, pop).astype(np.float32)
        sigma = np.full(pop, 0.08, np.float32)
        Pf = patches.reshape(-1, 75)
        for gen in range(1, gens + 1):
            ai = np.round(act).astype(np.int64) % 8
            if dbf is not None:
                fit = dbf(K, bb, ai)
            else:
                resp = (Pf @ K.T + bb).reshape(len(idx), RESP, RESP, pop)
                fit = mp._fisher_all(_pool_all(resp, ai, POOLS5), ys, nc=2)
            if gen == gens:
                o = np.argsort(fit)[::-1][:harvest_top]
                for j in o:
                    harvest.append((float(fit[j]), K[j].copy(), float(bb[j]),
                                    int(ai[j]), (a, b)))
            pd = {"K": K, "b": bb, "act": act, "sigma": sigma}
            mp.ga_step(pd, fit, rng0)
            K, bb, act, sigma = pd["K"], pd["b"], pd["act"], pd["sigma"]
            act = np.round(act) % 8
        if (ni + 1) % 11 == 0:
            log(f"  [niche {ni + 1}/{len(niches)}] best fisher "
                f"{max(h[0] for h in harvest):.3f}, harvested {len(harvest)}")
    # round-robin diverse selection: one candidate per niche per pass (Fisher
    # scores are not comparable ACROSS niches — a cat/dog specialist matters
    # even if its Fisher is lower than an easy pair's), correlation-capped on
    # a common whitened reference set
    rr = np.random.default_rng(seed + 9999)
    ridx = np.concatenate([rr.choice(np.where(ytr == c)[0], 100, replace=False)
                           for c in range(10)])
    Pref = _whiten_patches(_im2col5c(Xtr[ridx]).reshape(-1, 75))
    by_niche = {}
    for h in harvest:
        by_niche.setdefault(h[4], []).append(h)
    for v in by_niche.values():
        v.sort(key=lambda h: -h[0])
    picked, feats = [], []
    for rnd in range(harvest_top):
        for niche in by_niche:
            if rnd >= len(by_niche[niche]) or len(picked) >= n_bank:
                continue
            f, K, bb, a, _ = by_niche[niche][rnd]
            resp = (Pref @ K + bb).reshape(len(ridx), RESP, RESP)
            v = _pool_resp(mp._acts(resp, a), POOLS5).reshape(-1)
            v = (v - v.mean()) / (v.std() + 1e-8)
            if any(abs(float(v @ u) / len(v)) > corr_cap for u in feats):
                continue
            picked.append((K, bb, a)); feats.append(v)
    bank = {"K": np.stack([p[0] for p in picked]),
            "b": np.array([p[1] for p in picked], np.float32),
            "act": np.array([p[2] for p in picked], np.int64),
            "whitened": True, "pools": POOLS5}
    out = out or DETBANK5
    with open(out, "wb") as f:
        pickle.dump(bank, f)
    log(f"niched bank: {len(picked)} genomes (from {len(harvest)} harvested, "
        f"{len(niches)} niches) -> {out}")
    return bank


# --------------------------------------------------------------------------
# v6 environment — DIVERSITY-DRIVEN bank (label-free). Genomes evolve for one
# reason: extract information about images that no other genome extracts.
# Behavior = pooled response pattern over an unlabeled probe set; fitness =
# novelty (vs peers + archive) x information (response entropy). No labels,
# no Fisher, no class niches, no hand statistics, no whitening — the
# anti-collapse pressure IS the novelty term. Labels never touch this layer.
# --------------------------------------------------------------------------
DETBANK6 = os.path.join(ROOT, "demo", "cifar_detbank_v6.pkl")


def evolve_detbank_diverse(n_bank=256, max_rounds=64, pop=64, gens=50,
                           probe_n=2000, admit_per_round=8, dup_cap=0.85,
                           seed=7, log=print, out=None, stability=False,
                           aug="jitter"):
    """Novelty-search bank evolution. Fresh population per round; at round end
    the fittest genomes are admitted to the archive unless their behavior is a
    near-duplicate of an archived one (|cos| > dup_cap). Stops when the
    archive reaches n_bank or max_rounds is exhausted."""
    Xtr = load_cifar()[0]
    rng0 = np.random.default_rng(seed)
    rp = np.random.default_rng(seed + 1)
    probe = Xtr[rp.choice(len(Xtr), probe_n, replace=False)]   # no labels used
    patches = _im2col5c(probe)
    patches_jit = None
    if stability:
        if aug == "occlude":
            # OCCLUSION contrastive signal: the same image with one random
            # 12x12 region blanked to the image mean. Invariance to this
            # cannot be satisfied by any single local measurement — the
            # genome must integrate structure from across the image.
            ro = np.random.default_rng(seed + 2)
            pj = probe.copy()
            ys_o = ro.integers(0, 32 - 12, probe_n)
            xs_o = ro.integers(0, 32 - 12, probe_n)
            means = probe.reshape(probe_n, -1, 3).mean(1)
            for i in range(probe_n):
                pj[i, ys_o[i]:ys_o[i] + 12, xs_o[i]:xs_o[i] + 12, :] = means[i]
        else:                                      # 'jitter': 1px shift
            pj = np.zeros_like(probe)
            pj[:, :-1, :-1, :] = probe[:, 1:, 1:, :]
        patches_jit = _im2col5c(pj)
    if not evo_gpu.HAS_GPU:
        raise RuntimeError("diversity bank evolution requires the GPU backend")
    dfit = evo_gpu.DiversityFitGPU(patches, RESP, POOLS5, patches_jit=patches_jit)
    picked = []
    for rd in range(max_rounds):
        rs = np.random.default_rng(seed + 313 * rd)
        K = (rs.standard_normal((pop, 75)) * 0.25).astype(np.float32)
        bb = np.zeros(pop, np.float32)
        act = rs.integers(0, 8, pop).astype(np.float32)
        sigma = np.full(pop, 0.08, np.float32)
        fit, B = None, None
        for gen in range(1, gens + 1):
            ai = np.round(act).astype(np.int64) % 8
            fit, B = dfit(K, bb, ai)
            pd = {"K": K, "b": bb, "act": act, "sigma": sigma}
            mp.ga_step(pd, fit, rng0)
            K, bb, act, sigma = pd["K"], pd["b"], pd["act"], pd["sigma"]
            act = np.round(act) % 8
        ai = np.round(act).astype(np.int64) % 8
        fit, B = dfit(K, bb, ai)                   # final population state
        order = np.argsort(fit)[::-1]
        dup = dfit.novelty_vs_archive(B)
        admitted = 0
        for j in order:
            if admitted >= admit_per_round or len(picked) >= n_bank:
                break
            if dup[j] > dup_cap:
                continue
            picked.append((K[j].copy(), float(bb[j]), int(ai[j])))
            dfit.admit(B[j:j + 1])
            admitted += 1
        if (rd + 1) % 8 == 0 or len(picked) >= n_bank:
            log(f"  [divbank] round {rd + 1}: archive {len(picked)}, "
                f"best fit {fit[order[0]]:.3f}")
        if len(picked) >= n_bank:
            break
    bank = {"K": np.stack([p[0] for p in picked]),
            "b": np.array([p[1] for p in picked], np.float32),
            "act": np.array([p[2] for p in picked], np.int64),
            "whitened": False, "pools": POOLS5, "mode": "diversity"}
    out = out or DETBANK6
    with open(out, "wb") as f:
        pickle.dump(bank, f)
    log(f"diversity bank: {len(picked)} genomes ({rd + 1} rounds, "
        f"label-free) -> {out}")
    return bank


# --------------------------------------------------------------------------
# v9 environment — UTILITY-DRIVEN ENCODER (user directive 2026-07-09): a
# genome is kept only if adding its output INCREASES the system's fitness.
# Diversity is the consequence, not the objective: a redundant genome adds
# no marginal fitness and dies; a novel-but-useless one likewise. The
# admitted bank acts as an encoder, built one proven genome at a time.
# --------------------------------------------------------------------------
DETBANK9 = os.path.join(ROOT, "demo", "cifar_detbank_v9.pkl")


def evolve_detbank_utility(n_bank=512, max_rounds=768, pop=64, gens=30,
                           probe_n=4000, min_gain=0.0005, patience=40,
                           seed=7, log=print, out=None):
    """Greedy encoder growth: per round, evolve a candidate population whose
    fitness IS its marginal utility (change in nearest-centroid fitness on a
    balanced labelled probe when appended to the encoder so far); admit the
    champion if it clears `min_gain`. Stops at n_bank, max_rounds, or after
    `patience` consecutive rounds without an admission."""
    Xtr, ytr = load_cifar()[0], load_cifar()[1]
    rp = np.random.default_rng(seed + 1)
    idx = np.concatenate([rp.choice(np.where(ytr == c)[0], probe_n // 10,
                                    replace=False) for c in range(10)])
    probe, yp = Xtr[idx], ytr[idx]
    patches = _im2col5c(probe)
    if not evo_gpu.HAS_GPU:
        raise RuntimeError("utility encoder evolution requires the GPU backend")
    ufit = evo_gpu.UtilityFitGPU(patches, yp, RESP, POOLS5)
    rng0 = np.random.default_rng(seed)
    picked, dry = [], 0
    for rd in range(max_rounds):
        rs = np.random.default_rng(seed + 611 * rd)
        K = (rs.standard_normal((pop, 75)) * 0.25).astype(np.float32)
        bb = np.zeros(pop, np.float32)
        act = rs.integers(0, 8, pop).astype(np.float32)
        sigma = np.full(pop, 0.08, np.float32)
        fit, pooled = None, None
        for gen in range(1, gens + 1):
            ai = np.round(act).astype(np.int64) % 8
            fit, pooled = ufit(K, bb, ai)
            pd = {"K": K, "b": bb, "act": act, "sigma": sigma}
            mp.ga_step(pd, fit, rng0)
            K, bb, act, sigma = pd["K"], pd["b"], pd["act"], pd["sigma"]
            act = np.round(act) % 8
        ai = np.round(act).astype(np.int64) % 8
        fit, pooled = ufit(K, bb, ai)
        j = int(np.argmax(fit))
        if fit[j] > min_gain:
            acc = ufit.admit(pooled[j:j + 1])
            picked.append((K[j].copy(), float(bb[j]), int(ai[j])))
            dry = 0
            if len(picked) % 16 == 0:
                log(f"  [utilbank] round {rd + 1}: encoder {len(picked)} genomes, "
                    f"probe fitness {acc:.4f} (last gain +{fit[j]:.4f})")
        else:
            dry += 1
            if dry >= patience:
                log(f"  [utilbank] {patience} dry rounds — utility exhausted")
                break
        if len(picked) >= n_bank:
            break
    bank = {"K": np.stack([p[0] for p in picked]),
            "b": np.array([p[1] for p in picked], np.float32),
            "act": np.array([p[2] for p in picked], np.int64),
            "whitened": False, "pools": POOLS5, "mode": "utility"}
    out = out or DETBANK9
    with open(out, "wb") as f:
        pickle.dump(bank, f)
    log(f"utility encoder: {len(picked)} genomes ({rd + 1} rounds) -> {out}")
    return bank


# --------------------------------------------------------------------------
# v10 environment — HYBRID encoder: occlusion-contrastive evolution GENERATES
# candidates (the objective that produced the best label-free environment);
# the UTILITY GATE decides admission (a genome joins the encoder only if it
# increases the system's fitness — dense margin utility, no plateaus).
# Diversity from the generator, usefulness from the gate.
# --------------------------------------------------------------------------
DETBANK10 = os.path.join(ROOT, "demo", "cifar_detbank_v10.pkl")


def evolve_detbank_hybrid(n_bank=512, max_rounds=640, pop=64, gens=40,
                          probe_n=2000, util_probe_n=4000, patience=48,
                          seed=7, log=print, out=None):
    Xtr, ytr = load_cifar()[0], load_cifar()[1]
    # contrastive generator state (unlabeled probe + occluded copies)
    rp = np.random.default_rng(seed + 1)
    probe = Xtr[rp.choice(len(Xtr), probe_n, replace=False)]
    ro = np.random.default_rng(seed + 2)
    pj = probe.copy()
    ys_o = ro.integers(0, 20, probe_n); xs_o = ro.integers(0, 20, probe_n)
    means = probe.reshape(probe_n, -1, 3).mean(1)
    for i in range(probe_n):
        pj[i, ys_o[i]:ys_o[i] + 12, xs_o[i]:xs_o[i] + 12, :] = means[i]
    dfit = evo_gpu.DiversityFitGPU(_im2col5c(probe), RESP, POOLS5,
                                   patches_jit=_im2col5c(pj))
    # utility gate state (balanced labelled probe)
    ru = np.random.default_rng(seed + 3)
    uidx = np.concatenate([ru.choice(np.where(ytr == c)[0], util_probe_n // 10,
                                     replace=False) for c in range(10)])
    ufit = evo_gpu.UtilityFitGPU(_im2col5c(Xtr[uidx]), ytr[uidx], RESP, POOLS5)
    rng0 = np.random.default_rng(seed)
    picked, dry = [], 0
    for rd in range(max_rounds):
        rs = np.random.default_rng(seed + 733 * rd)
        K = (rs.standard_normal((pop, 75)) * 0.25).astype(np.float32)
        bb = np.zeros(pop, np.float32)
        act = rs.integers(0, 8, pop).astype(np.float32)
        sigma = np.full(pop, 0.08, np.float32)
        def combined(Kc, bc, aic):
            fc, Bc = dfit(Kc, bc, aic)             # occlusion-contrastive
            ug, Pu = ufit(Kc, bc, aic, dense=True) # marginal utility (dense)
            return fc / (1.0 + np.exp(-50.0 * ug)), Bc, ug, Pu

        for gen in range(1, gens + 1):
            ai = np.round(act).astype(np.int64) % 8
            fit, B, ug, Pu = combined(K, bb, ai)
            pd = {"K": K, "b": bb, "act": act, "sigma": sigma}
            mp.ga_step(pd, fit, rng0)
            K, bb, act, sigma = pd["K"], pd["b"], pd["act"], pd["sigma"]
            act = np.round(act) % 8
        ai = np.round(act).astype(np.int64) % 8
        fit, B, ug, Pu = combined(K, bb, ai)
        # admission: best combined candidate whose utility is strictly positive
        admitted = 0
        for j in np.argsort(fit)[::-1][:8]:
            if ug[j] > 0:
                ufit.admit(Pu[j:j + 1])
                dfit.admit(B[j:j + 1])             # stays out of future novelty
                picked.append((K[j].copy(), float(bb[j]), int(ai[j])))
                admitted += 1
                break                              # one admission per round
        dry = 0 if admitted else dry + 1
        if len(picked) % 32 == 0 and admitted:
            log(f"  [hybrid] round {rd + 1}: encoder {len(picked)}, "
                f"probe acc {ufit.base_acc:.4f}")
        if dry >= patience:
            log(f"  [hybrid] {patience} dry rounds — gate closed")
            break
        if len(picked) >= n_bank:
            break
    bank = {"K": np.stack([p[0] for p in picked]),
            "b": np.array([p[1] for p in picked], np.float32),
            "act": np.array([p[2] for p in picked], np.int64),
            "whitened": False, "pools": POOLS5, "mode": "hybrid"}
    out = out or DETBANK10
    with open(out, "wb") as f:
        pickle.dump(bank, f)
    log(f"hybrid encoder: {len(picked)} genomes ({rd + 1} rounds) -> {out}")
    return bank


# --------------------------------------------------------------------------
# v7 environment — LAYER-2 diversity genomes over the frozen layer-1 bank.
# Composition as the source of new information: a layer-2 genome selects 8
# layer-1 channels (evolved genes), convolves them with an evolved 3x3x8
# kernel, applies its evolved activation. Same novelty x entropy x stability
# fitness, zero labels at every level.
# --------------------------------------------------------------------------
DETBANK7 = os.path.join(ROOT, "demo", "cifar_detbank_v7.pkl")
POOLS_L2 = ((3, 3), (4, 2), (2, 4))                # on the 12x12 L2 map -> 25


def _l1_chunk_fn(bank1):
    return lambda Xc: evo_gpu.l1_maps(_im2col5c, Xc, bank1, RESP)


def evolve_detbank_l2(n_bank=512, max_rounds=96, pop=48, gens=40, probe_n=1200,
                      admit_per_round=8, dup_cap=0.85, seed=7, log=print,
                      out=None):
    with open(DETBANK6, "rb") as f:
        bank1 = pickle.load(f)
    nb = len(bank1["K"])
    Xtr = load_cifar()[0]
    rp = np.random.default_rng(seed + 1)
    probe = Xtr[rp.choice(len(Xtr), probe_n, replace=False)]
    pj = np.zeros_like(probe)
    pj[:, :-1, :-1, :] = probe[:, 1:, 1:, :]
    log(f"materialising layer-1 maps ({nb} channels, {probe_n} probe images)...")
    L1 = evo_gpu.l1_maps(_im2col5c, probe, bank1, RESP)
    L1j = evo_gpu.l1_maps(_im2col5c, pj, bank1, RESP)
    lfit = evo_gpu.Layer2FitGPU(L1, POOLS_L2, L1_jit=L1j)
    rng0 = np.random.default_rng(seed)
    picked = []
    for rd in range(max_rounds):
        rs = np.random.default_rng(seed + 401 * rd)
        ch = rs.integers(0, nb, (pop, 8))
        K = (rs.standard_normal((pop, 72)) * 0.2).astype(np.float32)
        bb = np.zeros(pop, np.float32)
        act = rs.integers(0, 8, pop).astype(np.float32)
        sigma = np.full(pop, 0.08, np.float32)
        fit, B = None, None
        for gen in range(1, gens + 1):
            ai = np.round(act).astype(np.int64) % 8
            fit, B = lfit(ch, K, bb, ai)
            pd = {"K": K, "b": bb, "act": act, "sigma": sigma}
            mp.ga_step(pd, fit, rng0)
            K, bb, act, sigma = pd["K"], pd["b"], pd["act"], pd["sigma"]
            act = np.round(act) % 8
            # channel genes: discrete resample mutation at a low rate
            mut = rng0.random((pop, 8)) < 0.03
            ch = np.where(mut, rng0.integers(0, nb, (pop, 8)), ch)
        ai = np.round(act).astype(np.int64) % 8
        fit, B = lfit(ch, K, bb, ai)
        order = np.argsort(fit)[::-1]
        dup = lfit.novelty_vs_archive(B)
        admitted = 0
        for j in order:
            if admitted >= admit_per_round or len(picked) >= n_bank:
                break
            if dup[j] > dup_cap:
                continue
            picked.append((ch[j].copy(), K[j].copy(), float(bb[j]), int(ai[j])))
            lfit.admit(B[j:j + 1])
            admitted += 1
        if (rd + 1) % 8 == 0 or len(picked) >= n_bank:
            log(f"  [l2bank] round {rd + 1}: archive {len(picked)}, "
                f"best fit {fit[order[0]]:.3f}")
        if len(picked) >= n_bank:
            break
    bank2 = {"ch": np.stack([p[0] for p in picked]),
             "K": np.stack([p[1] for p in picked]),
             "b": np.array([p[2] for p in picked], np.float32),
             "act": np.array([p[3] for p in picked], np.int64),
             "H": 14, "pools": POOLS_L2, "layer1": DETBANK6}
    out = out or DETBANK7
    with open(out, "wb") as f:
        pickle.dump(bank2, f)
    log(f"layer-2 bank: {len(picked)} genomes ({rd + 1} rounds, label-free) -> {out}")
    return bank2


# --------------------------------------------------------------------------
# Feature assembly (mirrors mp.build_features versions 2 and 4)
# --------------------------------------------------------------------------
_STATCACHE = {}


def build_features(version=2, pca_scale="unit"):
    """`pca_scale`: 'unit' standardises every PCA dim to variance 1 (the MNIST
    recipe); 'eig' keeps the natural eigenvalue-proportional amplitudes.
    Unit scaling amplifies deep-tail noise components to the same magnitude
    as signal — measured on CIFAR v5, that makes the binary-genome landscape
    unclimbable (chance at 2048 dims, 85% at top-256). Eigen scaling keeps
    the tail small so mutation-based search explores where variance lives,
    at no ceiling cost."""
    key = ("F", version, pca_scale)
    if key in _STATCACHE:
        return _STATCACHE[key]
    Xtr, ytr, Xva, yva, Xte, yte = load_cifar()
    pca = build_pca(Xtr)
    Ftr = stat_features(Xtr, pca)
    mu = Ftr.mean(axis=0); sd = Ftr.std(axis=0) + 1e-6
    Ftr = (Ftr - mu) / sd
    Fva = (stat_features(Xva, pca) - mu) / sd
    Fte = (stat_features(Xte, pca) - mu) / sd
    if version in (4, 5, 6, 7, 9, 10):
        tag = "" if pca_scale == "unit" else f"_{pca_scale}"
        cache_f = os.path.join(os.path.dirname(DATA_DIR), f"feats_v{version}{tag}.npz")
        if os.path.exists(cache_f):
            z = np.load(cache_f)
            Ftr, Fva, Fte = z["Ftr"], z["Fva"], z["Fte"]
        else:
            bank_p = {4: DETBANK, 5: DETBANK5, 6: DETBANK6, 7: DETBANK6,
                      9: DETBANK9, 10: DETBANK10}[version]
            with open(bank_p, "rb") as fh:
                bank = pickle.load(fh)
            def _bank_all(X):
                B = bank_features(X, bank)
                if version == 7:
                    with open(DETBANK7, "rb") as fh2:
                        bank2 = pickle.load(fh2)
                    B2 = evo_gpu.l2_features_gpu(_l1_chunk_fn(bank), X, bank2,
                                                 tuple(tuple(p) for p in bank2["pools"]))
                    B = np.concatenate([B, B2], axis=1)
                return B
            Btr = _bank_all(Xtr)
            bmu = Btr.mean(0); bsd = Btr.std(0) + 1e-6
            if version in (6, 7, 9, 10):
                # PURE evolved environment: bank outputs only — no hand-built
                # statistics anywhere (the label-free configuration)
                Ftr = (Btr - bmu) / bsd
                Fva = (_bank_all(Xva) - bmu) / bsd
                Fte = (_bank_all(Xte) - bmu) / bsd
            else:
                Ftr = np.concatenate([Ftr, (Btr - bmu) / bsd], axis=1)
                Fva = np.concatenate([Fva, (bank_features(Xva, bank) - bmu) / bsd], axis=1)
                Fte = np.concatenate([Fte, (bank_features(Xte, bank) - bmu) / bsd], axis=1)
            Dp = 1024 if version == 4 else 2048
            if Ftr.shape[1] > 12000 and evo_gpu.HAS_GPU:
                # high-dim bank: exact eigh is infeasible (dims^2 float64);
                # GPU randomized SVD gives the top-Dp principal directions
                import torch
                with torch.no_grad():
                    Fg = evo_gpu.to_dev(Ftr)
                    Fg = Fg - Fg.mean(dim=0, keepdim=True)
                    _, _, V = torch.svd_lowrank(Fg, q=Dp + 128, niter=4)
                    comps = V[:, :Dp].cpu().numpy().astype(np.float32)
                    del Fg
                    torch.cuda.empty_cache()
            else:
                C = (Ftr.T @ Ftr).astype(np.float64) / len(Ftr)
                m = Ftr.mean(0).astype(np.float64)
                C -= np.outer(m, m)
                w, v = np.linalg.eigh(C)
                comps = v[:, ::-1][:, :Dp].astype(np.float32)
            Ftr = Ftr @ comps; Fva = Fva @ comps; Fte = Fte @ comps
            pmu = Ftr.mean(0)
            if pca_scale == "unit":
                psd = Ftr.std(0) + 1e-6
            else:                                  # 'eig': global scale only
                psd = np.full(Ftr.shape[1], Ftr.std() + 1e-6, np.float32)
            Ftr = (Ftr - pmu) / psd; Fva = (Fva - pmu) / psd; Fte = (Fte - pmu) / psd
            np.savez_compressed(cache_f, Ftr=Ftr, Fva=Fva, Fte=Fte)
    _STATCACHE[key] = {"Ftr": Ftr, "ytr": ytr, "Fva": Fva, "yva": yva,
                       "Fte": Fte, "yte": yte, "nf": Ftr.shape[1],
                       "version": version}
    return _STATCACHE[key]


def centroid_baseline(version=2, pca_scale="unit"):
    D = build_features(version, pca_scale=pca_scale)
    cents = np.stack([D["Ftr"][D["ytr"] == c].mean(0) for c in range(10)])
    d2 = ((D["Fte"][:, None, :] - cents[None]) ** 2).sum(-1)
    return float((d2.argmin(1) == D["yte"]).mean())


# --------------------------------------------------------------------------
# Battery — the classifier genomes are LITERALLY mnist_pipe's (D passed in)
# --------------------------------------------------------------------------
def evaluate(champs, split="test", use_mixer=True, use_pairs=True,
             pair_margin=3.0, use_joint=True):
    D = build_features(champs.get("feat_version", 2),
                       pca_scale=champs.get("pca_scale", "unit"))
    F, y = (D["Fte"], D["yte"]) if split == "test" else (D["Fva"], D["yva"])
    pred, _ = mp.predict(champs, F, use_mixer, use_pairs, pair_margin, use_joint)
    acc = float((pred == y).mean())
    conf = np.zeros((10, 10), np.int64)
    np.add.at(conf, (y, pred), 1)
    return {"acc": round(acc, 4), "confusion": conf.tolist(), "n": len(y)}


def tune_pair_margin(champs, log=print):
    best_m, best_acc = 0.0, evaluate(champs, "val", True, False)["acc"]
    for m in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        acc = evaluate(champs, "val", True, True, m)["acc"]
        log(f"  margin {m}: val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_m = acc, m
    return best_m, best_acc


def _finish(champs, version, t0, log, pca_scale="unit"):
    champs["feat_version"] = version
    champs["pca_scale"] = pca_scale
    m, vacc = tune_pair_margin(champs, log=log)
    champs["pair_margin"] = m
    log(f"chosen pair_margin={m} (val_acc={vacc:.4f})")
    res = {
        "centroid_test": centroid_baseline(version, pca_scale),
        "joint_test": evaluate(champs, "test", True, False)["acc"],
        "full_test": evaluate(champs, "test", True, True, m)["acc"],
    }
    champs["results"] = res
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(champs, f)
    log(f"saved champions -> {CACHE}")
    log(f"TEST: centroid {res['centroid_test']:.4f} | joint {res['joint_test']:.4f} "
        f"| +pairwise {res['full_test']:.4f}   ({time.time() - t0:.0f}s)")
    return res


def run_battery(version=2, det_gens=0, joint_gens=6000, pair_gens=1500, seed=7,
                log=print, warm="centroid", pca_scale="unit"):
    """The classifier battery. `warm` selects the joint head's starting point:
      "centroid" — class-mean head (a train statistic; the MNIST v4 recipe)
      "fold"     — PURE GENOME STACK: evolve 10 one-vs-rest detector genomes
                   from random, evolve a 10x10 mixer genome to calibrate them,
                   fold algebraically, continue evolving the folded genome.
                   Every parameter in the classifier chain is then produced by
                   selection; no statistic ever initialises it."""
    t0 = time.time()
    log(f"=== CIFAR-Pipe battery (environment v{version}, warm={warm}) ===")
    D = build_features(version, pca_scale=pca_scale)
    log(f"environment v{version} ({pca_scale}-scaled): {D['nf']} dims")
    log(f"centroid floor (reference only): {centroid_baseline(version):.4f}")
    champs = {}
    if warm == "fold" or det_gens > 0:
        log("--- detector genomes (10x one-vs-rest, evolved from random) ---")
        champs = mp.train_detectors(gens=max(det_gens, 3000), seed=seed, log=log, D=D, mag_scale=True)
    if warm == "fold":
        log("--- mixer genome (calibrates the folded stack, evolved) ---")
        champs.update(mp.train_mixer(champs["det"], gens=2000, seed=seed,
                                     log=log, D=D))
        warm_init = mp.fold_stack(champs)          # algebra on evolved genomes
        log("--- joint refine (continues evolution from the folded stack) ---")
    else:
        mu_c = np.stack([D["Ftr"][D["ytr"] == c].mean(0) for c in range(10)], axis=1)
        warm_init = (mu_c.astype(np.float32),
                     (-0.5 * (mu_c ** 2).sum(0)).astype(np.float32))
        log("--- joint refine (centroid warm start) ---")
    champs.update(mp.train_joint(champs, gens=joint_gens, seed=seed, log=log,
                                 D=D, minibatch=0, warm_init=warm_init))
    log("--- pairwise referees (45x one-vs-one) ---")
    champs.update(mp.train_pairwise(gens=pair_gens, seed=seed, log=log, D=D, mag_scale=True))
    return _finish(champs, version, t0, log, pca_scale=pca_scale)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--detbank", action="store_true",
                    help="evolve the 5x5x3 conv detector bank (do this before --v4)")
    ap.add_argument("--detbank5", action="store_true",
                    help="evolve the NICHED whitened bank (do this before --v5)")
    ap.add_argument("--detbank6", action="store_true",
                    help="evolve the DIVERSITY bank, label-free (before --v6)")
    ap.add_argument("--v4", action="store_true",
                    help="battery on the round-1 evolved environment")
    ap.add_argument("--v5", action="store_true",
                    help="battery on the niched whitened environment")
    ap.add_argument("--v6", action="store_true",
                    help="battery on the label-free diversity environment")
    ap.add_argument("--pure", action="store_true",
                    help="pure genome stack: evolved detectors+mixer fold as the "
                         "joint warm start (no statistics in the classifier)")
    ap.add_argument("--det-gens", type=int, default=0)
    ap.add_argument("--joint-gens", type=int, default=6000)
    ap.add_argument("--pair-gens", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eig", action="store_true",
                    help="eigenvalue-proportional PCA scaling (GA-climbable)")
    args = ap.parse_args()
    if args.detbank:
        evolve_detbank(seed=args.seed)
    elif args.detbank5:
        evolve_detbank_niched(seed=args.seed)
    elif args.detbank6:
        evolve_detbank_diverse(seed=args.seed)
    else:
        v = 6 if args.v6 else (5 if args.v5 else (4 if args.v4 else 2))
        run_battery(v, args.det_gens, args.joint_gens, args.pair_gens, args.seed,
                    warm="fold" if args.pure else "centroid",
                    pca_scale="eig" if args.eig else "unit")
