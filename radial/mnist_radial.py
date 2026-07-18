"""mnist_radial.py — MNIST on the radial stack, plus the SEED-AXIS stack
(manufactured rotation for static classification).

Two things live here.

1. THE PORT. MNIST moved off the old WordPipe recipe (stat bank + linear
   detector genomes + mixer + joint, see genreg_train/mnist_pipe.py) and onto
   the radial stack: the environment is label-free patch-PCA maps, evolution
   searches the radial_evo2 GRAMMAR (scale / terms / op / soft-window pooling
   all genes), a closed-form ridge solves the head, test is touched once.
   Reuses radial_evo2.{Env, new_genome, mutate, feature, make_scorer} and
   radial_evo._ridge_soft verbatim — nothing hand-crafted, no gradients.

2. THE SEED-AXIS STACK (the new science). In a TEMPORAL clip the data rotates,
   so the stack composes across the time axis (motion = centroid@t5 -
   centroid@t0). A static digit does not rotate, so there is no signal between
   "frames". The move: MANUFACTURE the rotation by re-origining the radial
   space per seed. Roles (genomes) are evolved ONCE; each SEED applies a small
   fixed rotation to the patch-PCA feature frame (radial_stack._rotate_features
   — the exact "relative motion between data and lens" primitive). Role r under
   seed s gives f[r,s]; laid out (role x seed) this is byte-identical to the
   temporal (genome x step) hand-off. Seed plays the role of step.

   The claim is tested by a 3-rung ladder over ONE seed tensor (the only
   variable is composition):
     rung 1  single  — seed-0 slice, ridge head.
     rung 2  union   — flatten the whole (role x seed) tensor, ridge head.
                       (the "seed union" of the guide's section 5 — nearly-free
                       accuracy; the shallow version.)
     rung 3  composed — the radial_evo2 grammar composes ACROSS the seed axis:
                       |f[r,seedA] - f[r,seedB]| is the motion analog, plus
                       explicit cross-seed mean/std/range channels. The std
                       channel is the per-image GENERALISATION signal (feature
                       disagreement across viewpoints ~ uncertainty).
   Claim holds iff rung3 > rung2 > rung1.

   Roles are evolved on a small train subsample (cheap, and a bootstrap), so
   cross-seed disagreement is also a jackknife over the fit — the architecture
   measuring its own generalisation, per the user's directive.

Numerical rails per TEMPORAL_RADIAL_STACK_GUIDE section 4: TF32 off, gram in
fp32 / factor+solve in fp64 (inside _ridge_soft / make_scorer), every genome
column sanitised, standardise with TRAIN statistics only.
"""
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import argparse
import gc
import hashlib
import json
import os
import pickle
import time

import numpy as np

from radial_evo import _ridge_soft, _tprims, _STOP
from radial_evo2 import (new_genome, mutate, feature, make_scorer,
                         SCALES, C_PER_SCALE, _PRIMS, _OPS)
from radial_stack import _rotate_features

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_HERE, "corpora", "mnist")
OUT_DIR = os.path.join(_HERE, "radial_data")
RUNS_DIR = os.path.join(_HERE, "runs", "mnist_radial")
GRID = 4                         # spatial resolution carried on the seed axis


# ---------------------------------------------------------------------------
# data — MNIST as (N,28,28,3) [0,1] floats (patch-PCA Env expects 3 channels;
# grayscale tiled to 3 is a no-op waste but reuses the validated Env unchanged).
# Optional moment deskew (a label-free data statistic — enriches the
# environment, never the organism; the same deskew the old pipe used).
# ---------------------------------------------------------------------------
import gzip
import struct


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def _deskew(X):
    """Moment-based shear correction, batch-vectorised. Pure image statistics."""
    N = len(X)
    ys, xs = np.mgrid[0:28, 0:28].astype(np.float32)
    m = X.sum(axis=(1, 2)) + 1e-8
    cy = (X * ys).sum(axis=(1, 2)) / m
    cx = (X * xs).sum(axis=(1, 2)) / m
    dy = ys[None] - cy[:, None, None]
    dx = xs[None] - cx[:, None, None]
    mu11 = (X * dx * dy).sum(axis=(1, 2)) / m
    mu02 = (X * dy * dy).sum(axis=(1, 2)) / m
    alpha = np.clip(mu11 / (mu02 + 1e-8), -1.0, 1.0)
    out = np.empty_like(X)
    idx = np.arange(N)
    for lo in range(0, N, 4096):
        sl = idx[lo:lo + 4096]
        src_x = xs[None] + alpha[sl, None, None] * (ys[None] - cy[sl, None, None])
        x0 = np.floor(src_x).astype(np.int64)
        w = (src_x - x0).astype(np.float32)
        x0c = np.clip(x0, 0, 27); x1c = np.clip(x0 + 1, 0, 27)
        rows = np.broadcast_to(ys.astype(np.int64)[None], x0c.shape)
        n3 = sl[:, None, None]
        valid = ((src_x >= -1) & (src_x <= 28)).astype(np.float32)
        out[sl] = (X[n3, rows, x0c] * (1 - w) + X[n3, rows, x1c] * w) * valid
    return out


def load_data(n_train=None, n_test=None, deskew=True, val_frac=0.15, seed=0):
    """Returns dict with Xtr/Xte (N,28,28,3) [0,1], ytr/yte, n_fit (val is the
    tail of the train block: rows [n_fit:] gate champions, test is untouched)."""
    X = _read_idx(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    y = _read_idx(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")).astype(np.int64)
    Xt = _read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    yt = _read_idx(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)
    if deskew:
        X, Xt = _deskew(X), _deskew(Xt)
    # shuffle the train block once (deterministic) so fit/val are class-mixed
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    if n_train:
        X, y = X[:n_train], y[:n_train]
    if n_test:
        Xt, yt = Xt[:n_test], yt[:n_test]
    n_fit = int(round(len(X) * (1 - val_frac)))
    X3 = np.repeat(X[..., None], 3, axis=3)
    Xt3 = np.repeat(Xt[..., None], 3, axis=3)
    return {"Xtr": X3, "ytr": y, "Xte": Xt3, "yte": yt, "n_fit": n_fit}


# ---------------------------------------------------------------------------
# EnvLite — patch-PCA environment with a FIXED canonical basis (comps, mu,
# train-sd) per scale. Unlike radial_evo2.Env it can project an ARBITRARY posed
# image batch through the SAME lens, so an image-space rotation per seed keeps
# role r meaning the same feature across seeds (channel coherence). Drop-in
# .maps(ps) for the radial_evo2 grammar; .project_posed(imgs_gpu, ps) for the
# seed tensor. Normalises test with TRAIN sd everywhere (guide rail 5).
# ---------------------------------------------------------------------------

def _pose_imgs(torch, dev, Xnhwc, deg):
    """Rotate (N,28,28,3) images by `deg` degrees about the centre (bilinear).
    Returns a GPU tensor (N,3,H,W). deg=0 is the identity (upright)."""
    import torch.nn.functional as Fn
    x = torch.tensor(Xnhwc, device=dev).permute(0, 3, 1, 2).contiguous()
    if abs(deg) < 1e-6:
        return x
    c, s = float(np.cos(np.radians(deg))), float(np.sin(np.radians(deg)))
    mat = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=dev).unsqueeze(0).repeat(len(x), 1, 1)
    grid = Fn.affine_grid(mat, list(x.shape), align_corners=False)
    return Fn.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


class EnvLite:
    def __init__(self, torch, dev, Xtr, Xte):
        self.torch, self.dev = torch, dev
        self.Xtr, self.Xte = Xtr, Xte            # (N,28,28,3) float [0,1]
        self.basis = {}                          # ps -> comps/mu/sd/stride/H/W
        self.cache = {}                          # ps -> (Mtr, Mte, H, W)
        self.last_used = {}

    def _to_gpu(self, X):
        return self.torch.tensor(X, device=self.dev).permute(0, 3, 1, 2).contiguous()

    def _project_gpu(self, imgs_gpu, comps, mu, ps, stride, bs=400):
        import torch.nn.functional as Fn
        torch = self.torch
        out = None
        for b in range(0, len(imgs_gpu), bs):
            U = Fn.unfold(imgs_gpu[b:b + bs], ps, stride=stride)
            M = torch.einsum("cd,bdl->bcl", comps, U - mu.view(1, -1, 1))
            if out is None:
                out = torch.zeros((len(imgs_gpu), M.shape[1], M.shape[2]),
                                  device=self.dev, dtype=torch.float16)
            out[b:b + len(U)] = M.half()
        return out.float()

    def basis_for(self, ps):
        import torch.nn.functional as Fn
        torch = self.torch
        if ps in self.basis:
            return self.basis[ps]
        stride = max(2, ps // 2); d = ps * ps * 3
        i2k = self._to_gpu(self.Xtr[:2000])
        P = Fn.unfold(i2k, ps, stride=stride)
        cols = P.permute(0, 2, 1).reshape(-1, d)
        g = torch.Generator(device="cpu").manual_seed(ps)
        cols = cols[torch.randperm(len(cols), generator=g)[:100000].to(self.dev)]
        mu = cols.mean(0)
        _, _, V = torch.linalg.svd(cols - mu, full_matrices=False)
        comps = V[:min(C_PER_SCALE, d)]
        Sres = self.Xtr.shape[1]; H = W = (Sres - ps) // stride + 1
        Mtr = self._project_gpu(self._to_gpu(self.Xtr), comps, mu, ps, stride)
        sd = Mtr.std((0, 2), keepdim=True) + 1e-6
        self.basis[ps] = {"comps": comps, "mu": mu, "sd": sd,
                          "stride": stride, "H": H, "W": W}
        return self.basis[ps]

    def maps(self, ps):
        """Canonical (upright) maps — drop-in for radial_evo2.Env.maps."""
        if ps not in self.cache:
            bd = self.basis_for(ps)
            Mtr = (self._project_gpu(self._to_gpu(self.Xtr), bd["comps"], bd["mu"],
                                     ps, bd["stride"]) / bd["sd"]).half()
            Mte = (self._project_gpu(self._to_gpu(self.Xte), bd["comps"], bd["mu"],
                                     ps, bd["stride"]) / bd["sd"]).half()
            self.cache[ps] = (Mtr, Mte, bd["H"], bd["W"])
        self.last_used[ps] = self.last_used.get(ps, 0) + 1
        return self.cache[ps]

    def project_posed(self, imgs_gpu, ps):
        """Project an already-posed image tensor (N,3,H,W) through the canonical
        basis -> (M (N,C,L) half, H, W)."""
        bd = self.basis_for(ps)
        M = (self._project_gpu(imgs_gpu, bd["comps"], bd["mu"], ps, bd["stride"])
             / bd["sd"]).half()
        return M, bd["H"], bd["W"]


# ---------------------------------------------------------------------------
# role grammar helpers — role = a radial_evo2 genome; we need both its pooled
# SCALAR (for the evolve fitness) and its GRID response (for the seed tensor).
# ---------------------------------------------------------------------------

def _combined_map(torch, tp, M, g, H, W):
    """Fold a genome's terms into one (N,H,W) map over patch-PCA maps M
    (already the correct scale). This is feature()'s term loop, verbatim, but
    stopping before the soft-window pool so we can hand off the spatial grid."""
    z = None
    for t in g["terms"]:
        v = M[:, t["c"] % M.shape[1], :].float().view(len(M), H, W)
        for prim, a, b in t["prog"]:
            v = tp[_PRIMS[prim]](a * v + b)
        if z is None:
            z = v
        else:
            op = _OPS[g["op"]]
            z = z * v if op == "mult" else (torch.minimum(z, v) if op == "min"
                                            else torch.abs(z - v))
    return z


def role_grid(torch, M, g, H, W, tp, grid=GRID):
    """(N, grid, grid) pooled response of role g over maps M — the spatial
    hand-off unit for the seed tensor."""
    import torch.nn.functional as Fn
    z = _combined_map(torch, tp, M, g, H, W).unsqueeze(1)     # (N,1,H,W)
    z = Fn.adaptive_avg_pool2d(z, (grid, grid)).squeeze(1)    # (N,grid,grid)
    return torch.nan_to_num(z, 0.0, 30.0, -30.0)


# ---------------------------------------------------------------------------
# Phase 1 — evolve the ROLES on the canonical (seed-0) lens. Standard single
# radial space: comma GA, freeze-and-compose, empty base, honest val gate.
# (This is also rung 1's substrate — its own ridge head is the single-seed acc.)
# ---------------------------------------------------------------------------

def evolve_roles(torch, env, tp, ytr, n_fit, seed=5, rounds=40, pop_size=64,
                 gens=12, freeze_top=8, cap=0.0005, max_roles=None, log=print):
    dev = env.dev
    rng = np.random.default_rng(seed)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0

    frozen, fcols = [], []
    empty = 0
    t0 = time.time()
    for rnd in range(rounds):
        base = torch.stack(fcols, 1) if fcols else torch.zeros((len(env.Xtr), 0), device=dev)
        scorer, s0, a0 = make_scorer(torch, base, n_fit, Yf, yv)
        pop = [new_genome(rng) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)

        def fit_pop(gs):
            cols = [feature(torch, tp, env, g) for g in gs]
            ok = [i for i, c in enumerate(cols)
                  if float(c.std()) > 1e-6 and bool(torch.isfinite(c).all())]
            softs = np.full(len(gs), -1e9); accs = np.zeros(len(gs))
            if ok:
                C = torch.stack([cols[i] for i in ok], 1)
                sf, ac = scorer(C)
                for j, i in enumerate(ok):
                    softs[i] = sf[j] - s0; accs[i] = ac[j]
            return softs, accs, cols

        fits, accs, cols = fit_pop(pop)
        for _ in range(gens):
            order = np.argsort(fits)[::-1]
            keep = list(order[:6])
            kids, ksc = [], []
            while len(kids) < pop_size - 6:
                cand = rng.choice(pop_size, 3)
                pi = cand[np.argmax(fits[cand])]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                kids.append(mutate(rng, pop[pi], sc)); ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            pop = [pop[i] for i in keep] + kids
            scales = np.concatenate([scales[keep], ksc])
            fits = np.concatenate([fits[keep], kf])
            accs = np.concatenate([accs[keep], ka])
            cols = [cols[i] for i in keep] + kc
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= cap or added >= freeze_top:
                break
            col = cols[idx]
            colz = (col - col.mean()) / (col.std() + 1e-9)
            dup = any(float(torch.abs((colz * ((fc - fc.mean()) / (fc.std() + 1e-9))).mean())) > 0.95
                      for fc in fcols[-60:])
            if not dup:
                frozen.append(pop[idx]); fcols.append(col); added += 1
        if fcols:
            base = torch.stack(fcols, 1)
            _, a1 = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv)
        else:
            a1 = 0.0
        log(f"  [roles] round {rnd:3d}  +{added} (total {len(frozen)})  "
            f"val {a1:.4f}  ({round(time.time()-t0)}s)")
        empty = empty + 1 if added == 0 else 0
        if empty >= 3:
            break
        if max_roles and len(frozen) >= max_roles:
            log(f"  [roles] hit max_roles={max_roles}"); break
    return frozen[:max_roles] if max_roles else frozen


# ---------------------------------------------------------------------------
# Phase 2/3 — build the SEED TENSOR f[r,s] = role r under seed s.
# Seed s re-origins the patch-PCA frame by a fixed rotation angle theta_s
# (radial_stack._rotate_features across the channel axis). Roles fixed, so
# role r means the same feature under every seed -> the seed axis is composable.
# Returns numpy arrays (kept on CPU; can be large) Gtr (N, R, S, grid, grid).
# ---------------------------------------------------------------------------

def _rotate_maps(torch, M, deg):
    """Rotate the CHANNEL frame of patch-PCA maps M (N,C,L) — L=H*W flattened —
    by `deg`, the per-seed re-origining. Same rotation for train and test
    (deterministic)."""
    if deg == 0.0:
        return M
    N, C, Lp = M.shape
    flat = M.permute(0, 2, 1).reshape(-1, C).float()         # (N*L, C)
    rot = _rotate_features(torch, flat, deg)
    return rot.reshape(N, Lp, C).permute(0, 2, 1).contiguous().half()


def build_seed_tensor(torch, env, tp, roles, angles, grid=GRID, mode="image", log=print):
    """Gtr (N_tr, R, S, grid, grid), Gte (...) fp16 on CPU. Seed s applies angle
    angles[s]. Two mechanisms for "seed = a manufactured viewpoint":
      mode="image"   — rotate the actual IMAGE by the angle, then project through
                       the fixed canonical basis. The data itself takes a new
                       pose — the truest analog of the temporal object rotating.
      mode="feature" — rotate the patch-PCA FEATURE frame (channel Givens). The
                       image is untouched; the lens re-origins.
    Roles are grouped by scale so each (seed, scale) projection/rotation is
    computed once for all roles at that scale."""
    R, S = len(roles), len(angles)
    ntr, nte = len(env.Xtr), len(env.Xte)
    Gtr = np.zeros((ntr, R, S, grid, grid), np.float16)
    Gte = np.zeros((nte, R, S, grid, grid), np.float16)
    by_scale = {}
    for ri, g in enumerate(roles):
        by_scale.setdefault(g["ps"], []).append(ri)
    t0 = time.time()
    for si, deg in enumerate(angles):
        Xtr_g = Xte_g = None
        if mode == "image":
            Xtr_g = _pose_imgs(torch, env.dev, env.Xtr, deg)
            Xte_g = _pose_imgs(torch, env.dev, env.Xte, deg)
        for ps, ridx in by_scale.items():
            if mode == "image":
                Rtr, H, W = env.project_posed(Xtr_g, ps)
                Rte, _, _ = env.project_posed(Xte_g, ps)
            else:
                Mtr, Mte, H, W = env.maps(ps)
                Rtr = _rotate_maps(torch, Mtr, deg)
                Rte = _rotate_maps(torch, Mte, deg)
            for ri in ridx:
                Gtr[:, ri, si] = role_grid(torch, Rtr, roles[ri], H, W, tp, grid).cpu().numpy()
                Gte[:, ri, si] = role_grid(torch, Rte, roles[ri], H, W, tp, grid).cpu().numpy()
            del Rtr, Rte
            torch.cuda.empty_cache()
        if Xtr_g is not None:
            del Xtr_g, Xte_g
            torch.cuda.empty_cache()
        log(f"  [seedtensor:{mode}] seed {si+1}/{S} (deg {deg:.1f}) done ({round(time.time()-t0)}s)")
    return Gtr, Gte


# ---------------------------------------------------------------------------
# cross-seed reduction channels: mean / std / range per role over the seed
# axis. std is the GENERALISATION signal (disagreement across viewpoints).
# ---------------------------------------------------------------------------

def cross_seed_stats(G):
    """(N,R,S,g,g) -> (N, 3R, g, g): [mean_s | std_s | (max-min)_s] per role."""
    mean = G.mean(axis=2)
    std = G.std(axis=2)
    rng = G.max(axis=2) - G.min(axis=2)
    N, R = G.shape[0], G.shape[1]
    return np.concatenate([mean, std, rng], axis=1)          # (N, 3R, g, g)


# ---------------------------------------------------------------------------
# a MapBank that presents an arbitrary (N,C,grid,grid) tensor through the
# Env.maps(ps) interface, so radial_evo2's grammar/feature run UNCHANGED over
# the seed tensor. The scale gene is inert (one bank); channel picks index the
# (role x seed) + stat channels, so a multi-term absdiff genome composes across
# the seed axis with zero new grammar code.
# ---------------------------------------------------------------------------

class SeedBank:
    def __init__(self, torch, Mtr, Mte, grid):
        # Mtr/Mte: (N, C, grid*grid) fp16 tensors on GPU
        self.torch, self.grid = torch, grid
        self._pack = (Mtr, Mte, grid, grid)

    def maps(self, ps):
        return self._pack


def _bank_from_tensor(torch, dev, G, chan_stats=True):
    """(N,R,S,g,g) numpy -> (N, C, g*g) fp16 GPU tensor. C = R*S raw seed
    channels, then optionally 3R cross-seed stat channels appended."""
    N, R, S, g, _ = G.shape
    raw = G.reshape(N, R * S, g * g)
    if chan_stats:
        st = cross_seed_stats(G).reshape(N, 3 * R, g * g)
        raw = np.concatenate([raw, st], axis=1)
    return torch.tensor(raw, device=dev, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Phase 4 — evolve the COMPOSED-across-seed space over the seed bank (rung 3).
# Same comma-GA / freeze-and-compose as the roles, but new_genome/feature run
# on the SeedBank so terms address (role x seed) channels and absdiff composes
# across seeds.
# ---------------------------------------------------------------------------

def evolve_composed(torch, bank, tp, ytr, n_fit, C, seed=5, rounds=60,
                    pop_size=64, gens=12, freeze_top=8, cap=0.0005,
                    base_extra=None, log=print):
    """bank: a SeedBank. C: channel count (so genomes pick valid channels).
    base_extra: (n, F0) tensor put in the fitness BASE so genomes must earn
    RESIDUAL over it (force-residual arm — the deterministic stats). None = lean
    arm (empty base, genomes carry the model)."""
    dev = bank.torch.device if hasattr(bank.torch, "device") else "cuda"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    ntr = n_fit + (len(ytr) - n_fit)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0

    def seed_genome(r):
        g = new_genome(r)
        for t in g["terms"]:                 # channels index the FULL seed bank
            t["c"] = int(r.integers(C))
        return g

    def seed_mutate(r, g, sc):
        # radial_evo2.mutate resets channels into [0, C_PER_SCALE=40) — which on
        # this bank cannot reach the cross-seed stat channels (appended after the
        # R*S raw block). So take mutate for its prog/op/window/depth edits, then
        # OVERRIDE every channel gene with full-bank drift from the parent.
        parent_ch = [t["c"] for t in g["terms"]]
        c = mutate(r, g, sc)
        for i, t in enumerate(c["terms"]):
            pc = parent_ch[i] if i < len(parent_ch) else int(r.integers(C))
            if r.random() < 0.15:            # global jump anywhere in the bank
                t["c"] = int(r.integers(C))
            else:                            # local drift (locality law, section 3.8)
                t["c"] = int(np.clip(pc + r.integers(-24, 25), 0, C - 1))
        return c

    frozen, fcols = [], []
    empty = 0
    t0 = time.time()
    for rnd in range(rounds):
        parts = ([base_extra] if base_extra is not None else []) + \
                ([torch.stack(fcols, 1)] if fcols else [])
        base = torch.cat(parts, 1) if parts else torch.zeros((len(bank._pack[0]), 0), device=dev)
        scorer, s0, a0 = make_scorer(torch, base, n_fit, Yf, yv)
        pop = [seed_genome(rng) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)

        def fit_pop(gs):
            cols = [feature(torch, tp, bank, g) for g in gs]
            ok = [i for i, c in enumerate(cols)
                  if float(c.std()) > 1e-6 and bool(torch.isfinite(c).all())]
            softs = np.full(len(gs), -1e9); accs = np.zeros(len(gs))
            if ok:
                Cc = torch.stack([cols[i] for i in ok], 1)
                sf, ac = scorer(Cc)
                for j, i in enumerate(ok):
                    softs[i] = sf[j] - s0; accs[i] = ac[j]
            return softs, accs, cols

        fits, accs, cols = fit_pop(pop)
        for _ in range(gens):
            order = np.argsort(fits)[::-1]
            keep = list(order[:6])
            kids, ksc = [], []
            while len(kids) < pop_size - 6:
                cand = rng.choice(pop_size, 3)
                pi = cand[np.argmax(fits[cand])]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                kids.append(seed_mutate(rng, pop[pi], sc)); ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            pop = [pop[i] for i in keep] + kids
            scales = np.concatenate([scales[keep], ksc])
            fits = np.concatenate([fits[keep], kf])
            accs = np.concatenate([accs[keep], ka])
            cols = [cols[i] for i in keep] + kc
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= cap or added >= freeze_top:
                break
            col = cols[idx]
            colz = (col - col.mean()) / (col.std() + 1e-9)
            dup = any(float(torch.abs((colz * ((fc - fc.mean()) / (fc.std() + 1e-9))).mean())) > 0.95
                      for fc in fcols[-60:])
            if not dup:
                frozen.append(pop[idx]); fcols.append(col); added += 1
        vparts = ([base_extra] if base_extra is not None else []) + \
                 ([torch.stack(fcols, 1)] if fcols else [])
        if vparts:
            vb = torch.cat(vparts, 1)
            _, a1 = _ridge_soft(torch, vb[:n_fit], vb[n_fit:], Yf, yv)
        else:
            a1 = 0.0
        log(f"  [composed] round {rnd:3d}  +{added} (total {len(frozen)})  "
            f"val {a1:.4f}  ({round(time.time()-t0)}s)")
        empty = empty + 1 if added == 0 else 0
        if empty >= 3:
            break
    return frozen, fcols


# ---------------------------------------------------------------------------
# ridge head over a train/test channel bank, lambda picked on val, test once.
# ---------------------------------------------------------------------------

def _flatten(G, sl=None):
    """(N,R,S,g,g) -> (N, R*S*g*g) fp32."""
    if sl is not None:
        G = G[:, :, sl]
    return G.reshape(len(G), -1).astype(np.float32)


def _stdclip(tr, te):
    """Guide rails 4+5: kill inf/NaN, standardize by TRAIN stats, clamp to +-8 sd
    on BOTH sides. A genome tame on train can explode out-of-distribution, and one
    such column makes the ridge gram singular (torch 2.8 raises where 2.6 limped)."""
    tr = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)
    te = np.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0)
    mu = tr.mean(0); sd = tr.std(0) + 1e-6
    tr = np.clip((tr - mu) / sd, -8.0, 8.0).astype(np.float32)
    te = np.clip((te - mu) / sd, -8.0, 8.0).astype(np.float32)
    return tr, te


def _pca_base(torch, dev, block_tr, n_fit, K=256):
    """(N, F) numpy -> (N, K) GPU tensor: top-K PCA (fit on the n_fit rows) of a
    feature block. Used to compress the deterministic stat/single columns into a
    small fitness BASE so the force-residual border ridge stays cheap while still
    representing the block's full linear-predictive content."""
    X = torch.tensor(block_tr, device=dev)
    Xf = X[:n_fit]
    mu = Xf.mean(0)
    Cov = ((Xf - mu).T @ (Xf - mu)).double() / n_fit
    _, V = torch.linalg.eigh(Cov)
    comps = V[:, -min(K, V.shape[1]):].float()
    return (X - mu) @ comps


def ridge_eval(torch, Ftr, Fva, Fte, ytr, yva, yte, dev,
               lams=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)):
    """Fit on train, pick lambda on val, evaluate test ONCE at the best lambda."""
    Xf = torch.tensor(Ftr, device=dev); Xv = torch.tensor(Fva, device=dev)
    Xe = torch.tensor(Fte, device=dev)
    Yf = -torch.ones((len(ytr), 10), device=dev)
    Yf[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0
    yv = torch.tensor(yva, device=dev); ye = torch.tensor(yte, device=dev)
    best_lam, best_va = None, -1.0
    for lam in lams:
        try:                                    # wide heads can be singular at
            _, va = _ridge_soft(torch, Xf, Xv, Yf, yv, lam=lam)   # small lambda
        except Exception:                       # (esp. torch 2.8 / Blackwell) —
            continue                            # skip and let a heavier lam win
        if va > best_va:
            best_va, best_lam = va, lam
    if best_lam is None:                        # every lam singular -> force heavy
        best_lam = max(max(lams), 1000.0)
    try:
        _, te = _ridge_soft(torch, Xf, Xe, Yf, ye, lam=best_lam)
    except Exception:
        _, te = _ridge_soft(torch, Xf, Xe, Yf, ye, lam=max(max(lams), 1000.0))
    del Xf, Xv, Xe, Yf                          # release GPU tensors promptly
    torch.cuda.empty_cache()
    return round(float(best_va), 4), round(float(te), 4), best_lam


# ---------------------------------------------------------------------------
# run recording (AGENTS.md rule 4: 5 files in runs/<env>/<run-id>/)
# ---------------------------------------------------------------------------

def _record_run(cfg, hist, stats, log_lines, tags, env_name="mnist_radial"):
    try:
        h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
        rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{env_name}-{h}"
        d = os.path.join(_HERE, "runs", env_name, rid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"id": rid, "environment": env_name,
                       "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "config": cfg, "status": "done"}, f, indent=1)
        with open(os.path.join(d, "history.jsonl"), "w") as f:
            for row in hist:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump({"status": "done", **stats}, f, indent=1)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"label": f"{env_name} ({cfg.get('n_seeds')} seeds)",
                       "tags": tags, "favorite": False, "group": env_name}, f, indent=1)
        with open(os.path.join(d, "report.json"), "w") as f:
            json.dump({"config": cfg, "stats": stats,
                       "log_tail": log_lines[-40:]}, f, indent=1)
        return rid
    except Exception as exc:                          # pragma: no cover
        print(f"[record] failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# the full battery: roles -> seed tensor -> 3-rung ladder -> record -> notice
# ---------------------------------------------------------------------------

def run(n_seeds=8, step_deg=6.0, n_train=None, n_test=None, deskew=True,
        role_rounds=40, role_pop=64, role_gens=12, comp_rounds=60,
        comp_pop=64, comp_gens=12, grid=GRID, seed=5, max_roles=None,
        heavy_union=True, seed_mode="image", max_deg=15.0, ab=False,
        data=None, dataset="mnist", smoke=False, record=True, log=print):
    """`data`: pre-built {Xtr(N,H,W,3)[0,1], ytr, Xte, yte, n_fit} — pass this to
    run ANY dataset (CIFAR etc.); None = load MNIST via load_data. `dataset`
    labels the exports/records."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False     # rail 1
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    lines = []

    def L(m):
        lines.append(str(m)); log(m)

    if smoke:
        if data is None:
            n_train = n_train or 3000; n_test = n_test or 1000
        n_seeds = min(n_seeds, 3); role_rounds = min(role_rounds, 4)
        comp_rounds = min(comp_rounds, 4)

    L(f"=== {dataset} radial seed-stack === mode={seed_mode} seeds={n_seeds} "
      f"step_deg={step_deg} max_deg={max_deg} grid={grid} smoke={smoke}")
    D = data if data is not None else load_data(n_train, n_test, deskew=deskew)
    Xtr, ytr, Xte, yte, n_fit = D["Xtr"], D["ytr"], D["Xte"], D["yte"], D["n_fit"]
    yva = ytr[n_fit:]
    L(f"data: fit={n_fit} val={len(ytr)-n_fit} test={len(yte)}  "
      f"({Xtr.shape[1]}x{Xtr.shape[2]}x{Xtr.shape[3]})")

    env = EnvLite(torch, dev, Xtr, Xte)

    # anchor (honesty instrument): raw patch-PCA ridge, reported never used
    L("--- anchor: raw patch-PCA ridge (baseline, not a component) ---")
    Mtr8, Mte8, H8, W8 = env.maps(8)
    A_tr = Mtr8.reshape(len(Xtr), -1).float().cpu().numpy()
    A_te = Mte8.reshape(len(Xte), -1).float().cpu().numpy()
    a_va, a_te, a_lam = ridge_eval(torch, A_tr[:n_fit], A_tr[n_fit:], A_te,
                                   ytr[:n_fit], yva, yte, dev)
    L(f"anchor (ps8 patch-PCA ridge): val {a_va} test {a_te} (lam {a_lam})")

    L("--- phase 1: evolve roles (canonical lens, seed 0) ---")
    roles = evolve_roles(torch, env, tp, ytr, n_fit, seed=seed, rounds=role_rounds,
                         pop_size=role_pop, gens=role_gens, max_roles=max_roles, log=L)
    L(f"roles frozen: {len(roles)}")
    if not roles:
        L("no roles frozen — aborting"); return {"err": "no roles"}

    if seed_mode == "image":
        # seed 0 = upright (matches the roles' training pose = the single-seed
        # baseline); the rest spread symmetrically across +-max_deg.
        angles = [0.0] + [round(float(a), 2)
                          for a in np.linspace(-max_deg, max_deg, n_seeds - 1)]
    else:
        angles = [i * step_deg for i in range(n_seeds)]
    L(f"--- phase 2/3: seed tensor ({seed_mode}) over angles {angles} ---")
    Gtr, Gte = build_seed_tensor(torch, env, tp, roles, angles, grid=grid,
                                 mode=seed_mode, log=L)
    R, S = len(roles), n_seeds
    L(f"seed tensor: train {Gtr.shape} test {Gte.shape}")
    env.cache.clear(); env.last_used.clear()          # patch-PCA maps no longer needed
    torch.cuda.empty_cache()

    # rung 1 — single seed (seed-0 slice)
    r1_tr, r1_te = _stdclip(_flatten(Gtr[:, :, 0:1]), _flatten(Gte[:, :, 0:1]))
    r1_va, r1_te_acc, r1_lam = ridge_eval(torch, r1_tr[:n_fit], r1_tr[n_fit:],
                                          r1_te, ytr[:n_fit], yva, yte, dev)
    L(f"RUNG 1 single-seed:   val {r1_va}  TEST {r1_te_acc}  (R={R} roles, lam {r1_lam})")

    st_tr, st_te = _stdclip(cross_seed_stats(Gtr).reshape(len(Xtr), -1),
                            cross_seed_stats(Gte).reshape(len(Xte), -1))

    # ===================================================================
    # A/B EXPERIMENT: can evolution EARN on the composed seed axis, or is the
    # cross-seed signal fully tabulatable? Shares the roles + seed tensor.
    #   stats-only  = the deterministic baseline (mean/std/range head).
    #   arm A (force residual) = genomes evolve with the stats in the fitness
    #      BASE, so only genuinely non-linear cross-pose structure freezes; head
    #      = stats + single + genomes_A. Beats stats-only ONLY if evolution earns.
    #   arm B (lean) = genomes evolve empty-base; head reads GENOMES ONLY. Tests
    #      how far evolution carries the model alone (accuracy per parameter).
    # ===================================================================
    if ab:
        so_va, so_te, _ = ridge_eval(torch, st_tr[:n_fit], st_tr[n_fit:], st_te,
                                     ytr[:n_fit], yva, yte, dev)
        L(f"AB baseline: stats-only (deterministic) TEST {so_te}  single {r1_te_acc}")
        bank_tr = _bank_from_tensor(torch, dev, Gtr, chan_stats=True)
        bank_te = _bank_from_tensor(torch, dev, Gte, chan_stats=True)
        C = bank_tr.shape[1]
        bank = SeedBank(torch, bank_tr, bank_te, grid)

        # base_extra = PCA-256 of [single | stats] (keeps the border-ridge base
        # small so the per-round Cholesky stays cheap while still representing
        # everything a LINEAR head gets from the deterministic features).
        base_block = np.concatenate([r1_tr, st_tr], axis=1)
        base_extra = _pca_base(torch, dev, base_block, n_fit, K=256)

        L("--- ARM A: force residual (stats in the fitness base) ---")
        frozenA, fcolsA = evolve_composed(torch, bank, tp, ytr, n_fit, C, seed=seed,
                                          rounds=comp_rounds, pop_size=comp_pop,
                                          gens=comp_gens, base_extra=base_extra, log=L)
        FAtr = torch.stack(fcolsA, 1).float().cpu().numpy() if fcolsA else np.zeros((len(Xtr), 0), np.float32)
        FAte = (torch.stack([feature(torch, tp, SeedBank(torch, bank_te, bank_te, grid), g)
                             for g in frozenA], 1).float().cpu().numpy()
                if frozenA else np.zeros((len(Xte), 0), np.float32))
        if FAtr.shape[1] > 0:
            FAtr, FAte = _stdclip(FAtr, FAte)
        HA_tr = np.concatenate([r1_tr, st_tr, FAtr], axis=1)
        HA_te = np.concatenate([r1_te, st_te, FAte], axis=1)
        a_va, a_te, a_lam = ridge_eval(torch, HA_tr[:n_fit], HA_tr[n_fit:], HA_te,
                                       ytr[:n_fit], yva, yte, dev)
        residual_A = round(a_te - so_te, 4)
        L(f"ARM A force-residual: TEST {a_te}  ({len(frozenA)} genomes froze over "
          f"the stats base) | residual over stats-only = {residual_A}")

        L("--- ARM B: lean (head reads genomes ONLY) ---")
        frozenB, fcolsB = evolve_composed(torch, bank, tp, ytr, n_fit, C, seed=seed + 1,
                                          rounds=comp_rounds, pop_size=comp_pop,
                                          gens=comp_gens, base_extra=None, log=L)
        FBtr = torch.stack(fcolsB, 1).float().cpu().numpy() if fcolsB else np.zeros((len(Xtr), 0), np.float32)
        FBte = (torch.stack([feature(torch, tp, SeedBank(torch, bank_te, bank_te, grid), g)
                             for g in frozenB], 1).float().cpu().numpy()
                if frozenB else np.zeros((len(Xte), 0), np.float32))
        if FBtr.shape[1] > 0:
            FBtr, FBte = _stdclip(FBtr, FBte)
        b_va, b_te, b_lam = ridge_eval(torch, FBtr[:n_fit], FBtr[n_fit:], FBte,
                                       ytr[:n_fit], yva, yte, dev)
        del bank_tr, bank_te, bank
        torch.cuda.empty_cache()
        lean_p = (FBtr.shape[1] + 1) * 10
        L(f"ARM B lean: TEST {b_te}  ({len(frozenB)} genomes, head {lean_p} params, "
          f"{round(b_te - r1_te_acc, 4)} vs single seed)")

        dt = round(time.time() - t0)
        stats = {"anchor_test": a_te if False else so_te, "stats_only_test": so_te,
                 "single_test": r1_te_acc, "armA_force_residual_test": a_te,
                 "armA_residual_over_stats": residual_A, "armA_n_frozen": len(frozenA),
                 "armB_lean_test": b_te, "armB_n_frozen": len(frozenB),
                 "armB_head_params": lean_p, "n_roles": R, "n_seeds": S,
                 "seed_mode": seed_mode, "grid": grid, "seconds": dt,
                 "evolution_earns_residual": bool(residual_A > 0.0005)}
        cfg = {"experiment": "ab", "seed_mode": seed_mode, "max_deg": max_deg,
               "n_seeds": n_seeds, "grid": grid, "max_roles": max_roles,
               "comp_rounds": comp_rounds, "angles": angles, "smoke": smoke}
        hist = [{"arm": "stats-only", "test": so_te},
                {"arm": "A-force-residual", "test": a_te, "residual": residual_A},
                {"arm": "B-lean", "test": b_te, "params": lean_p}]
        L(f"=== AB: stats-only {so_te} | force-residual {a_te} (earns {residual_A}) "
          f"| lean {b_te} @ {lean_p}p  ({dt}s) ===")
        out = {"stats": stats, "config": cfg, "hist": hist, "roles": len(roles)}
        with open(os.path.join(OUT_DIR, f"{dataset}_radial_ab.json"), "w") as f:
            json.dump(out, f, indent=1)
        if record and not smoke:
            rid = _record_run(cfg, hist, stats, lines, [dataset, "seed-stack", "ab"],
                              env_name=f"{dataset}_radial")
            out["run_id"] = rid
            L(f"recorded run {rid}")
        return out

    # rung 2 — naive union (flatten whole seed tensor). Measured to REGRESS vs a
    # single seed AND it is the head's memory hog (R*S*grid^2 fp32 cols); skip it
    # at scale unless explicitly asked (heavy_union).
    if heavy_union:
        r2_tr = _flatten(Gtr); r2_te = _flatten(Gte)
        r2_va, r2_te_acc, r2_lam = ridge_eval(torch, r2_tr[:n_fit], r2_tr[n_fit:],
                                              r2_te, ytr[:n_fit], yva, yte, dev)
        L(f"RUNG 2 seed-union:    val {r2_va}  TEST {r2_te_acc}  "
          f"({r2_tr.shape[1]} cols = {R}x{S}x{grid*grid}, lam {r2_lam})")
    else:
        r2_tr = r2_te = None
        r2_va = r2_te_acc = r2_lam = None
        L("RUNG 2 seed-union:    SKIPPED (heavy_union off — union regresses & is "
          "the memory hog; composed head does not need it)")

    # rung 3 — composed across the seed axis
    L("--- rung 3: evolve composed-across-seed space ---")
    bank_tr = _bank_from_tensor(torch, dev, Gtr, chan_stats=True)
    bank_te = _bank_from_tensor(torch, dev, Gte, chan_stats=True)
    C = bank_tr.shape[1]
    bank = SeedBank(torch, bank_tr, bank_te, grid)
    frozen, fcols = evolve_composed(torch, bank, tp, ytr, n_fit, C, seed=seed,
                                    rounds=comp_rounds, pop_size=comp_pop,
                                    gens=comp_gens, log=L)
    # composed head reads: cross-seed stat channels (union rung-2 already has raw)
    # + the frozen composed genomes' columns. Build test columns and ridge once.
    Ftr = torch.stack(fcols, 1).float().cpu().numpy() if fcols else np.zeros((len(Xtr), 0), np.float32)
    Fte = (torch.stack([feature(torch, tp, SeedBank(torch, bank_te, bank_te, grid), g)
                        for g in frozen], 1).float().cpu().numpy()
           if frozen else np.zeros((len(Xte), 0), np.float32))
    if Ftr.shape[1] > 0:                        # sanitize genome cols (rail 4+5):
        Ftr, Fte = _stdclip(Ftr, Fte)           # a test-side explosion was making
                                                # the head singular on Blackwell
    # (st_tr/st_te already computed above, before the AB branch)
    # free the GPU seed-bank AND the big Gtr numpy tensor (both fully consumed
    # now: st_tr done, bank built) before the large ridge heads — a late RAM
    # spike here (bank + Gtr + the fp32 heads) was killing full CIFAR runs.
    del bank_tr, bank_te, bank, Gtr
    torch.cuda.empty_cache(); gc.collect()

    # rung 3a — COMPOSED-ONLY: cross-seed stats + across-seed genome columns,
    # NO raw union columns. Tests expressivity-per-feature vs the union.
    Ca_tr = np.concatenate([st_tr, Ftr], axis=1)
    Ca_te = np.concatenate([st_te, Fte], axis=1)
    r3a_va, r3a_te, r3a_lam = ridge_eval(torch, Ca_tr[:n_fit], Ca_tr[n_fit:],
                                         Ca_te, ytr[:n_fit], yva, yte, dev)
    L(f"RUNG 3a composed-only: val {r3a_va}  TEST {r3a_te}  "
      f"({Ca_tr.shape[1]} cols, {len(frozen)} across-seed genomes, lam {r3a_lam})")
    del Ca_tr, Ca_te                       # free BEFORE building the bigger Cc head
    torch.cuda.empty_cache(); gc.collect()

    # rung 3c — SINGLE + COMPOSED (the lean production head): clean seed-0 role
    # columns + cross-seed stats + across-seed genomes. Skips the noisy 7 rotated
    # union copies but keeps the clean base and the composition. Cheap, and the
    # 99% candidate.
    Cc_tr = np.concatenate([r1_tr, st_tr, Ftr], axis=1)
    Cc_te = np.concatenate([r1_te, st_te, Fte], axis=1)
    r3c_va, r3c_te, r3c_lam = ridge_eval(torch, Cc_tr[:n_fit], Cc_tr[n_fit:],
                                         Cc_te, ytr[:n_fit], yva, yte, dev)
    L(f"RUNG 3c single+composed: val {r3c_va}  TEST {r3c_te}  "
      f"({Cc_tr.shape[1]} cols, lam {r3c_lam})  <-- production head")

    # rung 3b — UNION + COMPOSED: does across-seed composition earn RESIDUAL on
    # top of the union? (heavy; only when heavy_union is on)
    if heavy_union and r2_tr is not None:
        C3_tr = np.concatenate([r2_tr, st_tr, Ftr], axis=1)
        C3_te = np.concatenate([r2_te, st_te, Fte], axis=1)
        r3_va, r3_te_acc, r3_lam = ridge_eval(torch, C3_tr[:n_fit], C3_tr[n_fit:],
                                              C3_te, ytr[:n_fit], yva, yte, dev)
        L(f"RUNG 3b union+composed: val {r3_va}  TEST {r3_te_acc}  "
          f"({C3_tr.shape[1]} cols, lam {r3_lam})")
    else:
        r3_va = r3_te_acc = r3_lam = None
        L("RUNG 3b union+composed: SKIPPED (heavy_union off)")

    head_cols = Cc_tr.shape[1]
    # Defaults so a crash in the (memory-heavy) diagnostics still exports the
    # LADDER — a late OOM here killed a full CIFAR run once.
    corr = so_te = ns_te_acc = go_te = genome_residual = None
    role_p = comp_p = basis_p = 0
    head_p = (head_cols + 1) * 10
    try:
        # generalisation signal: does cross-seed std flag the errors? refit the
        # rung-3c head at its lambda for test predictions.
        Xf = torch.tensor(Cc_tr, device=dev); Xe = torch.tensor(Cc_te, device=dev)
        Yf = -torch.ones((len(ytr), 10), device=dev)
        Yf[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0
        mu = Xf.mean(0); sd = Xf.std(0) + 1e-6
        Xfz = torch.cat([(Xf - mu) / sd, torch.ones(len(Xf), 1, device=dev)], 1)
        Xez = torch.cat([(Xe - mu) / sd, torch.ones(len(Xe), 1, device=dev)], 1)
        A = (Xfz.T @ Xfz).double() + r3c_lam * torch.eye(Xfz.shape[1], device=dev, dtype=torch.float64)
        W = torch.linalg.solve(A, (Xfz.T @ Yf).double()).float()
        pred = (Xez @ W).argmax(1).cpu().numpy()
        err = (pred != yte)
        seed_std_te = Gte.std(axis=2).mean(axis=(1, 2, 3))
        corr = float(np.corrcoef(seed_std_te, err.astype(np.float32))[0, 1])
        L(f"generalisation signal: corr(cross-seed std, error) = {corr:.4f}  "
          f"(err rate {err.mean():.4f})")
        Wnp = W.cpu().numpy()
        del Xf, Xe, Xfz, Xez, A, W, Yf; torch.cuda.empty_cache(); gc.collect()
        del Cc_tr, Cc_te; gc.collect()

        # ABLATION: do the evolved GENOMES earn, or is the head feasting on the
        # DETERMINISTIC cross-seed stat channels?
        so_va, so_te, _ = ridge_eval(torch, st_tr[:n_fit], st_tr[n_fit:], st_te,
                                     ytr[:n_fit], yva, yte, dev)
        ns_tr = np.concatenate([r1_tr, st_tr], axis=1)
        ns_te2 = np.concatenate([r1_te, st_te], axis=1)
        ns_va, ns_te_acc, _ = ridge_eval(torch, ns_tr[:n_fit], ns_tr[n_fit:], ns_te2,
                                         ytr[:n_fit], yva, yte, dev)
        del ns_tr, ns_te2; gc.collect()
        if Ftr.shape[1] > 0:
            _, go_te, _ = ridge_eval(torch, Ftr[:n_fit], Ftr[n_fit:], Fte,
                                     ytr[:n_fit], yva, yte, dev)
        else:
            go_te = 0.0
        genome_residual = round(r3c_te - ns_te_acc, 4)
        L(f"ABLATION: stats-only {so_te} | single+stats(NO genomes) {ns_te_acc} | "
          f"genomes-only {go_te} | production(+genomes) {r3c_te}")
        L(f"ABLATION: genome residual over single+stats = {genome_residual}  "
          f"(the {len(frozen)} evolved across-seed genomes earn THIS much)")

        def _gp(g):
            return 6 + sum(1 + 3 * len(t["prog"]) for t in g["terms"])
        role_p = int(sum(_gp(g) for g in roles))
        comp_p = int(sum(_gp(g) for g in frozen))
        basis_p = int(sum(env.basis[ps]["comps"].numel() + env.basis[ps]["mu"].numel()
                          + env.basis[ps]["sd"].numel() for ps in env.basis))
        L(f"PARAMS: head(ridge) {head_p} | evolved genomes {role_p + comp_p} "
          f"(roles {role_p}, composed {comp_p}) | pca basis {basis_p} (data-built)  "
          f"-> fitted {head_p + role_p + comp_p}")

        model = {"roles": roles, "composed": frozen, "angles": angles,
                 "seed_mode": seed_mode, "grid": grid, "head_cols": head_cols,
                 "head_lam": r3c_lam, "head_W": Wnp,
                 "basis": {ps: {k: (v.cpu().numpy() if hasattr(v, "cpu") else v)
                                for k, v in b.items()} for ps, b in env.basis.items()},
                 "params": {"head": head_p, "roles": role_p, "composed": comp_p,
                            "basis": basis_p}}
        with open(os.path.join(_HERE, "demo", f"{dataset}_radial_model.pkl"), "wb") as fh:
            pickle.dump(model, fh)
    except Exception as exc:
        L(f"[diagnostics] failed (LADDER kept): {type(exc).__name__}: {exc}")

    dt = round(time.time() - t0)
    best_test = max(x for x in (r1_te_acc, r3a_te, r3c_te, r3_te_acc) if x is not None)
    stats = {"anchor_test": a_te, "rung1_single_test": r1_te_acc,
             "rung2_union_test": r2_te_acc, "rung3a_composed_only_test": r3a_te,
             "rung3c_single_composed_test": r3c_te,
             "rung3b_union_composed_test": r3_te_acc,
             "rung1_val": r1_va, "rung2_val": r2_va,
             "rung3a_val": r3a_va, "rung3c_val": r3c_va, "rung3b_val": r3_va,
             "best_test": best_test, "n_roles": R, "n_seeds": S,
             "n_composed": len(frozen), "step_deg": step_deg, "grid": grid,
             "deskew": deskew, "seconds": dt,
             "uncertainty_corr": (round(corr, 4) if corr is not None else None),
             "stats_only_test": so_te, "single_stats_no_genomes_test": ns_te_acc,
             "genomes_only_test": go_te, "genome_residual": genome_residual,
             "head_params": head_p, "evolved_params": role_p + comp_p,
             "basis_params": basis_p, "seed_mode": seed_mode,
             "composition_beats_single": bool(max(r3a_te, r3c_te) > r1_te_acc),
             "reached_99": bool(best_test >= 0.99)}
    cfg = {"seed_mode": seed_mode, "max_deg": max_deg, "n_seeds": n_seeds,
           "step_deg": step_deg, "n_train": n_train, "n_test": n_test,
           "deskew": deskew, "grid": grid, "seed": seed, "angles": angles,
           "role_rounds": role_rounds, "comp_rounds": comp_rounds,
           "max_roles": max_roles, "heavy_union": heavy_union, "smoke": smoke}
    hist = [{"rung": 1, "test": r1_te_acc, "val": r1_va},
            {"rung": 2, "test": r2_te_acc, "val": r2_va},
            {"rung": "3a", "test": r3a_te, "val": r3a_va},
            {"rung": "3c", "test": r3c_te, "val": r3c_va},
            {"rung": "3b", "test": r3_te_acc, "val": r3_va}]
    L(f"=== LADDER: single {r1_te_acc} -> composed-only {r3a_te} -> "
      f"single+composed {r3c_te}  | BEST {best_test}  99%={stats['reached_99']}  "
      f"({dt}s) ===")

    out = {"stats": stats, "config": cfg, "hist": hist, "roles": len(roles)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{dataset}_radial.json"), "w") as f:
        json.dump(out, f, indent=1)
    if record and not smoke:
        rid = _record_run(cfg, hist, stats, lines, [dataset, "seed-stack"],
                          env_name=f"{dataset}_radial")
        out["run_id"] = rid
        L(f"recorded run {rid}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--step-deg", type=float, default=6.0)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--role-rounds", type=int, default=40)
    ap.add_argument("--comp-rounds", type=int, default=60)
    ap.add_argument("--max-roles", type=int, default=None)
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--no-heavy-union", action="store_true",
                    help="skip the naive-union rungs (they regress + hog memory)")
    ap.add_argument("--seed-mode", choices=["image", "feature"], default="image",
                    help="image = rotate the digit per seed; feature = rotate the PCA frame")
    ap.add_argument("--max-deg", type=float, default=15.0,
                    help="image mode: seeds spread over +-max_deg (seed 0 = upright)")
    ap.add_argument("--ab", action="store_true",
                    help="A/B: force-residual vs lean (do the genomes earn?)")
    args = ap.parse_args()
    run(n_seeds=args.seeds, step_deg=args.step_deg, n_train=args.n_train,
        n_test=args.n_test, deskew=not args.no_deskew, role_rounds=args.role_rounds,
        comp_rounds=args.comp_rounds, max_roles=args.max_roles, grid=args.grid,
        heavy_union=not args.no_heavy_union, seed_mode=args.seed_mode,
        max_deg=args.max_deg, ab=args.ab, smoke=args.smoke)
