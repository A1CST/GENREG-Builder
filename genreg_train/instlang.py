"""Internal-language experiment (user idea, 2026-07-11).

ONE genome, TEN images (one per CIFAR class). The genome is never given
targets: its only survival condition is to emit a CONSISTENT code for
occluded/jittered views of the same image and DISTINCT codes across images.
The code is the genome's own invented language — nothing external defines
what any output dimension means.

  Genome: 12 evolved 5x5x3 conv kernels (+ activation genes) -> 4x4 adaptive
  mean pool -> flatten (192) -> W1 -> hidden 24 (tanh) -> W2 -> code vector.
  Round 1: fixed code width 64. Round 2: the WIDTH itself is a gene (8..128)
  — the genome decides how long its words are.

  Fitness (soft, dense, deterministic): mean within-image cosine across a
  FIXED bank of occluded views minus mean between-image cosine. No labels,
  no targets, no reconstruction.

  Emergence test (after evolution): encode UNSEEN validation images; assign
  each to the nearest anchor code. If accuracy beats 10% chance, the genome's
  instance language accidentally encodes class essence.
"""
import os
import pickle
import time

import numpy as np

from genreg_train import cifar_pipe as cp
from genreg_train import mnist_pipe as mp
from genreg_train import evo_gpu

torch = evo_gpu.torch
DEV = evo_gpu.DEV
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NK, HID, MAXW = 12, 24, 128                      # kernels, hidden dim, max code width
N_VIEWS = 24                                     # fixed occluded views per anchor


def make_views(img, n, rng):
    """n occluded+jittered views of one 32x32x3 image (deterministic given rng)."""
    out = np.empty((n, 32, 32, 3), np.float32)
    mean = img.reshape(-1, 3).mean(0)
    for i in range(n):
        v = img.copy()
        if rng.random() < 0.8:                   # occlusion (the main pressure)
            y0, x0 = rng.integers(0, 20, 2)
            v[y0:y0 + 12, x0:x0 + 12, :] = mean
        if rng.random() < 0.5:                   # 1px jitter
            v2 = np.zeros_like(v)
            v2[:-1, :-1, :] = v[1:, 1:, :]
            v = v2
        out[i] = v
    return out


class InstPop:
    """Population of full encoder genomes."""

    def __init__(self, pop, seed, width_gene=False):
        rng = np.random.default_rng(seed)
        self.pop = pop
        self.K = (rng.standard_normal((pop, NK, 75)) * 0.25).astype(np.float32)
        self.kact = rng.integers(0, 8, (pop, NK)).astype(np.float32)
        self.W1 = (rng.standard_normal((pop, NK * 16, HID)) / np.sqrt(NK * 16)).astype(np.float32)
        self.b1 = np.zeros((pop, HID), np.float32)
        self.W2 = (rng.standard_normal((pop, HID, MAXW)) / np.sqrt(HID)).astype(np.float32)
        self.b2 = np.zeros((pop, MAXW), np.float32)
        self.width = (rng.integers(8, MAXW + 1, pop).astype(np.float32)
                      if width_gene else np.full(pop, 64.0, np.float32))
        self.width_gene = width_gene
        self.sigma = np.full(pop, 0.08, np.float32)

    def params(self):
        d = {"K": self.K, "kact": self.kact, "W1": self.W1, "b1": self.b1,
             "W2": self.W2, "b2": self.b2, "sigma": self.sigma}
        if self.width_gene:
            d["width"] = self.width
        return d

    def set_params(self, d):
        self.K, self.kact, self.W1, self.b1 = d["K"], d["kact"], d["W1"], d["b1"]
        self.W2, self.b2, self.sigma = d["W2"], d["b2"], d["sigma"]
        self.kact = np.round(self.kact) % 8
        if self.width_gene:
            self.width = np.clip(np.round(d["width"]), 8, MAXW).astype(np.float32)

    def champion(self, i):
        return {"K": self.K[i].copy(), "kact": self.kact[i].astype(np.int64).copy(),
                "W1": self.W1[i].copy(), "b1": self.b1[i].copy(),
                "W2": self.W2[i].copy(), "b2": self.b2[i].copy(),
                "width": int(self.width[i])}


class InstFitGPU:
    """Codes + fitness for the whole population over the fixed view bank."""

    def __init__(self, views, owner):
        """views (N,32,32,3), owner (N,) anchor index 0..9."""
        self.N = len(views)
        self.Pf = evo_gpu.to_dev(cp._im2col5c(views).reshape(-1, 75))
        self.owner = evo_gpu.to_dev(owner.astype(np.int64))
        same = owner[:, None] == owner[None, :]
        eye = np.eye(self.N, dtype=bool)
        self.m_within = evo_gpu.to_dev((same & ~eye).astype(np.float32))
        self.m_between = evo_gpu.to_dev((~same).astype(np.float32))
        self.nw = float(self.m_within.sum().item())
        self.nb = float(self.m_between.sum().item())

    @torch.no_grad()
    def codes(self, popn):
        P = popn.pop
        Kg = evo_gpu.to_dev(popn.K.reshape(P * NK, 75))
        resp = (self.Pf @ Kg.T).reshape(self.N, 28, 28, P, NK) \
            + 0.0                                       # bias-free kernels
        feats = torch.empty((P, self.N, NK, 16), device=DEV)
        ka = popn.kact.astype(np.int64) % 8
        for a in range(8):
            pi, ki = np.where(ka == a)
            if len(pi) == 0:
                continue
            sel = resp[:, :, :, evo_gpu.to_dev(pi.astype(np.int64)),
                       evo_gpu.to_dev(ki.astype(np.int64))]
            blk = evo_gpu._acts_t(sel, a).permute(3, 0, 1, 2) \
                .reshape(len(pi) * self.N, 28, 28)
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                blk[:, None, :, :], 4).reshape(len(pi), self.N, 16)
            feats[evo_gpu.to_dev(pi.astype(np.int64)),
                  :, evo_gpu.to_dev(ki.astype(np.int64)), :] = pooled
        x = feats.reshape(P, self.N, NK * 16)
        x = x - x.mean(dim=2, keepdim=True)
        x = x / (x.std(dim=2, keepdim=True) + 1e-6)
        W1 = evo_gpu.to_dev(popn.W1); b1 = evo_gpu.to_dev(popn.b1)
        W2 = evo_gpu.to_dev(popn.W2); b2 = evo_gpu.to_dev(popn.b2)
        h = torch.tanh(torch.einsum("pnf,pfh->pnh", x, W1) + b1[:, None, :])
        code = torch.einsum("pnh,phw->pnw", h, W2) + b2[:, None, :]
        # width gene: mask beyond the genome's word length
        wmask = (torch.arange(MAXW, device=DEV)[None, :]
                 < evo_gpu.to_dev(popn.width)[:, None]).float()
        code = code * wmask[:, None, :]
        code = code - code.mean(dim=2, keepdim=True) * 0   # keep raw, just norm
        code = code / (code.norm(dim=2, keepdim=True) + 1e-8)
        return code                                       # (P,N,MAXW) unit rows

    @torch.no_grad()
    def fitness(self, popn):
        code = self.codes(popn)
        S = torch.einsum("pnw,pmw->pnm", code, code)      # (P,N,N) cosines
        within = (S * self.m_within).sum(dim=(1, 2)) / self.nw
        between = (S * self.m_between).sum(dim=(1, 2)) / self.nb
        return (within - between).cpu().numpy(), \
            within.cpu().numpy(), between.cpu().numpy()


def run(gens=3000, pop=200, seed=7, width_gene=False, log=print):
    Xtr, ytr, Xva, yva, _, _ = cp.load_cifar()
    rng = np.random.default_rng(seed + 1)
    anchors = np.stack([Xtr[rng.choice(np.where(ytr == c)[0])] for c in range(10)])
    vr = np.random.default_rng(seed + 2)
    views, owner = [], []
    for c in range(10):
        views.append(make_views(anchors[c], N_VIEWS, vr))
        owner += [c] * N_VIEWS
    views = np.concatenate(views); owner = np.array(owner)
    fitgpu = InstFitGPU(views, owner)
    popn = InstPop(pop, seed, width_gene=width_gene)
    rng0 = np.random.default_rng(seed)
    best, champ = -1e9, None
    for gen in range(1, gens + 1):
        fit, wi, be = fitgpu.fitness(popn)
        if float(fit[0]) > best:
            best = float(fit[0]); champ = popn.champion(0)
        pd = popn.params()
        mp.ga_step(pd, fit, rng0, mag_scale=True)
        popn.set_params(pd)
        if gen % 250 == 0 or gen == 1:
            j = int(np.argmax(fit))
            log(f"  [instlang] gen {gen}: sep={fit[j]:.4f} "
                f"(within {wi[j]:.3f} / between {be[j]:.3f})"
                + (f" width={int(popn.width[j])}" if width_gene else ""))
    out = {"champ": champ, "separation": round(best, 4), "anchors": anchors,
           "views": views, "owner": owner, "width_gene": width_gene}
    return out


def emergence_test(result, n_eval=2000, log=print):
    """Do UNSEEN images land nearest their category's anchor code?"""
    Xtr, ytr, Xva, yva, _, _ = cp.load_cifar()
    champ = result["champ"]
    popn = InstPop(1, 0, width_gene=True)
    popn.K[0] = champ["K"]; popn.kact[0] = champ["kact"]
    popn.W1[0] = champ["W1"]; popn.b1[0] = champ["b1"]
    popn.W2[0] = champ["W2"]; popn.b2[0] = champ["b2"]
    popn.width[0] = champ["width"]

    def encode(imgs):
        f = InstFitGPU(imgs, np.zeros(len(imgs), np.int64))
        return f.codes(popn)[0].cpu().numpy()

    # anchor codes = mean code over each anchor's view bank
    vc = encode(result["views"])
    anchor_codes = np.stack([vc[result["owner"] == c].mean(0) for c in range(10)])
    anchor_codes /= np.linalg.norm(anchor_codes, axis=1, keepdims=True) + 1e-8
    rng = np.random.default_rng(11)
    idx = rng.choice(len(Xva), n_eval, replace=False)
    codes = encode(Xva[idx])
    pred = (codes @ anchor_codes.T).argmax(1)
    acc = float((pred == yva[idx]).mean())
    log(f"EMERGENCE: unseen-image nearest-anchor accuracy {acc:.4f} "
        f"(chance 0.10) over {n_eval} val images")
    # similarity matrix of anchor codes (how distinct the invented words are)
    sim = np.round(anchor_codes @ anchor_codes.T, 2)
    log("anchor code similarity matrix (off-diagonal should be low):")
    for row in sim:
        log("  " + " ".join(f"{v:+.2f}" for v in row))
    return acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=3000)
    ap.add_argument("--width-gene", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    res = run(gens=a.gens, width_gene=a.width_gene, seed=a.seed)
    print(f"separation {res['separation']} ({time.time() - t0:.0f}s)")
    with open(os.path.join(ROOT, "demo",
                           f"instlang{'_wg' if a.width_gene else ''}.pkl"), "wb") as f:
        pickle.dump(res, f)
    emergence_test(res)
