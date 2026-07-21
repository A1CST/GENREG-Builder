"""replicate_nonlin.py — module 46: a NONLINEAR patch environment.

m44/m45 located the perception lever precisely: not the readout, not spatial
pooling, not hand-added linear channels (patch-PCA already spans edges, being
linear). The ONE thing patch-PCA structurally cannot represent is a NONLINEAR
patch encoding. This is the Coates-Ng move — the single-layer method that hit
~80% on CIFAR gradient-free: a learned patch DICTIONARY (k-means prototypes,
data-statistics only, no gradients/labels) with a rectified activation, in
place of linear PCA axes.

NonlinEnv is a drop-in for Env (same maps() interface), so the IDENTICAL
grammar/evolution runs over it. Per scale: contrast-normalize patches, learn
K=40 k-means atoms, encode each patch by signed rectification
[relu(<x,atom>), relu(-<x,atom>)] -> 80 channels (same budget as 40 signed
PCA axes, but nonlinear + prototype-based). Matched A/B vs the raw-PCA
baseline (0.6694 @ 573 genomes, module 45): does a nonlinear perception
primitive break past linear patch-PCA?

    python3 replicate/replicate_nonlin.py --rounds 80 --cap 2500
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
for _p in ("replicate", "radial", "ocr"):
    sys.path.insert(0, os.path.join(_HERE, _p))
import genreg_paths                               # noqa: F401

from radial_evo import _tprims, _ridge_soft
from radial_evo2 import make_scorer, new_genome, mutate, feature

RD = os.path.join(_HERE, "radial_data")
CACHE = os.path.join(_HERE, "replicate", "cache")
STATE = os.path.join(CACHE, "state_nonlin.json")
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


class NonlinEnv:
    """Drop-in for radial_evo2.Env: maps(ps) -> (Mtr, Mte, H, W) with M shaped
    (N, 2K, H*W) fp16. Linear PCA axes replaced by k-means prototypes + signed
    rectification — a nonlinear patch code outside patch-PCA's linear span."""

    def __init__(self, torch, dev, Xtr, Xte, K=40, max_cached=6):
        self.torch, self.dev = torch, dev
        self.Xtr, self.Xte = Xtr, Xte
        self.K = K
        self.cache, self.last_used, self.tick = {}, {}, 0
        self.max_cached = max_cached
        self._dict = {}                              # ps -> (atoms, )

    def _normseq(self, cols):
        mu = cols.mean(1, keepdim=True)
        x = cols - mu
        x = x / (x.var(1, keepdim=True).sqrt() + 1e-2)
        return x

    def _learn(self, ps, d, stride):
        torch = self.torch
        import torch.nn.functional as Fn
        i2k = torch.tensor(self.Xtr[:3000], device=self.dev
                           ).permute(0, 3, 1, 2).contiguous()
        P = Fn.unfold(i2k, ps, stride=stride)
        cols = P.permute(0, 2, 1).reshape(-1, d)
        g = torch.Generator(device="cpu").manual_seed(ps)
        cols = cols[torch.randperm(len(cols), generator=g)[:100000].to(self.dev)]
        x = self._normseq(cols)                      # contrast-normalized
        # k-means (Lloyd), cosine assignment on unit-norm atoms
        g2 = torch.Generator(device="cpu").manual_seed(ps + 1)
        atoms = x[torch.randperm(len(x), generator=g2)[:self.K].to(self.dev)].clone()
        atoms = atoms / (atoms.norm(dim=1, keepdim=True) + 1e-8)
        for _ in range(15):
            sim = x @ atoms.T                        # (n, K)
            a = sim.argmax(1)
            for k in range(self.K):
                m = a == k
                if int(m.sum()) > 0:
                    atoms[k] = x[m].mean(0)
            atoms = atoms / (atoms.norm(dim=1, keepdim=True) + 1e-8)
        return atoms

    def maps(self, ps):
        torch = self.torch
        import torch.nn.functional as Fn
        self.tick += 1
        if ps in self.cache:
            self.last_used[ps] = self.tick
            return self.cache[ps]
        if len(self.cache) >= self.max_cached:
            ev = min(self.last_used, key=self.last_used.get)
            del self.cache[ev]; del self.last_used[ev]
            torch.cuda.empty_cache()
        stride = max(2, ps // 2)
        d = ps * ps * self.Xtr.shape[3]
        atoms = self._dict.get(ps)
        if atoms is None:
            atoms = self._learn(ps, d, stride)
            self._dict[ps] = atoms
        S = self.Xtr.shape[1]
        H = W = (S - ps) // stride + 1

        def build(X, bs=400):
            out, sd = None, None
            for b in range(0, len(X), bs):
                imgs = torch.tensor(X[b:b + bs], device=self.dev
                                    ).permute(0, 3, 1, 2).contiguous()
                U = Fn.unfold(imgs, ps, stride=stride)      # (B, d, L)
                Bn, _, L = U.shape
                xc = self._normseq(U.permute(0, 2, 1).reshape(-1, d))
                s = xc @ atoms.T                             # (B*L, K)
                enc = torch.cat([torch.relu(s), torch.relu(-s)], 1)  # (B*L, 2K)
                M = enc.reshape(Bn, L, 2 * self.K).permute(0, 2, 1)  # (B,2K,L)
                if out is None:
                    out = torch.zeros((len(X), 2 * self.K, L),
                                      device=self.dev, dtype=torch.float16)
                if sd is None:
                    sd = M.std((0, 2), keepdim=True) + 1e-6
                out[b:b + Bn] = (M / sd).half()
            return out

        Mtr, Mte = build(self.Xtr), build(self.Xte)
        self.cache[ps] = (Mtr, Mte, H, W)
        self.last_used[ps] = self.tick
        return self.cache[ps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=80)
    ap.add_argument("--cap", type=int, default=2500)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--K", type=int, default=40)
    args = ap.parse_args()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    tp = _tprims(torch)
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"].astype(np.int64), z["yte"].astype(np.int64)
    N = len(ytr)
    env = NonlinEnv(torch, dev, Xtr, Xte, K=args.K, max_cached=6)
    log(f"[nl] nonlinear k-means env K={args.K} (2K chans), "
        f"{torch.cuda.get_device_name(0)}")

    n_fit = int(N * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yfull = -torch.ones((N, 10), device=dev)
    Yfull[torch.arange(N), torch.tensor(ytr, device=dev)] = 1.0

    state = {"genomes": [], "round": 0}
    if os.path.isfile(STATE):
        state = json.load(open(STATE, encoding="utf-8"))
    rng = np.random.default_rng(4500 + 1000 * state["round"])
    cols_tr = [feature(torch, tp, env, g) for g in state["genomes"]]
    cols_te = [feature(torch, tp, env, g, test=True) for g in state["genomes"]]
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:3000],
                         device=dev)
    log(f"[nl] {len(state['genomes'])} genomes, round {state['round']}")

    for rnd in range(args.rounds):
        if len(state["genomes"]) >= args.cap:
            break
        pr = torch.tensor(rng.permutation(N), device=dev)
        yy = torch.tensor(ytr, device=dev)[pr]
        Yf = -torch.ones((n_fit, 10), device=dev)
        Yf[torch.arange(n_fit), yy[:n_fit]] = 1.0
        base = (torch.stack(cols_tr, 1)[pr] if cols_tr
                else torch.zeros((N, 0), device=dev))
        scorer, s0, a0 = make_scorer(torch, base, n_fit, Yf, yy[n_fit:])
        pr2 = torch.tensor(rng.permutation(N), device=dev)
        yy2 = torch.tensor(ytr, device=dev)[pr2]
        Yf2 = -torch.ones((n_fit, 10), device=dev)
        Yf2[torch.arange(n_fit), yy2[:n_fit]] = 1.0
        base2 = (torch.stack(cols_tr, 1)[pr2] if cols_tr
                 else torch.zeros((N, 0), device=dev))
        scorer2, _s2, a02 = make_scorer(torch, base2, n_fit, Yf2, yy2[n_fit:])

        pop = [new_genome(rng) for _ in range(args.pop)]
        for gen in range(args.gens):
            cc = torch.stack([feature(torch, tp, env, g) for g in pop], 1)
            _, accs = scorer(cc[pr])
            accs = np.array(accs)
            order = np.argsort(-accs)
            elite = order[:max(2, args.pop // 4)]
            npop = [pop[i] for i in elite]
            while len(npop) < args.pop:
                npop.append(mutate(rng, pop[elite[rng.integers(len(elite))]],
                                   float(rng.uniform(0.25, 0.5))))
            pop = npop
        cc = torch.stack([feature(torch, tp, env, g) for g in pop], 1)
        _, accs = scorer(cc[pr])
        accs = np.array(accs)
        order = np.argsort(-accs)
        adm, sigs, n_o, n_v = [], [], 0, 0
        for c in cols_tr[-64:]:
            s = c[probe] - c[probe].mean()
            sigs.append(s / (s.norm() + 1e-8))
        pcc = cc[probe]
        for i in order:
            if accs[i] - a0 < 0.0004 or len(adm) >= 8:
                break
            s = pcc[:, i] - pcc[:, i].mean()
            s = s / (s.norm() + 1e-8)
            if any(float(torch.abs(s @ t)) > 0.85 for t in sigs):
                n_o += 1
                continue
            _, a2 = scorer2(cc[pr2][:, i:i + 1])
            if a2[0] - a02 < 0.0002:
                n_v += 1
                continue
            adm.append(i)
            sigs.append(s)
        for i in adm:
            state["genomes"].append(pop[i])
            cols_tr.append(cc[:, i])
            cols_te.append(feature(torch, tp, env, pop[i], test=True))
        state["round"] += 1
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(state, f)
        if rnd % 5 == 0 or len(state["genomes"]) >= args.cap:
            Ftr = torch.stack(cols_tr, 1)
            Yf0 = -torch.ones((n_fit, 10), device=dev)
            Yf0[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
            bv = max(_ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf0, yv,
                                 lam=l)[1] for l in (1.0, 3.0, 10.0, 30.0))
            log(f"[nl] round {state['round']-1}: +{len(adm)} -> "
                f"{len(state['genomes'])} (rej {n_o},{n_v}); val {bv:.4f} "
                f"({round(time.time()-t0)}s)")
        else:
            log(f"[nl] round {state['round']-1}: +{len(adm)} -> "
                f"{len(state['genomes'])} (rej {n_o},{n_v}) "
                f"({round(time.time()-t0)}s)")

    Ftr = torch.stack(cols_tr, 1)
    Fte = torch.stack(cols_te, 1)
    bl, bv = 3.0, -1.0
    Yf0 = -torch.ones((n_fit, 10), device=dev)
    Yf0[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    for lam in (1.0, 3.0, 10.0, 30.0, 100.0):
        _, a = _ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf0, yv, lam=lam)
        if a > bv:
            bl, bv = lam, a
    n, d = Ftr.shape
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    A = torch.hstack([(Ftr - mu) / sd, torch.ones(n, 1, device=dev)])
    G = (A.T @ A).double() + bl * torch.eye(d + 1, device=dev,
                                            dtype=torch.float64)
    W = torch.linalg.solve(G, (A.T @ Yfull).double()).float()
    B = torch.hstack([(Fte - mu) / sd, torch.ones(len(Fte), 1, device=dev)])
    acc = float(((B @ W).argmax(1) == yte_t).float().mean())
    log(f"[nl] FINAL [nonlinear]: {len(state['genomes'])} genomes, "
        f"val {bv:.4f} TEST {acc:.4f} (raw-PCA control 0.6694 @ 573)")
    json.dump({"module": "nonlin", "n": len(state["genomes"]),
               "test": round(acc, 4)},
              open(os.path.join(RD, "replicate_nonlin.json"), "w"))


if __name__ == "__main__":
    main()
