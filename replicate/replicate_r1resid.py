"""replicate_r1resid.py — module 49: R1 evolved against R0's RESIDUAL.

Diagnosis of the m38 null (R0|R1 = 0.7731 ~ R0 = 0.7698): R1 got rich
information (R0's 4x4 grids + a raw skip bank) but the wrong FITNESS. m38
scored each R1 genome by make_scorer with base = R1-columns-only — i.e.
"classify from scratch," competing against other R1 genomes toward the raw
labels, with R0 ABSENT from the base and the orthogonality check blind to R0.
So evolution rediscovered R0's own discriminative directions and their union
landed back at ~0.77. Textbook suppression law: R1 was asked the SAME
question R0 already answered, so its answer was already in R0.

The fix is the fitness-as-answerable-question law made literal: R1's question
must be what R0 CANNOT answer. Fit R0's linear ridge, take its RESIDUAL (the
per-class error it can't explain — the nonlinear structure a linear head
can't reach), and evolve R1 as matching-pursuit against that residual. A
genome earns fitness only for residual variance it explains, verified on a
held split, and admitted only if decorrelated from the R1 already frozen.
Same rich R0-grid + raw-skip input as m38 — only the fitness changes.

De-risk: does R0 | R1 now lift the classification ridge above R0's 0.77? If
yes, depth was never the problem — the fitness was. If no even against the
residual, the residual has no grammar-recoverable structure.

    python3 replicate/replicate_r1resid.py --rounds 60 --cap 500
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
from radial_evo2 import Env, SCALES
import radial_stack as rk
from replicate_r1build import union_genomes

RD = os.path.join(_HERE, "radial_data")
CACHE = os.path.join(_HERE, "replicate", "cache")
STATE = os.path.join(CACHE, "state_r1resid.json")
LOG = []


def log(m):
    print(m, flush=True)
    LOG.append(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--gens", type=int, default=12)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as Fn
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda"
    tp = _tprims(torch)
    rk.GRID = 4
    G = rk.GRID
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"].astype(np.int64), z["yte"].astype(np.int64)
    N = len(ytr)
    env = Env(torch, dev, Xtr, Xte, max_cached=6)
    genomes = union_genomes()
    log(f"[r1r] union {len(genomes)} R0 genomes; {torch.cuda.get_device_name(0)}")

    gtr = torch.stack([rk.feature_r0(torch, tp, env, g, want_grid=True)
                       for g in genomes], 1).half()
    gte = torch.stack([rk.feature_r0(torch, tp, env, g, test=True,
                                     want_grid=True) for g in genomes], 1).half()
    bt, be = [], []
    for ps in SCALES:
        Mtr, Mte, H, W = env.maps(ps)
        bt.append(Fn.adaptive_avg_pool2d(Mtr.float().view(len(Mtr), -1, H, W),
                                         (G, G)))
        be.append(Fn.adaptive_avg_pool2d(Mte.float().view(len(Mte), -1, H, W),
                                         (G, G)))
    raw_tr, raw_te = torch.cat(bt, 1).half(), torch.cat(be, 1).half()
    del bt, be; torch.cuda.empty_cache()
    R0tr = torch.stack([rk.feature_r0(torch, tp, env, g) for g in genomes], 1)
    R0te = torch.stack([rk.feature_r0(torch, tp, env, g, test=True)
                        for g in genomes], 1)
    C_prev = gtr.shape[1]
    log(f"[r1r] R0 grid {tuple(gtr.shape)} + raw skip {tuple(raw_tr.shape)} "
        f"({round(time.time()-t0)}s)")

    n_fit = int(N * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yfull = -torch.ones((N, 10), device=dev)
    Yfull[torch.arange(N), torch.tensor(ytr, device=dev)] = 1.0

    def zc(F):
        return (F - F.mean(0)) / (F.std(0) + 1e-6)

    def test_of(F, Fte_):
        Yf = Yfull[:n_fit]
        bl, bv = 3.0, -1.0
        for lam in (1.0, 3.0, 10.0, 30.0, 100.0):
            _, a = _ridge_soft(torch, F[:n_fit], F[n_fit:], Yf, yv, lam=lam)
            if a > bv:
                bl, bv = lam, a
        n, d = F.shape
        mu, sd = F.mean(0), F.std(0) + 1e-6
        A = torch.hstack([(F - mu) / sd, torch.ones(n, 1, device=dev)])
        Gm = (A.T @ A).double() + bl * torch.eye(d + 1, device=dev,
                                                 dtype=torch.float64)
        W = torch.linalg.solve(Gm, (A.T @ Yfull).double()).float()
        B = torch.hstack([(Fte_ - mu) / sd, torch.ones(len(Fte_), 1, device=dev)])
        return float(((B @ W).argmax(1) == yte_t).float().mean()), bv

    # R0 baseline + its RESIDUAL (the target R1 must explain)
    r0_test, r0_val = test_of(R0tr, R0te)
    Z = zc(R0tr)
    A = torch.hstack([Z[:n_fit], torch.ones(n_fit, 1, device=dev)])
    W0 = torch.linalg.solve((A.T @ A).double() + 10.0 * torch.eye(
        A.shape[1], device=dev, dtype=torch.float64),
        (A.T @ Yfull[:n_fit]).double()).float()
    P = torch.hstack([Z, torch.ones(N, 1, device=dev)]) @ W0
    Res = (Yfull - P)                                    # (N,10) R0 can't explain
    log(f"[r1r] R0 ridge val {r0_val:.4f} TEST {r0_test:.4f}; residual RMS "
        f"{float(Res.pow(2).mean().sqrt()):.4f}")

    def resid_gain(C, Rc, idx):
        """explained-variance gain of each candidate col over residual Rc,
        on rows idx. C:(N,K), Rc:(N,10). Returns (K,) fit gains + fit coef."""
        c = C[idx] - C[idx].mean(0)
        r = Rc[idx]
        num = (c.T @ r)                                  # (K,10)
        den = (c * c).sum(0) + 1e-6                      # (K,)
        gain = (num.pow(2).sum(1)) / den                 # (K,)
        coef = num / den.unsqueeze(1)                    # (K,10)
        return gain, coef

    state = {"genomes": [], "round": 0}
    if os.path.isfile(STATE):
        state = json.load(open(STATE, encoding="utf-8"))
    rng = np.random.default_rng(909 + 1000 * state["round"])
    cols_tr = [rk.feature_grid_g(torch, tp, gtr, g, rawT=raw_tr)
               for g in state["genomes"]]
    cols_te = [rk.feature_grid_g(torch, tp, gte, g, rawT=raw_te)
               for g in state["genomes"]]
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:3000],
                         device=dev)
    fit_idx = torch.arange(n_fit, device=dev)
    val_idx = torch.arange(n_fit, N, device=dev)

    for rnd in range(args.rounds):
        if len(state["genomes"]) >= args.cap:
            break
        # current residual after what R1 already explains (refit each round)
        if cols_tr:
            R1 = zc(torch.stack(cols_tr, 1))
            Af = torch.hstack([R1[:n_fit], torch.ones(n_fit, 1, device=dev)])
            Wr = torch.linalg.solve((Af.T @ Af).double() + 5.0 * torch.eye(
                Af.shape[1], device=dev, dtype=torch.float64),
                (Af.T @ Res[:n_fit]).double()).float()
            Rcur = Res - torch.hstack([R1, torch.ones(N, 1, device=dev)]) @ Wr
        else:
            Rcur = Res
        pop = [rk.new_grid_genome(rng, C_prev) for _ in range(args.pop)]
        for gen in range(args.gens):
            cc = torch.stack([rk.feature_grid_g(torch, tp, gtr, g, rawT=raw_tr)
                              for g in pop], 1)
            gain, _ = resid_gain(cc, Rcur, fit_idx)
            order = torch.argsort(-gain).cpu().numpy()
            elite = order[:max(2, args.pop // 4)]
            npop = [pop[i] for i in elite]
            while len(npop) < args.pop:
                src = elite[rng.integers(len(elite))]
                npop.append(rk.mutate_grid_g(rng, pop[src],
                            float(rng.uniform(0.25, 0.5)), C_prev))
            pop = npop
        cc = torch.stack([rk.feature_grid_g(torch, tp, gtr, g, rawT=raw_tr)
                          for g in pop], 1)
        gain_f, coef_f = resid_gain(cc, Rcur, fit_idx)
        # verify on val: does the fit-derived direction still explain residual?
        cv = cc[val_idx] - cc[val_idx].mean(0)
        val_num = (cv.T @ Rcur[val_idx])                 # (K,10)
        val_gain = (coef_f * val_num).sum(1)             # generalizes if >0
        order = torch.argsort(-gain_f).cpu().numpy()
        adm, sigs, n_o, n_v = [], [], 0, 0
        for c in cols_tr[-96:]:
            s = c[probe] - c[probe].mean()
            sigs.append(s / (s.norm() + 1e-8))
        pcc = cc[probe]
        for i in order:
            if float(gain_f[i]) <= 0 or len(adm) >= 10:
                break
            if float(val_gain[i]) <= 0:                  # doesn't generalize
                n_v += 1
                continue
            s = pcc[:, i] - pcc[:, i].mean()
            s = s / (s.norm() + 1e-8)
            if any(float(torch.abs(s @ t)) > 0.9 for t in sigs):
                n_o += 1
                continue
            adm.append(i); sigs.append(s)
        for i in adm:
            state["genomes"].append(pop[i])
            cols_tr.append(cc[:, i])
            cols_te.append(rk.feature_grid_g(torch, tp, gte, pop[i],
                                             rawT=raw_te))
        state["round"] += 1
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(state, f)
        if rnd % 5 == 0 or len(state["genomes"]) >= args.cap:
            jt, jv = test_of(torch.cat([R0tr, torch.stack(cols_tr, 1)], 1),
                             torch.cat([R0te, torch.stack(cols_te, 1)], 1))
            log(f"[r1r] round {rnd}: +{len(adm)} -> {len(state['genomes'])} "
                f"(rej {n_o} redund, {n_v} noverify); R0|R1 TEST {jt:.4f} "
                f"(R0 {r0_test:.4f}, {jt - r0_test:+.4f}) ({round(time.time()-t0)}s)")
        else:
            log(f"[r1r] round {rnd}: +{len(adm)} -> {len(state['genomes'])} "
                f"(rej {n_o},{n_v}) ({round(time.time()-t0)}s)")

    R1tr = torch.stack(cols_tr, 1)
    R1te = torch.stack(cols_te, 1)
    r1_test, r1_val = test_of(R1tr, R1te)
    j_test, j_val = test_of(torch.cat([R0tr, R1tr], 1),
                            torch.cat([R0te, R1te], 1))
    log(f"[r1r] FINAL: R1 alone ({len(state['genomes'])}) TEST {r1_test:.4f}; "
        f"R0|R1 TEST {j_test:.4f} val {j_val:.4f} (R0 {r0_test:.4f}, "
        f"delta {j_test - r0_test:+.4f})")
    json.dump({"module": "r1resid", "n": len(state["genomes"]),
               "r0": round(r0_test, 4), "joint": round(j_test, 4)},
              open(os.path.join(RD, "replicate_r1resid.json"), "w"))


if __name__ == "__main__":
    main()
