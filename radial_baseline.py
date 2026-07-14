"""radial_baseline.py — the domain baselines from documentation/RADIAL_BASELINES.md:
the FULL radial lens bank + a closed-form linear head, no genomes. These numbers
define exactly where genome evolution needs to begin.

Input-format decision (logged per run): FLAT VECTOR, POINTWISE application —
each lens transforms every pixel independently, the linear head sees all
n_pixels x n_lens features. Because the head is linear over pointwise features,
the whole model is a GAM over pixels; anything above raw-linear accuracy is
bought purely by lens diversity. Implemented as kernel ridge (the summed linear
kernel of per-lens feature blocks) so the full bank fits in memory; runs on
CUDA when available. No gradients anywhere — kmeans-free, closed-form solve.

Data: radial-owned copies only (radial_data/*.npz); source projects untouched.
"""
import os
import json
import time
import numpy as np

import radial_map as rmap

_HERE = os.path.dirname(os.path.abspath(__file__))
_MNIST_NPZ = os.path.join(_HERE, "radial_data", "mnist_radial.npz")


def mnist_data():
    """Radial-owned MNIST copy (8k train / 2k test), built once from the
    corpora/mnist idx files (read-only) like the CIFAR copy was."""
    if not os.path.exists(_MNIST_NPZ):
        import gzip
        import struct

        def read_idx(path):
            with gzip.open(path, "rb") as f:
                magic = struct.unpack(">I", f.read(4))[0]
                shape = struct.unpack(">" + "I" * (magic & 0xFF), f.read(4 * (magic & 0xFF)))
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

        d = os.path.join(_HERE, "corpora", "mnist")
        np.savez(_MNIST_NPZ,
                 Xtr=read_idx(os.path.join(d, "train-images-idx3-ubyte.gz"))[:8000],
                 ytr=read_idx(os.path.join(d, "train-labels-idx1-ubyte.gz"))[:8000].astype(np.int64),
                 Xte=read_idx(os.path.join(d, "t10k-images-idx3-ubyte.gz"))[:2000],
                 yte=read_idx(os.path.join(d, "t10k-labels-idx1-ubyte.gz"))[:2000].astype(np.int64))
    z = np.load(_MNIST_NPZ)
    return (z["Xtr"].astype(np.float32) / 255.0, z["ytr"],
            z["Xte"].astype(np.float32) / 255.0, z["yte"])


# ---------------------------------------------------------------------------
# torch lens application (same programs, same math as radial_map.PRIMS)
# ---------------------------------------------------------------------------

def _tprims(torch):
    import math
    return {
        "id":    lambda z: z,
        "sin":   torch.sin,
        "cos":   torch.cos,
        "tanh":  torch.tanh,
        "relu":  lambda z: torch.clamp(z, min=0.0),
        "abs":   torch.abs,
        "sq":    lambda z: torch.clamp(z * z, -30, 30),
        "sign":  torch.sign,
        "gauss": lambda z: torch.exp(-torch.clamp(z * z, 0, 30)),
        "soft":  lambda z: z / (1.0 + torch.abs(z)),
        "sqrt":  lambda z: torch.sqrt(torch.abs(z)),
        "step":  lambda z: (z > 0).float(),
        "logp":  lambda z: torch.log1p(torch.abs(z)) * torch.sign(z),
        "sinc":  lambda z: torch.sinc(z / math.pi),
    }


def _lens_apply_t(tp, prog, x):
    y = x
    for name, a, b in prog:
        y = tp[name](a * y + b)
    return y


def _kernels(Xtr, Xte, ytr, lens_idx, snapshots, solve_fn, verbose=False):
    """Accumulate the summed per-lens linear kernel over the bank; at each
    snapshot lens-count, run the solver and record accuracy. Returns
    (per-snapshot results, per-lens label-alignment scores, kept lens ids)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    Ftr = torch.tensor(Xtr.reshape(len(Xtr), -1), device=dev)
    Fte = torch.tensor(Xte.reshape(len(Xte), -1), device=dev)
    Ktr = torch.zeros((len(Ftr), len(Ftr)), device=dev)
    Kte = torch.zeros((len(Fte), len(Ftr)), device=dev)
    n = len(ytr)
    Yc = -torch.ones((n, 10), device=dev) / 10
    Yc[torch.arange(n), torch.tensor(ytr, device=dev)] += 1.0
    yn = torch.norm(Yc)
    results, aligns, kept = [], [], []
    t0 = time.time()
    stop_at = max(snapshots)
    for i in lens_idx:
        prog = rmap.lens_program(i)
        Ztr = _lens_apply_t(tp, prog, Ftr)
        sd = Ztr.std(0)
        if float(sd.mean()) < 1e-6:
            continue
        mu = Ztr.mean(0)
        ok = sd > 1e-6
        Ztr = torch.where(ok, (Ztr - mu) / (sd + 1e-9), torch.zeros_like(Ztr))
        Zte = _lens_apply_t(tp, prog, Fte)
        Zte = torch.where(ok, (Zte - mu) / (sd + 1e-9), torch.zeros_like(Zte))
        Ktr += Ztr @ Ztr.T
        Kte += Zte @ Ztr.T
        kept.append(int(i))
        # label alignment scored NOW so the feature tensor is never retained
        aligns.append((int(i), round(float(torch.norm(Ztr.T @ Yc) /
                                           (torch.norm(Ztr) * yn + 1e-9)), 4)))
        if len(kept) in snapshots:
            acc = solve_fn(Ktr, Kte, len(kept))
            results.append(acc)
            if verbose:
                print(f"  L={len(kept):4d}  acc {acc['acc']:.4f}  (lam {acc['lam']}, "
                      f"{round(time.time()-t0)}s)", flush=True)
        if len(kept) >= stop_at:
            break
    return results, aligns, kept


def _kridge_factory(ytr, yte, lams=(1.0, 10.0, 100.0, 1000.0)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(ytr)
    Y = -torch.ones((n, 10), device=dev)
    Y[torch.arange(n), torch.tensor(ytr, device=dev)] = 1.0
    yte_t = torch.tensor(yte, device=dev)
    eye = torch.eye(n, device=dev)

    def solve(Ktr, Kte, n_lens):
        best = {"acc": 0.0, "lam": None, "n_lens": n_lens}
        for lam in lams:
            a = torch.linalg.solve(Ktr + lam * n_lens * eye, Y)
            pred = (Kte @ a).argmax(1)
            acc = float((pred == yte_t).float().mean())
            if acc > best["acc"]:
                pr = pred
                best = {"acc": round(acc, 4), "lam": lam, "n_lens": n_lens}
                best_pred = pr
        best["pred"] = best_pred.cpu().numpy()
        return best

    return solve


def mnist_baseline(n_lens=400, snapshots=(8, 32, 128, 400), verbose=True):
    """The roadmap's MNIST baseline: raw-pixel linear head vs the same head on
    the pointwise lens bank, accuracy-vs-bank-size curve (coverage vs
    compounding), per-class recall, top label-aligned lenses."""
    import torch
    Xtr, ytr, Xte, yte = mnist_data()
    solve = _kridge_factory(ytr, yte)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # raw-pixel linear baseline == kernel ridge with the plain linear kernel
    Ftr = torch.tensor(Xtr.reshape(len(Xtr), -1), device=dev)
    Fte = torch.tensor(Xte.reshape(len(Xte), -1), device=dev)
    raw = solve(Ftr @ Ftr.T, Fte @ Ftr.T, 1)
    if verbose:
        print(f"raw-pixel linear head:  acc {raw['acc']:.4f}  (lam {raw['lam']})", flush=True)

    results, aligns, kept = _kernels(Xtr, Xte, ytr, range(n_lens * 2), set(snapshots),
                                     solve, verbose=verbose)
    final = results[-1]

    # per-class recall at the full bank
    per_class = {}
    for c in range(10):
        m = yte == c
        per_class[str(c)] = round(float((final["pred"][m] == c).mean()), 4)

    scored = sorted(((s, i) for i, s in aligns), reverse=True)
    top = [{"i": i, "align": s, "prog": rmap.prog_str(rmap.lens_program(i))}
           for s, i in scored[:10]]

    out = {
        "domain": "mnist", "input_format": "flat 784 vector, pointwise lenses (GAM)",
        "train": len(ytr), "test": len(yte),
        "raw_linear_acc": raw["acc"],
        "curve": [{"n_lens": r["n_lens"], "acc": r["acc"], "lam": r["lam"]} for r in results],
        "acc": final["acc"], "per_class": per_class, "top_lenses": top,
    }
    if verbose:
        print(f"per-class: {per_class}", flush=True)
        print("top lenses:", *(f"\n  {t['align']}  #{t['i']}  {t['prog']}" for t in top[:5]),
              flush=True)
    return out


def mnist_rotation(n_lens=400, step_deg=1.0, frac=0.06, verbose=True):
    """The 360-degree rotation probe on MNIST classification: slices come from
    the MNIST-domain map (signatures on the mnist stream), per angle the
    kernel head sees only the co-planar lens slice."""
    import torch
    Xtr, ytr, Xte, yte = mnist_data()
    solve = _kridge_factory(ytr, yte)

    Sraw, _, _, _ = rmap.build_signatures(n_lens, "mnist")
    S = (Sraw - Sraw.mean(0)) / (Sraw.std(0) + 1e-9)
    X3 = rmap._mds(S, 3)
    shape = [round(float(v), 2) for v in X3.std(0)]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    Ftr = torch.tensor(Xtr.reshape(len(Xtr), -1), device=dev)
    Fte = torch.tensor(Xte.reshape(len(Xte), -1), device=dev)

    def slice_acc(idx):
        Ktr = torch.zeros((len(Ftr), len(Ftr)), device=dev)
        Kte = torch.zeros((len(Fte), len(Ftr)), device=dev)
        used = 0
        for i in idx:
            prog = rmap.lens_program(int(i))
            Ztr = _lens_apply_t(tp, prog, Ftr)
            sd = Ztr.std(0)
            if float(sd.mean()) < 1e-6:
                continue
            mu = Ztr.mean(0)
            ok = sd > 1e-6
            Ztr = torch.where(ok, (Ztr - mu) / (sd + 1e-9), torch.zeros_like(Ztr))
            Zte = _lens_apply_t(tp, prog, Fte)
            Zte = torch.where(ok, (Zte - mu) / (sd + 1e-9), torch.zeros_like(Zte))
            Ktr += Ztr @ Ztr.T
            Kte += Zte @ Ztr.T
            used += 1
        if used == 0:
            return 0.0
        return solve(Ktr, Kte, used)["acc"]

    angles, accs = [], []
    t0 = time.time()
    for k in range(int(round(360.0 / step_deg))):
        th = np.deg2rad(k * step_deg)
        z2 = -X3[:, 0] * np.sin(th) + X3[:, 2] * np.cos(th)
        idx = np.where(np.abs(z2) <= np.quantile(np.abs(z2), frac))[0]
        angles.append(round(k * step_deg, 2))
        accs.append(round(slice_acc(idx), 4))
        if verbose and k % 30 == 0:
            print(f"  {k * step_deg:5.0f} deg  acc {accs[-1]:.4f}  ({round(time.time()-t0)}s)",
                  flush=True)
    m = int(np.median([len(np.where(np.abs(-X3[:, 0] * np.sin(np.deg2rad(a)) +
            X3[:, 2] * np.cos(np.deg2rad(a))) <= np.quantile(np.abs(-X3[:, 0] *
            np.sin(np.deg2rad(a)) + X3[:, 2] * np.cos(np.deg2rad(a))), frac))[0])
            for a in (0, 90, 180, 270)]))
    rng = np.random.default_rng(3)
    rand = [slice_acc(rng.choice(n_lens, m, replace=False)) for _ in range(5)]
    best, worst = int(np.argmax(accs)), int(np.argmin(accs))
    return {
        "domain": "mnist", "n_lens": n_lens, "step_deg": step_deg, "slice_size": m,
        "map_shape_axis_std": shape,
        "angles": angles, "acc": accs,
        "best": {"deg": angles[best], "acc": accs[best]},
        "worst": {"deg": angles[worst], "acc": accs[worst]},
        "angular_spread": round(float(max(accs) - min(accs)), 4),
        "baseline_random_mean": round(float(np.mean(rand)), 4),
        "baseline_random_std": round(float(np.std(rand)), 4),
    }


def run_mnist(out_path=None):
    """Full MNIST baseline per the roadmap's success criteria; writes the JSON
    export (probe + rotation_probe + map shape)."""
    t0 = time.time()
    probe = mnist_baseline()
    rot = mnist_rotation()
    out = {"format": "radial-baseline-export", "domain": "mnist",
           "probe": probe, "rotation_probe": rot,
           "seconds": round(time.time() - t0)}
    path = out_path or os.path.join(_HERE, "radial_data", "baseline_mnist.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwritten {path} ({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "rotate" in sys.argv:
        r = mnist_rotation()
        print(f"best {r['best']}, worst {r['worst']}, spread {r['angular_spread']}, "
              f"random {r['baseline_random_mean']} +/- {r['baseline_random_std']}")
    elif "full" in sys.argv:
        run_mnist()
    else:
        mnist_baseline()
