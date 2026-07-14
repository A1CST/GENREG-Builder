"""radial_baseline.py — the domain baselines from documentation/RADIAL_BASELINES.md:
the FULL radial lens bank + a closed-form linear head, no genomes. These numbers
define exactly where genome evolution needs to begin.

Input-format decision (logged per run): FLAT VECTOR, POINTWISE application —
each lens transforms every input value independently, the linear head sees all
n_values x n_lens features. Because the head is linear over pointwise features,
the whole model is a GAM over positions; anything above raw-linear accuracy is
bought purely by lens diversity. Images/audio use kernel ridge (summed per-lens
linear kernels, CUDA when available); text uses primal ridge (scalar input).
No gradients anywhere — closed-form solves only.

Data: radial-owned copies only (radial_data/*.npz); source projects read once,
never written.
"""
import os
import json
import time
import numpy as np

import radial_map as rmap

_HERE = os.path.dirname(os.path.abspath(__file__))
_MNIST_NPZ = os.path.join(_HERE, "radial_data", "mnist_radial.npz")
_CIFAR_NPZ = os.path.join(_HERE, "radial_data", "cifar_radial.npz")
_WIKI = os.path.join(_HERE, "corpora", "wikipedia", "wiki_corpus.txt")


# ---------------------------------------------------------------------------
# domain data (radial-owned)
# ---------------------------------------------------------------------------

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


def cifar_data():
    z = np.load(_CIFAR_NPZ)
    return (z["Xtr"].astype(np.float32) / 255.0, z["ytr"],
            z["Xte"].astype(np.float32) / 255.0, z["yte"])


_TEXT = None


def text_data(n_train=60000, n_test=15000):
    """Char-level next-char prediction from the wiki corpus (read-only).
    Vocab = the 50 most common chars (rest -> space). The pointwise input is
    the CURRENT char id only, so the theoretical ceiling is the bigram table —
    exactly what the roadmap wants to locate."""
    global _TEXT
    if _TEXT is None:
        with open(_WIKI, encoding="utf-8", errors="ignore") as f:
            raw = f.read(400000).lower()
        from collections import Counter
        keep = [c for c, _ in Counter(raw).most_common(50)]
        cmap = {c: i for i, c in enumerate(sorted(keep))}
        sp = cmap.get(" ", 0)
        ids = np.array([cmap.get(c, sp) for c in raw], dtype=np.int64)
        x, y = ids[:-1], ids[1:]
        n = n_train + n_test
        idx = np.random.default_rng(11).permutation(len(x))[:n]   # random split (rules SS VII.2)
        tr, te = idx[:n_train], idx[n_train:]
        _TEXT = (x[tr], y[tr], x[te], y[te], len(cmap))
    return _TEXT


def text_stream(n=2400):
    x = text_data()[0][:n].astype(np.float64)
    x = x - x.mean()
    return x / (x.std() + 1e-9)


_AUDIO = None


def audio_data(n_train=8000, n_test=2000, T=256):
    """Synthetic tone detection: 10 classes = 10 frequencies, random phase and
    amplitude, additive noise. Raw waveform in, class out. Pointwise lenses see
    each sample independently — with random phase every position's value
    DISTRIBUTION is identical across classes, so this is the roadmap's designed
    stress test of 'is temporal rotation mandatory?'."""
    global _AUDIO
    if _AUDIO is None:
        rng = np.random.default_rng(5)
        n = n_train + n_test
        freqs = np.linspace(4, 40, 10)
        y = rng.integers(0, 10, n)
        t = np.arange(T) / T
        A = rng.uniform(0.7, 1.3, n)[:, None]
        ph = rng.uniform(0, 2 * np.pi, n)[:, None]
        X = (A * np.sin(2 * np.pi * freqs[y][:, None] * t[None, :] + ph)
             + 0.1 * rng.standard_normal((n, T))).astype(np.float32)
        _AUDIO = (X[:n_train], y[:n_train], X[n_train:], y[n_train:])
    return _AUDIO


_DOMAINS = {"mnist": mnist_data, "cifar": cifar_data, "audio": audio_data}


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


def _lens_block(tp, prog, Ftr, Fte):
    """Per-lens feature blocks, z-scored by train stats; None if degenerate."""
    import torch
    Ztr = _lens_apply_t(tp, prog, Ftr)
    sd = Ztr.std(0)
    if float(sd.mean()) < 1e-6:
        return None, None
    mu = Ztr.mean(0)
    ok = sd > 1e-6
    Ztr = torch.where(ok, (Ztr - mu) / (sd + 1e-9), torch.zeros_like(Ztr))
    Zte = _lens_apply_t(tp, prog, Fte)
    Zte = torch.where(ok, (Zte - mu) / (sd + 1e-9), torch.zeros_like(Zte))
    return Ztr, Zte


def _kridge_factory(ytr, yte, n_class, lams=(1.0, 10.0, 100.0, 1000.0)):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(ytr)
    Y = -torch.ones((n, n_class), device=dev)
    Y[torch.arange(n), torch.tensor(ytr, device=dev)] = 1.0
    yte_t = torch.tensor(yte, device=dev)
    eye = torch.eye(n, device=dev)

    def solve(Ktr, Kte, n_lens):
        best, best_pred = {"acc": 0.0, "lam": None, "n_lens": n_lens}, None
        for lam in lams:
            a = torch.linalg.solve(Ktr + lam * n_lens * eye, Y)
            pred = (Kte @ a).argmax(1)
            acc = float((pred == yte_t).float().mean())
            if acc > best["acc"]:
                best = {"acc": round(acc, 4), "lam": lam, "n_lens": n_lens}
                best_pred = pred
        best["pred"] = (best_pred if best_pred is not None else
                        torch.zeros(len(yte_t), dtype=torch.long, device=dev)).cpu().numpy()
        return best

    return solve


# ---------------------------------------------------------------------------
# image/audio baseline: pointwise bank -> kernel head
# ---------------------------------------------------------------------------

def domain_baseline(domain, n_lens=400, snapshots=(8, 32, 128, 400), verbose=True):
    import torch
    Xtr, ytr, Xte, yte = _DOMAINS[domain]()
    n_class = int(max(ytr.max(), yte.max())) + 1
    solve = _kridge_factory(ytr, yte, n_class)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    Ftr = torch.tensor(Xtr.reshape(len(Xtr), -1), device=dev)
    Fte = torch.tensor(Xte.reshape(len(Xte), -1), device=dev)

    raw = solve(Ftr @ Ftr.T, Fte @ Ftr.T, 1)
    if verbose:
        print(f"[{domain}] raw linear head:  acc {raw['acc']:.4f}  (lam {raw['lam']})", flush=True)

    n = len(ytr)
    Yc = -torch.ones((n, n_class), device=dev) / n_class
    Yc[torch.arange(n), torch.tensor(ytr, device=dev)] += 1.0
    yn = torch.norm(Yc)
    Ktr = torch.zeros((len(Ftr), len(Ftr)), device=dev)
    Kte = torch.zeros((len(Fte), len(Ftr)), device=dev)
    results, aligns, kept = [], [], []
    t0 = time.time()
    for i in range(n_lens * 2):
        Ztr, Zte = _lens_block(tp, rmap.lens_program(i), Ftr, Fte)
        if Ztr is None:
            continue
        Ktr += Ztr @ Ztr.T
        Kte += Zte @ Ztr.T
        kept.append(i)
        aligns.append((i, round(float(torch.norm(Ztr.T @ Yc) /
                                      (torch.norm(Ztr) * yn + 1e-9)), 4)))
        if len(kept) in snapshots:
            acc = solve(Ktr, Kte, len(kept))
            results.append(acc)
            if verbose:
                print(f"  L={len(kept):4d}  acc {acc['acc']:.4f}  (lam {acc['lam']}, "
                      f"{round(time.time()-t0)}s)", flush=True)
        if len(kept) >= max(snapshots):
            break
    final = results[-1]

    per_class = {str(c): round(float((final["pred"][yte == c] == c).mean()), 4)
                 for c in range(n_class)}
    scored = sorted(((s, i) for i, s in aligns), reverse=True)
    top = [{"i": i, "align": s, "prog": rmap.prog_str(rmap.lens_program(i))}
           for s, i in scored[:10]]
    out = {
        "domain": domain, "input_format": "flat vector, pointwise lenses (GAM)",
        "n_class": n_class, "train": len(ytr), "test": len(yte),
        "raw_linear_acc": raw["acc"],
        "curve": [{"n_lens": r["n_lens"], "acc": r["acc"], "lam": r["lam"]} for r in results],
        "acc": final["acc"], "per_class": per_class, "top_lenses": top,
        "majority_class_acc": round(float(np.mean(yte == np.bincount(ytr).argmax())), 4),
    }
    if verbose:
        print(f"  per-class: {per_class}", flush=True)
    return out


def domain_rotation(domain, n_lens=400, step_deg=1.0, frac=0.06, verbose=True):
    """The 360-degree rotation probe on the domain task: slices come from the
    DOMAIN's own map (signatures on that data's stream)."""
    import torch
    Xtr, ytr, Xte, yte = _DOMAINS[domain]()
    n_class = int(max(ytr.max(), yte.max())) + 1
    solve = _kridge_factory(ytr, yte, n_class)

    Sraw, _, _, _ = rmap.build_signatures(n_lens, domain)
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
            Ztr, Zte = _lens_block(tp, rmap.lens_program(int(i)), Ftr, Fte)
            if Ztr is None:
                continue
            Ktr += Ztr @ Ztr.T
            Kte += Zte @ Ztr.T
            used += 1
        return solve(Ktr, Kte, used)["acc"] if used else 0.0

    angles, accs, sizes = [], [], []
    t0 = time.time()
    for k in range(int(round(360.0 / step_deg))):
        th = np.deg2rad(k * step_deg)
        z2 = -X3[:, 0] * np.sin(th) + X3[:, 2] * np.cos(th)
        idx = np.where(np.abs(z2) <= np.quantile(np.abs(z2), frac))[0]
        sizes.append(len(idx))
        angles.append(round(k * step_deg, 2))
        accs.append(round(slice_acc(idx), 4))
        if verbose and k % 45 == 0:
            print(f"  [{domain}] {k*step_deg:5.0f} deg  acc {accs[-1]:.4f}  "
                  f"({round(time.time()-t0)}s)", flush=True)
    m = int(np.median(sizes))
    rng = np.random.default_rng(3)
    rand = [slice_acc(rng.choice(n_lens, m, replace=False)) for _ in range(5)]
    best, worst = int(np.argmax(accs)), int(np.argmin(accs))
    return {
        "domain": domain, "n_lens": n_lens, "step_deg": step_deg, "slice_size": m,
        "map_shape_axis_std": shape, "angles": angles, "acc": accs,
        "best": {"deg": angles[best], "acc": accs[best]},
        "worst": {"deg": angles[worst], "acc": accs[worst]},
        "angular_spread": round(float(max(accs) - min(accs)), 4),
        "baseline_random_mean": round(float(np.mean(rand)), 4),
        "baseline_random_std": round(float(np.std(rand)), 4),
    }


# ---------------------------------------------------------------------------
# text baseline: scalar char id -> primal ridge (bigram-table ceiling hunt)
# ---------------------------------------------------------------------------

def _primal_ridge_acc(Ftr, ytr, Fte, yte, n_class, lams=(1e-2, 1e-1, 1.0, 10.0)):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    Ftr = np.hstack([(Ftr - mu) / sd, np.ones((len(Ftr), 1))])
    Fte = np.hstack([(Fte - mu) / sd, np.ones((len(Fte), 1))])
    Y = -np.ones((len(ytr), n_class), np.float64)
    Y[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr
    best = 0.0
    for lam in lams:
        W = np.linalg.solve(A + lam * np.eye(A.shape[0]), Ftr.T @ Y)
        best = max(best, float(((Fte @ W).argmax(1) == yte).mean()))
    return round(best, 4)


def text_baseline(n_lens=400, verbose=True):
    xtr, ytr, xte, yte, V = text_data()
    maj = round(float((yte == np.bincount(ytr).argmax()).mean()), 4)
    # bigram table ceiling (the thing we are NOT allowed to ship, only measure)
    big = np.zeros((V, V))
    np.add.at(big, (xtr, ytr), 1)
    bigram = round(float((big.argmax(1)[xte] == yte).mean()), 4)
    raw = _primal_ridge_acc(xtr[:, None].astype(np.float64), ytr,
                            xte[:, None].astype(np.float64), yte, V)
    mu, sd = xtr.mean(), xtr.std()
    ztr, zte = (xtr - mu) / sd, (xte - mu) / sd
    cols_tr, cols_te, kept = [], [], []
    for i in range(n_lens * 2):
        p = rmap.lens_program(i)
        a, b = rmap.lens_apply(p, ztr), rmap.lens_apply(p, zte)
        if a.std() > 1e-9:
            cols_tr.append(a); cols_te.append(b); kept.append(i)
        if len(kept) >= n_lens:
            break
    accs = {}
    for L in (8, 32, 128, n_lens):
        accs[L] = _primal_ridge_acc(np.stack(cols_tr[:L], 1), ytr,
                                    np.stack(cols_te[:L], 1), yte, V)
        if verbose:
            print(f"  [text] L={L:4d}  acc {accs[L]:.4f}", flush=True)
    out = {
        "domain": "text", "input_format": "scalar char id, pointwise lenses",
        "task": "next-char (50-char vocab, wiki corpus)",
        "train": len(ytr), "test": len(yte),
        "majority_class_acc": maj, "bigram_table_ceiling": bigram,
        "raw_linear_acc": raw,
        "curve": [{"n_lens": L, "acc": a} for L, a in accs.items()],
        "acc": accs[n_lens],
        "gap_to_bigram": round(bigram - accs[n_lens], 4),
    }
    if verbose:
        print(f"  [text] majority {maj}, bigram ceiling {bigram}, raw-linear {raw}", flush=True)
    return out


def text_rotation(n_lens=400, step_deg=1.0, frac=0.06):
    xtr, ytr, xte, yte, V = text_data()
    Sraw, _, _, _ = rmap.build_signatures(n_lens, "text")
    S = (Sraw - Sraw.mean(0)) / (Sraw.std(0) + 1e-9)
    X3 = rmap._mds(S, 3)
    mu, sd = xtr.mean(), xtr.std()
    ztr, zte = (xtr - mu) / sd, (xte - mu) / sd
    bank_tr, bank_te = [], []
    for i in range(n_lens):
        p = rmap.lens_program(i)
        bank_tr.append(rmap.lens_apply(p, ztr)); bank_te.append(rmap.lens_apply(p, zte))
    bank_tr, bank_te = np.stack(bank_tr, 1), np.stack(bank_te, 1)
    ok = bank_tr.std(0) > 1e-9
    bank_tr, bank_te, X3 = bank_tr[:, ok], bank_te[:, ok], X3[ok]
    angles, accs, sizes = [], [], []
    for k in range(int(round(360.0 / step_deg))):
        th = np.deg2rad(k * step_deg)
        z2 = -X3[:, 0] * np.sin(th) + X3[:, 2] * np.cos(th)
        idx = np.where(np.abs(z2) <= np.quantile(np.abs(z2), frac))[0]
        sizes.append(len(idx))
        angles.append(round(k * step_deg, 2))
        accs.append(_primal_ridge_acc(bank_tr[:, idx], ytr, bank_te[:, idx], yte, V))
    best, worst = int(np.argmax(accs)), int(np.argmin(accs))
    return {"domain": "text", "slice_size": int(np.median(sizes)),
            "map_shape_axis_std": [round(float(v), 2) for v in X3.std(0)],
            "angles": angles, "acc": accs,
            "best": {"deg": angles[best], "acc": accs[best]},
            "worst": {"deg": angles[worst], "acc": accs[worst]},
            "angular_spread": round(float(max(accs) - min(accs)), 4)}


# ---------------------------------------------------------------------------
# roadmap runner
# ---------------------------------------------------------------------------

def run_domain(domain, out_path=None):
    t0 = time.time()
    if domain == "text":
        probe, rot = text_baseline(), text_rotation()
    else:
        probe, rot = domain_baseline(domain), domain_rotation(domain)
    out = {"format": "radial-baseline-export", "domain": domain,
           "probe": probe, "rotation_probe": rot, "seconds": round(time.time() - t0)}
    path = out_path or os.path.join(_HERE, "radial_data", f"baseline_{domain}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{domain}] written {path} ({round(time.time()-t0)}s)", flush=True)
    return out


def prebaseline_fixes():
    """The roadmap's pre-baseline experiments on loops: (a) Z-axis expansion
    (whitened MDS axes) — does it shrink rotation dead zones?; (b) rotation
    axis lock — how do X/Y/Z spins differ?"""
    out = {}
    for axis in ("y", "x", "z"):
        r = rmap.rotation_probe(axis=axis)
        out[f"axis_{axis}"] = {k: r[k] for k in
                               ("best", "worst", "angular_spread",
                                "baseline_random_mean")}
        print(f"  [fixes] axis {axis}: best {r['best']}, worst {r['worst']}, "
              f"spread {r['angular_spread']}", flush=True)
    r = rmap.rotation_probe(axis="y", whiten=True)
    out["whitened_y"] = {k: r[k] for k in
                         ("best", "worst", "angular_spread", "baseline_random_mean")}
    print(f"  [fixes] whitened y: best {r['best']}, worst {r['worst']}, "
          f"spread {r['angular_spread']}", flush=True)
    path = os.path.join(_HERE, "radial_data", "prebaseline_fixes.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


def run_all():
    print("=== pre-baseline fixes (loops) ===", flush=True)
    prebaseline_fixes()
    for domain in ("cifar", "text", "audio"):
        print(f"\n=== {domain} baseline ===", flush=True)
        run_domain(domain)
    print("\nroadmap sweep complete", flush=True)


if __name__ == "__main__":
    import sys
    if "all" in sys.argv:
        run_all()
    elif "fixes" in sys.argv:
        prebaseline_fixes()
    elif len(sys.argv) > 1 and sys.argv[1] in ("mnist", "cifar", "text", "audio"):
        run_domain(sys.argv[1])
    else:
        print("usage: radial_baseline.py all | fixes | mnist | cifar | text | audio")
