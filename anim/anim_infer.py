"""anim_infer.py — run the saved temporal-radial checkpoints on demand.

Two checkpoints, same sequences, opposite labels:
  task="path"  — which animation (motion answers, shape is the decoy)
  task="shape" — which shape (shape answers, motion is the decoy)

Loads the genome checkpoint (radial_data/anim_model[_shape].json) and
replays the exact feature pipeline from radial_anim.run() with the genomes
FIXED (no evolution). The ridge head is refit locally (closed-form, the
only "training"), so patch-PCA sign conventions can never mismatch between
machines. Predictions are served for HELD-OUT test sequences the model
never trained on; the locally measured test accuracy is reported so the
page's number is honest for this machine.

Build is one-time per process per task (background thread, ~20 s on the
4080); after that a classify call is a table lookup. No gradients anywhere.
"""
import os as _os, sys as _sys                     # repo-root shim
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import genreg_paths                               # noqa: F401
import base64
import json
import os
import threading

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCK = threading.Lock()
TASKS = ("path", "shape")
_STATE = {t: {"status": "idle", "error": None} for t in TASKS}
_M = {t: {} for t in TASKS}                   # built artifacts per task


def count_params(obj):
    """Numeric leaves in a genome structure = evolved parameters."""
    if isinstance(obj, (bool, int, float)):
        return 1
    if isinstance(obj, (list, tuple)):
        return sum(count_params(v) for v in obj)
    if isinstance(obj, dict):
        return sum(count_params(v) for v in obj.values())
    return 0                                  # strings = structure, not params


def _build(task):
    import torch

    from radial_evo import _tprims
    from radial_evo2 import Env
    import radial_stack as rk
    from radial_anim import T, PATH_NAMES, SHAPE_NAMES

    st, M = _STATE[task], _M[task]
    suffix = "" if task == "path" else "_shape"
    names = PATH_NAMES if task == "path" else SHAPE_NAMES

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)

    with open(os.path.join(_HERE, "radial_data",
                           f"anim_model{suffix}.json")) as f:
        ckpt = json.load(f)
    rk.GRID = int(ckpt["grid"])
    G = rk.GRID

    z = np.load(os.path.join(_HERE, "radial_data", "anim_seq.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr" + suffix], z["yte" + suffix]
    Ntr, Nte = len(ytr), len(yte)
    env = Env(torch, dev, Xtr.reshape(Ntr * T, 32, 32, 3),
              Xte.reshape(Nte * T, 32, 32, 3), max_cached=6)

    all_tr, all_te = [], []
    prev_tr = prev_te = None
    for si, genomes in enumerate(ckpt["spaces"]):
        if si == 0:
            f_tr = lambda g: rk.feature_r0(torch, tp, env, g).view(Ntr, T).mean(1)
            f_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True).view(Nte, T).mean(1)
            g_tr = lambda g: rk.feature_r0(torch, tp, env, g,
                                           want_grid=True).view(Ntr, T, G, G)
            g_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True,
                                           want_grid=True).view(Nte, T, G, G)
        else:
            f_tr = lambda g: rk.feature_grid_g(torch, tp, prev_tr, g)
            f_te = lambda g: rk.feature_grid_g(torch, tp, prev_te, g)
            g_tr = lambda g: rk.feature_grid_g(torch, tp, prev_tr, g, want_grid=True)
            g_te = lambda g: rk.feature_grid_g(torch, tp, prev_te, g, want_grid=True)
        all_tr.extend(f_tr(g) for g in genomes)
        all_te.extend(f_te(g) for g in genomes)
        st["progress"] = f"space {si}: {len(genomes)} genomes replayed"
        if si + 1 < len(ckpt["spaces"]):
            if si == 0:      # temporal hand-off: (genome x frame) channels
                prev_tr = torch.cat([g_tr(g) for g in genomes], 1).half()
                prev_te = torch.cat([g_te(g) for g in genomes], 1).half()
            else:
                prev_tr = torch.stack([g_tr(g) for g in genomes], 1).half()
                prev_te = torch.stack([g_te(g) for g in genomes], 1).half()
    del prev_tr, prev_te

    Ftr = torch.stack(all_tr, 1).float()
    Fte = torch.stack(all_te, 1).float()
    ytr_t = torch.tensor(ytr, device=dev)
    Y = -torch.ones((Ntr, 10), device=dev)
    Y[torch.arange(Ntr), ytr_t] = 1.0

    def _fit(Xf, Yf, lam):
        n, d = Xf.shape
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
        A = torch.hstack([(Xf - mu) / sd, torch.ones(n, 1, device=dev)])
        W = torch.linalg.solve(A.T @ A + lam * torch.eye(d + 1, device=dev),
                               A.T @ Yf)
        return mu, sd, W

    n_fit = int(Ntr * 0.8)                     # pick lambda on a val split
    best_lam, best_acc = 3.0, -1.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        mu, sd, W = _fit(Ftr[:n_fit], Y[:n_fit], lam)
        s = torch.hstack([(Ftr[n_fit:] - mu) / sd,
                          torch.ones(Ntr - n_fit, 1, device=dev)]) @ W
        acc = float((s.argmax(1) == ytr_t[n_fit:]).float().mean())
        if acc > best_acc:
            best_lam, best_acc = lam, acc
    mu, sd, W = _fit(Ftr, Y, best_lam)         # refit on full train
    s = torch.hstack([(Fte - mu) / sd, torch.ones(Nte, 1, device=dev)]) @ W
    preds = s.argmax(1).cpu().numpy()

    M.update(
        preds=preds, yte=yte, names=names, frames=T,
        Xte_u8=z["Xte"][..., 0],               # (Nte, T, 32, 32) grayscale
        test_acc=float((preds == yte).mean()),
        per_class={names[c]: float((preds[yte == c] == c).mean())
                   for c in range(len(names))},
        lam=best_lam,
        n_genomes=sum(len(sp) for sp in ckpt["spaces"]),
        genome_params=sum(count_params(g) for sp in ckpt["spaces"] for g in sp),
        head_params=int(W.numel()),
    )
    M["total_params"] = M["genome_params"] + M["head_params"]
    st["status"] = "ready"


def _build_safe(task):
    try:
        _build(task)
    except Exception as exc:                   # surfaced via the endpoint
        _STATE[task].update(status="error", error=str(exc))


def status(task="path"):
    return dict(_STATE[task])


def classify(n=12, seed=None, task="path"):
    """n random held-out test sequences with the checkpoint's predictions.
    Kicks off the one-time build if needed; returns building status until
    it is ready."""
    if task not in TASKS:
        return {"error": f"unknown task {task!r} (have {TASKS})"}
    st, M = _STATE[task], _M[task]
    with _LOCK:
        if st["status"] in ("idle", "error"):
            st.update(status="building", error=None, progress="starting")
            threading.Thread(target=_build_safe, args=(task,), daemon=True,
                             name=f"anim-infer-build-{task}").start()
    if st["status"] != "ready":
        return {"building": st["status"] == "building",
                "status": st["status"],
                "progress": st.get("progress"), "error": st["error"]}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(M["yte"]), size=min(int(n), 32), replace=False)
    items = []
    for i in idx:
        i = int(i)
        items.append({
            "true": M["names"][int(M["yte"][i])],
            "pred": M["names"][int(M["preds"][i])],
            "correct": bool(M["preds"][i] == M["yte"][i]),
            "size": 32, "frames": M["frames"],
            "data": base64.b64encode(M["Xte_u8"][i].tobytes()).decode("ascii"),
        })
    return {"items": items, "task": task, "test_acc": M["test_acc"],
            "per_class": M["per_class"], "n_test": len(M["yte"]),
            "n_genomes": M["n_genomes"], "lam": M["lam"],
            "genome_params": M["genome_params"],
            "head_params": M["head_params"],
            "total_params": M["total_params"]}
