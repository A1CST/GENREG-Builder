"""lm_word_infer.py — interactive autocomplete from the latest word-level
radial checkpoint.

Loads radial_data/lm_model_word.json (genome structure only) and replays
the exact pipeline from radial_lm_word.run() with everything FROZEN:
banks rebuilt, per-space standardization stats recomputed from the
training windows, closed-form head refit (fp32 gram / fp64 solve), decode
scale recalibrated on the val split. One background build per process
(~2-3 min); after that each autocomplete request is ~a second.

Decoding matches the run's sample generator: calibrated sharpness, top-5
sampling, repetition penalty. No gradients anywhere.
"""
import json
import os
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCK = threading.Lock()
_STATE = {"status": "idle", "error": None, "progress": None}
_M = {}


def count_params(obj):
    if isinstance(obj, (bool, int, float)):
        return 1
    if isinstance(obj, (list, tuple)):
        return sum(count_params(v) for v in obj)
    if isinstance(obj, dict):
        return sum(count_params(v) for v in obj.values())
    return 0


def _build():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    import radial_lm_word as rw
    from radial_evo import _tprims
    import radial_stack as rk

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()

    with open(os.path.join(_HERE, "radial_data", "lm_model_word.json")) as f:
        ckpt = json.load(f)
    rw.W = int(ckpt["context_words"])
    rw.V = int(ckpt["vocab"])
    W, V, D = rw.W, rw.V, rw.D

    vocab, feat, _ = rw._load_embed()
    feat_t = torch.tensor(feat, device=dev)
    w2i = {w: i for i, w in enumerate(vocab)}
    z = np.load(os.path.join(_HERE, "radial_data", "lm_word.npz"),
                allow_pickle=True)
    assert z["ctx_tr"].shape[1] == W and len(z["targets"]) == V, \
        "lm_word.npz does not match the checkpoint (regenerate the data)"
    ctx_tr, ytr = z["ctx_tr"], z["ytr"]
    targets = [str(w) for w in z["targets"]]
    tgt_i = {w: k for k, w in enumerate(targets)}
    tv = {w2i[w]: k for k, w in enumerate(targets) if w in w2i}
    Ntr = len(ytr)
    mu, sd = feat_t.mean(0), feat_t.std(0) + 1e-6

    _STATE["progress"] = "loading continuation tables"
    import pickle
    with open(os.path.join(_HERE, "radial_data", "lm_cont_tables.pkl"),
              "rb") as f:
        uni_c, bi_c, tri_c = pickle.load(f)

    def _cont_vec(dist):
        v = np.zeros(D, np.float32)
        tot = 0
        for w, c in dist.items():
            j = w2i.get(w)
            if j is not None:
                v += c * feat[j]
                tot += c
        return v / tot if tot else v

    def _cont_prob(dist):
        v = np.zeros(V, np.float32)
        for w, c in dist.items():
            k = tgt_i.get(w)
            if k is not None:
                v[k] = c
        s = v.sum()
        return v / s if s else v

    _uni_vec, _uni_prob = _cont_vec(uni_c), _cont_prob(uni_c)
    N_CONT = 2 * D + V

    def _cont_raw(ctx):
        N = len(ctx)
        out = np.zeros((N, N_CONT), np.float32)
        for i in range(N):
            j1, j2 = int(ctx[i, W - 2]), int(ctx[i, W - 1])
            w1 = vocab[j1] if j1 >= 0 else None
            w2 = vocab[j2] if j2 >= 0 else None
            key = (w1, w2)
            if key in tri_c:
                out[i, :D] = _cont_vec(tri_c[key])
                out[i, 2 * D:] = _cont_prob(tri_c[key])
            elif w2 in bi_c:
                out[i, :D] = _cont_vec(bi_c[w2])
                out[i, 2 * D:] = _cont_prob(bi_c[w2])
            else:
                out[i, :D] = _uni_vec
                out[i, 2 * D:] = _uni_prob
            out[i, D:2 * D] = _cont_vec(bi_c[w2]) if w2 in bi_c else _uni_vec
        return torch.tensor(out, device=dev)

    _STATE["progress"] = "building banks"
    cont_tr = _cont_raw(ctx_tr)
    cmu, csd = cont_tr.mean(0), cont_tr.std(0) + 1e-6

    def _identity(ctx, slot):
        N = len(ctx)
        M = torch.zeros((N, V), device=dev)
        rows, cols = [], []
        for i in range(N):
            k = tv.get(int(ctx[i, slot]), -1)
            if k >= 0:
                rows.append(i); cols.append(k)
        M[torch.tensor(rows, device=dev, dtype=torch.long),
          torch.tensor(cols, device=dev, dtype=torch.long)] = 1.0
        return M

    def _embed_bank(ctx):
        idx = torch.tensor(np.maximum(ctx.astype(np.int64), 0), device=dev)
        mask = torch.tensor((ctx >= 0).astype(np.float32), device=dev)
        cols = []
        for f in range(W):
            v = feat_t[idx[:, f]] * mask[:, f:f + 1]
            cols.append((v - mu) / sd)
        return torch.cat(cols, 1)

    def _bank0(ctx, cont=None):
        c = cont if cont is not None else _cont_raw(ctx)
        return torch.cat([_embed_bank(ctx), _identity(ctx, W - 2),
                          _identity(ctx, W - 1),
                          ((c - cmu) / csd).clamp(-8, 8)], 1)

    def _rows(ctx):
        idx = torch.tensor(np.maximum(ctx.astype(np.int64), 0), device=dev)
        mask = torch.tensor((ctx >= 0).astype(np.float32), device=dev)
        r = feat_t[idx.reshape(-1)] * mask.reshape(-1, 1)
        return (r - mu) / sd

    def _san(v):
        return torch.nan_to_num(v, nan=0.0, posinf=0.0,
                                neginf=0.0).clamp(-1e6, 1e6)

    B0_tr = _bank0(ctx_tr, cont_tr)

    _STATE["progress"] = "replaying genome spaces"
    all_cols = [B0_tr[:, j] for j in range(B0_tr.shape[1])]
    space_stats, handoff_stats = [], None
    bank = None
    for si, sp in enumerate(ckpt["spaces"]):
        if si == 0:
            r_tr = _rows(ctx_tr)
            slot_vals = [_san(rk.feature_vec(torch, tp, r_tr, g)
                              ).view(Ntr, W) for g in sp]
            f = torch.cat([v.mean(1, keepdim=True) for v in slot_vals], 1)
        else:
            f = torch.stack([_san(rk.feature_vec(torch, tp, bank, g))
                             for g in sp], 1)
        zmu, zsd = f.mean(0), f.std(0) + 1e-6
        space_stats.append((zmu, zsd))
        f = ((f - zmu) / zsd).clamp(-8, 8)
        all_cols.extend(f[:, j] for j in range(f.shape[1]))
        if si == 0:
            s1 = torch.cat(slot_vals, 1)
            pmu, psd = s1.mean(0), s1.std(0) + 1e-6
            handoff_stats = (pmu, psd)
            bank = torch.cat([((s1 - pmu) / psd).clamp(-8, 8),
                              _identity(ctx_tr, W - 2),
                              _identity(ctx_tr, W - 1),
                              B0_tr[:, -N_CONT:]], 1)
        else:
            bank = torch.cat([bank, f], 1)
    del bank

    _STATE["progress"] = "fitting the head"
    Ftr = torch.stack(all_cols, 1)
    Y = -torch.ones((Ntr, V), device=dev)
    Y[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0
    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)

    def _fit(Xf, Yf, lam):
        n, d = Xf.shape
        hm, hs = Xf.mean(0), Xf.std(0) + 1e-6
        A = torch.hstack([(Xf - hm) / hs, torch.ones(n, 1, device=dev)])
        G = (A.T @ A).double() + lam * torch.eye(d + 1, device=dev,
                                                 dtype=torch.float64)
        Wm = torch.linalg.solve(G, (A.T @ Yf).double()).float()
        return hm, hs, Wm

    best = (3.0, -1.0)
    for lam in (3.0, 10.0, 30.0):
        hm, hs, Wm = _fit(Ftr[:n_fit], Y[:n_fit], lam)
        s = torch.hstack([(Ftr[n_fit:] - hm) / hs,
                          torch.ones(Ntr - n_fit, 1, device=dev)]) @ Wm
        a = float((s.argmax(1) == yv).float().mean())
        if a > best[1]:
            best = (lam, a)
    hm, hs, Wm = _fit(Ftr, Y, best[0])
    val_acc = best[1]

    s_val = torch.hstack([(Ftr[n_fit:] - hm) / hs,
                          torch.ones(Ntr - n_fit, 1, device=dev)]) @ Wm
    s_cal, best_nll = 1.0, 1e9
    for sc in (1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 35.0):
        nll = float(-torch.log_softmax(s_val * sc, 1)
                    [torch.arange(Ntr - n_fit), yv].mean())
        if nll < best_nll:
            s_cal, best_nll = sc, nll
    del Ftr, s_val, Y

    genome_params = sum(count_params(g) for sp in ckpt["spaces"] for g in sp)
    head_params = int(Wm.numel())

    def _step_logits(win_ids):
        ctx1 = np.array([win_ids], np.int32)
        B01 = _bank0(ctx1)
        cols = [B01[0]]
        if ckpt["spaces"]:
            r1 = _rows(ctx1)
            slot_vals = [_san(rk.feature_vec(torch, tp, r1, g)).view(1, W)
                         for g in ckpt["spaces"][0]]
            f0 = torch.cat([v.mean(1, keepdim=True) for v in slot_vals], 1)
            zmu, zsd = space_stats[0]
            cols.append((((f0 - zmu) / zsd).clamp(-8, 8))[0])
            pmu, psd = handoff_stats
            b = torch.cat([(torch.cat(slot_vals, 1) - pmu).div(psd).clamp(-8, 8),
                           _identity(ctx1, W - 2), _identity(ctx1, W - 1),
                           B01[:, -N_CONT:]], 1)
            for sp, (zmu, zsd) in zip(ckpt["spaces"][1:], space_stats[1:]):
                f = torch.stack([_san(rk.feature_vec(torch, tp, b, g))
                                 for g in sp], 1)
                f = ((f - zmu) / zsd).clamp(-8, 8)
                cols.append(f[0])
                b = torch.cat([b, f], 1)
        F1 = torch.cat(cols).view(1, -1)
        return (torch.hstack([(F1 - hm) / hs,
                              torch.ones(1, 1, device=dev)]) @ Wm)[0]

    _M.update(step=_step_logits, torch=torch, w2i=w2i, targets=targets,
              tgt_i=tgt_i, W=W, V=V, s_cal=s_cal, val_acc=val_acc,
              genome_params=genome_params, head_params=head_params,
              total_params=genome_params + head_params,
              n_genomes=sum(len(sp) for sp in ckpt["spaces"]),
              build_seconds=round(time.time() - t0))
    _STATE["status"] = "ready"


def _build_safe():
    try:
        _build()
    except Exception as exc:
        _STATE.update(status="error", error=str(exc))


def complete(prompt, n_words=24, temp=1.0, seed=None):
    """Autocomplete `prompt` with the latest word checkpoint."""
    with _LOCK:
        if _STATE["status"] in ("idle", "error"):
            _STATE.update(status="building", error=None, progress="starting")
            threading.Thread(target=_build_safe, daemon=True,
                             name="lm-word-infer-build").start()
    if _STATE["status"] != "ready":
        return {"building": _STATE["status"] == "building",
                "status": _STATE["status"],
                "progress": _STATE.get("progress"), "error": _STATE["error"]}
    rng = np.random.default_rng(seed)
    words = [w for w in prompt.lower().split() if w][:64]
    if not words:
        return {"error": "empty prompt"}
    W, V = _M["W"], _M["V"]
    win = [_M["w2i"].get(w, -1) for w in words][-W:]
    while len(win) < W:
        win.insert(0, -1)
    out_words = []
    for _ in range(max(1, min(60, int(n_words)))):
        lg = _M["step"](win).detach().cpu().numpy().astype(np.float64)
        lg = lg * _M["s_cal"] / max(float(temp), 1e-3)
        for wd in out_words[-8:]:
            kr = _M["tgt_i"].get(wd)
            if kr is not None:
                lg[kr] -= 2.0
        top = np.argsort(lg)[-5:]
        z = lg[top] - lg[top].max()
        p = np.exp(z)
        p /= p.sum()
        k = int(rng.choice(top, p=p))
        wd = _M["targets"][k]
        out_words.append(wd)
        win = win[1:] + [_M["w2i"].get(wd, -1)]
    return {"prompt": " ".join(words), "completion": " ".join(out_words),
            "params": {"genome": _M["genome_params"],
                       "head": _M["head_params"],
                       "total": _M["total_params"],
                       "n_genomes": _M["n_genomes"]},
            "vocab": V, "context": W, "val_acc": round(_M["val_acc"], 4),
            "decode_scale": _M["s_cal"]}
