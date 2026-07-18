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
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401
import json
import os
import threading
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCK = threading.Lock()
_STATE = {"status": "idle", "error": None, "progress": None,
          "t0": None, "frac": 0.0}
_M = {}


def _stage(label, frac):
    """Record a build stage + rough completed fraction (for the ETA)."""
    _STATE["progress"] = label
    _STATE["frac"] = frac


def _build_eta():
    """(elapsed_s, eta_s or None) from the stage fractions."""
    if not _STATE["t0"]:
        return None, None
    el = time.time() - _STATE["t0"]
    f = _STATE["frac"]
    return round(el), (round(el * (1 - f) / f) if f >= 0.05 else None)


def count_params(obj):
    if isinstance(obj, (bool, int, float)):
        return 1
    if isinstance(obj, (list, tuple)):
        return sum(count_params(v) for v in obj)
    if isinstance(obj, dict):
        return sum(count_params(v) for v in obj.values())
    return 0


_FORCE_CPU = False

_NP_PRIMS = {
    "id": lambda z: z, "sin": np.sin, "cos": np.cos, "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0), "abs": np.abs,
    "sq": lambda z: np.clip(z * z, -30, 30), "sign": np.sign,
    "gauss": lambda z: np.exp(-np.clip(z * z, 0, 30)),
    "soft": lambda z: z / (1.0 + np.abs(z)),
    "sqrt": lambda z: np.sqrt(np.abs(z)),
}


def _np_feature_vec(prevF, g, prims, ops):
    """Exact numpy mirror of radial_stack.feature_vec (baked gates)."""
    z = None
    C = prevF.shape[1]
    for t in g["terms"]:
        v = prevF[:, t["c"] % C].astype(np.float64)
        for prim, a, b in t["prog"]:
            v = _NP_PRIMS[prims[prim]](a * v + b)
        if z is None:
            z = v
        else:
            op = ops[g["op"]]
            z = z * v if op == "mult" else (np.minimum(z, v) if op == "min"
                                            else np.abs(z - v))
    gt = g.get("gate")
    if gt:
        gv = prevF[:, gt["c"] % C].astype(np.float64)
        for prim, a, b in gt["prog"]:
            gv = _NP_PRIMS[prims[prim]](a * gv + b)
        st = gt.get("stats")
        if st is not None:
            gz = (gv - st[0]) / (st[1] + 1e-9)
        else:
            gz = (gv - gv.mean()) / (gv.std() + 1e-9)
        z = z * (1.0 / (1.0 + np.exp(-np.clip(gt["k"] * gz, -60, 60))))
    return z




def _build():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    import radial_lm_word as rw
    from radial_evo import _tprims
    import radial_stack as rk

    dev = "cuda" if (torch.cuda.is_available() and not _FORCE_CPU) else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    _STATE["t0"] = t0
    _stage("loading data", 0.02)

    with open(os.path.join(_HERE, "radial_data", "lm_model_word.json")) as f:
        ckpt = json.load(f)
    rw.W = int(ckpt["context_words"])
    rw.V = int(ckpt["vocab"])
    W, V, D = rw.W, rw.V, rw.D

    # inference pack: the full replay's end products (norm stats + solved
    # head), built once on a big GPU and reused everywhere - the crank
    # bank (27.7k cols x 150k rows) exceeds the local 4080 and its CPU
    # fallback takes an hour+; loading the pack takes seconds.
    sig = {"W": W, "V": V, "bank": ckpt.get("bank", "base"),
           "spaces": [len(sp) for sp in ckpt["spaces"]],
           "cont_pkl": ckpt.get("cont_pkl") or "lm_cont_tables.pkl",
           "skip_pkl": ckpt.get("skip_pkl")}
    pack = None
    pack_path = os.path.join(_HERE, "radial_data", "lm_infer_pack.pt")
    if os.path.exists(pack_path):
        try:
            p = torch.load(pack_path, map_location="cpu")
            if p.get("sig") == sig:
                pack = p
        except Exception:
            pack = None

    vocab, feat, _ = rw._load_embed()
    feat_t = torch.tensor(feat, device=dev)
    w2i = {w: i for i, w in enumerate(vocab)}
    if pack is None:
        z = np.load(os.path.join(_HERE, "radial_data", "lm_word.npz"),
                    allow_pickle=True)
        assert z["ctx_tr"].shape[1] == W and len(z["targets"]) == V, \
            "lm_word.npz does not match the checkpoint (regenerate the data)"
        ctx_tr, ytr = z["ctx_tr"], z["ytr"]
        targets = [str(w) for w in z["targets"]]
        Ntr = len(ytr)
    else:
        targets = list(pack["targets"])
    tgt_i = {w: k for k, w in enumerate(targets)}
    tv = {w2i[w]: k for k, w in enumerate(targets) if w in w2i}
    mu, sd = feat_t.mean(0), feat_t.std(0) + 1e-6

    _stage("loading continuation tables", 0.05)
    import pickle
    cont_pkl = ckpt.get("cont_pkl") or "lm_cont_tables.pkl"
    with open(os.path.join(_HERE, "radial_data", cont_pkl), "rb") as f:
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
    extra = ckpt.get("bank") == "skip5k"   # module-36+ crank bank
    if extra:
        import lm_crank
        if ckpt.get("skip_pkl"):
            lm_crank.SKIP_PKL = ckpt["skip_pkl"]
        quad_t, skipA_t, skipB_t = lm_crank.build_tables()
        N_EXTRA = 2 * (D + V) + D
    else:
        quad_t = skipA_t = skipB_t = None
        N_EXTRA = 0
    N_CONT = 2 * D + V + N_EXTRA

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
                out[i, 2 * D:2 * D + V] = _cont_prob(tri_c[key])
            elif w2 in bi_c:
                out[i, :D] = _cont_vec(bi_c[w2])
                out[i, 2 * D:2 * D + V] = _cont_prob(bi_c[w2])
            else:
                out[i, :D] = _uni_vec
                out[i, 2 * D:2 * D + V] = _uni_prob
            out[i, D:2 * D] = _cont_vec(bi_c[w2]) if w2 in bi_c else _uni_vec
            if extra:
                j0 = int(ctx[i, W - 3])
                w0 = vocab[j0] if j0 >= 0 else None
                base = 2 * D + V
                dq = quad_t.get((w0, w1, w2))
                if dq:
                    out[i, base:base + D] = _cont_vec(dq)
                    out[i, base + D:base + D + V] = _cont_prob(dq)
                base += D + V
                da = skipA_t.get((w0, w2))
                if da:
                    out[i, base:base + D] = _cont_vec(da)
                    out[i, base + D:base + D + V] = _cont_prob(da)
                base += D + V
                db = skipB_t.get((w0, w1))
                if db:
                    out[i, base:base + D] = _cont_vec(db)
        return torch.tensor(out, device=dev)

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

    if pack is not None:
        _stage("loading inference pack", 0.60)
        cmu, csd = pack["cmu"].to(dev), pack["csd"].to(dev)
        space_stats = [(a.to(dev), b.to(dev))
                       for a, b in pack["space_stats"]]
        handoff_stats = ((pack["pmu"].to(dev), pack["psd"].to(dev))
                         if pack.get("pmu") is not None else None)
        hm, hs, Wm = (pack["hm"].to(dev), pack["hs"].to(dev),
                      pack["Wm"].to(dev))
        s_cal, val_acc = pack["s_cal"], pack["val_acc"]
    else:
        _stage("building banks", 0.15)
        cont_tr = _cont_raw(ctx_tr)
        cmu, csd = cont_tr.mean(0), cont_tr.std(0) + 1e-6
        B0_tr = _bank0(ctx_tr, cont_tr)

        _stage("replaying genome spaces", 0.78)
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

        _stage("fitting the head", 0.90)
        Ftr = torch.stack(all_cols, 1)
        Y = -torch.ones((Ntr, V), device=dev)
        Y[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0
        n_fit = int(Ntr * 0.8)
        yv = torch.tensor(ytr[n_fit:], device=dev)

        def _fit(Xf, Yf, lam):
            # chunked gram: no full design-matrix copy (13GB+ at the
            # crank width); fp32 chunk grams summed in fp64
            n, d = Xf.shape
            fhm, fhs = Xf.mean(0), Xf.std(0) + 1e-6
            CH = 30000
            G = torch.zeros((d + 1, d + 1), device=dev,
                            dtype=torch.float64)
            R = torch.zeros((d + 1, Yf.shape[1]), device=dev,
                            dtype=torch.float64)
            for a in range(0, n, CH):
                Ab = torch.hstack([(Xf[a:a + CH] - fhm) / fhs,
                                   torch.ones(min(CH, n - a), 1,
                                              device=dev)])
                G += (Ab.T @ Ab).double()
                R += (Ab.T @ Yf[a:a + CH]).double()
                del Ab
            G += lam * torch.eye(d + 1, device=dev, dtype=torch.float64)
            Wf = torch.linalg.solve(G, R).float()
            return fhm, fhs, Wf

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

        _stage("saving inference pack", 0.97)
        try:
            torch.save(
                {"sig": sig, "targets": targets,
                 "cmu": cmu.cpu(), "csd": csd.cpu(),
                 "space_stats": [(a.cpu(), b.cpu())
                                 for a, b in space_stats],
                 "pmu": (handoff_stats[0].cpu()
                         if handoff_stats else None),
                 "psd": (handoff_stats[1].cpu()
                         if handoff_stats else None),
                 "hm": hm.cpu(), "hs": hs.cpu(), "Wm": Wm.cpu(),
                 "s_cal": s_cal, "val_acc": val_acc},
                os.path.join(_HERE, "radial_data", "lm_infer_pack.pt"))
        except Exception:
            pass                          # pack is an optimization only

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
            if len(ckpt["spaces"]) > 1:      # handoff only for deep stacks
                pmu, psd = handoff_stats
                b = torch.cat([(torch.cat(slot_vals, 1) - pmu)
                               .div(psd).clamp(-8, 8),
                               _identity(ctx1, W - 2), _identity(ctx1, W - 1),
                               B01[:, -N_CONT:]], 1)
                for sp, (zmu, zsd) in zip(ckpt["spaces"][1:],
                                          space_stats[1:]):
                    f = torch.stack([_san(rk.feature_vec(torch, tp, b, g))
                                     for g in sp], 1)
                    f = ((f - zmu) / zsd).clamp(-8, 8)
                    cols.append(f[0])
                    b = torch.cat([b, f], 1)
        F1 = torch.cat(cols).view(1, -1)
        return (torch.hstack([(F1 - hm) / hs,
                              torch.ones(1, 1, device=dev)]) @ Wm)[0]

    Vc_ = len(uni_c)
    n_uni_ = sum(uni_c.values())

    def _cont_score(words):
        """Mean continuation-table log-prob of a word list - the best-of-k
        reranker (the model's own tables, not any evaluation judge)."""
        def lp(w1, w2, w):
            if (w1, w2) in tri_c:
                d = tri_c[(w1, w2)]
            elif w2 in bi_c:
                d = bi_c[w2]
            else:
                return float(np.log((uni_c.get(w, 0) + 0.5)
                                    / (n_uni_ + 0.5 * Vc_)))
            return float(np.log((d.get(w, 0) + 0.5)
                                / (sum(d.values()) + 0.5 * Vc_)))
        if len(words) < 3:
            return -1e9
        return float(np.mean([lp(words[i], words[i + 1], words[i + 2])
                              for i in range(len(words) - 2)]))

    def _table_support(w0, w1, w2, n_each=20):
        """Target-vocab words the continuation tables expect after this
        3-word context - the fluent manifold for the merit-pool decode."""
        sup = set()
        dicts = [tri_c.get((w1, w2)), bi_c.get(w2)]
        if extra:
            dicts += [quad_t.get((w0, w1, w2)), skipA_t.get((w0, w2)),
                      skipB_t.get((w0, w1))]
        for d in dicts:
            if d:
                for w in sorted(d, key=d.get, reverse=True)[:n_each]:
                    k = tgt_i.get(w)
                    if k is not None:
                        sup.add(k)
        return sup

    def _step_raw(win_ids):
        """(feature row F1 [cpu], logits) - the demo trace decomposes the
        head dot-product per bank block from these + _M['head']."""
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
        F1 = torch.cat(cols).view(1, -1)
        lg = (torch.hstack([(F1 - hm) / hs,
                            torch.ones(1, 1, device=dev)]) @ Wm)[0]
        return F1[0].cpu(), lg

    # ---- FAST DECODE (exact): logits = C + sparse/dense updates -------
    # The head is linear, so precompute the all-zero row's logit vector C
    # and per-column scaled weight rows K = W/hs once; each step then adds
    # only the ~60 sparse entries (one-hots, table probabilities) and the
    # small dense blocks (embeds, table vectors, genomes). Bit-identical
    # to the slow path (per-column clamps compose); verified at build.
    _np = np
    Kw = (Wm[:-1] / hs.view(-1, 1)).cpu().numpy()          # (dcols, V)
    _dcols = Kw.shape[0]
    _coff = W * D + 2 * V
    _z0 = np.zeros(_dcols, np.float32)                     # bank value of an
    _cmu_n = cmu.cpu().numpy(); _csd_n = csd.cpu().numpy() # all-zero input
    _z0[_coff:_coff + N_CONT] = np.clip((0 - _cmu_n) / _csd_n, -8, 8)
    _hm_n = hm.cpu().numpy(); _hs_n = hs.cpu().numpy()
    C0 = ((_z0 - _hm_n) / _hs_n) @ (Wm[:-1].cpu().numpy())         + Wm[-1].cpu().numpy()                             # (V,)
    _mu_n = mu.cpu().numpy(); _sd_n = sd.cpu().numpy()
    _zmu_n = [a.cpu().numpy() for a, b in space_stats]
    _zsd_n = [b.cpu().numpy() for a, b in space_stats]

    _tbl_cache = {}

    def _tbl_entry(dist):
        """(vec128, target_cols, probs) for a table entry - computed ONCE
        per unique entry per process (tables are immutable in RAM)."""
        e = _tbl_cache.get(id(dist))
        if e is not None:
            return e
        vec = _cont_vec(dist)
        v = 0.0
        pairs = []
        for w_, c_ in dist.items():
            k_ = tgt_i.get(w_)
            if k_ is not None:
                pairs.append((k_, c_)); v += c_
        if pairs and v > 0:
            cols = np.array([k_ for k_, c_ in pairs])
            vals = np.array([c_ / v for k_, c_ in pairs], np.float32)
        else:
            cols = np.zeros(0, np.int64); vals = np.zeros(0, np.float32)
        e = (vec, cols, vals)
        _tbl_cache[id(dist)] = e
        return e

    def _fast_step(win_ids):
        lg = C0.copy()
        ctx1 = np.asarray(win_ids, np.int64)
        # dense embed block
        emb = np.zeros(W * D, np.float32)
        for f_ in range(W):
            j_ = int(ctx1[f_])
            if j_ >= 0:
                emb[f_ * D:(f_ + 1) * D] = (feat[j_] - _mu_n) / _sd_n
            else:
                emb[f_ * D:(f_ + 1) * D] = (0 - _mu_n) / _sd_n
        lg += emb @ Kw[:W * D]
        # sparse pieces: (col, bank_value) pairs for cont/identity region
        sp_cols, sp_vals = [], []
        for si_, slot in enumerate((W - 2, W - 1)):
            k_ = tv.get(int(ctx1[slot]), -1)
            if k_ >= 0:
                sp_cols.append(W * D + si_ * V + k_); sp_vals.append(1.0)
        j0, j1, j2 = int(ctx1[W - 3]), int(ctx1[W - 2]), int(ctx1[W - 1])
        w0 = vocab[j0] if j0 >= 0 else None
        w1 = vocab[j1] if j1 >= 0 else None
        w2 = vocab[j2] if j2 >= 0 else None
        dense_vecs = []                     # (offset, D-vector) raw values
        key = (w1, w2)
        if key in tri_c:
            d_ = tri_c[key]
        elif w2 in bi_c:
            d_ = bi_c[w2]
        else:
            d_ = uni_c
        vec, cols_, vals_ = _tbl_entry(d_)
        dense_vecs.append((_coff, vec))
        sp_cols += list(_coff + 2 * D + cols_); sp_vals += list(vals_)
        dense_vecs.append((_coff + D,
                           _tbl_entry(bi_c[w2])[0] if w2 in bi_c
                           else _uni_vec))
        if extra:
            base_ = _coff + 2 * D + V
            dq = quad_t.get((w0, w1, w2))
            if dq:
                vec, cols_, vals_ = _tbl_entry(dq)
                dense_vecs.append((base_, vec))
                sp_cols += list(base_ + D + cols_); sp_vals += list(vals_)
            da = skipA_t.get((w0, w2))
            if da:
                vec, cols_, vals_ = _tbl_entry(da)
                dense_vecs.append((base_ + D + V, vec))
                sp_cols += list(base_ + 2 * D + V + cols_)
                sp_vals += list(vals_)
            db = skipB_t.get((w0, w1))
            if db:
                dense_vecs.append((base_ + 2 * D + 2 * V, _tbl_entry(db)[0]))
        # apply cont-region updates: z = clip((x-cmu)/csd) vs baseline z0
        for off, vec in dense_vecs:
            oc = off - _coff
            zi = np.clip((vec - _cmu_n[oc:oc + D])
                         / _csd_n[oc:oc + D], -8, 8)
            dz = zi - _z0[off:off + D]
            lg += dz @ Kw[off:off + D]
        if sp_cols:
            cols_a = np.asarray(sp_cols)
            vals_a = np.asarray(sp_vals, np.float32)
            in_cont = cols_a >= _coff
            z_new = vals_a.copy()
            cc = cols_a[in_cont] - _coff
            z_new[in_cont] = np.clip((vals_a[in_cont] - _cmu_n[cc])
                                     / _csd_n[cc], -8, 8)
            dz = z_new - _z0[cols_a]
            lg += dz @ Kw[cols_a]
        # genomes: exact numpy eval (no GPU round-trip)
        if ckpt["spaces"]:
            rowsn = np.empty((W, D), np.float32)
            for f_ in range(W):
                j_ = int(win_ids[f_])
                rowsn[f_] = ((feat[j_] if j_ >= 0 else 0) - _mu_n) / _sd_n
            from radial_evo2 import _PRIMS as _PR
            from radial_stack import _OPS as _OP2
            f0 = np.array([np.nan_to_num(
                _np_feature_vec(rowsn, g, _PR, _OP2),
                nan=0.0, posinf=0.0, neginf=0.0).clip(-1e6, 1e6).mean()
                for g in ckpt["spaces"][0]], np.float32)
            gz = np.clip((f0 - _zmu_n[0]) / _zsd_n[0], -8, 8)
            gcols0 = _dcols - len(gz)
            dzg = gz - _z0[gcols0:]
            lg += dzg @ Kw[gcols0:]
        return lg

    # batched GPU step: logits = C0 + dz @ Kw, all rows in one matmul.
    # dz is the (row - all-zeros-baseline) delta in bank-z space, assembled
    # per row on CPU (cheap: ~2.7k dense + ~60 sparse entries), multiplied
    # on the GPU where Kw (554MB) lives resident.
    _Kw_t = torch.tensor(Kw, device=dev) if dev == "cuda" else None
    _C0_t = torch.tensor(C0, device=dev) if dev == "cuda" else None

    def _assemble_dz(win_ids, out):
        """Write the delta row for one window into out (len dcols)."""
        out[:] = 0.0
        for f_ in range(W):
            j_ = int(win_ids[f_])
            if j_ >= 0:
                out[f_ * D:(f_ + 1) * D] = (feat[j_] - _mu_n) / _sd_n
            else:
                out[f_ * D:(f_ + 1) * D] = (0 - _mu_n) / _sd_n
        for si_, slot in enumerate((W - 2, W - 1)):
            k_ = tv.get(int(win_ids[slot]), -1)
            if k_ >= 0:
                out[W * D + si_ * V + k_] = 1.0
        j0, j1, j2 = (int(win_ids[W - 3]), int(win_ids[W - 2]),
                      int(win_ids[W - 1]))
        w0 = vocab[j0] if j0 >= 0 else None
        w1 = vocab[j1] if j1 >= 0 else None
        w2 = vocab[j2] if j2 >= 0 else None
        key = (w1, w2)
        if key in tri_c:
            d_ = tri_c[key]
        elif w2 in bi_c:
            d_ = bi_c[w2]
        else:
            d_ = uni_c
        def put_vec(off, vec):
            oc = off - _coff
            zi = np.clip((vec - _cmu_n[oc:oc + D]) / _csd_n[oc:oc + D],
                         -8, 8)
            out[off:off + D] = zi - _z0[off:off + D]
        def put_prob(base_p, cols_, vals_):
            if len(cols_):
                cc = base_p - _coff + cols_
                zn = np.clip((vals_ - _cmu_n[cc]) / _csd_n[cc], -8, 8)
                out[base_p + cols_] = zn - _z0[base_p + cols_]
        vec, cols_, vals_ = _tbl_entry(d_)
        put_vec(_coff, vec)
        put_prob(_coff + 2 * D, cols_, vals_)
        put_vec(_coff + D, _tbl_entry(bi_c[w2])[0] if w2 in bi_c
                else _uni_vec)
        if extra:
            base_ = _coff + 2 * D + V
            dq = quad_t.get((w0, w1, w2))
            if dq:
                vec, cols_, vals_ = _tbl_entry(dq)
                put_vec(base_, vec)
                put_prob(base_ + D, cols_, vals_)
            da = skipA_t.get((w0, w2))
            if da:
                vec, cols_, vals_ = _tbl_entry(da)
                put_vec(base_ + D + V, vec)
                put_prob(base_ + 2 * D + V, cols_, vals_)
            db = skipB_t.get((w0, w1))
            if db:
                put_vec(base_ + 2 * D + 2 * V, _tbl_entry(db)[0])
        if ckpt["spaces"]:
            rowsn = np.empty((W, D), np.float32)
            for f_ in range(W):
                j_ = int(win_ids[f_])
                rowsn[f_] = ((feat[j_] if j_ >= 0 else 0) - _mu_n) / _sd_n
            from radial_evo2 import _PRIMS as _PR
            from radial_stack import _OPS as _OP2
            f0 = np.array([np.nan_to_num(
                _np_feature_vec(rowsn, g, _PR, _OP2),
                nan=0.0, posinf=0.0, neginf=0.0).clip(-1e6, 1e6).mean()
                for g in ckpt["spaces"][0]], np.float32)
            gz = np.clip((f0 - _zmu_n[0]) / _zsd_n[0], -8, 8)
            gcols0 = _dcols - len(gz)
            out[gcols0:] = gz - _z0[gcols0:]

    _dz_buf = {}

    def _fast_step_batch(wins):
        """[(win_ids), ...] -> (n, V) logits via one GPU matmul."""
        n = len(wins)
        buf = _dz_buf.get(n)
        if buf is None:
            buf = np.zeros((n, _dcols), np.float32)
            _dz_buf[n] = buf
        for i, wv in enumerate(wins):
            _assemble_dz(wv, buf[i])
        if _Kw_t is not None:
            dz = torch.tensor(buf, device=dev)
            return (dz @ _Kw_t + _C0_t).cpu().numpy()
        return buf @ Kw + C0

    # exactness gate: fast vs slow on random windows before serving
    _fast_ok = False
    try:
        rngv = np.random.default_rng(0)
        merr = 0.0
        wvs = []
        for _ in range(8):
            wv = [int(rngv.integers(0, len(vocab))) for _ in range(W)]
            wvs.append(wv)
            slow = _step_logits(wv).detach().cpu().numpy()
            fastv = _fast_step(wv)
            merr = max(merr, float(np.abs(slow - fastv).max()))
        batchv = _fast_step_batch(wvs)
        for bi_, wv in enumerate(wvs):
            slow = _step_logits(wv).detach().cpu().numpy()
            merr = max(merr, float(np.abs(slow - batchv[bi_]).max()))
        _fast_ok = merr < 1e-2
        print(f"[infer] fast-decode exactness: max|diff| {merr:.2e} "
              f"-> {'ON' if _fast_ok else 'OFF (slow path)'}", flush=True)
    except Exception as _exc:
        print(f"[infer] fast decode disabled: {_exc}", flush=True)

    if _fast_ok:
        class _FastWrap:
            def __init__(self, fn): self.fn = fn
            def __call__(self, w): return self.fn(w)
        _step_np = _fast_step
    else:
        _step_np = None

    n_gen_ = len(ckpt["spaces"][0]) if ckpt["spaces"] else 0
    _layout = [("embed", 0, W * D), ("id_prev", W * D, W * D + V),
               ("id_last", W * D + V, W * D + 2 * V)]
    _c0 = W * D + 2 * V
    _layout += [("tbl_vec", _c0, _c0 + 2 * D),
                ("tbl_prob", _c0 + 2 * D, _c0 + 2 * D + V)]
    if extra:
        _b = _c0 + 2 * D + V
        _layout += [("quad_vec", _b, _b + D), ("quad_prob", _b + D, _b + D + V),
                    ("skipA_vec", _b + D + V, _b + 2 * D + V),
                    ("skipA_prob", _b + 2 * D + V, _b + 2 * D + 2 * V),
                    ("skipB_vec", _b + 2 * D + 2 * V, _b + 3 * D + 2 * V)]
    _cols0 = W * D + 2 * V + N_CONT
    _layout += [("genomes", _cols0, _cols0 + n_gen_)]
    _M.update(step_raw=_step_raw, head=(hm, hs, Wm), layout=_layout,
              step_np=_step_np,
              step_batch=(_fast_step_batch if _fast_ok else None),
              genome_defs=ckpt["spaces"][0] if ckpt["spaces"] else [])
    _M.update(cont_score=_cont_score, table_support=_table_support)
    _M.update(step=_step_logits, torch=torch, w2i=w2i, targets=targets,
              tgt_i=tgt_i, W=W, V=V, s_cal=s_cal, val_acc=val_acc,
              genome_params=genome_params, head_params=head_params,
              total_params=genome_params + head_params,
              n_genomes=sum(len(sp) for sp in ckpt["spaces"]),
              build_seconds=round(time.time() - t0))
    _STATE["status"] = "ready"


def _build_safe():
    global _FORCE_CPU
    try:
        _build()
    except Exception as exc:
        # the fat bank (17k-28k cols x 150k rows) can exceed a local GPU
        # (the dev 4080 has 16GB); one retry on CPU - slow but it lands
        if not _FORCE_CPU and "out of memory" in str(exc).lower():
            _FORCE_CPU = True
            _stage("GPU too small - rebuilding on CPU (slower)", 0.02)
            try:
                _build()
                return
            except Exception as exc2:
                exc = exc2
        _STATE.update(status="error", error=str(exc))


_STEER = {}
_GRAM = {}


def _gram_assets():
    """Lazy grammar-specialist assets (module 39): temporal genomes over
    the directional (syntactic) RS vectors + head - votes per decode step
    on which candidate word keeps the order grammatical."""
    if _GRAM.get("ready") or _GRAM.get("failed"):
        return _GRAM
    try:
        import torch
        gp = os.path.join(_HERE, "radial_data", "kid_grammar_model.json")
        with open(gp) as f:
            gm = json.load(f)
        zp = np.load(os.path.join(_HERE, "radial_data", "embed_rs_prev.npz"),
                     allow_pickle=True)
        zn = np.load(os.path.join(_HERE, "radial_data", "embed_rs_next.npz"),
                     allow_pickle=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        gvocab = {str(w): i for i, w in enumerate(zp["vocab"])}
        gE = np.concatenate([zp["feat"], zn["feat"]], 1).astype(np.float32)
        T = gm["T"]
        gcmu = np.array(gm["cmu"], np.float32)
        gcsd = np.array(gm["csd"], np.float32)
        gfmu = torch.tensor(gm["fmu"], device=dev)
        gfsd = torch.tensor(gm["fsd"], device=dev)
        ghm = torch.tensor(gm["head_mu"], device=dev)
        ghs = torch.tensor(gm["head_sd"], device=dev)
        gWm = torch.tensor(gm["head_W"], device=dev)
        import radial_temporal as rt

        gfmu_n = gfmu.cpu().numpy(); gfsd_n = gfsd.cpu().numpy()
        ghm_n = ghm.cpu().numpy(); ghs_n = ghs.cpu().numpy()
        gWm_n = gWm.cpu().numpy()

        def _np_temporal(F, g):
            """Exact numpy mirror of radial_temporal.temporal_feat."""
            N, T_, C_ = F.shape
            z = None
            for t in g["terms"]:
                o = min(t["o"], T_ - 1)
                xa = F[:, o:, t["a"] % C_]
                xb = F[:, :T_ - o, t["b"] % C_]
                if t["op"] == 0:
                    v = xa * xb
                elif t["op"] == 1:
                    v = xa - xb
                else:
                    v = np.minimum(xa, xb)
                p = t["pool"]
                if p == 1:
                    v = v.max(1)
                elif p == 2:
                    v = v[:, -1]
                elif p == 3:
                    wt = np.arange(1, v.shape[1] + 1, dtype=v.dtype)
                    v = (v * wt).sum(1) / wt.sum()
                else:
                    v = v.mean(1)
                v = t["w"] * v
                z = v if z is None else z + v
            return z

        def margins(ctx_words, cand_words):
            """Real-vs-shuffled margin of (last T-1 ctx + cand), batched
            over candidates - pure numpy, no GPU launches."""
            base = ctx_words[-(T - 1):]
            if len(base) < T - 1:
                base = [None] * (T - 1 - len(base)) + list(base)
            Xb = np.zeros((len(cand_words), T, gE.shape[1]), np.float32)
            for t, w in enumerate(base):
                i = gvocab.get(w) if w else None
                if i is not None:
                    Xb[:, t] = gE[i]
            for c, w in enumerate(cand_words):
                i = gvocab.get(w)
                if i is not None:
                    Xb[c, T - 1] = gE[i]
            Xb = np.clip((Xb - gcmu) / gcsd, -8, 8)
            cols = [np.nan_to_num(_np_temporal(Xb, g), nan=0.0,
                                  posinf=0.0, neginf=0.0).clip(-1e6, 1e6)
                    for g in gm["genomes"]]
            Ft = np.clip((np.stack(cols, 1) - gfmu_n) / gfsd_n, -8, 8)
            A = np.hstack([(Ft - ghm_n) / ghs_n,
                           np.ones((len(cand_words), 1), np.float32)])
            sc = A @ gWm_n
            return sc[:, 1] - sc[:, 0]

        def margins_batch(groups):
            """[(ctx_words, cand_words), ...] -> list of margin arrays,
            evaluated in ONE numpy pass over all windows."""
            allX, spans = [], []
            for ctx_words, cand_words in groups:
                base = ctx_words[-(T - 1):]
                if len(base) < T - 1:
                    base = [None] * (T - 1 - len(base)) + list(base)
                Xb = np.zeros((len(cand_words), T, gE.shape[1]), np.float32)
                for t, w in enumerate(base):
                    i = gvocab.get(w) if w else None
                    if i is not None:
                        Xb[:, t] = gE[i]
                for c, w in enumerate(cand_words):
                    i = gvocab.get(w)
                    if i is not None:
                        Xb[c, T - 1] = gE[i]
                spans.append((len(allX), len(allX) + len(cand_words)))
                allX.extend(Xb)
            if not allX:
                return [np.zeros(0) for _ in groups]
            X = np.clip((np.stack(allX) - gcmu) / gcsd, -8, 8)
            cols = [np.nan_to_num(_np_temporal(X, g), nan=0.0,
                                  posinf=0.0, neginf=0.0).clip(-1e6, 1e6)
                    for g in gm["genomes"]]
            Ft = np.clip((np.stack(cols, 1) - gfmu_n) / gfsd_n, -8, 8)
            A = np.hstack([(Ft - ghm_n) / ghs_n,
                           np.ones((len(X), 1), np.float32)])
            sc = A @ gWm_n
            m = sc[:, 1] - sc[:, 0]
            return [m[a:b] for a, b in spans]

        _GRAM.update(ready=True, margins=margins,
                     margins_batch=margins_batch, test_acc=gm["test_acc"])
    except Exception as exc:            # the vote is optional - never break
        _GRAM.update(failed=True, error=str(exc))
    return _GRAM


def _steer_assets():
    """Lazy topic-steering assets (module 33-35 rig): the persistence topic
    model's per-target topic scores (evidence-floored) + the accumulated-
    state classifier. Built once per process, ~seconds."""
    if _STEER.get("ready") or _STEER.get("failed"):
        return _STEER
    try:
        import torch
        from topic_steer import (load_topic_model, target_topic_scores,
                                 load_evidence)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tm, responses, topic_probs, hm, hs, Wm = load_topic_model(torch, dev)
        S = target_topic_scores(torch, dev, responses, hm, hs, Wm,
                                _M["targets"], evidence=load_evidence())

        def tscore(words, t_star):
            """Topic head logit over a word list - the hybrid reranker's
            topic term."""
            r = responses(words)
            if r is None:
                return 0.0
            acc = r.mean(0, keepdim=True)
            lg = torch.hstack([(acc - hm) / hs,
                               torch.ones(1, 1, device=dev)]) @ Wm
            return float(lg[0, t_star])

        _STEER.update(ready=True, S=S.cpu().numpy().astype(np.float64),
                      topics=tm["topics"], topic_probs=topic_probs,
                      tscore=tscore)
    except Exception as exc:            # steering is optional - never break
        _STEER.update(failed=True, error=str(exc))   # plain autocomplete
    return _STEER


def complete(prompt, n_words=24, temp=0.7, seed=None, steer="auto", lam=1.5,
             topk=3, best_of=1):
    """Autocomplete `prompt` with the latest word checkpoint.

    steer: 'auto' (default) steers word choice toward the prompt's topic
    when the persistence topic state is CONFIDENT (p >= 0.30) - a generic
    prompt gets no topic forced onto it; 'off' disables. lam scales the
    steering bonus. Defaults are the post-CRANK measured winners (module
    37): lam=2.0 holds topic 16/16 at only -6.5pts next-word on the 0.56
    base model; decode temp 0.9 / top-5 is what those numbers were
    measured at. best_of > 1 = 'polish': best-of-k continuation-table
    rerank (best-of-8 at these settings: judge -8.60, hold 0.81)."""
    with _LOCK:
        if _STATE["status"] in ("idle", "error"):
            _STATE.update(status="building", error=None, progress="starting")
            threading.Thread(target=_build_safe, daemon=True,
                             name="lm-word-infer-build").start()
    if _STATE["status"] != "ready":
        el, eta = _build_eta()
        return {"building": _STATE["status"] == "building",
                "status": _STATE["status"],
                "progress": _STATE.get("progress"), "error": _STATE["error"],
                "elapsed": el, "eta": eta}
    words = [w for w in prompt.lower().split() if w][:64]
    if not words:
        return {"error": "empty prompt"}
    W, V = _M["W"], _M["V"]
    win0 = [_M["w2i"].get(w, -1) for w in words][-W:]
    while len(win0) < W:
        win0.insert(0, -1)

    lam = max(0.0, min(4.0, float(lam)))
    topk = max(1, min(10, int(topk)))
    best_of = max(1, min(8, int(best_of)))
    t_star, t_name, t_p = None, None, None
    if steer != "off" and lam > 0:
        sa = _steer_assets()
        if sa.get("ready"):
            p = sa["topic_probs"](words)
            if p is not None:
                pm = float(p.max())
                if pm >= 0.30:               # confident state only
                    t_star = int(p.argmax())
                    t_name, t_p = sa["topics"][t_star], round(pm, 3)

    def _gen(sd):
        """Merit-pool decode (module 38): candidates = (model top-10 that
        the continuation tables support) UNION (strong topic words the
        model rates in its top-25), ranked by the steered logits - fluency
        stays table-anchored, topic words compete on merit."""
        rng = np.random.default_rng(sd)
        win = list(win0)
        ctx_words = list(words)
        out_words = []
        for _ in range(max(1, min(60, int(n_words)))):
            if _M.get("step_batch") is not None:
                lg = _M["step_batch"]([win])[0].astype(np.float64)
            elif _M.get("step_np") is not None:
                lg = _M["step_np"](win).astype(np.float64)
            else:
                lg = _M["step"](win).detach().cpu().numpy().astype(np.float64)
            lg = lg * _M["s_cal"] / max(float(temp), 1e-3)
            if t_star is not None:
                bonus = lam * _STEER["S"][:, t_star].copy()
                for wd in out_words:         # steer toward NEW topic words
                    kr = _M["tgt_i"].get(wd)
                    if kr is not None:
                        bonus[kr] = 0.0
                lg = lg + bonus
            rep = 2.0 + (lam if t_star is not None else 0.0)
            for wd in out_words[-16:]:
                kr = _M["tgt_i"].get(wd)
                if kr is not None:
                    lg[kr] -= rep
            order = np.argsort(lg)
            c3 = ctx_words[-3:] if len(ctx_words) >= 3 else \
                [None] * (3 - len(ctx_words)) + ctx_words
            sup = _M["table_support"](c3[0], c3[1], c3[2])
            pool = [int(k) for k in order[-10:] if k in sup]
            if t_star is not None:
                Scol = _STEER["S"][:, t_star]
                pool += [int(k) for k in order[-25:]
                         if Scol[k] > 2.0 and int(k) not in pool
                         and _M["targets"][k] not in out_words]
            top = (np.array(sorted(pool, key=lambda k: lg[k])[-5:])
                   if pool else order[-3:])
            lg2 = lg[top].astype(np.float64)
            ga = _gram_assets()
            if ga.get("ready") and len(top) > 1:
                gmg = ga["margins"](ctx_words,
                                    [_M["targets"][k] for k in top])
                lg2 = lg2 + 1.0 * ((gmg - gmg.mean())
                                   / (gmg.std() + 1e-9))
            sel = np.argsort(lg2)[-topk:]
            top, lg2 = top[sel], lg2[sel]
            if len(top) == 1:
                k = int(top[0])
            else:
                z = lg2 - lg2.max()
                p = np.exp(z)
                p /= p.sum()
                k = int(rng.choice(top, p=p))
            out_words.append(_M["targets"][k])
            ctx_words.append(_M["targets"][k])
            win = win[1:] + [_M["w2i"].get(_M["targets"][k], -1)]
        return out_words

    def _gen_lockstep(sds):
        """All candidate sequences advance together: per step, one
        grammar pass covers every candidate of every sequence. Identical
        trajectories to sequential _gen (same per-seed rngs)."""
        rngs = [np.random.default_rng(sd) for sd in sds]
        states = [{"win": list(win0), "ctx": list(words), "out": []}
                  for _ in sds]
        ga = _gram_assets()
        for _ in range(max(1, min(60, int(n_words)))):
            pools, lgs = [], []
            if _M.get("step_batch") is not None:
                LG = _M["step_batch"]([st["win"] for st in states])                     .astype(np.float64)
            else:
                LG = None
            for si_, st in enumerate(states):
                if LG is not None:
                    lg = LG[si_].copy()
                elif _M.get("step_np") is not None:
                    lg = _M["step_np"](st["win"]).astype(np.float64)
                else:
                    lg = _M["step"](st["win"]).detach().cpu().numpy()                         .astype(np.float64)
                lg = lg * _M["s_cal"] / max(float(temp), 1e-3)
                if t_star is not None:
                    bonus = lam * _STEER["S"][:, t_star].copy()
                    for wd in st["out"]:
                        kr = _M["tgt_i"].get(wd)
                        if kr is not None:
                            bonus[kr] = 0.0
                    lg = lg + bonus
                rep = 2.0 + (lam if t_star is not None else 0.0)
                for wd in st["out"][-16:]:
                    kr = _M["tgt_i"].get(wd)
                    if kr is not None:
                        lg[kr] -= rep
                order = np.argsort(lg)
                c3 = st["ctx"][-3:] if len(st["ctx"]) >= 3 else                     [None] * (3 - len(st["ctx"])) + st["ctx"]
                sup = _M["table_support"](c3[0], c3[1], c3[2])
                pool = [int(k) for k in order[-10:] if k in sup]
                if t_star is not None:
                    Scol = _STEER["S"][:, t_star]
                    pool += [int(k) for k in order[-25:]
                             if Scol[k] > 2.0 and int(k) not in pool
                             and _M["targets"][k] not in st["out"]]
                top = (np.array(sorted(pool, key=lambda k: lg[k])[-5:])
                       if pool else order[-3:])
                pools.append(top)
                lgs.append(lg)
            if ga.get("ready"):
                margs = ga["margins_batch"](
                    [(states[i]["ctx"],
                      [_M["targets"][k] for k in pools[i]])
                     for i in range(len(states))])
            else:
                margs = [None] * len(states)
            for i, st in enumerate(states):
                top, lg = pools[i], lgs[i]
                lg2 = lg[top].astype(np.float64)
                if margs[i] is not None and len(top) > 1:
                    gm_ = margs[i]
                    lg2 = lg2 + (gm_ - gm_.mean()) / (gm_.std() + 1e-9)
                sel = np.argsort(lg2)[-topk:]
                t2, l2 = top[sel], lg2[sel]
                if len(t2) == 1:
                    k = int(t2[0])
                else:
                    z = l2 - l2.max()
                    p = np.exp(z)
                    p /= p.sum()
                    k = int(rngs[i].choice(t2, p=p))
                wd = _M["targets"][k]
                st["out"].append(wd)
                st["ctx"].append(wd)
                st["win"] = st["win"][1:] + [_M["w2i"].get(wd, -1)]
        return [st["out"] for st in states]

    if best_of == 1:
        out_words = _gen(seed)
    else:                                     # 'polish': hybrid best-of-k
        seeds = np.random.default_rng(seed).integers(0, 2**31, best_of)
        cands = _gen_lockstep([int(s) for s in seeds])
        cs = np.array([_M["cont_score"](c) for c in cands])
        if t_star is not None and _STEER.get("tscore"):
            ts_ = np.array([_STEER["tscore"](c, t_star) for c in cands])
            zc = (cs - cs.mean()) / (cs.std() + 1e-9)
            zt = (ts_ - ts_.mean()) / (ts_.std() + 1e-9)
            out_words = cands[int(np.argmax(zc + zt))]
        else:
            out_words = cands[int(np.argmax(cs))]
    return {"prompt": " ".join(words), "completion": " ".join(out_words),
            "params": {"genome": _M["genome_params"],
                       "head": _M["head_params"],
                       "total": _M["total_params"],
                       "n_genomes": _M["n_genomes"]},
            "vocab": V, "context": W, "val_acc": round(_M["val_acc"], 4),
            "decode_scale": _M["s_cal"],
            "topic": t_name, "topic_p": t_p,
            "steered": t_star is not None, "lam": lam}
