"""build_pack_local.py - build the inference pack on a 32GB workstation.

The standard build (lm_word_infer._build) holds the bank TWICE (B0 + the
stacked feature matrix) - ~34GB at the 27.7k-col crank bank, which is why
it needs the 96GB pod. This builder:
  - allocates ONE bank tensor F (rows x cols+genomes, CPU RAM, ~17GB) and
    fills every block in place (embed, identities, continuation z-scored
    on a streaming second pass, genome columns);
  - streams 20k-row chunks through the LOCAL GPU to accumulate the fp64
    gram G and cross-product R (lam-independent, so the lam sweep and the
    decode calibration reuse them);
  - saves the pack in lm_word_infer's exact format+signature.
Only valid for single-space checkpoints (spaces==[k]) - the handoff bank
for deeper stacks is not built. ~30-45 min, one time.

  python lm/build_pack_local.py
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401

import json
import os
import pickle
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_ROOT, "radial_data")
CH = 20000


def main():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    import radial_lm_word as rw
    from radial_evo import _tprims
    import radial_stack as rk
    t0 = time.time()

    def log(m):
        print(f"[{round(time.time() - t0):4d}s] {m}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(RD, "lm_model_word.json")) as f:
        ckpt = json.load(f)
    assert len(ckpt["spaces"]) == 1, "single-space checkpoints only"
    rw.W = W = int(ckpt["context_words"])
    rw.V = V = int(ckpt["vocab"])
    D = rw.D
    sig = {"W": W, "V": V, "bank": ckpt.get("bank", "base"),
           "spaces": [len(sp) for sp in ckpt["spaces"]],
           "cont_pkl": ckpt.get("cont_pkl") or "lm_cont_tables.pkl",
           "skip_pkl": ckpt.get("skip_pkl")}
    log(f"target sig: {sig}")

    vocab, feat, _ = rw._load_embed()
    w2i = {w: i for i, w in enumerate(vocab)}
    z = np.load(os.path.join(RD, "lm_word.npz"), allow_pickle=True)
    assert z["ctx_tr"].shape[1] == W and len(z["targets"]) == V
    ctx_tr, ytr = z["ctx_tr"], z["ytr"]
    targets = [str(w) for w in z["targets"]]
    tgt_i = {w: k for k, w in enumerate(targets)}
    tv = {w2i[w]: k for k, w in enumerate(targets) if w in w2i}
    Ntr = len(ytr)
    mu_np = feat.mean(0)
    sd_np = feat.std(0) + 1e-6

    log("loading tables")
    with open(os.path.join(RD, sig["cont_pkl"]), "rb") as f:
        uni_c, bi_c, tri_c = pickle.load(f)
    import lm_crank
    if sig["skip_pkl"]:
        lm_crank.SKIP_PKL = sig["skip_pkl"]
    quad_t, skipA_t, skipB_t = lm_crank.build_tables()

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
    N_EXTRA = 2 * (D + V) + D
    N_CONT = 2 * D + V + N_EXTRA
    n_gen = len(ckpt["spaces"][0])
    COLS = W * D + 2 * V + N_CONT
    log(f"bank F: {Ntr} x {COLS + n_gen} fp32 "
        f"({round(Ntr * (COLS + n_gen) * 4 / 1e9, 1)}GB CPU)")
    F = torch.zeros((Ntr, COLS + n_gen), dtype=torch.float32)

    # embed block, in place
    log("filling embed block")
    idx = np.maximum(ctx_tr.astype(np.int64), 0)
    mask = (ctx_tr >= 0).astype(np.float32)
    for f_ in range(W):
        blk = (feat[idx[:, f_]] * mask[:, f_:f_ + 1] - mu_np) / sd_np
        F[:, f_ * D:(f_ + 1) * D] = torch.tensor(blk.astype(np.float32))
    del idx, mask

    # identity blocks, in place
    log("filling identity blocks")
    for si_, slot in enumerate((W - 2, W - 1)):
        off = W * D + si_ * V
        for i in range(Ntr):
            k = tv.get(int(ctx_tr[i, slot]), -1)
            if k >= 0:
                F[i, off + k] = 1.0

    # continuation block: python pass fills RAW + accumulates stats,
    # then z-score in place
    log("filling continuation block (the long pass)")
    coff = W * D + 2 * V
    ssum = np.zeros(N_CONT, np.float64)
    ssq = np.zeros(N_CONT, np.float64)
    row = np.zeros(N_CONT, np.float32)
    for i in range(Ntr):
        row[:] = 0.0
        j0, j1, j2 = (int(ctx_tr[i, W - 3]), int(ctx_tr[i, W - 2]),
                      int(ctx_tr[i, W - 1]))
        w0 = vocab[j0] if j0 >= 0 else None
        w1 = vocab[j1] if j1 >= 0 else None
        w2 = vocab[j2] if j2 >= 0 else None
        key = (w1, w2)
        if key in tri_c:
            row[:D] = _cont_vec(tri_c[key])
            row[2 * D:2 * D + V] = _cont_prob(tri_c[key])
        elif w2 in bi_c:
            row[:D] = _cont_vec(bi_c[w2])
            row[2 * D:2 * D + V] = _cont_prob(bi_c[w2])
        else:
            row[:D] = _uni_vec
            row[2 * D:2 * D + V] = _uni_prob
        row[D:2 * D] = _cont_vec(bi_c[w2]) if w2 in bi_c else _uni_vec
        base = 2 * D + V
        dq = quad_t.get((w0, w1, w2))
        if dq:
            row[base:base + D] = _cont_vec(dq)
            row[base + D:base + D + V] = _cont_prob(dq)
        base += D + V
        da = skipA_t.get((w0, w2))
        if da:
            row[base:base + D] = _cont_vec(da)
            row[base + D:base + D + V] = _cont_prob(da)
        base += D + V
        db = skipB_t.get((w0, w1))
        if db:
            row[base:base + D] = _cont_vec(db)
        F[i, coff:coff + N_CONT] = torch.tensor(row)
        ssum += row
        ssq += row.astype(np.float64) ** 2
        if i % 25000 == 0:
            log(f"  cont row {i}/{Ntr}")
    cmu_np = (ssum / Ntr).astype(np.float32)
    csd_np = np.sqrt(np.maximum(ssq / Ntr - (ssum / Ntr) ** 2, 0)) \
        .astype(np.float32) + 1e-6
    log("z-scoring continuation block in place")
    cmu_t = torch.tensor(cmu_np)
    csd_t = torch.tensor(csd_np)
    for a in range(0, Ntr, CH):
        F[a:a + CH, coff:coff + N_CONT] = (
            (F[a:a + CH, coff:coff + N_CONT] - cmu_t) / csd_t
        ).clamp_(-8, 8)

    # genome columns (space 0: per-slot rows, window-mean)
    log("genome columns")
    tp = _tprims(torch)
    idx = np.maximum(ctx_tr.astype(np.int64), 0)
    mask = (ctx_tr >= 0).astype(np.float32)
    rows_np = ((feat[idx.reshape(-1)] * mask.reshape(-1, 1) - mu_np)
               / sd_np).astype(np.float32)
    rows_t = torch.tensor(rows_np, device=dev)
    del rows_np, idx, mask
    gcols = []
    for g in ckpt["spaces"][0]:
        c = torch.nan_to_num(rk.feature_vec(torch, tp, rows_t, g),
                             nan=0.0, posinf=0.0, neginf=0.0) \
            .clamp(-1e6, 1e6).view(Ntr, W).mean(1)
        gcols.append(c.cpu())
    del rows_t
    gmat = torch.stack(gcols, 1)
    zmu, zsd = gmat.mean(0), gmat.std(0) + 1e-6
    F[:, COLS:] = ((gmat - zmu) / zsd).clamp(-8, 8)
    space_stats = [(zmu, zsd)]

    # ---- GPU-chunked gram + head ----
    dcols = COLS + n_gen
    n_fit = int(Ntr * 0.8)
    hm_f = F[:n_fit].mean(0)
    hs_f = F[:n_fit].std(0) + 1e-6

    def gram(rows_lo, rows_hi, hm, hs):
        G = torch.zeros((dcols + 1, dcols + 1), device=dev,
                        dtype=torch.float64)
        R = torch.zeros((dcols + 1, V), device=dev, dtype=torch.float64)
        hm_d, hs_d = hm.to(dev), hs.to(dev)
        for a in range(rows_lo, rows_hi, CH):
            b = min(a + CH, rows_hi)
            Ab = torch.hstack([((F[a:b].to(dev) - hm_d) / hs_d),
                               torch.ones(b - a, 1, device=dev)])
            Yb = -torch.ones((b - a, V), device=dev)
            Yb[torch.arange(b - a),
               torch.tensor(ytr[a:b].astype(np.int64), device=dev)] = 1.0
            G += (Ab.T @ Ab).double()
            R += (Ab.T @ Yb).double()
            del Ab, Yb
        return G, R

    log("gram over fit rows (GPU-chunked)")
    Gf, Rf = gram(0, n_fit, hm_f, hs_f)
    yv = torch.tensor(ytr[n_fit:].astype(np.int64), device=dev)

    def val_acc(Wm, hm, hs):
        hits = 0
        hm_d, hs_d = hm.to(dev), hs.to(dev)
        for a in range(n_fit, Ntr, CH):
            b = min(a + CH, Ntr)
            Ab = torch.hstack([((F[a:b].to(dev) - hm_d) / hs_d),
                               torch.ones(b - a, 1, device=dev)])
            s = Ab @ Wm
            hits += int((s.argmax(1) == yv[a - n_fit:b - n_fit]).sum())
        return hits / (Ntr - n_fit)

    best = (3.0, -1.0, None)
    for lam in (3.0, 10.0, 30.0):
        Wm = torch.linalg.solve(
            Gf + lam * torch.eye(dcols + 1, device=dev,
                                 dtype=torch.float64), Rf).float()
        a = val_acc(Wm, hm_f, hs_f)
        log(f"  lam {lam}: val {a:.4f}")
        if a > best[1]:
            best = (lam, a, Wm)
    lam, va, Wm_f = best

    log("decode calibration")
    s_cal, best_nll = 1.0, 1e9
    hm_d, hs_d = hm_f.to(dev), hs_f.to(dev)
    nll_sums = {sc: 0.0 for sc in (1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 35.0)}
    cnt = 0
    for a in range(n_fit, Ntr, CH):
        b = min(a + CH, Ntr)
        Ab = torch.hstack([((F[a:b].to(dev) - hm_d) / hs_d),
                           torch.ones(b - a, 1, device=dev)])
        s = Ab @ Wm_f
        yb = yv[a - n_fit:b - n_fit]
        for sc in nll_sums:
            nll_sums[sc] += float(-torch.log_softmax(s * sc, 1)
                                  [torch.arange(b - a), yb].sum())
        cnt += b - a
    for sc, tot in nll_sums.items():
        if tot / cnt < best_nll:
            s_cal, best_nll = sc, tot / cnt

    log("final head on ALL rows")
    hm = F.mean(0)
    hs = F.std(0) + 1e-6
    G, R = gram(0, Ntr, hm, hs)
    Wm = torch.linalg.solve(
        G + lam * torch.eye(dcols + 1, device=dev,
                            dtype=torch.float64), R).float()

    pack = {"sig": sig, "targets": targets,
            "cmu": cmu_t, "csd": csd_t,
            "space_stats": [(a.cpu(), b.cpu()) for a, b in space_stats],
            "pmu": None, "psd": None,
            "hm": hm.cpu(), "hs": hs.cpu(), "Wm": Wm.cpu(),
            "s_cal": s_cal, "val_acc": va}
    torch.save(pack, os.path.join(RD, "lm_infer_pack.pt"))
    log(f"PACK SAVED: val {va:.4f} lam {lam} s_cal {s_cal} "
        f"({round(time.time() - t0)}s total)")
    print("PACK DONE", flush=True)


if __name__ == "__main__":
    main()
