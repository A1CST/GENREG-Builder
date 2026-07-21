"""replicate_autostack.py — module 60: the SELF-ASSEMBLING model.

Automate the question search. Walk a library of label-free questions; for each,
evolve a space asking it over the current representation; KEEP the space only
if it raises held-out (val) accuracy, DROP it otherwise; continue until the
library is exhausted. The stack builds itself; only questions that earn stay.

  - every question is INVARIANCE to a transform (flip, crop, photo, scale,
    blur, cutout, gray, rotate...). A word answering question q fires the same
    on two q-transformed views -> it names content the transform destroys as
    nuisance. Different transforms = different, composable questions.
  - space 1 reads the image (patch-PCA); every later space reads the full
    current representation (all kept words) and appends its own.
  - labels enter ONLY at the final ridge head. Spaces oscillate (energy on),
    never stopped early.

    python3 replicate/replicate_autostack.py --rounds 40 --cap 1600
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
import radial_evo2 as re2
from radial_evo2 import Env, SCALES, new_genome, mutate, feature
from replicate_langspace import _fp
from replicate_langstack import evolve, new_vec, mut_vec, feat_vec

RD = os.path.join(_HERE, "radial_data")


def log(m):
    print(m, flush=True)


def transform(torch, X, seed, kind):
    """label-free VIEW generator; invariance to `kind` is the question."""
    import torch.nn.functional as Fn
    g = torch.Generator(device=X.device).manual_seed(seed)
    t = X.permute(0, 3, 1, 2)
    n = len(t)
    if kind == "hflip":
        t = torch.flip(t, dims=[3])
    elif kind == "crop":
        t = Fn.pad(t, (3, 3, 3, 3), mode="reflect")
        ox = int(torch.randint(0, 7, (1,), generator=g, device=X.device))
        oy = int(torch.randint(0, 7, (1,), generator=g, device=X.device))
        t = t[:, :, oy:oy + 32, ox:ox + 32]
    elif kind == "photo":
        b = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=X.device) - .5) * .5
        con = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=X.device) - .5) * .5
        col = 1 + (torch.rand(n, 3, 1, 1, generator=g, device=X.device) - .5) * .3
        m = t.mean((2, 3), keepdim=True)
        t = (((t - m) * con + m) * b * col).clamp(0, 1)
    elif kind == "scale":
        s = float(0.8 + torch.rand(1, generator=g, device=X.device) * 0.4)
        hw = max(16, int(32 * s))
        t = Fn.interpolate(t, size=(hw, hw), mode="bilinear", align_corners=False)
        if hw >= 32:
            o = (hw - 32) // 2; t = t[:, :, o:o + 32, o:o + 32]
        else:
            p = (32 - hw) // 2; t = Fn.pad(t, (p, 32 - hw - p, p, 32 - hw - p))
    elif kind == "blur":
        k = torch.ones(3, 1, 3, 3, device=X.device) / 9.0
        t = Fn.conv2d(Fn.pad(t, (1, 1, 1, 1), mode="reflect"), k, groups=3)
    elif kind == "cutout":
        cx = int(torch.randint(0, 24, (1,), generator=g, device=X.device))
        cy = int(torch.randint(0, 24, (1,), generator=g, device=X.device))
        t = t.clone(); t[:, :, cy:cy + 8, cx:cx + 8] = 0.5
    elif kind == "gray":
        lum = (t * torch.tensor([.299, .587, .114], device=X.device
                                ).view(1, 3, 1, 1)).sum(1, keepdim=True)
        a = float(0.3 + torch.rand(1, generator=g, device=X.device) * 0.7)
        t = a * lum + (1 - a) * t
    elif kind == "rotate":
        ang = float((torch.rand(1, generator=g, device=X.device) - .5) * 0.5)
        c, s = np.cos(ang), np.sin(ang)
        th = torch.tensor([[c, -s, 0], [s, c, 0]], device=X.device,
                          dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        grid = Fn.affine_grid(th, t.shape, align_corners=False)
        t = Fn.grid_sample(t, grid, align_corners=False, padding_mode="reflection")
    return t.permute(0, 2, 3, 1).contiguous()


WORD_QUESTIONS = {"worddrop", "wordhalf", "wordnoise"}


def word_view(torch, Wz, seed, kind):
    """generate a view by perturbing the WORDS (not the image). Invariance to
    this is a SEMANTIC question about the vocabulary, not the pixels."""
    g = torch.Generator(device=Wz.device).manual_seed(seed)
    F = Wz.shape[1]
    if kind == "worddrop":                       # drop half the vocabulary
        m = (torch.rand(F, generator=g, device=Wz.device) > 0.5).float()
        return Wz * m
    if kind == "wordhalf":                        # keep one contiguous half
        perm = torch.randperm(F, generator=g, device=Wz.device)
        keep = perm[:F // 2]
        m = torch.zeros(F, device=Wz.device); m[keep] = 1.0
        return Wz * m
    if kind == "wordnoise":                       # jitter word activations
        return Wz + torch.randn(Wz.shape, generator=g, device=Wz.device) * 0.5
    return Wz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--cap", type=int, default=1600)
    ap.add_argument("--freeze", type=float, default=0.02)
    ap.add_argument("--keep_thresh", type=float, default=0.002,
                    help="min val gain to keep a question's space")
    ap.add_argument("--per_space_k", type=int, default=300,
                    help="compress each space's words to top-K (0=off) so many "
                         "questions compose without the head overfitting")
    ap.add_argument("--questions",
                    default="photo,worddrop,wordhalf,wordnoise,worddrop,wordhalf",
                    help="space 1 = image question (perceptual); rest = WORD "
                         "questions (semantic, about the vocabulary itself)")
    args = ap.parse_args()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr_np = z["Xtr"].astype(np.float32) / 255.0
    Xte_np = z["Xte"].astype(np.float32) / 255.0
    ytr = torch.tensor(z["ytr"].astype(np.int64), device=dev)
    yte = torch.tensor(z["yte"].astype(np.int64), device=dev)
    N = len(z["ytr"])
    Xtr_t = torch.tensor(Xtr_np, device=dev)
    env0 = Env(torch, dev, Xtr_np, Xte_np, max_cached=len(SCALES))
    for ps in SCALES:
        env0.maps(ps)
    basis = {ps: re2._SVD_CACHE[(_fp(Xtr_np), ps)] for ps in SCALES}
    probe = torch.tensor(np.random.default_rng(5).permutation(N)[:4000], device=dev)
    rng = np.random.default_rng(60)
    n_fit = int(N * 0.8); yv = ytr[n_fit:]
    Y = -torch.ones((N, 10), device=dev); Y[torch.arange(N), ytr] = 1.
    device_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    COST = {"evals": 0}          # genome x image feature evaluations (compute proxy)

    def counted(base, n):
        def w(gs):
            COST["evals"] += len(gs) * n
            return base(gs)
        return w

    def env_for(Xnp):
        e = Env(torch, dev, Xnp, Xnp[:100], max_cached=len(SCALES))
        for ps in SCALES:
            re2._SVD_CACHE[(_fp(Xnp), ps)] = basis[ps]
        return e

    def repr_forward(spaces, Xnp=None, env=None, test=False):
        """run patch-PCA + all kept spaces -> full representation (N, F)."""
        e = env if env is not None else env_for(Xnp)
        W = None
        for sp in spaces:
            if sp["kind"] == "img":
                new = torch.stack([feature(torch, tp, e, g, test=test)
                                   for g in sp["genomes"]], 1)
            else:
                Wz = (W - sp["in_mu"]) / sp["in_sd"]
                new = torch.stack([feat_vec(torch, tp, Wz, g)
                                   for g in sp["genomes"]], 1)
            COST["evals"] += len(sp["genomes"]) * new.shape[0]
            pj = sp.get("proj")
            if pj is not None:
                new = ((new - pj[0]) / pj[1]) @ pj[2]      # (cm, csd, Vk)
            W = new if W is None else torch.cat([W, new], 1)
        return W

    def head_val(Wtr):
        mu, sd = Wtr.mean(0), Wtr.std(0) + 1e-6
        Z = (Wtr - mu) / sd
        bv = -1
        for lam in (3., 10., 30., 100., 300.):
            _, a = _ridge_soft(torch, Z[:n_fit], Z[n_fit:], Y[:n_fit], yv, lam=lam)
            bv = max(bv, a)
        return bv

    def head_test(Wtr, Wte):
        mu, sd = Wtr.mean(0), Wtr.std(0) + 1e-6
        Z, Zt = (Wtr - mu) / sd, (Wte - mu) / sd
        bl, bv = 30., -1
        for lam in (3., 10., 30., 100., 300.):
            _, a = _ridge_soft(torch, Z[:n_fit], Z[n_fit:], Y[:n_fit], yv, lam=lam)
            if a > bv:
                bl, bv = lam, a
        A = torch.hstack([Z, torch.ones(N, 1, device=dev)])
        W = torch.linalg.solve((A.T @ A).double() + bl * torch.eye(A.shape[1],
                               device=dev, dtype=torch.float64),
                               (A.T @ Y).double()).float()
        B = torch.hstack([Zt, torch.ones(len(yte), 1, device=dev)])
        return float(((B @ W).argmax(1) == yte).float().mean()), bv

    kept = []            # frozen spaces that earned their place
    kept_names = []
    W_clean = None       # current representation on clean train
    W_clean_te = None
    cur_val = 0.10       # chance
    remaining = args.questions.split(",")
    log(f"[auto] question library: {remaining} ({dev})")

    space_times = []
    for q in remaining:
        tq = time.time()
        is_word = q in WORD_QUESTIONS
        if is_word and not kept:
            log(f"[auto] question '{q}': word-question can't be space 1 -> skip")
            continue
        if not kept:                              # first space: image (patch-PCA)
            XA = transform(torch, Xtr_t, 1, q).cpu().numpy()
            XB = transform(torch, Xtr_t, 2, q).cpu().numpy()
            eA, eB = env_for(XA), env_for(XB)
            feat_A = counted(lambda gs: torch.stack([feature(torch, tp, eA, g) for g in gs], 1), N)
            feat_B = counted(lambda gs: torch.stack([feature(torch, tp, eB, g) for g in gs], 1), N)
            new_fn, mut_fn = (lambda r: new_genome(r)), (lambda r, g, s: mutate(r, g, s))
            kind = "img"; in_mu = in_sd = None
        else:                                     # later space: over current WORDS
            in_mu, in_sd = W_clean.mean(0), W_clean.std(0) + 1e-6
            Wz = (W_clean - in_mu) / in_sd
            if is_word:                           # SEMANTIC: perturb the words
                WAz, WBz = word_view(torch, Wz, 1, q), word_view(torch, Wz, 2, q)
            else:                                 # image question over repr (perceptual)
                XA = transform(torch, Xtr_t, 1, q).cpu().numpy()
                XB = transform(torch, Xtr_t, 2, q).cpu().numpy()
                WAz = (repr_forward(kept, Xnp=XA) - in_mu) / in_sd
                WBz = (repr_forward(kept, Xnp=XB) - in_mu) / in_sd
            F = W_clean.shape[1]
            feat_A = counted(lambda gs: torch.stack([feat_vec(torch, tp, WAz, g) for g in gs], 1), N)
            feat_B = counted(lambda gs: torch.stack([feat_vec(torch, tp, WBz, g) for g in gs], 1), N)
            new_fn, mut_fn = (lambda r: new_vec(r, F)), (lambda r, g, s: mut_vec(r, g, s, F))
            kind = "vec"
        genomes = evolve(torch, rng, None, args.pop, args.gens, args.rounds,
                         args.cap, args.freeze, new_fn, mut_fn, feat_A, feat_B,
                         None, probe, f"q:{q}")
        if not genomes:
            log(f"[auto] question '{q}': no words evolved -> DROP"); continue
        proj = None
        # candidate representation on clean train, measure val gain
        if kind == "img":
            newc = torch.stack([feature(torch, tp, env0, g) for g in genomes], 1)
            newc_te = torch.stack([feature(torch, tp, env0, g, test=True)
                                   for g in genomes], 1)
        else:
            Wz = (W_clean - in_mu) / in_sd
            Wz_te = (W_clean_te - in_mu) / in_sd
            newc = torch.stack([feat_vec(torch, tp, Wz, g) for g in genomes], 1)
            newc_te = torch.stack([feat_vec(torch, tp, Wz_te, g) for g in genomes], 1)
        # compress this space's contribution to top-K (no width explosion): the
        # space evolves its full oscillating population, but only its top-K
        # informative directions enter the shared representation.
        if args.per_space_k and newc.shape[1] > args.per_space_k:
            cm = newc.mean(0); csd = newc.std(0) + 1e-6
            cc = (newc - cm) / csd
            cc_te = (newc_te - cm) / csd
            _, _, V = torch.linalg.svd(cc[:n_fit], full_matrices=False)
            Vk = V[:args.per_space_k].T
            newc = cc @ Vk
            newc_te = cc_te @ Vk
            proj = (cm, csd, Vk)
        sp = {"kind": kind, "genomes": genomes, "in_mu": in_mu, "in_sd": in_sd,
              "proj": proj}
        candW = newc if W_clean is None else torch.cat([W_clean, newc], 1)
        candW_te = newc_te if W_clean_te is None else torch.cat([W_clean_te, newc_te], 1)
        COST["evals"] += len(genomes) * (N + len(yte))   # clean-repr words
        v = head_val(candW)
        gain = v - cur_val
        verdict = "KEEP" if gain >= args.keep_thresh else "DROP"
        dtq = time.time() - tq
        space_times.append((q, dtq, verdict))
        log(f"[auto] question '{q}': {len(genomes)} words, val {v:.4f} "
            f"(gain {gain:+.4f}) -> {verdict}  [{dtq:.0f}s this space, "
            f"{round(time.time()-t0)}s total]")
        if gain >= args.keep_thresh:
            kept.append(sp); kept_names.append(q)
            W_clean, W_clean_te = candW, candW_te; cur_val = v

    # final test + COMPUTE COST report
    total_s = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else 0.0
    if W_clean is not None:
        acc, v = head_test(W_clean, W_clean_te)
        sizes = ",".join(str(len(k["genomes"])) for k in kept)
        log(f"[auto] FINAL STACK: {len(kept)} spaces [{sizes}], "
            f"{W_clean.shape[1]} words, val {v:.4f} TEST {acc:.4f}")
        log(f"[auto] kept questions (in order): {kept_names}")
    log("[auto] ===== TRAINING TIME + COMPUTE COST =====")
    log(f"[auto]   device            : {device_name}")
    log(f"[auto]   total train time  : {total_s:.0f}s ({total_s/60:.1f} min)")
    log(f"[auto]   peak GPU memory   : {peak_gb:.2f} GB")
    log(f"[auto]   genome-image evals: {COST['evals']/1e9:.2f} billion "
        f"(one eval = a tiny program over patch-PCA maps / prev words for 1 image)")
    log(f"[auto]   questions tried   : {len(remaining)}  "
        f"(evolved a full space for each; kept {len(kept)})")
    for q, dt, vd in space_times:
        log(f"[auto]     {q:8s} {dt:5.0f}s  {vd}")
    json.dump({"module": "autostack", "n_spaces": len(kept),
               "kept_questions": kept_names,
               "words": W_clean.shape[1] if W_clean is not None else 0,
               "test": round(acc, 4) if W_clean is not None else None,
               "device": device_name, "train_seconds": round(total_s),
               "peak_gpu_gb": round(peak_gb, 2),
               "genome_image_evals": COST["evals"],
               "per_space_seconds": [(q, round(dt)) for q, dt, _ in space_times]},
              open(os.path.join(RD, "replicate_autostack.json"), "w"))


if __name__ == "__main__":
    main()
