"""persistence_lang.py - the PERSISTENCE operator on LANGUAGE.

Port of persistence_test.py (corrupted-letter views: single 0.3835 ->
feature-accumulate 0.93) to word streams. The recurring signal is the TOPIC:
every word of a W-word window is one noisy view of it - topical words fire a
topic-detector wherever they land in the window, function words are the
per-view corruption. Test windows come from articles never seen in training
(build_topic_stream.py), so the question is the topic itself.

Arms, all evolving the SAME vec-genome detectors, same budget, genome-only
readout:
  SINGLE   - detector on ONE word of the window (no accumulation)
  VECMEAN  - detector on the MEAN embed vector (accumulate in RAW space -
             the pixel-mean analog)
  ACCUM    - detector applied to EVERY word, response MEANed over W
             (persistence: accumulate in FEATURE space - the operator)
Each arm reports its own linear anchor (ridge straight on the arm's raw
input, no genomes) per the suppression-law discipline; the fat anchor is
ridge on the full W*D concat. If ACCUM >> SINGLE and beats VECMEAN, the
letters result transfers: accumulation must be over detector RESPONSES,
not the raw signal.

  python persistence_lang.py [--smoke]
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401
import json
import os
import sys
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
CACHE = "wf_topic_stream.pt"


def _fin(torch, c):
    return torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e6, 1e6)


def run(smoke=False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    rng = np.random.default_rng(0)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    log_lines = []

    def log(m, v=True):
        log_lines.append(m); print(m, flush=True)

    blob = torch.load(os.path.join(RD, CACHE), map_location=dev)
    F_tr = torch.stack([t.float().to(dev) for t in blob["tr"]], 1)  # (N,W,D)
    F_te = torch.stack([t.float().to(dev) for t in blob["te"]], 1)
    ytr = blob["ytr"].to(dev)
    yte = blob["yte"].to(dev)
    topics = blob["topics"]
    K = len(topics)
    if smoke:
        F_tr, ytr = F_tr[:3000], ytr[:3000]
        F_te, yte = F_te[:800], yte[:800]
    Ntr, W, D = F_tr.shape
    Nte = F_te.shape[0]
    pop, gens, rounds = (48, 8, 20) if smoke else (64, 12, 60)
    log(f"[plang] {K} topics, W={W} words, D={D}, Ntr={Ntr} Nte={Nte} "
        f"(test = UNSEEN articles), chance {1 / K:.4f}")

    # z-score per dim over ALL train word positions (one shared scale, so
    # single/mean/accum read the same representation)
    flat = F_tr.reshape(-1, D)
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    F_tr = ((F_tr - mu) / sd).clamp(-8, 8)
    F_te = ((F_te - mu) / sd).clamp(-8, 8)

    single_tr, single_te = F_tr[:, W // 2], F_te[:, W // 2]   # middle word
    mean_tr, mean_te = F_tr.mean(1), F_te.mean(1)
    stack_tr = F_tr.reshape(Ntr * W, D)                        # all words
    stack_te = F_te.reshape(Nte * W, D)

    n_fit = int(Ntr * 0.8)
    yv = ytr[n_fit:]
    Yf = -torch.ones((n_fit, K), device=dev)
    Yf[torch.arange(n_fit), ytr[:n_fit]] = 1.0
    Yfull = -torch.ones((Ntr, K), device=dev)
    Yfull[torch.arange(Ntr), ytr] = 1.0

    def readout(Ftr, Fte):
        best = 0.0
        for lam in (1.0, 3.0, 10.0):
            _, a = _ridge_soft(torch, Ftr, Fte, Yfull, yte, lam=lam)
            best = max(best, a)
        return round(float(best), 4)

    def anchor(Atr, Ate, tag):
        """Linear ridge straight on a raw representation - no genomes."""
        va = _ridge_soft(torch, Atr[:n_fit], Atr[n_fit:], Yf, yv)[1]
        ta = readout(Atr, Ate)
        log(f"[anchor {tag}] val {va:.4f} TEST {ta}")
        return {"val": round(float(va), 4), "test": ta}

    frozen_by_tag = {}

    def arm(feat_tr, feat_te, bake_bank, tag):
        base0 = torch.zeros((Ntr, 0), device=dev)
        new_fn = lambda r: rk.new_vec_genome(r, D)
        mut_fn = lambda r, g, sc: rk.mutate_vec(r, g, sc, D)
        frozen, fcols = rk._evolve_space(torch, rng, pop, gens, rounds, n_fit,
                                         Yf, yv, base0, new_fn, mut_fn,
                                         feat_tr, log, True)
        if not frozen:
            log(f"[{tag}] earned nothing")
            return {"test": 0.0, "genomes": 0}
        rk.bake_gate_stats(torch, tp, frozen, bake_bank)
        frozen_by_tag[tag] = (frozen, feat_tr, feat_te)
        Ftr = torch.stack([feat_tr(g) for g in frozen], 1)
        Fte = torch.stack([feat_te(g) for g in frozen], 1)
        acc = readout(Ftr, Fte)
        log(f"[{tag}] TEST {acc} ({len(frozen)} genomes)")
        return {"test": acc, "genomes": len(frozen)}

    res = {"K": K, "W": W, "D": D, "Ntr": Ntr, "Nte": Nte,
           "topics": list(topics), "chance": round(1 / K, 4)}

    log("=== anchors (ridge on raw input, no genomes) ===")
    res["anchor_single"] = anchor(single_tr, single_te, "single word")
    res["anchor_mean"] = anchor(mean_tr, mean_te, "mean vec")
    res["anchor_concat"] = anchor(F_tr.reshape(Ntr, W * D),
                                  F_te.reshape(Nte, W * D), "concat (fat)")

    log("=== SINGLE: one word, no accumulation ===")
    res["single"] = arm(
        lambda g: _fin(torch, rk.feature_vec(torch, tp, single_tr, g)),
        lambda g: _fin(torch, rk.feature_vec(torch, tp, single_te, g)),
        single_tr, "single")
    log("=== VECMEAN: accumulate in RAW embed space, then detect ===")
    res["vecmean"] = arm(
        lambda g: _fin(torch, rk.feature_vec(torch, tp, mean_tr, g)),
        lambda g: _fin(torch, rk.feature_vec(torch, tp, mean_te, g)),
        mean_tr, "vecmean")
    log("=== ACCUM: detect per word, accumulate over W (persistence) ===")
    res["accum"] = arm(
        lambda g: _fin(torch, rk.feature_vec(torch, tp, stack_tr, g)
                       .view(Ntr, W).mean(1)),
        lambda g: _fin(torch, rk.feature_vec(torch, tp, stack_te, g)
                       .view(Nte, W).mean(1)),
        stack_tr, "accum")

    # persist the ACCUM topic model (genomes + closed-form head + norm stats)
    # so downstream consumers (topic-steered generation) can replay it as a
    # pure frozen function. All post-evolution: touches no rng state.
    if "accum" in frozen_by_tag:
        frozen, feat_tr_fn, _ = frozen_by_tag["accum"]
        Ftr = torch.stack([feat_tr_fn(g) for g in frozen], 1)
        best = (1.0, -1.0)
        for lam in (1.0, 3.0, 10.0):
            _, a = _ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf, yv, lam=lam)
            if a > best[1]:
                best = (lam, float(a))
        hm, hs = Ftr.mean(0), Ftr.std(0) + 1e-6
        A = torch.hstack([(Ftr - hm) / hs, torch.ones(Ntr, 1, device=dev)])
        G = (A.T @ A).double() + best[0] * torch.eye(
            Ftr.shape[1] + 1, device=dev, dtype=torch.float64)
        Wm = torch.linalg.solve(G, (A.T @ Yfull).double()).float()
        with open(os.path.join(RD, "kid_plang_model.json"), "w") as f:
            json.dump({"genomes": frozen, "topics": list(topics),
                       "embed_mu": mu.tolist(), "embed_sd": sd.tolist(),
                       "head_mu": hm.tolist(), "head_sd": hs.tolist(),
                       "head_W": Wm.tolist(), "lam": best[0], "W": W, "D": D,
                       "val_acc": best[1]}, f)
        log(f"[plang] ACCUM model saved: kid_plang_model.json "
            f"({len(frozen)} genomes, head lam {best[0]}, val {best[1]:.4f})")

    res["seconds"] = round(time.time() - t0)
    with open(os.path.join(RD, "persistence_lang_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("PLANG RESULT: " + json.dumps(res))
    s, m, a = (res["single"]["test"], res["vecmean"]["test"],
               res["accum"]["test"])
    log(f"VERDICT: single {s} | vec-mean {m} | ACCUM(persistence) {a} "
        f"| anchors single {res['anchor_single']['test']} "
        f"mean {res['anchor_mean']['test']} "
        f"concat {res['anchor_concat']['test']} -> "
        f"{'persistence TRANSFERS to language' if a > s and a >= m else 'accumulation does NOT transfer as-is'}")
    print("PLANG DONE", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
