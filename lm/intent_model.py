"""intent_model.py - the INTENT SPECIALIST (user's call, and genome #1's
homecoming: punctuation as free ground truth, 2026-07-09's opener lesson
now with the specialist machinery).

One model, one question: WHAT KIND OF RESPONSE does this prompt expect?
The label is free - every sentence already carries its mark:
  .  statement -> declarative continuation
  ?  question  -> an answer
  !  exclaim   -> emphasis
Substrate: first T=10 words as embed_rs vectors; TEMPORAL genomes (intent
is positional - openers, auxiliary inversion). Genome-only head, balanced
classes, BALANCED accuracy is the headline (the 07-09 collapse lesson).

Also builds the union's steering table for free from sentence ADJACENCY:
the words of the sentence FOLLOWING each mark are response evidence ->
per-target-word log-odds per intent (train slice), plus a DISJOINT
held-out slice for judging (the steering table must not grade itself).

  python lm/intent_model.py [--smoke]
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401

import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
import radial_stack as rk
import radial_temporal as rt
from radial_lm import _clean

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_ROOT, "radial_data")
CORPUS = os.path.join(_ROOT, "corpora", "combined", "combined_corpus.txt")
T = 10
MARKS = {".": 0, "?": 1, "!": 2}
NAMES = ["statement", "question", "exclaim"]


def sentences(seek, nbytes):
    """(words, mark_class, next_words) triples from raw corpus text."""
    with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(seek)
        raw = f.read(nbytes)
    parts = re.split(r"([.?!])", raw)
    out = []
    for i in range(0, len(parts) - 3, 2):
        body = _clean(parts[i]).split()
        mark = parts[i + 1]
        nxt = _clean(parts[i + 2]).split()
        if mark in MARKS and 4 <= len(body) <= 40:
            out.append((body[:T], MARKS[mark], nxt[:T]))
    return out


def build_dataset(vocab, E, cap, seek=5_000_000, nbytes=55_000_000):
    sents = sentences(seek, nbytes)
    rng = np.random.default_rng(0)
    rng.shuffle(sents)
    by_c = {0: [], 1: [], 2: []}
    for body, c, _ in sents:
        if len(by_c[c]) >= cap:
            continue
        ids = [vocab.get(w) for w in body]
        if sum(i is not None for i in ids) >= max(4, len(ids) - 2):
            by_c[c].append((ids, c))
    n = min(len(v) for v in by_c.values())
    data = by_c[0][:n] + by_c[1][:n] + by_c[2][:n]
    rng.shuffle(data)
    X = np.zeros((len(data), T, E.shape[1]), np.float32)
    y = np.zeros(len(data), np.int64)
    for i, (ids, c) in enumerate(data):
        for t, j in enumerate(ids):
            if j is not None:
                X[i, t] = E[j]
        y[i] = c
    return X, y, n


def response_counts(targets, seek, nbytes):
    """Per-target-word counts in the sentence FOLLOWING each mark."""
    tset = set(targets)
    cnt = {w: [0, 0, 0] for w in targets}
    tot = [0, 0, 0]
    for _, c, nxt in sentences(seek, nbytes):
        for w in nxt:
            if w in tset:
                cnt[w][c] += 1
                tot[c] += 1
    return cnt, tot


def run(smoke=False):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    def log(m, v=True):
        print(m, flush=True)

    ze = np.load(os.path.join(RD, "embed_rs.npz"), allow_pickle=True)
    vocab = {str(w): i for i, w in enumerate(ze["vocab"])}
    E = ze["feat"].astype(np.float32)
    D = E.shape[1]
    cap = 2000 if smoke else 40000
    X, y, per_class = build_dataset(vocab, E, cap)
    n_test = min(6000, len(X) // 6)
    log(f"[intent] {len(X)} sentences ({per_class}/class balanced), "
        f"T={T}, D={D}, test {n_test}")

    flat = X[:-n_test].reshape(-1, D)
    cmu, csd = flat.mean(0), flat.std(0) + 1e-6
    Xz = np.clip((X - cmu) / csd, -8, 8).astype(np.float32)
    Xtr, ytr = Xz[:-n_test], y[:-n_test]
    Xte, yte = Xz[-n_test:], y[-n_test:]
    Ntr, Nte = len(ytr), len(yte)
    F_tr = torch.tensor(Xtr, device=dev)
    F_te = torch.tensor(Xte, device=dev)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 3), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, 3), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    cat_tr = F_tr.reshape(Ntr, T * D)
    _, anchor = _ridge_soft(torch, cat_tr[:n_fit], cat_tr[n_fit:], Yf, yv)
    log(f"[intent] linear anchor (position concat ridge): val {anchor:.4f} "
        f"(chance 0.333)")

    pop, gens, rounds = (48, 8, 20) if smoke else (96, 12, 200)
    new_fn = lambda r: rt.new_temporal(r, D, T)
    mut_fn = lambda r, g, sc: rt.mutate_temporal(r, g, sc, D, T)
    feat_tr = lambda g: rt._finite(torch, rt.temporal_feat(torch, F_tr, g))
    feat_te = lambda g: rt._finite(torch, rt.temporal_feat(torch, F_te, g))
    base0 = torch.zeros((Ntr, 0), device=dev)
    frozen, fcols = rk._evolve_space(torch, rng, pop, gens, rounds, n_fit,
                                     Yf, yv, base0, new_fn, mut_fn,
                                     feat_tr, log, True)
    if not frozen:
        log("[intent] earned nothing - no model saved")
        return
    Ftr = torch.stack(fcols, 1)
    Fte = torch.stack([feat_te(g) for g in frozen], 1)
    fmu, fsd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = ((Ftr - fmu) / fsd).clamp(-8, 8)
    Fte = ((Fte - fmu) / fsd).clamp(-8, 8)

    best = (1.0, -1.0)
    for lam in (1.0, 3.0, 10.0):
        _, a = _ridge_soft(torch, Ftr[:n_fit], Ftr[n_fit:], Yf, yv, lam=lam)
        if a > best[1]:
            best = (lam, float(a))
    hm2 = Ftr.mean(0); hs2 = Ftr.std(0) + 1e-6
    A = torch.hstack([(Ftr - hm2) / hs2, torch.ones(Ntr, 1, device=dev)])
    G = (A.T @ A).double() + best[0] * torch.eye(Ftr.shape[1] + 1, device=dev,
                                                dtype=torch.float64)
    Wm = torch.linalg.solve(G, (A.T @ Yfull).double()).float()
    sc = torch.hstack([(Fte - hm2) / hs2,
                       torch.ones(Nte, 1, device=dev)]) @ Wm
    preds = sc.argmax(1)
    per_recall = {}
    for c, nm in enumerate(NAMES):
        m = yte_t == c
        per_recall[nm] = round(float((preds[m] == c).float().mean()), 4)
    bal = round(float(np.mean(list(per_recall.values()))), 4)
    top1 = round(float((preds == yte_t).float().mean()), 4)
    log(f"[intent] TEST top1 {top1} BALANCED {bal} per-class {per_recall} "
        f"({len(frozen)} temporal genomes, anchor {anchor:.4f})")

    with open(os.path.join(RD, "kid_intent_model.json"), "w") as f:
        json.dump({"genomes": frozen, "T": T, "D": D, "names": NAMES,
                   "cmu": cmu.tolist(), "csd": csd.tolist(),
                   "fmu": fmu.tolist(), "fsd": fsd.tolist(),
                   "head_mu": hm2.tolist(), "head_sd": hs2.tolist(),
                   "head_W": Wm.tolist(), "lam": best[0],
                   "balanced_acc": bal, "test_acc": top1,
                   "per_class": per_recall,
                   "anchor": round(float(anchor), 4)}, f)

    # response steering table (train slice) + judge counts (held-out)
    z = np.load(os.path.join(RD, "lm_word.npz"), allow_pickle=True)
    targets = [str(w) for w in z["targets"]]
    log("[intent] counting response words (train slice 60-100MB)")
    cnt, tot = response_counts(targets, 60_000_000, 40_000_000)
    with open(os.path.join(RD, "intent_response_counts.json"), "w") as f:
        json.dump({"counts": cnt, "totals": tot, "names": NAMES}, f)
    log("[intent] counting judge words (held-out slice 100-120MB)")
    jcnt, jtot = response_counts(targets, 100_000_000, 20_000_000)
    with open(os.path.join(RD, "intent_judge_counts.json"), "w") as f:
        json.dump({"counts": jcnt, "totals": jtot, "names": NAMES}, f)

    res = {"test_acc": top1, "balanced_acc": bal, "per_class": per_recall,
           "anchor": round(float(anchor), 4), "chance": 0.3333,
           "n_genomes": len(frozen), "Ntr": Ntr, "Nte": Nte,
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "intent_model_result.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("INTENT RESULT: " + json.dumps(res))
    print("INTENT DONE", flush=True)


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
