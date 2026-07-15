"""radial_lm.py — the radial stack on language, ISOLATED: character IDs in,
sequential composition tested, nothing else.

One problem at a time (the glyph-frame variant — vision AND language at
once — is preserved as stage 2 in lm_radial_glyph.json / git history; it
reached 0.2950 vs bigram 0.2894). Here the model gets character IDENTITY
for free: each window is T one-hot characters, (position x char) = T*27
binary channels. The label is the next character (27 classes).

What the ladder means:
  one-hot ridge — the closed-form head on the raw channels, NO genomes.
        This is the additive positional model; it literally contains the
        bigram table (slot-T weights) plus additive contributions from
        earlier slots. It CANNOT represent interactions ("t then h").
  genome spaces — vec-grammar genomes fold channels with mult / min /
        |a-b| and gates: min(t@5, h@6) IS a trigram detector. Stacked
        emergent-cap spaces, scalar hand-off (there is no spatial
        structure to preserve), raw one-hot skip bank in every space.
  n-gram ceilings — fit on the FULL train region (their best shot).

The claim under test: the genome layer learns COMPOSITIONAL CONTEXT — it
must beat the one-hot ridge (= beat additivity) and climb from the bigram
ceiling toward the trigram/4-gram marks. Test windows come from a
DISJOINT corpus region. No gradients anywhere.

Exports radial_data/lm_radial.json (+ lm_model.json) for the LM page.
"""
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(_HERE, "corpora", "combined", "combined_corpus.txt")
T = 6                                    # context characters per window
CHARS = "abcdefghijklmnopqrstuvwxyz "
N_CLASSES = len(CHARS)                   # 27
_IDX = {c: i for i, c in enumerate(CHARS)}


def _clean(raw):
    out = []
    prev_space = True
    for ch in raw.lower():
        if ch in _IDX and ch != " ":
            out.append(ch)
            prev_space = False
        elif not prev_space:
            out.append(" ")
            prev_space = True
    return "".join(out)


def _load_regions(train_mb=2.0, test_mb=0.5):
    """Two disjoint corpus regions (train first, test after a 1MB gap)."""
    with open(CORPUS, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(10_000_000)               # skip headers/front matter
        train = _clean(f.read(int(train_mb * 1_000_000)))
        f.seek(10_000_000 + int(train_mb * 1_000_000) + 1_000_000)
        test = _clean(f.read(int(test_mb * 1_000_000)))
    return train, test


def _windows(text, n, rng):
    ctx = np.zeros((n, T), np.int8)
    y = np.zeros(n, np.int64)
    pos = rng.integers(0, len(text) - T - 1, n)
    for i, p in enumerate(pos):
        for f in range(T):
            ctx[i, f] = _IDX[text[p + f]]
        y[i] = _IDX[text[p + T]]
    return ctx, y


def make_lm_data(n_train=20000, n_test=5000, seed=0,
                 path=os.path.join(_HERE, "radial_data", "lm_ids.npz")):
    rng = np.random.default_rng(seed)
    train_text, test_text = _load_regions()
    ctx_tr, y_tr = _windows(train_text, n_train, rng)
    ctx_te, y_te = _windows(test_text, n_test, rng)
    np.savez(path, ctx_tr=ctx_tr, ytr=y_tr, ctx_te=ctx_te, yte=y_te)
    print(f"lm_ids: {n_train}/{n_test} x {T}-char windows (IDs only, "
          f"disjoint regions) -> {path}", flush=True)
    return path


def ngram_baselines(ctx_te, y_te):
    """Greedy top-1 n-gram predictors with backoff, fit on the FULL train
    region — the honest ceilings."""
    train_text, _ = _load_regions()
    ids = np.array([_IDX[c] for c in train_text], np.int32)
    counts = [np.zeros((N_CLASSES ** k, N_CLASSES), np.int64) for k in range(4)]
    for k in range(4):
        if k == 0:
            np.add.at(counts[0], (np.zeros(len(ids) - 1, np.int64), ids[1:]), 1)
        else:
            kk = np.zeros(len(ids) - k, np.int64)
            for j in range(k):
                kk = kk * N_CLASSES + ids[j:len(ids) - k + j]
            np.add.at(counts[k], (kk, ids[k:]), 1)
    out = {}
    for k, name in enumerate(["unigram", "bigram", "trigram", "4-gram"]):
        hits = 0
        for i in range(len(y_te)):
            kb = k
            while True:
                kk = 0
                for j in range(T - kb, T):
                    kk = kk * N_CLASSES + int(ctx_te[i, j])
                row = counts[kb][kk]
                if row.sum() > 0 or kb == 0:
                    break
                kb -= 1                  # backoff to shorter context
            hits += int(row.argmax() == y_te[i])
        out[name] = round(hits / len(y_te), 4)
    return out


def _onehot(torch, dev, ctx):
    """(N, T) ids -> (N, T*27) float one-hot channels."""
    N = len(ctx)
    B = torch.zeros((N, T * N_CLASSES), device=dev)
    idx = torch.tensor(ctx.astype(np.int64), device=dev)
    for f in range(T):
        B[torch.arange(N, device=dev), f * N_CLASSES + idx[:, f]] = 1.0
    return B


def run(pop_size=64, gens=12, max_rounds=400, seed=5, max_spaces=16,
        out_path=None, verbose=True):
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    z = np.load(os.path.join(_HERE, "radial_data", "lm_ids.npz"))
    ctx_tr, ytr = z["ctx_tr"], z["ytr"]
    ctx_te, yte = z["ctx_te"], z["yte"]
    Ntr, Nte = len(ytr), len(yte)
    B0_tr = _onehot(torch, dev, ctx_tr)          # (Ntr, 162)
    B0_te = _onehot(torch, dev, ctx_te)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, N_CLASSES), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, N_CLASSES), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    # the no-genome anchor: additive positional model (contains bigram)
    oh_val = _ridge_soft(torch, B0_tr[:n_fit], B0_tr[n_fit:], Yf, yv)[1]
    oh_test = max(_ridge_soft(torch, B0_tr, B0_te, Yfull, yte_t, lam=lam)[1]
                  for lam in (1.0, 3.0, 10.0, 30.0))
    print(f"one-hot ridge (no genomes): val {oh_val:.4f} test {oh_test:.4f}",
          flush=True)

    def log(m, v=True):
        if v:
            print(m, flush=True)

    spaces, all_tr, all_te = [], [], []
    space_genomes = []
    # every space reads [raw one-hots  |  previous space's outputs] — the
    # raw skip bank keeps character identity reachable at any depth
    bank_tr, bank_te = B0_tr, B0_te
    # the border ridge starts FROM the one-hot ridge: genomes only earn by
    # beating additivity, never by re-deriving it
    base_cols_tr = [B0_tr[:, j] for j in range(B0_tr.shape[1])]
    base_cols_te = [B0_te[:, j] for j in range(B0_te.shape[1])]
    all_tr.extend(base_cols_tr); all_te.extend(base_cols_te)
    val_prev = oh_val

    for si in range(max_spaces):
        C = bank_tr.shape[1]
        new_fn = lambda r: rk.new_vec_genome(r, C)
        mut_fn = lambda r, g, sc: rk.mutate_vec(r, g, sc, C)
        feat_tr = lambda g: rk.feature_vec(torch, tp, bank_tr, g)
        feat_te = lambda g: rk.feature_vec(torch, tp, bank_te, g)
        src = (f"{C} channels (raw one-hots"
               + (f" + space {si-1} outputs)" if si else ")"))
        log(f"  [space {si}] opening — reads {src}")
        base_prev = torch.stack(all_tr, 1)
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens, max_rounds,
                                         n_fit, Yf, yv, base_prev, new_fn, mut_fn,
                                         feat_tr, log, verbose)
        if not frozen:
            log(f"  [space {si}] produced nothing — stop")
            break
        space_genomes.append(frozen)
        fte = [feat_te(g) for g in frozen]
        all_tr.extend(fcols); all_te.extend(fte)
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:], Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "source": src, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4)})
        log(f"  [space {si}] FULL: {len(frozen)} genomes, val {val_now:.4f} "
            f"(+{gain:.4f}) ({round(time.time()-t0)}s)")
        f_tr = torch.stack(fcols, 1)
        f_te2 = torch.stack(fte, 1)
        zmu, zsd = f_tr.mean(0), f_tr.std(0) + 1e-6
        bank_tr = torch.cat([B0_tr, (f_tr - zmu) / zsd], 1)
        bank_te = torch.cat([B0_te, (f_te2 - zmu) / zsd], 1)
        val_prev = val_now
        if si > 0 and gain < rk.MIN_SPACE_GAIN:
            log(f"  [space {si}] gain {gain:.4f} < {rk.MIN_SPACE_GAIN} — done")
            break
        if os.path.exists(_STOP):
            log("[radial-lm] STOP lever pulled")
            break

    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
        best = max(best, acc)

    baselines = ngram_baselines(ctx_te, yte)
    n_genomes = sum(len(sp) for sp in space_genomes)
    out = {"phase": "radial-lm (IDs — isolated sequential composition)",
           "frames": T, "n_train": Ntr, "n_test": Nte, "n_classes": N_CLASSES,
           "chance": round(1.0 / N_CLASSES, 4),
           "n_spaces": len(spaces), "space_caps": [s["n_frozen"] for s in spaces],
           "test_acc": round(best, 4),
           "onehot_ridge_test": round(float(oh_test), 4),
           "onehot_ridge_val": round(float(oh_val), 4),
           "val_final": spaces[-1]["val_after"] if spaces else round(float(oh_val), 4),
           "spaces": spaces, "classes": list(CHARS), "n_genomes": n_genomes,
           "baselines": baselines,
           "task": f"predict the NEXT character from {T} char IDs (one-hot "
                   "channels). The one-hot ridge is the no-genome additive "
                   "anchor (contains the bigram table); genomes only score "
                   "by composing interactions across positions",
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(_HERE, "radial_data", "lm_radial.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    model = {"frames": T, "spaces": space_genomes, "label": "next-char",
             "input": "onehot-ids"}
    with open(os.path.join(_HERE, "radial_data", "lm_model.json"), "w") as f:
        json.dump(model, f)
    print(f"[radial-lm] DONE: {len(spaces)} spaces {out['space_caps']}, "
          f"one-hot ridge {oh_test:.4f}, stack TEST {best:.4f} "
          f"vs baselines {baselines} ({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if not os.path.exists(os.path.join(_HERE, "radial_data", "lm_ids.npz")) \
            or "--regen" in sys.argv:
        make_lm_data()
    run()
