"""radial_kid.py - the ENGLISH CURRICULUM: models learn language like a
kid (user's design). Each stage is its own model, trained on top of the
previous stage's frozen genomes (the continue-train warm start):

  Stage A  letters  - identify single rendered letters (26 classes).
                      Pure perception; the gate is near-perfect accuracy.
  Stage B  words    - a word is a STRIP of letter tiles; identify the
                      word (V classes) with stage A frozen as the eye.
  Stage C  cloze    - short phrase with a blank; name the missing word.
  Stage D  autoregression - next word, on top of C.

No stage advances until the one below earns its gate. Pixels are the
whole environment; no embeddings, tables, or one-hots anywhere in the
curriculum. Stage A here.

    python radial_kid.py A
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, new_genome, mutate
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _font(size, cache={}):
    from PIL import ImageFont
    if size not in cache:
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                     "arial.ttf"):
            try:
                cache[size] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            cache[size] = ImageFont.load_default()
    return cache[size]


def render_letter(ch, rng, size=None):
    """One 32x32 letter tile: random size, position jitter, noise later."""
    from PIL import Image, ImageDraw
    im = Image.new("L", (32, 32), 0)
    d = ImageDraw.Draw(im)
    f = _font(int(size or rng.integers(14, 27)))
    x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=f)
    dx = int(rng.integers(-4, 5))
    dy = int(rng.integers(-4, 5))
    d.text(((32 - (x1 - x0)) / 2 - x0 + dx, (32 - (y1 - y0)) / 2 - y0 + dy),
           ch, fill=255, font=f)
    return np.asarray(im, np.float32) / 255.0


def make_letters(n_train=20000, n_test=5000, seed=0, noise=0.05,
                 path=os.path.join(RD, "kid_letters.npz")):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = rng.integers(0, 26, n)
    X = np.zeros((n, 32, 32), np.float32)
    for i in range(n):
        X[i] = render_letter(LETTERS[y[i]], rng)
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    X8 = (np.repeat(X[..., None], 3, axis=3) * 255).astype(np.uint8)
    np.savez(path, Xtr=X8[:n_train], ytr=y[:n_train].astype(np.int64),
             Xte=X8[n_train:], yte=y[n_train:].astype(np.int64))
    print(f"kid_letters: {n_train}/{n_test} single-letter tiles "
          f"(size 14-26px, jitter +-4, noise {noise}) -> {path}", flush=True)
    return path


def stage_a(pop_size=64, gens=12, max_rounds=200, seed=5, max_spaces=6,
            grid_size=8, verbose=True):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    rk.GRID = int(grid_size)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    z = np.load(os.path.join(RD, "kid_letters.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    Ntr, Nte = len(ytr), len(yte)
    env = Env(torch, dev, Xtr, Xte, max_cached=6)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 26), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, 26), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    log_lines = []

    def log(m, v=True):
        log_lines.append(m)
        if v:
            print(m, flush=True)

    G = rk.GRID
    spaces, all_tr, all_te, space_genomes = [], [], [], []
    prev_tr = prev_te = None
    val_prev = 0.0
    for si in range(max_spaces):
        base_prev = (torch.stack(all_tr, 1) if all_tr
                     else torch.zeros((Ntr, 0), device=dev))
        if si == 0:
            new_fn = new_genome
            mut_fn = lambda r, g, sc: mutate(r, g, sc)
            feat_tr = lambda g: rk.feature_r0(torch, tp, env, g)
            feat_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True)
            g_tr = lambda g: rk.feature_r0(torch, tp, env, g, want_grid=True)
            g_te = lambda g: rk.feature_r0(torch, tp, env, g, test=True,
                                           want_grid=True)
        else:
            C = prev_tr.shape[1]
            new_fn = lambda r: rk.new_grid_genome(r, C)
            mut_fn = lambda r, g, sc: rk.mutate_grid_g(r, g, sc, C)
            feat_tr = lambda g: rk.feature_grid_g(torch, tp, prev_tr, g)
            feat_te = lambda g: rk.feature_grid_g(torch, tp, prev_te, g)
            g_tr = lambda g: rk.feature_grid_g(torch, tp, prev_tr, g,
                                               want_grid=True)
            g_te = lambda g: rk.feature_grid_g(torch, tp, prev_te, g,
                                               want_grid=True)
        log(f"  [A space {si}] opening")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_tr, log, verbose)
        if not frozen:
            break
        space_genomes.append(frozen)
        all_tr.extend(fcols)
        all_te.extend(feat_te(g) for g in frozen)
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                                 Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4)})
        log(f"  [A space {si}] FULL: {len(frozen)} genomes, val "
            f"{val_now:.4f} (+{gain:.4f}) ({round(time.time()-t0)}s)")
        if si == 0:
            prev_tr = torch.cat([g_tr(g).unsqueeze(1) if g_tr(g).dim() == 3
                                 else g_tr(g) for g in []], 1) \
                if False else torch.stack([g_tr(g) for g in frozen], 1).half()
            prev_te = torch.stack([g_te(g) for g in frozen], 1).half()
        else:
            prev_tr = torch.stack([g_tr(g) for g in frozen], 1).half()
            prev_te = torch.stack([g_te(g) for g in frozen], 1).half()
        val_prev = val_now
        if si > 0 and gain < rk.MIN_SPACE_GAIN:
            break
        if os.path.exists(_STOP):
            break

    if not all_tr:
        print("[kid A] nothing earned - stop", flush=True)
        return None
    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best1 = 0.0
    for lam in (1.0, 3.0, 10.0):
        _, a = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t, lam=lam)
        best1 = max(best1, a)

    from anim_infer import count_params
    gp = sum(count_params(g) for sp in space_genomes for g in sp)
    n_genomes = sum(len(sp) for sp in space_genomes)
    out = {"phase": "curriculum stage A: letters", "n_classes": 26,
           "chance": round(1 / 26, 4), "n_train": Ntr, "n_test": Nte,
           "test_acc": round(best1, 4),
           "val_final": spaces[-1]["val_after"] if spaces else 0,
           "n_spaces": len(spaces),
           "space_caps": [s["n_frozen"] for s in spaces],
           "spaces": spaces, "n_genomes": n_genomes, "genome_params": gp,
           "head_params": (n_genomes + 1) * 26,
           "total_params": gp + (n_genomes + 1) * 26,
           "task": "identify single rendered letters (26 classes; size "
                   "14-26px, position jitter, brightness + noise). The "
                   "curriculum's first gate: the eye must work before "
                   "words exist.",
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "kid_stageA.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, "kid_modelA.json"), "w") as f:
        json.dump({"grid": G, "spaces": space_genomes,
                   "label": "letter", "stage": "A"}, f)
    rk._record_run(
        {"env": "kid-stageA", "n_classes": 26, "n_train": Ntr,
         "pop_size": pop_size, "gens": gens, "seed": seed},
        [{"round": s["space"], "added": s["n_frozen"],
          "val_acc": s["val_after"], "n": s["n_frozen"]} for s in spaces],
        {"test_acc": round(best1, 4), "n_frozen_total": n_genomes,
         "total_params": out["total_params"]},
        log_lines, ["lm", "kid", "curriculum", "radial"])
    print(f"[kid A] DONE: {len(spaces)} spaces {out['space_caps']}, "
          f"TEST {best1:.4f} ({round(time.time()-t0)}s)", flush=True)
    return out




# -- Stage B: words - a word is a strip of letter tiles -----------------
L_MAX = 8                                # letters per word (padded)
V_B = 500                                # word vocabulary


def make_words_b(n_train=25000, n_test=5000, seed=0, noise=0.05,
                 path=os.path.join(RD, "kid_words.npz")):
    from radial_lm import _clean
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        f.seek(10_000_000)
        toks = _clean(f.read(4_000_000)).split()
    cnt = {}
    for w in toks:
        if 1 <= len(w) <= L_MAX:
            cnt[w] = cnt.get(w, 0) + 1
    vocab = sorted(cnt, key=cnt.get, reverse=True)[:V_B]
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    y = rng.integers(0, V_B, n)
    X = np.zeros((n, L_MAX, 32, 32), np.float32)
    for i in range(n):
        for s, ch in enumerate(vocab[y[i]]):
            X[i, s] = render_letter(ch, rng)
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    X8 = (np.repeat(X[..., None], 3, axis=4) * 255).astype(np.uint8)
    np.savez(path, Xtr=X8[:n_train], ytr=y[:n_train].astype(np.int64),
             Xte=X8[n_train:], yte=y[n_train:].astype(np.int64),
             vocab=np.array(vocab))
    print(f"kid_words: {n_train}/{n_test} words as {L_MAX}-letter strips "
          f"(V={V_B}) -> {path}", flush=True)
    return path


def stage_b(pop_size=64, gens=12, max_rounds=300, seed=7, max_spaces=8,
            grid_size=8, verbose=True):
    """Stage A frozen as the eye; new spaces compose letters -> word."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    rk.GRID = int(grid_size)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    with open(os.path.join(RD, "kid_modelA.json")) as f:
        A = json.load(f)["spaces"][0]            # the eye: R0 letter genomes
    z = np.load(os.path.join(RD, "kid_words.npz"), allow_pickle=True)
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    Ntr, Nte = len(ytr), len(yte)
    env = Env(torch, dev, Xtr.reshape(Ntr * L_MAX, 32, 32, 3),
              Xte.reshape(Nte * L_MAX, 32, 32, 3), max_cached=6)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, V_B), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, V_B), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    log_lines = []

    def log(m, v=True):
        log_lines.append(m)
        if v:
            print(m, flush=True)

    G = rk.GRID
    log(f"[B] replaying the Stage-A eye: {len(A)} frozen letter genomes")
    a_tr = [rk.feature_r0(torch, tp, env, g).view(Ntr, L_MAX).mean(1)
            for g in A]
    a_te = [rk.feature_r0(torch, tp, env, g, test=True).view(Nte, L_MAX)
            .mean(1) for g in A]
    prev_tr = torch.cat([rk.feature_r0(torch, tp, env, g, want_grid=True)
                         .view(Ntr, L_MAX, G, G) for g in A], 1).half()
    prev_te = torch.cat([rk.feature_r0(torch, tp, env, g, test=True,
                         want_grid=True).view(Nte, L_MAX, G, G)
                         for g in A], 1).half()
    log(f"[B] eye bank: {prev_tr.shape[1]} channels "
        f"({len(A)} letter genomes x {L_MAX} slots)")

    all_tr, all_te = list(a_tr), list(a_te)
    spaces, space_genomes = [], []
    base_all = torch.stack(all_tr, 1)
    _, val_prev = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                              Yf, yv)
    log(f"[B] orderless letter-bag baseline (eye only): val {val_prev:.4f}")

    for si in range(max_spaces):
        C = prev_tr.shape[1]
        new_fn = lambda r: rk.new_grid_genome(r, C)
        mut_fn = lambda r, g, sc: rk.mutate_grid_g(r, g, sc, C)
        feat_tr = lambda g: rk.feature_grid_g(torch, tp, prev_tr, g)
        feat_te = lambda g: rk.feature_grid_g(torch, tp, prev_te, g)
        g_tr = lambda g: rk.feature_grid_g(torch, tp, prev_tr, g,
                                           want_grid=True)
        g_te = lambda g: rk.feature_grid_g(torch, tp, prev_te, g,
                                           want_grid=True)
        base_prev = torch.stack(all_tr, 1)
        log(f"  [B space {si}] opening - bank {C} channels")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_tr, log, verbose)
        if not frozen:
            log(f"  [B space {si}] produced nothing - stop")
            break
        space_genomes.append(frozen)
        all_tr.extend(fcols)
        all_te.extend(feat_te(g) for g in frozen)
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                                 Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4)})
        log(f"  [B space {si}] FULL: {len(frozen)} genomes, val "
            f"{val_now:.4f} (+{gain:.4f}) ({round(time.time()-t0)}s)")
        prev_tr = torch.stack([g_tr(g) for g in frozen], 1).half()
        prev_te = torch.stack([g_te(g) for g in frozen], 1).half()
        val_prev = val_now
        if si > 0 and gain < rk.MIN_SPACE_GAIN:
            break
        if os.path.exists(_STOP):
            break

    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best1 = best5 = 0.0
    for lam in (1.0, 3.0, 10.0):
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        Am = torch.hstack([(Ftr - mu) / sd, torch.ones(Ntr, 1, device=dev)])
        Gm = (Am.T @ Am).double() + lam * torch.eye(
            Ftr.shape[1] + 1, device=dev, dtype=torch.float64)
        Wm = torch.linalg.solve(Gm, (Am.T @ Yfull).double()).float()
        sc = torch.hstack([(Fte - mu) / sd,
                           torch.ones(Nte, 1, device=dev)]) @ Wm
        a1 = float((sc.argmax(1) == yte_t).float().mean())
        a5 = float((sc.topk(5, 1).indices == yte_t.view(-1, 1))
                   .any(1).float().mean())
        if a1 > best1:
            best1, best5 = a1, a5

    from anim_infer import count_params
    gp = (sum(count_params(g) for g in A)
          + sum(count_params(g) for sp in space_genomes for g in sp))
    n_genomes = len(A) + sum(len(sp) for sp in space_genomes)
    out = {"phase": "curriculum stage B: words from letter strips",
           "n_classes": V_B, "chance": round(1 / V_B, 4),
           "n_train": Ntr, "n_test": Nte,
           "test_acc": round(best1, 4), "test_top5": round(best5, 4),
           "val_final": spaces[-1]["val_after"] if spaces else None,
           "n_spaces": 1 + len(spaces),
           "space_caps": [len(A)] + [s["n_frozen"] for s in spaces],
           "spaces": spaces, "n_genomes": n_genomes, "genome_params": gp,
           "head_params": (len(all_tr) + 1) * V_B,
           "total_params": gp + (len(all_tr) + 1) * V_B,
           "task": ("identify the word (V=%d) from its strip of rendered "
                    "letter tiles. Stage A's letter genomes are FROZEN as "
                    "the eye; new spaces compose letters into words via "
                    "the (letter-genome x slot) hand-off.") % V_B,
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "kid_stageB.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, "kid_modelB.json"), "w") as f:
        json.dump({"grid": G, "eye": A, "spaces": space_genomes,
                   "label": "word", "stage": "B"}, f)
    rk._record_run(
        {"env": "kid-stageB", "n_classes": V_B, "n_train": Ntr,
         "pop_size": pop_size, "gens": gens, "seed": seed},
        [{"round": s["space"], "added": s["n_frozen"],
          "val_acc": s["val_after"], "n": s["n_frozen"]} for s in spaces],
        {"test_acc": round(best1, 4), "test_top5": round(best5, 4),
         "n_frozen_total": n_genomes, "total_params": out["total_params"]},
        log_lines, ["lm", "kid", "curriculum", "radial"])
    print("[kid B] DONE: eye %d + %d new genomes, TEST top1 %.4f top5 %.4f"
          " (%ds)" % (len(A), sum(len(s) for s in space_genomes), best1,
                      best5, round(time.time() - t0)), flush=True)
    return out




# -- Stage C: cloze - name the blanked word in a short phrase -----------
P_C = 4                                  # words per phrase
ROW_CH = 10000                           # phrases per B-chain row chunk
#                                          (bounds GPU peak in word_feats;
#                                          per-row chain, so exact)


def make_cloze_c(n_train=20000, n_test=5000, seed=0, noise=0.05,
                 path=os.path.join(RD, "kid_cloze.npz")):
    from radial_lm import _clean
    zb = np.load(os.path.join(RD, "kid_words.npz"), allow_pickle=True)
    vocab = [str(w) for w in zb["vocab"]]
    v2i = {w: i for i, w in enumerate(vocab)}
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        f.seek(10_000_000)
        toks = _clean(f.read(8_000_000)).split()
    ze = np.load(os.path.join(RD, "embed_rs.npz"), allow_pickle=True)
    rs_vocab = {str(w): i for i, w in enumerate(ze["vocab"])}
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = np.zeros((n, P_C, L_MAX, 32, 32), np.float32)
    y = np.zeros(n, np.int64)
    blank = np.zeros(n, np.int64)
    ctx = np.full((n, P_C), -1, np.int32)     # RS-vocab ids; blank/OOV -1
    got = 0
    order = rng.permutation(len(toks) - P_C)
    for p in order:
        wds = toks[p:p + P_C]
        if any(len(w) > L_MAX for w in wds):
            continue
        b = int(rng.integers(0, P_C))
        if wds[b] not in v2i:
            continue
        y[got] = v2i[wds[b]]
        blank[got] = b
        for s, w in enumerate(wds):
            if s == b:
                continue                      # the blank: empty tiles
            ctx[got, s] = rs_vocab.get(w, -1)
            for l, ch in enumerate(w):
                X[got, s, l] = render_letter(ch, rng)
        got += 1
        if got == n:
            break
    if got < n:
        raise RuntimeError(f"only {got}/{n} phrases")
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    X8 = (X * 255).astype(np.uint8)
    np.savez(path, Xtr=X8[:n_train], ytr=y[:n_train],
             Xte=X8[n_train:], yte=y[n_train:],
             btr=blank[:n_train], bte=blank[n_train:],
             ctr=ctx[:n_train], cte=ctx[n_train:])
    print(f"kid_cloze: {n_train}/{n_test} {P_C}-word phrases, one blank, "
          f"target in V={V_B} -> {path}", flush=True)
    return path


# -- Stage D: autoregression - name the word that comes NEXT -------------
def make_next_d(n_train=50000, n_test=10000, seed=0, noise=0.05,
                path=os.path.join(RD, "kid_next.npz")):
    """P_C context words as tile strips; the target is the word that FOLLOWS
    them and is never rendered. Same shape as the cloze set, so stage D reads
    it through the same A->B word eye - only the QUESTION changes (which word
    comes next, vs which word is missing). Unlike cloze, the model sees no
    right-hand context: this is the strictly harder, causal question."""
    from radial_lm import _clean
    zb = np.load(os.path.join(RD, "kid_words.npz"), allow_pickle=True)
    vocab = [str(w) for w in zb["vocab"]]
    v2i = {w: i for i, w in enumerate(vocab)}
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        f.seek(10_000_000)
        toks = _clean(f.read(8_000_000)).split()
    ze = np.load(os.path.join(RD, "embed_rs.npz"), allow_pickle=True)
    rs_vocab = {str(w): i for i, w in enumerate(ze["vocab"])}
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = np.zeros((n, P_C, L_MAX, 32, 32), np.float32)
    y = np.zeros(n, np.int64)
    blank = np.full(n, -1, np.int64)          # no blank slot in D
    ctx = np.full((n, P_C), -1, np.int32)
    got = 0
    order = rng.permutation(len(toks) - P_C - 1)
    for p in order:
        wds = toks[p:p + P_C]                 # context
        nxt = toks[p + P_C]                   # the answer - never rendered
        if any(len(w) > L_MAX for w in wds) or nxt not in v2i:
            continue
        y[got] = v2i[nxt]
        for s, w in enumerate(wds):
            ctx[got, s] = rs_vocab.get(w, -1)
            for l, ch in enumerate(w):
                X[got, s, l] = render_letter(ch, rng)
        got += 1
        if got == n:
            break
    if got < n:
        raise RuntimeError(f"only {got}/{n} next-word windows")
    X = X * rng.uniform(0.7, 1.0, (n, 1, 1, 1, 1)).astype(np.float32)
    X = np.clip(X + rng.normal(0, noise, X.shape).astype(np.float32), 0, 1)
    X8 = (X * 255).astype(np.uint8)
    np.savez(path, Xtr=X8[:n_train], ytr=y[:n_train],
             Xte=X8[n_train:], yte=y[n_train:],
             btr=blank[:n_train], bte=blank[n_train:],
             ctr=ctx[:n_train], cte=ctx[n_train:])
    print(f"kid_next: {n_train}/{n_test} {P_C}-word contexts -> NEXT word "
          f"(V={V_B}, target never shown) -> {path}", flush=True)
    return path


def stage_d(pop_size=96, gens=12, max_rounds=600, seed=11, max_spaces=8,
            grid_size=8, verbose=True, ears=True, warm=None,
            head_mode="genomes"):
    """Stage D: autoregression - predict the NEXT word.

    head_mode="genomes" (default, the user's rule): the head sees ONLY frozen
    genome outputs. The word bag and the ears are ENVIRONMENT the genomes read
    from - never head inputs. The first D run fed the raw bank to the head and
    the head became 98% of the params while genomes earned ZERO; genomes must
    carry the weight, so they now build the model from an empty head (exactly
    how stage A earns its 515).

    warm defaults to None: stage C's genomes measured NEGATIVE as a warm base
    (0.1703 -> 0.1699), so there is nothing there worth inheriting."""
    return stage_c(pop_size=pop_size, gens=gens, max_rounds=max_rounds,
                   seed=seed, max_spaces=max_spaces, grid_size=grid_size,
                   verbose=verbose, ears=ears, data="kid_next.npz",
                   stage="D", warm=warm, head_mode=head_mode,
                   task=("name the word that comes NEXT after a %d-word "
                         "context (V=%d, pixels only, target never shown). "
                         "A->B chain frozen as the per-slot word eye; C's "
                         "composition genomes replayed as the warm base.")
                        % (P_C, V_B))


def stage_d_part(target, pop_size=96, gens=12, max_rounds=600, seed=13,
                 max_spaces=8, ears=True, verbose=True):
    """One DECOMPOSED next-word question - a piece the kid can actually
    answer, asked on the same pixels and the same split as D.

    D2 measured 140 genomes of signal found instantly, then a stall 2.75
    points short of a plain ridge head: the signature of a question that is
    too big to answer in one leap, not one without signal. So stop asking
    "which of 500 words comes next" and ask what the kid's earned competence
    can reach: the next word's FIRST LETTER (26 - stage B already spells) or
    its LENGTH (8). Their answers become environment for the composed stage.
    """
    tag = {"first_letter": "DFL", "length": "DLEN"}[target]
    return stage_c(pop_size=pop_size, gens=gens, max_rounds=max_rounds,
                   seed=seed, max_spaces=max_spaces, verbose=verbose,
                   ears=ears, data="kid_next.npz", stage=tag, warm=None,
                   head_mode="genomes", target=target,
                   task=("name the %s of the word that comes NEXT after a "
                         "%d-word context (pixels only, target never shown) "
                         "- a decomposed piece of stage D.")
                        % (target.replace("_", " "), P_C))


def stage_d_compose(pop_size=96, gens=12, max_rounds=600, seed=17,
                    max_spaces=8, ears=True, verbose=True,
                    parts=("kid_modelDFL3.json", "kid_modelDLEN3.json")):
    """The COMPOSE step: the full 500-way next-word question, with the
    decomposed answers available as ENVIRONMENT channels (not head inputs).
    Head stays genome-only, so the composed stage still earns its own model.
    Anchor to beat: D2's 0.1403 test / 0.1514 val, and the 0.1703 reference."""
    return stage_c(pop_size=pop_size, gens=gens, max_rounds=max_rounds,
                   seed=seed, max_spaces=max_spaces, verbose=verbose,
                   ears=ears, data="kid_next.npz", stage="DC", warm=None,
                   head_mode="genomes", target="word",
                   bank_from=list(parts),
                   task=("name the word that comes NEXT after a %d-word "
                         "context (V=%d, pixels only, target never shown), "
                         "COMPOSED: the decomposed first-letter and length "
                         "genomes are environment channels.") % (P_C, V_B))


def stage_c(pop_size=64, gens=12, max_rounds=400, seed=9, max_spaces=8,
            grid_size=8, verbose=True, ears=False, data="kid_cloze.npz",
            stage="C", warm=None, task=None, head_mode="all",
            target="word", bank_from=None):
    """B frozen as the word-eye per slot; new spaces compose context.
    ears=True (Stage C2): each OBSERVED context word also arrives as its
    RS-evolved semantic vector (evolution's own listening experience,
    embed_rs.npz). The blank word gets no vector - it stays the unknown.
    The curriculum remains fully evolution-made: eyes from pixels,
    spelling on the eyes, meaning from the evolved distributional space."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    rk.GRID = int(grid_size)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    with open(os.path.join(RD, "kid_modelA.json")) as f:
        A = json.load(f)["spaces"][0]
    with open(os.path.join(RD, "kid_modelB.json")) as f:
        B = json.load(f)["spaces"]              # B composition spaces
    z = np.load(os.path.join(RD, data))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    Ntr, Nte = len(ytr), len(yte)
    G = rk.GRID

    # DECOMPOSITION: the same environment, a smaller question. "which of 500
    # words comes next" is one leap; its first letter (26) and its length (8)
    # are questions the kid's earned spelling can actually answer. Labels are
    # DERIVED from the same targets - no new data, no new pixels, and the
    # test split is untouched.
    n_cls = V_B
    if target != "word":
        zb = np.load(os.path.join(RD, "kid_words.npz"), allow_pickle=True)
        vocab = [str(w) for w in zb["vocab"]]
        if target == "first_letter":
            m = np.array([LETTERS.index(w[0]) if w[0] in LETTERS else 0
                          for w in vocab], np.int64)
            n_cls = 26
        elif target == "length":
            m = np.array([min(len(w), L_MAX) - 1 for w in vocab], np.int64)
            n_cls = L_MAX
        else:
            raise ValueError(f"unknown target {target!r}")
        ytr, yte = m[ytr], m[yte]
        log_pre = (f"[{stage}] TARGET = {target}: {n_cls} classes "
                   f"(chance {1/n_cls:.4f}); labels derived from the same "
                   f"next-word answers, same pixels, same split")
    else:
        log_pre = None
    rows_tr = np.repeat(Xtr.reshape(Ntr * P_C * L_MAX, 32, 32)[..., None],
                        3, axis=3)
    rows_te = np.repeat(Xte.reshape(Nte * P_C * L_MAX, 32, 32)[..., None],
                        3, axis=3)
    # 640k rows: one cached scale is ~14GB of maps - cache exactly one and
    # sort the eye by scale so each scale builds once per pass
    env = Env(torch, dev, rows_tr, rows_te, max_cached=1)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, n_cls), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, n_cls), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    log_lines = []

    def log(m, v=True):
        log_lines.append(m)
        if v:
            print(m, flush=True)

    if log_pre:
        log(log_pre)

    # replay the A->B chain PER WORD SLOT -> word features per slot
    log(f"[{stage}] replaying A({len(A)}) -> B({sum(len(s) for s in B)}) "
        f"per word slot")

    A_sorted = sorted(range(len(A)), key=lambda k: A[k].get("ps", 8))

    def word_feats(test):
        N = Nte if test else Ntr
        # SLOT-OUTER sweep: park ONE slot bank at a time. Parking all P_C
        # slots together costs P_C x len(A) x N x L_MAX x G x G fp16 - 211GB
        # at 100k phrases, which the 251GB container OOM-kills. One slot is
        # ~53GB at 100k. The price is P_C eye sweeps instead of one; the
        # features are byte-identical (feature_r0 is per-row and its gate
        # stats are baked into the genes, so order never changes a value).
        out_slots = []
        for s in range(P_C):
            parts = [None] * len(A)
            for k in A_sorted:                # scale-sorted: one build/scale
                g8 = rk.feature_r0(torch, tp, env, A[k], test=test,
                                   want_grid=True).view(N, P_C, L_MAX, G, G)
                parts[k] = g8[:, s].reshape(N, L_MAX, G, G).half().cpu()
                del g8
            torch.cuda.empty_cache()
            a_cpu = torch.cat(parts, 1)       # (N, len(A)*L_MAX, G, G) fp16
            parts = None
            # B chain in ROW CHUNKS - it is per-row, so chunking is exact and
            # the whole slot bank never has to sit on the GPU at once.
            chunks = []
            for lo in range(0, N, ROW_CH):
                bank = a_cpu[lo:lo + ROW_CH].to(dev)
                cols = []
                for sp in B:
                    f = torch.stack([rk.feature_grid_g(torch, tp, bank, bg)
                                     for bg in sp], 1)
                    cols.append(f)
                    bank = torch.stack(
                        [rk.feature_grid_g(torch, tp, bank, bg, want_grid=True)
                         for bg in sp], 1).half()
                chunks.append(torch.cat(cols, 1).cpu())
                del bank, cols
                torch.cuda.empty_cache()
            del a_cpu
            out_slots.append(torch.cat(chunks, 0).to(dev))
            chunks = None
            log(f"  [{stage}] slot {s} word features: {out_slots[-1].shape[1]} "
                f"({round(time.time()-t0)}s)")
        return out_slots                      # P_C x (N, nB)

    # The A->B replay is deterministic (frozen genomes, baked gate stats), so
    # every stage over the same data + same eye rebuilds the SAME word
    # features at ~21 min a run. Cache them: the decomposed stages (first
    # letter / length / compose) all read the same sweep. Key on the data
    # file, grid, and the exact eye+word-space sizes so a changed A or B can
    # never silently hit a stale cache.
    wf_key = (f"wf_{os.path.basename(data).replace('.npz', '')}_g{G}"
              f"_A{len(A)}_B{sum(len(s) for s in B)}_N{Ntr}x{Nte}.pt")
    wf_path = os.path.join(RD, wf_key)
    if os.path.exists(wf_path):
        blob = torch.load(wf_path, map_location=dev)
        wf_tr = [t.to(dev) for t in blob["tr"]]
        wf_te = [t.to(dev) for t in blob["te"]]
        log(f"[{stage}] word features from cache {wf_key} "
            f"({round(time.time()-t0)}s - skipped the eye sweep)")
    else:
        wf_tr = word_feats(False)
        wf_te = word_feats(True)
        try:
            torch.save({"tr": [t.cpu() for t in wf_tr],
                        "te": [t.cpu() for t in wf_te]}, wf_path)
            log(f"[{stage}] word features cached -> {wf_key}")
        except Exception as exc:               # cache is an optimisation only
            log(f"[{stage}] word-feature cache write failed ({exc}) - "
                f"continuing without it")
    nB = wf_tr[0].shape[1]
    zmu = torch.cat(wf_tr, 1).mean(0)
    zsd = torch.cat(wf_tr, 1).std(0) + 1e-6
    bank_tr = ((torch.cat(wf_tr, 1) - zmu) / zsd).clamp(-8, 8)
    bank_te = ((torch.cat(wf_te, 1) - zmu) / zsd).clamp(-8, 8)

    ears_tag = ""
    if ears:
        # three evolved ears: similarity (what words mean like), and the
        # DIRECTIONAL pair - next (what follows a word) and prev (what
        # precedes it) - sequence experience, all evolution-made. Vocab
        # can differ per table; look up by word string.
        tables = []
        names = [("embed_rs.npz", "sim"), ("embed_rs_next.npz", "next"),
                 ("embed_rs_prev.npz", "prev")]
        for fn, nm in names:
            fp = os.path.join(RD, fn)
            if os.path.exists(fp):
                ze = np.load(fp, allow_pickle=True)
                v2 = {str(w): i for i, w in enumerate(ze["vocab"])}
                tables.append((nm, v2,
                               torch.tensor(ze["feat"].astype(np.float32),
                                            device=dev)))
        zc = np.load(os.path.join(RD, data), allow_pickle=True)
        # RS-vocab id -> word string (ctx ids index the SIM table's vocab)
        ze0 = np.load(os.path.join(RD, "embed_rs.npz"), allow_pickle=True)
        rs_words = [str(w) for w in ze0["vocab"]]

        def ear_bank(ctx):
            N = len(ctx)
            parts = []
            for nm, v2, E in tables:
                D_ = E.shape[1]
                block = torch.zeros((N, P_C * D_), device=dev)
                for s in range(P_C):
                    ids = ctx[:, s]
                    rows, feats = [], []
                    for i in range(N):
                        if ids[i] >= 0:
                            j = v2.get(rs_words[ids[i]])
                            if j is not None:
                                rows.append(i)
                                feats.append(j)
                    if rows:
                        block[torch.tensor(rows, device=dev,
                                           dtype=torch.long),
                              s * D_:(s + 1) * D_] = E[
                            torch.tensor(feats, device=dev,
                                         dtype=torch.long)]
                parts.append(block)
            return torch.cat(parts, 1)

        eb_tr = ear_bank(zc["ctr"])
        emu, esd = eb_tr.mean(0), eb_tr.std(0) + 1e-6
        eb_tr = ((eb_tr - emu) / esd).clamp(-8, 8)
        eb_te = ((ear_bank(zc["cte"]) - emu) / esd).clamp(-8, 8)
        bank_tr = torch.cat([bank_tr, eb_tr], 1)
        bank_te = torch.cat([bank_te, eb_te], 1)
        ears_tag = "3" if len(tables) == 3 else "2"
        log(f"[{stage}{ears_tag}] ears attached: +{eb_tr.shape[1]} channels from "
            f"{[t[0] for t in tables]} (observed words only)")

    if bank_from:
        # COMPOSE: the answers to the smaller questions become ENVIRONMENT for
        # the bigger one - sub-model genomes are appended to the BANK, never to
        # the head. So the composed stage still has to earn its own model, but
        # it can now read "what letter does the next word start with" and "how
        # long is it" as channels.
        #
        # Each sub-model replays from the SAME base bank and grows only within
        # itself, then all outputs are appended at the end. This is load-
        # bearing: a vec genome indexes channels as (c % C), so a genome
        # evolved on a 2724-wide bank reads DIFFERENT channels on a wider one.
        # Chaining the sub-models into one growing bank would silently rewire
        # every genome after the first.
        base_bt, base_be = bank_tr, bank_te
        extra_tr, extra_te = [], []
        for mf in bank_from:
            mp = os.path.join(RD, mf)
            if not os.path.exists(mp):
                raise RuntimeError(f"{stage} bank_from missing: {mp}")
            with open(mp) as f:
                md = json.load(f)
            b_tr, b_te = base_bt, base_be     # fresh base per sub-model
            n_m = 0
            for sp in md["spaces"]:
                if not sp:
                    continue
                s_tr = torch.stack([torch.nan_to_num(
                    rk.feature_vec(torch, tp, b_tr, g), nan=0.0, posinf=0.0,
                    neginf=0.0).clamp(-1e6, 1e6) for g in sp], 1)
                s_te = torch.stack([torch.nan_to_num(
                    rk.feature_vec(torch, tp, b_te, g), nan=0.0, posinf=0.0,
                    neginf=0.0).clamp(-1e6, 1e6) for g in sp], 1)
                smu, ssd = s_tr.mean(0), s_tr.std(0) + 1e-6
                s_tr = ((s_tr - smu) / ssd).clamp(-8, 8)
                s_te = ((s_te - smu) / ssd).clamp(-8, 8)
                extra_tr.append(s_tr)
                extra_te.append(s_te)
                b_tr = torch.cat([b_tr, s_tr], 1)   # grow WITHIN this model
                b_te = torch.cat([b_te, s_te], 1)
                n_m += len(sp)
            log(f"[{stage}] composed in {mf} ({md.get('label', '?')}): "
                f"+{n_m} frozen genomes -> bank")
        if extra_tr:
            bank_tr = torch.cat([bank_tr] + extra_tr, 1)
            bank_te = torch.cat([bank_te] + extra_te, 1)
            log(f"[{stage}] bank after compose: {bank_tr.shape[1]} channels")
        del extra_tr, extra_te

    # orderless bag of word features (mean over slots)
    bag_tr = torch.stack(wf_tr, 2).mean(2)
    bag_te = torch.stack(wf_te, 2).mean(2)
    bmu, bsd = bag_tr.mean(0), bag_tr.std(0) + 1e-6
    bag_tr = ((bag_tr - bmu) / bsd).clamp(-8, 8)
    bag_te = ((bag_te - bmu) / bsd).clamp(-8, 8)

    # The REFERENCE anchor: what a plain linear head gets from the raw
    # environment. Always measured and reported - it is the number every
    # result must be read against - but whether the MODEL's head is allowed
    # to see it is head_mode's call.
    ref_tr = [bag_tr[:, j] for j in range(nB)]
    ref_te = [bag_te[:, j] for j in range(nB)]
    if ears:
        ref_tr.extend(eb_tr[:, j] for j in range(eb_tr.shape[1]))
        ref_te.extend(eb_te[:, j] for j in range(eb_te.shape[1]))
    ref_all = torch.stack(ref_tr, 1)
    _, val_ref = _ridge_soft(torch, ref_all[:n_fit], ref_all[n_fit:], Yf, yv)
    del ref_all

    if head_mode == "genomes":
        # GENOMES CARRY THE WEIGHT (user's rule). The head sees ONLY frozen
        # genome outputs; the word bag and the ears are ENVIRONMENT that
        # genomes read from, never head inputs. Feeding the raw bank to the
        # head is what killed C and D: the head read all 1835 raw columns
        # linearly, so a genome could only earn by beating a full linear
        # readout of the whole environment - an unpayable bar, and the
        # measured result was genomes 10 -> 5 -> 2 -> 0 while the head grew
        # to 98% of the params. Stage A earns 515 genomes precisely because
        # it starts from an EMPTY head. Same rule here.
        all_tr, all_te = [], []
        val_prev = 0.0
        log(f"[{stage}] head = GENOMES ONLY. Reference linear anchor "
            f"(bag+ears straight to a ridge head, NOT model input): "
            f"val {val_ref:.4f} - genomes must build the model from zero.")
    else:
        all_tr, all_te = ref_tr, ref_te
        val_prev = val_ref
        log(f"[{stage}] orderless word-bag baseline: val {val_prev:.4f}")

    if warm:
        # the curriculum's warm start: the PREVIOUS stage's frozen genomes
        # replayed as base columns. Their gate stats are baked into the genes,
        # so replaying them on this stage's bank is exact (the environment
        # law: a genome must never be a function of the batch it lands in).
        # This is also the honest anchor - the baseline D must beat CONTAINS
        # stage C, so any gain logged below is D's own.
        wp = os.path.join(RD, warm)
        if not os.path.exists(wp):
            raise RuntimeError(f"stage {stage} warm start missing: {wp}")
        with open(wp) as f:
            wspaces = json.load(f)["spaces"]
        n_w = sum(len(sp) for sp in wspaces)
        for sp in wspaces:
            if not sp:
                continue
            fw_tr = torch.stack([torch.nan_to_num(
                rk.feature_vec(torch, tp, bank_tr, g), nan=0.0, posinf=0.0,
                neginf=0.0).clamp(-1e6, 1e6) for g in sp], 1)
            fw_te = torch.stack([torch.nan_to_num(
                rk.feature_vec(torch, tp, bank_te, g), nan=0.0, posinf=0.0,
                neginf=0.0).clamp(-1e6, 1e6) for g in sp], 1)
            wmu, wsd = fw_tr.mean(0), fw_tr.std(0) + 1e-6
            fw_tr = ((fw_tr - wmu) / wsd).clamp(-8, 8)
            fw_te = ((fw_te - wmu) / wsd).clamp(-8, 8)
            all_tr.extend(fw_tr[:, j] for j in range(fw_tr.shape[1]))
            all_te.extend(fw_te[:, j] for j in range(fw_te.shape[1]))
            bank_tr = torch.cat([bank_tr, fw_tr], 1)
            bank_te = torch.cat([bank_te, fw_te], 1)
        if n_w:
            base_all = torch.stack(all_tr, 1)
            _, val_prev = _ridge_soft(torch, base_all[:n_fit],
                                      base_all[n_fit:], Yf, yv)
            log(f"[{stage}] warm base: +{n_w} frozen genomes from {warm} -> "
                f"val {val_prev:.4f}  (the anchor {stage} must beat)")
        else:
            log(f"[{stage}] warm {warm}: 0 genomes to inherit - the anchor "
                f"stays the word bag (stage C earned nothing to pass on)")

    spaces, space_genomes = [], []
    for si in range(max_spaces):
        C = bank_tr.shape[1]
        new_fn = lambda r: rk.new_vec_genome(r, C)
        mut_fn = lambda r, g, sc: rk.mutate_vec(r, g, sc, C)
        feat_tr = lambda g, b=bank_tr: torch.nan_to_num(
            rk.feature_vec(torch, tp, b, g), nan=0.0, posinf=0.0,
            neginf=0.0).clamp(-1e6, 1e6)
        feat_te = lambda g, b=bank_te: torch.nan_to_num(
            rk.feature_vec(torch, tp, b, g), nan=0.0, posinf=0.0,
            neginf=0.0).clamp(-1e6, 1e6)
        base_prev = (torch.stack(all_tr, 1) if all_tr
                     else torch.zeros((Ntr, 0), device=dev))
        log(f"  [{stage} space {si}] opening - bank {C} channels, base "
            f"{base_prev.shape[1]} cols")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_tr, log, verbose)
        if not frozen:
            log(f"  [{stage} space {si}] produced nothing - stop")
            break
        rk.bake_gate_stats(torch, tp, frozen, bank_tr)
        space_genomes.append(frozen)
        f_tr = torch.stack(fcols, 1)
        f_te = torch.stack([feat_te(g) for g in frozen], 1)
        fmu, fsd = f_tr.mean(0), f_tr.std(0) + 1e-6
        f_tr = ((f_tr - fmu) / fsd).clamp(-8, 8)
        f_te = ((f_te - fmu) / fsd).clamp(-8, 8)
        all_tr.extend(f_tr[:, j] for j in range(f_tr.shape[1]))
        all_te.extend(f_te[:, j] for j in range(f_te.shape[1]))
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                                 Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4)})
        log(f"  [{stage} space {si}] FULL: {len(frozen)} genomes, val "
            f"{val_now:.4f} (+{gain:.4f}) ({round(time.time()-t0)}s)")
        bank_tr = torch.cat([bank_tr, f_tr], 1)
        bank_te = torch.cat([bank_te, f_te], 1)
        val_prev = val_now
        if si > 0 and gain < rk.MIN_SPACE_GAIN:
            break
        if os.path.exists(_STOP):
            break

    if not all_tr:
        # genome-only head and evolution earned nothing: there is no model.
        # Say so, instead of reporting a head-only number as a result.
        log(f"[kid {stage}] NOTHING EARNED - genomes built no model. "
            f"Reference linear anchor was val {val_ref:.4f}.")
        print(f"[kid {stage}] DONE: NO MODEL (0 genomes earned); reference "
              f"linear anchor val {val_ref:.4f} ({round(time.time()-t0)}s)",
              flush=True)
        return {"phase": f"curriculum stage {stage}", "test_acc": 0.0,
                "test_top5": 0.0, "n_genomes": 0, "n_spaces": 0,
                "ref_anchor": round(float(val_ref), 4),
                "note": "genome-only head; evolution earned nothing",
                "seconds": round(time.time() - t0)}
    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    best1 = best5 = 0.0
    for lam in (1.0, 3.0, 10.0):
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        Am = torch.hstack([(Ftr - mu) / sd, torch.ones(Ntr, 1, device=dev)])
        Gm = (Am.T @ Am).double() + lam * torch.eye(
            Ftr.shape[1] + 1, device=dev, dtype=torch.float64)
        Wm = torch.linalg.solve(Gm, (Am.T @ Yfull).double()).float()
        sc = torch.hstack([(Fte - mu) / sd,
                           torch.ones(Nte, 1, device=dev)]) @ Wm
        a1 = float((sc.argmax(1) == yte_t).float().mean())
        a5 = float((sc.topk(5, 1).indices == yte_t.view(-1, 1))
                   .any(1).float().mean())
        if a1 > best1:
            best1, best5 = a1, a5

    from anim_infer import count_params
    gp = (sum(count_params(g) for g in A)
          + sum(count_params(g) for sp in B for g in sp)
          + sum(count_params(g) for sp in space_genomes for g in sp))
    n_genomes = (len(A) + sum(len(sp) for sp in B)
                 + sum(len(sp) for sp in space_genomes))
    out = {"phase": f"curriculum stage {stage}",
           "n_classes": n_cls, "chance": round(1 / n_cls, 4),
           "n_train": Ntr, "n_test": Nte,
           "test_acc": round(best1, 4), "test_top5": round(best5, 4),
           "val_final": spaces[-1]["val_after"] if spaces else None,
           "n_spaces": len(spaces),
           "space_caps": [s["n_frozen"] for s in spaces],
           "spaces": spaces, "n_genomes": n_genomes, "genome_params": gp,
           "head_params": (len(all_tr) + 1) * n_cls,
           "total_params": gp + (len(all_tr) + 1) * n_cls,
           "task": task or (("name the BLANKED word in a %d-word phrase "
                             "(V=%d, pixels only). A->B chain frozen as the "
                             "per-slot word eye; new spaces compose words "
                             "into context.") % (P_C, V_B)),
           "seconds": round(time.time() - t0)}
    out["ears"] = bool(ears)
    out["warm"] = warm
    out["head_mode"] = head_mode
    out["ref_anchor"] = round(float(val_ref), 4)
    # the honest split: how much of the model is evolved vs a linear head
    out["genome_share"] = round(gp / max(1, out["total_params"]), 4)
    with open(os.path.join(RD, f"kid_stage{stage}{ears_tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, f"kid_model{stage}{ears_tag}.json"), "w") as f:
        json.dump({"grid": G, "eye": A, "word_spaces": B,
                   "spaces": space_genomes,
                   "label": "cloze" if stage == "C" else "next-word",
                   "stage": stage}, f)
    rk._record_run(
        {"env": f"kid-stage{stage}", "n_classes": n_cls, "n_train": Ntr,
         "pop_size": pop_size, "gens": gens, "seed": seed},
        [{"round": s["space"], "added": s["n_frozen"],
          "val_acc": s["val_after"], "n": s["n_frozen"]} for s in spaces],
        {"test_acc": round(best1, 4), "test_top5": round(best5, 4),
         "n_frozen_total": n_genomes, "total_params": out["total_params"]},
        log_lines, ["lm", "kid", "curriculum", "radial"])
    print("[kid %s] DONE: TEST top1 %.4f top5 %.4f (%ds)"
          % (stage, best1, best5, round(time.time() - t0)), flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "D" in sys.argv:
        if not os.path.exists(os.path.join(RD, "kid_next.npz")) \
                or "--regen" in sys.argv:
            make_next_d()
        stage_d()
    elif "C3" in sys.argv:
        if not os.path.exists(os.path.join(RD, "kid_words.npz")):
            make_words_b()
        make_cloze_c()
        stage_c(ears=True)
    elif "C2" in sys.argv:
        make_cloze_c()                        # regen: ctx ids now stored
        stage_c(ears=True)
    elif "C" in sys.argv:
        if not os.path.exists(os.path.join(RD, "kid_cloze.npz")) \
                or "--regen" in sys.argv:
            make_cloze_c()
        stage_c()
    elif "B" in sys.argv:
        if not os.path.exists(os.path.join(RD, "kid_words.npz")) \
                or "--regen" in sys.argv:
            make_words_b()
        stage_b()
    else:
        if not os.path.exists(os.path.join(RD, "kid_letters.npz")) \
                or "--regen" in sys.argv:
            make_letters()
        stage_a()
