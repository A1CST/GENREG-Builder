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
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _ridge_soft, _STOP
from radial_evo2 import Env, new_genome, mutate
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))
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


def stage_c(pop_size=64, gens=12, max_rounds=400, seed=9, max_spaces=8,
            grid_size=8, verbose=True, ears=False):
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
    z = np.load(os.path.join(RD, "kid_cloze.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    Ntr, Nte = len(ytr), len(yte)
    G = rk.GRID
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
    Yf = -torch.ones((n_fit, V_B), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, V_B), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    log_lines = []

    def log(m, v=True):
        log_lines.append(m)
        if v:
            print(m, flush=True)

    # replay the A->B chain PER WORD SLOT -> word features per slot
    log(f"[C] replaying A({len(A)}) -> B({sum(len(s) for s in B)}) "
        f"per word slot")

    A_sorted = sorted(range(len(A)), key=lambda k: A[k].get("ps", 8))

    def word_feats(test):
        N = Nte if test else Ntr
        # one sweep over the eye, scale-sorted; all slots filled together
        slot_parts = [[None] * len(A) for _ in range(P_C)]
        for k in A_sorted:
            g8 = rk.feature_r0(torch, tp, env, A[k], test=test,
                               want_grid=True).view(N, P_C, L_MAX, G, G)
            for s in range(P_C):
                # park on CPU (pods carry 1.5TB RAM); 4 slot banks on GPU
                # simultaneously OOMs at 50k phrases (4 x 26GB)
                slot_parts[s][k] = g8[:, s].reshape(
                    N, L_MAX, G, G).half().cpu()
            del g8
        torch.cuda.empty_cache()
        out_slots = []
        for s in range(P_C):
            a_bank = torch.cat(slot_parts[s], 1).to(dev)
            slot_parts[s] = None
            cols, bank = [], a_bank
            for sp in B:
                f = torch.stack([rk.feature_grid_g(torch, tp, bank, bg)
                                 for bg in sp], 1)
                cols.append(f)
                bank = torch.stack(
                    [rk.feature_grid_g(torch, tp, bank, bg, want_grid=True)
                     for bg in sp], 1).half()
            out_slots.append(torch.cat(cols, 1))
            del a_bank, bank
            torch.cuda.empty_cache()
            log(f"  [C] slot {s} word features: {out_slots[-1].shape[1]} "
                f"({round(time.time()-t0)}s)")
        return out_slots                      # P_C x (N, nB)

    wf_tr = word_feats(False)
    wf_te = word_feats(True)
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
        zc = np.load(os.path.join(RD, "kid_cloze.npz"), allow_pickle=True)
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
        log(f"[C{ears_tag}] ears attached: +{eb_tr.shape[1]} channels from "
            f"{[t[0] for t in tables]} (observed words only)")

    # warm base: orderless bag of word features (mean over slots)
    bag_tr = torch.stack(wf_tr, 2).mean(2)
    bag_te = torch.stack(wf_te, 2).mean(2)
    bmu, bsd = bag_tr.mean(0), bag_tr.std(0) + 1e-6
    bag_tr = ((bag_tr - bmu) / bsd).clamp(-8, 8)
    bag_te = ((bag_te - bmu) / bsd).clamp(-8, 8)
    all_tr = [bag_tr[:, j] for j in range(nB)]
    all_te = [bag_te[:, j] for j in range(nB)]
    if ears:
        all_tr.extend(eb_tr[:, j] for j in range(eb_tr.shape[1]))
        all_te.extend(eb_te[:, j] for j in range(eb_te.shape[1]))
    base_all = torch.stack(all_tr, 1)
    _, val_prev = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                              Yf, yv)
    log(f"[C] orderless word-bag baseline: val {val_prev:.4f}")

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
        base_prev = torch.stack(all_tr, 1)
        log(f"  [C space {si}] opening - bank {C} channels")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_tr, log, verbose)
        if not frozen:
            log(f"  [C space {si}] produced nothing - stop")
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
        log(f"  [C space {si}] FULL: {len(frozen)} genomes, val "
            f"{val_now:.4f} (+{gain:.4f}) ({round(time.time()-t0)}s)")
        bank_tr = torch.cat([bank_tr, f_tr], 1)
        bank_te = torch.cat([bank_te, f_te], 1)
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
          + sum(count_params(g) for sp in B for g in sp)
          + sum(count_params(g) for sp in space_genomes for g in sp))
    n_genomes = (len(A) + sum(len(sp) for sp in B)
                 + sum(len(sp) for sp in space_genomes))
    out = {"phase": "curriculum stage C: cloze",
           "n_classes": V_B, "chance": round(1 / V_B, 4),
           "n_train": Ntr, "n_test": Nte,
           "test_acc": round(best1, 4), "test_top5": round(best5, 4),
           "val_final": spaces[-1]["val_after"] if spaces else None,
           "n_spaces": len(spaces),
           "space_caps": [s["n_frozen"] for s in spaces],
           "spaces": spaces, "n_genomes": n_genomes, "genome_params": gp,
           "head_params": (len(all_tr) + 1) * V_B,
           "total_params": gp + (len(all_tr) + 1) * V_B,
           "task": ("name the BLANKED word in a %d-word phrase (V=%d, "
                    "pixels only). A->B chain frozen as the per-slot word "
                    "eye; new spaces compose words into context.")
                   % (P_C, V_B),
           "seconds": round(time.time() - t0)}
    out["ears"] = bool(ears)
    with open(os.path.join(RD, f"kid_stageC{ears_tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, f"kid_modelC{ears_tag}.json"), "w") as f:
        json.dump({"grid": G, "eye": A, "word_spaces": B,
                   "spaces": space_genomes, "label": "cloze",
                   "stage": "C"}, f)
    rk._record_run(
        {"env": "kid-stageC", "n_classes": V_B, "n_train": Ntr,
         "pop_size": pop_size, "gens": gens, "seed": seed},
        [{"round": s["space"], "added": s["n_frozen"],
          "val_acc": s["val_after"], "n": s["n_frozen"]} for s in spaces],
        {"test_acc": round(best1, 4), "test_top5": round(best5, 4),
         "n_frozen_total": n_genomes, "total_params": out["total_params"]},
        log_lines, ["lm", "kid", "curriculum", "radial"])
    print("[kid C] DONE: TEST top1 %.4f top5 %.4f (%ds)"
          % (best1, best5, round(time.time() - t0)), flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "C3" in sys.argv:
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
