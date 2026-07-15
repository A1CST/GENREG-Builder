"""radial_lm_word.py — the radial stack on WORD-level next-word prediction.

The pivot from characters: at char level the stack was fighting n-gram
tables at their own game. Words move the fight to composition over
MEANING: context words enter as their corpus-SVD embeddings (the
pre-built semantic space — "features are the environment"), and the
stack must compose them into a next-word prediction.

  input   — W context words -> their 128-d wiki-SVD vectors, concatenated
            (W x 128 channels). OOV context words = zero vector.
  target  — the next word, restricted to the top-V vocabulary.
  anchor  — ridge on the raw embedding channels, NO genomes: the additive
            linear-from-embeddings model. Genomes only earn beyond it.
  genomes — vec-grammar conjunctions across (position x embedding-dim)
            channels, emergent-cap stacked spaces, drift-op mutation.
  ceilings— word unigram / bigram / trigram with backoff, fit on the full
            train region.

Train/test from disjoint corpus regions. Exports
radial_data/lm_radial_word.json (+ lm_model_word.json). No gradients.
"""
import json
import os
import time

import numpy as np

from radial_evo import _tprims, _STOP
from radial_lm import _clean, _load_regions
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = os.path.join(_HERE, "corpora", "wikipedia", "wiki_feats.npz")
W = 6                                    # context words per window
V = 500                                  # prediction vocabulary (top-V)
D = 128                                  # embedding dims


def _load_embed():
    z = np.load(FEATS, allow_pickle=True)
    vocab = [str(w) for w in z["vocab"]]
    return vocab, z["feat"].astype(np.float32), z["freq"]


def _tokens(text):
    return text.split()


def make_word_data(n_train=60000, n_test=8000, seed=0,
                   path=os.path.join(_HERE, "radial_data", "lm_word.npz")):
    rng = np.random.default_rng(seed)
    vocab, feat, freq = _load_embed()
    w2i = {w: i for i, w in enumerate(vocab)}
    train_text, test_text = _load_regions(train_mb=8.0, test_mb=2.0)
    tr_tok, te_tok = _tokens(train_text), _tokens(test_text)

    # top-V target vocabulary by TRAIN-REGION frequency (not global wiki)
    cnt = {}
    for w in tr_tok:
        cnt[w] = cnt.get(w, 0) + 1
    top = sorted(cnt, key=cnt.get, reverse=True)[:V]
    tgt2i = {w: i for i, w in enumerate(top)}
    coverage = sum(cnt[w] for w in top) / max(1, len(tr_tok))

    def windows(toks, n):
        ctx = np.zeros((n, W), np.int32)          # index into wiki vocab; -1 OOV
        y = np.zeros(n, np.int64)
        got = 0
        order = rng.permutation(len(toks) - W - 1)
        for p in order:
            tgt = toks[p + W]
            if tgt not in tgt2i:
                continue
            for f in range(W):
                ctx[got, f] = w2i.get(toks[p + f], -1)
            y[got] = tgt2i[tgt]
            got += 1
            if got == n:
                break
        if got < n:
            raise RuntimeError(f"only {got}/{n} windows found")
        return ctx, y

    ctx_tr, y_tr = windows(tr_tok, n_train)
    ctx_te, y_te = windows(te_tok, n_test)
    np.savez(path, ctx_tr=ctx_tr, ytr=y_tr, ctx_te=ctx_te, yte=y_te,
             targets=np.array(top), coverage=coverage)
    print(f"lm_word: {n_train}/{n_test} x {W}-word windows, V={V} "
          f"(covers {coverage:.1%} of train tokens) -> {path}", flush=True)
    return path


def word_ngram_baselines(ctx_words_te, y_te, targets):
    """Word unigram/bigram/trigram with backoff, fit on the train region.
    ctx_words_te: (n, W) arrays of the actual context WORD STRINGS."""
    train_text, _ = _load_regions(train_mb=8.0, test_mb=2.0)
    toks = _tokens(train_text)
    uni, bi, tri = {}, {}, {}
    for i in range(len(toks) - 2):
        uni[toks[i]] = uni.get(toks[i], 0) + 1
        bi.setdefault(toks[i], {})
        bi[toks[i]][toks[i + 1]] = bi[toks[i]].get(toks[i + 1], 0) + 1
        key = (toks[i], toks[i + 1])
        tri.setdefault(key, {})
        tri[key][toks[i + 2]] = tri[key].get(toks[i + 2], 0) + 1
    uni_best = max(uni, key=uni.get)
    out = {}
    n = len(y_te)
    hits_u = hits_b = hits_t = 0
    for i in range(n):
        true = targets[y_te[i]]
        w1, w2 = ctx_words_te[i][-2], ctx_words_te[i][-1]
        hits_u += int(uni_best == true)
        pb = max(bi[w2], key=bi[w2].get) if w2 in bi else uni_best
        hits_b += int(pb == true)
        key = (w1, w2)
        if key in tri:
            pt = max(tri[key], key=tri[key].get)
        elif w2 in bi:
            pt = max(bi[w2], key=bi[w2].get)
        else:
            pt = uni_best
        hits_t += int(pt == true)
    out["unigram"] = round(hits_u / n, 4)
    out["bigram"] = round(hits_b / n, 4)
    out["trigram"] = round(hits_t / n, 4)
    return out


def _embed_bank(torch, dev, ctx, feat_t, mu, sd):
    """(N, W) wiki-vocab ids (-1 = OOV) -> (N, W*D) z-scored channels."""
    N = len(ctx)
    idx = torch.tensor(np.maximum(ctx.astype(np.int64), 0), device=dev)
    mask = torch.tensor((ctx >= 0).astype(np.float32), device=dev)
    cols = []
    for f in range(W):
        v = feat_t[idx[:, f]] * mask[:, f:f + 1]
        cols.append((v - mu) / sd)
    return torch.cat(cols, 1)


def run(pop_size=64, gens=12, max_rounds=400, seed=5, max_spaces=16,
        out_path=None, verbose=True):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    from radial_evo import _ridge_soft
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)

    vocab, feat, _freq = _load_embed()
    feat_t = torch.tensor(feat, device=dev)
    z = np.load(os.path.join(_HERE, "radial_data", "lm_word.npz"),
                allow_pickle=True)
    ctx_tr, ytr = z["ctx_tr"], z["ytr"]
    ctx_te, yte = z["ctx_te"], z["yte"]
    targets = [str(w) for w in z["targets"]]
    Ntr, Nte = len(ytr), len(yte)
    mu, sd = feat_t.mean(0), feat_t.std(0) + 1e-6

    # identity channels: top-V one-hots of the LAST TWO context words, so
    # the anchor CONTAINS the bigram and skip-bigram tables (embeddings
    # alone blur identity — first run: embed ridge 0.1514 < bigram 0.1727)
    w2i = {w: i for i, w in enumerate(vocab)}
    tv = {w2i[w]: k for k, w in enumerate(targets) if w in w2i}

    # CONTINUATION channels — co-occurrence belongs to the ENVIRONMENT,
    # not to evolution (three runs proved genomes can't rediscover lookup
    # tables one conjunction at a time). For each window: the empirical
    # next-word distribution after the last bigram (backoff: last word;
    # backoff: unigram), as its expected EMBEDDING — "what the corpus
    # expects next, semantically". Fit on a SEPARATE corpus slice,
    # independent of the training windows AND the test region — otherwise
    # val leaks through the tables and evolution's fitness compass chases
    # memorization (run 4: val +1.5pt genome gain, test +0.2).
    tbl_pkl = os.path.join(_HERE, "radial_data", "lm_cont_tables.pkl")
    if os.path.exists(tbl_pkl):
        import pickle
        with open(tbl_pkl, "rb") as fh:
            uni_c, bi_c, tri_c = pickle.load(fh)
    else:
        with open(os.path.join(_HERE, "corpora", "combined",
                               "combined_corpus.txt"), "r",
                  encoding="utf-8", errors="ignore") as fh:
            fh.seek(30_000_000)          # past the window + test regions
            table_text = _clean(fh.read(16_000_000))
        toks = _tokens(table_text)
        uni_c, bi_c, tri_c = {}, {}, {}
        for i in range(len(toks) - 2):
            uni_c[toks[i + 1]] = uni_c.get(toks[i + 1], 0) + 1
            bi_c.setdefault(toks[i], {})
            bi_c[toks[i]][toks[i + 1]] = bi_c[toks[i]].get(toks[i + 1], 0) + 1
            key = (toks[i], toks[i + 1])
            tri_c.setdefault(key, {})
            tri_c[key][toks[i + 2]] = tri_c[key].get(toks[i + 2], 0) + 1
        import pickle
        with open(tbl_pkl, "wb") as fh:
            pickle.dump((uni_c, bi_c, tri_c), fh)

    def _cont_vec(dist):
        v = np.zeros(D, np.float32)
        tot = 0
        for w, c in dist.items():
            j = w2i.get(w)
            if j is not None:
                v += c * feat[j]
                tot += c
        return v / tot if tot else v

    _uni_vec = _cont_vec(uni_c)

    tgt_i = {w: k for k, w in enumerate(targets)}

    def _cont_prob(dist):
        """Distribution over the V targets (mass outside vocab dropped)."""
        v = np.zeros(V, np.float32)
        for w, c in dist.items():
            k = tgt_i.get(w)
            if k is not None:
                v[k] = c
        s = v.sum()
        return v / s if s else v

    _uni_prob = _cont_prob(uni_c)

    def _cont_raw(ctx):
        """Continuation channels: expected next EMBEDDING (2D dims, lossy
        but generalizing) + the exact next-word PROBABILITY VECTOR over
        the V targets (the table's candidate list, which the embedding
        blurs — run 6: trigram top-5 0.639 vs our 0.548)."""
        N = len(ctx)
        out = np.zeros((N, 2 * D + V), np.float32)
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

    # standardization stats from TRAIN only — per-batch stats NaN on a
    # single generation row (std of n=1) and shift between batches
    _cont_tr_raw = _cont_raw(ctx_tr)
    _cmu = _cont_tr_raw.mean(0)
    _csd = _cont_tr_raw.std(0) + 1e-6

    def _cont_bank(ctx, _raw=None):
        t = _raw if _raw is not None else _cont_raw(ctx)
        return ((t - _cmu) / _csd).clamp(-8, 8)

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

    N_CONT = 2 * D + V                   # continuation block width

    def _bank0(ctx, cont_raw=None):
        emb = _embed_bank(torch, dev, ctx, feat_t, mu, sd)
        return torch.cat([emb, _identity(ctx, W - 2), _identity(ctx, W - 1),
                          _cont_bank(ctx, cont_raw)], 1)

    B0_tr = _bank0(ctx_tr, _cont_tr_raw)  # (Ntr, W*D + 2V + N_CONT)
    B0_te = _bank0(ctx_te)

    n_fit = int(Ntr * 0.8)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, V), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((Ntr, V), device=dev)
    Yfull[torch.arange(Ntr), torch.tensor(ytr, device=dev)] = 1.0

    oh_val = _ridge_soft(torch, B0_tr[:n_fit], B0_tr[n_fit:], Yf, yv)[1]
    oh_test = max(_ridge_soft(torch, B0_tr, B0_te, Yfull, yte_t, lam=lam)[1]
                  for lam in (3.0, 10.0, 30.0))
    print(f"embed ridge (no genomes): val {oh_val:.4f} test {oh_test:.4f} "
          f"({round(time.time()-t0)}s)", flush=True)

    log_lines = []

    def log(m, v=True):
        log_lines.append(m)
        if v:
            print(m, flush=True)

    def _san(v):
        return torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1e6, 1e6)

    # ---- R0: PER-WORD perception, the animation wiring ----------------
    # each context word is a row; a genome is a word-property detector
    # over the 128 embedding dims; its fitness column is the WINDOW MEAN
    # of its per-slot values (the orderless bag, like frame-mean). R0's
    # base is EMPTY — perception earns freely, it never fights the anchor.
    def _rows(ctx):
        idx = torch.tensor(np.maximum(ctx.astype(np.int64), 0), device=dev)
        mask = torch.tensor((ctx >= 0).astype(np.float32), device=dev)
        r = feat_t[idx.reshape(-1)] * mask.reshape(-1, 1)
        return (r - mu) / sd                      # (N*W, D)

    rows_tr, rows_te = _rows(ctx_tr), _rows(ctx_te)
    spaces, space_genomes, space_stats = [], [], []
    handoff_stats = None
    all_tr = [B0_tr[:, j] for j in range(B0_tr.shape[1])]
    all_te = [B0_te[:, j] for j in range(B0_te.shape[1])]
    bank_tr = bank_te = None
    val_prev = oh_val

    for si in range(max_spaces):
        if si == 0:
            C = D
            new_fn = lambda r: rk.new_vec_genome(r, C)
            mut_fn = lambda r, g, sc: rk.mutate_vec(r, g, sc, C)
            feat_tr = lambda g: _san(rk.feature_vec(torch, tp, rows_tr, g)
                                     ).view(Ntr, W).mean(1)
            feat_te = lambda g: _san(rk.feature_vec(torch, tp, rows_te, g)
                                     ).view(Nte, W).mean(1)
            slot_tr = lambda g: _san(rk.feature_vec(torch, tp, rows_tr, g)
                                     ).view(Ntr, W)
            slot_te = lambda g: _san(rk.feature_vec(torch, tp, rows_te, g)
                                     ).view(Nte, W)
            base_prev = torch.zeros((Ntr, 0), device=dev)
            src = f"{D} embedding dims per word (per-slot perception, window-mean)"
        else:
            C = bank_tr.shape[1]
            new_fn = lambda r: rk.new_vec_genome(r, C)
            mut_fn = lambda r, g, sc: rk.mutate_vec(r, g, sc, C)
            feat_tr = lambda g: _san(rk.feature_vec(torch, tp, bank_tr, g))
            feat_te = lambda g: _san(rk.feature_vec(torch, tp, bank_te, g))
            base_prev = torch.stack(all_tr, 1)    # anchor + everything frozen
            src = f"{C} channels (R0-genome x POSITION + identity one-hots)"
        log(f"  [space {si}] opening — reads {src}")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens, max_rounds,
                                         n_fit, Yf, yv, base_prev, new_fn, mut_fn,
                                         feat_tr, log, verbose)
        if not frozen:
            log(f"  [space {si}] produced nothing — stop")
            break
        space_genomes.append(frozen)
        fte = [feat_te(g) for g in frozen]
        f_tr = torch.stack(fcols, 1)
        f_te2 = torch.stack(fte, 1)
        zmu, zsd = f_tr.mean(0), f_tr.std(0) + 1e-6
        space_stats.append((zmu, zsd))
        f_tr = ((f_tr - zmu) / zsd).clamp(-8, 8)
        f_te2 = ((f_te2 - zmu) / zsd).clamp(-8, 8)
        all_tr.extend(f_tr[:, j] for j in range(f_tr.shape[1]))
        all_te.extend(f_te2[:, j] for j in range(f_te2.shape[1]))
        base_all = torch.stack(all_tr, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:], Yf, yv)
        gain = val_now - val_prev
        spaces.append({"space": si, "source": src, "n_frozen": len(frozen),
                       "val_after": round(float(val_now), 4),
                       "val_gain": round(float(gain), 4)})
        log(f"  [space {si}] FULL: {len(frozen)} genomes, val {val_now:.4f} "
            f"(+{gain:.4f}) ({round(time.time()-t0)}s)")
        if si == 0:
            # THE HAND-OFF: every R0 genome's value at EVERY position —
            # (genome x slot) channels; order enters here, like frames
            g_tr = [slot_tr(g) for g in frozen]
            g_te = [slot_te(g) for g in frozen]
            s_tr = torch.cat(g_tr, 1)             # (Ntr, nR0*W)
            s_te = torch.cat(g_te, 1)
            pmu, psd = s_tr.mean(0), s_tr.std(0) + 1e-6
            handoff_stats = (pmu, psd)
            s_tr = ((s_tr - pmu) / psd).clamp(-8, 8)
            s_te = ((s_te - pmu) / psd).clamp(-8, 8)
            id_tr = torch.cat([_identity(ctx_tr, W - 2), _identity(ctx_tr, W - 1)], 1)
            id_te = torch.cat([_identity(ctx_te, W - 2), _identity(ctx_te, W - 1)], 1)
            bank_tr = torch.cat([s_tr, id_tr, B0_tr[:, -N_CONT:]], 1)
            bank_te = torch.cat([s_te, id_te, B0_te[:, -N_CONT:]], 1)
            log(f"  [hand-off] positional bank: {bank_tr.shape[1]} channels "
                f"({len(frozen)} R0 genomes x {W} slots + 2x{V} identity "
                f"+ {N_CONT} continuation)")
        else:
            bank_tr = torch.cat([bank_tr, f_tr], 1)
            bank_te = torch.cat([bank_te, f_te2], 1)
        val_prev = val_now
        if si > 0 and gain < rk.MIN_SPACE_GAIN:
            log(f"  [space {si}] gain {gain:.4f} < {rk.MIN_SPACE_GAIN} — done")
            break
        if os.path.exists(_STOP):
            log("[radial-lm-word] STOP lever pulled")
            break

    def _head_topk(Ftr, Fte, ks=(1, 5)):
        """fp64 ridge head, lam picked on the val split, test touched once;
        returns {k: top-k accuracy}."""
        def _fit(Xf, Yf, lam):
            n, d = Xf.shape
            hm, hs = Xf.mean(0), Xf.std(0) + 1e-6
            A = torch.hstack([(Xf - hm) / hs, torch.ones(n, 1, device=dev)])
            G = (A.T @ A).double() + lam * torch.eye(d + 1, device=dev,
                                                     dtype=torch.float64)
            Wm = torch.linalg.solve(G, (A.T @ Yf).double()).float()
            return hm, hs, Wm

        best_lam, best_v = 3.0, -1.0
        for lam in (3.0, 10.0, 30.0):
            hm, hs, Wm = _fit(Ftr[:n_fit], Yf, lam)
            s = torch.hstack([(Ftr[n_fit:] - hm) / hs,
                              torch.ones(Ntr - n_fit, 1, device=dev)]) @ Wm
            v = float((s.argmax(1) == yv).float().mean())
            if v > best_v:
                best_lam, best_v = lam, v
        hm, hs, Wm = _fit(Ftr, Yfull, best_lam)
        s = torch.hstack([(Fte - hm) / hs,
                          torch.ones(Nte, 1, device=dev)]) @ Wm
        out = {}
        for k in ks:
            tk = s.topk(k, dim=1).indices
            out[k] = round(float((tk == yte_t.view(-1, 1)).any(1)
                                 .float().mean()), 4)
        return out, (hm, hs, Wm)

    Ftr = torch.stack(all_tr, 1)
    Fte = torch.stack(all_te, 1)
    stack_k, head = _head_topk(Ftr, Fte)
    anchor_k, _ = _head_topk(B0_tr, B0_te)
    best = stack_k[1]

    # decode calibration: ridge margins are nearly flat, so raw softmax
    # samples word salad. Fit ONE scale on the val split's log-likelihood
    hm, hs, Wm = head
    s_val = torch.hstack([(Ftr[n_fit:] - hm) / hs,
                          torch.ones(Ntr - n_fit, 1, device=dev)]) @ Wm
    s_cal, best_nll = 1.0, 1e9
    for sc in (1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 35.0):
        nll = float(-torch.log_softmax(s_val * sc, 1)
                    [torch.arange(Ntr - n_fit), yv].mean())
        if nll < best_nll:
            s_cal, best_nll = sc, nll
    log(f"  [decode] calibrated scale {s_cal} (val nll {best_nll:.3f})")

    # ---- generation: slide a 6-word window, sample from the head ------
    def _step_logits(win_ids):
        ctx1 = np.array([win_ids], np.int32)
        B01 = _bank0(ctx1)
        cols = [B01[0]]
        if space_genomes:
            r1 = _rows(ctx1)
            slot_vals = [
                _san(rk.feature_vec(torch, tp, r1, g)).view(1, W)
                for g in space_genomes[0]]
            f0 = torch.cat([v.mean(1, keepdim=True) for v in slot_vals], 1)
            zmu, zsd = space_stats[0]
            cols.append((((f0 - zmu) / zsd).clamp(-8, 8))[0])
            s1 = torch.cat(slot_vals, 1)
            pmu, psd = handoff_stats
            bank = torch.cat([((s1 - pmu) / psd).clamp(-8, 8),
                              _identity(ctx1, W - 2), _identity(ctx1, W - 1),
                              B01[:, -N_CONT:]], 1)
            for sp, (zmu, zsd) in zip(space_genomes[1:], space_stats[1:]):
                f = torch.stack([_san(rk.feature_vec(torch, tp, bank, g))
                                 for g in sp], 1)
                f = ((f - zmu) / zsd).clamp(-8, 8)
                cols.append(f[0])
                bank = torch.cat([bank, f], 1)
        F1 = torch.cat(cols).view(1, -1)
        hm, hs, Wm = head
        return (torch.hstack([(F1 - hm) / hs,
                              torch.ones(1, 1, device=dev)]) @ Wm)[0]

    def _gen(prompt_words, n_words=24, temp=0.8, seed=0):
        grng = np.random.default_rng(seed)
        win = [w2i.get(w, -1) for w in prompt_words][-W:]
        while len(win) < W:
            win.insert(0, -1)
        out_words = []
        for _ in range(n_words):
            lg = _step_logits(win).detach().cpu().numpy().astype(np.float64)
            lg = lg * s_cal / max(temp, 1e-3)     # calibrated sharpness
            for wd_r in out_words[-8:]:           # repetition penalty
                kr = tgt_i.get(wd_r)
                if kr is not None:
                    lg[kr] -= 2.0
            top = np.argsort(lg)[-5:]             # sample within top-5
            z = lg[top] - lg[top].max()
            p = np.exp(z)
            p /= p.sum()
            k = int(grng.choice(top, p=p))
            wd = targets[k]
            out_words.append(wd)
            win = win[1:] + [w2i.get(wd, -1)]
        return " ".join(prompt_words) + " | " + " ".join(out_words)

    samples = []
    for i, pr in enumerate([["the", "first", "part", "of", "the"],
                            ["he", "was", "born", "in", "the"],
                            ["one", "of", "the", "most", "important"]]):
        try:
            samples.append({"prompt": " ".join(pr), "temp": 0.8,
                            "text": _gen(pr, seed=i)})
        except Exception as exc:
            samples.append({"prompt": " ".join(pr), "temp": 0.8,
                            "text": f"(generation failed: {exc})"})
    for s_ in samples:
        print("  sample:", s_["text"], flush=True)

    # trigram top-5 ceiling: top-5 continuations by count, with backoff
    hits5 = 0
    for i in range(Nte):
        j1, j2 = int(ctx_te[i, W - 2]), int(ctx_te[i, W - 1])
        w1 = vocab[j1] if j1 >= 0 else None
        w2 = vocab[j2] if j2 >= 0 else None
        cand = []
        key = (w1, w2)
        if key in tri_c:
            cand = sorted(tri_c[key], key=tri_c[key].get, reverse=True)[:5]
        if len(cand) < 5 and w2 in bi_c:
            for w in sorted(bi_c[w2], key=bi_c[w2].get, reverse=True):
                if w not in cand:
                    cand.append(w)
                if len(cand) == 5:
                    break
        if len(cand) < 5:
            for w in sorted(uni_c, key=uni_c.get, reverse=True):
                if w not in cand:
                    cand.append(w)
                if len(cand) == 5:
                    break
        hits5 += int(targets[yte[i]] in cand)
    trigram_top5 = round(hits5 / Nte, 4)

    vocab_arr = [str(w) for w in z["targets"]]
    ctx_words_te = np.array([[vocab[c] if c >= 0 else "<oov>" for c in row]
                             for row in ctx_te])
    baselines = word_ngram_baselines(ctx_words_te, yte, vocab_arr)
    out = {"phase": "radial-lm-word (embeddings in, next word out)",
           "context_words": W, "vocab": V, "n_train": Ntr, "n_test": Nte,
           "coverage": round(float(z["coverage"]), 4),
           "chance": round(1.0 / V, 4),
           "n_spaces": len(spaces), "space_caps": [s["n_frozen"] for s in spaces],
           "test_acc": round(best, 4),
           "test_top5": stack_k[5],
           "anchor_top5": anchor_k[5],
           "trigram_top5": trigram_top5,
           "samples": samples,
           "embed_ridge_test": round(float(anchor_k[1]), 4),
           "val_final": spaces[-1]["val_after"] if spaces else round(float(oh_val), 4),
           "spaces": spaces, "baselines": baselines,
           "n_genomes": sum(len(sp) for sp in space_genomes),
           "task": f"predict the NEXT WORD (top-{V} vocab) from {W} context "
                   "words entering as 128-d corpus-SVD embeddings; the "
                   "embed-ridge anchor is linear-from-embeddings — genomes "
                   "only earn by composing interactions",
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(_HERE, "radial_data", "lm_radial_word.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    model = {"context_words": W, "vocab": V, "spaces": space_genomes,
             "label": "next-word", "input": "wiki-svd-embeddings"}
    with open(os.path.join(_HERE, "radial_data", "lm_model_word.json"), "w") as f:
        json.dump(model, f)
    rk._record_run(
        {"env": "lm-word", "context_words": W, "vocab": V,
         "n_train": Ntr, "n_test": Nte, "pop_size": pop_size, "gens": gens,
         "max_rounds": max_rounds, "seed": seed},
        [{"round": s["space"], "added": s["n_frozen"],
          "val_acc": s["val_after"], "n": s["n_frozen"]} for s in spaces],
        {"test_acc": best, "test_top5": stack_k[5],
         "anchor_top1": anchor_k[1], "anchor_top5": anchor_k[5],
         "trigram_top5": trigram_top5,
         "n_frozen_total": sum(len(sp) for sp in space_genomes),
         "baselines": baselines},
        log_lines, ["lm", "word", "radial"])
    print(f"[radial-lm-word] DONE: {len(spaces)} spaces "
          f"{out['space_caps']}, anchor top1 {anchor_k[1]:.4f} top5 "
          f"{anchor_k[5]:.4f}, stack TEST top1 {best:.4f} TOP5 "
          f"{stack_k[5]:.4f} vs {baselines} (trigram top5 {trigram_top5}) "
          f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if not os.path.exists(os.path.join(_HERE, "radial_data", "lm_word.npz")) \
            or "--regen" in sys.argv:
        make_word_data()
    run()
