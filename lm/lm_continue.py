"""lm_continue.py - probe the lean word model's weaknesses, then repair
them by CONTINUE-TRAINING: the animation line's Module-3 method (keep the
evolved model frozen as a warm base, stack NEW spaces on a failure-weighted
mix). Never rebuild, never grow a fat head.

  probe()    - read-only. Replays the lean checkpoint, fits its head, and
               breaks test accuracy down by where the continuation tables
               stand: trigram hit / bigram backoff / unigram backoff, and
               whether the table's own top-5 contains the target (the
               composition-needed cases). Emits the worst target words and
               20 raw example predictions.
  continue_train() - warm start. Frozen genome banks unchanged; new spaces
               evolve on a mix that over-weights the probe's failure class
               (table-top5-miss windows, weight 3x). Head refit once on the
               ORIGINAL train distribution; the SAME untouched test set is
               scored once and the breakdown recomputed for deltas.

Everything gradient-free. Run on the pod:
    python lm_continue.py probe
    python lm_continue.py continue
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import json
import os
import pickle
import sys
import time

import numpy as np

from radial_evo import _tprims, _STOP
import radial_lm_word as rw
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD = os.path.join(_HERE, "radial_data")


def _san(torch, v):
    return torch.nan_to_num(v, nan=0.0, posinf=0.0,
                            neginf=0.0).clamp(-1e6, 1e6)


class LeanModel:
    """Rebuild the lean checkpoint's world: banks, per-space stats, head."""

    def __init__(self):
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tp = _tprims(torch)
        with open(os.path.join(RD, "lm_model_word_lean.json")) as f:
            self.ckpt = json.load(f)
        rw.W = self.W = int(self.ckpt["context_words"])
        rw.V = self.V = int(self.ckpt["vocab"])
        self.D = rw.D

        self.vocab, self.feat, _ = rw._load_embed()
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        z = np.load(os.path.join(RD, "lm_word.npz"), allow_pickle=True)
        assert z["ctx_tr"].shape[1] == self.W
        self.ctx_tr, self.ytr = z["ctx_tr"], z["ytr"]
        self.ctx_te, self.yte = z["ctx_te"], z["yte"]
        self.targets = [str(w) for w in z["targets"]]
        self.tgt_i = {w: k for k, w in enumerate(self.targets)}
        self.Ntr, self.Nte = len(self.ytr), len(self.yte)

        with open(os.path.join(RD, "lm_cont_tables.pkl"), "rb") as f:
            self.uni_c, self.bi_c, self.tri_c = pickle.load(f)

        t0 = time.time()
        self._build_env()
        self._replay()
        self._fit_head()
        print(f"[lean-model] rebuilt: test top1 {self.base_top1:.4f} "
              f"({round(time.time()-t0)}s)", flush=True)

    # -- environment (mirrors radial_lm_word.run, RS embeddings) --------
    def _build_env(self):
        torch, dev = self.torch, self.dev
        W, V, D = self.W, self.V, self.D
        feat_t = torch.tensor(self.feat, device=dev)
        self.feat_t = feat_t
        self.mu, self.sd = feat_t.mean(0), feat_t.std(0) + 1e-6
        self.tv = {self.w2i[w]: k for k, w in enumerate(self.targets)
                   if w in self.w2i}

        def _cont_vec(dist):
            v = np.zeros(D, np.float32)
            tot = 0
            for w, c in dist.items():
                j = self.w2i.get(w)
                if j is not None:
                    v += c * self.feat[j]
                    tot += c
            return v / tot if tot else v

        def _cont_prob(dist):
            v = np.zeros(V, np.float32)
            for w, c in dist.items():
                k = self.tgt_i.get(w)
                if k is not None:
                    v[k] = c
            s = v.sum()
            return v / s if s else v

        uni_vec, uni_prob = _cont_vec(self.uni_c), _cont_prob(self.uni_c)
        self.N_CONT = 2 * D + V

        def cont_raw(ctx):
            N = len(ctx)
            out = np.zeros((N, self.N_CONT), np.float32)
            for i in range(N):
                j1, j2 = int(ctx[i, W - 2]), int(ctx[i, W - 1])
                w1 = self.vocab[j1] if j1 >= 0 else None
                w2 = self.vocab[j2] if j2 >= 0 else None
                key = (w1, w2)
                if key in self.tri_c:
                    out[i, :D] = _cont_vec(self.tri_c[key])
                    out[i, 2 * D:] = _cont_prob(self.tri_c[key])
                elif w2 in self.bi_c:
                    out[i, :D] = _cont_vec(self.bi_c[w2])
                    out[i, 2 * D:] = _cont_prob(self.bi_c[w2])
                else:
                    out[i, :D] = uni_vec
                    out[i, 2 * D:] = uni_prob
                out[i, D:2 * D] = (_cont_vec(self.bi_c[w2])
                                   if w2 in self.bi_c else uni_vec)
            return torch.tensor(out, device=dev)

        self.cont_tr = cont_raw(self.ctx_tr)
        self.cont_te = cont_raw(self.ctx_te)
        self.cmu = self.cont_tr.mean(0)
        self.csd = self.cont_tr.std(0) + 1e-6

        def identity(ctx, slot):
            N = len(ctx)
            M = torch.zeros((N, V), device=dev)
            rows, cols = [], []
            for i in range(N):
                k = self.tv.get(int(ctx[i, slot]), -1)
                if k >= 0:
                    rows.append(i); cols.append(k)
            M[torch.tensor(rows, device=dev, dtype=torch.long),
              torch.tensor(cols, device=dev, dtype=torch.long)] = 1.0
            return M

        def embed_bank(ctx):
            idx = torch.tensor(np.maximum(ctx.astype(np.int64), 0), device=dev)
            mask = torch.tensor((ctx >= 0).astype(np.float32), device=dev)
            cols = []
            for f in range(W):
                v = feat_t[idx[:, f]] * mask[:, f:f + 1]
                cols.append((v - self.mu) / self.sd)
            return torch.cat(cols, 1)

        def bank0(ctx, cont):
            return torch.cat([embed_bank(ctx), identity(ctx, W - 2),
                              identity(ctx, W - 1),
                              ((cont - self.cmu) / self.csd).clamp(-8, 8)], 1)

        self.B0_tr = bank0(self.ctx_tr, self.cont_tr)
        self.B0_te = bank0(self.ctx_te, self.cont_te)

        featV_np = np.zeros((V, D), np.float32)
        for k, wd in enumerate(self.targets):
            j = self.w2i.get(wd)
            if j is not None:
                featV_np[k] = self.feat[j]
        featV = torch.tensor(featV_np, device=dev)
        featV = featV / (featV.norm(dim=1, keepdim=True) + 1e-6)

        def slot_scores(ctx):
            idx = torch.tensor(np.maximum(ctx.astype(np.int64), 0), device=dev)
            mask = torch.tensor((ctx >= 0).astype(np.float32), device=dev)
            out = []
            for f in range(W):
                e = feat_t[idx[:, f]] * mask[:, f:f + 1]
                e = e / (e.norm(dim=1, keepdim=True) + 1e-6)
                out.append((e @ featV.T).half())
            return out

        self.ss_tr = slot_scores(self.ctx_tr)
        self.ss_te = slot_scores(self.ctx_te)
        self.prob_tr = self.cont_tr[:, 2 * D:].clone()
        self.prob_te = self.cont_te[:, 2 * D:].clone()
        del self.cont_tr, self.cont_te      # slices live inside B0 now
        torch.cuda.empty_cache()

    def _feat(self, g, bank, ss, prob):
        torch = self.torch
        if g.get("kind") == "attend":
            return _san(torch, rw.feature_attend(torch, g, ss, prob))
        return _san(torch, rk.feature_vec(torch, self.tp, bank, g))

    def _replay(self):
        torch = self.torch
        self.stats = []
        self.cols_tr, self.cols_te = [], []
        bank_tr, bank_te = self.B0_tr, self.B0_te
        for sp in self.ckpt["spaces"]:
            f_tr = torch.stack([self._feat(g, bank_tr, self.ss_tr,
                                           self.prob_tr) for g in sp], 1)
            f_te = torch.stack([self._feat(g, bank_te, self.ss_te,
                                           self.prob_te) for g in sp], 1)
            zmu, zsd = f_tr.mean(0), f_tr.std(0) + 1e-6
            self.stats.append((zmu, zsd))
            f_tr = ((f_tr - zmu) / zsd).clamp(-8, 8)
            f_te = ((f_te - zmu) / zsd).clamp(-8, 8)
            self.cols_tr.append(f_tr)
            self.cols_te.append(f_te)
            bank_tr = torch.cat([bank_tr, f_tr], 1)
            bank_te = torch.cat([bank_te, f_te], 1)
        self.bank_tr, self.bank_te = bank_tr, bank_te

    def _head_cols(self, extra_tr=None, extra_te=None):
        torch, D, V = self.torch, self.D, self.V
        tr = list(self.cols_tr) + ([extra_tr] if extra_tr is not None else [])
        te = list(self.cols_te) + ([extra_te] if extra_te is not None else [])
        # hybrid head: + the continuation-EMBEDDING block (never prob/identity)
        ce_tr = self.B0_tr[:, -(self.N_CONT):-V]
        ce_te = self.B0_te[:, -(self.N_CONT):-V]
        return (torch.cat(tr + [ce_tr], 1), torch.cat(te + [ce_te], 1))

    def _fit_head(self, extra_tr=None, extra_te=None):
        torch, dev = self.torch, self.dev
        Ftr, Fte = self._head_cols(extra_tr, extra_te)
        Ntr, V = self.Ntr, self.V
        Y = -torch.ones((Ntr, V), device=dev)
        Y[torch.arange(Ntr), torch.tensor(self.ytr, device=dev)] = 1.0
        n_fit = int(Ntr * 0.8)
        yv = torch.tensor(self.ytr[n_fit:], device=dev)

        def fit(Xf, Yf, lam):
            n, d = Xf.shape
            hm, hs = Xf.mean(0), Xf.std(0) + 1e-6
            A = torch.hstack([(Xf - hm) / hs, torch.ones(n, 1, device=dev)])
            G = (A.T @ A).double() + lam * torch.eye(
                d + 1, device=dev, dtype=torch.float64)
            Wm = torch.linalg.solve(G, (A.T @ Yf).double()).float()
            return hm, hs, Wm

        best = (3.0, -1.0)
        for lam in (1.0, 3.0, 10.0, 30.0):
            hm, hs, Wm = fit(Ftr[:n_fit], Y[:n_fit], lam)
            s = torch.hstack([(Ftr[n_fit:] - hm) / hs,
                              torch.ones(Ntr - n_fit, 1, device=dev)]) @ Wm
            a = float((s.argmax(1) == yv).float().mean())
            if a > best[1]:
                best = (lam, a)
        hm, hs, Wm = fit(Ftr, Y, best[0])
        self._hm, self._hs, self._Wm = hm, hs, Wm
        s = torch.hstack([(Fte - hm) / hs,
                          torch.ones(self.Nte, 1, device=dev)]) @ Wm
        self.preds = s.argmax(1).cpu().numpy()
        top5 = s.topk(5, dim=1).indices.cpu().numpy()
        self.top5_hit = (top5 == self.yte[:, None]).any(1)
        self.base_top1 = float((self.preds == self.yte).mean())
        self.base_top5 = float(self.top5_hit.mean())
        self.head_params = int(Wm.numel())

    # -- table diagnostics per window ------------------------------------
    def table_info(self, ctx, y):
        W = self.W
        level = np.zeros(len(y), np.int8)       # 2 tri / 1 bi / 0 uni
        tab5 = np.zeros(len(y), bool)           # table top5 contains target
        for i in range(len(y)):
            j1, j2 = int(ctx[i, W - 2]), int(ctx[i, W - 1])
            w1 = self.vocab[j1] if j1 >= 0 else None
            w2 = self.vocab[j2] if j2 >= 0 else None
            key = (w1, w2)
            if key in self.tri_c:
                level[i] = 2
                dist = self.tri_c[key]
            elif w2 in self.bi_c:
                level[i] = 1
                dist = self.bi_c[w2]
            else:
                dist = self.uni_c
            top = sorted(dist, key=dist.get, reverse=True)[:5]
            tab5[i] = self.targets[y[i]] in top
        return level, tab5


def probe():
    m = LeanModel()
    lev, tab5 = m.table_info(m.ctx_te, m.yte)
    hit = m.preds == m.yte
    br = {}
    for name, mask in [("trigram-hit", lev == 2), ("bigram-backoff", lev == 1),
                       ("unigram-backoff", lev == 0),
                       ("table-top5-has-target", tab5),
                       ("table-top5-MISSES-target", ~tab5)]:
        n = int(mask.sum())
        br[name] = {"n": n, "top1": round(float(hit[mask].mean()), 4) if n else None,
                    "top5": round(float(m.top5_hit[mask].mean()), 4) if n else None}
        print(f"  {name:26s} n={n:6d} top1 {br[name]['top1']} "
              f"top5 {br[name]['top5']}", flush=True)

    miss_counts = {}
    for i in range(m.Nte):
        if not hit[i]:
            t = m.targets[m.yte[i]]
            miss_counts.setdefault(t, [0, {}])
            miss_counts[t][0] += 1
            p = m.targets[m.preds[i]]
            miss_counts[t][1][p] = miss_counts[t][1].get(p, 0) + 1
    worst = sorted(miss_counts.items(), key=lambda kv: -kv[1][0])[:15]
    worst_out = [{"target": t, "misses": c,
                  "usually_predicted": max(d, key=d.get)}
                 for t, (c, d) in worst]
    for wj in worst_out:
        print(f"  MISS {wj['target']:14s} x{wj['misses']:4d} -> usually "
              f"'{wj['usually_predicted']}'", flush=True)

    rng = np.random.default_rng(3)
    ex = []
    for i in rng.choice(m.Nte, 20, replace=False):
        i = int(i)
        ctxw = " ".join(m.vocab[c] if c >= 0 else "?" for c in m.ctx_te[i])
        ex.append({"context": ctxw, "pred": m.targets[m.preds[i]],
                   "true": m.targets[m.yte[i]],
                   "correct": bool(hit[i])})
    out = {"phase": "lean-model probe (read-only)",
           "test_acc": round(m.base_top1, 4), "test_top5": round(m.base_top5, 4),
           "vocab": m.V, "context_words": m.W,
           "n_genomes": sum(len(sp) for sp in m.ckpt["spaces"]),
           "breakdown": br, "worst_targets": worst_out, "examples": ex,
           "head_params": m.head_params}
    with open(os.path.join(RD, "lm_probe_lean.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[probe] DONE top1 {m.base_top1:.4f} top5 {m.base_top5:.4f} -> "
          f"lm_probe_lean.json", flush=True)
    return out


def continue_train(pop_size=64, gens=12, max_rounds=600, seed=11,
                   max_new_spaces=8, miss_weight=3.0, mix_size=60000):
    import torch
    m = LeanModel()
    dev = m.dev
    t0 = time.time()
    log_lines = []

    def log(msg, v=True):
        log_lines.append(msg)
        if v:
            print(msg, flush=True)

    # failure-weighted mix over TRAIN windows: over-sample the windows the
    # TABLES cannot answer (top-5 miss) - the composition-needed class.
    # A SUBSAMPLE, not a full resample: the attend substrate is ~10GB and
    # duplicating it full-size OOMs even 96GB (measured).
    lev_tr, tab5_tr = m.table_info(m.ctx_tr, m.ytr)
    wgt = 1.0 + miss_weight * (~tab5_tr)
    rng = np.random.default_rng(seed)
    mix = rng.choice(m.Ntr, size=mix_size, p=wgt / wgt.sum())
    mix_t = torch.tensor(mix, device=dev, dtype=torch.long)
    log(f"[continue] mix: {mix_size} rows, {int((~tab5_tr).sum())}/{m.Ntr} "
        f"table-miss windows at weight {miss_weight}x")

    bank_mix = m.bank_tr[mix_t]
    ss_mix = [s[mix_t] for s in m.ss_tr]
    prob_mix = m.prob_tr[mix_t]
    old_cols_mix = torch.cat(m.cols_tr, 1)[mix_t]
    torch.cuda.empty_cache()
    n_fit = int(mix_size * 0.8)
    ymix = m.ytr[mix]
    yv = torch.tensor(ymix[n_fit:], device=dev)
    Yf = -torch.ones((n_fit, m.V), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ymix[:n_fit], device=dev)] = 1.0

    new_spaces, new_stats = [], []
    new_mix_cols, new_tr_cols, new_te_cols = [], [], []
    bank_tr_full, bank_te_full = m.bank_tr, m.bank_te
    val_prev = None
    from radial_evo import _ridge_soft
    for si in range(max_new_spaces):
        C = bank_mix.shape[1]

        def new_fn(r, C=C):
            if r.random() < 0.5:
                return rw.new_attend_genome(r, m.W)
            return rk.new_vec_genome(r, C)

        def mut_fn(r, g, sc, C=C):
            if g.get("kind") == "attend":
                return rw.mutate_attend(r, g, sc)
            return rk.mutate_vec(r, g, sc, C)

        def feat_mix(g, b=bank_mix):
            return m._feat(g, b, ss_mix, prob_mix)

        base_prev = torch.cat([old_cols_mix] + new_mix_cols, 1)
        log(f"  [cont space {si}] opening - warm base {base_prev.shape[1]} "
            f"frozen cols, bank {C} channels")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_mix, log, True)
        if not frozen:
            log(f"  [cont space {si}] produced nothing - stop")
            break
        new_spaces.append(frozen)
        f_mix = torch.stack(fcols, 1)
        zmu, zsd = f_mix.mean(0), f_mix.std(0) + 1e-6
        new_stats.append((zmu, zsd))
        f_mix = ((f_mix - zmu) / zsd).clamp(-8, 8)
        # the same genomes evaluated on the FULL original train + test rows
        f_tr = torch.stack([m._feat(g, bank_tr_full, m.ss_tr, m.prob_tr)
                            for g in frozen], 1)
        f_te = torch.stack([m._feat(g, bank_te_full, m.ss_te, m.prob_te)
                            for g in frozen], 1)
        f_tr = ((f_tr - zmu) / zsd).clamp(-8, 8)
        f_te = ((f_te - zmu) / zsd).clamp(-8, 8)
        new_mix_cols.append(f_mix)
        new_tr_cols.append(f_tr)
        new_te_cols.append(f_te)
        bank_mix = torch.cat([bank_mix, f_mix], 1)
        bank_tr_full = torch.cat([bank_tr_full, f_tr], 1)
        bank_te_full = torch.cat([bank_te_full, f_te], 1)
        base_all = torch.cat([old_cols_mix] + new_mix_cols, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                                 Yf, yv)
        gain = None if val_prev is None else val_now - val_prev
        log(f"  [cont space {si}] FULL: {len(frozen)} genomes, mix-val "
            f"{val_now:.4f}" + (f" (+{gain:.4f})" if gain is not None else "")
            + f" ({round(time.time()-t0)}s)")
        if gain is not None and gain < rk.MIN_SPACE_GAIN:
            log(f"  [cont space {si}] gain {gain:.4f} - done")
            break
        val_prev = val_now
        if os.path.exists(_STOP):
            break

    # head refit ONCE on the ORIGINAL train distribution; test scored once
    extra_tr = torch.cat(new_tr_cols, 1) if new_tr_cols else None
    extra_te = torch.cat(new_te_cols, 1) if new_te_cols else None
    m._fit_head(extra_tr, extra_te)
    lev, tab5 = m.table_info(m.ctx_te, m.yte)
    hit = m.preds == m.yte
    br = {}
    for name, mask in [("trigram-hit", lev == 2), ("bigram-backoff", lev == 1),
                       ("unigram-backoff", lev == 0),
                       ("table-top5-has-target", tab5),
                       ("table-top5-MISSES-target", ~tab5)]:
        n = int(mask.sum())
        br[name] = {"n": n, "top1": round(float(hit[mask].mean()), 4) if n else None,
                    "top5": round(float(m.top5_hit[mask].mean()), 4) if n else None}
        log(f"  AFTER {name:26s} top1 {br[name]['top1']} top5 {br[name]['top5']}")

    rng2 = np.random.default_rng(3)
    ex = []
    for i in rng2.choice(m.Nte, 20, replace=False):
        i = int(i)
        ctxw = " ".join(m.vocab[c] if c >= 0 else "?" for c in m.ctx_te[i])
        ex.append({"context": ctxw, "pred": m.targets[m.preds[i]],
                   "true": m.targets[m.yte[i]], "correct": bool(hit[i])})

    from anim_infer import count_params
    n_old = sum(len(sp) for sp in m.ckpt["spaces"])
    n_new = sum(len(sp) for sp in new_spaces)
    gp = (sum(count_params(g) for sp in m.ckpt["spaces"] for g in sp)
          + sum(count_params(g) for sp in new_spaces for g in sp))
    out = {"phase": "continue-train on table-miss mix (warm start)",
           "test_acc": round(m.base_top1, 4), "test_top5": round(m.base_top5, 4),
           "vocab": m.V, "context_words": m.W,
           "n_genomes": n_old + n_new, "n_new_genomes": n_new,
           "n_spaces": len(m.ckpt["spaces"]) + len(new_spaces),
           "spaces": [{"space": i, "n_frozen": len(sp),
                       "val_after": None, "val_gain": None}
                      for i, sp in enumerate(new_spaces)],
           "genome_params": gp, "head_params": m.head_params,
           "total_params": gp + m.head_params,
           "breakdown": br, "examples": ex,
           "miss_weight": miss_weight,
           "task": f"warm start on the frozen lean model ({n_old} genomes "
                   f"kept); {n_new} new genomes evolved on a {miss_weight}x "
                   "table-miss-weighted mix; head refit once on the original "
                   "train distribution; same untouched test set",
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "lm_radial_word_cont.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, "lm_model_word_cont.json"), "w") as f:
        json.dump({"context_words": m.W, "vocab": m.V,
                   "spaces": m.ckpt["spaces"] + new_spaces,
                   "label": "next-word", "mode": "lean-cont"}, f)
    rk._record_run(
        {"env": "lm-word-continue", "vocab": m.V, "context_words": m.W,
         "miss_weight": miss_weight, "pop_size": pop_size, "seed": seed},
        [{"round": i, "added": len(sp), "val_acc": None, "n": len(sp)}
         for i, sp in enumerate(new_spaces)],
        {"test_acc": round(m.base_top1, 4), "test_top5": round(m.base_top5, 4),
         "n_new_genomes": n_new, "total_params": gp + m.head_params,
         "breakdown": br},
        log_lines, ["lm", "word", "continue", "radial"])
    print(f"[continue] DONE: +{n_new} genomes in {len(new_spaces)} new "
          f"spaces, TEST top1 {m.base_top1:.4f} top5 {m.base_top5:.4f} "
          f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    rw.EMBED_NPZ = os.path.join(RD, "embed_rs.npz")
    if "continue" in sys.argv:
        continue_train()
    else:
        probe()
