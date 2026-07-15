"""lm_continue2.py - Module 14: give the environment WIDER CONTEXT, then
continue-train the same frozen base.

Module 13 measured that the table-miss class (38% of test, top-1 0.004)
cannot be repaired by more search: the signal is absent from the
environment. This module adds it as corpus statistics, same independent
slice, same legitimacy as the existing tables:

  skipA - continuation distribution keyed on (w[-3], w[-1]): wider left
          context WITH the adjacent word
  skipB - continuation distribution keyed on (w[-3], w[-2]): the pair
          BEFORE the last word, predicting across it

Each contributes a probability vector over the V targets and an expected
embedding (2V + 2D new bank channels). Attend genomes get a SRC gene so
content addressing can attend over the NEW candidate lists, not just the
old trigram one.

Before evolving anything, the HEADROOM diagnostic: what fraction of the
miss-class windows have the target inside skipA/skipB top-5 - the ceiling
this repair can possibly reach. Printed first, so a null result is
interpretable.

Warm start otherwise identical to Module 13: frozen lean base, 3x
miss-weighted subsampled mix, head refit once on the true distribution,
untouched test set scored once.

    python lm_continue2.py
"""
import json
import os
import pickle
import time

import numpy as np

from radial_evo import _STOP
from lm_continue import LeanModel, RD, _san
import radial_lm_word as rw
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))


def _build_skip_tables():
    pkl = os.path.join(RD, "lm_skip_tables.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    from radial_lm import _clean
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        f.seek(30_000_000)               # the same independent slice
        toks = _clean(f.read(16_000_000)).split()
    skipA, skipB = {}, {}
    for i in range(3, len(toks)):
        a3, a2, a1, t = toks[i - 3], toks[i - 2], toks[i - 1], toks[i]
        kA = (a3, a1)
        skipA.setdefault(kA, {})
        skipA[kA][t] = skipA[kA].get(t, 0) + 1
        kB = (a3, a2)
        skipB.setdefault(kB, {})
        skipB[kB][t] = skipB[kB].get(t, 0) + 1
    with open(pkl, "wb") as f:
        pickle.dump((skipA, skipB), f)
    return skipA, skipB


class Env2(LeanModel):
    def __init__(self):
        super().__init__()
        torch, dev = self.torch, self.dev
        t0 = time.time()
        self.skipA, self.skipB = _build_skip_tables()
        W, V, D = self.W, self.V, self.D

        def prob_and_emb(ctx, table, sa, sb):
            N = len(ctx)
            P = np.zeros((N, V), np.float32)
            E = np.zeros((N, D), np.float32)
            for i in range(N):
                ja, jb = int(ctx[i, sa]), int(ctx[i, sb])
                wa = self.vocab[ja] if ja >= 0 else None
                wb = self.vocab[jb] if jb >= 0 else None
                dist = table.get((wa, wb))
                if not dist:
                    continue
                tot = 0
                for w, c in dist.items():
                    k = self.tgt_i.get(w)
                    if k is not None:
                        P[i, k] = c
                    j = self.w2i.get(w)
                    if j is not None:
                        E[i] += c * self.feat[j]
                        tot += c
                s = P[i].sum()
                if s:
                    P[i] /= s
                if tot:
                    E[i] /= tot
            return (torch.tensor(P, device=dev), torch.tensor(E, device=dev))

        W3, W1, W2 = W - 3, W - 1, W - 2
        self.skA_p_tr, skA_e_tr = prob_and_emb(self.ctx_tr, self.skipA, W3, W1)
        self.skA_p_te, skA_e_te = prob_and_emb(self.ctx_te, self.skipA, W3, W1)
        self.skB_p_tr, skB_e_tr = prob_and_emb(self.ctx_tr, self.skipB, W3, W2)
        self.skB_p_te, skB_e_te = prob_and_emb(self.ctx_te, self.skipB, W3, W2)

        new_tr = torch.cat([self.skA_p_tr, self.skB_p_tr, skA_e_tr, skB_e_tr], 1)
        new_te = torch.cat([self.skA_p_te, self.skB_p_te, skA_e_te, skB_e_te], 1)
        nmu, nsd = new_tr.mean(0), new_tr.std(0) + 1e-6
        self.nmu, self.nsd = nmu, nsd
        self.new_tr = ((new_tr - nmu) / nsd).clamp(-8, 8)
        self.new_te = ((new_te - nmu) / nsd).clamp(-8, 8)
        # the skip EMBEDDING blocks (2D cols at the end) also join the head
        self.new_emb_tr = self.new_tr[:, 2 * V:]
        self.new_emb_te = self.new_te[:, 2 * V:]
        print(f"[env2] skip channels: +{self.new_tr.shape[1]} "
              f"({round(time.time()-t0)}s)", flush=True)

    def headroom(self):
        """Of the miss-class test windows, how many could the new tables
        answer (target in skipA/skipB top-5)?"""
        W = self.W
        _, tab5 = self.table_info(self.ctx_te, self.yte)
        miss = ~tab5
        inA = np.zeros(self.Nte, bool)
        inB = np.zeros(self.Nte, bool)
        for i in np.where(miss)[0]:
            t = self.targets[self.yte[i]]
            for arr, table, sa, sb in ((inA, self.skipA, W - 3, W - 1),
                                       (inB, self.skipB, W - 3, W - 2)):
                ja, jb = int(self.ctx_te[i, sa]), int(self.ctx_te[i, sb])
                wa = self.vocab[ja] if ja >= 0 else None
                wb = self.vocab[jb] if jb >= 0 else None
                dist = table.get((wa, wb))
                if dist:
                    top = sorted(dist, key=dist.get, reverse=True)[:5]
                    arr[i] = t in top
        n = int(miss.sum())
        out = {"miss_n": n,
               "skipA_top5": round(float(inA[miss].mean()), 4),
               "skipB_top5": round(float(inB[miss].mean()), 4),
               "either_top5": round(float((inA | inB)[miss].mean()), 4)}
        print(f"[headroom] miss-class n={n}: skipA {out['skipA_top5']} "
              f"skipB {out['skipB_top5']} either {out['either_top5']}",
              flush=True)
        return out


def run(pop_size=64, gens=12, max_rounds=600, seed=13, max_new_spaces=8,
        miss_weight=3.0, mix_size=60000):
    import torch
    m = Env2()
    dev = m.dev
    t0 = time.time()
    log_lines = []

    def log(msg, v=True):
        log_lines.append(msg)
        if v:
            print(msg, flush=True)

    head = m.headroom()

    lev_tr, tab5_tr = m.table_info(m.ctx_tr, m.ytr)
    wgt = 1.0 + miss_weight * (~tab5_tr)
    rng = np.random.default_rng(seed)
    mix = rng.choice(m.Ntr, size=mix_size, p=wgt / wgt.sum())
    mix_t = torch.tensor(mix, device=dev, dtype=torch.long)

    bank_full_tr = torch.cat([m.bank_tr, m.new_tr], 1)
    bank_full_te = torch.cat([m.bank_te, m.new_te], 1)
    bank_mix = bank_full_tr[mix_t]
    ss_mix = [s[mix_t] for s in m.ss_tr]
    probs_tr = {"cont": m.prob_tr, "skipA": m.skA_p_tr, "skipB": m.skB_p_tr}
    probs_te = {"cont": m.prob_te, "skipA": m.skA_p_te, "skipB": m.skB_p_te}
    probs_mix = {k: v[mix_t] for k, v in probs_tr.items()}
    old_cols_mix = torch.cat(m.cols_tr, 1)[mix_t]
    torch.cuda.empty_cache()
    n_fit = int(mix_size * 0.8)
    ymix = m.ytr[mix]
    yv = torch.tensor(ymix[n_fit:], device=dev)
    Yf = -torch.ones((n_fit, m.V), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ymix[:n_fit], device=dev)] = 1.0
    SRCS = list(probs_tr.keys())

    def _feat2(g, bank, ss, probs):
        if g.get("kind") == "attend":
            return _san(m.torch, rw.feature_attend(
                m.torch, g, ss, probs[g.get("src", "cont")]))
        return _san(m.torch, rk.feature_vec(m.torch, m.tp, bank, g))

    new_spaces, new_stats = [], []
    new_mix_cols, new_tr_cols, new_te_cols = [], [], []
    val_prev = None
    from radial_evo import _ridge_soft
    for si in range(max_new_spaces):
        C = bank_mix.shape[1]

        def new_fn(r, C=C):
            if r.random() < 0.6:
                g = rw.new_attend_genome(r, m.W)
                g["src"] = SRCS[int(r.integers(len(SRCS)))]
                return g
            return rk.new_vec_genome(r, C)

        def mut_fn(r, g, sc, C=C):
            if g.get("kind") == "attend":
                g2 = rw.mutate_attend(r, g, sc)
                g2["src"] = (SRCS[int(r.integers(len(SRCS)))]
                             if r.random() < 0.15 else g.get("src", "cont"))
                return g2
            return rk.mutate_vec(r, g, sc, C)

        def feat_mix(g, b=bank_mix):
            return _feat2(g, b, ss_mix, probs_mix)

        base_prev = torch.cat([old_cols_mix] + new_mix_cols, 1)
        log(f"  [cont2 space {si}] opening - warm base {base_prev.shape[1]} "
            f"cols, bank {C} channels, attend srcs {SRCS}")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_mix, log, True)
        if not frozen:
            log(f"  [cont2 space {si}] produced nothing - stop")
            break
        new_spaces.append(frozen)
        f_mix = torch.stack(fcols, 1)
        zmu, zsd = f_mix.mean(0), f_mix.std(0) + 1e-6
        new_stats.append((zmu, zsd))
        f_mix = ((f_mix - zmu) / zsd).clamp(-8, 8)
        f_tr = torch.stack([_feat2(g, bank_full_tr, m.ss_tr, probs_tr)
                            for g in frozen], 1)
        f_te = torch.stack([_feat2(g, bank_full_te, m.ss_te, probs_te)
                            for g in frozen], 1)
        f_tr = ((f_tr - zmu) / zsd).clamp(-8, 8)
        f_te = ((f_te - zmu) / zsd).clamp(-8, 8)
        new_mix_cols.append(f_mix)
        new_tr_cols.append(f_tr)
        new_te_cols.append(f_te)
        bank_mix = torch.cat([bank_mix, f_mix], 1)
        bank_full_tr = torch.cat([bank_full_tr, f_tr], 1)
        bank_full_te = torch.cat([bank_full_te, f_te], 1)
        base_all = torch.cat([old_cols_mix] + new_mix_cols, 1)
        _, val_now = _ridge_soft(torch, base_all[:n_fit], base_all[n_fit:],
                                 Yf, yv)
        gain = None if val_prev is None else val_now - val_prev
        n_att = sum(1 for g in frozen if g.get("kind") == "attend")
        srcs = {}
        for g in frozen:
            if g.get("kind") == "attend":
                srcs[g.get("src", "cont")] = srcs.get(g.get("src", "cont"), 0) + 1
        log(f"  [cont2 space {si}] FULL: {len(frozen)} genomes ({n_att} "
            f"attend {srcs}), mix-val {val_now:.4f}"
            + (f" (+{gain:.4f})" if gain is not None else "")
            + f" ({round(time.time()-t0)}s)")
        if gain is not None and gain < rk.MIN_SPACE_GAIN:
            break
        val_prev = val_now
        if os.path.exists(_STOP):
            break

    # head: old cols + new cols + cont-emb + skip-emb blocks, refit ONCE on
    # the original train distribution
    extra_tr = torch.cat(new_tr_cols + [m.new_emb_tr], 1) if new_tr_cols \
        else m.new_emb_tr
    extra_te = torch.cat(new_te_cols + [m.new_emb_te], 1) if new_te_cols \
        else m.new_emb_te
    m._fit_head(extra_tr, extra_te)
    lev, tab5 = m.table_info(m.ctx_te, m.yte)
    hit = m.preds == m.yte
    br = {}
    for name, mask in [("trigram-hit", lev == 2), ("bigram-backoff", lev == 1),
                       ("unigram-backoff", lev == 0),
                       ("table-top5-has-target", tab5),
                       ("table-top5-MISSES-target", ~tab5)]:
        n = int(mask.sum())
        br[name] = {"n": n,
                    "top1": round(float(hit[mask].mean()), 4) if n else None,
                    "top5": round(float(m.top5_hit[mask].mean()), 4) if n else None}
        log(f"  AFTER {name:26s} top1 {br[name]['top1']} "
            f"top5 {br[name]['top5']}")

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
    out = {"phase": "continue-train 2: skip-gram environment channels",
           "test_acc": round(m.base_top1, 4),
           "test_top5": round(m.base_top5, 4),
           "vocab": m.V, "context_words": m.W,
           "n_genomes": n_old + n_new, "n_new_genomes": n_new,
           "n_spaces": len(m.ckpt["spaces"]) + len(new_spaces),
           "spaces": [{"space": i, "n_frozen": len(sp), "val_after": None,
                       "val_gain": None} for i, sp in enumerate(new_spaces)],
           "genome_params": gp, "head_params": m.head_params,
           "total_params": gp + m.head_params,
           "headroom": head, "breakdown": br, "examples": ex,
           "task": "skipA (w-3,w-1) + skipB (w-3,w-2) continuation channels "
                   "added to the environment; attend genomes address the new "
                   "candidate lists via a src gene; warm start on the frozen "
                   f"lean base ({n_old} genomes kept, {n_new} new)",
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "lm_radial_word_cont2.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, "lm_model_word_cont2.json"), "w") as f:
        json.dump({"context_words": m.W, "vocab": m.V,
                   "spaces": m.ckpt["spaces"] + new_spaces,
                   "label": "next-word", "mode": "lean-cont2"}, f)
    rk._record_run(
        {"env": "lm-word-continue2", "vocab": m.V, "context_words": m.W,
         "miss_weight": miss_weight, "pop_size": pop_size, "seed": seed},
        [{"round": i, "added": len(sp), "val_acc": None, "n": len(sp)}
         for i, sp in enumerate(new_spaces)],
        {"test_acc": round(m.base_top1, 4),
         "test_top5": round(m.base_top5, 4), "n_new_genomes": n_new,
         "headroom": head, "breakdown": br},
        log_lines, ["lm", "word", "continue", "skipgram", "radial"])
    print(f"[continue2] DONE: +{n_new} genomes, TEST top1 "
          f"{m.base_top1:.4f} top5 {m.base_top5:.4f} "
          f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    rw.EMBED_NPZ = os.path.join(RD, "embed_rs.npz")
    run()
