"""lm_continue3.py - Module 15: the second turn of the crank.

Module 14 validated the loop (probe -> environment gap -> headroom ->
continue-train). This module repeats it on the Module-14 model:

  1. rebuild the cont2 model exactly (original lean spaces over the base
     bank, then the skipA/B channels, then the cont2 spaces over the
     extended bank - every genome must see the same channels it evolved on)
  2. add LONGER-RANGE skip tables: skipC keyed (w-4, w-1) and skipD keyed
     (w-5, w-1), from the same independent slice
  3. HEADROOM first: of the windows the cont2 model still gets wrong, how
     many have the target inside skipC/skipD top-5
  4. continue-train the frozen 350-genome base on a still-wrong-weighted
     mix with the new channels + attend srcs {cont, skipA..skipD}
  5. head refit once on the true distribution; untouched test scored once

    python lm_continue3.py
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

from radial_evo import _STOP
from lm_continue import RD, _san
from lm_continue2 import Env2
import radial_lm_word as rw
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_far_tables():
    pkl = os.path.join(RD, "lm_skip_tables2.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    from radial_lm import _clean
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        f.seek(30_000_000)
        toks = _clean(f.read(16_000_000)).split()
    skipC, skipD = {}, {}
    for i in range(5, len(toks)):
        t = toks[i]
        kC = (toks[i - 4], toks[i - 1])
        skipC.setdefault(kC, {})
        skipC[kC][t] = skipC[kC].get(t, 0) + 1
        kD = (toks[i - 5], toks[i - 1])
        skipD.setdefault(kD, {})
        skipD[kD][t] = skipD[kD].get(t, 0) + 1
    with open(pkl, "wb") as f:
        pickle.dump((skipC, skipD), f)
    return skipC, skipD


class Env3(Env2):
    """Env2 (lean base + skipA/B channels) + exact replay of the cont2
    spaces + the new far-skip channels."""

    def __init__(self):
        # Env2.__init__ replays the ORIGINAL lean checkpoint over the base
        # bank and builds skipA/B. We then must replay the cont2 extra
        # spaces over the SAME extended bank they evolved on.
        super().__init__()
        torch, dev = self.torch, self.dev
        t0 = time.time()
        with open(os.path.join(RD, "lm_model_word_cont2.json")) as f:
            cont2 = json.load(f)
        n_orig = len(self.ckpt["spaces"])
        self.cont2_spaces = cont2["spaces"][n_orig:]

        self.probs_tr = {"cont": self.prob_tr, "skipA": self.skA_p_tr,
                         "skipB": self.skB_p_tr}
        self.probs_te = {"cont": self.prob_te, "skipA": self.skA_p_te,
                         "skipB": self.skB_p_te}

        def feat2(g, bank, ss, probs):
            if g.get("kind") == "attend":
                return _san(torch, rw.feature_attend(
                    torch, g, ss, probs[g.get("src", "cont")]))
            return _san(torch, rk.feature_vec(torch, self.tp, bank, g))

        self._feat2 = feat2
        bank_tr = torch.cat([self.bank_tr, self.new_tr], 1)
        bank_te = torch.cat([self.bank_te, self.new_te], 1)
        self.cont2_cols_tr, self.cont2_cols_te = [], []
        self.cont2_stats = []
        for sp in self.cont2_spaces:
            f_tr = torch.stack([feat2(g, bank_tr, self.ss_tr, self.probs_tr)
                                for g in sp], 1)
            f_te = torch.stack([feat2(g, bank_te, self.ss_te, self.probs_te)
                                for g in sp], 1)
            zmu, zsd = f_tr.mean(0), f_tr.std(0) + 1e-6
            self.cont2_stats.append((zmu, zsd))
            f_tr = ((f_tr - zmu) / zsd).clamp(-8, 8)
            f_te = ((f_te - zmu) / zsd).clamp(-8, 8)
            self.cont2_cols_tr.append(f_tr)
            self.cont2_cols_te.append(f_te)
            bank_tr = torch.cat([bank_tr, f_tr], 1)
            bank_te = torch.cat([bank_te, f_te], 1)
        self.bank2_tr, self.bank2_te = bank_tr, bank_te

        # verify the rebuild reproduces Module 14 before adding anything
        extra_tr = torch.cat(self.cont2_cols_tr + [self.new_emb_tr], 1)
        extra_te = torch.cat(self.cont2_cols_te + [self.new_emb_te], 1)
        self._fit_head(extra_tr, extra_te)
        print(f"[env3] cont2 rebuilt: test top1 {self.base_top1:.4f} "
              f"(Module 14 reported 0.3992) ({round(time.time()-t0)}s)",
              flush=True)

        # far-skip channels
        self.skipC, self.skipD = _build_far_tables()
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

        self.skC_p_tr, cE_tr = prob_and_emb(self.ctx_tr, self.skipC, W - 4, W - 1)
        self.skC_p_te, cE_te = prob_and_emb(self.ctx_te, self.skipC, W - 4, W - 1)
        self.skD_p_tr, dE_tr = prob_and_emb(self.ctx_tr, self.skipD, W - 5, W - 1)
        self.skD_p_te, dE_te = prob_and_emb(self.ctx_te, self.skipD, W - 5, W - 1)
        far_tr = torch.cat([self.skC_p_tr, self.skD_p_tr, cE_tr, dE_tr], 1)
        far_te = torch.cat([self.skC_p_te, self.skD_p_te, cE_te, dE_te], 1)
        fmu, fsd = far_tr.mean(0), far_tr.std(0) + 1e-6
        self.fmu, self.fsd = fmu, fsd
        self.far_tr = ((far_tr - fmu) / fsd).clamp(-8, 8)
        self.far_te = ((far_te - fmu) / fsd).clamp(-8, 8)
        self.far_emb_tr = self.far_tr[:, 2 * V:]
        self.far_emb_te = self.far_te[:, 2 * V:]
        self.probs_tr.update({"skipC": self.skC_p_tr, "skipD": self.skD_p_tr})
        self.probs_te.update({"skipC": self.skC_p_te, "skipD": self.skD_p_te})
        torch.cuda.empty_cache()
        print(f"[env3] far-skip channels: +{self.far_tr.shape[1]}", flush=True)


def run(pop_size=64, gens=12, max_rounds=600, seed=17, max_new_spaces=8,
        wrong_weight=3.0, mix_size=60000):
    import torch
    m = Env3()
    dev = m.dev
    t0 = time.time()
    log_lines = []

    def log(msg, v=True):
        log_lines.append(msg)
        if v:
            print(msg, flush=True)

    # headroom: of windows the CONT2 MODEL still gets wrong, how many can
    # the far tables answer
    wrong_te = m.preds != m.yte
    W = m.W
    inC = np.zeros(m.Nte, bool)
    inD = np.zeros(m.Nte, bool)
    for i in np.where(wrong_te)[0]:
        t = m.targets[m.yte[i]]
        for arr, table, sa, sb in ((inC, m.skipC, W - 4, W - 1),
                                   (inD, m.skipD, W - 5, W - 1)):
            ja, jb = int(m.ctx_te[i, sa]), int(m.ctx_te[i, sb])
            wa = m.vocab[ja] if ja >= 0 else None
            wb = m.vocab[jb] if jb >= 0 else None
            dist = table.get((wa, wb))
            if dist:
                top = sorted(dist, key=dist.get, reverse=True)[:5]
                arr[i] = t in top
    nw = int(wrong_te.sum())
    head = {"wrong_n": nw,
            "skipC_top5": round(float(inC[wrong_te].mean()), 4),
            "skipD_top5": round(float(inD[wrong_te].mean()), 4),
            "either_top5": round(float((inC | inD)[wrong_te].mean()), 4)}
    log(f"[headroom] cont2 still-wrong n={nw}: skipC {head['skipC_top5']} "
        f"skipD {head['skipD_top5']} either {head['either_top5']}")

    # mix weighted toward TRAIN windows the cont2 model gets wrong
    hm_, hs_, = None, None
    # cheap proxy for train wrongness: table-top5 miss OR skip answerable -
    # use actual train predictions instead: fit-head preds on train rows
    lev_tr, tab5_tr = m.table_info(m.ctx_tr, m.ytr)
    wgt = 1.0 + wrong_weight * (~tab5_tr)
    rng = np.random.default_rng(seed)
    mix = rng.choice(m.Ntr, size=mix_size, p=wgt / wgt.sum())
    mix_t = torch.tensor(mix, device=dev, dtype=torch.long)

    bank_full_tr = torch.cat([m.bank2_tr, m.far_tr], 1)
    bank_full_te = torch.cat([m.bank2_te, m.far_te], 1)
    bank_mix = bank_full_tr[mix_t]
    ss_mix = [s[mix_t] for s in m.ss_tr]
    probs_mix = {k: v[mix_t] for k, v in m.probs_tr.items()}
    old_cols_full = torch.cat([torch.cat(m.cols_tr, 1)]
                              + m.cont2_cols_tr, 1)
    old_cols_mix = old_cols_full[mix_t]
    torch.cuda.empty_cache()
    n_fit = int(mix_size * 0.8)
    ymix = m.ytr[mix]
    yv = torch.tensor(ymix[n_fit:], device=dev)
    Yf = -torch.ones((n_fit, m.V), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ymix[:n_fit], device=dev)] = 1.0
    SRCS = list(m.probs_tr.keys())

    new_spaces = []
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
            return m._feat2(g, b, ss_mix, probs_mix)

        base_prev = torch.cat([old_cols_mix] + new_mix_cols, 1)
        log(f"  [cont3 space {si}] opening - warm base {base_prev.shape[1]} "
            f"cols, bank {C} channels")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_mix, log, True)
        if not frozen:
            log(f"  [cont3 space {si}] produced nothing - stop")
            break
        new_spaces.append(frozen)
        f_mix = torch.stack(fcols, 1)
        zmu, zsd = f_mix.mean(0), f_mix.std(0) + 1e-6
        f_mix = ((f_mix - zmu) / zsd).clamp(-8, 8)
        f_tr = torch.stack([m._feat2(g, bank_full_tr, m.ss_tr, m.probs_tr)
                            for g in frozen], 1)
        f_te = torch.stack([m._feat2(g, bank_full_te, m.ss_te, m.probs_te)
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
                srcs[g.get("src")] = srcs.get(g.get("src"), 0) + 1
        log(f"  [cont3 space {si}] FULL: {len(frozen)} genomes ({n_att} "
            f"attend {srcs}), mix-val {val_now:.4f}"
            + (f" (+{gain:.4f})" if gain is not None else "")
            + f" ({round(time.time()-t0)}s)")
        if gain is not None and gain < rk.MIN_SPACE_GAIN:
            break
        val_prev = val_now
        if os.path.exists(_STOP):
            break

    extra_parts_tr = m.cont2_cols_tr + [m.new_emb_tr, m.far_emb_tr] + new_tr_cols
    extra_parts_te = m.cont2_cols_te + [m.new_emb_te, m.far_emb_te] + new_te_cols
    m._fit_head(torch.cat(extra_parts_tr, 1), torch.cat(extra_parts_te, 1))
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
    all_spaces = m.ckpt["spaces"] + m.cont2_spaces + new_spaces
    n_new = sum(len(sp) for sp in new_spaces)
    gp = sum(count_params(g) for sp in all_spaces for g in sp)
    out = {"phase": "continue-train 3: far skip-grams (w-4,w-1)/(w-5,w-1)",
           "test_acc": round(m.base_top1, 4),
           "test_top5": round(m.base_top5, 4),
           "vocab": m.V, "context_words": m.W,
           "n_genomes": sum(len(sp) for sp in all_spaces),
           "n_new_genomes": n_new,
           "n_spaces": len(all_spaces),
           "spaces": [{"space": i, "n_frozen": len(sp), "val_after": None,
                       "val_gain": None} for i, sp in enumerate(new_spaces)],
           "genome_params": gp, "head_params": m.head_params,
           "total_params": gp + m.head_params,
           "headroom": head, "breakdown": br, "examples": ex,
           "task": "second crank turn: far-skip tables added, continue-train "
                   "on the frozen Module-14 base",
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "lm_radial_word_cont3.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, "lm_model_word_cont3.json"), "w") as f:
        json.dump({"context_words": m.W, "vocab": m.V, "spaces": all_spaces,
                   "label": "next-word", "mode": "lean-cont3"}, f)
    rk._record_run(
        {"env": "lm-word-continue3", "vocab": m.V, "context_words": m.W,
         "pop_size": pop_size, "seed": seed},
        [{"round": i, "added": len(sp), "val_acc": None, "n": len(sp)}
         for i, sp in enumerate(new_spaces)],
        {"test_acc": round(m.base_top1, 4),
         "test_top5": round(m.base_top5, 4), "n_new_genomes": n_new,
         "headroom": head, "breakdown": br},
        log_lines, ["lm", "word", "continue", "skipgram", "radial"])
    print(f"[continue3] DONE: +{n_new} genomes, TEST top1 "
          f"{m.base_top1:.4f} top5 {m.base_top5:.4f} "
          f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    rw.EMBED_NPZ = os.path.join(RD, "embed_rs.npz")
    run()
