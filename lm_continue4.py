"""lm_continue4.py - Module 16: third crank turn.

New environment keys, sharpest first: the true 4-GRAM table (keyed on the
last THREE words - never added before; the deepest exact-context table)
plus skipE (w-6,w-1) and skipF (w-5,w-2). Headroom against the Module-15
model's remaining errors, then continue-train its frozen base.

    python lm_continue4.py
"""
import json
import os
import pickle
import time

import numpy as np

from radial_evo import _STOP
from lm_continue import RD, _san
from lm_continue3 import Env3
import radial_lm_word as rw
import radial_stack as rk

_HERE = os.path.dirname(os.path.abspath(__file__))


def _build_t4_tables():
    pkl = os.path.join(RD, "lm_skip_tables3.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    from radial_lm import _clean
    with open(os.path.join(_HERE, "corpora", "combined",
                           "combined_corpus.txt"), "r",
              encoding="utf-8", errors="ignore") as f:
        f.seek(30_000_000)
        toks = _clean(f.read(16_000_000)).split()
    quad, skipE, skipF = {}, {}, {}
    for i in range(6, len(toks)):
        t = toks[i]
        kQ = (toks[i - 3], toks[i - 2], toks[i - 1])
        quad.setdefault(kQ, {})
        quad[kQ][t] = quad[kQ].get(t, 0) + 1
        kE = (toks[i - 6], toks[i - 1])
        skipE.setdefault(kE, {})
        skipE[kE][t] = skipE[kE].get(t, 0) + 1
        kF = (toks[i - 5], toks[i - 2])
        skipF.setdefault(kF, {})
        skipF[kF][t] = skipF[kF].get(t, 0) + 1
    with open(pkl, "wb") as f:
        pickle.dump((quad, skipE, skipF), f)
    return quad, skipE, skipF


class Env4(Env3):
    def __init__(self):
        super().__init__()
        torch, dev = self.torch, self.dev
        t0 = time.time()
        with open(os.path.join(RD, "lm_model_word_cont3.json")) as f:
            cont3 = json.load(f)
        n_prev = (len(self.ckpt["spaces"]) + len(self.cont2_spaces))
        self.cont3_spaces = cont3["spaces"][n_prev:]

        bank_tr = torch.cat([self.bank2_tr, self.far_tr], 1)
        bank_te = torch.cat([self.bank2_te, self.far_te], 1)
        self.cont3_cols_tr, self.cont3_cols_te = [], []
        self.cont3_stats = []
        for sp in self.cont3_spaces:
            f_tr = torch.stack([self._feat2(g, bank_tr, self.ss_tr,
                                            self.probs_tr) for g in sp], 1)
            f_te = torch.stack([self._feat2(g, bank_te, self.ss_te,
                                            self.probs_te) for g in sp], 1)
            zmu, zsd = f_tr.mean(0), f_tr.std(0) + 1e-6
            self.cont3_stats.append((zmu, zsd))
            f_tr = ((f_tr - zmu) / zsd).clamp(-8, 8)
            f_te = ((f_te - zmu) / zsd).clamp(-8, 8)
            self.cont3_cols_tr.append(f_tr)
            self.cont3_cols_te.append(f_te)
            bank_tr = torch.cat([bank_tr, f_tr], 1)
            bank_te = torch.cat([bank_te, f_te], 1)
        self.bank3_tr, self.bank3_te = bank_tr, bank_te

        extra_tr = torch.cat(self.cont2_cols_tr + [self.new_emb_tr,
                             self.far_emb_tr] + self.cont3_cols_tr, 1)
        extra_te = torch.cat(self.cont2_cols_te + [self.new_emb_te,
                             self.far_emb_te] + self.cont3_cols_te, 1)
        self._fit_head(extra_tr, extra_te)
        print(f"[env4] cont3 rebuilt: test top1 {self.base_top1:.4f} "
              f"(Module 15 reported 0.4832) ({round(time.time()-t0)}s)",
              flush=True)

        self.quad, self.skipE, self.skipF = _build_t4_tables()
        W, V, D = self.W, self.V, self.D

        def prob_and_emb(ctx, table, slots):
            N = len(ctx)
            P = np.zeros((N, V), np.float32)
            E = np.zeros((N, D), np.float32)
            for i in range(N):
                key = []
                for s in slots:
                    j = int(ctx[i, s])
                    key.append(self.vocab[j] if j >= 0 else None)
                dist = table.get(tuple(key)) if len(key) > 1 else None
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
                s2 = P[i].sum()
                if s2:
                    P[i] /= s2
                if tot:
                    E[i] /= tot
            return (torch.tensor(P, device=dev), torch.tensor(E, device=dev))

        self.q_p_tr, qE_tr = prob_and_emb(self.ctx_tr, self.quad,
                                          (W - 3, W - 2, W - 1))
        self.q_p_te, qE_te = prob_and_emb(self.ctx_te, self.quad,
                                          (W - 3, W - 2, W - 1))
        self.e_p_tr, eE_tr = prob_and_emb(self.ctx_tr, self.skipE,
                                          (W - 6, W - 1))
        self.e_p_te, eE_te = prob_and_emb(self.ctx_te, self.skipE,
                                          (W - 6, W - 1))
        self.f_p_tr, fE_tr = prob_and_emb(self.ctx_tr, self.skipF,
                                          (W - 5, W - 2))
        self.f_p_te, fE_te = prob_and_emb(self.ctx_te, self.skipF,
                                          (W - 5, W - 2))
        t4_tr = torch.cat([self.q_p_tr, self.e_p_tr, self.f_p_tr,
                           qE_tr, eE_tr, fE_tr], 1)
        t4_te = torch.cat([self.q_p_te, self.e_p_te, self.f_p_te,
                           qE_te, eE_te, fE_te], 1)
        tmu, tsd = t4_tr.mean(0), t4_tr.std(0) + 1e-6
        self.tmu, self.tsd = tmu, tsd
        self.t4_tr = ((t4_tr - tmu) / tsd).clamp(-8, 8)
        self.t4_te = ((t4_te - tmu) / tsd).clamp(-8, 8)
        self.t4_emb_tr = self.t4_tr[:, 3 * V:]
        self.t4_emb_te = self.t4_te[:, 3 * V:]
        self.probs_tr.update({"quad": self.q_p_tr, "skipE": self.e_p_tr,
                              "skipF": self.f_p_tr})
        self.probs_te.update({"quad": self.q_p_te, "skipE": self.e_p_te,
                              "skipF": self.f_p_te})
        # free the redundant intermediate banks: bank3 CONTAINS bank2 which
        # contains the lean bank and the skipA/B + far channels - keeping
        # them all resident is what OOMed the first attempt (89GB)
        self.new_emb_tr = self.new_emb_tr.clone()
        self.new_emb_te = self.new_emb_te.clone()
        self.far_emb_tr = self.far_emb_tr.clone()
        self.far_emb_te = self.far_emb_te.clone()
        self.t4_emb_tr = self.t4_emb_tr.clone()
        self.t4_emb_te = self.t4_emb_te.clone()
        del (self.bank_tr, self.bank_te, self.bank2_tr, self.bank2_te,
             self.new_tr, self.new_te, self.far_tr, self.far_te)
        torch.cuda.empty_cache()
        print(f"[env4] quad + far channels: +{self.t4_tr.shape[1]} "
              "(intermediate banks freed)", flush=True)


def run(pop_size=48, gens=12, max_rounds=600, seed=19, max_new_spaces=8,
        wrong_weight=3.0, mix_size=40000):
    import torch
    m = Env4()
    dev = m.dev
    t0 = time.time()
    log_lines = []

    def log(msg, v=True):
        log_lines.append(msg)
        if v:
            print(msg, flush=True)

    wrong_te = m.preds != m.yte
    W = m.W
    inQ = np.zeros(m.Nte, bool)
    inE = np.zeros(m.Nte, bool)
    inF = np.zeros(m.Nte, bool)
    for i in np.where(wrong_te)[0]:
        t = m.targets[m.yte[i]]
        for arr, table, slots in ((inQ, m.quad, (W - 3, W - 2, W - 1)),
                                  (inE, m.skipE, (W - 6, W - 1)),
                                  (inF, m.skipF, (W - 5, W - 2))):
            key = []
            for s in slots:
                j = int(m.ctx_te[i, s])
                key.append(m.vocab[j] if j >= 0 else None)
            dist = table.get(tuple(key))
            if dist:
                top = sorted(dist, key=dist.get, reverse=True)[:5]
                arr[i] = t in top
    nw = int(wrong_te.sum())
    head = {"wrong_n": nw,
            "quad_top5": round(float(inQ[wrong_te].mean()), 4),
            "skipE_top5": round(float(inE[wrong_te].mean()), 4),
            "skipF_top5": round(float(inF[wrong_te].mean()), 4),
            "any_top5": round(float((inQ | inE | inF)[wrong_te].mean()), 4)}
    log(f"[headroom] cont3 still-wrong n={nw}: quad {head['quad_top5']} "
        f"skipE {head['skipE_top5']} skipF {head['skipF_top5']} "
        f"any {head['any_top5']}")

    lev_tr, tab5_tr = m.table_info(m.ctx_tr, m.ytr)
    wgt = 1.0 + wrong_weight * (~tab5_tr)
    rng = np.random.default_rng(seed)
    mix = rng.choice(m.Ntr, size=mix_size, p=wgt / wgt.sum())
    mix_t = torch.tensor(mix, device=dev, dtype=torch.long)

    bank_full_tr = torch.cat([m.bank3_tr, m.t4_tr], 1)
    bank_full_te = torch.cat([m.bank3_te, m.t4_te], 1)
    bank_mix = bank_full_tr[mix_t]
    ss_mix = [s[mix_t] for s in m.ss_tr]
    probs_mix = {k: v[mix_t] for k, v in m.probs_tr.items()}
    old_cols_full = torch.cat([torch.cat(m.cols_tr, 1)] + m.cont2_cols_tr
                              + m.cont3_cols_tr, 1)
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
        log(f"  [cont4 space {si}] opening - warm base {base_prev.shape[1]} "
            f"cols, bank {C} channels")
        frozen, fcols = rk._evolve_space(torch, rng, pop_size, gens,
                                         max_rounds, n_fit, Yf, yv, base_prev,
                                         new_fn, mut_fn, feat_mix, log, True)
        if not frozen:
            log(f"  [cont4 space {si}] produced nothing - stop")
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
        log(f"  [cont4 space {si}] FULL: {len(frozen)} genomes, mix-val "
            f"{val_now:.4f}"
            + (f" (+{gain:.4f})" if gain is not None else "")
            + f" ({round(time.time()-t0)}s)")
        if gain is not None and gain < rk.MIN_SPACE_GAIN:
            break
        val_prev = val_now
        if os.path.exists(_STOP):
            break

    extra_tr = torch.cat(m.cont2_cols_tr + [m.new_emb_tr, m.far_emb_tr]
                         + m.cont3_cols_tr + [m.t4_emb_tr] + new_tr_cols, 1)
    extra_te = torch.cat(m.cont2_cols_te + [m.new_emb_te, m.far_emb_te]
                         + m.cont3_cols_te + [m.t4_emb_te] + new_te_cols, 1)
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
    all_spaces = (m.ckpt["spaces"] + m.cont2_spaces + m.cont3_spaces
                  + new_spaces)
    n_new = sum(len(sp) for sp in new_spaces)
    gp = sum(count_params(g) for sp in all_spaces for g in sp)
    out = {"phase": "continue-train 4: quad + (w-6,w-1) + (w-5,w-2) tables",
           "test_acc": round(m.base_top1, 4),
           "test_top5": round(m.base_top5, 4),
           "vocab": m.V, "context_words": m.W,
           "n_genomes": sum(len(sp) for sp in all_spaces),
           "n_new_genomes": n_new, "n_spaces": len(all_spaces),
           "spaces": [{"space": i, "n_frozen": len(sp), "val_after": None,
                       "val_gain": None} for i, sp in enumerate(new_spaces)],
           "genome_params": gp, "head_params": m.head_params,
           "total_params": gp + m.head_params,
           "headroom": head, "breakdown": br, "examples": ex,
           "task": "third crank: true 4-gram table + two more far skips, "
                   "continue-train on the frozen Module-15 base",
           "seconds": round(time.time() - t0)}
    with open(os.path.join(RD, "lm_radial_word_cont4.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RD, "lm_model_word_cont4.json"), "w") as f:
        json.dump({"context_words": m.W, "vocab": m.V, "spaces": all_spaces,
                   "label": "next-word", "mode": "lean-cont4"}, f)
    rk._record_run(
        {"env": "lm-word-continue4", "vocab": m.V, "context_words": m.W,
         "pop_size": pop_size, "seed": seed},
        [{"round": i, "added": len(sp), "val_acc": None, "n": len(sp)}
         for i, sp in enumerate(new_spaces)],
        {"test_acc": round(m.base_top1, 4),
         "test_top5": round(m.base_top5, 4), "n_new_genomes": n_new,
         "headroom": head, "breakdown": br},
        log_lines, ["lm", "word", "continue", "skipgram", "radial"])
    print(f"[continue4] DONE: +{n_new} genomes, TEST top1 "
          f"{m.base_top1:.4f} top5 {m.base_top5:.4f} "
          f"({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    rw.EMBED_NPZ = os.path.join(RD, "embed_rs.npz")
    run()
