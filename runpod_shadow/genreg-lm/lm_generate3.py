"""lm_generate3.py - generate text from the Module-16 (cont4) model.

Rebuilds the full environment chain (Env4 + cont4 spaces), verifies the
test number, calibrates a decode scale on val, then generates word by
word: each step builds the single-row bank chain exactly as training did
(B0 -> lean spaces -> skipA/B -> cont2 -> far -> cont3 -> quad/E/F ->
cont4) and samples from the head's top-5. Samples are appended into
lm_radial_word_cont4.json so the /lm page shows them on Module 16.
"""
import json
import os
import time

import numpy as np

from lm_continue import RD, _san
from lm_continue4 import Env4
import radial_lm_word as rw
import radial_stack as rk


class Gen(Env4):
    def __init__(self):
        super().__init__()
        torch = self.torch
        with open(os.path.join(RD, "lm_model_word_cont4.json")) as f:
            cont4 = json.load(f)
        n_prev = (len(self.ckpt["spaces"]) + len(self.cont2_spaces)
                  + len(self.cont3_spaces))
        self.cont4_spaces = cont4["spaces"][n_prev:]
        bank_tr = torch.cat([self.bank3_tr, self.t4_tr], 1)
        bank_te = torch.cat([self.bank3_te, self.t4_te], 1)
        self.cont4_cols_tr, self.cont4_cols_te, self.cont4_stats = [], [], []
        for sp in self.cont4_spaces:
            f_tr = torch.stack([self._feat2(g, bank_tr, self.ss_tr,
                                            self.probs_tr) for g in sp], 1)
            f_te = torch.stack([self._feat2(g, bank_te, self.ss_te,
                                            self.probs_te) for g in sp], 1)
            zmu, zsd = f_tr.mean(0), f_tr.std(0) + 1e-6
            self.cont4_stats.append((zmu, zsd))
            f_tr = ((f_tr - zmu) / zsd).clamp(-8, 8)
            f_te = ((f_te - zmu) / zsd).clamp(-8, 8)
            self.cont4_cols_tr.append(f_tr)
            self.cont4_cols_te.append(f_te)
            bank_tr = torch.cat([bank_tr, f_tr], 1)
            bank_te = torch.cat([bank_te, f_te], 1)
        extra_tr = torch.cat(self.cont2_cols_tr + [self.new_emb_tr,
                             self.far_emb_tr] + self.cont3_cols_tr
                             + [self.t4_emb_tr] + self.cont4_cols_tr, 1)
        extra_te = torch.cat(self.cont2_cols_te + [self.new_emb_te,
                             self.far_emb_te] + self.cont3_cols_te
                             + [self.t4_emb_te] + self.cont4_cols_te, 1)
        self._fit_head(extra_tr, extra_te)
        print(f"[gen] cont4 rebuilt: test top1 {self.base_top1:.4f} "
              "(Module 16 reported 0.5663)", flush=True)
        # decode scale on the val split
        Ftr, _ = self._head_cols(extra_tr, extra_te)
        n_fit = int(self.Ntr * 0.8)
        yv = torch.tensor(self.ytr[n_fit:], device=self.dev)
        hm, hs, Wm = self._hm, self._hs, self._Wm
        s_val = torch.hstack([(Ftr[n_fit:] - hm) / hs,
                              torch.ones(self.Ntr - n_fit, 1,
                                         device=self.dev)]) @ Wm
        self.s_cal, best = 1.0, 1e9
        for sc in (1.0, 2.0, 4.0, 7.0, 12.0, 20.0, 35.0):
            nll = float(-torch.log_softmax(s_val * sc, 1)
                        [torch.arange(self.Ntr - n_fit), yv].mean())
            if nll < best:
                self.s_cal, best = sc, nll
        print(f"[gen] decode scale x{self.s_cal}", flush=True)
        featV = np.zeros((self.V, self.D), np.float32)
        for k, wd in enumerate(self.targets):
            j = self.w2i.get(wd)
            if j is not None:
                featV[k] = self.feat[j]
        fV = torch.tensor(featV, device=self.dev)
        self.featV = fV / (fV.norm(dim=1, keepdim=True) + 1e-6)
        # gate stats in feature_vec are BATCH statistics (block-diff: env
        # blocks exact, genome spaces diverge on 1-row banks). Steps are
        # evaluated appended to a fixed reference batch.
        R = 2048
        B0w = self.B0_te.shape[1]
        leanw = sum(c.shape[1] for c in self.cols_te)
        c2w = sum(c.shape[1] for c in self.cont2_cols_te)
        bk = self.bank3_te
        self.refB0 = self.B0_te[:R]
        self.refAB = bk[:R, B0w + leanw:B0w + leanw + 4256]
        self.refFar = bk[:R, B0w + leanw + 4256 + c2w:
                         B0w + leanw + 4256 + c2w + 4256]
        self.refT4 = self.t4_te[:R]
        self.ss_ref = [x[:R] for x in self.ss_te]
        self.probs_ref = {k: v[:R] for k, v in self.probs_te.items()}

    # -- single-row builders (mirror the batch ones exactly) ------------
    def _dist_pe(self, table, key):
        V, D = self.V, self.D
        P = np.zeros(V, np.float32)
        E = np.zeros(D, np.float32)
        dist = table.get(key)
        if dist:
            tot = 0
            for w, c in dist.items():
                k = self.tgt_i.get(w)
                if k is not None:
                    P[k] = c
                j = self.w2i.get(w)
                if j is not None:
                    E += c * self.feat[j]
                    tot += c
            s = P.sum()
            if s:
                P /= s
            if tot:
                E /= tot
        return P, E

    def _wd(self, win, s):
        j = win[s]
        return self.vocab[j] if j >= 0 else None

    def _step_logits(self, win, debug=False):
        torch, dev = self.torch, self.dev
        W, V, D = self.W, self.V, self.D
        ctx = np.array([win], np.int32)
        idx = torch.tensor(np.maximum(ctx.astype(np.int64), 0), device=dev)
        mask = torch.tensor((ctx >= 0).astype(np.float32), device=dev)
        emb = torch.cat([(self.feat_t[idx[:, f]] * mask[:, f:f + 1]
                          - self.mu) / self.sd for f in range(W)], 1)
        ident = torch.zeros((1, 2 * V), device=dev)
        for pos, slot in ((0, W - 2), (V, W - 1)):
            k = self.tv.get(int(ctx[0, slot]), -1)
            if k >= 0:
                ident[0, pos + k] = 1.0
        w1, w2 = self._wd(win, W - 2), self._wd(win, W - 1)
        cont = np.zeros(2 * D + V, np.float32)
        if (w1, w2) in self.tri_c:
            P, E = self._dist_pe(self.tri_c, (w1, w2))
        elif w2 in self.bi_c:
            P, E = self._dist_pe(self.bi_c, w2)
        else:
            P, E = self._dist_pe(self.uni_c, None), None
            P, E = P[0] if isinstance(P, tuple) else P, np.zeros(D, np.float32)
        cont[:D] = E
        cont[2 * D:] = P
        Pb, Eb = self._dist_pe(self.bi_c, w2) if w2 in self.bi_c else (
            np.zeros(V, np.float32), np.zeros(D, np.float32))
        cont[D:2 * D] = Eb
        cont_t = torch.tensor(cont[None], device=dev)
        cont_z = ((cont_t - self.cmu) / self.csd).clamp(-8, 8)
        B0 = torch.cat([emb, ident, cont_z], 1)

        def pe_row(table, key):
            P, E = self._dist_pe(table, key)
            return P, E

        pA, eA = pe_row(self.skipA, (self._wd(win, W - 3), w2))
        pB, eB = pe_row(self.skipB, (self._wd(win, W - 3), w1))
        skipAB = torch.tensor(np.concatenate([pA, pB, eA, eB])[None],
                              device=dev)
        skipAB = ((skipAB - self.nmu) / self.nsd).clamp(-8, 8)
        pC, eC = pe_row(self.skipC, (self._wd(win, W - 4), w2))
        pD, eD = pe_row(self.skipD, (self._wd(win, W - 5), w2))
        far = torch.tensor(np.concatenate([pC, pD, eC, eD])[None], device=dev)
        far = ((far - self.fmu) / self.fsd).clamp(-8, 8)
        pQ, eQ = pe_row(self.quad, (self._wd(win, W - 3), w1, w2))
        pE_, eE_ = pe_row(self.skipE, (self._wd(win, W - 6), w2))
        pF, eF = pe_row(self.skipF, (self._wd(win, W - 5), w1))
        t4 = torch.tensor(np.concatenate([pQ, pE_, pF, eQ, eE_, eF])[None],
                          device=dev)
        t4 = ((t4 - self.tmu) / self.tsd).clamp(-8, 8)

        e = self.feat_t[idx.reshape(-1)] * mask.reshape(-1, 1)
        en = e / (e.norm(dim=1, keepdim=True) + 1e-6)
        ss = [(en[f:f + 1] @ self.featV.T).half() for f in range(W)]
        probs = {"cont": cont_t[:, 2 * D:], "skipA": torch.tensor(pA[None], device=dev),
                 "skipB": torch.tensor(pB[None], device=dev),
                 "skipC": torch.tensor(pC[None], device=dev),
                 "skipD": torch.tensor(pD[None], device=dev),
                 "quad": torch.tensor(pQ[None], device=dev),
                 "skipE": torch.tensor(pE_[None], device=dev),
                 "skipF": torch.tensor(pF[None], device=dev)}

        def space_cols(spaces, stats, bank):
            cols = []
            for sp, (zmu, zsd) in zip(spaces, stats):
                f = torch.stack([self._feat2(g, bank, ss, probs)
                                 for g in sp], 1)
                f = ((f - zmu) / zsd).clamp(-8, 8)
                cols.append(f)
                bank = torch.cat([bank, f], 1)
            return cols, bank

        B0 = torch.cat([self.refB0, B0], 0)
        skipAB = torch.cat([self.refAB, skipAB], 0)
        far = torch.cat([self.refFar, far], 0)
        t4 = torch.cat([self.refT4, t4], 0)
        ss = [torch.cat([r, x], 0) for r, x in zip(self.ss_ref, ss)]
        probs = {k: torch.cat([self.probs_ref[k], probs[k]], 0)
                 for k in probs}
        lean_cols, bank = space_cols(self.ckpt["spaces"], self.stats, B0)
        bank = torch.cat([bank, skipAB], 1)
        c2_cols, bank = space_cols(self.cont2_spaces, self.cont2_stats, bank)
        bank = torch.cat([bank, far], 1)
        c3_cols, bank = space_cols(self.cont3_spaces, self.cont3_stats, bank)
        bank = torch.cat([bank, t4], 1)
        c4_cols, bank = space_cols(self.cont4_spaces, self.cont4_stats, bank)
        if debug:
            return {"B0": B0, "skipAB": skipAB, "far": far, "t4": t4,
                    "lean": lean_cols, "c2": c2_cols, "c3": c3_cols,
                    "c4": c4_cols}
        F1 = torch.cat([x[-1:] for x in lean_cols + c2_cols]
                       + [skipAB[-1:, 2 * V:], far[-1:, 2 * V:]]
                       + [x[-1:] for x in c3_cols] + [t4[-1:, 3 * V:]]
                       + [x[-1:] for x in c4_cols]
                       + [B0[-1:, -(2 * D + V):-V]], 1)
        hm, hs, Wm = self._hm, self._hs, self._Wm
        return (torch.hstack([(F1 - hm) / hs,
                              torch.ones(1, 1, device=dev)]) @ Wm)[0]

    def generate(self, prompt, n_words=28, temp=1.0, seed=0):
        rng = np.random.default_rng(seed)
        words = prompt.lower().split()
        win = [self.w2i.get(w, -1) for w in words][-self.W:]
        while len(win) < self.W:
            win.insert(0, -1)
        out = []
        for _ in range(n_words):
            lg = self._step_logits(win).detach().cpu().numpy().astype(float)
            lg = lg * self.s_cal / max(temp, 1e-3)
            for wd in out[-8:]:
                k = self.tgt_i.get(wd)
                if k is not None:
                    lg[k] -= 2.0
            top = np.argsort(lg)[-5:]
            p = np.exp(lg[top] - lg[top].max())
            p /= p.sum()
            wd = self.targets[int(rng.choice(top, p=p))]
            out.append(wd)
            win = win[1:] + [self.w2i.get(wd, -1)]
        return prompt + " | " + " ".join(out)


def main():
    g = Gen()
    samples = []
    for i, pr in enumerate(["the first part of the", "he was born in the",
                            "one of the most important",
                            "i think we should"]):
        txt = g.generate(pr, seed=i)
        samples.append({"prompt": pr, "temp": 1.0, "text": txt})
        print("SAMPLE:", txt, flush=True)
    p = os.path.join(RD, "lm_radial_word_cont4.json")
    with open(p) as f:
        out = json.load(f)
    out["samples"] = samples
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("[gen] samples appended to lm_radial_word_cont4.json", flush=True)


if __name__ == "__main__":
    rw.EMBED_NPZ = os.path.join(RD, "embed_rs.npz")
    main()
