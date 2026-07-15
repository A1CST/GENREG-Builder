"""CIFAR "internal language" — a SINGLE end-to-end genome, everything evolved.

The departure from cifar_pipe's bank->features->classifier stack: here ONE
genome does the whole job. It carries its own small set of 5x5x3 convolution
kernels, a per-filter evolved activation from the 8-function catalog, multi-
shape mean pooling, and a linear readout to a single scalar. sigmoid(score) =
P(class A). Nothing is pre-built: no PCA, no Fisher pre-selection, no logistic
regression. The pooled filter responses are the genome's private "internal
language" for telling the two categories apart, and the readout reads that
language. Evolved jointly, gradient-free, by the shared ga_step engine
(tournament + elitism + starvation homeostasis + self-adaptive sigma).

Soft fitness only (mean log-prob / BCE), balanced two-class minibatches, ZCA-
whitened patches, champion chosen on a held-out split, reported on the CIFAR
test set against the 50% majority-class baseline.
"""
import os
import pickle

import numpy as np

from genreg_train import cifar_pipe as cp
from genreg_train import mnist_pipe as mp
from genreg_train import evo_gpu

RESP = cp.RESP                                      # 28
POOLS = mp.POOLS                                    # ((3,3),(4,2),(2,4)) -> 25
PD = sum(r * c for r, c in POOLS)
KD = 75                                             # 5x5x3
LABELS = cp.LABELS
OUT = os.path.join(cp.ROOT, "demo", "cifar_internal.pkl")


# --------------------------------------------------------------------------
# Whole-population fitness for the single end-to-end genome. Each genome's M
# filters are laid out as M "detectors" so the conv+activation+pool reuses the
# validated detbank kernels; the reshape back to (P, M*PD) then feeds the
# per-genome readout. positives / negatives patch pools uploaded once.
# --------------------------------------------------------------------------
class SingleGenomeBinaryGPU:
    def __init__(self, Pp, Pn, M, pools=POOLS):
        import torch
        self.torch = torch
        self.M, self.pools = M, pools
        self.PD = sum(r * c for r, c in pools)
        self.Np, self.RR, self.kd = Pp.shape
        self.Nn = Pn.shape[0]
        self.Pp = evo_gpu.to_dev(Pp.astype(np.float32))     # (Np, RR, KD)
        self.Pn = evo_gpu.to_dev(Pn.astype(np.float32))

    def _feats(self, batch, K, kb, act):
        """batch (B,RR,KD) on device; K (P,M,KD), kb (P,M), act (P,M) ints ->
        (P, B, M*PD) pooled internal representation."""
        torch = self.torch
        P, M = K.shape[0], self.M
        D = P * M
        B = batch.shape[0]
        Kf = evo_gpu.to_dev(K.reshape(D, self.kd))
        bf = evo_gpu.to_dev(kb.reshape(D).astype(np.float32))
        resp = (batch.reshape(B * self.RR, self.kd) @ Kf.T + bf) \
            .reshape(B, RESP, RESP, D)
        pooled = torch.empty((D, B, self.PD), device=evo_gpu.DEV)
        actf = act.reshape(D)
        for a in range(8):
            ids = np.where(actf == a)[0]
            if len(ids) == 0:
                continue
            idt = evo_gpu.to_dev(ids.astype(np.int64))
            blk = evo_gpu._acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * B, RESP, RESP)
            pooled[idt] = evo_gpu._pool_t(blk, RESP, self.pools) \
                .reshape(len(ids), B, self.PD)
        return pooled.reshape(P, M, B, self.PD).permute(0, 2, 1, 3) \
            .reshape(P, B, M * self.PD)

    def _apply_gate(self, feat, gate):
        """Multiply each filter's pooled block by g=sigmoid(gate). gate (P,M);
        an off filter (g~0) contributes ~nothing to the score. None = all-on."""
        if gate is None:
            return feat
        torch = self.torch
        g = torch.sigmoid(evo_gpu.to_dev(gate.astype(np.float32)))   # (P,M)
        P, B, _ = feat.shape
        return (feat.reshape(P, B, self.M, self.PD) * g[:, None, :, None]) \
            .reshape(P, B, self.M * self.PD)

    def _scores(self, feat, w, rb):
        wg = evo_gpu.to_dev(w)                              # (P, M*PD)
        rbg = evo_gpu.to_dev(rb.astype(np.float32))         # (P,)
        return (feat * wg[:, None, :]).sum(-1) + rbg[:, None]

    def __call__(self, K, kb, act, w, rb, ip, inn, gate=None):
        torch = self.torch
        with torch.no_grad():
            fp = self._apply_gate(self._feats(self.Pp[evo_gpu.to_dev(ip.astype(np.int64))],
                                              K, kb, act), gate)
            fn = self._apply_gate(self._feats(self.Pn[evo_gpu.to_dev(inn.astype(np.int64))],
                                              K, kb, act), gate)
            zp = self._scores(fp, w, rb).clamp(-30, 30)
            zn = self._scores(fn, w, rb).clamp(-30, 30)
            lp = -torch.log1p(torch.exp(-zp)).mean(1)
            ln = -torch.log1p(torch.exp(zn)).mean(1)
            acc = ((zp > 0).float().mean(1) + (zn < 0).float().mean(1)) / 2
            return (lp + ln).cpu().numpy(), acc.cpu().numpy()

    def eval_full(self, K, kb, act, w, rb, chunk=250, gate=None):
        """Champion (single genome, P=1) accuracy + prob spread over the whole
        uploaded pos/neg pools, chunked over images."""
        torch = self.torch
        with torch.no_grad():
            def probs(Pool, n):
                out = []
                for lo in range(0, n, chunk):
                    idx = np.arange(lo, min(lo + chunk, n))
                    z = self._scores(self._apply_gate(
                        self._feats(Pool[evo_gpu.to_dev(idx.astype(np.int64))],
                                    K, kb, act), gate), w, rb).clamp(-30, 30)
                    out.append(torch.sigmoid(z)[0].cpu().numpy())
                return np.concatenate(out)
            pp = probs(self.Pp, self.Np)
            pn = probs(self.Pn, self.Nn)
        acc = ((pp > 0.5).mean() + (pn < 0.5).mean()) / 2
        return float(acc), pp, pn


# --------------------------------------------------------------------------
# Data: fixed per-class patch pools (im2col'd once, ZCA-whitened)
# --------------------------------------------------------------------------
def _patch_pool(X, whiten=True):
    P = cp._im2col5c(X).reshape(-1, KD)
    if whiten:
        P = cp._whiten_patches(P)
    return P.reshape(len(X), RESP * RESP, KD).astype(np.float32)


def _sample(split_X, split_y, cls, n, seed):
    rng = np.random.default_rng(seed)
    idx = np.where(split_y == cls)[0]
    if n < len(idx):
        idx = rng.choice(idx, n, replace=False)
    return split_X[idx]


# --------------------------------------------------------------------------
# Evolution of the single genome
# --------------------------------------------------------------------------
def evolve_single(pos=1, neg=2, M=8, pop=64, gens=2500, minibatch=128,
                  n_train=2000, n_val=500, whiten=True, seed=7,
                  log_every=100, log=print):
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    a_name, b_name = LABELS[pos], LABELS[neg]
    log(f"[internal] single-genome {a_name}(+) vs {b_name}(-)  "
        f"M={M} pop={pop} gens={gens} whiten={whiten}", flush=True)

    if whiten:
        cp.build_zca()                                       # unsupervised, cached
    Pp = _patch_pool(_sample(Xtr, ytr, pos, n_train, seed), whiten)
    Pn = _patch_pool(_sample(Xtr, ytr, neg, n_train, seed + 1), whiten)
    Vp = _patch_pool(_sample(Xva, yva, pos, n_val, seed + 2), whiten)
    Vn = _patch_pool(_sample(Xva, yva, neg, n_val, seed + 3), whiten)
    log(f"[internal] train pool {len(Pp)}/{len(Pn)}  val pool {len(Vp)}/{len(Vn)}",
        flush=True)

    if not evo_gpu.HAS_GPU:
        raise RuntimeError("this experiment expects the GPU backend")
    trainf = SingleGenomeBinaryGPU(Pp, Pn, M)
    valf = SingleGenomeBinaryGPU(Vp, Vn, M)

    rng = np.random.default_rng(seed)
    MPD = M * PD
    params = {
        "K": (rng.standard_normal((pop, M, KD)) * 0.15).astype(np.float32),
        "kb": np.zeros((pop, M), np.float32),
        "act": rng.integers(0, 8, (pop, M)).astype(np.float32),
        "w": (rng.standard_normal((pop, MPD)) / np.sqrt(MPD)).astype(np.float32),
        "rb": np.zeros(pop, np.float32),
        "sigma": np.full(pop, 0.08, np.float32),
    }

    def act_ids(a):
        return np.round(a).astype(np.int64) % 8

    best = {"fit": -1e9, "acc": 0.0, "gen": 0, "champ": None}
    for gen in range(1, gens + 1):
        ai = act_ids(params["act"])
        ip = rng.integers(0, len(Pp), minibatch)
        inn = rng.integers(0, len(Pn), minibatch)
        fit, _ = trainf(params["K"], params["kb"], ai, params["w"], params["rb"],
                        ip, inn)
        if gen >= int(0.8 * gens):                           # anneal late
            params["sigma"] = np.minimum(params["sigma"], 0.04)
        mp.ga_step(params, fit, rng, mag_scale=True)
        params["act"] = np.round(params["act"]) % 8

        if gen % log_every == 0 or gen == 1 or gen == gens:
            ai = act_ids(params["act"])
            vfit, vacc = valf(params["K"], params["kb"], ai, params["w"],
                              params["rb"], np.arange(len(Vp)), np.arange(len(Vn)))
            j = int(np.argmax(vfit))
            if float(vfit[j]) > best["fit"]:
                best.update(fit=float(vfit[j]), acc=float(vacc[j]), gen=gen,
                            champ={"K": params["K"][j].copy(),
                                   "kb": params["kb"][j].copy(),
                                   "act": ai[j].copy(),
                                   "w": params["w"][j].copy(),
                                   "rb": float(params["rb"][j])})
            log(f"  gen {gen:5d}: val_logprob={vfit[j]:.4f} val_acc={vacc[j]:.4f} "
                f"(best {best['acc']:.4f} @ gen {best['gen']})", flush=True)

    c = best["champ"]
    # single-genome (P=1) evaluation on held-out test set
    Tp = _patch_pool(Xte[yte == pos], whiten)
    Tn = _patch_pool(Xte[yte == neg], whiten)
    testf = SingleGenomeBinaryGPU(Tp, Tn, M)
    K1 = c["K"][None]; kb1 = c["kb"][None]; act1 = c["act"][None]
    w1 = c["w"][None]; rb1 = np.array([c["rb"]], np.float32)
    test_acc, pp, pn = testf.eval_full(K1, kb1, act1, w1, rb1)
    val_acc, _, _ = valf.eval_full(K1, kb1, act1, w1, rb1)

    log("", flush=True)
    log(f"[internal] RESULT {a_name}(+) vs {b_name}(-)", flush=True)
    log(f"  majority-class baseline : 0.5000", flush=True)
    log(f"  val  acc (held-out)     : {val_acc:.4f}", flush=True)
    log(f"  test acc (held-out)     : {test_acc:.4f}  "
        f"(champion from gen {best['gen']})", flush=True)
    drop = (val_acc - test_acc) / max(val_acc, 1e-9)
    log(f"  val->test rel. drop     : {drop * 100:.1f}%", flush=True)
    log(f"  activations used        : {[int(x) for x in c['act']]}", flush=True)
    # inspect real predictions: a few examples each way
    log(f"  example P({a_name}) on real {a_name} imgs: "
        f"{np.round(pp[:6], 3).tolist()}", flush=True)
    log(f"  example P({a_name}) on real {b_name} imgs: "
        f"{np.round(pn[:6], 3).tolist()}", flush=True)

    payload = {"pos": pos, "neg": neg, "M": M, "whiten": whiten,
               "champ": c, "val_acc": val_acc, "test_acc": test_acc,
               "gen": best["gen"], "labels": (a_name, b_name)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)
    log(f"[internal] saved champion -> {OUT}", flush=True)
    return payload


# --------------------------------------------------------------------------
# Masked-capacity variant: the genome EVOLVES ITS OWN HIDDEN DIM. Each of
# M_max filters carries a soft gate gene g=sigmoid(gate); fitness pays a cost
# proportional to sum(g), annealed in after a warm-up so filters can specialise
# before pruning starts (the maturation principle). A filter survives only if
# its discriminative contribution beats its cost -> effective width is an
# evolved trait under landscape pressure, not a hyperparameter.
# --------------------------------------------------------------------------
OUT_MASKED = os.path.join(cp.ROOT, "demo", "cifar_internal_masked.pkl")


def evolve_single_masked(pos=1, neg=2, M_max=24, pop=64, gens=2500,
                         minibatch=128, n_train=2000, n_val=500, whiten=True,
                         cost_lambda=0.015, warm_frac=0.30, full_frac=0.60,
                         seed=7, log_every=100, log=print):
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    a_name, b_name = LABELS[pos], LABELS[neg]
    log(f"[masked] single-genome {a_name}(+) vs {b_name}(-)  M_max={M_max} "
        f"pop={pop} gens={gens} cost_lambda={cost_lambda} whiten={whiten}",
        flush=True)

    if whiten:
        cp.build_zca()
    Pp = _patch_pool(_sample(Xtr, ytr, pos, n_train, seed), whiten)
    Pn = _patch_pool(_sample(Xtr, ytr, neg, n_train, seed + 1), whiten)
    Vp = _patch_pool(_sample(Xva, yva, pos, n_val, seed + 2), whiten)
    Vn = _patch_pool(_sample(Xva, yva, neg, n_val, seed + 3), whiten)
    log(f"[masked] train pool {len(Pp)}/{len(Pn)}  val pool {len(Vp)}/{len(Vn)}",
        flush=True)

    if not evo_gpu.HAS_GPU:
        raise RuntimeError("this experiment expects the GPU backend")
    trainf = SingleGenomeBinaryGPU(Pp, Pn, M_max)
    valf = SingleGenomeBinaryGPU(Vp, Vn, M_max)

    rng = np.random.default_rng(seed)
    MPD = M_max * PD
    params = {
        "K": (rng.standard_normal((pop, M_max, KD)) * 0.15).astype(np.float32),
        "kb": np.zeros((pop, M_max), np.float32),
        "act": rng.integers(0, 8, (pop, M_max)).astype(np.float32),
        "gate": np.full((pop, M_max), 1.0, np.float32),      # start all-on (g~0.73)
        "w": (rng.standard_normal((pop, MPD)) / np.sqrt(MPD)).astype(np.float32),
        "rb": np.zeros(pop, np.float32),
        "sigma": np.full(pop, 0.08, np.float32),
    }

    def act_ids(a):
        return np.round(a).astype(np.int64) % 8

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    warm, full = int(warm_frac * gens), int(full_frac * gens)

    def lam_at(gen):                                         # annealed cost
        if gen <= warm:
            return 0.0
        return cost_lambda * min(1.0, (gen - warm) / max(1, full - warm))

    best = {"fit": -1e9, "acc": 0.0, "gen": 0, "champ": None, "width": 0}
    for gen in range(1, gens + 1):
        ai = act_ids(params["act"])
        ip = rng.integers(0, len(Pp), minibatch)
        inn = rng.integers(0, len(Pn), minibatch)
        fit, _ = trainf(params["K"], params["kb"], ai, params["w"], params["rb"],
                        ip, inn, gate=params["gate"])
        fit = fit - lam_at(gen) * sig(params["gate"]).sum(1)   # capacity cost
        if gen >= int(0.8 * gens):
            params["sigma"] = np.minimum(params["sigma"], 0.04)
        mp.ga_step(params, fit, rng, mag_scale=True)
        params["act"] = np.round(params["act"]) % 8

        if gen % log_every == 0 or gen == 1 or gen == gens:
            ai = act_ids(params["act"])
            vfit, vacc = valf(params["K"], params["kb"], ai, params["w"],
                              params["rb"], np.arange(len(Vp)), np.arange(len(Vn)),
                              gate=params["gate"])
            # score on val ACCURACY-fitness minus current cost, pick champion
            vscore = vfit - lam_at(gen) * sig(params["gate"]).sum(1)
            j = int(np.argmax(vscore))
            width = int((sig(params["gate"][j]) > 0.5).sum())
            if float(vfit[j]) > best["fit"]:
                best.update(fit=float(vfit[j]), acc=float(vacc[j]), gen=gen,
                            width=width,
                            champ={"K": params["K"][j].copy(),
                                   "kb": params["kb"][j].copy(),
                                   "act": ai[j].copy(),
                                   "gate": params["gate"][j].copy(),
                                   "w": params["w"][j].copy(),
                                   "rb": float(params["rb"][j])})
            mean_w = float((sig(params["gate"]) > 0.5).sum(1).mean())
            log(f"  gen {gen:5d}: val_logprob={vfit[j]:.4f} val_acc={vacc[j]:.4f} "
                f"width={width} pop_mean_width={mean_w:.1f} lam={lam_at(gen):.4f} "
                f"(best acc {best['acc']:.4f} w={best['width']} @ gen {best['gen']})",
                flush=True)

    c = best["champ"]
    Tp = _patch_pool(Xte[yte == pos], whiten)
    Tn = _patch_pool(Xte[yte == neg], whiten)
    testf = SingleGenomeBinaryGPU(Tp, Tn, M_max)
    K1 = c["K"][None]; kb1 = c["kb"][None]; act1 = c["act"][None]
    w1 = c["w"][None]; rb1 = np.array([c["rb"]], np.float32); g1 = c["gate"][None]
    test_acc, pp, pn = testf.eval_full(K1, kb1, act1, w1, rb1, gate=g1)
    val_acc, _, _ = valf.eval_full(K1, kb1, act1, w1, rb1, gate=g1)

    g = sig(c["gate"])
    on = np.where(g > 0.5)[0]
    log("", flush=True)
    log(f"[masked] RESULT {a_name}(+) vs {b_name}(-)", flush=True)
    log(f"  majority-class baseline : 0.5000", flush=True)
    log(f"  val  acc (held-out)     : {val_acc:.4f}", flush=True)
    log(f"  test acc (held-out)     : {test_acc:.4f}  (champion from gen {best['gen']})",
        flush=True)
    drop = (val_acc - test_acc) / max(val_acc, 1e-9)
    log(f"  val->test rel. drop     : {drop * 100:.1f}%", flush=True)
    log(f"  EVOLVED hidden dim      : {len(on)} / {M_max} filters active "
        f"(g>0.5)", flush=True)
    log(f"  gate strengths (sorted) : "
        f"{np.round(np.sort(g)[::-1], 2).tolist()}", flush=True)
    log(f"  active filter activations: {[int(c['act'][m]) for m in on]}", flush=True)
    log(f"  example P({a_name}) on real {a_name}: {np.round(pp[:6], 3).tolist()}",
        flush=True)
    log(f"  example P({a_name}) on real {b_name}: {np.round(pn[:6], 3).tolist()}",
        flush=True)

    payload = {"pos": pos, "neg": neg, "M_max": M_max, "whiten": whiten,
               "champ": c, "val_acc": val_acc, "test_acc": test_acc,
               "gen": best["gen"], "width": len(on), "labels": (a_name, b_name),
               "cost_lambda": cost_lambda}
    with open(OUT_MASKED, "wb") as f:
        pickle.dump(payload, f)
    log(f"[masked] saved champion -> {OUT_MASKED}", flush=True)
    return payload


# --------------------------------------------------------------------------
# TWO-LAYER internal vocabulary (filters-of-filters). Layer 1 = L1 evolved
# 5x5x3 texture filters (like the single-layer genome) whose activated maps are
# avg-pooled 28->14; layer 2 = L2 evolved 3x3xL1 filters that combine layer-1
# channels into PART detectors, pooled and read out. Everything (both filter
# banks, both activation sets, the readout) is one genome, evolved jointly. The
# question: does composing texture into parts lift the cat/dog wall that a flat
# 5x5 representation cannot cross?
# --------------------------------------------------------------------------
H1 = 14                                             # layer-1 map size after pool
R2 = H1 - 2                                         # 12, after 3x3 layer-2 conv
OUT2 = os.path.join(cp.ROOT, "demo", "cifar_internal2.pkl")


class SingleGenome2LayerBinaryGPU:
    def __init__(self, Pp, Pn, L1, L2, pools=POOLS):
        import torch
        self.torch = torch
        self.L1, self.L2, self.pools = L1, L2, pools
        self.PD = sum(r * c for r, c in pools)
        self.Np, self.RR, self.kd = Pp.shape
        self.Nn = Pn.shape[0]
        self.Pp = evo_gpu.to_dev(Pp.astype(np.float32))
        self.Pn = evo_gpu.to_dev(Pn.astype(np.float32))

    def _l1maps(self, batch, K1, kb1, act1):
        """batch (B,RR,KD) -> (P, L1, B, 14, 14) activated, avg-pooled maps."""
        torch = self.torch
        F = torch.nn.functional
        P, L1, B = K1.shape[0], self.L1, batch.shape[0]
        D = P * L1
        Kf = evo_gpu.to_dev(K1.reshape(D, self.kd))
        bf = evo_gpu.to_dev(kb1.reshape(D).astype(np.float32))
        resp = (batch.reshape(B * self.RR, self.kd) @ Kf.T + bf).reshape(B, RESP, RESP, D)
        m = torch.empty((B, RESP, RESP, D), device=evo_gpu.DEV)
        actf = act1.reshape(D)
        for a in range(8):
            ids = np.where(actf == a)[0]
            if len(ids) == 0:
                continue
            idt = evo_gpu.to_dev(ids.astype(np.int64))
            m.index_copy_(3, idt, evo_gpu._acts_t(resp.index_select(3, idt), a))
        m = F.avg_pool2d(m.permute(3, 0, 1, 2).reshape(D * B, 1, RESP, RESP), 2)
        return m.reshape(P, L1, B, H1, H1)

    def _feats(self, batch, K1, kb1, act1, K2, kb2, act2, chunk=64):
        """-> (P, B, L2*PD) layer-2 part features (the deep internal language)."""
        torch = self.torch
        F = torch.nn.functional
        P, L1, L2, B = K1.shape[0], self.L1, self.L2, batch.shape[0]
        m1 = self._l1maps(batch, K1, kb1, act1)             # (P,L1,B,14,14)
        K2g = evo_gpu.to_dev(K2)                             # (P, L2, 9*L1)
        b2g = evo_gpu.to_dev(kb2.astype(np.float32))         # (P, L2)
        feat = torch.empty((P, B, L2 * self.PD), device=evo_gpu.DEV)
        for lo in range(0, B, chunk):
            n = min(chunk, B - lo)
            sub = m1[:, :, lo:lo + n]                        # (P,L1,n,14,14)
            sub = sub.permute(0, 2, 1, 3, 4).reshape(P * n, L1, H1, H1)
            u = F.unfold(sub, 3).reshape(P, n, 9 * L1, R2 * R2)   # (P,n,9L1,144)
            z = torch.einsum("pnfk,plf->pnlk", u, K2g) + b2g[:, None, :, None]
            z = z.reshape(P, n, L2, R2, R2)
            # per-(genome,filter) activation from the catalog; act2 is (P,L2)
            pooled = torch.zeros((P, n, L2, self.PD), device=evo_gpu.DEV)
            for a in range(8):
                mask = (act2 == a)                           # (P,L2) bool
                if not mask.any():
                    continue
                blk = evo_gpu._acts_t(z, a).reshape(P * n * L2, R2, R2)
                pl = evo_gpu._pool_t(blk, R2, self.pools).reshape(P, n, L2, self.PD)
                mm = evo_gpu.to_dev(mask.astype(np.float32))[:, None, :, None]
                pooled = pooled + pl * mm
            feat[:, lo:lo + n] = pooled.reshape(P, n, L2 * self.PD)
        return feat

    def _scores(self, feat, w, rb):
        wg = evo_gpu.to_dev(w)
        rbg = evo_gpu.to_dev(rb.astype(np.float32))
        return (feat * wg[:, None, :]).sum(-1) + rbg[:, None]

    def __call__(self, K1, kb1, act1, K2, kb2, act2, w, rb, ip, inn):
        torch = self.torch
        with torch.no_grad():
            fp = self._feats(self.Pp[evo_gpu.to_dev(ip.astype(np.int64))],
                             K1, kb1, act1, K2, kb2, act2)
            fn = self._feats(self.Pn[evo_gpu.to_dev(inn.astype(np.int64))],
                             K1, kb1, act1, K2, kb2, act2)
            zp = self._scores(fp, w, rb).clamp(-30, 30)
            zn = self._scores(fn, w, rb).clamp(-30, 30)
            lp = -torch.log1p(torch.exp(-zp)).mean(1)
            ln = -torch.log1p(torch.exp(zn)).mean(1)
            acc = ((zp > 0).float().mean(1) + (zn < 0).float().mean(1)) / 2
            return (lp + ln).cpu().numpy(), acc.cpu().numpy()

    def eval_full(self, champ, chunk=100):
        """Single-genome (P=1) probabilities over the uploaded pos/neg pools."""
        torch = self.torch
        K1 = champ["K1"][None]; kb1 = champ["kb1"][None]; act1 = champ["act1"][None]
        K2 = champ["K2"][None]; kb2 = champ["kb2"][None]; act2 = champ["act2"][None]
        w = champ["w"][None]; rb = np.array([champ["rb"]], np.float32)
        with torch.no_grad():
            def probs(Pool, n):
                out = []
                for lo in range(0, n, chunk):
                    idx = np.arange(lo, min(lo + chunk, n))
                    f = self._feats(Pool[evo_gpu.to_dev(idx.astype(np.int64))],
                                    K1, kb1, act1, K2, kb2, act2)
                    out.append(torch.sigmoid(self._scores(f, w, rb).clamp(-30, 30))[0]
                               .cpu().numpy())
                return np.concatenate(out)
            pp = probs(self.Pp, self.Np); pn = probs(self.Pn, self.Nn)
        acc = ((pp > 0.5).mean() + (pn < 0.5).mean()) / 2
        return float(acc), pp, pn

    def logits(self, champ, chunk=100):
        """Single-genome raw logits over the uploaded positive pool (for arbiter)."""
        torch = self.torch
        K1 = champ["K1"][None]; kb1 = champ["kb1"][None]; act1 = champ["act1"][None]
        K2 = champ["K2"][None]; kb2 = champ["kb2"][None]; act2 = champ["act2"][None]
        w = champ["w"][None]; rb = np.array([champ["rb"]], np.float32)
        out = []
        with torch.no_grad():
            for lo in range(0, self.Np, chunk):
                idx = np.arange(lo, min(lo + chunk, self.Np))
                f = self._feats(self.Pp[evo_gpu.to_dev(idx.astype(np.int64))],
                                K1, kb1, act1, K2, kb2, act2)
                out.append(self._scores(f, w, rb)[0].cpu().numpy())
        return np.concatenate(out)


def evolve_single2(pos=3, neg=5, L1=8, L2=8, pop=48, gens=1500, minibatch=96,
                   n_train=2000, n_val=500, whiten=True, seed=7, log_every=100,
                   log=print):
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    a_name, b_name = LABELS[pos], LABELS[neg]
    log(f"[2layer] {a_name}(+) vs {b_name}(-)  L1={L1} L2={L2} pop={pop} "
        f"gens={gens} whiten={whiten} seed={seed}", flush=True)
    if whiten:
        cp.build_zca()
    Pp = _patch_pool(_sample(Xtr, ytr, pos, n_train, seed), whiten)
    Pn = _patch_pool(_sample(Xtr, ytr, neg, n_train, seed + 1), whiten)
    Vp = _patch_pool(_sample(Xva, yva, pos, n_val, seed + 2), whiten)
    Vn = _patch_pool(_sample(Xva, yva, neg, n_val, seed + 3), whiten)
    if not evo_gpu.HAS_GPU:
        raise RuntimeError("this experiment expects the GPU backend")
    trainf = SingleGenome2LayerBinaryGPU(Pp, Pn, L1, L2)
    valf = SingleGenome2LayerBinaryGPU(Vp, Vn, L1, L2)

    rng = np.random.default_rng(seed)
    params = {
        "K1": (rng.standard_normal((pop, L1, KD)) * 0.15).astype(np.float32),
        "kb1": np.zeros((pop, L1), np.float32),
        "act1": rng.integers(0, 8, (pop, L1)).astype(np.float32),
        "K2": (rng.standard_normal((pop, L2, 9 * L1)) * (1.0 / np.sqrt(9 * L1))).astype(np.float32),
        "kb2": np.zeros((pop, L2), np.float32),
        "act2": rng.integers(0, 8, (pop, L2)).astype(np.float32),
        "w": (rng.standard_normal((pop, L2 * PD)) / np.sqrt(L2 * PD)).astype(np.float32),
        "rb": np.zeros(pop, np.float32),
        "sigma": np.full(pop, 0.08, np.float32),
    }

    def ids1():
        return np.round(params["act1"]).astype(np.int64) % 8

    def ids2():
        return np.round(params["act2"]).astype(np.int64) % 8

    best = {"fit": -1e9, "acc": 0.0, "gen": 0, "champ": None}
    for gen in range(1, gens + 1):
        ip = rng.integers(0, len(Pp), minibatch)
        inn = rng.integers(0, len(Pn), minibatch)
        fit, _ = trainf(params["K1"], params["kb1"], ids1(),
                        params["K2"], params["kb2"], ids2(),
                        params["w"], params["rb"], ip, inn)
        if gen >= int(0.8 * gens):
            params["sigma"] = np.minimum(params["sigma"], 0.04)
        mp.ga_step(params, fit, rng, mag_scale=True)
        params["act1"] = np.round(params["act1"]) % 8
        params["act2"] = np.round(params["act2"]) % 8
        if gen % log_every == 0 or gen == 1 or gen == gens:
            vfit, vacc = valf(params["K1"], params["kb1"], ids1(),
                              params["K2"], params["kb2"], ids2(),
                              params["w"], params["rb"],
                              np.arange(len(Vp)), np.arange(len(Vn)))
            j = int(np.argmax(vfit))
            if float(vfit[j]) > best["fit"]:
                best.update(fit=float(vfit[j]), acc=float(vacc[j]), gen=gen,
                            champ={"K1": params["K1"][j].copy(),
                                   "kb1": params["kb1"][j].copy(),
                                   "act1": ids1()[j].copy(),
                                   "K2": params["K2"][j].copy(),
                                   "kb2": params["kb2"][j].copy(),
                                   "act2": ids2()[j].copy(),
                                   "w": params["w"][j].copy(),
                                   "rb": float(params["rb"][j]),
                                   "L1": L1, "L2": L2})
            log(f"  gen {gen:5d}: val_logprob={vfit[j]:.4f} val_acc={vacc[j]:.4f} "
                f"(best {best['acc']:.4f} @ gen {best['gen']})", flush=True)

    c = best["champ"]
    Tp = _patch_pool(Xte[yte == pos], whiten)
    Tn = _patch_pool(Xte[yte == neg], whiten)
    testf = SingleGenome2LayerBinaryGPU(Tp, Tn, L1, L2)
    test_acc, pp, pn = testf.eval_full(c)
    val_acc, _, _ = valf.eval_full(c)
    log("", flush=True)
    log(f"[2layer] RESULT {a_name}(+) vs {b_name}(-)", flush=True)
    log(f"  majority baseline   : 0.5000", flush=True)
    log(f"  val  acc (held-out) : {val_acc:.4f}", flush=True)
    log(f"  test acc (held-out) : {test_acc:.4f}  (champion gen {best['gen']})",
        flush=True)
    log(f"  val->test rel drop  : {(val_acc - test_acc) / max(val_acc, 1e-9) * 100:.1f}%",
        flush=True)
    log(f"  L1 acts {[int(x) for x in c['act1']]}  L2 acts {[int(x) for x in c['act2']]}",
        flush=True)
    log(f"  P({a_name}) on real {a_name}: {np.round(pp[:6], 3).tolist()}", flush=True)
    log(f"  P({a_name}) on real {b_name}: {np.round(pn[:6], 3).tolist()}", flush=True)
    payload = {"pos": pos, "neg": neg, "L1": L1, "L2": L2, "whiten": whiten,
               "champ": c, "val_acc": val_acc, "test_acc": test_acc,
               "gen": best["gen"], "labels": (a_name, b_name)}
    with open(OUT2, "wb") as f:
        pickle.dump(payload, f)
    log(f"[2layer] saved -> {OUT2}", flush=True)
    return payload


def champ_logits2(champ, X, L1, L2, whiten=True, chunk=100):
    Pool = _patch_pool(X, whiten)
    inst = SingleGenome2LayerBinaryGPU(Pool, Pool[:2], L1, L2)
    return inst.logits(champ, chunk)


# --------------------------------------------------------------------------
# ARBITER + CHECKER: two decider genomes seeded differently (the "same model
# from a different seed"), and a checker genome that verifies the answer before
# outputting it. The checker reads both deciders' raw scores AND their
# DISAGREEMENT: when the two seeds agree it is a confident case; when the
# answer changes with the seed (they disagree) it is uncertain, and the checker
# is evolved to resolve those. All gradient-free.
# --------------------------------------------------------------------------
OUT_ARBITER = os.path.join(cp.ROOT, "demo", "cifar_arbiter.pkl")


def champ_logits(champ, X, M, whiten=True, chunk=250):
    """Raw decision logit of one decider champion over images X -> (len(X),)."""
    import torch
    Pool = _patch_pool(X, whiten)
    inst = SingleGenomeBinaryGPU(Pool, Pool[:2], M)
    K1, kb1, act1 = champ["K"][None], champ["kb"][None], champ["act"][None]
    w1, rb1 = champ["w"][None], np.array([champ["rb"]], np.float32)
    out = []
    with torch.no_grad():
        for lo in range(0, inst.Np, chunk):
            idx = np.arange(lo, min(lo + chunk, inst.Np))
            z = inst._scores(inst._feats(inst.Pp[evo_gpu.to_dev(idx.astype(np.int64))],
                                         K1, kb1, act1), w1, rb1)
            out.append(z[0].cpu().numpy())
    return np.concatenate(out)


def _balanced(X, y, pos, neg, n, seed):
    rng = np.random.default_rng(seed)
    ip = np.where(y == pos)[0]; inn = np.where(y == neg)[0]
    ip = rng.choice(ip, min(n, len(ip)), replace=False)
    inn = rng.choice(inn, min(n, len(inn)), replace=False)
    Xb = np.concatenate([X[ip], X[inn]])
    lab = np.concatenate([np.ones(len(ip)), np.zeros(len(inn))]).astype(np.float32)
    return Xb, lab


def _checker_feats(sA, sB):
    """Features the checker sees per image: the two decider logits, their sum,
    their signed and absolute disagreement, and their product. The absolute
    disagreement |sA-sB| is the verification signal — large means the answer
    changes with the seed."""
    return np.stack([sA, sB, sA + sB, sA - sB, np.abs(sA - sB), sA * sB],
                    axis=1).astype(np.float32)


def _checker_scores(F, g, act_ids):
    """F (B,nf); g = {W1(P,H,nf), b1(P,H), w2(P,H), b2(P)} -> (P,B) logits.
    One evolved hidden layer with per-unit activation from the 8-catalog."""
    z1 = np.einsum("bf,phf->pbh", F, g["W1"]) + g["b1"][:, None, :]
    h = np.zeros_like(z1)
    for a in range(8):
        ma = act_ids == a
        if ma.any():
            h += np.where(ma[:, None, :], mp._acts(z1, a), 0.0)
    return (h * g["w2"][:, None, :]).sum(-1) + g["b2"][:, None]


def evolve_checker(Ftr, ytr, Fva, yva, H=8, pop=200, gens=1500, minibatch=256,
                   seed=13, log_every=200, log=print):
    nf = Ftr.shape[1]
    rng = np.random.default_rng(seed)
    params = {
        "W1": (rng.standard_normal((pop, H, nf)) / np.sqrt(nf)).astype(np.float32),
        "b1": np.zeros((pop, H), np.float32),
        "act": rng.integers(0, 8, (pop, H)).astype(np.float32),
        "w2": (rng.standard_normal((pop, H)) / np.sqrt(H)).astype(np.float32),
        "b2": np.zeros(pop, np.float32),
        "sigma": np.full(pop, 0.08, np.float32),
    }
    ip = np.where(ytr == 1)[0]; inn = np.where(ytr == 0)[0]

    def fit_on(F, y, ai):
        z = np.clip(_checker_scores(F, params, ai), -30, 30)
        ly = y[None, :]
        bce = ly * -np.log1p(np.exp(-z)) + (1 - ly) * -np.log1p(np.exp(z))
        acc = ((z > 0) == (ly > 0.5)).mean(1)
        return bce.mean(1), acc

    best = {"fit": -1e9, "acc": 0.0, "champ": None}
    for gen in range(1, gens + 1):
        bi = np.concatenate([rng.choice(ip, minibatch // 2),
                             rng.choice(inn, minibatch // 2)])
        ai = np.round(params["act"]).astype(np.int64) % 8
        fit, _ = fit_on(Ftr[bi], ytr[bi], ai)
        mp.ga_step(params, fit, rng, mag_scale=True)
        params["act"] = np.round(params["act"]) % 8
        if gen % log_every == 0 or gen == 1 or gen == gens:
            ai = np.round(params["act"]).astype(np.int64) % 8
            vfit, vacc = fit_on(Fva, yva, ai)
            j = int(np.argmax(vfit))
            if float(vfit[j]) > best["fit"]:
                best.update(fit=float(vfit[j]), acc=float(vacc[j]),
                            champ={k: params[k][j].copy() for k in
                                   ("W1", "b1", "w2", "b2")} | {"act": ai[j].copy()})
            log(f"  [checker] gen {gen:5d}: val_logprob={vfit[j]:.4f} "
                f"val_acc={vacc[j]:.4f} (best {best['acc']:.4f})", flush=True)
    return best["champ"]


def evolve_arbiter(pos=3, neg=5, M=8, seedA=7, seedB=101, dec_gens=2000,
                   checker_gens=1500, n_train=2000, n_check=2000, whiten=True,
                   log=print):
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    a_name, b_name = LABELS[pos], LABELS[neg]
    log(f"=== ARBITER+CHECKER: {a_name}(+) vs {b_name}(-) ===", flush=True)

    log(f"\n--- decider A (seed {seedA}) ---", flush=True)
    A = evolve_single(pos, neg, M=M, gens=dec_gens, n_train=n_train,
                      whiten=whiten, seed=seedA, log=log)["champ"]
    log(f"\n--- decider B (seed {seedB}) ---", flush=True)
    B = evolve_single(pos, neg, M=M, gens=dec_gens, n_train=n_train,
                      whiten=whiten, seed=seedB, log=log)["champ"]

    def feats_for(X):
        return _checker_feats(champ_logits(A, X, M, whiten),
                              champ_logits(B, X, M, whiten))

    # checker train / val / test feature sets from the two deciders
    Xct, yct = _balanced(Xtr, ytr, pos, neg, n_check, seedA + 555)
    Xcv, ycv = _balanced(Xva, yva, pos, neg, 10 ** 9, seedA + 556)
    Xte2, yte2 = _balanced(Xte, yte, pos, neg, 10 ** 9, seedA + 557)
    Ftr, Fva, Fte = feats_for(Xct), feats_for(Xcv), feats_for(Xte2)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6                # standardise on train
    Ftr, Fva, Fte = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd

    log(f"\n--- checker genome (verifier) ---", flush=True)
    C = evolve_checker(Ftr, yct, Fva, ycv, gens=checker_gens, log=log)

    # ---- evaluation on the held-out test set ----
    sA, sB = champ_logits(A, Xte2, M, whiten), champ_logits(B, Xte2, M, whiten)
    accA = ((sA > 0) == (yte2 > 0.5)).mean()
    accB = ((sB > 0) == (yte2 > 0.5)).mean()
    pAB = ((1 / (1 + np.exp(-sA)) + 1 / (1 + np.exp(-sB))) / 2 > 0.5)
    acc_avg = (pAB == (yte2 > 0.5)).mean()
    ai = np.round(C["act"]).astype(np.int64) % 8
    zc = _checker_scores(Fte, {k: C[k][None] for k in ("W1", "b1", "w2", "b2")}, ai[None])[0]
    acc_chk = ((zc > 0) == (yte2 > 0.5)).mean()

    agree = (sA > 0) == (sB > 0)                           # do the two seeds agree?
    dis = ~agree
    acc_agree = (((sA > 0) == (yte2 > 0.5))[agree]).mean() if agree.any() else 0.0
    accA_dis = (((sA > 0) == (yte2 > 0.5))[dis]).mean() if dis.any() else 0.0
    acc_chk_dis = ((zc > 0) == (yte2 > 0.5))[dis].mean() if dis.any() else 0.0

    log("", flush=True)
    log(f"=== RESULT {a_name}(+) vs {b_name}(-) -- arbiter + checker ===", flush=True)
    log(f"  majority baseline        : 0.5000", flush=True)
    log(f"  decider A alone (test)   : {accA:.4f}", flush=True)
    log(f"  decider B alone (test)   : {accB:.4f}", flush=True)
    log(f"  average ensemble (test)  : {acc_avg:.4f}", flush=True)
    log(f"  CHECKER genome (test)    : {acc_chk:.4f}  <- verified output", flush=True)
    log(f"  --- seed-consistency (verification) ---", flush=True)
    log(f"  seeds AGREE on           : {agree.mean() * 100:.1f}% of test images", flush=True)
    log(f"    acc where they agree   : {acc_agree:.4f}  (confident cases)", flush=True)
    log(f"    acc where they DISAGREE: A={accA_dis:.4f}  (answer changes w/ seed)",
        flush=True)
    log(f"    checker on disagree    : {acc_chk_dis:.4f}  (does it resolve them?)",
        flush=True)

    payload = {"pos": pos, "neg": neg, "M": M, "deciderA": A, "deciderB": B,
               "checker": C, "feat_mu": mu, "feat_sd": sd,
               "labels": (a_name, b_name),
               "acc": {"A": float(accA), "B": float(accB), "avg": float(acc_avg),
                       "checker": float(acc_chk), "agree_frac": float(agree.mean()),
                       "acc_agree": float(acc_agree), "checker_disagree": float(acc_chk_dis)}}
    with open(OUT_ARBITER, "wb") as f:
        pickle.dump(payload, f)
    log(f"\n[arbiter] saved -> {OUT_ARBITER}", flush=True)
    return payload


# --------------------------------------------------------------------------
# TEN-CLASS single genome — the real test. One genome must evolve an internal
# language that organises ALL of CIFAR-10, not just one boundary. Conv filters
# + per-filter activation + pooling exactly as the binary genome, but the
# readout is now (M*PD -> 10) with a softmax and SOFT cross-entropy fitness
# (mean log-prob of the true class, GENREG rule IV.1). Chance = 10%; the old
# bank->features->logreg pipeline ceiling (~65%) is the upper reference.
# --------------------------------------------------------------------------
OUT10 = os.path.join(cp.ROOT, "demo", "cifar_internal10.pkl")


class SingleGenome10ClassGPU:
    def __init__(self, patches, y, M, nc=10, pools=POOLS):
        import torch
        self.torch = torch
        self.M, self.nc, self.pools = M, nc, pools
        self.PD = sum(r * c for r, c in pools)
        self.N, self.RR, self.kd = patches.shape
        self.Pf = evo_gpu.to_dev(patches.astype(np.float32))    # (N,RR,KD)
        self.y = y.astype(np.int64)

    def _feats(self, batch, K, kb, act):
        torch = self.torch
        P, M, B = K.shape[0], self.M, batch.shape[0]
        D = P * M
        Kf = evo_gpu.to_dev(K.reshape(D, self.kd))
        bf = evo_gpu.to_dev(kb.reshape(D).astype(np.float32))
        resp = (batch.reshape(B * self.RR, self.kd) @ Kf.T + bf) \
            .reshape(B, RESP, RESP, D)
        pooled = torch.empty((D, B, self.PD), device=evo_gpu.DEV)
        actf = act.reshape(D)
        for a in range(8):
            ids = np.where(actf == a)[0]
            if len(ids) == 0:
                continue
            idt = evo_gpu.to_dev(ids.astype(np.int64))
            blk = evo_gpu._acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * B, RESP, RESP)
            pooled[idt] = evo_gpu._pool_t(blk, RESP, self.pools) \
                .reshape(len(ids), B, self.PD)
        return pooled.reshape(P, M, B, self.PD).permute(0, 2, 1, 3) \
            .reshape(P, B, M * self.PD)

    def __call__(self, K, kb, act, W, b, idx):
        torch = self.torch
        with torch.no_grad():
            feat = self._feats(self.Pf[evo_gpu.to_dev(idx.astype(np.int64))],
                               K, kb, act)                      # (P,B,MPD)
            Wg = evo_gpu.to_dev(W); bg = evo_gpu.to_dev(b)      # (P,MPD,nc),(P,nc)
            z = torch.einsum("pbf,pfc->pbc", feat, Wg) + bg[:, None, :]
            logp = torch.log_softmax(z, dim=-1)
            yb = evo_gpu.to_dev(self.y[idx])
            P, B = len(K), len(idx)
            ch = logp.gather(2, yb[None, :, None].expand(P, B, 1))[..., 0]
            fit = ch.mean(1)                                    # mean log-prob true
            acc = (z.argmax(-1) == yb[None]).float().mean(1)
            return fit.cpu().numpy(), acc.cpu().numpy()

    def eval_full(self, champ, chunk=250):
        """Single-genome (P=1) top-1 accuracy + confusion matrix over the pool."""
        torch = self.torch
        K1, kb1, act1 = champ["K"][None], champ["kb"][None], champ["act"][None]
        W1, b1 = champ["W"][None], champ["b"][None]
        preds = np.empty(self.N, np.int64)
        with torch.no_grad():
            for lo in range(0, self.N, chunk):
                idx = np.arange(lo, min(lo + chunk, self.N))
                feat = self._feats(self.Pf[evo_gpu.to_dev(idx.astype(np.int64))],
                                   K1, kb1, act1)
                z = torch.einsum("pbf,pfc->pbc", feat, evo_gpu.to_dev(W1)) \
                    + evo_gpu.to_dev(b1)[:, None, :]
                preds[idx] = z.argmax(-1)[0].cpu().numpy()
        acc = float((preds == self.y).mean())
        conf = np.zeros((self.nc, self.nc), np.int64)
        for t, p in zip(self.y, preds):
            conf[t, p] += 1
        return acc, preds, conf


def _labeled_pool(X, y, per_class, seed, whiten=True):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                     min(per_class, int((y == c).sum())),
                                     replace=False) for c in range(10)])
    return _patch_pool(X[idx], whiten), y[idx]


def evolve_ten(M=16, pop=48, gens=3000, minibatch=300, n_train=700, n_val=300,
               whiten=True, seed=7, log_every=100, log=print):
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    log(f"[ten] CIFAR-10 single genome  M={M} pop={pop} gens={gens} "
        f"whiten={whiten} seed={seed}  (chance=0.10, pipeline ceiling ~0.65)",
        flush=True)
    if whiten:
        cp.build_zca()
    Ptr, ytr2 = _labeled_pool(Xtr, ytr, n_train, seed, whiten)
    Pva, yva2 = _labeled_pool(Xva, yva, n_val, seed + 1, whiten)
    log(f"[ten] train pool {len(Ptr)} imgs, val pool {len(Pva)} imgs", flush=True)
    if not evo_gpu.HAS_GPU:
        raise RuntimeError("this experiment expects the GPU backend")
    trainf = SingleGenome10ClassGPU(Ptr, ytr2, M)
    valf = SingleGenome10ClassGPU(Pva, yva2, M)
    cls_idx = [np.where(ytr2 == c)[0] for c in range(10)]

    rng = np.random.default_rng(seed)
    MPD = M * PD
    params = {
        "K": (rng.standard_normal((pop, M, KD)) * 0.15).astype(np.float32),
        "kb": np.zeros((pop, M), np.float32),
        "act": rng.integers(0, 8, (pop, M)).astype(np.float32),
        "W": (rng.standard_normal((pop, MPD, 10)) / np.sqrt(MPD)).astype(np.float32),
        "b": np.zeros((pop, 10), np.float32),
        "sigma": np.full(pop, 0.08, np.float32),
    }

    def ids():
        return np.round(params["act"]).astype(np.int64) % 8

    per = minibatch // 10
    best = {"fit": -1e9, "acc": 0.0, "gen": 0, "champ": None}
    for gen in range(1, gens + 1):
        bi = np.concatenate([rng.choice(cls_idx[c], per) for c in range(10)])
        fit, _ = trainf(params["K"], params["kb"], ids(), params["W"],
                        params["b"], bi)
        if gen >= int(0.8 * gens):
            params["sigma"] = np.minimum(params["sigma"], 0.04)
        mp.ga_step(params, fit, rng, mag_scale=True)
        params["act"] = np.round(params["act"]) % 8
        if gen % log_every == 0 or gen == 1 or gen == gens:
            vfit, vacc = valf(params["K"], params["kb"], ids(), params["W"],
                              params["b"], np.arange(len(Pva)))
            j = int(np.argmax(vfit))
            if float(vfit[j]) > best["fit"]:
                best.update(fit=float(vfit[j]), acc=float(vacc[j]), gen=gen,
                            champ={"K": params["K"][j].copy(),
                                   "kb": params["kb"][j].copy(),
                                   "act": ids()[j].copy(),
                                   "W": params["W"][j].copy(),
                                   "b": params["b"][j].copy(), "M": M})
            log(f"  gen {gen:5d}: val_logprob={vfit[j]:.4f} val_top1={vacc[j]:.4f} "
                f"(best {best['acc']:.4f} @ gen {best['gen']})", flush=True)

    c = best["champ"]
    del trainf, valf
    import torch
    torch.cuda.empty_cache()
    Pte, yte2 = _labeled_pool(Xte, yte, 1000, seed + 2, whiten)
    testf = SingleGenome10ClassGPU(Pte, yte2, M)
    test_acc, preds, conf = testf.eval_full(c)

    log("", flush=True)
    log(f"[ten] RESULT CIFAR-10 (all classes), single genome", flush=True)
    log(f"  chance (majority)   : 0.1000", flush=True)
    log(f"  pipeline ref ceiling: ~0.6500 (bank+features+logreg, old approach)",
        flush=True)
    log(f"  val  top-1          : {best['acc']:.4f}", flush=True)
    log(f"  test top-1          : {test_acc:.4f}  (champion gen {best['gen']})",
        flush=True)
    log(f"  activations used    : {[int(x) for x in c['act']]}", flush=True)
    per_cls = conf.diagonal() / conf.sum(1).clip(1)
    log(f"  per-class top-1     :", flush=True)
    for ci in range(10):
        row = conf[ci]
        top_conf = int(np.argsort(row)[::-1][0])
        top_conf = int(np.argsort(row)[::-1][1]) if top_conf == ci else top_conf
        log(f"    {LABELS[ci]:>6}: {per_cls[ci]:.3f}   most-confused-with "
            f"{LABELS[top_conf]}", flush=True)
    payload = {"champ": c, "M": M, "whiten": whiten, "val_acc": best["acc"],
               "test_acc": test_acc, "gen": best["gen"], "confusion": conf}
    with open(OUT10, "wb") as f:
        pickle.dump(payload, f)
    log(f"[ten] saved -> {OUT10}", flush=True)
    return payload


# ==========================================================================
# CONTRASTIVE ENCODER — the pivot. Not a classifier: evolve an ENCODER whose
# private latent code captures similarity/difference structure. The encoder
# NEVER sees a label. Positives = two augmentations of the same image;
# negatives = other images (SimCLR / NT-Xent, gradient-free). Afterwards we
# CHECK whether class geometry emerged on its own: kNN accuracy and the
# inter-class similarity matrix, with labels used ONLY at evaluation.
# ==========================================================================
OUT_ENC = os.path.join(cp.ROOT, "demo", "cifar_encoder.pkl")


def _augment(X, rng, hard=False, color=False, kind=None):
    """Label-free views. Each `kind` is a QUESTION posed by what it VARIES (the
    encoder must be invariant to it, so it learns the complement):
      standard - flip/shift/occlusion/brightness (coarse identity)
      color    - vary COLOUR strongly            -> learn SHAPE
      crop     - vary CROP/SCALE strongly        -> learn scale/position-invariant global identity
      occlude  - vary WHICH PARTS are visible    -> infer identity from parts/context
      warp     - vary SHAPE (rotate/stretch)     -> learn COLOUR/TEXTURE/appearance
    Applied to (N,32,32,3) before im2col."""
    if kind is None:
        kind = "color" if color else ("hard" if hard else "standard")
    N = len(X)
    Y = X.copy()
    flip = rng.random(N) < 0.5
    Y[flip] = Y[flip, :, ::-1, :]

    if kind == "crop":
        for i in range(N):
            cs = int(rng.integers(16, 33))
            ty, tx = rng.integers(0, 33 - cs, 2)
            patch = Y[i, ty:ty + cs, tx:tx + cs]
            idx = (np.arange(32) * cs / 32).astype(np.int64)
            Y[i] = patch[idx][:, idx]
        Y *= rng.uniform(0.85, 1.15, (N, 1, 1, 1)).astype(np.float32)
        return np.clip(Y, 0.0, 1.0)

    if kind == "warp":
        from scipy import ndimage
        for i in range(N):
            ang = float(rng.uniform(-22, 22))
            Y[i] = ndimage.rotate(Y[i], ang, reshape=False, order=1, mode="reflect")
            if rng.random() < 0.5:                       # axis stretch
                ax = int(rng.integers(0, 2))
                f = float(rng.uniform(0.8, 1.25))
                z = ndimage.zoom(Y[i], (f, 1, 1) if ax == 0 else (1, f, 1), order=1)
                z = z[:32, :32] if z.shape[0] >= 32 or z.shape[1] >= 32 else z
                zz = np.zeros_like(Y[i]); zz[:z.shape[0], :z.shape[1]] = z[:32, :32]
                Y[i] = zz
        Y *= rng.uniform(0.85, 1.15, (N, 1, 1, 1)).astype(np.float32)
        return np.clip(Y, 0.0, 1.0)

    sh = 2 if kind == "color" else 4
    for i in range(N):
        dy, dx = rng.integers(-sh, sh + 1, 2)
        Y[i] = np.roll(Y[i], (int(dy), int(dx)), axis=(0, 1))
        if kind == "occlude":                            # large cutouts: vary visible parts
            for _ in range(int(rng.integers(1, 3))):
                oy, ox = rng.integers(0, 20, 2)
                s = int(rng.integers(10, 18))
                Y[i, oy:oy + s, ox:ox + s, :] = 0.0
        elif kind in ("standard", "hard"):
            for _ in range(2 if kind == "hard" else 1):
                if rng.random() < 0.5:
                    oy, ox = rng.integers(0, 24, 2)
                    s = int(rng.integers(6, 12))
                    Y[i, oy:oy + s, ox:ox + s, :] = 0.0

    if kind == "color":
        Y *= rng.uniform(0.4, 1.6, (N, 1, 1, 3)).astype(np.float32)
        gray = rng.random(N) < 0.4
        g = Y[gray].mean(axis=3, keepdims=True)
        Y[gray] = np.repeat(g, 3, axis=3)
    else:
        Y *= rng.uniform(0.8, 1.2, (N, 1, 1, 1)).astype(np.float32)
    return np.clip(Y, 0.0, 1.0)


class ContrastiveEncoderGPU:
    """NT-Xent fitness over a population of encoder genomes. Views bank uploaded
    once: (K anchors x V augmented views), im2col'd + whitened."""

    def __init__(self, bank, K, V, M, d, tau=0.2, pools=POOLS):
        import torch
        self.torch = torch
        self.K, self.V, self.M, self.d, self.tau = K, V, M, d, tau
        self.pools = pools
        self.PD = sum(r * c for r, c in pools)
        self.NV, self.RR, self.kd = bank.shape          # (K*V, RR, KD)
        self.Pf = evo_gpu.to_dev(bank.astype(np.float32))

    def _feats(self, batch, K, kb, act):
        torch = self.torch
        P, M, B = K.shape[0], self.M, batch.shape[0]
        D = P * M
        Kf = evo_gpu.to_dev(K.reshape(D, self.kd))
        bf = evo_gpu.to_dev(kb.reshape(D).astype(np.float32))
        resp = (batch.reshape(B * self.RR, self.kd) @ Kf.T + bf) \
            .reshape(B, RESP, RESP, D)
        pooled = torch.empty((D, B, self.PD), device=evo_gpu.DEV)
        actf = act.reshape(D)
        for a in range(8):
            ids = np.where(actf == a)[0]
            if len(ids) == 0:
                continue
            idt = evo_gpu.to_dev(ids.astype(np.int64))
            blk = evo_gpu._acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * B, RESP, RESP)
            pooled[idt] = evo_gpu._pool_t(blk, RESP, self.pools) \
                .reshape(len(ids), B, self.PD)
        return pooled.reshape(P, M, B, self.PD).permute(0, 2, 1, 3) \
            .reshape(P, B, M * self.PD)

    def _codes(self, batch, K, kb, act, W):
        """-> (P, B, d) unit-norm latent codes (the private language)."""
        torch = self.torch
        feat = self._feats(batch, K, kb, act)               # (P,B,MPD)
        Wg = evo_gpu.to_dev(W)                              # (P,MPD,d)
        z = torch.einsum("pbf,pfd->pbd", feat, Wg)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

    def __call__(self, K, kb, act, W, gidx):
        """gidx (2N,) interleaved view indices; positive of 2i is 2i+1.
        -> (NT-Xent fitness (P,), positive-retrieval acc (P,)) numpy."""
        torch = self.torch
        with torch.no_grad():
            c = self._codes(self.Pf[evo_gpu.to_dev(gidx.astype(np.int64))],
                            K, kb, act, W)                  # (P, 2N, d)
            P, B, _ = c.shape
            S = torch.einsum("pid,pjd->pij", c, c) / self.tau
            eye = torch.eye(B, device=evo_gpu.DEV).bool()
            S = S.masked_fill(eye[None], -1e9)              # mask self
            tgt = torch.arange(B, device=evo_gpu.DEV) ^ 1   # adjacent partner
            logp = torch.log_softmax(S, dim=2)
            fit = logp.gather(2, tgt[None, :, None].expand(P, B, 1))[..., 0].mean(1)
            acc = (S.argmax(2) == tgt[None]).float().mean(1)
            return fit.cpu().numpy(), acc.cpu().numpy()

    def encode(self, champ, X, whiten=True, chunk=500):
        """Champion encoder over raw images X (single view, no aug) -> (n,d)."""
        torch = self.torch
        Pool = _patch_pool(X, whiten)
        inst_pf = evo_gpu.to_dev(Pool.astype(np.float32))
        K1, kb1, act1, W1 = (champ["K"][None], champ["kb"][None],
                             champ["act"][None], champ["W"][None])
        out = []
        with torch.no_grad():
            for lo in range(0, len(X), chunk):
                b = inst_pf[lo:lo + chunk]
                out.append(self._codes(b, K1, kb1, act1, W1)[0].cpu().numpy())
        return np.concatenate(out)


def _knn_acc(codes_g, y_g, codes_q, y_q, k=10):
    """Label-free encoder, labels used ONLY here: cosine kNN top-1."""
    sim = codes_q @ codes_g.T
    nn = np.argsort(-sim, axis=1)[:, :k]
    votes = y_g[nn]
    pred = np.array([np.bincount(v, minlength=10).argmax() for v in votes])
    return float((pred == y_q).mean())


def evolve_encoder(M=16, d=16, pop=48, gens=2500, n_anchor=1500, V=4, N=64,
                   tau=0.2, whiten=True, seed=7, hard_aug=False, out=None,
                   log_every=100, log=print):
    out = out or OUT_ENC
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    log(f"[enc] contrastive encoder (LABEL-FREE)  M={M} d={d} pop={pop} "
        f"gens={gens} anchors={n_anchor} views={V} tau={tau} seed={seed} "
        f"hard_aug={hard_aug}", flush=True)
    if whiten:
        cp.build_zca()
    rng = np.random.default_rng(seed)
    anchors = Xtr[rng.choice(len(Xtr), n_anchor, replace=False)]     # no labels
    log(f"[enc] building {V}-view augmented bank of {n_anchor} anchors...", flush=True)
    views = [_patch_pool(_augment(anchors, np.random.default_rng(seed + 10 + v),
                                  hard=hard_aug), whiten) for v in range(V)]
    bank = np.stack(views, axis=1).reshape(n_anchor * V, RESP * RESP, KD)  # k*V+v
    if not evo_gpu.HAS_GPU:
        raise RuntimeError("this experiment expects the GPU backend")
    ef = ContrastiveEncoderGPU(bank, n_anchor, V, M, d, tau)

    MPD = M * PD
    params = {
        "K": (rng.standard_normal((pop, M, KD)) * 0.15).astype(np.float32),
        "kb": np.zeros((pop, M), np.float32),
        "act": rng.integers(0, 8, (pop, M)).astype(np.float32),
        "W": (rng.standard_normal((pop, MPD, d)) / np.sqrt(MPD)).astype(np.float32),
        "sigma": np.full(pop, 0.08, np.float32),
    }

    def ids():
        return np.round(params["act"]).astype(np.int64) % 8

    best = {"fit": -1e9, "acc": 0.0, "gen": 0, "champ": None}
    for gen in range(1, gens + 1):
        a = rng.choice(n_anchor, N, replace=False)
        vv = np.stack([rng.choice(V, 2, replace=False) for _ in range(N)])
        gidx = (a[:, None] * V + vv).reshape(-1)             # (2N,) interleaved
        fit, acc = ef(params["K"], params["kb"], ids(), params["W"], gidx)
        if gen >= int(0.8 * gens):
            params["sigma"] = np.minimum(params["sigma"], 0.04)
        j = int(np.argmax(fit))
        if float(fit[j]) > best["fit"]:
            best.update(fit=float(fit[j]), acc=float(acc[j]), gen=gen,
                        champ={"K": params["K"][j].copy(), "kb": params["kb"][j].copy(),
                               "act": ids()[j].copy(), "W": params["W"][j].copy(),
                               "M": M, "d": d})
        mp.ga_step(params, fit, rng, mag_scale=True)
        params["act"] = np.round(params["act"]) % 8
        if gen % log_every == 0 or gen == 1 or gen == gens:
            log(f"  gen {gen:5d}: NT-Xent={fit[j]:.4f} pos-retrieval={acc[j]:.4f} "
                f"(best retr {best['acc']:.4f} @ gen {best['gen']})", flush=True)

    c = best["champ"]
    # ---- EMERGENCE CHECK (labels touched for the FIRST time, eval only) ----
    log("\n[enc] --- emergence check (labels used only now, at eval) ---", flush=True)
    gsel = np.concatenate([np.where(ytr == cc)[0][:200] for cc in range(10)])
    qsel = np.concatenate([np.where(yte == cc)[0][:300] for cc in range(10)])
    cg = ef.encode(c, Xtr[gsel], whiten); yg = ytr[gsel]
    cq = ef.encode(c, Xte[qsel], whiten); yq = yte[qsel]
    knn = _knn_acc(cg, yg, cq, yq, k=10)

    # inter-class similarity in the private language (class centroids, cosine)
    cent = np.stack([cq[yq == cc].mean(0) for cc in range(10)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
    sim = cent @ cent.T
    log(f"  kNN top-1 (k=10)     : {knn:.4f}   (chance 0.10)", flush=True)
    log(f"  --> class geometry emerged label-free: {'YES' if knn > 0.15 else 'weak'}",
        flush=True)
    log(f"  nearest class for each (in the encoder's private language):", flush=True)
    for cc in range(10):
        order = np.argsort(-sim[cc])
        nn = [LABELS[o] for o in order if o != cc][:3]
        log(f"    {LABELS[cc]:>6} ~ {', '.join(nn)}", flush=True)

    payload = {"champ": c, "M": M, "d": d, "whiten": whiten, "knn": knn,
               "sim_matrix": sim, "pos_retrieval": best["acc"], "gen": best["gen"],
               "seed": seed, "hard_aug": hard_aug}
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    log(f"\n[enc] saved -> {out}", flush=True)
    return payload


# ==========================================================================
# FREEZE-AND-STACK: layer 2 of the contrastive encoder. Layer 1 (a proven,
# frozen encoder's conv filters) is infrastructure. We take its ACTIVATED
# feature maps BEFORE the collapsing pool (C1 channels, spatially downsampled
# to H1xH1) and evolve a SECOND conv layer on top of them — filters that read
# edges-of-edges, textures-of-textures — then project to a code under the SAME
# contrastive objective, still zero labels. The frozen layer-1 maps are
# precomputed once (the whole point of grow-don't-train). This is what v5's
# joint-from-random 2-layer genome could not do (GENREG rules VI/X).
# ==========================================================================
OUT_L2 = os.path.join(cp.ROOT, "demo", "cifar_encoder_l2.pkl")


def _frozen_l1_maps(champ1, X, out_hw=14, whiten=True, chunk=None):
    """Frozen layer-1 activated feature maps for images X -> (n, C1, hw, hw)."""
    bank1 = {"K": champ1["K"], "b": champ1["kb"], "act": champ1["act"]}

    def pf(Xc):
        P = cp._im2col5c(Xc).reshape(-1, KD)
        if whiten:
            P = cp._whiten_patches(P)
        return P.reshape(len(Xc), RESP * RESP, KD)

    return evo_gpu.l1_maps(pf, X, bank1, RESP, chunk=chunk, out_hw=out_hw)


class ContrastiveL2GPU:
    """NT-Xent over layer-2 genomes reading FROZEN layer-1 maps. L1 maps
    (K*V, C1, H1, H1) uploaded once; layer-1 is shared so its 3x3 unfold is
    computed once per batch, not per genome."""

    def __init__(self, L1maps, K, V, L2, d, tau=0.2, pools=POOLS, l1codes=None):
        import torch
        self.torch = torch
        self.K, self.V, self.L2, self.d, self.tau = K, V, L2, d, tau
        self.pools = pools
        self.PD = sum(r * c for r, c in pools)
        self.NV, self.C1, self.H1, _ = L1maps.shape
        self.R2 = self.H1 - 2
        self.L1 = evo_gpu.to_dev(L1maps.astype(np.float32))
        self.l1codes = evo_gpu.to_dev(l1codes.astype(np.float32)) \
            if l1codes is not None else None       # (NV, d1) frozen L1 codes
        self.shuf = None                           # (K, C1, H1, H1) shuffled-layout maps

    def set_shuf(self, shufmaps):
        self.shuf = evo_gpu.to_dev(shufmaps.astype(np.float32))

    def arrange(self, K2, kb2, act2, W2, gidx, a):
        """Mode #arrange: real-vs-shuffled layout discriminator. Positives = two
        augmentations of the real image; each anchor's QUADRANT-SHUFFLED self is
        added as a hard negative (identical local content, wrong arrangement).
        To push it away, the code must encode part LAYOUT, not a bag of features."""
        torch = self.torch
        with torch.no_grad():
            pos = self._codes(gidx, K2, kb2, act2, W2)          # (P,2N,d)
            shuf = self._emt(self.shuf[evo_gpu.to_dev(a.astype(np.int64))],
                             K2, kb2, act2, W2)                 # (P,N,d)
            c = torch.cat([pos, shuf], dim=1)                   # (P,2N+N,d)
            P, tot, _ = c.shape
            npos = pos.shape[1]
            S = torch.einsum("pid,pjd->pij", c, c) / self.tau
            S = S.masked_fill(torch.eye(tot, device=evo_gpu.DEV).bool()[None], -1e9)
            tgt = torch.arange(npos, device=evo_gpu.DEV) ^ 1    # partners (first 2N)
            logp = torch.log_softmax(S[:, :npos], dim=2)        # score only real rows
            fit = logp.gather(2, tgt[None, :, None].expand(P, npos, 1))[..., 0].mean(1)
            acc = (S[:, :npos].argmax(2) == tgt[None]).float().mean(1)
            return fit.cpu().numpy(), acc.cpu().numpy()

    def _ntxent(self, c):
        """c (P,B,d) unit codes; positive of 2i is 2i+1 -> (fit(P,), acc(P,))."""
        torch = self.torch
        P, B, _ = c.shape
        S = torch.einsum("pid,pjd->pij", c, c) / self.tau
        S = S.masked_fill(torch.eye(B, device=evo_gpu.DEV).bool()[None], -1e9)
        tgt = torch.arange(B, device=evo_gpu.DEV) ^ 1
        logp = torch.log_softmax(S, dim=2)
        fit = logp.gather(2, tgt[None, :, None].expand(P, B, 1))[..., 0].mean(1)
        acc = (S.argmax(2) == tgt[None]).float().mean(1)
        return fit, acc

    def _codes(self, idx, K2, kb2, act2, W2):
        return self._emt(self.L1[evo_gpu.to_dev(idx.astype(np.int64))],
                         K2, kb2, act2, W2)

    def _emt(self, maps, K2, kb2, act2, W2):
        """Encode an explicit maps tensor (B,C1,H1,H1) -> (P,B,d) unit codes."""
        torch = self.torch
        F = torch.nn.functional
        B, P, L2 = maps.shape[0], K2.shape[0], self.L2
        u = F.unfold(maps, 3)                                    # (B,9*C1,R2*R2)
        z = torch.einsum("bfk,plf->pblk", u, evo_gpu.to_dev(K2)) \
            + evo_gpu.to_dev(kb2.astype(np.float32))[:, None, :, None]
        z = z.reshape(P, B, L2, self.R2, self.R2)
        pooled = torch.zeros((P, B, L2, self.PD), device=evo_gpu.DEV)
        for a in range(8):
            mask = (act2 == a)
            if not mask.any():
                continue
            blk = evo_gpu._acts_t(z, a).reshape(P * B * L2, self.R2, self.R2)
            pl = evo_gpu._pool_t(blk, self.R2, self.pools).reshape(P, B, L2, self.PD)
            mm = evo_gpu.to_dev(mask.astype(np.float32))[:, None, :, None]
            pooled = pooled + pl * mm
        feat = pooled.reshape(P, B, L2 * self.PD)
        code = torch.einsum("pbf,pfd->pbd", feat, evo_gpu.to_dev(W2))
        return code / (code.norm(dim=-1, keepdim=True) + 1e-8)

    def __call__(self, K2, kb2, act2, W2, gidx):
        """Plain NT-Xent (modes #2 different-invariance and #3 hard-negative use
        this; the batch composition differs, not the fitness)."""
        torch = self.torch
        with torch.no_grad():
            fit, acc = self._ntxent(self._codes(gidx, K2, kb2, act2, W2))
            return fit.cpu().numpy(), acc.cpu().numpy()

    def decorr(self, K2, kb2, act2, W2, gidx, lam=2.0):
        """Mode #1: NT-Xent minus redundancy with the frozen L1 code. L2 is
        rewarded for being augment-invariant AND statistically independent of
        L1 (Barlow-style squared cross-correlation) -> it encodes the residual."""
        torch = self.torch
        with torch.no_grad():
            c = self._codes(gidx, K2, kb2, act2, W2)            # (P,B,d)
            fit, acc = self._ntxent(c)
            l1 = self.l1codes[evo_gpu.to_dev(gidx.astype(np.int64))]   # (B,d1)
            l1 = (l1 - l1.mean(0)) / (l1.std(0) + 1e-6)
            z = (c - c.mean(1, keepdim=True)) / (c.std(1, keepdim=True) + 1e-6)
            B = c.shape[1]
            C = torch.einsum("pbd,be->pde", z, l1) / B          # (P,d,d1)
            redundancy = (C * C).mean(dim=(1, 2))               # (P,)
            return (fit - lam * redundancy).cpu().numpy(), acc.cpu().numpy()

    def swav(self, K2, kb2, act2, W2, protos, gidx, beta=0.5, temp=0.1):
        """Mode #4: online clustering. Genome also carries K prototypes; the two
        views of an image must agree on their soft cluster assignment AND the
        batch must spread over clusters (equipartition) -> fine discrete
        structure. Returns (fitness, view-agreement acc)."""
        torch = self.torch
        with torch.no_grad():
            c = self._codes(gidx, K2, kb2, act2, W2)            # (P,B,d)
            pr = evo_gpu.to_dev(protos)                         # (P,Kc,d)
            pr = pr / (pr.norm(dim=-1, keepdim=True) + 1e-8)
            sc = torch.einsum("pbd,pkd->pbk", c, pr) / temp
            q = torch.softmax(sc, dim=2)                        # (P,B,Kc)
            qa, qb = q[:, 0::2], q[:, 1::2]                     # partner views
            agree = torch.sqrt(qa * qb + 1e-12).sum(2).mean(1)  # Bhattacharyya
            mbar = q.mean(1)                                    # (P,Kc)
            bal = -(mbar * torch.log(mbar + 1e-12)).sum(1) / np.log(q.shape[2])
            fit = agree + beta * bal
            acc = (qa.argmax(2) == qb.argmax(2)).float().mean(1)
            return fit.cpu().numpy(), acc.cpu().numpy()

    def encode_maps(self, champ2, L1maps, chunk=500):
        """Champion layer-2 codes over precomputed L1 maps -> (n,d)."""
        torch = self.torch
        saved = self.L1
        self.L1 = evo_gpu.to_dev(L1maps.astype(np.float32))
        K2, kb2, act2, W2 = (champ2["K2"][None], champ2["kb2"][None],
                             champ2["act2"][None], champ2["W2"][None])
        out = []
        with torch.no_grad():
            for lo in range(0, len(L1maps), chunk):
                idx = np.arange(lo, min(lo + chunk, len(L1maps)))
                out.append(self._codes(idx, K2, kb2, act2, W2)[0].cpu().numpy())
        self.L1 = saved
        return np.concatenate(out)


def evolve_encoder_l2(encoder_pkl="cifar_encoder_seed7.pkl", L2=32, d=16, pop=48,
                      gens=2500, n_anchor=1500, V=4, N=64, tau=0.2, out_hw=14,
                      whiten=True, seed=7, log_every=100, log=print):
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    enc1 = pickle.load(open(os.path.join(cp.ROOT, "demo", encoder_pkl), "rb"))
    c1 = enc1["champ"]
    C1 = c1["K"].shape[0]
    log(f"[l2] FREEZE-AND-STACK on {encoder_pkl} (L1 kNN {enc1.get('knn'):.4f}). "
        f"Layer2: L2={L2} filters on {C1}x{out_hw}x{out_hw} frozen maps -> d={d} "
        f"code. pop={pop} gens={gens} seed={seed} (LABEL-FREE)", flush=True)
    if whiten:
        cp.build_zca()
    rng = np.random.default_rng(seed)
    anchors = Xtr[rng.choice(len(Xtr), n_anchor, replace=False)]
    log(f"[l2] precomputing frozen layer-1 maps for {V}-view bank "
        f"({n_anchor} anchors)...", flush=True)
    vmaps = [_frozen_l1_maps(c1, _augment(anchors, np.random.default_rng(seed + 10 + v)),
                             out_hw, whiten) for v in range(V)]
    bank = np.stack(vmaps, axis=1).reshape(n_anchor * V, C1, out_hw, out_hw)
    if not evo_gpu.HAS_GPU:
        raise RuntimeError("this experiment expects the GPU backend")
    ef = ContrastiveL2GPU(bank, n_anchor, V, L2, d, tau)

    FIN = 9 * C1                                             # 3x3 x C1
    L2PD = L2 * PD
    params = {
        "K2": (rng.standard_normal((pop, L2, FIN)) / np.sqrt(FIN)).astype(np.float32),
        "kb2": np.zeros((pop, L2), np.float32),
        "act2": rng.integers(0, 8, (pop, L2)).astype(np.float32),
        "W2": (rng.standard_normal((pop, L2PD, d)) / np.sqrt(L2PD)).astype(np.float32),
        "sigma": np.full(pop, 0.08, np.float32),
    }

    def ids():
        return np.round(params["act2"]).astype(np.int64) % 8

    best = {"fit": -1e9, "acc": 0.0, "gen": 0, "champ": None}
    for gen in range(1, gens + 1):
        a = rng.choice(n_anchor, N, replace=False)
        vv = np.stack([rng.choice(V, 2, replace=False) for _ in range(N)])
        gidx = (a[:, None] * V + vv).reshape(-1)
        fit, acc = ef(params["K2"], params["kb2"], ids(), params["W2"], gidx)
        if gen >= int(0.8 * gens):
            params["sigma"] = np.minimum(params["sigma"], 0.04)
        j = int(np.argmax(fit))
        if float(fit[j]) > best["fit"]:
            best.update(fit=float(fit[j]), acc=float(acc[j]), gen=gen,
                        champ={"K2": params["K2"][j].copy(), "kb2": params["kb2"][j].copy(),
                               "act2": ids()[j].copy(), "W2": params["W2"][j].copy(),
                               "L2": L2, "d": d, "out_hw": out_hw, "C1": C1})
        mp.ga_step(params, fit, rng, mag_scale=True)
        params["act2"] = np.round(params["act2"]) % 8
        if gen % log_every == 0 or gen == 1 or gen == gens:
            log(f"  gen {gen:5d}: NT-Xent={fit[j]:.4f} pos-retrieval={acc[j]:.4f} "
                f"(best retr {best['acc']:.4f} @ gen {best['gen']})", flush=True)

    c2 = best["champ"]
    log("\n[l2] --- emergence check (labels used only now) ---", flush=True)
    gsel = np.concatenate([np.where(ytr == cc)[0][:200] for cc in range(10)])
    qsel = np.concatenate([np.where(yte == cc)[0][:300] for cc in range(10)])
    mg = _frozen_l1_maps(c1, Xtr[gsel], out_hw, whiten)
    mq = _frozen_l1_maps(c1, Xte[qsel], out_hw, whiten)
    cg, yg = ef.encode_maps(c2, mg), ytr[gsel]
    cq, yq = ef.encode_maps(c2, mq), yte[qsel]
    knn = _knn_acc(cg, yg, cq, yq, 10)
    cent = np.stack([cq[yq == cc].mean(0) for cc in range(10)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
    sim = cent @ cent.T
    VEH, ANI = {0, 1, 8, 9}, {2, 3, 4, 5, 6, 7}
    clean = sum(all(o in (VEH if k in VEH else ANI)
                    for o in [o for o in np.argsort(-sim[k]) if o != k][:3])
                for k in range(10))
    log(f"  L1 (frozen) kNN     : {enc1.get('knn'):.4f}", flush=True)
    log(f"  L2 (stacked)  kNN   : {knn:.4f}   (chance 0.10)  {'RAISED' if knn > enc1.get('knn') else 'no gain'} the ceiling",
        flush=True)
    log(f"  supergroup-clean    : {clean}/10", flush=True)
    # fine-grained: intra-animal separations (the thing L1 could not resolve)
    fine = [(2, 4, "bird/deer"), (3, 5, "cat/dog"), (4, 7, "deer/horse"),
            (2, 6, "bird/frog"), (5, 7, "dog/horse")]
    log(f"  fine-grained animal similarity (lower=better separated):", flush=True)
    for i, jj, nm in fine:
        log(f"    {nm:>10}: {sim[i, jj]:.3f}", flush=True)
    log(f"  nearest class per class (private language, layer 2):", flush=True)
    for cc in range(10):
        nn = [LABELS[o] for o in np.argsort(-sim[cc]) if o != cc][:3]
        log(f"    {LABELS[cc]:>6} ~ {', '.join(nn)}", flush=True)
    payload = {"champ": c2, "encoder1": encoder_pkl, "L2": L2, "d": d,
               "out_hw": out_hw, "knn": knn, "knn_l1": enc1.get("knn"),
               "sim_matrix": sim, "pos_retrieval": best["acc"], "seed": seed}
    with open(OUT_L2, "wb") as f:
        pickle.dump(payload, f)
    log(f"\n[l2] saved -> {OUT_L2}", flush=True)
    return payload


def _shuffle_quadrants(X, rng):
    """Permute the four 16x16 quadrants of each image (same local content, wrong
    global arrangement). Guarantees a non-identity permutation per image."""
    Y = X.copy()
    qs = [(slice(0, 16), slice(0, 16)), (slice(0, 16), slice(16, 32)),
          (slice(16, 32), slice(0, 16)), (slice(16, 32), slice(16, 32))]
    orig = [X[:, a, b] for a, b in qs]
    for i in range(len(X)):
        p = rng.permutation(4)
        while (p == np.arange(4)).all():
            p = rng.permutation(4)
        for k, (a, b) in enumerate(qs):
            Y[i, a, b] = orig[p[k]][i]
    return Y


def _patch_swap_fake(X, rng):
    """'Real-vs-fake' corruption that is NATURAL, not systematic: paste a region
    from a different image, FEATHERED (cosine-window alpha) so there is no seam
    to detect. The pixels are all real; only the CONTENT is out of place. To
    flag it the encoder must know what belongs where -- coherence, not an edge."""
    N = len(X)
    Y = X.copy()
    donor = rng.permutation(N)
    same = donor == np.arange(N)
    donor[same] = (donor[same] + 1) % N
    for i in range(N):
        h, w = int(rng.integers(10, 18)), int(rng.integers(10, 18))
        ty, tx = int(rng.integers(0, 33 - h)), int(rng.integers(0, 33 - w))
        my = np.cos(np.linspace(-1, 1, h) * (np.pi / 2)) ** 2
        mx = np.cos(np.linspace(-1, 1, w) * (np.pi / 2)) ** 2
        alpha = (0.95 * np.outer(my, mx))[..., None].astype(np.float32)
        real = Y[i, ty:ty + h, tx:tx + w]
        don = X[donor[i], ty:ty + h, tx:tx + w]
        Y[i, ty:ty + h, tx:tx + w] = real * (1 - alpha) + don * alpha
    return np.clip(Y, 0.0, 1.0)


def _concat_knn(cg1, cg2, yg, cq1, cq2, yq, k=10):
    """kNN on the L1||L2 concatenation (each block L2-normalised first)."""
    def unit(A):
        return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    g = np.concatenate([unit(cg1), unit(cg2)], axis=1)
    q = np.concatenate([unit(cq1), unit(cq2)], axis=1)
    return _knn_acc(g, yg, q, yq, k)


def evolve_l2(mode="decorr", encoder_pkl="cifar_encoder_seed7.pkl", L2=32, d=16,
              pop=48, gens=2500, n_anchor=1500, V=4, N=64, tau=0.2, out_hw=14,
              lam=2.0, Kc=50, whiten=True, seed=7, log_every=100, log=print):
    """Stack a layer-2 encoder on the FROZEN L1 with a DIFFERENT fitness than L1
    (the point: L2 must encode what L1 does not). modes:
    #1 decorr  - NT-Xent minus redundancy with L1 code (encode the residual)
    #3 hardneg - NT-Xent on batches = L1 nearest-neighbourhoods (split confusions)
    #2 color   - NT-Xent but positives vary in COLOUR (learn shape not colour)
    #4 swav    - online clustering + equipartition (fine discrete structure)"""
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    enc1 = pickle.load(open(os.path.join(cp.ROOT, "demo", encoder_pkl), "rb"))
    c1 = enc1["champ"]; C1 = c1["K"].shape[0]; d1 = enc1["d"]
    log(f"=== L2 mode={mode} on FROZEN {encoder_pkl} (L1 kNN {enc1['knn']:.4f}) "
        f"L2={L2} d={d} pop={pop} gens={gens} (LABEL-FREE) ===", flush=True)
    if whiten:
        cp.build_zca()
    rng = np.random.default_rng(seed)
    anchors = Xtr[rng.choice(len(Xtr), n_anchor, replace=False)]
    log(f"[l2:{mode}] precomputing frozen L1 maps + codes for {V}-view bank...",
        flush=True)
    AUGQ = {"color", "crop", "occlude", "warp"}          # augmentation-question modes
    augkind = mode if mode in AUGQ else "standard"
    view_imgs = [_augment(anchors, np.random.default_rng(seed + 10 + v),
                          kind=augkind) for v in range(V)]
    vmaps = [_frozen_l1_maps(c1, vi, out_hw, whiten) for vi in view_imgs]
    bank = np.stack(vmaps, axis=1).reshape(n_anchor * V, C1, out_hw, out_hw)
    l1codes = None
    if mode in ("decorr", "hardneg"):
        vcodes = [_encode_set(enc1, vi, whiten) for vi in view_imgs]
        l1codes = np.stack(vcodes, axis=1).reshape(n_anchor * V, d1)
    if not evo_gpu.HAS_GPU:
        raise RuntimeError("this experiment expects the GPU backend")
    ef = ContrastiveL2GPU(bank, n_anchor, V, L2, d, tau, l1codes=l1codes)

    nbr = None
    if mode == "hardneg":                                   # anchor L1 neighbours
        ac = np.stack([_encode_set(enc1, vi, whiten) for vi in view_imgs]).mean(0)
        ac /= np.linalg.norm(ac, axis=1, keepdims=True) + 1e-8
        nbr = np.argsort(-(ac @ ac.T), axis=1)[:, :N]       # (n_anchor, N) incl self
    if mode in ("arrange", "realfake"):                     # hard-negative fakes
        if mode == "arrange":
            fakes = _shuffle_quadrants(anchors, np.random.default_rng(seed + 77))
        else:
            fakes = _patch_swap_fake(anchors, np.random.default_rng(seed + 77))
        ef.set_shuf(_frozen_l1_maps(c1, fakes, out_hw, whiten))

    FIN = 9 * C1; L2PD = L2 * PD
    params = {
        "K2": (rng.standard_normal((pop, L2, FIN)) / np.sqrt(FIN)).astype(np.float32),
        "kb2": np.zeros((pop, L2), np.float32),
        "act2": rng.integers(0, 8, (pop, L2)).astype(np.float32),
        "W2": (rng.standard_normal((pop, L2PD, d)) / np.sqrt(L2PD)).astype(np.float32),
        "sigma": np.full(pop, 0.08, np.float32),
    }
    if mode == "swav":
        params["protos"] = (rng.standard_normal((pop, Kc, d))).astype(np.float32)

    def ids():
        return np.round(params["act2"]).astype(np.int64) % 8

    best = {"fit": -1e9, "acc": 0.0, "gen": 0, "champ": None}
    for gen in range(1, gens + 1):
        if mode == "hardneg":
            a = nbr[rng.integers(n_anchor)]                 # one L1 neighbourhood
        else:
            a = rng.choice(n_anchor, N, replace=False)
        vv = np.stack([rng.choice(V, 2, replace=False) for _ in range(len(a))])
        gidx = (a[:, None] * V + vv).reshape(-1)
        if mode == "decorr":
            fit, acc = ef.decorr(params["K2"], params["kb2"], ids(), params["W2"],
                                 gidx, lam=lam)
        elif mode == "swav":
            fit, acc = ef.swav(params["K2"], params["kb2"], ids(), params["W2"],
                               params["protos"], gidx)
        elif mode in ("arrange", "realfake"):
            fit, acc = ef.arrange(params["K2"], params["kb2"], ids(), params["W2"],
                                  gidx, a)
        else:                                               # color / hardneg / crop / ...
            fit, acc = ef(params["K2"], params["kb2"], ids(), params["W2"], gidx)
        if gen >= int(0.8 * gens):
            params["sigma"] = np.minimum(params["sigma"], 0.04)
        j = int(np.argmax(fit))
        if float(fit[j]) > best["fit"]:
            ch = {"K2": params["K2"][j].copy(), "kb2": params["kb2"][j].copy(),
                  "act2": ids()[j].copy(), "W2": params["W2"][j].copy(),
                  "L2": L2, "d": d, "out_hw": out_hw, "C1": C1}
            best.update(fit=float(fit[j]), acc=float(acc[j]), gen=gen, champ=ch)
        mp.ga_step(params, fit, rng, mag_scale=True)
        params["act2"] = np.round(params["act2"]) % 8
        if gen % log_every == 0 or gen == 1 or gen == gens:
            log(f"  gen {gen:5d}: fit={fit[j]:.4f} agree/retr={acc[j]:.4f} "
                f"(best {best['acc']:.4f} @ gen {best['gen']})", flush=True)

    c2 = best["champ"]
    # ---- eval: L2 alone AND the L1||L2 concatenation, vs L1's ceiling ----
    gsel = np.concatenate([np.where(ytr == cc)[0][:200] for cc in range(10)])
    qsel = np.concatenate([np.where(yte == cc)[0][:300] for cc in range(10)])
    yg, yq = ytr[gsel], yte[qsel]
    mg = _frozen_l1_maps(c1, Xtr[gsel], out_hw, whiten)
    mq = _frozen_l1_maps(c1, Xte[qsel], out_hw, whiten)
    cg2, cq2 = ef.encode_maps(c2, mg), ef.encode_maps(c2, mq)
    cg1, cq1 = _encode_set(enc1, Xtr[gsel], whiten), _encode_set(enc1, Xte[qsel], whiten)
    knn_l2 = _knn_acc(cg2, yg, cq2, yq, 10)
    knn_cat = _concat_knn(cg1, cg2, yg, cq1, cq2, yq, 10)
    cent = np.stack([cq2[yq == cc].mean(0) for cc in range(10)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8
    sim = cent @ cent.T
    log("", flush=True)
    log(f"=== RESULT L2 mode={mode} ===", flush=True)
    log(f"  L1 alone kNN         : {enc1['knn']:.4f}", flush=True)
    log(f"  L2 alone kNN         : {knn_l2:.4f}", flush=True)
    log(f"  L1||L2 concat kNN    : {knn_cat:.4f}   "
        f"{'RAISED CEILING (+%.3f)' % (knn_cat - enc1['knn']) if knn_cat > enc1['knn'] else 'no gain'}",
        flush=True)
    log(f"  fine pairs (lower=separated): cat/dog {sim[3,5]:.3f} bird/deer "
        f"{sim[2,4]:.3f} deer/horse {sim[4,7]:.3f}", flush=True)
    payload = {"mode": mode, "champ": c2, "encoder1": encoder_pkl, "L2": L2, "d": d,
               "out_hw": out_hw, "knn_l2": knn_l2, "knn_concat": knn_cat,
               "knn_l1": enc1["knn"], "sim_matrix": sim, "seed": seed}
    outp = os.path.join(cp.ROOT, "demo", f"cifar_l2_{mode}.pkl")
    with open(outp, "wb") as f:
        pickle.dump(payload, f)
    log(f"[l2:{mode}] saved -> {outp}", flush=True)
    return payload


# --------------------------------------------------------------------------
# READ-HEAD on a FROZEN encoder. The encoder is infrastructure now -- a retina
# whose weights are done. We encode every image through it ONCE (its output is
# fixed) and evolve a small read-head genome on the cached 16-d codes: code ->
# 10-class decision, soft cross-entropy fitness on LABELED data. This is the
# first time labels enter the pipeline at all. hidden=0 => pure linear probe
# (the canonical SSL measure); hidden>0 => evolved MLP head (8-catalog acts).
# --------------------------------------------------------------------------
OUT_HEAD = os.path.join(cp.ROOT, "demo", "cifar_readhead.pkl")


def _encode_set(payload, X, whiten=True):
    dummy = np.zeros((2, RESP * RESP, KD), np.float32)
    ef = ContrastiveEncoderGPU(dummy, 1, 2, payload["M"], payload["d"])
    return ef.encode(payload["champ"], X, whiten)


def _head_logits(C, g, act_ids, hidden):
    """C (B,dc) codes -> (P,B,10) logits. Pure numpy (codes are tiny)."""
    if hidden == 0:
        return np.einsum("bf,pcf->pbc", C, g["W"]) + g["b"][:, None, :]
    z1 = np.einsum("bf,phf->pbh", C, g["W1"]) + g["b1"][:, None, :]
    h = np.zeros_like(z1)
    for a in range(8):
        m = act_ids == a
        if m.any():
            h += np.where(m[:, None, :], mp._acts(z1, a), 0.0)
    return np.einsum("pbh,pch->pbc", h, g["W2"]) + g["b2"][:, None, :]


def evolve_readhead(encoder_pkl="cifar_encoder_seed7.pkl", hidden=64, H=64,
                    pop=160, gens=1500, minibatch=400, n_head=1500, whiten=True,
                    seed=13, log_every=150, log=print):
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    enc = pickle.load(open(os.path.join(cp.ROOT, "demo", encoder_pkl), "rb"))
    dc = enc["d"]
    kind = "linear probe" if hidden == 0 else f"evolved MLP head (H={H})"
    log(f"[head] FROZEN encoder {encoder_pkl} (d={dc}, kNN={enc.get('knn'):.4f}) "
        f"+ {kind}  pop={pop} gens={gens}", flush=True)
    log(f"[head] encoding images through the frozen retina (one-time)...", flush=True)
    tr = np.concatenate([np.where(ytr == c)[0][:n_head] for c in range(10)])
    Ctr, ytr2 = _encode_set(enc, Xtr[tr], whiten), ytr[tr]
    Cva, yva2 = _encode_set(enc, Xva, whiten), yva
    Cte, yte2 = _encode_set(enc, Xte, whiten), yte
    log(f"[head] codes: train {Ctr.shape}, val {Cva.shape}, test {Cte.shape}",
        flush=True)

    rng = np.random.default_rng(seed)
    if hidden == 0:
        params = {"W": (rng.standard_normal((pop, 10, dc)) / np.sqrt(dc)).astype(np.float32),
                  "b": np.zeros((pop, 10), np.float32),
                  "sigma": np.full(pop, 0.08, np.float32)}
    else:
        params = {"W1": (rng.standard_normal((pop, H, dc)) / np.sqrt(dc)).astype(np.float32),
                  "b1": np.zeros((pop, H), np.float32),
                  "act": rng.integers(0, 8, (pop, H)).astype(np.float32),
                  "W2": (rng.standard_normal((pop, 10, H)) / np.sqrt(H)).astype(np.float32),
                  "b2": np.zeros((pop, 10), np.float32),
                  "sigma": np.full(pop, 0.08, np.float32)}
    cls_idx = [np.where(ytr2 == c)[0] for c in range(10)]
    per = minibatch // 10

    def aids():
        return np.round(params["act"]).astype(np.int64) % 8 if hidden else None

    def fit_on(C, y, ai):
        z = _head_logits(C, params, ai, hidden)              # (P,B,10)
        z = z - z.max(-1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        P, B = z.shape[0], z.shape[1]
        ll = logp[np.arange(P)[:, None], np.arange(B)[None], y[None]].mean(1)
        acc = (z.argmax(-1) == y[None]).mean(1)
        return ll, acc

    best = {"fit": -1e9, "acc": 0.0, "gen": 0, "champ": None}
    for gen in range(1, gens + 1):
        bi = np.concatenate([rng.choice(cls_idx[c], per) for c in range(10)])
        ai = aids()
        fit, _ = fit_on(Ctr[bi], ytr2[bi], ai)
        if gen >= int(0.8 * gens):
            params["sigma"] = np.minimum(params["sigma"], 0.04)
        mp.ga_step(params, fit, rng, mag_scale=True)
        if hidden:
            params["act"] = np.round(params["act"]) % 8
        if gen % log_every == 0 or gen == 1 or gen == gens:
            vfit, vacc = fit_on(Cva, yva2, aids())
            j = int(np.argmax(vfit))
            if float(vfit[j]) > best["fit"]:
                champ = {k: params[k][j].copy() for k in params if k != "sigma"}
                best.update(fit=float(vfit[j]), acc=float(vacc[j]), gen=gen,
                            champ=champ)
            log(f"  gen {gen:5d}: val_logprob={vfit[j]:.4f} val_top1={vacc[j]:.4f} "
                f"(best {best['acc']:.4f} @ gen {best['gen']})", flush=True)

    # test with champion (P=1)
    g1 = {k: v[None] for k, v in best["champ"].items()}
    ai1 = np.round(best["champ"]["act"]).astype(np.int64)[None] % 8 if hidden else None
    z = _head_logits(Cte, g1, ai1, hidden)[0]
    pred = z.argmax(-1)
    test_acc = float((pred == yte2).mean())
    conf = np.zeros((10, 10), np.int64)
    for t, p in zip(yte2, pred):
        conf[t, p] += 1
    per_cls = conf.diagonal() / conf.sum(1).clip(1)

    log("", flush=True)
    log(f"[head] RESULT -- {kind} on FROZEN label-free encoder", flush=True)
    log(f"  chance                : 0.1000", flush=True)
    log(f"  label-free kNN (ref)  : {enc.get('knn'):.4f}", flush=True)
    log(f"  val  top-1            : {best['acc']:.4f}", flush=True)
    log(f"  test top-1            : {test_acc:.4f}  (champion gen {best['gen']})",
        flush=True)
    log(f"  per-class test top-1  : " +
        ", ".join(f"{LABELS[c]} {per_cls[c]:.2f}" for c in range(10)), flush=True)
    payload = {"encoder": encoder_pkl, "hidden": hidden, "H": H,
               "champ": best["champ"], "test_acc": test_acc, "val_acc": best["acc"],
               "confusion": conf, "gen": best["gen"]}
    with open(OUT_HEAD, "wb") as f:
        pickle.dump(payload, f)
    log(f"[head] saved -> {OUT_HEAD}", flush=True)
    return payload


def evolve_arbiter2(pos=3, neg=5, L1=8, L2=8, seedA=7, seedB=101, dec_gens=1500,
                    checker_gens=1500, n_train=2000, n_check=2000, whiten=True,
                    log=print):
    """Same arbiter+checker as evolve_arbiter, but the two deciders are the
    TWO-LAYER (filters-of-filters) genome. Tests whether a deeper internal
    vocabulary lifts the deciders (and thus grows the checker's confident
    'agree' bucket)."""
    Xtr, ytr, Xva, yva, Xte, yte = cp.load_cifar()
    a_name, b_name = LABELS[pos], LABELS[neg]
    log(f"=== 2-LAYER ARBITER+CHECKER: {a_name}(+) vs {b_name}(-) ===", flush=True)

    log(f"\n--- 2-layer decider A (seed {seedA}) ---", flush=True)
    A = evolve_single2(pos, neg, L1=L1, L2=L2, gens=dec_gens, n_train=n_train,
                       whiten=whiten, seed=seedA, log=log)["champ"]
    log(f"\n--- 2-layer decider B (seed {seedB}) ---", flush=True)
    B = evolve_single2(pos, neg, L1=L1, L2=L2, gens=dec_gens, n_train=n_train,
                       whiten=whiten, seed=seedB, log=log)["champ"]

    def feats_for(X):
        return _checker_feats(champ_logits2(A, X, L1, L2, whiten),
                              champ_logits2(B, X, L1, L2, whiten))

    Xct, yct = _balanced(Xtr, ytr, pos, neg, n_check, seedA + 555)
    Xcv, ycv = _balanced(Xva, yva, pos, neg, 10 ** 9, seedA + 556)
    Xte2, yte2 = _balanced(Xte, yte, pos, neg, 10 ** 9, seedA + 557)
    Ftr, Fva, Fte = feats_for(Xct), feats_for(Xcv), feats_for(Xte2)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr, Fva, Fte = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd

    log(f"\n--- checker genome (verifier) ---", flush=True)
    C = evolve_checker(Ftr, yct, Fva, ycv, gens=checker_gens, log=log)

    sA = champ_logits2(A, Xte2, L1, L2, whiten)
    sB = champ_logits2(B, Xte2, L1, L2, whiten)
    accA = ((sA > 0) == (yte2 > 0.5)).mean()
    accB = ((sB > 0) == (yte2 > 0.5)).mean()
    acc_avg = (((1 / (1 + np.exp(-sA)) + 1 / (1 + np.exp(-sB))) / 2 > 0.5)
               == (yte2 > 0.5)).mean()
    ai = np.round(C["act"]).astype(np.int64) % 8
    zc = _checker_scores(Fte, {k: C[k][None] for k in ("W1", "b1", "w2", "b2")}, ai[None])[0]
    acc_chk = ((zc > 0) == (yte2 > 0.5)).mean()
    agree = (sA > 0) == (sB > 0); dis = ~agree
    acc_agree = (((sA > 0) == (yte2 > 0.5))[agree]).mean() if agree.any() else 0.0
    accA_dis = (((sA > 0) == (yte2 > 0.5))[dis]).mean() if dis.any() else 0.0
    acc_chk_dis = ((zc > 0) == (yte2 > 0.5))[dis].mean() if dis.any() else 0.0

    log("", flush=True)
    log(f"=== RESULT {a_name}(+) vs {b_name}(-) -- 2-LAYER arbiter + checker ===", flush=True)
    log(f"  majority baseline        : 0.5000", flush=True)
    log(f"  1-layer reference (v2)   : 0.6705", flush=True)
    log(f"  2-layer decider A (test) : {accA:.4f}", flush=True)
    log(f"  2-layer decider B (test) : {accB:.4f}", flush=True)
    log(f"  average ensemble (test)  : {acc_avg:.4f}", flush=True)
    log(f"  CHECKER genome (test)    : {acc_chk:.4f}  <- verified output", flush=True)
    log(f"  --- seed-consistency (verification) ---", flush=True)
    log(f"  seeds AGREE on           : {agree.mean() * 100:.1f}% of test images", flush=True)
    log(f"    acc where they agree   : {acc_agree:.4f}  (confident cases)", flush=True)
    log(f"    acc where they DISAGREE: A={accA_dis:.4f}  (answer changes w/ seed)", flush=True)
    log(f"    checker on disagree    : {acc_chk_dis:.4f}", flush=True)

    payload = {"pos": pos, "neg": neg, "L1": L1, "L2": L2, "deciderA": A,
               "deciderB": B, "checker": C, "feat_mu": mu, "feat_sd": sd,
               "labels": (a_name, b_name),
               "acc": {"A": float(accA), "B": float(accB), "avg": float(acc_avg),
                       "checker": float(acc_chk), "agree_frac": float(agree.mean()),
                       "acc_agree": float(acc_agree),
                       "checker_disagree": float(acc_chk_dis)}}
    with open(os.path.join(cp.ROOT, "demo", "cifar_arbiter2.pkl"), "wb") as f:
        pickle.dump(payload, f)
    log(f"\n[arbiter2] saved -> demo/cifar_arbiter2.pkl", flush=True)
    return payload


if __name__ == "__main__":
    evolve_single()
