"""genreg_trustmix — trust-weighted mixture of heterogeneous predictors, the
composition the A-series (A_66..A_70) validated. Channels:

    unigram · bigram · trigram · 4-gram   (exact count tables — carry COHERENCE)
    neural                                (the evolved recurrent model — generalization)

Combined P = Σ_c softmax(trust)_c · P_c(next | context). The `trust` vector is
EVOLVED (gradient-free) — that's the only thing trained here; the channels are
frozen. Docs finding: the n-gram channels supply the context-sensitive
distribution SHAPE a single neural head can't, restoring readable generation.
Not one model doing everything — many parts, gated by trust.
"""
import json
import os

import numpy as np

from .genreg_lm import (LMPopulation, DEFAULTS, load_char_corpus, sample_windows,
                        load_population, CHARS, RARE, V)

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "lm")
CHANNELS = ["uni", "bi", "tri", "4g", "5g", "6g", "7g", "neural"]
NG_ORDERS = {"uni": 0, "bi": 1, "tri": 2, "4g": 3, "5g": 4}   # dense: #prev chars
HASH_ORDERS = {"6g": 5, "7g": 6}          # hashed higher orders: #prev chars
HASH_BITS = 22                            # 4.2M-row hashed tables
ALPHA = 0.02          # add-alpha smoothing for the count tables


# --------------------------------------------------------------------------
# Exact n-gram probability tables (char-level, V small so dense is fine)
# --------------------------------------------------------------------------
def _table(seg, order):
    """Dense (V^order, V) count table for the given #prev-chars conditioned on.
    Returns (prob (rows,V), evidence (rows,)) — evidence = raw row total."""
    import torch
    rows = V ** order
    C = np.zeros((rows, V), np.float64)
    if order == 0:
        C[0] = np.bincount(seg, minlength=V)
    else:
        key = seg[:-order].astype(np.int64)
        for j in range(1, order):
            key = key * V + seg[j:-order + j]
        np.add.at(C, (key, seg[order:]), 1.0)
    ev = C.sum(1)                                       # evidence BEFORE smoothing
    Cs = C + ALPHA
    P = torch.tensor(Cs / Cs.sum(1, keepdims=True), dtype=torch.float32, device="cuda")
    E = torch.tensor(ev, dtype=torch.float32, device="cuda")
    return P, E


def _hash_key(cols, M):
    """Hash a stack of char-id columns (numpy int64 arrays) to [0,M)."""
    key = np.zeros(len(cols[0]), np.int64)
    for c in cols:
        key = (key * np.int64(1000003) + c) & np.int64(0x7FFFFFFFFFFFFFFF)
    return (key % M).astype(np.int64)


def _hashed_table(seg, nprev, M):
    """Hashed (M,V) count table for an n-gram of `nprev` context chars.
    Collisions add bounded noise (the docs' hash trick). Returns (prob, evid)."""
    import torch
    cols = [seg[j:len(seg) - nprev + j] for j in range(nprev)]
    key = _hash_key(cols, M)
    nxt = seg[nprev:]
    C = np.zeros((M, V), np.float32)
    np.add.at(C, (key, nxt), 1.0)
    ev = C.sum(1)
    Cs = C + ALPHA
    P = torch.tensor(Cs / Cs.sum(1, keepdims=True), dtype=torch.float32, device="cuda")
    E = torch.tensor(ev, dtype=torch.float32, device="cuda")
    return P, E


def build_ngrams(ids, n_train=3_000_000):
    """Dense tables orders 1..5 + hashed tables orders 6..7."""
    seg = ids[:n_train].astype(np.int64)
    tabs, evid = {}, {}
    for name, order in NG_ORDERS.items():
        tabs[name], evid[name] = _table(seg, order)
    tabs["uni"] = tabs["uni"].reshape(V)               # (1,V) -> (V,)
    M = 1 << HASH_BITS
    for name, nprev in HASH_ORDERS.items():
        tabs[name], evid[name] = _hashed_table(seg, nprev, M)
    tabs["_evid"] = evid
    tabs["_M"] = M
    return tabs


def _ctx_key(ctx, order):
    """ctx:(B,>=order) -> (B,) flat index into a V^order table."""
    if order == 0:
        import torch
        return torch.zeros(ctx.shape[0], dtype=torch.long, device=ctx.device)
    key = ctx[:, -order].long()
    for j in range(-order + 1, 0):
        key = key * V + ctx[:, j].long()
    return key


def _hash_key_t(ctx, nprev, M):
    """Torch hash of the last nprev columns of ctx:(B,>=nprev) -> (B,) in [0,M)."""
    import torch
    key = torch.zeros(ctx.shape[0], dtype=torch.long, device=ctx.device)
    for j in range(-nprev, 0):
        key = (key * 1000003 + ctx[:, j].long()) & 0x7FFFFFFFFFFFFFFF
    return key % M


def ngram_probs(tabs, ctx):
    """ctx:(B,>=6) recent char ids. Returns (dists {name:(B,V)}, evid {name:(B,)})."""
    import torch
    B = ctx.shape[0]; dists, ev = {}, {}
    for name, order in NG_ORDERS.items():
        if order == 0:
            dists[name] = tabs["uni"][None, :].expand(B, V)
            ev[name] = torch.full((B,), 1e9, device=ctx.device)   # unigram always sure
        else:
            k = _ctx_key(ctx, order)
            dists[name] = tabs[name][k]; ev[name] = tabs["_evid"][name][k]
    for name, nprev in HASH_ORDERS.items():
        k = _hash_key_t(ctx, nprev, tabs["_M"])
        dists[name] = tabs[name][k]; ev[name] = tabs["_evid"][name][k]
    return dists, ev


# --------------------------------------------------------------------------
# Neural channel: per-position next-char distributions over held-out windows
# --------------------------------------------------------------------------
def load_predictor(rid, device="cuda"):
    d = os.path.join(RUNS, rid)
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))["config"]
    pop = LMPopulation({**DEFAULTS, "pop": 400, "trigram": bool(cfg.get("trigram")),
                        "tie_readout": bool(cfg.get("tie_readout"))}, device)
    load_population(pop, os.path.join(d, "checkpoint.npz"))
    ids = load_char_corpus()
    W = sample_windows(ids, 256, 64, np.random.default_rng(31337))
    fit, t1, _ = pop.evaluate(W, 8)
    return pop, int(fit.argmax()), float(t1[int(fit.argmax())])


def _neural_step(pop, b, cur, prev, h):
    import torch
    t = torch
    e_c = pop.E[b:b + 1, cur, :][:, None].reshape(1, -1, pop.D)
    e_p = pop.E[b:b + 1, prev, :][:, None].reshape(1, -1, pop.D)
    x = t.cat([e_c, e_p, t.tanh(h)], dim=2)
    h = pop._act(t.einsum("pbi,pih->pbh", x, pop.W_in[b:b + 1]) + pop.b_h[b:b + 1, None, :], act=pop.act[b:b + 1])
    if pop.tie:
        z = t.einsum("pbh,phd->pbd", h, pop.W_po[b:b + 1])
        lg = (t.einsum("pbd,pvd->pbv", z, pop.E[b:b + 1]) + pop.b_out[b:b + 1, None, :])[0]
    else:
        lg = (t.einsum("pbh,phv->pbv", h, pop.W_out[b:b + 1]) + pop.b_out[b:b + 1, None, :])[0]
    if getattr(pop, "trigram", False):
        inter = t.einsum("bl,lv->bv", pop.E1[b, cur] * pop.E2[b, prev], pop.O_lr[b])
        lg = lg + pop.alpha_lr[b] * (pop.bigram_lr[b, cur] + inter)
    return lg, h


def collect(pop, b, tabs, windows, warmup=8):
    """Run neural + n-gram channels over held-out windows. Returns per-position
    channel prob tensors {name:(N,V)}, evidence {name:(N,)}, and targets (N,)."""
    import torch
    t = torch
    w = t.as_tensor(windows, device=pop.dev); B, T1 = w.shape; T = T1 - 1
    h = t.zeros(1, B, pop.H, device=pop.dev)
    chans = {c: [] for c in CHANNELS}; evs = {c: [] for c in CHANNELS}; tgt = []
    ONG = [c for c in CHANNELS if c != "neural"]
    with t.inference_mode():
        for i in range(T):
            cur = w[:, i]; prev = w[:, max(i - 1, 0)]
            lg, h = _neural_step(pop, b, cur, prev, h)
            if i < warmup:
                continue
            ctx = w[:, max(i - 6, 0):i + 1]
            if ctx.shape[1] < 7:                       # pad early positions
                ctx = t.cat([w[:, :1].expand(B, 7 - ctx.shape[1]), ctx], 1)
            ng, ev = ngram_probs(tabs, ctx)
            for c in ONG:
                chans[c].append(ng[c]); evs[c].append(ev[c])
            chans["neural"].append(lg.softmax(1))
            evs["neural"].append(t.full((B,), 1e9, device=pop.dev))
            tgt.append(w[:, i + 1])
    return ({c: t.cat(chans[c], 0) for c in CHANNELS},
            {c: t.cat(evs[c], 0) for c in CHANNELS}, t.cat(tgt, 0))


# --------------------------------------------------------------------------
# Evolve the trust vector (gradient-free ES). Fitness = mixture held-out top-1.
# --------------------------------------------------------------------------
def _gate(genome, P, Ev):
    """Context-conditional backoff gate (evolved, Witten-Bell style).
    genome = [trust(C) | logkappa(C)]. Per position, channel weight =
    softmax(trust)_c · confidence_c, confidence = evidence/(evidence+kappa_c).
    Returns mixed distribution (pop, N, V)."""
    import torch
    t = torch
    C = len(CHANNELS)
    trust = genome[:, :C].softmax(1)                     # (pop, C)
    kappa = genome[:, C:].exp()                          # (pop, C) > 0
    conf = Ev[None, :, :] / (Ev[None, :, :] + kappa[:, :, None])   # (pop, C, N)
    g = trust[:, :, None] * conf                         # (pop, C, N)
    g = g / g.sum(1, keepdim=True).clamp_min(1e-9)
    return t.einsum("pcn,cnv->pnv", g, P)


def evolve_trust(chans, evid, tgt, generations=600, pop=96, seed=1, log=print):
    """Evolve trust + per-channel backoff kappa (context-conditional gate)."""
    import torch
    t = torch
    P = t.stack([chans[c] for c in CHANNELS], 0)         # (C, N, V)
    Ev = t.stack([evid[c] for c in CHANNELS], 0)         # (C, N)
    y = tgt; C = len(CHANNELS)
    rng = np.random.default_rng(seed)
    # genome: trust(C) init ~0, logkappa(C) init log(5)
    W = t.tensor(np.hstack([rng.standard_normal((pop, C)) * 0.5,
                            np.full((pop, C), np.log(5.0)) + rng.standard_normal((pop, C)) * 0.3]),
                 dtype=torch.float32, device="cuda")
    def acc(w):
        mix = _gate(w, P, Ev)
        return (mix.argmax(2) == y[None, :]).float().mean(1)
    best = None
    for gen in range(generations):
        a = acc(W); top = int(a.argmax())
        if best is None or float(a[top]) > best[0]:
            best = (float(a[top]), W[top].clone())
        k = max(2, pop // 4)
        elite = W[a.topk(k).indices]
        children = elite[t.randint(0, k, (pop - k,), device="cuda")] + \
            t.randn(pop - k, 2 * C, device="cuda") * 0.15
        W = t.cat([elite, children], 0)
        if gen % 150 == 0 or gen == generations - 1:
            tr = best[1][:C].softmax(0)
            log(f"gate gen {gen:4d} mix top1 {best[0]*100:.2f}%  " +
                " ".join(f"{c}={float(tr[i]):.2f}" for i, c in enumerate(CHANNELS)))
    return best[1]                                       # full genome (trust+kappa)


# --------------------------------------------------------------------------
# Generation from the mixture
# --------------------------------------------------------------------------
def generate_mix(pop, b, tabs, genome, prompt, length=90, temperature=0.6,
                 seed=None, only=None):
    """Generate from the context-conditional gated mixture.
    only: None (full mix) | 'neural' | 'ngram' — for A/B."""
    import torch
    t = torch
    rng = np.random.default_rng(seed)
    C = len(CHANNELS)
    trust = genome[:C].clone(); kappa = genome[C:].exp()
    if only == "neural":
        m = t.full((C,), -1e9, device=pop.dev); m[CHANNELS.index("neural")] = 10.0; trust = m
    elif only == "ngram":
        trust = trust.clone(); trust[CHANNELS.index("neural")] = -1e9
    tr = trust.softmax(0)
    lut = {c: i for i, c in enumerate(CHARS)}
    seq = [lut.get(c, RARE) for c in prompt.lower()] or [0]
    n = len(seq)
    with t.inference_mode():
        h = t.zeros(1, 1, pop.H, device=pop.dev)
        for i in range(n + length - 1):
            cur = seq[i]; prev = seq[i - 1] if i > 0 else seq[0]
            lg, h = _neural_step(pop, b, t.tensor([cur], device=pop.dev),
                                 t.tensor([prev], device=pop.dev), h)
            if i >= n - 1:
                tail = seq[-7:] if len(seq) >= 7 else [seq[0]] * (7 - len(seq)) + seq
                ctx = t.tensor([tail[-7:]], device=pop.dev)
                ng, ev = ngram_probs(tabs, ctx)
                ngram_names = [c for c in CHANNELS if c != "neural"]
                dists = {**{c: ng[c][0] for c in ngram_names}, "neural": lg.softmax(1)[0]}
                evs = {**{c: ev[c][0] for c in ngram_names}, "neural": t.tensor(1e9, device=pop.dev)}
                conf = t.stack([evs[c] / (evs[c] + kappa[j]) for j, c in enumerate(CHANNELS)])
                g = tr * conf; g = g / g.sum().clamp_min(1e-9)
                mix = sum(g[j] * dists[c] for j, c in enumerate(CHANNELS))
                p = (mix.clamp_min(1e-9).log() / max(temperature, 1e-6)).softmax(0).cpu().numpy()
                seq.append(int(rng.choice(V, p=p / p.sum())))
    return "".join(CHARS[i] if i < len(CHARS) else "?" for i in seq[n:])
