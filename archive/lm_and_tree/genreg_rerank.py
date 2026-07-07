"""genreg_rerank — a SECOND evolved component: a coherence scorer (discriminator)
that reranks the predictor's proposals. The documented best-for-coherence method
(§VI rerank): the predictor proposes top-K next chars; the scorer judges which
candidate keeps the text looking like real corpus, and we pick that one.

Two components composed at decode time — generator + critic. Both evolved,
gradient-free.

Scorer: window of chars -> realness scalar. Fitness = distinguish REAL corpus
windows from the PREDICTOR's own sampled windows (soft log-prob of the correct
label). Energy homeostasis + tournament, same GENREG machinery.
"""
import glob
import json
import os

import numpy as np

from .genreg_lm import (LMPopulation, DEFAULTS, load_char_corpus, sample_windows,
                        load_population, CHARS, RARE, V, N_ACT)

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "lm")
SCW = 28           # scorer context window (chars) — phrase scale


# --------------------------------------------------------------------------
# Load a trained char predictor (the proposer) by run id, return (pop, champ).
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


def _step(pop, b, cur, prev, h):
    """One predictor step for a single genome. cur/prev int, h (1,1,H). -> (logits(V,), h)."""
    import torch
    t = torch
    e_c = pop.E[b:b + 1, cur:cur + 1, :]
    e_p = pop.E[b:b + 1, prev:prev + 1, :]
    x = t.cat([e_c, e_p, t.tanh(h)], dim=2)
    h2 = pop._act(t.einsum("pbi,pih->pbh", x, pop.W_in[b:b + 1]) + pop.b_h[b:b + 1, None, :],
                  act=pop.act[b:b + 1])
    if pop.tie:
        z = t.einsum("pbh,phd->pbd", h2, pop.W_po[b:b + 1])
        logits = (t.einsum("pbd,pvd->pbv", z, pop.E[b:b + 1]) + pop.b_out[b:b + 1, None, :])[0, 0]
    else:
        logits = (t.einsum("pbh,phv->pbv", h2, pop.W_out[b:b + 1]) + pop.b_out[b:b + 1, None, :])[0, 0]
    if getattr(pop, "trigram", False):
        inter = (pop.E1[b, cur] * pop.E2[b, prev]) @ pop.O_lr[b]
        logits = logits + pop.alpha_lr[b] * (pop.bigram_lr[b, cur] + inter)
    return logits, h2


def make_fakes(pop, b, ids, n, W, rng, temperature=0.8):
    """n length-W windows: half REAL corpus, half the PREDICTOR's sampled text
    (primed on a real prefix, then it generates, all n in parallel). Returns
    (windows (2n,W), labels)."""
    import torch
    t = torch
    real = sample_windows(ids, n, W - 1, rng)[:, :W]                 # (n, W) real
    prime = t.as_tensor(sample_windows(ids, n, W - 1, rng)[:, :W - 1], device=pop.dev)  # (n,W-1)
    P1 = pop.W_in[b:b + 1]; b1 = pop.b_h[b:b + 1]; act = pop.act[b:b + 1]
    with t.inference_mode():
        h = t.zeros(1, n, pop.H, device=pop.dev)                     # batch over n
        cur = prime[:, 0].clone(); prev = cur.clone()
        gen = W - 1                                                  # chars to generate
        outbuf = []
        L = prime.shape[1] + gen
        for i in range(L - 1):
            e_c = pop.E[b:b + 1, cur, :][:, None].reshape(1, n, pop.D)
            e_p = pop.E[b:b + 1, prev, :][:, None].reshape(1, n, pop.D)
            x = t.cat([e_c, e_p, t.tanh(h)], dim=2)
            h = pop._act(t.einsum("pbi,pih->pbh", x, P1) + b1[:, None, :], act=act)
            lg = (t.einsum("pbh,phv->pbv", h, pop.W_out[b:b + 1]) + pop.b_out[b:b + 1, None, :])[0] \
                if not pop.tie else None
            if pop.tie:
                z = t.einsum("pbh,phd->pbd", h, pop.W_po[b:b + 1])
                lg = (t.einsum("pbd,pvd->pbv", z, pop.E[b:b + 1]) + pop.b_out[b:b + 1, None, :])[0]
            if getattr(pop, "trigram", False):
                inter = t.einsum("bl,lv->bv", pop.E1[b, cur] * pop.E2[b, prev], pop.O_lr[b])
                lg = lg + pop.alpha_lr[b] * (pop.bigram_lr[b, cur] + inter)
            if i < prime.shape[1] - 1:                              # teacher-force the prime
                prev = cur; cur = prime[:, i + 1]
            else:                                                   # sample own output
                p = (lg / temperature).softmax(1)
                nxt = t.multinomial(p, 1).squeeze(1)
                outbuf.append(nxt); prev = cur; cur = nxt
    fakes_gen = t.stack(outbuf, 1).cpu().numpy()                    # (n, gen)
    fakes = np.hstack([prime[:, -1:].cpu().numpy(), fakes_gen])[:, :W]
    X = np.vstack([real, fakes])
    y = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)
    return X, y


# --------------------------------------------------------------------------
# Coherence scorer population (window -> realness). Own tiny embedding + MLP.
# --------------------------------------------------------------------------
class ScorerPop:
    def __init__(self, cfg, device):
        import torch
        self.t = torch
        self.dev = device
        P, D, Hs = cfg["pop"], cfg["sd"], cfg["sh"]
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        mk = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(device)
        self.E = mk(P, V, D, sc=0.1)
        self.W1 = mk(P, SCW * D, Hs, sc=1.0 / np.sqrt(SCW * D)); self.b1 = torch.zeros(P, Hs, device=device)
        self.a1 = torch.randint(0, N_ACT, (P, Hs), generator=g).to(device)
        self.wo = mk(P, Hs, sc=1.0 / np.sqrt(Hs)); self.bo = torch.zeros(P, device=device)
        self.mut_rate = torch.full((P,), 0.05, device=device)
        self.mut_scale = torch.full((P,), 0.05, device=device)
        self.energy = torch.full((P,), 1.0, device=device)
        self.age = torch.zeros(P, dtype=torch.long, device=device)
        self.fit_ema = torch.full((P,), float("nan"), device=device)
        self.P, self.D, self.Hs = P, D, Hs

    _T = ("E", "W1", "b1", "a1", "wo", "bo", "mut_rate", "mut_scale")

    def _act(self, pre):
        t = self.t; a = self.a1[:, None, :].expand_as(pre)
        o = t.tanh(pre)
        o = t.where(a == 1, t.sigmoid(pre), o); o = t.where(a == 2, t.relu(pre).clamp(max=4), o)
        o = t.where(a == 3, t.sin(pre), o); o = t.where(a == 4, t.exp(-pre * pre), o)
        o = t.where(a == 5, pre.clamp(-4, 4), o); o = t.where(a == 6, pre / (1 + pre.abs()), o)
        o = t.where(a == 7, pre.abs().clamp(max=4), o)
        return o

    def score(self, X):
        """X:(B,SCW) -> (P,B) realness logits."""
        t = self.t
        w = t.as_tensor(X, device=self.dev)
        emb = self.E[:, w, :].reshape(self.P, w.shape[0], SCW * self.D)
        h = self._act(t.einsum("pbi,pih->pbh", emb, self.W1) + self.b1[:, None, :])
        return t.einsum("pbh,ph->pb", h, self.wo) + self.bo[:, None]

    def evaluate(self, X, y):
        t = self.t
        with t.inference_mode():
            s = self.score(X)                                 # (P,B) logit
            yv = t.as_tensor(y, device=self.dev)[None, :]
            lp = -t.nn.functional.softplus(-s) * yv - t.nn.functional.softplus(s) * (1 - yv)
            fit = lp.mean(1)                                  # soft log-prob of correct label
            acc = ((s > 0).float() == yv).float().mean(1)
            return fit, acc

    def _clone(self, dst, src):
        for n in self._T:
            getattr(self, n)[dst] = getattr(self, n)[src]

    def _mutate(self, idx):
        t = self.t
        self.mut_rate[idx] = (self.mut_rate[idx] * t.exp(0.2 * t.randn(len(idx), device=self.dev))).clamp(0.005, 0.2)
        self.mut_scale[idx] = (self.mut_scale[idx] * t.exp(0.2 * t.randn(len(idx), device=self.dev))).clamp(0.02, 0.5)
        for n in ("E", "W1", "b1", "wo", "bo"):
            ten = getattr(self, n); sub = ten[idx]
            mask = t.rand(sub.shape, device=self.dev) < self.mut_rate[idx].view(-1, *([1] * (sub.dim() - 1)))
            ten[idx] = sub + mask * t.randn(sub.shape, device=self.dev) * self.mut_scale[idx].view(-1, *([1] * (sub.dim() - 1)))
        am = t.rand(self.a1[idx].shape, device=self.dev) < (self.mut_rate[idx] / 4).view(-1, 1)
        self.a1[idx] = t.where(am, t.randint(0, N_ACT, self.a1[idx].shape, device=self.dev), self.a1[idx])

    def step(self, fitness, cfg):
        t = self.t
        with t.inference_mode():
            fresh = t.isnan(self.fit_ema)
            self.fit_ema = t.where(fresh, fitness, cfg["fit_ema"] * self.fit_ema + (1 - cfg["fit_ema"]) * fitness)
            f = self.fit_ema
            self.energy = (self.energy * cfg["energy_decay"] + cfg["energy_gain"] * (f - f.median())).clamp(0, cfg["energy_max"])
            dead = self.energy < cfg["energy_floor"]; n_dead = int(dead.sum())
            alive = (~dead).nonzero().squeeze(1); mat = alive[self.age[alive] >= 1]
            pool = mat if len(mat) >= 2 else alive
            if n_dead and len(pool):
                di = dead.nonzero().squeeze(1)
                cand = pool[t.randint(0, len(pool), (n_dead, cfg["tournament_k"]), device=self.dev)]
                win = cand[t.arange(n_dead, device=self.dev), f[cand].argmax(1)]
                self._clone(di, win); self._mutate(di)
                self.energy[di] = 1.0; self.age[di] = 0; self.fit_ema[di] = float("nan")
            self.age += 1
            return n_dead


SDEF = dict(pop=300, sd=8, sh=64, seed=1, fit_ema=0.75, energy_decay=0.90,
            energy_gain=8.0, energy_floor=0.20, energy_max=1.5, tournament_k=3)


def evolve_scorer(pred, b, generations=2500, batch=128, log=print):
    """Evolve the coherence scorer against the predictor's fakes."""
    import torch
    dev = pred.dev
    sp = ScorerPop({**SDEF, "pop": 300}, dev)
    ids = load_char_corpus()
    rng = np.random.default_rng(3)
    for gen in range(generations):
        X, y = make_fakes(pred, b, ids, batch // 2, SCW, rng, temperature=0.8)
        fit, acc = sp.evaluate(X, y)
        starved = sp.step(fit, SDEF)
        if gen % 250 == 0 or gen == generations - 1:
            bb = int(fit.argmax())
            log(f"scorer gen {gen:5d} real-vs-fake acc {float(acc[bb])*100:5.2f}% starved {starved}")
    Xh, yh = make_fakes(pred, b, ids, 256, SCW, np.random.default_rng(999))
    fit_h, acc_h = sp.evaluate(Xh, yh)
    return sp, int(fit_h.argmax()), float(acc_h[int(fit_h.argmax())])


def rerank_generate(pred, pb, scorer, sb, prompt, length=90, K=6, beta=1.5,
                    temperature=0.6, seed=None):
    """Predictor proposes top-K next chars; scorer picks the one that keeps the
    trailing window most 'real'. score = logP_pred + beta*realness."""
    import torch
    t = torch
    rng = np.random.default_rng(seed)
    lut = {c: i for i, c in enumerate(CHARS)}
    seq = [lut.get(c, RARE) for c in prompt.lower()] or [0]
    n = len(seq)
    with t.inference_mode():
        h = t.zeros(1, 1, pred.H, device=pred.dev)
        for i in range(n + length - 1):
            cur = seq[i]; prev = seq[i - 1] if i > 0 else seq[0]
            lg, h = _step(pred, pb, cur, prev, h)
            if i >= n - 1:
                logp = (lg / max(temperature, 1e-6)).log_softmax(0)
                topk = lg.topk(K).indices                       # candidate next chars
                # build the trailing SCW window for each candidate, score realness
                tail = seq[-(SCW - 1):] if len(seq) >= SCW - 1 else [seq[0]] * (SCW - 1 - len(seq)) + seq
                cand_windows = np.array([[*tail, int(c)][-SCW:] for c in topk.tolist()], np.int64)
                real = scorer.score(cand_windows)[sb]           # (K,) realness logit
                combined = logp[topk] + beta * real
                choice = int(topk[int(combined.argmax())])
                seq.append(choice)
    return "".join(CHARS[i] if i < len(CHARS) else "?" for i in seq[n:])


def raw_generate(pred, pb, prompt, length=90, temperature=0.6, seed=None):
    from .genreg_lm import generate
    return generate(pred, pb, prompt, length=length, temperature=temperature, seed=seed)
