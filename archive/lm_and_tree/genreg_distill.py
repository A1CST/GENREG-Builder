"""genreg_distill — distill the n-gram trust-mix TEACHER into a compact,
table-free evolved model. The point (user, 2026-07-06): n-gram lookup tables
are 1990s tech and huge; a real model should INTERNALIZE the statistics in its
weights so we don't ship the tables.

Teacher: the trust-mix (genreg_trustmix, ~55% held-out) — gives a full smoothed
distribution P_teacher(next | context) at each position.
Student: a feedforward neural n-gram — last K chars → embeddings → MLP (per-neuron
evolved activations) → V logits. No tables at inference.
Fitness: SOFT distillation — maximize Σ_v P_teacher(v)·log softmax(student)(v)
(cross-entropy to the teacher's whole distribution, not a one-hot target). Soft
targets are what the docs found unblock evolution (hard→soft: 10%→24%).

Everything gradient-free: torch inference-mode, energy homeostasis, tournament.
"""
import datetime
import hashlib
import json
import os

import numpy as np

from .genreg_lm import load_char_corpus, sample_windows, CHARS, RARE, V, N_ACT
from . import genreg_trustmix as tm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTILL_RUNS = os.path.join(ROOT, "runs", "distill")


# --------------------------------------------------------------------------
# Teacher pool: (last-K context chars, teacher distribution, true target)
# --------------------------------------------------------------------------
def build_teacher_pool(teacher_rid, K, n_windows=600, seq=64, seed=7, log=print):
    """Run the trust-mix teacher over windows; collect per-position
    (context K chars, teacher dist (V,), true next char)."""
    import torch
    t = torch
    ids = load_char_corpus()
    tabs = tm.build_ngrams(ids)
    pop, b, t1 = tm.load_predictor(teacher_rid)
    genome = t.tensor(np.load(os.path.join(ROOT, "runs", "pure", "trustmix_genome.npy")),
                      device="cuda")
    log(f"teacher {teacher_rid} loaded (neural top1 {t1*100:.1f}%)")
    W = sample_windows(ids, n_windows, seq, np.random.default_rng(seed))
    chans, evid, tgt = tm.collect(pop, b, tabs, W, warmup=8)
    P = t.stack([chans[c] for c in tm.CHANNELS], 0)
    Ev = t.stack([evid[c] for c in tm.CHANNELS], 0)
    teach = tm._gate(genome[None, :], P, Ev)[0]                  # (N, V) teacher dist
    # reconstruct the K-char context for each collected position. collect scores
    # positions i>=warmup of each window; context = chars [i-K+1 .. i].
    w = t.as_tensor(W, device="cuda"); B, T1 = w.shape; T = T1 - 1
    ctxs = []
    for i in range(8, T):
        c = w[:, max(i - K + 1, 0):i + 1]
        if c.shape[1] < K:
            c = t.cat([w[:, :1].expand(B, K - c.shape[1]), c], 1)
        ctxs.append(c)
    ctx = t.cat(ctxs, 0)                                         # (N, K)
    log(f"teacher pool: {ctx.shape[0]} positions, K={K}")
    return ctx.cpu().numpy().astype(np.int64), teach.cpu().numpy(), tgt.cpu().numpy()


# --------------------------------------------------------------------------
# Student: feedforward neural n-gram (no tables)
# --------------------------------------------------------------------------
class Student:
    def __init__(self, cfg, device):
        import torch
        self.t = torch
        self.dev = device
        P, K, D, H1, H2 = cfg["pop"], cfg["K"], cfg["embed"], cfg["h1"], cfg["h2"]
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        mk = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(device)
        self.E = mk(P, V, D, sc=0.1)
        self.W1 = mk(P, K * D, H1, sc=1.0 / np.sqrt(K * D)); self.b1 = torch.zeros(P, H1, device=device)
        self.a1 = torch.randint(0, N_ACT, (P, H1), generator=g).to(device)
        self.W2 = mk(P, H1, H2, sc=1.0 / np.sqrt(H1)); self.b2 = torch.zeros(P, H2, device=device)
        self.a2 = torch.randint(0, N_ACT, (P, H2), generator=g).to(device)
        self.Wo = mk(P, H2, V, sc=1.0 / np.sqrt(H2)); self.bo = torch.zeros(P, V, device=device)
        self.mut_rate = torch.full((P,), 0.05, device=device)
        self.mut_scale = torch.full((P,), 0.05, device=device)
        self.energy = torch.full((P,), 1.0, device=device)
        self.age = torch.zeros(P, dtype=torch.long, device=device)
        self.fit_ema = torch.full((P,), float("nan"), device=device)
        self.P, self.K, self.D = P, K, D

    _T = ("E", "W1", "b1", "a1", "W2", "b2", "a2", "Wo", "bo", "mut_rate", "mut_scale")

    def _act(self, pre, aid):
        t = self.t; a = aid[:, None, :].expand_as(pre)
        o = t.tanh(pre)
        o = t.where(a == 1, t.sigmoid(pre), o); o = t.where(a == 2, t.relu(pre).clamp(max=4), o)
        o = t.where(a == 3, t.sin(pre), o); o = t.where(a == 4, t.exp(-pre * pre), o)
        o = t.where(a == 5, pre.clamp(-4, 4), o); o = t.where(a == 6, pre / (1 + pre.abs()), o)
        o = t.where(a == 7, pre.abs().clamp(max=4), o)
        return o

    def logits(self, X):
        """X:(B,K) -> (P,B,V)."""
        t = self.t
        w = t.as_tensor(X, device=self.dev)
        emb = self.E[:, w, :].reshape(self.P, w.shape[0], self.K * self.D)
        h1 = self._act(t.einsum("pbi,pih->pbh", emb, self.W1) + self.b1[:, None, :], self.a1)
        h2 = self._act(t.einsum("pbi,pih->pbh", h1, self.W2) + self.b2[:, None, :], self.a2)
        return t.einsum("pbh,phv->pbv", h2, self.Wo) + self.bo[:, None, :]

    def evaluate(self, X, teach, y, lam=2.0):
        """HYBRID distillation fitness (docs §XI fix): soft CE to the teacher's
        distribution (shape) + lam · log P(true target) (sharpens the argmax the
        soft term alone leaves at frequent chars). Returns top1/top5 vs TRUE."""
        t = self.t
        with t.inference_mode():
            lg = self.logits(X)
            lp = lg.log_softmax(2)                                # (P,B,V)
            te = t.as_tensor(teach, device=self.dev)              # (B,V)
            yv = t.as_tensor(y, device=self.dev)
            soft = (lp * te[None, :, :]).sum(2).mean(1)           # match teacher shape
            hard = lp[:, t.arange(len(y), device=self.dev), yv].mean(1)   # nail true next
            fit = soft + lam * hard
            top1 = (lg.argmax(2) == yv[None, :]).float().mean(1)
            top5 = (lg.topk(5, 2).indices == yv[None, :, None]).any(2).float().mean(1)
            return fit, top1, top5

    def _clone(self, dst, src):
        for n in self._T:
            getattr(self, n)[dst] = getattr(self, n)[src]

    def _mutate(self, idx):
        t = self.t
        self.mut_rate[idx] = (self.mut_rate[idx] * t.exp(0.2 * t.randn(len(idx), device=self.dev))).clamp(0.005, 0.2)
        self.mut_scale[idx] = (self.mut_scale[idx] * t.exp(0.2 * t.randn(len(idx), device=self.dev))).clamp(0.02, 0.5)
        for n in ("E", "W1", "b1", "W2", "b2", "Wo", "bo"):
            ten = getattr(self, n); sub = ten[idx]
            mask = t.rand(sub.shape, device=self.dev) < self.mut_rate[idx].view(-1, *([1] * (sub.dim() - 1)))
            ten[idx] = sub + mask * t.randn(sub.shape, device=self.dev) * self.mut_scale[idx].view(-1, *([1] * (sub.dim() - 1)))
        for n in ("a1", "a2"):
            ten = getattr(self, n)
            am = t.rand(ten[idx].shape, device=self.dev) < (self.mut_rate[idx] / 4).view(-1, 1)
            ten[idx] = t.where(am, t.randint(0, N_ACT, ten[idx].shape, device=self.dev), ten[idx])

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


DEFAULTS = dict(pop=400, K=6, embed=16, h1=256, h2=256, batch=256,
                generations=12000, seed=1, fit_ema=0.75, energy_decay=0.90,
                energy_gain=8.0, energy_floor=0.20, energy_max=1.5,
                tournament_k=3, log_every=500)


def run(teacher_rid="20260706-150603-lm-724543", cfg=None, log=print):
    import torch
    cfg = {**DEFAULTS, **(cfg or {})}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ts = datetime.datetime.now()
    rid = f"{ts.strftime('%Y%m%d-%H%M%S')}-distill-{hashlib.sha1(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:6]}"
    run_dir = os.path.join(DISTILL_RUNS, rid); os.makedirs(run_dir, exist_ok=True)

    ctx, teach, tgt = build_teacher_pool(teacher_rid, cfg["K"], log=log)
    # split pool: 90% train, 10% held-out (never sampled in training)
    n = len(ctx); rng = np.random.default_rng(0); perm = rng.permutation(n)
    ntr = int(n * 0.9); tr, ho = perm[:ntr], perm[ntr:]
    teacher_top1 = float(np.mean(teach[ho].argmax(1) == tgt[ho]) * 100)
    log(f"teacher held-out top1 {teacher_top1:.2f}%  (this is the ceiling to distill toward)")

    stu = Student(cfg, dev)
    grng = np.random.default_rng(cfg["seed"])
    for gen in range(cfg["generations"]):
        bi = tr[grng.integers(0, ntr, cfg["batch"])]              # fresh minibatch
        fit, t1, _ = stu.evaluate(ctx[bi], teach[bi], tgt[bi])
        starved = stu.step(fit, cfg)
        if gen % cfg["log_every"] == 0 or gen == cfg["generations"] - 1:
            f2, t1h, t5h = stu.evaluate(ctx[ho], teach[ho], tgt[ho])
            b = int(fit.argmax())
            log(f"gen {gen:6d} student held-out top1 {float(t1h[b])*100:5.2f}% top5 {float(t5h[b])*100:5.2f}% "
                f"(teacher {teacher_top1:.1f}%) starved {starved}")
    # final held-out + save
    f2, t1h, t5h = stu.evaluate(ctx[ho], teach[ho], tgt[ho])
    b = int(f2.argmax())
    np.savez_compressed(os.path.join(run_dir, "student.npz"),
                        **{k: getattr(stu, k).cpu().numpy() for k in stu._T},
                        K=cfg["K"], best=b)
    summary = {"id": rid, "teacher": teacher_rid, "K": cfg["K"],
               "teacher_top1": round(teacher_top1, 2),
               "student_top1": round(float(t1h[b]) * 100, 2),
               "student_top5": round(float(t5h[b]) * 100, 2),
               "recovered_pct": round(float(t1h[b]) * 100 / teacher_top1 * 100, 1)}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    log(f"FINAL table-free student: top1 {summary['student_top1']}% top5 {summary['student_top5']}% "
        f"= {summary['recovered_pct']}% of the teacher, NO tables")
    return rid, run_dir, stu, b, summary
