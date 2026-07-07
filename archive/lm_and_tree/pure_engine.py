"""pure_engine — multi-layer evolved model with PER-LAYER (localized) constraints.

Model card: documentation/PURE_PER_LAYER_CONSTRAINTS.md. The mechanic (user's
"tissue differentiation"): a constraint wired to a layer evaluates its penalty
from THAT layer's activations, not the whole network's. Same tournament / energy
/ whole-organism reproduction — only the fitness landscape becomes locally
uneven. This is the evolving backend PURE's node graph feeds.

Charter: torch inference-mode only (no autograd), per-neuron evolved activations,
soft multiplicative fitness, mandatory energy homeostasis, no gradients.

Task: next-char prediction from the last K chars — input embeddings → L1 → L2 →
readout. Everything evolved.
"""
import datetime
import hashlib
import json
import os

import numpy as np

from .genreg_lm import load_char_corpus, CHARS, V, N_ACT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PURE_RUNS = os.path.join(ROOT, "runs", "pure")


def char_windows(ids, n, K, rng):
    """n samples of (K context chars -> next char)."""
    starts = rng.choice(len(ids) - K - 1, size=n, replace=False)
    X = ids[starts[:, None] + np.arange(K)[None, :]]
    y = ids[starts + K]
    return X, y


class PurePopulation:
    """Two-hidden-layer MLP population. Constraints is a list of dicts:
    {"type": "energy"|"consequential", "layer": 1|2|None, "budget": float}.
    layer=None → global (whole-network) evaluation (the control)."""

    ACT = ("tanh", "sigmoid", "relu", "sin", "gauss", "id", "softsign", "abs")

    def __init__(self, cfg, device):
        import torch
        self.t = torch
        self.dev = device
        P, K, D = cfg["pop"], cfg["ctx"], cfg["embed"]
        H1, H2 = cfg["h1"], cfg["h2"]
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
        self.P, self.K, self.D, self.H1, self.H2 = P, K, D, H1, H2
        self.constraints = cfg.get("constraints", [])
        self.per_layer = bool(cfg.get("per_layer", True))

    _TENSORS = ("E", "W1", "b1", "a1", "W2", "b2", "a2", "Wo", "bo",
                "mut_rate", "mut_scale")

    def _act(self, pre, aid):
        t = self.t
        a = aid[:, None, :].expand_as(pre)
        o = t.tanh(pre)
        o = t.where(a == 1, t.sigmoid(pre), o)
        o = t.where(a == 2, t.relu(pre).clamp(max=4.0), o)
        o = t.where(a == 3, t.sin(pre), o)
        o = t.where(a == 4, t.exp(-pre * pre), o)
        o = t.where(a == 5, pre.clamp(-4.0, 4.0), o)
        o = t.where(a == 6, pre / (1 + pre.abs()), o)
        o = t.where(a == 7, pre.abs().clamp(max=4.0), o)
        return o

    def forward(self, X):
        """X:(B,K) -> (logits (P,B,V), h1 (P,B,H1), h2 (P,B,H2))."""
        t = self.t
        w = t.as_tensor(X, device=self.dev)
        emb = self.E[:, w, :].reshape(self.P, w.shape[0], self.K * self.D)  # (P,B,KD)
        h1 = self._act(t.einsum("pbi,pih->pbh", emb, self.W1) + self.b1[:, None, :], self.a1)
        h2 = self._act(t.einsum("pbi,pih->pbh", h1, self.W2) + self.b2[:, None, :], self.a2)
        logits = t.einsum("pbh,phv->pbv", h2, self.Wo) + self.bo[:, None, :]
        return logits, h1, h2

    def _penalty(self, c, h1, h2):
        """Constraint penalty per genome (P,), read from the wired layer."""
        t = self.t
        layer = c.get("layer")
        budget = c.get("budget", 1.0)
        if c["type"] == "energy":
            # power = mean |activation|. Wired to a layer → that layer only;
            # global → concat of both layers.
            if not self.per_layer or layer is None:
                power = t.cat([h1.abs().mean(2), h2.abs().mean(2)], 1).mean(1)
            else:
                h = h1 if layer == 1 else h2
                power = h.abs().mean(dim=(1, 2))
            return 1.0 / (1.0 + power / budget)
        if c["type"] == "consequential":
            # a neuron matters if it VARIES across inputs AND feeds the output.
            # dead_frac = fraction of the layer's neurons with low consequence.
            def dead(h, Wnext):
                std = h.std(dim=1)                              # (P, Hn) across batch
                # downstream weight magnitude per neuron (mean over its fan-out)
                wmag = Wnext.abs().mean(dim=2)                  # (P, Hn)
                cons = std * wmag                               # (P, Hn)
                thr = 0.05 * cons.mean(dim=1, keepdim=True) + 1e-6
                return (cons < thr).float().mean(dim=1)         # (P,)
            if not self.per_layer or layer is None:
                d = 0.5 * (dead(h1, self.W2) + dead(h2, self.Wo))
            else:
                d = dead(h1, self.W2) if layer == 1 else dead(h2, self.Wo)
            return 1.0 / (1.0 + d / budget)
        return t.ones(self.P, device=self.dev)

    def evaluate(self, X, y, want_stats=False):
        t = self.t
        with t.inference_mode():
            logits, h1, h2 = self.forward(X)
            yv = t.as_tensor(y, device=self.dev)
            lp = logits.log_softmax(2)
            base = lp[:, t.arange(len(y), device=self.dev), yv].mean(1)   # soft (P,)
            top1 = (logits.argmax(2) == yv[None, :]).float().mean(1)
            # localized penalties multiply into base (exp so base<0 stays sane)
            fit = t.exp(base)                                  # (0,1] geometric base
            for c in self.constraints:
                fit = fit * self._penalty(c, h1, h2)
            fit = t.log(fit.clamp_min(1e-12))                 # back to log space
            if want_stats:
                stats = {"L1_power": float(h1.abs().mean().cpu()),
                         "L2_power": float(h2.abs().mean().cpu()),
                         "L1_dead": float(self._dead_frac(h1, self.W2).mean().cpu()),
                         "L2_dead": float(self._dead_frac(h2, self.Wo).mean().cpu())}
                return fit, top1, stats
            return fit, top1

    def _dead_frac(self, h, Wnext):
        t = self.t
        std = h.std(dim=1); wmag = Wnext.abs().mean(dim=2)
        cons = std * wmag
        thr = 0.05 * cons.mean(dim=1, keepdim=True) + 1e-6
        return (cons < thr).float().mean(dim=1)

    # -- GENREG machinery (energy homeostasis + tournament, from genreg_lm) --
    def _clone(self, dst, src):
        for n in self._TENSORS:
            getattr(self, n)[dst] = getattr(self, n)[src]

    def _mutate(self, idx):
        t = self.t
        self.mut_rate[idx] = (self.mut_rate[idx] * t.exp(0.2 * t.randn(len(idx), device=self.dev))).clamp(0.005, 0.2)
        self.mut_scale[idx] = (self.mut_scale[idx] * t.exp(0.2 * t.randn(len(idx), device=self.dev))).clamp(0.02, 0.5)
        for n in ("E", "W1", "b1", "W2", "b2", "Wo", "bo"):
            ten = getattr(self, n); sub = ten[idx]
            mask = t.rand(sub.shape, device=self.dev) < self.mut_rate[idx].view(-1, *([1] * (sub.dim() - 1)))
            noise = t.randn(sub.shape, device=self.dev) * self.mut_scale[idx].view(-1, *([1] * (sub.dim() - 1)))
            ten[idx] = sub + mask * noise
        for n in ("a1", "a2"):
            ten = getattr(self, n)
            am = t.rand(ten[idx].shape, device=self.dev) < (self.mut_rate[idx] / 4).view(-1, 1)
            rnd = t.randint(0, N_ACT, ten[idx].shape, device=self.dev)
            ten[idx] = t.where(am, rnd, ten[idx])

    def step(self, fitness, cfg):
        t = self.t
        with t.inference_mode():
            a = cfg["fit_ema"]; fresh = t.isnan(self.fit_ema)
            self.fit_ema = t.where(fresh, fitness, a * self.fit_ema + (1 - a) * fitness)
            f = self.fit_ema
            self.energy = (self.energy * cfg["energy_decay"] + cfg["energy_gain"] * (f - f.median())).clamp(0, cfg["energy_max"])
            dead = self.energy < cfg["energy_floor"]
            n_dead = int(dead.sum()); alive = (~dead).nonzero().squeeze(1)
            mature = alive[self.age[alive] >= 1]; pool = mature if len(mature) >= 2 else alive
            if n_dead and len(pool):
                di = dead.nonzero().squeeze(1)
                cand = pool[t.randint(0, len(pool), (n_dead, cfg["tournament_k"]), device=self.dev)]
                win = cand[t.arange(n_dead, device=self.dev), f[cand].argmax(1)]
                self._clone(di, win); self._mutate(di)
                self.energy[di] = 1.0; self.age[di] = 0; self.fit_ema[di] = float("nan")
            self.age += 1
            return n_dead


DEFAULTS = dict(pop=400, ctx=4, embed=8, h1=48, h2=48, batch=128,
                generations=3000, seed=1, fit_ema=0.75,
                energy_decay=0.90, energy_gain=8.0, energy_floor=0.20,
                energy_max=1.5, tournament_k=3, log_every=100,
                per_layer=True, constraints=[])


def run(cfg=None, log=print, tag=""):
    import torch
    cfg = {**DEFAULTS, **(cfg or {})}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = load_char_corpus()
    rng = np.random.default_rng(cfg["seed"])
    ts = datetime.datetime.now()
    h8 = hashlib.sha1((json.dumps(cfg, sort_keys=True, default=str) + tag).encode()).hexdigest()[:6]
    run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}-pure-{h8}"
    run_dir = os.path.join(PURE_RUNS, run_id); os.makedirs(run_dir, exist_ok=True)
    mode = "per-layer" if cfg["per_layer"] else "global"
    log(f"[{run_id}] {tag} constraints={[ (c['type'], c.get('layer')) for c in cfg['constraints'] ]} mode={mode}")
    open(os.path.join(run_dir, "history.jsonl"), "w").close()

    pop = PurePopulation(cfg, dev)
    for gen in range(cfg["generations"]):
        X, y = char_windows(ids, cfg["batch"], cfg["ctx"], rng)
        fit, top1 = pop.evaluate(X, y)
        starved = pop.step(fit, cfg)
        if gen % cfg["log_every"] == 0 or gen == cfg["generations"] - 1:
            Xs, ys = char_windows(ids, 512, cfg["ctx"], np.random.default_rng(999))
            _, t1, st = pop.evaluate(Xs, ys, want_stats=True)
            b = int(fit.argmax())
            rec = {"gen": gen, "top1": round(float(t1[b]), 4), **st, "starved": starved}
            with open(os.path.join(run_dir, "history.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"gen {gen:5d} top1 {rec['top1']*100:5.2f}% | L1 pow {st['L1_power']:.3f} dead {st['L1_dead']*100:4.1f}% "
                f"| L2 pow {st['L2_power']:.3f} dead {st['L2_dead']*100:4.1f}% | starved {starved}")

    Xh, yh = char_windows(ids, 4096, cfg["ctx"], np.random.default_rng(31337))
    fit_h, t1_h, st_h = pop.evaluate(Xh, yh, want_stats=True)
    b = int(fit_h.argmax())
    summary = {"id": run_id, "tag": tag, "mode": mode,
               "constraints": [(c["type"], c.get("layer")) for c in cfg["constraints"]],
               "heldout_top1": round(float(t1_h[b]), 4), "stats": st_h}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
    log(f"HELD-OUT {tag}: top1 {float(t1_h[b])*100:.2f}% | "
        f"L1 pow {st_h['L1_power']:.3f}/dead {st_h['L1_dead']*100:.1f}% "
        f"L2 pow {st_h['L2_power']:.3f}/dead {st_h['L2_dead']*100:.1f}%")
    return summary


if __name__ == "__main__":
    run()
