"""enc_char_v1 — the recurrent encoder as its own GENREG model (component-first).

Model card: documentation/LM_ENCODER_COMPONENT.md. Fitness = evolved-head
decodability of the future at horizons {1, 2, 4} from the hidden state —
breeding the STATE, not the prediction. Heads are scaffolding, discarded at
composition; the frozen deliverable is (E, W_in, b_h, act).

Warm-startable from any lm_char_v1 checkpoint (the substrate's W_out becomes
the h=1 head). Same energy/EMA/tournament machinery, inherited.
"""

import datetime
import hashlib
import json
import os

import numpy as np

from .genreg_lm import (LMPopulation, load_char_corpus, sample_windows,
                        load_population, V)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENC_RUNS = os.path.join(ROOT, "runs", "enc")

HORIZONS = (1, 2, 4)


def skip_bigram_baselines(ids, rng, n_train=200000, n_test=20000):
    """P(c_{t+h} | c_t) count-table top-1 per horizon — the state-richness bars."""
    out = {}
    idx = rng.choice(len(ids) - max(HORIZONS) - 1, size=n_train + n_test, replace=False)
    tr, te = idx[:n_train], idx[n_train:]
    for h in HORIZONS:
        C = np.zeros((V, V), np.int64)
        np.add.at(C, (ids[tr], ids[tr + h]), 1)
        pred = C.argmax(1)[ids[te]]
        out[f"skip{h}_top1"] = float(np.mean(pred == ids[te + h]))
    return out


class EncPopulation(LMPopulation):
    """LMPopulation + two extra horizon heads (parent W_out/b_out = h=1 head)."""

    def __init__(self, cfg, device):
        super().__init__({**cfg, "attention": False}, device)
        import torch
        P, H = self.P, self.H
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"] + 7)
        mk = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to(device)
        self.W_out2 = mk(P, H, V, scale=1.0 / np.sqrt(H))
        self.b_out2 = torch.zeros(P, V, device=device)
        self.W_out4 = mk(P, H, V, scale=1.0 / np.sqrt(H))
        self.b_out4 = torch.zeros(P, V, device=device)

    def _genome_tensors(self):
        return super()._genome_tensors() + ["W_out2", "b_out2", "W_out4", "b_out4"]

    def _mutate(self, idx, rng_t=None):
        super()._mutate(idx, rng_t)
        t = self.t
        for name in ("W_out2", "b_out2", "W_out4", "b_out4"):
            ten = getattr(self, name)
            sub = ten[idx]
            mask = (t.rand(sub.shape, device=self.dev)
                    < self.mut_rate[idx].view(-1, *([1] * (sub.dim() - 1))))
            noise = (t.randn(sub.shape, device=self.dev)
                     * self.mut_scale[idx].view(-1, *([1] * (sub.dim() - 1))))
            ten[idx] = sub + mask * noise

    def evaluate_multi(self, windows, warmup, weights=(1 / 3, 1 / 3, 1 / 3)):
        """Fitness = weighted mean log-prob across horizons {1,2,4}.
        Equal weights spent too much h1 sharpness (29.7% vs the substrate's
        31.9%); h1-dominant weights buy state richness at a price h1 can pay.
        Returns (fitness, per-horizon top1 tensor (P, 3))."""
        t = self.t
        heads = ((1, self.W_out, self.b_out), (2, self.W_out2, self.b_out2),
                 (4, self.W_out4, self.b_out4))
        with t.inference_mode():
            w = t.as_tensor(windows, device=self.dev)
            B, T1 = w.shape
            T = T1 - 1
            h = t.zeros(self.P, B, self.H, device=self.dev)
            logp = t.zeros(self.P, len(HORIZONS), device=self.dev)
            top1 = t.zeros(self.P, len(HORIZONS), device=self.dev)
            steps = 0
            last = T - max(HORIZONS)                     # all horizons in range
            for i in range(last + 1):
                cur, prev = w[:, i], w[:, max(i - 1, 0)]
                x = t.cat([self.E[:, cur, :], self.E[:, prev, :], t.tanh(h)], dim=2)
                h = self._act(t.einsum("pbi,pih->pbh", x, self.W_in)
                              + self.b_h[:, None, :])
                if i < warmup:
                    continue
                for j, (hz, Wo, bo) in enumerate(heads):
                    logits = (t.einsum("pbh,phv->pbv", h, Wo) + bo[:, None, :])
                    tgt = w[:, i + hz]
                    lp = logits.log_softmax(dim=2)
                    logp[:, j] += lp[:, t.arange(B, device=self.dev), tgt].mean(dim=1)
                    top1[:, j] += (logits.argmax(dim=2) == tgt[None, :]).float().mean(dim=1)
                steps += 1
            wts = t.as_tensor(weights, device=self.dev, dtype=logp.dtype)
            return (logp / steps) @ wts, (top1 / steps)


def run(cfg=None, log=print, resume=None):
    """Standalone encoder sweep. `resume` may be an enc OR lm checkpoint
    (lm: encoder tensors + h=1 head transfer; h=2/4 heads stay fresh)."""
    import torch
    from .genreg_lm import DEFAULTS, save_population
    cfg = {**DEFAULTS, **(cfg or {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = load_char_corpus()
    rng = np.random.default_rng(cfg["seed"])

    ts = datetime.datetime.now()
    h8 = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}-enc-{h8}"
    run_dir = os.path.join(ENC_RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)

    bl = skip_bigram_baselines(ids, np.random.default_rng(999))
    log(f"[{run_id}] device={device} baselines={json.dumps({k: round(v, 4) for k, v in bl.items()})}")
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"id": run_id, "environment": "enc",
                   "created": ts.isoformat(timespec="seconds"),
                   "config": {**cfg, "device": device, "population": cfg["pop"],
                              "horizons": list(HORIZONS), "baselines": bl},
                   "started": {"population": cfg["pop"],
                               "generations": cfg["generations"],
                               "notes": "enc_char_v1 (card: LM_ENCODER_COMPONENT.md)"},
                   "status": "running"}, f, indent=2)
    open(os.path.join(run_dir, "history.jsonl"), "w").close()

    pop = EncPopulation(cfg, device)
    if resume:
        load_population(pop, resume)          # missing keys (h2/h4 heads) stay fresh
        log(f"warm start from {resume}")

    annealed = False
    weights = tuple(cfg.get("horizon_weights", (1 / 3, 1 / 3, 1 / 3)))
    for gen in range(cfg["generations"]):
        W = sample_windows(ids, cfg["batch"], cfg["seq_len"], rng)
        fit, top1 = pop.evaluate_multi(W, cfg["warmup"], weights)
        starved = pop.step_selection(fit, cfg)
        if not annealed and gen >= cfg["anneal_after"] * cfg["generations"]:
            pop.mut_scale.mul_(0.5).clamp_(min=0.02)
            annealed = True
        if gen % cfg["log_every"] == 0 or gen == cfg["generations"] - 1:
            b = int(fit.argmax())
            hz = {f"h{h}": round(float(top1[b, j]), 4) for j, h in enumerate(HORIZONS)}
            rec = {"gen": gen,
                   "fitness": {"best": round(float(fit[b]), 4),
                               "mean": round(float(fit.mean()), 4)},
                   "best": {"score": hz["h1"], **hz}, "starved": starved}
            with open(os.path.join(run_dir, "history.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"gen {gen:5d} soft {rec['fitness']['best']:+.4f} "
                f"h1 {hz['h1']*100:5.2f}% h2 {hz['h2']*100:5.2f}% h4 {hz['h4']*100:5.2f}% "
                f"starved {starved:3d}")

    W_ho = sample_windows(ids, 256, cfg["seq_len"], np.random.default_rng(31337))
    fit_ho, top1_ho = pop.evaluate_multi(W_ho, cfg["warmup"], weights)
    b = int(fit_ho.argmax())
    per_h = {f"h{h}": round(float(top1_ho[b, j]), 4) for j, h in enumerate(HORIZONS)}
    save_population(pop, os.path.join(run_dir, "checkpoint.npz"), gen=cfg["generations"])
    bars = {"h1_vs_bigram": round(per_h["h1"] - 0.2730, 4),
            "h2_vs_skip2": round(per_h["h2"] - bl["skip2_top1"], 4),
            "h4_vs_skip4": round(per_h["h4"] - bl["skip4_top1"], 4)}
    summary = {"id": run_id, "environment": "enc", "status": "finished",
               "finished": datetime.datetime.now().isoformat(timespec="seconds"),
               "gen": cfg["generations"],
               "best": {"score": per_h["h1"], **per_h, "soft": round(float(fit_ho[b]), 4)},
               "baselines": bl, "bars": bars, "checkpoint": "checkpoint.npz"}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    cfgj = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))
    cfgj["status"] = "finished"
    json.dump(cfgj, open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8"), indent=2)
    log(f"HELD-OUT: {per_h} · bars {bars}")
    try:
        import agent_board
        agent_board.post_run_event("enc", {"type": "done", "run_id": run_id,
                                           "best": summary["best"]})
    except Exception:
        pass
    return run_id, run_dir, pop, summary


if __name__ == "__main__":
    run()
