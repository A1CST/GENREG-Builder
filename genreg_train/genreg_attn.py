"""attn_copy_v1 — evolved selective retrieval on offset-k copy (component 2).

Model card: documentation/LM_ATTENTION_COMPONENT.md. Charter: no gradients
(torch inference-mode only), constraints shape the environment, per-k
equal-weight soft fitness so partial solutions can't win.

Episode: [K_TOK(k), sym, sym, ..., FLAG, ..., sym] (length L in 24..64).
Target: the symbol k positions before FLAG. Query = state at FLAG position.
"""

import datetime
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ATTN_RUNS = os.path.join(ROOT, "runs", "attn")

S = 16                                # symbol alphabet
OFFSETS = (1, 2, 5, 10, 20)
FLAG = S                              # token ids: 0..15 symbols, 16 flag,
K_BASE = S + 1                        # 17..21 the five k-announce tokens
V = K_BASE + len(OFFSETS)             # 22
L_MAX = 64
N_ACT = 8


def make_episodes(n_per_k, rng, offsets=OFFSETS, l_lo=24, l_hi=L_MAX):
    """Batch of episodes, n_per_k for each ACTIVE offset (curriculum funnel:
    the environment only poses questions the population has earned). Returns
    (seq, flag_pos, target, k_index-within-offsets)."""
    B = n_per_k * len(offsets)
    seq = rng.integers(0, S, size=(B, L_MAX))
    ep_len = np.zeros(B, dtype=np.int64)
    flag_pos = np.zeros(B, dtype=np.int64)
    target = np.zeros(B, dtype=np.int64)
    k_idx = np.zeros(B, dtype=np.int64)
    for j, k in enumerate(offsets):
        kj = OFFSETS.index(k)                         # stable announce token
        for i in range(n_per_k):
            b = j * n_per_k + i
            L = int(rng.integers(max(l_lo, k + 2), max(l_hi, k + 3) + 1))
            fp = int(rng.integers(k + 1, L))          # k symbols must precede
            seq[b, 0] = K_BASE + kj                   # environment states k
            seq[b, fp] = FLAG
            ep_len[b] = L                             # keys beyond L are masked
            flag_pos[b] = fp
            target[b] = seq[b, fp - k]
            k_idx[b] = j
    return seq, flag_pos, target, k_idx, ep_len


class AttnPopulation:
    def __init__(self, cfg, device):
        import torch
        self.t = torch
        self.dev = device
        P, D, A, H = cfg["pop"], cfg["embed"], cfg["attn_dim"], cfg["hidden"]
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        mk = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to(device)
        self.E = mk(P, V, D, scale=0.1)
        # transparent init (the cascade lesson: structured components can't
        # evolve from random init) — deterministic sinusoid position code;
        # relative-offset retrieval is EXPRESSIBLE from this basin, and the
        # table remains fully evolvable afterwards.
        pos = np.zeros((L_MAX, D), np.float32)
        for i in range(L_MAX):
            for j in range(D):
                if j % 2 == 0:
                    pos[i, j] = np.sin(i / (10000 ** (j / D)))
                else:
                    pos[i, j] = np.cos(i / (10000 ** ((j - 1) / D)))
        self.Pp = (torch.as_tensor(pos).expand(P, L_MAX, D).clone()
                   * 0.5).to(device)
        # query reads BOTH the flag state and the k-announcement state at
        # position 0 — without it the offset is invisible to the query
        # (information must flow; v1 failed at chance for exactly this)
        # MULTIPLICATIVE query (§VI: interaction structure additive can't
        # express): q = (flag_state ⊙ k_state) @ Wq is a DIFFERENT effective
        # linear map per offset token — per-k rotations become expressible.
        self.Wq = mk(P, D, A, scale=1.0 / np.sqrt(D))
        self.Wk = mk(P, D, A, scale=1.0 / np.sqrt(D))
        # values init as IDENTITY (A=D forced) and the readout is weight-tied
        # to the symbol embeddings (logits = ctx · E[sym]^T): the value→symbol
        # path is solved BY CONSTRUCTION, so the only thing evolution must
        # discover is where attention points — one basin, immediate credit.
        assert A == D, "attn_dim must equal embed (identity value path)"
        self.Wv = torch.eye(D).expand(P, D, D).clone().to(device)
        # evolved RELATIVE-POSITION bias per offset token, zero-init (a
        # transparent no-op): score(l) += B_rel[k, fp−l]. Makes "attend
        # exactly k back" directly expressible; evolution grows the bump.
        self.B_rel = torch.zeros(P, len(OFFSETS), L_MAX, device=device)
        self.mut_rate = torch.full((P,), 0.05, device=device)
        self.mut_scale = torch.full((P,), 0.05, device=device)
        self.energy = torch.full((P,), 1.0, device=device)
        self.age = torch.zeros(P, dtype=torch.long, device=device)
        self.fit_ema = torch.full((P,), float("nan"), device=device)
        self.P_, self.D, self.A, self.H = P, D, A, H

    def _act_fn(self, pre, act=None):
        t = self.t
        a = (act if act is not None else self.act)[:, None, :].expand_as(pre)
        out = t.tanh(pre)
        out = t.where(a == 1, t.sigmoid(pre), out)
        out = t.where(a == 2, t.relu(pre).clamp(max=4.0), out)
        out = t.where(a == 3, t.sin(pre), out)
        out = t.where(a == 4, t.exp(-pre * pre), out)
        out = t.where(a == 5, pre.clamp(-4.0, 4.0), out)
        out = t.where(a == 6, pre / (1 + pre.abs()), out)
        out = t.where(a == 7, pre.abs().clamp(max=4.0), out)
        return out

    def evaluate(self, seq, flag_pos, target, k_idx, ep_len):
        """Returns per-genome (fitness [equal-weight per k], per-k top1 (P, K))."""
        t = self.t
        with t.inference_mode():
            seq_t = t.as_tensor(seq, device=self.dev)
            fp = t.as_tensor(flag_pos, device=self.dev)
            tgt = t.as_tensor(target, device=self.dev)
            ki = t.as_tensor(k_idx, device=self.dev)
            el = t.as_tensor(ep_len, device=self.dev)
            B = seq_t.shape[0]
            nk = int(ki.max()) + 1
            H_states = self.E[:, seq_t, :] + self.Pp[:, None, :, :]   # (P,B,L,D)
            flag_state = H_states[:, t.arange(B, device=self.dev), fp, :]
            k_state = H_states[:, :, 0, :]                            # (P,B,D)
            q = t.einsum("pbd,pda->pba", flag_state * k_state, self.Wq)
            K = t.einsum("pbld,pda->pbla", H_states, self.Wk)
            Vv = t.einsum("pbld,pda->pbla", H_states, self.Wv)
            att = t.einsum("pba,pbla->pbl", q, K) / np.sqrt(self.A)
            # evolved relative-position bias: distance from the flag
            # (active offsets are a prefix of OFFSETS, so ki is global)
            d = (fp[:, None] - t.arange(L_MAX, device=self.dev)[None, :]
                 ).clamp(0, L_MAX - 1)                                # (B, L)
            bias = self.B_rel[:, ki, :]                               # (P, B, L)
            att = att + t.gather(bias, 2, d[None, :, :].expand(self.P_, -1, -1))
            # the episode ENDS at ep_len — keys beyond it don't exist
            beyond = (t.arange(L_MAX, device=self.dev)[None, :]
                      >= el[:, None])                                 # (B, L)
            att = att.masked_fill(beyond[None, :, :], float("-inf"))
            att = att.softmax(dim=2)                                  # head norm
            ctx = t.einsum("pbl,pbla->pba", att, Vv)
            # weight-tied readout: score against the symbol embeddings
            logits = t.einsum("pbd,psd->pbs", ctx, self.E[:, :S, :])
            lp = logits.log_softmax(dim=2)
            lp_t = lp[:, t.arange(B, device=self.dev), tgt]           # (P,B)
            top1 = (logits.argmax(dim=2) == tgt[None, :]).float()
            fit = t.zeros(self.P_, device=self.dev)
            acc_k = t.zeros(self.P_, nk, device=self.dev)
            for j in range(nk):
                m = ki == j
                fit += lp_t[:, m].mean(dim=1) / nk                    # equal weight
                acc_k[:, j] = top1[:, m].mean(dim=1)
            return fit, acc_k

    TENSORS = ("E", "Pp", "Wq", "Wk", "Wv", "B_rel", "mut_rate", "mut_scale")

    def _clone_into(self, dst, src):
        for n in self.TENSORS:
            getattr(self, n)[dst] = getattr(self, n)[src]

    def _mutate(self, idx):
        t = self.t
        self.mut_rate[idx] = (self.mut_rate[idx]
                              * t.exp(0.2 * t.randn(len(idx), device=self.dev))
                              ).clamp(0.005, 0.2)
        self.mut_scale[idx] = (self.mut_scale[idx]
                               * t.exp(0.2 * t.randn(len(idx), device=self.dev))
                               ).clamp(0.02, 0.5)
        for n in ("E", "Pp", "Wq", "Wk", "Wv", "B_rel"):
            ten = getattr(self, n)
            sub = ten[idx]
            mask = (t.rand(sub.shape, device=self.dev)
                    < self.mut_rate[idx].view(-1, *([1] * (sub.dim() - 1))))
            noise = (t.randn(sub.shape, device=self.dev)
                     * self.mut_scale[idx].view(-1, *([1] * (sub.dim() - 1))))
            ten[idx] = sub + mask * noise

    def step_selection(self, fitness, cfg):
        t = self.t
        with t.inference_mode():
            a = cfg["fit_ema"]
            fresh = t.isnan(self.fit_ema)
            self.fit_ema = t.where(fresh, fitness,
                                   a * self.fit_ema + (1 - a) * fitness)
            f = self.fit_ema
            self.energy = (self.energy * cfg["energy_decay"]
                           + cfg["energy_gain"] * (f - f.median())
                           ).clamp(0.0, cfg["energy_max"])
            dead = self.energy < cfg["energy_floor"]
            n_dead = int(dead.sum())
            alive = (~dead).nonzero().squeeze(1)
            mature = alive[self.age[alive] >= 1]
            pool = mature if len(mature) >= 2 else alive
            if n_dead and len(pool):
                di = dead.nonzero().squeeze(1)
                cand = pool[t.randint(0, len(pool), (n_dead, cfg["tournament_k"]),
                                      device=self.dev)]
                winners = cand[t.arange(n_dead, device=self.dev),
                               f[cand].argmax(dim=1)]
                self._clone_into(di, winners)
                self._mutate(di)
                self.energy[di] = 1.0
                self.age[di] = 0
                self.fit_ema[di] = float("nan")
            self.age += 1
            return n_dead


DEFAULTS = dict(pop=400, embed=32, attn_dim=32, hidden=32,
                n_per_k=12, generations=3000, seed=1, fit_ema=0.75,
                energy_decay=0.90, energy_gain=8.0, energy_floor=0.20,
                energy_max=1.5, tournament_k=3, anneal_after=0.8,
                log_every=25)


_STATE = ("E", "Pp", "Wq", "Wk", "Wv", "B_rel", "mut_rate", "mut_scale",
          "energy", "age", "fit_ema")


def save_population(pop, path, gen=0):
    np.savez_compressed(path, gen=gen, **{
        n: getattr(pop, n).cpu().numpy() for n in _STATE})


def load_population(pop, path):
    import torch
    data = np.load(path)
    for n in _STATE:
        if n in data:
            getattr(pop, n).copy_(torch.as_tensor(data[n], device=pop.dev))


def run(cfg=None, log=print, resume=None):
    import torch
    cfg = {**DEFAULTS, **(cfg or {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg["seed"])

    ts = datetime.datetime.now()
    h8 = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}-attn-{h8}"
    run_dir = os.path.join(ATTN_RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)
    log(f"[{run_id}] device={device} chance={1/S*100:.2f}% bar=95% on every k {OFFSETS}")
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"id": run_id, "environment": "attn",
                   "created": ts.isoformat(timespec="seconds"),
                   "config": {**cfg, "device": device, "population": cfg["pop"],
                              "offsets": list(OFFSETS)},
                   "started": {"population": cfg["pop"],
                               "generations": cfg["generations"],
                               "notes": "attn_copy_v1 (card: LM_ATTENTION_COMPONENT.md)"},
                   "status": "running"}, f, indent=2)
    open(os.path.join(run_dir, "history.jsonl"), "w").close()

    pop = AttnPopulation(cfg, device)
    if resume:
        load_population(pop, resume)
        log(f"resumed from {resume}")

    annealed = False
    n_active = cfg.get("start_offsets", 1)     # curriculum funnel (§IV.6):
    mastery_streak = 0                         # offsets unlock when earned
    # fine-grained funnel (§IV.6): length grows first (attention over 3-4
    # positions has dense reward), then offsets unlock, then full length.
    LADDER = [(1, 3, 5), (1, 4, 8), (1, 6, 12), (1, 8, 24),
              (2, 8, 24), (2, 12, 32), (3, 12, 32), (3, 16, 48),
              (4, 16, 48), (4, 24, 64), (5, 24, 64)]
    stage = cfg.get("start_stage", 0)
    for gen in range(cfg["generations"]):
        n_active, l_lo, l_hi = LADDER[stage]
        active = OFFSETS[:n_active]
        ep = make_episodes(cfg["n_per_k"], rng, offsets=active,
                           l_lo=l_lo, l_hi=l_hi)
        fit, acc_k = pop.evaluate(*ep)
        starved = pop.step_selection(fit, cfg)
        # rung gate: best genome ≥80% on every active offset for 5
        # consecutive gens → the environment poses the next question
        b = int(fit.argmax())
        if stage < len(LADDER) - 1:
            if float(acc_k[b].min()) >= cfg.get("gate", 0.80):
                mastery_streak += 1
                if mastery_streak >= 5:
                    stage += 1
                    mastery_streak = 0
                    log(f"gen {gen}: LADDER RUNG {stage} -> "
                        f"offsets {OFFSETS[:LADDER[stage][0]]} L {LADDER[stage][1]}-{LADDER[stage][2]}")
            else:
                mastery_streak = 0
        if not annealed and gen >= cfg["anneal_after"] * cfg["generations"]:
            pop.mut_scale.mul_(0.5).clamp_(min=0.02)
            annealed = True
        if gen % cfg["log_every"] == 0 or gen == cfg["generations"] - 1:
            accs = [round(float(a), 3) for a in acc_k[b]]
            rec = {"gen": gen,
                   "fitness": {"best": round(float(fit[b]), 4),
                               "mean": round(float(fit.mean()), 4)},
                   "best": {"score": round(float(acc_k[b].mean()), 4),
                            "per_k": accs, "active_k": list(active)},
                   "starved": starved}
            with open(os.path.join(run_dir, "history.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"gen {gen:5d} soft {rec['fitness']['best']:+.4f} "
                f"active{list(active)} per-k {accs} starved {starved:3d}")

    # held-out verification: fresh episodes, unseen rng stream
    ep_ho = make_episodes(60, np.random.default_rng(31337))
    fit_ho, acc_ho = pop.evaluate(*ep_ho)
    b = int(fit_ho.argmax())
    per_k = {str(k): round(float(acc_ho[b, j]), 4) for j, k in enumerate(OFFSETS)}
    passed = all(v >= 0.95 for v in per_k.values())
    save_population(pop, os.path.join(run_dir, "checkpoint.npz"), gen=cfg["generations"])
    summary = {"id": run_id, "environment": "attn", "status": "finished",
               "finished": datetime.datetime.now().isoformat(timespec="seconds"),
               "gen": cfg["generations"],
               "best": {"score": round(float(acc_ho[b].mean()), 4), "per_k": per_k},
               "bar_95_all_k": passed, "checkpoint": "checkpoint.npz"}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    cfgj = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))
    cfgj["status"] = "finished"
    json.dump(cfgj, open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8"), indent=2)
    log(f"HELD-OUT per-k: {per_k} -> bar(95% all k): {'PASSED' if passed else 'not yet'}")
    try:
        import agent_board
        agent_board.post_run_event("attn", {"type": "done", "run_id": run_id,
                                            "best": summary["best"]})
    except Exception:
        pass
    return run_id, run_dir, pop, summary


if __name__ == "__main__":
    run()
