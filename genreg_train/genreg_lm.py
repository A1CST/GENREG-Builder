"""lm_char_v1 — recurrent char-level GENREG substrate (stage 1).

Model card: documentation/LM_STAGE1_SUBSTRATE.md (drafted first, per §II).
Charter: no gradients (torch used in inference_mode only — tensor math, no
autograd, no pretrained weights), no gradient-like mechanics, constraints
shape the landscape, information flows (recurrent state; fresh windows each
generation).

Architecture (the documented A_89/A_98 structural wins):
    step t:  x = [E[c_t], E[c_{t-1}], tanh(h_prev)]
             h = act_per_neuron(x @ W_in + b_h)          # 8-function catalog
             logits_{t+1} = h @ W_out + b_out
Fitness = mean log softmax(logits)[target] over fresh windows (soft,
multiplicative). Energy homeostasis culls independently of tournament rank.
"""

import datetime
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS_PATH = os.path.join(ROOT, "project", "EEC-main", "engine", "corpus.txt")
LM_RUNS = os.path.join(ROOT, "runs", "lm")

# charset: space + lowercase + digits-as-# + basic punctuation + rare bucket
CHARS = " abcdefghijklmnopqrstuvwxyz.,;:'\"!?-#"
RARE = len(CHARS)                    # id for anything else
V = len(CHARS) + 1

N_ACT = 8                            # per-neuron activation catalog size


def load_char_corpus():
    """Corpus as char ids over the fixed charset (digits → '#', other → rare)."""
    with open(CORPUS_PATH, "rb") as fh:
        text = fh.read().decode("utf-8", errors="replace").lower()
    text = (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("—", "-").replace("–", "-"))
    lut = np.full(1114112, RARE, dtype=np.int64)
    for i, c in enumerate(CHARS):
        lut[ord(c)] = i
    for d in "0123456789":
        lut[ord(d)] = CHARS.index("#")
    for ws in "\n\r\t":
        lut[ord(ws)] = CHARS.index(" ")
    ids = lut[np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)]
    return ids


def sample_windows(ids, n, seq_len, rng):
    starts = rng.choice(len(ids) - seq_len - 1, size=n, replace=False)
    return ids[starts[:, None] + np.arange(seq_len + 1)[None, :]]   # (n, T+1)


def baselines(ids, seq_len, rng, n_train=200000, n_test=20000):
    """§VII bars: majority, bigram, trigram top-1/top-5 on a random window split."""
    W = sample_windows(ids, n_train + n_test, seq_len, rng)
    tr, te = W[:n_train], W[n_train:]
    p_tr = tr[:, :-1].ravel()
    y_tr = tr[:, 1:].ravel()
    maj = np.bincount(y_tr, minlength=V).argmax()

    big = np.zeros((V, V), np.int64)
    np.add.at(big, (p_tr, y_tr), 1)
    tri = np.zeros((V * V, V), np.int32)
    p2 = tr[:, :-2].ravel() * V + tr[:, 1:-1].ravel()
    np.add.at(tri, (p2, tr[:, 2:].ravel()), 1)

    p_te, y_te = te[:, :-1].ravel(), te[:, 1:].ravel()
    out = {"majority": float(np.mean(y_te == maj)),
           "bigram_top1": float(np.mean(big.argmax(1)[p_te] == y_te))}
    big_top5 = np.argsort(big, axis=1)[:, -5:]
    out["bigram_top5"] = float(np.mean((big_top5[p_te] == y_te[:, None]).any(1)))
    q2 = te[:, :-2].ravel() * V + te[:, 1:-1].ravel()
    y3 = te[:, 2:].ravel()
    tri_pred = tri[q2].argmax(1)
    seen = tri[q2].sum(1) > 0
    tri_pred[~seen] = big.argmax(1)[te[:, 1:-1].ravel()[~seen]]     # backoff
    out["trigram_top1"] = float(np.mean(tri_pred == y3))
    return out


# ---------------------------------------------------------------------------
# Population (torch, inference-mode only — never autograd)
# ---------------------------------------------------------------------------
class LMPopulation:
    def __init__(self, cfg, device):
        import torch
        self.t = torch
        self.cfg = cfg
        self.dev = device
        P, D, H = cfg["pop"], cfg["embed"], cfg["hidden"]
        g = torch.Generator(device="cpu").manual_seed(cfg["seed"])
        mk = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).to(device)
        self.E = mk(P, V, D, scale=0.1)
        self.W_in = mk(P, 2 * D + H, H, scale=1.0 / np.sqrt(2 * D + H))
        self.b_h = torch.zeros(P, H, device=device)
        self.W_out = mk(P, H, V, scale=1.0 / np.sqrt(H))
        self.b_out = torch.zeros(P, V, device=device)
        self.act = torch.randint(0, N_ACT, (P, H), generator=g).to(device)
        self.mut_rate = torch.full((P,), 0.05, device=device)
        self.mut_scale = torch.full((P,), 0.05, device=device)
        self.energy = torch.full((P,), 1.0, device=device)
        self.age = torch.zeros(P, dtype=torch.long, device=device)
        # EMA-smoothed fitness for selection/energy (§IV.7: noise-driven
        # culling destroys ratchets). NaN = no estimate yet (first eval sets it).
        self.fit_ema = torch.full((P,), float("nan"), device=device)
        # -- assembly: evolved copy-attention channel (cfg["attention"]) ----
        # q from h_t, keys from past char embeddings, evolved relative bias,
        # retrieved content read out through the TIED embedding table, all
        # behind gate alpha init 0 — gen 0 behaves exactly like the substrate.
        self.attention = bool(cfg.get("attention"))
        if self.attention:
            Aa = cfg.get("attn_dim", 32)
            self.Aa = Aa
            self.Wq_a = mk(P, H, Aa, scale=1.0 / np.sqrt(H))
            self.Wk_a = mk(P, D, Aa, scale=1.0 / np.sqrt(D))
            self.B_rel = torch.zeros(P, cfg["seq_len"] + 1, device=device)
            self.alpha = torch.zeros(P, device=device)
        self.P, self.D, self.H = P, D, H

    def _act(self, pre, act=None):
        """Per-neuron activation from the 8-catalog. pre: (P, B, H)."""
        t = self.t
        a = (act if act is not None else self.act)[:, None, :].expand_as(pre)
        out = t.tanh(pre)                                        # 0 tanh
        out = t.where(a == 1, t.sigmoid(pre), out)               # 1 sigmoid
        out = t.where(a == 2, t.relu(pre).clamp(max=4.0), out)   # 2 relu (capped)
        out = t.where(a == 3, t.sin(pre), out)                   # 3 sin
        out = t.where(a == 4, t.exp(-pre * pre), out)            # 4 gaussian
        out = t.where(a == 5, pre.clamp(-4.0, 4.0), out)         # 5 identity (clipped)
        out = t.where(a == 6, pre / (1 + pre.abs()), out)        # 6 softsign
        out = t.where(a == 7, pre.abs().clamp(max=4.0), out)     # 7 abs
        return out

    def evaluate(self, windows, warmup):
        """windows: (B, T+1) numpy. Returns (fitness, top1, top5) per genome."""
        t = self.t
        with t.inference_mode():
            w = t.as_tensor(windows, device=self.dev)
            B, T1 = w.shape
            T = T1 - 1
            h = t.zeros(self.P, B, self.H, device=self.dev)
            logp_sum = t.zeros(self.P, device=self.dev)
            top1 = t.zeros(self.P, device=self.dev)
            top5 = t.zeros(self.P, device=self.dev)
            steps = 0
            if self.attention:                     # precompute keys per char
                E_all = self.E[:, w[:, :T], :]                  # (P,B,T,D)
                K_all = t.einsum("pbtd,pda->pbta", E_all, self.Wk_a)
                # induction values: the char that FOLLOWED each position —
                # match past context, retrieve its successor
                E_next = self.E[:, w[:, 1:T + 1], :]            # (P,B,T,D)
            for i in range(T):
                cur, prev = w[:, i], w[:, max(i - 1, 0)]
                e_cur = self.E[:, cur, :]                       # (P, B, D)
                e_prev = self.E[:, prev, :]
                x = t.cat([e_cur, e_prev, t.tanh(h)], dim=2)    # (P, B, 2D+H)
                h = self._act(t.einsum("pbi,pih->pbh", x, self.W_in)
                              + self.b_h[:, None, :])
                if i < warmup:
                    continue
                logits = (t.einsum("pbh,phv->pbv", h, self.W_out)
                          + self.b_out[:, None, :])
                if self.attention and i > 0:
                    # keys/values over j ≤ i−1 ONLY: the value at j is char
                    # j+1, so including j=i would hand the model its target
                    q = t.einsum("pbh,pha->pba", h, self.Wq_a)
                    scores = (t.einsum("pba,pbja->pbj", q, K_all[:, :, :i])
                              / np.sqrt(self.Aa))
                    d_rel = (i - t.arange(i, device=self.dev)).clamp(
                        0, self.B_rel.shape[1] - 1)             # (i,)
                    scores = scores + self.B_rel[:, d_rel][:, None, :]
                    att = scores.softmax(dim=2)
                    ctx = t.einsum("pbj,pbjd->pbd", att, E_next[:, :, :i])
                    copy_logits = t.einsum("pbd,pvd->pbv", ctx, self.E)
                    logits = logits + self.alpha[:, None, None] * copy_logits
                lp = logits.log_softmax(dim=2)
                tgt = w[:, i + 1]                               # (B,)
                lp_t = lp[:, t.arange(B, device=self.dev), tgt] # (P, B)
                logp_sum += lp_t.mean(dim=1)
                pred5 = logits.topk(5, dim=2).indices           # (P, B, 5)
                top1 += (pred5[:, :, 0] == tgt[None, :]).float().mean(dim=1)
                top5 += (pred5 == tgt[None, :, None]).any(dim=2).float().mean(dim=1)
                steps += 1
            return (logp_sum / steps), (top1 / steps), (top5 / steps)

    def evaluate_rollout(self, windows, warmup, R, temperature=0.8, seed=None):
        """Closed-loop fitness (stage 4, the DiffEvo unrolled-training lesson):
        after a teacher-forced warmup, each genome consumes ITS OWN SAMPLED
        chars for R steps while being scored against the TRUE corpus
        continuation at every position (the Goodhart anchor). Survival =
        staying calibrated on your own output distribution.

        Returns (fitness = mean log-prob over rollout steps, top1, top5)."""
        t = self.t
        with t.inference_mode():
            g = t.Generator(device=self.dev)
            g.manual_seed(int(seed) if seed is not None else 0)
            w = t.as_tensor(windows, device=self.dev)
            B, T1 = w.shape
            T = min(T1 - 1, warmup + R)
            h = t.zeros(self.P, B, self.H, device=self.dev)
            pi = t.arange(self.P, device=self.dev)[:, None]     # genome index
            # per-genome running inputs (diverge once sampling starts)
            cur = w[:, 0][None, :].expand(self.P, B).contiguous()
            prev = cur.clone()
            logp_sum = t.zeros(self.P, device=self.dev)
            top1 = t.zeros(self.P, device=self.dev)
            top5 = t.zeros(self.P, device=self.dev)
            steps = 0
            for i in range(T):
                e_cur = self.E[pi, cur, :]                      # (P, B, D)
                e_prev = self.E[pi, prev, :]
                x = t.cat([e_cur, e_prev, t.tanh(h)], dim=2)
                h = self._act(t.einsum("pbi,pih->pbh", x, self.W_in)
                              + self.b_h[:, None, :])
                logits = (t.einsum("pbh,phv->pbv", h, self.W_out)
                          + self.b_out[:, None, :])
                tgt = w[:, i + 1]                               # TRUE next char
                # BLENDED objective: score the teacher-forced segment too
                # (after a short state-fill) — pure-rollout fitness bred
                # hedging (flat marginals score "safely" on drifted context,
                # open-loop sharpness collapsed 31.9%→23.1%). Both regimes
                # must pay simultaneously.
                score_now = i >= min(4, warmup)
                if score_now:
                    lp = logits.log_softmax(dim=2)
                    lp_t = lp[:, t.arange(B, device=self.dev), tgt]
                    logp_sum += lp_t.mean(dim=1)
                    pred5 = logits.topk(5, dim=2).indices
                    top1 += (pred5[:, :, 0] == tgt[None, :]).float().mean(dim=1)
                    top5 += (pred5 == tgt[None, :, None]).any(dim=2).float().mean(dim=1)
                    steps += 1
                if i >= warmup:                                 # rollout zone
                    # next input = the genome's OWN sample (experience)
                    probs = (logits / max(temperature, 1e-6)).softmax(dim=2)
                    nxt = t.multinomial(probs.reshape(-1, probs.shape[-1]),
                                        1, generator=g).reshape(self.P, B)
                    prev, cur = cur, nxt
                else:                                           # teacher-forced
                    prev = cur
                    cur = tgt[None, :].expand(self.P, B).contiguous()
            return (logp_sum / max(steps, 1)), (top1 / max(steps, 1)), (top5 / max(steps, 1))

    # -- reproduction --------------------------------------------------------
    def _genome_tensors(self):
        names = ["E", "W_in", "b_h", "W_out", "b_out", "act",
                 "mut_rate", "mut_scale"]
        if self.attention:
            names += ["Wq_a", "Wk_a", "B_rel", "alpha"]
        return names

    def _clone_into(self, dst, src):
        for name in self._genome_tensors():
            getattr(self, name)[dst] = getattr(self, name)[src]

    def _mutate(self, idx, rng_t):
        t = self.t
        # self-adaptive step sizes first (log-normal), with the hard floor
        self.mut_rate[idx] = (self.mut_rate[idx]
                              * t.exp(0.2 * t.randn(len(idx), device=self.dev))
                              ).clamp(0.005, 0.2)
        self.mut_scale[idx] = (self.mut_scale[idx]
                               * t.exp(0.2 * t.randn(len(idx), device=self.dev))
                               ).clamp(0.02, 0.5)
        weighted = [("E", 1.0), ("W_in", 1.0), ("b_h", 0.5),
                    ("W_out", 1.0), ("b_out", 0.5)]
        if self.attention:
            weighted += [("Wq_a", 1.0), ("Wk_a", 1.0), ("B_rel", 0.5),
                         ("alpha", 0.3)]
        for name, per in weighted:
            ten = getattr(self, name)
            sub = ten[idx]
            mask = (t.rand(sub.shape, device=self.dev)
                    < self.mut_rate[idx].view(-1, *([1] * (sub.dim() - 1))))
            noise = (t.randn(sub.shape, device=self.dev)
                     * self.mut_scale[idx].view(-1, *([1] * (sub.dim() - 1))) * per)
            ten[idx] = sub + mask * noise
        # activation ids: rare reassignment
        am = (t.rand(self.act[idx].shape, device=self.dev)
              < (self.mut_rate[idx] / 4).view(-1, 1))
        rnd = t.randint(0, N_ACT, self.act[idx].shape, device=self.dev)
        self.act[idx] = t.where(am, rnd, self.act[idx])

    def step_selection(self, fitness, cfg):
        """Energy update -> cull -> tournament refill (maturation-gated).
        Selection and energy use the EMA-smoothed fitness, not the raw
        single-batch measurement."""
        t = self.t
        with t.inference_mode():
            a = cfg.get("fit_ema", 0.75)
            fresh = t.isnan(self.fit_ema)
            self.fit_ema = t.where(fresh, fitness,
                                   a * self.fit_ema + (1 - a) * fitness)
            fitness = self.fit_ema
            med = fitness.median()
            self.energy = (self.energy * cfg["energy_decay"]
                           + cfg["energy_gain"] * (fitness - med)
                           ).clamp(0.0, cfg["energy_max"])
            dead = self.energy < cfg["energy_floor"]
            n_dead = int(dead.sum())
            alive = (~dead).nonzero().squeeze(1)
            # breeders: alive AND mature (survived >= 1 full generation)
            mature = alive[self.age[alive] >= 1]
            pool = mature if len(mature) >= 2 else alive
            if n_dead and len(pool):
                dead_idx = dead.nonzero().squeeze(1)
                k = cfg["tournament_k"]
                cand = pool[t.randint(0, len(pool), (n_dead, k), device=self.dev)]
                winners = cand[t.arange(n_dead, device=self.dev),
                               fitness[cand].argmax(dim=1)]
                self._clone_into(dead_idx, winners)
                self._mutate(dead_idx, None)
                self.energy[dead_idx] = 1.0
                self.age[dead_idx] = 0
                self.fit_ema[dead_idx] = float("nan")   # child measures fresh
            self.age += 1
            return n_dead


# ---------------------------------------------------------------------------
# Trainer / run loop with runstore-layout persistence (shows on /runs as "lm")
# ---------------------------------------------------------------------------
DEFAULTS = dict(pop=400, embed=24, hidden=64, seq_len=64, batch=96,
                generations=3000, warmup=8, seed=1, fit_ema=0.75,
                energy_decay=0.90, energy_gain=8.0, energy_floor=0.20,
                energy_max=1.5, tournament_k=3, anneal_after=0.8,
                log_every=25,
                rollout_len=0, rollout_temp=0.8)   # stage 4: closed loop


def run(cfg=None, log=print, resume=None):
    import torch
    cfg = {**DEFAULTS, **(cfg or {})}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = load_char_corpus()
    rng = np.random.default_rng(cfg["seed"])

    ts = datetime.datetime.now()
    h8 = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"{ts.strftime('%Y%m%d-%H%M%S')}-lm-{h8}"
    run_dir = os.path.join(LM_RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)

    bl = baselines(ids, cfg["seq_len"], np.random.default_rng(999))
    log(f"[{run_id}] device={device} V={V} baselines={json.dumps({k: round(v, 4) for k, v in bl.items()})}")

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"id": run_id, "environment": "lm",
                   "created": ts.isoformat(timespec="seconds"),
                   "config": {**cfg, "device": device, "population": cfg["pop"],
                              "vocab": V, "baselines": bl},
                   "started": {"population": cfg["pop"],
                               "generations": cfg["generations"],
                               "notes": "lm_char_v1 substrate (model card: LM_STAGE1_SUBSTRATE.md)"},
                   "status": "running"}, f, indent=2)
    open(os.path.join(run_dir, "history.jsonl"), "w").close()

    pop = LMPopulation(cfg, device)
    if resume:                                   # bootstrap, don't relearn
        load_population(pop, resume)
        log(f"resumed population from {resume}")

    best_hist = []
    annealed = False
    R = int(cfg.get("rollout_len", 0))
    for gen in range(cfg["generations"]):
        W = sample_windows(ids, cfg["batch"], cfg["seq_len"], rng)   # fresh every gen
        if R > 0:      # stage 4: survive generating your own future
            fit, top1, top5 = pop.evaluate_rollout(
                W, cfg["warmup"], R, cfg.get("rollout_temp", 0.8), seed=gen)
        else:
            fit, top1, top5 = pop.evaluate(W, cfg["warmup"])
        starved = pop.step_selection(fit, cfg)
        if not annealed and gen >= cfg["anneal_after"] * cfg["generations"]:
            pop.mut_scale.mul_(0.5).clamp_(min=0.02)
            annealed = True
            log(f"gen {gen}: annealed mut_scale (late fine-tuning)")
        if gen % cfg["log_every"] == 0 or gen == cfg["generations"] - 1:
            b = int(fit.argmax())
            rec = {"gen": gen,
                   "fitness": {"best": round(float(fit[b]), 4),
                               "mean": round(float(fit.mean()), 4)},
                   "best": {"score": round(float(top1[b]), 4),
                            "top5": round(float(top5[b]), 4)},
                   "starved": starved,
                   "mut_scale_mean": round(float(pop.mut_scale.mean()), 4)}
            with open(os.path.join(run_dir, "history.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"gen {gen:5d} soft {rec['fitness']['best']:+.4f} "
                f"top1 {rec['best']['score']*100:5.2f}% top5 {rec['best']['top5']*100:5.2f}% "
                f"starved {starved:3d} ({starved/cfg['pop']*100:4.1f}%)")
            best_hist.append(rec)

    # held-out verification (§VII): fresh windows never used in training
    W_ho = sample_windows(ids, 256, cfg["seq_len"], np.random.default_rng(31337))
    fit_ho, top1_ho, top5_ho = pop.evaluate(W_ho, cfg["warmup"])
    b = int(fit_ho.argmax())
    ro_summary = None
    if R > 0:      # also verify in the CLOSED LOOP the model was bred for
        fit_ro, top1_ro, top5_ro = pop.evaluate_rollout(
            W_ho, cfg["warmup"], R, cfg.get("rollout_temp", 0.8), seed=777)
        br = int(fit_ro.argmax())
        ro_summary = {"soft": round(float(fit_ro[br]), 4),
                      "top1": round(float(top1_ro[br]), 4),
                      "top5": round(float(top5_ro[br]), 4), "R": R}
        log(f"HELD-OUT rollout(R={R}): soft {ro_summary['soft']} "
            f"top1 {ro_summary['top1']*100:.2f}% top5 {ro_summary['top5']*100:.2f}%")
    save_population(pop, os.path.join(run_dir, "checkpoint.npz"), gen=cfg["generations"])
    summary = {"id": run_id, "environment": "lm", "status": "finished",
               "finished": datetime.datetime.now().isoformat(timespec="seconds"),
               "gen": cfg["generations"],
               "best": {"score": round(float(top1_ho[b]), 4),
                        "top5": round(float(top5_ho[b]), 4),
                        "soft": round(float(fit_ho[b]), 4)},
               "baselines": bl, "checkpoint": "checkpoint.npz",
               "rollout": ro_summary,
               "vs_bigram_pts": round((float(top1_ho[b]) - bl["bigram_top1"]) * 100, 2)}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    cfgj = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))
    cfgj["status"] = "finished"
    json.dump(cfgj, open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8"), indent=2)
    log(f"HELD-OUT best: top1 {float(top1_ho[b])*100:.2f}% top5 {float(top5_ho[b])*100:.2f}% "
        f"(bigram bar {bl['bigram_top1']*100:.2f}%, majority {bl['majority']*100:.2f}%)")
    try:
        import agent_board
        agent_board.post_run_event("lm", {"type": "done", "run_id": run_id,
                                          "best": summary["best"],
                                          "accuracy": summary["best"]["score"],
                                          "bigram_accuracy": bl["bigram_top1"]})
    except Exception:
        pass
    return run_id, run_dir, pop, summary


def save_population(pop, path, gen=0):
    """ALL state (§IX): tensors, act ids, energy, mutation params, age, gen."""
    names = pop._genome_tensors() + ["energy", "age", "fit_ema"]
    np.savez_compressed(path, gen=gen, **{
        name: getattr(pop, name).cpu().numpy() for name in names})


def load_population(pop, path):
    import torch
    data = np.load(path)
    names = pop._genome_tensors() + ["energy", "age", "fit_ema"]
    for name in names:
        # substrate checkpoints lack the attention tensors — those keep
        # their transparent no-op init (alpha 0), which is the point
        if name in data:
            getattr(pop, name).copy_(torch.as_tensor(data[name], device=pop.dev))


def generate(pop, idx, prompt, length=200, temperature=0.8, seed=None):
    """Sample a continuation from one genome (inspect-actual-predictions duty)."""
    import torch
    t = torch
    rng = np.random.default_rng(seed)
    lut = {c: i for i, c in enumerate(CHARS)}
    seq = [lut.get(c, RARE) for c in prompt.lower()] or [0]
    n_prompt = len(seq)
    with t.inference_mode():
        h = t.zeros(1, 1, pop.H, device=pop.dev)
        for i in range(n_prompt + length - 1):
            cur = seq[i]
            prev = seq[i - 1] if i > 0 else seq[0]
            e_c = pop.E[idx:idx + 1, cur, :][:, None, :]
            e_p = pop.E[idx:idx + 1, prev, :][:, None, :]
            x = t.cat([e_c, e_p, t.tanh(h)], dim=2)
            h = pop._act(t.einsum("pbi,pih->pbh", x, pop.W_in[idx:idx + 1])
                         + pop.b_h[idx:idx + 1, None, :],
                         act=pop.act[idx:idx + 1])
            if i >= n_prompt - 1:                      # start emitting
                logits = (t.einsum("pbh,phv->pbv", h, pop.W_out[idx:idx + 1])
                          + pop.b_out[idx:idx + 1, None, :])[0, 0]
                p = (logits / max(temperature, 1e-6)).softmax(0).cpu().numpy()
                seq.append(int(rng.choice(V, p=p / p.sum())))
        return "".join(CHARS[i] if i < len(CHARS) else "¿" for i in seq[n_prompt:])


if __name__ == "__main__":
    run()
