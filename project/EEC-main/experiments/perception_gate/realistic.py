"""Scale distributional/experience comprehension toward REALISTIC language structure, and map the
experience-vs-comprehension curve as it gets harder.

Builds on distributional.py (synonymy: generalise to held-out surface forms). Adds the messiness real
corpora have, one knob at a time:
  - NOISE   : filler/off-topic tokens padding every sentence (dilutes the signal).
  - ZIPF    : topic + within-group word frequencies are Zipfian (rare synonyms get little exposure).
  - POLYSEMY: a fraction of words belong to TWO meaning-groups (context-dependent / ambiguous).
For each realism level we sweep total experience and measure held-out-synonym comprehension. Question:
does the distributional engine still generalise as the world gets language-like, and how much more
exposure does it need?
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G, M, T, GP, DIM = 18, 6, 14, 4, 56
NOISE_V = 40
LOG = open(os.path.join(HERE, "realistic_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def build(cfg, seed):
    rng = np.random.default_rng(seed)
    V = G * M
    group = np.repeat(np.arange(G), M)
    held = np.zeros(V, bool)
    for g in range(G):
        toks = np.where(group == g)[0]; held[toks[rng.choice(M, 2, replace=False)]] = True
    train_syn = [np.where((group == g) & ~held)[0] for g in range(G)]
    held_syn = [np.where((group == g) & held)[0] for g in range(G)]
    sec = {}                                                  # polysemy: token -> secondary group
    if cfg.get("poly", 0):
        for tk in range(V):
            if rng.random() < cfg["poly"]:
                sec[tk] = int(rng.integers(G))
    tg = [rng.choice(G, GP, replace=False) for _ in range(T)]
    # frequencies
    tfreq = 1.0 / (np.arange(1, T + 1) ** 1.1) if cfg.get("zipf") else np.ones(T)
    tfreq /= tfreq.sum()
    def wpick(pool):
        if cfg.get("zipf"):
            w = 1.0 / (np.arange(1, len(pool) + 1) ** 1.1); w /= w.sum()
            return pool[rng.choice(len(pool), p=w)]
        return rng.choice(pool)
    Vtot = V + (NOISE_V if cfg.get("noise") else 0)

    def topic_sent(pool):
        t = int(rng.choice(T, p=tfreq))
        s = [wpick(pool[g]) for g in tg[t]]
        if cfg.get("noise"):
            s += list(V + rng.integers(NOISE_V, size=cfg["noise"]))
        return np.array(s), t

    def synonymy_sent():
        g = int(rng.integers(G))
        s = list(rng.choice(train_syn[g], 3)) + [rng.choice(held_syn[g])]
        if sec:                                              # polysemy: also link a secondary-group pair
            cand = [k for k in sec if sec[k] == g]
            if cand: s.append(int(rng.choice(cand)))
        return np.array(s)

    return dict(V=Vtot, tg=tg, train_syn=train_syn, held_syn=held_syn, topic_sent=topic_sent,
                synonymy_sent=synonymy_sent, tfreq=tfreq, wpick=wpick, cfg=cfg)


def ppmi_svd(corpus, V):
    co = np.zeros((V, V), np.float32)
    for s in corpus:
        for a in s:
            for b in s:
                if a != b: co[a, b] += 1
    tot = co.sum(); rk = co.sum(1, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log((co * tot) / (rk @ rk.T) + 1e-9), 0)
    U, S, _ = np.linalg.svd(ppmi.astype(np.float64))
    E = U[:, :DIM] * np.sqrt(S[:DIM]); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E


def emb(E, toks): v = E[toks].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else v


def run_cfg(cfg, nsyn, seed):
    w = build(cfg, seed)
    Xtr = [w["topic_sent"](w["train_syn"]) for _ in range(2500)]
    Xte = [w["topic_sent"](w["held_syn"]) for _ in range(1000)]
    corpus = [x for x, _ in Xtr] + [w["synonymy_sent"]() for _ in range(nsyn)]
    E = ppmi_svd(corpus, w["V"])
    cent = np.stack([np.mean([emb(E, x) for x, t in Xtr if t == k] or [np.zeros(DIM)], 0) for k in range(T)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    return float(np.mean([int((cent @ emb(E, x)).argmax()) == t for x, t in Xte]))


if __name__ == "__main__":
    CONDS = [("clean", {}), ("+noise", {"noise": 4}), ("+zipf", {"zipf": 1}),
             ("+polysemy", {"poly": 0.3}), ("realistic (all)", {"noise": 4, "zipf": 1, "poly": 0.3})]
    EXP = [1000, 4000, 12000, 30000]
    out(f"REALISTIC distributional comprehension. {G} groups x {M} words, {T} topics (chance {1/T:.3f}). "
        "held-out-synonym generalisation vs experience.")
    out(f"{'condition':>17} | " + " | ".join(f"{e:>10} exp" for e in EXP))
    out("=" * 78)
    res = {}
    for name, cfg in CONDS:
        row = []
        for e in EXP:
            a = [run_cfg(cfg, e, s) for s in range(2)]; row.append(np.mean(a))
        res[name] = row
        out(f"{name:>17} | " + " | ".join(f"{v:>13.3f}" for v in row))
    out("=" * 78)
    out("READING: does the distributional engine still generalise as structure gets language-like, and")
    out("how much more experience does each realism factor cost?")
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    for name in res:
        plt.plot(EXP, res[name], "o-", lw=2, label=name)
    plt.axhline(1 / T, color="#999", ls=":", label="chance")
    plt.xscale("log"); plt.xlabel("experience (synonymy exposures)"); plt.ylabel("held-out comprehension")
    plt.title("Distributional comprehension vs experience, by realism", weight="bold")
    plt.legend(); plt.grid(alpha=.3); plt.ylim(0, 1); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "realistic.png"), dpi=130); print("saved realistic.png")
