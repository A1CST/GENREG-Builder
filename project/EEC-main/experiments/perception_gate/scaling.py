"""SCALE the two-timescale organism: does experience-perception + evolved policy still generalise to
never-seen surface forms as the MEANING SPACE grows from ~14 to 200+ meanings? Self-contained, with a
fast (vectorised) co-occurrence builder so large scales are feasible."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "scaling_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"
M, GP, DIM = 6, 4, 64                                          # words/group, groups/topic, embedding dim


def build(G, T, seed, poly=0.2, noise=3, zipf=True):
    rng = np.random.default_rng(seed); V = G * M
    group = np.repeat(np.arange(G), M)
    held = np.zeros(V, bool)
    for g in range(G):
        tk = np.where(group == g)[0]; held[tk[rng.choice(M, 2, replace=False)]] = True
    train_syn = [np.where((group == g) & ~held)[0] for g in range(G)]
    held_syn = [np.where((group == g) & held)[0] for g in range(G)]
    sec = {tk: int(rng.integers(G)) for tk in range(V) if rng.random() < poly}
    tg = [rng.choice(G, GP, replace=False) for _ in range(T)]
    tfreq = (1.0 / (np.arange(1, T + 1) ** 1.1)) if zipf else np.ones(T); tfreq /= tfreq.sum()
    Vtot = V + (noise and 50)

    def wpick(pool):
        if zipf:
            w = 1.0 / (np.arange(1, len(pool) + 1) ** 1.1); w /= w.sum(); return pool[rng.choice(len(pool), p=w)]
        return rng.choice(pool)

    def topic(pool):
        t = int(rng.choice(T, p=tfreq)); s = [wpick(pool[g]) for g in tg[t]]
        if noise: s += list(V + rng.integers(50, size=noise))
        return np.array(s), t

    def synonymy():
        g = int(rng.integers(G)); s = list(rng.choice(train_syn[g], 3)) + [int(rng.choice(held_syn[g]))]
        cand = [k for k in sec if sec[k] == g]
        if cand: s.append(int(rng.choice(cand)))
        return np.array(s)

    return dict(V=Vtot, train_syn=train_syn, held_syn=held_syn, topic=topic, synonymy=synonymy, T=T)


def ppmi_svd(corpus, V):
    co = np.zeros((V, V), np.float32)
    for s in corpus:                                          # vectorised per-sentence outer update
        co[np.ix_(s, s)] += 1.0
    np.fill_diagonal(co, 0)
    tot = co.sum(); rk = co.sum(1, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log((co * tot) / (rk @ rk.T) + 1e-9), 0)
    U, S, _ = np.linalg.svd(ppmi.astype(np.float64))
    E = U[:, :DIM] * np.sqrt(S[:DIM]); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E


def emb(E, toks): v = E[toks].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else v


def run(G, T, seed):
    rng = np.random.default_rng(100 + seed)
    while True:
        r = rng.permutation(T)
        if np.all(r != np.arange(T)): break                   # transform policy: reply != meaning
    w = build(G, T, seed)
    Xtr = [w["topic"](w["train_syn"]) for _ in range(40 * T)]
    Xte = [w["topic"](w["held_syn"]) for _ in range(1000)]
    nsyn = 220 * G                                            # experience scaled with the meaning space
    corpus = [x for x, _ in Xtr] + [w["synonymy"]() for _ in range(nsyn)]
    E = ppmi_svd(corpus, w["V"])
    cent = np.stack([np.mean([emb(E, x) for x, t in Xtr if t == k] or [np.zeros(DIM)], 0) for k in range(T)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    two = np.mean([r[int((cent @ emb(E, x)).argmax())] == r[t] for x, t in Xte])
    return float(two), nsyn


if __name__ == "__main__":
    SCALES = [(18, 14), (36, 30), (60, 50), (96, 80), (150, 120), (220, 180)]
    out("SCALING the two-timescale organism: held-out reply accuracy as the meaning space grows.")
    out(f"{'meanings':>9} {'vocab':>7} {'chance':>8} {'experience':>11} | {'held-out reply accuracy':>24}")
    out("=" * 76)
    res = []
    for G, T in SCALES:
        a, nsyn = zip(*[run(G, T, s) for s in range(2)])
        res.append((T, np.mean(a))); out(f"{T:>9} {G*M:>7} {1/T:>8.3f} {nsyn[0]:>11} | {ms(a):>24}")
    out("=" * 76)
    out("READING: if held-out reply accuracy stays high (>> chance) as meanings grow to 180+, the")
    out("architecture SCALES: experience-perception generalises and the policy maps it, at any size,")
    out("given experience scaled to the meaning space.")
    out("done"); LOG.close()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ts = [t for t, _ in res]
    plt.figure(figsize=(8, 5))
    plt.plot(Ts, [a for _, a in res], "o-", color="#1b5e9e", lw=2.5, label="two-timescale (held-out reply)")
    plt.plot(Ts, [1 / t for t in Ts], ":", color="#999", label="chance")
    plt.xlabel("number of meanings (meaning-space size)"); plt.ylabel("held-out reply accuracy"); plt.ylim(0, 1)
    plt.title("Does the two-timescale organism scale?", weight="bold"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "scaling.png"), dpi=130); print("saved scaling.png")
