"""DISTRIBUTIONAL / EXPERIENCE LEARNING as the comprehension engine.

Gating was a dead end; the thing that worked was the learned MODEL. So lean in: does gradient-free
DISTRIBUTIONAL learning from experience (PPMI-SVD over co-occurrence) build comprehension that
generalises across SURFACE FORMS -- the real snack<->hungry gap?

World with synonymy: V tokens in G groups of interchangeable words (a group = words that mean the same).
Topics use GP groups; a sentence picks one word per group. Each group's words split into TRAIN words
and HELD-OUT words. The organism is exposed to: (a) topic sentences using TRAIN words, and (b) synonymy
sentences where a held-out word co-occurs with train words of its OWN group (this is how it learns
"snack" goes with "eat/meal"). TEST = topic sentences built from HELD-OUT words only.

  - SURFACE (token identity): the held-out words never appeared in topic sentences -> it cannot place
    them -> fails. (This is exactly where the always-on perception map died.)
  - DISTRIBUTIONAL model: held-out words embed near their group (via synonymy co-occurrence) -> a
    held-out sentence lands near the right topic -> it GENERALISES. And it should scale with experience.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G, M, T, GP, DIM = 16, 6, 12, 4, 48
LOG = open(os.path.join(HERE, "distributional_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"

rng0 = np.random.default_rng(0)
V = G * M
group = np.repeat(np.arange(G), M)
held = np.zeros(V, bool)
for g in range(G):
    toks = np.where(group == g)[0]; held[toks[rng0.choice(M, 2, replace=False)]] = True
train_syn = [np.where((group == g) & ~held)[0] for g in range(G)]
held_syn = [np.where((group == g) & held)[0] for g in range(G)]
topic_groups = [rng0.choice(G, GP, replace=False) for _ in range(T)]


def topic_sentences(n, pool, seed):
    rng = np.random.default_rng(seed); y = rng.integers(T, size=n)
    X = [np.array([rng.choice(pool[g]) for g in topic_groups[t]]) for t in y]
    return X, y


def synonymy_sentences(n, seed):
    rng = np.random.default_rng(seed); out_ = []
    for _ in range(n):
        g = rng.integers(G)
        s = list(rng.choice(train_syn[g], 3)) + [rng.choice(held_syn[g])]
        out_.append(np.array(s))
    return out_


def ppmi_svd(corpus):
    co = np.zeros((V, V))
    for s in corpus:
        for a in s:
            for b in s:
                if a != b: co[a, b] += 1
    tot = co.sum(); rk = co.sum(1, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log((co * tot) / (rk @ rk.T) + 1e-9), 0)
    U, S, _ = np.linalg.svd(ppmi)
    E = U[:, :DIM] * np.sqrt(S[:DIM]); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E


def emb(E, toks): v = E[toks].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else v


def comprehend(E, Xtr, ytr, Xte, yte):
    cent = np.stack([np.mean([emb(E, Xtr[i]) for i in np.where(ytr == t)[0]], 0) for t in range(T)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    return float(np.mean([int((cent @ emb(E, x)).argmax()) == y for x, y in zip(Xte, yte)]))


def surface(Xtr, ytr, Xte, yte):
    def bag(t): b = np.zeros(V); b[t] = 1; return b
    B = np.stack([bag(x) for x in Xtr]);
    return float(np.mean([ytr[int((B @ bag(x)).argmax())] == y for x, y in zip(Xte, yte)]))


if __name__ == "__main__":
    out(f"DISTRIBUTIONAL comprehension: generalise to HELD-OUT synonyms. {G} groups x {M} words, "
        f"{T} topics (chance {1/T:.3f}).")
    Xtr, ytr = topic_sentences(2500, train_syn, 1)            # exposure: topic sentences (train words)
    Xte, yte = topic_sentences(1200, held_syn, 2)             # test: same topics, HELD-OUT words only
    base = surface(Xtr, ytr, Xte, yte)
    out(f"SURFACE baseline (token identity, no model): {base:.3f}  <- held-out words unseen in topics")
    out("=" * 70)
    out("experience (# synonymy sentences) -> held-out comprehension (distributional model):")
    res = {}
    for nsyn in [0, 500, 1500, 4000, 9000]:
        accs = []
        for s in range(3):
            corpus = list(topic_sentences(2500, train_syn, 10 + s)[0]) + synonymy_sentences(nsyn, 50 + s)
            E = ppmi_svd(corpus); accs.append(comprehend(E, Xtr, ytr, Xte, yte))
        res[nsyn] = np.mean(accs); out(f"   {nsyn:>6} synonymy exposures : {ms(accs)}")
    out("=" * 70)
    out(f"no synonymy experience {res[0]:.3f} (~surface) -> rich experience {res[9000]:.3f}")
    out("READING: with no synonymy exposure the model can't bridge surface forms (~surface/chance). As")
    out("experience accumulates, held-out synonyms embed near their meaning and comprehension GENERALISES")
    out("-- distributional/experience learning is the comprehension engine, and it scales with exposure.")
    out("done"); LOG.close()
