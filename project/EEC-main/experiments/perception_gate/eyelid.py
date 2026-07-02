"""THE EYELID -- does gating perception build a GENERALISING internal model (comprehension)?

v2, after a flawed first run (distinctive-token world let surface bag-of-words win 0.95, and a plain
Hebbian rule collapsed). Fixes:
  - SHARED-TOKEN world: tokens belong to several topics (each token ambiguous); the topic is only
    identifiable from which tokens CO-OCCUR. So surface (token identity) cannot cleanly classify --
    you need a model of co-occurrence structure. This is the real perception gap.
  - CONTRASTIVE local rule (pull toward context, push random negatives) -> embeddings do not collapse.
  - The EYELID = masking the CONTEXT (dropout-like): predict each token from only a fraction (1-g) of
    the other tokens. g=0 = full sight; higher g = must model from less. Local, no backprop, within-life.

Test = classify the topic of HELD-OUT sentences (chance 1/T). If the gated representation generalises
where surface fails, and there is a Goldilocks in g, gating builds the comprehending model.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
T, V, D, K, PER = 12, 60, 40, 3, 8          # tiny ambiguous sentences, heavy overlap -> surface MUST fail
GATES = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75]
LOG = open(os.path.join(HERE, "eyelid_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def make_world(seed=0):
    rng = np.random.default_rng(seed)
    members = [[] for _ in range(T)]                       # each token assigned to PER topics -> shared/ambiguous
    for tok in range(V):
        for t in rng.choice(T, PER, replace=False): members[t].append(tok)
    cliques = [np.array(m) for m in members]
    return cliques


def sentences(cliques, n, seed):
    rng = np.random.default_rng(seed); y = rng.integers(T, size=n)
    X = np.stack([rng.choice(cliques[t], K, replace=True) for t in y]); return X, y


def learn(X, gate, seed=0, epochs=4, lr=0.05, negs=4):
    rng = np.random.default_rng(seed); E = rng.normal(0, 0.1, (V, D))
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    for ep in range(epochs):
        for i in rng.permutation(len(X)):
            toks = X[i]
            for j in range(K):
                others = [toks[k] for k in range(K) if k != j]
                vis = [o for o in others if rng.random() >= gate]   # eyelid: drop fraction g of the context
                if not vis: continue
                ctx = E[vis].mean(0)
                E[toks[j]] += lr * ctx                              # pull true token toward its (gated) context
                for _ in range(negs):
                    neg = int(rng.integers(V)); E[neg] -= lr * ctx * max(0.0, float(E[neg] @ ctx))
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return E


def emb(E, toks): v = E[toks].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else v


def comprehend(E, Xtr, ytr, Xte, yte):
    cent = np.stack([np.mean([emb(E, Xtr[i]) for i in np.where(ytr == t)[0][:300]], 0) for t in range(T)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    return float(np.mean([int((cent @ emb(E, x)).argmax()) == y for x, y in zip(Xte, yte)]))


def surface(Xtr, ytr, Xte, yte, n=600):
    def bag(t): b = np.zeros(V); b[t] = 1; return b
    B = np.stack([bag(x) for x in Xtr[:3000]]); yb = ytr[:3000]
    return np.mean([yb[int((B @ bag(x)).argmax())] == y for x, y in zip(Xte[:n], yte[:n])])


if __name__ == "__main__":
    out(f"THE EYELID v2: shared-token world (each token in {PER}/{T} topics). classify HELD-OUT topic "
        f"(chance {1/T:.3f}).")
    cl = make_world(0); Xtr, ytr = sentences(cl, 5000, 1); Xte, yte = sentences(cl, 1200, 2)
    base = surface(Xtr, ytr, Xte, yte)
    out(f"SURFACE baseline (always-on, no model): {base:.3f}")
    out("=" * 70)
    out(f"{'gate g':>8} | {'held-out comprehension (eyelid organism)':>42}")
    res = {}
    for g in GATES:
        a = [comprehend(learn(Xtr, g, seed=s), Xtr, ytr, Xte, yte) for s in range(3)]
        res[g] = np.mean(a); out(f"{g:>8} | {ms(a):>42}")
    out("=" * 70)
    best = max(res, key=res.get)
    out(f"best g={best} -> {res[best]:.3f}   vs g=0 {res[0.0]:.3f}   vs surface {base:.3f}")
    out("READING: eyelid (g>0) > g=0 AND > surface => gating builds the comprehending model; peak at")
    out("moderate g = Goldilocks (some blindness forces modelling, too much starves it).")
    out("done"); LOG.close()
