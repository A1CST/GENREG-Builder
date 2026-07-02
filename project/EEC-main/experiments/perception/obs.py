"""obs.py -- OBSERVE the evolved organism, don't score it.

No accuracy, no baselines. We ask: what did this thing BECOME under the
existence constraints? What does it attend to, what structure self-organized in
its internal representation, did it carve the world into its own categories,
what is its behavioral repertoire.
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import pickle
import numpy as np

from perc import PercGenome, make_sequences_L, L
from evolve import build_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "best", "perc_best.pkl")


def kmeans(Hn, k, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    C = Hn[rng.choice(len(Hn), k, replace=False)].copy()
    for _ in range(iters):
        d = ((Hn[:, None, :] - C[None]) ** 2).sum(2)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = Hn[m].mean(0)
    return a


def main():
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences_L(ids)
    with open(CKPT, "rb") as f:
        saved = pickle.load(f)
    g = PercGenome.__new__(PercGenome)
    g.E, g.g, g.W1, g.b1, g.W2, g.b2 = saved["genome"]
    p = g.perception()
    print(f"=== organism from gen {saved['gen']} ===")
    print(f"PERCEPTION (what it looks at, oldest->newest of {L}):")
    for i in range(L):
        bar = "#" * int(round(p[i] * 20))
        print(f"  pos {i} (t-{L-i}): {p[i]:.2f} {bar}")
    print(f"  total perception load: {p.sum():.2f}/{L}")

    rng = np.random.default_rng(1)
    N = 20000
    idx = rng.permutation(len(X))[:N]
    Xs, ys = X[idx], y[idx]
    # internal representation (the 64-d hidden code) for each context
    pe = (g.E[Xs] * p[None, :, None].astype(np.float32)).reshape(N, -1)
    H = np.tanh(pe @ g.W1 + g.b1)
    logits = H @ g.W2 + g.b2
    preds = logits.argmax(1)

    # how much structure is in the representation?
    Hc = H - H.mean(0)
    cov = Hc.T @ Hc / N
    ev = np.linalg.eigvalsh(cov)[::-1]
    cum = np.cumsum(ev) / ev.sum()
    rank90 = int(np.searchsorted(cum, 0.90)) + 1
    print(f"\nREPRESENTATION: 64-d hidden, {rank90} dims hold 90% of variance "
          f"(active units: {(H.std(0) > 0.05).sum()}/64)")

    # behavioral repertoire
    uniq, cnt = np.unique(preds, return_counts=True)
    order = np.argsort(cnt)[::-1]
    print(f"\nBEHAVIORAL REPERTOIRE: {len(uniq)} distinct outputs over {N} contexts")
    print("  top behaviors: " + ", ".join(
        f"{vocab[uniq[o]]!r}={100*cnt[o]/N:.1f}%" for o in order[:8]))

    # did it carve its own categories? cluster the hidden code.
    Hn = (H - H.mean(0)) / (H.std(0) + 1e-6)
    a = kmeans(Hn, 10)
    print("\nSELF-ORGANIZED CATEGORIES (clusters of internal state):")
    for j in range(10):
        m = a == j
        if m.sum() < 20:
            continue
        lastw = Xs[m][:, -1]
        lw, lc = np.unique(lastw, return_counts=True)
        topw = lw[np.argsort(lc)[::-1][:3]]
        pr = preds[m]
        pu, pc = np.unique(pr, return_counts=True)
        topp = pu[np.argsort(pc)[::-1][:3]]
        ex = " ".join(vocab[w] for w in Xs[m][0])
        print(f"  cluster {j} ({100*m.mean():4.1f}%): "
              f"last-word~[{', '.join(repr(vocab[w]) for w in topw)}] "
              f"-> emits~[{', '.join(repr(vocab[w]) for w in topp)}]")
        print(f"      e.g. ...{ex[-60:]}")


if __name__ == "__main__":
    main()
