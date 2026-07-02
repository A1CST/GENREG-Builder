"""REAL-CORPUS-grounded communication.
Vocab, frequencies and embeddings come from the actual corpus (grounding.npz).
Meanings are real English words; the listener decodes a signal into the real EMBEDDING
space and picks the nearest word -- so mistakes are forced through real semantics
(a confusion is a semantically NEAR word, not a random one). Tests whether:
  (1) grounding yields real-English naming, (2) errors follow real semantic similarity,
  (3) frequent words are communicated more reliably (Zipf), (4) sex stabilises it.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(HERE, "grounding.npz"), allow_pickle=True)
VOCAB = list(G["vocab"]); EMB = G["emb"].astype(float); FREQ = G["freq"].astype(float)
K, DIM = EMB.shape
SIM = EMB @ EMB.T                                   # cosine sim (emb already unit-norm)


def new_pop(N, rng):
    return dict(spk=rng.normal(0, 0.3, (N, K, K)), lis=rng.normal(0, 0.3, (N, K, DIM)))


def eff_spk(g, anchor):
    return g["spk"] + (anchor * np.eye(K) if anchor > 0 else 0)


def eff_lis(g, anchor):
    pr = anchor * EMB if anchor > 0 else 0          # signal s -> embedding of word s
    return g["lis"] + pr                            # (N, K, DIM)


def decode(ehat):                                   # nearest word in real embedding space
    return int(np.argmax(EMB @ ehat))


def play(g, anchor, rng, rounds=6, noise=0.0):
    N = len(g["spk"]); en = np.zeros(N)
    ES = eff_spk(g, anchor); EL = eff_lis(g, anchor)
    msample = rng.choice(K, size=N * rounds * 2, p=FREQ)   # talk about frequent things more
    mi = 0; errors = []
    for _ in range(rounds):
        perm = rng.permutation(N)
        for a in range(0, N - 1, 2):
            i, j = perm[a], perm[a + 1]
            for sp, ls in ((i, j), (j, i)):
                m = int(msample[mi]); mi += 1
                s = int(np.argmax(ES[sp, m]))
                ehat = EL[ls, s] + (rng.normal(0, noise, DIM) if noise > 0 else 0)
                mh = decode(ehat)
                if mh == m: en[sp] += 1; en[ls] += 1
                else: errors.append((m, mh))
    return en, errors


def english_usage(g, anchor):
    emit = eff_spk(g, anchor).argmax(2)
    return float((emit == np.arange(K)[None, :]).mean())


def evolve(anchor=2.0, repro="sexual", gens=250, N=40, mut=0.25, cull=0.25, seed=1, noise=0.0):
    rng = np.random.default_rng(seed); g = new_pop(N, rng)
    for t in range(gens):
        if noise > 0:
            g["spk"] += rng.normal(0, noise, g["spk"].shape)
            g["lis"] += rng.normal(0, noise, g["lis"].shape)
        en, _ = play(g, anchor, rng, noise=noise)
        Kc = max(1, int(cull * N)); order = np.argsort(en)
        worst = order[:Kc]; top = order[N - max(2, N // 3):]
        for w in worst:
            if repro == "clone":
                p = int(top[rng.integers(len(top))])
                g["spk"][w] = g["spk"][p] + rng.normal(0, mut, (K, K))
                g["lis"][w] = g["lis"][p] + rng.normal(0, mut, (K, DIM))
            else:
                pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
                ms = rng.random((K, K)) < 0.5; ml = rng.random((K, DIM)) < 0.5
                g["spk"][w] = np.where(ms, g["spk"][pa], g["spk"][pb]) + rng.normal(0, mut * 0.6, (K, K))
                g["lis"][w] = np.where(ml, g["lis"][pa], g["lis"][pb]) + rng.normal(0, mut * 0.6, (K, DIM))
    return g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"REAL-CORPUS grounding: {K} words, {DIM}-d embeddings  (chance accuracy ~ {1/K:.3f})")
    print("vocab:", " ".join(VOCAB[:18]), "...")
    print("=" * 72)
    print("T1 — does real-corpus grounding produce real-English naming?")
    for anchor in [0.0, 2.0]:
        eu = np.mean([english_usage(evolve(anchor=anchor, gens=200, seed=s), anchor) for s in range(3)])
        print(f"  anchor={anchor:.0f}: english_usage={eu:.2f}")
    print("=" * 72)
    print("T2 — do ERRORS follow real semantic similarity? (the key real-corpus test)")
    g = evolve(anchor=2.0, gens=220, seed=1)
    _, errs = play(g, 2.0, rng, rounds=40, noise=0.6)            # inject noise to elicit errors
    if errs:
        es = np.mean([SIM[m, mh] for m, mh in errs])
        rs = np.mean([SIM[m, int(rng.integers(K))] for m, _ in errs])
        print(f"  mean similarity(target, CONFUSED word) = {es:+.3f}")
        print(f"  mean similarity(target, RANDOM word)   = {rs:+.3f}")
        print(f"  -> confusions are {'MORE' if es > rs else 'not more'} semantically similar than chance"
              f"  ({len(errs)} errors)")
        # show example confusions
        from collections import Counter
        cc = Counter((VOCAB[m], VOCAB[mh]) for m, mh in errs).most_common(6)
        print("  example confusions (target -> heard):")
        for (a, b), n in cc:
            print(f"    {a:8} -> {b:8}  (x{n}, sim {SIM[VOCAB.index(a), VOCAB.index(b)]:+.2f})")
    print("=" * 72)
    print("T3 — FREQUENCY effect: are frequent words communicated more reliably?")
    g = evolve(anchor=2.0, gens=220, seed=2)
    perword_ok = np.zeros(K); perword_n = np.zeros(K)
    _, errs = play(g, 2.0, rng, rounds=60, noise=0.5)
    # recompute per-word accuracy directly
    ES = eff_spk(g, 2.0); EL = eff_lis(g, 2.0)
    for trial in range(4000):
        m = int(rng.choice(K, p=FREQ)); j = int(rng.integers(len(g["spk"])))
        s = int(np.argmax(ES[j, m])); ehat = EL[j, s] + rng.normal(0, 0.5, DIM)
        perword_n[m] += 1; perword_ok[m] += (decode(ehat) == m)
    acc = perword_ok / np.maximum(perword_n, 1)
    mask = perword_n > 5
    r = np.corrcoef(np.log(FREQ[mask]), acc[mask])[0, 1]
    print(f"  corr(log word-frequency, accuracy) = {r:+.2f}  (positive = Zipf effect)")
    print("=" * 72)
    print("T4 — sample real-English conversation (anchor=2):")
    g = evolve(anchor=2.0, repro="sexual", gens=220, seed=1)
    ES = eff_spk(g, 2.0); EL = eff_lis(g, 2.0)
    for _ in range(7):
        m = int(rng.choice(K, p=FREQ)); s = int(np.argmax(ES[0, m])); mh = decode(EL[1, s])
        print(f'    mean "{VOCAB[m]:8}" -> says "{VOCAB[s]:8}" -> heard "{VOCAB[mh]:8}"  {"OK" if mh==m else "x"}')
