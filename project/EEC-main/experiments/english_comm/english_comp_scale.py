"""ENGLISH COMP SCALE -- scale REAL English vocabulary 300 -> 2000 at CONSTANT cost.

The organism's genome is per-slot machinery S x V x V -- INDEPENDENT of vocabulary K. So training a
2000-word vocab costs the same as 300; vocabulary is free in compute. Each real word gets a
compositional code by product-quantising its embedding (S chunks, V centroids). Train on 70% of
words, measure on HELD-OUT words (zero-shot). Report code-comprehension (transmission fidelity) AND
word-recovery (collision-aware: decoded code identifies the unique word). If held-out comprehension
stays flat as K grows, real-English vocabulary scales by composition -- impossible for a flat lexicon.
"""
import os, numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(HERE, "grounding_xl.npz"), allow_pickle=True)
EMB = G["emb"].astype(np.float64); VOCAB = list(G["vocab"])
P_NATIVE, FULL_BONUS = 0.7, 2.0
N = int(os.environ.get("EC_N", "64")); BUDGET = int(os.environ.get("EC_BUDGET", "4000"))
SEEDS = int(os.environ.get("EC_SEEDS", "3")); S = int(os.environ.get("EC_S", "6")); V = int(os.environ.get("EC_V", "8"))
KS = [int(x) for x in os.environ.get("EC_KS", "300,1000,2000").split(",")]; TRAIN_FRAC = 0.7

LOG = open(os.path.join(HERE, "english_comp_scale_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def quantize(E, seed):
    K = len(E); ch = E.shape[1] // S; codes = np.zeros((K, S), int)
    for s in range(S):
        X = E[:, s * ch:(s + 1) * ch]; rng = np.random.default_rng(seed * 31 + s)
        C = X[rng.choice(K, V, replace=False)].copy()
        for _ in range(25):
            a = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
            for v in range(V):
                if (a == v).any(): C[v] = X[a == v].mean(0)
        codes[:, s] = a
    return codes


def breed(P, en, rng, mut=0.22):
    Ng = len(en); order = np.argsort(en); worst = order[:int(0.25 * Ng)]; top = order[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        msk = rng.random(P[pa].shape) < 0.5
        P[w] = np.where(msk, P[pa], P[pb]) + rng.normal(0, mut * 0.6, P[pa].shape)


def run(K, seed):
    E = EMB[:K]; codes = quantize(E, seed)
    cc = Counter(tuple(c) for c in codes); uniq_word = np.array([cc[tuple(codes[w])] == 1 for w in range(K)])
    rng = np.random.default_rng(seed); ntr = int(TRAIN_FRAC * K); perm = rng.permutation(K)
    seen, held = perm[:ntr], perm[ntr:]; B = N * 16
    A = rng.normal(0, 0.3, (N, S, V, V))
    for t in range(BUDGET):
        emit = A.argmax(3); hear = A.argmax(2); en = np.zeros(N)
        w = seen[rng.integers(len(seen), size=B)]; cw = codes[w]
        sp = rng.integers(N, size=B); ls = rng.integers(N, size=B)
        nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5; rr = ~nat; allc = np.ones(B, bool)
        for s in range(S):
            cs = cw[:, s]; cor = hear[ls, s, emit[sp, s, cs]] == cs; allc &= cor
            ok = cor & rr; np.add.at(en, sp[ok], 1.0); np.add.at(en, ls[ok], 1.0)
            np.add.at(en, ls[(hear[ls, s, cs] == cs) & nat & half], 1.0)
            np.add.at(en, sp[(emit[sp, s, cs] == cs) & nat & ~half], 1.0)
        fb = allc & rr; np.add.at(en, sp[fb], FULL_BONUS); np.add.at(en, ls[fb], FULL_BONUS)
        breed(A, en, rng)
    return ev(A, codes, held, uniq_word) + (uniq_word.mean(),)


def ev(A, codes, words, uniq_word, T=8000):
    emit = A.argmax(3); hear = A.argmax(2); Ng = len(A)
    w = words[np.random.default_rng(7).integers(len(words), size=T)]; cw = codes[w]
    sp = np.random.default_rng(8).integers(Ng, size=T); ls = np.random.default_rng(9).integers(Ng, size=T)
    okall = np.ones(T, bool)
    for s in range(S):
        okall &= hear[ls, s, emit[sp, s, cw[:, s]]] == cw[:, s]
    code_comp = float(okall.mean())
    word_rec = float((okall & uniq_word[w]).mean())              # recovered: code right AND unique to the word
    return code_comp, word_rec


if __name__ == "__main__":
    out(f"ENGLISH COMP SCALE: real vocab {KS} via S={S} x V={V} codes (V^S={V**S:,}). "
        f"GENOME {S}x{V}x{V} is CONSTANT in K. pop {N}, budget {BUDGET}, {SEEDS} seeds, train {int(TRAIN_FRAC*100)}%.")
    out(f"{'K (real words)':>14} | {'HELD-OUT code-comp':>19} | {'word-recovery':>14} | {'uniqueness':>11}")
    out("=" * 72)
    res = {}
    for K in KS:
        r = [run(K, s) for s in range(SEEDS)]
        cc = [x[0] for x in r]; wr = [x[1] for x in r]; uq = [x[2] for x in r]
        res[K] = (np.mean(cc), np.mean(wr))
        out(f"{K:>14} | {ms(cc):>19} | {ms(wr):>14} | {ms(uq):>11}")
    out("=" * 72)
    out("READING: held-out comprehension ~flat as K grows = real-English vocabulary scales by")
    out("composition, zero-shot, at CONSTANT compute. A flat lexicon needs O(K^2) params and cannot")
    out("generalise to held-out words at all.")
    for K in KS:
        out(f"  K={K:>4}: held-out code-comp {res[K][0]:.3f}, word-recovery {res[K][1]:.3f}")
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(KS, [res[K][0] for K in KS], "o-", color="#1b5e9e", lw=2.5, label="held-out code-comprehension")
    plt.plot(KS, [res[K][1] for K in KS], "s-", color="#2a8", lw=2.5, label="held-out word-recovery")
    plt.xlabel("real English vocabulary size K"); plt.ylabel("zero-shot comprehension"); plt.ylim(0, 1)
    plt.title("Real-English vocabulary scales by composition (constant compute)", weight="bold")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    p = os.path.join(HERE, "english_comp_scale.png"); plt.savefig(p, dpi=120); print("saved", p)
