"""ENGLISH COMPOSITIONAL -- bring the multislot scaling win to REAL English words.

Multislot composition gave fluent comprehension over exponentially-large abstract meaning spaces. To
make it real-English vocabulary: give each real word a COMPOSITIONAL code by product-quantising its
embedding -- split the 64-d embedding into S chunks, k-means each chunk into V centroids, so each word
becomes an S-tuple code (semantically-grounded: similar words share slot-codes). The organism learns
per-slot machinery (shared across all words) and communicates a word by its code. Code grows as V^S
(>> vocabulary), params are O(S*V^2) independent of vocab.

Decisive test = ZERO-SHOT to HELD-OUT words: train on a subset of words, measure code-comprehension on
words never seen in training (their code is a new combination of seen slot-values). Full-meaning bonus
(partner identifies the word only on the full code). Metric: code-comprehension (all S slots conveyed).
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(HERE, "grounding_large.npz"), allow_pickle=True)
EMB = G["emb"].astype(np.float64); VOCAB = list(G["vocab"]); KMAX, D = EMB.shape
P_NATIVE, FULL_BONUS = 0.7, 2.0
N = int(os.environ.get("EC_N", "64")); BUDGET = int(os.environ.get("EC_BUDGET", "3000"))
SEEDS = int(os.environ.get("EC_SEEDS", "3")); S = int(os.environ.get("EC_S", "4"))
V = int(os.environ.get("EC_V", "8")); K = int(os.environ.get("EC_K", "300")); TRAIN_FRAC = 0.7

LOG = open(os.path.join(HERE, "english_compositional_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def quantize(E, S, V, seed):
    K = len(E); chunk = E.shape[1] // S; codes = np.zeros((K, S), int)
    for s in range(S):
        X = E[:, s * chunk:(s + 1) * chunk]; rng = np.random.default_rng(seed * 31 + s)
        C = X[rng.choice(K, V, replace=False)].copy()
        for _ in range(20):
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


def run(seed):
    E = EMB[:K]; codes = quantize(E, S, V, seed)
    uniq = len({tuple(c) for c in codes}) / K                     # code uniqueness (1.0 = no collisions)
    rng = np.random.default_rng(seed); ntr = int(TRAIN_FRAC * K); perm = rng.permutation(K)
    seen, held = perm[:ntr], perm[ntr:]; B = N * 16
    A = rng.normal(0, 0.3, (N, S, V, V))
    for t in range(BUDGET):
        emit = A.argmax(3); hear = A.argmax(2); en = np.zeros(N)
        w = seen[rng.integers(len(seen), size=B)]; cw = codes[w]    # (B,S) word codes
        sp = rng.integers(N, size=B); ls = rng.integers(N, size=B)
        nat = rng.random(B) < P_NATIVE; half = rng.random(B) < 0.5; rr = ~nat; allc = np.ones(B, bool)
        for s in range(S):
            cs = cw[:, s]; sig = emit[sp, s, cs]; cor = hear[ls, s, sig] == cs; allc &= cor
            ok = cor & rr; np.add.at(en, sp[ok], 1.0); np.add.at(en, ls[ok], 1.0)
            np.add.at(en, ls[(hear[ls, s, cs] == cs) & nat & half], 1.0)
            np.add.at(en, sp[(emit[sp, s, cs] == cs) & nat & ~half], 1.0)
        fb = allc & rr; np.add.at(en, sp[fb], FULL_BONUS); np.add.at(en, ls[fb], FULL_BONUS)
        breed(A, en, rng)
    return evaluate(A, codes, seen, held) + (uniq,)


def evaluate(A, codes, seen, held, T=6000):
    emit = A.argmax(3); hear = A.argmax(2); Ng = len(A)
    def comp(words):
        w = words[np.random.default_rng(7).integers(len(words), size=T)]; cw = codes[w]
        sp = np.random.default_rng(8).integers(Ng, size=T); ls = np.random.default_rng(9).integers(Ng, size=T)
        okall = np.ones(T, bool)
        for s in range(S):
            cs = cw[:, s]; okall &= hear[ls, s, emit[sp, s, cs]] == cs
        return float(okall.mean())                                # full-code comprehension
    return comp(seen), comp(held)


if __name__ == "__main__":
    out(f"ENGLISH COMPOSITIONAL: {K} real words -> product-quantised S={S} x V={V} codes (V^S={V**S}). "
        f"pop {N}, budget {BUDGET}, {SEEDS} seeds, train {int(TRAIN_FRAC*100)}%.")
    out("zero-shot = code-comprehension on HELD-OUT real words never seen in training.")
    out("=" * 70)
    rs = [run(s) for s in range(SEEDS)]
    se = [r[0] for r in rs]; he = [r[1] for r in rs]; uq = [r[2] for r in rs]
    out(f"  seen-word code-comprehension : {ms(se)}")
    out(f"  HELD-OUT (zero-shot)         : {ms(he)}   (chance ~ {1/(V**S):.1e})")
    out(f"  code uniqueness (1=no collisions): {ms(uq)}")
    out("=" * 70)
    out("READING: held-out >> chance => the organism comprehends REAL words it never trained on,")
    out("via shared per-slot machinery. Real-English vocabulary scales by composition, zero-shot.")
    out("done"); LOG.close()
