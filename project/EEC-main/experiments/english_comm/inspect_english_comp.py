"""Sample output from the compositional English model: real words spoken -> code -> heard word,
including HELD-OUT (zero-shot) words the model never trained on. The 'heard' word is the real word
whose code the listener reconstructed (nearest by code), so errors are real, usually-related words."""
import os, numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(HERE, "grounding_xl.npz"), allow_pickle=True)
EMB = G["emb"].astype(np.float64); VOCAB = list(G["vocab"])
P_NATIVE, FULL_BONUS = 0.7, 2.0
N, BUDGET, S, V, K = 64, 6000, 6, 8, 2000
TRAIN_FRAC = 0.7


def quantize(E, seed):
    ch = E.shape[1] // S; codes = np.zeros((len(E), S), int)
    for s in range(S):
        X = E[:, s * ch:(s + 1) * ch]; rng = np.random.default_rng(seed * 31 + s)
        C = X[rng.choice(len(E), V, replace=False)].copy()
        for _ in range(25):
            a = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
            for v in range(V):
                if (a == v).any(): C[v] = X[a == v].mean(0)
        codes[:, s] = a
    return codes


def breed(P, en, rng, mut=0.22):
    Ng = len(en); o = np.argsort(en); worst = o[:int(0.25 * Ng)]; top = o[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        m = rng.random(P[pa].shape) < 0.5
        P[w] = np.where(m, P[pa], P[pb]) + rng.normal(0, mut * 0.6, P[pa].shape)


def train(codes, seen, seed):
    rng = np.random.default_rng(seed); B = N * 16; A = rng.normal(0, 0.3, (N, S, V, V))
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
    return A


if __name__ == "__main__":
    codes = quantize(EMB[:K], 0)
    code_to_word = defaultdict(list)
    for w in range(K): code_to_word[tuple(codes[w])].append(w)
    allcodes = codes.copy()
    rng = np.random.default_rng(0); perm = rng.permutation(K); ntr = int(TRAIN_FRAC * K)
    seen, held = perm[:ntr], perm[ntr:]
    print(f"training compositional model on {len(seen)} real words (S={S},V={V}, {BUDGET} gens)...", flush=True)
    A = train(codes, seen, 0); emit = A.argmax(3); hear = A.argmax(2)

    def converse(word, sp, ls):
        cw = codes[word]; dec = np.array([hear[ls, s, emit[sp, s, cw[s]]] for s in range(S)])
        cand = code_to_word.get(tuple(dec))
        if cand:
            heardw = word if word in cand else cand[0]
        else:
            heardw = int(((allcodes != dec).sum(1)).argmin())          # nearest word by code
        return int(heardw)

    def sample(words, label, n=14):
        print(f"\n=== {label} ===")
        rng = np.random.default_rng(3); pick = rng.choice(words, n, replace=False); ok = 0
        for w in pick:
            sp, ls = 0, 1; h = converse(int(w), sp, ls); good = (h == w); ok += good
            print(f"  [{'OK' if good else 'x '}] speaker means \"{VOCAB[w]:11}\" -> listener hears \"{VOCAB[h]:11}\"")
        print(f"  -> {ok}/{n} recovered")

    sample(held, "HELD-OUT words (ZERO-SHOT — never trained on these)")
    sample(seen, "SEEN words (trained)")
