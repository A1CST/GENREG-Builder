"""Find a regime where anchor-0 acquisition BOOTSTRAPS to a pidgin (English usage > chance, tracking
frequency) -- the precondition for testing whether criticality breaks the pidgin. Impose a real
Zipfian frequency spread over the grounded words (the corpus top-K are near-equal frequency)."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GR = np.load(os.path.join(HERE, "grounding.npz"), allow_pickle=True)
EMB = GR["emb"].astype(float); Kfull, DIM = EMB.shape


def zipf_freq(K, a=1.1):
    f = 1.0 / (np.arange(1, K + 1) ** a); return f / f.sum()


def run(K, gens, anchor, seed, p_native=0.7, N=40, mut=0.25):
    rng = np.random.default_rng(seed)
    emb = EMB[:K]; FREQ = zipf_freq(K)
    spk = rng.normal(0, 0.3, (N, K, K)); lis = rng.normal(0, 0.3, (N, K, DIM))
    def emitm(): return (spk + (anchor * np.eye(K) if anchor > 0 else 0)).argmax(2)
    def dec(e): return int(np.argmax(emb @ e))
    ELp = lambda: lis + (anchor * emb if anchor > 0 else 0)
    for t in range(gens):
        en = np.zeros(N); emit = emitm(); EL = ELp()
        nm = N * 8; msample = rng.choice(K, size=nm, p=FREQ); coin = rng.random(nm); mi = 0
        for _ in range(4):
            perm = rng.permutation(N)
            for a in range(0, N - 1, 2):
                i, j = perm[a], perm[a + 1]
                for sp, ls in ((i, j), (j, i)):
                    m = int(msample[mi % nm]); mi += 1
                    if coin[m] < p_native:                       # native exchange (both directions)
                        if dec(EL[ls, m]) == m: en[ls] += 1
                        if emit[sp, m] == m: en[sp] += 1
                    else:
                        s = int(emit[sp, m])
                        if dec(EL[ls, s]) == m: en[sp] += 1; en[ls] += 1
        order = np.argsort(en); worst = order[:int(0.25 * N)]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            ms_ = rng.random((K, K)) < 0.5; ml = rng.random((K, DIM)) < 0.5
            spk[w] = np.where(ms_, spk[pa], spk[pb]) + rng.normal(0, mut * 0.6, (K, K))
            lis[w] = np.where(ml, lis[pa], lis[pb]) + rng.normal(0, mut * 0.6, (K, DIM))
    emit = emitm(); u = (emit == np.arange(K)[None, :]).mean(0)
    return float(u.mean()), float(np.corrcoef(np.log(FREQ), u)[0, 1])


if __name__ == "__main__":
    print(f"probe: which (K, gens, anchor) bootstraps acquisition above chance?  (emb dim {DIM})")
    for K in [16, 24, 40]:
        for anchor in [0.0, 0.4]:
            us, cs = [], []
            for s in range(3):
                u, c = run(K, gens=400, anchor=anchor, seed=s)
                us.append(u); cs.append(c)
            print(f"  K={K:2} anchor={anchor:.1f} chance={1/K:.3f} | "
                  f"usage={np.mean(us):.3f}+/-{np.std(us):.3f} | corr(logFreq,usage)={np.mean(cs):+.2f}")
