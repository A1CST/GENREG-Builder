"""TEST 1 -- Zipfian scarcity / criticality: escaping the pidgin ceiling.

The pidgin ceiling is P1 at the per-word level: a word English-ifies only where THAT word pays its
rent, rent = frequency x stake. Under FLAT stakes rent ~ frequency, so only frequent words clear
threshold -> the rare tail stays un-English (pidgin; signature corr(logFreq,usage) > 0). Make rare
referents survival-critical (rare = deadly, stake ~ 1/freq), rent flattens across the lexicon, and
the rare tail English-ifies too -- escaping pidgin with NO designed push toward English (stake is a
world fact: a poison berry is rare but lethal; English usage is MEASURED, never rewarded).

Regime found by probe_bootstrap.py: real PPMI-SVD embeddings, a real Zipfian frequency spread over
K referents, anchor 0 (English acquired from fixed native speakers, not innate). At K=24 the pidgin
is clear (usage ~0.19 >> chance 0.04, corr(logFreq,usage) ~ +0.5).
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GR = np.load(os.path.join(HERE, "grounding.npz"), allow_pickle=True)
EMB_ALL = GR["emb"].astype(float); VOCAB_ALL = list(GR["vocab"])

K, DIM = 24, GR["emb"].shape[1]
EMB = EMB_ALL[:K]; VOCAB = VOCAB_ALL[:K]
FREQ = (1.0 / (np.arange(1, K + 1) ** 1.1)); FREQ /= FREQ.sum()   # real Zipf spread over referents
ANCHOR, P_NATIVE = 0.0, 0.7

LOG = open(os.path.join(HERE, "zipf_scarcity_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def stakes(kind):
    if kind == "flat":        s = np.ones(K)
    elif kind == "criticality": s = 1.0 / FREQ                       # rare = deadly
    elif kind == "random":    s = np.random.default_rng(0).permutation(1.0 / FREQ)  # freq-independent
    s = np.clip(s / s.mean(), 0.2, 12.0)
    return s / s.mean()                                              # equal TOTAL survival value across worlds


def emit_of(spk): return (spk + (ANCHOR * np.eye(K) if ANCHOR > 0 else 0)).argmax(2)
def lis_of(lis): return lis + (ANCHOR * EMB if ANCHOR > 0 else 0)
def decode(e): return int(np.argmax(EMB @ e))


def play(spk, lis, rng, stake, rounds=4):
    N = len(spk); en = np.zeros(N); emit = emit_of(spk); EL = lis_of(lis)
    nm = N * 8; msample = rng.choice(K, size=nm, p=FREQ); coin = rng.random(nm); mi = 0
    for _ in range(rounds):
        perm = rng.permutation(N)
        for a in range(0, N - 1, 2):
            i, j = perm[a], perm[a + 1]
            for sp, ls in ((i, j), (j, i)):
                m = int(msample[mi % nm]); mi += 1
                if coin[m] < P_NATIVE:                              # native exchange (both directions)
                    if decode(EL[ls, m]) == m: en[ls] += stake[m]   # native speaks English -> resident listens
                    if emit[sp, m] == m: en[sp] += stake[m]         # resident speaks -> native understands iff English
                else:
                    s = int(emit[sp, m])
                    if decode(EL[ls, s]) == m: en[sp] += stake[m]; en[ls] += stake[m]
    return en


def evolve(kind, gens=450, N=40, mut=0.25, seed=1):
    rng = np.random.default_rng(seed)
    spk = rng.normal(0, 0.3, (N, K, K)); lis = rng.normal(0, 0.3, (N, K, DIM)); st = stakes(kind)
    for t in range(gens):
        en = play(spk, lis, rng, st)
        order = np.argsort(en); worst = order[:int(0.25 * N)]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            ms_ = rng.random((K, K)) < 0.5; ml = rng.random((K, DIM)) < 0.5
            spk[w] = np.where(ms_, spk[pa], spk[pb]) + rng.normal(0, mut * 0.6, (K, K))
            lis[w] = np.where(ml, lis[pa], lis[pb]) + rng.normal(0, mut * 0.6, (K, DIM))
    return spk, lis, st


def per_word_usage(spk): return (emit_of(spk) == np.arange(K)[None, :]).mean(0)


if __name__ == "__main__":
    rank = np.argsort(-FREQ); rare_half = rank[K // 2:]; logF = np.log(FREQ)
    out(f"ZIPFIAN SCARCITY / criticality (real embeddings, Zipf freq over K={K} referents, "
        f"chance {1/K:.3f}, anchor {ANCHOR})")
    out(f"frequency spread {FREQ.max()/FREQ.min():.0f}x; English usage MEASURED never rewarded; "
        f"English acquired from natives")
    out("=" * 86)
    SEEDS = 5; rows = {}
    for kind in ["flat", "criticality", "random"]:
        eu, eu_rare, eu_freq, corr, pw = [], [], [], [], []
        for s in range(SEEDS):
            spk, lis, st = evolve(kind, seed=s); u = per_word_usage(spk)
            eu.append(u.mean()); eu_rare.append(u[rare_half].mean()); eu_freq.append(u[rank[:K//2]].mean())
            corr.append(np.corrcoef(logF, u)[0, 1]); pw.append(u)
        rows[kind] = (np.mean(pw, 0), stakes(kind))
        out(f"  {kind:12} | usage {ms(eu)} | freq-half {ms(eu_freq)} | rare-half {ms(eu_rare)} | "
            f"corr(logF,usage) {ms(corr)}")
    out("-" * 86)
    out("READING: flat = PIDGIN (usage tracks frequency: corr>0, rare-half low).")
    out("         criticality = rare half RISES and corr FLATTENS toward 0 => full lexicon English-ifies.")
    out("=" * 86)
    out("THE LAW: per-word usage vs rent = freq x stake (all three worlds should collapse onto one curve)")
    allr = np.concatenate([FREQ * st for _, st in rows.values()])
    allu = np.concatenate([pw for pw, _ in rows.values()])
    q = np.quantile(allr, [0, .25, .5, .75, 1.0])
    out("  rent quartile (pooled flat/criticality/random)   mean usage")
    for b in range(4):
        m = (allr >= q[b]) & (allr <= q[b + 1])
        out(f"    {q[b]:.4f}..{q[b+1]:.4f}    usage {allu[m].mean():.3f}  (n={m.sum()})")
    out(f"  corr(log rent, usage) pooled = {np.corrcoef(np.log(allr+1e-9), allu)[0,1]:+.2f}  "
        f"(positive => rent, not raw frequency, is the law)")
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for kind, col in [("flat", "#c44"), ("criticality", "#1b5e9e")]:
        pw = rows[kind][0]
        ax1.scatter(FREQ, pw, color=col, s=40, alpha=.8, label=kind)
    ax1.set_xscale("log"); ax1.set_xlabel("word frequency (Zipf)"); ax1.set_ylabel("English usage")
    ax1.set_title("Pidgin tail: flat stakes vs criticality (rare=deadly)", weight="bold")
    ax1.legend(); ax1.grid(alpha=.3)
    ax2.scatter(allr, allu, c="#2a8", s=30, alpha=.7)
    ax2.set_xscale("log"); ax2.set_xlabel("rent = frequency x stake"); ax2.set_ylabel("English usage")
    ax2.set_title("The law: usage follows RENT (all 3 worlds pooled)", weight="bold")
    ax2.grid(alpha=.3)
    plt.tight_layout(); p = os.path.join(HERE, "zipf_scarcity.png"); plt.savefig(p, dpi=120)
    print("saved", p)
