"""TEST 2 (constraint #7) -- EXPOSURE / urgency-gated communication: finishing what scarcity started.

Test 1 showed criticality makes the rare lexicon PAY (rent = freq x stake) but cannot flatten the
pidgin, because frequency independently controls EXPOSURE -- how often selection acts on a word.
Rent sets the prize; frequency sets the airtime. Rare-critical words are worth learning but seen
too rarely to consolidate.

#7 is the law that fixes exposure WITHOUT a designed gradient: in a real world you communicate what
MATTERS, not what is merely frequent (alarm calls are about the rare predator, not the grass). So the
channel's referent distribution is a mixture: (1-u) frequency-chatter + u stake-driven urgency. The
'urgency fraction' u is the new axis. Energy from a correct decode is still stake[m] (criticality
value); English usage is MEASURED, never rewarded; English is acquired from natives (anchor 0).

Prediction: u=0 reproduces Test-1 criticality (pidgin tail survives). Raising u lifts rare-word
EXPOSURE -> the tail English-ifies and corr(logFreq,usage) flattens. Too much u starves the frequent
words -> expect a P2 Goldilocks interior optimum on overall usage, not a monotone win.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GR = np.load(os.path.join(HERE, "grounding.npz"), allow_pickle=True)
EMB = GR["emb"].astype(float)[:24]; VOCAB = list(GR["vocab"])[:24]
K, DIM = EMB.shape
FREQ = 1.0 / (np.arange(1, K + 1) ** 1.1); FREQ /= FREQ.sum()
ANCHOR, P_NATIVE = 0.0, 0.7

LOG = open(os.path.join(HERE, "exposure_teaching_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def stakes(kind):
    s = np.ones(K) if kind == "flat" else 1.0 / FREQ
    s = np.clip(s / s.mean(), 0.2, 12.0); return s / s.mean()


def emit_of(spk): return (spk + (ANCHOR * np.eye(K) if ANCHOR > 0 else 0)).argmax(2)
def lis_of(lis): return lis + (ANCHOR * EMB if ANCHOR > 0 else 0)
def decode(e): return int(np.argmax(EMB @ e))


def play(spk, lis, rng, stake, sample_p, rounds=4):
    N = len(spk); en = np.zeros(N); emit = emit_of(spk); EL = lis_of(lis)
    nm = N * 8; msample = rng.choice(K, size=nm, p=sample_p); coin = rng.random(nm); mi = 0
    for _ in range(rounds):
        perm = rng.permutation(N)
        for a in range(0, N - 1, 2):
            i, j = perm[a], perm[a + 1]
            for sp, ls in ((i, j), (j, i)):
                m = int(msample[mi % nm]); mi += 1
                if coin[m] < P_NATIVE:
                    if decode(EL[ls, m]) == m: en[ls] += stake[m]
                    if emit[sp, m] == m: en[sp] += stake[m]
                else:
                    s = int(emit[sp, m])
                    if decode(EL[ls, s]) == m: en[sp] += stake[m]; en[ls] += stake[m]
    return en


def evolve(kind, u, gens=450, N=40, mut=0.25, seed=1):
    rng = np.random.default_rng(seed)
    spk = rng.normal(0, 0.3, (N, K, K)); lis = rng.normal(0, 0.3, (N, K, DIM))
    st = stakes(kind); ps = st / st.sum()
    sample_p = (1 - u) * FREQ + u * ps                       # urgency-gated channel
    for t in range(gens):
        en = play(spk, lis, rng, st, sample_p)
        order = np.argsort(en); worst = order[:int(0.25 * N)]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            ms_ = rng.random((K, K)) < 0.5; ml = rng.random((K, DIM)) < 0.5
            spk[w] = np.where(ms_, spk[pa], spk[pb]) + rng.normal(0, mut * 0.6, (K, K))
            lis[w] = np.where(ml, lis[pa], lis[pb]) + rng.normal(0, mut * 0.6, (K, DIM))
    return spk


def per_word_usage(spk): return (emit_of(spk) == np.arange(K)[None, :]).mean(0)


if __name__ == "__main__":
    rank = np.argsort(-FREQ); rare = rank[K // 2:]; freqh = rank[:K // 2]; logF = np.log(FREQ); SEEDS = 5
    out(f"EXPOSURE / urgency-gated communication (#7), on top of criticality. K={K}, chance {1/K:.3f}, "
        f"anchor {ANCHOR}, {SEEDS} seeds")
    out("u = urgency fraction of the channel (0 = pure frequency chatter = Test-1 criticality)")
    out("=" * 90)
    out(f"{'condition':>26} | {'usage':>13} | {'freq-half':>13} | {'rare-half':>13} | {'corr(logF,use)':>14}")

    def report(tag, kind, u):
        eu, ef, er, co = [], [], [], []
        for s in range(SEEDS):
            uarr = per_word_usage(evolve(kind, u, seed=s))
            eu.append(uarr.mean()); ef.append(uarr[freqh].mean()); er.append(uarr[rare].mean())
            co.append(np.corrcoef(logF, uarr)[0, 1])
        out(f"{tag:>26} | {ms(eu):>13} | {ms(ef):>13} | {ms(er):>13} | {ms(co):>14}")
        return np.mean(eu), np.mean(er), np.mean(co)

    report("flat,u=0 (PIDGIN ref)", "flat", 0.0)
    out("-" * 90)
    sweep = [0.0, 0.25, 0.5, 0.75, 1.0]
    res = {u: report(f"criticality,u={u}", "criticality", u) for u in sweep}
    out("=" * 90)
    best_u = max(res, key=lambda k: res[k][0])
    out(f"OVERALL-USAGE optimum at u={best_u} (usage {res[best_u][0]:.3f}). "
        f"Monotone-down corr = lexicon flattening; interior usage peak = P2 Goldilocks on the exposure axis.")
    out(f"corr(logF,usage): " + "  ".join(f"u{u}={res[u][2]:+.2f}" for u in sweep))
    out(f"rare-half usage : " + "  ".join(f"u{u}={res[u][1]:.3f}" for u in sweep))
    out("done"); LOG.close()

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep, [res[u][0] for u in sweep], "o-", color="#1b5e9e", lw=2.5, label="overall usage")
    ax.plot(sweep, [res[u][1] for u in sweep], "^-", color="#2a8", lw=2, label="rare-half usage")
    ax2 = ax.twinx(); ax2.plot(sweep, [res[u][2] for u in sweep], "s--", color="#c44", lw=2, label="corr(logFreq,usage)")
    ax.set_xlabel("urgency fraction u (exposure axis)"); ax.set_ylabel("English usage"); ax2.set_ylabel("freq-usage corr", color="#c44")
    ax.set_title("Exposure (#7): urgency-gating lifts the rare tail, flattens the pidgin", weight="bold")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right"); ax.grid(alpha=.3)
    plt.tight_layout(); p = os.path.join(HERE, "exposure_teaching.png"); plt.savefig(p, dpi=120); print("saved", p)
