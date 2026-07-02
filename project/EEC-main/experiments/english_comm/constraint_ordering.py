"""TEST 3 -- CONSTRAINT ORDERING: does the ORDER we apply the exposure constraint change the outcome?

Test 2 found a P2 Goldilocks: urgency-gating (#7) lifts the rare tail but, applied all at once, starves
the frequent core (finite exposure channel) -> overall usage peaks early then falls. The hypothesis
(Payton): the ORDER constraints are applied matters -- like the curriculum ramp beating cold-start, and
the coma needing graduated re-grounding. Bootstrap the frequent core FIRST (u=0), THEN ramp urgency:
the common words consolidate and, retained by selection, survive the later airtime drop, while the rare
tail gets its late boost. That should beat applying everything at once -- escaping the P2 tradeoff by
SCHEDULING, not by a stronger constraint.

All schedules use criticality stakes and the SAME total urgency budget (mean u = 0.5 over training), so
the ONLY difference is the order. English usage MEASURED, never rewarded; acquired from natives (anchor 0).
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GR = np.load(os.path.join(HERE, "grounding.npz"), allow_pickle=True)
EMB = GR["emb"].astype(float)[:24]; K, DIM = EMB.shape
FREQ = 1.0 / (np.arange(1, K + 1) ** 1.1); FREQ /= FREQ.sum()
ANCHOR, P_NATIVE = 0.0, 0.7

LOG = open(os.path.join(HERE, "constraint_ordering_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"

st = np.clip((1.0 / FREQ) / (1.0 / FREQ).mean(), 0.2, 12.0); st = st / st.mean()   # criticality stakes
PS = st / st.sum()


def emit_of(spk): return (spk + (ANCHOR * np.eye(K) if ANCHOR > 0 else 0)).argmax(2)
def decode(e): return int(np.argmax(EMB @ e))


def play(spk, lis, rng, sample_p, rounds=4):
    N = len(spk); en = np.zeros(N); emit = emit_of(spk); EL = lis + (ANCHOR * EMB if ANCHOR > 0 else 0)
    nm = N * 8; msample = rng.choice(K, size=nm, p=sample_p); coin = rng.random(nm); mi = 0
    for _ in range(rounds):
        perm = rng.permutation(N)
        for a in range(0, N - 1, 2):
            i, j = perm[a], perm[a + 1]
            for sp, ls in ((i, j), (j, i)):
                m = int(msample[mi % nm]); mi += 1
                if coin[m] < P_NATIVE:
                    if decode(EL[ls, m]) == m: en[ls] += st[m]
                    if emit[sp, m] == m: en[sp] += st[m]
                else:
                    s = int(emit[sp, m])
                    if decode(EL[ls, s]) == m: en[sp] += st[m]; en[ls] += st[m]
    return en


def u_at(sched, t, gens):
    """All schedules have the SAME mean urgency (0.25, the Goldilocks budget); only ORDER differs."""
    f = t / (gens - 1)
    if sched == "constant":      return 0.25            # best static from Test 2 (champion)
    if sched == "freq_first":    return 0.5 * f         # 0 -> 0.5 (ground common core, then urgency)
    if sched == "urgent_first":  return 0.5 * (1 - f)   # 0.5 -> 0 (time-reverse of freq_first)
    if sched == "staged":        return 0.0 if f < 0.5 else 0.5
    return 0.25


def evolve(sched, gens=450, N=40, mut=0.25, seed=1):
    rng = np.random.default_rng(seed)
    spk = rng.normal(0, 0.3, (N, K, K)); lis = rng.normal(0, 0.3, (N, K, DIM))
    for t in range(gens):
        u = u_at(sched, t, gens); sample_p = (1 - u) * FREQ + u * PS
        en = play(spk, lis, rng, sample_p)
        order = np.argsort(en); worst = order[:int(0.25 * N)]; top = order[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            ms_ = rng.random((K, K)) < 0.5; ml = rng.random((K, DIM)) < 0.5
            spk[w] = np.where(ms_, spk[pa], spk[pb]) + rng.normal(0, mut * 0.6, (K, K))
            lis[w] = np.where(ml, lis[pa], lis[pb]) + rng.normal(0, mut * 0.6, (K, DIM))
    return spk


def per_word_usage(spk): return (emit_of(spk) == np.arange(K)[None, :]).mean(0)


if __name__ == "__main__":
    rank = np.argsort(-FREQ); rare = rank[K // 2:]; freqh = rank[:K // 2]; logF = np.log(FREQ); SEEDS = 10
    out(f"CONSTRAINT ORDERING: same urgency budget (mean u=0.25), different schedule. K={K}, {SEEDS} seeds")
    out("can reordering BEAT the best constant? freq_first vs urgent_first are exact time-reverses")
    out("=" * 92)
    out(f"{'schedule':>16} | {'usage':>13} | {'freq-half':>13} | {'rare-half':>13} | {'corr(logF,use)':>14}")
    res = {}
    for sched in ["constant", "urgent_first", "staged", "freq_first"]:
        eu, ef, er, co = [], [], [], []
        for s in range(SEEDS):
            u = per_word_usage(evolve(sched, seed=s))
            eu.append(u.mean()); ef.append(u[freqh].mean()); er.append(u[rare].mean())
            co.append(np.corrcoef(logF, u)[0, 1])
        res[sched] = (np.mean(eu), np.std(eu))
        out(f"{sched:>16} | {ms(eu):>13} | {ms(ef):>13} | {ms(er):>13} | {ms(co):>14}")
    out("=" * 92)
    best = max(res, key=lambda k: res[k][0])
    aa = res["constant"][0]
    out(f"BEST schedule: {best} (usage {res[best][0]:.3f}) vs constant {aa:.3f} "
        f"(+{(res[best][0]-aa)/aa*100:.0f}%). If freq_first/staged > all_at_once, ORDER is load-bearing.")
    out("done"); LOG.close()
