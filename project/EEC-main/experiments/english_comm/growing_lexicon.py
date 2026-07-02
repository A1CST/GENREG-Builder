"""TEST 4 (#1) -- GROWING LEXICON: where constraint ORDER should finally be load-bearing.

Test 3 showed reordering a FIXED-task reweighting washes out. The continuity principle bites only when
a constraint changes the REACHABLE COMPLEXITY -- a GROWING world (curriculum.py: G 8->128 unreachable
cold). So here the lexicon GROWS: start with a few words, add more in stages, CARRYING the population.
A small vocab is far easier to align on BOTH speaker and listener sides; extend from a stable core.

We track TWO-WAY COMPREHENSION (listener decodes correctly = the survival-relevant metric), not just
speaker usage -- the status check showed comprehension (0.30) is the honest number, not usage (0.31).

Conditions (same total generation budget; criticality stakes + urgency u=0.25 as in the champion):
  cold          all K words active from gen 0 (no ramp)
  freq_first    grow the active vocab most-frequent-first (curriculum)   -- carry population
  rare_first    grow rarest-first (reverse word order)                   -- carry population
  random_order  grow in random word order                               -- carry population
If freq_first (and grows generally) beat cold on comprehension, ORDER is load-bearing in a growing
world; the order AMONG words (freq vs rare first) tests which curriculum the world favors.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GR = np.load(os.path.join(HERE, "grounding.npz"), allow_pickle=True)
import os as _os
K = int(_os.environ.get("EEC_K","24"))
EMB = GR["emb"].astype(float)[:K]; VOCAB = list(GR["vocab"])[:K]; DIM = EMB.shape[1]
FREQ = 1.0 / (np.arange(1, K + 1) ** 1.1); FREQ /= FREQ.sum()
P_NATIVE, U = 0.7, 0.25
st = np.clip((1.0 / FREQ) / (1.0 / FREQ).mean(), 0.2, 12.0); st = st / st.mean()

LOG = open(os.path.join(HERE, "growing_lexicon_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"
def decode(e): return int(np.argmax(EMB @ e))


def sample_p(active):
    p = np.zeros(K); p[active] = (1 - U) * FREQ[active] + U * (st[active] / st[active].sum())
    return p / p.sum()


def play(spk, lis, rng, active, rounds=4):
    N = len(spk); en = np.zeros(N); emit = spk.argmax(2); sp_p = sample_p(active)
    nm = N * 8; msample = rng.choice(K, size=nm, p=sp_p); coin = rng.random(nm); mi = 0
    for _ in range(rounds):
        perm = rng.permutation(N)
        for a in range(0, N - 1, 2):
            i, j = perm[a], perm[a + 1]
            for sp, ls in ((i, j), (j, i)):
                m = int(msample[mi % nm]); mi += 1
                if coin[m] < P_NATIVE:
                    if decode(lis[ls, m]) == m: en[ls] += st[m]
                    if emit[sp, m] == m: en[sp] += st[m]
                else:
                    s = int(emit[sp, m])
                    if decode(lis[ls, s]) == m: en[sp] += st[m]; en[ls] += st[m]
    return en


def step(spk, lis, rng, active, N, mut):
    en = play(spk, lis, rng, active)
    order = np.argsort(en); worst = order[:int(0.25 * N)]; top = order[N - max(2, N // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        ms_ = rng.random((K, K)) < 0.5; ml = rng.random((K, DIM)) < 0.5
        spk[w] = np.where(ms_, spk[pa], spk[pb]) + rng.normal(0, mut * 0.6, (K, K))
        lis[w] = np.where(ml, lis[pa], lis[pb]) + rng.normal(0, mut * 0.6, (K, DIM))


def evolve(cond, gens=480, N=40, mut=0.25, seed=1):
    rng = np.random.default_rng(seed)
    spk = rng.normal(0, 0.3, (N, K, K)); lis = rng.normal(0, 0.3, (N, K, DIM))
    rank = np.argsort(-FREQ)
    if cond == "cold":
        for t in range(gens): step(spk, lis, rng, np.arange(K), N, mut)
        return spk, lis
    if cond == "freq_first":  word_order = rank
    elif cond == "rare_first": word_order = rank[::-1]
    else:                      word_order = rng.permutation(K)
    sizes = [K//4, K//2, 3*K//4, K]; per = gens // len(sizes)
    for sz in sizes:
        active = word_order[:sz]
        for t in range(per): step(spk, lis, rng, active, N, mut)
    return spk, lis


def metrics(spk, lis, rng):
    emit = spk.argmax(2); usage = (emit == np.arange(K)[None, :]).mean()
    N = len(spk); comp = 0; T = 5000
    for _ in range(T):
        sp, ls = rng.integers(N), rng.integers(N); m = int(rng.choice(K, p=(0.75 * FREQ + 0.25 * st / st.sum())))
        if decode(lis[ls, int(emit[sp, m])]) == m: comp += 1
    return usage, comp / T


if __name__ == "__main__":
    out(f"GROWING LEXICON (#1): does ORDER matter when the world GROWS? K={K}, criticality+u={U}, "
        f"tracking TWO-WAY COMPREHENSION (chance {1/K:.3f})")
    out("cold = all words at once; grow = carry population while adding words in stages")
    out("=" * 82)
    out(f"{'condition':>14} | {'speaker usage':>16} | {'TWO-WAY comprehension':>22}")
    SEEDS = 6; res = {}
    for cond in ["cold", "freq_first", "rare_first", "random_order"]:
        us, cp = [], []
        for s in range(SEEDS):
            spk, lis = evolve(cond, seed=s)
            u, c = metrics(spk, lis, np.random.default_rng(900 + s)); us.append(u); cp.append(c)
        res[cond] = (np.mean(cp), np.std(cp))
        out(f"{cond:>14} | {ms(us):>16} | {ms(cp):>22}")
    out("=" * 82)
    cold = res["cold"][0]
    for cond in ["freq_first", "rare_first", "random_order"]:
        d = (res[cond][0] - cold) / cold * 100
        out(f"  {cond:>12}: comprehension {res[cond][0]:.3f} vs cold {cold:.3f}  ({d:+.0f}%)")
    out("READING: grow > cold => ORDER (a reachability ramp) is load-bearing in a GROWING world.")
    out("         freq_first vs rare_first => which word-curriculum the world favors.")
    out("done"); LOG.close()
