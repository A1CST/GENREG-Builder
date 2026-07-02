"""FLUENCY PUSH -- can two-way comprehension climb toward fluency on the KNOWN-CORRECT landscape?

The probes fixed the landscape: grounding + cultural-anchor + criticality(rent=freq x stake) +
exposure(urgency u=0.25), and the curriculum: grow the lexicon FREQUENT-FIRST, carry the population.
Status check said the honest metric is TWO-WAY COMPREHENSION (listener decodes), and BOTH sides leak
(speaker says English 0.31, listener understands English 0.59). Here we run that landscape with real
exposure and TRACK comprehension per stage:
  - climbs steadily  -> fluency was just a training-budget question (scale training = required & enough)
  - plateaus low     -> the landscape needs another constraint (the listener bottleneck), not more gens
We separate speaker-loss from listener-loss each stage so we can see WHICH side caps fluency.
Knobs scaled vs the 480-gen probe: bigger pop, more rounds (exposure), more gens, frequent-first grow.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GR = np.load(os.path.join(HERE, "grounding.npz"), allow_pickle=True)
K = 24
EMB = GR["emb"].astype(float)[:K]; VOCAB = list(GR["vocab"])[:K]; DIM = EMB.shape[1]
FREQ = 1.0 / (np.arange(1, K + 1) ** 1.1); FREQ /= FREQ.sum()
P_NATIVE, U = 0.7, 0.25
st = np.clip((1.0 / FREQ) / (1.0 / FREQ).mean(), 0.2, 12.0); st = st / st.mean()
RANK = np.argsort(-FREQ)

LOG = open(os.path.join(HERE, "fluency_push_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"
def decode(e): return int(np.argmax(EMB @ e))


def sample_p(active):
    p = np.zeros(K); p[active] = (1 - U) * FREQ[active] + U * (st[active] / st[active].sum())
    return p / p.sum()


def step(spk, lis, rng, active, mut, rounds):
    N = len(spk); en = np.zeros(N); emit = spk.argmax(2); sp_p = sample_p(active)
    nm = N * 2 * rounds; msample = rng.choice(K, size=nm, p=sp_p); coin = rng.random(nm); mi = 0
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
    order = np.argsort(en); worst = order[:int(0.25 * N)]; top = order[N - max(2, N // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        ms_ = rng.random((K, K)) < 0.5; ml = rng.random((K, DIM)) < 0.5
        spk[w] = np.where(ms_, spk[pa], spk[pb]) + rng.normal(0, mut * 0.6, (K, K))
        lis[w] = np.where(ml, lis[pa], lis[pb]) + rng.normal(0, mut * 0.6, (K, DIM))


def evaluate(spk, lis, rng, T=5000):
    emit = spk.argmax(2); N = len(spk)
    usage = float((emit == np.arange(K)[None, :]).mean())
    comp = comp_en = en_n = 0
    for _ in range(T):
        sp, ls = rng.integers(N), rng.integers(N)
        m = int(rng.choice(K, p=(0.75 * FREQ + 0.25 * st / st.sum())))
        ok = decode(lis[ls, int(emit[sp, m])]) == m; comp += ok
        if emit[sp, m] == m: en_n += 1; comp_en += ok
    # listener-only: native English (signal=m) -> does listener decode m?
    lis_ok = np.mean([decode(lis[rng.integers(N), m]) == m for m in range(K) for _ in range(40)])
    return usage, comp / T, comp_en / max(en_n, 1), float(lis_ok)


def run(seed, N, rounds, gens_per_stage, mut=0.22, grow=True):
    rng = np.random.default_rng(seed)
    spk = rng.normal(0, 0.3, (N, K, K)); lis = rng.normal(0, 0.3, (N, K, DIM))
    sizes = [K // 4, K // 2, 3 * K // 4, K] if grow else [K, K, K, K]
    traj = []
    for sz in sizes:
        active = RANK[:sz]
        for t in range(gens_per_stage):
            step(spk, lis, rng, active, mut, rounds)
        traj.append((sz, evaluate(spk, lis, np.random.default_rng(7000 + seed))))
    return spk, lis, traj


if __name__ == "__main__":
    N = int(os.environ.get("FP_N", "60")); ROUNDS = int(os.environ.get("FP_ROUNDS", "8"))
    GPS = int(os.environ.get("FP_GPS", "350")); SEEDS = int(os.environ.get("FP_SEEDS", "4"))
    out(f"FLUENCY PUSH on the correct landscape (freq-first grow, criticality+u={U}). "
        f"pop={N}, rounds={ROUNDS}, gens/stage={GPS} (total {GPS*4}), {SEEDS} seeds")
    out(f"metric = TWO-WAY COMPREHENSION (chance {1/K:.3f}); also split speaker vs listener loss")
    out("=" * 92)
    out(f"{'stage(vocab)':>13} | {'comprehension':>14} | {'speaker usage':>14} | "
        f"{'comp|saidEN':>12} | {'listener-only':>13}")
    by_stage = {sz: {"c": [], "u": [], "ce": [], "l": []} for sz in [6, 12, 18, 24]}
    finals = []
    last = None
    for s in range(SEEDS):
        spk, lis, traj = run(s, N, ROUNDS, GPS)
        for sz, (u, c, ce, l) in traj:
            by_stage[sz]["c"].append(c); by_stage[sz]["u"].append(u)
            by_stage[sz]["ce"].append(ce); by_stage[sz]["l"].append(l)
        finals.append(traj[-1][1]); last = (spk, lis)
    for sz in [6, 12, 18, 24]:
        d = by_stage[sz]
        out(f"{sz:>13} | {ms(d['c']):>14} | {ms(d['u']):>14} | {ms(d['ce']):>12} | {ms(d['l']):>13}")
    out("=" * 92)
    fc = [f[1] for f in finals]
    out(f"FINAL comprehension {ms(fc)}  vs 480-gen probe 0.362 vs fixed-champion 0.30")
    out("READING: rising across stages = training-limited (push longer). speaker usage << listener-only")
    out("=> speaker is the cap; listener-only << 1 => listener is the cap. Fix the lower one next.")
    spk, lis = last; emit = spk.argmax(2); rng = np.random.default_rng(11)
    sp = int(np.argmax((emit == np.arange(K)[None, :]).sum(1))); ls = (sp + 1) % len(spk)
    out("-" * 92)
    out(f"SAMPLE CONVERSATION from a trained pair (speaker {sp} -> listener {ls}), 14 frequency-sampled turns:")
    good = 0
    for _ in range(14):
        m = int(rng.choice(K, p=(0.75 * FREQ + 0.25 * st / st.sum())))
        s = int(emit[sp, m]); heard = decode(lis[ls, s]); ok = heard == m; good += ok
        out(f"  [{'OK' if ok else 'x '}] mean '{VOCAB[m]:8}' -> says '{VOCAB[s]:8}' -> heard '{VOCAB[heard]:8}'")
    out(f"  -> {good}/14 understood in this sample")
    out("done"); LOG.close()
