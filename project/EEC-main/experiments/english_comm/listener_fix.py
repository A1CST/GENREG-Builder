"""LISTENER FIX -- shared lexicon (tie speak & hear) vs the two-matrix organism, for more fluency.

Diagnosis from fluency_push.py: the LISTENER is the cap. The current organism has an INDEPENDENT
speaker (spk K x K) and listener (lis K x DIM, decoded by nearest embedding), so (a) semantic
neighbours in the embedding collide on the listener side, and (b) the two sides must align by luck.

Paradigm-correct fix ("understanding before expression"): ONE shared lexicon used both to speak and
to hear. Genome = a single association matrix A (K x K). Speak meaning m -> signal = argmax_s A[m,s]
(row). Hear signal s -> meaning = argmax_m A[m,s] (column). Learning a word in EITHER direction fixes
BOTH, and there is no embedding-cluster confusion. Natives anchor A toward identity (English).

Head-to-head on the SAME known-correct landscape (frequent-first grow + criticality + urgency u=0.25).
Metric = two-way comprehension; also listener-uniform (decode native English over all words equally).
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

LOG = open(os.path.join(HERE, "listener_fix_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"
def decode(e): return int(np.argmax(EMB @ e))


def sample_p(active):
    p = np.zeros(K); p[active] = (1 - U) * FREQ[active] + U * (st[active] / st[active].sum())
    return p / p.sum()


# ---------------- two-matrix organism (current) ----------------
def tm_init(N, rng): return dict(spk=rng.normal(0, 0.3, (N, K, K)), lis=rng.normal(0, 0.3, (N, K, DIM)))
def tm_step(g, rng, active, mut, rounds):
    spk, lis = g["spk"], g["lis"]; N = len(spk); en = np.zeros(N); emit = spk.argmax(2)
    nm = N * 2 * rounds; msample = rng.choice(K, size=nm, p=sample_p(active)); coin = rng.random(nm); mi = 0
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
    _breed(g, en, rng, mut, ("spk", "lis"))
def tm_eval(g, rng, T=6000):
    spk, lis = g["spk"], g["lis"]; emit = spk.argmax(2); N = len(spk)
    comp = sum(decode(lis[rng.integers(N), int(emit[rng.integers(N), m])]) == m
               for m in rng.choice(K, T, p=(0.75 * FREQ + 0.25 * st / st.sum()))) / T
    lis_u = np.mean([decode(lis[rng.integers(N), m]) == m for m in range(K) for _ in range(60)])
    return comp, float(lis_u)


# ---------------- shared-lexicon organism (the fix) ----------------
def sh_init(N, rng): return dict(A=rng.normal(0, 0.3, (N, K, K)))
def sh_step(g, rng, active, mut, rounds):
    A = g["A"]; N = len(A); en = np.zeros(N); emit = A.argmax(2); hear = A.argmax(1)
    nm = N * 2 * rounds; msample = rng.choice(K, size=nm, p=sample_p(active)); coin = rng.random(nm); mi = 0
    for _ in range(rounds):
        perm = rng.permutation(N)
        for a in range(0, N - 1, 2):
            i, j = perm[a], perm[a + 1]
            for sp, ls in ((i, j), (j, i)):
                m = int(msample[mi % nm]); mi += 1
                if coin[m] < P_NATIVE:
                    if hear[ls, m] == m: en[ls] += st[m]          # native speaks English m -> resident hears
                    if emit[sp, m] == m: en[sp] += st[m]          # resident speaks -> native understands iff English
                else:
                    s = int(emit[sp, m])
                    if hear[ls, s] == m: en[sp] += st[m]; en[ls] += st[m]
    _breed(g, en, rng, mut, ("A",))
def sh_eval(g, rng, T=6000):
    A = g["A"]; emit = A.argmax(2); hear = A.argmax(1); N = len(A)
    comp = sum(hear[rng.integers(N), int(emit[rng.integers(N), m])] == m
               for m in rng.choice(K, T, p=(0.75 * FREQ + 0.25 * st / st.sum()))) / T
    lis_u = np.mean([hear[rng.integers(N), m] == m for m in range(K) for _ in range(60)])
    return comp, float(lis_u)


def _breed(g, en, rng, mut, keys):
    N = len(en); order = np.argsort(en); worst = order[:int(0.25 * N)]; top = order[N - max(2, N // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        for k in keys:
            msk = rng.random(g[k][pa].shape) < 0.5
            g[k][w] = np.where(msk, g[k][pa], g[k][pb]) + rng.normal(0, mut * 0.6, g[k][pa].shape)


def run(arch, seed, N=60, rounds=8, gps=350, mut=0.22):
    rng = np.random.default_rng(seed)
    init, stepf, evalf = (tm_init, tm_step, tm_eval) if arch == "two_matrix" else (sh_init, sh_step, sh_eval)
    g = init(N, rng); traj = []
    for sz in [K // 4, K // 2, 3 * K // 4, K]:
        active = RANK[:sz]
        for t in range(gps): stepf(g, rng, active, mut, rounds)
        traj.append((sz, evalf(g, np.random.default_rng(7000 + seed))))
    return traj, g


if __name__ == "__main__":
    GPS, SEEDS = 350, 4
    out(f"LISTENER FIX: shared lexicon vs two-matrix, freq-first grow + criticality + u={U}. "
        f"pop60 rounds8 gens/stage={GPS} ({GPS*4} total), {SEEDS} seeds")
    out(f"metric = two-way comprehension (chance {1/K:.3f}); lis-uniform = decode native English over all words")
    out("=" * 88)
    out(f"{'stage(vocab)':>13} | {'two_matrix comp':>16} {'(lis-unif)':>11} | {'SHARED comp':>13} {'(lis-unif)':>11}")
    tmj = [run("two_matrix", s, gps=GPS)[0] for s in range(SEEDS)]
    shr = [run("shared", s, gps=GPS) for s in range(SEEDS)]; shj = [r[0] for r in shr]
    for k, sz in enumerate([6, 12, 18, 24]):
        tc = [tmj[s][k][1][0] for s in range(SEEDS)]; tl = [tmj[s][k][1][1] for s in range(SEEDS)]
        sc = [shj[s][k][1][0] for s in range(SEEDS)]; sl = [shj[s][k][1][1] for s in range(SEEDS)]
        out(f"{sz:>13} | {ms(tc):>16} {ms(tl):>11} | {ms(sc):>13} {ms(sl):>11}")
    out("=" * 88)
    tf = [tmj[s][-1][1][0] for s in range(SEEDS)]; sf = [shj[s][-1][1][0] for s in range(SEEDS)]
    out(f"FINAL comprehension: two_matrix {ms(tf)}  |  SHARED {ms(sf)}  "
        f"(shared/two = {np.mean(sf)/np.mean(tf):.2f}x)")
    out("READING: shared >> two_matrix => tying speak+hear (understanding before expression) is the listener fix.")
    # transcript + per-word fluency from the best shared population
    g = max(shr, key=lambda r: r[0][-1][1][0])[1]; A = g["A"]; emit = A.argmax(2); hear = A.argmax(1)
    sp = int(np.argmax((emit == np.arange(K)[None, :]).sum(1))); ls = (sp + 1) % len(A)
    pop_emit = (emit == np.arange(K)[None, :]).mean(0)
    out("-" * 88)
    out(f"SHARED model — per-word fluency (frequent->rare), pop-level speak accuracy:")
    out("  " + "  ".join(f"{VOCAB[r]}:{pop_emit[r]*100:.0f}%" for r in RANK))
    rng = np.random.default_rng(5)
    out(f"SAMPLE CONVERSATION (shared, speaker {sp} -> listener {ls}):")
    good = 0
    for _ in range(14):
        m = int(rng.choice(K, p=(0.75 * FREQ + 0.25 * st / st.sum())))
        s = int(emit[sp, m]); h = int(hear[ls, s]); ok = h == m; good += ok
        out(f"  [{'OK' if ok else 'x '}] mean '{VOCAB[m]:8}' -> says '{VOCAB[s]:8}' -> heard '{VOCAB[h]:8}'")
    out(f"  -> {good}/14 understood")
    out("done"); LOG.close()