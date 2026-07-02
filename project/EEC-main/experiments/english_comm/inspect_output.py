"""STATUS CHECK -- what does the evolved population actually OUTPUT? Evolve the champion regime
(criticality stakes + urgency u=0.25) and print real conversations: for each meaning, the English
word a resident SAYS and the word the listener HEARS. Signals are indices into the real vocab, so
every utterance is a real English word; emit==meaning => spoke the correct English word.
Self-contained (does not import the experiment scripts, to avoid truncating their result logs)."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GR = np.load(os.path.join(HERE, "grounding.npz"), allow_pickle=True)
EMB = GR["emb"].astype(float)[:24]; VOCAB = list(GR["vocab"])[:24]; K, DIM = EMB.shape
FREQ = 1.0 / (np.arange(1, K + 1) ** 1.1); FREQ /= FREQ.sum()
P_NATIVE = 0.7
st = np.clip((1.0 / FREQ) / (1.0 / FREQ).mean(), 0.2, 12.0); st = st / st.mean()
PS = st / st.sum()


def decode(e): return int(np.argmax(EMB @ e))


def evolve(u=0.25, gens=450, N=40, mut=0.25, seed=0):
    rng = np.random.default_rng(seed)
    spk = rng.normal(0, 0.3, (N, K, K)); lis = rng.normal(0, 0.3, (N, K, DIM))
    sample_p = (1 - u) * FREQ + u * PS
    for t in range(gens):
        emit = spk.argmax(2); en = np.zeros(N)
        nm = N * 8; msample = rng.choice(K, size=nm, p=sample_p); coin = rng.random(nm); mi = 0
        for _ in range(4):
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
    return spk, lis


rank = np.argsort(-FREQ)
print("STATUS of the evolved population — regime: criticality + urgency u=0.25 (the champion)")
print(f"vocab ({K} real words): {' '.join(VOCAB)}")
print("=" * 80)
spk, lis = evolve(seed=0)
emit = spk.argmax(2)
usage = (emit == np.arange(K)[None, :]).mean(0)
j = int(np.argmax((emit == np.arange(K)[None, :]).sum(1)))   # most-fluent resident

print(f"PER-WORD STATUS (frequent -> rare): meaning -> resident {j} SAYS  [English? / pop-usage]")
for r in rank:
    said = VOCAB[int(emit[j, r])]; ok = "EN  " if emit[j, r] == r else "code"
    print(f"  {VOCAB[r]:9} -> says {said:9}  [{ok}  pop {usage[r]*100:3.0f}%]  freq={FREQ[r]:.3f}")

print("=" * 80)
print(f"LIVE conversation (resident {j} speaks -> a peer listener decodes), meanings Zipf+urgency sampled:")
rng = np.random.default_rng(7); ls = (j + 1) % len(spk); good = 0; N_DEMO = 12
for _ in range(N_DEMO):
    m = int(rng.choice(K, p=(0.75 * FREQ + 0.25 * PS)))
    s = int(emit[j, m]); heard = decode(lis[ls, s])
    mark = "OK" if heard == m else "x "
    print(f"  [{mark}] means '{VOCAB[m]:8}' -> says '{VOCAB[s]:8}' -> heard '{VOCAB[heard]:8}'")
    good += heard == m
print("=" * 80)
# population-wide TWO-WAY comprehension: over many (speaker,listener,meaning), does the listener decode it?
rng2 = np.random.default_rng(123); N = len(spk); comp = 0; comp_en = 0; en_n = 0; T = 6000
for _ in range(T):
    sp, lsr = rng2.integers(N), rng2.integers(N); m = int(rng2.choice(K, p=(0.75 * FREQ + 0.25 * PS)))
    s = int(emit[sp, m]); ok = decode(lis[lsr, s]) == m; comp += ok
    if emit[sp, m] == m: en_n += 1; comp_en += ok           # comprehension when speaker DID use English
comp_rate = comp / T; comp_en_rate = comp_en / max(en_n, 1)
n_en = int((emit[j] == np.arange(K)).sum())
print(f"resident {j}: speaks {n_en}/{K} meanings in correct English; live demo {good}/{N_DEMO} understood")
print(f"population mean SPEAKER usage {usage.mean():.2f}  |  frequent half {usage[rank[:K//2]].mean():.2f}  "
      f"|  rare half {usage[rank[K//2:]].mean():.2f}")
print(f"population TWO-WAY comprehension (listener decodes correctly): {comp_rate:.2f}  "
      f"(chance {1/K:.2f}); when speaker used English: {comp_en_rate:.2f}")
print("READING: 'EN' = spoke the correct real English word; 'code' = uses another real word as a private")
print("symbol (functional, understood by peers, but not English). Every utterance is a real word.")
