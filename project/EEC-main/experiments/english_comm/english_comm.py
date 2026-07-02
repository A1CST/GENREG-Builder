"""ENGLISH-GROUNDED COMMUNICATION.

Lesson stack we are applying:
  - arbitrary emergent codes DRIFT (agents invent their own symbols, not English)
  - a FROZEN grounded channel + evolved residual keeps real structure (skip connection)
  - COUPLED survival makes communication evolve at all (free-rider problem otherwise)
  - SEXUAL reproduction floors communication pro-social (this session's finding)

So: give agents an English-grounded PRIOR (innate word<->meaning mapping) plus an evolved
residual, put them under coupled-survival communicative pressure, and ask:
  (1) do they communicate in ENGLISH (use the English word for each meaning), or drift?
  (2) is English an evolutionarily STABLE Schelling point under mutation/noise?
  (3) does the reproduction operator (clone vs sexual crossover) change English stability?
"""
import numpy as np

ENGLISH = ["food", "water", "danger", "friend", "home", "big", "red", "cold", "run", "sleep"]
DISTRACT = ["z3", "qk", "vx", "wb", "fm", "jl"]            # non-English symbols agents COULD drift to
WORDS = ENGLISH + DISTRACT
M, V = len(ENGLISH), len(WORDS)                            # 10 meanings, 16 possible signals
# ground-truth English mapping: meaning m <-> word m (the first M words are the English names)


def new_pop(N, rng):
    return dict(spk=rng.normal(0, 0.3, (N, M, V)), lis=rng.normal(0, 0.3, (N, V, M)))


def eff_spk(g, anchor):
    """effective speak logits = frozen English prior (skip) + evolved residual."""
    pr = np.zeros((M, V))
    if anchor > 0:
        pr[np.arange(M), np.arange(M)] = anchor          # English: meaning m -> word m
    return g["spk"] + pr                                  # broadcast over population


def eff_lis(g, anchor):
    pr = np.zeros((V, M))
    if anchor > 0:
        pr[np.arange(M), np.arange(M)] = anchor          # English: word m -> meaning m
    return g["lis"] + pr


def play(g, anchor, rng, rounds=6, n_native=0, native_only=False):
    """coupled-survival referential game; returns per-agent energy + metrics.
    native_only: energy is awarded only for exchanges that involve a native speaker
    (i.e. the resource itself speaks English -- the WORLD requires English, not a grader)."""
    N = len(g["spk"]); en = np.zeros(N)
    ES = eff_spk(g, anchor); EL = eff_lis(g, anchor)
    succ = 0; tot = 0

    def award(i, j, ok):
        if not ok:
            return
        if native_only and i >= n_native and j >= n_native:
            return                                        # no native involved -> no food
        en[i] += 1; en[j] += 1

    for _ in range(rounds):
        perm = rng.permutation(N)
        for a in range(0, N - 1, 2):
            i, j = perm[a], perm[a + 1]
            m = rng.integers(M)
            w = int(np.argmax(ES[i, m])); mh = int(np.argmax(EL[j, w])); ok = (mh == m)
            award(i, j, ok)
            m2 = rng.integers(M); w2 = int(np.argmax(ES[j, m2])); mh2 = int(np.argmax(EL[i, w2])); ok2 = (mh2 == m2)
            award(j, i, ok2)
            succ += ok + ok2; tot += 2
    return en, succ / max(tot, 1)


def english_usage(g, anchor):
    """fraction of (agent, meaning) where the agent emits the ENGLISH word for that meaning."""
    ES = eff_spk(g, anchor); emit = ES.argmax(2)          # (N, M)
    return float((emit == np.arange(M)[None, :]).mean())


def _make_native(n, rng, big=5.0):
    """fixed native English speakers: residual dominated by the English identity mapping."""
    spk = rng.normal(0, 0.1, (n, M, V)); lis = rng.normal(0, 0.1, (n, V, M))
    spk[:, np.arange(M), np.arange(M)] += big
    lis[:, np.arange(M), np.arange(M)] += big
    return spk, lis


def evolve(anchor=2.0, repro="clone", gens=250, N=40, mut=0.25, noise=0.0,
           cull=0.25, seed=1, n_native=0, native_only=False):
    """n_native > 0 installs fixed English speakers the residents must understand to eat
    (cultural transmission): they are reset to English each gen, never culled, never parents."""
    rng = np.random.default_rng(seed)
    g = new_pop(N, rng)
    res = np.arange(n_native, N)                          # resident (evolving) indices
    traj = []
    for t in range(gens):
        if n_native > 0:                                  # refresh the native speakers (the English in the world)
            ns, nl = _make_native(n_native, rng)
            g["spk"][:n_native] = ns; g["lis"][:n_native] = nl
        if noise > 0:
            g["spk"][res] += rng.normal(0, noise, (len(res), M, V))
            g["lis"][res] += rng.normal(0, noise, (len(res), V, M))
        en, acc = play(g, anchor, rng, n_native=n_native, native_only=native_only)
        if n_native > 0:
            en[:n_native] = 1e9                           # natives never culled / not in breeding pool
        K = max(1, int(cull * len(res)))
        ren = en[res]; order = res[np.argsort(ren)]
        worst = order[:K]; top = order[max(0, len(order) - max(2, len(res) // 3)):]
        for w in worst:
            if repro == "clone":
                p = int(top[rng.integers(len(top))])
                g["spk"][w] = g["spk"][p] + rng.normal(0, mut, (M, V))
                g["lis"][w] = g["lis"][p] + rng.normal(0, mut, (V, M))
            else:
                pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
                ms = rng.random((M, V)) < 0.5; ml = rng.random((V, M)) < 0.5
                g["spk"][w] = np.where(ms, g["spk"][pa], g["spk"][pb]) + rng.normal(0, mut * 0.6, (M, V))
                g["lis"][w] = np.where(ml, g["lis"][pa], g["lis"][pb]) + rng.normal(0, mut * 0.6, (V, M))
        if t % 25 == 0 or t == gens - 1:
            eu = float((eff_spk(g, anchor)[res].argmax(2) == np.arange(M)[None, :]).mean())
            traj.append((t, acc, eu))
    return g, traj


def transcript(g, anchor, rng, n=8):
    ES = eff_spk(g, anchor); EL = eff_lis(g, anchor)
    i = 0; j = 1; lines = []
    for _ in range(n):
        m = int(rng.integers(M)); w = int(np.argmax(ES[i, m])); mh = int(np.argmax(EL[j, w]))
        lines.append(f'    mean "{ENGLISH[m]:7}" -> says "{WORDS[w]:7}" -> heard "{ENGLISH[mh]:7}"  {"OK" if mh==m else "x"}')
    return "\n".join(lines)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=" * 70)
    print("TEST 1 — does grounding produce ENGLISH? (anchor off vs on)")
    print("  english_usage: fraction of meanings spoken with the real English word")
    print("  (chance ~ 1/16 = 0.06)")
    for anchor in [0.0, 1.0, 2.0, 3.0]:
        accs, eng = [], []
        for sd in range(4):
            _, tr = evolve(anchor=anchor, repro="clone", gens=250, seed=sd)
            accs.append(tr[-1][1]); eng.append(tr[-1][2])
        print(f"  anchor={anchor:.0f}: accuracy={np.mean(accs):.2f}  english_usage={np.mean(eng):.2f}")

    print("=" * 70)
    print("TEST 2 — STABILITY under channel noise: clone vs sexual reproduction")
    for noise in [0.0, 0.15, 0.3]:
        for repro in ["clone", "sexual"]:
            eng = []
            for sd in range(4):
                _, tr = evolve(anchor=2.0, repro=repro, gens=300, noise=noise, seed=sd)
                eng.append(tr[-1][2])
            print(f"  noise={noise:.2f} {repro:7}: english_usage={np.mean(eng):.2f}")

    print("=" * 70)
    print("TEST 3 — sample conversation (anchor=2, after evolution):")
    g, _ = evolve(anchor=2.0, repro="sexual", gens=250, seed=1)
    print(transcript(g, 2.0, np.random.default_rng(2)))

    print("=" * 70)
    print("TEST 4 — ACQUIRING English from EXPOSURE (no innate prior, anchor=0)")
    print("  residents start with NO English; they must understand native speakers to eat.")
    print("  does the population converge to English purely from social pressure?")
    for nn in [0, 4, 8]:
        for repro in ["clone", "sexual"]:
            eng = []
            for sd in range(4):
                _, tr = evolve(anchor=0.0, repro=repro, gens=400, n_native=nn, seed=sd)
                eng.append(tr[-1][2])
            tag = "no natives" if nn == 0 else f"{nn} natives"
            print(f"  {tag:11} {repro:7}: resident english_usage={np.mean(eng):.2f}")

    print("=" * 70)
    print("TEST 5 — acquisition trajectory (8 natives, sexual, anchor=0, seed 1):")
    _, tr = evolve(anchor=0.0, repro="sexual", gens=400, n_native=8, seed=1)
    for t, acc, eu in tr:
        if t % 50 == 0:
            print(f"    gen {t:3}: english_usage={eu:.2f}")
    g, _ = evolve(anchor=0.0, repro="sexual", gens=400, n_native=8, seed=1)
    print("  conversation between two RESIDENTS who acquired English (no innate prior):")
    print(transcript({"spk": g["spk"][8:], "lis": g["lis"][8:]}, 0.0, np.random.default_rng(3)))
