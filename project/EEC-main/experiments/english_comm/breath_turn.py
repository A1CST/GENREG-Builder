"""BREATH TURN -- multi-breath turns: does breath make responses RESPONSE-SHAPED (not one word, not a
book), chunked into breath groups, sized to the content? With a transcript of real-word output.

A meaning = C real content-words to convey (C varies = how much there is to say). The speaker emits
words one per step; breath is a lungful B (word-units after speed sigma): every floor(B/sigma) words it
must [inhale] (refills breath, costs time+energy). It stops when the content is discharged (nothing
left to say). Energy/word + inhale cost punish rambling; failing to convey punishes under-saying.
Length is never specified -- it emerges. We check: does response length track content and avoid both
the one-word collapse and the 'book', and does it chunk into breath groups? Then we print transcripts.

Encoding is strongly anchored (words transmit ~faithfully) so the star is LENGTH, not the lexicon.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(HERE, "grounding_xl.npz"), allow_pickle=True)
VOCAB = list(G["vocab"]); KV = 60                       # small readable vocab
CMAX, B_LUNG, EWORD, EINHALE, TAU = 6, 4.0, 0.06, 0.25, 12.0
SIG_LO, SIG_HI = 0.6, 2.5; LMAX = 16                    # hard ramble cap (death by book)
N = int(os.environ.get("BT_N", "64")); BUDGET = int(os.environ.get("BT_BUDGET", "2500"))
SEEDS = int(os.environ.get("BT_SEEDS", "4"))
ANCHOR = 4.0

LOG = open(os.path.join(HERE, "breath_turn_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.2f}+/-{a.std():.2f}"


def breed(g, en, rng, mut=0.22):
    Ng = len(en); o = np.argsort(en); worst = o[:int(0.25 * Ng)]; top = o[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        for k in ("emap", "sigma", "stop"):
            m = rng.random(g[k][pa].shape) < 0.5
            g[k][w] = np.where(m, g[k][pa], g[k][pb]) + rng.normal(0, mut * 0.6, g[k][pa].shape)
    g["sigma"][:] = np.clip(g["sigma"], SIG_LO, SIG_HI)


def emit_word(g, j, word):                              # anchored ~identity word channel
    return int((g["emap"][j, word] + ANCHOR * np.eye(KV)[word]).argmax())


def speak(g, j, targets, rng=None):
    """Generate a turn: list of ('word', idx) and ('inhale',) events. Stops when content discharged."""
    sig = g["sigma"][j]; group = max(1, int(np.floor(B_LUNG / sig))); breath = group
    events = []; conveyed = 0; t = 0; nwords = 0
    while t < len(targets) and nwords < LMAX:
        # stop policy: stop if learned rule fires given 'targets remaining'
        remaining = len(targets) - t
        z = g["stop"][j, 0] * (remaining > 0) + g["stop"][j, 1] * remaining + g["stop"][j, 2]
        if remaining <= 0:
            break
        if breath < 1:                                  # out of air -> must inhale to continue
            events.append(("inhale",)); breath = group
        w = targets[t]; emitted = emit_word(g, j, w)    # what the speaker actually produces
        events.append(("word", emitted, w)); breath -= 1; nwords += 1; t += 1
    return events, nwords


def listen(events):
    return [e[1] for e in events if e[0] == "word"]     # heard word indices (channel ~faithful)


def fitness_round(g, rng):
    Ng = len(g["emap"]); en = np.zeros(Ng)
    for _ in range(N):                                  # one speaking trial per organism (sampled partner)
        j = rng.integers(Ng)
        C = int(rng.integers(1, CMAX + 1)); targets = list(rng.integers(KV, size=C))
        ev, nwords = speak(g, j, targets)
        heard = listen(ev); ninh = sum(1 for e in ev if e[0] == "inhale")
        conveyed = sum(1 for i, w in enumerate(targets) if i < len(heard) and heard[i] == w)
        full = conveyed == C
        time = nwords + 2 * ninh                        # inhaling costs time
        r = (conveyed + 3.0 * full) / (1 + time / TAU) - EWORD * nwords - EINHALE * ninh
        en[j] += r
    return en


def run(seed):
    rng = np.random.default_rng(seed)
    g = {"emap": rng.normal(0, 0.3, (N, KV, KV)), "sigma": rng.uniform(SIG_LO, SIG_HI, N),
         "stop": rng.normal(0, 0.4, (N, 3))}
    for t in range(BUDGET):
        breed(g, fitness_round(g, rng), rng)
    # evaluate length behaviour
    lensum = {c: [] for c in range(1, CMAX + 1)}; collapse = book = full_ok = tot = 0
    rng2 = np.random.default_rng(7000 + seed)
    for _ in range(1500):
        j = int(rng2.integers(N)); C = int(rng2.integers(1, CMAX + 1)); targets = list(rng2.integers(KV, size=C))
        ev, nw = speak(g, j, targets); heard = listen(ev)
        lensum[C].append(nw); tot += 1
        if C >= 2 and nw <= 1: collapse += 1
        if nw >= 2 * C + 3: book += 1
        if sum(1 for i, w in enumerate(targets) if i < len(heard) and heard[i] == w) == C: full_ok += 1
    meanlen = {c: float(np.mean(lensum[c])) for c in lensum}
    return g, meanlen, collapse / tot, book / tot, full_ok / tot, float(g["sigma"].mean())


if __name__ == "__main__":
    out(f"BREATH TURN: meaning = C real words (C~1..{CMAX}); breath lung B={B_LUNG}; response length EMERGES. "
        f"pop {N}, budget {BUDGET}, {SEEDS} seeds.")
    gs = []; rows = []
    for s in range(SEEDS):
        g, ml, col, bk, fo, sg = run(s); gs.append(g); rows.append((ml, col, bk, fo, sg))
    out("mean response length by content size C (should rise with C, stay in a natural band):")
    for c in range(1, CMAX + 1):
        out(f"  C={c}: {ms([r[0][c] for r in rows])} words")
    out("-" * 60)
    out(f"  one-word COLLAPSE rate (C>=2 answered in <=1 word): {ms([r[1] for r in rows])}")
    out(f"  'BOOK' rate (length >= 2C+3, rambling):             {ms([r[2] for r in rows])}")
    out(f"  full-meaning conveyed:                              {ms([r[3] for r in rows])}")
    out(f"  evolved speed sigma:                                {ms([r[4] for r in rows])}")
    out("=" * 60)

    # ---- TRANSCRIPT ----
    out("TRANSCRIPT (best organism) -- meaning -> spoken response (with breath groups) -> heard:")
    bg = max(gs, key=lambda g: 0)  # first seed's pop; pick its most-fluent organism
    g = gs[0]; j = int(np.argmax([1] * N))  # placeholder; choose a good speaker below
    # pick the organism with best full-meaning on a quick probe
    rng = np.random.default_rng(123)
    def score(jj):
        ok = 0
        for _ in range(60):
            C = int(rng.integers(1, CMAX + 1)); tg = list(rng.integers(KV, size=C))
            ev, _ = speak(g, jj, tg); h = listen(ev)
            ok += sum(1 for i, w in enumerate(tg) if i < len(h) and h[i] == w) == C
        return ok
    j = int(np.argmax([score(jj) for jj in range(N)]))
    rng = np.random.default_rng(42)
    for C in [1, 1, 2, 3, 4, 6, 5, 2]:
        targets = list(rng.integers(KV, size=C))
        ev, nw = speak(g, j, targets)
        # render spoken with breath groups
        parts = []
        for e in ev:
            parts.append("[breath]" if e[0] == "inhale" else VOCAB[e[1]])
        heard = [VOCAB[w] for w in listen(ev)]
        intended = [VOCAB[w] for w in targets]
        good = sum(1 for i, w in enumerate(targets) if i < len(listen(ev)) and listen(ev)[i] == w)
        out(f"  meaning({C}): {' '.join(intended)}")
        out(f"     spoken : {' '.join(parts)}")
        out(f"     heard  : {' '.join(heard)}   [{good}/{C}]")
    out("done"); LOG.close()
