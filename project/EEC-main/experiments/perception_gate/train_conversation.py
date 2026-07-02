"""TRAIN a conversational organism correctly -- no templates, no hand-written rules, gradient-free.

The law: BE-UNDERSTOOD-OR-DIE. An organism hears a turn, must emit a reply (real words). A listener
(the distributional perception engine) decodes the reply back to a meaning. The organism survives only
if the listener recovers a meaning that KEEPS THE CONVERSATION GOING -- i.e. matches how real
conversations actually flow. The organism is NEVER told the right answer; it only feels survival.
Everything is discovered: intents are clustered from the corpus, the conversational law is the corpus's
own turn-to-turn flow, expression emits the intent's real words. The reply policy is EVOLVED (mutation +
selection), gradient-free. Coherence/appropriateness EMERGE because un-understood organisms die.
"""
import os, re, random, collections, numpy as np
from real_perslot import build_embeddings, CORP
HERE = os.path.dirname(os.path.abspath(__file__))
SPK = re.compile(r"(?i)\b(?:person\s*[ab12]?|friend\s*\d*|other\s+(?:friend|person)|me|you|a|b)\s*:")
STOP = set("i i'm im a an the to of and you it's it is so that this my me we he she they them their was "
           "were be been have has had do did doing don't just like really for in on at with but yeah oh "
           "what how when too not no your we're they're get got going gonna wanna up out about".split())
K, POP, GENS = 14, 80, 120
random.seed(0); np.random.seed(0)
LOG = open(os.path.join(HERE, "train_conversation_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def dialogues():
    D = []
    with open(CORP, encoding="utf-8", errors="ignore") as f:
        for line in f:
            ts = []
            for part in re.split(r'\||"', SPK.sub(" | ", line)):
                w = [x for x in re.findall(r"[a-z']+", part.lower())]
                if len(w) >= 2: ts.append(w)
            if len(ts) >= 2: D.append(ts)
    return D


def kmeans(X, k, iters=25):
    rng = np.random.default_rng(0); C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            m = X[a == j]
            if len(m): C[j] = m.mean(0)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    return C


if __name__ == "__main__":
    out("building distributional perception from the corpus...")
    idx, E, n = build_embeddings(CORP)
    inv = {i: w for w, i in idx.items()}
    D = dialogues()

    def temb(words):
        v = [E[idx[w]] for w in words if w in idx and w not in STOP]
        return np.mean(v, 0) if v else None

    # collect turns + a meaning vector each
    turns, vecs = [], []
    for d in D:
        for t in d:
            v = temb(t)
            if v is not None: turns.append(t); vecs.append(v / (np.linalg.norm(v) + 1e-9))
    V = np.array(vecs)
    out(f"{len(turns):,} turns, {len(D):,} dialogues. clustering into {K} discovered intents...")
    C = kmeans(V, K)
    intent_of = ((V[:, None] - C[None]) ** 2).sum(2).argmin(1)

    # label each intent by its most distinctive words (for us to read; not used in training)
    words_by = [collections.Counter() for _ in range(K)]
    for t, k in zip(turns, intent_of):
        for w in t:
            if w in idx and w not in STOP: words_by[k][w] += 1
    labels = [" ".join(w for w, _ in words_by[k].most_common(5)) for k in range(K)]

    # the WORLD's conversational law: how real turns flow, intent(t) -> intent(t+1)
    pos = {}; i = 0
    for d in D:
        for t in d:
            if temb(t) is not None: pos[id(t)] = i; i += 1
    trans = np.zeros((K, K))
    for d in D:
        prev = None
        for t in d:
            v = temb(t)
            if v is None: prev = None; continue
            k = int(((v / (np.linalg.norm(v) + 1e-9) - C) ** 2).sum(1).argmin())
            if prev is not None: trans[prev, k] += 1
            prev = k
    law = trans.argmax(1)                                  # the viable reply intent for each heard intent
    freq = np.bincount(intent_of, minlength=K).astype(float); freq /= freq.sum()

    # expression: emit a few of intent b's real words (sampled, so being understood is NOT guaranteed)
    emit_pool = [[w for w, _ in words_by[b].most_common(25)] for b in range(K)]

    def express(b):
        pool = emit_pool[b] or ["..."]
        return random.sample(pool, min(4, len(pool)))

    def perceive(words):                                  # listener decodes emitted words -> intent
        v = [E[idx[w]] for w in words if w in idx]
        if not v: return -1
        u = np.mean(v, 0); u /= np.linalg.norm(u) + 1e-9
        return int(((u - C) ** 2).sum(1).argmin())

    # FITNESS = survival: heard intent A -> organism replies pi[A] -> emit words -> listener decodes B'.
    # survive iff B' == law[A]  (understood AND keeps the conversation going). organism never told law.
    def fitness(pi, trials=600):
        ok = 0
        for _ in range(trials):
            a = int(np.random.choice(K, p=freq))
            b_intended = pi[a]
            bp = perceive(express(b_intended))
            ok += (bp == law[a])
        return ok / trials

    out(f"chance survival ~ {1/K:.3f}. evolving the reply policy (gradient-free, pop {POP})...")
    pop = [np.random.randint(0, K, K) for _ in range(POP)]
    best = None
    for g in range(GENS):
        scored = sorted(((fitness(p), p) for p in pop), key=lambda x: -x[0])
        best = scored[0]
        if g % 20 == 0 or g == GENS - 1:
            out(f"  gen {g:3d}: best survival {best[0]:.3f}   mean {np.mean([s for s,_ in scored]):.3f}")
        elite = [p for _, p in scored[:POP // 4]]
        pop = elite[:]
        while len(pop) < POP:
            p = random.choice(elite).copy()
            for _ in range(random.randint(1, 3)): p[random.randrange(K)] = random.randrange(K)
            pop.append(p)
    pi = best[1]
    out("=" * 74)
    out(f"final survival {best[0]:.3f}  (chance {1/K:.3f}).  policy matches world-law on "
        f"{int((pi==law).sum())}/{K} intents -- DISCOVERED from survival alone, never told.")
    out("=" * 74)
    out("real conversations (input turn -> organism perceives intent -> EVOLVED reply -> emitted words):")
    random.seed(7)
    shown = 0
    for d in D:
        if shown >= 8: break
        if len(d) >= 2 and temb(d[0]) is not None:
            inp = d[0]; v = temb(inp); v /= np.linalg.norm(v) + 1e-9
            a = int(((v - C) ** 2).sum(1).argmin())
            reply_intent = pi[a]; words = express(reply_intent)
            out(f"  you: {' '.join(inp)[:60]}")
            out(f"       [heard intent: {labels[a]}]")
            out(f"  bot: {' '.join(words)}   [reply intent: {labels[reply_intent]}]")
            out("")
            shown += 1
    out("done"); LOG.close()
