"""Kill the generic mode-collapse with a CONSTRAINT, not a tweak.

v2 collapsed: it learned to say the safe generic thing to everything, because 'be an appropriate reply'
is satisfied by filler. New law, reasoned: YOUR REPLY MUST IDENTIFY WHAT YOU ANSWERED. A listener hearing
the reply must be able to recover which input it responded to. Generic replies fit any input -> the
listener can't tell -> they die. Specific (on-topic) replies survive. Formally the reply must be both
APPROPRIATE to the input (follows it in real talk) AND CHARACTERISTIC of it (identifies it) = mutual
information. Still gradient-free, still no templates; only the survival law changed.
"""
import os, re, math, random, collections, numpy as np
from real_perslot import build_embeddings, CORP
from chatbot import build as build_ng, sample_next, EOT
from train_conversation import dialogues, kmeans
HERE = os.path.dirname(os.path.abspath(__file__))
STOP = set("i i'm im a an the to of and you it's it is so that this my me we he she they them their was "
           "were be been have has had do did doing don't just like really for in on at with but yeah oh "
           "what how when too not no your we're they're get got going gonna wanna up out about".split())
K, POP, GENS = 40, 80, 110
random.seed(0); np.random.seed(0)
LOG = open(os.path.join(HERE, "conversation_v3_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

if __name__ == "__main__":
    out(f"building perception + language (K={K})...")
    idx, E, n = build_embeddings(CORP); ng, uni, cooc = build_ng(); D = dialogues()

    def temb(t):
        v = [E[idx[w]] for w in t if w in idx and w not in STOP]
        return np.mean(v, 0) if v else None

    turns, vecs = [], []
    for d in D:
        for t in d:
            v = temb(t)
            if v is not None: turns.append(t); vecs.append(v / (np.linalg.norm(v) + 1e-9))
    Vv = np.array(vecs)
    C = kmeans(Vv, K); intent_of = ((Vv[:, None] - C[None]) ** 2).sum(2).argmin(1)
    words_by = [collections.Counter() for _ in range(K)]
    for t, k in zip(turns, intent_of):
        for w in t:
            if w in idx and w not in STOP: words_by[k][w] += 1
    labels = [" ".join(w for w, _ in words_by[k].most_common(4)) for k in range(K)]

    trans = np.zeros((K, K))
    for d in D:
        prev = None
        for t in d:
            v = temb(t)
            if v is None: prev = None; continue
            k = int(((v / (np.linalg.norm(v) + 1e-9) - C) ** 2).sum(1).argmin())
            if prev is not None: trans[prev, k] += 1
            prev = k
    freq = np.bincount(intent_of, minlength=K).astype(float); freq /= freq.sum()
    Pb_a = trans / (trans.sum(1, keepdims=True) + 1e-9)        # P(reply | heard) -- appropriateness
    Pa_b = trans / (trans.sum(0, keepdims=True) + 1e-9)        # P(heard | reply) -- identifiability

    def perceive(words):
        v = [E[idx[w]] for w in words if w in idx]
        if not v: return -1
        u = np.mean(v, 0); u /= np.linalg.norm(u) + 1e-9
        return int(((u - C) ** 2).sum(1).argmin())

    pool = [[w for w, _ in words_by[b].most_common(25)] for b in range(K)]
    def cheap(b): p = pool[b] or ["..."]; return random.sample(p, min(4, len(p)))

    def fitness(pi, trials=500):                               # be-understood AND mutually-informative
        s = 0.0
        for _ in range(trials):
            a = int(np.random.choice(K, p=freq)); b = pi[a]
            understood = (perceive(cheap(b)) == b)
            s += understood * math.sqrt(max(Pb_a[a, b], 0) * max(Pa_b[a, b], 0))   # appropriate x identifying
        return s / trials

    out(f"evolving reply policy under be-understood + identify-the-input (chance-ish; mutual info)...")
    popl = [np.random.randint(0, K, K) for _ in range(POP)]; best = None
    for g in range(GENS):
        scored = sorted(((fitness(p), p) for p in popl), key=lambda x: -x[0]); best = scored[0]
        if g % 30 == 0 or g == GENS - 1: out(f"  gen {g:3d}: best {best[0]:.3f}  mean {np.mean([s for s,_ in scored]):.3f}")
        elite = [p for _, p in scored[:POP // 4]]; popl = elite[:]
        while len(popl) < POP:
            p = random.choice(elite).copy()
            for _ in range(random.randint(1, 3)): p[random.randrange(K)] = random.randrange(K)
            popl.append(p)
    pi = best[1]
    distinct = len(set(pi.tolist()))
    out(f"distinct reply-intents used: {distinct}/{K}  (v2 collapsed toward 1; higher = less generic)\n")

    DANGLE = set("a an the to of and or but so i you it is are was for in on at with my your we that this maybe just about like".split())
    def gen_c(topic, maxlen=11, beta=4.0):
        st = {w: 1.0 for w in topic}; ctx = [EOT]; o = []
        for _ in range(maxlen):
            nxt = sample_next(ng, uni, ctx, st, cooc, beta)
            if nxt == EOT:
                if len(o) >= 3: break
                continue
            if len(o) >= 2 and nxt == o[-1] == o[-2]: continue
            o.append(nxt); ctx = (ctx + [nxt])[-3:]
        while o and o[-1] in DANGLE: o.pop()
        return o

    def fluency(w):
        return -9 if len(w) < 3 else sum(math.log1p(ng[3][tuple(w[i-2:i])].get(w[i], 0)) for i in range(2, len(w))) / (len(w) - 2)

    def margin(w, B):                                          # how UNAMBIGUOUSLY the utterance means B
        v = [E[idx[x]] for x in w if x in idx]
        if not v: return -9
        u = np.mean(v, 0); u /= np.linalg.norm(u) + 1e-9
        d = ((u - C) ** 2).sum(1); o = d.argsort()
        return (d[o[1]] - d[o[0]]) if o[0] == B else -9        # decodes to B, by how clear a margin

    def express(B, nc=90):                                     # identify-the-intent applied to the utterance
        topic = [w for w, _ in words_by[B].most_common(15)]
        cands = [g for g in (gen_c(topic, beta=7.0) for _ in range(nc)) if len(g) >= 3]
        scored = [(margin(g, B), fluency(g), g) for g in cands]
        scored = [s for s in scored if s[0] > 0 and s[1] > -1]   # must clearly mean B, and be fluent-ish
        if not scored: return topic[:3]
        return max(scored, key=lambda s: s[0] + 0.15 * s[1])[2]  # most clearly-B, fluency as a nudge

    out("=" * 76)
    out("conversations (input -> perceived -> EVOLVED reply -> FLUENT utterance). nothing authored.")
    out("=" * 76)
    random.seed(11); shown = 0
    for d in D:
        if shown >= 12: break
        if temb(d[0]) is not None:
            inp = d[0]; v = temb(inp); v /= np.linalg.norm(v) + 1e-9
            a = int(((v - C) ** 2).sum(1).argmin()); B = int(pi[a]); utt = express(B); shown += 1
            out(f"  you: {' '.join(inp)[:62]}")
            out(f"  bot: {' '.join(utt)}    [reply: {labels[B]}]\n")
    out("done"); LOG.close()
