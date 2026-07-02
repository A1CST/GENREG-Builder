"""v4: fix input perception -- read CONTENT, not boilerplate.

v3's collapse traced to perception: 'hey i'm exhausted' perceived as a greeting because 'hey' dominates.
Fix (principled, not a hack): weight each word by its INFORMATIVENESS (inverse frequency / SIF). Glue
words ('hey','yeah','the') are common -> tiny weight; content words ('exhausted','bike') are rare ->
dominate the meaning. This is what a perception under 'distinguish your inputs efficiently' does. Then
the de-lumped inputs actually use the policy's distinct replies. Still gradient-free, no templates.
"""
import os, re, math, random, collections, numpy as np
from real_perslot import build_embeddings, CORP
from chatbot import build as build_ng, sample_next, EOT
from train_conversation import dialogues, kmeans
HERE = os.path.dirname(os.path.abspath(__file__))
STOPLBL = set("i i'm im a an the to of and you it's it is so that this my me we he she they them their "
              "was were be been have has had do did doing don't just like really for in on at with but "
              "yeah oh what how when too not no your we're they're get got going gonna wanna up out about".split())
K, POP, GENS, SIF_A = 40, 80, 110, 1e-3
random.seed(0); np.random.seed(0)
LOG = open(os.path.join(HERE, "conversation_v5_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

if __name__ == "__main__":
    out(f"building perception + language (K={K}, SIF informativeness weighting)...")
    idx, E, n = build_embeddings(CORP); ng, uni, cooc = build_ng(); D = dialogues()
    tot = sum(uni.values())
    sif = {w: SIF_A / (SIF_A + uni[w] / tot) for w in uni}     # informativeness weight per word

    def temb(t):                                              # informativeness-weighted meaning (reads content)
        ws = [(idx[w], sif.get(w, 1.0)) for w in t if w in idx]
        if not ws: return None
        v = np.sum([w * E[i] for i, w in ws], 0); nv = np.linalg.norm(v)
        return v / nv if nv > 0 else None

    turns, vecs = [], []
    for d in D:
        for t in d:
            v = temb(t)
            if v is not None: turns.append(t); vecs.append(v)
    Vv = np.array(vecs)
    C = kmeans(Vv, K); intent_of = ((Vv[:, None] - C[None]) ** 2).sum(2).argmin(1)
    words_by = [collections.Counter() for _ in range(K)]
    for t, k in zip(turns, intent_of):
        for w in t:
            if w in idx and w not in STOPLBL: words_by[k][w] += 1
    labels = [" ".join(w for w, _ in words_by[k].most_common(4)) for k in range(K)]

    trans = np.zeros((K, K))
    for d in D:
        prev = None
        for t in d:
            v = temb(t)
            if v is None: prev = None; continue
            k = int(((v - C) ** 2).sum(1).argmin())
            if prev is not None: trans[prev, k] += 1
            prev = k
    freq = np.bincount(intent_of, minlength=K).astype(float); freq /= freq.sum()
    Pb_a = trans / (trans.sum(1, keepdims=True) + 1e-9)
    Pa_b = trans / (trans.sum(0, keepdims=True) + 1e-9)

    def perceive(words):
        v = temb(words); return -1 if v is None else int(((v - C) ** 2).sum(1).argmin())

    pool = [[w for w, _ in words_by[b].most_common(25)] for b in range(K)]
    def cheap(b): p = pool[b] or ["..."]; return random.sample(p, min(4, len(p)))

    def fitness(pi, trials=500):
        s = 0.0
        for _ in range(trials):
            a = int(np.random.choice(K, p=freq)); b = pi[a]
            s += (perceive(cheap(b)) == b) * math.sqrt(max(Pb_a[a, b], 0) * max(Pa_b[a, b], 0))
        return s / trials

    out("evolving reply policy (be-understood + identify-the-input)...")
    popl = [np.random.randint(0, K, K) for _ in range(POP)]; best = None
    for g in range(GENS):
        scored = sorted(((fitness(p), p) for p in popl), key=lambda x: -x[0]); best = scored[0]
        if g % 30 == 0 or g == GENS - 1: out(f"  gen {g:3d}: best {best[0]:.3f}")
        elite = [p for _, p in scored[:POP // 4]]; popl = elite[:]
        while len(popl) < POP:
            p = random.choice(elite).copy()
            for _ in range(random.randint(1, 3)): p[random.randrange(K)] = random.randrange(K)
            popl.append(p)
    pi = best[1]
    out(f"distinct reply-intents: {len(set(pi.tolist()))}/{K}\n")

    DANGLE = set("a an the to of and or but so i you it is are was for in on at with my your we that this maybe just about like".split())
    def gen_c(topic, maxlen, beta=7.0):                       # breath: length is a sampled budget
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

    def margin(w, B):
        v = temb(w)
        if v is None: return -9
        d = ((v - C) ** 2).sum(1); o = d.argsort()
        return (d[o[1]] - d[o[0]]) if o[0] == B else -9

    def express(B, nc=160):                                   # be-understood + breath (short) + grammatical (fluency floor)
        topic = [w for w, _ in words_by[B].most_common(15)]
        cands = [g for g in (gen_c(topic, random.randint(4, 9)) for _ in range(nc)) if 4 <= len(g) <= 9]
        full = [(margin(g, B), fluency(g), len(g), g) for g in cands]
        scored = [s for s in full if s[0] > 0 and s[1] > 0.7]      # clearly means B AND grammatical
        if not scored: scored = [s for s in full if s[0] > 0]      # fall back if floor too strict
        return max(scored, key=lambda s: s[0] + 0.15 * s[1] - 0.12 * s[2])[3] if scored else topic[:3]

    out("=" * 76)
    out("conversations (informativeness perception -> EVOLVED reply -> FLUENT utterance). nothing authored.")
    out("=" * 76)
    tests = ["hey looks like rain again", "hey i'm so exhausted today", "i'm so hungry right now",
             "hey i'm having some trouble with my bike", "i'm just feeling really down today",
             "hey i'm so bored", "thanks so much for helping me out", "it's freezing out here",
             "do you wanna grab coffee", "i had such a great day today", "how was your weekend",
             "i can't sleep at all lately"]
    for u in tests:
        t = re.findall(r"[a-z']+", u.lower()); a = perceive(t)
        if a < 0: continue
        B = int(pi[a]); utt = express(B)
        out(f"  you: {u}")
        out(f"       [heard: {labels[a]}]  ->  bot: {' '.join(utt)}\n")
    out("done"); LOG.close()
