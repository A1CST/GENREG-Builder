"""Scale + enrich the conversational organism -- still no templates, no hand-written replies, gradient-free.

Two enrichments over train_conversation.py:
  1) MORE intents (finer resolution) -- discovered by clustering, not authored.
  2) FLUENT expression by CONSTRAINT: to express reply-intent B, the organism generates many real
     candidate utterances (from the language it absorbed) and keeps only ones a listener DECODES BACK
     as B (be-understood), then picks the most fluent. The utterance is selected by survival, not written.

The reply policy is still EVOLVED under be-understood-or-die (now over more intents). Everything --
intents, the conversational law, the words, the sentences -- is discovered or selected, never authored.
"""
import os, re, math, random, collections, numpy as np
from real_perslot import build_embeddings, CORP
from chatbot import build as build_ng, sample_next, EOT
from train_conversation import dialogues, kmeans
HERE = os.path.dirname(os.path.abspath(__file__))
STOP = set("i i'm im a an the to of and you it's it is so that this my me we he she they them their was "
           "were be been have has had do did doing don't just like really for in on at with but yeah oh "
           "what how when too not no your we're they're get got going gonna wanna up out about".split())
K, POP, GENS = 40, 80, 90
random.seed(0); np.random.seed(0)
LOG = open(os.path.join(HERE, "conversation_v2_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

if __name__ == "__main__":
    out(f"building perception + language from corpus (K={K} intents)...")
    idx, E, n = build_embeddings(CORP)
    ng, uni, cooc = build_ng()
    D = dialogues()

    def temb(t):
        v = [E[idx[w]] for w in t if w in idx and w not in STOP]
        return np.mean(v, 0) if v else None

    turns, vecs = [], []
    for d in D:
        for t in d:
            v = temb(t)
            if v is not None: turns.append(t); vecs.append(v / (np.linalg.norm(v) + 1e-9))
    Vv = np.array(vecs)
    out(f"{len(turns):,} turns; clustering into {K} discovered intents...")
    C = kmeans(Vv, K)
    intent_of = ((Vv[:, None] - C[None]) ** 2).sum(2).argmin(1)
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
    law = trans.argmax(1)
    freq = np.bincount(intent_of, minlength=K).astype(float); freq /= freq.sum()

    def perceive(words):
        v = [E[idx[w]] for w in words if w in idx]
        if not v: return -1
        u = np.mean(v, 0); u /= np.linalg.norm(u) + 1e-9
        return int(((u - C) ** 2).sum(1).argmin())

    cheap_pool = [[w for w, _ in words_by[b].most_common(25)] for b in range(K)]
    def cheap_express(b):
        p = cheap_pool[b] or ["..."]; return random.sample(p, min(4, len(p)))

    def fitness(pi, trials=500):                            # evolve the reply policy (cheap expression)
        ok = 0
        for _ in range(trials):
            a = int(np.random.choice(K, p=freq))
            ok += (perceive(cheap_express(pi[a])) == law[a])
        return ok / trials

    out(f"chance {1/K:.3f}. evolving reply policy over {K} intents (gradient-free)...")
    pop = [np.random.randint(0, K, K) for _ in range(POP)]; best = None
    for g in range(GENS):
        scored = sorted(((fitness(p), p) for p in pop), key=lambda x: -x[0]); best = scored[0]
        if g % 30 == 0 or g == GENS - 1: out(f"  gen {g:3d}: best {best[0]:.3f}  mean {np.mean([s for s,_ in scored]):.3f}")
        elite = [p for _, p in scored[:POP // 4]]; pop = elite[:]
        while len(pop) < POP:
            p = random.choice(elite).copy()
            for _ in range(random.randint(1, 3)): p[random.randrange(K)] = random.randrange(K)
            pop.append(p)
    pi = best[1]
    out(f"reply policy matches world-law on {int((pi==law).sum())}/{K} intents.\n")

    DANGLE = set("a an the to of and or but so i you it is are was for in on at with my your we that this "
                 "maybe just about like".split())
    def gen_candidate(topicwords, maxlen=11, beta=4.0):
        st = {w: 1.0 for w in topicwords}; ctx = [EOT]; o = []
        for _ in range(maxlen):
            nxt = sample_next(ng, uni, ctx, st, cooc, beta)
            if nxt == EOT:
                if len(o) >= 3: break
                continue
            if len(o) >= 2 and nxt == o[-1] == o[-2]: continue
            o.append(nxt); ctx = (ctx + [nxt])[-3:]
        while o and o[-1] in DANGLE: o.pop()
        return o

    def fluency(words):
        if len(words) < 3: return -9
        return sum(math.log1p(ng[3][tuple(words[i - 2:i])].get(words[i], 0)) for i in range(2, len(words))) / (len(words) - 2)

    def fluent_express(B, n_cand=60):                      # generate -> keep only those UNDERSTOOD as B -> most fluent
        topic = [w for w, _ in words_by[B].most_common(15)]
        cands = [g for g in (gen_candidate(topic) for _ in range(n_cand)) if len(g) >= 3]
        good = [g for g in cands if perceive(g) == B]       # be-understood filter (no template)
        pool = good or cands
        return max(pool, key=fluency) if pool else topic[:3]

    out("=" * 76)
    out("real conversations (input -> perceived intent -> EVOLVED reply -> FLUENT utterance selected by")
    out("being-understood; understood-rate shown). nothing here is authored.")
    out("=" * 76)
    random.seed(11); shown = 0; und = 0
    for d in D:
        if shown >= 12: break
        if len(d) >= 1 and temb(d[0]) is not None:
            inp = d[0]; v = temb(inp); v /= np.linalg.norm(v) + 1e-9
            a = int(((v - C) ** 2).sum(1).argmin()); B = int(pi[a])
            utt = fluent_express(B); ok = (perceive(utt) == B); und += ok; shown += 1
            out(f"  you: {' '.join(inp)[:62]}")
            out(f"  bot: {' '.join(utt)}")
            out(f"       reply-intent[{labels[B]}]  understood={'yes' if ok else 'no'}\n")
    out(f"understood-as-intended: {und}/{shown}")
    out("done"); LOG.close()
