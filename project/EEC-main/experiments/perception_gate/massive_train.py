"""MASSIVE retrain of the conversational organism on big real corpora -- the experience lever.
Embeddings (perception): dialogue + WikiText-103 subset, efficient sparse PPMI + truncated SVD.
Dialogue model (intents/law/n-grams): ~4M words of real structured dialogue.
Same gradient-free, no-template pipeline as v5 (informativeness perception, mutual-info reply policy,
breath expression). Only the corpus changed -- 500K -> ~15M tokens."""
import os, re, math, random, collections, numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
HERE = os.path.dirname(os.path.abspath(__file__))
WIKI = os.path.join(HERE, "wiki_corpus.txt")
DIALOG = [os.path.join(HERE, "emotional_dialog.txt"), os.path.join(os.path.dirname(HERE), "english_comm", "chat_corpus.txt")]
DIM, WIN, KVOCAB, MAXTOK = 100, 5, 14000, 16_000_000
K, POP, GENS = 45, 80, 120
random.seed(0); np.random.seed(0)
LOG = open(os.path.join(HERE, "massive_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
STOPLBL = set("i i'm im a an the to of and you it's it is so that this my me we he she they them their was were be been have has had do did doing don't just like really for in on at with but yeah oh what how when too not no your we're they're get got going gonna wanna up out about".split())


def stream_tokens(paths, maxtok):
    nt = 0
    for p in paths:
        if not os.path.exists(p): continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                for w in re.findall(r"[a-z']+", line.lower()):
                    if len(w) >= 2:
                        yield w; nt += 1
                        if nt >= maxtok: return


def build_embeddings_big():
    out("  counting word frequencies...")
    cnt = collections.Counter()
    for w in stream_tokens(DIALOG + [WIKI], MAXTOK): cnt[w] += 1
    vocab = [w for w, _ in cnt.most_common(KVOCAB)]
    idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    out(f"  vocab {V:,}. building sparse co-occurrence...")
    co = sparse.csr_matrix((V, V), dtype=np.float32)
    buf = []
    def flush(buf):
        if not buf: return sparse.csr_matrix((V, V), dtype=np.float32)
        a = np.fromiter((x for p in buf for x in p), dtype=np.int32).reshape(-1, 2)
        r, c = a[:, 0], a[:, 1]
        return sparse.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(V, V)).tocsr()
    ids = []
    for w in stream_tokens(DIALOG + [WIKI], MAXTOK):
        ids.append(idx.get(w, -1))
        if len(ids) >= 2_000_000:
            arr = np.array(ids, np.int32); pairs = []
            for d in range(1, WIN + 1):
                r, c = arr[:-d], arr[d:]; m = (r >= 0) & (c >= 0)
                pairs.append(np.stack([r[m], c[m]], 1)); pairs.append(np.stack([c[m], r[m]], 1))
            P = np.concatenate(pairs)
            co = co + sparse.coo_matrix((np.ones(len(P), np.float32), (P[:, 0], P[:, 1])), shape=(V, V)).tocsr()
            ids = []
    if ids:
        arr = np.array(ids, np.int32); pairs = []
        for d in range(1, WIN + 1):
            r, c = arr[:-d], arr[d:]; m = (r >= 0) & (c >= 0)
            pairs.append(np.stack([r[m], c[m]], 1)); pairs.append(np.stack([c[m], r[m]], 1))
        P = np.concatenate(pairs)
        co = co + sparse.coo_matrix((np.ones(len(P), np.float32), (P[:, 0], P[:, 1])), shape=(V, V)).tocsr()
    out(f"  co-occurrence nnz {co.nnz:,}. PPMI + truncated SVD (dim {DIM})...")
    co = co.tocoo(); tot = co.data.sum(); rs = np.asarray(co.tocsr().sum(1)).ravel() + 1e-9
    pmi = np.log(co.data * tot / (rs[co.row] * rs[co.col]) + 1e-12)
    keep = pmi > 0
    ppmi = sparse.coo_matrix((pmi[keep], (co.row[keep], co.col[keep])), shape=(V, V)).tocsr()
    E = TruncatedSVD(n_components=DIM, random_state=0).fit_transform(ppmi)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return idx, E.astype(np.float32), cnt


def read_dialogues():
    SPK = re.compile(r"(?i)\b(?:person\s*[ab12]?|friend\s*\d*|other\s+(?:friend|person)|me|you|a|b)\s*:")
    D = []
    for p in DIALOG:
        if not os.path.exists(p): continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                ts = []
                for part in re.split(r'\|', SPK.sub(" | ", line)):
                    for sub in part.split("  "):
                        w = re.findall(r"[a-z']+", sub.lower())
                        if len(w) >= 2: ts.append(w)
                if len(ts) >= 2: D.append(ts)
    return D


def build_ngrams(D):
    EOT = "<eot>"; stream = []
    for d in D:
        for t in d: stream += t + [EOT]
    ng = {n: collections.defaultdict(collections.Counter) for n in (4, 3, 2)}
    uni = collections.Counter(stream)
    for i in range(len(stream)):
        for n in (4, 3, 2):
            if i >= n - 1: ng[n][tuple(stream[i - n + 1:i])][stream[i]] += 1
    cooc = collections.defaultdict(collections.Counter)
    for d in D:
        for t in d:
            u = set(t)
            for a in u:
                for b in u:
                    if a != b: cooc[a][b] += 1
    return ng, uni, cooc, EOT


def kmeans(X, k, iters=25):
    rng = np.random.default_rng(0); C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            m = X[a == j]
            if len(m): C[j] = m.mean(0)
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)


if __name__ == "__main__":
    out(f"MASSIVE build: embeddings on dialogue+wiki ({MAXTOK:,} tok cap), dialogue model on real dialogue.")
    idx, E, cnt = build_embeddings_big()
    out(f"embeddings: {E.shape[0]:,} x {E.shape[1]}")
    D = read_dialogues(); out(f"dialogues: {len(D):,}")
    ng, uni, cooc, EOT = build_ngrams(D); out(f"n-gram vocab {len(uni):,}, 4-gram ctx {len(ng[4]):,}")
    tot = sum(cnt.values()); sif = {w: 1e-3 / (1e-3 + cnt[w] / tot) for w in cnt}

    def temb(t):
        ws = [(idx[w], sif.get(w, 1.0)) for w in t if w in idx]
        if not ws: return None
        v = np.sum([w * E[i] for i, w in ws], 0); nv = np.linalg.norm(v)
        return v / nv if nv > 0 else None

    turns, vecs = [], []
    for d in D:
        for t in d:
            v = temb(t)
            if v is not None: turns.append(t); vecs.append(v)
    sub = np.random.choice(len(vecs), min(40000, len(vecs)), replace=False)
    out(f"clustering {len(sub):,} sampled turns into {K} intents...")
    C = kmeans(np.array(vecs)[sub], K)
    Vv = np.array(vecs); intent_of = ((Vv[:, None] - C[None]) ** 2).sum(2).argmin(1)
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
    Pb_a = trans / (trans.sum(1, keepdims=True) + 1e-9); Pa_b = trans / (trans.sum(0, keepdims=True) + 1e-9)

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
        if g % 40 == 0 or g == GENS - 1: out(f"  gen {g:3d}: best {best[0]:.3f}")
        elite = [p for _, p in scored[:POP // 4]]; popl = elite[:]
        while len(popl) < POP:
            p = random.choice(elite).copy()
            for _ in range(random.randint(1, 3)): p[random.randrange(K)] = random.randrange(K)
            popl.append(p)
    pi = best[1]; out(f"distinct reply-intents: {len(set(pi.tolist()))}/{K}\n")

    DANGLE = set("a an the to of and or but so i you it is are was for in on at with my your we that this maybe just about like".split())
    def sample_next(ctx, st, beta=7.0, temp=0.7):
        for n in (4, 3, 2):
            cand = ng[n].get(tuple(ctx[-(n - 1):]) if n > 1 else ())
            if cand and sum(cand.values()) >= 2:
                ws, cs = zip(*cand.items())
                w = [(c ** (1 / temp)) * math.exp(beta * (sum(math.log1p(cooc[x].get(c2, 0)) for c2 in st) / math.log1p(uni.get(x, 1) + 5) if x in st or st else 0)) for x, c in zip(ws, cs)]
                tt = sum(w); r = random.random() * tt
                for x, wt in zip(ws, w):
                    r -= wt
                    if r <= 0: return x
        return uni.most_common(1)[0][0]

    def gen_c(topic, maxlen):
        st = set(topic); ctx = [EOT]; o = []
        for _ in range(maxlen):
            nxt = sample_next(ctx, st)
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
    def express(B, nc=160):
        topic = [w for w, _ in words_by[B].most_common(15)]
        cands = [g for g in (gen_c(topic, random.randint(4, 9)) for _ in range(nc)) if 4 <= len(g) <= 9]
        full = [(margin(g, B), fluency(g), len(g), g) for g in cands]
        sc = [s for s in full if s[0] > 0 and s[1] > 0.7] or [s for s in full if s[0] > 0]
        return max(sc, key=lambda s: s[0] + 0.15 * s[1] - 0.12 * s[2])[3] if sc else topic[:3]

    out("=" * 76)
    tests = ["hey looks like rain again", "hey i'm so exhausted today", "i'm so hungry right now",
             "i'm having some trouble with my bike", "i'm just feeling really down today", "i'm so bored",
             "thanks so much for helping me out", "it's freezing out here", "do you wanna grab coffee",
             "i had such a great day today", "how was your weekend", "i can't sleep at all lately",
             "what music do you like", "my dog is sick and i'm worried"]
    for u in tests:
        t = re.findall(r"[a-z']+", u.lower()); a = perceive(t)
        if a < 0: continue
        B = int(pi[a]); utt = express(B)
        out(f"  you: {u}")
        out(f"       [heard: {labels[a]}] -> bot: {' '.join(utt)}\n")
    out("done"); LOG.close()
