"""Build REAL English grounding from the corpus: vocabulary, frequencies, and
PPMI-SVD word embeddings (genuine semantic geometry). Cached to grounding.npz.
"""
import os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "engine", "corpus.txt")
OUT = os.path.join(HERE, "grounding.npz")

K = 60          # vocabulary size (content words)
DIM = 24        # embedding dimension
WIN = 4         # co-occurrence window
MAXTOK = 2_000_000

STOP = set((
    "the of and to a in that is was it for as with his he be not by but at this had which "
    "have they from or her she him you all are we their has been would s t said were there "
    "what one when out them who could now then very more will your into some did may must "
    "shall can should our its any such only see well much little up down so no nor too most "
    "other another every each either neither both few many own being do does done am being "
    "where why how than after before while because though although however these those here "
    "him himself herself myself yourself itself about over under again further once also just "
    "even still back away off down upon without within among between against toward "
    "are us our ours yours hers theirs whom whose around almost ever never always often "
    "soon yet thus hence therefore moreover indeed perhaps rather quite somewhat "
    "having make made made get got go went come came know knew think thought say says "
    "seemed seem felt feel look looked good great long way thing things").split())

print("reading corpus...", flush=True)
toks = []
with open(CORPUS, encoding="utf-8", errors="ignore") as f:
    for line in f:
        for w in re.findall(r"[a-z]+", line.lower()):
            if len(w) >= 3:
                toks.append(w)
        if len(toks) >= MAXTOK:
            break
print(f"  {len(toks):,} tokens", flush=True)

freq = collections.Counter(toks)
cand = [w for w, _ in freq.most_common() if w not in STOP]
vocab = cand[:K]
idx = {w: i for i, w in enumerate(vocab)}
print("vocab:", " ".join(vocab[:20]), "...", flush=True)

# co-occurrence over the chosen vocab
co = np.zeros((K, K))
window = []
for w in toks:
    if w in idx:
        wi = idx[w]
        for v in window[-WIN:]:
            co[wi, v] += 1; co[v, wi] += 1
        window.append(wi)
        if len(window) > WIN:
            window.pop(0)
# PPMI
tot = co.sum(); rk = co.sum(1, keepdims=True) + 1e-9
pmi = np.log((co * tot) / (rk @ rk.T) + 1e-9)
ppmi = np.maximum(pmi, 0)
# SVD -> embeddings
U, S, _ = np.linalg.svd(ppmi)
emb = U[:, :DIM] * np.sqrt(S[:DIM])
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
freqs = np.array([freq[w] for w in vocab], float); freqs /= freqs.sum()

np.savez(OUT, vocab=np.array(vocab), emb=emb.astype(np.float32), freq=freqs.astype(np.float32))
print(f"saved {OUT}: emb {emb.shape}", flush=True)

# sanity: nearest neighbours show real semantic structure
sim = emb @ emb.T
print("nearest-neighbour sanity check:")
for w in ["mr", "elizabeth", "could", "good", "house"]:
    if w in idx:
        i = idx[w]; order = np.argsort(-sim[i])
        nn = [vocab[j] for j in order[1:5]]
        print(f"  {w:10} ~ {', '.join(nn)}")
