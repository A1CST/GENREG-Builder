"""Is DIM=100 saturating? Build the PPMI matrix, run SVD out to 600 dims, and read the singular-value
spectrum + cumulative energy. If energy is still climbing fast at 100, we truncated real signal."""
import os, re, collections, numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
HERE = os.path.dirname(os.path.abspath(__file__))
WIKI = os.path.join(HERE, "wiki_corpus.txt")
DIALOG = [os.path.join(HERE, "emotional_dialog.txt"), os.path.join(os.path.dirname(HERE), "english_comm", "chat_corpus.txt")]
WIN, KVOCAB, MAXTOK, NMAX = 5, 14000, 16_000_000, 600
def log(s): print(s, flush=True)


def stream(paths, maxtok):
    nt = 0
    for p in paths:
        if not os.path.exists(p): continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                for w in re.findall(r"[a-z']+", line.lower()):
                    if len(w) >= 2:
                        yield w; nt += 1
                        if nt >= maxtok: return


log("counting + building co-occurrence...")
cnt = collections.Counter()
for w in stream(DIALOG + [WIKI], MAXTOK): cnt[w] += 1
vocab = [w for w, _ in cnt.most_common(KVOCAB)]; idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
co = sparse.csr_matrix((V, V), dtype=np.float32); ids = []
def add(ids):
    arr = np.array(ids, np.int32); ps = []
    for d in range(1, WIN + 1):
        r, c = arr[:-d], arr[d:]; m = (r >= 0) & (c >= 0)
        ps.append(np.stack([r[m], c[m]], 1)); ps.append(np.stack([c[m], r[m]], 1))
    P = np.concatenate(ps)
    return sparse.coo_matrix((np.ones(len(P), np.float32), (P[:, 0], P[:, 1])), shape=(V, V)).tocsr()
for w in stream(DIALOG + [WIKI], MAXTOK):
    ids.append(idx.get(w, -1))
    if len(ids) >= 2_000_000: co = co + add(ids); ids = []
if ids: co = co + add(ids)
co = co.tocoo(); tot = co.data.sum(); rs = np.asarray(co.tocsr().sum(1)).ravel() + 1e-9
pmi = np.log(co.data * tot / (rs[co.row] * rs[co.col]) + 1e-12); keep = pmi > 0
ppmi = sparse.coo_matrix((pmi[keep], (co.row[keep], co.col[keep])), shape=(V, V)).tocsr()
log(f"vocab {V:,}, ppmi nnz {ppmi.nnz:,}. running SVD to {NMAX} dims...")

svd = TruncatedSVD(n_components=NMAX, random_state=0).fit(ppmi)
sv = svd.singular_values_
evr = svd.explained_variance_ratio_
cum = np.cumsum(evr)
log("=" * 64)
log(f"{'dim':>5} {'singular val':>13} {'cum.energy(sv^2)':>17} {'cum.var.ratio':>14}")
energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
for d in [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600]:
    if d <= NMAX:
        log(f"{d:>5} {sv[d-1]:>13.2f} {energy[d-1]:>17.3f} {cum[d-1]:>14.3f}")
log("=" * 64)
# saturation readout: how much signal lives beyond 100?
log(f"singular value drop 100->600: {sv[99]:.1f} -> {sv[-1]:.1f}  (ratio {sv[99]/sv[-1]:.2f}x)")
log(f"marginal energy in dims 100-200: {energy[199]-energy[99]:.3f}   200-600: {energy[-1]-energy[199]:.3f}")
log(f"variance-ratio captured by first 100 of {NMAX}: {cum[99]/cum[-1]:.1%}")
log("READING: if sv[100] is still large and energy keeps climbing past 100, dim-100 truncated real")
log("signal -> perception was bottlenecked by my hardcoded DIM, not the data.")
