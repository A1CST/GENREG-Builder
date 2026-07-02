"""Compress the retrieval index into a compact LEARNED model: the organism DISCOVERS its conversational
situations by clustering its own utterance embeddings (k-means), and keeps a few representative replies
per cluster (from the corpus). ~250 centroids + short reply lists instead of 23k stored pairs.
Discovered, not hand-authored; grows with the corpus. Tiny: ~100KB vs ~13MB."""
import os, re, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "demo")
K = int(os.environ.get("K", "250")); PER = 8; ITERS = 30

E = np.load(os.path.join(DEMO, "retrieval_emb.npy")).astype(np.float32)
RESP = json.load(open(os.path.join(DEMO, "retrieval_pairs.json")))["responses"]


def kmeans(X, k, iters, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)                              # cosine assign (unit vectors)
        for j in range(k):
            m = a == j
            if m.any():
                v = X[m].mean(0); n = np.linalg.norm(v)
                if n > 0: C[j] = v / n
    return C, (X @ C.T).argmax(1)


def main():
    print(f"clustering {len(E)} utterance embeddings into {K} discovered situations...", flush=True)
    C, assign = kmeans(E, K, ITERS)
    replies = []
    for j in range(K):
        members = np.where(assign == j)[0]
        if len(members) == 0:
            replies.append([]); continue
        order = members[np.argsort(-(E[members] @ C[j]))]     # responses closest to the situation centroid
        seen, reps = set(), []
        for i in order:
            r = RESP[i]
            if r not in seen and len(r.split()) >= 2:
                reps.append(r); seen.add(r)
            if len(reps) >= PER: break
        replies.append(reps)
    keep = [j for j in range(K) if replies[j]]                 # drop empty clusters
    C = C[keep]; replies = [replies[j] for j in keep]
    np.savez_compressed(os.path.join(DEMO, "clusters.npz"), cent=C.astype(np.float32))
    json.dump(replies, open(os.path.join(DEMO, "cluster_replies.json"), "w"))
    sz = os.path.getsize(os.path.join(DEMO, "clusters.npz")) + os.path.getsize(os.path.join(DEMO, "cluster_replies.json"))
    print(f"  {len(C)} situations kept; model size {sz//1024} KB (was ~13 MB)")
    # probe
    z = np.load(os.path.join(DEMO, "self_emb.npz"), allow_pickle=True)
    vocab = list(z["vocab"]); emb = z["emb"].astype(np.float32); idx = {w: i for i, w in enumerate(vocab)}
    def embed(t):
        ws = [idx[w] for w in re.findall(r"[a-z']+", t.lower()) if w in idx]
        if not ws: return None
        v = emb[ws].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else None
    print("=" * 60)
    for q in ["hi there", "i am really hungry", "i had a long day", "i feel sad today",
              "what do you do for fun", "thanks so much", "good night"]:
        e = embed(q); j = int(np.argmax(C @ e))
        print(f'  "{q:20}" -> {replies[j][:3]}')


if __name__ == "__main__":
    main()
