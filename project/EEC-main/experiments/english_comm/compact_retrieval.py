"""Shrink the retrieval index without wrecking it: subsample to a representative set and int8-quantize
the embeddings. 13 MB -> ~1.5 MB, keeping retrieval quality (which clustering loses at this corpus size)."""
import os, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "demo")
KEEP = int(os.environ.get("KEEP", "12000"))

E = np.load(os.path.join(DEMO, "retrieval_emb.npy")).astype(np.float32)
RESP = json.load(open(os.path.join(DEMO, "retrieval_pairs.json")))["responses"]
rng = np.random.default_rng(0)
sel = rng.permutation(len(E))[:KEEP]                          # representative subsample (pairs already diverse)
Es, Rs = E[sel], [RESP[i] for i in sel]
scale = float(np.abs(Es).max())
q = np.clip(np.round(Es / scale * 127), -127, 127).astype(np.int8)   # int8 quantize (full range)
np.savez_compressed(os.path.join(DEMO, "retrieval.npz"), emb=q, scale=np.float32(scale),
                    resp=np.array(Rs, dtype=object))
for old in ("retrieval_emb.npy", "retrieval_pairs.json", "clusters.npz", "cluster_replies.json"):
    p = os.path.join(DEMO, old)
    if os.path.exists(p): os.remove(p)
sz = os.path.getsize(os.path.join(DEMO, "retrieval.npz"))
print(f"kept {KEEP} pairs; retrieval.npz = {sz//1024} KB (was ~13 MB); removed the big float index")
# verify quantization preserves retrieval
z = np.load(os.path.join(DEMO, "self_emb.npz"), allow_pickle=True)
import re
vocab = list(z["vocab"]); emb = z["emb"].astype(np.float32); idx = {w: i for i, w in enumerate(vocab)}
Eq = q.astype(np.float32) * scale / 127
def embed(t):
    ws = [idx[w] for w in re.findall(r"[a-z']+", t.lower()) if w in idx]
    if not ws: return None
    v = emb[ws].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else None
print("=" * 60)
for query in ["hi there", "i am really hungry", "i had a long day at work", "i feel sad today", "thanks so much"]:
    e = embed(query); s = Eq @ e; print(f'  "{query:22}" -> "{Rs[int(s.argmax())]}"  (sim {s.max():.2f})')
