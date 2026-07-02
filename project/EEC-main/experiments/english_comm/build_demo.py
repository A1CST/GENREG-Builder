"""Build the self-contained demo: the organism's OWN embeddings (PPMI-SVD over its conversational
corpus) -> demo/self_emb.npz, then validate comprehension on the demo's situation examples."""
import os, re, json, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "chat_corpus.txt")
DEMO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "demo")
DIM, WIN, K = int(os.environ.get("DIM", "120")), 5, int(os.environ.get("K", "4000"))


def build():
    text = open(CORPUS, encoding="utf-8", errors="ignore").read().lower()
    toks = [w for w in re.findall(r"[a-z']+", text) if len(w) >= 2]
    cnt = collections.Counter(toks); vocab = [w for w, _ in cnt.most_common(K)]
    idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    co = np.zeros((V, V), np.float32); win = []
    for w in toks:
        if w in idx:
            wi = idx[w]
            for v in win[-WIN:]: co[wi, v] += 1; co[v, wi] += 1
            win.append(wi)
            if len(win) > WIN: win.pop(0)
    tot = co.sum(); rk = co.sum(1, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log((co * tot) / (rk @ rk.T) + 1e-9), 0)
    U, S, _ = np.linalg.svd(ppmi.astype(np.float64))
    emb = U[:, :DIM] * np.sqrt(S[:DIM]); emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return vocab, idx, emb.astype(np.float32), len(toks)


def main():
    print("building the organism's own embeddings (PPMI-SVD over chat_corpus.txt)...", flush=True)
    vocab, idx, emb, ntok = build()
    print(f"  corpus {ntok} tokens, vocab {len(vocab)}, dim {emb.shape[1]}", flush=True)
    os.makedirs(DEMO, exist_ok=True)
    np.savez(os.path.join(DEMO, "self_emb.npz"), vocab=np.array(vocab), emb=emb)
    # sanity: semantic separation
    def se(w): return emb[idx[w]] if w in idx else None
    for a, b, c in [("hungry", "food", "goodbye"), ("tired", "sleep", "hello"), ("happy", "glad", "cold")]:
        if all(se(x) is not None for x in (a, b, c)):
            print(f"  cos({a},{b})={float(se(a)@se(b)):+.2f}  cos({a},{c})={float(se(a)@se(c)):+.2f}", flush=True)
    # validate comprehension on the demo situations (leave-one-example-out)
    sit = json.load(open(os.path.join(DEMO, "situations.json"))); names = list(sit)
    def embed(msg):
        ws = [idx[w] for w in re.findall(r"[a-z']+", msg.lower()) if w in idx]
        if not ws: return None
        v = emb[ws].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else None
    correct = tot_ = 0
    for held in names:
        for hi, hp in enumerate(sit[held]["examples"]):
            he = embed(hp)
            if he is None: continue
            protos = {}
            for s in names:
                ex = [embed(p) for j, p in enumerate(sit[s]["examples"]) if not (s == held and j == hi)]
                ex = [e for e in ex if e is not None]
                if ex: protos[s] = np.mean(ex, 0)
            ns = list(protos); Pm = np.stack([protos[s] for s in ns])
            pred = ns[int(np.argmax(Pm @ he))]; correct += (pred == held); tot_ += 1
    print("=" * 60)
    print(f"DEMO comprehension self-test (leave-one-example-out): {correct/max(1,tot_):.2f}  "
          f"(chance {1/len(names):.2f}, {tot_} examples)")
    print(f"saved demo/self_emb.npz -- run:  python3 {os.path.join(DEMO,'chat.py')}")


if __name__ == "__main__":
    main()
