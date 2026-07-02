"""Make the demo learn what to say FROM the corpus (no hand-authored situations/replies). Turn the
conversational corpus into (utterance -> response) pairs, embed each utterance in the organism's OWN
embedding space, and save an index. The demo replies by retrieving the response that followed the most
similar utterance it has heard. Growing = growing the corpus."""
import os, re, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "chat_corpus.txt")
DEMO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "demo")
MAXPAIRS = int(os.environ.get("MAXPAIRS", "40000"))

z = np.load(os.path.join(DEMO, "self_emb.npz"), allow_pickle=True)
vocab = list(z["vocab"]); EMB = z["emb"].astype(np.float32); idx = {w: i for i, w in enumerate(vocab)}


def embed(text):
    ws = [idx[w] for w in re.findall(r"[a-z']+", text.lower()) if w in idx]
    if len(ws) < 1: return None
    v = EMB[ws].mean(0); n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else None


def clean(s):
    s = re.sub(r'["""\'`*\-_]+', " ", s)                     # strip quotes/markup
    s = re.sub(r"^\s*[a-z]{1,12}\s?\d?\s*:\s*", "", s)       # strip a leading speaker label  (girl: / person 1:)
    s = re.sub(r"\b[a-z]{1,12}\s?\d?\s*:\s*", " ", s)        # strip mid-string speaker labels
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    print("extracting (utterance -> response) pairs from the corpus...", flush=True)
    embs, resp, ctx = [], [], []
    seen = set()
    with open(CORPUS, encoding="utf-8", errors="ignore") as f:
        for line in f:
            sents = [clean(s) for s in re.split(r"[.?!]+", line.lower()) if s.strip()]
            sents = [s for s in sents if 3 <= len(s.split()) <= 12 and ":" not in s]
            for i in range(len(sents) - 1):
                u, r = sents[i], sents[i + 1]
                if r in seen or len(r) < 6: continue          # dedupe identical responses (variety)
                e = embed(u)
                if e is None: continue
                embs.append(e); resp.append(r); ctx.append(u); seen.add(r)
                if len(embs) >= MAXPAIRS: break
            if len(embs) >= MAXPAIRS: break
    E = np.stack(embs)
    np.save(os.path.join(DEMO, "retrieval_emb.npy"), E)
    json.dump({"responses": resp, "contexts": ctx}, open(os.path.join(DEMO, "retrieval_pairs.json"), "w"))
    print(f"  {len(resp)} pairs, emb {E.shape}; saved demo/retrieval_emb.npy + retrieval_pairs.json")
    # quick quality probe
    print("=" * 60)
    print("probe (input -> retrieved response that followed a similar utterance):")
    for q in ["hello there", "i am really hungry", "i am so tired", "thank you so much",
              "can you help me", "what a nice day", "i feel sad", "see you later"]:
        e = embed(q); sims = E @ e; top = sims.argsort()[-3:][::-1]
        print(f'  "{q:22}" -> "{resp[top[0]]}"   (sim {sims[top[0]]:.2f})')


if __name__ == "__main__":
    main()
