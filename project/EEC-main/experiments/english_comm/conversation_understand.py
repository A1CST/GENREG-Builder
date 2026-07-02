"""CONVERSATION UNDERSTAND -- the organism comprehends varied real English itself (no LLM matching).

The live-chat crutch was: the LLM mapped each incoming message to the organism's nearest prompt. Here
the ORGANISM comprehends, grounded in real embeddings: it perceives a message as the mean of its word
embeddings and classifies the conversational situation. We test generalisation to HELD-OUT PARAPHRASES
(different ways to say the same thing, LLM-generated) -- the real off-vocab fix. Then a live pass where
the organism comprehends the running LLM's actual messages and replies, no oracle matching.
"""
import os, json, re, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
G = np.load(os.path.join(HERE, "grounding_xl.npz"), allow_pickle=True)
VOCAB = list(G["vocab"]); EMB = G["emb"].astype(float); widx = {w: i for i, w in enumerate(VOCAB)}
SIT = {  # situation -> seed phrase, and a compositional reply
    "greeting": ("hello", "hello there friend"), "thanks": ("thank you", "you are welcome"),
    "farewell": ("goodbye", "see you later"), "ask_name": ("what is your name", "i am a friend"),
    "hungry": ("i am hungry", "go find food"), "ask_help": ("can you help me", "i will help"),
    "lost": ("i am lost", "stay calm there"), "tired": ("i am tired", "you should rest"),
    "happy": ("i am happy", "that is great"), "danger": ("there is danger", "run away now")}
LOG = open(os.path.join(HERE, "conversation_understand_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def ask(prompt, npred=80, temp=0.8):
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "num_predict": npred}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["response"].strip()


def embed(msg):
    ws = [widx[w] for w in re.findall(r"[a-z]+", msg.lower()) if w in widx]
    if not ws: return None
    v = EMB[ws].mean(0); return v / (np.linalg.norm(v) + 1e-9)


if __name__ == "__main__":
    names = list(SIT)
    out("Generating paraphrases per situation from the LLM...")
    para = {}
    for s in names:
        seed = SIT[s][0]
        txt = ask(f'Give 8 different short everyday ways to say "{seed}". One per line, lowercase, no numbering.')
        lines = [l.strip(" -.\t") for l in txt.splitlines() if l.strip()]
        ps = [seed] + [l for l in lines if embed(l) is not None][:8]
        para[s] = ps
        out(f"   {s:9}: {ps[:4]} ...")
    # embed, split train/held-out, nearest-centroid comprehension
    rng = np.random.default_rng(0); train_c = {}; held = []
    for s in names:
        embs = [(embed(p), p) for p in para[s] if embed(p) is not None]
        rng.shuffle(embs); k = max(2, int(0.6 * len(embs)))
        train_c[s] = np.mean([e for e, _ in embs[:k]], 0)
        for e, p in embs[k:]: held.append((s, e, p))
    cent = np.stack([train_c[s] for s in names])
    correct = 0
    for s, e, p in held:
        pred = names[int(np.argmax(cent @ e))]; correct += (pred == s)
    out("=" * 70)
    out(f"HELD-OUT paraphrase comprehension (organism classifies UNSEEN phrasings): "
        f"{correct}/{len(held)} = {correct/len(held):.2f}  (chance {1/len(names):.2f})")
    out("=" * 70)
    out("LIVE: organism comprehends the running LLM's own messages (no oracle matching) and replies:")
    msg = "hey there how is it going"
    for turn in range(8):
        e = embed(msg)
        s = names[int(np.argmax(cent @ e))] if e is not None else "greeting"
        reply = SIT[s][1]
        out(f'   turn {turn+1}: LLM: "{msg:34}" -> [understood: {s:9}] -> organism: "{reply}"')
        msg = " ".join(re.findall(r"[a-z]+", ask(f'Friendly chat. You said "{msg}", friend said "{reply}". '
                                                 f'Reply in one short natural sentence (3-7 words):', temp=0.5).lower())[:8])
    out("done"); LOG.close()
