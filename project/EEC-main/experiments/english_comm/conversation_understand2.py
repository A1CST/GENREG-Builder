"""CONVERSATION UNDERSTAND v2 -- organism comprehends real English via a proper embedding model
(nomic-embed-text), removing both the literary-corpus limitation and the LLM-matching crutch.
Same test: held-out paraphrase comprehension + a live pass comprehending the running LLM's messages."""
import os, json, re, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); GEN = os.environ.get("LLM_MODEL", "llama3.2:3b")
SIT = {"greeting": ("hello", "hello there friend"), "thanks": ("thank you", "you are welcome"),
       "farewell": ("goodbye", "see you later"), "ask_name": ("what is your name", "i am a friend"),
       "hungry": ("i am hungry", "go find food"), "ask_help": ("can you help me", "i will help"),
       "lost": ("i am lost", "stay calm there"), "tired": ("i am tired", "you should rest"),
       "happy": ("i am happy", "that is great"), "danger": ("there is danger", "run away now")}
LOG = open(os.path.join(HERE, "conversation_understand2_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def post(ep, payload):
    req = urllib.request.Request(f"http://localhost:11434/api/{ep}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
def embed(t):
    v = np.array(post("embed", {"model": "nomic-embed-text", "input": t})["embeddings"][0]); return v/(np.linalg.norm(v)+1e-9)
def gen(p, temp=0.8, n=80): return post("generate", {"model": GEN, "prompt": p, "stream": False, "options": {"temperature": temp, "num_predict": n}})["response"].strip()

names = list(SIT)
out("Generating paraphrases + embedding with nomic-embed-text...")
para = {}
for s in names:
    txt = gen(f'Give 8 different short everyday ways to say "{SIT[s][0]}". One per line, lowercase, no numbering.')
    lines = [l.strip(" -.\t") for l in txt.splitlines() if l.strip() and len(l) < 60]
    para[s] = [SIT[s][0]] + lines[:8]
rng = np.random.default_rng(0); cent = []; held = []
for s in names:
    embs = [(embed(p), p) for p in para[s]]; rng.shuffle(embs); k = max(2, int(0.6*len(embs)))
    cent.append(np.mean([e for e,_ in embs[:k]], 0))
    for e,p in embs[k:]: held.append((s,e,p))
cent = np.stack(cent); correct = sum(names[int(np.argmax(cent@e))] == s for s,e,p in held)
out("="*70); out(f"HELD-OUT paraphrase comprehension (nomic): {correct}/{len(held)} = {correct/len(held):.2f}  (chance {1/len(names):.2f})")
out("="*70); out("LIVE: organism comprehends the running LLM's messages via nomic embeddings, replies:")
msg = "hey there how is it going"
for turn in range(8):
    s = names[int(np.argmax(cent@embed(msg)))]; reply = SIT[s][1]
    out(f'   turn {turn+1}: LLM: "{msg:38}" -> [understood: {s:9}] -> "{reply}"')
    msg = " ".join(re.findall(r"[a-z]+", gen(f'Friendly chat. You said "{msg}", friend said "{reply}". Reply in one short natural sentence (3-7 words):', 0.6).lower())[:9])
out("done"); LOG.close()
