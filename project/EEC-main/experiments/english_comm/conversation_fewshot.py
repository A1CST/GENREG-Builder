"""FEW-SHOT conversational acquisition: learn to comprehend a NEW situation from a few exposures.
Prototype formed by experience-averaging (non-gradient memory). Sweep #exposures; measure comprehension
of the novel situation's held-out paraphrases against the existing situations as distractors."""
import os, json, re, urllib.request, numpy as np
GEN="llama3.2:3b"
SEED={"greeting":"hello","thanks":"thank you","farewell":"goodbye","hungry":"i am hungry",
      "ask_help":"can you help me","tired":"i am tired","happy":"i am happy","sad":"i feel sad",
      "danger":"there is danger","lost":"i am lost","cold":"i am cold","thirsty":"i need water"}
def out(s): print(s,flush=True)
def post(ep,p):
    r=urllib.request.Request(f"http://localhost:11434/api/{ep}",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=120).read())
def embed(t):
    v=np.array(post("embed",{"model":"nomic-embed-text","input":t})["embeddings"][0]); return v/(np.linalg.norm(v)+1e-9)
def gen(p): return post("generate",{"model":GEN,"prompt":p,"stream":False,"options":{"temperature":0.8,"num_predict":90}})["response"].strip()
names=list(SEED); out("embedding paraphrases (nomic)...")
E={}
for s in names:
    txt=gen(f'Give 10 short everyday ways to say "{SEED[s]}". one per line, lowercase.')
    ps=[SEED[s]]+[l.strip(" -.\t") for l in txt.splitlines() if l.strip() and len(l)<60][:10]
    E[s]=np.array([embed(p) for p in ps])
out("="*60); out("FEW-SHOT acquisition of a NEW situation (prototype = mean of k exposures):")
rng=np.random.default_rng(0)
for k in [1,2,3,5,8]:
    accs=[]
    for novel in names:                                  # leave-one-out: 'novel' is the new concept
        others=[s for s in names if s!=novel]
        base=np.stack([E[s][:6].mean(0) for s in others])   # existing situations (well-known)
        for _ in range(8):
            perm=rng.permutation(len(E[novel])); expo,test=perm[:k],perm[k:]
            proto=E[novel][expo].mean(0)                   # learn novel from k exposures (averaging)
            cents=np.vstack([base,proto[None]]); lab=others+[novel]
            ok=np.mean([lab[int(np.argmax(cents@E[novel][i]))]==novel for i in test])
            accs.append(ok)
    out(f"  {k} exposure(s): novel-situation comprehension {np.mean(accs):.2f}  (chance {1/len(names):.2f})")
out("READING: good comprehension from few exposures => the organism ACQUIRES a new conversational")
out("situation from experience (prototype memory), gradient-free -- few-shot learning to understand.")
