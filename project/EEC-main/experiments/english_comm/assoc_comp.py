"""Self-contained comprehension WITHOUT embeddings: the organism learns which WORDS signal which
situation, from exposure (PMI of word|situation). Its own representation of meaning, no pretrained
perception, no huge corpus, no Ollama at inference -- just learned word->situation associations."""
import os, re, json, urllib.request, collections, math, numpy as np
GEN="llama3.2:3b"
SEED={"greeting":"hello","thanks":"thank you","farewell":"goodbye","hungry":"i am hungry",
      "ask_help":"can you help me","tired":"i am tired","happy":"i am happy","sad":"i feel sad",
      "lost":"i am lost","cold":"i am cold","bored":"i am bored","thirsty":"i need water"}
STOP=set("i am a the is are you to it my me of and so we be do that this can could will would have has".split())
def post(ep,p):
    r=urllib.request.Request(f"http://localhost:11434/api/{ep}",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=120).read())
def gen(p): return post("generate",{"model":GEN,"prompt":p,"stream":False,"options":{"temperature":0.85,"num_predict":90}})["response"].strip()
def words(t): return [w for w in re.findall(r"[a-z']+",t.lower()) if w not in STOP and len(w)>=2]
names=list(SEED); para={}
print("generating paraphrases...",flush=True)
for s in names:
    txt=gen(f'Give 12 short everyday ways to say "{SEED[s]}". one per line, lowercase.')
    para[s]=[SEED[s]]+[l.strip(" -.\t") for l in txt.splitlines() if l.strip() and len(l)<60][:12]
rng=np.random.default_rng(0); accs=[]
for trial in range(6):
    train={s:[] for s in names}; held=[]
    for s in names:
        ps=para[s][:]; rng.shuffle(ps); k=max(2,int(0.6*len(ps)))
        train[s]=ps[:k]; held+=[(s,p) for p in ps[k:]]
    # learn word->situation PMI from exposure
    wc={s:collections.Counter(w for p in train[s] for w in words(p)) for s in names}
    tot=collections.Counter(); 
    for s in names: tot+=wc[s]
    def score(msg):
        ws=words(msg); sc={}
        for s in names:
            n=sum(wc[s].values())+1
            sc[s]=sum(math.log((wc[s][w]+0.1)/n*(sum(tot.values()))/(tot[w]+0.1)) for w in ws) if ws else 0
        return max(sc,key=sc.get)
    accs.append(np.mean([score(p)==s for s,p in held]))
print("="*60)
print(f"SELF associative comprehension (word->situation PMI from exposure): {np.mean(accs):.2f} (chance {1/len(names):.2f})")
print("READING: if decent, the organism comprehends via its OWN learned word-situation associations --")
print("self-contained, no pretrained embeddings, no Ollama at inference.")
