"""The organism builds its OWN embeddings from the conversational corpus (PPMI-SVD over co-occurrence),
then we test whether they comprehend situations as well as nomic. No pretrained perception."""
import os, re, json, urllib.request, collections, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
DIM, WIN, K = 96, 5, 2500
def out(s): print(s,flush=True)

def build_self_embeddings():
    text=open(os.path.join(HERE,"chat_corpus.txt"),encoding="utf-8",errors="ignore").read().lower()
    toks=re.findall(r"[a-z']+",text); toks=[w for w in toks if len(w)>=2]
    cnt=collections.Counter(toks); vocab=[w for w,_ in cnt.most_common(K)]
    idx={w:i for i,w in enumerate(vocab)}; V=len(vocab)
    co=np.zeros((V,V)); win=[]
    for w in toks:
        if w in idx:
            wi=idx[w]
            for v in win[-WIN:]: co[wi,v]+=1; co[v,wi]+=1
            win.append(wi)
            if len(win)>WIN: win.pop(0)
    tot=co.sum(); rk=co.sum(1,keepdims=True)+1e-9
    ppmi=np.maximum(np.log((co*tot)/(rk@rk.T)+1e-9),0)
    U,S,_=np.linalg.svd(ppmi); emb=U[:,:DIM]*np.sqrt(S[:DIM])
    emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-9)
    return vocab, idx, emb, len(toks)

def post(ep,p):
    r=urllib.request.Request(f"http://localhost:11434/api/{ep}",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=120).read())
def gen(p): return post("generate",{"model":"llama3.2:3b","prompt":p,"stream":False,"options":{"temperature":0.8,"num_predict":90}})["response"].strip()
def nomic(t):
    v=np.array(post("embed",{"model":"nomic-embed-text","input":t})["embeddings"][0]); return v/(np.linalg.norm(v)+1e-9)

SEED={"greeting":"hello","thanks":"thank you","farewell":"goodbye","hungry":"i am hungry",
      "ask_help":"can you help me","tired":"i am tired","happy":"i am happy","sad":"i feel sad",
      "lost":"i am lost","cold":"i am cold","bored":"i am bored","thirsty":"i need water"}

if __name__=="__main__":
    out("building the organism's OWN embeddings (PPMI-SVD over the chat corpus)...")
    vocab,idx,emb,ntok=build_self_embeddings()
    out(f"  corpus {ntok} tokens, vocab {len(vocab)}, dim {DIM}")
    # sanity: do self-embeddings separate meaning?
    def se(w): return emb[idx[w]] if w in idx else None
    for a,b,c in [("hungry","food","goodbye"),("tired","sleep","hello")]:
        if se(a) is not None and se(b) is not None and se(c) is not None:
            out(f"  self-emb cos({a},{b})={float(se(a)@se(b)):+.2f}  cos({a},{c})={float(se(a)@se(c)):+.2f}")
    def selfembed(msg):
        ws=[idx[w] for w in re.findall(r"[a-z']+",msg.lower()) if w in idx]
        if not ws: return None
        v=emb[ws].mean(0); return v/(np.linalg.norm(v)+1e-9)
    # comprehension test: self-emb vs nomic on held-out paraphrases
    names=list(SEED); para={}
    out("generating paraphrases...")
    for s in names:
        txt=gen(f'Give 8 short everyday ways to say "{SEED[s]}". one per line, lowercase.')
        para[s]=[SEED[s]]+[l.strip(" -.\t") for l in txt.splitlines() if l.strip() and len(l)<60][:8]
    rng=np.random.default_rng(0)
    for tag,embfn in [("SELF (organism-built)",selfembed),("nomic (external, for reference)",nomic)]:
        accs=[]
        for trial in range(6):
            cent={}; held=[]
            for s in names:
                es=[(embfn(p),p) for p in para[s]]; es=[(e,p) for e,p in es if e is not None]
                rng.shuffle(es); k=max(2,int(0.6*len(es)))
                if k>=len(es): continue
                cent[s]=np.mean([e for e,_ in es[:k]],0)
                for e,p in es[k:]: held.append((s,e))
            cs=np.stack([cent[s] for s in names if s in cent]); cn=[s for s in names if s in cent]
            accs.append(np.mean([cn[int(np.argmax(cs@e))]==s for s,e in held]))
        out(f"  {tag}: held-out situation comprehension {np.mean(accs):.2f} (chance {1/len(names):.2f})")
    out("READING: if SELF ~ nomic, the organism's own exposure-built embeddings perceive conversation")
    out("well enough -- no pretrained perception needed; self-contained.")
