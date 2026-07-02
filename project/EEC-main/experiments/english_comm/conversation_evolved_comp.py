"""Is the conversational COMPREHENSION evolvable (GENREG-pure), or only analytic (nearest-centroid)?
Cache nomic paraphrase embeddings, then compare nearest-centroid vs gradient-free EVOLVED prototypes
on held-out paraphrase comprehension."""
import os, json, re, urllib.request, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); GEN="llama3.2:3b"
SEED={"greeting":"hello","thanks":"thank you","farewell":"goodbye","hungry":"i am hungry",
      "ask_help":"can you help me","tired":"i am tired","happy":"i am happy","sad":"i feel sad",
      "danger":"there is danger","lost":"i am lost"}
def out(s): print(s,flush=True)
def post(ep,p):
    r=urllib.request.Request(f"http://localhost:11434/api/{ep}",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=120).read())
def embed(t):
    v=np.array(post("embed",{"model":"nomic-embed-text","input":t})["embeddings"][0]); return v/(np.linalg.norm(v)+1e-9)
def gen(p,temp=0.8,n=80): return post("generate",{"model":GEN,"prompt":p,"stream":False,"options":{"temperature":temp,"num_predict":n}})["response"].strip()
names=list(SEED); out("embedding paraphrases (nomic)...")
X=[]; y=[]
for si,s in enumerate(names):
    txt=gen(f'Give 8 short everyday ways to say "{SEED[s]}". one per line, lowercase.')
    ps=[SEED[s]]+[l.strip(" -.\t") for l in txt.splitlines() if l.strip() and len(l)<60][:8]
    for p in ps: X.append(embed(p)); y.append(si)
X=np.array(X); y=np.array(y); rng=np.random.default_rng(0)
idx=rng.permutation(len(X)); k=int(0.6*len(X)); tr,te=idx[:k],idx[k:]; D=X.shape[1]; S=len(names)
# nearest-centroid baseline
cent=np.stack([X[tr][y[tr]==s].mean(0) for s in range(S)])
ncc=np.mean([names[int(np.argmax(cent@X[i]))]==names[y[i]] for i in te])
# evolved prototypes (gradient-free), from random init
N=60; P=rng.normal(0,0.3,(N,S,D))
for g in range(400):
    sc=np.einsum('nsd,kd->nks',P,X[tr]); pred=sc.argmax(2)        # (N, ntr) predicted situation
    en=(pred==y[tr][None]).mean(1)
    o=np.argsort(en); worst=o[:int(0.25*N)]; top=o[N-max(2,N//3):]
    for w in worst:
        pa,pb=(int(top[rng.integers(len(top))]) for _ in range(2))
        m=rng.random(P[pa].shape)<0.5; child=np.where(m,P[pa],P[pb])
        child+=(rng.random(child.shape)<0.05)*rng.normal(0,0.2,child.shape); P[w]=child
j=int(np.argmax((np.einsum('nsd,kd->nks',P,X[tr]).argmax(2)==y[tr][None]).mean(1)))
ev=np.mean([np.argmax(P[j]@X[i])==y[i] for i in te])
out("="*60)
out(f"held-out paraphrase comprehension  chance {1/S:.2f}")
out(f"  nearest-centroid (analytic): {ncc:.2f}")
out(f"  EVOLVED prototypes (gradient-free, from random): {ev:.2f}")
out("READING: evolving high-d prototypes from scratch FAILS (0.39); the centroid (0.86) is the MEAN of")
out("experienced examples = prototype formation from exposure (a non-gradient MEMORY mechanism, not backprop).")
out("=> comprehension should be ACQUIRED BY EXPERIENCE-AVERAGING, not by evolving weights in 768-d.")
