"""Seed map: every genome from every converged substrate, fingerprinted by
behavior on 512 probe images, MDS to 3D, tagged by seed. Answers visually:
do independent seeds occupy complementary regions of behavior space?"""
import json, time, numpy as np, torch
import radial_evo2 as e2
from radial_evo import _tprims

t0 = time.time(); dev = "cuda"; tp = _tprims(torch)
z = np.load("radial_data/cifar_full.npz")
Xtr = z["Xtr"].astype(np.float32)/255.0
env = e2.Env(torch, dev, Xtr[:2000], Xtr[:8], max_cached=6)  # probe from train head
SRCS = [("s7",  "radial_data/evo2_s7_ckpt.json"),
        ("s13", "radial_data/evo2x_ckpt.json"),
        ("s19", "radial_data/evo2_s19_ckpt.json"),
        ("s29", "radial_data/evo2_s29_ckpt.json"),
        ("s37", "radial_data/evo2_s37_ckpt.json"),
        ("s43", "radial_data/evo2_s43_ckpt.json")]
sigs, tags = [], []
for name, src in SRCS:
    try:
        gs = json.load(open(src))["frozen"]
    except Exception as exc:
        print("skip", name, exc, flush=True); continue
    for g in gs:
        g["terms"] = [{"c": t["c"], "prog": [tuple(s) for s in t["prog"]]}
                      for t in g["terms"]]
        v = e2.feature(torch, tp, env, g)[:512]
        v = (v - v.mean()) / (v.std() + 1e-9)
        sigs.append(v.cpu().numpy()); tags.append(name)
    print(name, len(gs), flush=True)
S = np.array(sigs)
Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
n = len(Sz)
D2 = np.maximum(0.0, (Sz**2).sum(1)[:,None] + (Sz**2).sum(1)[None,:] - 2*Sz@Sz.T)
J = np.eye(n) - 1.0/n
w, V = np.linalg.eigh(-0.5 * J @ D2 @ J)
X3 = V[:, -3:][:, ::-1] * np.sqrt(np.maximum(w[-3:][::-1], 0))
X3 = X3 - X3.mean(0)
# complementarity: nearest-neighbour same-seed fraction (high = seeds cluster)
from numpy.linalg import norm
nn_same = 0
for i in range(0, n, 7):        # sample every 7th for speed
    d = norm(X3 - X3[i], axis=1); d[i] = 1e9
    nn_same += (tags[int(np.argmin(d))] == tags[i])
frac = nn_same / len(range(0, n, 7))
out = {"n": n, "nn_same_seed_frac": round(float(frac), 3),
       "axis_std": [round(float(v), 2) for v in X3.std(0)],
       "pts": [{"x": round(float(X3[i,0]),3), "y": round(float(X3[i,1]),3),
                "z": round(float(X3[i,2]),3), "s": tags[i]} for i in range(n)]}
json.dump(out, open("radial_data/seed_map.json", "w"))
print(f"seed map: {n} genomes, NN-same-seed {frac:.3f} "
      f"(1/6=0.167 would mean fully mixed), {round(time.time()-t0)}s", flush=True)
