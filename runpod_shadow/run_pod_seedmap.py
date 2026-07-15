"""Seed map v2: per-genome 3D coords + structure + activation profile."""
import json, time, numpy as np, torch
import radial_evo2 as e2
from radial_evo import _tprims

t0 = time.time(); dev = "cuda"; tp = _tprims(torch)
z = np.load("radial_data/cifar_full.npz")
Xtr = z["Xtr"].astype(np.float32)/255.0
env = e2.Env(torch, dev, Xtr[:2000], Xtr[:8], max_cached=6)
SRCS = [("s7", "radial_data/evo2_s7_ckpt.json"), ("s13", "radial_data/evo2x_ckpt.json"),
        ("s19", "radial_data/evo2_s19_ckpt.json"), ("s29", "radial_data/evo2_s29_ckpt.json"),
        ("s37", "radial_data/evo2_s37_ckpt.json"), ("s43", "radial_data/evo2_s43_ckpt.json")]
PRIMS = e2._PRIMS; OPS = e2._OPS; STATS = e2._STATS
sigs, meta = [], []
for name, src in SRCS:
    gs = json.load(open(src))["frozen"]
    for k, g in enumerate(gs):
        g["terms"] = [{"c": t["c"], "prog": [tuple(s) for s in t["prog"]]}
                      for t in g["terms"]]
        v = e2.feature(torch, tp, env, g)[:512]
        v = (v - v.mean()) / (v.std() + 1e-9)
        sigs.append(v.cpu().numpy())
        prog = " ".join(
            "+".join(PRIMS[st[0]] for st in t["prog"]) + f"(c{t['c']})"
            for t in g["terms"])
        meta.append({"s": name, "i": k, "ps": g["ps"], "op": OPS[g["op"]],
                     "st": STATS[g["stat"]],
                     "w": f"({g['cx']:.2f},{g['cy']:.2f},sig {np.exp(g['lsig']):.2f})",
                     "prog": prog,
                     "act": [round(float(x), 2) for x in sigs[-1][:48]]})
    print(name, len(gs), flush=True)
S = np.array(sigs)
Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
n = len(Sz)
D2 = np.maximum(0.0, (Sz**2).sum(1)[:,None] + (Sz**2).sum(1)[None,:] - 2*Sz@Sz.T)
J = np.eye(n) - 1.0/n
w, V = np.linalg.eigh(-0.5 * J @ D2 @ J)
X3 = V[:, -3:][:, ::-1] * np.sqrt(np.maximum(w[-3:][::-1], 0))
X3 = X3 - X3.mean(0)
pts = []
for i in range(n):
    m = meta[i]
    m.update({"x": round(float(X3[i,0]),2), "y": round(float(X3[i,1]),2),
              "z": round(float(X3[i,2]),2)})
    pts.append(m)
out = {"n": n, "axis_std": [round(float(v),2) for v in X3.std(0)], "pts": pts}
json.dump(out, open("radial_data/seed_map.json", "w"))
print(f"seed map v2: {n} genomes, {round(time.time()-t0)}s", flush=True)
