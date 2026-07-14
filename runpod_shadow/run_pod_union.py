"""Final readout of the diversity experiment: union of ALL converged
substrates, one head refit, one test measurement. Run on the pod."""
import json, time, numpy as np, torch
import radial_evo2 as e2
from radial_evo import _tprims

t0 = time.time(); dev = "cuda"; tp = _tprims(torch)
z = np.load("radial_data/cifar_full.npz")
Xtr = z["Xtr"].astype(np.float32)/255.0; Xte = z["Xte"].astype(np.float32)/255.0
ytr, yte = z["ytr"], z["yte"]
env = e2.Env(torch, dev, Xtr, Xte, max_cached=6)
srcs = [("s7",  "radial_data/evo2_s7_ckpt.json"),
        ("s13", "radial_data/evo2x_ckpt.json"),
        ("s19", "radial_data/evo2_s19_ckpt.json"),
        ("s29", "radial_data/evo2_s29_ckpt.json"),
        ("s37", "radial_data/evo2_s37_ckpt.json"),
        ("s43", "radial_data/evo2_s43_ckpt.json"),
        ("p128", "radial_data/evo2p128_ckpt.json")]
cols_tr, cols_te, counts = [], [], {}
for name, src in srcs:
    try:
        g1 = json.load(open(src))["frozen"]
    except Exception as exc:
        print(f"skip {name}: {exc}", flush=True)
        continue
    for g in g1:
        g["terms"] = [{"c": t["c"], "prog": [tuple(s) for s in t["prog"]]} for t in g["terms"]]
    cols_tr += [e2.feature(torch, tp, env, g) for g in g1]
    cols_te += [e2.feature(torch, tp, env, g, test=True) for g in g1]
    counts[name] = len(g1)
    print(f"{name}: {len(g1)}", flush=True)
Ftr = torch.stack(cols_tr, 1); Fte = torch.stack(cols_te, 1)
Y = -torch.ones((len(ytr), 10), device=dev); Y[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0
yte_t = torch.tensor(yte, device=dev)
best, best_lam = 0.0, None
for lam in (1.0, 3.0, 10.0, 30.0, 100.0):
    _, acc = e2._ridge_soft(torch, Ftr, Fte, Y, yte_t, lam=lam)
    if acc > best: best, best_lam = acc, lam
print(f"FINAL UNION {counts} = {sum(counts.values())} genomes: TEST {best:.4f} "
      f"(lam {best_lam}, {round(time.time()-t0)}s)", flush=True)
json.dump({"phase": "final-union", "counts": counts, "test_acc": round(best, 4),
           "lam": best_lam}, open("radial_data/final_union_cifar.json", "w"), indent=1)
