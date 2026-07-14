"""radial_evo.py — evolution takes the lead on CIFAR (autonomous campaign).

Phase A — genome #2, the REGION genome: search ON the map, not over indices.
  Q4 showed index-space fitness is flat (random 24-subsets differ by ±0.002)
  while the map's geometry finds 0.4015 for free. So the genome's parameters
  are now map-space coordinates: a plane orientation (theta, phi), an offset
  along that direction, and a slice size. Nearby genomes select overlapping
  lens sets — the landscape is smooth by construction. Honest control: pure
  random-genome search with the SAME evaluation budget.

Phase B — INTERACTION feature genomes: past the pointwise ceiling.
  The pointwise bank cannot see pixel interactions (CIFAR gain 0.0). The
  environment provides label-free patch statistics (PCA basis of 6x6x3
  patches, built once from train images — "the features are the environment");
  a feature genome picks two basis components, two lens programs, and a
  combine op, producing ONE pooled scalar feature per image:
      f(img) = mean_over_positions  act_a(c_i(pos)) * act_b(c_j(pos))
  Evolution breeds a population of such genomes; fitness is soft residual
  gain on a validation split; the best are FROZEN and composed (component-
  first discipline) and the head stays closed-form ridge.

Phase C — THE GENOME MAP: fingerprint every evolved genome by its behavior
  (feature values over a fixed probe set), MDS to 3D — do good genomes
  cluster? Is the map usable as diversity pressure?

Everything logged to radial_data/evo_*.json; no gradients anywhere.
"""
import json
import os
import time

import numpy as np

import radial_map as rm
from radial_baseline import cifar_data, _tprims, _lens_apply_t, _kridge_factory
from radial_slice_ga import _cache_grams, N_FIT, N_VAL

_HERE = os.path.dirname(os.path.abspath(__file__))
L = 300


def _setup(torch):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    Xtr, ytr, Xte, yte = cifar_data()
    Ffit = torch.tensor(Xtr[:N_FIT].reshape(N_FIT, -1), device=dev)
    Fval = torch.tensor(Xtr[N_FIT:N_FIT + N_VAL].reshape(N_VAL, -1), device=dev)
    grams, kept = _cache_grams(torch, tp, Ffit, Fval)
    # NOTE: grams stay fp32 — z-scored gram entries reach ~1.5e5, past fp16 max
    torch.cuda.empty_cache()
    yfit = torch.tensor(ytr[:N_FIT], device=dev)
    yval = torch.tensor(ytr[N_FIT:N_FIT + N_VAL], device=dev)
    Y = -torch.ones((N_FIT, 10), device=dev)
    Y[torch.arange(N_FIT), yfit] = 1.0
    eye = torch.eye(N_FIT, device=dev)
    return dev, tp, (Xtr, ytr, Xte, yte), grams, kept, yval, Y, eye


def _batch_fitness(torch, grams, sels, Y, eye, yval, dev, chunk=8):
    """Batched kernel-ridge fitness, evaluated in small chunks so the batch
    tensors stay ~128MB and never fight the gram cache for VRAM."""
    softs, accs = [], []
    n = Y.shape[0]
    for c in range(0, len(sels), chunk):
        part = sels[c:c + chunk]
        B = len(part)
        Ktr = torch.zeros((B, n, n), device=dev)
        Kva = torch.zeros((B, grams[0][1].shape[0], n), device=dev)
        for b, sel in enumerate(part):
            for g in sel:
                Ktr[b] += grams[g][0]
                Kva[b] += grams[g][1]
        ks = torch.tensor([float(max(len(s), 1)) for s in part], device=dev)
        Ktr += (100.0 * ks.view(B, 1, 1)) * eye
        a = torch.linalg.solve(Ktr, Y.expand(B, -1, -1))
        scores = Kva @ a
        nv = scores.shape[1]
        softs += [float(s) for s in
                  torch.log_softmax(scores, 2)[:, torch.arange(nv), yval].mean(1)]
        accs += [float(x) for x in (scores.argmax(2) == yval).float().mean(1)]
        del Ktr, Kva, a, scores
    return softs, accs


def _final_test(torch, tp, data, lens_ids, dev):
    """Honest final: full 8k train, real 2k test, chosen lenses only."""
    Xtr, ytr, Xte, yte = data
    solve = _kridge_factory(ytr, yte, 10)
    Ftr = torch.tensor(Xtr.reshape(len(Xtr), -1), device=dev)
    Fte = torch.tensor(Xte.reshape(len(Xte), -1), device=dev)
    Ktr = torch.zeros((len(Xtr), len(Xtr)), device=dev)
    Kte = torch.zeros((len(Xte), len(Xtr)), device=dev)
    for i in lens_ids:
        prog = rm.lens_program(int(i))
        Z = _lens_apply_t(tp, prog, Ftr)
        mu, sd = Z.mean(0), Z.std(0)
        ok = sd > 1e-6
        Z = torch.where(ok, (Z - mu) / (sd + 1e-9), torch.zeros_like(Z))
        Zt = _lens_apply_t(tp, prog, Fte)
        Zt = torch.where(ok, (Zt - mu) / (sd + 1e-9), torch.zeros_like(Zt))
        Ktr += Z @ Z.T
        Kte += Zt @ Z.T
    return solve(Ktr, Kte, max(len(lens_ids), 1))["acc"]


# ---------------------------------------------------------------------------
# Phase A — the region genome
# ---------------------------------------------------------------------------

def _region_select(X3, genome):
    """Genome -> lens ids: (theta, phi) orient a direction u; project the map
    onto u; take the K lenses nearest the offset-quantile point along u."""
    th, ph, off, logk = genome
    u = np.array([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)])
    proj = X3 @ u
    d = np.quantile(proj, 1.0 / (1.0 + np.exp(-off)))       # off unbounded -> (0,1)
    K = int(np.clip(round(np.exp(logk)), 8, 48))
    return list(np.argsort(np.abs(proj - d))[:K])


def phase_a(pop_size=32, gens=60, seed=0, verbose=True):
    import torch
    rng = np.random.default_rng(seed)
    dev, tp, data, grams, kept, yval, Y, eye = _setup(torch)

    Sraw, _, _, _ = rm.build_signatures(max(kept) + 1, "cifar")
    S = Sraw[kept]
    S = (S - S.mean(0)) / (S.std(0) + 1e-9)
    X3 = rm._mds(S, 3)

    def new_genome():
        return np.array([rng.uniform(0, 2 * np.pi), rng.uniform(-1.4, 1.4),
                         rng.normal(0, 1.5), np.log(24.0) + rng.normal(0, 0.4)])

    pop = [new_genome() for _ in range(pop_size)]
    scales = np.full(pop_size, 0.3)
    sels = [_region_select(X3, g) for g in pop]
    fits, accs = _batch_fitness(torch, grams, sels, Y, eye, yval, dev)
    fits, accs = np.array(fits), np.array(accs)
    energy = np.ones(pop_size)
    hist, best_seen = [], (-1e9, None, 0.0)
    n_evals = pop_size
    t0 = time.time()
    for gen in range(gens):
        med = np.median(fits)
        energy = np.clip(energy * 0.9 + 1.5 * (fits - med), 0.0, 1.5)
        starved = energy < 0.2
        order = np.argsort(fits)[::-1]
        if fits[order[0]] > best_seen[0]:
            best_seen = (fits[order[0]], pop[order[0]].copy(), accs[order[0]])
        keep = list(order[:4])
        alive = [i for i in range(pop_size) if not starved[i]] or list(range(pop_size))
        children, cscales = [], []
        while len(children) < pop_size - 4:
            cand = rng.choice(alive, 3)
            pi = cand[np.argmax(fits[cand])]
            sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.05, 1.0))
            child = pop[pi] + rng.normal(0, sc, 4) * np.array([1.0, 0.6, 0.8, 0.25])
            children.append(child)
            cscales.append(sc)
        csels = [_region_select(X3, g) for g in children]
        cf, ca = _batch_fitness(torch, grams, csels, Y, eye, yval, dev)
        n_evals += len(children)
        pop = [pop[i] for i in keep] + children
        scales = np.concatenate([scales[keep], cscales])
        fits = np.concatenate([fits[keep], cf])
        accs = np.concatenate([accs[keep], ca])
        energy = np.concatenate([energy[keep], np.ones(len(children))])
        hist.append({"gen": gen, "best_soft": round(float(fits.max()), 4),
                     "best_val_acc": round(float(accs[np.argmax(fits)]), 4),
                     "starved": int(starved.sum()),
                     "mean_scale": round(float(scales.mean()), 3)})
        if verbose and gen % 10 == 0:
            h = hist[-1]
            print(f"  [A] gen {gen:3d}  soft {h['best_soft']}  val {h['best_val_acc']}  "
                  f"starved {h['starved']}  scale {h['mean_scale']}", flush=True)

    # honest control: random genomes, same evaluation budget, scored on the
    # same validation split (final 8k test is run for the evolved winner only)
    rbest, rbest_g, rbest_acc = -1e9, None, 0.0
    done = 0
    while done < n_evals:
        chunk = [new_genome() for _ in range(min(pop_size, n_evals - done))]
        rf, ra = _batch_fitness(torch, grams, [_region_select(X3, g) for g in chunk],
                                Y, eye, yval, dev)
        j = int(np.argmax(rf))
        if rf[j] > rbest:
            rbest, rbest_g, rbest_acc = rf[j], chunk[j], ra[j]
        done += len(chunk)

    best_g = pop[int(np.argmax(fits))] if fits.max() >= best_seen[0] else best_seen[1]
    sel = _region_select(X3, best_g)
    test_acc = _final_test(torch, tp, data, [kept[i] for i in sel], dev)
    out = {"phase": "A-region-genome", "domain": "cifar",
           "evolved": {"genome": [round(float(v), 4) for v in best_g],
                       "K": len(sel), "test_acc": test_acc,
                       "val_acc": round(float(accs[np.argmax(fits)]), 4),
                       "val_soft": round(float(fits.max()), 4)},
           "random_search_same_budget": {"K": len(_region_select(X3, rbest_g)),
                                         "val_acc": round(float(rbest_acc), 4),
                                         "val_soft": round(float(rbest), 4)},
           "references": {"rotation_slice": 0.4015, "bank400": 0.3815,
                          "raw": 0.3820, "random24": 0.3774,
                          "index_ga": 0.376},
           "evals": n_evals * 2, "seconds": round(time.time() - t0),
           "history": hist}
    with open(os.path.join(_HERE, "radial_data", "evo_region_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[A] evolved region: K={len(sel)} TEST {test_acc} (val {out['evolved']['val_acc']}) | "
              f"random-search control val {out['random_search_same_budget']['val_acc']} | "
              f"rotation ref 0.4015", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase B — interaction feature genomes (past the pointwise ceiling)
# ---------------------------------------------------------------------------

_OPS = ["mult", "absdiff", "min"]
_POOLS = ["mean", "max"]
_PRIMS_B = ["id", "abs", "relu", "tanh", "gauss", "sq", "soft", "sin"]


def _patch_env(torch, dev, C=40, ps=6, stride=2, seed=0):
    """The environment: label-free patch-PCA component maps per image.
    Built from data statistics once; evolution never touches it."""
    import torch.nn.functional as Fn
    Xtr, ytr, Xte, yte = cifar_data()
    itr = torch.tensor(Xtr, device=dev).permute(0, 3, 1, 2).contiguous()
    ite = torch.tensor(Xte, device=dev).permute(0, 3, 1, 2).contiguous()
    P = Fn.unfold(itr[:2000], ps, stride=stride)          # (n, 108, L)
    cols = P.permute(0, 2, 1).reshape(-1, ps * ps * 3)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cols = cols[torch.randperm(len(cols), generator=g)[:120000].to(dev)]
    mu = cols.mean(0)
    _, _, V = torch.linalg.svd(cols - mu, full_matrices=False)
    comps = V[:C]                                          # (C, 108)

    def maps(imgs, bs=500):
        out = []
        for b in range(0, len(imgs), bs):
            U = Fn.unfold(imgs[b:b + bs], ps, stride=stride)   # (b,108,L)
            M = torch.einsum("cd,bdl->bcl", comps, U - mu.view(1, -1, 1))
            out.append(M)
        M = torch.cat(out)
        return M
    Mtr = maps(itr)
    sd = Mtr.std((0, 2), keepdim=True) + 1e-6
    Mtr = Mtr / sd
    Mte = maps(ite) / sd
    return Mtr, Mte, (Xtr, ytr, Xte, yte), C


def _feat(torch, M, g):
    """One genome -> one scalar feature per image. M: (N, C, L)."""
    tp = _tprims(torch)
    za = tp[_PRIMS_B[g["pa"]]](g["a1"] * M[:, g["i"], :] + g["b1"])
    zb = tp[_PRIMS_B[g["pb"]]](g["a2"] * M[:, g["j"], :] + g["b2"])
    op = _OPS[g["op"]]
    z = za * zb if op == "mult" else (torch.abs(za - zb) if op == "absdiff"
                                      else torch.minimum(za, zb))
    return z.mean(1) if _POOLS[g["pool"]] == "mean" else z.amax(1)


def _new_gen_b(rng, C):
    return {"i": int(rng.integers(C)), "j": int(rng.integers(C)),
            "pa": int(rng.integers(len(_PRIMS_B))), "pb": int(rng.integers(len(_PRIMS_B))),
            "a1": float(rng.uniform(0.5, 2.5)), "b1": float(rng.uniform(-1, 1)),
            "a2": float(rng.uniform(0.5, 2.5)), "b2": float(rng.uniform(-1, 1)),
            "op": int(rng.integers(len(_OPS))), "pool": int(rng.integers(len(_POOLS)))}


def _mut_b(rng, g, C, sc):
    c = dict(g)
    for k in ("i", "j"):
        if rng.random() < 0.15:
            c[k] = int(rng.integers(C))
    for k in ("pa", "pb"):
        if rng.random() < 0.15:
            c[k] = int(rng.integers(len(_PRIMS_B)))
    for k in ("a1", "a2"):
        c[k] = float(np.clip(c[k] + rng.normal(0, sc), 0.1, 4.0))
    for k in ("b1", "b2"):
        c[k] = float(np.clip(c[k] + rng.normal(0, sc), -2.0, 2.0))
    if rng.random() < 0.1:
        c["op"] = int(rng.integers(len(_OPS)))
    if rng.random() < 0.1:
        c["pool"] = int(rng.integers(len(_POOLS)))
    return c


def _ridge_soft(torch, Xf, Xv, Yf, yval, lam=3.0):
    n, d = Xf.shape
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    A = torch.hstack([(Xf - mu) / sd, torch.ones(n, 1, device=Xf.device)])
    B = torch.hstack([(Xv - mu) / sd, torch.ones(len(Xv), 1, device=Xf.device)])
    W = torch.linalg.solve(A.T @ A + lam * torch.eye(d + 1, device=Xf.device), A.T @ Yf)
    s = B @ W
    soft = float(torch.log_softmax(s, 1)[torch.arange(len(yval)), yval].mean())
    acc = float((s.argmax(1) == yval).float().mean())
    return soft, acc


def phase_b(rounds=14, pop_size=64, gens=12, freeze_top=8, seed=0, verbose=True):
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    Mtr, Mte, data, C = _patch_env(torch, dev)
    Xtr, ytr, Xte, yte = data
    n_fit = 6000
    yf = torch.tensor(ytr[:n_fit], device=dev)
    yv = torch.tensor(ytr[n_fit:], device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), yf] = 1.0

    frozen, fcols_tr = [], []
    base_soft = None
    hist = []
    for rnd in range(rounds):
        Ffr = (torch.stack(fcols_tr, 1) if fcols_tr
               else torch.zeros((len(Xtr), 0), device=dev))
        s0, a0 = _ridge_soft(torch, Ffr[:n_fit], Ffr[n_fit:], Yf, yv) \
            if fcols_tr else (-np.log(10.0), 0.1)
        pop = [_new_gen_b(rng, C) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)

        def fit_pop(genomes):
            softs, accs, cols = [], [], []
            for g in genomes:
                col = _feat(torch, Mtr, g)
                sd = col.std()
                if float(sd) < 1e-6:
                    softs.append(-1e9); accs.append(0.0); cols.append(col)
                    continue
                X = torch.cat([Ffr, col.view(-1, 1)], 1)
                s, a = _ridge_soft(torch, X[:n_fit], X[n_fit:], Yf, yv)
                softs.append(s - s0); accs.append(a); cols.append(col)
            return np.array(softs), np.array(accs), cols

        fits, accs, cols = fit_pop(pop)
        energy = np.ones(pop_size)
        for gen in range(gens):
            med = np.median(fits)
            energy = np.clip(energy * 0.9 + 1.5 * (fits - med) * 50, 0.0, 1.5)
            starved = energy < 0.2
            order = np.argsort(fits)[::-1]
            keep = list(order[:6])
            alive = [i for i in range(pop_size) if not starved[i]] or list(range(pop_size))
            kids, ksc = [], []
            while len(kids) < pop_size - 6:
                cand = rng.choice(alive, 3)
                pi = cand[np.argmax(fits[cand])]
                sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                kids.append(_mut_b(rng, pop[pi], C, sc)); ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            pop = [pop[i] for i in keep] + kids
            scales = np.concatenate([scales[keep], ksc])
            fits = np.concatenate([fits[keep], kf])
            accs = np.concatenate([accs[keep], ka])
            cols = [cols[i] for i in keep] + kc
            energy = np.concatenate([energy[keep], np.ones(len(kids))])

        # freeze the top genomes, decorrelated against each other + the frozen set
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= 0.0005 or added >= freeze_top:
                break
            col = cols[idx]
            colz = (col - col.mean()) / (col.std() + 1e-9)
            dup = False
            for fc in fcols_tr[-60:]:
                fz = (fc - fc.mean()) / (fc.std() + 1e-9)
                if float(torch.abs((colz * fz).mean())) > 0.95:
                    dup = True
                    break
            if not dup:
                frozen.append(pop[idx]); fcols_tr.append(col); added += 1
        Ffr = torch.stack(fcols_tr, 1) if fcols_tr else torch.zeros((len(Xtr), 0), device=dev)
        s1, a1 = _ridge_soft(torch, Ffr[:n_fit], Ffr[n_fit:], Yf, yv)
        hist.append({"round": rnd, "added": added, "n_frozen": len(frozen),
                     "val_soft": round(s1, 4), "val_acc": round(a1, 4),
                     "starved_last_gen": int(starved.sum())})
        if verbose:
            print(f"  [B] round {rnd:2d}  +{added} frozen (total {len(frozen)})  "
                  f"val acc {a1:.4f}  soft {s1:.4f}  ({round(time.time()-t0)}s)", flush=True)
        if added == 0:
            break

    # final honest eval: features on full train + real test, one measurement
    Fte = torch.stack([_feat(torch, Mte, g) for g in frozen], 1)
    Ftr = torch.stack(fcols_tr, 1)
    ytr_t = torch.tensor(ytr, device=dev)
    Yfull = -torch.ones((len(ytr), 10), device=dev)
    Yfull[torch.arange(len(ytr)), ytr_t] = 1.0
    yte_t = torch.tensor(yte, device=dev)
    _, test_acc = _ridge_soft(torch, Ftr, Fte, Yfull, yte_t)
    out = {"phase": "B-interaction-genomes", "domain": "cifar",
           "n_frozen": len(frozen), "test_acc": round(test_acc, 4),
           "genomes": frozen, "history": hist,
           "references": {"pointwise_ceiling_honest": 0.3845, "bank400": 0.3815,
                          "raw": 0.3820, "coates_ng_handcrafted_8k": 0.493},
           "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "radial_data", "evo_interact_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[B] {len(frozen)} evolved features -> TEST {test_acc:.4f}  "
              f"(pointwise ceiling 0.3845, hand-crafted patch bar 0.493)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase C — the genome map
# ---------------------------------------------------------------------------

def phase_c(verbose=True):
    """Map the evolved genomes the same way lenses were mapped: fingerprint
    each by its BEHAVIOR (feature values over a fixed probe set), MDS to 3D.
    Questions: does the genome population have real structure? does evolution
    expand OUTWARD over discovery time (the activation-galaxy claim)? what is
    the effective dimensionality of the evolved feature set?"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(_HERE, "radial_data", "evo_interact_cifar.json")) as f:
        run = json.load(f)
    genomes = run["genomes"]
    Mtr, Mte, data, C = _patch_env(torch, dev)
    probe = Mtr[:512]
    sigs = []
    for g in genomes:
        v = _feat(torch, probe, g)
        v = (v - v.mean()) / (v.std() + 1e-9)
        sigs.append(v.cpu().numpy())
    S = np.array(sigs)
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
    X3 = rm._mds(Sz, 3)
    X3 = X3 - X3.mean(0)
    r = np.linalg.norm(X3, axis=1)
    order = np.arange(len(genomes))                     # freeze order = discovery time
    expand = rm._safe_corr(order.astype(float), r)      # does radius grow with time?
    sv = np.linalg.svd(Sz - Sz.mean(0), compute_uv=False)
    eff = int((sv > 0.01 * sv[0]).sum())
    # family structure: mean intra-op vs inter-op signature distance
    ops = np.array([g["op"] for g in genomes])
    D = np.linalg.norm(Sz[:, None, :100] - Sz[None, :, :100], axis=2)
    intra = float(np.mean([D[np.ix_(ops == o, ops == o)].mean()
                           for o in set(ops)]))
    inter = float(D[np.ix_(ops == 0, ops != 0)].mean()) if (ops == 0).any() else 0.0
    out = {"phase": "C-genome-map", "n_genomes": len(genomes),
           "effective_dim": eff,
           "expansion_corr_time_vs_radius": round(float(expand), 3),
           "axis_std": [round(float(v), 2) for v in X3.std(0)],
           "op_intra_dist": round(intra, 2), "op_inter_dist": round(inter, 2),
           "pts": [{"i": int(i), "x": round(float(X3[i, 0]), 3),
                    "y": round(float(X3[i, 1]), 3), "z": round(float(X3[i, 2]), 3),
                    "op": _OPS[genomes[i]["op"]], "pool": _POOLS[genomes[i]["pool"]],
                    "round": int(i // 8)} for i in range(len(genomes))]}
    with open(os.path.join(_HERE, "radial_data", "evo_genome_map.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[C] genome map: {len(genomes)} genomes, effective dim {eff}, "
              f"shape {out['axis_std']}", flush=True)
        print(f"    discovery expands outward? corr(time, radius) = {expand:+.3f}", flush=True)
        print(f"    op families: intra-dist {intra:.2f} vs inter {inter:.2f}", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "c" in sys.argv:
        phase_c()
    elif "b" in sys.argv:
        phase_b()
    elif "a" in sys.argv or len(sys.argv) == 1:
        phase_a()
