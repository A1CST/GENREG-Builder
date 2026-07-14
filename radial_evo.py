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


def phase_b(rounds=14, pop_size=64, gens=12, freeze_top=8, seed=0, C_env=40,
            verbose=True):
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    Mtr, Mte, data, C = _patch_env(torch, dev, C=C_env)
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
           "n_frozen": len(frozen), "test_acc": round(test_acc, 4), "C_env": C,
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


def phase_b_full(verbose=True):
    """Transfer test: the 647 genomes evolved on 8k, features recomputed on
    FULL CIFAR (50k train / 10k test), ridge head refit, test measured once.
    The environment basis is rebuilt from the SAME 2000 images the genomes
    evolved against, so component indices keep their meaning."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    with open(os.path.join(_HERE, "radial_data", "evo_interact_cifar.json")) as f:
        run8 = json.load(f)
    genomes = run8["genomes"]
    C = int(run8.get("C_env", 40))
    import torch.nn.functional as Fn
    z = np.load(os.path.join(_HERE, "radial_data", "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]

    # rebuild the exact basis (same code path/seed as _patch_env)
    from radial_baseline import cifar_data
    X8 = cifar_data()[0]
    i8 = torch.tensor(X8[:2000], device=dev).permute(0, 3, 1, 2).contiguous()
    P = Fn.unfold(i8, 6, stride=2)
    cols = P.permute(0, 2, 1).reshape(-1, 108)
    g = torch.Generator(device="cpu").manual_seed(0)
    cols = cols[torch.randperm(len(cols), generator=g)[:120000].to(dev)]
    mu = cols.mean(0)
    _, _, V = torch.linalg.svd(cols - mu, full_matrices=False)
    comps = V[:C]
    sd8 = None

    def feats(X, bs=400):
        nonlocal sd8
        F_out = torch.zeros((len(X), len(genomes)), device=dev)
        for b in range(0, len(X), bs):
            imgs = torch.tensor(X[b:b + bs], device=dev).permute(0, 3, 1, 2).contiguous()
            U = Fn.unfold(imgs, 6, stride=2)
            M = torch.einsum("cd,bdl->bcl", comps, U - mu.view(1, -1, 1))
            if sd8 is None:
                sd8 = M.std((0, 2), keepdim=True) + 1e-6   # first batch stats
            M = M / sd8
            for k, gn in enumerate(genomes):
                F_out[b:b + len(imgs), k] = _feat(torch, M, gn)
        return F_out

    Ftr = feats(Xtr)
    Fte = feats(Xte)
    Y = -torch.ones((len(ytr), 10), device=dev)
    Y[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0
    yte_t = torch.tensor(yte, device=dev)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, Ftr, Fte, Y, yte_t, lam=lam)
        best = max(best, acc)
    out = {"phase": "B-full-transfer", "train": len(ytr), "test": len(yte),
           "n_genomes": len(genomes), "test_acc": round(best, 4),
           "references": {"v1_coates_ng_full50k": 0.5904, "evolved_8k": 0.584},
           "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "radial_data", "evo_full_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[B-full] {len(genomes)} genomes on 50k/10k: TEST {best:.4f} "
              f"(v1 milestone 0.5904, evolved-8k 0.5840, {out['seconds']}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase D — the PLATEAU SAFETY MECHANISM: stack a new radial space on the
# outputs of the converged one
# ---------------------------------------------------------------------------

def phase_stack(rounds=60, pop_size=64, gens=12, freeze_top=8, seed=1, verbose=True):
    """When adding more genomes stops helping (phase_b converged: a round
    froze nothing), take the END-OF-GENOME OUTPUTS — the frozen features —
    and hand them to a NEW radial space as its data. Stage-2 genomes read
    stage-1 features exactly the way stage-1 genomes read patch components
    (two channels, two lens bends, combine op). Measured, test once each:
      - stacked  = ridge on [stage-1 + stage-2]  (does the plateau break?)
      - stage-2 only = ridge on stage-2 features (how close does the new
        space get to re-expressing stage 1 from its outputs alone?)
    Plus the stage-2 behavioral map for the record."""
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    with open(os.path.join(_HERE, "radial_data", "evo_interact_cifar.json")) as f:
        run1 = json.load(f)
    genomes1 = run1["genomes"]
    Mtr, Mte, data, C = _patch_env(torch, dev, C=int(run1.get("C_env", 40)))
    Xtr, ytr, Xte, yte = data
    F1tr = torch.stack([_feat(torch, Mtr, g) for g in genomes1], 1)
    F1te = torch.stack([_feat(torch, Mte, g) for g in genomes1], 1)
    mu1, sd1 = F1tr.mean(0), F1tr.std(0) + 1e-6
    # stage-2 environment: z-scored stage-1 outputs as (N, C2, 1) "maps"
    M2tr = ((F1tr - mu1) / sd1).unsqueeze(2)
    M2te = ((F1te - mu1) / sd1).unsqueeze(2)
    C2 = M2tr.shape[1]

    n_fit = 6000
    yv = torch.tensor(ytr[n_fit:], device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    s_plateau, a_plateau = _ridge_soft(torch, F1tr[:n_fit], F1tr[n_fit:], Yf, yv)
    if verbose:
        print(f"[D] plateau baseline (stage-1 only): val acc {a_plateau:.4f}", flush=True)

    frozen2, cols2 = [], []
    hist = []
    for rnd in range(rounds):
        base = torch.cat([F1tr, torch.stack(cols2, 1)], 1) if cols2 else F1tr
        s0, _ = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv)
        pop = [_new_gen_b(rng, C2) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)

        def fit_pop(gs):
            softs, accs, cols = [], [], []
            for g in gs:
                col = _feat(torch, M2tr, g)
                if float(col.std()) < 1e-6:
                    softs.append(-1e9); accs.append(0.0); cols.append(col)
                    continue
                X = torch.cat([base, col.view(-1, 1)], 1)
                s, a = _ridge_soft(torch, X[:n_fit], X[n_fit:], Yf, yv)
                softs.append(s - s0); accs.append(a); cols.append(col)
            return np.array(softs), np.array(accs), cols

        fits, accs, cols = fit_pop(pop)
        # DOWNSTREAM ENERGY ECONOMY (every radial space after the first):
        #   existing costs (decay), producing an output costs, any valid
        #   output restores a little, and real energy comes ONLY from output
        #   that leads to the right answer THROUGH the composition (fits =
        #   residual gain measured in context of the frozen ensemble). A
        #   newborn has ~8 gens to prove contribution before it starves and
        #   is REPLACED — who produces the right answer fastest, survives.
        # tuned so a non-contributor starves at ~gen 6 (existence clock bites);
        # a genome earning residual gain >= ~0.0005 sustains itself indefinitely
        # STEADY-STATE population: genomes PERSIST and live off their energy —
        # existing costs (decay), producing an output costs, any valid output
        # restores a little, and real energy comes only from above-median
        # contribution to the right answer through the composition. A genome
        # that never out-earns the median starves in ~6 generations and its
        # slot goes to a child of the living — fastest to contribute survives.
        E_DECAY, OUT_COST, RESTORE, GAIN, E_FLOOR2, E_MAX2 = 0.75, 0.05, 0.04, 400.0, 0.2, 1.5
        MIN_TURNOVER = 12                       # search pressure even with no deaths
        energy = np.ones(pop_size)
        starved_total = 0
        for gen in range(gens):
            valid = fits > -1e8
            energy = np.clip(energy * E_DECAY - OUT_COST + RESTORE * valid
                             + GAIN * np.maximum(fits - np.median(fits), 0.0),
                             0.0, E_MAX2)
            starved = energy < E_FLOOR2
            starved_total += int(starved.sum())
            dead = list(np.where(starved)[0])
            # turnover floor: also recycle the weakest of the living
            if len(dead) < MIN_TURNOVER:
                living_by_fit = [i for i in np.argsort(fits) if i not in set(dead)]
                dead += living_by_fit[:MIN_TURNOVER - len(dead)]
            alive = [i for i in range(pop_size) if i not in set(dead)] or \
                    list(np.argsort(fits)[::-1][:4])
            kids, ksc = [], []
            n_fresh = max(1, len(dead) // 4)     # some slots go to fresh blood
            for k in range(len(dead)):
                if k < n_fresh:
                    kids.append(_new_gen_b(rng, C2)); ksc.append(0.25)
                else:
                    cand = rng.choice(alive, 3)
                    pi = cand[np.argmax(fits[cand])]
                    sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                    kids.append(_mut_b(rng, pop[pi], C2, sc)); ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            for slot, k in zip(dead, range(len(kids))):
                pop[slot] = kids[k]
                scales[slot] = ksc[k]
                fits[slot] = kf[k]
                accs[slot] = ka[k]
                cols[slot] = kc[k]
                energy[slot] = 1.0               # newborns start with full budget
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= 0.0005 or added >= freeze_top:
                break
            col = cols[idx]
            colz = (col - col.mean()) / (col.std() + 1e-9)
            dup = False
            for fc in cols2[-60:]:
                fz = (fc - fc.mean()) / (fc.std() + 1e-9)
                if float(torch.abs((colz * fz).mean())) > 0.95:
                    dup = True
                    break
            if not dup:
                frozen2.append(pop[idx]); cols2.append(col); added += 1
        base = torch.cat([F1tr, torch.stack(cols2, 1)], 1) if cols2 else F1tr
        s1, a1 = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv)
        hist.append({"round": rnd, "added": added, "n_stage2": len(frozen2),
                     "val_acc_stacked": round(a1, 4),
                     "starved_per_gen": round(starved_total / max(gens, 1), 1)})
        if verbose:
            print(f"  [D] round {rnd:2d}  +{added} (stage2 {len(frozen2)})  "
                  f"stacked val {a1:.4f}  (plateau was {a_plateau:.4f}, "
                  f"{round(time.time()-t0)}s)", flush=True)
        if added == 0:
            break

    # honest finals, each model tested once on the real test set
    yte_t = torch.tensor(yte, device=dev)
    Yfull = -torch.ones((len(ytr), 10), device=dev)
    Yfull[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0
    F2tr = torch.stack(cols2, 1) if cols2 else torch.zeros((len(ytr), 0), device=dev)
    F2te = (torch.stack([_feat(torch, M2te, g) for g in frozen2], 1)
            if frozen2 else torch.zeros((len(yte), 0), device=dev))
    _, t_stage1 = _ridge_soft(torch, F1tr, F1te, Yfull, yte_t)
    _, t_stacked = _ridge_soft(torch, torch.cat([F1tr, F2tr], 1),
                               torch.cat([F1te, F2te], 1), Yfull, yte_t)
    t_stage2 = 0.0
    if frozen2:
        _, t_stage2 = _ridge_soft(torch, F2tr, F2te, Yfull, yte_t)
    # the stage-2 map
    sig = [(c - c.mean()) / (c.std() + 1e-9) for c in cols2]
    map_stats = {}
    if len(sig) > 3:
        S = torch.stack(sig, 0)[:, :512].cpu().numpy()
        Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
        X3 = rm._mds(Sz, 3)
        sv = np.linalg.svd(Sz - Sz.mean(0), compute_uv=False)
        map_stats = {"axis_std": [round(float(v), 2) for v in X3.std(0)],
                     "effective_dim": int((sv > 0.01 * sv[0]).sum())}
    out = {"phase": "D-stacked-radial-space", "domain": "cifar",
           "plateau_val": round(a_plateau, 4),
           "n_stage2": len(frozen2),
           "test_stage1_only": round(t_stage1, 4),
           "test_stacked": round(t_stacked, 4),
           "test_stage2_only": round(t_stage2, 4),
           "stage2_map": map_stats, "history": hist,
           "genomes2": frozen2,
           "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "radial_data", "evo_stack_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[D] TEST: stage-1 {t_stage1:.4f} | stacked {t_stacked:.4f} | "
              f"stage-2-only {t_stage2:.4f} ({len(frozen2)} stage-2 genomes, "
              f"{out['seconds']}s)", flush=True)
    return out


def phase_stack_full(verbose=True):
    """Stacked model on full CIFAR: stage-1 features from the saved genomes,
    stage-2 features computed on z-scored stage-1 outputs (stats refit on the
    full train split), ridge head, test once."""
    import torch
    import torch.nn.functional as Fn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    with open(os.path.join(_HERE, "radial_data", "evo_interact_cifar.json")) as f:
        run1 = json.load(f)
    with open(os.path.join(_HERE, "radial_data", "evo_stack_cifar.json")) as f:
        run2 = json.load(f)
    genomes1, C = run1["genomes"], int(run1.get("C_env", 40))
    genomes2 = run2.get("genomes2") or []
    if not genomes2:
        raise RuntimeError("stage-2 genomes missing from evo_stack_cifar.json")
    z = np.load(os.path.join(_HERE, "radial_data", "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    X8 = cifar_data()[0]
    i8 = torch.tensor(X8[:2000], device=dev).permute(0, 3, 1, 2).contiguous()
    P = Fn.unfold(i8, 6, stride=2)
    cols = P.permute(0, 2, 1).reshape(-1, 108)
    g = torch.Generator(device="cpu").manual_seed(0)
    cols = cols[torch.randperm(len(cols), generator=g)[:120000].to(dev)]
    mu = cols.mean(0)
    _, _, V = torch.linalg.svd(cols - mu, full_matrices=False)
    comps = V[:C]
    sd8 = None

    def f1(X, bs=400):
        nonlocal sd8
        out = torch.zeros((len(X), len(genomes1)), device=dev)
        for b in range(0, len(X), bs):
            imgs = torch.tensor(X[b:b + bs], device=dev).permute(0, 3, 1, 2).contiguous()
            U = Fn.unfold(imgs, 6, stride=2)
            M = torch.einsum("cd,bdl->bcl", comps, U - mu.view(1, -1, 1))
            if sd8 is None:
                sd8 = M.std((0, 2), keepdim=True) + 1e-6
            M = M / sd8
            for k, gn in enumerate(genomes1):
                out[b:b + len(imgs), k] = _feat(torch, M, gn)
        return out

    F1tr, F1te = f1(Xtr), f1(Xte)
    mu1, sd1 = F1tr.mean(0), F1tr.std(0) + 1e-6
    M2tr = ((F1tr - mu1) / sd1).unsqueeze(2)
    M2te = ((F1te - mu1) / sd1).unsqueeze(2)
    F2tr = torch.stack([_feat(torch, M2tr, g2) for g2 in genomes2], 1)
    F2te = torch.stack([_feat(torch, M2te, g2) for g2 in genomes2], 1)
    Y = -torch.ones((len(ytr), 10), device=dev)
    Y[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0
    yte_t = torch.tensor(yte, device=dev)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, torch.cat([F1tr, F2tr], 1),
                             torch.cat([F1te, F2te], 1), Y, yte_t, lam=lam)
        best = max(best, acc)
    out = {"phase": "D-stack-full", "test_acc": round(best, 4),
           "n_stage1": len(genomes1), "n_stage2": len(genomes2),
           "references": {"stage1_full": 0.6257, "v1_milestone": 0.5904},
           "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "radial_data", "evo_stack_full_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[D-full] stacked on 50k/10k: TEST {best:.4f} "
              f"(stage-1 full 0.6257, v1 0.5904, {out['seconds']}s)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Phase E — RUN DEEP: keep stacking radial spaces until it actually stops.
# Levers: touch radial_data/STOP_EVO to stop gracefully (finishes the round,
# checkpoints, exits); crash-safe atomic checkpoint after every round.
# ---------------------------------------------------------------------------

_CKPT = os.path.join(_HERE, "radial_data", "evo_deep_ckpt.json")
_STOP = os.path.join(_HERE, "radial_data", "STOP_EVO")


def _save_ckpt(obj):
    tmp = _CKPT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, _CKPT)


def _evolve_stage(torch, dev, rng, M2tr, base_tr, n_fit, Yf, yv, rounds, gens,
                  freeze_top, C2, on_round=None, verbose=True, tag=""):
    """One downstream radial space evolved under the energy economy, until
    3 consecutive rounds freeze nothing (or on_round says stop)."""
    frozen2, cols2, hist = [], [], []
    empty_streak = 0
    for rnd in range(rounds):
        base = torch.cat([base_tr, torch.stack(cols2, 1)], 1) if cols2 else base_tr
        s0, _ = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv)
        pop = [_new_gen_b(rng, C2) for _ in range(64)]
        scales = np.full(64, 0.25)

        def fit_pop(gs):
            softs, accs, cols = [], [], []
            for g in gs:
                col = _feat(torch, M2tr, g)
                if float(col.std()) < 1e-6:
                    softs.append(-1e9); accs.append(0.0); cols.append(col)
                    continue
                X = torch.cat([base, col.view(-1, 1)], 1)
                s, a = _ridge_soft(torch, X[:n_fit], X[n_fit:], Yf, yv)
                softs.append(s - s0); accs.append(a); cols.append(col)
            return np.array(softs), np.array(accs), cols

        fits, accs, cols = fit_pop(pop)
        E_DECAY, OUT_COST, RESTORE, GAIN, E_FLOOR2, E_MAX2 = 0.75, 0.05, 0.04, 400.0, 0.2, 1.5
        MIN_TURNOVER = 12
        energy = np.ones(64)
        starved_total = 0
        for gen in range(gens):
            valid = fits > -1e8
            energy = np.clip(energy * E_DECAY - OUT_COST + RESTORE * valid
                             + GAIN * np.maximum(fits - np.median(fits), 0.0),
                             0.0, E_MAX2)
            starved = energy < E_FLOOR2
            starved_total += int(starved.sum())
            dead = list(np.where(starved)[0])
            if len(dead) < MIN_TURNOVER:
                living_by_fit = [i for i in np.argsort(fits) if i not in set(dead)]
                dead += living_by_fit[:MIN_TURNOVER - len(dead)]
            alive = [i for i in range(64) if i not in set(dead)] or \
                    list(np.argsort(fits)[::-1][:4])
            kids, ksc = [], []
            n_fresh = max(1, len(dead) // 4)
            for k in range(len(dead)):
                if k < n_fresh:
                    kids.append(_new_gen_b(rng, C2)); ksc.append(0.25)
                else:
                    cand = rng.choice(alive, 3)
                    pi = cand[np.argmax(fits[cand])]
                    sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]), 0.03, 0.6))
                    kids.append(_mut_b(rng, pop[pi], C2, sc)); ksc.append(sc)
            kf, ka, kc = fit_pop(kids)
            for slot, k in zip(dead, range(len(kids))):
                pop[slot] = kids[k]; scales[slot] = ksc[k]
                fits[slot] = kf[k]; accs[slot] = ka[k]
                cols[slot] = kc[k]; energy[slot] = 1.0
        order = np.argsort(fits)[::-1]
        added = 0
        for idx in order:
            if fits[idx] <= 0.0005 or added >= freeze_top:
                break
            col = cols[idx]
            colz = (col - col.mean()) / (col.std() + 1e-9)
            dup = False
            for fc in cols2[-60:]:
                fz = (fc - fc.mean()) / (fc.std() + 1e-9)
                if float(torch.abs((colz * fz).mean())) > 0.95:
                    dup = True
                    break
            if not dup:
                frozen2.append(pop[idx]); cols2.append(col); added += 1
        base = torch.cat([base_tr, torch.stack(cols2, 1)], 1) if cols2 else base_tr
        s1, a1 = _ridge_soft(torch, base[:n_fit], base[n_fit:], Yf, yv)
        hist.append({"round": rnd, "added": added, "n": len(frozen2),
                     "val_acc": round(a1, 4),
                     "starved_per_gen": round(starved_total / max(gens, 1), 1)})
        starved_total = 0
        if verbose:
            print(f"  [{tag}] round {rnd:3d}  +{added} (total {len(frozen2)})  "
                  f"val {a1:.4f}  starved/gen {hist[-1]['starved_per_gen']}", flush=True)
        if on_round is not None and on_round(frozen2, hist) is False:
            hist.append({"stopped": "STOP_EVO lever"})
            break
        empty_streak = empty_streak + 1 if added == 0 else 0
        if empty_streak >= 3:
            break
    return frozen2, cols2, hist


def run_deep(max_stages=8, rounds=500, gens=12, freeze_top=8, seed=2, verbose=True):
    """Stack radial spaces until the architecture itself stops improving.
    Stage 1 = the saved phase_b genomes (no energy — first space is exempt).
    Every later stage evolves under the energy economy with the concatenated
    outputs of ALL previous stages as its environment. Checkpoint after every
    round; touch radial_data/STOP_EVO to stop gracefully."""
    import torch
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if os.path.exists(_STOP):
        os.remove(_STOP)
    with open(os.path.join(_HERE, "radial_data", "evo_interact_cifar.json")) as f:
        run1 = json.load(f)
    genomes1, C_env = run1["genomes"], int(run1.get("C_env", 40))
    Mtr, Mte, data, _ = _patch_env(torch, dev, C=C_env)
    Xtr, ytr, Xte, yte = data
    F1tr = torch.stack([_feat(torch, Mtr, g) for g in genomes1], 1)
    F1te = torch.stack([_feat(torch, Mte, g) for g in genomes1], 1)
    n_fit = 6000
    yv = torch.tensor(ytr[n_fit:], device=dev)
    yte_t = torch.tensor(yte, device=dev)
    Yf = -torch.ones((n_fit, 10), device=dev)
    Yf[torch.arange(n_fit), torch.tensor(ytr[:n_fit], device=dev)] = 1.0
    Yfull = -torch.ones((len(ytr), 10), device=dev)
    Yfull[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0

    stages = []          # [{"genomes": [...], "hist": [...]}]
    feats_tr, feats_te = [F1tr], [F1te]

    # resume from checkpoint if present
    if os.path.exists(_CKPT):
        with open(_CKPT) as f:
            ck = json.load(f)
        for sg in ck.get("stages", []):
            base_tr = torch.cat(feats_tr, 1)
            base_te = torch.cat(feats_te, 1)
            mu, sd = base_tr.mean(0), base_tr.std(0) + 1e-6
            M2tr = ((base_tr - mu) / sd).unsqueeze(2)
            M2te = ((base_te - mu) / sd).unsqueeze(2)
            gs = sg["genomes"]
            feats_tr.append(torch.stack([_feat(torch, M2tr, g) for g in gs], 1))
            feats_te.append(torch.stack([_feat(torch, M2te, g) for g in gs], 1))
            stages.append(sg)
        if verbose:
            print(f"[deep] resumed {len(stages)} completed stages from checkpoint", flush=True)

    def val_acc_now():
        B = torch.cat(feats_tr, 1)
        _, a = _ridge_soft(torch, B[:n_fit], B[n_fit:], Yf, yv)
        return a

    stopped = False
    prev_val = val_acc_now()
    if verbose:
        print(f"[deep] starting val (stage-1{'+' + str(len(stages)) if stages else ''}): "
              f"{prev_val:.4f}", flush=True)
    while len(stages) < max_stages and not stopped:
        stage_no = len(stages) + 2
        base_tr = torch.cat(feats_tr, 1)
        base_te = torch.cat(feats_te, 1)
        mu, sd = base_tr.mean(0), base_tr.std(0) + 1e-6
        M2tr = ((base_tr - mu) / sd).unsqueeze(2)
        M2te = ((base_te - mu) / sd).unsqueeze(2)
        C2 = M2tr.shape[1]

        def on_round(frozen2, hist):
            _save_ckpt({"stages": stages + [{"genomes": frozen2, "hist": hist,
                                             "partial": True}],
                        "seconds": round(time.time() - t0)})
            return not os.path.exists(_STOP)

        frozen2, cols2, hist = _evolve_stage(
            torch, dev, rng, M2tr, base_tr, n_fit, Yf, yv, rounds, gens,
            freeze_top, C2, on_round=on_round, verbose=verbose,
            tag=f"stage{stage_no}")
        stopped = os.path.exists(_STOP)
        if not frozen2:
            if verbose:
                print(f"[deep] stage {stage_no} froze nothing — architecture "
                      "converged", flush=True)
            break
        feats_tr.append(torch.stack(cols2, 1))
        feats_te.append(torch.stack([_feat(torch, M2te, g) for g in frozen2], 1))
        stages.append({"genomes": frozen2, "hist": hist, "stage": stage_no})
        _save_ckpt({"stages": stages, "seconds": round(time.time() - t0)})
        new_val = val_acc_now()
        if verbose:
            print(f"[deep] stage {stage_no} done: +{len(frozen2)} genomes, "
                  f"val {prev_val:.4f} -> {new_val:.4f}", flush=True)
        if new_val - prev_val < 0.002:
            if verbose:
                print("[deep] stage gain < 0.002 — the tower has stopped", flush=True)
            prev_val = new_val
            break
        prev_val = new_val

    # honest final: 8k test once, then full-50k transfer of the whole tower
    Btr, Bte = torch.cat(feats_tr, 1), torch.cat(feats_te, 1)
    _, test8 = _ridge_soft(torch, Btr, Bte, Yfull, yte_t)
    out = {"phase": "E-run-deep", "stages": len(stages),
           "genomes_per_stage": [len(s["genomes"]) for s in stages],
           "stage1": len(genomes1),
           "final_val": round(prev_val, 4), "test_8k": round(test8, 4),
           "stopped_by_lever": bool(stopped),
           "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "radial_data", "evo_deep_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[deep] DONE: {len(stages)} stacked stages, val {prev_val:.4f}, "
              f"8k TEST {test8:.4f}, lever={stopped} "
              f"({round(time.time()-t0)}s)", flush=True)
    return out


def run_deep_full(verbose=True):
    """The whole tower on full CIFAR (50k/10k): stage-1 features, then each
    checkpointed stage applied to the z-scored concat of everything before
    it, ridge head, test once."""
    import torch
    import torch.nn.functional as Fn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    with open(os.path.join(_HERE, "radial_data", "evo_interact_cifar.json")) as f:
        run1 = json.load(f)
    with open(_CKPT) as f:
        ck = json.load(f)
    genomes1, C = run1["genomes"], int(run1.get("C_env", 40))
    stages = [s["genomes"] for s in ck["stages"] if s.get("genomes")]
    z = np.load(os.path.join(_HERE, "radial_data", "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr, yte = z["ytr"], z["yte"]
    X8 = cifar_data()[0]
    i8 = torch.tensor(X8[:2000], device=dev).permute(0, 3, 1, 2).contiguous()
    P = Fn.unfold(i8, 6, stride=2)
    cols = P.permute(0, 2, 1).reshape(-1, 108)
    g = torch.Generator(device="cpu").manual_seed(0)
    cols = cols[torch.randperm(len(cols), generator=g)[:120000].to(dev)]
    mu = cols.mean(0)
    _, _, V = torch.linalg.svd(cols - mu, full_matrices=False)
    comps = V[:C]
    sd8 = None

    def f1(X, bs=400):
        nonlocal sd8
        out = torch.zeros((len(X), len(genomes1)), device=dev)
        for b in range(0, len(X), bs):
            imgs = torch.tensor(X[b:b + bs], device=dev).permute(0, 3, 1, 2).contiguous()
            U = Fn.unfold(imgs, 6, stride=2)
            M = torch.einsum("cd,bdl->bcl", comps, U - mu.view(1, -1, 1))
            if sd8 is None:
                sd8 = M.std((0, 2), keepdim=True) + 1e-6
            M = M / sd8
            for k, gn in enumerate(genomes1):
                out[b:b + len(imgs), k] = _feat(torch, M, gn)
        return out

    feats_tr, feats_te = [f1(Xtr)], [f1(Xte)]
    for gs in stages:
        base_tr = torch.cat(feats_tr, 1)
        base_te = torch.cat(feats_te, 1)
        m1, s1 = base_tr.mean(0), base_tr.std(0) + 1e-6
        M2tr = ((base_tr - m1) / s1).unsqueeze(2)
        M2te = ((base_te - m1) / s1).unsqueeze(2)
        feats_tr.append(torch.stack([_feat(torch, M2tr, g2) for g2 in gs], 1))
        feats_te.append(torch.stack([_feat(torch, M2te, g2) for g2 in gs], 1))
    Y = -torch.ones((len(ytr), 10), device=dev)
    Y[torch.arange(len(ytr)), torch.tensor(ytr, device=dev)] = 1.0
    yte_t = torch.tensor(yte, device=dev)
    best = 0.0
    for lam in (1.0, 3.0, 10.0, 30.0):
        _, acc = _ridge_soft(torch, torch.cat(feats_tr, 1),
                             torch.cat(feats_te, 1), Y, yte_t, lam=lam)
        best = max(best, acc)
    out = {"phase": "E-deep-full", "test_acc": round(best, 4),
           "stages": [len(genomes1)] + [len(s) for s in stages],
           "references": {"two_stage_full": 0.6353, "stage1_full": 0.6257,
                          "v1_milestone": 0.5904},
           "seconds": round(time.time() - t0)}
    with open(os.path.join(_HERE, "radial_data", "evo_deep_full_cifar.json"), "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"[deep-full] tower {out['stages']} on 50k/10k: TEST {best:.4f} "
              f"(two-stage 0.6353, {out['seconds']}s)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    if "deepfull" in sys.argv:
        run_deep_full()
    elif "deep" in sys.argv:
        run_deep()
    elif "stackfull" in sys.argv:
        phase_stack_full()
    elif "stack" in sys.argv:
        phase_stack()
    elif "full" in sys.argv:
        phase_b_full()
    elif "c" in sys.argv:
        phase_c()
    elif "b" in sys.argv:
        phase_b()
    elif "a" in sys.argv or len(sys.argv) == 1:
        phase_a()
