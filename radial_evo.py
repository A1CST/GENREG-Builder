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


if __name__ == "__main__":
    import sys
    if "a" in sys.argv or len(sys.argv) == 1:
        phase_a()
