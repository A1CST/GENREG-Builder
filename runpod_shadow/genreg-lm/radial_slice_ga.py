"""radial_slice_ga.py — the FIRST radial genome: an evolved slice picker.

The baselines measured that on CIFAR a good ~24-lens slice (0.4015) beats the
full 400-lens bank (0.3815), raw pixels (0.3820) and random 24-lens subsets
(0.3774). The rotation probe finds good slices geometrically; this genome
EVOLVES one directly. Genome = 24 lens indices. Fitness is soft (mean
log-softmax of the true class on a held-back validation split, never test).
Energy homeostasis per GENREG rules; selection is tournament + elitism. The
classifier stays closed-form kernel ridge — evolution only chooses the VIEWS,
exactly the division of labor the radial thesis proposes.

Per-lens Gram matrices are cached on the GPU once, and fitness is evaluated
for the WHOLE population in a single batched solve per generation — the 4080
does one (POP, N, N) LU instead of thousands of sequential small calls.
"""
import json
import os
import time

import numpy as np

import radial_map as rm
from radial_baseline import cifar_data, _tprims, _lens_apply_t, _kridge_factory

_HERE = os.path.dirname(os.path.abspath(__file__))

K = 24            # genome length (matches the measured slice size)
L = 300           # lens pool
N_FIT, N_VAL = 2000, 1000
POP, GENS = 32, 60
ELITE = 4
E_DECAY, E_GAIN, E_FLOOR, E_MAX = 0.90, 1.5, 0.2, 1.5


def _cache_grams(torch, tp, Ffit, Fval):
    grams = []
    kept = []
    for i in range(L * 2):
        prog = rm.lens_program(i)
        Z = _lens_apply_t(tp, prog, Ffit)
        sd = Z.std(0)
        if float(sd.mean()) < 1e-6:
            continue
        mu = Z.mean(0)
        ok = sd > 1e-6
        Z = torch.where(ok, (Z - mu) / (sd + 1e-9), torch.zeros_like(Z))
        Zv = _lens_apply_t(tp, prog, Fval)
        Zv = torch.where(ok, (Zv - mu) / (sd + 1e-9), torch.zeros_like(Zv))
        grams.append((Z @ Z.T, Zv @ Z.T))          # fp32: no per-eval converts
        kept.append(i)
        if len(kept) >= L:
            break
    return grams, kept


def run(seed=0, verbose=True):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    rng = np.random.default_rng(seed)

    Xtr, ytr, Xte, yte = cifar_data()
    fit_idx = np.arange(N_FIT)
    val_idx = np.arange(N_FIT, N_FIT + N_VAL)
    Ffit = torch.tensor(Xtr[fit_idx].reshape(N_FIT, -1), device=dev)
    Fval = torch.tensor(Xtr[val_idx].reshape(N_VAL, -1), device=dev)
    yfit = torch.tensor(ytr[fit_idx], device=dev)
    yval = torch.tensor(ytr[val_idx], device=dev)

    t0 = time.time()
    grams, kept = _cache_grams(torch, tp, Ffit, Fval)
    if verbose:
        print(f"cached {len(grams)} per-lens grams in {round(time.time()-t0)}s", flush=True)

    Y = -torch.ones((N_FIT, 10), device=dev)
    Y[torch.arange(N_FIT), yfit] = 1.0
    eye = torch.eye(N_FIT, device=dev)

    def fitness_batch(genomes):
        """One batched LU for the whole set — this is where the 4080 earns it."""
        B = len(genomes)
        Ktr = torch.zeros((B, N_FIT, N_FIT), device=dev)
        Kva = torch.zeros((B, N_VAL, N_FIT), device=dev)
        for b, genome in enumerate(genomes):
            for g in genome:
                Ktr[b] += grams[g][0]
                Kva[b] += grams[g][1]
        Ktr += (100.0 * K) * eye
        a = torch.linalg.solve(Ktr, Y.expand(B, -1, -1))
        scores = Kva @ a                                    # (B, N_VAL, 10)
        soft = torch.log_softmax(scores, 2)[:, torch.arange(N_VAL), yval].mean(1)
        acc = (scores.argmax(2) == yval).float().mean(1)
        return [float(s) for s in soft], [float(a2) for a2 in acc]

    pop = [np.sort(rng.choice(L, K, replace=False)) for _ in range(POP)]
    f0, a0 = fitness_batch(pop)
    fits, accs = np.array(f0), np.array(a0)
    energy = np.ones(POP)
    hist = []
    for gen in range(GENS):
        med = np.median(fits)
        energy = np.clip(energy * E_DECAY + E_GAIN * (fits - med), 0.0, E_MAX)
        starved = energy < E_FLOOR
        order = np.argsort(fits)[::-1]
        keep = list(order[:ELITE])
        alive = [i for i in range(POP) if not starved[i]] or list(range(POP))
        children = []
        while len(children) < POP - ELITE:
            cand = rng.choice(alive, 3)
            parent = pop[cand[np.argmax(fits[cand])]]
            child = parent.copy()
            for _ in range(int(rng.integers(1, 4))):        # swap 1-3 lenses
                pos = int(rng.integers(K))
                pool = np.setdiff1d(np.arange(L), child)
                child[pos] = pool[int(rng.integers(len(pool)))]
            children.append(np.sort(child))
        cf, ca = fitness_batch(children)
        pop = [pop[i] for i in keep] + children
        fits = np.concatenate([fits[keep], cf])
        accs = np.concatenate([accs[keep], ca])
        energy = np.concatenate([energy[keep], np.ones(len(children))])
        hist.append({"gen": gen, "best_soft": round(float(fits.max()), 4),
                     "best_val_acc": round(float(accs[np.argmax(fits)]), 4),
                     "starved": int(starved.sum())})
        if verbose and gen % 10 == 0:
            h = hist[-1]
            print(f"  gen {gen:3d}  soft {h['best_soft']}  val {h['best_val_acc']}  "
                  f"starved {h['starved']}", flush=True)

    best = pop[int(np.argmax(fits))]
    best_ids = [int(kept[g]) for g in best]

    # final honest eval: full 8k train, real 2k test, chosen lenses only
    solve = _kridge_factory(ytr, yte, 10)
    Ftr = torch.tensor(Xtr.reshape(len(Xtr), -1), device=dev)
    Fte = torch.tensor(Xte.reshape(len(Xte), -1), device=dev)
    Ktr = torch.zeros((len(Xtr), len(Xtr)), device=dev)
    Kte = torch.zeros((len(Xte), len(Xtr)), device=dev)
    for i in best_ids:
        prog = rm.lens_program(i)
        Z = _lens_apply_t(tp, prog, Ftr)
        mu, sd = Z.mean(0), Z.std(0)
        ok = sd > 1e-6
        Z = torch.where(ok, (Z - mu) / (sd + 1e-9), torch.zeros_like(Z))
        Zt = _lens_apply_t(tp, prog, Fte)
        Zt = torch.where(ok, (Zt - mu) / (sd + 1e-9), torch.zeros_like(Zt))
        Ktr += Z @ Z.T
        Kte += Zt @ Z.T
    final = solve(Ktr, Kte, K)
    out = {"domain": "cifar", "genome_len": K, "lens_pool": L,
           "test_acc": final["acc"], "genome": best_ids, "history": hist,
           "reference": {"rotation_best_slice": 0.4015, "full_bank_400": 0.3815,
                         "raw_linear": 0.3820, "random_24": 0.3774}}
    path = os.path.join(_HERE, "radial_data", "slice_ga_cifar.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    if verbose:
        print(f"\nEVOLVED 24-lens slice test acc: {final['acc']}  "
              f"(rotation slice 0.4015, bank 0.3815, random 0.3774)", flush=True)
        print(f"written {path}", flush=True)
    return out


if __name__ == "__main__":
    run()
