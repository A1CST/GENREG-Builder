"""replicate_wordspace.py — module 61: R1, the first WORD space.

R0 = image space (perceptual words, photometric-invariant). R1 = the first
word space: its genomes fire on WORDS, not images. Each R0-word is a data
point whose features are its activation profile across images; two words
"appear together" when their profiles correlate (they fire on the same
images). R1 evolves coordinates that are SMOOTH over that co-occurrence graph
-- co-firing words land near each other, non-co-firing words apart. Stack a
few coordinates and you have a word-embedding; each IMAGE is then placed in it
by mixing the embeddings of the words it activates, and the head reads that.

R1 alone is a linear reshaping of R0 so it is not expected to raise accuracy
(the user's call); it builds the geometry that R2 (similarity/opposition) will
read nonlinearly. This module builds R0 -> R1 -> image aggregation -> head and
reports honestly, plus a check that the embedding actually clusters
co-occurring words.

    python3 replicate/replicate_wordspace.py --r0_rounds 30 --r1_rounds 30
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)
for _p in ("replicate", "radial", "ocr"):
    sys.path.insert(0, os.path.join(_HERE, _p))
import genreg_paths                               # noqa: F401

from radial_evo import _tprims, _ridge_soft
import radial_evo2 as re2
from radial_evo2 import Env, SCALES, new_genome, mutate, feature
from replicate_langspace import _fp
from replicate_langstack import new_vec, mut_vec, feat_vec, quality

RD = os.path.join(_HERE, "radial_data")
E_DECAY, OUT_COST, RESTORE, E_GAIN, E_FLOOR, E_MAX = 0.75, 0.05, 0.04, 6.0, 0.2, 1.5
MIN_TURNOVER = 12


def log(m):
    print(m, flush=True)


def energy_evolve(rng, pop_size, gens, rounds, cap, freeze, new_fn, mut_fn,
                  fit_fn, sig_fn, tag, log_every=5):
    """generic label-free space under the energy economy. fit_fn(genomes)->
    np.array of fitness; sig_fn(genomes)->(D, K) signatures for diversity."""
    import torch
    frozen, fsig = [], []
    for rnd in range(rounds):
        if len(frozen) >= cap:
            break
        pop = [new_fn(rng) for _ in range(pop_size)]
        scales = np.full(pop_size, 0.25)
        fits = fit_fn(pop)
        energy = np.ones(pop_size); starved_total = 0
        for _g in range(gens):
            valid = fits > -1e8
            energy = np.clip(energy * E_DECAY - OUT_COST + RESTORE * valid
                             + E_GAIN * np.maximum(fits - np.median(fits), 0.),
                             0., E_MAX)
            starved = energy < E_FLOOR
            starved_total += int(starved.sum())
            dead = list(np.where(starved)[0])
            if len(dead) < MIN_TURNOVER:
                living = [i for i in np.argsort(fits) if i not in set(dead)]
                dead += living[:MIN_TURNOVER - len(dead)]
            alive = [i for i in range(pop_size) if i not in set(dead)] or \
                    list(np.argsort(fits)[::-1][:4])
            n_fresh = max(1, len(dead) // 4)
            kids, ksc = [], []
            for k in range(len(dead)):
                if k < n_fresh:
                    kids.append(new_fn(rng)); ksc.append(0.25)
                else:
                    cand = rng.choice(alive, 3)
                    pi = cand[int(np.argmax(fits[cand]))]
                    sc = float(np.clip(scales[pi] * rng.choice([1.3, 1 / 1.3]),
                                       0.03, 0.6))
                    kids.append(mut_fn(rng, pop[pi], sc)); ksc.append(sc)
            kf = fit_fn(kids)
            for slot, k in zip(dead, range(len(kids))):
                pop[slot] = kids[k]; scales[slot] = ksc[k]; fits[slot] = kf[k]
                energy[slot] = 1.0
        sigs = sig_fn(pop)
        order = np.argsort(fits)[::-1]; added = 0
        for i in order:
            if fits[i] < freeze:
                break
            s = sigs[:, i] - sigs[:, i].mean()
            s = s / (s.norm() + 1e-8)
            if any(float(torch.abs(s @ t)) > 0.9 for t in fsig[-120:]):
                continue
            frozen.append(pop[i]); fsig.append(s); added += 1
        if rnd % log_every == 0:
            log(f"[{tag}] round {rnd:3d}  +{added} ({tag} {len(frozen)})  "
                f"starved/gen {round(starved_total/max(gens,1),1)}  "
                f"best-fit {float(fits.max()):.3f}")
    return frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--r0_rounds", type=int, default=30)
    ap.add_argument("--r1_rounds", type=int, default=30)
    ap.add_argument("--r0_cap", type=int, default=1200)
    ap.add_argument("--r1_cap", type=int, default=400)
    ap.add_argument("--word_feats", type=int, default=48)
    args = ap.parse_args()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    t0 = time.time()
    z = np.load(os.path.join(RD, "cifar_full.npz"))
    Xtr = z["Xtr"].astype(np.float32) / 255.0
    Xte = z["Xte"].astype(np.float32) / 255.0
    ytr = torch.tensor(z["ytr"].astype(np.int64), device=dev)
    yte = torch.tensor(z["yte"].astype(np.int64), device=dev)
    N = len(z["ytr"])
    env0 = Env(torch, dev, Xtr, Xte, max_cached=len(SCALES))
    for ps in SCALES:
        env0.maps(ps)
    basis = {ps: re2._SVD_CACHE[(_fp(Xtr), ps)] for ps in SCALES}
    n_fit = int(N * 0.8); yv = ytr[n_fit:]
    Y = -torch.ones((N, 10), device=dev); Y[torch.arange(N), ytr] = 1.
    dev_name = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"

    def augment(Xnp, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        t = torch.tensor(Xnp, device=dev).permute(0, 3, 1, 2); n = len(t)
        b = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=dev) - .5) * .5
        con = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=dev) - .5) * .5
        col = 1 + (torch.rand(n, 3, 1, 1, generator=g, device=dev) - .5) * .3
        m = t.mean((2, 3), keepdim=True)
        t = (((t - m) * con + m) * b * col).clamp(0, 1)
        return t.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    def env_for(Xa):
        e = Env(torch, dev, Xa, Xa[:100], max_cached=len(SCALES))
        for ps in SCALES:
            re2._SVD_CACHE[(_fp(Xa), ps)] = basis[ps]
        return e

    def head(F, Fte, tag):
        mu, sd = F.mean(0), F.std(0) + 1e-6
        Z, Zt = (F - mu) / sd, (Fte - mu) / sd
        bl, bv = 30., -1
        for lam in (1., 3., 10., 30., 100., 300.):
            _, a = _ridge_soft(torch, Z[:n_fit], Z[n_fit:], Y[:n_fit], yv, lam=lam)
            if a > bv:
                bl, bv = lam, a
        A = torch.hstack([Z, torch.ones(N, 1, device=dev)])
        W = torch.linalg.solve((A.T @ A).double() + bl * torch.eye(A.shape[1],
                               device=dev, dtype=torch.float64),
                               (A.T @ Y).double()).float()
        B = torch.hstack([Zt, torch.ones(len(yte), 1, device=dev)])
        acc = float(((B @ W).argmax(1) == yte).float().mean())
        log(f"[word] HEAD [{tag}]: val {bv:.4f} TEST {acc:.4f}")
        return acc

    # ---- R0 : image space (photometric-invariant perceptual words) --------
    envA, envB = env_for(augment(Xtr, 1)), env_for(augment(Xtr, 2))
    probe_img = torch.tensor(np.random.default_rng(5).permutation(N)[:4000], device=dev)
    rng = np.random.default_rng(61)

    def r0_fit(gs):
        CA = torch.stack([feature(torch, tp, envA, g) for g in gs], 1)
        CB = torch.stack([feature(torch, tp, envB, g) for g in gs], 1)
        return quality(torch, CA, CB).cpu().numpy()

    def r0_sig(gs):
        return torch.stack([feature(torch, tp, env0, g) for g in gs], 1)[probe_img]
    R0 = energy_evolve(rng, args.pop, args.gens, args.r0_rounds, args.r0_cap,
                       0.02, lambda r: new_genome(r), lambda r, g, s: mutate(r, g, s),
                       r0_fit, r0_sig, "R0")
    W0 = torch.stack([feature(torch, tp, env0, g) for g in R0], 1)
    W0te = torch.stack([feature(torch, tp, env0, g, test=True) for g in R0], 1)
    mu0, sd0 = W0.mean(0), W0.std(0) + 1e-6
    Wz, Wzte = (W0 - mu0) / sd0, (W0te - mu0) / sd0
    F0 = len(R0)
    log(f"[word] R0 done: {F0} perceptual words ({round(time.time()-t0)}s)")

    # ---- word co-occurrence + per-word features (words are the data now) ---
    C = (Wz.T @ Wz) / N                                   # (F0,F0) co-occurrence
    Xw = Wz.T                                             # (F0, N) word profiles
    xm = Xw.mean(0)
    _, _, V = torch.linalg.svd(Xw - xm, full_matrices=False)
    Xword = (Xw - xm) @ V[:args.word_feats].T            # (F0, d) word features
    Xword = (Xword - Xword.mean(0)) / (Xword.std(0) + 1e-6)
    probe_w = torch.arange(F0, device=dev)               # signatures over words

    # ---- R1 : cluster co-occurring words (smooth over the co-occ graph) ----
    def r1_fit(gs):
        coords = torch.stack([feat_vec(torch, tp, Xword, g) for g in gs], 1)  # (F0,K)
        cz = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
        num = (cz * (C @ cz)).sum(0)                     # Rayleigh: c^T C c
        q = num / F0                                     # high = co-firing words share c
        q[~torch.isfinite(q)] = -1
        return q.cpu().numpy()

    def r1_sig(gs):
        coords = torch.stack([feat_vec(torch, tp, Xword, g) for g in gs], 1)
        return coords                                    # (F0, K) per-word signatures
    R1 = energy_evolve(rng, args.pop, args.gens, args.r1_rounds, args.r1_cap,
                       0.02, lambda r: new_vec(r, args.word_feats),
                       lambda r, g, s: mut_vec(r, g, s, args.word_feats),
                       r1_fit, r1_sig, "R1")
    if not R1:
        log("[word] R1 evolved no coordinates"); return
    E = torch.stack([feat_vec(torch, tp, Xword, g) for g in R1], 1)   # (F0, F1) word embedding
    E = (E - E.mean(0)) / (E.std(0) + 1e-6)
    log(f"[word] R1 done: {len(R1)} embedding coords ({round(time.time()-t0)}s)")

    # ---- R2 : opposition — words that NEVER fire together -> opposite poles -
    # fires on words too, but reads R1's EMBEDDING as each word's features and
    # finds axes that split never-together words to opposite ends (same C, sign
    # flipped; folds similarity+antonym into one RS). decorrelated from R1.
    F1 = len(R1)

    def r2_fit(gs):
        coords = torch.stack([feat_vec(torch, tp, E, g) for g in gs], 1)  # (F0,K)
        cz = (coords - coords.mean(0)) / (coords.std(0) + 1e-6)
        q = (cz * (C @ cz)).sum(0) / F0
        q[~torch.isfinite(q)] = -1
        return q.cpu().numpy()

    def r2_sig(gs):
        return torch.stack([feat_vec(torch, tp, E, g) for g in gs], 1)
    # seed R2's diversity signatures with R1's coords so R2 must find NEW axes
    R2 = energy_evolve(rng, args.pop, args.gens, args.r1_rounds, args.r1_cap,
                       0.02, lambda r: new_vec(r, F1),
                       lambda r, g, s: mut_vec(r, g, s, F1),
                       r2_fit, r2_sig, "R2")
    if R2:
        E2 = torch.stack([feat_vec(torch, tp, E, g) for g in R2], 1)
        E2 = (E2 - E2.mean(0)) / (E2.std(0) + 1e-6)
        Efull = torch.cat([E, E2], 1)
        log(f"[word] R2 done: {len(R2)} opposition coords ({round(time.time()-t0)}s)")
    else:
        Efull = E
        log("[word] R2 evolved no coords")

    # sanity: do co-occurring words actually sit closer in the embedding?
    import torch as _t
    hi = C > (C.mean() + 2 * C.std())                    # strongly co-firing pairs
    d2 = torch.cdist(E, E)                               # word-word embedding dist
    m_hi = float(d2[hi].mean()); m_all = float(d2.mean())
    log(f"[word] co-occurrence check: mean embed-dist for co-firing pairs "
        f"{m_hi:.3f} vs all pairs {m_all:.3f} "
        f"({'clusters' if m_hi < m_all else 'no clustering'})")

    # ---- place images in the word-embedding, then head --------------------
    img_emb = Wz @ Efull                                 # (N, F_emb)
    img_emb_te = Wzte @ Efull
    a0 = head(W0, W0te, f"R0 alone ({F0} words)")
    a1 = head(img_emb, img_emb_te, f"embedding alone ({Efull.shape[1]} dims)")
    aj = head(torch.cat([W0, img_emb], 1), torch.cat([W0te, img_emb_te], 1),
              f"R0 | embedding ({F0}+{Efull.shape[1]})")
    # NONLINEAR read of the semantic geometry: pairwise products of embedding
    # axes = "near cluster A AND far from cluster B" — what a linear head can't
    # form. this is where the geometry is supposed to pay off.
    g2 = torch.Generator(device=dev).manual_seed(9)
    D = min(4000, Efull.shape[1] ** 2)
    ia = torch.randint(0, Efull.shape[1], (D,), generator=g2, device=dev)
    ib = torch.randint(0, Efull.shape[1], (D,), generator=g2, device=dev)
    nl = img_emb[:, ia] * img_emb[:, ib]
    nl_te = img_emb_te[:, ia] * img_emb_te[:, ib]
    anl = head(torch.cat([W0, img_emb, nl], 1),
               torch.cat([W0te, img_emb_te, nl_te], 1),
               f"R0 | embedding | NONLINEAR products ({D})")
    log(f"[word] R0 {a0:.4f}  embed {a1:.4f}  R0|embed {aj:.4f}  "
        f"R0|embed|nonlinear {anl:.4f}")
    log(f"[word] COMPUTE: {round(time.time()-t0)}s on {dev_name}, peak "
        f"{torch.cuda.max_memory_allocated()/1e9 if dev=='cuda' else 0:.1f}GB")
    json.dump({"module": "wordspace", "F0": F0, "F1": len(R1),
               "r0_test": round(a0, 4), "r1_embed_test": round(a1, 4),
               "joint_test": round(aj, 4), "cooc_hi": round(m_hi, 3),
               "cooc_all": round(m_all, 3)},
              open(os.path.join(RD, "replicate_wordspace.json"), "w"))


if __name__ == "__main__":
    main()
