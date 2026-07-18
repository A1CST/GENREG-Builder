"""radial_embed.py — an RS embedding model: evolution separates words.

No prediction anywhere. Each word's ENVIRONMENT is its distributional
profile — co-occurrence with the top-K context words (window +-3) over an
independent corpus slice, PPMI-weighted. Genomes are vec-grammar programs
over those profile channels; EACH FROZEN GENOME IS ONE EMBEDDING
DIMENSION (the CIFAR contrastive encoder's private-language design,
replayed on words — diversity-first, labels never touch the environment).

Fitness = separation quality, per dimension:
  agreement — over sampled word pairs, does |d(i)-d(j)| order pairs the
              way profile dissimilarity orders them?
  novelty   — decorrelation from every frozen dimension (a dimension
              earns nothing for restating another).
  spread    — degenerate near-constant outputs die.

Honesty rails: genomes see only HALF the context channels (split A);
the evaluation similarity is computed from the held-out half (split B),
so a dimension cannot score by memorizing its own inputs. The SVD
baseline (wiki_feats, built from the full wikipedia corpus) is evaluated
on the identical held-out metric.

Exports radial_data/embed_rs.npz (vocab + evolved table) and
radial_data/embed_report.json. No gradients anywhere.
"""
import os as _os, sys as _sys                     # repo-root shim
for _p in (_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
           _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import genreg_paths                               # noqa: F401
import json
import os
import time

import numpy as np

from radial_evo import _tprims
from radial_lm import _clean
import radial_stack as rk

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(_HERE, "corpora", "combined", "combined_corpus.txt")
N_WORDS = 5000                    # vocabulary to embed (by slice frequency)
K_CTX = 2000                      # context channels (top-K words)
WIN = 3                           # co-occurrence window
N_DIMS = 128                      # stop after this many frozen dimensions
N_PAIRS = 20000                   # sampled pairs for fitness/eval


def build_profiles(seed=0, corpus=CORPUS, offset=30_000_000,
                   span_mb=16.0, mode="window"):
    """Word x context PPMI profiles from a corpus slice.
    mode="window": symmetric +-WIN co-occurrence (semantic similarity).
    mode="next":   the word AFTER only - separates words by what FOLLOWS
                   them (sequence experience, the continuation ear).
    mode="prev":   the word BEFORE only."""
    with open(corpus, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(offset)
        text = _clean(f.read(int(span_mb * 1_000_000)))
    toks = text.split()
    cnt = {}
    for w in toks:
        cnt[w] = cnt.get(w, 0) + 1
    vocab = sorted(cnt, key=cnt.get, reverse=True)[:N_WORDS]
    ctx_words = vocab[:K_CTX]
    w2i = {w: i for i, w in enumerate(vocab)}
    c2i = {w: i for i, w in enumerate(ctx_words)}
    M = np.zeros((N_WORDS, K_CTX), np.float32)
    if mode == "window":
        rng_js = lambda i: range(max(0, i - WIN),
                                 min(len(toks), i + WIN + 1))
    elif mode == "next":
        rng_js = lambda i: range(i + 1, min(len(toks), i + 2))
    else:
        rng_js = lambda i: range(max(0, i - 1), i)
    for i, w in enumerate(toks):
        wi = w2i.get(w)
        if wi is None:
            continue
        for j in rng_js(i):
            if j == i:
                continue
            cj = c2i.get(toks[j])
            if cj is not None:
                M[wi, cj] += 1.0
    # PPMI (positive pointwise mutual information) — environment stats
    row = M.sum(1, keepdims=True) + 1e-9
    col = M.sum(0, keepdims=True) + 1e-9
    tot = M.sum()
    ppmi = np.log(np.maximum(M * tot / (row * col), 1.0))
    return vocab, ppmi.astype(np.float32)


def run(pop_size=96, gens=10, seed=5, out_path=None, verbose=True,
        corpus=CORPUS, offset=30_000_000, span_mb=16.0, mode="window",
        n_dims=None, npz_name="embed_rs.npz"):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tp = _tprims(torch)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    log_lines = []

    def log(m):
        log_lines.append(m)
        print(m, flush=True)

    global N_DIMS
    if n_dims:
        N_DIMS = int(n_dims)
    vocab, P = build_profiles(corpus=corpus, offset=offset,
                              span_mb=span_mb, mode=mode)
    N = len(vocab)
    # split channels: genomes see A; evaluation similarity comes from B
    perm = rng.permutation(K_CTX)
    A_idx, B_idx = perm[:K_CTX // 2], perm[K_CTX // 2:]
    PA = torch.tensor(P[:, A_idx], device=dev)
    PA = (PA - PA.mean(0)) / (PA.std(0) + 1e-6)
    PB = P[:, B_idx]
    PB_t = torch.tensor(PB, device=dev)
    PB_n = PB_t / (PB_t.norm(dim=1, keepdim=True) + 1e-9)
    log(f"profiles: {N} words x {K_CTX} ctx (A={len(A_idx)} for genomes, "
        f"B={len(B_idx)} held out) ({round(time.time()-t0)}s)")

    # fixed probe pairs; target = held-in (A-side) dissimilarity for
    # fitness, held-out (B-side) for the final report
    pi = torch.tensor(rng.integers(0, N, N_PAIRS), device=dev)
    pj = torch.tensor(rng.integers(0, N, N_PAIRS), device=dev)
    PA_n = torch.tensor(P[:, A_idx], device=dev)
    PA_n = PA_n / (PA_n.norm(dim=1, keepdim=True) + 1e-9)
    tA = 1.0 - (PA_n[pi] * PA_n[pj]).sum(1)         # fitness target
    tA = (tA - tA.mean()) / (tA.std() + 1e-9)

    def _san(v):
        return torch.nan_to_num(v, nan=0.0, posinf=0.0,
                                neginf=0.0).clamp(-1e6, 1e6)

    C = PA.shape[1]
    frozen, dims = [], []

    def score(cols):
        """(N, K) candidate dims -> fitness per candidate."""
        z = (cols - cols.mean(0)) / (cols.std(0) + 1e-6)
        d = (z[pi] - z[pj]).abs()                    # pair separations
        dz = (d - d.mean(0)) / (d.std(0) + 1e-6)
        agree = (dz * tA.view(-1, 1)).mean(0)        # corr with target
        if dims:
            D = torch.stack(dims, 1)
            Dz = (D - D.mean(0)) / (D.std(0) + 1e-6)
            red = ((Dz.T @ z) / N).abs().max(0).values
        else:
            red = torch.zeros(cols.shape[1], device=dev)
        spread = (cols.std(0) > 1e-4).float()
        return (agree * (1.0 - red) * spread)

    pop = [rk.new_vec_genome(rng, C) for _ in range(pop_size)]
    sc = 0.35
    round_i = 0
    while len(dims) < N_DIMS:
        for _g in range(gens):
            cols = torch.stack([_san(rk.feature_vec(torch, tp, PA, g))
                                for g in pop], 1)
            fit = score(cols).cpu().numpy()
            order = np.argsort(-fit)
            elite = [pop[k] for k in order[:pop_size // 4]]
            pop = list(elite)
            while len(pop) < pop_size:
                p = elite[rng.integers(0, len(elite))]
                pop.append(rk.mutate_vec(rng, p, sc, C))
        # freeze the best decorrelated survivors of the round
        cols = torch.stack([_san(rk.feature_vec(torch, tp, PA, g))
                            for g in pop], 1)
        fit = score(cols).cpu().numpy()
        order = np.argsort(-fit)
        took = 0
        for k in order:
            if fit[k] <= 0.02 or took >= 8 or len(dims) >= N_DIMS:
                break
            z = cols[:, k]
            z = (z - z.mean()) / (z.std() + 1e-6)
            if dims:
                D = torch.stack(dims, 1)
                Dz = (D - D.mean(0)) / (D.std(0) + 1e-6)
                if float((Dz.T @ z).abs().max() / N) > 0.55:
                    continue
            dims.append(cols[:, k].clone())
            frozen.append(pop[k])
            took += 1
        round_i += 1
        log(f"  round {round_i}: +{took} dims (total {len(dims)}) "
            f"best-fit {fit[order[0]]:.4f} ({round(time.time()-t0)}s)")
        if took == 0:
            log("  round dry — stopping")
            break
        pop = [rk.new_vec_genome(rng, C) for _ in range(pop_size)]

    E = torch.stack(dims, 1)                          # (N, n_dims)
    Ez = (E - E.mean(0)) / (E.std(0) + 1e-6)

    # ---- the gates, on the HELD-OUT half ----------------------------
    def spearman_vs_heldout(X):
        Xn = X / (X.norm(dim=1, keepdim=True) + 1e-9)
        d_emb = 1.0 - (Xn[pi] * Xn[pj]).sum(1)
        d_tgt = 1.0 - (PB_n[pi] * PB_n[pj]).sum(1)
        a = d_emb.cpu().numpy()
        b = d_tgt.cpu().numpy()
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    sp_rs = spearman_vs_heldout(Ez)
    # SVD baseline on the SAME metric, shared vocab
    zf = np.load(os.path.join(_HERE, "corpora", "wikipedia",
                              "wiki_feats.npz"), allow_pickle=True)
    svd_vocab = {str(w): i for i, w in enumerate(zf["vocab"])}
    S = np.zeros((N, 128), np.float32)
    hit = 0
    for i, w in enumerate(vocab):
        j = svd_vocab.get(w)
        if j is not None:
            S[i] = zf["feat"][j]
            hit += 1
    sp_svd = spearman_vs_heldout(torch.tensor(S, device=dev))
    log(f"GATE spearman vs held-out profiles: RS {sp_rs:.4f} "
        f"({len(dims)} dims) vs SVD {sp_svd:.4f} (128 dims, "
        f"{hit}/{N} vocab hit)")

    # neighbor sanity, verbatim for the report
    probes = ["cat", "dog", "king", "three", "red", "water", "city",
              "said", "war", "music"]
    Ezn = (Ez / (Ez.norm(dim=1, keepdim=True) + 1e-9)).cpu().numpy()
    w2i = {w: i for i, w in enumerate(vocab)}
    neighbors = {}
    for w in probes:
        i = w2i.get(w)
        if i is None:
            continue
        simi = Ezn @ Ezn[i]
        top = np.argsort(-simi)[1:7]
        neighbors[w] = [vocab[k] for k in top]
        log(f"  NN[{w}]: {' '.join(neighbors[w])}")

    from anim_infer import count_params
    gp = sum(count_params(g) for g in frozen)
    out = {"phase": "radial-embed (RS embedding space, separation only)",
           "n_words": N, "n_dims": len(dims), "k_ctx": K_CTX,
           "spearman_rs": round(sp_rs, 4), "spearman_svd": round(sp_svd, 4),
           "neighbors": neighbors, "genome_params": gp,
           "seconds": round(time.time() - t0)}
    op = out_path or os.path.join(_HERE, "radial_data", "embed_report.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(_HERE, "radial_data", npz_name),
             vocab=np.array(vocab),
             feat=Ez.cpu().numpy().astype(np.float32))
    with open(os.path.join(_HERE, "radial_data", "embed_rs_model.json"),
              "w") as f:
        json.dump({"dims": frozen, "n_words": N}, f)
    rk._record_run(
        {"env": "rs-embed", "n_words": N, "k_ctx": K_CTX, "n_dims": N_DIMS,
         "pop_size": pop_size, "gens": gens, "seed": seed},
        [{"round": i, "added": 8, "val_acc": None, "n": None}
         for i in range(round_i)],
        {"spearman_rs": round(sp_rs, 4), "spearman_svd": round(sp_svd, 4),
         "n_dims": len(dims), "genome_params": gp},
        log_lines, ["lm", "embed", "radial"])
    print(f"[radial-embed] DONE: {len(dims)} dims, spearman RS {sp_rs:.4f} "
          f"vs SVD {sp_svd:.4f} ({round(time.time()-t0)}s)", flush=True)
    return out


if __name__ == "__main__":
    run()
