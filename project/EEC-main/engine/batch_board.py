"""Full-board constraint batch. Three sub-experiments, each measured by EMERGENT
INTERNALS on CLEAN segments (per the paradigm: read the memory machinery, not the
token output). Modes write JSON; `plot` mode renders the 2 comparison PNGs.

  occ   (A) OCCLUSION rho-sweep -- harden the occlusion law across rho and seeds.
  noise (C) SENSORY NOISE sigma-sweep -- a NEW orthogonal observability law:
            the world CORRUPTS the input (additive noise) instead of HIDING it.
            Does corruption drive memory the way blackout does? (memory as denoiser)
  stack (B) OCCLUSION x ENTROPY 2x2 -- do the two active-maintenance drivers
            COMPOUND (super-additive gain) or share one mechanism?

Worlds: text (large-vocab, thin signal) vs long-range (periodic, recoverable).
Memory machinery = recurrent gain (spectral radius of W_rec) + memory horizon.
"""
import os, sys, json
import numpy as np
from multiprocessing import Pool

import run_matrix as RM
import mind as MIND
from mind import Mind, RENT, START_ENERGY, reproduce
from evolve import POP_SIZE, EMBED

HERE = os.path.dirname(os.path.abspath(__file__))
# board_* artifacts are tracked under the observability_board experiment folder
OUT = os.path.join(HERE, "..", "experiments", "observability_board")
os.makedirs(OUT, exist_ok=True)
GENS = int(os.environ.get("EEC_MGENS", "80"))
SEG_TEXT = int(os.environ.get("EEC_SEGTEXT", "300"))   # text is the null control; keep cheap
SEG_LR = int(os.environ.get("EEC_SEGLR", "400"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "3"))))


def make_emb(m, seg, mask, noise_pat, sigma):
    emb = m.E[seg].copy()
    if mask is not None:
        emb[mask] = 0.0
    if sigma > 0.0 and noise_pat is not None:
        emb = emb + noise_pat * (sigma * float(emb.std()))   # relative SNR
    return emb


def eval_lives(pop, seg, mask, noise_pat, sigma):
    lives = []
    for m in pop:
        S = m.run_states(make_emb(m, seg, mask, noise_pat, sigma))
        preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
        hits = preds[:-1] == seg[1:]
        cum = np.cumsum(RENT * m.M + (~hits))
        life = len(hits) if cum[-1] < START_ENERGY \
            else int(np.searchsorted(cum, START_ENERGY)) + 1
        lives.append(life)
    return np.array(lives)


def evolve(world_ids, V, seed, seg_len, rho, sigma, decay):
    MIND.DECAY = decay
    rng = np.random.default_rng(seed)
    pop = [Mind(V, rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        start = int(rng.integers(0, len(world_ids) - seg_len - 1))
        seg = world_ids[start:start + seg_len]
        mask = (rng.random(len(seg)) < rho) if rho > 0 else None
        noise_pat = rng.normal(0, 1, (len(seg), EMBED)).astype(np.float32) \
            if sigma > 0 else None
        lives = eval_lives(pop, seg, mask, noise_pat, sigma)
        pop = reproduce(pop, lives, rng)
    return pop


def internals(world_ids, V, seed, seg_len, rho, sigma, decay):
    """Evolve one config, return memory-machinery internals of the best organism,
    measured on a CLEAN (rho=0,sigma=0) segment."""
    pop = evolve(world_ids, V, seed, seg_len, rho, sigma, decay)
    lives = eval_lives(pop, world_ids[seg_len:2 * seg_len], None, None, 0.0)
    bi = int(np.argmax(lives))
    mi = RM.measure_internals(pop[bi], world_ids, V, decay, np.random.default_rng(seed + 700))
    return dict(M=mi["M"], gain=mi["gain"], horizon=mi["horizon"])


def worlds():
    text_ids, textV, _ = RM.world_text()
    lr_ids, lrV, _ = RM.world_longrange(np.random.default_rng(999))
    return {"text": (text_ids, textV, SEG_TEXT), "longrange": (lr_ids, lrV, SEG_LR)}


def agg(world_ids, V, seg, rho, sigma, decay):
    rows = [internals(world_ids, V, s, seg, rho, sigma, decay) for s in SEEDS]
    return {k: float(np.mean([r[k] for r in rows])) for k in ("M", "gain", "horizon")}


# ---- multiprocessing driver: one cell per (experiment, world, param, seed) ----
WORLDS = {}
STACK_CONDS = {"none": (0.0, 1.0), "occ": (0.4, 1.0),
               "entropy": (0.0, 0.6), "occ+entropy": (0.4, 0.6)}


def cell(task):
    exp, wn, param, seed = task
    ids, V, seg = WORLDS[wn]
    if exp == "occ":
        rho, sig, dec = param, 0.0, 1.0
    elif exp == "noise":
        rho, sig, dec = 0.0, param, 1.0
    else:  # stack
        rho, dec = STACK_CONDS[param]; sig = 0.0
    return (exp, wn, str(param), seed, internals(ids, V, seed, seg, rho, sig, dec))


def main_all():
    global WORLDS
    WORLDS = worlds()                      # built ONCE; workers inherit via fork
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
    sigs = [0.0, 0.25, 0.5, 1.0, 2.0]
    tasks = []
    for wn in WORLDS:
        for s in SEEDS:
            tasks += [("occ", wn, r, s) for r in rhos]
            tasks += [("noise", wn, sg, s) for sg in sigs]
            tasks += [("stack", wn, c, s) for c in STACK_CONDS]
    nproc = min(18, len(tasks))
    print(f"running {len(tasks)} cells on {nproc} workers "
          f"(gens={GENS} seg_text={SEG_TEXT} seg_lr={SEG_LR} seeds={SEEDS})", flush=True)
    with Pool(nproc) as p:
        results = p.map(cell, tasks, chunksize=1)   # even load-balancing (cells vary 17-49s)

    buckets = {"occ": {}, "noise": {}, "stack": {}}
    for exp, wn, param, seed, mi in results:
        buckets[exp].setdefault(f"{wn}|{param}", []).append(mi)
    out = {}
    for exp, d in buckets.items():
        out[exp] = {k: {m: float(np.mean([r[m] for r in rows])) for m in ("M", "gain", "horizon")}
                    for k, rows in d.items()}
        json.dump(out[exp], open(os.path.join(OUT, f"board_{exp}.json"), "w"), indent=1)
        print(f"  wrote board_{exp}.json ({len(d)} configs)", flush=True)
    plot()


def run_occ():
    W = worlds(); out = {}
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for wn, (ids, V, seg) in W.items():
            out[f"{wn}|{rho}"] = agg(ids, V, seg, rho, 0.0, 1.0)
            print(f"[occ {wn:9} rho={rho}] {out[f'{wn}|{rho}']}", flush=True)
    json.dump(out, open(os.path.join(OUT, "board_occ.json"), "w"), indent=1)


def run_noise():
    W = worlds(); out = {}
    for sig in [0.0, 0.25, 0.5, 1.0, 2.0]:
        for wn, (ids, V, seg) in W.items():
            out[f"{wn}|{sig}"] = agg(ids, V, seg, 0.0, sig, 1.0)
            print(f"[noise {wn:9} sig={sig}] {out[f'{wn}|{sig}']}", flush=True)
    json.dump(out, open(os.path.join(OUT, "board_noise.json"), "w"), indent=1)


def run_stack():
    W = worlds(); out = {}
    conds = [("none", 0.0, 1.0), ("occ", 0.4, 1.0), ("entropy", 0.0, 0.6), ("occ+entropy", 0.4, 0.6)]
    for cname, rho, decay in conds:
        for wn, (ids, V, seg) in W.items():
            out[f"{wn}|{cname}"] = agg(ids, V, seg, rho, 0.0, decay)
            print(f"[stack {wn:9} {cname:12}] {out[f'{wn}|{cname}']}", flush=True)
    json.dump(out, open(os.path.join(OUT, "board_stack.json"), "w"), indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    occ = json.load(open(os.path.join(OUT, "board_occ.json")))
    noi = json.load(open(os.path.join(OUT, "board_noise.json")))
    stk = json.load(open(os.path.join(OUT, "board_stack.json")))

    # ---- PNG 1: observability sweeps (occlusion rho vs noise sigma) ----------
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
    sigs = [0.0, 0.25, 0.5, 1.0, 2.0]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for col, (data, xs, xlab, tag) in enumerate(
            [(occ, rhos, "occlusion rho (fraction hidden)", "OCCLUSION (missing)"),
             (noi, sigs, "noise sigma (rel. SNR)", "NOISE (corrupted)")]):
        for row, metric in enumerate(["gain", "horizon"]):
            for wn, color in [("text", "#999"), ("longrange", "#377eb8")]:
                ys = [data[f"{wn}|{x}"][metric] for x in xs]
                ax[row, col].plot(xs, ys, "o-", color=color, lw=2, label=wn)
            ax[row, col].set_xlabel(xlab)
            ax[row, col].set_ylabel("recurrent gain" if metric == "gain" else "memory horizon")
            ax[row, col].set_title(f"{tag} -> {metric}")
            ax[row, col].legend(); ax[row, col].grid(alpha=.3)
    fig.suptitle("Observability laws: does degrading perception build memory machinery?\n"
                 "(measured on CLEAN segments; long-range = recoverable structure, text = thin signal)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = os.path.join(OUT, "board_observability.png")
    plt.savefig(p1, dpi=115); print("saved", p1)

    # ---- PNG 2: occlusion x entropy compounding -----------------------------
    conds = ["none", "occ", "entropy", "occ+entropy"]
    colors = ["#bbb", "#377eb8", "#e41a1c", "#984ea3"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, metric in enumerate(["gain", "horizon"]):
        x = np.arange(2); w = 0.2
        for i, c in enumerate(conds):
            vals = [stk[f"{wn}|{c}"][metric] for wn in ["text", "longrange"]]
            ax[k].bar(x + (i - 1.5) * w, vals, w, label=c, color=colors[i])
        ax[k].set_xticks(x); ax[k].set_xticklabels(["text", "long-range"])
        ax[k].set_ylabel("recurrent gain" if metric == "gain" else "memory horizon")
        ax[k].set_title(f"occlusion x entropy -> {metric}")
        ax[k].legend()
    # super-additivity annotation for long-range gain
    g = {c: stk[f"longrange|{c}"]["gain"] for c in conds}
    add = (g["occ"] - g["none"]) + (g["entropy"] - g["none"])
    obs = g["occ+entropy"] - g["none"]
    verdict = "SUPER-additive" if obs > add * 1.15 else ("sub-additive" if obs < add * 0.85 else "~additive")
    fig.suptitle(f"Compounding of active-maintenance drivers (long-range gain): "
                 f"occ+entropy uplift {obs:.3f} vs sum-of-parts {add:.3f}  ->  {verdict}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(OUT, "board_compound.png")
    plt.savefig(p2, dpi=115); print("saved", p2)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    {"all": main_all, "occ": run_occ, "noise": run_noise,
     "stack": run_stack, "plot": plot}[mode]()
