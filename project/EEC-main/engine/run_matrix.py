"""Cross-comparison sweep. Three axes, one orchestrator, internals not accuracy:
  (1) seeds      -- is the entropy->maintenance shift real across seeds?
  (2) long-range -- when memory PAYS, does capacity/maintenance grow?
  (3) scarcity   -- does a shared-food law turn a monoculture into an ecosystem?

All measured by what EMERGED inside the organism: recurrent gain (active
maintenance), memory horizon, evolved memory M, population diversity.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mind as MIND
from mind import Mind, RENT, START_ENERGY, MAX_M, reproduce
from evolve import build_corpus, POP_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
GENS = int(os.environ.get("EEC_MGENS", "220"))
SEG_TEXT = 1500
SEG_LR = 600
CAP_MULT = 3.0          # shared-food capacity = CAP_MULT * natural token count


# ---------- worlds ----------------------------------------------------------
def world_text():
    ids, vocab, _ = build_corpus()
    return ids, len(vocab), "text"


def world_longrange(rng, n=240000, alph=6, blk=24, noise=0.05):
    block = rng.integers(0, alph, blk)
    stream = np.tile(block, n // blk + 1)[:n].copy()
    m = rng.random(n) < noise
    stream[m] = rng.integers(0, alph, m.sum())
    return stream.astype(np.int32), alph, "longrange"


# ---------- evaluation (shared segment; optional shared food) ---------------
def population_eval(pop, seg, scarcity, vocab_size):
    """Returns lives (array), preds_list (per organism, len T-1)."""
    T = len(seg)
    hits_list, preds_list = [], []
    for m in pop:
        S = m.run_states(m.E[seg])
        preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
        hits = preds[:-1] == seg[1:]
        hits_list.append(hits); preds_list.append(preds[:-1])
    nxt = seg[1:]
    if scarcity:
        natural = np.bincount(nxt, minlength=vocab_size).astype(np.float64)
        demand = np.zeros(vocab_size)
        for hits, preds in zip(hits_list, preds_list):
            np.add.at(demand, preds[hits], 1.0)
        cap = CAP_MULT * np.maximum(natural, 1.0)
        scale = np.minimum(1.0, cap / np.maximum(demand, 1.0))   # per token type
    lives = []
    for hits, preds, m in zip(hits_list, preds_list, pop):
        rent = RENT * m.M
        hit_cost = np.zeros(len(hits))
        if scarcity:
            hit_cost = (1.0 - scale[preds]) * hits          # contested hits still drain
        cost = rent + (~hits) + hit_cost
        cum = np.cumsum(cost)
        life = len(hits) if cum[-1] < START_ENERGY else int(np.searchsorted(cum, START_ENERGY)) + 1
        lives.append(life)
    return np.array(lives), preds_list


def evolve_config(world_ids, vocab_size, decay, scarcity, seed, seg_len):
    MIND.DECAY = decay
    rng = np.random.default_rng(seed)
    pop = [Mind(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(GENS):
        start = int(rng.integers(0, len(world_ids) - seg_len - 1))
        seg = world_ids[start:start + seg_len]
        lives, _ = population_eval(pop, seg, scarcity, vocab_size)
        pop = reproduce(pop, lives, rng)
    return pop


# ---------- emergent measurements ------------------------------------------
def measure_internals(m, world_ids, vocab_size, decay, rng):
    MIND.DECAY = decay
    Wr = m.W_rec[:m.M, :m.M]
    gain = float(np.max(np.abs(np.linalg.eigvals(Wr)))) if m.M else 0.0
    start = int(rng.integers(0, len(world_ids) - 1100))
    seg = world_ids[start:start + 1000]
    S = m.run_states(m.E[seg])
    hors = []
    for t0 in rng.integers(50, 900, 16):
        s2 = seg.copy(); s2[t0] = int((s2[t0] + 3) % vocab_size)
        S2 = m.run_states(m.E[s2])
        div = np.linalg.norm(S - S2, axis=1)
        peak = div[t0:t0 + 3].max()
        if peak < 1e-6:
            continue
        after = div[t0:]; below = np.where(after < 0.1 * peak)[0]
        hors.append(int(below[0]) if len(below) else len(after))
    horizon = float(np.mean(hors)) if hors else 0.0
    return dict(M=m.M, gain=gain, eff_gain=decay * gain, horizon=horizon)


def measure_diversity(pop, world_ids, seg_len, rng):
    start = int(rng.integers(0, len(world_ids) - seg_len - 1))
    seg = world_ids[start:start + seg_len]
    profs = []
    for m in pop:
        S = m.run_states(m.E[seg])
        preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
        profs.append(preds)
    profs = np.array(profs)
    dominant = [np.bincount(p).argmax() for p in profs]
    uniq_dominant = len(set(dominant))
    # mean pairwise disagreement
    dis = []
    for i in range(0, len(profs), 3):
        for j in range(i + 1, len(profs), 3):
            dis.append(float((profs[i] != profs[j]).mean()))
    return dict(uniq_dominant=uniq_dominant, disagreement=float(np.mean(dis)))


# ---------- the sweep -------------------------------------------------------
def main():
    rng0 = np.random.default_rng(999)
    text_ids, textV, _ = world_text()
    lr_ids, lrV, _ = world_longrange(rng0)
    print(f"text V={textV}, longrange V={lrV}, gens={GENS}")

    results = {"seeds": [], "lr": [], "scar": []}

    # (1) seeds: text world, decay on/off, several seeds
    for seed in [0, 1, 2]:
        for decay in [1.0, 0.6]:
            pop = evolve_config(text_ids, textV, decay, False, seed, SEG_TEXT)
            lives, _ = population_eval(pop, text_ids[1000:1000 + SEG_TEXT], False, textV)
            bi = int(np.argmax(lives))
            mi = measure_internals(pop[bi], text_ids, textV, decay, np.random.default_rng(seed + 50))
            mi.update(seed=seed, decay=decay)
            results["seeds"].append(mi)
            print(f"  [seeds] seed{seed} decay{decay}: gain {mi['gain']:.3f} horizon {mi['horizon']:.2f} M {mi['M']}")

    # (2) long-range world, decay on/off
    for seed in [0, 1]:
        for decay in [1.0, 0.6]:
            pop = evolve_config(lr_ids, lrV, decay, False, seed, SEG_LR)
            lives, _ = population_eval(pop, lr_ids[600:600 + SEG_LR], False, lrV)
            bi = int(np.argmax(lives))
            mi = measure_internals(pop[bi], lr_ids, lrV, decay, np.random.default_rng(seed + 70))
            mi.update(seed=seed, decay=decay)
            results["lr"].append(mi)
            print(f"  [longrange] seed{seed} decay{decay}: M {mi['M']} gain {mi['gain']:.3f} eff_gain {mi['eff_gain']:.3f} horizon {mi['horizon']:.2f}")

    # (3) scarcity: text world, shared food on/off
    for seed in [0, 1]:
        for scar in [False, True]:
            pop = evolve_config(text_ids, textV, 1.0, scar, seed, SEG_TEXT)
            dv = measure_diversity(pop, text_ids, SEG_TEXT, np.random.default_rng(seed + 90))
            dv.update(seed=seed, scarcity=scar)
            results["scar"].append(dv)
            print(f"  [scarcity] seed{seed} scar{scar}: uniq_dominant {dv['uniq_dominant']} disagreement {dv['disagreement']:.3f}")

    _plot(results)
    _summary(results)


def _grp(rows, key, val):
    return [r[val] for r in rows if r[key]]


def _summary(R):
    print("\n===== CROSS-COMPARISON SUMMARY (emergent internals) =====")
    s_ctrl = [r for r in R["seeds"] if r["decay"] == 1.0]
    s_ent = [r for r in R["seeds"] if r["decay"] == 0.6]
    def mean(rows, k): return np.mean([r[k] for r in rows])
    print(f"(1) ENTROPY across seeds  gain: ctrl {mean(s_ctrl,'gain'):.3f} -> ent {mean(s_ent,'gain'):.3f} | "
          f"horizon: ctrl {mean(s_ctrl,'horizon'):.2f} -> ent {mean(s_ent,'horizon'):.2f}")
    l_ctrl = [r for r in R["lr"] if r["decay"] == 1.0]
    l_ent = [r for r in R["lr"] if r["decay"] == 0.6]
    print(f"(2) LONG-RANGE world      M: ctrl {mean(l_ctrl,'M'):.1f}/ent {mean(l_ent,'M'):.1f} (text M~3) | "
          f"gain ctrl {mean(l_ctrl,'gain'):.3f}/ent {mean(l_ent,'gain'):.3f} | "
          f"horizon ctrl {mean(l_ctrl,'horizon'):.2f}/ent {mean(l_ent,'horizon'):.2f}")
    c_off = [r for r in R["scar"] if not r["scarcity"]]
    c_on = [r for r in R["scar"] if r["scarcity"]]
    print(f"(3) SCARCITY              uniq-dominant: off {mean(c_off,'uniq_dominant'):.1f} -> on {mean(c_on,'uniq_dominant'):.1f} | "
          f"disagreement: off {mean(c_off,'disagreement'):.3f} -> on {mean(c_on,'disagreement'):.3f}")


def _plot(R):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    # (1)
    for r in R["seeds"]:
        x = 0 if r["decay"] == 1.0 else 1
        ax[0].scatter(x + np.random.uniform(-.05, .05), r["gain"], c="#999" if x == 0 else "#e41a1c", s=60)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["control", "entropy"])
    ax[0].set_title("(1) recurrent gain across seeds\n(entropy -> more active maintenance?)")
    ax[0].set_ylabel("W_rec gain")
    # (2)
    lr = R["lr"]
    labels = [f"d{r['decay']}\ns{r['seed']}" for r in lr]
    ax[1].bar(range(len(lr)), [r["M"] for r in lr], color=["#377eb8" if r["decay"] == 1 else "#e41a1c" for r in lr])
    ax[1].axhline(3, ls="--", color="k", lw=1); ax[1].text(0, 3.1, "text-world M=3", fontsize=8)
    ax[1].set_xticks(range(len(lr))); ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_title("(2) evolved memory M in long-range world\n(does memory grow when it pays?)")
    ax[1].set_ylabel("evolved M")
    # (3)
    off = [r["uniq_dominant"] for r in R["scar"] if not r["scarcity"]]
    on = [r["uniq_dominant"] for r in R["scar"] if r["scarcity"]]
    ax[2].bar([0, 1], [np.mean(off), np.mean(on)], color=["#999", "#4daf4a"])
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["no scarcity", "scarcity"])
    ax[2].set_title("(3) population diversity\n(distinct dominant behaviors)")
    ax[2].set_ylabel("unique dominant outputs / 30")
    plt.tight_layout()
    out = os.path.join(HERE, "..", "experiments", "entropy_memory", "matrix_compare.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=115); print("\nsaved", out)


if __name__ == "__main__":
    main()
