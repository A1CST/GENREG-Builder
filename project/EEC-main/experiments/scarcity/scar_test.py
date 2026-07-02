"""Scarcity x world (2x2). Prediction: scarcity diversifies the population ONLY
where distinct niches are REACHABLE (long-range world, organisms can specialize)
-- not in the text world where they're reflex monocultures with nowhere to go.
Measured by emergent population structure, not accuracy."""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_matrix as RM
RM.GENS = int(os.environ.get("EEC_MGENS", "250"))
from run_matrix import evolve_config, world_text, world_longrange

HERE = os.path.dirname(os.path.abspath(__file__))


def diversity(pop, world_ids, seg_len, rng):
    start = int(rng.integers(0, len(world_ids) - seg_len - 1))
    seg = world_ids[start:start + seg_len]
    profs = []
    for m in pop:
        S = m.run_states(m.E[seg])
        profs.append((S @ m.W_out[:m.M, :] + m.b_out).argmax(1))
    profs = np.array(profs)
    dom = [int(np.bincount(p).argmax()) for p in profs]
    dis = []
    for i in range(0, len(profs), 3):
        for j in range(i + 1, len(profs), 3):
            dis.append(float((profs[i] != profs[j]).mean()))
    return len(set(dom)), float(np.mean(dis)), sorted(set(dom))


def main():
    text_ids, textV, _ = world_text()
    lr_ids, lrV, _ = world_longrange(np.random.default_rng(999))
    worlds = [("text", text_ids, textV, RM.SEG_TEXT),
              ("longrange", lr_ids, lrV, RM.SEG_LR)]
    res = {}
    for wname, ids, V, seg in worlds:
        for scar in [False, True]:
            uds, diss, doms = [], [], []
            for seed in [0, 1]:
                pop = evolve_config(ids, V, 1.0, scar, seed, seg)
                u, d, dom = diversity(pop, ids, seg, np.random.default_rng(seed + 200))
                uds.append(u); diss.append(d); doms.append(dom)
            res[(wname, scar)] = (np.mean(uds), np.mean(diss), doms)
            print(f"[{wname:9} scarcity={str(scar):5}] uniq_dominant={np.mean(uds):.1f} "
                  f"disagreement={np.mean(diss):.3f}  niches(seed0)={doms[0]}")

    print("\n===== INTERACTION (does scarcity diversify where niches are reachable?) =====")
    for w in ["text", "longrange"]:
        off = res[(w, False)][0]; on = res[(w, True)][0]
        do = res[(w, False)][1]; dn = res[(w, True)][1]
        print(f"  {w:9}: uniq-dominant {off:.1f} -> {on:.1f}   disagreement {do:.3f} -> {dn:.3f}")

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(2); w = 0.35
    for k, metric, title in [(0, 0, "distinct dominant behaviors"), (1, 1, "mean pairwise disagreement")]:
        off = [res[(wn, False)][metric] for wn in ["text", "longrange"]]
        on = [res[(wn, True)][metric] for wn in ["text", "longrange"]]
        ax[k].bar(x - w/2, off, w, label="no scarcity", color="#999")
        ax[k].bar(x + w/2, on, w, label="scarcity", color="#4daf4a")
        ax[k].set_xticks(x); ax[k].set_xticklabels(["text\n(no reachable niches)", "long-range\n(niches reachable)"])
        ax[k].set_title(title); ax[k].legend()
    fig.suptitle("Scarcity diversifies ONLY where the alternative is reachable")
    plt.tight_layout()
    out = os.path.join(HERE, "scarcity_interaction.png")
    plt.savefig(out, dpi=115); print("\nsaved", out)


if __name__ == "__main__":
    main()
