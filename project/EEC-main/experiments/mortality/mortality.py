"""MORTALITY law: a hard maximum age. An organism dies at MAX_AGE even with
energy to spare -- lifespan is decoupled from fitness at the top end.

Paradigm: this is a law of existence (finite life / forced generational
turnover), NOT a reward. We never grade by accuracy.

Prediction: without mortality, the longest-surviving genome is preferentially
cloned -> the population monocultures around one reflex. With mortality, every
*fit* organism dies at the same age and TIES at the top of the lifespan ranking,
so no single genome is preferentially bred; selection only culls the genuinely
bad (who die sub-cap). Expected emergent effect: HIGHER population diversity
without a collapse of the competence floor -- and more so where distinct niches
are reachable (long-range world) than in the text reflex-monoculture.

Measured by emergent population structure (distinct dominant behaviours, pairwise
disagreement) plus a floor check (cap-hit rate, mean sub-cap death fraction).
"""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_matrix as RM
from mind import Mind, RENT, START_ENERGY, reproduce
from evolve import POP_SIZE

RM.GENS = int(os.environ.get("EEC_MGENS", "220"))
GENS = RM.GENS
# text SEG drives cost (recurrent python loop ~linear in seg). The population-
# structure signal we measure does not need long segments, so allow shrinking it.
RM.SEG_TEXT = int(os.environ.get("EEC_SEGTEXT", str(RM.SEG_TEXT)))
HERE = os.path.dirname(os.path.abspath(__file__))
AGE_FRAC = float(os.environ.get("EEC_AGEFRAC", "0.40"))   # MAX_AGE = AGE_FRAC * seg_len


def eval_lives(pop, seg, max_age):
    """Survival-only fitness with an optional hard age cap.
    Returns lives (capped), and the per-organism uncapped survival point."""
    lives, uncapped = [], []
    for m in pop:
        S = m.run_states(m.E[seg])
        preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
        hits = preds[:-1] == seg[1:]
        cost = RENT * m.M + (~hits)
        cum = np.cumsum(cost)
        life = len(hits) if cum[-1] < START_ENERGY \
            else int(np.searchsorted(cum, START_ENERGY)) + 1
        uncapped.append(life)
        lives.append(min(life, max_age) if max_age else life)
    return np.array(lives), np.array(uncapped)


def evolve_mortal(world_ids, vocab_size, seed, seg_len, mortal):
    RM.MIND.DECAY = 1.0
    rng = np.random.default_rng(seed)
    pop = [Mind(vocab_size, rng) for _ in range(POP_SIZE)]
    max_age = int(AGE_FRAC * seg_len) if mortal else 0
    cap_hits = []
    for gen in range(GENS):
        start = int(rng.integers(0, len(world_ids) - seg_len - 1))
        seg = world_ids[start:start + seg_len]
        lives, uncapped = eval_lives(pop, seg, max_age)
        if mortal:
            cap_hits.append(float((uncapped >= max_age).mean()))
        pop = reproduce(pop, lives, rng)
    cap_rate = float(np.mean(cap_hits[-30:])) if cap_hits else 0.0
    return pop, max_age, cap_rate


def diversity(pop, world_ids, seg_len, rng):
    start = int(rng.integers(0, len(world_ids) - seg_len - 1))
    seg = world_ids[start:start + seg_len]
    profs = np.array([(m.run_states(m.E[seg]) @ m.W_out[:m.M, :] + m.b_out).argmax(1)
                      for m in pop])
    dom = [int(np.bincount(p).argmax()) for p in profs]
    dis = [float((profs[i] != profs[j]).mean())
           for i in range(0, len(profs), 3) for j in range(i + 1, len(profs), 3)]
    return len(set(dom)), float(np.mean(dis)), sorted(set(dom))


def floor_check(pop, world_ids, seg_len, max_age, rng):
    """Did the competence floor hold? Fraction of the pop that would die BEFORE
    the cap on a fresh held-out segment (these are the genuinely-bad)."""
    start = int(rng.integers(0, len(world_ids) - seg_len - 1))
    seg = world_ids[start:start + seg_len]
    _, uncapped = eval_lives(pop, seg, 0)
    ref = max_age if max_age else int(AGE_FRAC * seg_len)
    return float((uncapped < ref).mean()), float(np.median(uncapped))


def main():
    text_ids, textV, _ = RM.world_text()
    lr_ids, lrV, _ = RM.world_longrange(np.random.default_rng(999))
    worlds = [("text", text_ids, textV, RM.SEG_TEXT),
              ("longrange", lr_ids, lrV, RM.SEG_LR)]
    print(f"gens={GENS} age_frac={AGE_FRAC} | text V={textV} lr V={lrV}")
    res = {}
    for wname, ids, V, seg in worlds:
        for mortal in [False, True]:
            uds, diss, frac_bad, med_life, caps = [], [], [], [], []
            for seed in [0, 1]:
                pop, max_age, cap_rate = evolve_mortal(ids, V, seed, seg, mortal)
                u, d, dom = diversity(pop, ids, seg, np.random.default_rng(seed + 300))
                fb, ml = floor_check(pop, ids, seg, max_age, np.random.default_rng(seed + 400))
                uds.append(u); diss.append(d); frac_bad.append(fb)
                med_life.append(ml); caps.append(cap_rate)
            res[(wname, mortal)] = dict(ud=np.mean(uds), dis=np.mean(diss),
                                        bad=np.mean(frac_bad), med=np.mean(med_life),
                                        cap=np.mean(caps), dom0=dom)
            r = res[(wname, mortal)]
            print(f"[{wname:9} mortal={str(mortal):5}] uniq_dom={r['ud']:.1f} "
                  f"disagree={r['dis']:.3f} | cap_hit={r['cap']:.2f} "
                  f"frac_sub_cap={r['bad']:.2f} med_life={r['med']:.0f}")

    print("\n===== INTERACTION (does mortality diversify without breaking the floor?) =====")
    for w in ["text", "longrange"]:
        off, on = res[(w, False)], res[(w, True)]
        print(f"  {w:9}: uniq-dom {off['ud']:.1f} -> {on['ud']:.1f}   "
              f"disagree {off['dis']:.3f} -> {on['dis']:.3f}   "
              f"frac-sub-cap floor {off['bad']:.2f} -> {on['bad']:.2f} "
              f"(cap-hit {on['cap']:.2f})")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(2); w = 0.35
    for k, key, title in [(0, "ud", "distinct dominant behaviours"),
                          (1, "dis", "mean pairwise disagreement")]:
        off = [res[(wn, False)][key] for wn in ["text", "longrange"]]
        on = [res[(wn, True)][key] for wn in ["text", "longrange"]]
        ax[k].bar(x - w/2, off, w, label="no mortality", color="#999")
        ax[k].bar(x + w/2, on, w, label="mortality", color="#984ea3")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(["text\n(reflex monoculture)", "long-range\n(niches reachable)"])
        ax[k].set_title(title); ax[k].legend()
    fig.suptitle("MORTALITY: does a hard age cap flatten selection -> more diversity?")
    plt.tight_layout()
    out = os.path.join(HERE, "mortality_interaction.png")
    plt.savefig(out, dpi=115); print("\nsaved", out)


if __name__ == "__main__":
    main()
