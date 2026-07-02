"""REPRODUCTION COST law (the coupled sibling of mortality).

Mortality-as-age-cap is degenerate in this substrate: lifespan IS already the
fitness, so capping it just truncates the existing energy selection (below the
life-band it ties everyone -> neutral drift; above it never binds). A finite
life only MATTERS if living buys something you must spend before you die.

So: make energy a BANK, and make children cost energy.
  - Each step: +FOOD if the organism predicted the next token (food = anticipating
    the world), minus metabolic rent RENT*M. Energy accumulates (surplus) or
    drains. Starve (energy<=0) -> dead, no offspring.
  - At the end of life, offspring = floor(surplus / CHILD_COST). Prediction skill
    is converted into FERTILITY. We never grade by accuracy; we count children.
  - Carrying capacity POP_SIZE: if the offspring pool overflows, sample POP_SIZE
    weighted by fertility; if it underflows, pad from the pool.

Control = the original pure-lifespan selection (steady-state reproduce()).

Emergent measurements (not accuracy): fertility distribution (mean/max/Gini ->
is it a monoculture-of-clones or an ecosystem of lineages?), evolved memory M,
recurrent gain, and a held-out competence proxy (net surplus rate).
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
import mind as MIND
from mind import Mind, RENT, reproduce
from evolve import POP_SIZE

RM.SEG_TEXT = int(os.environ.get("EEC_SEGTEXT", "600"))
GENS = int(os.environ.get("EEC_MGENS", "180"))
HERE = os.path.dirname(os.path.abspath(__file__))
E0 = float(os.environ.get("EEC_E0", "5.0"))          # starting bank (small: must earn)
FOOD = float(os.environ.get("EEC_FOOD", "1.0"))      # energy per correct anticipation
CHILD_COST = float(os.environ.get("EEC_CHILD", "10.0"))


def life_and_surplus(m, seg):
    """Returns (survived_bool, surplus_energy). Energy banks FOOD per hit, pays
    rent every step; death if it ever hits <=0."""
    S = m.run_states(m.E[seg])
    preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
    hits = preds[:-1] == seg[1:]
    rent = RENT * m.M
    gain = hits * FOOD - rent                          # per-step net energy
    bank = E0 + np.cumsum(gain)
    if bank.min() <= 0.0:
        return False, 0.0                              # starved at some point
    return True, float(bank[-1] - E0)                  # surplus earned over the life


def fertility_reproduce(pop, surplus, survived, rng):
    """Endogenous, energy-funded reproduction under carrying capacity POP_SIZE."""
    offspring = np.where(survived, np.floor(surplus / CHILD_COST).astype(int), 0)
    parents = [(pop[i], int(offspring[i])) for i in range(len(pop)) if offspring[i] > 0]
    if not parents:                                    # total reproductive failure
        return [Mind_clone_mut(pop[int(np.argmax(surplus))], rng) for _ in range(POP_SIZE)], offspring
    pool = []
    for par, n in parents:
        for _ in range(n):
            pool.append((par, 1))
    if len(pool) >= POP_SIZE:
        idx = rng.choice(len(pool), POP_SIZE, replace=False)
    else:
        idx = np.concatenate([np.arange(len(pool)),
                              rng.choice(len(pool), POP_SIZE - len(pool), replace=True)])
    new = [Mind_clone_mut(pool[i][0], rng) for i in idx]
    return new, offspring


def Mind_clone_mut(parent, rng):
    c = parent.copy(); c.mutate(rng); return c


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def evolve(world_ids, vocab_size, seed, seg_len, repro_cost):
    MIND.DECAY = 1.0
    rng = np.random.default_rng(seed)
    pop = [Mind(vocab_size, rng) for _ in range(POP_SIZE)]
    last_off = np.zeros(POP_SIZE)
    for gen in range(GENS):
        start = int(rng.integers(0, len(world_ids) - seg_len - 1))
        seg = world_ids[start:start + seg_len]
        if repro_cost:
            sv, sp = [], []
            for m in pop:
                s, e = life_and_surplus(m, seg)
                sv.append(s); sp.append(e)
            pop, last_off = fertility_reproduce(pop, np.array(sp), np.array(sv), rng)
        else:
            # control: pure-lifespan steady-state selection (original law)
            lives = []
            for m in pop:
                S = m.run_states(m.E[seg])
                preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
                hits = preds[:-1] == seg[1:]
                cost = RENT * m.M + (~hits)
                cum = np.cumsum(cost)
                from mind import START_ENERGY
                life = len(hits) if cum[-1] < START_ENERGY \
                    else int(np.searchsorted(cum, START_ENERGY)) + 1
                lives.append(life)
            pop = reproduce(pop, np.array(lives), rng)
    return pop, last_off


def measure(pop, world_ids, seg_len, rng):
    start = int(rng.integers(0, len(world_ids) - seg_len - 1))
    seg = world_ids[start:start + seg_len]
    profs, surplus, Ms, gains = [], [], [], []
    for m in pop:
        S = m.run_states(m.E[seg])
        profs.append((S @ m.W_out[:m.M, :] + m.b_out).argmax(1))
        _, e = life_and_surplus(m, seg)
        surplus.append(e); Ms.append(m.M)
        Wr = m.W_rec[:m.M, :m.M]
        gains.append(float(np.max(np.abs(np.linalg.eigvals(Wr)))) if m.M else 0.0)
    profs = np.array(profs)
    dom = [int(np.bincount(p).argmax()) for p in profs]
    return dict(uniq_dom=len(set(dom)), surplus_med=float(np.median(surplus)),
                surplus_max=float(np.max(surplus)), M=float(np.mean(Ms)),
                gain=float(np.mean(gains)))


def main():
    text_ids, textV, _ = RM.world_text()
    lr_ids, lrV, _ = RM.world_longrange(np.random.default_rng(999))
    worlds = [("text", text_ids, textV, RM.SEG_TEXT),
              ("longrange", lr_ids, lrV, RM.SEG_LR)]
    print(f"gens={GENS} E0={E0} FOOD={FOOD} CHILD_COST={CHILD_COST} | text V={textV} lr V={lrV}")
    res = {}
    for wname, ids, V, seg in worlds:
        for rc in [False, True]:
            ginis, ms, gains, uds, smed = [], [], [], [], []
            for seed in [0, 1]:
                pop, off = evolve(ids, V, seed, seg, rc)
                mm = measure(pop, ids, seg, np.random.default_rng(seed + 500))
                ginis.append(gini(off) if rc else 0.0)
                ms.append(mm["M"]); gains.append(mm["gain"])
                uds.append(mm["uniq_dom"]); smed.append(mm["surplus_med"])
            res[(wname, rc)] = dict(gini=np.mean(ginis), M=np.mean(ms), gain=np.mean(gains),
                                    ud=np.mean(uds), smed=np.mean(smed))
            r = res[(wname, rc)]
            print(f"[{wname:9} repro_cost={str(rc):5}] uniq_dom={r['ud']:.1f} "
                  f"M={r['M']:.1f} gain={r['gain']:.3f} surplus_med={r['smed']:.1f} "
                  f"fert_gini={r['gini']:.3f}")

    print("\n===== INTERACTION (does fertility-funded reproduction change what evolves?) =====")
    for w in ["text", "longrange"]:
        off, on = res[(w, False)], res[(w, True)]
        print(f"  {w:9}: M {off['M']:.1f}->{on['M']:.1f}  gain {off['gain']:.3f}->{on['gain']:.3f}  "
              f"uniq_dom {off['ud']:.1f}->{on['ud']:.1f}  surplus_med {off['smed']:.1f}->{on['smed']:.1f}  "
              f"fert_gini(on)={on['gini']:.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(2); w = 0.35
    for k, key, title in [(0, "M", "evolved memory M"),
                          (1, "gain", "recurrent gain (active maintenance)"),
                          (2, "smed", "held-out surplus (competence)")]:
        off = [res[(wn, False)][key] for wn in ["text", "longrange"]]
        on = [res[(wn, True)][key] for wn in ["text", "longrange"]]
        ax[k].bar(x - w/2, off, w, label="lifespan only", color="#999")
        ax[k].bar(x + w/2, on, w, label="reproduction cost", color="#ff7f00")
        ax[k].set_xticks(x); ax[k].set_xticklabels(["text", "long-range"])
        ax[k].set_title(title); ax[k].legend()
    fig.suptitle("REPRODUCTION COST: prediction -> energy surplus -> fertility")
    plt.tight_layout()
    out = os.path.join(HERE, "reproduce_cost.png")
    plt.savefig(out, dpi=115); print("\nsaved", out)


if __name__ == "__main__":
    main()
