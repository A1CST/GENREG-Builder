"""PARTIAL OBSERVABILITY (occlusion) law -- a genuinely new axis, not redundant
with lifespan/energy.

The world hides a fraction RHO of tokens from the organism's INPUT: on a masked
step it receives NO sensory embedding (sensory blackout) and must run on its
recurrent state alone. The TARGET it eats by anticipating is still the real next
token. Occlusion does not cap life or change the energy economy -- it degrades
OBSERVABILITY, so the only way to keep predicting through a blank is to have held
the relevant information in memory.

Prediction: occlusion makes memory INSTRUMENTAL -> evolved memory M, recurrent
gain (active maintenance), and memory horizon should GROW vs the fully-observed
control, and MORE in the long-range world (hidden tokens are recoverable from
periodic structure) than in thin text (less recoverable).

Measured by EMERGENT INTERNALS on CLEAN (unoccluded) segments -- did occlusion
build more memory machinery? -- per the paradigm (read state, not output).
Reuses run_matrix.measure_internals.
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
from mind import Mind, RENT, START_ENERGY, reproduce
from evolve import POP_SIZE

RM.SEG_TEXT = int(os.environ.get("EEC_SEGTEXT", "600"))
GENS = int(os.environ.get("EEC_MGENS", "150"))
HERE = os.path.dirname(os.path.abspath(__file__))


def occluded_emb(m, seg, mask):
    """Per-genome embedding of the segment with masked steps blanked to zero
    (sensory blackout: no input drive on those steps)."""
    emb = m.E[seg].copy()
    emb[mask] = 0.0
    return emb


def eval_lives(pop, seg, mask):
    """Survival-only fitness under occlusion. State runs on the occluded input;
    the organism is still graded on anticipating the REAL next token."""
    lives = []
    for m in pop:
        S = m.run_states(occluded_emb(m, seg, mask))
        preds = (S @ m.W_out[:m.M, :] + m.b_out).argmax(1)
        hits = preds[:-1] == seg[1:]
        cost = RENT * m.M + (~hits)
        cum = np.cumsum(cost)
        life = len(hits) if cum[-1] < START_ENERGY \
            else int(np.searchsorted(cum, START_ENERGY)) + 1
        lives.append(life)
    return np.array(lives)


def evolve(world_ids, vocab_size, seed, seg_len, rho):
    MIND.DECAY = 1.0
    rng = np.random.default_rng(seed)
    pop = [Mind(vocab_size, rng) for _ in range(POP_SIZE)]
    for gen in range(GENS):
        start = int(rng.integers(0, len(world_ids) - seg_len - 1))
        seg = world_ids[start:start + seg_len]
        # one occlusion pattern per generation, shared by all genomes (fair eval)
        mask = rng.random(len(seg)) < rho if rho > 0 else np.zeros(len(seg), bool)
        lives = eval_lives(pop, seg, mask)
        pop = reproduce(pop, lives, rng)
    return pop


def main():
    text_ids, textV, _ = RM.world_text()
    lr_ids, lrV, _ = RM.world_longrange(np.random.default_rng(999))
    worlds = [("text", text_ids, textV, RM.SEG_TEXT),
              ("longrange", lr_ids, lrV, RM.SEG_LR)]
    rhos = [0.0, 0.4]
    print(f"gens={GENS} rhos={rhos} | text V={textV} lr V={lrV}")
    res = {}
    for wname, ids, V, seg in worlds:
        for rho in rhos:
            Ms, gains, hors = [], [], []
            for seed in [0, 1]:
                pop = evolve(ids, V, seed, seg, rho)
                # internals measured on CLEAN segments (intrinsic memory machinery)
                lives = eval_lives(pop, ids[seg:seg + seg], np.zeros(seg, bool))
                bi = int(np.argmax(lives))
                mi = RM.measure_internals(pop[bi], ids, V, 1.0, np.random.default_rng(seed + 600))
                Ms.append(mi["M"]); gains.append(mi["gain"]); hors.append(mi["horizon"])
            res[(wname, rho)] = dict(M=np.mean(Ms), gain=np.mean(gains), hor=np.mean(hors))
            r = res[(wname, rho)]
            print(f"[{wname:9} rho={rho:.1f}] M={r['M']:.1f} gain={r['gain']:.3f} horizon={r['hor']:.2f}")

    print("\n===== INTERACTION (does occlusion grow memory machinery?) =====")
    for w in ["text", "longrange"]:
        off, on = res[(w, 0.0)], res[(w, 0.4)]
        print(f"  {w:9}: M {off['M']:.1f}->{on['M']:.1f}  gain {off['gain']:.3f}->{on['gain']:.3f}  "
              f"horizon {off['hor']:.2f}->{on['hor']:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(2); w = 0.35
    for k, key, title in [(0, "M", "evolved memory M"),
                          (1, "gain", "recurrent gain (active maintenance)"),
                          (2, "hor", "memory horizon")]:
        off = [res[(wn, 0.0)][key] for wn in ["text", "longrange"]]
        on = [res[(wn, 0.4)][key] for wn in ["text", "longrange"]]
        ax[k].bar(x - w/2, off, w, label="fully observed", color="#999")
        ax[k].bar(x + w/2, on, w, label="occluded (rho=0.4)", color="#377eb8")
        ax[k].set_xticks(x); ax[k].set_xticklabels(["text", "long-range"])
        ax[k].set_title(title); ax[k].legend()
    fig.suptitle("OCCLUSION: hiding tokens makes memory instrumental")
    plt.tight_layout()
    out = os.path.join(HERE, "occlusion.png")
    plt.savefig(out, dpi=115); print("\nsaved", out)


if __name__ == "__main__":
    main()
