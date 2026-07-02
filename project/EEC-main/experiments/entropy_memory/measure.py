"""Measure what the organism BECAME -- not accuracy. Compare conditions on
emergent structure: recurrent gain (active maintenance), memory horizon,
internal dimensionality, evolved memory size, survival."""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mind as MIND
from evolve import build_corpus

import os
HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    d = pickle.load(open(os.path.join(HERE, "best", f"mind_{tag}.pkl"), "rb"))
    g = MIND.Mind.__new__(MIND.Mind)
    g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out = d["genome"]
    g.M = d["M"]
    return g, d


def measure(tag, ids, vocab_size):
    g, d = load(tag)
    MIND.DECAY = d["decay"]                      # run it under its own law
    M = g.M
    Wr = g.W_rec[:M, :M]
    sr = float(np.max(np.abs(np.linalg.eigvals(Wr))))   # recurrent gain
    eff_gain = d["decay"] * sr                  # effective per-step retention

    rng = np.random.default_rng(123)
    start = int(rng.integers(0, len(ids) - 1600))
    seg = ids[start:start + 1500]
    S = g.run_states(g.E[seg])
    Sc = S - S.mean(0)
    ev = np.linalg.eigvalsh(Sc.T @ Sc / len(S))[::-1]
    eff_dim = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9)) + 1

    # memory horizon: how long a single flipped token keeps the state diverged
    hors = []
    for t0 in rng.integers(60, 1400, 16):
        s2 = seg.copy()
        s2[t0] = int((s2[t0] + 911) % vocab_size)
        S2 = g.run_states(g.E[s2])
        div = np.linalg.norm(S - S2, axis=1)
        peak = div[t0:t0 + 3].max()
        if peak < 1e-6:
            continue
        after = div[t0:]
        below = np.where(after < 0.1 * peak)[0]
        hors.append(int(below[0]) if len(below) else len(after))
    horizon = float(np.mean(hors)) if hors else 0.0

    lives = []
    for _ in range(8):
        st = int(rng.integers(0, len(ids) - 2100)); sg = ids[st:st + 2000]
        lives.append(MIND.live(g, sg, g.E[sg])[0])
    return dict(tag=tag, decay=d["decay"], M=M, gain=sr, eff_gain=eff_gain,
                eff_dim=eff_dim, horizon=horizon, lifespan=float(np.mean(lives)),
                gen=d["gen"])


def main():
    tags = sys.argv[1:] or ["ctrl", "ent"]
    ids, vocab, _ = build_corpus()
    V = len(vocab)
    res = [measure(t, ids, V) for t in tags]
    keys = ["decay", "M", "gain", "eff_gain", "eff_dim", "horizon", "lifespan"]
    labels = {"decay": "world decay", "M": "evolved memory M", "gain": "W_rec gain",
              "eff_gain": "effective gain (decay*gain)", "eff_dim": "state eff-dim",
              "horizon": "memory horizon (steps)", "lifespan": "survival (steps)"}
    print(f"\n{'metric':<32}" + "".join(f"{r['tag']:>12}" for r in res))
    for k in keys:
        print(f"{labels[k]:<32}" + "".join(f"{r[k]:>12.3f}" for r in res))

    # comparison bars for the emergent metrics
    show = ["M", "gain", "eff_gain", "eff_dim", "horizon", "lifespan"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, k in zip(axs.ravel(), show):
        vals = [r[k] for r in res]
        ax.bar([r["tag"] for r in res], vals, color=["#999", "#e41a1c"][:len(res)])
        ax.set_title(labels[k])
        if k == "eff_gain":
            ax.axhline(1.0, ls="--", color="k", lw=1)
            ax.text(0, 1.02, "maintenance threshold", fontsize=8)
    fig.suptitle("What the entropy law produced (emergent structure, not accuracy)")
    plt.tight_layout()
    out = os.path.join(HERE, "entropy_compare.png")
    plt.savefig(out, dpi=115)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
