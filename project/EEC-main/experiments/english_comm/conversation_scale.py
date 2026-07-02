"""CONVERSATION SCALE -- zero-shot conversation: reply correctly to prompts never seen in training.

Compose the transform-don't-copy mechanism (conversation.py) with composition (vocabulary for free).
A prompt is (a,b); the appropriate reply is per-slot r(a,b)=(g0(a),g1(b)), g_s a derangement (reply!=
prompt). The organism learns per-slot reply machinery. Train on a SUBSET of prompt combinations; measure
reply accuracy + conversation length on HELD-OUT prompt combinations it never saw. If held-out ~ seen,
the organism converses with novel inputs zero-shot -- it generalises the conversation, not memorises it.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S = 2; N = 64; BUDGET = int(os.environ.get("CS_BUDGET", "1500")); SEEDS = 4; TMAX = 50; TRAIN_FRAC = 0.7
FULL_BONUS = float(os.environ.get("CS_FULLBONUS", "0"))   # reward a fully-correct reply (whole turn survives)
VS = [int(x) for x in os.environ.get("CS_VS", "6,10,16").split(",")]
LOG = open(os.path.join(HERE, "conversation_scale_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def derange(V, rng):
    while True:
        p = rng.permutation(V)
        if np.all(p != np.arange(V)): return p


def conv_len(resp, g, prompts):
    Ng = resp.shape[0]; alive = np.ones(Ng, bool); length = np.zeros(Ng, int)
    for t in range(prompts.shape[1]):
        a, b = prompts[:, t, 0], prompts[:, t, 1]
        ok = (resp[np.arange(Ng), 0, a] == g[0, a]) & (resp[np.arange(Ng), 1, b] == g[1, b])
        length += (alive & ok); alive &= ok
    return length


def run(V, seed):
    rng = np.random.default_rng(seed)
    g = np.stack([derange(V, rng) for _ in range(S)])
    combos = np.array([(a, b) for a in range(V) for b in range(V)])
    perm = rng.permutation(len(combos)); ntr = int(TRAIN_FRAC * len(combos))
    seen, held = combos[perm[:ntr]], combos[perm[ntr:]]
    R = rng.normal(0, 0.3, (N, S, V, V))
    for t in range(BUDGET):
        resp = R.argmax(3)
        c0 = resp[:, 0, seen[:, 0]] == g[0, seen[:, 0]][None, :]
        c1 = resp[:, 1, seen[:, 1]] == g[1, seen[:, 1]][None, :]
        en = (c0.sum(1) + c1.sum(1) + FULL_BONUS * (c0 & c1).sum(1)).astype(float)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            m = rng.random(R[pa].shape) < 0.5
            R[w] = np.where(m, R[pa], R[pb]) + rng.normal(0, 0.22 * 0.6, R[pa].shape)
    resp = R.argmax(3)
    def acc(combos_):
        ok = (resp[:, 0, combos_[:, 0]] == g[0, combos_[:, 0]][None, :]) & \
             (resp[:, 1, combos_[:, 1]] == g[1, combos_[:, 1]][None, :])
        return float(ok.mean())
    # conversation length on held-out prompts
    hp = held[rng.integers(len(held), size=(N, TMAX))]
    clen = conv_len(resp, g, hp).mean()
    return acc(seen), acc(held), float(clen)


if __name__ == "__main__":
    out(f"CONVERSATION SCALE: zero-shot reply to unseen prompts. S={S}, pop {N}, budget {BUDGET}, "
        f"{SEEDS} seeds, train {int(TRAIN_FRAC*100)}% of V^2 prompts.")
    out(f"{'V':>4} {'prompts V^2':>12} {'chance':>8} | {'SEEN reply acc':>15} | {'HELD-OUT reply acc':>19} | {'held conv len':>14}")
    out("=" * 86)
    for V in VS:
        r = np.array([run(V, s) for s in range(SEEDS)])
        out(f"{V:>4} {V*V:>12} {1/V:>8.3f} | {ms(r[:,0]):>15} | {ms(r[:,1]):>19} | {ms(r[:,2]):>14}")
    out("=" * 86)
    out("READING: HELD-OUT reply accuracy ~ SEEN => the organism replies correctly to prompt combinations")
    out("it NEVER trained on -> zero-shot conversation. It generalises the chat, doesn't memorise it.")
    out("done"); LOG.close()
