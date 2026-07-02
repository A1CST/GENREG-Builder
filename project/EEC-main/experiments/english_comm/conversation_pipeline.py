"""CONVERSATION PIPELINE -- full hear -> understand -> reply, removing the oracle prompt.

So far the organism was handed the prompt id. Real conversation: it must COMPREHEND the incoming
message itself, then RESPOND. Two stacked components: comprehension C (input signal -> understood
value, per slot) and response R (understood value -> reply phrase). We test JOINT training (one fitness
on the final reply) vs SERIES (train comprehension first, freeze, then response) -- the 'series not
parallel' caution -- and generalisation to held-out prompts.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S, V, WORDS_PER = 2, 10, 2; N = 100; BUDGET = 2500; SEEDS = 4; TRAIN_FRAC = 0.7
RV = V * WORDS_PER
LOG = open(os.path.join(HERE, "conversation_pipeline_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def evolve_step(G, en, rng, MUT):
    o = np.argsort(en); worst = o[:int(0.25 * len(en))]; top = o[len(en) - max(2, len(en) // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        out_shape = G[pa].shape; m = rng.random(out_shape) < 0.5; child = np.where(m, G[pa], G[pb])
        mm = rng.random(out_shape) < MUT
        hi = G.max() + 1
        G[w] = np.where(mm, rng.integers(hi, size=out_shape), child)


def run(mode, seed):
    rng = np.random.default_rng(seed)
    phrase = rng.integers(RV, size=(S, V, WORDS_PER))          # world reply phrases
    combos = np.array([(a, b) for a in range(V) for b in range(V)])
    perm = rng.permutation(len(combos)); ntr = int(TRAIN_FRAC * len(combos))
    seen, held = combos[perm[:ntr]], combos[perm[ntr:]]
    # input signal = value (identity channel) -> comprehension must learn it; response on understood value
    Gc = rng.integers(V, size=(N, S, V))                       # comprehension: signal -> understood value
    Gr = rng.integers(RV, size=(N, S, V, WORDS_PER))           # response: understood value -> reply words

    def understood(combos_):
        u = np.zeros((N, len(combos_), S), int)
        for s in range(S): u[:, :, s] = Gc[:, s, combos_[:, s]]
        return u

    def reply_acc(combos_):
        u = understood(combos_); ok = np.ones((N, len(combos_)), bool)
        for s in range(S):
            uv = np.clip(u[:, :, s], 0, V - 1)
            for wp in range(WORDS_PER):
                pred = np.take_along_axis(Gr[:, s, :, wp], uv, 1)
                ok &= (pred == phrase[s, combos_[:, s], wp][None, :])
        return ok.mean()

    if mode == "series":
        for t in range(BUDGET // 2):                            # phase 1: comprehension (understand the input)
            en = np.zeros(N)
            for s in range(S): en += (Gc[:, s, seen[:, s]] == seen[:, s][None, :]).sum(1)
            evolve_step(Gc, en, rng, 0.06)
        for t in range(BUDGET // 2):                            # phase 2: response on understood values
            u = understood(seen); en = np.zeros(N)
            for s in range(S):
                uv = np.clip(u[:, :, s], 0, V - 1)
                for wp in range(WORDS_PER):
                    pred = np.take_along_axis(Gr[:, s, :, wp], uv, 1)
                    en += (pred == phrase[s, seen[:, s], wp][None, :]).sum(1)
            evolve_step(Gr, en, rng, 0.06)
    else:                                                       # joint: one fitness on the final reply
        for t in range(BUDGET):
            u = understood(seen); en = np.zeros(N)
            for s in range(S):
                uv = np.clip(u[:, :, s], 0, V - 1)
                for wp in range(WORDS_PER):
                    pred = np.take_along_axis(Gr[:, s, :, wp], uv, 1)
                    en += (pred == phrase[s, seen[:, s], wp][None, :]).sum(1)
            # mutate both components together
            o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
            for w in worst:
                pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
                for Gx, hi in ((Gc, V), (Gr, RV)):
                    m = rng.random(Gx[pa].shape) < 0.5; child = np.where(m, Gx[pa], Gx[pb])
                    mm = rng.random(child.shape) < 0.06
                    Gx[w] = np.where(mm, rng.integers(hi, size=child.shape), child)
    return float(reply_acc(seen)), float(reply_acc(held))


if __name__ == "__main__":
    out(f"CONVERSATION PIPELINE: hear->understand->reply. S={S}, V={V}, {WORDS_PER}-word phrases. "
        f"pop {N}, budget {BUDGET}, {SEEDS} seeds.")
    out(f"{'training':>10} | {'SEEN full-reply':>16} | {'HELD-OUT full-reply':>20}")
    out("=" * 56)
    for mode in ["joint", "series"]:
        r = np.array([run(mode, s) for s in range(SEEDS)])
        out(f"{mode:>10} | {ms(r[:,0]):>16} | {ms(r[:,1]):>20}")
    out("=" * 56)
    out("READING: if series >> joint, stacked comprehend+respond must be trained in SERIES (parallel")
    out("collapses). held-out ~ seen => the full pipeline generalises to unseen prompts.")
    out("done"); LOG.close()
