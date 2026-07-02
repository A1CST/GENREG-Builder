"""CONVERSATION COMPOSE -- multi-word replies that GENERALISE to unseen prompts (kill the lookup).

The llm_chat organism is a fixed-prompt lookup. Here the prompt is compositional (a,b) and the reply
is a multi-word phrase built per-component: reply = phrase(a) ++ phrase(b), each phrase WORDS_PER words.
The organism learns per-component phrase production, so an unseen (a,b) combination still yields the
correct multi-word reply -- zero-shot conversation with sentence-length output. Train on a subset of
prompt combinations, test reply accuracy on HELD-OUT combinations never seen.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S = 2; WORDS_PER = 2; N = 100; BUDGET = 3000; SEEDS = 4; TRAIN_FRAC = 0.7
VS = [6, 10, 16]
LOG = open(os.path.join(HERE, "conversation_compose_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def run(V, seed):
    rng = np.random.default_rng(seed)
    # per-component phrase target: phrase[s][value] = WORDS_PER word-ids (reply vocab = V*WORDS_PER words)
    RV = V * WORDS_PER
    phrase = rng.integers(RV, size=(S, V, WORDS_PER))      # the world's reply phrases (compositional)
    combos = np.array([(a, b) for a in range(V) for b in range(V)])
    perm = rng.permutation(len(combos)); ntr = int(TRAIN_FRAC * len(combos))
    seen, held = combos[perm[:ntr]], combos[perm[ntr:]]
    # organism genome: G[n, s, value, word] = predicted reply word for that component value
    G = rng.integers(RV, size=(N, S, V, WORDS_PER)); MUT = 0.06
    def reply_correct(combos_):
        # for each combo, all S*WORDS_PER reply words correct
        ok = np.ones((N, len(combos_)), bool)
        for s in range(S):
            cv = combos_[:, s]
            for wp in range(WORDS_PER):
                ok &= (G[:, s, cv, wp] == phrase[s, cv, wp][None, :])
        return ok
    for t in range(BUDGET):
        # fitness = correct reply-words over SEEN prompts (per-component, partial credit)
        en = np.zeros(N)
        for s in range(S):
            cv = seen[:, s]
            for wp in range(WORDS_PER):
                en += (G[:, s, cv, wp] == phrase[s, cv, wp][None, :]).sum(1)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            m = rng.random(G[pa].shape) < 0.5; child = np.where(m, G[pa], G[pb])
            mm = rng.random(child.shape) < MUT
            G[w] = np.where(mm, rng.integers(RV, size=child.shape), child)
    seen_acc = reply_correct(seen).mean(); held_acc = reply_correct(held).mean()
    return float(seen_acc), float(held_acc)


if __name__ == "__main__":
    out(f"CONVERSATION COMPOSE: {WORDS_PER}-word phrase per component, S={S} -> {WORDS_PER*S}-word replies. "
        f"pop {N}, budget {BUDGET}, {SEEDS} seeds, train {int(TRAIN_FRAC*100)}% of V^2 prompts.")
    out(f"{'V':>4} {'prompts':>8} | {'SEEN reply-correct':>19} | {'HELD-OUT reply-correct':>23}")
    out("=" * 64)
    for V in VS:
        r = np.array([run(V, s) for s in range(SEEDS)])
        out(f"{V:>4} {V*V:>8} | {ms(r[:,0]):>19} | {ms(r[:,1]):>23}")
    out("=" * 64)
    out("READING: HELD-OUT full-reply-correct ~ SEEN => the organism produces the correct multi-word reply")
    out("to prompt combinations it never trained on -> generalising sentence-level conversation (no lookup).")
    out("done"); LOG.close()
