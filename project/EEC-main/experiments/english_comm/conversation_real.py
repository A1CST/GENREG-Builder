"""CONVERSATION REAL -- generalising English conversation: answer situations never trained on.

A situation is (intent, topic); the appropriate reply composes an opener (from the intent) and a
content (from the topic): reply = opener(intent) ++ content(topic). The organism learns per-component
production, so it answers UNSEEN (intent, topic) combinations in real English -- generalising the
conversation instead of looking it up. Train on a subset of situations; transcript on HELD-OUT ones.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
INTENTS = {"greet": "hello there", "ask": "tell me", "want": "i need",
           "thank": "thanks for", "warn": "watch out", "leave": "see you"}
TOPICS = {"you": "my friend", "name": "your name", "food": "some food",
          "water": "the water", "help": "your help", "home": "going home"}
N, BUDGET, SEEDS, TRAIN_FRAC = 100, 2500, 4, 0.6
LOG = open(os.path.join(HERE, "conversation_real_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"

II = list(INTENTS); TT = list(TOPICS)
vocab = sorted(set(" ".join(INTENTS.values()).split()) | set(" ".join(TOPICS.values()).split()))
idx = {w: i for i, w in enumerate(vocab)}; Vv = len(vocab)
op_target = np.array([[idx[w] for w in INTENTS[i].split()] for i in II])      # (I,2)
co_target = np.array([[idx[w] for w in TOPICS[t].split()] for t in TT])       # (T,2)
I, T = len(II), len(TT)
combos = np.array([(a, b) for a in range(I) for b in range(T)])


def run(seed, want_org=False):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(combos)); ntr = int(TRAIN_FRAC * len(combos))
    seen, held = combos[perm[:ntr]], combos[perm[ntr:]]
    Go = rng.integers(Vv, size=(N, I, 2)); Gc = rng.integers(Vv, size=(N, T, 2))   # per-component openers/contents
    for t in range(BUDGET):
        # fitness over SEEN situations: correct opener words (by intent) + content words (by topic)
        en = np.zeros(N)
        si, st = seen[:, 0], seen[:, 1]
        en += (Go[:, si, :] == op_target[si][None]).reshape(N, -1).sum(1)
        en += (Gc[:, st, :] == co_target[st][None]).reshape(N, -1).sum(1)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            for Gx in (Go, Gc):
                m = rng.random(Gx[pa].shape) < 0.5; child = np.where(m, Gx[pa], Gx[pb])
                mm = rng.random(child.shape) < 0.06
                Gx[w] = np.where(mm, rng.integers(Vv, size=child.shape), child)
    j = int(np.argmax([(Go[k, II_i := combos[:, 0]] == op_target[combos[:, 0]]).sum() for k in range(N)]))

    def reply_ok(cset):
        ok = np.ones(len(cset), bool)
        for k, (a, b) in enumerate(cset):
            ok[k] = np.array_equal(Go[j, a], op_target[a]) and np.array_equal(Gc[j, b], co_target[b])
        return ok.mean()
    if want_org: return Go[j], Gc[j], seen, held
    return float(reply_ok(seen)), float(reply_ok(held))


if __name__ == "__main__":
    out(f"CONVERSATION REAL: generalising English conversation. {I} intents x {T} topics = {I*T} "
        f"situations. pop {N}, budget {BUDGET}, {SEEDS} seeds, train {int(TRAIN_FRAC*100)}%.")
    r = np.array([run(s) for s in range(SEEDS)])
    out(f"  SEEN situation reply-correct:     {ms(r[:,0])}")
    out(f"  HELD-OUT situation reply-correct: {ms(r[:,1])}   (never trained on these situations)")
    out("=" * 72)
    Go, Gc, seen, held = run(0, want_org=True)
    def render(a, b): return " ".join(vocab[w] for w in list(Go[a]) + list(Gc[b]))
    out("TRANSCRIPT on HELD-OUT situations (organism never trained on these intent+topic combos):")
    for (a, b) in held[:12]:
        tgt = INTENTS[II[a]] + " " + TOPICS[TT[b]]
        out(f'   situation [{II[a]:6} + {TT[b]:5}] -> organism: "{render(a,b):22}"  (target "{tgt}")')
    out("done"); LOG.close()
