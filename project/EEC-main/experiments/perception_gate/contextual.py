"""Break the polysemy ceiling with CONTEXT. The one thing that lowered distributional comprehension was
polysemy (a word means different things in different sentences; a single blended embedding can't
resolve it). Fix without gradients: represent a sentence by an iteratively REWEIGHTED mean -- each word
is weighted by how consistent it is with the sentence's emerging meaning, so an ambiguous word is
pulled toward (or down-weighted against) the sense the rest of the sentence implies. Compare plain mean
vs contextual on the polysemy / realistic worlds."""
import os, numpy as np
from realistic import build, ppmi_svd, DIM, T
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "contextual_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def plain(E, toks):
    v = E[toks].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else v


def contextual(E, toks, iters=4):
    em = E[toks]; w = np.ones(len(toks))
    for _ in range(iters):
        v = (w[:, None] * em).sum(0); nv = np.linalg.norm(v)
        if nv == 0: return em.mean(0)
        v /= nv; w = np.clip(em @ v, 0, None) ** 2          # keep words that agree with the consensus meaning
        if w.sum() == 0: w = np.ones(len(toks))
    v = (w[:, None] * em).sum(0); n = np.linalg.norm(v); return v / n if n > 0 else v


def evaluate(cfg, nsyn, seed, repf):
    w = build(cfg, seed)
    Xtr = [w["topic_sent"](w["train_syn"]) for _ in range(2500)]
    Xte = [w["topic_sent"](w["held_syn"]) for _ in range(1000)]
    corpus = [x for x, _ in Xtr] + [w["synonymy_sent"]() for _ in range(nsyn)]
    E = ppmi_svd(corpus, w["V"])
    cent = np.stack([np.mean([repf(E, x) for x, t in Xtr if t == k] or [np.zeros(DIM)], 0) for k in range(T)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9
    return float(np.mean([int((cent @ repf(E, x)).argmax()) == t for x, t in Xte]))


if __name__ == "__main__":
    out("CONTEXT vs polysemy: does a context-reweighted sentence representation break the polysemy ceiling?")
    out(f"(chance {1/T:.3f}, high experience = 20000 synonymy exposures)")
    out("=" * 70)
    out(f"{'condition':>17} | {'plain mean':>14} | {'CONTEXTUAL':>14}")
    for name, cfg in [("+polysemy", {"poly": 0.3}),
                      ("heavy polysemy", {"poly": 0.5}),
                      ("realistic (all)", {"noise": 4, "zipf": 1, "poly": 0.3}),
                      ("realistic + heavy", {"noise": 4, "zipf": 1, "poly": 0.5})]:
        p = [evaluate(cfg, 20000, s, plain) for s in range(2)]
        c = [evaluate(cfg, 20000, s, contextual) for s in range(2)]
        out(f"{name:>17} | {ms(p):>14} | {ms(c):>14}")
    out("=" * 70)
    out("READING: contextual > plain on polysemy => resolving a word's sense from its sentence context")
    out("recovers the comprehension that a single blended embedding loses. Context is the polysemy fix.")
    out("done"); LOG.close()
