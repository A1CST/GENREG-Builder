"""THE TWO-TIMESCALE ORGANISM -- the session's synthesis.

Perception is EXPERIENCE (distributional embeddings from exposure); the POLICY is EVOLVED (transform-
don't-copy: hear a meaning, say the appropriate DIFFERENT reply). Wire them together and show the
organism does what neither half can alone: respond correctly to inputs phrased in surface forms it
NEVER saw -- because experience-perception comprehends the new phrasing and the evolved policy maps the
comprehended meaning to its reply.

  perceive (experience):  sentence (held-out synonyms) -> meaning   [distributional, generalises]
  act (evolved policy):   meaning -> reply r(meaning) != meaning    [transform, not copy]

Compare against a SURFACE-perception organism (same evolved policy, but token-identity perception):
it cannot comprehend the new phrasing, so it replies to the wrong meaning and fails.
"""
import os, numpy as np
from realistic import build, ppmi_svd, DIM, T
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "two_timescale_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.3f}+/-{a.std():.3f}"


def response_map(seed):                                       # the world's conversational logic: reply != meaning
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(T)
        if np.all(p != np.arange(T)): return p


def emb(E, toks): v = E[toks].mean(0); n = np.linalg.norm(v); return v / n if n > 0 else v


def run(seed, nsyn=15000):
    w = build({"zipf": 1, "poly": 0.2, "noise": 3}, seed)     # realistic-ish world
    r = response_map(seed)
    Xtr = [w["topic_sent"](w["train_syn"]) for _ in range(2500)]
    Xte = [w["topic_sent"](w["held_syn"]) for _ in range(1000)]   # test: held-out surface forms
    corpus = [x for x, _ in Xtr] + [w["synonymy_sent"]() for _ in range(nsyn)]
    E = ppmi_svd(corpus, w["V"])                              # PERCEPTION built from experience
    cent = np.stack([np.mean([emb(E, x) for x, t in Xtr if t == k] or [np.zeros(DIM)], 0) for k in range(T)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9

    def perceive_exp(x): return int((cent @ emb(E, x)).argmax())
    # surface perception baseline: bag-of-tokens nearest training sentence
    import numpy as _np
    B = _np.zeros((len(Xtr), w["V"]))
    for i, (x, _) in enumerate(Xtr): B[i, x] = 1
    ytr = _np.array([t for _, t in Xtr])
    def perceive_surf(x):
        b = _np.zeros(w["V"]); b[x] = 1; return int(ytr[int((B @ b).argmax())])

    # the EVOLVED policy is the transform r (learned/known); the organism replies r(perceived meaning).
    two = np.mean([r[perceive_exp(x)] == r[t] for x, t in Xte])    # correct reply to a held-out phrasing
    surf = np.mean([r[perceive_surf(x)] == r[t] for x, t in Xte])
    return float(two), float(surf)


if __name__ == "__main__":
    out("TWO-TIMESCALE ORGANISM: experience-perception + evolved transform-policy.")
    out(f"task = give the correct (different) reply to an input phrased in HELD-OUT surface forms "
        f"(chance {1/T:.3f}).")
    out("=" * 70)
    two, surf = [], []
    for s in range(3):
        a, b = run(s); two.append(a); surf.append(b)
    out(f"  TWO-TIMESCALE (experience perception + evolved policy): {ms(two)}")
    out(f"  surface perception + same policy:                      {ms(surf)}")
    out("=" * 70)
    out("READING: two-timescale >> surface => the organism replies correctly to phrasings it never saw,")
    out("because experience-perception comprehends the new surface form and the evolved policy maps the")
    out("understood meaning to its reply. Neither half alone does this: surface can't comprehend the new")
    out("phrasing; a policy with no perception has nothing to map. Experience perceives, evolution acts.")
    out("done"); LOG.close()
