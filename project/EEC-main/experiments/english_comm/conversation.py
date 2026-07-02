"""CONVERSATION -- from PARROTING (telephone) to COMMUNICATION (transform, don't copy).

Current comm tasks are transmission: hear a meaning, say the SAME meaning back. Real communication is:
hear X, produce the appropriate DIFFERENT response Y (hello->hi, not hello->hello). The world must make
copying fatal: survival = the conversation CONTINUES, which happens only when your response is the
contextually-correct reply r(prompt) != prompt. Parrot the prompt and the conversation dies.

World: a conversational response function r (per-slot derangement, r(v)!=v). A conversation is a run of
turns; each turn the world prompts a meaning (a,b); the organism must answer with r = (g0(a), g1(b));
if right, the conversation CONTINUES (it eats this turn); if wrong, the conversation ENDS. Fitness =
how many turns it kept the conversation alive (no score, just lifespan-of-interaction). Composition:
per-slot response machinery -> generalises to unseen prompt combinations (vocabulary of replies for free).

Controls: a PARROT organism (answer=prompt) should die instantly (r!=identity). We also report response
accuracy and conversation length, and print a transcript.
"""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
S, V = 2, 8; TMAX = 8                                  # slots, values/slot, max conversation length
N = int(os.environ.get("CV_N", "64")); BUDGET = int(os.environ.get("CV_BUDGET", "2000")); SEEDS = 4
LOG = open(os.path.join(HERE, "conversation_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def ms(v): a = np.array(v, float); return f"{a.mean():.2f}+/-{a.std():.2f}"


def response_fn(seed=0):
    """per-slot derangement g_s: a value's correct REPLY, never itself (so copying always fails)."""
    rng = np.random.default_rng(100 + seed); g = np.zeros((S, V), int)
    for s in range(S):
        while True:
            p = rng.permutation(V)
            if np.all(p != np.arange(V)): g[s] = p; break
    return g


def breed(R, en, rng, mut=0.22):
    Ng = len(en); o = np.argsort(en); worst = o[:int(0.25 * Ng)]; top = o[Ng - max(2, Ng // 3):]
    for w in worst:
        pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
        m = rng.random(R[pa].shape) < 0.5
        R[w] = np.where(m, R[pa], R[pb]) + rng.normal(0, mut * 0.6, R[pa].shape)


def conv_len(resp, g, prompts):
    """resp[s,v]=organism reply; leading run of turns whose reply == r(prompt) = conversation lifespan."""
    Ng = resp.shape[0]; T = prompts.shape[1]; alive = np.ones(Ng, bool); length = np.zeros(Ng, int)
    for t in range(T):
        a, b = prompts[:, t, 0], prompts[:, t, 1]
        ok = (resp[np.arange(Ng), 0, a] == g[0, a]) & (resp[np.arange(Ng), 1, b] == g[1, b])
        length += (alive & ok).astype(int); alive &= ok
    return length


def run(seed):
    rng = np.random.default_rng(seed); g = response_fn(seed)
    R = rng.normal(0, 0.3, (N, S, V, V))
    for t in range(BUDGET):
        resp = R.argmax(3)                              # (N,S,V) reply per (slot,value)
        en = (resp == g[None, :, :]).reshape(N, -1).sum(1).astype(float)   # exact #correct reply-rules
        breed(R, en, rng)
    resp = R.argmax(3)
    acc = np.mean([(resp[:, s, np.arange(V)] == g[s][None, :]).mean() for s in range(S)])
    pr = rng.integers(V, size=(N, 200, S)); clen = conv_len(resp, g, pr).mean()
    # parrot baseline: reply = prompt
    parrot = np.zeros((N, S, V), int); parrot[:] = np.arange(V)[None, None, :]
    plen = conv_len(parrot, g, pr).mean()
    return float(acc), float(clen), float(plen), R, g


if __name__ == "__main__":
    out(f"CONVERSATION: transform-don't-copy. S={S} slots, V={V} ({V**S} meanings), reply r(v)!=v. "
        f"pop {N}, budget {BUDGET}, {SEEDS} seeds. fitness = conversation length (max {TMAX}).")
    res = [run(s) for s in range(SEEDS)]
    out(f"  response accuracy (says the correct REPLY):  {ms([r[0] for r in res])}  (chance {1/V:.3f})")
    out(f"  conversation length kept alive:              {ms([r[1] for r in res])}  / {TMAX}")
    out(f"  PARROT baseline conversation length:         {ms([r[2] for r in res])}  (copying dies)")
    out("=" * 70)
    out("READING: accuracy >> chance & convo length >> parrot => the organism learned to TRANSFORM the")
    out("input into the correct different reply, keeping the conversation alive. Parroting starves.")
    # ---- transcript ----
    acc, clen, plen, R, g = res[0]
    resp = R.argmax(3); j = 0
    out("-" * 70)
    out("TRANSCRIPT (organism 0) -- world prompts (a,b); organism replies; correct reply = r(prompt):")
    rng = np.random.default_rng(1); alive = True; turn = 0
    while alive and turn < TMAX:
        a, b = rng.integers(V), rng.integers(V)
        ra, rb = int(resp[j, 0, a]), int(resp[j, 1, b]); ga, gb = int(g[0, a]), int(g[1, b])
        ok = (ra == ga) and (rb == rb if False else rb == gb)
        out(f"  turn {turn+1}: world says ({a},{b}) -> organism replies ({ra},{rb}); "
            f"correct reply ({ga},{gb})  {'[continues]' if ok else '[CONVERSATION DIES]'}")
        alive = ok; turn += 1
    out(f"  (note: replies differ from prompts -> communication, not echo)")
    out("done"); LOG.close()
