"""CONVERSATION ACCEPT -- communicate, don't mimic: discover your OWN acceptable response.

Copying the LLM's exact reply is mimicry. Real communication: MANY replies are acceptable, and the
organism should discover one of its own -- judged by whether the partner accepts it, not by matching a
target. We sample the LLM several times per prompt to get the SET of acceptable replies; the organism
evolves to land in that set (its reply is accepted = conversation continues), free to pick a different
valid reply than the LLM's modal one. Then the running LLM JUDGES the organism's self-chosen replies
("is Y a sensible reply to X?") as an independent check that it really communicates.
"""
import os, json, re, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
N, BUDGET, KSAMP = 80, 2000, 8
PROMPTS = ["hello", "thanks", "goodbye", "how are you", "who are you", "i am hungry",
           "i am lost", "help", "good morning", "what is your name", "i am tired", "see you",
           "are you ok", "i am happy"]
LOG = open(os.path.join(HERE, "conversation_accept_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def ask(prompt, npred=10, temp=0.0):
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "num_predict": npred}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["response"].strip()


def w1(s):
    m = re.findall(r"[a-z]+", s.lower()); return m[0] if m else "ok"


if __name__ == "__main__":
    out(f"Sampling acceptable reply SETS from {MODEL} ({KSAMP} samples/prompt)...")
    accept = {}
    for p in PROMPTS:
        s = set()
        for k in range(KSAMP):
            s.add(w1(ask(f'Casual chat. Someone says "{p}". Reply with ONE common lowercase word:',
                         temp=0.9 if k else 0.0)))
        accept[p] = s
        out(f'   "{p:18}" acceptable replies: {", ".join(sorted(s))}')
    vocab = sorted({w for s in accept.values() for w in s})
    idx = {w: i for i, w in enumerate(vocab)}; Vv = len(vocab); P = len(PROMPTS)
    accept_mask = np.zeros((P, Vv), bool)
    for pi, p in enumerate(PROMPTS):
        for w in accept[p]: accept_mask[pi, idx[w]] = True
    modal = {p: ask(f'Casual chat. Someone says "{p}". Reply with ONE common lowercase word:') for p in PROMPTS}
    modal = {p: w1(v) for p, v in modal.items()}
    out("=" * 70)
    out(f"vocab {Vv}; evolving organism to land in the acceptable set ({BUDGET} gens)...")
    rng = np.random.default_rng(0); G = rng.integers(Vv, size=(N, P))
    for t in range(BUDGET):
        en = accept_mask[np.arange(P)[None, :], G].sum(1).astype(float)   # # prompts answered acceptably
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            m = rng.random(P) < 0.5; child = np.where(m, G[pa], G[pb])
            mm = rng.random(P) < 0.08
            G[w] = np.where(mm, rng.integers(Vv, size=P), child)
    j = int(np.argmax(accept_mask[np.arange(P)[None, :], G].sum(1)))
    reply = {p: vocab[int(G[j, pi])] for pi, p in enumerate(PROMPTS)}
    acc = float(np.mean([reply[p] in accept[p] for p in PROMPTS]))
    own = float(np.mean([reply[p] in accept[p] and reply[p] != modal[p] for p in PROMPTS]))
    out(f"organism {j}: acceptable-reply rate {acc:.2f}; chose a DIFFERENT valid reply than the LLM modal: {own:.2f}")
    out("=" * 70)
    out("LLM JUDGES the organism's self-chosen replies (independent check it really communicates):")
    yes = 0
    for p in PROMPTS:
        verdict = w1(ask(f'In a casual chat, someone says "{p}" and the reply is "{reply[p]}". '
                         f'Is that a sensible reply? Answer yes or no:'))
        ok = verdict.startswith("y"); yes += ok
        tag = "own-choice" if (reply[p] != modal[p]) else "modal"
        out(f'   "{p:18}" -> organism: "{reply[p]:10}" [{tag}]   LLM judge: {verdict:4} {"OK" if ok else ""}')
    out(f"LLM accepted {yes}/{P} of the organism's self-chosen replies.")
    out("done"); LOG.close()
