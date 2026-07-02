"""LLM AS ENVIRONMENT -- the conversational partner is a real LLM, not a hand-coded proxy.

The LLM defines the world's conversational logic: for each prompt word it gives the appropriate reply
(the move that keeps a chat going). The organism evolves to produce those replies -> it learns to
CONVERSE (transform prompt -> appropriate different reply), not parrot. Survival = the conversation
continues = the reply is the one the LLM would accept. The LLM never sees a score; it just talks, and
the organism becomes the thing that keeps it talking. Cheap: the LLM's reply table is cached, then
evolution runs against it; a final LIVE pass actually converses with the running model.
"""
import os, json, re, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
N, BUDGET = 64, 1500
PROMPTS = ["hello", "hi", "thanks", "sorry", "goodbye", "yes", "no", "how",
           "what", "who", "help", "food", "danger", "water", "friend", "name"]
LOG = open(os.path.join(HERE, "llm_world_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def ask(prompt, npred=12, temp=0.0):
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "num_predict": npred}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())["response"].strip()


def word(s):
    m = re.findall(r"[a-z]+", s.lower())
    return m[0] if m else "ok"


def build_world():
    """LLM defines r(prompt) = the one-word reply that keeps the chat alive."""
    out(f"Building the conversational world from {MODEL} (caching its replies)...")
    r = {}
    for p in PROMPTS:
        rep = word(ask(f'Casual chat. Someone says "{p}". Reply with exactly ONE common English word:'))
        r[p] = rep
        out(f"   LLM: \"{p}\" -> \"{rep}\"")
    return r


def evolve(prompt_ids, reply_ids, vocsize, seed=0):
    rng = np.random.default_rng(seed); R = rng.normal(0, 0.3, (N, vocsize, vocsize))
    target = np.full(vocsize, -1); target[prompt_ids] = reply_ids       # learn reply for each prompt
    pmask = prompt_ids
    for t in range(BUDGET):
        resp = R.argmax(2)
        en = (resp[:, pmask] == reply_ids[None, :]).sum(1).astype(float)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            m = rng.random(R[pa].shape) < 0.5
            R[w] = np.where(m, R[pa], R[pb]) + rng.normal(0, 0.22 * 0.6, R[pa].shape)
    return R


if __name__ == "__main__":
    world = build_world()
    vocab = sorted(set(PROMPTS) | set(world.values())); idx = {w: i for i, w in enumerate(vocab)}
    prompt_ids = np.array([idx[p] for p in PROMPTS]); reply_ids = np.array([idx[world[p]] for p in PROMPTS])
    out("=" * 64)
    out(f"vocab ({len(vocab)}): {' '.join(vocab)}")
    out(f"evolving organism to learn the LLM's conversational replies ({BUDGET} gens)...")
    R = evolve(prompt_ids, reply_ids, len(vocab))
    resp = R.argmax(2); j = int(np.argmax([(resp[k, prompt_ids] == reply_ids).sum() for k in range(N)]))
    acc = float((resp[j, prompt_ids] == reply_ids).mean())
    parrot_ok = float(np.mean([world[p] == p for p in PROMPTS]))   # would echoing ever be right?
    out(f"organism {j}: learned-reply accuracy {acc:.2f}  (parrot would score {parrot_ok:.2f} — copying fails)")
    out("=" * 64)
    out("TRANSCRIPT — organism answering the LLM's prompts (reply differs from prompt = communication):")
    for p in PROMPTS:
        r_org = vocab[int(resp[j, p_i := idx[p]])]; r_llm = world[p]
        ok = r_org == r_llm
        out(f'   LLM says "{p:8}" -> organism replies "{r_org:9}"  (LLM-correct "{r_llm}")  '
            f"{'[chat continues]' if ok else '[chat would stall]'}")
    out("=" * 64)
    out("LIVE conversation with the running LLM (organism keeps it going):")
    msg = "hello"
    for turn in range(6):
        # organism replies to the current message (nearest known prompt, else echo-less fallback)
        pid = idx.get(word(msg), None)
        r_org = vocab[int(resp[j, pid])] if pid is not None else vocab[int(resp[j, prompt_ids[turn % len(prompt_ids)]])]
        out(f'   turn {turn+1}: LLM: "{msg}"   |   organism: "{r_org}"')
        # LLM continues the chat given the organism's reply
        msg = word(ask(f'Casual chat. You said "{msg}", friend replied "{r_org}". '
                       f'Say the next short message in ONE word:', temp=0.3))
    out("done"); LOG.close()
