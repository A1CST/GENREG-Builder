"""LLM CONVERSE -- longer (sentence-length) replies, conversing with a running LLM.

Step up from one-word replies: the LLM gives a short multi-word reply to each prompt (a little
sentence), and the organism learns to PRODUCE that word SEQUENCE in response -- transform the prompt
into an appropriate multi-word answer, not echo it. A PAD token at the end is the discharge/stop, so
reply length emerges per prompt. Cached LLM reply table for cheap evolution; a live pass actually
converses with the model in multi-word turns.
"""
import os, json, re, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
N, BUDGET, LMAX = 80, int(os.environ.get("LC_BUDGET", "6000")), 5
MUT_MASK, MUT_SD = 0.10, 0.12                            # sparse, gentle mutation -> fine consolidation of many word-rules
PROMPTS = ["hello", "how are you", "what is your name", "thanks", "i am hungry", "goodbye",
           "who are you", "i am lost", "help me", "are you ok", "what happened", "where are you",
           "good morning", "i am scared", "tell me more", "see you later"]
LOG = open(os.path.join(HERE, "llm_converse_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def ask(prompt, npred=24, temp=0.0):
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "num_predict": npred}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["response"].strip()


def words(s, k=LMAX):
    return re.findall(r"[a-z]+", s.lower())[:k]


def build_world():
    out(f"Building multi-word reply world from {MODEL}...")
    table = {}
    for p in PROMPTS:
        rep = words(ask(f'Casual chat. Someone says: "{p}". Reply in 2 to 4 simple common '
                        f'lowercase words only:'))
        if not rep: rep = ["ok"]
        table[p] = rep
        out(f'   "{p}"  ->  "{" ".join(rep)}"')
    return table


if __name__ == "__main__":
    table = build_world()
    vocab = ["<end>"] + sorted({w for rep in table.values() for w in rep} | {w for p in PROMPTS for w in p.split()})
    idx = {w: i for i, w in enumerate(vocab)}; Vv = len(vocab)
    P = len(PROMPTS)
    target = np.zeros((P, LMAX), int)                       # padded target reply sequences (<end>=0)
    tlen = np.zeros(P, int)
    for pi, p in enumerate(PROMPTS):
        rep = table[p]; tlen[pi] = len(rep)
        for t, w in enumerate(rep): target[pi, t] = idx[w]
    out("=" * 70)
    out(f"vocab {Vv} words; evolving organism to produce the LLM's sentence replies ({BUDGET} gens)...")
    rng = np.random.default_rng(0); full_target = target.copy()
    G = rng.integers(Vv, size=(N, P, LMAX))                # direct genome: a word index per reply position
    MUT_RATE = 0.08
    for t in range(BUDGET):
        en = (G == full_target[None]).reshape(N, -1).sum(1).astype(float)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            m = rng.random((P, LMAX)) < 0.5
            child = np.where(m, G[pa], G[pb])
            mm = rng.random((P, LMAX)) < MUT_RATE          # mutation = reassign a position to a random word
            child = np.where(mm, rng.integers(Vv, size=(P, LMAX)), child)
            G[w] = child
    resp = G; j = int(np.argmax([(resp[k] == full_target).sum() for k in range(N)]))

    def render(seq):
        ws = []
        for w in seq:
            if w == 0: break                               # <end>
            ws.append(vocab[w])
        return " ".join(ws) if ws else "..."

    word_acc = float((resp[j] == full_target).mean())
    full_ok = float(np.mean([np.array_equal(resp[j, pi], full_target[pi]) for pi in range(P)]))
    out(f"organism {j}: word accuracy {word_acc:.2f}, whole-sentence-correct {full_ok:.2f}")
    out("=" * 70)
    out("TRANSCRIPT -- prompt -> organism's multi-word reply (target = what the LLM said):")
    for pi, p in enumerate(PROMPTS):
        out(f'   "{p:18}" -> organism: "{render(resp[j, pi]):26}"  (LLM: "{" ".join(table[p])}")')
    out("=" * 70)
    out("LIVE conversation with the running LLM (organism replies in sentences):")
    msg = "hello"
    for turn in range(6):
        # nearest known prompt by word overlap
        mw = set(words(msg)); pi = int(np.argmax([len(mw & set(p.split())) for p in PROMPTS]))
        reply = render(resp[j, pi])
        out(f'   turn {turn+1}:  LLM: "{msg}"   ->   organism: "{reply}"')
        msg = ask(f'Casual chat. You said "{msg}", friend replied "{reply}". '
                  f'Continue with a short natural message (2-5 words):', temp=0.3)
        msg = " ".join(words(msg, 6))
    out("done"); LOG.close()
