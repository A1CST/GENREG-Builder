"""LLM CHAT -- a longer, flowing conversation: bigger repertoire + the LLM does the input matching.

The organism learns multi-word replies to a broad set of conversational prompts (the LLM supplies the
appropriate replies). For the live conversation, the LLM ITSELF maps each incoming message to the
organism's nearest known prompt, so novel phrasings still land -> the chat flows over many turns. The
organism produces appropriate, multi-word, non-echo replies that keep the model talking.
"""
import os, json, re, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("LLM_MODEL", "llama3.2:3b")
N, BUDGET, LMAX = 100, 5000, 5
PROMPTS = ["hello", "hi there", "how are you", "what is your name", "nice to meet you", "thanks",
           "i am hungry", "i am tired", "goodbye", "who are you", "i am lost", "help me",
           "are you ok", "what happened", "where are you", "good morning", "i am scared",
           "tell me more", "see you later", "what do you do", "i like you", "that is funny",
           "i am sad", "can you help", "what time is it", "i am bored", "let us go", "wait for me",
           "i dont understand", "good night", "how was your day", "i am happy", "what is that",
           "come here", "be careful", "i need water"]
LOG = open(os.path.join(HERE, "llm_chat_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def ask(prompt, npred=24, temp=0.0):
    data = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temp, "num_predict": npred}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["response"].strip()


def words(s, k=LMAX): return re.findall(r"[a-z]+", s.lower())[:k]


if __name__ == "__main__":
    out(f"Building conversational repertoire ({len(PROMPTS)} prompts) from {MODEL}...")
    table = {}
    for p in PROMPTS:
        rep = words(ask(f'Casual chat. Someone says: "{p}". Reply in 2 to 4 simple common lowercase words:'))
        table[p] = rep or ["ok"]
    vocab = ["<end>"] + sorted({w for r in table.values() for w in r})
    idx = {w: i for i, w in enumerate(vocab)}; Vv = len(vocab); P = len(PROMPTS)
    target = np.zeros((P, LMAX), int)
    for pi, p in enumerate(PROMPTS):
        for t, w in enumerate(table[p]): target[pi, t] = idx[w]
    out(f"vocab {Vv}; evolving organism over {P} prompts ({BUDGET} gens)...")
    rng = np.random.default_rng(0); G = rng.integers(Vv, size=(N, P, LMAX)); MUT = 0.06
    for t in range(BUDGET):
        en = (G == target[None]).reshape(N, -1).sum(1).astype(float)
        o = np.argsort(en); worst = o[:int(0.25 * N)]; top = o[N - max(2, N // 3):]
        for w in worst:
            pa, pb = (int(top[rng.integers(len(top))]) for _ in range(2))
            m = rng.random((P, LMAX)) < 0.5; child = np.where(m, G[pa], G[pb])
            mm = rng.random((P, LMAX)) < MUT
            G[w] = np.where(mm, rng.integers(Vv, size=(P, LMAX)), child)
    resp = G; j = int(np.argmax([(resp[k] == target).sum() for k in range(N)]))
    word_acc = float((resp[j] == target).mean())
    full_ok = float(np.mean([np.array_equal(resp[j, pi], target[pi]) for pi in range(P)]))
    def render(seq):
        ws = [vocab[w] for w in seq if w != 0]
        return " ".join(ws) if ws else "..."
    out(f"organism {j}: word accuracy {word_acc:.2f}, whole-sentence-correct {full_ok:.2f}")
    out("=" * 70)
    out("LIVE CONVERSATION (the LLM matches each message to the organism's nearest known prompt):")
    plist = "\n".join(f"{i}: {p}" for i, p in enumerate(PROMPTS))
    msg = "hey there friend"
    for turn in range(12):
        sel = ask(f"Known messages:\n{plist}\n\nWhich number is closest in meaning to: \"{msg}\"? "
                  f"Answer with ONLY the number.", npred=6)
        m = re.findall(r"\d+", sel); pi = int(m[0]) % P if m else 0
        reply = render(resp[j, pi])
        out(f'   turn {turn+1:2}:  LLM: "{msg:34}" -> organism: "{reply}"')
        msg = " ".join(words(ask(f'Casual friendly chat. You said "{msg}", friend replied "{reply}". '
                                 f'Reply with one short natural message (3-6 words):', temp=0.4), 7))
    out("done"); LOG.close()
