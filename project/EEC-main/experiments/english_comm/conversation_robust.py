"""CONVERSATION ROBUST -- honest quality of the integrated organism over MANY conversations.

One 16-turn run is anecdotal. Run several independent conversations (different openers) through the
full perceive(nomic)->comprehend->reply loop with variation+memory+pivot, judge each reply by the LLM
(majority of 3), and report mean +/- std coherence. A robust number for the integrated system.
"""
import os, json, re, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); GEN = os.environ.get("LLM_MODEL", "llama3.2:3b")
REPLIES = {
 "greeting": ["hello there friend", "hey good to see you", "hi how are you"],
 "thanks": ["you are welcome", "no problem at all", "happy to help"],
 "farewell": ["see you later", "take care now", "goodbye for now"],
 "hungry": ["go find some food", "let us get a snack", "you should eat something"],
 "ask_help": ["i will help you", "sure what do you need", "of course i can help"],
 "tired": ["you should get some rest", "go take a nap", "sleep would do you good"],
 "happy": ["that is great to hear", "i am glad for you", "wonderful news"],
 "sad": ["i am sorry to hear that", "things will get better", "i am here for you"],
 "weather": ["the weather is lovely", "what a nice day", "perfect day outside"],
 "bored": ["let us do something fun", "want to go for a walk", "lets find an adventure"],
 "agree": ["yes i think so too", "you are right", "i agree completely"],
 "compliment": ["that is very kind", "you are too nice", "thank you so much"]}
SEED = {"greeting": "hello", "thanks": "thanks a lot", "farewell": "goodbye", "hungry": "i am hungry",
        "ask_help": "can you help me", "tired": "i am so tired", "happy": "i feel great", "sad": "i feel down",
        "weather": "what a nice day", "bored": "i am bored", "agree": "i agree", "compliment": "you are wonderful"}
PIVOTS = {"hungry": "by the way are you getting hungry", "weather": "anyway hows the weather there",
          "bored": "what do you do for fun", "happy": "what made you happy today", "tired": "are you feeling tired"}
OPENERS = ["hey there good to see you", "i could really use some help", "what a beautiful morning",
           "i am feeling pretty tired today", "thanks so much for everything"]
LOG = open(os.path.join(HERE, "conversation_robust_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def post(ep, p):
    req = urllib.request.Request(f"http://localhost:11434/api/{ep}", data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
def embed(t):
    v = np.array(post("embed", {"model": "nomic-embed-text", "input": t})["embeddings"][0]); return v/(np.linalg.norm(v)+1e-9)
def gen(p, temp=0.7, n=40): return post("generate", {"model": GEN, "prompt": p, "stream": False, "options": {"temperature": temp, "num_predict": n}})["response"].strip()

names = list(REPLIES)
out("Building prototypes (nomic)...")
cent = []
for s in names:
    txt = gen(f'Give 6 short everyday ways to say "{SEED[s]}". One per line, lowercase.', 0.8, 80)
    lines = [SEED[s]] + [l.strip(" -.\t") for l in txt.splitlines() if l.strip() and len(l) < 60][:6]
    cent.append(np.mean([embed(l) for l in lines], 0))
cent = np.stack(cent)


def converse(opener, turns=12):
    msg = opener; recent = []; used = {s: 0 for s in names}; pv = list(PIVOTS); transcript = []
    for _ in range(turns):
        s = names[int(np.argmax(cent @ embed(msg)))]
        if recent[-2:] == [s, s] and pv:
            nt = next((p for p in pv if p != s), pv[0]); pv.remove(nt); reply = PIVOTS[nt]
        else:
            reply = REPLIES[s][used[s] % len(REPLIES[s])]; used[s] += 1
        recent.append(s); transcript.append((msg, reply))
        msg = " ".join(re.findall(r"[a-z]+", gen(f'Friendly chat. You said "{msg}", friend replied "{reply}". '
                                  f'Reply with one short natural sentence (3-8 words):').lower())[:10]) or "ok"
    good = 0
    for m, r in transcript:
        v = sum(gen(f'Message: "{m}". Reply: "{r}". Is the reply sensible? Answer only yes or no.', 0.4, 4)
                .lower().strip().startswith("y") for _ in range(3))
        good += v >= 2
    return good / len(transcript), len({r for _, r in transcript})


if __name__ == "__main__":
    out("=" * 70)
    coh, dist = [], []
    for i, op in enumerate(OPENERS):
        c, d = converse(op); coh.append(c); dist.append(d)
        out(f"  conversation {i+1} (opener: \"{op[:28]}\"): coherence {c:.2f}, {d} distinct replies")
    out("=" * 70)
    out(f"INTEGRATED SYSTEM over {len(OPENERS)} conversations: coherence "
        f"{np.mean(coh):.2f}+/-{np.std(coh):.2f}, distinct replies {np.mean(dist):.1f}")
    out("READING: a robust coherence number for the full perceive->comprehend->reply loop with a live LLM.")
    out("done"); LOG.close()
