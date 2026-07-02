"""CONVERSATION LIVE v2 -- break the loop with VARIED replies + MEMORY + initiative.

v1 looped on one situation (fixed reply -> feedback loop). Here each situation has several replies; the
organism remembers recent situations/replies and (a) does not repeat a reply, (b) when the same
situation recurs, PIVOTS -- introduces a new topic with a question (conversational initiative). Same
nomic-embedding perception + comprehension; LLM judges coherence. Tests whether variation+memory
sustain a flowing conversation.
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
PIVOTS = {"hungry": "by the way are you getting hungry", "weather": "anyway how is the weather there",
          "bored": "what do you do for fun", "happy": "what made you happy today", "tired": "are you feeling tired"}
LOG = open(os.path.join(HERE, "conversation_live2_results.txt"), "w")
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
out("=" * 74)
out("LIVE CONVERSATION v2 (varied replies + memory + pivot on repeat):")
msg = "hey there good to see you"; recent = []; used = {s: 0 for s in names}; transcript = []
pivot_list = list(PIVOTS)
for turn in range(16):
    s = names[int(np.argmax(cent @ embed(msg)))]
    if recent[-2:] == [s, s] and pivot_list:                    # stuck -> pivot to a new topic (initiative)
        nt = next((p for p in pivot_list if p != s), pivot_list[0]); pivot_list.remove(nt)
        reply = PIVOTS[nt]; tag = f"{s}->PIVOT:{nt}"
    else:
        opts = REPLIES[s]; reply = opts[used[s] % len(opts)]; used[s] += 1; tag = s
    recent.append(s); transcript.append((msg, reply))
    out(f'   {turn+1:2}. LLM: "{msg:40}" -> [{tag:18}] -> "{reply}"')
    msg = " ".join(re.findall(r"[a-z]+", gen(f'Friendly chat. You said "{msg}", friend replied "{reply}". '
                                            f'Reply with one short natural sentence (3-8 words):').lower())[:10]) or "ok"
out("=" * 74)
good = 0
for m, r in transcript:
    votes = sum(gen(f'Message: "{m}". Reply: "{r}". Is the reply a sensible response? Answer only yes or no.',
                    0.5, 4).lower().strip().startswith("y") for _ in range(3))
    good += votes >= 2
distinct = len({r for _, r in transcript})
out(f"  coherent replies (LLM-judged, majority): {good}/{len(transcript)} = {good/len(transcript):.2f}")
out(f"  distinct replies used: {distinct}/{len(transcript)} (v1 looped on ~2)")
out("done"); LOG.close()
