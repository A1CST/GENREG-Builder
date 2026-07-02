"""CONVERSATION LIVE -- the integrated organism: perceive (nomic embeddings) -> comprehend the
situation -> reply, holding a long live conversation with the running LLM, then JUDGED for quality.

Brings the suite together: real-embedding perception (conversation_understand2), situation comprehension
(nearest evolved prototype), per-situation reply. Runs a long conversation with the LLM and has the LLM
independently JUDGE each organism reply (majority of 3 votes) -> a quantitative coherence score for the
full perceive->comprehend->reply loop, with no matching crutch.
"""
import os, json, re, urllib.request, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); GEN = os.environ.get("LLM_MODEL", "llama3.2:3b")
SIT = {"greeting": "hello there friend", "thanks": "you are welcome", "farewell": "see you later",
       "ask_name": "i am a friend", "hungry": "go find food", "ask_help": "i will help you",
       "lost": "stay calm i am here", "tired": "you should get some rest", "happy": "that is great to hear",
       "danger": "watch out be careful", "sad": "i am sorry to hear", "agree": "yes i think so too",
       "question": "let me think about it", "compliment": "that is very kind of you",
       "weather": "the weather is nice today", "bored": "let us do something fun"}
SEED = {"greeting": "hello", "thanks": "thank you so much", "farewell": "goodbye", "ask_name": "what is your name",
        "hungry": "i am so hungry", "ask_help": "can you help me", "lost": "i am completely lost",
        "tired": "i am exhausted", "happy": "i feel wonderful today", "danger": "look out there is danger",
        "sad": "i feel really down", "agree": "i totally agree with you", "question": "what do you think",
        "compliment": "you are amazing", "weather": "what a beautiful day", "bored": "i am so bored"}
LOG = open(os.path.join(HERE, "conversation_live_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def post(ep, p):
    req = urllib.request.Request(f"http://localhost:11434/api/{ep}", data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())
def embed(t):
    v = np.array(post("embed", {"model": "nomic-embed-text", "input": t})["embeddings"][0]); return v/(np.linalg.norm(v)+1e-9)
def gen(p, temp=0.7, n=40): return post("generate", {"model": GEN, "prompt": p, "stream": False, "options": {"temperature": temp, "num_predict": n}})["response"].strip()

names = list(SIT)
out("Building situation prototypes (paraphrase embeddings) ...")
cent = []
for s in names:
    txt = gen(f'Give 6 short everyday ways to say "{SEED[s]}". One per line, lowercase.', 0.8, 80)
    lines = [SEED[s]] + [l.strip(" -.\t") for l in txt.splitlines() if l.strip() and len(l) < 60][:6]
    cent.append(np.mean([embed(l) for l in lines], 0))
cent = np.stack(cent)
def understand(msg): return names[int(np.argmax(cent @ embed(msg)))]

out("=" * 74)
out("LIVE CONVERSATION (organism perceives + comprehends + replies; LLM judges each reply):")
msg = "hey there good to see you"; transcript = []
for turn in range(16):
    s = understand(msg); reply = SIT[s]
    transcript.append((msg, s, reply))
    out(f'   {turn+1:2}. LLM: "{msg:40}" -> [{s:10}] -> "{reply}"')
    msg = " ".join(re.findall(r"[a-z]+", gen(f'Friendly chat. You said "{msg}", friend replied "{reply}". '
                                            f'Reply with one short natural sentence (3-8 words):').lower())[:10]) or "ok"
out("=" * 74)
out("LLM JUDGE (majority of 3): was each organism reply a sensible response to the message?")
good = 0
for m, s, r in transcript:
    votes = sum(gen(f'Message: "{m}". Reply: "{r}". Is the reply a sensible response? Answer only yes or no.',
                    0.5, 4).lower().strip().startswith("y") for _ in range(3))
    ok = votes >= 2; good += ok
out(f"  coherent replies: {good}/{len(transcript)} = {good/len(transcript):.2f} (LLM-judged, majority vote)")
out("done"); LOG.close()
