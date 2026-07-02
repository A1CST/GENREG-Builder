"""Generate a CONVERSATIONAL world (casual dialogue corpus) -- the LLM is just a text source (a book),
not the perceiver. The organism will build its OWN embeddings from co-occurrence over this."""
import json, urllib.request, time, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_corpus.txt")
def gen(p, n=480, temp=1.0):
    d = json.dumps({"model":"llama3.2:3b","prompt":p,"stream":False,"options":{"temperature":temp,"num_predict":n}}).encode()
    r = urllib.request.Request("http://localhost:11434/api/generate", data=d, headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=180).read())["response"]
TOPICS = ["the weather","food and being hungry","feeling tired","greetings between friends","saying goodbye",
 "asking for help","being lost and needing directions","a happy day","feeling sad or down","plans for the weekend",
 "thanking someone","being bored","meeting someone new","feeling cold","needing water","a long day at work",
 "morning routines","how are you doing","making small talk","running into a friend","feeling scared",
 "asking someone's name","offering help","complimenting a friend","being excited","needing a break",
 "talking about sleep","being thirsty","feeling great","casual catching up","leaving a party","good night"]
n=0
with open(OUT,"w") as f:
    for rep in range(4):
        for t in TOPICS:
            try:
                txt = gen(f"Write a short natural casual conversation between two friends about {t}. "
                          f"Only their spoken lines, simple everyday words, lowercase, no names or labels.")
                f.write(txt.lower().replace("\n"," ") + "\n"); f.flush(); n+=1
                if n % 20 == 0: print(f"  {n} dialogues, ~{os.path.getsize(OUT)//1024}KB", flush=True)
            except Exception as e: print("skip", e, flush=True)
print(f"done: {n} dialogues, {os.path.getsize(OUT)//1024}KB")
