"""Build a large CONVERSATIONAL world in parallel (the organism's language exposure).
llama3.2:1b as a fast text source; a few workers; diverse topics/styles; append until target words."""
import json, urllib.request, threading, random, os, time, re
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "chat_corpus.txt")
TARGET = int(os.environ.get("TARGET", "500000")); WORKERS = int(os.environ.get("WORKERS", "4"))
TOPICS = ("weather sunny rainy food hungry meal cooking dinner tired sleep waking greetings hello goodbye party help "
"directions lost happy sad weekend trip vacation thanks grateful bored fun cold water thirsty work boss coworker coffee "
"how-are-you smalltalk catching-up friend scared movie name introduce offering compliment excited news celebration break "
"dreams great workout exercise walk park pets dog cat hobbies music concert book reading shopping money busy tonight "
"family kids school homework game sports phones internet problem advice decision nervous interview goodnight relax stress "
"better neighbor recipe gardening morning afternoon evening plans feelings weekend-plans lunch breakfast").split()
TMPL = ["write a natural casual conversation between two friends about {} and {}",
        "two people making everyday small talk about {} and {}, only spoken lines",
        "a short friendly chat where one asks the other about {} and {}",
        "someone talking to a friend about {} and {} in simple everyday words",
        "a relaxed conversation about {} and {} between people who know each other"]
lock = threading.Lock(); stop = threading.Event(); written = [0]


def gen(p):
    d = json.dumps({"model": "llama3.2:1b", "prompt": p, "stream": False,
                    "options": {"temperature": 1.05, "num_predict": 450}}).encode()
    r = urllib.request.Request("http://localhost:11434/api/generate", data=d,
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=180).read())["response"]


def words():
    try:
        with open(OUT) as f: return len(f.read().split())
    except Exception: return 0


def worker(wid):
    rnd = random.Random(wid * 13 + 7)
    while not stop.is_set():
        a, b = rnd.choice(TOPICS), rnd.choice(TOPICS)
        try:
            txt = gen(rnd.choice(TMPL).format(a, b) + ". simple lowercase words, no names or labels.")
        except Exception:
            time.sleep(3); continue
        txt = re.sub(r"\s+", " ", txt.lower()).strip()
        if len(txt) < 20: continue
        with lock:
            with open(OUT, "a") as f: f.write(txt + "\n")
            written[0] += len(txt.split())


if __name__ == "__main__":
    n0 = words(); print(f"start words={n0} target={TARGET} workers={WORKERS}", flush=True)
    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(WORKERS)]
    for x in ths: x.start()
    t0 = time.time()
    while True:
        time.sleep(30); n = words(); rate = (n - n0) / max(1, time.time() - t0)
        print(f"  words={n} (+{n-n0}) rate={rate*60:.0f}/min eta={(TARGET-n)/max(1,rate)/60:.0f}min", flush=True)
        if n >= TARGET:
            stop.set(); break
    time.sleep(4); print(f"DONE words={words()}", flush=True)
