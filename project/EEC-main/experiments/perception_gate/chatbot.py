"""Gradient-free generative chatbot with COMPRESSED CONVERSATION STATE.
n-gram back-off LM (local fluency) + a decaying topic-state carried across turns (thread coherence).
The state is a compressed word->weight vector: each turn it decays and absorbs the new content words
(yours AND the bot's), and generation is biased toward it -- so the bot stays on the conversation
instead of resetting every 3 words. Run:  python3 chatbot.py         (state OFF vs ON, same convo)
                                           python3 chatbot.py chat    (talk to it, stateful)"""
import os, re, sys, random, math, collections
HERE = os.path.dirname(os.path.abspath(__file__))
CORP = os.path.join(os.path.dirname(HERE), "english_comm", "chat_corpus.txt")
EOT = "<eot>"; random.seed(0)
SPK = re.compile(r"(?i)\b(?:person\s*[ab12]?|friend\s*\d*|other\s+(?:friend|person)|me|you|a|b)\s*:")
STOP = set("i i'm im a an the to of and you it's it is so that this my me we he she they them their "
           "was were be been have has had do did doing don't just like really kind for in on at with "
           "but what's how's hey yeah oh um umm get got going gonna wanna too not no your you're".split())


def turns_of(line):
    line = SPK.sub(" | ", line); out = []
    for part in re.split(r'\||"', line):
        w = re.findall(r"[a-z']+", part.lower())
        if len(w) >= 2: out.append(w)
    return out


def build():
    stream, turns = [], []
    with open(CORP, encoding="utf-8", errors="ignore") as f:
        for line in f:
            for t in turns_of(line):
                stream += t + [EOT]; turns.append(t)
    ng = {n: collections.defaultdict(collections.Counter) for n in (4, 3, 2)}
    uni = collections.Counter(stream)
    for i in range(len(stream)):
        for n in (4, 3, 2):
            if i >= n - 1: ng[n][tuple(stream[i - n + 1:i])][stream[i]] += 1
    cooc = collections.defaultdict(collections.Counter)
    for t in turns:
        u = set(t)
        for a in u:
            for b in u:
                if a != b: cooc[a][b] += 1
    return ng, uni, cooc


def topic_score(w, state, cooc, uni):
    if not state or w in STOP or w == EOT: return 0.0
    s = sum(wt * math.log1p(cooc[w].get(c, 0)) for c, wt in state.items())
    return s / math.log1p(uni.get(w, 1) + 5)


def sample_next(ng, uni, ctx, state, cooc, beta, temp=0.7):
    for n in (4, 3, 2):
        cand = ng[n].get(tuple(ctx[-(n - 1):]) if n > 1 else ())
        if cand and sum(cand.values()) >= 2:
            words, counts = zip(*cand.items())
            ws = [(c ** (1.0 / temp)) * math.exp(beta * topic_score(w, state, cooc, uni))
                  for w, c in zip(words, counts)]
            tot = sum(ws); r = random.random() * tot
            for word, wt in zip(words, ws):
                r -= wt
                if r <= 0: return word
    return uni.most_common(1)[0][0]


def respond(ng, uni, cooc, user, state, beta=3.0, max_len=30):
    ctx = re.findall(r"[a-z']+", user.lower())[-3:] + [EOT]; out = []
    for _ in range(max_len):
        nxt = sample_next(ng, uni, ctx, state, cooc, beta)
        if nxt == EOT:
            if len(out) >= 3: break
            continue
        if len(out) >= 2 and nxt == out[-1] == out[-2]: continue
        out.append(nxt); ctx = (ctx + [nxt])[-3:]
    return out


GREET_W = {"hey", "hi", "hello", "yo", "sup", "morning"}
FEEL = {"tired", "sleepy", "exhausted", "wiped", "hungry", "starving", "thirsty", "cold", "freezing",
        "hot", "sad", "down", "happy", "glad", "scared", "bored", "stressed", "sick", "excited",
        "nervous", "angry", "mad", "good", "great", "lonely", "anxious", "relaxed", "busy"}
QW = re.compile(r"(do|did|are|is|was|were|what|how|when|where|why|who|can|could|would|will|have|you)\b")


DANGLE = set("a an the some or and to of for with my your our that this we i you it is are was been "
             "maybe kinda about like just so we're they're what how when if but at on in".split())


def detect_act(text, content):                                  # comprehend the user's MOVE (greeting beats '?')
    t = text.strip().lower(); ws = set(re.findall(r"[a-z']+", t))
    if ws & GREET_W or "how's it going" in t or "what's up" in t or "how are you" in t: return "greeting"
    if any(w in FEEL for w in ws): return "feeling"
    if "?" in text or QW.match(t): return "question"
    return "statement"


def gen_short(ng, uni, cooc, topic, maxlen=9, beta=5.0):        # express ONE short topical clause, end clean
    ctx = [EOT]; out = []
    for _ in range(maxlen):
        nxt = sample_next(ng, uni, ctx, topic, cooc, beta)
        if nxt == EOT:
            if len(out) >= 3: break
            continue
        if len(out) >= 2 and nxt == out[-1] == out[-2]: continue
        out.append(nxt); ctx = (ctx + [nxt])[-3:]
    while out and out[-1] in DANGLE: out.pop()                  # don't end on a dangling function word
    return " ".join(out)


POSITIVE = {"happy", "glad", "good", "great", "excited", "relaxed"}
SKIP_TOPIC = set("now today later sometime much time thing stuff bit lot day week something anything "
                 "back out here there one some together really actually pretty kinda also even still "
                 "again maybe sure okay though anyway around about".split())


def respond_intent(ng, uni, cooc, user, state):                 # decide a move -> say it cleanly (no chaining)
    content = [w for w in re.findall(r"[a-z']+", user.lower()) if w not in STOP and uni.get(w, 0) > 3]
    topic = next((w for w in reversed(content) if w not in SKIP_TOPIC and w not in FEEL), None)
    act = detect_act(user, content)
    if act == "greeting":
        return random.choice(["hey! pretty good, you?", "good, just chilling — how about you?", "not bad! how are you?"])
    if act == "feeling":
        f = next((w for w in content if w in FEEL), None)
        if f in POSITIVE: return random.choice([f"nice, glad you're {f}!", f"aw {f}, love that — what's up?"])
        return random.choice([f"ugh, {f} too — that's rough", f"aw, {f}? hope it picks up", "yeah, me too honestly"]) \
            if f else "yeah, i feel that"
    if act == "question":
        return random.choice(["honestly not much — you?", "yeah, for sure! how about you?", "hmm, not really — what about you?"])
    if topic:                                                   # statement: acknowledge + topical follow-up (clean slot)
        return random.choice([f"oh nice! how was the {topic}?", f"nice — how's the {topic} going?",
                              f"cool, tell me more about the {topic}"])
    return random.choice(["oh nice! tell me more", "haha, sounds good to me", "yeah? how'd that go?"])


class Convo:
    def __init__(self, decay=0.5): self.s = {}; self.decay = decay
    def add(self, words, uni):
        for k in list(self.s): self.s[k] *= self.decay
        for x in words:
            if x not in STOP and uni.get(x, 0) > 3: self.s[x] = self.s.get(x, 0) + 1.0
        self.s = {k: v for k, v in self.s.items() if v > 0.1}
    def topic(self): return dict(self.s)


def say(words): return (" ".join(words)[0:1].upper() + " ".join(words)[1:]) if words else "..."


CONV = ["hey, how's it going?", "i just got back from the gym", "yeah i'm pretty wiped out now",
        "do you exercise much?", "we should work out together sometime", "what are you doing later?"]

if __name__ == "__main__":
    print("building n-gram model + topic co-occurrence...", flush=True)
    ng, uni, cooc = build()
    print(f"vocab {len(uni):,}, 4-gram contexts {len(ng[4]):,}\n")
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        st = Convo()
        print("chat (stateful, ctrl-C to quit):")
        while True:
            try: u = input("you: ")
            except (EOFError, KeyboardInterrupt): break
            st.add(re.findall(r"[a-z']+", u.lower()), uni)   # state absorbs only your turns
            print("bot:", respond_intent(ng, uni, cooc, u, st.topic()),
                  f"   [state: {', '.join(list(st.topic())[:6])}]")
    else:
        print("======  CHAIN (old: n-gram + topic state)  ======")
        st = Convo()
        for u in CONV:
            st.add(re.findall(r"[a-z']+", u.lower()), uni)
            print(f"you: {u}\nbot: {say(respond(ng, uni, cooc, u, st.topic()))}\n")
        print("======  INTENT (new: decide a move, say it short)  ======")
        st = Convo()
        for u in CONV:
            st.add(re.findall(r"[a-z']+", u.lower()), uni)
            print(f"you: {u}\nbot: {respond_intent(ng, uni, cooc, u, st.topic())}\n")
