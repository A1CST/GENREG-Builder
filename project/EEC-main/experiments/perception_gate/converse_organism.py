"""The REAL GENREG organism on dialogue, with EVOLVABLE n-gram depth -- the last experimenter knob gone.

The organism is OFFERED a bank of frozen channels (unigram, bigram, trigram, 4-gram). Each channel it
keeps active costs RENT every step, same law as memory. It evolves WHICH channels to maintain and how
much to trust each (a_chan), alongside its memory size M and its neural channel. An organism paying rent
on a channel it doesn't need starves; one too shallow to predict dies. The surviving channel set is the
intersection of what the world demands and what energy can afford -- discovered, not prescribed.

Substrate: rich FROZEN PPMI-SVD embeddings (ED). Engine: ENERGY (channel rent + memory rent + misses),
EVOLVABLE memory M, EVOLVABLE channel depth, ENTROPY decay, selection by SURVIVAL (lifespan)."""
import os, sys, pickle
import numpy as np
from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD
ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "engine")
sys.path.insert(0, ENGINE)
os.environ.setdefault("EEC_VOCAB", "3000")
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE, START_ENERGY, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ED = int(os.environ.get("EEC_ED", "128"))
MAX_M, MIN_M = 48, 2
RENT = float(os.environ.get("EEC_RENT", "0.003"))          # memory rent per unit per step
CHAN_RENT = float(os.environ.get("EEC_CRENT", "0.02"))     # rent per ACTIVE n-gram channel per step
DECAY = 0.97
SEG = 1500
GENS = int(os.environ.get("EEC_GENS", "220"))
CHAN_NAMES = ["uni", "bi", "tri", "4g"]
K = len(CHAN_NAMES)
N_ELITE = round(0.20 * POP_SIZE); N_BOTTOM = round(0.20 * POP_SIZE)
DIALOG = [os.path.join(HERE, "emotional_dialog.txt"), os.path.join(os.path.dirname(HERE), "english_comm", "chat_corpus.txt"), os.path.join(HERE, "dialog_extra.txt")]
def log(s): print(s, flush=True)


def build_corpus():
    text = []
    for p in DIALOG:
        if os.path.exists(p): text.append(open(p, encoding="utf-8", errors="ignore").read())
    toks = tokenize("\n".join(text)); V = int(os.environ["EEC_VOCAB"])
    vocab = ["<unk>"] + [w for w, _ in Counter(toks).most_common(V - 1)]
    w2i = {w: i for i, w in enumerate(vocab)}
    return np.array([w2i.get(t, 0) for t in toks], dtype=np.int32), vocab, w2i


def svd_embeddings(ids, V, dim, WIN=5):
    co = np.zeros(V * V, np.float64); a = ids.astype(np.int64)
    for d in range(1, WIN + 1):
        co += np.bincount(a[:-d] * V + a[d:], minlength=V * V); co += np.bincount(a[d:] * V + a[:-d], minlength=V * V)
    co = co.reshape(V, V); tot = co.sum(); rs = co.sum(1, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log(co * tot / (rs @ rs.T) + 1e-12), 0)
    E = TruncatedSVD(n_components=dim, random_state=0).fit_transform(ppmi)
    return (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def build_ngram(ids, n):                                   # context (n-1 prev tokens) -> next, pruned
    d = defaultdict(Counter)
    for i in range(len(ids) - n + 1):
        d[tuple(int(x) for x in ids[i:i + n - 1])][int(ids[i + n - 1])] += 1
    return {k: v for k, v in d.items() if sum(v.values()) >= 3}


def channels_for_segment(seg, uni_log, bg, grams, V, floor=-12.0):
    """Frozen logits (T,V) for each offered channel: unigram, bigram, trigram, 4-gram (each backs off)."""
    T = len(seg); out = []
    out.append(uni_log[None, :])                            # unigram (broadcast over T)
    out.append(bg[seg])                                     # bigram
    for n, base_idx in ((3, 1), (4, 2)):                   # trigram backs off to bigram; 4-gram to trigram
        chan = np.array(out[base_idx]) if out[base_idx].shape[0] == T else np.broadcast_to(out[base_idx], (T, V)).copy()
        chan = chan.copy()
        g = grams[n]
        for t in range(n - 1, T):
            c = g.get(tuple(int(x) for x in seg[t - n + 2:t + 1]))
            if c:
                tot = sum(c.values()); row = np.full(V, floor, np.float32)
                for nx, cnt in c.items(): row[nx] = np.log(cnt / tot)
                chan[t] = row
        out.append(chan)
    return out


class Organism:
    def __init__(self, V, rng):
        self.W_in = rng.normal(0, 1/np.sqrt(ED), (ED, MAX_M)).astype(np.float32)
        self.W_rec = rng.normal(0, 1/np.sqrt(MAX_M), (MAX_M, MAX_M)).astype(np.float32)
        self.b = np.zeros(MAX_M, np.float32)
        self.W_out = rng.normal(0, 1/np.sqrt(MAX_M), (MAX_M, V)).astype(np.float32)
        self.b_out = np.zeros(V, np.float32)
        self.M = int(rng.integers(MIN_M, MAX_M + 1))
        self.active = rng.random(K) < 0.5                  # EVOLVABLE: which n-gram channels are maintained
        self.a_chan = np.ones(K, np.float32)               # EVOLVABLE: trust per channel
        self.a_nn = np.float32(0.3)

    def params(self): return [self.W_in, self.W_rec, self.b, self.W_out, self.b_out]

    def states(self, emb):
        M = self.M; drive = emb @ self.W_in[:, :M]; Wr = self.W_rec[:M, :M]; bb = self.b[:M]
        T = emb.shape[0]; S = np.empty((T, M), np.float32); s = np.zeros(M, np.float32)
        for t in range(T): s = np.tanh(drive[t] + (DECAY * s) @ Wr + bb); S[t] = s
        return S

    def copy(self):
        g = Organism.__new__(Organism)
        g.W_in, g.W_rec, g.b, g.W_out, g.b_out = [p.copy() for p in self.params()]
        g.M, g.a_nn = self.M, self.a_nn; g.active = self.active.copy(); g.a_chan = self.a_chan.copy()
        return g

    def mutate(self, rng):
        for p in self.params():
            m = rng.random(p.shape) < MUT_RATE
            p += m * rng.normal(0, 1, p.shape).astype(np.float32) * (MUT_SCALE * (np.abs(p) + 1e-3))
        if rng.random() < 0.25:                            # flip a channel on/off
            k = rng.integers(K); self.active[k] = ~self.active[k]
        cm = rng.random(K) < MUT_RATE                      # channel-trust mutation
        self.a_chan = np.maximum(0, self.a_chan + cm * rng.normal(0, 1, K).astype(np.float32) * (MUT_SCALE * (np.abs(self.a_chan) + 1e-2)))
        self.a_nn = np.float32(max(0.0, self.a_nn + rng.normal(0, MUT_SCALE * (abs(self.a_nn) + 1e-2))))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE * self.M)), MIN_M, MAX_M))


def live(org, seg_ids, emb, chans):
    S = org.states(emb)
    combined = org.a_nn * (S @ org.W_out[:org.M, :] + org.b_out)
    for k in range(K):
        if org.active[k]: combined = combined + org.a_chan[k] * chans[k]
    preds = combined.argmax(1); hits = preds[:-1] == seg_ids[1:]
    rent = CHAN_RENT * int(org.active.sum()) + RENT * org.M    # pay for every channel + memory unit
    cum = np.cumsum(rent + (~hits))
    if cum[-1] < START_ENERGY: return len(hits), int(hits.sum()), org.M, int(org.active.sum())
    life = int(np.searchsorted(cum, START_ENERGY)) + 1
    return life, int(hits[:life].sum()), org.M, int(org.active.sum())


def reproduce(pop, lives, rng):
    order = np.argsort(lives)[::-1]; new = [pop[i] for i in order[:POP_SIZE - N_BOTTOM]]; top = order[:N_ELITE]
    while len(new) < POP_SIZE:
        c = pop[int(top[rng.integers(0, len(top))])].copy(); c.mutate(rng); new.append(c)
    return new


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    log(f"dialogue world + rich frozen embeddings (ED={ED}) + EVOLVABLE channel bank {CHAN_NAMES}...")
    ids, vocab, w2i = build_corpus(); V = len(vocab)
    E = svd_embeddings(ids, V, ED)
    uni = np.bincount(ids, minlength=V).astype(np.float64) + 0.1; uni_log = np.log(uni / uni.sum()).astype(np.float32)
    C2 = np.zeros((V, V), np.float32); np.add.at(C2, (ids[:-1], ids[1:]), 1.0); C2 += 0.1
    bg = np.log(C2 / C2.sum(1, keepdims=True)).astype(np.float32)
    grams = {3: build_ngram(ids, 3), 4: build_ngram(ids, 4)}
    log(f"world {len(ids):,} tok, V={V}, ED={ED}, tri {len(grams[3]):,} 4g {len(grams[4]):,}, "
        f"mem_rent={RENT} chan_rent={CHAN_RENT}")
    pop = [Organism(V, rng) for _ in range(POP_SIZE)]; bi = 0; Ms = np.array([2])
    for gen in range(1, GENS + 1):
        start = int(rng.integers(0, len(ids) - SEG - 1)); seg = ids[start:start + SEG]
        emb = E[seg]; chans = channels_for_segment(seg, uni_log, bg, grams, V)
        res = [live(o, seg, emb, chans) for o in pop]
        lives = np.array([r[0] for r in res]); corrs = np.array([r[1] for r in res])
        Ms = np.array([r[2] for r in res]); ncs = np.array([r[3] for r in res])
        bi = int(np.argmax(lives)); b = pop[bi]
        if gen % 10 == 0 or gen <= 5:
            kept = [CHAN_NAMES[k] for k in range(K) if b.active[k]]
            log(f"gen {gen:>4} | life {lives[bi]:>5} | M {Ms[bi]:>2} | chans {str(kept):<22} | "
                f"avg_M {Ms.mean():>4.1f} avg_chans {ncs.mean():>3.1f} | corr {corrs[bi]:>4} ({100*corrs[bi]/max(1,lives[bi]):.0f}%)")
        pop = reproduce(pop, lives, rng)
    b = pop[bi]
    with open(os.path.join(HERE, "convo_organism.pkl"), "wb") as f:
        pickle.dump({"genome": b.params(), "E": E, "M": int(Ms[bi]), "active": b.active, "a_chan": b.a_chan, "a_nn": float(b.a_nn), "vocab": vocab}, f)
    log(f"final channels kept: {[CHAN_NAMES[k] for k in range(K) if b.active[k]]}  trust {np.round(b.a_chan,2)}")
    log("done")
