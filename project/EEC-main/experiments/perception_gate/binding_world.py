"""MEMORY-DEMANDING WORLD (synthetic, clean) for the real GENREG organism.

Each conversation binds arbitrary marker->value pairs at the start (value random from a pool, so
unknowable in advance), then a stretch of 4-gram-predictable filler punctuated by recurring queries
'GET marker_b -> value_b'. Because the value is random PER CONVERSATION, the frozen 4-gram is uniform
over the pool at every query -- it CANNOT predict the value. Only an organism that wrote value_b into
memory at the binding can. Queries recur through the stretch: a forgetful organism misses every one and
starves; distance grows with D. The organism (energy, EVOLVABLE memory M, EVOLVABLE channels, recurrent,
entropy decay, selection=lifespan) must grow memory to live.

Sweep the WORLD (distance D, binding count B); evolve the ORGANISM. Success: organism beats the frozen
4-gram at recall points, AND M climbs with D.
"""
import os, sys
import numpy as np
from collections import Counter, defaultdict
ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "engine")
sys.path.insert(0, ENGINE)
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE, START_ENERGY
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- vocab: specials + markers + value pool + filler ----
SET, GET, SEP = 1, 2, 3
BMAX, POOL, NFILL = 4, 40, 40
MARK0 = 4; VAL0 = MARK0 + BMAX; FILL0 = VAL0 + POOL
VOCAB = FILL0 + NFILL
ED, MAX_M, MIN_M = 64, 48, 2
RENT = 0.003; CHAN_RENT = 0.02; DECAY = 0.9
BUDGET = 150                                                  # tight energy: forgetting must be FATAL mid-conversation
QUERY_RATE = 0.6                                              # ~60% of events are recalls -> binding governs the stretch
CHAN_NAMES = ["uni", "bi", "tri", "4g"]; K = len(CHAN_NAMES)
N_ELITE = round(0.20 * POP_SIZE); N_BOTTOM = round(0.20 * POP_SIZE)
def log(s): print(s, flush=True)


def filler_step(prev, rng):                                   # order-1 markov filler (4g-predictable)
    return FILL0 + int((prev * 7 + 3) % NFILL) if rng.random() < 0.6 else FILL0 + int(rng.integers(NFILL))


def gen_conversation(B, D, rng):
    toks = []; vals = {}
    for b in range(B):
        v = VAL0 + int(rng.integers(POOL)); toks += [SET, MARK0 + b, v]; vals[MARK0 + b] = v
    val_pos = []; f = FILL0; last = {b: -999 for b in range(B)}; GAP = 8   # >4-gram window: no local copy
    while len(toks) < 3 * B + D:
        avail = [b for b in range(B) if len(toks) - last[b] > GAP]
        if avail and rng.random() < QUERY_RATE:              # dense across markers, but each value out-of-window
            b = int(rng.choice(avail)); toks += [GET, MARK0 + b]
            val_pos.append(len(toks)); toks += [vals[MARK0 + b]]; last[b] = len(toks)
        else:
            f = filler_step(f, rng); toks.append(f)
    toks.append(SEP)
    return np.array(toks, np.int32), val_pos


def make_stream(B, D, n_conv, rng):
    stream, positions = [], []
    for _ in range(n_conv):
        c, vp = gen_conversation(B, D, rng)
        base = len(stream); stream += list(c); positions += [base + p for p in vp]
    return np.array(stream, np.int32), positions


# ---- frozen channels (uni/bi/tri/4g) + random frozen embeddings ----
def bigram_logits(ids, V):
    C = np.zeros((V, V), np.float32); np.add.at(C, (ids[:-1], ids[1:]), 1.0); C += 0.1
    return np.log(C / C.sum(1, keepdims=True)).astype(np.float32)


def build_ngram(ids, n):
    d = defaultdict(Counter)
    for i in range(len(ids) - n + 1):
        d[tuple(int(x) for x in ids[i:i + n - 1])][int(ids[i + n - 1])] += 1
    return {k: v for k, v in d.items() if sum(v.values()) >= 3}


def channels(seg, uni_log, bg, grams, V, floor=-12.0):
    T = len(seg); out = [uni_log[None, :], bg[seg]]
    for n, bi in ((3, 1), (4, 2)):
        chan = np.broadcast_to(out[bi], (T, V)).copy() if out[bi].shape[0] != T else out[bi].copy()
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
    def __init__(self, rng):
        self.W_in = rng.normal(0, 1/np.sqrt(ED), (ED, MAX_M)).astype(np.float32)
        self.W_rec = rng.normal(0, 1/np.sqrt(MAX_M), (MAX_M, MAX_M)).astype(np.float32)
        self.b = np.zeros(MAX_M, np.float32)
        self.W_out = rng.normal(0, 1/np.sqrt(MAX_M), (MAX_M, VOCAB)).astype(np.float32)
        self.b_out = np.zeros(VOCAB, np.float32)
        self.M = int(rng.integers(MIN_M, MAX_M + 1))
        self.active = rng.random(K) < 0.5; self.a_chan = np.ones(K, np.float32); self.a_nn = np.float32(0.3)

    def params(self): return [self.W_in, self.W_rec, self.b, self.W_out, self.b_out]

    def states(self, emb):
        M = self.M; drive = emb @ self.W_in[:, :M]; Wr = self.W_rec[:M, :M]; bb = self.b[:M]
        T = emb.shape[0]; S = np.empty((T, M), np.float32); s = np.zeros(M, np.float32)
        for t in range(T): s = np.tanh(drive[t] + (DECAY * s) @ Wr + bb); S[t] = s
        return S

    def logits(self, emb, chans):
        out = self.a_nn * (self.states(emb) @ self.W_out[:self.M, :] + self.b_out)
        for k in range(K):
            if self.active[k]: out = out + self.a_chan[k] * chans[k]
        return out

    def copy(self):
        g = Organism.__new__(Organism)
        g.W_in, g.W_rec, g.b, g.W_out, g.b_out = [p.copy() for p in self.params()]
        g.M, g.a_nn = self.M, self.a_nn; g.active = self.active.copy(); g.a_chan = self.a_chan.copy()
        return g

    def mutate(self, rng):
        for p in self.params():
            m = rng.random(p.shape) < MUT_RATE
            p += m * rng.normal(0, 1, p.shape).astype(np.float32) * (MUT_SCALE * (np.abs(p) + 1e-3))
        if rng.random() < 0.25:
            k = rng.integers(K); self.active[k] = ~self.active[k]
        cm = rng.random(K) < MUT_RATE
        self.a_chan = np.maximum(0, self.a_chan + cm * rng.normal(0, 1, K).astype(np.float32) * (MUT_SCALE * (np.abs(self.a_chan) + 1e-2)))
        self.a_nn = np.float32(max(0.0, self.a_nn + rng.normal(0, MUT_SCALE * (abs(self.a_nn) + 1e-2))))
        if rng.random() < 0.3:
            self.M = int(np.clip(round(self.M + rng.normal(0, MUT_SCALE * self.M)), MIN_M, MAX_M))


def live(org, seg, emb, chans):
    preds = org.logits(emb, chans).argmax(1); hits = preds[:-1] == seg[1:]
    cum = np.cumsum(CHAN_RENT * int(org.active.sum()) + RENT * org.M + (~hits))
    if cum[-1] < BUDGET: return len(hits), org.M
    return int(np.searchsorted(cum, BUDGET)) + 1, org.M


def reproduce(pop, lives, rng):
    order = np.argsort(lives)[::-1]; new = [pop[i] for i in order[:POP_SIZE - N_BOTTOM]]; top = order[:N_ELITE]
    while len(new) < POP_SIZE:
        c = pop[int(top[rng.integers(0, len(top))])].copy(); c.mutate(rng); new.append(c)
    return new


def recall_acc(org, B, D, emb_table, uni_log, bg, grams, rng, ntest=6):
    """accuracy specifically at the value-recall positions (organism vs frozen 4-gram-only)."""
    oh = ot = bh = 0
    for _ in range(ntest):
        seg, vp = gen_conversation(B, D, rng)
        if not vp: continue
        ch = channels(seg, uni_log, bg, grams, VOCAB); emb = emb_table[seg]
        op = org.logits(emb, ch).argmax(1)
        bp = (ch[3]).argmax(1)                               # 4-gram-only baseline
        for p in vp:
            ot += 1; oh += int(op[p - 1] == seg[p]); bh += int(bp[p - 1] == seg[p])
    return oh / max(1, ot), bh / max(1, ot)


def run_cell(B, D, gens=120):
    rng = np.random.default_rng(0)
    emb_table = rng.normal(0, 1, (VOCAB, ED)).astype(np.float32)   # frozen random distinct embeddings
    emb_table /= np.linalg.norm(emb_table, axis=1, keepdims=True)
    train, _ = make_stream(B, D, 200, rng)                   # frozen channels learn filler+structure (NOT the random values)
    uni = np.bincount(train, minlength=VOCAB).astype(np.float64) + 0.1; uni_log = np.log(uni / uni.sum()).astype(np.float32)
    bg = bigram_logits(train, VOCAB); grams = {3: build_ngram(train, 3), 4: build_ngram(train, 4)}
    pop = [Organism(rng) for _ in range(POP_SIZE)]; bi = 0; Ms = np.array([2])
    for gen in range(1, gens + 1):
        seg, _ = gen_conversation(B, D, rng); emb = emb_table[seg]; ch = channels(seg, uni_log, bg, grams, VOCAB)
        res = [live(o, seg, emb, ch) for o in pop]
        lives = np.array([r[0] for r in res]); Ms = np.array([r[1] for r in res]); bi = int(np.argmax(lives))
        pop = reproduce(pop, lives, rng)
    oacc, bacc = recall_acc(pop[bi], B, D, emb_table, uni_log, bg, grams, rng)
    return int(Ms[bi]), float(Ms.mean()), oacc, bacc


if __name__ == "__main__":
    log("MEMORY-DEMANDING WORLD: dense recalls, tight energy. Can evolution solve binding at all (B=1)?")
    log(f"{'B':>2} {'D':>5} | {'best_M':>6} {'avg_M':>6} | {'organism recall':>15} {'4gram recall':>12}")
    log("=" * 60)
    log("-- B=3 (interleaved, out-of-window): distance sweep -> does M climb, does organism beat 4g? --")
    for D in [300, 700, 1300]:
        m, am, oa, ba = run_cell(3, D, gens=200)
        log(f"{3:>2} {D:>5} | {m:>6} {am:>6.1f} | {oa:>15.2f} {ba:>12.2f}")
    log("-- binding-count sweep (D=700) --")
    for B in [1, 2]:
        m, am, oa, ba = run_cell(B, 700, gens=200)
        log(f"{B:>2} {700:>5} | {m:>6} {am:>6.1f} | {oa:>15.2f} {ba:>12.2f}")
    log("=" * 60)
    log("SUCCESS = organism recall >> 4gram recall (memory does what n-gram can't), AND best_M climbs with D.")
    log("done")
