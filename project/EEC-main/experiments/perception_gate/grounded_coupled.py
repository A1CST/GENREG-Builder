"""THE CONVERGENCE: English-grounded coupled communication. Unifies the three proven pieces.

  1. DISTRIBUTIONAL EXPERIENCE  -> both organisms perceive through FROZEN English PPMI-SVD embeddings
     (real word meanings learned from a real corpus; the perception that hit 0.72 on real text).
  2. COUPLED SURVIVAL           -> a LISTENER predicts the speaker's next word through those English
     embeddings; it can only predict English-shaped speech, so un-English speech starves it and both die.
  3. EVOLVED MEMORY GENERATION  -> the speaker generates from its own recurrent state (no channel),
     memory is the generator.

Plus the laws (no formulas): SCARCITY (words deplete/regen -> variety) and OUTPUT COST (speech drains).
Because the listener is grounded in English, "be understood" = "speak English-shaped, coherent words" --
so the speaker's memory-driven generation is pulled toward real English, with nothing grading it but the
listener's survival. Everything evolvable (ED<=EDIM, M<=256, decay, mutation, temp); caps raised."""
import os, sys, math, re
import numpy as np
from collections import Counter
from sklearn.decomposition import TruncatedSVD
ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "engine")
sys.path.insert(0, ENGINE)
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
DIALOG = [os.path.join(HERE, "emotional_dialog.txt"), os.path.join(os.path.dirname(HERE), "english_comm", "chat_corpus.txt")]
VOC, EDIM, MIN_M, INIT_M = 600, 256, 2, 48                     # English vocab; grounding dim; NO memory cap (M grows)
L = 500; START_E = 28.0; GENS = int(os.environ.get("EEC_GLGENS", "300"))
OUTPUT_COST = 0.25; SC, STOCK0, USE, REGEN = 1.6, 2.0, 1.0, 0.08   # scarcity strong: a depleted word costs
# more than the understanding income, so repetition starves -> only VARIED-but-predictable speech survives.
RENT = 0.002                                                   # memory rent / unit / step -- the ONLY thing bounding M
UGAIN, UMISS, BASE_L = 1.0, 0.1, 0.15; TAU = 0.25
N_BOTTOM = round(0.2 * POP_SIZE); N_ELITE = round(0.2 * POP_SIZE)
def log(s): print(s, flush=True)


def build_english():
    text = []
    for p in DIALOG:
        if os.path.exists(p): text.append(open(p, encoding="utf-8", errors="ignore").read())
    toks = tokenize("\n".join(text))
    vocab = ["<unk>"] + [w for w, _ in Counter(toks).most_common(VOC - 1)]
    w2i = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
    ids = np.array([w2i.get(t, 0) for t in toks], dtype=np.int64)
    co = np.zeros(V * V, np.float64)                                                   # co-occurrence (window 5)
    for d in range(1, 6):
        co += np.bincount(ids[:-d] * V + ids[d:], minlength=V * V); co += np.bincount(ids[d:] * V + ids[:-d], minlength=V * V)
    co = co.reshape(V, V); tot = co.sum(); rs = co.sum(1, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log(co * tot / (rs @ rs.T) + 1e-12), 0)
    E = TruncatedSVD(n_components=EDIM, random_state=0).fit_transform(ppmi)
    E = (E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)).astype(np.float32)     # FROZEN English perception
    return E, vocab, V


class Org:
    """Perception (E) is the FROZEN shared English grounding. Memory M is UNBOUNDED: the weight arrays
    GROW when M mutates up (new units born small/random) and shrink when it mutates down. There is NO
    capacity ceiling -- memory rent (energy/unit/step) is the only thing bounding size. Everything evolves."""
    def __init__(self, V, rng):
        self.V = V; self.ED = int(rng.integers(8, EDIM + 1)); self.M = int(rng.integers(MIN_M, INIT_M + 1))
        self.decay = float(rng.uniform(0.5, 0.98)); self.mr = float(rng.uniform(0.05, 0.3))
        self.ms = float(rng.uniform(0.1, 0.4)); self.temp = float(rng.uniform(0.4, 1.1))
        M = self.M
        self.W_in = rng.normal(0, 0.2, (EDIM, M)).astype(np.float32)
        self.W_rec = rng.normal(0, 1/np.sqrt(M), (M, M)).astype(np.float32)
        self.b = np.zeros(M, np.float32); self.W_out = rng.normal(0, 1/np.sqrt(M), (M, V)).astype(np.float32)
        self.b_out = np.zeros(V, np.float32)

    def step(self, e_in, s):                                                            # arrays are exactly M-sized
        s2 = np.tanh(e_in[:self.ED] @ self.W_in[:self.ED, :] + self.decay * (s @ self.W_rec) + self.b)
        return s2, s2 @ self.W_out + self.b_out

    def copy(self):
        g = Org.__new__(Org); g.V = self.V
        for k in ("ED", "M", "decay", "mr", "ms", "temp"): setattr(g, k, getattr(self, k))
        g.W_in = self.W_in.copy(); g.W_rec = self.W_rec.copy(); g.b = self.b.copy()
        g.W_out = self.W_out.copy(); g.b_out = self.b_out.copy(); return g

    def _resize(self, newM, rng):                                                       # grow / shrink the memory
        M, V = self.M, self.V
        if newM > M:
            d = newM - M
            self.W_in = np.concatenate([self.W_in, rng.normal(0, 0.2, (EDIM, d)).astype(np.float32)], axis=1)
            Wr = rng.normal(0, 1/np.sqrt(newM), (newM, newM)).astype(np.float32); Wr[:M, :M] = self.W_rec; self.W_rec = Wr
            self.b = np.concatenate([self.b, np.zeros(d, np.float32)])
            self.W_out = np.concatenate([self.W_out, rng.normal(0, 1/np.sqrt(newM), (d, V)).astype(np.float32)], axis=0)
        elif newM < M:
            self.W_in = self.W_in[:, :newM].copy(); self.W_rec = self.W_rec[:newM, :newM].copy()
            self.b = self.b[:newM].copy(); self.W_out = self.W_out[:newM, :].copy()
        self.M = newM

    def mutate(self, rng):
        self.mr = float(np.clip(self.mr * math.exp(TAU * rng.normal()), 0.01, 0.6))
        self.ms = float(np.clip(self.ms * math.exp(TAU * rng.normal()), 0.02, 0.8))
        self._resize(max(MIN_M, int(round(self.M + rng.normal(0, self.ms * self.M)))), rng)   # M has NO upper cap
        for p in (self.W_in, self.W_rec, self.b, self.W_out, self.b_out):
            m = rng.random(p.shape) < self.mr
            p += m * rng.normal(0, 1, p.shape).astype(np.float32) * (self.ms * (np.abs(p) + 1e-3))
        self.ED = int(np.clip(round(self.ED + rng.normal(0, self.ms * self.ED)), 8, EDIM))
        self.decay = float(np.clip(self.decay + rng.normal(0, 0.05), 0.3, 0.995))
        self.temp = float(np.clip(self.temp + rng.normal(0, 0.1), 0.2, 1.5))


def couple_life(S, Li, E, V, rng, record=False):
    sS = np.zeros(S.M, np.float32); sL = np.zeros(Li.M, np.float32)                      # state sized to each genome's M
    eS = eL = START_E; stock = np.full(V, STOCK0, np.float32)
    rentS = RENT * S.M; rentL = RENT * Li.M                                              # bigger memory = more rent
    x = int(rng.integers(V)); pred = -1; out = []; correct = npred = 0
    for t in range(L):
        sS, lS = S.step(E[x], sS)
        p = lS / S.temp; p = np.exp(p - p.max()); p /= p.sum(); y = int(rng.choice(V, p=p))
        eS -= OUTPUT_COST + SC * max(0.0, 1.0 - stock[y]) + rentS
        stock[y] = max(0.0, stock[y] - USE); stock = np.minimum(STOCK0, stock + REGEN)
        if pred >= 0:
            npred += 1
            if pred == y: eL += UGAIN; eS += UGAIN; correct += 1
            else: eL -= UMISS
        eL -= BASE_L + rentL
        sL, lL = Li.step(E[y], sL); pred = int(lL.argmax())
        out.append(y); x = y
        if eS <= 0 or eL <= 0: break
    return (t + 1, out, correct / max(1, npred)) if record else (t + 1)


def evaluate(pop, E, V, rng, rounds=3):
    fit = np.zeros(POP_SIZE)
    for _ in range(rounds):
        order = rng.permutation(POP_SIZE)
        for i in range(0, POP_SIZE - 1, 2):
            a, b = int(order[i]), int(order[i + 1])
            life = couple_life(pop[a], pop[b], E, V, rng); fit[a] += life; fit[b] += life
    return fit / rounds


def reproduce(pop, fits, rng):
    order = np.argsort(fits)[::-1]; new = [pop[i] for i in order[:POP_SIZE - N_BOTTOM]]; top = order[:N_ELITE]
    while len(new) < POP_SIZE:
        c = pop[int(top[rng.integers(0, len(top))])].copy(); c.mutate(rng); new.append(c)
    return new


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    log("building FROZEN English perception (PPMI-SVD over real dialogue)...")
    E, vocab, V = build_english()
    log(f"English grounding: V={V}, EDIM={EDIM}. coupled+scarcity+output-cost, M UNBOUNDED (rent={RENT}), gens={GENS}.")
    pop = [Org(V, rng) for _ in range(POP_SIZE)]
    for gen in range(1, GENS + 1):
        fits = evaluate(pop, E, V, rng); bi = int(np.argmax(fits))
        if gen == 1 or gen % 30 == 0 or gen == GENS:
            top = list(np.argsort(fits)[::-1]); S = pop[top[0]]; Li = pop[top[1]]
            life, out, acc = couple_life(S, Li, E, V, np.random.default_rng(7), record=True)
            text = " ".join(vocab[t] for t in out[:42])
            log(f"gen {gen:>4} | couple-life {fits[bi]:>5.1f}/{L} | understanding {acc:.2f} | M {S.M} ED {S.ED} "
                f"temp {S.temp:.2f} | distinct {len(set(out))}")
            if gen == 1 or gen >= GENS - 1: log(f'        speaker: "{text}"')
        pop = reproduce(pop, fits, rng)
    log("\nSUCCESS = understanding climbs (the English-grounded listener follows the speaker) AND the speaker's")
    log("output reads as English-shaped, coherent words -- because that is the only thing the listener can")
    log("understand. Experience grounds it, coupled survival makes it load-bearing, memory generates it.")
    log("done")
