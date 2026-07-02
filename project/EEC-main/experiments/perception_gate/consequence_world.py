"""CONSEQUENCE / ENVIRONMENT COUPLING -- the law that makes GENERATION the survival task.

The organism generates a stream autoregressively: its output at step T becomes its input at step T+1.
No channel -- it generates from its own recurrent state, so memory IS the generator. Survival depends on
the COHERENCE of what it generates: the world rewards staying on the primed topic with varied content,
PACED by neutral words. Because a neutral word is topic-ambiguous, the organism must RECALL the topic
through the gap to keep producing on-topic -> memory is required to generate coherently. Incoherent
streams (topic-drift, repetition) drain energy and die. The structure lives in the FITNESS, never in the
generation pathway -- it cannot mask memory the way a channel does.

EVERYTHING the organism uses is a GENE: embedding dim ED, memory M, its own decay, mutation rate+scale
(self-adapting), generation temperature -- plus all weights. The world sets only the laws."""
import os, sys, math
import numpy as np
from collections import deque
ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "engine")
sys.path.insert(0, ENGINE)
from evolve import POP_SIZE                                    # world carrying capacity (a population law, not an organism knob)

NEUTRAL = "i you the to a it is we do so today really just and".split()
TOPICS = {"FOOD": "eat hungry lunch dinner pizza cook".split(), "WEATHER": "rain cold sunny snow warm storm".split(),
          "WORK": "job boss meeting office email deadline".split(), "SLEEP": "tired sleep bed nap rest dream".split(),
          "SPORT": "game gym ball team play score".split(), "MUSIC": "song band guitar sing concert album".split()}
TN = list(TOPICS); WORDS = NEUTRAL + [w for t in TN for w in TOPICS[t]]
V = len(WORDS); w2i = {w: i for i, w in enumerate(WORDS)}
NEU_IDS = set(range(len(NEUTRAL))); TOPIC_OF = {w2i[w]: t for t in TN for w in TOPICS[t]}
MAX_ED, MAX_M = 256, 128                                       # raised: organism hit 64/44; let energy/rent set the real size
L = 600; START_E = 30.0; GENS = int(os.environ.get("EEC_QGENS", "400"))   # L raised to un-saturate survival (was 300/300)
# the world's energy ECONOMY: income ONLY from good paced varied on-topic content; silence/drift just
# pay BASE metabolism (-> starve). no penalty for *trying* topical content (off-topic costs nothing extra),
# so partial coherence beats silence and there is a RAMP from silence up to coherent generation.
REWARD, REP, SHIFT, BASE = -2.5, 0.6, 0.3, 0.5
# coherence = SELF-CONSISTENCY: income for maintaining your OWN current topic across a neutral gap (varied,
# not repeated); deliberately shifting topic costs a little. No external target -> the organism must recall
# the topic IT is generating through the ambiguous neutral words = memory is the only way to hold the thread.
N_BOTTOM = round(0.2 * POP_SIZE); N_ELITE = round(0.2 * POP_SIZE); TAU = 0.25
def log(s): print(s, flush=True)


class Org:
    def __init__(self, rng):
        self.ED = int(rng.integers(4, MAX_ED + 1)); self.M = int(rng.integers(2, MAX_M + 1))
        self.decay = float(rng.uniform(0.5, 0.98)); self.mr = float(rng.uniform(0.05, 0.3))
        self.ms = float(rng.uniform(0.1, 0.4)); self.temp = float(rng.uniform(0.4, 1.1))
        self.E = rng.normal(0, 0.1, (V, MAX_ED)).astype(np.float32)
        self.W_in = rng.normal(0, 0.2, (MAX_ED, MAX_M)).astype(np.float32)
        self.W_rec = rng.normal(0, 1/np.sqrt(MAX_M), (MAX_M, MAX_M)).astype(np.float32)
        self.b = np.zeros(MAX_M, np.float32); self.W_out = rng.normal(0, 1/np.sqrt(MAX_M), (MAX_M, V)).astype(np.float32)
        self.b_out = np.zeros(V, np.float32)

    def weights(self): return [self.E, self.W_in, self.W_rec, self.b, self.W_out, self.b_out]

    def step(self, x, s):
        ed, M = self.ED, self.M
        s = np.tanh(self.E[x, :ed] @ self.W_in[:ed, :M] + self.decay * (s[:M] @ self.W_rec[:M, :M]) + self.b[:M])
        out = np.zeros(MAX_M, np.float32); out[:M] = s
        return out, s @ self.W_out[:M, :] + self.b_out

    def copy(self):
        g = Org.__new__(Org)
        for k in ("ED", "M", "decay", "mr", "ms", "temp"): setattr(g, k, getattr(self, k))
        g.E, g.W_in, g.W_rec, g.b, g.W_out, g.b_out = [p.copy() for p in self.weights()]
        return g

    def mutate(self, rng):
        self.mr = float(np.clip(self.mr * math.exp(TAU * rng.normal()), 0.01, 0.6))    # self-adaptive
        self.ms = float(np.clip(self.ms * math.exp(TAU * rng.normal()), 0.02, 0.8))
        for p in self.weights():
            m = rng.random(p.shape) < self.mr
            p += m * rng.normal(0, 1, p.shape).astype(np.float32) * (self.ms * (np.abs(p) + 1e-3))
        self.ED = int(np.clip(round(self.ED + rng.normal(0, self.ms * self.ED)), 4, MAX_ED))
        self.M = int(np.clip(round(self.M + rng.normal(0, self.ms * self.M)), 2, MAX_M))
        self.decay = float(np.clip(self.decay + rng.normal(0, 0.05), 0.3, 0.995))
        self.temp = float(np.clip(self.temp + rng.normal(0, 0.1), 0.2, 1.5))


def live(org, rng, record=False):
    s = np.zeros(MAX_M, np.float32); x = w2i[NEUTRAL[0]]
    tw = [w2i[w] for w in TOPICS[TN[int(rng.integers(len(TN)))]]]
    for seed in (w2i[NEUTRAL[0]], int(rng.choice(tw))): s, _ = org.step(seed, s); x = seed
    energy = START_E; recent = deque(maxlen=3); prev_neu = True; out = []; running = None
    for t in range(L):
        s, logits = org.step(x, s)
        p = logits / org.temp; p = np.exp(p - p.max()); p /= p.sum(); y = int(rng.choice(V, p=p))
        cost = BASE
        if y in NEU_IDS:
            prev_neu = True
        else:
            ty = TOPIC_OF[y]
            if running is None: running = ty
            if ty != running: cost += SHIFT; running = ty                                # deliberate topic shift
            elif prev_neu and y not in recent: cost += REWARD                            # held my own topic through the gap
            prev_neu = False
        energy -= cost; recent.append(y); out.append(y); x = y                          # CONSEQUENCE: output -> input
        if energy <= 0: break
    return (t + 1, out) if record else (t + 1)


def fitness(org, rng): return float(np.mean([live(org, rng) for _ in range(3)]))


def reproduce(pop, fits, rng):
    order = np.argsort(fits)[::-1]; new = [pop[i] for i in order[:POP_SIZE - N_BOTTOM]]; top = order[:N_ELITE]
    while len(new) < POP_SIZE:
        c = pop[int(top[rng.integers(0, len(top))])].copy(); c.mutate(rng); new.append(c)
    return new


def show(org, rng):
    life, out = live(org, rng, record=True)
    words = [WORDS[t] for t in out[:44]]
    rend = " ".join((TOPIC_OF[w2i[w]][:3] + ":" + w if w not in NEUTRAL else w) for w in words)
    running = None; pn = True; held = ngap = 0
    for t in out:
        if t in NEU_IDS: pn = True
        else:
            ty = TOPIC_OF[t]
            if running is None: running = ty
            if pn: ngap += 1; held += (ty == running)                                   # maintained own topic across gap?
            if ty != running: running = ty
            pn = False
    return life, rend, (held / max(1, ngap))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    log(f"CONSEQUENCE world: generate-to-survive, memory is the generator, everything evolvable. "
        f"V={V}, {len(TN)} topics, gens={GENS}, pop={POP_SIZE}.")
    pop = [Org(rng) for _ in range(POP_SIZE)]
    for gen in range(1, GENS + 1):
        fits = np.array([fitness(o, rng) for o in pop]); bi = int(np.argmax(fits)); b = pop[bi]
        if gen == 1 or gen % 50 == 0 or gen == GENS:
            life, rend, held = show(b, np.random.default_rng(7))
            log(f"gen {gen:>4} | best life {fits[bi]:>5.0f}/{L} | M {b.M:>2} ED {b.ED:>2} decay {b.decay:.2f} "
                f"mr {b.mr:.2f} temp {b.temp:.2f} | held-own-topic-through-gaps {held:.2f}")
            if gen == 1 or gen >= GENS - 1: log(f"        {rend}")
        pop = reproduce(pop, fits, rng)
    log("\nSUCCESS = best life climbs AND held-own-topic-through-gaps >> chance (1/6=0.17): the organism")
    log("GENERATES self-consistent topical text from its own memory, holding its thread across neutral gaps.")
    log("done")
