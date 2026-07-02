"""COUPLED communication world -- formulas replaced by LAWS OF EXISTENCE.

No grammar judge, no coherence reward formula, no shift penalty, no variety clause. Instead:
  - LISTENER organism: reads the speaker's stream and predicts its next token. It survives only while it
    can predict (understand). Coherence/grammar become load-bearing because un-understandable speech
    starves the listener -- discovered through death, not graded by a formula.
  - COUPLED survival: speaker is rewarded only when understood (listener predicted it); if EITHER dies the
    life ends. Both survive or both die.
  - SCARCITY: each word depletes when used and regenerates slowly. Overusing a word makes it expensive ->
    variety emerges from depletion, not a no-repeat clause.
  - OUTPUT COST: every spoken token drains energy. Silence is cheaper than speech -> speak only when
    being understood pays for the words.
Same fully-evolvable organism (raised caps: ED to 256, M to 128). Both roles use the same genome, so the
population co-evolves a shared protocol: the speaker must be predictable-but-varied (= structure), and
the listener must learn to predict it. Fitness = how long the COUPLE survives."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from consequence_world import Org, MAX_ED, MAX_M, V, WORDS, w2i, TOPIC_OF, NEU_IDS, TN, reproduce, POP_SIZE

START_E = 28.0; L = 500; GENS = int(os.environ.get("EEC_LGENS", "350"))
OUTPUT_COST = 0.25                                             # speaking drains energy (silence cheaper)
SC, STOCK0, USE, REGEN = 0.6, 3.0, 1.0, 0.12                   # scarcity: deplete on use, slow regen
UGAIN, UMISS, BASE_L = 1.0, 0.1, 0.15                          # understood -> both gain; listener metabolism
def log(s): print(s, flush=True)


def couple_life(S, Li, rng, record=False):
    sS = np.zeros(MAX_M, np.float32); sL = np.zeros(MAX_M, np.float32)
    eS = eL = START_E; stock = np.full(V, STOCK0, np.float32)
    x = int(rng.integers(V)); pred = -1; out = []; correct = npred = 0
    for t in range(L):
        sS, lS = S.step(x, sS)
        p = lS / S.temp; p = np.exp(p - p.max()); p /= p.sum(); y = int(rng.choice(V, p=p))
        eS -= OUTPUT_COST + SC * max(0.0, 1.0 - stock[y])                                # output cost + scarcity
        stock[y] = max(0.0, stock[y] - USE); stock = np.minimum(STOCK0, stock + REGEN)
        if pred >= 0:                                                                    # score listener's last guess
            npred += 1
            if pred == y: eL += UGAIN; eS += UGAIN; correct += 1                         # understood -> BOTH gain
            else: eL -= UMISS
        eL -= BASE_L
        sL, lL = Li.step(y, sL); pred = int(lL.argmax())                                 # listener predicts speaker's next
        out.append(y); x = y                                                             # consequence: output -> input
        if eS <= 0 or eL <= 0: break                                                     # COUPLED: either dies, life ends
    return (t + 1, out, correct / max(1, npred)) if record else (t + 1)


def evaluate(pop, rng, rounds=3):
    fit = np.zeros(POP_SIZE)
    for _ in range(rounds):
        order = rng.permutation(POP_SIZE)
        for i in range(0, POP_SIZE - 1, 2):
            a, b = int(order[i]), int(order[i + 1])
            life = couple_life(pop[a], pop[b], rng)
            fit[a] += life; fit[b] += life
    return fit / rounds


def render(out):
    return " ".join((TOPIC_OF[t][:3] + ":" + WORDS[t] if t not in NEU_IDS else WORDS[t]) for t in out[:48])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    log(f"COUPLED world: speaker+listener, scarcity, output cost, NO reward formulas. V={V}, pop={POP_SIZE}, "
        f"L={L}, gens={GENS}. fitness = couple lifespan.")
    pop = [Org(rng) for _ in range(POP_SIZE)]
    for gen in range(1, GENS + 1):
        fits = evaluate(pop, rng); bi = int(np.argmax(fits))
        if gen == 1 or gen % 40 == 0 or gen == GENS:
            top = list(np.argsort(fits)[::-1]); S = pop[top[0]]; Li = pop[top[1]]
            life, out, acc = couple_life(S, Li, np.random.default_rng(7), record=True)
            sw = sorted(set(out), key=lambda t: -out.count(t))[:8]; nuniq = len(set(out))
            log(f"gen {gen:>4} | couple-life {fits[bi]:>5.1f}/{L} | understanding {acc:.2f} | speaker M {S.M} ED {S.ED} "
                f"temp {S.temp:.2f} | distinct words used {nuniq}")
            if gen == 1 or gen >= GENS - 1: log(f"        speaker: {render(out)}")
        pop = reproduce(pop, fits, rng)
    log("\nSUCCESS = couple-life climbs (mutual understanding sustains) AND understanding >> chance (1/V): the")
    log("speaker generates predictable-but-VARIED structure that an evolved listener learns to follow.")
    log("No formula graded it -- structure emerged because un-understood speech starved the listener.")
    log("done")
