"""HELLO -- organism-to-HUMAN communication. The GPT-2 moment: prove an evolved
organism learns to talk to a person, not co-evolve a private code with a partner.

An organism lives in a turn-based dialogue with a FIXED rule-based human-proxy.
The proxy opens with a greeting and emits prompts from a tiny grammar; the organism
emits a response word each turn. A COHERENT response (greeting answered with a
greeting, "how" with "good", etc.) keeps the conversation alive (+energy); an
incoherent one makes the proxy disengage (-energy). When energy runs out the
conversation dies. Nothing labels any word "correct" -- the only signal is that
CONTINUED INTERACTION = SURVIVAL. Selection is conversation length.

No architecture for language, no examples, no reward for "good" replies, no
gradient. If the organism learns to answer HELLO with a greeting because that is
how it keeps existing, that is communication with a human from first principles.

We read: protocol mastery (accepted-response rate), the LEARNED PHRASEBOOK (what it
says to each prompt), a transcript, and a SEVERED control (deaf organism -> dies).
"""
# --- EEC path bootstrap ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))

import os
import numpy as np
from evolve import MUT_RATE, MUT_SCALE, POP_SIZE
from mind import reproduce

GENS = int(os.environ.get("EEC_MGENS", "150"))
SEEDS = list(range(int(os.environ.get("EEC_SEEDS", "5"))))
E0, TMAX = 15.0, 50

# ---- the shared "human" vocabulary ----
VOCAB = ["HELLO", "HI", "HOW", "GOOD", "NAME", "BOT", "BYE", "SEEYA",
         "THANKS", "WELCOME", "YES", "NO", "uh", "blah", "zzz", "??"]
IX = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)

# ---- the human-proxy grammar: prompt -> set of ACCEPTED responses ----
ACCEPT = {
    IX["HELLO"]:  {IX["HI"], IX["HELLO"]},     # greet back
    IX["HOW"]:    {IX["GOOD"]},                # "how are you" -> good
    IX["NAME"]:   {IX["BOT"]},                 # "your name?" -> bot
    IX["THANKS"]: {IX["WELCOME"]},             # thanks -> you're welcome
    IX["BYE"]:    {IX["SEEYA"], IX["BYE"]},    # farewell
}
PROMPTS = list(ACCEPT.keys())
CHANCE = np.mean([len(ACCEPT[p]) for p in PROMPTS]) / V   # random-response accept rate


class Speaker:
    """A learnable phrasebook: prompt -> response logits (V x V)."""
    def __init__(self, rng):
        self.W = rng.normal(0, 1.0, (V, V)).astype(np.float32)

    def copy(self):
        g = Speaker.__new__(Speaker); g.W = self.W.copy(); return g

    def mutate(self, rng):
        m = rng.random(self.W.shape) < MUT_RATE
        self.W += m * rng.normal(0, 1, self.W.shape).astype(np.float32) * (MUT_SCALE * (np.abs(self.W) + 0.2))

    def reply(self, prompt):
        return int(self.W[prompt].argmax())


def converse(org, rng, deaf=False):
    """One dialogue. Returns (turns_survived, accepted_count). Always opens HELLO."""
    e = E0; prompt = IX["HELLO"]; turns = 0; ok = 0
    while e > 0 and turns < TMAX:
        seen = rng.integers(0, V) if deaf else prompt      # severed: organism can't hear the prompt
        r = org.reply(seen)
        if r in ACCEPT[prompt]:
            e += 1.0; ok += 1
        else:
            e -= 2.0
        turns += 1
        prompt = PROMPTS[rng.integers(len(PROMPTS))]
    return turns, ok


def fitness(org, rng):
    t, ok = converse(org, rng)
    return t + 0.01 * ok        # length is survival; tiny tie-break by coherence


def evolve(seed):
    rng = np.random.default_rng(seed)
    pop = [Speaker(rng) for _ in range(POP_SIZE)]
    for _ in range(GENS):
        fits = np.array([fitness(m, np.random.default_rng(seed * 7919 + _)) for m in pop])
        pop = reproduce(pop, fits, rng)
    return max(pop, key=lambda m: fitness(m, np.random.default_rng(seed + 12345)))


def mastery(org):
    """fraction of prompts answered with an ACCEPTED response (deterministic phrasebook)."""
    return float(np.mean([org.reply(p) in ACCEPT[p] for p in PROMPTS]))


def transcript(org):
    rng = np.random.default_rng(7)
    lines = []; e = E0; prompt = IX["HELLO"]; turns = 0
    while e > 0 and turns < 12:
        r = org.reply(prompt)
        good = r in ACCEPT[prompt]
        lines.append(f"   HUMAN: {VOCAB[prompt]:<8}   ORGANISM: {VOCAB[r]:<8}  {'[ok]' if good else '[proxy disengages]'}")
        e += 1.0 if good else -2.0
        turns += 1; prompt = PROMPTS[rng.integers(len(PROMPTS))]
    return lines


def main():
    print(f"HELLO experiment: vocab={V} words, {len(PROMPTS)} exchange types, "
          f"random-reply accept rate={CHANCE:.3f}  (gens={GENS})\n")
    best, bestm = None, -1
    for seed in SEEDS:
        org = evolve(seed)
        m = mastery(org)
        t_alive, _ = converse(org, np.random.default_rng(seed + 1))
        t_deaf, _ = converse(org, np.random.default_rng(seed + 1), deaf=True)
        print(f"  seed {seed}: protocol mastery {m*100:5.1f}%  | conversation {t_alive:>2}/{TMAX} turns  "
              f"| SEVERED (deaf) {t_deaf:>2}/{TMAX} turns  | greets HELLO with '{VOCAB[org.reply(IX['HELLO'])]}'")
        if m > bestm:
            best, bestm = org, m

    print(f"\n===== BEST ORGANISM: learned phrasebook (mastery {bestm*100:.0f}%) =====")
    for p in PROMPTS:
        r = best.reply(p)
        print(f"   hears '{VOCAB[p]:<7}' -> says '{VOCAB[r]:<8}'  {'OK' if r in ACCEPT[p] else 'x'}")
    print("\n===== a conversation with the organism =====")
    for ln in transcript(best):
        print(ln)
    print(f"\nGreeting check: when the human says HELLO, the organism says "
          f"'{VOCAB[best.reply(IX['HELLO'])]}'.  Severing the channel (deaf) collapses it to chance.")


if __name__ == "__main__":
    main()
