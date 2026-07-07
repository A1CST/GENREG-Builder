"""EEC Track B — the CONVERSATIONAL memory environment.

The static-corpus worlds were local: nothing referenced anything far away, so a
survival organism grew ~1 step of memory (P1 — the world must pay). Conversation
is different: a reply depends on the earlier turn, so to anticipate the next token
an organism must HOLD the conversation state across turn boundaries. If that
cross-turn dependency is real and exploitable, memory finally pays and evolves.

Same EEC paradigm as eec_memory.py: survival = lifespan on energy (prediction =
metabolism, miss burns energy), memory-rent, graded by emergent STATE (recurrent
gain, horizon, ablation) — never accuracy. Corpus: the Cornell Movie-Dialogs turn
stream (project/conversational/conversations.txt), turns marked by __eou__.

MAKE-OR-BREAK first: `cross_turn_diagnostic` measures whether remembering the
prompt actually predicts the response (real vs a shuffled-prompt control) BEFORE
any evolution run. If the corpus has no exploitable cross-turn signal, we stop —
just like the topic-memory idea died on the static corpus.
"""
import collections
import os

import numpy as np

from genreg_train import eec_memory as eec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONVO_PATH = os.path.join(ROOT, "project", "conversational", "conversations.txt")

# common function words to ignore when measuring lexical cross-turn reference
_STOP = set("the a an of to and in is it you that he she was for on are with as i "
            "his her they be this have from at not but had by we do what all one there "
            "were so my me your him them then me s t re ll m d ve".split())


def _read_dialogues(max_dialogues=None):
    with open(CONVO_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if max_dialogues:
        lines = lines[:max_dialogues]
    return [ln.split(" __eou__ ") for ln in lines if ln.strip()]


# --------------------------------------------------------------------------
# MAKE-OR-BREAK diagnostic: does remembering the prompt predict the response?
# --------------------------------------------------------------------------
def cross_turn_diagnostic(max_dialogues=40000, seed=0, log=print):
    """For adjacent (prompt, response) turns, fraction of the response's CONTENT
    words that already appeared in the prompt — real pairing vs a shuffled
    (random-prompt) control. real >> shuffled => cross-turn memory pays."""
    dias = _read_dialogues(max_dialogues)
    prompts, responses = [], []
    for turns in dias:
        for i in range(len(turns) - 1):
            p = set(w for w in turns[i].split() if w not in _STOP and len(w) > 2)
            r = [w for w in turns[i + 1].split() if w not in _STOP and len(w) > 2]
            if p and r:
                prompts.append(p); responses.append(r)

    def overlap(prompt_sets):
        hit = tot = 0
        for ps, r in zip(prompt_sets, responses):
            for w in r:
                tot += 1; hit += (w in ps)
        return hit / max(1, tot)

    real = overlap(prompts)
    rng = np.random.default_rng(seed)
    shuf = overlap([prompts[i] for i in rng.permutation(len(prompts))])
    log(f"cross-turn lexical reference: REAL {real:.4f} | shuffled-prompt {shuf:.4f} "
        f"| lift x{real / max(shuf, 1e-9):.1f}")
    log(f"  ({len(responses):,} turn pairs). memory pays if REAL >> shuffled.")
    return {"real": round(real, 4), "shuffled": round(shuf, 4),
            "lift": round(real / max(shuf, 1e-9), 2), "pairs": len(responses)}


# --------------------------------------------------------------------------
# Conversational token stream (word-level, __eou__ as its own token)
# --------------------------------------------------------------------------
def build_convo_stream(vocab_n=2000, max_dialogues=None):
    dias = _read_dialogues(max_dialogues)
    counts = collections.Counter()
    for turns in dias:
        for t in turns:
            counts.update(w for w in t.split())
    top = [w for w, _ in counts.most_common(vocab_n)]
    vocab = ["<unk>", "__eou__"] + [w for w in top if w != "__eou__"]
    stoi = {w: i for i, w in enumerate(vocab)}
    eou = stoi["__eou__"]
    parts = []
    for turns in dias:
        toks = (" __eou__ ".join(turns)).split()
        parts.append(np.fromiter((stoi.get(w, 0) for w in toks), np.int64, len(toks)))
        parts.append(np.array([eou], np.int64))
    return np.concatenate(parts), vocab, stoi, eou


def turn_shuffled(stream, eou, seed=0):
    """Control: shuffle the ORDER of turns (destroy cross-turn dependency) while
    preserving within-turn token structure. Memory should NOT emerge here."""
    rng = np.random.default_rng(seed)
    bounds = np.where(stream == eou)[0]
    turns, prev = [], 0
    for b in bounds:
        turns.append(stream[prev:b + 1]); prev = b + 1
    if prev < len(stream):
        turns.append(stream[prev:])
    rng.shuffle(turns)
    return np.concatenate(turns)


# --------------------------------------------------------------------------
# The experiment: evolve on the real convo stream vs the turn-shuffled control,
# read the state (gain / horizon) + ablation. Reuses the eec_memory substrate.
# --------------------------------------------------------------------------
def run_convo_memory(vocab_n=2000, max_dialogues=40000, gens=250, pop=140,
                     seg=600, start_energy=60, rent=0.008, decay=0.9, seed=0,
                     log=print):
    stream, vocab, stoi, eou = build_convo_stream(vocab_n, max_dialogues)
    V = len(vocab)
    log(f"convo stream: {len(stream):,} tokens, vocab {V}, eou-rate {(stream==eou).mean():.3f}")
    out = {}
    for tag, s in (("REAL", stream), ("TURN-SHUFFLED", turn_shuffled(stream, eou, seed))):
        log(f"=== {tag} ===")
        r = eec.evolve_world(s, V, rho=0.0, burst=1, gens=gens, pop=pop, seg=seg,
                             start_energy=start_energy, rent=rent, decay=decay,
                             tag=tag, seed=seed, log=log)
        mi = eec.measure_internals(r["champ"], s, V, decay, np.random.default_rng(seed + 50))
        ab = eec.ablation_test({**r, "rho": 0.0, "burst": 1}, log=log)
        out[tag] = {**mi, **ab}
        log(f"  {tag}: gain {mi['gain']} horizon {mi['horizon']} M {mi['M']} "
            f"ablation-drop {ab['survival_drop']:.1%}")
    log("MEMORY EMERGES if REAL shows higher gain/horizon AND a larger ablation drop "
        "than TURN-SHUFFLED.")
    return out
