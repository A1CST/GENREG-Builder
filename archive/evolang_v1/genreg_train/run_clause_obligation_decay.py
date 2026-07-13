"""Clause-obligation tracker, DECAY version — follow-up to run_clause_obligation.py
(cut: flat-gamma suppression reduced never-closed-relative-rate only 13.1%->12.5%,
plateauing immediately, while making dangling-rate worse — samples showed some
sentences running very long, suggesting flat suppression holds boundary probability
down too aggressively once triggered, forcing run-ons instead of letting a
genuinely-unresolved clause end naturally).

This version decays the suppression with word-distance since the obligation opened:
full strength right after a relative pronoun/subordinator appears (give the clause a
real chance to resolve in the next few words), fading toward zero if it hasn't closed
soon (so an unresolved clause doesn't force an indefinitely long run-on — better to
let the sentence end than keep suppressing forever). Sweeps GAMMA (peak suppression
strength) x TAU (decay time constant, in words) against the same battery + the
never-closed-relative-rate metric. Runs on the I2 primary.
"""
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if "genreg_train" not in sys.modules:
    _pkg = types.ModuleType("genreg_train")
    _pkg.__path__ = [os.path.join(ROOT, "genreg_train")]
    sys.modules["genreg_train"] = _pkg

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "clause_obligation_decay.log")
open(LOG, "w").close()
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    open(LOG, "a", encoding="utf-8").write(line + "\n")

from genreg_train import evolang
evolang.CORPUS_PATH = os.path.join(ROOT, "corpora", "combined", "combined_corpus.txt")
evolang._IDS = None
if not os.path.exists(evolang.CORPUS_PATH):
    log("FATAL: combined corpus missing"); sys.exit(1)

import numpy as np
from genreg_train import wordpipe_service as ws
from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import agreement as ag
from genreg_train import repetition as rp

ws.CACHE = os.path.join(ROOT, "corpora", "combined", "combined_genomes.pkl")

log("loading Service...")
svc = ws.Service()
svc._load()
if not svc.ready:
    log(f"LOAD FAILED: {svc.err}"); sys.exit(1)
log(f"ready. champs keys: {sorted(svc.champs.keys())}")

ids, vocab, stoi = wp.build_word_corpus(4000)
BIGSET = set(zip(ids[:-1].tolist(), ids[1:].tolist()))
is_content = svc.is_content
CLOSED_CLASS = al.PREPS | al.ARTICLES | al.DET | al.TO
RELATIVE = al.SUBORD | set("who which that whom whose where when".split())
VERBLIKE = ag.MODALS | ag.COPULA | ag.AUX | ag.FIN_3SG | ag.FIN_NON3 | ag.BARE | ag.PARTICIPLE
REL_IDS = set(stoi[w] for w in RELATIVE if w in stoi)
VERB_IDS_STATIC = set(stoi[w] for w in VERBLIKE if w in stoi)


def is_verblike(w):
    if w in VERB_IDS_STATIC:
        return True
    word = vocab[w]
    return word.endswith("ed") or word.endswith("ing")


def generate_with_clause_obligation_decay(en, n, seed, gamma, tau):
    """Same shape as the flat-gamma version, but suppression strength decays
    with word-distance since the MOST RECENT unclosed obligation opened
    (age), instead of staying constant at full strength the whole time it's
    open. factor = exp(-gamma * depth * exp(-age / tau))."""
    rng = np.random.default_rng(seed)
    order_reranks = []
    if en.get("altern") and "altern" in svc.champs:
        order_reranks.append((svc.altern_classfeat, svc.champs["altern"], 2.0))
    if en.get("agree") and "agree" in svc.champs:
        order_reranks.append((svc.agree_classfeat, svc.champs["agree"], 1.0))
    cls_seq = wp.gen_class_seq(svc.champs["order"], 4, n, svc.cids[500:504], rng, 0.8,
                               reranks=order_reranks or None)
    reranks = []
    if en.get("altern") and "altern" in svc.champs:
        reranks.append((svc.altern_feats, svc.champs["altern"], 3.0))
    if en.get("agree") and "agree" in svc.champs:
        reranks.append((svc.agree_feats, svc.champs["agree"], 2.5))
    if en.get("sem") and "sem" in svc.champs:
        reranks.append((svc.feat, svc.champs["sem"], 2.5))
    reranks = reranks or None

    recent, parts, prev, cur, clause = [], [], None, 0, 0
    clause_depth = 0
    age = 0   # words since the most recent obligation was pushed
    for j, cl in enumerate(cls_seq):
        cl = int(cl)
        if cl not in svc.table:
            continue
        mem = svc.table[cl][0]
        bonus = (rp.penalty(svc.champs["rep"], recent, mem, is_content)
                if en.get("rep") and prev is not None else None)
        if prev is not None:
            nxt = next((int(cls_seq[k]) for k in range(j + 1, len(cls_seq)) if int(cls_seq[k]) in svc.table), cl)
            w = wp._fill_bisel(prev, cl, nxt, svc.table, svc.feat, svc.logfreq, svc.cents,
                               svc.champs["bisel"], rng, reranks=reranks, bonus=bonus)
        else:
            w = int(rng.choice(mem, p=svc.table[cl][1]))
        parts.append(vocab[w]); prev = w; recent.append(w); cur += 1; clause += 1

        if w in REL_IDS:
            clause_depth = min(4, clause_depth + 1)
            age = 0   # reset: a fresh obligation just opened
        elif is_verblike(w) and clause_depth > 0:
            clause_depth -= 1
        elif clause_depth > 0:
            age += 1

        pb = wp.boundary_prob(svc.champs["bound"], cl, cur)
        if gamma > 0 and clause_depth > 0 and 0 < pb < 1:
            decay = np.exp(-age / tau)
            pb = pb * np.exp(-gamma * clause_depth * decay)
        if cur >= 4 and rng.random() < pb:
            parts.append("."); cur = 0; clause = 0; clause_depth = 0; age = 0
        elif clause >= 3 and rng.random() < wp.boundary_prob(svc.champs["comma"], cl, clause):
            parts.append(","); clause = 0
    text = " ".join(parts).replace(" .", ".").replace(" ,", ",")
    return re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def measure(gamma, tau, n_samples=30, seed0=80000, n=180):
    en = {"vocab": True, "altern": True, "agree": True, "sem": True, "rep": True}
    ff_pairs = 0; total_pairs = 0
    hits = 0; total_bg = 0
    all_words = []
    dangling = 0; total_sents = 0
    rel_opened = 0; rel_never_closed = 0
    sent_lens = []
    for i in range(n_samples):
        text = generate_with_clause_obligation_decay(en, n, seed0 + i, gamma, tau)
        words = re.findall(r"[a-zA-Z']+", text)
        wids = [stoi.get(w.lower(), 0) for w in words]
        all_words.extend(w.lower() for w in words)
        for k in range(len(wids) - 1):
            if wids[k] and wids[k + 1]:
                total_pairs += 1
                if not is_content[wids[k]] and not is_content[wids[k + 1]]:
                    ff_pairs += 1
                total_bg += 1
                if (wids[k], wids[k + 1]) in BIGSET:
                    hits += 1
        for s in re.split(r"(?<=[.])\s+", text):
            sw = re.findall(r"[a-zA-Z']+", s)
            if not sw:
                continue
            total_sents += 1
            sent_lens.append(len(sw))
            sw_ids = [stoi.get(w.lower(), 0) for w in sw]
            if sw[-1].lower() in CLOSED_CLASS:
                dangling += 1
            depth = 0
            for wid in sw_ids:
                if wid in REL_IDS:
                    depth += 1; rel_opened += 1
                elif is_verblike(wid) and depth > 0:
                    depth -= 1
            if depth > 0:
                rel_never_closed += 1
    return {
        "ff_rate": ff_pairs / max(1, total_pairs),
        "adj_hit": hits / max(1, total_bg),
        "distinct": len(set(all_words)) / max(1, len(all_words)),
        "dangling": dangling / max(1, total_sents),
        "mean_len": sum(sent_lens) / max(1, len(sent_lens)),
        "rel_opened": rel_opened,
        "rel_never_closed_rate": rel_never_closed / max(1, rel_opened) if rel_opened else 0.0,
    }


log("\nbaseline (no tracker, gamma=0):")
r0 = measure(0.0, 1.0)
log(f"  ff-adj={r0['ff_rate']:.3f}  adj-hit={r0['adj_hit']:.3f}  distinct={r0['distinct']:.3f}  "
   f"dangling={r0['dangling']:.3f}  mean-len={r0['mean_len']:.1f}  "
   f"relatives-opened={r0['rel_opened']}  never-closed-rate={r0['rel_never_closed_rate']:.3f}")

log("\nprior flat-gamma result for reference: gamma=1.5, never-closed-rate=0.125, dangling=0.533")

log("\nsweeping GAMMA x TAU (decay time constant, in words):")
for gamma in (3.0, 6.0, 10.0):
    for tau in (2.0, 4.0, 8.0):
        r = measure(gamma, tau)
        log(f"  gamma={gamma:4.1f} tau={tau:4.1f}  ff-adj={r['ff_rate']:.3f}  adj-hit={r['adj_hit']:.3f}  "
           f"distinct={r['distinct']:.3f}  dangling={r['dangling']:.3f}  mean-len={r['mean_len']:.1f}  "
           f"never-closed-rate={r['rel_never_closed_rate']:.3f}")

log("\nsamples at gamma=6.0, tau=4.0:")
en = {"vocab": True, "altern": True, "agree": True, "sem": True, "rep": True}
for seed in range(4):
    log(generate_with_clause_obligation_decay(en, 120, 81000 + seed, 6.0, 4.0))
    log("")

log("DONE")
