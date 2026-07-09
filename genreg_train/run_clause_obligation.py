"""Clause-obligation tracker experiment — direct fix attempt for the
clause-completeness gap diagnosed earlier this session: relative pronouns
(who/which/that/whom/whose/where/when) and subordinators open a clause that
frequently never resolves ("Particular whom state is near" — "whom" opens,
nothing ever closes it). Same shape as the earlier open-obligation tracker
(cut for a specific double-counting bug, not a wrong idea) but at the
CLAUSE level instead of the phrase level: a relative/subordinate word pushes
an obligation, the next verb-like word pops it, sentence-boundary
probability is suppressed while any obligation is open. Heuristic first
(no training, a stateful generation-time rule, exactly like the original
open-obligation experiment) -- measure the REAL effect before deciding
whether it earns a permanent gamma. Runs on the I2 primary.
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
LOG = os.path.join(HERE, "clause_obligation.log")
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

# demo/genomes.pkl is NOT pushed to the primary (not in PUSH_WHITELIST) --
# point CACHE at the combined-corpus training output directly. Without
# this, self.champs is silently EMPTY on the primary and this script's
# direct svc.champs["order"] indexing crashes with KeyError (confirmed:
# this bug already happened once).
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


CLAUSE_GAMMA_DEFAULT = 3.0


def generate_with_clause_obligation(en, n, seed, clause_gamma):
    """Reimplements the forward generate() loop with a clause-obligation
    tracker added on top -- suppresses sentence-boundary probability while
    a relative/subordinate clause is still open."""
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
        elif is_verblike(w) and clause_depth > 0:
            clause_depth -= 1

        pb = wp.boundary_prob(svc.champs["bound"], cl, cur)
        if clause_gamma > 0 and clause_depth > 0 and 0 < pb < 1:
            pb = pb * np.exp(-clause_gamma * clause_depth)
        if cur >= 4 and rng.random() < pb:
            parts.append("."); cur = 0; clause = 0; clause_depth = 0
        elif clause >= 3 and rng.random() < wp.boundary_prob(svc.champs["comma"], cl, clause):
            parts.append(","); clause = 0
    text = " ".join(parts).replace(" .", ".").replace(" ,", ",")
    return re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def measure(clause_gamma, n_samples=30, seed0=70000, n=180):
    en = {"vocab": True, "altern": True, "agree": True, "sem": True, "rep": True}
    ff_pairs = 0; total_pairs = 0
    hits = 0; total_bg = 0
    all_words = []
    dangling = 0; total_sents = 0
    rel_opened = 0; rel_never_closed = 0
    for i in range(n_samples):
        text = generate_with_clause_obligation(en, n, seed0 + i, clause_gamma)
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
        "rel_opened": rel_opened,
        "rel_never_closed_rate": rel_never_closed / max(1, rel_opened) if rel_opened else 0.0,
    }


log("\nsweeping CLAUSE_GAMMA (0=off baseline):")
for cg in (0.0, 1.5, 3.0, 5.0, 8.0):
    r = measure(cg)
    log(f"  clause_gamma={cg:4.1f}  ff-adj={r['ff_rate']:.3f}  adj-hit={r['adj_hit']:.3f}  "
       f"distinct={r['distinct']:.3f}  dangling={r['dangling']:.3f}  "
       f"relatives-opened={r['rel_opened']}  never-closed-rate={r['rel_never_closed_rate']:.3f}")

log("\nsamples at clause_gamma=3.0:")
en = {"vocab": True, "altern": True, "agree": True, "sem": True, "rep": True}
for seed in range(4):
    log(generate_with_clause_obligation(en, 120, 71000 + seed, 3.0))
    log("")

log("DONE")
