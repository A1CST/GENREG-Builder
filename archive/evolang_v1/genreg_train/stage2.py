"""Stage-2 candidate genomes (cross-sentence / structural). These target capabilities
nothing in the current stack has: verb-per-sentence completeness, content memory across
a window wider than +/-1, and cross-sentence coherence. Each is small and rides existing
machinery; the battery decides which earn their place.

  wider co-occurrence : reuse the evolved `sem` head, but score a candidate against the
                        last K CONTENT words, not just the previous word (memory beyond +-1).
  sentence-has-verb   : per-class verb-ness (data-derived), used to nudge a verb in when a
                        sentence is running long without one.
  lexical bridge      : carry a content word across a sentence boundary — boost words from
                        the previous sentence in the next (the minimum unit of coherence).
  discourse connector : open some sentences with a connector (however/then/also/but/so).

Reference corpus rates (verb-per-sentence, cross-sentence carryover, connector-opening) are
computed from the corpus so the battery can gate against them, not invent targets.
"""
import numpy as np

from genreg_train import wordpipe as wp
from genreg_train import altern as al
from genreg_train import tense_consist as tc

CONNECTORS = set("however then also but so therefore thus yet still meanwhile "
                 "nevertheless moreover besides hence consequently".split())


def is_verb_word(vocab):
    return np.array([bool(tc.tense_feats([w])[0, 7]) for w in vocab], dtype=bool)


def is_content_word(vocab):
    return np.array([w.isalpha() and w not in al.FUNCTION and len(w) > 2 for w in vocab], dtype=bool)


def is_connector_word(vocab):
    return np.array([w in CONNECTORS for w in vocab], dtype=bool)


def class_verbness(table, isv, nc):
    """Per-class freq-weighted verb fraction (data-derived, ~nc params)."""
    cv = np.zeros(nc, np.float32)
    for cl, (mem, p) in table.items():
        cv[cl] = float(p @ isv[mem])
    return cv


def corpus_reference(n_classes=32, vocab_n=4000, scan=800000):
    """Reference rates from the corpus: verb-per-sentence, cross-sentence content
    carryover, connector-opening. Sentences split on . ! ? in the raw token stream."""
    _, vocab, stoi = wp.build_word_corpus(vocab_n)
    toks = wp.decode(wp.corpus_ids()).split()[:scan]
    isv = is_verb_word(vocab); isc = is_content_word(vocab)
    sents, cur = [], []
    for t in toks:
        s = t.strip(".,!?;:'\"")
        wid = stoi.get(s, 0)
        cur.append(wid)
        if t and t[-1] in ".!?":
            if cur:
                sents.append(cur); cur = []
    have_verb = np.mean([any(isv[w] for w in s if w) for s in sents])
    conn = np.mean([bool(s) and s[0] != 0 and vocab[s[0]] in CONNECTORS for s in sents])
    carry = []
    for a, b in zip(sents, sents[1:]):
        ca = {w for w in a if w and isc[w]}
        carry.append(any(isc[w] and w in ca for w in b))
    return {"verb_per_sent": round(float(have_verb), 3),
            "carryover": round(float(np.mean(carry)), 3),
            "connector_open": round(float(np.mean(conn)), 3),
            "n_sents": len(sents)}
