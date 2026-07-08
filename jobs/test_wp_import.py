import _pkg_stub  # noqa: F401 — must be first, see _pkg_stub.py docstring
from genreg_train import wordpipe as wp
from genreg_train import genelib as gl

ids, vocab, stoi = wp.build_word_corpus(4000)
print("corpus tokens:", len(ids))
print("vocab size:", len(vocab))
print("sample words:", vocab[1:6])
print("wordpipe + genelib import and corpus build OK on this node")
