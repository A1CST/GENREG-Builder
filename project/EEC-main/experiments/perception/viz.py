"""Visualize the evolved organism's internal world -- the representations that
paint the real picture, not the token output."""

# --- EEC path bootstrap: shared engine + corpus live in EEC/engine ---
import os as _o, sys as _s
_s.path.insert(0, _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "..", "..", "engine"))
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from perc import PercGenome, make_sequences_L, L
from evolve import build_corpus

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "best", "perc_best.pkl")
OUT = os.path.join(HERE, "organism_world.png")

PUNCT = set(list(",.!?;:\"'`-_()[]") + ["“", "”", "‘", "’", "—", "–"])
FUNC = {"the", "of", "and", "a", "to", "in", "that", "it", "is", "was",
        "he", "she", "i", "you", "for", "with", "as", "his", "her", "had"}


def category(word):
    if word == "<unk>":
        return 3, "<unk>"
    if word in PUNCT:
        return 0, "punctuation"
    if word in FUNC:
        return 1, "function word"
    return 2, "content word"


def main():
    ids, vocab, word2id = build_corpus()
    X, y = make_sequences_L(ids)
    with open(CKPT, "rb") as f:
        saved = pickle.load(f)
    g = PercGenome.__new__(PercGenome)
    g.E, g.g, g.W1, g.b1, g.W2, g.b2 = saved["genome"]
    p = g.perception()

    rng = np.random.default_rng(7)
    N = 4000
    idx = rng.permutation(len(X))[:N]
    Xs = X[idx]
    pe = (g.E[Xs] * p[None, :, None].astype(np.float32)).reshape(N, -1)
    Hs = np.tanh(pe @ g.W1 + g.b1)

    cats = np.array([category(vocab[w])[0] for w in Xs[:, -1]])
    cat_names = ["punctuation", "function word", "content word", "<unk>"]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))

    # 1) perception map
    ax[0, 0].bar(range(L), p, color="#444")
    ax[0, 0].set_title(f"PERCEPTION: what it looks at (gen {saved['gen']}, load {p.sum():.1f}/{L})")
    ax[0, 0].set_xlabel("context position (oldest -> newest)")
    ax[0, 0].set_ylabel("perception intensity"); ax[0, 0].set_ylim(0, 1)

    # 2) hidden-state PCA colored by context category
    Hp = PCA(2).fit_transform(Hs)
    for c in range(4):
        m = cats == c
        ax[0, 1].scatter(Hp[m, 0], Hp[m, 1], s=6, alpha=0.4, c=colors[c], label=cat_names[c])
    ax[0, 1].set_title("INTERNAL STATE (PCA) colored by last-word type")
    ax[0, 1].legend(markerscale=2, fontsize=8)

    # 3) hidden-state t-SNE (nonlinear structure) on a subset
    sub = rng.permutation(N)[:1500]
    Ht = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Hs[sub])
    for c in range(4):
        m = cats[sub] == c
        ax[1, 0].scatter(Ht[m, 0], Ht[m, 1], s=8, alpha=0.5, c=colors[c], label=cat_names[c])
    ax[1, 0].set_title("INTERNAL STATE (t-SNE) -- self-organized categories")
    ax[1, 0].legend(markerscale=2, fontsize=8)

    # 4) learned word-embedding space, colored by frequency rank
    counts = np.bincount(y, minlength=len(vocab))
    topn = 1200
    top_ids = np.argsort(counts)[::-1][:topn]
    Ep = PCA(2).fit_transform(g.E[top_ids])
    sc = ax[1, 1].scatter(Ep[:, 0], Ep[:, 1], s=8, alpha=0.5,
                          c=np.log1p(np.arange(topn)), cmap="viridis")
    for w in ["the", ",", ".", "and", "<unk>", "he", "she", "said", "?", "“"]:
        if w in word2id and word2id[w] in top_ids:
            k = list(top_ids).index(word2id[w])
            ax[1, 1].annotate(w, (Ep[k, 0], Ep[k, 1]), fontsize=9, color="black")
    ax[1, 1].set_title("LEARNED WORD EMBEDDINGS (PCA, top 1200, dark=frequent)")

    plt.tight_layout()
    plt.savefig(OUT, dpi=110)
    print("saved", OUT)


if __name__ == "__main__":
    main()
