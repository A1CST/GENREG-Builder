"""COMMUNICATION MICROSCOPE (working substrate). Organisms live at fixed 2D spots and
talk to NEARBY neighbours via a referential game with COUPLED SURVIVAL (a cooperation
law -- a successful exchange feeds BOTH parties, so honest signalling is the only way
to live; the free-rider problem that breaks lone-agent signalling is gone).

Each organism has an evolvable SPEAK map (meaning -> signal) and LISTEN map (signal ->
meaning). Nothing about meaning is coded. We then MONITOR + DECIPHER:
  - the emergent CODEBOOK: which signal each meaning maps to, and back (the 'dictionary')
  - MI(meaning; signal) and listener accuracy
  - WHO TALKS TO WHOM: the spatial network of successful exchanges
  - DIALECTS: do separated neighbourhoods invent different codes?
Outputs comm_lang.png (4-panel analysis) and comm_lang.mp4 (live talking).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
L, N, V, VSIG, R = 100.0, 44, 5, 6, 26.0      # meanings, signals, talk radius
MEAN = ["FOOD", "DANGER", "MATE", "WATER", "HOME"]
SIGCOL = ["#e74c3c", "#3498db", "#f1c40f", "#2ecc71", "#9b59b6", "#e67e22"]
rng = np.random.default_rng(5)


def init():
    # two loose clusters far apart -> mixing is local -> dialects can form
    c = np.where(np.arange(N) < N//2, 0, 1)
    centers = np.array([[28, 50], [72, 50]])
    pos = centers[c] + rng.normal(0, 13, (N, 2))
    pos = np.clip(pos, 3, L-3)
    g = dict(spk=rng.normal(0, 1, (N, V, VSIG)), lis=rng.normal(0, 1, (N, VSIG, V)))
    return g, pos, np.full(N, 6.0), c


def neighbours(pos):
    nb = []
    for i in range(N):
        d = np.linalg.norm(pos - pos[i], axis=1)
        nb.append(np.where((d > 1e-6) & (d < R))[0])
    return nb


def step(g, pos, energy, nb, cull=True, log=None):
    energy[:] = 0.0                                     # energy = THIS round's communication success
    for i in range(N):                                 # each organism talks to ALL its neighbours
        for j in nb[i]:
            m = int(rng.integers(V))                   # meaning to convey
            sig = int(np.argmax(g["spk"][i][m]))       # speak
            mhat = int(np.argmax(g["lis"][j][sig]))    # listen
            ok = (mhat == m)
            if ok:
                energy[i] += 1.0; energy[j] += 1.0      # COUPLED: a good exchange feeds both
            if log is not None:
                log["ex"].append((int(i), int(j), m, sig, mhat, ok))
    if cull:
        order = np.argsort(energy); worst = order[:4]; top = order[N - N//3:]
        for w in worst:
            p = int(top[rng.integers(len(top))])
            g["spk"][w] = g["spk"][p] + rng.normal(0, 0.22, (V, VSIG))
            g["lis"][w] = g["lis"][p] + rng.normal(0, 0.22, (VSIG, V))


def codebook(g, idx):
    """consensus speak map over a set of organisms: meaning -> most-used signal."""
    book = []
    for m in range(V):
        sigs = [int(np.argmax(g["spk"][i][m])) for i in idx]
        book.append(int(np.bincount(sigs, minlength=VSIG).argmax()))
    return book


def mutual_info(log):
    J = np.zeros((V, VSIG))
    for i, j, m, sig, mhat, ok in log["ex"]:
        J[m, sig] += 1
    P = J / J.sum(); pm = P.sum(1, keepdims=True); ps = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(P * (np.log2(P) - np.log2(pm) - np.log2(ps)))
    acc = np.mean([ok for *_, ok in log["ex"]])
    return float(mi), float(acc), J


def main():
    g, pos, energy, clust = init()
    nb = neighbours(pos)
    print("evolving the language (coupled-survival referential game)...", flush=True)
    accs = []
    for t in range(1300):
        step(g, pos, energy, nb, cull=True)
    print("observing + monitoring...", flush=True)
    log = {"ex": []}
    frames = []
    for t in range(260):
        step(g, pos, energy, nb, cull=False, log=log)
        if t % 2 == 0:
            # snapshot the conversations of this step
            snap = [(i, j, m, sig, mhat, ok) for (i, j, m, sig, mhat, ok) in log["ex"][-N:]]
            frames.append(snap)
    mi, acc, J = mutual_info(log)
    bookA = codebook(g, np.where(clust == 0)[0]); bookB = codebook(g, np.where(clust == 1)[0])
    print(f"\nlistener accuracy = {acc*100:.0f}%   MI(meaning;signal) = {mi:.2f} bits (max {np.log2(V):.1f})")
    print("DECIPHERED CODEBOOK (consensus):")
    for m in range(V):
        print(f"  meaning '{MEAN[m]:7}'  ->  signal {bookA[m]}  ({SIGCOL[bookA[m]]})")
    print(f"dialects: cluster-A code {bookA}  |  cluster-B code {bookB}  "
          f"({'SAME' if bookA == bookB else 'DIFFERENT -> dialects formed'})")

    # who-talks-to-whom adjacency (successful exchanges)
    A = np.zeros((N, N))
    for i, j, m, sig, mhat, ok in log["ex"]:
        if ok:
            A[i, j] += 1
    _figure(g, pos, clust, mi, acc, J, bookA, bookB, A)
    _clip(pos, frames, g)


def _figure(g, pos, clust, mi, acc, J, bookA, bookB, A):
    fig = plt.figure(figsize=(15.5, 8)); fig.patch.set_facecolor("#0d0d12")
    # A) who talks to whom (spatial network, coloured by dialect)
    axn = fig.add_axes([0.04, 0.08, 0.44, 0.82]); axn.set_facecolor("#10131a")
    axn.set_xlim(0, L); axn.set_ylim(0, L); axn.set_xticks([]); axn.set_yticks([])
    mx = A.max() or 1
    for i in range(N):
        for j in range(N):
            if A[i, j] > mx*0.2:
                axn.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-", mutation_scale=6,
                              color="#5a6", alpha=min(0.7, A[i, j]/mx), lw=0.5+1.6*A[i, j]/mx))
    deg = A.sum(1) + A.sum(0)
    axn.scatter(pos[:, 0], pos[:, 1], s=40+deg*1.2, c=["#e67e22" if c == 0 else "#3498db" for c in clust],
                edgecolor="white", linewidth=0.5, zorder=5)
    axn.set_title("WHO TALKS TO WHOM  (lines = successful exchanges; colour = neighbourhood/dialect)",
                  color="white", fontsize=11)
    # B) the deciphered codebook (dictionary)
    axd = fig.add_axes([0.53, 0.55, 0.20, 0.35]); axd.set_facecolor("#0d0d12"); axd.axis("off")
    axd.text(0.5, 1.02, "DECIPHERED DICTIONARY", color="white", ha="center", fontsize=11, weight="bold", transform=axd.transAxes)
    for m in range(V):
        y = 0.86 - m*0.17
        axd.text(0.02, y, MEAN[m], color="white", fontsize=11, va="center", transform=axd.transAxes)
        axd.annotate("", xy=(0.55, y), xytext=(0.42, y), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="#888"))
        axd.add_patch(plt.Rectangle((0.6, y-0.05), 0.16, 0.1, color=SIGCOL[bookA[m]], transform=axd.transAxes))
        axd.text(0.68, y, f"sig {bookA[m]}", color="black", fontsize=9, ha="center", va="center", weight="bold", transform=axd.transAxes)
    # C) meaning x signal matrix
    axm = fig.add_axes([0.80, 0.57, 0.17, 0.33]); axm.set_facecolor("#10131a")
    Pn = J / (J.sum(1, keepdims=True) + 1e-9)
    axm.imshow(Pn, aspect="auto", cmap="magma")
    axm.set_yticks(range(V)); axm.set_yticklabels(MEAN, color="white", fontsize=8)
    axm.set_xticks(range(VSIG)); axm.set_xticklabels([f"s{s}" for s in range(VSIG)], color="#aaa", fontsize=7)
    axm.set_title(f"meaning->signal\nMI={mi:.2f} bits · {acc*100:.0f}% understood", color="white", fontsize=9)
    # D) dialect comparison
    axx = fig.add_axes([0.53, 0.08, 0.44, 0.37]); axx.set_facecolor("#0d0d12"); axx.axis("off")
    axx.text(0.5, 1.0, "DIALECTS  (do separated neighbourhoods invent different codes?)",
             color="white", ha="center", fontsize=11, weight="bold", transform=axx.transAxes)
    for m in range(V):
        y = 0.82 - m*0.16
        axx.text(0.04, y, MEAN[m], color="white", fontsize=10, va="center", transform=axx.transAxes)
        axx.add_patch(plt.Rectangle((0.40, y-0.045), 0.1, 0.09, color=SIGCOL[bookA[m]], transform=axx.transAxes))
        axx.text(0.45, y, str(bookA[m]), color="black", ha="center", va="center", fontsize=9, transform=axx.transAxes)
        axx.add_patch(plt.Rectangle((0.62, y-0.045), 0.1, 0.09, color=SIGCOL[bookB[m]], transform=axx.transAxes))
        axx.text(0.67, y, str(bookB[m]), color="black", ha="center", va="center", fontsize=9, transform=axx.transAxes)
    axx.text(0.45, 0.92, "WEST", color="#e67e22", ha="center", fontsize=9, transform=axx.transAxes)
    axx.text(0.67, 0.92, "EAST", color="#3498db", ha="center", fontsize=9, transform=axx.transAxes)
    same = bookA == bookB
    axx.text(0.5, -0.02, "same code" if same else "DIFFERENT codes -> dialects emerged",
             color="#2ecc71" if same else "#e74c3c", ha="center", fontsize=11, weight="bold", transform=axx.transAxes)
    fig.suptitle(f"COMMUNICATION MICROSCOPE  —  an emergent language, deciphered   "
                 f"({acc*100:.0f}% understood, MI {mi:.2f} bits)", color="white", fontsize=14, weight="bold")
    out = os.path.join(HERE, "comm_lang.png")
    plt.savefig(out, dpi=115, facecolor=fig.get_facecolor()); print("saved", out)


def _clip(pos, frames, g):
    fig, ax = plt.subplots(figsize=(8.5, 8)); fig.patch.set_facecolor("#0d0d12")

    def draw(k):
        ax.clear(); ax.set_facecolor("#10131a"); ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_xticks([]); ax.set_yticks([])
        ax.scatter(pos[:, 0], pos[:, 1], s=45, c="#444", edgecolor="white", linewidth=0.4, zorder=3)
        for (i, j, m, sig, mhat, ok) in frames[k]:
            ax.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-|>", mutation_scale=12,
                         color=SIGCOL[sig], alpha=0.85 if ok else 0.25, lw=2.0 if ok else 0.8, zorder=4))
        ax.set_title("live conversations  —  arrow = signal (colour), solid = understood, faint = misunderstood",
                     color="white", fontsize=11)
        return []
    ani = FuncAnimation(fig, draw, frames=len(frames), interval=120, blit=False)
    out = os.path.join(HERE, "comm_lang.mp4")
    ani.save(out, writer=FFMpegWriter(fps=8, bitrate=1800)); print("saved", out)


if __name__ == "__main__":
    main()
