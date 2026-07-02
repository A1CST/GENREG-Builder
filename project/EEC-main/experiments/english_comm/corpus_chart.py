"""Charts for real-corpus-grounded communication: semantic confusion structure + Zipf."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corpus_comm as C

HERE = os.path.dirname(os.path.abspath(__file__))
ACC = "#1b5e9e"; ACC2 = "#c0392b"; GOOD = "#2e8b57"
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.color": "#eee"})
rng = np.random.default_rng(0)

print("computing...", flush=True)
# semantic confusion structure
g = C.evolve(anchor=2.0, gens=220, seed=1)
_, errs = C.play(g, 2.0, rng, rounds=60, noise=0.6)
conf_sim = [C.SIM[m, mh] for m, mh in errs]
rand_sim = [C.SIM[m, int(rng.integers(C.K))] for m, _ in errs]
# frequency effect
ES = C.eff_spk(g, 2.0); EL = C.eff_lis(g, 2.0)
ok = np.zeros(C.K); n = np.zeros(C.K)
for _ in range(8000):
    m = int(rng.choice(C.K, p=C.FREQ)); j = int(rng.integers(len(g["spk"])))
    s = int(np.argmax(ES[j, m])); n[m] += 1; ok[m] += (C.decode(EL[j, s] + rng.normal(0, 0.5, C.DIM)) == m)
acc = ok / np.maximum(n, 1); mask = n > 5

fig = plt.figure(figsize=(11, 4.3)); fig.patch.set_facecolor("white")
ax1 = fig.add_axes([0.07, 0.16, 0.32, 0.68])
ax1.bar([0, 1], [np.mean(conf_sim), np.mean(rand_sim)], color=[ACC2, "#999"], width=0.6)
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["confused\nword", "random\nword"])
ax1.set_ylabel("semantic similarity to target")
ax1.set_title("A. Errors are semantically structured", fontsize=10)
ax1.text(0, np.mean(conf_sim) + 0.01, f"{np.mean(conf_sim):+.2f}", ha="center", fontsize=9)
ax1.text(1, np.mean(rand_sim) + 0.01, f"{np.mean(rand_sim):+.2f}", ha="center", fontsize=9)

ax2 = fig.add_axes([0.46, 0.16, 0.5, 0.68])
ax2.scatter(np.log10(C.FREQ[mask]), acc[mask], s=22, color=ACC, alpha=0.8)
z = np.polyfit(np.log10(C.FREQ[mask]), acc[mask], 1); xs = np.array([np.log10(C.FREQ[mask]).min(), np.log10(C.FREQ[mask]).max()])
ax2.plot(xs, np.polyval(z, xs), color=ACC2, lw=2)
r = np.corrcoef(np.log(C.FREQ[mask]), acc[mask])[0, 1]
ax2.set_xlabel("log10 word frequency (corpus)"); ax2.set_ylabel("communication accuracy")
ax2.set_title(f"B. Frequent words communicated better  (Zipf, r={r:+.2f})", fontsize=10)
# annotate a few words
for i in np.where(mask)[0][:40:5]:
    ax2.annotate(C.VOCAB[i], (np.log10(C.FREQ[i]), acc[i]), fontsize=6.5, alpha=0.7)

fig.suptitle("Real-corpus-grounded communication carries real English structure",
             fontsize=12, weight="bold", y=0.99)
out = os.path.join(HERE, "corpus_findings.png")
plt.savefig(out, dpi=120, facecolor="white"); print("saved", out)
