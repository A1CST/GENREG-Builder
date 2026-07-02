"""COMMUNICATION MICROSCOPE. Organisms in a grid have a SYMBOLIC channel: each emits
one of K evolved signals based on what it sees (food direction), and a lost organism
(small vision) hears the nearest neighbour's signal and decodes it into a heading.
Survival depends on finding food, much of which is only reachable by READING signals.

Nothing about meaning is coded. We then MONITOR + DECIPHER the emergent protocol:
  - what does each signal MEAN? (signal -> mean food-direction it is emitted for; MI)
  - who is talking to whom? (speaker -> listener edges = who responded to whom)
Outputs an analysis figure (comm_scope.png) and a short annotated clip (comm_scope.mp4).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
L, N, NF, K = 100.0, 30, 16, 4          # food findable but small vision
VIS, HEAR = 4.5, 45.0                    # tiny vision -> mostly lost -> must listen
SIGCOL = ["#e74c3c", "#3498db", "#f1c40f", "#2ecc71", "#9b59b6", "#e67e22"]
rng = np.random.default_rng(3)


def init():
    g = dict(Wspk=rng.normal(0, 1, (N, 3, K)), Wlis=rng.normal(0, 1, (N, K, 2)),
             wsig=rng.uniform(0.5, 1.5, N))
    st = dict(pos=rng.uniform(0, L, (N, 2)), energy=np.full(N, 10.0),
              food_xy=rng.uniform(6, L-6, (NF, 2)), food_amt=rng.uniform(0.6, 1.0, NF),
              sig=np.zeros(N, int), heard_from=np.full(N, -1), seen=np.zeros(N, bool),
              fdir=np.zeros((N, 2)))
    return g, st


def sense_emit(g, st):
    pos = st["pos"]; fx = st["food_xy"]; fa = st["food_amt"]
    for i in range(N):
        d = fx - pos[i]; dist = np.linalg.norm(d, axis=1)
        vis = dist < VIS
        if vis.any():
            j = np.where(vis)[0][np.argmin(dist[vis])]
            u = d[j] / max(dist[j], 1e-6); st["fdir"][i] = u; st["seen"][i] = True
            feat = np.array([u[0], u[1], 1.0])
        else:
            st["seen"][i] = False; st["fdir"][i] = 0; feat = np.array([0.0, 0.0, 1.0])
        st["sig"][i] = int(np.argmax(feat @ g["Wspk"][i]))


def step(g, st, cull=True, log=None):
    sense_emit(g, st)
    pos = st["pos"]; fx = st["food_xy"]; fa = st["food_amt"]; energy = st["energy"]
    st["heard_from"][:] = -1
    newpos = pos.copy()
    for i in range(N):
        dd = np.zeros(2)
        if st["seen"][i]:
            dd += st["fdir"][i]                          # go to visible food
        else:                                            # lost: go to a signaller WHO SEES FOOD, offset by its message
            dn = pos - pos[i]; dist = np.linalg.norm(dn, axis=1)
            cand = np.where((dist > 1e-6) & (dist < HEAR) & st["seen"])[0]
            if len(cand):
                j = cand[np.argmin(dist[cand])]
                st["heard_from"][i] = j
                target = pos[j] + g["Wlis"][i][st["sig"][j]] * 8.0   # speaker pos + decoded offset
                v = target - pos[i]
                dd += g["wsig"][i] * v / max(np.linalg.norm(v), 1e-6)
        dd += 0.5 * rng.normal(0, 1, 2)
        nd = np.linalg.norm(dd)
        newpos[i] = np.clip(pos[i] + (dd / nd if nd > 1e-6 else 0), 0, L)
    st["pos"] = newpos; pos = newpos
    # eat: lone forager survives modestly; a GROUP gets a big COOPERATION bonus -> recruit via signal
    for j in range(NF):
        if fa[j] > 0.05:
            on = np.linalg.norm(pos - fx[j], axis=1) < 5; cnt = int(on.sum())
            if cnt >= 1:
                per = min(0.6, fa[j]) * (2.2 if cnt >= 2 else 1.0)   # group ~doubles per-capita yield
                energy[on] += per * 1.5; fa[j] -= min(0.6, fa[j])
    energy -= 0.4
    fa[:] = np.minimum(1.0, fa + 0.004)
    if log is not None:                                  # MONITOR
        for i in range(N):
            if st["seen"][i]:
                log["emit"].append((st["sig"][i], float(np.arctan2(st["fdir"][i][1], st["fdir"][i][0]))))
            hf = st["heard_from"][i]
            if hf >= 0 and not st["seen"][i]:            # i responded to hf's signal
                log["edges"].append((int(hf), int(i), int(st["sig"][hf])))
    if cull:                                             # steady-state survival
        order = np.argsort(energy); worst = order[:2]; top = order[N - N//2:]
        med = float(np.median(energy))
        for w in worst:
            p = int(top[rng.integers(len(top))])
            for key, sc in (("Wspk", 0.12), ("Wlis", 0.12), ("wsig", 0.1)):
                g[key][w] = g[key][p] + rng.normal(0, sc, g[key][p].shape)
            st["pos"][w] = st["pos"][p] + rng.normal(0, 3, 2); energy[w] = med
    energy[:] = np.clip(energy, -1, 26)


def mutual_info(sigs, angs, nbin=8):
    a = ((np.array(angs) + np.pi) / (2*np.pi) * nbin).astype(int) % nbin
    s = np.array(sigs)
    J = np.zeros((K, nbin))
    for ss, aa in zip(s, a): J[ss, aa] += 1
    if J.sum() == 0: return 0.0, J
    P = J / J.sum(); ps = P.sum(1, keepdims=True); pa = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = P * (np.log2(P) - np.log2(ps) - np.log2(pa))
    return float(np.nansum(m)), J


def decipher(log):
    sigs = [s for s, a in log["emit"]]; angs = [a for s, a in log["emit"]]
    mi, J = mutual_info(sigs, angs)
    # meaning of each signal = circular mean of food-direction angles it is emitted for
    meaning = {}
    for s in range(K):
        aa = [a for ss, a in log["emit"] if ss == s]
        if aa:
            meaning[s] = (float(np.arctan2(np.mean(np.sin(aa)), np.mean(np.cos(aa)))), len(aa))
        else:
            meaning[s] = (None, 0)
    return mi, meaning, J, np.array(sigs)


def main():
    g, st = init()
    print("evolving the protocol (perception-limited, comm-coupled)...", flush=True)
    for t in range(1400):
        step(g, st, cull=True)
    print("observing + monitoring...", flush=True)
    log = {"emit": [], "edges": []}
    frames_pos = []
    for t in range(420):
        step(g, st, cull=False, log=log)
        if t % 3 == 0:
            frames_pos.append((st["pos"].copy(), st["sig"].copy(), st["heard_from"].copy(),
                               st["seen"].copy(), st["food_xy"].copy(), st["food_amt"].copy()))
    mi, meaning, J, sigs = decipher(log)
    # who-talks-to-whom adjacency
    A = np.zeros((N, N))
    for sp, li, sg in log["edges"]:
        A[sp, li] += 1
    print(f"\nMI(signal; food-direction) = {mi:.2f} bits (max {np.log2(8):.0f})")
    print("DECIPHERED DICTIONARY:")
    comp = {0: "E", 1: "NE", 2: "N", 3: "NW", 4: "W", 5: "SW", 6: "S", 7: "SE"}
    for s in range(K):
        ang, n = meaning[s]
        if ang is not None:
            sect = comp[int(((ang+np.pi)/(2*np.pi)*8)) % 8]
            print(f"  signal {s} ({SIGCOL[s]}): '{sect}'  (food to the {sect}; used {n}x)")
    _figure(mi, meaning, J, A, frames_pos, g, st)
    _clip(frames_pos)


def _figure(mi, meaning, J, A, frames, g, st):
    fig = plt.figure(figsize=(15, 8)); fig.patch.set_facecolor("#0d0d12")
    comp8 = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    # A) world snapshot with signals + comm arrows
    axw = fig.add_axes([0.04, 0.08, 0.42, 0.82]); axw.set_facecolor("#10131a")
    axw.set_xlim(0, L); axw.set_ylim(0, L); axw.set_xticks([]); axw.set_yticks([])
    pos, sig, hf, seen, fx, fa = frames[-1]
    axw.scatter(fx[:, 0], fx[:, 1], s=50*fa+6, c="#2ecc71", alpha=0.4, marker="s")
    for i in range(N):                                   # comm arrows speaker->listener
        if hf[i] >= 0 and not seen[i]:
            sp = hf[i]
            axw.add_patch(FancyArrowPatch(pos[sp], pos[i], arrowstyle="-|>", mutation_scale=10,
                          color=SIGCOL[sig[sp]], alpha=0.6, lw=1.2))
    for i in range(N):
        axw.scatter(pos[i, 0], pos[i, 1], s=90, c=SIGCOL[sig[i]], edgecolor="white", linewidth=0.6, zorder=5)
    axw.set_title("who is talking to whom  (arrow = listener responded to speaker; colour = signal)",
                  color="white", fontsize=11)
    # B) deciphered dictionary (compass per signal)
    axd = fig.add_axes([0.52, 0.52, 0.20, 0.38], projection="polar"); axd.set_facecolor("#10131a")
    for s in range(K):
        ang, n = meaning[s]
        if ang is not None:
            axd.annotate("", xy=(ang, 1), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=SIGCOL[s], lw=2.5))
    axd.set_yticks([]); axd.set_title("deciphered MEANING\n(each signal -> food direction)", color="white", fontsize=10)
    axd.tick_params(colors="#888")
    # C) signal x direction matrix
    axm = fig.add_axes([0.80, 0.55, 0.17, 0.33]); axm.set_facecolor("#10131a")
    Pn = J / (J.sum(1, keepdims=True) + 1e-9)
    axm.imshow(Pn, aspect="auto", cmap="magma")
    axm.set_yticks(range(K)); axm.set_yticklabels([f"sig {s}" for s in range(K)], color="white", fontsize=8)
    axm.set_xticks(range(8)); axm.set_xticklabels(comp8, color="#aaa", fontsize=7)
    axm.set_title(f"signal -> direction\nMI = {mi:.2f} bits", color="white", fontsize=9)
    # D) communication network (who talks to whom)
    axn = fig.add_axes([0.52, 0.08, 0.45, 0.36]); axn.set_facecolor("#10131a")
    axn.set_xlim(0, L); axn.set_ylim(0, L); axn.set_xticks([]); axn.set_yticks([])
    talk = A.sum(1); listen = A.sum(0)
    axn.scatter(pos[:, 0], pos[:, 1], s=30+talk*4, c=[SIGCOL[s] for s in sig], edgecolor="white", lw=0.4, zorder=5)
    mx = A.max() if A.max() > 0 else 1
    for sp in range(N):
        for li in range(N):
            if A[sp, li] > mx*0.18:
                axn.add_patch(FancyArrowPatch(pos[sp], pos[li], arrowstyle="-|>", mutation_scale=8,
                              color="white", alpha=min(0.8, A[sp, li]/mx), lw=0.6+1.5*A[sp, li]/mx))
    axn.set_title("communication NETWORK over the whole observation  (node size = how much it speaks)",
                  color="white", fontsize=10)
    fig.suptitle(f"COMMUNICATION MICROSCOPE  —  an emergent protocol, deciphered   (MI = {mi:.2f} bits)",
                 color="white", fontsize=14, weight="bold")
    out = os.path.join(HERE, "comm_scope.png")
    plt.savefig(out, dpi=115, facecolor=fig.get_facecolor()); print("saved", out)


def _clip(frames):
    fig, ax = plt.subplots(figsize=(8, 8)); fig.patch.set_facecolor("#0d0d12")

    def draw(k):
        ax.clear(); ax.set_facecolor("#10131a"); ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_xticks([]); ax.set_yticks([])
        pos, sig, hf, seen, fx, fa = frames[k]
        ax.scatter(fx[:, 0], fx[:, 1], s=50*fa+6, c="#2ecc71", alpha=0.4, marker="s")
        for i in range(N):
            if hf[i] >= 0 and not seen[i]:
                sp = hf[i]
                ax.add_patch(FancyArrowPatch(pos[sp], pos[i], arrowstyle="-|>", mutation_scale=11,
                             color=SIGCOL[sig[sp]], alpha=0.7, lw=1.4))
        for i in range(N):
            mk = "*" if seen[i] else "o"
            ax.scatter(pos[i, 0], pos[i, 1], s=140 if seen[i] else 80, marker=mk,
                       c=SIGCOL[sig[i]], edgecolor="white", linewidth=0.6, zorder=5)
        ax.set_title("live communication  —  ★ = sees food (speaking) · ○ = lost (listening) · colour = signal",
                     color="white", fontsize=11)
        return []
    ani = FuncAnimation(fig, draw, frames=len(frames), interval=90, blit=False)
    out = os.path.join(HERE, "comm_scope.mp4")
    ani.save(out, writer=FFMpegWriter(fps=11, bitrate=2000)); print("saved", out)


if __name__ == "__main__":
    main()
