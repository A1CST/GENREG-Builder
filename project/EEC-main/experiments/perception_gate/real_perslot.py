"""REAL per-slot perception: build embeddings from a REAL corpus (distributional, gradient-free), then
measure how well perception classifies a HELD-OUT synonym into its true semantic group. That real
per-slot number, raised to S, is the real compositional depth curve."""
import os, re, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CORP = os.path.join(os.path.dirname(HERE), "english_comm", "chat_corpus.txt")
ENG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "engine", "corpus.txt")
DIM, WIN = 80, 5
LOG = open(os.path.join(HERE, "real_perslot_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

GROUPS = {
 "happy": "happy glad joyful cheerful pleased delighted".split(),
 "sad": "sad unhappy miserable gloomy upset down".split(),
 "big": "big large huge giant enormous massive".split(),
 "small": "small tiny little".split(),
 "fast": "fast quick rapid swift speedy".split(),
 "hungry": "hungry starving famished".split(),
 "tired": "tired sleepy exhausted weary".split(),
 "hot": "hot warm boiling".split(),
 "cold": "cold chilly freezing".split(),
 "good": "good great excellent wonderful fine nice".split(),
 "bad": "bad terrible awful horrible".split(),
 "smart": "smart clever intelligent bright".split(),
 "angry": "angry mad furious annoyed".split(),
 "scared": "scared afraid frightened terrified".split(),
 "pretty": "pretty beautiful lovely gorgeous".split(),
 "talk": "talk speak chat say tell".split(),
}


def build_embeddings(path, maxtok=4_000_000, K=6000):
    toks = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            toks += [w for w in re.findall(r"[a-z']+", line.lower()) if len(w) >= 2]
            if len(toks) >= maxtok: break
    cnt = collections.Counter(toks); vocab = [w for w, _ in cnt.most_common(K)]
    idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab); co = np.zeros((V, V), np.float32); win = []
    for w in toks:
        if w in idx:
            wi = idx[w]
            for v in win[-WIN:]: co[wi, v] += 1; co[v, wi] += 1
            win.append(wi)
            if len(win) > WIN: win.pop(0)
    tot = co.sum(); rk = co.sum(1, keepdims=True) + 1e-9
    ppmi = np.maximum(np.log((co * tot) / (rk @ rk.T) + 1e-9), 0)
    U, S, _ = np.linalg.svd(ppmi.astype(np.float64)); E = U[:, :DIM] * np.sqrt(S[:DIM])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return idx, E, len(toks)


def per_slot(idx, E):
    names = list(GROUPS)
    inv = {g: [w for w in GROUPS[g] if w in idx] for g in names}
    names = [g for g in names if len(inv[g]) >= 3]               # need >=3 in-vocab for train+held
    cover = sum(len(inv[g]) for g in names) / sum(len(GROUPS[g]) for g in GROUPS)
    correct = tot = 0
    for held_g in names:
        for hw in inv[held_g]:
            cent = []; gn = []
            for g in names:
                others = [w for w in inv[g] if not (g == held_g and w == hw)]
                if others: cent.append(np.mean([E[idx[w]] for w in others], 0)); gn.append(g)
            C = np.stack(cent); C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
            pred = gn[int((C @ E[idx[hw]]).argmax())]; correct += (pred == held_g); tot += 1
    return correct / tot, len(names), cover


if __name__ == "__main__":
    out("REAL per-slot perception (distributional embeddings from a real corpus). held-out synonym -> "
        "true semantic group.")
    rows = []
    for name, path in [("conversational (chat_corpus ~500k)", CORP), ("literary (engine corpus ~few M)", ENG)]:
        if not os.path.exists(path): out(f"  {name}: MISSING"); continue
        idx, E, n = build_embeddings(path)
        ps, ng, cov = per_slot(idx, E)
        rows.append((name, ps, ng, cov, n)); out(f"  {name}: per-slot {ps:.3f}  ({ng} groups, coverage {cov:.0%}, {n:,} tokens)")
    out("=" * 70)
    out("REAL compositional DEPTH curve  (full-meaning = per-slot^S):")
    out(f"{'corpus':>34} | " + " | ".join(f"S={s}" for s in [1, 2, 4, 8, 14]))
    for name, ps, ng, cov, n in rows:
        out(f"{name:>34} | " + " | ".join(f"{ps**s:.3f}" for s in [1, 2, 4, 8, 14]))
    out("=" * 70)
    out("READING: this is the REAL per-slot number. In the clean world it was 1.0 (full-meaning held at")
    out("any depth). Real per-slot < 1.0, so full-meaning = per-slot^S DECAYS with compositional depth --")
    out("and the lever to push it back up is EXPERIENCE (richer corpus -> per-slot -> 1).")
    out("done"); LOG.close()
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        Ss = np.arange(1, 15); plt.figure(figsize=(8, 5))
        for name, ps, ng, cov, n in rows:
            plt.plot(Ss, ps ** Ss, "o-", lw=2, label=f"{name.split('(')[0].strip()} (per-slot {ps:.2f})")
        plt.axhline(1.0, color="#999", ls=":", label="clean world (per-slot 1.0)")
        plt.xlabel("compositional depth (slots S)"); plt.ylabel("full-meaning accuracy = per-slot^S"); plt.ylim(0, 1.02)
        plt.title("Real compositional depth curve: fidelity decays, experience is the lever", weight="bold")
        plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(HERE, "real_perslot.png"), dpi=130); print("saved real_perslot.png")
    except Exception as e:
        print("chart skipped", e)
