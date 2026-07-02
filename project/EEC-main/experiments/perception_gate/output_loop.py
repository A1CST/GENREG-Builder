"""STATE OF THE OUTPUT: close the loop on the real corpus. Input a real word -> perceive its meaning
(distributional, per-slot) -> EXPRESS that meaning with a DIFFERENT word (transform-don't-copy). Show
actual transcripts + how often the spoken word is a true same-meaning synonym (output fidelity is
bounded by perception: a misperceived input -> a wrong-meaning word out)."""
import os, numpy as np
from real_perslot import build_embeddings, GROUPS, ENG, CORP
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = open(os.path.join(HERE, "output_loop_results.txt"), "w")
def out(s): print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def main():
    idx, E, n = build_embeddings(ENG)
    inv = {g: [w for w in GROUPS[g] if w in idx] for g in GROUPS}
    names = [g for g in inv if len(inv[g]) >= 3]
    true_group = {w: g for g in names for w in inv[g]}

    def perceive(word):                                       # classify input word -> meaning group
        cent, gn = [], []
        for g in names:
            others = [w for w in inv[g] if w != word]
            if others: cent.append(np.mean([E[idx[w]] for w in others], 0)); gn.append(g)
        C = np.stack(cent); C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        return gn[int((C @ E[idx[word]]).argmax())]

    def express(group, avoid):                                # say the meaning in a DIFFERENT word
        opts = [w for w in inv[group] if w != avoid] or inv[group]
        return opts[hash(avoid) % len(opts)]

    out(f"STATE OF THE OUTPUT — closed loop on the literary corpus ({n:,} tokens, per-slot ~0.72).")
    out("input word  ->  [understood meaning]  ->  spoken reply (different word, same meaning)")
    out("=" * 74)
    correct = tot = 0
    shown = 0
    for g in names:
        for w in inv[g]:
            pg = perceive(w); said = express(pg, w)
            ok = (pg == true_group[w]); correct += ok; tot += 1
            mark = "" if ok else "   <- misheard"
            if shown < 26:
                out(f"  {w:>12}  ->  [{pg:^9}]  ->  {said:<12}{mark}"); shown += 1
    out("=" * 74)
    out(f"output fidelity (spoke a true same-meaning word): {correct}/{tot} = {correct/tot:.3f}")
    out("READING: the organism DOES produce real-word output -- it hears a word, understands the meaning,")
    out("and says a DIFFERENT word of that meaning (transform-don't-copy, not echo). Output quality is")
    out("bounded by perception (~per-slot): when it mishears the meaning, it speaks the wrong word. The")
    out("expression step itself is trivial; the bottleneck is comprehension, and the lever is experience.")
    out("done"); LOG.close()


if __name__ == "__main__":
    main()
