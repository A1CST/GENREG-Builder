"""Neuron + dimension saturation of the REAL organism. Two suspects for why the neural channel is
vestigial: (1) the recurrent tanh units saturate (pinned at +/-1 -> carry no varying info), or (2) the
input embedding (EMBED=16, hardcoded) is starved -- last check showed the data wants hundreds of dims."""
import os, pickle, numpy as np
import converse_organism as co

ids, vocab, w2i = co.build_corpus(); V = len(vocab)
seg = ids[50000:51500]
d = pickle.load(open(os.path.join(co.HERE, "convo_organism.pkl"), "rb"))
org = co.Organism.__new__(co.Organism)
org.E, org.W_in, org.W_rec, org.b, org.W_out, org.b_out = d["genome"]
org.M, org.a_ng, org.a_nn = d["M"], d["a_ng"], d["a_nn"]
print(f"evolved organism: M={org.M}, a_ng={org.a_ng:.3f}, a_nn={org.a_nn:.4f}, EMBED={co.EMBED}")


def report(S, tag):
    sat = float(np.mean(np.abs(S) > 0.9))                    # tanh saturation
    stds = S.std(0)
    dead = float(np.mean(stds < 0.05))                       # units that barely move
    print(f"  {tag}: |s|>0.9 saturated = {sat:.1%} | dead units (std<0.05) = {dead:.1%} | "
          f"mean|s| = {np.mean(np.abs(S)):.3f} | active range = {1-sat-dead:.1%}")


print("\n-- recurrent hidden-state neuron saturation --")
report(org.states(org.E[seg]), f"evolved (M={org.M})")
for m in (8, 16, 32, 48):                                   # force wider memory: does the recurrence saturate?
    o2 = org.copy(); o2.M = m
    report(o2.states(o2.E[seg]), f"forced M={m}")

print("\n-- input embedding dimension --")
Erank = np.linalg.matrix_rank(org.E[:V], tol=1e-3)
sv = np.linalg.svd(org.E[:2000], compute_uv=False)
print(f"  EMBED = {co.EMBED} dims for {V} words. embedding-matrix rank = {Erank}")
print(f"  embedding singular values: top {sv[0]:.2f} -> bottom {sv[-1]:.2f} (ratio {sv[0]/max(sv[-1],1e-6):.1f}x)")
print(f"  energy in first 8 of {co.EMBED} dims: {np.sum(sv[:8]**2)/np.sum(sv**2):.1%}")
print("\nREADING: high tanh-saturation or many dead units => recurrence carries no info. EMBED=16 vs the")
print("~400 dims the corpus wanted (saturation_results.txt) => the neural channel's INPUT is starved")
print("~25x before evolution even starts -- it literally can't see enough to beat the trigram.")
