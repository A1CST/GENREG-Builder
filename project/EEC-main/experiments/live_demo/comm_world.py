"""COMMUNICATION FROM LAWS (no engineered cooperation). Free-living organisms, PURE
INDIVIDUAL survival -- you eat only what you yourself reach. NO coupled fitness, NO
cooperation bonus, NO reward for communicating. Honest signalling is not designed in;
it must EMERGE because nothing else pays rent under the stacked laws:

  ENERGY      -- metabolism drains every step; run out -> outcompeted
  SCARCITY    -- food patches deplete and a crowd dilutes the share
  PERCEPTION  -- small vision: you cannot find enough food alone
  LOCAL REPRO -- offspring are placed by their PARENT -> neighbours are KIN

Under these, a lost organism that decodes a neighbour's signal finds food faster
(individually selected to listen); and because neighbours are kin, a lineage whose
members signal honestly forages better and spreads (kin selection -- emergent from
the reproduction operator, not engineered). A liar/hoarder just runs its kin-patch
into the ground and shrinks. We then DECIPHER the protocol if one emerges.

Diagnostic mode prints seen-fraction / survival / MI over generations.
"""
import os, sys
import numpy as np

L, N, NF, K = 100.0, 30, 12, 4
VIS, HEAR = 8.0, 42.0
MET, BITE, REGEN = 0.18, 0.5, 0.2
rng = np.random.default_rng(11)


def init():
    g = dict(Wspk=rng.normal(0, 1, (N, 3, K)), Wlis=rng.normal(0, 1, (N, K, 2)),
             wsig=rng.uniform(0.6, 1.4, N))
    st = dict(pos=rng.uniform(0, L, (N, 2)), energy=rng.uniform(6, 12, N),
              food_xy=rng.uniform(6, L-6, (NF, 2)), food_amt=rng.uniform(1.5, 3.0, NF),
              sig=np.zeros(N, int), heard=np.full(N, -1), seen=np.zeros(N, bool), fdir=np.zeros((N, 2)))
    return g, st


def step(g, st, log=None):
    pos, fx, fa, en = st["pos"], st["food_xy"], st["food_amt"], st["energy"]
    # sense + emit
    for i in range(N):
        d = fx - pos[i]; dist = np.linalg.norm(d, axis=1); v = (dist < VIS) & (fa > 0.06)
        if v.any():
            j = np.where(v)[0][np.argmin(dist[v])]; u = d[j]/max(dist[j], 1e-6)
            st["fdir"][i] = u; st["seen"][i] = True; feat = np.array([u[0], u[1], 1.0])
        else:
            st["seen"][i] = False; st["fdir"][i] = 0; feat = np.array([0., 0., 1.])
        st["sig"][i] = int(np.argmax(feat @ g["Wspk"][i]))
    # move
    st["heard"][:] = -1; newpos = pos.copy()
    for i in range(N):
        dd = np.zeros(2)
        if st["seen"][i]:
            dd += st["fdir"][i]
        else:
            dn = pos - pos[i]; dist = np.linalg.norm(dn, axis=1)
            cand = np.where((dist > 1e-6) & (dist < HEAR) & st["seen"])[0]
            if len(cand):
                j = int(cand[np.argmin(dist[cand])]); st["heard"][i] = j
                target = pos[j] + g["Wlis"][i][st["sig"][j]] * 8.0
                w = target - pos[i]; dd += g["wsig"][i]*w/max(np.linalg.norm(w), 1e-6)
        dd += 0.45*rng.normal(0, 1, 2); nd = np.linalg.norm(dd)
        newpos[i] = np.clip(pos[i] + (dd/nd if nd > 1e-6 else 0), 0, L)
    st["pos"] = newpos; pos = newpos
    # eat: INDIVIDUAL, scarcity dilution (no cooperation bonus)
    nearest = np.array([int(np.argmin(np.linalg.norm(fx - pos[i], axis=1))) for i in range(N)])
    ndist = np.array([np.linalg.norm(fx[nearest[i]] - pos[i]) for i in range(N)])
    onf = (ndist < 4) & (fa[nearest] > 0.02)
    for i in range(N):
        if onf[i]:
            j = nearest[i]
            b = min(BITE, fa[j]); en[i] += b*3.0; fa[j] -= b      # SCARCITY via DEPLETION
    en -= MET                                  # ENERGY law
    fa[:] = np.minimum(4.0, fa + REGEN)        # food regrows
    if log is not None:
        for i in range(N):
            if st["seen"][i]:
                log["emit"].append((st["sig"][i], float(np.arctan2(st["fdir"][i][1], st["fdir"][i][0]))))
            if st["heard"][i] >= 0:
                log["edges"].append((int(st["heard"][i]), int(i)))
    # steady-state survival + LOCAL reproduction (offspring by the PARENT -> kin clusters)
    order = np.argsort(en); worst = order[:3]; top = order[N - N//3:]
    med = float(np.median(en))
    for w in worst:
        p = int(top[rng.integers(len(top))])
        g["Wspk"][w] = g["Wspk"][p] + rng.normal(0, 0.18, (3, K))
        g["Wlis"][w] = g["Wlis"][p] + rng.normal(0, 0.18, (K, 2))
        g["wsig"][w] = g["wsig"][p] + rng.normal(0, 0.1)
        st["pos"][w] = np.clip(st["pos"][p] + rng.normal(0, 4, 2), 0, L)   # KIN placed by parent
        en[w] = med
    st["energy"][:] = np.clip(en, -2, 28)


def mi_of(log):
    sigs = np.array([s for s, a in log["emit"]]); angs = [a for s, a in log["emit"]]
    a = ((np.array(angs)+np.pi)/(2*np.pi)*8).astype(int) % 8
    J = np.zeros((K, 8))
    for ss, aa in zip(sigs, a): J[ss, aa] += 1
    if J.sum() == 0: return 0.0
    P = J/J.sum(); pm = P.sum(1, keepdims=True); ps = P.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(P*(np.log2(P)-np.log2(pm)-np.log2(ps))))


def main():
    global VIS
    g, st = init()
    print(f"params: VIS={VIS} food={NF} MET={MET}  (pure individual + local repro, no coupling)", flush=True)
    VISES = [20,17,14,12,10,8,7,6]
    for blk in range(8):
        VIS = VISES[blk]            # RAMP perception harder -> organisms get lost gradually
        for t in range(500):
            step(g, st)
        log = {"emit": [], "edges": []}; sf = []
        for t in range(60):
            step(g, st, log=log); sf.append(st["seen"].mean())
        print(f"  gen ~{(blk+1)*500:5}: seen-frac={np.mean(sf):.2f} meanE={st['energy'].mean():5.1f} "
              f"alive={(st['energy']>0).sum():2}/{N} edges={len(log['edges']):4} MI={mi_of(log):.2f} bits", flush=True)


if __name__ == "__main__":
    main()
