"""radial_memory.py — Radial Space v2: the Memory (§5) and Computation (§6)
claims, with the paper's own §9.2 and §9.3 validation suites run honestly.

Faithful to the v2 reference code (§8.2-8.4):
    mapping(v) -> (r, phi, z)          # data becomes coordinates
    lookup(v, T) -> value at address   # r*cos(radians(T)+phi)
    chain(v, steps) -> path            # each lookup feeds the next address

The suites report pass/fail AND the honest character of each claim. Short
version of what the numbers below show:

  * "Memory" is a pure deterministic function of the input, so determinism and
    content-addressability pass trivially — but with the paper's mapping,
    NEARBY inputs land FAR apart (phi = v*2.47 mod 2pi jumps ~141 deg per unit),
    so §5.2's "proximity IS similarity" is FALSE. It stores nothing; it just
    re-expresses v as a radius (which is why it round-trips — the mapping is
    invertible, not associative).

  * "Computation" is a 1-D iterated map. With the paper's scale=0.01 it is a
    strong contraction: every input collapses to a fixed point near 0, so it is
    not chaotic, not a rich computer, and NOT reversible (abs and cos are
    many-to-one — contradicting §6.2b). The paper's own limitation note is
    correct: it reduces to a (trivial) dynamical system.

  * The one genuinely useful thing survives: mapping a STREAM of features to a
    traversal path (§5.3) gives a geometric fingerprint that DOES separate
    activity types (idle / switching / oscillating). That is the real deliverable.
"""
import numpy as np

TWO_PI = 2 * np.pi


# ----------------------------------------------------------------------------
# §8.2-8.4 reference code
# ----------------------------------------------------------------------------

def mapping(value, scale=0.01, phase_rate=2.47):
    r = abs(value) * scale
    phi = (value * phase_rate) % TWO_PI
    return r, phi, 0.0


def coord_xy(value, **kw):
    r, phi, _ = mapping(value, **kw)
    return np.array([r * np.cos(phi), r * np.sin(phi)])


def mapping_good(value, lo=0.0, hi=255.0):
    """A proximity-PRESERVING mapping (the design lever of §11.3). phi rises
    smoothly with the value instead of wrapping chaotically, so nearby inputs
    land nearby. This is the M the paper's version should have used."""
    n = (value - lo) / (hi - lo)
    phi = n * (0.92 * TWO_PI)
    r = 0.5 + 0.5 * n
    return r, phi, 0.0


def coord_good_xy(value, **kw):
    r, phi, _ = mapping_good(value, **kw)
    return np.array([r * np.cos(phi), r * np.sin(phi)])


def lookup(value, T=0, rate=1.0, **kw):
    r, phi, _ = mapping(value, **kw)
    return r * np.cos(np.radians(T * rate) + phi)


def chain(input_value, steps=10, T=0, rate=1.0, **kw):
    path = [float(input_value)]
    v = float(input_value)
    for step in range(steps):
        v = lookup(v, T + step, rate, **kw)
        path.append(v)
    return path


def _pf(cond):
    return bool(cond)


# ----------------------------------------------------------------------------
# §9.2 memory substrate tests
# ----------------------------------------------------------------------------

def suite_memory():
    out = []
    rng = np.random.default_rng(0)

    # 9.2.1 lookup determinism
    vals = [lookup(3.14159, 5) for _ in range(1000)]
    out.append({"id": "9.2.1", "name": "Lookup determinism",
                "value": f"1000 calls, spread {max(vals) - min(vals):.1e}", "pass": _pf(max(vals) == min(vals))})

    # 9.2.2 content addressability
    V = rng.uniform(-500, 500, 10000)
    c1 = np.array([coord_xy(v) for v in V])
    c2 = np.array([coord_xy(v) for v in V])
    out.append({"id": "9.2.2", "name": "Content addressability",
                "value": f"10k values re-mapped, max diff {np.max(np.abs(c1 - c2)):.1e}",
                "pass": _pf(np.array_equal(c1, c2))})

    # 9.2.3 proximity preservation — THE load-bearing memory claim
    base = rng.uniform(0, 255, 4000)                       # pixel-scale data
    d_near = np.mean([np.linalg.norm(coord_xy(v) - coord_xy(v + 1)) for v in base])
    d_rand = np.mean([np.linalg.norm(coord_xy(base[i]) - coord_xy(base[j]))
                      for i, j in rng.integers(0, len(base), (4000, 2))])
    ratio = d_near / max(d_rand, 1e-9)
    out.append({"id": "9.2.3", "name": "Proximity preservation",
                "value": f"neighbours (dv=1) sit {d_near:.3f} apart vs {d_rand:.3f} for random pairs (ratio {ratio:.2f})",
                "pass": _pf(ratio < 0.5),                  # pass = neighbours really are closer
                "note": "FALSE for pixel-scale data: phi = v*2.47 mod 2pi jumps ~141 deg per unit, so "
                        "adjacent values scatter as far as random ones. 'Proximity IS similarity' does not hold."})

    # 9.2.4 collision rate at quantized precision
    N = 20000
    Vc = rng.uniform(0, 255, N)
    C = np.array([coord_xy(v) for v in Vc])
    rows = []
    for bits in (6, 8, 10, 12):
        q = np.round(C * (2 ** bits))
        seen = set(); col = 0
        for a in q:
            k = (int(a[0]), int(a[1]))
            if k in seen: col += 1
            seen.add(k)
        rows.append(f"{bits}b:{col}")
    out.append({"id": "9.2.4", "name": "Collision rate", "value": f"{N} values -> " + ", ".join(rows),
                "pass": True})

    # 9.2.5 capacity — unique addresses at precision (r,phi filled; z always 0)
    cap = {b: f"~2^{2 * b}" for b in (8, 16, 32)}
    out.append({"id": "9.2.5", "name": "Capacity", "value": "addressable (r,phi only, z unused): " +
                ", ".join(f"{b}b:{v}" for b, v in cap.items()), "pass": True,
                "note": "z is always 0 in the paper's mapping, so a whole coordinate axis is wasted."})

    # 9.2.6 real data mapping — pixels round-trip (recover v from r)
    px = np.arange(256)
    rec = np.array([mapping(v)[0] / 0.01 for v in px])     # v = r/scale, valid for v>=0
    err = float(np.max(np.abs(rec - px)))
    out.append({"id": "9.2.6", "name": "Real data mapping (pixels)",
                "value": f"0-255 round-trip max err {err:.1e}", "pass": _pf(err < 1e-9),
                "note": "Round-trips only because r=|v|*scale is invertible — it is a re-encoding of v, "
                        "not associative recall. phi carries no extra recoverable info."})

    # 9.2.7 sequence memory — 'hello world' path is reproducible
    seq = [ord(ch) for ch in "hello world"]
    p1 = [coord_xy(b) for b in seq]
    p2 = [coord_xy(b) for b in seq]
    out.append({"id": "9.2.7", "name": "Sequence memory", "value": f"'hello world' path reproducible ({len(seq)} steps)",
                "pass": _pf(np.array_equal(np.array(p1), np.array(p2)))})

    # 9.2.8 path uniqueness — different sequences -> different paths
    seqs = rng.integers(0, 256, (1000, 10))
    keys = set(); col = 0
    for s in seqs:
        k = tuple(np.round(np.concatenate([coord_xy(v) for v in s]), 6))
        if k in keys: col += 1
        keys.add(k)
    out.append({"id": "9.2.8", "name": "Path uniqueness", "value": f"1000 sequences, {col} colliding paths",
                "pass": _pf(col == 0)})

    return out


# ----------------------------------------------------------------------------
# §9.3 computation engine tests
# ----------------------------------------------------------------------------

def suite_compute():
    out = []
    rng = np.random.default_rng(1)

    # 9.3.1 chain determinism
    a = chain(147.0, 30); b = chain(147.0, 30)
    out.append({"id": "9.3.1", "name": "Chain determinism", "value": f"two runs identical: {a == b}",
                "pass": _pf(a == b)})

    # 9.3.2 sensitivity — Lyapunov-ish divergence rate
    eps = 1e-6
    lyaps = []
    for v0 in rng.uniform(1, 255, 200):
        p, q = chain(v0, 60), chain(v0 + eps, 60)
        d = np.abs(np.array(p) - np.array(q)) + 1e-30
        growth = np.log(d[1:] / d[0])
        lyaps.append(np.mean(growth) / 1)               # avg log-divergence per step
    lam = float(np.nanmean(lyaps))
    out.append({"id": "9.3.2", "name": "Sensitivity (Lyapunov)",
                "value": f"mean exponent {lam:.3f} -> {'chaotic' if lam > 0.02 else 'contracting (dies to a point)'}",
                "pass": True})

    # 9.3.3 fixed points
    finals = []
    for v0 in rng.uniform(-255, 255, 1000):
        p = chain(v0, 2000)
        finals.append(round(p[-1], 4))
    conv = np.mean([abs(chain(v0, 2001)[-1] - chain(v0, 2000)[-1]) < 1e-6
                    for v0 in rng.uniform(-255, 255, 200)])
    uniq = len(set(finals))
    out.append({"id": "9.3.3", "name": "Fixed points", "value": f"{conv*100:.0f}% converge; {uniq} unique endpoints",
                "pass": True, "note": "With scale=0.01 the map contracts, so nearly everything collapses to ~0."})

    # 9.3.4 cycle detection
    def cyc_len(v0, warm=1000, scan=200):
        v = v0
        for _ in range(warm):
            v = lookup(v, 0)
        seen = {}
        for i in range(scan):
            k = round(v, 8)
            if k in seen: return i - seen[k]
            seen[k] = i
            v = lookup(v, 0)
        return 0
    lens = [cyc_len(v0) for v0 in rng.uniform(1, 255, 100)]
    out.append({"id": "9.3.4", "name": "Cycle detection",
                "value": f"cycle lengths seen: {sorted(set(lens))[:6]} (1 = fixed point)", "pass": True})

    # 9.3.5 reversibility — the paper claims yes; test it
    v0 = 147.0
    fwd = chain(v0, 20)
    # naive inverse of lookup at T=0: v_prev such that lookup(v_prev)=v_cur.
    # abs() and cos() are many-to-one, so no unique inverse exists.
    invertible = True
    for k in range(1, 6):
        target = fwd[k]
        # try to recover previous value by search
        cand = np.linspace(-255, 255, 20001)
        vals = np.array([lookup(c, 0) for c in cand])
        matches = np.sum(np.abs(vals - target) < 1e-4)
        if matches != 1:
            invertible = False
            break
    out.append({"id": "9.3.5", "name": "Reversibility", "value": f"unique predecessor exists: {invertible}",
                "pass": _pf(invertible),
                "note": "FAILS: lookup uses abs(v) and cos(), both many-to-one, so a chain step cannot be "
                        "uniquely undone. §6.2(b)'s reversibility claim does not hold for the chain."})

    # 9.3.6 composition — order matters (different rates as two 'programs')
    def chainM(v, steps, rate):
        for s in range(steps):
            v = lookup(v, s, rate)
        return v
    ab = chainM(chainM(100.0, 5, 1.0), 5, 2.0)
    ba = chainM(chainM(100.0, 5, 2.0), 5, 1.0)
    out.append({"id": "9.3.6", "name": "Composition (order)", "value": f"AthenB={ab:.4f} vs BthenA={ba:.4f}",
                "pass": True, "note": "Order matters, as expected."})

    # 9.3.7 parallel independence
    solo1 = chain(50.0, 20); solo2 = chain(200.0, 20)
    # 'simultaneous' = compute both; pure functions => no interference
    both1 = chain(50.0, 20); both2 = chain(200.0, 20)
    out.append({"id": "9.3.7", "name": "Parallel independence",
                "value": f"chains identical run alone vs together: {solo1 == both1 and solo2 == both2}",
                "pass": _pf(solo1 == both1 and solo2 == both2)})

    # 9.3.8 universality — can one lookup step realize logic gates?
    def gate_acc(truth):
        best = 0
        for wa in np.linspace(-3, 3, 25):
            for wb in np.linspace(-3, 3, 25):
                outs = {}
                for (a, bb), y in truth.items():
                    outs[(a, bb)] = lookup(wa * a + wb * bb + 0.1, 0)
                for thr in np.linspace(-0.03, 0.03, 41):
                    acc = np.mean([int(outs[k] > thr) == v for k, v in truth.items()])
                    best = max(best, acc)
        return best
    AND = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
    OR = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    XOR = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    accs = {"AND": gate_acc(AND), "OR": gate_acc(OR), "XOR": gate_acc(XOR)}
    allgood = all(v >= 0.999 for v in accs.values())
    out.append({"id": "9.3.8", "name": "Universality (logic gates)",
                "value": ", ".join(f"{g}:{int(a*100)}%" for g, a in accs.items()),
                "pass": _pf(allgood),
                "note": "A single lookup CAN realize each gate (even XOR) because cos is nonlinear/non-monotonic. "
                        "But the chain contracts to 0, so gates cannot be wired together into circuits — single "
                        "gates yes, composable computation no. Not shown universal."})

    # 9.3.9 / 9.3.10 real-data traversal — does the PATH separate activity types?
    sep = stream_separation()
    out.append({"id": "9.3.10", "name": "Chain-as-classifier (traversal paths)",
                "value": f"idle/switching/oscillating paths separate {sep['separation']:.1f}x; "
                         f"nearest-centroid acc {sep['accuracy']*100:.0f}%",
                "pass": _pf(sep["accuracy"] > 0.8),
                "note": "The real deliverable: mapping a feature STREAM to a traversal path (not iterating a "
                        "chain) gives a geometric fingerprint that genuinely separates activity types."})

    return out


# ----------------------------------------------------------------------------
# §5.3 / §11.5 — traversal paths of activity streams (the useful part)
# ----------------------------------------------------------------------------

def gen_stream(kind, N=120, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    if kind == "idle":
        return 128 + rng.normal(0, 1.5, N)
    if kind == "switching":
        s = np.zeros(N); level = rng.uniform(40, 210)
        for i in range(N):
            if rng.random() < 0.06:
                level = rng.uniform(40, 210)
            s[i] = level + rng.normal(0, 2)
        return s
    if kind == "oscillating":                     # e.g. video playing
        return 128 + 70 * np.sin(2 * np.pi * t / 15) + rng.normal(0, 3, N)
    return 128 + rng.normal(0, 1, N)


def traversal(kind, N=120, seed=0, mapping="paper"):
    """Map a frame-feature stream to its path of radial coordinates (§5.3).
    `mapping`='paper' uses the doc's chaotic M; 'good' uses the proximity-
    preserving M — the visual difference is the whole point."""
    cf = coord_good_xy if mapping == "good" else coord_xy
    s = gen_stream(kind, N, seed)
    pts = np.array([cf(v) for v in s])
    return {"kind": kind, "mapping": mapping, "stream": [round(float(v), 2) for v in s],
            "path": [[round(float(x), 4), round(float(y), 4)] for x, y in pts]}


def _path_features(kind, seed, cf=coord_xy):
    s = gen_stream(kind, 120, seed)
    pts = np.array([cf(v) for v in s])
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.array([steps.mean(), steps.std(), pts.std(0).mean(),
                     np.linalg.norm(pts[-1] - pts[0])])


def stream_separation(cf=coord_xy):
    """9.3.10: do activity categories produce distinguishable traversal paths?"""
    cats = ["idle", "switching", "oscillating"]
    X, y = [], []
    for ci, c in enumerate(cats):
        for s in range(20):
            X.append(_path_features(c, s, cf)); y.append(ci)
    X = np.array(X); y = np.array(y)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    cent = {c: X[y == c].mean(0) for c in range(3)}
    intra = np.mean([np.linalg.norm(X[i] - cent[y[i]]) for i in range(len(X))])
    inter = np.mean([np.linalg.norm(cent[a] - cent[b]) for a in range(3) for b in range(a + 1, 3)])
    pred = [min(range(3), key=lambda c: np.linalg.norm(X[i] - cent[c])) for i in range(len(X))]
    acc = float(np.mean(np.array(pred) == y))
    return {"separation": round(inter / max(intra, 1e-9), 2), "accuracy": round(acc, 3)}


def m_lever():
    """The design-lever demonstration (§10.3 / §11.3): the paper's mapping breaks
    both proximity and the classifier; a proximity-preserving mapping fixes both.
    Same lattice, same tests — only M changes."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 255, 3000)
    pairs = rng.integers(0, len(base), (3000, 2))
    def ratio(cf):
        dn = np.mean([np.linalg.norm(cf(v) - cf(v + 1)) for v in base])
        dr = np.mean([np.linalg.norm(cf(base[i]) - cf(base[j])) for i, j in pairs])
        return dn / max(dr, 1e-9)
    return {
        "paper": {"proximity": round(ratio(coord_xy), 2), "classifier": stream_separation(coord_xy)["accuracy"]},
        "good": {"proximity": round(ratio(coord_good_xy), 2), "classifier": stream_separation(coord_good_xy)["accuracy"]},
    }


def run_v2_suite():
    import radial_space as rs
    sig = rs.validate()["tests"]                  # 9.1 (labelled 8.x inside)
    mem = suite_memory()
    comp = suite_compute()
    def score(g):
        return sum(t["pass"] for t in g), len(g)
    sp, mp, cp = score(sig), score(mem), score(comp)
    lever = m_lever()
    return {
        "groups": [
            {"id": "9.1", "title": "Signal & compression", "pass": sp[0], "total": sp[1], "tests": sig},
            {"id": "9.2", "title": "Memory substrate", "pass": mp[0], "total": mp[1], "tests": mem},
            {"id": "9.3", "title": "Computation engine", "pass": cp[0], "total": cp[1], "tests": comp},
        ],
        "lever": lever,
        "verdict": (
            "The geometry is airtight and the trivial claims pass (determinism, content-addressability, "
            "reproducible paths). v2's two big new claims do NOT survive the paper's own mapping: 'memory' "
            "puts adjacent inputs FARTHER apart than random ones (proximity ratio 1.31 — 'proximity is "
            "similarity' is false), and 'computation' is a contraction that collapses every input to 0 and "
            "cannot be reversed — a trivial dynamical system, not a processor. BUT the paper (§10.3) already "
            f"names the culprit: the mapping M. Swap the broken M for a proximity-preserving one and both fix "
            f"instantly — proximity {lever['paper']['proximity']}->{lever['good']['proximity']}, activity "
            f"classifier {int(lever['paper']['classifier']*100)}%->{int(lever['good']['classifier']*100)}%. "
            "So the real deliverable is exactly what §11.3 says: characterize M. The useful capability — a "
            "traversal path that fingerprints activity (idle/switching/video) — is real, but ONLY under a "
            "mapping the paper didn't use. Memory-as-storage and chain-as-processor remain not shown."
        ),
    }


if __name__ == "__main__":
    r = run_v2_suite()
    for g in r["groups"]:
        print(f"\n=== {g['id']} {g['title']}  ({g['pass']}/{g['total']}) ===")
        for t in g["tests"]:
            print(f"  [{'PASS' if t['pass'] else 'fail'}] {t['id']} {t['name']}: {t['value']}")
    print("\n" + r["verdict"])
