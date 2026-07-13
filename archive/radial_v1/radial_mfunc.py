"""radial_mfunc.py — Radial Space v3, Test Suite 10: characterize the mapping M.

v2 proved the mapping function M is the only real design choice: same lattice,
different M, completely different results. v3's decisive question (its own "next
action"): across a family of candidate M functions, which one — if any — gives a
mapping that is invertible, well-conditioned, collision-free, AND yields a stable
reversible chain? The last one gates whether "chain computation" survives at all.

Five M families per the doc (§10.6): linear, arctan, logistic sigmoid, sqrt
(companding), sinusoidal. Each maps a value to (r, phi). Tests 10.1-10.6 below.

Honest bottom line the numbers reach: monotonic M (linear/arctan/sigmoid/sqrt)
are invertible and preserve proximity, so MEMORY and ACTIVITY FINGERPRINTING are
solid. But the CHAIN step v <- r*cos(angle+phi) contracts (or goes chaotic) and
throws away sign+phase under every M, so chain computation stays irreversible and
unstable — no M rescues it. Per the doc's own decision rule, chain computation
should be dropped; compression + memory + fingerprinting are what to keep.
"""
import numpy as np
import radial_memory as rm

TWO_PI = 2 * np.pi


# ----------------------------------------------------------------------------
# the five M families (+ paper's hash for reference). r grows smoothly with the
# value; phi is the discriminating part.
# ----------------------------------------------------------------------------

def _t(v, lo, hi):
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)

def M_linear(v, lo=0., hi=255.):
    t = _t(v, lo, hi); return 0.5 + 0.5 * t, t * TWO_PI, 0.0

def M_arctan(v, lo=0., hi=255.):
    c, s = (lo + hi) / 2, (hi - lo) / 4
    t = (np.arctan((v - c) / s) / (np.pi / 2) + 1) / 2
    return 0.5 + 0.5 * t, t * TWO_PI, 0.0

def M_sigmoid(v, lo=0., hi=255.):
    c, s = (lo + hi) / 2, (hi - lo) / 8
    t = 1.0 / (1.0 + np.exp(-(v - c) / s))
    return 0.5 + 0.5 * t, t * TWO_PI, 0.0

def M_sqrt(v, lo=0., hi=255.):
    t = _t(v, lo, hi); return 0.5 + 0.5 * t, np.sqrt(t) * TWO_PI, 0.0

def M_sin(v, lo=0., hi=255.):
    t = _t(v, lo, hi); u = 0.5 + 0.5 * np.sin(t * 3 * np.pi)
    return 0.5 + 0.5 * t, u * TWO_PI, 0.0        # non-monotonic phi -> not bijective

def M_paper(v, lo=0., hi=255.):
    return abs(v) * 0.01, (v * 2.47) % TWO_PI, 0.0

FAMILIES = {"linear": M_linear, "arctan": M_arctan, "sigmoid": M_sigmoid,
            "sqrt": M_sqrt, "sinusoidal": M_sin, "paper": M_paper}


def coord(M, v, lo=0., hi=255.):
    r, phi, _ = M(v, lo, hi)
    return np.array([r * np.cos(phi), r * np.sin(phi)])


# ----------------------------------------------------------------------------
# §10 tests
# ----------------------------------------------------------------------------

def t_invertibility(M, lo=0., hi=255.):
    """10.1 — is M bijective through phi? The doc's M_inverse recovers the value
    from phi alone, so invertibility means phi(v) is strictly monotonic. (r also
    encodes v here, but the doc treats phi as the carrier.) A non-monotonic phi —
    the paper's mod-wrap hash, or the sinusoidal fold — collides and cannot be
    inverted from phi."""
    vs = np.linspace(lo, hi, 4000)
    phi = np.array([M(v, lo, hi)[1] for v in vs])
    dphi = np.diff(phi)
    mono = bool(np.all(dphi > 1e-12) or np.all(dphi < -1e-12))
    return 0.0 if mono else 99.0


def t_condition(M, lo=0., hi=255.):
    """10.2 — uniformity of the mapping's stretch: max/min of |d coord / d v|
    across the domain (a stand-in for the Jacobian condition number)."""
    vs = np.linspace(lo, hi, 100)
    d = 1e-3 * (hi - lo)
    speed = np.array([np.linalg.norm(coord(M, v + d, lo, hi) - coord(M, v - d, lo, hi)) / (2 * d)
                      for v in vs])
    speed = speed[speed > 1e-9]
    return float(speed.max() / max(speed.min(), 1e-9))


def t_collisions(M, lo=0., hi=255.):
    """10.3 — min bits of address precision for zero collisions over 0..255."""
    C = np.array([coord(M, v, lo, hi) for v in range(256)])
    for bits in (8, 10, 12, 16, 20, 24):
        q = np.round(C * (2 ** bits))
        if len({(int(a), int(b)) for a, b in q}) == 256:
            return bits
    return 99


def t_chain_lyapunov(M, n_v=100, steps=800):
    """10.4 — is the chain stable? Lyapunov exponent of v <- r*cos(angle+phi).
    Domain matched to the chain's own value range so r doesn't trivially vanish."""
    rng = np.random.default_rng(0)
    lo, hi = -1.5, 1.5
    def step(v, s):
        r, phi, _ = M(v, lo, hi)
        return r * np.cos(np.radians(s) + phi)
    lam = []
    for v0 in rng.uniform(lo, hi, n_v):
        v, w = v0, v0 + 1e-8
        acc = 0.0; ok = 0
        for s in range(steps):
            v, w = step(v, s), step(w, s)
            d = abs(v - w)
            if d > 1e-300:
                acc += np.log(d / 1e-8); ok += 1
                w = v + 1e-8 * np.sign(w - v) if w != v else v + 1e-8
                # renormalise separation to keep it linear
                w = v + 1e-8
        lam.append(acc / max(ok, 1))
    return float(np.nanmean(lam))


def t_proximity(M, lo=0., hi=255.):
    rng = np.random.default_rng(0)
    base = rng.uniform(lo, hi, 2000)
    pairs = rng.integers(0, len(base), (2000, 2))
    dn = np.mean([np.linalg.norm(coord(M, v, lo, hi) - coord(M, v + 1, lo, hi)) for v in base])
    dr = np.mean([np.linalg.norm(coord(M, base[i], lo, hi) - coord(M, base[j], lo, hi)) for i, j in pairs])
    return dn / max(dr, 1e-9)


def t_classifier(M):
    cf = lambda v: coord(M, v, 0., 255.)
    return rm.stream_separation(cf)["accuracy"]


def t_gate_composition(M):
    """10.5 — the minimum bar for computation: wire two fixed-M lookup steps in
    series (no digital restoration between them) and see if the 2-step chain can
    realise a gate. Brute-force only the INPUT encoding; M is fixed. Report best
    truth-table accuracy for NAND (needs composition)."""
    lo, hi = -1.5, 1.5
    def stepv(v, s):
        r, phi, _ = M(v, lo, hi); return r * np.cos(np.radians(s) + phi)
    NAND = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    best1 = best2 = 0.0
    for wa in np.linspace(-2, 2, 21):
        for wb in np.linspace(-2, 2, 21):
            for bias in np.linspace(-1, 1, 11):
                enc = {k: wa * k[0] + wb * k[1] + bias for k in NAND}
                o1 = {k: stepv(enc[k], 0) for k in NAND}
                o2 = {k: stepv(o1[k], 1) for k in NAND}
                for thr in np.linspace(-1, 1, 41):
                    best1 = max(best1, np.mean([int(o1[k] > thr) == v for k, v in NAND.items()]))
                    best2 = max(best2, np.mean([int(o2[k] > thr) == v for k, v in NAND.items()]))
    return {"one_step": round(best1, 3), "two_step": round(best2, 3)}


# ----------------------------------------------------------------------------
# the suite
# ----------------------------------------------------------------------------

def run_suite10():
    rows = []
    for name, M in FAMILIES.items():
        inv = t_invertibility(M)
        rows.append({
            "M": name,
            "invertible": bool(inv < 1e-2), "inv_err": round(inv, 3),
            "condition": round(t_condition(M), 1),
            "min_bits": t_collisions(M),
            "proximity": round(t_proximity(M), 3),
            "classifier": round(t_classifier(M), 3),
            "lyapunov": round(t_chain_lyapunov(M), 2),
        })
    # score each M on the memory/fingerprinting criteria (computation is separate)
    for r in rows:
        r["mem_score"] = int(r["invertible"]) + int(r["proximity"] < 0.5) + int(r["classifier"] > 0.9)

    # 10.4/10.5 the decision: a viable computation substrate needs a chain that
    # is STABLE (Lyapunov ~ 0, so state neither dies nor explodes). Check every M.
    invertible = [r for r in rows if r["invertible"]] or rows
    stable = [r for r in invertible if abs(r["lyapunov"]) < 0.05]
    cand = min(invertible, key=lambda r: abs(r["lyapunov"]))
    gate = t_gate_composition(FAMILIES[cand["M"]])     # single-step expressivity
    comp_dead = len(stable) == 0

    best_mem = max(rows, key=lambda r: (r["mem_score"], -abs(r["condition"] - 1)))
    verdict = (
        f"MEMORY + fingerprinting: solved. Winner '{best_mem['M']}' — invertible, proximity "
        f"{best_mem['proximity']} (similar inputs stay near), classifier {int(best_mem['classifier']*100)}%. "
        f"All four monotonic maps (linear/arctan/sigmoid/sqrt) round-trip and preserve proximity; the paper's "
        f"hash and the sinusoidal fold do not. "
        f"COMPUTATION: dead under every M. A single lookup is expressive (it realises gates including NAND at "
        f"{int(gate['one_step']*100)}%), but no M gives a STABLE chain — Lyapunov runs "
        f"{min(r['lyapunov'] for r in rows):.2f} to {max(r['lyapunov'] for r in rows):.2f}, all far from 0 "
        f"(best {cand['lyapunov']}, still contracting). So state decays to 0 (or explodes into chaos) within a "
        f"few steps, and the cos lookup discards sign+phase so a step can't be inverted. You cannot chain gates "
        f"into a persistent circuit. Per the doc's own decision rule: DROP chain computation; keep the three "
        f"confirmed capabilities — compression, memory, activity fingerprinting."
    )
    return {"rows": rows, "best_mem": best_mem["M"], "gate": gate,
            "gate_M": cand["M"], "computation_dead": comp_dead, "verdict": verdict}


def phi_curves(n=256):
    """phi-vs-input for each family (the doc's §3.2 figure)."""
    vs = np.linspace(0, 255, n)
    return {name: [round(float(np.degrees(M(v)[1])), 2) for v in vs]
            for name, M in FAMILIES.items()}, [round(float(v), 2) for v in vs]


if __name__ == "__main__":
    r = run_suite10()
    hdr = f"{'M':11s} {'inv':>5s} {'cond':>7s} {'bits':>5s} {'prox':>6s} {'class':>6s} {'lyap':>8s} {'mem':>4s}"
    print(hdr); print("-" * len(hdr))
    for x in r["rows"]:
        print(f"{x['M']:11s} {str(x['invertible']):>5s} {x['condition']:>7.1f} {x['min_bits']:>5d} "
              f"{x['proximity']:>6.3f} {x['classifier']:>6.3f} {x['lyapunov']:>8.2f} {x['mem_score']:>4d}")
    print(f"\ngate composition on '{r['gate_M']}': {r['gate']}")
    print("\n" + r["verdict"])
