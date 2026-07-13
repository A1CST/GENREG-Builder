"""radial_map.py — Radial map v2, the WAY-simpler rebuild.

The idea (from the user's activation-map research: tens of thousands of
activation functions characterized by how they TRANSFORM real data — behavior,
not formula — self-organize into a galaxy: linear near the centre,
nonlinearity outward, oscillators on their own branches):

  1. A deterministic, index-addressed lens space. Lens i's program is seeded
     by i alone, so the map is fixed: same address, same lens, every time.
  2. Feed every lens the SAME baseline data (numerical loops first; images and
     text later use the same machinery with a different stream) and record a
     behavioral signature — how the lens transforms that data.
  3. Project signatures to 2D (classical MDS). That picture is the baseline
     map for the data type.
  4. Stack a plain closed-form LINEAR model on the lens bank. If the lens
     views are diverse/orthogonal enough, linear-on-lenses solves what plain
     linear cannot — the diversity does the heavy lifting, not learned
     nonlinearity.

No gradients anywhere. No training in the map itself — it is enumerated, not
searched. Everything is deterministic given (index, data kind).
"""
import numpy as np

# ---------------------------------------------------------------------------
# 1. the lens space — primitive ops, index-addressed programs
# ---------------------------------------------------------------------------

PRIMS = {
    "id":    lambda z: z,
    "sin":   np.sin,
    "cos":   np.cos,
    "tanh":  np.tanh,
    "relu":  lambda z: np.maximum(0.0, z),
    "abs":   np.abs,
    "sq":    lambda z: np.clip(z * z, -30, 30),
    "sign":  np.sign,
    "gauss": lambda z: np.exp(-np.clip(z * z, 0, 30)),
    "soft":  lambda z: z / (1.0 + np.abs(z)),          # softsign
    "sqrt":  lambda z: np.sqrt(np.abs(z)),
    "step":  lambda z: (z > 0).astype(np.float64),
    "logp":  lambda z: np.log1p(np.abs(z)) * np.sign(z),
    "sinc":  lambda z: np.sinc(z / np.pi),
}
_PN = sorted(PRIMS)


def lens_program(i):
    """Deterministic program for lens i: a chain of 1-3 stages, each
    prim(a*x + b). Lens 0 is the pure identity anchor (the map's centre)."""
    if i == 0:
        return [("id", 1.0, 0.0)]
    rng = np.random.default_rng(i)
    depth = 1 + int(rng.integers(0, 3))
    return [(_PN[int(rng.integers(len(_PN)))],
             float(rng.uniform(0.5, 3.0)),
             float(rng.uniform(-1.0, 1.0))) for _ in range(depth)]


def lens_apply(prog, x):
    y = x
    for name, a, b in prog:
        y = PRIMS[name](a * y + b)
    return y


def prog_str(prog):
    # ASCII only — cp1252 console-safe (composition reads right-to-left)
    return " . ".join(f"{n}({a:.2f}x{b:+.2f})" for n, a, b in reversed(prog))


# ---------------------------------------------------------------------------
# 2. baseline data streams — numbers (loops) first; other kinds plug in here
# ---------------------------------------------------------------------------

def data_stream(kind="loops", n=2400, seed=0):
    """A 1D sample stream for a data type. 'loops' is the numeric baseline:
    simple periodic signals, exactly repeating — predictable input so the map
    shows what each lens does to KNOWN structure. 'noise' is the contrast."""
    t = np.arange(n)
    if kind == "loops":
        k = n // 6
        parts = [
            np.sin(2 * np.pi * t[:k] / 60),                       # slow sine loop
            np.sin(2 * np.pi * t[:k] / 13),                       # fast sine loop
            2 * (t[:k] % 40) / 40 - 1,                            # sawtooth loop
            2 * np.abs(2 * ((t[:k] % 50) / 50) - 1) - 1,          # triangle loop
            np.sign(np.sin(2 * np.pi * t[:k] / 30)),              # square loop
            np.sin(2 * np.pi * t[:k] / 60) * np.sin(2 * np.pi * t[:k] / 7),  # product loop
        ]
        x = np.concatenate(parts)
    elif kind == "noise":
        x = np.random.default_rng(seed).standard_normal(n)
    else:
        raise ValueError(f"unknown data kind '{kind}' (have: loops, noise)")
    x = x - x.mean()
    return x / (x.std() + 1e-9)


# ---------------------------------------------------------------------------
# 3. behavioral signature — how a lens transforms the stream
# ---------------------------------------------------------------------------

def _sig(prog, x, xq):
    """Signature = the lens's response curve over the data's own quantile grid
    (the transform's shape as seen through this data's distribution) plus a
    few scalar behavior stats. Everything computed from behavior, never from
    the formula."""
    y = lens_apply(prog, x)
    ys = y.std()
    yz = (y - y.mean()) / (ys + 1e-9)
    curve = lens_apply(prog, xq)
    cs = curve.std()
    curve = (curve - curve.mean()) / (cs + 1e-9) if cs > 1e-9 else curve * 0.0
    dc = np.diff(curve)
    corr = float(np.corrcoef(x, yz)[0, 1]) if ys > 1e-9 else 0.0
    stats = np.array([
        corr,                                                    # linearity
        float(np.mean(np.abs(yz) > 2.0)),                        # tail mass
        float(np.mean(np.abs(yz) < 0.1)),                        # dead zone
        float(np.mean(np.sign(yz[1:]) != np.sign(yz[:-1]))),     # zero-cross rate
        float(np.mean(np.sign(dc[1:]) != np.sign(dc[:-1]))) if len(dc) > 1 else 0.0,  # oscillation
        float(np.mean(dc >= 0)),                                 # monotonic frac
        float(np.tanh(ys)),                                      # gain (bounded)
    ])
    return np.concatenate([curve, stats]), corr, stats[4]


def build_signatures(n_lens=1200, kind="loops", n_curve=48):
    x = data_stream(kind)
    xq = np.quantile(x, np.linspace(0.01, 0.99, n_curve))
    sigs, nl, osc, progs = [], [], [], []
    for i in range(n_lens):
        p = lens_program(i)
        s, corr, o = _sig(p, x, xq)
        sigs.append(s); nl.append(1.0 - abs(corr)); osc.append(o)
        progs.append(prog_str(p))
    return np.array(sigs), np.array(nl), np.array(osc), progs


# ---------------------------------------------------------------------------
# 4. the map — classical MDS of signature distances, centred on identity
# ---------------------------------------------------------------------------

def build_map(n_lens=1200, kind="loops"):
    Sraw, nl, osc, progs = build_signatures(n_lens, kind)
    S = (Sraw - Sraw.mean(0)) / (Sraw.std(0) + 1e-9)
    D2 = np.maximum(0.0, (S ** 2).sum(1)[:, None] + (S ** 2).sum(1)[None, :] - 2 * S @ S.T)
    J = np.eye(n_lens) - 1.0 / n_lens
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    XY = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0.0))
    XY = XY - XY[0]                       # identity lens = origin
    r = np.linalg.norm(XY, axis=1)
    # honest checks: does the galaxy structure hold?
    rad_nl = float(np.corrcoef(r, nl)[0, 1])            # nonlinearity grows outward?
    k = min(n_lens, 50)                                 # raw signatures reproduce exactly
    s2, _, _, _ = build_signatures(k, kind)
    det_err = float(np.abs(Sraw[:k] - s2[:k]).max())
    return {
        "kind": kind, "n": n_lens,
        "pts": [{"i": i, "x": round(float(XY[i, 0]), 3), "y": round(float(XY[i, 1]), 3),
                 "nl": round(float(nl[i]), 3), "osc": round(float(osc[i]), 3),
                 "prog": progs[i]} for i in range(n_lens)],
        "checks": {"radius_vs_nonlinearity_corr": round(rad_nl, 3),
                   "determinism_err": det_err,
                   "identity_at_origin": True},
    }


def lens_detail(i, kind="loops", n_curve=120):
    """Response curve of lens i over the data's range — for click-inspection."""
    x = data_stream(kind)
    xs = np.linspace(float(x.min()), float(x.max()), n_curve)
    p = lens_program(int(i))
    return {"i": int(i), "prog": prog_str(p),
            "xs": [round(float(v), 4) for v in xs],
            "ys": [round(float(v), 4) for v in lens_apply(p, xs)]}


# ---------------------------------------------------------------------------
# 5. the linear test — does lens diversity alone make hard targets linear?
# ---------------------------------------------------------------------------

def _ridge_r2(F, y, lam=1e-3):
    """Closed-form ridge, 60/40 split, heldout R². No gradients."""
    n = len(y); ntr = int(n * 0.6)
    idx = np.random.default_rng(7).permutation(n)
    tr, te = idx[:ntr], idx[ntr:]
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    Ftr = np.hstack([(F[tr] - mu) / sd, np.ones((len(tr), 1))])
    Fte = np.hstack([(F[te] - mu) / sd, np.ones((len(te), 1))])
    W = np.linalg.solve(Ftr.T @ Ftr + lam * np.eye(Ftr.shape[1]), Ftr.T @ y[tr])
    resid = y[te] - Fte @ W
    return 1.0 - float(resid.var() / (y[te].var() + 1e-12))


TASKS = {
    "square":   lambda x: x * x,
    "abs":      lambda x: np.abs(x),
    "sin3x":    lambda x: np.sin(3 * x),
    "step":     lambda x: (x > 0.3).astype(np.float64),
    "ripple":   lambda x: np.sin(4 * x) * np.exp(-x * x),
}


def probe(n_lens=400, kind="loops"):
    """Plain linear on raw x vs the SAME linear model on the lens bank.
    Targets are nonlinear functions of the stream — unsolvable by a line."""
    x = data_stream(kind)
    bank = np.stack([lens_apply(lens_program(i), x) for i in range(n_lens)], 1)
    sd = bank.std(0)
    bank = bank[:, sd > 1e-9]
    rows = []
    for name, f in TASKS.items():
        y = f(x)
        rows.append({"task": name,
                     "r2_linear": round(_ridge_r2(x[:, None], y), 4),
                     "r2_lens": round(_ridge_r2(bank, y), 4),
                     "n_lens": int(bank.shape[1])})
    return {"kind": kind, "rows": rows}


if __name__ == "__main__":
    import sys, json
    if "probe" in sys.argv:
        r = probe()
        print(f"linear model, raw x vs {r['rows'][0]['n_lens']}-lens bank ({r['kind']}):")
        for row in r["rows"]:
            print(f"  {row['task']:<8} raw R2 {row['r2_linear']:+.4f}   lens R2 {row['r2_lens']:+.4f}")
    else:
        m = build_map(n_lens=int(sys.argv[sys.argv.index("map") + 1]) if "map" in sys.argv[:-1] else 1200)
        print(json.dumps(m["checks"], indent=2))
        r = np.array([[p["x"], p["y"]] for p in m["pts"]])
        print(f"{m['n']} lenses mapped, radius range {np.linalg.norm(r, axis=1).min():.2f}"
              f"–{np.linalg.norm(r, axis=1).max():.2f}")
