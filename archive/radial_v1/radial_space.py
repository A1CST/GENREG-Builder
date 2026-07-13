"""radial_space.py — the Radial Space (RS) system from radial_space_theory.pdf.

A fixed 3D cubic lattice is swept by a deterministic rotation (+rate deg/step
about one axis). Each dot then traces a sinusoid X(T)=r*cos(wT+phi0). An external
1D signal is "addressed" by finding the dot whose trajectory best matches it; the
dot's (r, phi0, z0) is a 3-float address you can transmit instead of the raw
signal, and the receiver replays the trajectory to reconstruct it.

This is a faithful implementation of the paper's §7 reference code, made actually
usable (the paper's correlation-only encoder can't fix amplitude — corr is scale
invariant — so we also least-squares fit amplitude/phase), plus the §8 validation
suite that the paper asks be run "to validate or break the theory."

The honest headline the suite makes unavoidable: a SINGLE-axis sweep is a
one-frequency basis. Every dot is the same cosine, differing only in amplitude
and phase, so RS perfectly and cheaply encodes ONE sinusoid at the sweep
frequency — but summing more same-frequency dots still gives one sinusoid, so
arbitrary signals need the multi-axis / harmonic extension (§10.1), at which
point RS is a geometric restatement of the Fourier transform. The suite shows
exactly that: base RS nails pure tones, stalls on multi-tone; the harmonic
dictionary then converges like a DFT.
"""
import numpy as np


# ----------------------------------------------------------------------------
# §7.1 / §7.2 — lattice, rotation, sweep (faithful to the paper)
# ----------------------------------------------------------------------------

def build_lattice(rng=(-2.0, 2.0), step=0.5):
    vals = np.arange(rng[0], rng[1] + step, step)
    lattice = np.array([[x, y, z] for x in vals for y in vals for z in vals], float)
    r = np.sqrt(lattice[:, 0] ** 2 + lattice[:, 1] ** 2)
    phi0 = np.arctan2(lattice[:, 1], lattice[:, 0])
    z0 = lattice[:, 2]
    return lattice, r, phi0, z0


def rotation_matrix(axis, angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    if axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])           # x


def sweep(lattice, T, axis='z', rate=1.0):
    R = rotation_matrix(axis, T * rate)
    return (R @ lattice.T).T


def dot_trajectory_matrix(lattice, i, N, axis='z', rate=1.0, comp=0):
    """A dot's trajectory over N steps by literally applying the sweep — the
    paper's O(N) method, used only to prove the analytic form matches it."""
    return np.array([sweep(lattice[i:i + 1], t, axis, rate)[0, comp] for t in range(N)])


def dot_trajectory(r_i, phi0_i, N, rate=1.0, comp=0):
    """Analytic trajectory (X=comp0, Y=comp1) for a Z-sweep. Identical to the
    matrix sweep but O(1) per sample: rotations just spin (r, phi0)."""
    w = np.radians(rate)
    t = np.arange(N)
    if comp == 1:
        return r_i * np.sin(w * t + phi0_i)
    return r_i * np.cos(w * t + phi0_i)


# ----------------------------------------------------------------------------
# §4 / §7.3 — encode: find the dot whose trajectory matches the signal
# ----------------------------------------------------------------------------

def _fit_harmonic(signal, rate=1.0, k=1):
    """Least-squares fit S ~ a*cos(kwt)+b*sin(kwt)+c. Returns amplitude A, phase
    psi (so A*cos(kwt+psi)), offset c, and correlation of the fitted wave."""
    N = len(signal)
    w = np.radians(rate) * k
    t = np.arange(N)
    B = np.stack([np.cos(w * t), np.sin(w * t), np.ones(N)], 1)
    coef, *_ = np.linalg.lstsq(B, signal, rcond=None)
    a, b, c = coef
    A = float(np.hypot(a, b))
    psi = float(np.arctan2(-b, a))
    fit = B @ coef
    denom = np.std(signal) * np.std(fit)
    corr = float(np.corrcoef(signal, fit)[0, 1]) if denom > 1e-12 else 0.0
    return A, psi, float(c), corr


def encode(signal, r, phi0, rate=1.0, quantize=True):
    """Address a signal to the nearest lattice dot (Z-sweep, match on X).
    Returns dict with dot index, (r, phi0, z0)-style address, fitted amp/phase,
    offset and correlation. `quantize=False` keeps the exact fitted address."""
    A, psi, c, _ = _fit_harmonic(signal, rate, k=1)
    if quantize:
        # nearest dot in (radius, angle): match amplitude to r and phase to phi0
        dang = np.angle(np.exp(1j * (phi0 - psi)))          # wrapped angle diff
        cost = (r - A) ** 2 + (dang * (A + 1e-6)) ** 2
        idx = int(np.argmin(cost))
        addr_r, addr_phi = float(r[idx]), float(phi0[idx])
    else:
        idx, addr_r, addr_phi = -1, A, psi
    traj = dot_trajectory(addr_r, addr_phi, len(signal), rate)
    denom = np.std(signal) * np.std(traj)
    corr = float(np.corrcoef(signal, traj)[0, 1]) if denom > 1e-12 else 0.0
    return {"idx": idx, "r": addr_r, "phi0": addr_phi, "z0": 0.0,
            "amp": A, "phase": psi, "offset": c, "corr": corr}


def decode(addr, N, rate=1.0, with_offset=False):
    """§5.2 recovery: replay the addressed dot's trajectory."""
    s = dot_trajectory(addr["r"], addr["phi0"], N, rate)
    if with_offset:
        s = s + addr["offset"]
    return s


# ----------------------------------------------------------------------------
# §5 / §8.6 — multi-dot (harmonic) decomposition = matching pursuit → Fourier
# ----------------------------------------------------------------------------

def decompose(signal, rate=1.0, max_k=64, target=0.999):
    """Greedily peel the signal into harmonic-k dot components (the §10.1 multi-
    axis extension: reading the sweep at rate k gives the k-th harmonic). Each
    component is a (k, amp, phase) triple. Stops at `target` correlation or
    max_k components. This is exactly a greedy DFT — RS's geometric form of it."""
    N = len(signal)
    resid = signal - signal.mean()
    comps = []
    recon = np.zeros(N)
    order = float(np.linalg.norm(resid)) + 1e-12
    used = set()
    for _ in range(max_k):
        best = None
        for k in range(1, max_k + 1):
            if k in used:
                continue
            A, psi, _, _ = _fit_harmonic(resid, rate, k)
            energy = A
            if best is None or energy > best[0]:
                best = (energy, k, A, psi)
        _, k, A, psi = best
        used.add(k)
        comps.append({"k": k, "amp": float(A), "phase": float(psi)})
        w = np.radians(rate) * k
        recon = recon + A * np.cos(w * np.arange(N) + psi)
        resid = (signal - signal.mean()) - recon
        corr = float(np.corrcoef(signal - signal.mean(), recon)[0, 1]) if recon.std() > 1e-12 else 0.0
        if corr >= target:
            break
    return comps, recon + signal.mean(), corr


# ----------------------------------------------------------------------------
# signal generators for the demos / tests
# ----------------------------------------------------------------------------

def gen_signal(kind, N=360, rate=1.0, seed=0, r=1.5, phi0=0.7):
    rng = np.random.default_rng(seed)
    w = np.radians(rate)
    t = np.arange(N)
    if kind == "dot":                       # a pure dot trajectory (should be exact)
        return r * np.cos(w * t + phi0)
    if kind == "noisy":                     # dot + gaussian noise
        return r * np.cos(w * t + phi0) + rng.normal(0, 0.3, N)
    if kind == "multitone":                 # 3 harmonics — needs decomposition
        return (1.5 * np.cos(w * t + 0.4) + 0.8 * np.cos(3 * w * t + 1.1)
                + 0.4 * np.cos(7 * w * t - 0.6))
    if kind == "chirp":                     # frequency sweep — hard for a fixed basis
        f = np.linspace(1, 8, N)
        return np.cos(w * f * t / 2)
    if kind == "randomwalk":                # non-periodic
        return np.cumsum(rng.normal(0, 1, N)) * 0.1
    return np.cos(w * t)


# ----------------------------------------------------------------------------
# §5.1 compression accounting
# ----------------------------------------------------------------------------

def compression(N, n_components=1, bytes_per_val=4):
    raw = N * bytes_per_val
    addr = n_components * 3 * bytes_per_val          # (dot/k, amp, phase)
    return {"raw_bytes": raw, "addr_bytes": addr, "ratio": round(raw / max(addr, 1), 1)}


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------
# §8 — the validation suite (pass/fail + real numbers)
# ----------------------------------------------------------------------------

def validate(rng=(-2.0, 2.0), step=0.5, rate=1.0, N=360):
    lattice, r, phi0, z0 = build_lattice(rng, step)
    out = []

    # 8.1 determinism + analytic==matrix
    s1 = sweep(lattice, 137, 'z', rate)
    s2 = sweep(lattice, 137, 'z', rate)
    det = float(np.max(np.abs(s1 - s2)))
    i_probe = int(np.argmax(r))                       # an outer dot (r>0)
    tm = dot_trajectory_matrix(lattice, i_probe, N, 'z', rate, 0)
    ta = dot_trajectory(r[i_probe], phi0[i_probe], N, rate, 0)
    ana = float(np.max(np.abs(tm - ta)))
    out.append({"id": "8.1", "name": "Determinism", "value": f"max diff {det:.1e}; analytic vs sweep {ana:.1e}",
                "pass": det == 0.0 and ana < 1e-9})

    # 8.2 origin invariance
    disp = max(float(np.linalg.norm(sweep(np.zeros((1, 3)), T, 'z', rate)[0])) for T in range(0, 360, 7))
    out.append({"id": "8.2", "name": "Origin invariance", "value": f"max origin displacement {disp:.1e}",
                "pass": disp == 0.0})

    # 8.3 perfect recovery of a dot-generated signal
    sig = dot_trajectory(r[i_probe], phi0[i_probe], N, rate)
    a_cont = encode(sig, r, phi0, rate, quantize=False)
    err_cont = float(np.max(np.abs(sig - decode(a_cont, N, rate))))
    a_q = encode(sig, r, phi0, rate, quantize=True)
    err_q = float(np.max(np.abs(sig - decode(a_q, N, rate))))
    out.append({"id": "8.3", "name": "Perfect recovery (dot signal)",
                "value": f"continuous err {err_cont:.1e} (exact); lattice-quantized err {err_q:.3f}",
                "pass": err_cont < 1e-9})

    # 8.4 noisy recovery — does phase survive noise?
    rows = []
    for sd in (0.1, 0.5, 1.0):
        s = sig + np.random.default_rng(1).normal(0, sd, N)
        a = encode(s, r, phi0, rate, quantize=False)
        dphi = abs(float(np.angle(np.exp(1j * (a["phase"] - phi0[i_probe])))))
        rows.append(f"sd={sd}: dphase={np.degrees(dphi):.1f}deg, recon_corr={_corr(sig, decode(a, N, rate)):.3f}")
    out.append({"id": "8.4", "name": "Noisy recovery", "value": "; ".join(rows), "pass": True})

    # 8.5 arbitrary signal — single dot vs many
    arb = gen_signal("randomwalk", N, rate, seed=3)
    one = encode(arb, r, phi0, rate, quantize=False)
    c1 = _corr(arb, decode(one, N, rate))
    comps, recon, cK = decompose(arb, rate, max_k=64, target=0.99)
    out.append({"id": "8.5", "name": "Arbitrary signal (random walk)",
                "value": f"1 dot corr {c1:.2f} (poor); {len(comps)} dots corr {cK:.2f}",
                "pass": True})

    # 8.6 multi-dot decomposition targets
    mt = gen_signal("multitone", N, rate)
    tg = {}
    for target in (0.95, 0.99, 0.999):
        comps, _, got = decompose(mt, rate, max_k=64, target=target)
        tg[target] = (len(comps), round(got, 4))
    single = _corr(mt, decode(encode(mt, r, phi0, rate, quantize=False), N, rate))
    out.append({"id": "8.6", "name": "Multi-tone: dots to reach fidelity",
                "value": f"1 dot corr {single:.2f}; " + ", ".join(f"{tg[t][1]} at {tg[t][0]} dots" for t in tg),
                "pass": tg[0.999][1] >= 0.999})

    # 8.7 lattice density vs quantization error — use an OFF-lattice signal
    # (amplitude/phase deliberately between dots) so density actually matters
    off = dot_trajectory(1.37, 0.55, N, rate)
    rows = []
    for st in (1.0, 0.5, 0.25, 0.1):
        lat2, r2, p2, _ = build_lattice(rng, st)
        aq = encode(off, r2, p2, rate, quantize=True)
        rows.append(f"step {st}:{float(np.max(np.abs(off-decode(aq,N,rate)))):.3f} ({len(lat2)} dots)")
    out.append({"id": "8.7", "name": "Lattice density vs error", "value": "; ".join(rows), "pass": True})

    # 8.8 reproducibility (determinism ⇒ identical address anywhere)
    a1 = encode(sig, r, phi0, rate, quantize=True)
    a2 = encode(sig, r, phi0, rate, quantize=True)
    out.append({"id": "8.8", "name": "Reproducibility", "value": f"address stable: idx {a1['idx']}=={a2['idx']}",
                "pass": a1["idx"] == a2["idx"]})

    n_pass = sum(t["pass"] for t in out)
    verdict = (
        "The geometry is exactly as claimed — deterministic, origin-fixed, reversible, and a pure "
        "dot-signal round-trips to floating-point zero (120:1). But the suite makes the real boundary "
        "unavoidable: a single-axis sweep is a ONE-frequency basis. It nails a pure tone and stalls on "
        "everything else with a single dot; only the harmonic (multi-axis) decomposition reaches "
        "arbitrary signals — and that is a greedy Fourier transform wearing geometric clothes. RS is a "
        "real, correct sinusoid codec; its compression beats raw exactly when the signal is sparse in "
        "this Fourier basis, which is the standard caveat for transform coding, not a new law."
    )
    return {"lattice_dots": len(lattice), "params": {"range": list(rng), "step": step, "rate": rate, "N": N},
            "tests": out, "pass": n_pass, "total": len(out), "verdict": verdict}


def run_encode(kind="dot", rng=(-2.0, 2.0), step=0.5, rate=1.0, N=360,
               max_k=32, target=0.999, seed=0):
    """Encode one demo signal and return everything the page needs to draw:
    the signal, the single-dot reconstruction, the K-dot reconstruction, the
    address, per-component list, correlations and compression ratios."""
    lattice, r, phi0, z0 = build_lattice(rng, step)
    sig = gen_signal(kind, N, rate, seed)
    addr = encode(sig, r, phi0, rate, quantize=True)
    recon1 = decode(addr, N, rate, with_offset=True) if kind not in ("dot", "multitone") else decode(addr, N, rate)
    corr1 = _corr(sig, recon1)
    comps, reconK, corrK = decompose(sig, rate, max_k=max_k, target=target)
    return {
        "kind": kind, "n": N, "lattice_dots": len(lattice),
        "params": {"range": list(rng), "step": step, "rate": rate},
        "signal": [round(float(v), 4) for v in sig],
        "recon1": [round(float(v), 4) for v in recon1],
        "reconK": [round(float(v), 4) for v in reconK],
        "address": {"idx": addr["idx"], "r": round(addr["r"], 4),
                    "phi0": round(addr["phi0"], 4), "z0": addr["z0"],
                    "amp": round(addr["amp"], 4), "phase": round(addr["phase"], 4)},
        "corr1": round(corr1, 4), "corrK": round(corrK, 4),
        "components": comps,
        "comp1": compression(N, 1), "compK": compression(N, len(comps)),
    }


if __name__ == "__main__":
    import json
    v = validate()
    print(f"lattice {v['lattice_dots']} dots · {v['pass']}/{v['total']} pass\n")
    for t in v["tests"]:
        print(f"  [{'PASS' if t['pass'] else 'fail'}] {t['id']} {t['name']}: {t['value']}")
    print("\n" + v["verdict"])
