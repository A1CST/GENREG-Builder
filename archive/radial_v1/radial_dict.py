"""radial_dict.py — Radial Space, past the uniform Fourier basis.

The base system's dots are eternal circles -> pure cosines -> a Fourier basis.
The radial coordinate did nothing (radius was just amplitude). Here we give the
radius a temporal job: let it rise and fall over the sweep so a dot traces a
WINDOWED oscillation instead of an endless circle. Geometrically that is a dot
spiralling in and back out; as a signal it is a Gabor atom

    g(t) = exp(-((t-tau)^2)/(2 s^2)) * cos(omega (t-tau) + phi)

optionally with a chirp (the rotation rate drifting during the sweep):

    cos( omega (t-tau) + 0.5 chirp (t-tau)^2 + phi )

Now the dictionary is overcomplete and time-localized. Matching pursuit over it
should beat top-K Fourier precisely on the signals Fourier is bad at: transients,
bursts, chirps, anything non-stationary. This file PROVES OR DISPROVES that
before any UI gets built.
"""
import numpy as np


def _atom(N, tau, omega, phi, s, chirp=0.0):
    t = np.arange(N) - tau
    env = np.exp(-(t ** 2) / (2 * s ** 2))
    g = env * np.cos(omega * t + 0.5 * chirp * t ** 2 + phi)
    n = np.linalg.norm(g)
    return g / n if n > 1e-9 else g


def build_dictionary(N, n_freq=28, n_tau=14, scales=None, chirps=(0.0,), phases=(0.0, np.pi / 2)):
    """Overcomplete Gabor(+chirp) dictionary indexed by geometry: frequency
    (rotation rate), time-centre tau (where in the sweep), scale s (radial
    envelope width), chirp (rate drift). Includes a near-global scale so the
    dictionary CONTAINS the Fourier atoms — MP can fall back to Fourier when
    that is genuinely best, making the comparison fair."""
    if scales is None:
        scales = [N / 2.5, N / 6, N / 14, N / 30]     # global -> narrow
    freqs = np.pi * np.linspace(0.01, 1.0, n_freq)     # up to Nyquist-ish
    taus = np.linspace(0, N - 1, n_tau)
    atoms, meta = [], []
    for s in scales:
        for tau in taus:
            for om in freqs:
                for ch in chirps:
                    for ph in phases:
                        atoms.append(_atom(N, tau, om, ph, s, ch))
                        meta.append((tau, om, ph, s, ch))
    return np.array(atoms), meta


def matching_pursuit(signal, D, K):
    """Greedy: repeatedly add the atom most correlated with the residual."""
    resid = signal - signal.mean()
    coefs, idxs = [], []
    recon = np.zeros_like(signal, dtype=float)
    for _ in range(K):
        proj = D @ resid
        j = int(np.argmax(np.abs(proj)))
        c = float(proj[j])
        coefs.append(c); idxs.append(j)
        recon = recon + c * D[j]
        resid = (signal - signal.mean()) - recon
    return recon + signal.mean(), idxs, coefs


def fourier_topk(signal, K):
    """Top-K complex DFT coefficients (the honest Fourier competitor)."""
    F = np.fft.rfft(signal - signal.mean())
    keep = np.argsort(np.abs(F))[::-1][:K]
    G = np.zeros_like(F)
    G[keep] = F[keep]
    return np.fft.irfft(G, n=len(signal)) + signal.mean()


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def gen(kind, N=360, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    w = 2 * np.pi / 90                       # base tone period 90
    if kind == "tone":
        return np.cos(w * t + 0.5)
    if kind == "burst":                      # a tone that exists only briefly
        env = np.exp(-((t - N * 0.5) ** 2) / (2 * (N * 0.05) ** 2))
        return env * np.cos(w * t)
    if kind == "two_bursts":                 # two different tones at two times
        e1 = np.exp(-((t - N * 0.3) ** 2) / (2 * (N * 0.04) ** 2))
        e2 = np.exp(-((t - N * 0.7) ** 2) / (2 * (N * 0.04) ** 2))
        return e1 * np.cos(2 * w * t) + e2 * np.cos(0.5 * w * t)
    if kind == "chirp":                      # frequency sweeps up over time
        return np.cos((w * 0.3 + w * 1.5 * t / N) * t)
    if kind == "am":                         # amplitude-modulated tone
        return (1 + 0.8 * np.cos(w * 0.1 * t)) * np.cos(w * t)
    if kind == "walk":
        return np.cumsum(rng.normal(0, 1, N)) * 0.1
    return np.cos(w * t)


def compare(kind, N=360, Ks=(1, 2, 3, 5, 8, 12, 20), D=None, meta=None):
    sig = gen(kind, N)
    if D is None:
        D, meta = build_dictionary(N)
    rows = []
    for K in Ks:
        rec_rs, _, _ = matching_pursuit(sig, D, K)
        rec_ft = fourier_topk(sig, K)
        rows.append({"K": K, "rs": round(_corr(sig, rec_rs), 4), "ft": round(_corr(sig, rec_ft), 4)})
    return sig, rows


_DICT = {}          # cache dictionaries by N


def _get_dict(N):
    if N not in _DICT:
        _DICT[N] = build_dictionary(N)
    return _DICT[N]


def run_compare(kind="burst", N=360, K=3, seed=0):
    """RS-Gabor matching pursuit vs top-K Fourier at K atoms, plus the full
    corr-vs-K curve for both — everything the page needs to draw the head-to-head."""
    N = max(64, min(1024, int(N)))
    K = max(1, min(40, int(K)))
    D, meta = _get_dict(N)
    sig = gen(kind, N, seed)
    rec_rs, idxs, coefs = matching_pursuit(sig, D, K)
    rec_ft = fourier_topk(sig, K)
    Ks = [k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 28, 40) if k <= max(40, K)]
    curve = []
    for k in Ks:
        rr, _, _ = matching_pursuit(sig, D, k)
        curve.append({"K": k, "rs": round(_corr(sig, rr), 4), "ft": round(_corr(sig, fourier_topk(sig, k)), 4)})
    rs_c, ft_c = _corr(sig, rec_rs), _corr(sig, rec_ft)

    # --- the building blocks each method uses (for the "why" picture) ---
    def _ds(a, m=140):                       # downsample for transport
        step = max(1, len(a) // m)
        return [round(float(v), 4) for v in a[::step]]
    rs_pieces = [_ds(c * D[j]) for j, c in list(zip(idxs, coefs))[:3]]
    # individual Fourier components: each kept bin reconstructed alone
    Ff = np.fft.rfft(sig - sig.mean())
    keep = np.argsort(np.abs(Ff))[::-1][:K]
    ft_pieces = []
    for b in keep[:3]:
        G = np.zeros_like(Ff); G[b] = Ff[b]
        ft_pieces.append(_ds(np.fft.irfft(G, n=N)))

    # --- plain scoreboard: pieces needed to reach 95% match ---
    def _pieces_to(target, method):
        for k in range(1, 41):
            if method == "rs":
                rr, _, _ = matching_pursuit(sig, D, k)
                if _corr(sig, rr) >= target:
                    return k
            else:
                if _corr(sig, fourier_topk(sig, k)) >= target:
                    return k
        return 41
    rs95, ft95 = _pieces_to(0.95, "rs"), _pieces_to(0.95, "ft")

    return {
        "kind": kind, "n": N, "K": K, "dict_atoms": len(D),
        "signal": [round(float(v), 4) for v in sig],
        "rs_recon": [round(float(v), 4) for v in rec_rs],
        "ft_recon": [round(float(v), 4) for v in rec_ft],
        "curve": curve, "rs_pieces": rs_pieces, "ft_pieces": ft_pieces,
        "rs95": rs95, "ft95": ft95,
        "rs_corr": round(rs_c, 4), "ft_corr": round(ft_c, 4),
        "winner": "RS-Gabor" if rs_c > ft_c + 1e-4 else ("Fourier" if ft_c > rs_c + 1e-4 else "tie"),
    }


if __name__ == "__main__":
    N = 360
    D, meta = build_dictionary(N)
    print(f"dictionary: {len(D)} atoms (overcomplete; N={N})\n")
    print(f"{'signal':12s}  {'K':>3s}  {'RS-Gabor':>9s}  {'FFT-topK':>9s}  winner")
    for kind in ["tone", "burst", "two_bursts", "chirp", "am", "walk"]:
        _, rows = compare(kind, N, D=D, meta=meta)
        for r in rows:
            win = "RS" if r["rs"] > r["ft"] + 1e-4 else ("FFT" if r["ft"] > r["rs"] + 1e-4 else "tie")
            mark = "  <--" if win == "RS" else ""
            print(f"{kind:12s}  {r['K']:3d}  {r['rs']:9.4f}  {r['ft']:9.4f}  {win}{mark}")
        print()
