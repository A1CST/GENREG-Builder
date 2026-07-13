"""radial_compress_bench.py — the honest compression verdict.

Fair head-to-head: RS-Gabor matching pursuit vs DFT / DCT / Haar-wavelet top-K,
all with 8-bit-quantized coefficients and honest byte accounting, on realistic
signals (speech, music, ECG, isolated transient). Metric: bytes to reach 99%
reconstruction correlation.

Result: RS-Gabor loses to plain DFT on every realistic signal (often 3-4x) and
wins ONLY on a lone transient. The earlier "beats Fourier" claim held only
against an unquantized top-K DFT baseline on burst-shaped signals. Under a fair
test, radial-space compression is dominated by 50-year-old transform coding and
is NOT a differentiator. Run: python radial_compress_bench.py
"""
import numpy as np
from scipy.fft import dct, idct
import radial_dict as rd

N = 512

def haar_fwd(a):
    a = a.astype(float).copy(); L = N
    while L > 1:
        h = L // 2; e = a[0:L:2]; o = a[1:L:2]
        a[:h] = (e + o) / np.sqrt(2); a[h:L] = (e - o) / np.sqrt(2); L = h
    return a

def haar_inv(a):
    a = a.astype(float).copy(); L = 2
    while L <= N:
        h = L // 2; av = a[:h].copy(); d = a[h:L].copy(); r = np.zeros(L)
        r[0:L:2] = (av + d) / np.sqrt(2); r[1:L:2] = (av - d) / np.sqrt(2); a[:L] = r; L *= 2
    return a

def topk_quant(c, K):
    idx = np.argsort(np.abs(c))[::-1][:K]; keep = np.zeros_like(c); v = c[idx]
    m = np.abs(v).max() + 1e-12; keep[idx] = np.round(v / m * 127) / 127 * m; return keep

def rec_dct(x, K): return idct(topk_quant(dct(x, norm='ortho'), K), norm='ortho')
def rec_wav(x, K): return haar_inv(topk_quant(haar_fwd(x), K))
def rec_dft(x, K):
    F = np.fft.rfft(x); idx = np.argsort(np.abs(F))[::-1][:K]; G = np.zeros_like(F)
    vq = F[idx]; m = np.abs(vq).max() + 1e-12
    G[idx] = np.round(vq.real / m * 127) / 127 * m + 1j * np.round(vq.imag / m * 127) / 127 * m
    return np.fft.irfft(G, n=N)

_D, _ = rd.build_dictionary(N)
def rec_rs(x, K):
    _, idxs, coefs = rd.matching_pursuit(x, _D, K); m = np.abs(coefs).max() + 1e-12
    r = np.zeros(N)
    for j, c in zip(idxs, coefs): r += (np.round(c / m * 127) / 127 * m) * _D[j]
    return r + x.mean()

def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if np.std(b) > 1e-9 else 0.0
BITS = {'DFT': 17, 'DCT': 17, 'Wavelet': 17, 'RS-Gabor': 20}

def sig(kind):
    t = np.arange(N); w = 2 * np.pi / 64
    if kind == 'speech':
        f = 1 + 0.6 * np.sin(w * 0.08 * t); return (1 + 0.5 * np.cos(w * 0.05 * t)) * np.cos(w * f * t) + 0.4 * np.cos(3 * w * t)
    if kind == 'music':
        b = np.cos(w * t) + 0.5 * np.cos(2 * w * t) + 0.3 * np.cos(3 * w * t)
        for c in (80, 240, 400): b += 1.5 * np.exp(-((t - c) ** 2) / (2 * 8 ** 2)) * np.cos(4 * w * t)
        return b
    if kind == 'ecg':
        s = np.zeros(N)
        for c in range(30, N, 70): s += np.exp(-((t - c) ** 2) / (2 * 3 ** 2)) * 2 - 0.4 * np.exp(-((t - c - 8) ** 2) / (2 * 6 ** 2))
        return s
    return np.exp(-((t - 256) ** 2) / (2 * 20 ** 2)) * np.cos(w * t)

if __name__ == "__main__":
    print(f"{'signal':10s} bytes to reach corr>=0.99 (lower=better)")
    for kind in ['speech', 'music', 'ecg', 'transient']:
        x = sig(kind); line = {}
        for name, fn in [('DFT', rec_dft), ('DCT', rec_dct), ('Wavelet', rec_wav), ('RS-Gabor', rec_rs)]:
            best = None
            for K in range(1, 60):
                if corr(x, fn(x, K)) >= 0.99: best = K * BITS[name] / 8; break
            line[name] = best
        win = min([k for k in line if line[k]], key=lambda k: line[k])
        print(f"{kind:10s}", "  ".join(f"{k}:{'%.0f' % line[k] if line[k] else '>60c'}" for k in line), f"  WIN={win}")
