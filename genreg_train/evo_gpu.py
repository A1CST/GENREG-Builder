"""GPU backend for the image-pipe genome batteries (MNIST-Pipe / CIFAR-Pipe).

Pure arithmetic acceleration — the evolution (ga_step, selection, mutation,
champion gating) stays in numpy on the CPU, byte-identical to the CPU path.
Only the FITNESS EVALUATIONS (dense GEMMs + elementwise activations + block
pooling + Fisher statistics) run on the GPU, because that is >95% of the
wall-clock. No gradients are ever computed: torch is used under no_grad as a
matrix calculator. TF32 is disabled so GPU fitness matches CPU fitness to
float32 rounding.

Everything degrades gracefully: HAS_GPU is False without torch/CUDA and the
pipelines fall back to their numpy paths.
"""
import numpy as np

try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
except ImportError:                                # pragma: no cover
    torch = None
    HAS_GPU = False

DEV = "cuda" if HAS_GPU else "cpu"


def to_dev(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


def _acts_t(x, a):
    """Torch mirror of mnist_pipe._acts (same 8-function catalog)."""
    if a == 0:
        return torch.relu(x)
    if a == 1:
        return torch.abs(x)
    if a == 2:
        return torch.sin(x)
    if a == 3:
        return torch.cos(x)
    if a == 4:
        return torch.exp(-x * x)
    if a == 5:
        return torch.where(x > 0, x, 0.1 * x)
    if a == 6:
        return x * x
    return torch.tanh(x)


def _pool_t(resp, R, pools):
    """(B,R,R) -> (B, sum r*c) crop-block mean pools (matches _pool_resp)."""
    B = resp.shape[0]
    parts = []
    for r, c in pools:
        hr, hc = R // r, R // c
        parts.append(resp[:, :r * hr, :c * hc].reshape(B, r, hr, c, hc)
                     .mean(dim=(2, 4)).reshape(B, r * c))
    return torch.cat(parts, dim=1)


# --------------------------------------------------------------------------
# Joint-head fitness (mean log-softmax of true class - L2), numpy in/out
# --------------------------------------------------------------------------
class JointFitGPU:
    def __init__(self, F, y):
        self.Fg = to_dev(F.astype(np.float32))
        self.yg = to_dev(y.astype(np.int64)).view(1, -1, 1)
        self.N, self.nf = F.shape

    @torch.no_grad()
    def __call__(self, W, b, l2=0.0):
        P = len(W)
        Wg = to_dev(W)                             # (P,nf,10)
        bg = to_dev(b)                             # (P,10)
        z = (self.Fg @ Wg.permute(1, 0, 2).reshape(self.nf, P * 10)) \
            .reshape(self.N, P, 10).permute(1, 0, 2) + bg[:, None, :]
        logp = torch.log_softmax(z, dim=-1)
        ch = logp.gather(2, self.yg.expand(P, self.N, 1))[..., 0]
        fit = ch.mean(dim=1)
        if l2 > 0:
            fit = fit - l2 * (Wg * Wg).sum(dim=(1, 2))
        return fit.cpu().numpy()

    @torch.no_grad()
    def acc1(self, W, b):
        """Top-1 accuracy of a single head (nf,10),(10,) on this pool."""
        Wg = to_dev(W); bg = to_dev(b)
        pred = (self.Fg @ Wg + bg).argmax(dim=1)
        return float((pred == self.yg.view(-1)).float().mean().item())


# --------------------------------------------------------------------------
# Binary (detector / pairwise) fitness: pools uploaded once, minibatch
# indices sampled on CPU per generation
# --------------------------------------------------------------------------
class BinaryFitGPU:
    def __init__(self, Fp, Fn):
        self.Fp = to_dev(Fp.astype(np.float32))
        self.Fn = to_dev(Fn.astype(np.float32))

    @torch.no_grad()
    def __call__(self, w, b, ip, inn, l2=0.0):
        """w (P,nf), b (P,), ip/inn int index arrays into the pools ->
        (fit (P,), acc (P,)) numpy — same math as LinearPop.fitness."""
        wg = to_dev(w); bg = to_dev(b)
        Fp = self.Fp[to_dev(ip.astype(np.int64))]
        Fn = self.Fn[to_dev(inn.astype(np.int64))]
        zp = (wg @ Fp.T + bg[:, None]).clamp(-30, 30)
        zn = (wg @ Fn.T + bg[:, None]).clamp(-30, 30)
        lp = -torch.log1p(torch.exp(-zp)).mean(dim=1)
        ln = -torch.log1p(torch.exp(zn)).mean(dim=1)
        acc = ((zp > 0).float().mean(dim=1) + (zn < 0).float().mean(dim=1)) / 2
        fit = lp + ln
        if l2 > 0:
            fit = fit - l2 * (wg * wg).sum(dim=1)
        return fit.cpu().numpy(), acc.cpu().numpy()


# --------------------------------------------------------------------------
# Detector-bank fitness: conv responses + activations + pools + Fisher,
# whole population per call
# --------------------------------------------------------------------------
class DetbankFitGPU:
    def __init__(self, patches, y, R, pools, nc=10):
        """patches (N, R*R, KD) im2col'd images, y (N,) labels."""
        self.N, self.npos, self.kd = patches.shape
        self.R, self.pools, self.nc = R, pools, nc
        self.Pf = to_dev(patches.reshape(-1, self.kd).astype(np.float32))
        self.masks = [to_dev((y == c).astype(np.float32)) for c in range(nc)]
        self.counts = [float(m.sum().item()) for m in self.masks]

    @torch.no_grad()
    def __call__(self, K, b, act):
        """K (P,KD), b (P,), act (P,) ints -> Fisher fitness (P,) numpy."""
        P = len(K)
        Kg = to_dev(K); bg = to_dev(b)
        resp = (self.Pf @ Kg.T + bg).reshape(self.N, self.R, self.R, P)
        D = sum(r * c for r, c in self.pools)
        pooled = torch.empty((P, self.N, D), device=DEV)
        for a in range(8):
            ids = np.where(act == a)[0]
            if len(ids) == 0:
                continue
            idt = to_dev(ids.astype(np.int64))
            blk = _acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * self.N, self.R, self.R)
            pooled[idt] = _pool_t(blk, self.R, self.pools).reshape(len(ids), self.N, D)
        mu = pooled.mean(dim=1)                    # (P,D)
        bt = torch.zeros((P, D), device=DEV)
        wi = torch.zeros((P, D), device=DEV)
        for c in range(self.nc):
            m = self.masks[c]
            n_c = self.counts[c]
            mc = (pooled * m[None, :, None]).sum(dim=1) / n_c
            bt += n_c * (mc - mu) ** 2
            d = (pooled - mc[:, None, :]) * m[None, :, None]
            wi += (d * d).sum(dim=1)
        return (bt / (wi + 1e-8)).mean(dim=1).cpu().numpy()


# --------------------------------------------------------------------------
# Diversity-driven bank fitness (label-free): behavior = z-normalised pooled
# response pattern over an unlabeled probe set; fitness = novelty x info.
# --------------------------------------------------------------------------
class DiversityFitGPU:
    def __init__(self, patches, R, pools, k=5, patches_jit=None):
        """`patches_jit`: im2col of the SAME probe images under a small jitter
        (e.g. 1px shift). When given, fitness gains a STABILITY term: response
        must be consistent for the same image under jitter while varying
        across images — an unsupervised signal-to-noise ratio that kills
        noise-measuring genomes which entropy alone admits."""
        self.N, self.npos, self.kd = patches.shape
        self.R, self.pools, self.k = R, pools, k
        self.PD = sum(r * c for r, c in pools)
        self.Pf = to_dev(patches.reshape(-1, self.kd).astype(np.float32))
        self.Pj = to_dev(patches_jit.reshape(-1, self.kd).astype(np.float32)) \
            if patches_jit is not None else None
        self.archive = None                        # (m, N*PD) unit rows on DEV

    def _behaviors(self, K, b, act, Pf=None):
        P = len(K)
        Kg = to_dev(K); bg = to_dev(b)
        resp = ((Pf if Pf is not None else self.Pf) @ Kg.T + bg) \
            .reshape(self.N, self.R, self.R, P)
        pooled = torch.empty((P, self.N, self.PD), device=DEV)
        for a in range(8):
            ids = np.where(act == a)[0]
            if len(ids) == 0:
                continue
            idt = to_dev(ids.astype(np.int64))
            blk = _acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * self.N, self.R, self.R)
            pooled[idt] = _pool_t(blk, self.R, self.pools).reshape(len(ids), self.N, self.PD)
        B = pooled.reshape(P, -1)
        B = B - B.mean(dim=1, keepdim=True)
        B = B / (B.norm(dim=1, keepdim=True) + 1e-8)   # unit rows: cos = dot
        return B, pooled

    @torch.no_grad()
    def __call__(self, K, b, act):
        """-> (fitness (P,), behaviors (P, N*PD) numpy unit rows).
        novelty = 1 - mean |cos| to the k nearest behaviors among peers+archive;
        info    = normalised entropy of the per-image mean-response histogram
                  (a constant, saturated, or duplicate-heavy output scores 0)."""
        P = len(K)
        B, pooled = self._behaviors(K, b, act)
        sims = [B @ B.T - 2.0 * torch.eye(P, device=DEV)]   # exclude self
        if self.archive is not None and len(self.archive):
            sims.append(B @ self.archive.T)
        S = torch.cat(sims, dim=1).abs()
        k = min(self.k, S.shape[1])
        novelty = 1.0 - S.topk(k, dim=1).values.mean(dim=1)
        # info: entropy of 16-bin histogram of per-image mean response
        m = pooled.mean(dim=2)                              # (P,N)
        lo = m.min(dim=1, keepdim=True).values
        hi = m.max(dim=1, keepdim=True).values
        idx = ((m - lo) / (hi - lo + 1e-8) * 15.999).long() # (P,N) in 0..15
        hist = torch.zeros((P, 16), device=DEV)
        hist.scatter_add_(1, idx, torch.ones_like(m))
        p = hist / self.N
        ent = -(p * torch.log(p + 1e-12)).sum(dim=1) / np.log(16.0)
        fit = novelty * ent
        if self.Pj is not None:                             # stability (SNR) term
            _, pooled_j = self._behaviors(K, b, act, Pf=self.Pj)
            mj = pooled_j.mean(dim=2)
            sig = ((m - m.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
            noi = ((m - mj) ** 2).mean(dim=1) + 1e-12
            snr = sig / noi
            fit = fit * (snr / (1.0 + snr))
        return fit.cpu().numpy(), B.cpu().numpy()

    def admit(self, behaviors):
        """Append accepted behaviors (numpy unit rows) to the archive."""
        Bg = to_dev(behaviors.astype(np.float32))
        self.archive = Bg if self.archive is None \
            else torch.cat([self.archive, Bg], dim=0)

    def novelty_vs_archive(self, behaviors):
        """Max |cos| of each row against the archive (1.0 means duplicate)."""
        if self.archive is None or not len(self.archive):
            return np.zeros(len(behaviors), np.float32)
        Bg = to_dev(behaviors.astype(np.float32))
        return (Bg @ self.archive.T).abs().max(dim=1).values.cpu().numpy()


# --------------------------------------------------------------------------
# Utility-driven encoder fitness: a candidate genome's fitness is the CHANGE
# in classification fitness when its pooled output dims are appended to the
# encoder built so far. Incremental nearest-centroid state makes each
# candidate's marginal utility O(N*PD) — the whole population per GEMM.
# --------------------------------------------------------------------------
class UtilityFitGPU:
    def __init__(self, patches, y, R, pools, nc=10):
        self.N, self.npos, self.kd = patches.shape
        self.R, self.pools, self.nc = R, pools, nc
        self.PD = sum(r * c for r, c in pools)
        self.Pf = to_dev(patches.reshape(-1, self.kd).astype(np.float32))
        self.yg = to_dev(y.astype(np.int64))
        # class one-hot / counts for centroid computation
        M = np.zeros((len(y), nc), np.float32)
        M[np.arange(len(y)), y] = 1.0
        self.Mg = to_dev(M / M.sum(0, keepdims=True))     # (N,nc) mean weights
        self.d2 = torch.zeros((self.N, nc), device=DEV)   # accumulated distances
        self.base_acc = 0.1

    def _pooled(self, K, b, act):
        P = len(K)
        Kg = to_dev(K); bg = to_dev(b)
        resp = (self.Pf @ Kg.T + bg).reshape(self.N, self.R, self.R, P)
        pooled = torch.empty((P, self.N, self.PD), device=DEV)
        for a in range(8):
            ids = np.where(act == a)[0]
            if len(ids) == 0:
                continue
            idt = to_dev(ids.astype(np.int64))
            blk = _acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * self.N, self.R, self.R)
            pooled[idt] = _pool_t(blk, self.R, self.pools).reshape(len(ids), self.N, self.PD)
        # per-dim standardisation so no genome wins by raw output scale
        mu = pooled.mean(dim=1, keepdim=True)
        sd = pooled.std(dim=1, keepdim=True) + 1e-6
        return (pooled - mu) / sd

    def _d2_cand(self, pooled):
        """(P,N,PD) standardised features -> (P,N,nc) squared distances to the
        candidate's own class centroids."""
        cent = torch.einsum("pnd,nc->pcd", pooled, self.Mg)     # (P,nc,PD)
        f2 = (pooled * pooled).sum(dim=2, keepdim=True)         # (P,N,1)
        c2 = (cent * cent).sum(dim=2)                           # (P,nc)
        fc = torch.einsum("pnd,pcd->pnc", pooled, cent)
        return f2 + c2[:, None, :] - 2.0 * fc

    def _margin(self, tot):
        """Dense fitness of a distance state: mean (min-other - true) margin,
        larger is better. No accuracy plateaus."""
        true_d = tot.gather(-1, self.yg[None, :, None].expand(tot.shape[0], -1, 1))[..., 0]
        masked = tot.scatter(-1, self.yg[None, :, None].expand(tot.shape[0], -1, 1),
                             torch.inf)
        other = masked.min(dim=-1).values
        return (other - true_d).mean(dim=1)

    @torch.no_grad()
    def __call__(self, K, b, act, dense=False):
        """fitness = marginal gain of the candidate over the encoder-so-far:
        accuracy gain (default) or dense margin gain (`dense=True`)."""
        pooled = self._pooled(K, b, act)
        d2c = self._d2_cand(pooled)
        tot = self.d2[None] + d2c
        if dense:
            base = self._margin(self.d2[None])[0]
            return (self._margin(tot) - base).cpu().numpy(), pooled
        acc = (tot.argmin(dim=2) == self.yg[None]).float().mean(dim=1)
        return (acc - self.base_acc).cpu().numpy(), pooled

    @torch.no_grad()
    def admit(self, pooled_row):
        """Fold the champion's distances into the encoder state."""
        d2c = self._d2_cand(pooled_row)
        self.d2 = self.d2 + d2c[0]
        self.base_acc = float((self.d2.argmin(dim=1) == self.yg).float()
                              .mean().item())
        return self.base_acc


# --------------------------------------------------------------------------
# Layer-2 diversity genomes: read a frozen layer-1 bank's response maps.
# Genome = 8 channel-selection genes + 3x3x8 kernel + activation.
# --------------------------------------------------------------------------
@torch.no_grad()
def l1_maps(patches_fn, X, bank, R, pools_unused=None, chunk=None, out_hw=14):
    """Frozen layer-1 maps for images X -> (N, nb, out_hw, out_hw) float32
    numpy (2x2-avg-pooled from R x R). Chunked for memory."""
    nb = len(bank["K"])
    if chunk is None:
        chunk = max(128, min(2048, int(2e9 / (R * R * nb * 4))))
    Kg = to_dev(bank["K"]); bg = to_dev(bank["b"])
    out = np.empty((len(X), nb, out_hw, out_hw), np.float32)
    for lo in range(0, len(X), chunk):
        Xc = X[lo:lo + chunk]
        Pf = to_dev(patches_fn(Xc).reshape(-1, bank["K"].shape[1]).astype(np.float32))
        resp = (Pf @ Kg.T + bg).reshape(len(Xc), R, R, nb)
        act_maps = torch.empty_like(resp)
        for a in range(8):
            ids = np.where(bank["act"] == a)[0]
            if len(ids):
                idt = to_dev(ids.astype(np.int64))
                act_maps.index_copy_(3, idt, _acts_t(resp.index_select(3, idt), a))
        m = act_maps.permute(0, 3, 1, 2)                 # (n, nb, R, R)
        m = torch.nn.functional.adaptive_avg_pool2d(m, out_hw)
        out[lo:lo + len(Xc)] = m.cpu().numpy()
        del resp, act_maps, m, Pf
        torch.cuda.empty_cache()
    return out


class Layer2FitGPU:
    """Diversity fitness for layer-2 genomes over precomputed L1 maps.
    L1 (N, nb, H, H) numpy; genome: ch (P,8) int, K (P,72), b (P,), act (P,)."""

    def __init__(self, L1, pools, k=5, L1_jit=None, chunk=500):
        self.N, self.nb, self.H, _ = L1.shape
        self.R2 = self.H - 2
        self.pools = pools
        self.PD = sum(r * c for r, c in pools)
        self.k, self.chunk = k, chunk
        self.L1 = to_dev(L1)
        self.L1j = to_dev(L1_jit) if L1_jit is not None else None
        self.archive = None

    def _pooled(self, ch, K, b, act, L1):
        P = len(K)
        Kg = to_dev(K); bg = to_dev(b)
        cht = to_dev(ch.astype(np.int64))
        pooled = torch.empty((P, self.N, self.PD), device=DEV)
        for lo in range(0, self.N, self.chunk):
            sub = L1[lo:lo + self.chunk]                 # (n, nb, H, H)
            n = len(sub)
            sel = sub[:, cht, :, :]                      # (n, P, 8, H, H)
            u = torch.nn.functional.unfold(
                sel.reshape(n * P, 8, self.H, self.H), 3)   # (nP, 72, R2*R2)
            z = torch.einsum("pnfk,pk->pnf",
                             u.reshape(n, P, 72, -1).permute(1, 0, 3, 2),
                             Kg) + bg[:, None, None]
            z = z.reshape(P, n, self.R2, self.R2)
            for a in range(8):
                ids = np.where(act == a)[0]
                if len(ids):
                    idt = to_dev(ids.astype(np.int64))
                    blk = _acts_t(z.index_select(0, idt), a) \
                        .reshape(len(ids) * n, self.R2, self.R2)
                    pooled[idt, lo:lo + n] = _pool_t(blk, self.R2, self.pools) \
                        .reshape(len(ids), n, self.PD)
            del sel, u, z
        return pooled

    @torch.no_grad()
    def __call__(self, ch, K, b, act):
        P = len(K)
        pooled = self._pooled(ch, K, b, act, self.L1)
        B = pooled.reshape(P, -1)
        B = B - B.mean(dim=1, keepdim=True)
        B = B / (B.norm(dim=1, keepdim=True) + 1e-8)
        sims = [B @ B.T - 2.0 * torch.eye(P, device=DEV)]
        if self.archive is not None and len(self.archive):
            sims.append(B @ self.archive.T)
        S = torch.cat(sims, dim=1).abs()
        novelty = 1.0 - S.topk(min(self.k, S.shape[1]), dim=1).values.mean(dim=1)
        m = pooled.mean(dim=2)
        lo = m.min(dim=1, keepdim=True).values
        hi = m.max(dim=1, keepdim=True).values
        idx = ((m - lo) / (hi - lo + 1e-8) * 15.999).long()
        hist = torch.zeros((P, 16), device=DEV)
        hist.scatter_add_(1, idx, torch.ones_like(m))
        p = hist / self.N
        ent = -(p * torch.log(p + 1e-12)).sum(dim=1) / np.log(16.0)
        fit = novelty * ent
        if self.L1j is not None:
            mj = self._pooled(ch, K, b, act, self.L1j).mean(dim=2)
            sig = ((m - m.mean(dim=1, keepdim=True)) ** 2).mean(dim=1)
            noi = ((m - mj) ** 2).mean(dim=1) + 1e-12
            snr = sig / noi
            fit = fit * (snr / (1.0 + snr))
        return fit.cpu().numpy(), B.cpu().numpy()

    def admit(self, behaviors):
        Bg = to_dev(behaviors.astype(np.float32))
        self.archive = Bg if self.archive is None \
            else torch.cat([self.archive, Bg], dim=0)

    def novelty_vs_archive(self, behaviors):
        if self.archive is None or not len(self.archive):
            return np.zeros(len(behaviors), np.float32)
        Bg = to_dev(behaviors.astype(np.float32))
        return (Bg @ self.archive.T).abs().max(dim=1).values.cpu().numpy()


@torch.no_grad()
def l2_features_gpu(L1_chunk_fn, X, bank2, pools, chunk=512):
    """Full-corpus layer-2 pooled features -> (N, n2*PD) numpy.
    `L1_chunk_fn(Xc) -> (n, nb, H, H)` numpy (frozen layer-1 maps)."""
    n2 = len(bank2["K"])
    H = bank2["H"]
    R2 = H - 2
    PD = sum(r * c for r, c in pools)
    cht = to_dev(bank2["ch"].astype(np.int64))
    Kg = to_dev(bank2["K"]); bg = to_dev(bank2["b"])
    out = np.empty((len(X), n2 * PD), np.float32)
    for lo in range(0, len(X), chunk):
        Xc = X[lo:lo + chunk]
        sub = to_dev(L1_chunk_fn(Xc))                    # (n, nb, H, H)
        n = len(sub)
        for gs in range(0, n2, 64):                      # genome blocks
            ge = min(gs + 64, n2)
            P = ge - gs
            sel = sub[:, cht[gs:ge], :, :]
            u = torch.nn.functional.unfold(
                sel.reshape(n * P, 8, H, H), 3)
            z = torch.einsum("pnfk,pk->pnf",
                             u.reshape(n, P, 72, -1).permute(1, 0, 3, 2),
                             Kg[gs:ge]) + bg[gs:ge, None, None]
            z = z.reshape(P, n, R2, R2)
            for a in range(8):
                ids = np.where(bank2["act"][gs:ge] == a)[0]
                if len(ids):
                    idt = to_dev(ids.astype(np.int64))
                    blk = _acts_t(z.index_select(0, idt), a) \
                        .reshape(len(ids) * n, R2, R2)
                    pooled = _pool_t(blk, R2, pools).reshape(len(ids), n, PD) \
                        .permute(1, 0, 2).cpu().numpy()
                    for k2, j in enumerate(ids):
                        col = (gs + j) * PD
                        out[lo:lo + n, col:col + PD] = pooled[:, k2, :]
            del sel, u, z
        del sub
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def bank_features_gpu(patches_fn, X, bank, R, pools, chunk=None):
    """Full-corpus bank features on GPU. `patches_fn(Xc) -> (n, R*R, KD)`.
    Chunk auto-sizes to keep the response tensor under ~3 GB."""
    nb = len(bank["K"])
    if chunk is None:
        chunk = max(256, min(4096, int(3e9 / (R * R * nb * 4))))
    D = sum(r * c for r, c in pools)
    Kg = to_dev(bank["K"]); bg = to_dev(bank["b"])
    out = np.empty((len(X), nb * D), np.float32)
    for lo in range(0, len(X), chunk):
        Xc = X[lo:lo + chunk]
        Pf = to_dev(patches_fn(Xc).reshape(-1, bank["K"].shape[1]).astype(np.float32))
        resp = (Pf @ Kg.T + bg).reshape(len(Xc), R, R, nb)
        for a in range(8):
            ids = np.where(bank["act"] == a)[0]
            if len(ids) == 0:
                continue
            idt = to_dev(ids.astype(np.int64))
            blk = _acts_t(resp.index_select(3, idt), a) \
                .permute(3, 0, 1, 2).reshape(len(ids) * len(Xc), R, R)
            pooled = _pool_t(blk, R, pools).reshape(len(ids), len(Xc), D) \
                .permute(1, 0, 2).cpu().numpy()
            for k, j in enumerate(ids):
                out[lo:lo + len(Xc), j * D:(j + 1) * D] = pooled[:, k, :]
        del resp, Pf
        torch.cuda.empty_cache()
    return out
