"""Zero-watermark fingerprint extractors and attack-aware bit selection.

A zero-watermark leaves the host signal untouched. It extracts a binary feature
sequence F (the "fingerprint") from the audio and registers K = F XOR W with a
trusted third party, where W is the owner's watermark bits. At verification time
F' is extracted from the (possibly attacked) audio and W' = F' XOR K is compared
with W. Because XOR is an involution, BER(W, W') == BER(F, F'), so every result in
this package is reported as the bit error rate between F and F'.

Three fingerprint families are implemented; each returns a uint8 bit vector.

- ``dwt_dct_bits``   : segment-wise DWT approximation -> DCT -> magnitude rank bits
                       (the DWT/DCT family used by many audio zero-watermarking papers)
- ``philips_bits``   : Haitsma & Kalker (2002) band-energy difference fingerprint
- ``mfcc_bits``      : segment-wise mean MFCCs thresholded at their temporal median

``CandidatePool`` concatenates higher-resolution versions of all three, and
``select_bits`` performs filter-based, multi-task (one task per simulated attack)
selection of the most stable, still discriminative bits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np
import pywt
from scipy.fft import dct
from scipy.signal import get_window

from . import SR

Extractor = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Family 1: DWT-DCT magnitude-rank bits
# ---------------------------------------------------------------------------
def dwt_dct_bits(y: np.ndarray, n_seg: int = 64, n_coef: int = 16,
                 wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """Segment -> ``level``-level DWT approximation -> DCT -> bit = |c_k| > median.

    Coefficient 0 (DC) is skipped. Each segment yields ``n_coef`` bits that are
    balanced by construction (half above the within-segment median).
    """
    y = np.asarray(y, dtype=np.float64)
    seg_len = len(y) // n_seg
    out = np.empty(n_seg * n_coef, dtype=np.uint8)
    for s in range(n_seg):
        seg = y[s * seg_len:(s + 1) * seg_len]
        ca = pywt.wavedec(seg, wavelet, level=level)[0]
        c = dct(ca, norm="ortho")
        v = np.abs(c[1:1 + n_coef])
        out[s * n_coef:(s + 1) * n_coef] = v > np.median(v)
    return out


# ---------------------------------------------------------------------------
# Family 2: Philips (Haitsma-Kalker) band-energy difference fingerprint
# ---------------------------------------------------------------------------
def _band_edges(n_bands: int, fmin: float, fmax: float) -> np.ndarray:
    return np.geomspace(fmin, fmax, n_bands + 1)


def philips_bits(y: np.ndarray, sr: int = SR, n_frames: int = 32, n_bands: int = 33,
                 fmin: float = 300.0, fmax: float = 2000.0, win_s: float = 0.37) -> np.ndarray:
    """Band-energy difference bits following Haitsma & Kalker (2002).

    ``n_frames + 1`` overlapping frames are placed uniformly over the clip. For
    frame n and band m the bit is 1 when
        (E[n,m] - E[n,m+1]) - (E[n-1,m] - E[n-1,m+1]) > 0.
    With ``n_bands`` = 33 this gives 32 bits per frame, 32 x 32 = 1024 bits.
    """
    y = np.asarray(y, dtype=np.float64)
    win = int(round(win_s * sr))
    nfft = int(2 ** np.ceil(np.log2(win)))
    w = get_window("hann", win, fftbins=True)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    edges = _band_edges(n_bands, fmin, fmax)
    band_idx = np.digitize(freqs, edges) - 1  # -1 below fmin, n_bands at/above fmax
    centers = np.linspace(win / 2, len(y) - win / 2, n_frames + 1)
    E = np.zeros((n_frames + 1, n_bands))
    for i, c in enumerate(centers):
        start = int(round(c - win / 2))
        frame = y[start:start + win]
        if len(frame) < win:
            frame = np.pad(frame, (0, win - len(frame)))
        p = np.abs(np.fft.rfft(frame * w, nfft)) ** 2
        for m in range(n_bands):
            E[i, m] = p[band_idx == m].sum()
    d = E[:, :-1] - E[:, 1:]                 # (n_frames+1, n_bands-1)
    bits = (d[1:] - d[:-1]) > 0               # (n_frames, n_bands-1)
    return bits.astype(np.uint8).ravel()


# ---------------------------------------------------------------------------
# Family 3: segment-wise MFCC means thresholded at the temporal median
# ---------------------------------------------------------------------------
def _mel_filterbank(sr: int, nfft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((nfft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, nfft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, ce, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, ce):
            fb[m - 1, k] = (k - lo) / max(ce - lo, 1)
        for k in range(ce, hi):
            fb[m - 1, k] = (hi - k) / max(hi - ce, 1)
    return fb


def _log_mel(y: np.ndarray, sr: int, n_mels: int, nfft: int = 512, hop: int = 160) -> np.ndarray:
    w = get_window("hann", nfft, fftbins=True)
    n_frames = 1 + (len(y) - nfft) // hop
    if n_frames < 1:
        y = np.pad(y, (0, nfft - len(y)))
        n_frames = 1
    idx = np.arange(nfft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * w
    p = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2  # (T, nfft/2+1)
    fb = _mel_filterbank(sr, nfft, n_mels)
    return np.log(p @ fb.T + 1e-10)                       # (T, n_mels)


def mfcc_bits(y: np.ndarray, sr: int = SR, n_seg: int = 64, n_coef: int = 16, n_mels: int = 40) -> np.ndarray:
    """Mean MFCC (coefficients 1..n_coef) per segment; bit = value > median over segments.

    Comparing against the temporal median makes each coefficient column balanced
    and removes any dependence on absolute gain.
    """
    y = np.asarray(y, dtype=np.float64)
    seg_len = len(y) // n_seg
    M = np.zeros((n_seg, n_coef))
    for s in range(n_seg):
        seg = y[s * seg_len:(s + 1) * seg_len]
        lm = _log_mel(seg, sr, n_mels)
        c = dct(lm, type=2, norm="ortho", axis=1)[:, 1:1 + n_coef]
        M[s] = c.mean(axis=0)
    bits = M > np.median(M, axis=0, keepdims=True)
    return bits.astype(np.uint8).ravel()


# ---------------------------------------------------------------------------
# Family 4: random-projection sign bits of segment-mean WavLM features (a locality-sensitive hash
# of a learned representation; no training, still a filter). The model is loaded lazily.
# ---------------------------------------------------------------------------
class WavLMLSH:
    """Sign bits of random projections of segment-mean hidden states of WavLM-Base-Plus.

    The clip is split into ``n_seg`` equal segments; the hidden states of ``layer`` (50 frames/s, 768-dim)
    are averaged within each segment, the per-clip mean over segments is subtracted (so that the bits
    describe content relative to the clip, not the model's constant offset), and each segment vector is
    projected onto ``n_proj`` fixed Gaussian directions; bit = projection > 0. 16 x 64 = 1024 bits.
    """

    def __init__(self, n_seg: int = 16, n_proj: int = 64, layer: int = 6, seed: int = 0,
                 model_id: str = "microsoft/wavlm-base-plus"):
        self.n_seg, self.n_proj, self.layer, self.model_id = n_seg, n_proj, layer, model_id
        self.W = np.random.default_rng(seed).standard_normal((768, n_proj)).astype(np.float32)
        self._model = None

    def _load(self):
        if self._model is None:
            import os
            import torch
            from transformers import AutoModel
            torch.set_num_threads(max(1, int(os.environ.get("ZWM_THREADS", os.cpu_count() or 1))))
            self._model = AutoModel.from_pretrained(self.model_id).eval()
        return self._model

    def features(self, y: np.ndarray) -> np.ndarray:
        import torch
        m = self._load()
        x = torch.from_numpy(np.asarray(y, dtype=np.float32))[None, :]
        x = (x - x.mean()) / (x.std() + 1e-7)
        with torch.no_grad():
            h = m(x, output_hidden_states=True).hidden_states[self.layer][0].numpy()   # (frames, 768)
        return h

    def __call__(self, y: np.ndarray) -> np.ndarray:
        h = self.features(y)
        edges = np.linspace(0, len(h), self.n_seg + 1).astype(int)
        S = np.stack([h[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])   # (n_seg, 768)
        S = S - S.mean(axis=0, keepdims=True)
        return (S @ self.W > 0).astype(np.uint8).ravel()


# ---------------------------------------------------------------------------
# Baseline scheme registry (each returns exactly 1024 bits on a 5 s clip)
# ---------------------------------------------------------------------------
BASELINES: Dict[str, Extractor] = {
    "DWT-DCT": lambda y: dwt_dct_bits(y, n_seg=64, n_coef=16),
    "Philips": lambda y: philips_bits(y, n_frames=32, n_bands=33),
    "MFCC": lambda y: mfcc_bits(y, n_seg=64, n_coef=16),
    "WavLM-LSH": WavLMLSH(),
}
N_BITS = 1024


# ---------------------------------------------------------------------------
# Candidate pool and attack-aware selection
# ---------------------------------------------------------------------------
@dataclass
class CandidatePool:
    """Concatenation of several extractors at higher resolution than the baselines."""
    members: List[tuple] = field(default_factory=lambda: [
        ("dwtdct32", lambda y: dwt_dct_bits(y, n_seg=64, n_coef=32)),            # 2048
        ("philips65_300-2000", lambda y: philips_bits(y, n_frames=32, n_bands=65)),  # 2048
        ("philips33_300-4000", lambda y: philips_bits(y, n_frames=32, n_bands=33, fmax=4000.0)),  # 1024
        ("mfcc24", lambda y: mfcc_bits(y, n_seg=64, n_coef=24)),                 # 1536
    ])

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return np.concatenate([f(y) for _, f in self.members]).astype(np.uint8)

    def family_of_bit(self) -> np.ndarray:
        """Return an array giving the member name of each pooled bit (for reporting)."""
        probe = np.zeros(5 * SR, dtype=np.float32)
        probe[::97] = 0.5  # non-silent dummy so extractors run
        names = []
        for name, f in self.members:
            names += [name] * len(f(probe))
        return np.array(names)


def inter_content_disagreement(bits: np.ndarray) -> np.ndarray:
    """Per-bit expected disagreement between two different clips (2p(1-p), p = P(bit=1))."""
    p = bits.mean(axis=0)
    return 2.0 * p * (1.0 - p)


def select_bits(train_orig: np.ndarray, train_attacked: Dict[str, np.ndarray], n_select: int = N_BITS,
                aggregate: str = "max", min_disagreement: float = 0.40) -> dict:
    """Filter-based multi-task selection of stable, discriminative bits.

    Parameters
    ----------
    train_orig : (N, P) uint8 bits of the clean training clips.
    train_attacked : mapping attack name -> (N, P) bits of the attacked training clips.
        Each attack is one *task*; a bit is good only if it survives every task.
    aggregate : 'max' (worst case over tasks, the multi-task min-max criterion) or 'mean'.
    min_disagreement : bits whose inter-content disagreement is below this are discarded
        (they are stable only because they are nearly constant across clips).

    Returns
    -------
    dict with 'idx' (selected bit indices, sorted), 'score', 'flip', 'disagreement'.
    """
    flips = np.stack([(train_orig != A).mean(axis=0) for A in train_attacked.values()])  # (T, P)
    agg = flips.max(axis=0) if aggregate == "max" else flips.mean(axis=0)
    q = inter_content_disagreement(train_orig)
    score = q - agg                           # high = discriminative and stable
    score = np.where(q >= min_disagreement, score, -np.inf)
    order = np.argsort(-score, kind="stable")
    idx = np.sort(order[:n_select])
    if not np.isfinite(score[idx]).all():
        raise ValueError("not enough candidate bits satisfy the disagreement constraint")
    return {"idx": idx, "score": score, "flip": agg, "disagreement": q, "flips_per_task": flips}


def make_selected_extractor(pool: CandidatePool, idx: np.ndarray) -> Extractor:
    idx = np.asarray(idx)
    return lambda y: pool(y)[idx]


# ---------------------------------------------------------------------------
# Watermark registration / verification (XOR zero-watermark)
# ---------------------------------------------------------------------------
def register(fingerprint: np.ndarray, watermark: np.ndarray) -> np.ndarray:
    """Zero-watermark key K = F XOR W (stored with the third party)."""
    return np.bitwise_xor(fingerprint.astype(np.uint8), watermark.astype(np.uint8))


def verify(fingerprint_attacked: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Recovered watermark W' = F' XOR K."""
    return np.bitwise_xor(fingerprint_attacked.astype(np.uint8), key.astype(np.uint8))
