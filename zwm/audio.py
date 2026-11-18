"""Audio I/O and alignment helpers. Working sample rate is 16 kHz mono float32."""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

from . import SR


def load_wav(path: str, target_sr: int = SR) -> np.ndarray:
    """Load any audio file as mono float32 at ``target_sr``."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != target_sr:
        y = resample(y, sr, target_sr)
    return y.astype(np.float32)


def save_wav(path: str, y: np.ndarray, sr: int = SR) -> None:
    sf.write(path, np.clip(y, -1.0, 1.0).astype(np.float32), sr, subtype="PCM_16")


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling with an anti-aliasing filter."""
    if sr_in == sr_out:
        return y.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(y, sr_out // g, sr_in // g).astype(np.float32)


def align(y: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad ``y`` to exactly ``n`` samples (keeps the start)."""
    y = np.asarray(y, dtype=np.float32).ravel()
    if len(y) >= n:
        return y[:n]
    return np.concatenate([y, np.zeros(n - len(y), dtype=np.float32)])


def peak_normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(y)) + 1e-9
    return (y / m * peak).astype(np.float32)


def crop_segment(y: np.ndarray, seconds: float, sr: int = SR, offset_frac: float = 0.0) -> np.ndarray:
    """Take a ``seconds``-long segment starting at ``offset_frac`` of the slack."""
    n = int(round(seconds * sr))
    if len(y) <= n:
        return align(y, n)
    start = int((len(y) - n) * offset_frac)
    return y[start:start + n].copy()
