"""Reimplementation of the audio zero-watermark of Chen and Zhu (EURASIP J. Adv. Signal Process., 2008, art. 453580).

Steps as published:
  1. Cut the audio into frames of 512 samples; rank frames by energy and keep the T most energetic (key K1).
  2. Three-level Haar DWT of each kept frame; keep the approximation subband (512 / 2^3 = 64 coefficients).
  3. DCT of the approximation subband.
  4. Fourth-order cumulant sequence of the DCT coefficients.
  5. In each frame keep the P cumulant values of largest magnitude (their positions are key K2).
  6. Bit = 1 when the kept cumulant value is >= 0.
Verification applies the registered keys K1 and K2 to the received copy.

Choices the paper leaves open, fixed here:
  * T = 128 frames and P = 8 values per frame, so that T * P = 1024 bits (the paper uses a 64x64 logo, T * P = 4096,
    on 28.7 s of 44.1 kHz audio; a 5 s clip at 16 kHz has only 156 frames).
  * The cumulant estimator is not given. We use the standard diagonal slice of the fourth-order cumulant of a
    zero-mean sequence, c4(tau) = m4(tau) - 3 c2(0) c2(tau), with m4(tau) = mean_n x(n)^3 x(n + tau) and
    c2(tau) = mean_n x(n) x(n + tau), for tau = 0..63 (biased estimates, sums over the overlap divided by 64).
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.fft import dct

FRAME = 512
T_FRAMES = 128
P_PER_FRAME = 8
N_COEF = FRAME // 8


def cumulant_slice(x: np.ndarray) -> np.ndarray:
    """c4(tau) for tau = 0..len(x)-1 of the zero-mean version of x."""
    x = x - x.mean()
    n = len(x)
    c2 = np.array([np.dot(x[: n - t], x[t:]) for t in range(n)]) / n
    m4 = np.array([np.dot(x[: n - t] ** 3, x[t:]) for t in range(n)]) / n
    return m4 - 3.0 * c2[0] * c2


def features(y: np.ndarray):
    """Per-frame energy (F,) and cumulant sequences (F, 64) for every full 512-sample frame of y."""
    nf = len(y) // FRAME
    fr = y[: nf * FRAME].reshape(nf, FRAME).astype(np.float64)
    energy = (fr ** 2).sum(axis=1)
    C = np.empty((nf, N_COEF))
    for i in range(nf):
        a = pywt.wavedec(fr[i], "haar", level=3)[0]
        C[i] = cumulant_slice(dct(a, type=2, norm="ortho"))
    return energy, C


def make_key(energy: np.ndarray, C: np.ndarray):
    """K1: indices of the T most energetic frames (in decreasing energy); K2: (T, P) positions of the largest |c4|."""
    k1 = np.argsort(-energy, kind="stable")[:T_FRAMES]
    k2 = np.argsort(-np.abs(C[k1]), axis=1, kind="stable")[:, :P_PER_FRAME]
    return k1, k2


def bits(C: np.ndarray, key) -> np.ndarray:
    """1024 bits of a clip (cumulants C) read with a registration key."""
    k1, k2 = key
    return (np.take_along_axis(C[k1], k2, axis=1) >= 0).astype(np.uint8).ravel()


def bits_many(C_stack: np.ndarray, key) -> np.ndarray:
    """bits for a stack of clips (N, F, 64) under one key -> (N, 1024)."""
    k1, k2 = key
    sel = C_stack[:, k1, :]                                          # (N, T, 64)
    return (np.take_along_axis(sel, k2[None], axis=2) >= 0).astype(np.uint8).reshape(len(C_stack), -1)
