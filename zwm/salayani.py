"""Reimplementation of the supervised zero-watermark RSZW of Salayani, Bakhtiari and Ghafarian
(Circuits Syst. Signal Process. 42:3668-3705, 2023), following Section 3 of that paper.

Registration (per host clip):
  1. Cut the clip into L equal, non-overlapping segments and pick M2 of them with a keyed pseudo-random permutation.
  2. Wavelet-transform each picked segment and keep the low-frequency (approximation) band; split it into M1 pieces.
  3. DCT each piece; feature F(k, j) = largest absolute DCT coefficient of piece j of segment k  (M2 x M1 matrix).
  4. Binary pattern P(i, j) = 1 if F(i, j) > 0, else 0 (Eq. 5).  ADW = P xor W', W' the Arnold-scrambled logo.
  5. For each simulated attack, compute F of the attacked host, map it to [0, 1], take the 5 x 5 window around
     every cell as a 25-dimensional sample whose label is the ADW bit of that cell, and train one SVM per attack.
Verification: compute F of the received clip, predict labels with every SVM, xor with the received P, and accept
if at least one SVM's logo is close enough to the owner's logo (the paper's "at least one" rule).

Choices the paper leaves open, fixed here and stated in the manuscript:
  * L = 64 segments of a 5 s clip, M1 = M2 = 32 (a 1024-bit logo, as for the other schemes).
  * Wavelet: one-dimensional three-level Daubechies-4 transform (the paper says "two-dimensional DWT" of a
    one-dimensional segment and names neither the wavelet nor the level).
  * SVM: RBF kernel, C = 100, gamma = 50, chosen on training-split clips only (scripts/tune_salayani.py: the
    setting with the lowest logo BER on MP3 128 kbps for SVMs trained on MP3 32 kbps); 5 x 5 windows use edge padding.
  * F is non-negative by construction, so Eq. 5 as printed gives P = all ones for every original recording,
    evidently a typo. We evaluate the rule as printed (the default); ZWM_SZ_RULE=mean thresholds at the mean of the
    clip's feature matrix instead.
  * The Arnold scrambling is a fixed permutation of the logo; it does not change any bit error rate, so it is omitted.
"""
from __future__ import annotations

import os

import numpy as np
import pywt
from scipy.fft import dct
from sklearn.svm import SVC

L_SEG, M1, M2 = 64, 32, 32


def selection_key(seed: int = 12345) -> np.ndarray:
    return np.random.default_rng(seed).permutation(L_SEG)[:M2]


def features(y: np.ndarray, key: np.ndarray) -> np.ndarray:
    """M2 x M1 feature matrix of a clip."""
    seg = len(y) // L_SEG
    F = np.empty((M2, M1))
    for k, i in enumerate(key):
        a = pywt.wavedec(y[i * seg:(i + 1) * seg].astype(np.float64), "db4", level=3)[0]
        m = len(a) // M1
        pieces = a[: m * M1].reshape(M1, m)
        F[k] = np.abs(dct(pieces, type=2, norm="ortho", axis=1)).max(axis=1)
    return F


RULE = os.environ.get("ZWM_SZ_RULE", "printed")


def pattern(F: np.ndarray) -> np.ndarray:
    """Binary pattern of Eq. 5. F is a magnitude, so "F > 0" as printed (the default, used for every result in the
    paper) sets every bit of an original recording to 1, evidently a typo. ZWM_SZ_RULE=mean thresholds each feature
    at the mean of the clip's feature matrix instead."""
    if RULE == "printed":
        return (F > 0).astype(np.uint8)
    return (F > F.mean()).astype(np.uint8)


def windows(F: np.ndarray) -> np.ndarray:
    """Map F to [0, 1] and return the 5 x 5 neighbourhood of every cell as rows (M2*M1, 25)."""
    G = (F - F.min()) / (F.max() - F.min() + 1e-12)
    P = np.pad(G, 2, mode="edge")
    return np.stack([P[i:i + 5, j:j + 5].ravel() for i in range(F.shape[0]) for j in range(F.shape[1])])


class RSZW:
    """One registration: a logo, the key, and one SVM per training attack."""

    def __init__(self, logo_bits: np.ndarray, key: np.ndarray):
        self.logo, self.key, self.svms = logo_bits.astype(np.uint8), key, []

    def fit(self, host_clean: np.ndarray, attacked_hosts: list[np.ndarray]):
        adw = pattern(features(host_clean, self.key)).ravel() ^ self.logo
        self.adw = adw
        for ya in attacked_hosts:
            X = windows(features(ya, self.key))
            if len(np.unique(adw)) < 2:
                raise ValueError("degenerate labels")
            self.svms.append(SVC(kernel="rbf", C=100.0, gamma=50.0).fit(X, adw))
        return self

    def logo_errors(self, Fs: list[np.ndarray]) -> np.ndarray:
        """For received feature matrices Fs, the per-bit logo error pattern of the best SVM, (N, 1024)."""
        X = np.concatenate([windows(F) for F in Fs])
        Pr = np.concatenate([pattern(F).ravel() for F in Fs]).reshape(len(Fs), -1)
        best = None
        for s in self.svms:
            C = s.predict(X).astype(np.uint8).reshape(len(Fs), -1)
            E = ((C ^ Pr) != self.logo).astype(np.uint8)           # extracted logo vs owner's logo
            if best is None:
                best = E
            else:
                take = E.mean(axis=1) < best.mean(axis=1)
                best[take] = E[take]
        return best
