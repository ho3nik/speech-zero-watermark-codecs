"""Evaluation metrics for zero-watermark fingerprints."""
from __future__ import annotations

import numpy as np


def ber(a: np.ndarray, b: np.ndarray) -> float:
    """Bit error rate between two equal-length bit vectors."""
    a = np.asarray(a, dtype=np.uint8).ravel()
    b = np.asarray(b, dtype=np.uint8).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.mean(a != b))


def nc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized correlation of bipolar bits; equals 1 - 2*BER."""
    return 1.0 - 2.0 * ber(a, b)


def pairwise_ber_matrix(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """BER between every row of A (N, L) and every row of B (M, L)."""
    A = np.asarray(A, dtype=np.float32)
    B = A if B is None else np.asarray(B, dtype=np.float32)
    # XOR count via two products: a(1-b) + (1-a)b, exact for 0/1 bits; memory O(N*M) instead of O(N*M*L)
    return (A @ (1.0 - B).T + (1.0 - A) @ B.T) / A.shape[1]


def impostor_bers(F: np.ndarray) -> np.ndarray:
    """BERs between all pairs of distinct rows of F (upper triangle)."""
    M = pairwise_ber_matrix(F)
    iu = np.triu_indices(len(F), k=1)
    return M[iu]


def threshold_at_fpr(impostor: np.ndarray, fpr: float = 0.01) -> float:
    """Largest BER threshold tau such that at most ``fpr`` of impostor pairs satisfy BER <= tau.

    A clip is declared a match when BER <= tau. Ties are handled exactly: if the
    allowed number of impostors cannot be reached without including a tied group,
    the threshold drops below that group. If no impostor may pass, tau is placed
    just below the smallest impostor BER.
    """
    s = np.sort(np.asarray(impostor, dtype=np.float64))
    n = len(s)
    k = int(np.floor(fpr * n))  # number of impostors allowed at or below tau
    for v in s[:k][::-1]:
        if np.searchsorted(s, v, side="right") <= k:
            return float(v)
    return float(np.nextafter(s[0], -np.inf))


def tpr_at_threshold(genuine: np.ndarray, tau: float) -> float:
    return float(np.mean(np.asarray(genuine) <= tau))


def auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Area under the ROC curve for the score -BER (lower BER = more genuine)."""
    g = -np.asarray(genuine, dtype=np.float64)
    i = -np.asarray(impostor, dtype=np.float64)
    # Mann-Whitney U with tie handling
    allv = np.concatenate([g, i])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(allv)
    sorted_v = allv[order]
    # average ranks for ties
    r = np.arange(1, len(allv) + 1, dtype=np.float64)
    uniq, inv, counts = np.unique(sorted_v, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    avg = sums / counts
    ranks[order] = avg[inv]
    rg = ranks[:len(g)].sum()
    n_g, n_i = len(g), len(i)
    return float((rg - n_g * (n_g + 1) / 2) / (n_g * n_i))


def identification_accuracy(F_attacked: np.ndarray, F_registered: np.ndarray) -> float:
    """Fraction of attacked clips whose nearest registered fingerprint is their own.

    Rows of ``F_attacked`` and ``F_registered`` must be in the same clip order.
    """
    M = pairwise_ber_matrix(F_attacked, F_registered)
    return float(np.mean(np.argmin(M, axis=1) == np.arange(len(F_attacked))))


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Equal error rate for the decision BER <= tau, scanning every distinct BER value."""
    g = np.sort(np.asarray(genuine)); i = np.sort(np.asarray(impostor))
    taus = np.unique(np.concatenate([g, i]))
    frr = 1.0 - np.searchsorted(g, taus, side="right") / len(g)
    far = np.searchsorted(i, taus, side="right") / len(i)
    k = int(np.argmin(np.abs(frr - far)))
    return float((frr[k] + far[k]) / 2)
