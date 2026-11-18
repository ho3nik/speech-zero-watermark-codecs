"""Error-correcting codes for reading an owner identifier out of a zero-watermark fingerprint.

The registered key is K = F xor x, where x is the codeword of the identifier; a copy with fingerprint F' gives
x xor (F xor F'), so the fingerprint's bit errors act as a binary symmetric channel on the codeword. Positions are
interleaved by a fixed permutation so that errors in neighbouring fingerprint bits fall on distant code bits.

* repetition(r): each message bit at positions i, i+m, ... (m = 1024 // r), majority decoding (as before).
* BCH(1023, 76): t = 187 correctable errors; success is exactly 'at most 187 errors' for its bounded-distance decoder.
* Polar(1024, 76): frozen set from Bhattacharyya parameters of a binary symmetric channel with p = 0.25,
  successive-cancellation decoding with min-sum updates.
"""
from __future__ import annotations

import numpy as np

N = 1024
PERM = np.random.default_rng(0).permutation(N)


# ---------------------------------------------------------------- polar code
def _bhattacharyya(n: int, z: float) -> np.ndarray:
    if n == 1:
        return np.array([z])
    return np.concatenate([_bhattacharyya(n // 2, 2 * z - z * z), _bhattacharyya(n // 2, z * z)])


def polar_info_set(k: int, p: float = 0.25, n: int = N) -> np.ndarray:
    z = _bhattacharyya(n, 2 * np.sqrt(p * (1 - p)))
    return np.sort(np.argsort(z, kind="stable")[:k])


def polar_encode(u: np.ndarray) -> np.ndarray:
    """u: (B, n) bits -> codewords (B, n); x = [enc(u1) ^ enc(u2), enc(u2)]."""
    n = u.shape[1]
    if n == 1:
        return u.copy()
    a, b = polar_encode(u[:, : n // 2]), polar_encode(u[:, n // 2:])
    return np.concatenate([a ^ b, b], axis=1)


def _sc(llr: np.ndarray, frozen: np.ndarray):
    n = llr.shape[1]
    if n == 1:
        u = np.zeros((llr.shape[0], 1), np.uint8) if frozen[0] else (llr < 0).astype(np.uint8)
        return u, u.copy()
    a, b = llr[:, : n // 2], llr[:, n // 2:]
    f = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
    u1, x1 = _sc(f, frozen[: n // 2])
    g = b + (1 - 2 * x1.astype(np.float64)) * a
    u2, x2 = _sc(g, frozen[n // 2:])
    return np.concatenate([u1, u2], axis=1), np.concatenate([x1 ^ x2, x2], axis=1)


def polar_decode(y: np.ndarray, info: np.ndarray) -> np.ndarray:
    """y: received hard bits (B, n). Returns the decoded information bits (B, k)."""
    frozen = np.ones(y.shape[1], bool); frozen[info] = False
    llr = 1.0 - 2.0 * y.astype(np.float64)               # min-sum is scale-invariant
    u, _ = _sc(llr, frozen)
    return u[:, info]


# ---------------------------------------------------------------- evaluation on real error patterns
def exact_recovery(E: np.ndarray, code: str, seed: int = 0) -> np.ndarray:
    """E: (B, 1024) per-bit fingerprint errors. Returns a boolean per row: identifier recovered with no wrong bit."""
    rng = np.random.default_rng(seed)
    Ei = E[:, PERM].astype(np.uint8)
    B = len(E)
    if code == "rep15":
        r, m = 15, N // 15
        rep = np.stack([E[:, i::m][:, :r] for i in range(m)], axis=1)
        return (rep.sum(axis=2) > r / 2).sum(axis=1) == 0
    if code == "polar76":
        info = polar_info_set(76)
        msg = rng.integers(0, 2, (B, 76)).astype(np.uint8)
        u = np.zeros((B, N), np.uint8); u[:, info] = msg
        y = polar_encode(u) ^ Ei
        return (polar_decode(y, info) == msg).all(axis=1)
    if code == "bch76":
        # A bounded-distance BCH decoder recovers the message exactly when at most t = 187 of the 1023 code bits
        # are wrong, and never otherwise, so its success is computed from the error count (decoding itself takes
        # about 20 s per word in pure Python). Verified against galois.BCH(1023, 76).decode on test patterns.
        return Ei[:, :1023].sum(axis=1) <= 187
    raise ValueError(code)


_BCH = None


def _bch():
    global _BCH
    if _BCH is None:
        import galois
        _BCH = galois.BCH(1023, 76)
    return _BCH
