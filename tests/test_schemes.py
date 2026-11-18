"""Invariants every fingerprint scheme must satisfy. Run with: pytest -q"""
import numpy as np
import pytest

from zwm import SR
from zwm.schemes import (BASELINES, N_BITS, CandidatePool, select_bits, register, verify,
                         make_selected_extractor, inter_content_disagreement)


def synth_speechlike(seed: int, seconds: float = 5.0) -> np.ndarray:
    """Harmonic tone with a wandering f0 and a modulated envelope (deterministic)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 120 + 40 * rng.random() + 20 * np.sin(2 * np.pi * (0.5 + rng.random()) * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    y = sum((1.0 / h) * np.sin(h * phase) for h in range(1, 12))
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * (2 + 3 * rng.random()) * t + rng.random()))
    y = y * env + 0.01 * rng.standard_normal(len(t))
    return (0.9 * y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture(scope="module")
def clips():
    return [synth_speechlike(i) for i in range(6)]


@pytest.mark.parametrize("name", list(BASELINES))
def test_length_and_dtype(name, clips):
    f = BASELINES[name](clips[0])
    assert f.dtype == np.uint8 and f.shape == (N_BITS,)
    assert set(np.unique(f)) <= {0, 1}


@pytest.mark.parametrize("name", list(BASELINES))
def test_deterministic_identity(name, clips):
    f = BASELINES[name]
    assert np.array_equal(f(clips[0]), f(clips[0].copy()))


@pytest.mark.parametrize("name", list(BASELINES))
def test_gain_invariance(name, clips):
    """All three families compare relative quantities, so scaling must not flip bits."""
    f = BASELINES[name]
    for g in (0.3, 0.7, 1.0):
        assert np.array_equal(f(clips[1]), f(clips[1] * g))


@pytest.mark.parametrize("name", list(BASELINES))
def test_balanced(name, clips):
    f = BASELINES[name]
    for c in clips:
        m = f(c).mean()
        assert 0.4 <= m <= 0.6, f"{name}: mean bit {m}"


@pytest.mark.parametrize("name", list(BASELINES))
def test_discriminative(name, clips):
    """Different content must give BER well away from zero (ideal is 0.5)."""
    f = BASELINES[name]
    if name == "WavLM-LSH":
        # a speech model sees the synthetic clips as alike; test this family on real clips instead
        from pathlib import Path
        from zwm.audio import load_wav
        paths = sorted((Path(__file__).resolve().parents[1] / "data" / "clips").glob("*.wav"))[:6]
        if len(paths) < 6:
            pytest.skip("real clips not present")
        clips = [load_wav(str(p)) for p in paths]
    fps = [f(c) for c in clips]
    bers = [np.mean(fps[i] != fps[j]) for i in range(len(fps)) for j in range(i + 1, len(fps))]
    assert min(bers) > 0.25 and 0.35 < np.mean(bers) < 0.65


def test_xor_roundtrip(clips):
    rng = np.random.default_rng(1)
    w = rng.integers(0, 2, N_BITS, dtype=np.uint8)
    f = BASELINES["Philips"](clips[2])
    key = register(f, w)
    assert np.array_equal(verify(f, key), w)
    # BER of recovered watermark equals BER of fingerprint under a corrupted fingerprint
    f2 = f.copy()
    f2[:100] ^= 1
    assert np.mean(verify(f2, key) != w) == pytest.approx(100 / N_BITS)


def test_pool_concatenates_members(clips):
    pool = CandidatePool()
    bits = pool(clips[0])
    sizes = [len(fn(clips[0])) for _, fn in pool.members]
    assert len(bits) == sum(sizes)
    assert len(pool.family_of_bit()) == len(bits)


def test_select_bits_prefers_stable_and_rejects_constant():
    rng = np.random.default_rng(0)
    N, P = 200, 50
    orig = rng.integers(0, 2, (N, P), dtype=np.uint8)
    orig[:, 0] = 1                       # constant bit: stable but useless
    flip_rate = np.linspace(0.0, 0.5, P)  # bit j flips in exactly round(flip_rate[j]*N) clips
    flip_rate[0] = 0.0
    att = orig.copy()
    for j in range(P):
        k = int(round(flip_rate[j] * N))
        att[rng.choice(N, k, replace=False), j] ^= 1
    res = select_bits(orig, {"a": att}, n_select=10)
    assert 0 not in res["idx"], "constant bit must be rejected"
    # the ten most stable non-constant bits are indices 1..10
    assert set(res["idx"]) == set(range(1, 11))


def test_select_bits_max_is_worst_case():
    rng = np.random.default_rng(0)
    N, P = 300, 4
    orig = rng.integers(0, 2, (N, P), dtype=np.uint8)
    a1 = orig.copy(); a1[:, 0] ^= 1                 # bit 0 always flips under attack 1
    a2 = orig.copy(); a2[:, 1] ^= (rng.random(N) < 0.3).astype(np.uint8)
    res = select_bits(orig, {"a1": a1, "a2": a2}, n_select=2)
    assert set(res["idx"]) == {2, 3}
    assert res["flip"][0] == pytest.approx(1.0)


def test_selected_extractor_subsets_pool(clips):
    pool = CandidatePool()
    idx = np.arange(0, 2048, 2)
    ext = make_selected_extractor(pool, idx)
    assert np.array_equal(ext(clips[3]), pool(clips[3])[idx])


def test_inter_content_disagreement_formula():
    bits = np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1]], dtype=np.uint8)
    q = inter_content_disagreement(bits)
    assert q[0] == 0.0 and q[1] == pytest.approx(0.5) and q[2] == pytest.approx(0.5)
