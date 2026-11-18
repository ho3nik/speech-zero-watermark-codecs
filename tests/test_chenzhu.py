import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import chenzhu as CZ


def _sig(seed=0, n=80000):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.05 * rng.standard_normal(n)).astype(np.float32)


def test_shapes_and_length():
    e, C = CZ.features(_sig())
    assert C.shape == (80000 // 512, 64) and e.shape == (80000 // 512,)
    assert CZ.bits(C, CZ.make_key(e, C)).shape == (1024,)


def test_gain_invariance():
    y = _sig()
    e, C = CZ.features(y)
    k = CZ.make_key(e, C)
    assert (CZ.bits(CZ.features(0.3 * y)[1], k) == CZ.bits(C, k)).all()


def test_key_selects_high_energy_frames():
    y = _sig()
    y[:512 * 20] *= 0.01
    e, C = CZ.features(y)
    k1, k2 = CZ.make_key(e, C)
    assert len(set(k1)) == CZ.T_FRAMES and not set(range(20)) & set(k1)
    assert k2.shape == (CZ.T_FRAMES, CZ.P_PER_FRAME)


def test_cumulant_of_gaussian_is_small():
    x = np.random.default_rng(1).standard_normal(4096)
    c = CZ.cumulant_slice(x)
    assert abs(c[0]) < 0.3 * 3 * np.var(x) ** 2


def test_bits_many_matches_bits():
    e, C = CZ.features(_sig())
    k = CZ.make_key(e, C)
    assert (CZ.bits_many(C[None], k)[0] == CZ.bits(C, k)).all()
