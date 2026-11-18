"""Attacks must preserve length and behave as specified; metrics must be exact."""
import shutil

import numpy as np
import pytest

from zwm import SR
from zwm import attacks as A
from zwm.audio import align, resample
from zwm.metrics import (ber, nc, pairwise_ber_matrix, impostor_bers, threshold_at_fpr,
                         tpr_at_threshold, auc, identification_accuracy)
from tests.test_schemes import synth_speechlike

ffmpeg_missing = shutil.which("ffmpeg") is None


@pytest.fixture(scope="module")
def y():
    return synth_speechlike(7)


@pytest.mark.parametrize("name", [k for k in A.CLASSICAL if not k.startswith("mp3") and not k.startswith("pitch")])
def test_classical_preserve_length(name, y):
    z = A.CLASSICAL[name](y)
    assert z.shape == y.shape and z.dtype == np.float32
    assert np.isfinite(z).all()


@pytest.mark.skipif(ffmpeg_missing, reason="ffmpeg not installed")
def test_mp3_roundtrip_is_close(y):
    z = A.mp3_attack(y, 128)
    assert z.shape == y.shape
    # 128 kbps MP3 of a clean tone should keep the waveform mostly intact
    snr = 10 * np.log10(np.mean(y ** 2) / (np.mean((y - z) ** 2) + 1e-12))
    assert snr > 10, f"MP3 128k SNR {snr:.1f} dB too low: misaligned decode?"


def test_awgn_snr(y):
    z = A.awgn_attack(y, 20.0)
    snr = 10 * np.log10(np.mean(y ** 2) / np.mean((z - y) ** 2))
    assert abs(snr - 20.0) < 0.5


def _spectrum(x):
    """Hann-windowed magnitude spectrum (a plain FFT leaks tone energy into all bands)."""
    return np.abs(np.fft.rfft(x * np.hanning(len(x))))


def test_lowpass_removes_high_band(y):
    z = A.lowpass_attack(y, 2000.0)
    Y, Z = _spectrum(y), _spectrum(z)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    hi = f > 4000
    assert Z[hi].sum() < 0.05 * Y[hi].sum()


def test_requantize_levels(y):
    z = A.requantize_attack(y, 8)
    assert len(np.unique(np.round(z * 127))) <= 256


def test_amplitude(y):
    assert np.allclose(A.amplitude_attack(y, 0.7), 0.7 * y)


def test_resample_roundtrip_lowpasses(y):
    z = A.resample_attack(y, 8000)
    assert z.shape == y.shape
    f = np.fft.rfftfreq(len(y), 1 / SR)
    Z = _spectrum(z)
    assert Z[f > 4500].sum() < 0.05 * Z.sum()


def test_align():
    assert len(align(np.ones(10), 5)) == 5
    out = align(np.ones(3), 5)
    assert len(out) == 5 and out[3] == 0


def test_resample_identity_rate():
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(resample(x, SR, SR), x)


def test_attack_registry_groups():
    for k in list(A.CLASSICAL) + list(A.ENCODEC) + list(A.DAC):
        assert k in A.ATTACK_GROUP
    assert set(A.ENCODEC) == {"encodec_1.5k", "encodec_3k", "encodec_6k", "encodec_12k", "encodec_24k"}


# ------------------------------------------------------------------ metrics
def test_ber_nc():
    a = np.array([0, 1, 1, 0], dtype=np.uint8)
    b = np.array([0, 1, 0, 1], dtype=np.uint8)
    assert ber(a, b) == 0.5 and nc(a, b) == 0.0 and nc(a, a) == 1.0
    with pytest.raises(ValueError):
        ber(a, b[:3])


def test_pairwise_and_impostor():
    F = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]], dtype=np.uint8)
    M = pairwise_ber_matrix(F)
    assert M.shape == (3, 3) and np.allclose(np.diag(M), 0)
    assert M[0, 1] == 1.0 and M[0, 2] == 0.5
    assert sorted(impostor_bers(F)) == [0.5, 0.5, 1.0]


def test_threshold_and_tpr():
    imp = np.linspace(0.3, 0.7, 101)
    tau = threshold_at_fpr(imp, 0.01)
    assert np.mean(imp <= tau) <= 0.01 + 1e-9
    assert tau == pytest.approx(0.3)            # exactly one impostor (1/101 < 1%) may pass
    assert tpr_at_threshold(np.array([0.0, 0.2, 0.5]), tau) == pytest.approx(2 / 3)
    # ties: ten identical minima and fpr that allows only five -> threshold must drop below them
    imp2 = np.concatenate([np.full(10, 0.2), np.linspace(0.3, 0.7, 90)])
    tau2 = threshold_at_fpr(imp2, 0.05)
    assert np.mean(imp2 <= tau2) <= 0.05
    # fpr too small for any impostor: threshold sits just below the minimum
    tau3 = threshold_at_fpr(imp, 0.001)
    assert tau3 < imp.min() and np.mean(imp <= tau3) == 0


def test_auc_extremes():
    assert auc(np.array([0.1, 0.2]), np.array([0.4, 0.5])) == 1.0
    assert auc(np.array([0.4, 0.5]), np.array([0.1, 0.2])) == 0.0
    assert auc(np.array([0.3, 0.3]), np.array([0.3, 0.3])) == 0.5


def test_identification_accuracy():
    reg = np.eye(4, dtype=np.uint8)
    att = reg.copy()
    assert identification_accuracy(att, reg) == 1.0
    att2 = np.roll(reg, 1, axis=0)
    assert identification_accuracy(att2, reg) == 0.0
