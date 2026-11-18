"""Attacks. Every attack maps a 16 kHz float32 waveform to a waveform of the same length.

Classical attacks: resampling, MP3 (ffmpeg), additive noise, low/high-pass filtering,
amplitude scaling, 8-bit requantization, pitch shifting.
Neural codec attacks: EnCodec 24 kHz (Defossez et al., 2022) at 1.5-24 kbps,
Descript Audio Codec 16 kHz (Kumar et al., 2023) with 3/6/12 codebooks.
Voice conversion: kNN-VC (Baas et al., 2023) with a self or cross-speaker matching set.

Models are loaded lazily and cached per process.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import SR
from .audio import align, resample, save_wav, load_wav

Attack = Callable[[np.ndarray], np.ndarray]

# --------------------------------------------------------------------------- classical
def resample_attack(y: np.ndarray, inter_sr: int = 8000) -> np.ndarray:
    return align(resample(resample(y, SR, inter_sr), inter_sr, SR), len(y))


def mp3_attack(y: np.ndarray, kbps: int = 32) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        src, mp3, out = Path(d) / "a.wav", Path(d) / "a.mp3", Path(d) / "b.wav"
        save_wav(str(src), y)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-codec:a", "libmp3lame",
                        "-b:a", f"{kbps}k", str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ar", str(SR), "-ac", "1",
                        str(out)], check=True)
        z = load_wav(str(out))
    # the LAME encoder adds a fixed encoder delay; ffmpeg strips most of it but
    # the decoded file can still differ by a few samples, so align to the input
    return align(z, len(y))


def awgn_attack(y: np.ndarray, snr_db: float = 20.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_sig = np.mean(y ** 2) + 1e-12
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (y + rng.normal(0, np.sqrt(p_noise), size=len(y))).astype(np.float32)


def lowpass_attack(y: np.ndarray, fc: float = 4000.0, order: int = 6) -> np.ndarray:
    sos = butter(order, fc, btype="low", fs=SR, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def highpass_attack(y: np.ndarray, fc: float = 200.0, order: int = 6) -> np.ndarray:
    sos = butter(order, fc, btype="high", fs=SR, output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def amplitude_attack(y: np.ndarray, gain: float = 0.7) -> np.ndarray:
    return (y * gain).astype(np.float32)


def requantize_attack(y: np.ndarray, bits: int = 8) -> np.ndarray:
    q = 2 ** (bits - 1)
    return (np.round(np.clip(y, -1, 1) * (q - 1)) / (q - 1)).astype(np.float32)


def pitch_shift_attack(y: np.ndarray, semitones: float = 2.0) -> np.ndarray:
    import librosa
    z = librosa.effects.pitch_shift(y.astype(np.float32), sr=SR, n_steps=semitones)
    return align(z, len(y))


def opus_attack(y: np.ndarray, kbps: int = 12) -> np.ndarray:
    """Opus (libopus via ffmpeg) at ``kbps``; Opus runs at 48 kHz internally, output resampled to 16 kHz."""
    with tempfile.TemporaryDirectory() as d:
        src, enc, out = Path(d) / "a.wav", Path(d) / "a.opus", Path(d) / "b.wav"
        save_wav(str(src), y)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-codec:a", "libopus",
                        "-b:a", f"{kbps}k", "-vbr", "on", "-application", "voip", str(enc)], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(enc), "-ar", str(SR), "-ac", "1",
                        str(out)], check=True)
        z = load_wav(str(out))
    return align(z, len(y))


# --------------------------------------------------------------------------- desynchronization
def shift_attack(y: np.ndarray, ms: float = 20.0) -> np.ndarray:
    """Circular time shift by ``ms`` milliseconds (content moves later; the tail wraps to the front)."""
    return np.roll(y, int(round(ms * SR / 1000))).astype(np.float32)


def delay_attack(y: np.ndarray, ms: float = 23.0) -> np.ndarray:
    """Non-circular delay: prepend ``ms`` milliseconds of silence and cut the end to keep the length
    (what a recording that starts a little early, or a stream joined late, looks like)."""
    n = int(round(ms * SR / 1000))
    return np.concatenate([np.zeros(n, np.float32), y[:len(y) - n]]).astype(np.float32)


def crop_attack(y: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Drop the first ``frac`` of the clip and pad silence at the end (same length, content shifted earlier)."""
    n = int(round(frac * len(y)))
    return np.concatenate([y[n:], np.zeros(n, np.float32)]).astype(np.float32)


def stretch_attack(y: np.ndarray, pct: float = 2.0) -> np.ndarray:
    """Time-stretch by ``pct`` percent (phase vocoder, pitch preserved), then cut or zero-pad to the original length."""
    import librosa
    z = librosa.effects.time_stretch(y.astype(np.float32), rate=1.0 / (1.0 + pct / 100.0))
    return align(z, len(y))


# --------------------------------------------------------------------------- EnCodec
@lru_cache(maxsize=None)
def _encodec_model(bandwidth: float):
    import torch
    from encodec import EncodecModel
    torch.set_num_threads(max(1, int(os.environ.get("ZWM_THREADS", os.cpu_count() or 1))))
    m = EncodecModel.encodec_model_24khz()
    m.set_target_bandwidth(bandwidth)
    return m.eval()


def encodec_attack(y: np.ndarray, kbps: float = 6.0) -> np.ndarray:
    """Encode/decode with EnCodec 24 kHz at ``kbps`` in {1.5, 3, 6, 12, 24}."""
    import torch
    m = _encodec_model(float(kbps))
    x24 = resample(y, SR, 24000)
    with torch.no_grad():
        t = torch.from_numpy(x24)[None, None, :]
        frames = m.encode(t)
        out = m.decode(frames)[0, 0].numpy()
    return align(resample(out, 24000, SR), len(y))


# --------------------------------------------------------------------------- DAC
@lru_cache(maxsize=None)
@lru_cache(maxsize=None)
def _dac_model():
    import torch
    import dac
    torch.set_num_threads(max(1, int(os.environ.get("ZWM_THREADS", os.cpu_count() or 1))))
    path = dac.utils.download(model_type="16khz")
    return dac.DAC.load(path).eval()


def dac_attack(y: np.ndarray, n_quantizers: int = 12) -> np.ndarray:
    """Encode/decode with DAC 16 kHz using the first ``n_quantizers`` of 12 codebooks.

    Each codebook carries 10 bits per 20 ms frame, so 12/6/3 codebooks correspond
    to 6/3/1.5 kbps.
    """
    import torch
    m = _dac_model()
    with torch.no_grad():
        x = torch.from_numpy(y.astype(np.float32))[None, None, :]
        x = m.preprocess(x, SR)
        z, codes, latents, _, _ = m.encode(x, n_quantizers=n_quantizers)
        out = m.decode(z)[0, 0].numpy()
    return align(out, len(y))


# --------------------------------------------------------------------------- kNN-VC
KNNVC_REPO = os.environ.get("KNNVC_REPO", str(Path(__file__).resolve().parents[2] / "ext" / "knn-vc"))


@lru_cache(maxsize=None)
def _knnvc():
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("ZWM_THREADS", os.cpu_count() or 1))))
    return torch.hub.load(KNNVC_REPO, "knn_vc", source="local", prematched=True, device="cpu", trust_repo=True)


class KNNVCAttack:
    """kNN-VC voice conversion with a fixed matching set.

    ``matching_wavs`` is a list of 16 kHz waveforms of the target speaker. For a
    *self* voice-conversion attack pass other utterances of the same speaker.
    """

    def __init__(self, matching_wavs: List[np.ndarray], topk: int = 4, chunk_s: float = 8.0):
        import torch
        self.model = _knnvc()
        self.topk = topk
        # WavLM self-attention is quadratic in length; feed the matching audio in chunks
        # (kNN-VC concatenates frame features, so chunking does not change the matching set)
        n = int(chunk_s * SR)
        tensors = []
        for w in matching_wavs:
            w = np.asarray(w, dtype=np.float32)
            for i in range(0, len(w), n):
                piece = w[i:i + n]
                if len(piece) >= SR:  # skip fragments shorter than 1 s
                    tensors.append(torch.from_numpy(piece)[None, :])
        self.matching_set = self.model.get_matching_set(tensors, vad_trigger_level=0)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        import torch
        q = self.model.get_features(torch.from_numpy(np.asarray(y, dtype=np.float32))[None, :])
        out = self.model.match(q, self.matching_set, topk=self.topk, target_duration=len(y) / SR)
        return align(out.numpy(), len(y))


# --------------------------------------------------------------------------- registry
CLASSICAL: Dict[str, Attack] = {
    "resample_8k": lambda y: resample_attack(y, 8000),
    "mp3_32k": lambda y: mp3_attack(y, 32),
    "mp3_128k": lambda y: mp3_attack(y, 128),
    "awgn_20dB": lambda y: awgn_attack(y, 20.0),
    "lowpass_4k": lambda y: lowpass_attack(y, 4000.0),
    "highpass_200": lambda y: highpass_attack(y, 200.0),
    "amp_0.7": lambda y: amplitude_attack(y, 0.7),
    "requant_8bit": lambda y: requantize_attack(y, 8),
    "pitch_+2st": lambda y: pitch_shift_attack(y, 2.0),
}

OPUS: Dict[str, Attack] = {
    f"opus_{k}k": (lambda k: (lambda y: opus_attack(y, k)))(k) for k in (6, 12, 16)
}

DESYNC: Dict[str, Attack] = {
    "delay_7ms": lambda y: delay_attack(y, 7.0),
    "delay_23ms": lambda y: delay_attack(y, 23.0),
    "delay_37ms": lambda y: delay_attack(y, 37.0),
    "delay_250ms": lambda y: delay_attack(y, 250.0),
    "crop_10pct": lambda y: crop_attack(y, 0.10),
    "stretch_+2pct": lambda y: stretch_attack(y, 2.0),
    "stretch_-2pct": lambda y: stretch_attack(y, -2.0),
    "stretch_+5pct": lambda y: stretch_attack(y, 5.0),
    "stretch_-5pct": lambda y: stretch_attack(y, -5.0),
}

ENCODEC: Dict[str, Attack] = {
    f"encodec_{b}k".replace(".0k", "k"): (lambda b: (lambda y: encodec_attack(y, b)))(b)
    for b in (1.5, 3.0, 6.0, 12.0, 24.0)
}

DAC: Dict[str, Attack] = {
    f"dac_{n}cb": (lambda n: (lambda y: dac_attack(y, n)))(n) for n in (3, 6, 12)
}

ATTACK_GROUP = {**{k: "classical" for k in CLASSICAL}, **{k: "opus" for k in OPUS}, **{k: "desync" for k in DESYNC},
                **{k: "encodec" for k in ENCODEC}, **{k: "dac" for k in DAC}, "knnvc_self": "vc", "knnvc_cross": "vc",
                "speecht5_cross": "vc"}
