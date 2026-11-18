"""zwm: audio zero-watermarking under neural codec and voice-conversion attacks.

Modules
-------
audio    : loading, resampling, alignment helpers (working rate 16 kHz)
schemes  : fingerprint (zero-watermark feature) extractors and attack-aware bit selection
attacks  : classical signal-processing attacks, neural codecs (EnCodec, DAC), kNN-VC
metrics  : BER, NC, threshold/FPR analysis, identification accuracy
"""
SR = 16000
__version__ = "0.1.0"
