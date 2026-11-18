"""Two checks behind the DWT-DCT explanation in Section 5.1.

1. BER of each handcrafted family when the clip is delayed by a few samples (silence in front, end cut).
2. Lag between each codec output and its original, from the cross-correlation peak (median over test clips).

Writes results/table_sample_shift.csv and results/codec_lag.csv.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm.audio import load_wav
from zwm.schemes import BASELINES

data, cache, out = Path("data"), Path("cache"), Path("results")
rows = list(csv.DictReader(open(data / "manifest.csv")))
test = [r["clip_id"] for r in rows if r["split"] == "test"]
fams = ["DWT-DCT", "Philips", "MFCC"]
clean = {c: load_wav(str(data / "clips" / f"{c}.wav")) for c in test}
F0 = {s: np.stack([BASELINES[s](clean[c]) for c in test]) for s in fams}
res = []
for k in [1, 2, 4, 8, 16, 32]:
    for s in fams:
        Fk = np.stack([BASELINES[s](np.concatenate([np.zeros(k, np.float32), clean[c][:-k]])) for c in test])
        res.append({"samples": k, "ms": k / 16, "scheme": s, "ber_mean": float((Fk != F0[s]).mean())})
pd.DataFrame(res).to_csv(out / "table_sample_shift.csv", index=False)
lag = []
for a in ["mp3_32k", "opus_16k", "opus_6k", "encodec_24k", "encodec_6k", "encodec_1.5k", "dac_12cb", "dac_3cb"]:
    L = []
    for c in test:
        x, y = clean[c], load_wav(str(cache / a / f"{c}.wav"))
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        X, Y = np.fft.rfft(x, 2 * n), np.fft.rfft(y, 2 * n)
        cc = np.fft.irfft(X.conj() * Y)
        cc = np.concatenate([cc[-400:], cc[:401]])          # lags -400..+400 samples
        L.append(int(np.argmax(np.abs(cc))) - 400)
    lag.append({"attack": a, "lag_median": float(np.median(L)), "lag_abs_max": int(np.max(np.abs(L)))})
pd.DataFrame(lag).to_csv(out / "codec_lag.csv", index=False)
print(pd.DataFrame(res).pivot(index="samples", columns="scheme", values="ber_mean").round(3).to_string())
print(pd.DataFrame(lag).to_string())
