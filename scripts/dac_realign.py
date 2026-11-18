"""Intervention for the DAC delay: advance each DAC output by its measured lag (8 samples) and re-evaluate the two
wavelet-based schemes (DWT-DCT and Chen-Zhu). If BER and acceptance recover, the DAC failure of these schemes is an
alignment effect. Writes results/table_dac_realign.csv."""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm.audio import load_wav
from zwm.schemes import BASELINES
from zwm import chenzhu as CZ

ids = [r["clip_id"] for r in csv.DictReader(open("data/manifest.csv")) if r["split"] == "test"]
far = pd.read_csv("results/table_far.csv"); farcz = pd.read_csv("results/table_far_cz.csv")
tau = {"DWT-DCT": float(far[far.scheme == "DWT-DCT"]["tau_1e-4"].iloc[0]), "Chen-Zhu": float(farcz["tau_1e-4"].iloc[0])}
lags = {}
rows = []
clean = {c: load_wav(f"data/clips/{c}.wav") for c in ids}
Fd = {c: BASELINES["DWT-DCT"](clean[c]) for c in ids}
cz = {c: CZ.features(clean[c]) for c in ids}
key = {c: CZ.make_key(*cz[c]) for c in ids}
Fc = {c: CZ.bits(cz[c][1], key[c]) for c in ids}
for a in ["dac_12cb", "dac_6cb", "dac_3cb"]:
    for shift in [0, "lag"]:
        bd, bc, L = [], [], []
        for c in ids:
            x, y = clean[c], load_wav(f"cache/{a}/{c}.wav")
            n = min(len(x), len(y))
            X, Y = np.fft.rfft(x[:n], 2 * n), np.fft.rfft(y[:n], 2 * n)
            cc = np.fft.irfft(X.conj() * Y); cc = np.concatenate([cc[-400:], cc[:401]])
            lag = int(np.argmax(np.abs(cc))) - 400                      # y is late by +lag samples if lag > 0; here DAC lag < 0 means ...
            L.append(lag)
            if shift == "lag" and lag != 0:
                # move y so that it lines up with x (positive lag: drop the first samples; negative: prepend zeros)
                y = np.concatenate([y[lag:], np.zeros(lag, np.float32)]) if lag > 0 else np.concatenate([np.zeros(-lag, np.float32), y[:lag]])
            bd.append(float(np.mean(BASELINES["DWT-DCT"](y) != Fd[c])))
            bc.append(float(np.mean(CZ.bits(CZ.features(y)[1], key[c]) != Fc[c])))
        bd, bc = np.array(bd), np.array(bc)
        rows.append({"attack": a, "realigned": shift == "lag", "lag_median": float(np.median(L)),
                     "ber_dwtdct": float(bd.mean()), "tpr_1e-4_dwtdct": float(np.mean(bd <= tau["DWT-DCT"])),
                     "ber_chenzhu": float(bc.mean()), "tpr_1e-4_chenzhu": float(np.mean(bc <= tau["Chen-Zhu"]))})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/table_dac_realign.csv", index=False)
