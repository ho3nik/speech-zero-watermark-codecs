"""Perceptual quality and intelligibility of the attacked copies: STOI and wideband PESQ of each attacked test clip
against its original (16 kHz). Writes results/table_quality.csv (mean and 10th percentile per attack)."""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
from pesq import pesq
from pystoi import stoi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm.audio import load_wav

ATT = ["mp3_32k", "opus_16k", "opus_6k", "encodec_24k", "encodec_6k", "encodec_1.5k", "dac_12cb", "dac_6cb", "dac_3cb", "knnvc_self", "knnvc_cross"]
ids = [r["clip_id"] for r in csv.DictReader(open("data/manifest_80.csv")) if r["split"] == "test"]
ref = {c: load_wav(f"data/clips/{c}.wav") for c in ids}
rows = []
for a in ATT:
    s, p = [], []
    for c in ids:
        x, y = ref[c], load_wav(f"cache/{a}/{c}.wav")
        n = min(len(x), len(y)); x, y = x[:n].astype(np.float64), y[:n].astype(np.float64)
        s.append(stoi(x, y, 16000, extended=False))
        try:
            p.append(pesq(16000, x, y, "wb"))
        except Exception:
            p.append(np.nan)
    s, p = np.array(s), np.array(p)
    rows.append({"attack": a, "stoi_mean": float(np.mean(s)), "stoi_p10": float(np.quantile(s, 0.1)),
                 "pesq_mean": float(np.nanmean(p)), "pesq_p10": float(np.nanquantile(p, 0.1)), "n": len(ids)})
    print(a, round(rows[-1]["stoi_mean"], 3), round(rows[-1]["pesq_mean"], 2), flush=True)
pd.DataFrame(rows).to_csv("results/table_quality.csv", index=False)
