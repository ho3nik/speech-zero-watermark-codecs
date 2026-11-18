"""Extraction time per 5 s clip for each fingerprint family on one CPU thread (median of 20 test clips).
Writes results/timing.csv."""
import csv, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm.audio import load_wav
from zwm.schemes import BASELINES
rows = list(csv.DictReader(open("data/manifest.csv")))
clips = [load_wav(f"data/clips/{r['clip_id']}.wav") for r in rows if r["split"] == "test"][:20]
out = []
for name, f in BASELINES.items():
    f(clips[0])                                     # warm-up (model load)
    t = []
    for y in clips:
        t0 = time.perf_counter(); f(y); t.append(time.perf_counter() - t0)
    out.append({"scheme": name, "ms_median": 1000 * float(np.median(t))})
pd.DataFrame(out).to_csv("results/timing.csv", index=False)
print(pd.DataFrame(out).round(1).to_string())
