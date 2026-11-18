"""Choose the RSZW SVM hyper-parameters on TRAINING-split clips only: for each (C, gamma), train one SVM per clip on
its MP3 32 kbps copy and measure the logo BER on its MP3 128 kbps copy (an unseen setting of a seen attack kind),
as the paper's results imply the SVMs must generalise across attack parameters. Writes results/salayani_tuning.csv."""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import salayani as SZ
from zwm.audio import load_wav
from sklearn.svm import SVC
ids = [r["clip_id"] for r in csv.DictReader(open("data/manifest.csv")) if r["split"] == "train"][:20]
key = SZ.selection_key(); rows = []
data = []
for k, c in enumerate(ids):
    logo = np.random.default_rng(5000 + k).integers(0, 2, 1024).astype(np.uint8)
    adw = SZ.pattern(SZ.features(load_wav(f"data/clips/{c}.wav"), key)).ravel() ^ logo
    data.append((logo, adw, SZ.features(load_wav(f"cache/mp3_32k/{c}.wav"), key), SZ.features(load_wav(f"cache/mp3_128k/{c}.wav"), key)))
for C in [1.0, 1e2, 1e3, 1e4, 1e6]:
    for g in ["scale", 10.0, 50.0]:
        b = []
        for logo, adw, Ftr, Fte in data:
            s = SVC(kernel="rbf", C=C, gamma=g).fit(SZ.windows(Ftr), adw)
            b.append(np.mean((s.predict(SZ.windows(Fte)).astype(np.uint8) ^ SZ.pattern(Fte).ravel()) != logo))
        rows.append({"C": C, "gamma": g, "ber_mp3_128k": float(np.mean(b))}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/salayani_tuning.csv", index=False)
