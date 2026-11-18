"""Identifier read-out with stronger codes: exact recovery of the identifier from the real per-clip error patterns
of the test clips, for the repetition code (68 bits), BCH(1023, 76) and a polar code (1024, 76) with
successive-cancellation decoding. Also the false-identifier rate: an unrelated clip read against a registration
(error pattern = fingerprint of clip j xor registered fingerprint of clip i) decoding exactly to i's identifier.

Writes results/table_codes.csv.
"""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import codes as C

ATT = ["mp3_32k", "opus_6k", "encodec_24k", "encodec_6k", "encodec_3k", "encodec_1.5k", "dac_12cb", "dac_6cb", "dac_3cb", "knnvc_self", "knnvc_cross", "speecht5_cross"]
CODES = ["rep15", "bch76", "polar76"]
rows = list(csv.DictReader(open("data/manifest.csv")))
test = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
z = np.load("results/fingerprints.npz", allow_pickle=True)
sel = np.load("results/selected_idx.npz")["SEL-codec"]


def bits(s, a):
    if s == "SEL-codec":
        k = f"POOL|{a}"
        return None if k not in z.files else z[k][:, sel]
    k = f"{s}|{a}"
    return None if k not in z.files else z[k]


out = []
rng = np.random.default_rng(0)
for s in ["DWT-DCT", "Philips", "MFCC", "WavLM-LSH", "SEL-codec"]:
    Fc = bits(s, "clean")[test]
    for a in ATT:
        Fa = bits(s, a)
        if Fa is None:
            continue
        Fa = Fa[test]; ok = Fa[:, 0] != 255
        E = (Fc[ok] != Fa[ok]).astype(np.uint8)
        for c in CODES:
            out.append({"scheme": s, "attack": a, "code": c, "n": int(ok.sum()), "exact_recovery": float(C.exact_recovery(E, c).mean())})
    # false identifiers: 4000 random registration/impostor pairs of different speakers
    spk = np.array([int(rows[i]["speaker"]) for i in test])
    i = rng.integers(0, len(test), 6000); j = rng.integers(0, len(test), 6000); keep = spk[i] != spk[j]
    E = (Fc[i[keep]][:4000] != Fc[j[keep]][:4000]).astype(np.uint8)
    for c in CODES:
        out.append({"scheme": s, "attack": "impostor", "code": c, "n": len(E), "exact_recovery": float(C.exact_recovery(E, c).mean())})
    print(s, flush=True)
pd.DataFrame(out).to_csv("results/table_codes.csv", index=False)
print(pd.DataFrame(out).pivot_table(index=["scheme", "attack"], columns="code", values="exact_recovery").round(2).to_string())
