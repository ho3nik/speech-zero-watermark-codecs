"""Same-sentence impostors: do the fingerprints accept the same text read by a different speaker?

Seven CMU ARCTIC speakers (bdl, rms, jmk, awb, ksp male; slt, clb female) read the same sentences. For each of N_TEXTS pairs of consecutive
sentences, every speaker's two recordings are trimmed of leading and trailing silence, joined with 100 ms of silence,
cut or padded to 5 s and peak-normalized, as for the LibriSpeech clips. Fingerprints of different speakers reading the
same pair are compared (same-text impostors), and so are different speakers reading different pairs (baseline).
Each pair is judged against the thresholds set on LibriSpeech: 1% FAR on the test impostors (summary.json) and 0.01%
FAR on the external impostors (table_far.csv).

Writes results/table_same_text.csv.
Usage: python scripts/run_same_text.py --arctic /path/to/arctic
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import SR
from zwm import chenzhu as CZ
from zwm.audio import load_wav, peak_normalize
from zwm.schemes import BASELINES, CandidatePool

SPEAKERS = ["bdl", "rms", "jmk", "awb", "ksp", "slt", "clb"]
N_TEXTS = 100


def trim(y, top_db=35.0, frame=400):
    """Drop leading and trailing frames more than top_db below the loudest frame."""
    e = np.array([np.sum(y[i:i + frame] ** 2) for i in range(0, len(y) - frame, frame // 2)]) + 1e-12
    keep = np.where(10 * np.log10(e / e.max()) > -top_db)[0]
    return y[keep[0] * frame // 2: keep[-1] * frame // 2 + frame] if len(keep) else y


def clip(paths):
    gap = np.zeros(int(0.1 * SR), np.float32)
    y = np.concatenate([np.concatenate([trim(load_wav(str(p))), gap]) for p in paths])
    n = int(5 * SR)
    y = y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y), np.float32)])
    return peak_normalize(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arctic", required=True); ap.add_argument("--out", default="results")
    args = ap.parse_args()
    root, out = Path(args.arctic), Path(args.out)
    ids = sorted(p.stem for p in (root / f"cmu_us_{SPEAKERS[0]}_arctic" / "wav").glob("arctic_a*.wav"))
    texts = [(ids[2 * k], ids[2 * k + 1]) for k in range(N_TEXTS)]
    sel = np.load(out / "selected_idx.npz")["SEL-codec"]
    pool = CandidatePool()
    schemes = ["DWT-DCT", "Philips", "MFCC", "WavLM-LSH", "SEL-codec", "Chen-Zhu"]
    F = {s: {} for s in schemes}
    czf = {}
    for t, (u1, u2) in enumerate(texts):
        for spk in SPEAKERS:
            y = clip([root / f"cmu_us_{spk}_arctic" / "wav" / f"{u}.wav" for u in (u1, u2)])
            for s in ["DWT-DCT", "Philips", "MFCC", "WavLM-LSH"]:
                F[s][(t, spk)] = BASELINES[s](y).astype(np.uint8)
            F["SEL-codec"][(t, spk)] = pool(y)[sel].astype(np.uint8)
            czf[(t, spk)] = CZ.features(y)
        if t % 20 == 0:
            print("text", t, flush=True)
    keys = {k: CZ.make_key(*v) for k, v in czf.items()}
    summ = json.load(open(out / "summary.json"))
    far = pd.read_csv(out / "table_far.csv")
    cz_tau = json.load(open(out / "summary_cz.json"))["thresholds"]["Chen-Zhu"] if (out / "summary_cz.json").exists() else None
    cz_tau4 = float(pd.read_csv(out / "table_far_cz.csv")["tau_1e-4"].iloc[0]) if (out / "table_far_cz.csv").exists() else None

    def ber(s, a, b):
        if s == "Chen-Zhu":                         # read clip b with the key of registration a
            return float(np.mean(CZ.bits(czf[b][1], keys[a]) != CZ.bits(czf[a][1], keys[a])))
        return float(np.mean(F[s][a] != F[s][b]))

    clips = [(t, spk) for t in range(N_TEXTS) for spk in SPEAKERS]
    idx = {c: i for i, c in enumerate(clips)}
    T = np.array([c[0] for c in clips]); S = np.array([SPEAKERS.index(c[1]) for c in clips])
    iu = np.triu_indices(len(clips), k=1)
    same_text = (T[iu[0]] == T[iu[1]]) & (S[iu[0]] != S[iu[1]])
    diff_text = (T[iu[0]] != T[iu[1]]) & (S[iu[0]] != S[iu[1]])
    pair_id = np.minimum(S[iu[0]], S[iu[1]]) * len(SPEAKERS) + np.maximum(S[iu[0]], S[iu[1]])
    rng = np.random.default_rng(0)
    rows = []
    for s in schemes:
        if s == "Chen-Zhu":                         # keyed: read clip j with the key of clip i (i < j)
            C = np.stack([czf[c][1] for c in clips])
            REG = np.stack([CZ.bits(C[i], keys[clips[i]]) for i in range(len(clips))])
            Mx = np.stack([(CZ.bits_many(C, keys[clips[i]]) != REG[i]).mean(axis=1) for i in range(len(clips))])
        else:
            Fm = np.stack([F[s][c] for c in clips]).astype(np.float32)
            Mx = (Fm @ (1 - Fm).T + (1 - Fm) @ Fm.T) / Fm.shape[1]
        b = Mx[iu]
        tau1 = cz_tau if s == "Chen-Zhu" else summ["thresholds"][s]
        tau4 = cz_tau4 if s == "Chen-Zhu" else float(far[far.scheme == s]["tau_1e-4"].iloc[0])
        tau_a = float(np.quantile(b[diff_text], 1e-3, method="lower"))        # 0.1% FAR on ARCTIC different-text pairs
        same, diff = b[same_text], b[diff_text]
        acc = (same <= tau_a).astype(float)
        pids = pair_id[same_text]; up = np.unique(pids)
        per = np.array([acc[pids == p].mean() for p in up])
        boot = [per[rng.integers(0, len(up), len(up))].mean() for _ in range(2000)]
        rows.append({"scheme": s, "n_same_text_pairs": int(same.size), "n_diff_text_pairs": int(diff.size), "n_speaker_pairs": int(len(up)),
                     "ber_same_text": float(same.mean()), "ber_diff_text": float(diff.mean()),
                     "accept_1e-4_same_text": float(np.mean(same <= tau4)), "accept_1e-4_diff_text": float(np.mean(diff <= tau4)),
                     "tau_arctic_1e-3": tau_a, "accept_arctic_1e-3_same_text": float(acc.mean()),
                     "accept_arctic_1e-3_lo": float(np.quantile(boot, 0.025)), "accept_arctic_1e-3_hi": float(np.quantile(boot, 0.975)),
                     "tau_1pct": tau1, "tau_1e-4": tau4})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(out / "table_same_text.csv", index=False)


if __name__ == "__main__":
    main()
