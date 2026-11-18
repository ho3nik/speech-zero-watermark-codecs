"""Offset-search verifier (desynchronization defence), non-circular.

Every extractor cuts the clip into fixed segments aligned to sample 0, so a small time shift moves every
segment boundary. The search verifier extracts the fingerprint of the received clip at every circular offset
in [-100, +100] ms (step 10 ms; 20 ms for WavLM-LSH) and takes the minimum BER against the registered
fingerprint. To keep the false-accept rate honest, the threshold is set with the same search on impostor
pairs of clean clips (an attacked clip is also compared, with search, against every other registered clip).

Outputs results/table_search.csv: for each scheme and attack, plain and searched BER, the searched
threshold at 1% FAR, TPR with search, and the false-accept rate under attack with search.

Usage: python scripts/run_search.py --schemes classical            (DWT-DCT, Philips, MFCC, SEL-codec; all attacks)
       python scripts/run_search.py --schemes classical --groups desync
       python scripts/run_search.py --schemes wavlm --groups desync
           (WavLM-LSH: five desynchronization attacks, +-60 ms on a 20 ms grid, 0.4 s per extraction)
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import SR
from zwm.audio import load_wav
from zwm.schemes import BASELINES, CandidatePool
from zwm.attacks import ATTACK_GROUP
from zwm import metrics as M

FPR = 0.01
ROOT = Path(__file__).resolve().parents[1]
_pool = None
_wavlm = None


def _realign(y: np.ndarray, o_ms: int) -> np.ndarray:
    """Undo a delay of o_ms (o > 0: drop the first o ms and pad zeros at the end; o < 0: prepend zeros)."""
    n = int(round(abs(o_ms) * SR / 1000))
    if n == 0:
        return y
    if o_ms > 0:
        return np.concatenate([y[n:], np.zeros(n, np.float32)])
    return np.concatenate([np.zeros(n, np.float32), y[:len(y) - n]])


def _wavlm_search(y: np.ndarray, offsets_ms):
    """WavLM-LSH searched in the feature domain: the model runs once and the 16 segment windows slide over
    its 20 ms frames, so each offset is a multiple of 20 ms (frames outside the clip are left out of a window)."""
    f = BASELINES["WavLM-LSH"]
    h = f.features(y)
    T = len(h)
    edges = np.linspace(0, T, f.n_seg + 1).astype(int)
    out = []
    for o in offsets_ms:
        sft = int(round(o / 20))
        S = []
        for a, b in zip(edges[:-1], edges[1:]):
            lo, hi = max(0, a + sft), min(T, b + sft)
            S.append(h[lo:hi].mean(axis=0) if hi > lo else np.zeros(h.shape[1], np.float32))
        S = np.stack(S); S = S - S.mean(axis=0, keepdims=True)
        out.append((S @ f.W > 0).astype(np.uint8).ravel())
    return out


def _extract(args):
    """Fingerprints of one clip at every offset: dict scheme -> (n_offsets, 1024)."""
    path, offsets_ms, schemes, sel_idx = args
    global _pool
    y = load_wav(path)
    out = {s: [] for s in schemes}
    if "WavLM-LSH" in schemes:
        out["WavLM-LSH"] = _wavlm_search(y, offsets_ms)
    for o in offsets_ms:
        if not any(s in schemes for s in ("SEL-codec", "DWT-DCT", "Philips", "MFCC")):
            break
        z = _realign(y, o)
        if "SEL-codec" in schemes:
            if _pool is None:
                _pool = CandidatePool()
            out["SEL-codec"].append(_pool(z)[sel_idx])
        for s in ("DWT-DCT", "Philips", "MFCC"):
            if s in schemes:
                out[s].append(BASELINES[s](z))
    return {s: np.stack(v).astype(np.uint8) for s, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="results")
    ap.add_argument("--schemes", choices=["classical", "wavlm"], default="classical")
    ap.add_argument("--groups", nargs="*", default=None, help="attack groups to run (default: all cached)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--attacks", nargs="*", default=None, help="explicit attack list (overrides --groups)")
    ap.add_argument("--max-offset", type=int, default=None, help="search range in ms (default 600)")
    args = ap.parse_args()
    data, cache, out = Path(args.data), Path(args.cache), Path(args.out)
    rows = list(csv.DictReader(open(data / "manifest.csv")))
    test = [r["clip_id"] for r in rows if r["split"] == "test"]
    speakers = np.array([int(r["speaker"]) for r in rows if r["split"] == "test"])
    if args.schemes == "classical":
        mo = args.max_offset or 600
        schemes, offsets = ["DWT-DCT", "Philips", "MFCC"], list(range(-mo, mo + 1, 10))
    else:
        mo = args.max_offset or 600
        schemes, offsets = ["WavLM-LSH"], list(range(-mo, mo + 1, 20))
    sel_idx = np.load(out / "selected_idx.npz")["SEL-codec"]
    attacks = [a for a in ATTACK_GROUP if (cache / a).is_dir() and all((cache / a / f"{c}.wav").exists() for c in test)]
    if args.groups:
        attacks = [a for a in attacks if ATTACK_GROUP[a] in args.groups]
    if args.attacks:
        attacks = [a for a in attacks if a in args.attacks]
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    log(f"schemes {schemes}, {len(offsets)} offsets, attacks {attacks}")

    def fps(attack):
        paths = [str(data / "clips" / f"{c}.wav") if attack == "clean" else str(cache / attack / f"{c}.wav") for c in test]
        jobs = [(p, offsets, schemes, sel_idx) for p in paths]
        if args.workers > 1 and args.schemes == "classical":
            with Pool(args.workers) as pool:
                res = pool.map(_extract, jobs, chunksize=4)
        else:
            res = [_extract(j) for j in jobs]
        return {s: np.stack([r[s] for r in res], axis=1) for s in schemes}     # (n_off, n_clips, 1024)

    t0 = time.time()
    clean = fps("clean"); log(f"clean done {time.time() - t0:.0f}s")
    i0 = offsets.index(0)
    table = []
    thresholds = {}
    for s in schemes:
        F0 = clean[s][i0]                                                        # registered fingerprints (no offset)
        # impostor BER with search: min over offsets of BER(registered i, clean j at offset o), i != j
        Mo = np.stack([M.pairwise_ber_matrix(F0, clean[s][k]) for k in range(len(offsets))]).min(axis=0)
        iu = np.triu_indices(len(test), k=1)
        imp_search = np.concatenate([Mo[iu], Mo.T[iu]])
        thresholds[s] = {"tau_search": M.threshold_at_fpr(imp_search, FPR), "impostor_search_mean": float(imp_search.mean())}
    log(f"thresholds {thresholds}")
    for a in attacks:
        t1 = time.time()
        att = fps(a)
        for s in schemes:
            F0 = clean[s][i0]
            Ms = np.stack([M.pairwise_ber_matrix(F0, att[s][k]) for k in range(len(offsets))])   # (n_off, reg i, received j)
            Mmin = Ms.min(axis=0)
            genuine_plain = np.diag(Ms[i0]); genuine_search = np.diag(Mmin)
            off = np.ones_like(Mmin, bool); np.fill_diagonal(off, False)
            tau = thresholds[s]["tau_search"]
            table.append({"scheme": s, "attack": a, "group": ATTACK_GROUP[a], "n_clips": len(test), "n_offsets": len(offsets),
                          "ber_plain_mean": float(genuine_plain.mean()), "ber_search_mean": float(genuine_search.mean()),
                          "tau_search": tau, "tpr_search_at_1pct_far": float(np.mean(genuine_search <= tau)),
                          "far_attack_search": float(np.mean(Mmin[off] <= tau)),
                          "best_offset_ms_median": float(np.median([offsets[k] for k in Ms[:, np.arange(len(test)), np.arange(len(test))].argmin(axis=0)]))})
        log(f"{a} done {time.time() - t1:.0f}s")
        pd.DataFrame(table).to_csv(out / f"table_search_{args.schemes}.csv", index=False)
    print(pd.DataFrame(table).pivot(index="attack", columns="scheme", values="ber_search_mean").round(3).to_string())


if __name__ == "__main__":
    main()
