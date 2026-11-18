"""Offset-search verifier at a 0.01% false-accept rate.

Impostors: the first N_EXT external clips (LibriSpeech test-clean/test-other, cut as in build_impostors.py) are
searched over the same offsets as genuine copies (+-600 ms; 10 ms steps for the handcrafted families, 20 ms for
WavLM-LSH) against every registered test clip; the threshold is the 0.01% quantile of these searched impostor BERs.
Genuine: the 80 original test clips under the nine misalignment attacks, searched the same way.

Writes results/table_search_far.csv.
Usage: python scripts/run_search_far.py --schemes classical|wavlm --librispeech /path/to/LibriSpeech
"""
from __future__ import annotations

import argparse, csv, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zwm import SR
from zwm import metrics as M
from zwm.audio import load_wav, crop_segment, peak_normalize
from run_search import _extract

N_EXT = 1000
N_EXT_CLASSICAL = 600   # fewer for the slower 121-offset search of the handcrafted families
DESYNC = ["delay_7ms", "delay_23ms", "delay_37ms", "delay_250ms", "crop_10pct", "stretch_+2pct", "stretch_-2pct", "stretch_+5pct", "stretch_-5pct"]


def ext_clips(root: Path, n: int):
    files = []
    for sub in ["test-clean", "test-other"]:
        files += sorted((root / sub).rglob("*.flac"))
    for f in files:
        y = load_wav(str(f))
        if len(y) < 5.0 * SR * 1.2:
            continue
        seg = crop_segment(y, 5.0, SR, offset_frac=0.5)
        if np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue
        yield peak_normalize(seg)
        n -= 1
        if n == 0:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", choices=["classical", "wavlm"], required=True)
    ap.add_argument("--librispeech", required=True)
    args = ap.parse_args()
    schemes = ["DWT-DCT", "Philips", "MFCC"] if args.schemes == "classical" else ["WavLM-LSH"]
    offsets = list(range(-600, 601, 10)) if args.schemes == "classical" else list(range(-600, 601, 20))
    i0 = offsets.index(0)
    rows = list(csv.DictReader(open("data/manifest.csv")))
    test_all = [r["clip_id"] for r in rows if r["split"] == "test"]
    orig = [r["clip_id"] for r in csv.DictReader(open("data/manifest_80.csv")) if r["split"] == "test"]
    t0 = time.time()
    import tempfile
    from zwm.audio import save_wav
    tmp = Path(tempfile.mkdtemp())

    def fp(y_or_path):
        if isinstance(y_or_path, str):
            return _extract((y_or_path, offsets, schemes, None))
        p = tmp / "x.wav"; save_wav(str(p), y_or_path)
        return _extract((str(p), offsets, schemes, None))

    # registrations: offset-0 fingerprints of all test clips
    from zwm.schemes import BASELINES
    REG = {s: np.stack([BASELINES[s](load_wav(f"data/clips/{c}.wav")) for c in test_all]).astype(np.uint8) for s in schemes}
    print(f"registered {len(test_all)} {time.time() - t0:.0f}s", flush=True)
    cache_f = Path(f"results/search_far_imp_{args.schemes}.npz")
    if cache_f.exists():
        zc = np.load(cache_f); imp = {s: [zc[s]] for s in schemes}
        print("loaded cached impostor search", flush=True)
    else:
        partial = Path(f"results/search_far_imp_{args.schemes}_partial.npz")
        imp = {s: [] for s in schemes}; k0 = 0
        if partial.exists():
            zp = np.load(partial); k0 = int(zp["k"]); imp = {s: list(zp[s]) for s in schemes}
            print("resuming at external clip", k0, flush=True)
        n_ext = N_EXT_CLASSICAL if args.schemes == "classical" else N_EXT
        for k, y in enumerate(ext_clips(Path(args.librispeech), n_ext)):
            if k < k0:
                continue
            F = fp(y)
            for s in schemes:
                imp[s].append(M.pairwise_ber_matrix(REG[s], F[s]).min(axis=1))      # (n_reg,) best offset per registration
            if k % 25 == 24:
                np.savez(partial, k=k + 1, **{s: np.stack(imp[s]) for s in schemes})
                print("ext", k + 1, f"{time.time() - t0:.0f}s", flush=True)
        np.savez(cache_f, **{s: np.concatenate(imp[s]) for s in schemes})
    tau = {s: M.threshold_at_fpr(np.concatenate(imp[s]), 1e-4) for s in schemes}
    n_pairs = {s: int(np.concatenate(imp[s]).size) for s in schemes}
    print("tau", tau, flush=True)
    idx = {c: i for i, c in enumerate(test_all)}
    part = Path(f"results/search_far_part_{args.schemes}.csv")
    out = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {r["attack"] for r in out}
    for a in DESYNC:
        if a in done:
            continue
        g = {s: [] for s in schemes}
        for c in orig:
            F = fp(f"cache/{a}/{c}.wav")
            for s in schemes:
                g[s].append(float(np.mean(F[s] != REG[s][idx[c]], axis=1).min()))
        for s in schemes:
            gs = np.array(g[s])
            out.append({"scheme": s, "attack": a, "n_clips": len(gs), "n_impostor_pairs": n_pairs[s], "tau_search_1e-4": tau[s],
                        "ber_search_mean": float(gs.mean()), "tpr_search_at_1e-4": float(np.mean(gs <= tau[s]))})
        print(a, {s: round(out[-len(schemes) + i]["tpr_search_at_1e-4"], 3) for i, s in enumerate(schemes)}, f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(out).to_csv(part, index=False)
    pd.DataFrame(out).to_csv(f"results/table_search_far_{args.schemes}.csv", index=False)
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
