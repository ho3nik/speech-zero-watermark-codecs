"""Large impostor set: 5 s clips from LibriSpeech test-clean and test-other (speakers disjoint from the
dev-clean clips of the main experiment), cut and normalized exactly as prepare_data.py cuts the main clips,
fingerprinted clean with every family. Used to estimate false-accept rates down to 1e-4.

Writes results/impostor_fingerprints.npz with one (N, bits) array per family and the clip ids.
Usage: python scripts/build_impostors.py --librispeech /path/LibriSpeech --out results
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import SR
from zwm.audio import load_wav, crop_segment, peak_normalize
from zwm.schemes import BASELINES, CandidatePool

CLIP_SECONDS = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True)
    ap.add_argument("--subsets", nargs="+", default=["test-clean", "test-other"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    files = []
    for sub in args.subsets:
        files += sorted((Path(args.librispeech) / sub).rglob("*.flac"))
    pool = CandidatePool()
    ext = {**BASELINES, "POOL": pool}
    store = {k: [] for k in ext}
    ids = []
    t0 = time.time()
    for n, f in enumerate(files):
        y = load_wav(str(f))
        if len(y) < CLIP_SECONDS * SR * 1.2:
            continue
        seg = crop_segment(y, CLIP_SECONDS, SR, offset_frac=0.5)
        if np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue
        seg = peak_normalize(seg)
        for k, fn in ext.items():
            store[k].append(fn(seg).astype(np.uint8))
        ids.append(f.stem)
        if len(ids) % 250 == 0:
            print(f"{len(ids)} clips ({n + 1}/{len(files)} files) {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(Path(args.out) / "impostor_fingerprints.npz", ids=np.array(ids),
                        **{k: np.stack(v) for k, v in store.items()})
    print("done", len(ids), "clips", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
