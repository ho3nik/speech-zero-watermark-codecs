"""Generate attacked versions of every clip and cache them as WAV files.

cache/<attack>/<clip_id>.wav. Re-running skips files that already exist.

Groups:
  classical  all clips
  encodec    all clips
  dac        test clips (held-out attack family)
  vc         test clips (kNN-VC self and cross-gender)

Usage: python scripts/run_attacks.py --data data --cache cache --groups classical encodec dac vc
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import SR
from zwm.audio import load_wav, save_wav
from zwm import attacks as A


def read_manifest(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def speaker_genders(librispeech_root: Path) -> dict:
    g = {}
    with open(librispeech_root / "SPEAKERS.TXT") as fh:
        for line in fh:
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[2] == "dev-clean":
                g[int(parts[0])] = parts[1]
    return g


def run_group(name_to_fn, rows, data: Path, cache: Path, log):
    for aname, fn in name_to_fn.items():
        d = cache / aname
        d.mkdir(parents=True, exist_ok=True)
        t0, n = time.time(), 0
        for r in rows:
            out = d / f"{r['clip_id']}.wav"
            if out.exists():
                continue
            y = load_wav(str(data / "clips" / f"{r['clip_id']}.wav"))
            z = fn(y)
            assert len(z) == len(y), (aname, r["clip_id"], len(z), len(y))
            save_wav(str(out), z)
            n += 1
        log(f"{aname}: {n} new files, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--librispeech", required=True)
    ap.add_argument("--groups", nargs="+", default=["classical", "opus", "desync", "encodec", "dac", "vc"])
    ap.add_argument("--vc-matching-seconds", type=float, default=90.0)
    args = ap.parse_args()
    data, cache = Path(args.data), Path(args.cache)
    rows = read_manifest(data / "manifest.csv")
    test_rows = [r for r in rows if r["split"] == "test"]
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)

    if "classical" in args.groups:
        run_group(A.CLASSICAL, rows, data, cache, log)
    if "opus" in args.groups:
        run_group(A.OPUS, rows, data, cache, log)
    if "desync" in args.groups:
        run_group(A.DESYNC, rows, data, cache, log)
    if "encodec" in args.groups:
        run_group(A.ENCODEC, rows, data, cache, log)
    if "dac" in args.groups:
        run_group(A.DAC, test_rows, data, cache, log)
    if "vc" in args.groups:
        genders = speaker_genders(Path(args.librispeech))
        train_spk = sorted({int(r["speaker"]) for r in rows if r["split"] == "train"})
        # fixed cross-gender targets drawn from the train split (never from test speakers)
        target_f = [s for s in train_spk if genders[s] == "F"][0]
        target_m = [s for s in train_spk if genders[s] == "M"][0]
        log(f"cross-VC targets: female {target_f}, male {target_m}")
        n_match = int(args.vc_matching_seconds * SR)
        matching = lambda spk: [load_wav(str(data / "vc_matching" / f"{spk}.wav"))[:n_match]]
        cross = {"F": A.KNNVCAttack(matching(target_m)), "M": A.KNNVCAttack(matching(target_f))}
        (cache / "knnvc_cross").mkdir(parents=True, exist_ok=True)
        (cache / "knnvc_self").mkdir(parents=True, exist_ok=True)
        by_spk = {}
        for r in test_rows:
            by_spk.setdefault(int(r["speaker"]), []).append(r)
        for spk, rs in by_spk.items():
            t0 = time.time()
            self_att = None
            for r in rs:
                y = load_wav(str(data / "clips" / f"{r['clip_id']}.wav"))
                out_c = cache / "knnvc_cross" / f"{r['clip_id']}.wav"
                if not out_c.exists():
                    save_wav(str(out_c), cross[genders[spk]](y))
                out_s = cache / "knnvc_self" / f"{r['clip_id']}.wav"
                if not out_s.exists():
                    if self_att is None:
                        self_att = A.KNNVCAttack(matching(spk))
                    save_wav(str(out_s), self_att(y))
            log(f"vc speaker {spk} ({genders[spk]}): {len(rs)} clips, {time.time() - t0:.0f}s")
    log("done")


if __name__ == "__main__":
    main()
