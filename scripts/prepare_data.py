"""Build the clip set from LibriSpeech dev-clean.

Selects ``--per-speaker`` utterances from each of the 40 dev-clean speakers, crops a
5 s segment from each, and writes 16 kHz WAVs plus a manifest. Speakers are split
alternately (sorted by id) into train (bit selection) and test (evaluation), so the
two splits share no speaker. Remaining utterances of each speaker are listed as
voice-conversion matching material.

Usage: python scripts/prepare_data.py --librispeech /path/LibriSpeech --out data
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import SR
from zwm.audio import load_wav, crop_segment, save_wav, peak_normalize

CLIP_SECONDS = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-speaker", type=int, default=4)
    ap.add_argument("--vc-matching-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.librispeech) / "dev-clean"
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "vc_matching").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    speakers = sorted(int(p.name) for p in root.iterdir() if p.is_dir())
    assert len(speakers) == 40, f"expected 40 dev-clean speakers, found {len(speakers)}"
    split = {s: ("train" if i % 2 == 0 else "test") for i, s in enumerate(speakers)}

    rows = []
    for spk in speakers:
        flacs = sorted((root / str(spk)).rglob("*.flac"))
        rng.shuffle(flacs)
        chosen, used = [], 0
        # clips: need at least CLIP_SECONDS of audio, take the middle 5 s
        for f in flacs:
            if len(chosen) >= args.per_speaker:
                break
            y = load_wav(str(f))
            if len(y) < CLIP_SECONDS * SR * 1.2:
                continue
            seg = crop_segment(y, CLIP_SECONDS, SR, offset_frac=0.5)
            if np.sqrt(np.mean(seg ** 2)) < 0.01:  # skip near-silent segments
                continue
            seg = peak_normalize(seg)
            cid = f"{spk}_{f.stem}"
            save_wav(str(out / "clips" / f"{cid}.wav"), seg)
            chosen.append(f)
            rows.append({"clip_id": cid, "speaker": spk, "split": split[spk], "source": str(f.relative_to(root))})
        assert len(chosen) == args.per_speaker, f"speaker {spk}: only {len(chosen)} usable clips"
        # matching material for voice conversion: other utterances, concatenated up to N seconds
        rest = [f for f in flacs if f not in chosen]
        buf, total = [], 0.0
        for f in rest:
            y = load_wav(str(f))
            buf.append(y)
            total += len(y) / SR
            if total >= args.vc_matching_seconds:
                break
        save_wav(str(out / "vc_matching" / f"{spk}.wav"), np.concatenate(buf))

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "speaker", "split", "source"])
        w.writeheader()
        w.writerows(rows)
    n_train = sum(r["split"] == "train" for r in rows)
    print(f"wrote {len(rows)} clips ({n_train} train / {len(rows) - n_train} test) to {out}")


if __name__ == "__main__":
    main()
