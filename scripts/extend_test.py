"""Extend the test split from 4 to 20 clips per test speaker (400 test clips), with the same selection rule as
prepare_data.py (same shuffled order, so the original 4 clips stay the first 4). The training split is unchanged,
so bit selection never sees a test speaker. Voice-conversion matching material of the test speakers is rebuilt
from utterances that are NOT among the 20 test clips (the original material could contain new test utterances).

Usage: python scripts/extend_test.py --librispeech /path/to/LibriSpeech --per-speaker 20
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm import SR
from zwm.audio import load_wav, save_wav, crop_segment, peak_normalize

CLIP_SECONDS = 5.0
ap = argparse.ArgumentParser()
ap.add_argument("--librispeech", required=True); ap.add_argument("--data", default="data")
ap.add_argument("--per-speaker", type=int, default=20); ap.add_argument("--vc-matching-seconds", type=float, default=120.0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
root, out = Path(args.librispeech) / "dev-clean", Path(args.data)
rows = list(csv.DictReader(open(out / "manifest.csv")))
have = {}
for r in rows:
    have.setdefault(int(r["speaker"]), []).append(r["source"])
rng = np.random.default_rng(args.seed)
speakers = sorted(int(p.name) for p in root.iterdir() if p.is_dir())
split = {s: ("train" if i % 2 == 0 else "test") for i, s in enumerate(speakers)}
new_rows = []
for spk in speakers:
    flacs = sorted((root / str(spk)).rglob("*.flac"))
    rng.shuffle(flacs)                                   # same generator sequence as prepare_data.py
    if split[spk] != "test":
        continue
    chosen = [root / s for s in have[spk]]
    for f in flacs:
        if len(chosen) >= args.per_speaker:
            break
        if f in chosen:
            continue
        y = load_wav(str(f))
        if len(y) < CLIP_SECONDS * SR * 1.2:
            continue
        seg = crop_segment(y, CLIP_SECONDS, SR, offset_frac=0.5)
        if np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue
        cid = f"{spk}_{f.stem}"
        save_wav(str(out / "clips" / f"{cid}.wav"), peak_normalize(seg))
        chosen.append(f)
        new_rows.append({"clip_id": cid, "speaker": spk, "split": "test", "source": str(f.relative_to(root))})
    assert len(chosen) >= 16, (spk, len(chosen))
    rest = [f for f in flacs if f not in chosen]
    buf, total = [], 0.0
    for f in rest:
        y = load_wav(str(f)); buf.append(y); total += len(y) / SR
        if total >= args.vc_matching_seconds:
            break
    save_wav(str(out / "vc_matching" / f"{spk}.wav"), np.concatenate(buf))
    print(spk, len(chosen), f"matching {total:.0f}s", flush=True)
with open(out / "manifest.csv", "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["clip_id", "speaker", "split", "source"])
    for r in new_rows:
        w.writerow(r)
print("added", len(new_rows))
