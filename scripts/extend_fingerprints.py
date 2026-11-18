"""Extend results/fingerprints.npz (computed on the 160-clip manifest) to the 478-clip manifest by computing
fingerprints only for the clips added by extend_test.py. Rows keep the manifest order (old rows first).

Run in parallel shards over attacks, then merge:
  python scripts/extend_fingerprints.py --shard 0 --nshards 2   (writes results/fp_ext_0.npz)
  python scripts/extend_fingerprints.py --shard 1 --nshards 2
  python scripts/extend_fingerprints.py --merge 2
"""
from __future__ import annotations

import argparse, csv, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm.audio import load_wav
from zwm.schemes import BASELINES, CandidatePool

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--merge", nargs="*", default=None); ap.add_argument("--attacks", nargs="*", default=None)
ap.add_argument("--tag", default=None)
args = ap.parse_args()
res = Path("results")
old_n = len(list(csv.DictReader(open("data/manifest_80.csv"))))
rows = list(csv.DictReader(open("data/manifest.csv")))
new = rows[old_n:]
assert all(r["split"] == "test" for r in new)

if args.merge is not None:
    z = np.load(res / "fingerprints.npz", allow_pickle=True)
    bits = {k: z[k] for k in z.files}
    for tag in args.merge:
        e = np.load(res / f"fp_ext_{tag}.npz")
        for k in e.files:
            if len(e[k]) == len(rows):
                bits[k] = e[k]
            else:
                assert k in bits and len(bits[k]) == old_n, k
                bits[k] = np.concatenate([bits[k], e[k]])
    stale = [k for k, v in bits.items() if "|" in k and len(v) != len(rows)]
    print("dropping keys not extended (no longer used):", sorted({k.split("|")[1] for k in stale}))
    for k in stale:
        del bits[k]
    np.savez_compressed(res / "fingerprints.npz", **bits)
    print("merged", len(bits), "keys", len(rows), "rows")
    sys.exit()

z = np.load(res / "fingerprints.npz", allow_pickle=True)
keys = [k for k in z.files if "|" in k]
attacks = sorted({k.split("|", 1)[1] for k in keys})
if args.attacks:
    attacks = [a for a in args.attacks]
attacks = [a for i, a in enumerate(attacks) if i % args.nshards == args.shard]
pool = CandidatePool()
ext = {**BASELINES, "POOL": pool}
out = {}
t0 = time.time()
for a in attacks:
    fresh = f"WavLM-LSH|{a}" not in z.files            # attack not in the old file: compute every row
    todo = rows if fresh else new
    paths = [f"data/clips/{r['clip_id']}.wav" if a == "clean" else f"cache/{a}/{r['clip_id']}.wav" for r in todo]
    for s, f in ext.items():
        width = z[f"{s}|clean"].shape[1] if f"{s}|clean" in z.files else 1024
        arr = np.full((len(todo), width), 255, np.uint8)
        for i, p in enumerate(paths):
            if Path(p).exists():
                arr[i] = f(load_wav(p))
        out[f"{s}|{a}"] = arr
    print(a, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(res / f"fp_ext_{args.tag or args.shard}.npz", **out)
print("done shard", args.shard)
