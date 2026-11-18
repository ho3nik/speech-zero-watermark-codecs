"""Evaluate the reimplemented Chen-Zhu (2008) zero-watermark under every cached attack, with the same measures as
run_eval.py. Because the scheme is keyed (frame and position keys come from the registered clip), every comparison
reads the received clip with the key of the registration it is compared against.

Writes results/table_ber_cz.csv, table_far_cz.csv, table_recovery_cz.csv, per_clip_cz.csv, summary_cz.json and
table_sample_shift_cz.csv; build_paper.py merges them with the main tables.

Usage: python scripts/run_chenzhu.py --librispeech /path/LibriSpeech
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zwm import SR
from zwm import chenzhu as CZ
from zwm import metrics as M
from zwm.attacks import ATTACK_GROUP
from zwm.audio import load_wav, crop_segment, peak_normalize
from run_eval import ATTACK_ORDER, REP_RATES, bootstrap_ci, repetition_recovery

NAME = "Chen-Zhu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--cache", default="cache"); ap.add_argument("--out", default="results")
    ap.add_argument("--librispeech", required=True)
    args = ap.parse_args()
    data, cache, out = Path(args.data), Path(args.cache), Path(args.out)
    t0 = time.time()
    rows = list(csv.DictReader(open(data / "manifest.csv")))
    test = [r for r in rows if r["split"] == "test"]
    ids = [r["clip_id"] for r in test]
    spk = np.array([int(r["speaker"]) for r in test])
    n = len(ids)

    clean = [CZ.features(load_wav(str(data / "clips" / f"{c}.wav"))) for c in ids]
    Cc = np.stack([c for _, c in clean])                                   # (n, F, 64)
    keys = [CZ.make_key(e, c) for e, c in clean]
    REG = np.stack([CZ.bits(Cc[i], keys[i]) for i in range(n)])           # registered bits (n, 1024)
    READ = lambda Cs: np.stack([CZ.bits_many(Cs, keys[i]) for i in range(n)])   # (reg i, clip j, 1024)

    # impostors among the test clips: clip j read with the key of registration i, i != j (ordered pairs)
    Bc = READ(Cc)
    Mc = (Bc != REG[:, None, :]).mean(axis=2)
    off = ~np.eye(n, dtype=bool)
    imp = Mc[off]
    same = (spk[:, None] == spk[None, :]) & off
    tau = M.threshold_at_fpr(imp, 0.01)
    print(f"internal impostors {len(imp)} tau {tau:.3f} {time.time() - t0:.0f}s", flush=True)

    # external impostors: the same cut as build_impostors.py
    files = []
    for sub in ["test-clean", "test-other"]:
        files += sorted((Path(args.librispeech) / sub).rglob("*.flac"))
    ext_e, ext_c = [], []
    for f in files:
        y = load_wav(str(f))
        if len(y) < 5.0 * SR * 1.2:
            continue
        seg = crop_segment(y, 5.0, SR, offset_frac=0.5)
        if np.sqrt(np.mean(seg ** 2)) < 0.01:
            continue
        e, c = CZ.features(peak_normalize(seg))
        ext_e.append(e); ext_c.append(c)
    Ce = np.stack(ext_c)
    ext_keys = [CZ.make_key(e, c) for e, c in zip(ext_e, ext_c)]
    EXT_REG = np.stack([CZ.bits(Ce[k], ext_keys[k]) for k in range(len(Ce))])
    Me = (READ(Ce) != REG[:, None, :]).mean(axis=2)                       # (n, N_ext)
    imp_ext = Me.ravel()
    tau4 = M.threshold_at_fpr(imp_ext, 1e-4)
    tau_half = M.threshold_at_fpr(Me[:, 0::2].ravel(), 1e-4)
    far_heldout = float(np.mean(Me[:, 1::2] <= tau_half))
    print(f"external impostors {len(Ce)} clips, tau4 {tau4:.3f} {time.time() - t0:.0f}s", flush=True)
    hist = {"scheme": NAME, "tau_1e-4": tau4, **{f"b{k:02d}": int(c) for k, c in enumerate(np.histogram(imp_ext, bins=np.linspace(0, 0.6, 61))[0])}}

    tab, far_rows, rec_rows, per_clip = [], [], [], []
    for a in [x for x in ATTACK_ORDER if (cache / x).is_dir()]:
        Ca = np.stack([CZ.features(load_wav(str(cache / a / f"{c}.wav")))[1] for c in ids])
        Ba = READ(Ca)                                                      # (reg i, received j, 1024)
        g_bits = Ba[np.arange(n), np.arange(n)]
        E = (g_bits != REG).astype(np.uint8)
        g = E.mean(axis=1)
        Ma = (Ba != REG[:, None, :]).mean(axis=2)
        far_attack = float(np.mean(Ma[off] <= tau))
        lo, hi = bootstrap_ci(g, spk)
        tab.append({"scheme": NAME, "attack": a, "group": ATTACK_GROUP[a], "n_clips": n, "seen_in_selection": False,
                    "ber_mean": float(g.mean()), "ber_ci_lo": lo, "ber_ci_hi": hi, "ber_std": float(g.std()),
                    "ber_max": float(g.max()), "frac_le_0.25": float(np.mean(g <= 0.25)), "nc_mean": float(1 - 2 * g.mean()),
                    "tpr_at_1pct_fpr": M.tpr_at_threshold(g, tau), "auc": M.auc(g, imp), "far_attack_at_clean_tau": far_attack,
                    "id_acc": float(np.mean(Ma.argmin(axis=0) == np.arange(n)))})
        # attacked copy j read with the key of each external registration e, against e's registered bits
        fa = 0
        for k in range(len(Ce)):
            fa += int(np.sum((CZ.bits_many(Ca, ext_keys[k]) != EXT_REG[k]).mean(axis=1) <= tau4))
        far_rows.append({"scheme": NAME, "attack": a, "n_external": len(Ce), "n_impostor_pairs": int(imp_ext.size),
                         "imp_mean": float(imp_ext.mean()), "imp_std": float(imp_ext.std()), "imp_min": float(imp_ext.min()),
                         "tau_1e-4": tau4, "tpr_at_1e-4": float(np.mean(g <= tau4)), "far_attack_at_1e-4": fa / (n * len(Ce)),
                         "eer": M.eer(g, imp_ext), "genuine_mean": float(g.mean()), "tau_half": tau_half, "far_heldout": far_heldout,
                         "tpr_at_tau_half": float(np.mean(g <= tau_half)), "n_accepted": int(np.sum(g <= tau4)), "n_genuine": n})
        for r in REP_RATES:
            mb = repetition_recovery(E, r)
            rec_rows.append({"scheme": NAME, "attack": a, "rate": r, "message_bits": 1024 // r, "msg_ber_mean": float(mb.mean()),
                             "frac_clips_msg_ber_le_0.05": float(np.mean(mb <= 0.05)), "frac_clips_msg_ber_0": float(np.mean(mb == 0))})
        per_clip += [{"scheme": NAME, "attack": a, "clip_id": ids[j], "ber": float(g[j])} for j in range(n)]
        print(f"{a}: BER {g.mean():.3f} TPR1e-4 {np.mean(g <= tau4):.2f} {time.time() - t0:.0f}s", flush=True)

    # sensitivity to a delay of a few samples (silence in front, end cut), as in sample_shift.py
    shift = []
    for k in [1, 2, 4, 8, 16, 32]:
        Ck = np.stack([CZ.features(np.concatenate([np.zeros(k, np.float32), load_wav(str(data / "clips" / f"{c}.wav"))[:-k]]))[1] for c in ids])
        Bk = np.stack([CZ.bits(Ck[i], keys[i]) for i in range(n)])
        shift.append({"samples": k, "ms": k / 16, "scheme": NAME, "ber_mean": float((Bk != REG).mean())})

    pd.DataFrame(tab).to_csv(out / "table_ber_cz.csv", index=False)
    pd.DataFrame(far_rows).to_csv(out / "table_far_cz.csv", index=False)
    pd.DataFrame(rec_rows).to_csv(out / "table_recovery_cz.csv", index=False)
    pd.DataFrame(per_clip).to_csv(out / "per_clip_cz.csv", index=False)
    pd.DataFrame([hist]).to_csv(out / "impostor_ext_hist_cz.csv", index=False)
    pd.DataFrame(shift).to_csv(out / "table_sample_shift_cz.csv", index=False)
    json.dump({"thresholds": {NAME: tau}, "impostor": {NAME: {"mean": float(imp.mean()), "std": float(imp.std()), "min": float(imp.min()),
               "n_pairs": int(len(imp)), "same_speaker_mean": float(Mc[same].mean()), "other_speaker_mean": float(Mc[off & ~same].mean())}}},
              open(out / "summary_cz.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
