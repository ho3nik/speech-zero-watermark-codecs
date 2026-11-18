"""Evaluate the reimplemented RSZW scheme of Salayani et al. (2023) on the test clips.

Each test clip is registered with its own random 1024-bit logo and seven SVMs, one per training attack of the
kinds the paper simulates and our attack set contains (amplitude change, white noise, high-pass, low-pass, MP3,
requantization, resampling). Verification uses the best of the seven SVMs ("at least one" rule) for genuine and
impostor pairs alike. Impostor pairs: every registration read on 10 random clips of other speakers and 5 other
clips of its own speaker.

The binary pattern of Eq. 5 follows ZWM_SZ_RULE (see zwm/salayani.py): "printed" (default; the equation as
printed, which sets every bit of an original recording to 1, and the rule behind every result in the paper) or
"mean" (a threshold at the mean of the clip's features). Outputs for the mean rule carry the suffix "_mean".

The work can be split over processes, because SVM prediction is slow and single-threaded:
  python scripts/run_salayani.py --part imp                       # impostor pairs
  python scripts/run_salayani.py --part att --shard 0 --nshards 3 # every third attack, starting with the first
  python scripts/run_salayani.py --part merge                     # thresholds, tables, summary
  python scripts/run_salayani.py                                  # all of the above in one process

Writes results/table_ber_sz{suffix}.csv, table_recovery_sz{suffix}.csv, summary_sz{suffix}.json.
"""
from __future__ import annotations

import argparse, csv, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zwm import metrics as M
from zwm import salayani as SZ
from zwm.attacks import ATTACK_GROUP
from zwm.audio import load_wav
from run_eval import ATTACK_ORDER, REP_RATES, bootstrap_ci, repetition_recovery

NAME = "Salayani"
TRAIN = ["amp_0.7", "awgn_20dB", "highpass_200", "lowpass_4k", "mp3_32k", "requant_8bit", "resample_8k"]
SFX = "" if SZ.RULE == "printed" else f"_{SZ.RULE}"
R = Path("results")


def setup():
    rows = [r for r in csv.DictReader(open("data/manifest.csv")) if r["split"] == "test"]
    ids = [r["clip_id"] for r in rows]
    spk = np.array([int(r["speaker"]) for r in rows])
    key = SZ.selection_key()
    clean = [load_wav(f"data/clips/{c}.wav") for c in ids]
    regs = []
    for i, c in enumerate(ids):
        logo = np.random.default_rng(1000 + i).integers(0, 2, SZ.M1 * SZ.M2).astype(np.uint8)
        att = [load_wav(f"cache/{a}/{c}.wav") for a in TRAIN]
        regs.append(SZ.RSZW(logo, key).fit(clean[i], att))
    return ids, spk, key, clean, regs


def part_imp(ids, spk, key, clean, regs, log):
    """Registration i read on 10 clips of other speakers and 5 other clips of its own speaker."""
    cache_f, partial = R / f"salayani_imp{SFX}.npz", R / f"salayani_imp_partial{SFX}.npz"
    if cache_f.exists():
        return
    Fclean = [SZ.features(y, key) for y in clean]
    n, rng = len(ids), np.random.default_rng(0)
    imp_list, imp_same, i0 = [], [], 0
    if partial.exists():
        zp = np.load(partial); i0 = int(zp["i"]); imp_list, imp_same = list(zp["other"]), list(zp["same"])
    for i in range(n):
        others = rng.choice(np.where(spk != spk[i])[0], 10, replace=False)
        sames = rng.choice([j for j in np.where(spk == spk[i])[0] if j != i], 5, replace=False)
        if i < i0:
            continue                                   # keep the random stream identical on resume
        e = regs[i].logo_errors([Fclean[j] for j in list(others) + list(sames)]).mean(axis=1)
        imp_list += list(e[:10]); imp_same += list(e[10:])
        if i % 20 == 19:
            np.savez(partial, i=i + 1, other=np.array(imp_list), same=np.array(imp_same))
            log(f"impostors {i + 1}/{n}")
    np.savez(cache_f, other=np.array(imp_list), same=np.array(imp_same))
    log("impostors done")


def part_att(ids, spk, key, regs, shard, nshards, log):
    out = R / f"salayani_att{SFX}_{shard}of{nshards}.pkl"
    res = pickle.load(open(out, "rb")) if out.exists() else {}
    attacks = [a for a in ATTACK_ORDER if Path(f"cache/{a}").is_dir()][shard::nshards]
    for a in [x for x in attacks if x not in res]:
        Fa = [SZ.features(load_wav(f"cache/{a}/{c}.wav"), key) for c in ids]
        E = np.stack([regs[i].logo_errors([Fa[i]])[0] for i in range(len(ids))])
        res[a] = {"g": E.mean(axis=1), "msg": {r: repetition_recovery(E, r) for r in REP_RATES}}
        pickle.dump(res, open(out, "wb"))
        log(f"{a}: BER {res[a]['g'].mean():.3f}")


def part_merge(spk, log):
    z = np.load(R / f"salayani_imp{SFX}.npz")
    imp_list, imp_same = list(z["other"]), list(z["same"])
    imp_v = np.array(imp_list + imp_same)
    tau = M.threshold_at_fpr(imp_v, 0.01)
    res = {}
    for f in sorted(R.glob(f"salayani_att{SFX}_*of*.pkl")):
        res.update(pickle.load(open(f, "rb")))
    tab, rec = [], []
    for a in [x for x in ATTACK_ORDER if x in res]:
        g, n = res[a]["g"], len(res[a]["g"])
        lo, hi = bootstrap_ci(g, spk)
        tab.append({"scheme": NAME, "attack": a, "group": ATTACK_GROUP[a], "n_clips": n, "seen_in_selection": a in TRAIN,
                    "ber_mean": float(g.mean()), "ber_ci_lo": lo, "ber_ci_hi": hi, "ber_std": float(g.std()), "ber_max": float(g.max()),
                    "frac_le_0.25": float(np.mean(g <= 0.25)), "nc_mean": float(1 - 2 * g.mean()),
                    "tpr_at_1pct_fpr": M.tpr_at_threshold(g, tau), "auc": M.auc(g, imp_v), "far_attack_at_clean_tau": np.nan, "id_acc": np.nan})
        for r in REP_RATES:
            mb = res[a]["msg"][r]
            rec.append({"scheme": NAME, "attack": a, "rate": r, "message_bits": 1024 // r, "msg_ber_mean": float(mb.mean()),
                        "frac_clips_msg_ber_le_0.05": float(np.mean(mb <= 0.05)), "frac_clips_msg_ber_0": float(np.mean(mb == 0))})
        log(f"{a}: BER {g.mean():.3f} TPR1% {tab[-1]['tpr_at_1pct_fpr']:.2f}")
    missing = [a for a in ATTACK_ORDER if Path(f"cache/{a}").is_dir() and a not in res]
    assert not missing, f"attacks not evaluated: {missing}"
    pd.DataFrame(tab).to_csv(R / f"table_ber_sz{SFX}.csv", index=False)
    pd.DataFrame(rec).to_csv(R / f"table_recovery_sz{SFX}.csv", index=False)
    json.dump({"rule": SZ.RULE, "thresholds": {NAME: tau}, "train_tasks": {NAME: TRAIN},
               "impostor": {NAME: {"mean": float(imp_v.mean()), "std": float(imp_v.std()), "min": float(imp_v.min()), "n_pairs": int(imp_v.size),
                                   "same_speaker_mean": float(np.mean(imp_same)), "other_speaker_mean": float(np.mean(imp_list))}}},
              open(R / f"summary_sz{SFX}.json", "w"), indent=1)
    log(f"impostors tau {tau:.3f} mean {imp_v.mean():.3f}; merged {len(tab)} attacks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "imp", "att", "merge"], default="all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[{SZ.RULE} {args.part}{args.shard if args.part == 'att' else ''}] {s} {time.time() - t0:.0f}s", flush=True)
    log(f"rule {SZ.RULE}")
    if args.part == "merge":
        spk = np.array([int(r["speaker"]) for r in csv.DictReader(open("data/manifest.csv")) if r["split"] == "test"])
        part_merge(spk, log)
        return
    ids, spk, key, clean, regs = setup()
    log(f"registered {len(ids)} clips")
    if args.part in ("all", "imp"):
        part_imp(ids, spk, key, clean, regs, log)
    if args.part in ("all", "att"):
        part_att(ids, spk, key, regs, args.shard, args.nshards, log)
    if args.part == "all":
        part_merge(spk, log)
    log("SZ_DONE")


if __name__ == "__main__":
    main()
