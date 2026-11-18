"""Extract fingerprints, run attack-aware selection on the train split, evaluate on the test split.

Outputs (in --out, default results/):
  fingerprints.npz           cached bits for every scheme/attack/clip
  per_clip.csv               BER for every (scheme, attack, test clip)
  table_ber.csv              mean BER, 95% bootstrap CI, NC, TPR@1%FPR, AUC, ID accuracy
  table_train_vs_test.csv    BER on train vs test clips for the selected schemes (over-fit check)
  selection_composition.csv  how many bits each selected scheme takes from each pool member
  per_family_flip.csv        mean flip rate of each candidate family under each attack (test split)
  summary.json               thresholds, impostor statistics, run metadata

Usage: python scripts/run_eval.py --data data --cache cache --out results
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
from zwm.audio import load_wav
from zwm.schemes import BASELINES, N_BITS, CandidatePool, select_bits
from zwm.attacks import ATTACK_GROUP
from zwm import metrics as M

TRAIN_TASKS_CLASSICAL = ["resample_8k", "mp3_32k", "awgn_20dB", "lowpass_4k", "highpass_200", "requant_8bit"]
TRAIN_TASKS_CODEC = TRAIN_TASKS_CLASSICAL + ["encodec_1.5k", "encodec_6k"]
ATTACK_ORDER = ["resample_8k", "mp3_128k", "mp3_32k", "opus_16k", "opus_12k", "opus_6k", "awgn_20dB", "lowpass_4k", "highpass_200", "amp_0.7",
                "requant_8bit", "pitch_+2st", "delay_7ms", "delay_23ms", "delay_37ms", "delay_250ms", "crop_10pct", "stretch_+2pct", "stretch_-2pct",
                "stretch_+5pct", "stretch_-5pct", "encodec_24k", "encodec_12k", "encodec_6k", "encodec_3k",
                "encodec_1.5k", "dac_12cb", "dac_6cb", "dac_3cb", "knnvc_self", "knnvc_cross", "speecht5_cross"]
REP_RATES = (5, 9, 15)          # repetition-code rates for message recovery (1024 // r message bits)
FPR = 0.01
SEED = 0


def bootstrap_ci(x: np.ndarray, groups=None, n: int = 2000, seed: int = SEED):
    """95% bootstrap interval of the mean. With ``groups`` (one label per element, e.g. the speaker)
    whole groups are resampled, because clips of one speaker are not independent."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    if groups is None:
        means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        gm = np.array([x[groups == g].mean() for g in uniq])
        gn = np.array([(groups == g).sum() for g in uniq])
        pick = rng.integers(0, len(uniq), size=(n, len(uniq)))
        means = (gm[pick] * gn[pick]).sum(axis=1) / gn[pick].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_bootstrap(d: np.ndarray, groups, n: int = 2000, seed: int = SEED):
    """Mean of paired differences with a speaker-level bootstrap interval."""
    lo, hi = bootstrap_ci(d, groups, n, seed)
    return float(np.mean(d)), lo, hi


def repetition_recovery(E: np.ndarray, r: int):
    """Message bit error after an interleaved repetition code of rate r on the per-bit error pattern E (N, 1024):
    message bit i occupies fingerprint bits i, i+m, i+2m, ... (m = 1024 // r), so the r copies are spread over the clip.
    Returns the per-clip message BER after majority decoding."""
    m = E.shape[1] // r
    rep = np.stack([E[:, i::m][:, :r] for i in range(m)], axis=1)      # (N, m, r)
    return (rep.sum(axis=2) > r / 2).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    data, cache, out = Path(args.data), Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    rows = list(csv.DictReader(open(data / "manifest.csv")))
    clip_ids = [r["clip_id"] for r in rows]
    speakers = np.array([int(r["speaker"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    train, test = np.where(split == "train")[0], np.where(split == "test")[0]
    attacks = [a for a in ATTACK_ORDER if (cache / a).is_dir() and any((cache / a).iterdir())]
    print("attacks found:", attacks)

    pool = CandidatePool()
    families = pool.family_of_bit()
    extractors = {**BASELINES, "POOL": pool}

    # ---------------------------------------------------------------- fingerprints
    fp_path = out / "fingerprints.npz"
    bits = {}
    if fp_path.exists():
        z = np.load(fp_path, allow_pickle=True)
        bits = {k: z[k] for k in z.files}
    def need(key):
        return key not in bits
    def wav_for(attack, cid):
        return data / "clips" / f"{cid}.wav" if attack == "clean" else cache / attack / f"{cid}.wav"
    for attack in ["clean"] + attacks:
        keys = {s: f"{s}|{attack}" for s in extractors}
        if not any(need(k) for k in keys.values()):
            continue
        present = [i for i, cid in enumerate(clip_ids) if wav_for(attack, cid).exists()]
        store = {s: np.full((len(clip_ids), len(pool(np.zeros(80000, np.float32) + 1e-3)) if s == "POOL" else N_BITS), 255, np.uint8) for s in extractors}
        for i in present:
            y = load_wav(str(wav_for(attack, clip_ids[i])))
            for s, f in extractors.items():
                store[s][i] = f(y)
        for s in extractors:
            bits[keys[s]] = store[s]
        print(f"fingerprints: {attack} ({len(present)} clips)", flush=True)
    np.savez_compressed(fp_path, **bits)

    def available(attack):
        """indices of clips that have this attack cached (255 marks missing)."""
        return np.where(bits[f"POOL|{attack}"][:, 0] != 255)[0]

    # ---------------------------------------------------------------- selection (train split only)
    P_clean = bits["POOL|clean"]
    def selection(tasks, aggregate="max"):
        for t in tasks:
            assert set(train) <= set(available(t)), f"train clips missing for task {t}"
        res = select_bits(P_clean[train], {t: bits[f"POOL|{t}"][train] for t in tasks},
                          n_select=N_BITS, aggregate=aggregate)
        assert not set(res["idx"]) - set(range(P_clean.shape[1]))
        return res

    rng = np.random.default_rng(SEED)
    selected = {
        "SEL-classical": selection(TRAIN_TASKS_CLASSICAL),
        "SEL-codec": selection(TRAIN_TASKS_CODEC),
        "SEL-codec-mean": selection(TRAIN_TASKS_CODEC, aggregate="mean"),
    }
    # selection over the handcrafted pool plus the 1024 WavLM-LSH bits (does a learned family change the selection?)
    for t in TRAIN_TASKS_CODEC + ["clean"]:
        bits[f"POOLW|{t}"] = np.concatenate([bits[f"POOL|{t}"], bits[f"WavLM-LSH|{t}"]], axis=1)
    resW = select_bits(bits["POOLW|clean"][train], {t: bits[f"POOLW|{t}"][train] for t in TRAIN_TASKS_CODEC},
                       n_select=N_BITS, aggregate="max")
    families_w = np.concatenate([families, np.array(["wavlm_lsh"] * N_BITS)])
    random_idx = np.sort(rng.choice(P_clean.shape[1], N_BITS, replace=False))
    np.savez(out / "selected_idx.npz", **{k: v["idx"] for k, v in selected.items()}, random=random_idx, SEL_codec_W=resW["idx"])

    def scheme_bits(scheme, attack, src=None):
        """Fingerprints of a scheme under an attack; src = a dict of family arrays (the impostor set) instead."""
        get = (lambda fam: bits[f"{fam}|{attack}"]) if src is None else (lambda fam: src[fam])
        if scheme == "PHIL+MFCC":            # trivial combination baseline: every other bit of each family
            return np.concatenate([get("Philips")[:, ::2], get("MFCC")[:, ::2]], axis=1)
        if scheme == "SEL-codec+W":
            return np.concatenate([get("POOL"), get("WavLM-LSH")], axis=1)[:, resW["idx"]]
        if src is not None:
            if scheme in BASELINES:
                return get(scheme)
            if scheme == "POOL-random":
                return get("POOL")[:, random_idx]
            return get("POOL")[:, selected[scheme]["idx"]]
        if scheme in BASELINES:
            return bits[f"{scheme}|{attack}"]
        if scheme == "POOL-all":
            return bits[f"POOL|{attack}"]
        if scheme == "POOL-random":
            return bits[f"POOL|{attack}"][:, random_idx]
        return bits[f"POOL|{attack}"][:, selected[scheme]["idx"]]

    schemes = list(BASELINES) + ["PHIL+MFCC", "POOL-all", "POOL-random"] + list(selected) + ["SEL-codec+W"]

    # selection composition
    comp = []
    for name, res in selected.items():
        fam, cnt = np.unique(families[res["idx"]], return_counts=True)
        for f_, c_ in zip(fam, cnt):
            comp.append({"scheme": name, "family": f_, "n_bits": int(c_), "fraction": c_ / N_BITS})
    fam, cnt = np.unique(families_w[resW["idx"]], return_counts=True)
    for f_, c_ in zip(fam, cnt):
        comp.append({"scheme": "SEL-codec+W", "family": f_, "n_bits": int(c_), "fraction": c_ / N_BITS})
    pd.DataFrame(comp).to_csv(out / "selection_composition.csv", index=False)

    # ---------------------------------------------------------------- evaluation (test split)
    per_clip, table, train_test, recovery, paired = [], [], [], [], []
    summary = {"n_train": int(len(train)), "n_test": int(len(test)), "fpr": FPR, "attacks": attacks,
               "train_tasks": {"SEL-classical": TRAIN_TASKS_CLASSICAL, "SEL-codec": TRAIN_TASKS_CODEC,
                               "SEL-codec-mean": TRAIN_TASKS_CODEC},
               "thresholds": {}, "impostor": {}}
    for s in schemes:
        F_clean = scheme_bits(s, "clean")
        imp = M.impostor_bers(F_clean[test])
        tau = M.threshold_at_fpr(imp, FPR)
        summary["thresholds"][s] = tau
        same_spk = speakers[test][:, None] == speakers[test][None, :]
        Mimp = M.pairwise_ber_matrix(F_clean[test])
        iu = np.triu_indices(len(test), k=1)
        summary["impostor"][s] = {"mean": float(imp.mean()), "std": float(imp.std()), "min": float(imp.min()),
                                  "n_pairs": int(len(imp)), "same_speaker_mean": float(Mimp[iu][same_spk[iu]].mean()),
                                  "other_speaker_mean": float(Mimp[iu][~same_spk[iu]].mean())}
        for a in attacks:
            idx = np.intersect1d(test, available(a))
            if len(idx) == 0:
                continue
            F_att = scheme_bits(s, a)
            b = (F_clean[idx] != F_att[idx]).mean(axis=1)
            for i, v in zip(idx, b):
                per_clip.append({"scheme": s, "attack": a, "clip_id": clip_ids[i], "ber": float(v)})
            lo, hi = bootstrap_ci(b, speakers[idx])
            # false-accept rate under attack: an attacked clip against every *other* registered clip, at the clean threshold
            Mx = M.pairwise_ber_matrix(F_att[idx], F_clean[test])
            own = np.array([list(test).index(i) for i in idx])
            mask = np.ones_like(Mx, bool); mask[np.arange(len(idx)), own] = False
            far_attack = float(np.mean(Mx[mask] <= tau))
            E = (F_clean[idx] != F_att[idx]).astype(int)
            for r_ in REP_RATES:
                mb = repetition_recovery(E, r_)
                recovery.append({"scheme": s, "attack": a, "rate": r_, "message_bits": E.shape[1] // r_,
                                 "msg_ber_mean": float(mb.mean()), "frac_clips_msg_ber_le_0.05": float(np.mean(mb <= 0.05)),
                                 "frac_clips_msg_ber_0": float(np.mean(mb == 0))})
            table.append({
                "scheme": s, "attack": a, "group": ATTACK_GROUP[a], "n_clips": int(len(idx)),
                "seen_in_selection": a in summary["train_tasks"].get(s, []),
                "ber_mean": float(b.mean()), "ber_ci_lo": lo, "ber_ci_hi": hi, "ber_std": float(b.std()),
                "ber_max": float(b.max()), "frac_le_0.25": float(np.mean(b <= 0.25)),
                "nc_mean": float(1 - 2 * b.mean()),
                "tpr_at_1pct_fpr": M.tpr_at_threshold(b, tau),
                "auc": M.auc(b, imp), "far_attack_at_clean_tau": far_attack,
                "id_acc": M.identification_accuracy(F_att[idx], F_clean[test]) if set(idx) == set(test) else np.nan,
            })
            if s in selected or s in BASELINES:
                idx_tr = np.intersect1d(train, available(a))
                if len(idx_tr):
                    b_tr = (F_clean[idx_tr] != F_att[idx_tr]).mean(axis=1)
                    train_test.append({"scheme": s, "attack": a, "ber_train": float(b_tr.mean()),
                                       "ber_test": float(b.mean()), "n_train": len(idx_tr), "n_test": len(idx)})
        print(f"evaluated {s}", flush=True)

    pd.DataFrame(per_clip).to_csv(out / "per_clip.csv", index=False)
    pd.DataFrame(recovery).to_csv(out / "table_recovery.csv", index=False)
    # paired, speaker-level bootstrap of SEL-codec minus each baseline, per attack
    pc = pd.DataFrame(per_clip)
    pairs = [("SEL-codec", b) for b in ["MFCC", "Philips", "DWT-DCT", "WavLM-LSH", "PHIL+MFCC", "SEL-classical"]] + [("SEL-codec+W", "WavLM-LSH")]
    for sel, base in pairs:
        for a in attacks:
            x = pc[(pc.scheme == sel) & (pc.attack == a)].set_index("clip_id").ber
            yb = pc[(pc.scheme == base) & (pc.attack == a)].set_index("clip_id").ber
            if len(x) == 0 or len(yb) == 0:
                continue
            common = x.index.intersection(yb.index)
            d = (x[common] - yb[common]).values
            spk = np.array([int(c.split("_")[0]) for c in common])
            mean, lo, hi = paired_bootstrap(d, spk)
            paired.append({"selected": sel, "baseline": base, "attack": a, "n": len(common), "diff_mean": mean, "diff_lo": lo, "diff_hi": hi,
                           "sel_better": bool(hi < 0), "base_better": bool(lo > 0)})
    pd.DataFrame(paired).to_csv(out / "table_paired.csv", index=False)
    # ---------------------------------------------------------------- large impostor set (LibriSpeech test-clean/test-other)
    ip = out / "impostor_fingerprints.npz"
    if ip.exists():
        zi = np.load(ip)
        ext = {k: zi[k] for k in zi.files if k != "ids"}
        far_rows, hist_rows = [], []
        for s_ in ["DWT-DCT", "Philips", "MFCC", "WavLM-LSH", "PHIL+MFCC", "SEL-classical", "SEL-codec", "SEL-codec+W"]:
            R = scheme_bits(s_, "clean")[test]
            E = scheme_bits(s_, None, ext)
            imp_ext = M.pairwise_ber_matrix(R, E).ravel()              # 80 registered x N external clean clips
            tau4 = M.threshold_at_fpr(imp_ext, 1e-4)
            Mx = M.pairwise_ber_matrix(R, E)
            tau_half = M.threshold_at_fpr(Mx[:, 0::2].ravel(), 1e-4)        # threshold from even-index external clips
            far_heldout = float(np.mean(Mx[:, 1::2] <= tau_half))            # false accepts on the odd-index clips
            hist_rows.append({"scheme": s_, "tau_1e-4": tau4, **{f"b{k:02d}": int(c) for k, c in enumerate(np.histogram(imp_ext, bins=np.linspace(0, 0.6, 61))[0])}})
            mu, sd = float(imp_ext.mean()), float(imp_ext.std())
            for a in attacks:
                A_ = scheme_bits(s_, a)[test]
                ok = A_[:, 0] != 255
                g = np.mean(R[ok] != A_[ok], axis=1)
                far_att = float(np.mean(M.pairwise_ber_matrix(A_[ok], E) <= tau4))
                far_rows.append({"scheme": s_, "attack": a, "n_external": int(E.shape[0]), "n_impostor_pairs": int(len(imp_ext)),
                                 "imp_mean": mu, "imp_std": sd, "imp_min": float(imp_ext.min()), "tau_1e-4": tau4,
                                 "tpr_at_1e-4": float(np.mean(g <= tau4)), "far_attack_at_1e-4": far_att,
                                 "eer": M.eer(g, imp_ext), "genuine_mean": float(g.mean()),
                                 "tau_half": tau_half, "far_heldout": far_heldout, "tpr_at_tau_half": float(np.mean(g <= tau_half)),
                                 "n_accepted": int(np.sum(g <= tau4)), "n_genuine": int(len(g))})
        pd.DataFrame(far_rows).to_csv(out / "table_far.csv", index=False)
        pd.DataFrame(hist_rows).to_csv(out / "impostor_ext_hist.csv", index=False)
        print("large impostor set:", E.shape[0], "clips", flush=True)
    pd.DataFrame(table).to_csv(out / "table_ber.csv", index=False)
    pd.DataFrame(train_test).to_csv(out / "table_train_vs_test.csv", index=False)

    # per-family flip rate (test split) for every candidate bit family
    fam_rows = []
    for a in attacks:
        idx = np.intersect1d(test, available(a))
        flip = (P_clean[idx] != bits[f"POOL|{a}"][idx]).mean(axis=0)
        for f_ in dict.fromkeys(families):
            fam_rows.append({"attack": a, "family": f_, "flip_rate": float(flip[families == f_].mean())})
    pd.DataFrame(fam_rows).to_csv(out / "per_family_flip.csv", index=False)

    summary["elapsed_s"] = time.time() - t_start
    summary["pool_size"] = int(P_clean.shape[1])
    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    print(pd.DataFrame(table).pivot(index="attack", columns="scheme", values="ber_mean").round(3).to_string())


if __name__ == "__main__":
    main()
