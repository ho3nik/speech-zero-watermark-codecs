"""Intelligibility of attacked copies: word error rate of Whisper (base.en) transcripts of each attacked test clip
against the transcript of its original clip. Writes results/table_wer.csv."""
import csv, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwm.audio import load_wav

torch.set_num_threads(1)
ATT = ["clean", "mp3_32k", "opus_6k", "encodec_6k", "encodec_1.5k", "dac_12cb", "dac_3cb", "knnvc_self", "knnvc_cross", "speecht5_cross"]
proc = WhisperProcessor.from_pretrained("openai/whisper-base.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").eval()
ids = [r["clip_id"] for r in csv.DictReader(open("data/manifest_80.csv")) if r["split"] == "test"]   # the 80 original test clips
norm = lambda t: re.sub(r"[^a-z' ]", " ", t.lower()).split()


def wer(ref, hyp):
    d = np.arange(len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1], len(ref)


def transcribe(paths):
    out = []
    for k in range(0, len(paths), 8):
        x = [load_wav(p) for p in paths[k:k + 8]]
        f = proc(x, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            g = model.generate(f, max_new_tokens=96)
        out += proc.batch_decode(g, skip_special_tokens=True)
    return out


ref = transcribe([f"data/clips/{c}.wav" for c in ids])
import os
prev = pd.read_csv("results/table_wer.csv") if os.path.exists("results/table_wer.csv") else None
rows = [] if prev is None else prev.to_dict("records")
done = set() if prev is None else set(prev.attack)
for a in ATT:
    if a in done:
        continue
    hyp = ref if a == "clean" else transcribe([f"cache/{a}/{c}.wav" for c in ids])
    e = [wer(norm(r), norm(h)) for r, h in zip(ref, hyp)]
    rows.append({"attack": a, "wer_vs_clean_transcript": sum(x for x, _ in e) / sum(n for _, n in e), "n_words": sum(n for _, n in e)})
    print(a, round(rows[-1]["wer_vs_clean_transcript"], 3), flush=True)
pd.DataFrame(rows).to_csv("results/table_wer.csv", index=False)
