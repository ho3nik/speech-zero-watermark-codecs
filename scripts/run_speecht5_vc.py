"""Second voice converter, not built on WavLM: SpeechT5 voice conversion (microsoft/speecht5_vc with the
microsoft/speecht5_hifigan vocoder). Cross-gender only: female test speakers are converted to the male CMU ARCTIC
voice bdl and male test speakers to the female voice slt, each given by the mean of its ARCTIC x-vectors.
SpeechT5 generates its own timing, so the output is cut or zero-padded to the length of the original, as for every
other attack. Writes cache/speecht5_cross/<clip>.wav.
Usage: python scripts/run_speecht5_vc.py --xvec /path/to/spkrec-xvect --librispeech /path/to/LibriSpeech
"""
import argparse, csv, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from zwm.audio import load_wav, save_wav, align
from run_attacks import speaker_genders

torch.set_num_threads(1)
ap = argparse.ArgumentParser(); ap.add_argument("--xvec", required=True); ap.add_argument("--librispeech", required=True)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
proc = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc").eval()
voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
xv = lambda spk: torch.tensor(np.mean([np.load(p) for p in sorted(Path(args.xvec).glob(f"cmu_us_{spk}_arctic-*.npy"))], axis=0), dtype=torch.float32)[None]
target = {"F": xv("bdl"), "M": xv("slt")}
genders = speaker_genders(Path(args.librispeech))
rows = [r for r in csv.DictReader(open("data/manifest.csv")) if r["split"] == "test"]
if args.limit:
    rows = rows[: args.limit]
out = Path("cache/speecht5_cross"); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for k, r in enumerate(rows):
    o = out / f"{r['clip_id']}.wav"
    if o.exists():
        continue
    y = load_wav(f"data/clips/{r['clip_id']}.wav")
    inp = proc(audio=y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        # maxlenratio bounds the decoder length below its positional-encoding limit (the default 20 can exceed it)
        z = model.generate_speech(inp["input_values"], target[genders[int(r["speaker"])]], vocoder=voc, maxlenratio=4.0).numpy()
    save_wav(str(o), align(z.astype(np.float32), len(y)))
    if k % 20 == 0:
        print(k, f"{time.time() - t0:.0f}s", len(z) / 16000, flush=True)
print("done", time.time() - t0)
