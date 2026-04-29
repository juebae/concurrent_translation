"""
phase4_baseline.py - Baseline Evaluation (Sequential Loading)
Runs models one at a time to stay within 4GB RAM
"""

import sys, os, json, time, tempfile
import numpy as np
import soundfile as sf
sys.path.insert(0, '/home/zubair/disso')

from phase3a_asr_module import WhisperASR
from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from jiwer import wer
from sacrebleu import corpus_bleu, corpus_chrf

FLEURS_PATH  = "/home/zubair/disso/datasets/fleurs_test/samples.json"
FLORES_PATH  = "/home/zubair/disso/datasets/flores_test/samples.json"
RESULTS_PATH = "/home/zubair/disso/results/baseline_results.json"
INTERIM_PATH = "/home/zubair/disso/results/interim.json"

os.makedirs("/home/zubair/disso/results", exist_ok=True)

with open(FLEURS_PATH) as f:
    fleurs = json.load(f)
with open(FLORES_PATH) as f:
    flores = {s["id"]: s for s in json.load(f)}

# ── PHASE 1: ASR ─────────────────────────────────────────────
print("=== PHASE 1: ASR ===")
asr = WhisperASR(model_size="tiny")
ok, msg = asr.load()
print(msg)

asr_results = []
for i, sample in enumerate(fleurs):
    audio_arr   = np.array(sample["audio_array"])
    sample_rate = sample["sampling_rate"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        sf.write(tmp_path, audio_arr, sample_rate)

    ok, transcription, asr_lat = asr.transcribe(tmp_path, language="en")
    os.unlink(tmp_path)

    asr_results.append({
        "id": sample["id"],
        "ref_text": sample["transcription"],
        "asr_text": transcription if ok else "",
        "asr_ok": ok,
        "asr_latency_ms": round(asr_lat, 1)
    })

    if (i+1) % 10 == 0:
        print(f"  ASR {i+1}/100 done")

asr.cleanup()
del asr

# Save interim
with open(INTERIM_PATH, "w") as f:
    json.dump(asr_results, f)
print(f"ASR done. Saved interim to {INTERIM_PATH}")

# ── PHASE 2: MT ──────────────────────────────────────────────
print("\n=== PHASE 2: MT ===")
mt = OpusMT()
ok, msg = mt.load()
print(msg)

for r in asr_results:
    if not r["asr_ok"] or not r["asr_text"]:
        r["mt_text"] = ""
        r["mt_latency_ms"] = 0.0
        continue
    ok, translation, mt_lat = mt.translate(r["asr_text"])
    r["mt_text"] = translation if ok else ""
    r["mt_ok"] = ok
    r["mt_latency_ms"] = round(mt_lat, 1)

mt.cleanup()
del mt
print("MT done.")

# ── PHASE 3: QE ──────────────────────────────────────────────
print("\n=== PHASE 3: QE ===")
qe = QualityEstimation()
ok, msg = qe.load()
print(msg)

for r in asr_results:
    if not r.get("mt_text"):
        r["qe_score"] = None
        r["qe_latency_ms"] = 0.0
        continue
    ok, qe_score, qe_lat = qe.score(r["asr_text"], r["mt_text"])
    r["qe_score"] = round(float(qe_score), 4) if ok else None
    r["qe_latency_ms"] = round(qe_lat, 1)

qe.cleanup()
del qe
print("QE done.")

# ── METRICS ──────────────────────────────────────────────────
valid = [r for r in asr_results if r.get("asr_text") and r.get("mt_text")]

asr_refs = [r["ref_text"] for r in valid]
asr_hyps = [r["asr_text"] for r in valid]
mt_hyps  = [r["mt_text"]  for r in valid if r["id"] in flores]
mt_refs  = [flores[r["id"]]["reference_es"] for r in valid if r["id"] in flores]

wer_score  = wer(" ".join(asr_refs), " ".join(asr_hyps))
bleu_score = corpus_bleu(mt_hyps, [mt_refs]).score
chrf_score = corpus_chrf(mt_hyps, [mt_refs]).score

for r in asr_results:
    r["total_latency_ms"] = round(
        r.get("asr_latency_ms", 0) +
        r.get("mt_latency_ms", 0) +
        r.get("qe_latency_ms", 0), 1)

summary = {
    "num_samples": len(valid),
    "wer":  round(wer_score, 4),
    "bleu": round(bleu_score, 2),
    "chrf": round(chrf_score, 2),
    "avg_asr_latency_ms": round(np.mean([r["asr_latency_ms"] for r in valid]), 1),
    "avg_mt_latency_ms":  round(np.mean([r["mt_latency_ms"]  for r in valid]), 1),
    "avg_qe_latency_ms":  round(np.mean([r["qe_latency_ms"]  for r in valid]), 1),
    "avg_total_latency_ms": round(np.mean([r["total_latency_ms"] for r in valid]), 1),
}

with open(RESULTS_PATH, "w") as f:
    json.dump({"summary": summary, "per_sample": asr_results}, f, indent=2)

print("\n========== BASELINE RESULTS ==========")
print(f"Samples evaluated : {summary['num_samples']}")
print(f"WER               : {summary['wer']:.4f}  (lower is better)")
print(f"BLEU              : {summary['bleu']:.2f}  (higher is better)")
print(f"ChrF              : {summary['chrf']:.2f}  (higher is better)")
print(f"Avg ASR latency   : {summary['avg_asr_latency_ms']:.1f} ms")
print(f"Avg MT latency    : {summary['avg_mt_latency_ms']:.1f} ms")
print(f"Avg QE latency    : {summary['avg_qe_latency_ms']:.1f} ms")
print(f"Avg total latency : {summary['avg_total_latency_ms']:.1f} ms")
print(f"\nResults saved to  : {RESULTS_PATH}")
