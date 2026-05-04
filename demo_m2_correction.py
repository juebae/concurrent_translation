#!/usr/bin/env python3

import sys, os, time, gc, csv, json, subprocess, tempfile, threading
sys.path.insert(0, os.path.expanduser("~/disso"))

import numpy as np
import sounddevice as sd
import soundfile as sf

from phase3a_asr_module import WhisperASR
from phase3a_mt_module  import OpusMT
from phase3a_qe_module  import QualityEstimation

OUTPUT_CSV  = os.path.expanduser("~/disso/results/demo_m2_log.csv")
OUTPUT_JSON = os.path.expanduser("~/disso/results/demo_m2_log.json")
os.makedirs(os.path.expanduser("~/disso/results"), exist_ok=True)

SAMPLE_RATE      = 16000
CHUNK_SIZE       = 1024
SILENCE_THRESH   = 0.018
SILENCE_GAP_SEC  = 1.0
MIN_SENTENCE_SEC = 0.8
QE_THRESHOLD     = 0.90
N_BEAMS          = 5

def get_ram_mb():
    try:
        lines = {}
        for l in open("/proc/meminfo"):
            if ":" in l:
                k, v = l.split(":")
                lines[k.strip()] = int(v.split()[0])
        return round((lines["MemTotal"] - lines["MemAvailable"]) / 1024, 1)
    except: return 0

def get_temp():
    try: return round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000, 1)
    except: return None

def speak_tts(text):
    t0 = time.time()
    try: subprocess.run(["espeak-ng", "-v", "es", "-s", "145", text],
                        capture_output=True, timeout=30)
    except: pass
    return round((time.time() - t0) * 1000, 1)

def mbr_select(candidates):
    try:
        from sacrebleu import corpus_chrf
        if len(candidates) <= 1: return candidates[0] if candidates else ""
        best, best_score = candidates[0], -1.0
        for i, hyp in enumerate(candidates):
            peers = [c for j, c in enumerate(candidates) if j != i]
            avg = sum(corpus_chrf([hyp], [[p]]).score for p in peers) / len(peers)
            if avg > best_score: best_score = avg; best = hyp
        return best
    except: return candidates[0]

def split_sentences(chunks, sr=SAMPLE_RATE):
    if not chunks: return []
    audio = np.concatenate(chunks)
    frame_len = int(sr * 0.05)
    sentences = []
    current = []
    silent_frames = 0
    in_speech = False
    max_silent = int(SILENCE_GAP_SEC / 0.05)

    for i in range(0, len(audio), frame_len):
        frame = audio[i:i + frame_len]
        if len(frame) == 0: continue
        energy = float(np.sqrt(np.mean(frame ** 2)))

        if energy > SILENCE_THRESH:
            in_speech = True
            silent_frames = 0
            current.append(frame)
        else:
            if in_speech:
                current.append(frame)
                silent_frames += 1
                if silent_frames >= max_silent:
                    seg = np.concatenate(current)
                    dur = len(seg) / sr
                    print(f"  [SPLIT] segment {len(sentences)+1}: {dur:.2f}s")
                    if dur >= MIN_SENTENCE_SEC: sentences.append(seg)
                    current = []; in_speech = False; silent_frames = 0

    if current:
        seg = np.concatenate(current)
        dur = len(seg) / sr
        print(f"  [SPLIT] final segment: {dur:.2f}s")
        if dur >= MIN_SENTENCE_SEC: sentences.append(seg)

    return sentences

# Record 
audio_chunks = []
stop_mic     = threading.Event()

def mic_thread_fn():
    def callback(indata, frames, time_info, status):
        audio_chunks.append(indata.copy().flatten())
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=CHUNK_SIZE, dtype="float32",
                        callback=callback):
        stop_mic.wait()

print("\n" + "="*60)
print(f" M2 SELF-CORRECTION  τ={QE_THRESHOLD}  N={N_BEAMS}")
print(" Loading ASR first...")
print("="*60)

asr = WhisperASR(model_size="tiny")
ok, msg = asr.load()
print(f"[ASR] {msg}")
if not ok: sys.exit("ASR failed")

print("\n" + "="*60)
print(" Recording... speak your sentences.")
print(" Press ENTER when done.")
print("="*60 + "\n")

mic_t = threading.Thread(target=mic_thread_fn, daemon=True)
mic_t.start()

try:
    input()
except KeyboardInterrupt:
    pass
stop_mic.set()
mic_t.join(timeout=2)

total_sec = round(len(audio_chunks) * CHUNK_SIZE / SAMPLE_RATE, 1)
print(f"\n[INFO] Recorded {len(audio_chunks)} chunks ({total_sec}s)")
print(f"[INFO] Splitting by silence (thresh={SILENCE_THRESH}, gap={SILENCE_GAP_SEC}s)...\n")

sentences_audio = split_sentences(audio_chunks)
print(f"\n[INFO] Found {len(sentences_audio)} sentences\n")

if not sentences_audio:
    sys.exit("[!] No sentences detected. Try speaking louder or adjust SILENCE_THRESH.")

wav_files = []
for i, seg in enumerate(sentences_audio, 1):
    path = os.path.expanduser(f"~/disso/results/tmp_sent_{i}.wav")
    sf.write(path, seg, SAMPLE_RATE)
    wav_files.append(path)
    print(f"  Saved sentence {i}: {round(len(seg)/SAMPLE_RATE,2)}s")

#ASR all 
print("\n[PHASE] ASR transcribing...")
transcripts = []; asr_times = []
for i, wav in enumerate(wav_files, 1):
    res = asr.transcribe(wav, language="en")
    ok_a, text, ms = res if len(res) == 3 else (*res, 0.0)
    ms = round(float(ms), 1); text = str(text).strip()
    print(f"  [{i}] {text!r}  ({ms:.0f}ms)")
    transcripts.append(text); asr_times.append(ms)

asr.cleanup(); gc.collect()
print(f"[ASR] done, unloaded. RAM={get_ram_mb()}MB\n")

#MT Pass 1 all
print("[PHASE] Loading MT...")
mt = OpusMT()
ok, msg = mt.load(); print(f"[MT] {msg}")

translations_p1 = []; mt1_times = []
for i, text in enumerate(transcripts, 1):
    if not text: translations_p1.append(""); mt1_times.append(0); continue
    ok_m, trans, ms = mt.translate(text, num_beams=1)
    ms = round(float(ms), 1)
    print(f"  [{i}] {trans!r}  ({ms:.0f}ms)")
    translations_p1.append(trans if ok_m else ""); mt1_times.append(ms)

# QE all 
print(f"\n[PHASE] Loading QE (threshold={QE_THRESHOLD})...")
qe = QualityEstimation()
ok, msg = qe.load(); print(f"[QE] {msg}")

qe_scores = []; qe_times = []; triggered = []
for i, (src, tgt) in enumerate(zip(transcripts, translations_p1), 1):
    if not src or not tgt:
        qe_scores.append(None); qe_times.append(0); triggered.append(False); continue
    t0 = time.time()
    ok_q, score, _ = qe.score(src, tgt)
    ms = round((time.time() - t0) * 1000, 1)
    score = round(float(score), 4) if ok_q else None
    trig = score is not None and score < QE_THRESHOLD
    print(f"  [{i}] score={score}  trigger={'YES ⚡' if trig else 'no'}  ({ms:.0f}ms)")
    qe_scores.append(score); qe_times.append(ms); triggered.append(trig)

qe.cleanup(); gc.collect()
print(f"[QE] done, unloaded. RAM={get_ram_mb()}MB\n")

# MT Pass 2 MBR for triggered sentences 
translations_final = list(translations_p1)
mt2_times = [0] * len(transcripts)

if any(triggered):
    print("[PHASE] MT Pass 2 — MBR for triggered sentences...")
    mt2 = OpusMT()
    ok, msg = mt2.load(); print(f"[MT2] {msg}")
    for i, (text, trig) in enumerate(zip(transcripts, triggered)):
        if not trig or not text: continue
        t0 = time.time()
        ok_nb, candidates = mt2.translate_nbest(text, num_beams=N_BEAMS)
        ms = round((time.time() - t0) * 1000, 1)
        if ok_nb and candidates:
            best = mbr_select(candidates)
            translations_final[i] = best
            print(f"  [{i+1}] MBR → {best!r}  ({ms:.0f}ms)")
        mt2_times[i] = ms
    mt2.cleanup(); gc.collect()
    print(f"[MT2] done. RAM={get_ram_mb()}MB\n")
else:
    print("[INFO] No sentences triggered correction, skipping MT Pass 2\n")

#TTS
print("[PHASE] TTS speaking translations...")
tts_times = []
for i, trans in enumerate(translations_final, 1):
    if not trans: tts_times.append(0); continue
    ms = speak_tts(trans)
    print(f"  [{i}] spoke ({ms:.0f}ms)")
    tts_times.append(ms)

#  Save
records = []
wall = 0
for i in range(len(transcripts)):
    total_ms = round(asr_times[i] + mt1_times[i] + mt2_times[i] + qe_times[i] + tts_times[i], 1)
    wall += total_ms / 1000
    rec = dict(
        sentence_id          = i + 1,
        wall_clock_s         = round(wall, 2),
        transcript           = transcripts[i],
        trans_pass1          = translations_p1[i],
        trans_final          = translations_final[i],
        qe_score             = qe_scores[i],
        correction_triggered = triggered[i],
        asr_ms               = asr_times[i],
        mt_pass1_ms          = mt1_times[i],
        mt_pass2_ms          = mt2_times[i],
        qe_ms                = qe_times[i],
        tts_ms               = tts_times[i],
        total_pipeline_ms    = total_ms,
        ram_used_mb          = get_ram_mb(),
        temp_c               = get_temp(),
    )
    records.append(rec)
    print(f"  [{i+1}] Total={total_ms:.0f}ms | triggered={triggered[i]}")

for w in wav_files:
    try: os.remove(w)
    except: pass

with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=records[0].keys())
    w.writeheader(); w.writerows(records)
with open(OUTPUT_JSON, "w") as f:
    json.dump(records, f, indent=2)

trig_count = sum(triggered)
avg = round(sum(r["total_pipeline_ms"] for r in records) / len(records), 1)
print(f"\n[SAVED] {len(records)} sentences → {OUTPUT_CSV}")
print(f"Avg latency: {avg}ms | Triggered: {trig_count}/{len(records)}")
