#!/usr/bin/env python3
"""
demo_baseline.py — Baseline: continuous record until Enter, then sentence-split + translate
Thread 1: mic recording (continuous)
Thread 2: pipeline (ASR->MT->QE->TTS) per detected sentence
"""
import sys, os, time, gc, csv, json, threading, queue, subprocess, tempfile
sys.path.insert(0, os.path.expanduser("~/disso"))

import numpy as np
import sounddevice as sd
import soundfile as sf

from phase3a_asr_module   import WhisperASR
from phase3a_mt_module    import OpusMT
from phase3a_qe_module    import QualityEstimation

OUTPUT_CSV  = os.path.expanduser("~/disso/results/demo_baseline_log.csv")
OUTPUT_JSON = os.path.expanduser("~/disso/results/demo_baseline_log.json")
os.makedirs(os.path.expanduser("~/disso/results"), exist_ok=True)

SAMPLE_RATE       = 16000
CHUNK_SIZE        = 1024
SILENCE_THRESH    = 0.025   # raise this if still too sensitive
MIN_SENTENCE_SEC  = 1.0     # ignore chunks shorter than this
SILENCE_GAP_SEC   = 1.2     # gap between words that counts as sentence boundary

#  System stats 
def get_ram_mb():
    try:
        lines = {}
        for l in open("/proc/meminfo"):
            if ":" in l:
                k,v = l.split(":")
                lines[k.strip()] = int(v.split()[0])
        return round((lines["MemTotal"]-lines["MemAvailable"])/1024,1)
    except: return 0

def get_temp():
    try:
        return round(int(open("/sys/class/thermal/thermal_zone0/temp").read())/1000,1)
    except: return None

def speak_tts(text):
    t0=time.time()
    try: subprocess.run(["espeak-ng","-v","es","-s","145",text],capture_output=True,timeout=30)
    except: pass
    return round((time.time()-t0)*1000,1)

# Load models
print("\n"+"="*60+"\n BASELINE — loading models\n"+"="*60)
asr = WhisperASR(model_size="tiny")
ok,msg = asr.load(); print(f"[ASR] {msg}")
if not ok: sys.exit("ASR failed")

mt = OpusMT()
ok,msg = mt.load(); print(f"[MT]  {msg}")
if not ok: sys.exit("MT failed")

qe = QualityEstimation()
ok,msg = qe.load(); print(f"[QE]  {msg}")
if not ok: sys.exit("QE failed")

# Shared state 
audio_chunks   = []          # raw recorded chunks (mic thread writes)
stop_mic       = threading.Event()
sentence_queue = queue.Queue()
records        = []
session_start  = time.time()
sentence_num   = [0]
print_lock     = threading.Lock()

# Mic thread: records until Enter is pressed
def mic_thread_fn():
    def callback(indata, frames, time_info, status):
        audio_chunks.append(indata.copy().flatten())
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=CHUNK_SIZE, dtype="float32",
                        callback=callback):
        stop_mic.wait() blocks until Enter pressed

#  Sentence splitter: chops recorded audio by silence gaps 
def split_sentences(chunks, sr=SAMPLE_RATE,
                    silence_thresh=SILENCE_THRESH,
                    silence_gap_sec=SILENCE_GAP_SEC,
                    min_sentence_sec=MIN_SENTENCE_SEC):
    audio = np.concatenate(chunks)
    chunk = int(sr * 0.05)   # 50ms analysis frames
    sentences, current, silent_frames, in_speech = [], [], 0, False
    max_silent = int(silence_gap_sec / 0.05)
    for i in range(0, len(audio), chunk):
        frame = audio[i:i+chunk]
        if len(frame) == 0: continue
        energy = float(np.sqrt(np.mean(frame**2)))
        if energy > silence_thresh:
            in_speech = True
            silent_frames = 0
            current.append(frame)
        elif in_speech:
            current.append(frame)
            silent_frames += 1
            if silent_frames >= max_silent:
                seg = np.concatenate(current)
                if len(seg)/sr >= min_sentence_sec:
                    sentences.append(seg)
                current = []
                in_speech = False
                silent_frames = 0
    if current:
        seg = np.concatenate(current)
        if len(seg)/sr >= min_sentence_sec:
            sentences.append(seg)
    return sentences

#  Pipeline thread: processes sentences from queue 
def pipeline_thread_fn():
    while True:
        item = sentence_queue.get()
        if item is None: break
        audio_seg, seg_idx = item

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_seg, SAMPLE_RATE)

        # ASR
        t0 = time.time()
        res = asr.transcribe(tmp.name, language="en")
        os.unlink(tmp.name)
        ok_asr, transcript, asr_ms = res if len(res)==3 else (*res, round((time.time()-t0)*1000,1))
        asr_ms = round(float(asr_ms),1)
        if not ok_asr or not str(transcript).strip():
            with print_lock: print(f"  [{seg_idx}] ASR empty, skipping")
            continue

        # MT
        ok_mt, translation, mt_ms = mt.translate(transcript, num_beams=1)
        mt_ms = round(float(mt_ms),1)
        if not ok_mt:
            with print_lock: print(f"  [{seg_idx}] MT failed, skipping")
            continue

        # QE
        t0=time.time()
        ok_qe, qe_score, _ = qe.score(transcript, translation)
        qe_ms = round((time.time()-t0)*1000,1)
        qe_score = round(float(qe_score),4) if ok_qe else None

        # TTS
        tts_ms = speak_tts(translation)

        total_ms = round(asr_ms+mt_ms+qe_ms+tts_ms,1)
        wall_s   = round(time.time()-session_start,2)
        ram      = get_ram_mb(); temp = get_temp()

        rec = dict(sentence_id=seg_idx, wall_clock_s=wall_s,
                   transcript=transcript, translation=translation,
                   qe_score=qe_score, correction_triggered=False,
                   asr_ms=asr_ms, mt_pass1_ms=mt_ms, mt_pass2_ms=0,
                   qe_ms=qe_ms, tts_ms=tts_ms,
                   total_pipeline_ms=total_ms,
                   ram_used_mb=ram, temp_c=temp)
        records.append(rec)

        with print_lock:
            print(f"\n  [{seg_idx}] {transcript!r}")
            print(f"         → {translation!r}")
            print(f"         QE={qe_score}  Total={total_ms:.0f}ms  RAM={ram}MB  Temp={temp}°C")
        gc.collect()

# Main
print("\n"+"="*60)
print(" Press ENTER to stop recording and process")
print(" Ctrl+C at any time to cancel")
print("="*60+"\n")

mic_t      = threading.Thread(target=mic_thread_fn, daemon=True)
pipeline_t = threading.Thread(target=pipeline_thread_fn, daemon=True)
mic_t.start()
pipeline_t.start()

try:
    while True:
        print("Recording... speak your sentences. Press ENTER when done.")
        input()   # blocks until Enter
        stop_mic.set()
        break
except KeyboardInterrupt:
    stop_mic.set()

mic_t.join(timeout=2)
print(f"\n[INFO] Recorded {len(audio_chunks)} chunks "
      f"({round(len(audio_chunks)*CHUNK_SIZE/SAMPLE_RATE,1)}s total). Splitting...")

sentences = split_sentences(audio_chunks)
print(f"[INFO] Found {len(sentences)} sentences. Processing...\n")

for i, seg in enumerate(sentences, 1):
    sentence_queue.put((seg, i))

sentence_queue.put(None)
pipeline_t.join()

if records:
    with open(OUTPUT_CSV,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys()); w.writeheader(); w.writerows(records)
    with open(OUTPUT_JSON,"w") as f:
        json.dump(records,f,indent=2)
    avg=round(sum(r["total_pipeline_ms"] for r in records)/len(records),1)
    print(f"\n[SAVED] {len(records)} sentences → {OUTPUT_CSV}")
    print(f"Avg latency: {avg}ms")
else:
    print("[!] No records saved.")
