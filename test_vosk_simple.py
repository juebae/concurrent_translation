#!/usr/bin/env python3
"""Simple Vosk test - no file checks, just try to load and transcribe"""

import os
import json
import wave
from vosk import Model, KaldiRecognizer

MODEL_PATH = "vosk-model-en-us-0.22-lgraph"
AUDIO_FILE = "test_audio.wav"

print("=" * 60)
print("Vosk Simple Test")
print("=" * 60)

# 1. Load model (Vosk will figure out the format)
print(f"Loading model: {MODEL_PATH}")
try:
    model = Model(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model load failed: {e}")
    exit(1)

# 2. Check audio exists
if not os.path.exists(AUDIO_FILE):
    print(f"✗ Audio file not found: {AUDIO_FILE}")
    print(f"\nRecord audio with:")
    print(f"  arecord -d 5 -f S16_LE -r 16000 -c 1 {AUDIO_FILE}")
    exit(1)

# 3. Open audio
try:
    wf = wave.open(AUDIO_FILE, "rb")
    print(f"✓ Audio: {wf.getframerate()} Hz, {wf.getnchannels()} channel(s)")
except Exception as e:
    print(f"✗ Cannot open audio: {e}")
    exit(1)

# 4. Recognize
print("Processing...")
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetMaxAlternatives(0)
rec.SetWords(False)

results = []
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        if result.get("text"):
            results.append(result["text"])

final = json.loads(rec.FinalResult())
if final.get("text"):
    results.append(final["text"])

wf.close()

# 5. Show result
print("=" * 60)
print("TRANSCRIPTION:")
transcription = " ".join(results).strip()
print(transcription if transcription else "[EMPTY - no speech detected]")
print("=" * 60)

if transcription:
    print("\n✓ SUCCESS - Vosk is working!")
else:
    print("\n⚠ Empty transcription - check audio quality")
