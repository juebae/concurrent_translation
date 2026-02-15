#!/usr/bin/env python3
"""Minimal Vosk ASR test for Jetson Nano"""

import os
import json
import wave
from vosk import Model, KaldiRecognizer

MODEL_PATH = "vosk-model-small-en-us-0.15"
AUDIO_FILE = "everytime.wav"  # ← Using your file

print("=" * 60)
print("Vosk ASR Test - Jetson Nano")
print("=" * 60)

# Load model
print(f"Loading model from {MODEL_PATH}...")
model = Model(MODEL_PATH)
print("✓ Model loaded")

# Open audio
wf = wave.open(AUDIO_FILE, "rb")
print(f"✓ Audio: {wf.getframerate()} Hz, {wf.getnchannels()} ch")

# Create recognizer
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetMaxAlternatives(0)
rec.SetWords(False)

# Process audio
print("Processing...")
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

# Show transcription
print("=" * 60)
print("TRANSCRIPTION:")
print(" ".join(results) if results else "[EMPTY]")
print("=" * 60)
