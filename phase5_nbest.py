#!/usr/bin/env python3
"""
PHASE 5 - Self-Correcting Speech Translation (Method 1: N-best QE Reranking)
Architecture: Mic → ASR → MT(beam=1) → QE gate → MT(beam=5 rerank) → TTS
"""

import sys, time, os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/zubair/disso')

from phase3a_asr_module import WhisperASR
from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from phase3a_tts_module import EspeakNGTTS
from phase3b_audio_module import MicrophoneAudioCapture

QE_THRESHOLD = 0.90
N_BEAMS      = 5

print("=" * 60)
print(" PHASE 5 — N-BEST QE RERANKING")
print(f" QE threshold: {QE_THRESHOLD} | Beams: {N_BEAMS}")
print("=" * 60)

# ── Load all models ───────────────────────────────────────────
print("\n[1] Loading ASR (Whisper tiny)...")
asr = WhisperASR(model_size="tiny")
ok, msg = asr.load()
print(f"    {'✓' if ok else '✗'} {msg}")

print("[2] Loading MT (Opus-MT en→es)...")
mt = OpusMT()
ok, msg = mt.load()
print(f"    {'✓' if ok else '✗'} {msg}")

print("[3] Loading QE (mBERT)...")
qe = QualityEstimation(model_type="mbert")
ok, msg = qe.load()
print(f"    {'✓' if ok else '✗'} {msg}")

print("[4] Loading TTS (espeak-ng)...")
tts = EspeakNGTTS(language="es")
ok, msg = tts.load()
print(f"    {'✓' if ok else '✗'} {msg}")

print("[5] Starting microphone...")
mic = MicrophoneAudioCapture(sample_rate=16000)
ok, msg = mic.calibrate_noise_profile(duration=2.0)
print(f"    {'✓' if ok else '✗'} {msg}")
ok, msg = mic.start_recording()
print(f"    {'✓' if ok else '✗'} {msg}")

print("\n" + "=" * 60)
print(" Ready. Press Enter, speak, pause 2s to end.")
print(" Ctrl+C to quit.")
print("=" * 60)

session_num = 0

try:
    while True:
        session_num += 1
        print(f"\n[Session #{session_num}] Press Enter to speak...")
        try:
            input()
        except KeyboardInterrupt:
            break

        # ── Record ───────────────────────────────────────────
        ok, audio, duration = mic.record_until_silence(timeout_sec=20)
        if not ok or len(audio) == 0:
            print("  ✗ No speech detected.")
            continue
        print(f"  ✓ Recorded {duration:.2f}s")

        # ── ASR ──────────────────────────────────────────────
        timestamp = int(time.time() * 1000)
        tmp = f"/tmp/rec_{timestamp}.wav"
        mic.save_audio(audio, tmp)
        ok, transcript, asr_lat = asr.transcribe(tmp, language="en")
        if not ok or not transcript.strip():
            print("  ✗ Transcription failed.")
            continue
        print(f"\n  ASR  [{asr_lat:.0f}ms]: \"{transcript}\"")

        # ── MT beam=1 first pass ─────────────────────────────
        t0 = time.time()
        ok, translation, mt_lat = mt.translate(transcript)
        if not ok:
            print("  ✗ Translation failed.")
            continue
        print(f"  MT1  [{mt_lat:.0f}ms]: \"{translation}\"")

        # ── QE gate ──────────────────────────────────────────
        ok, qe_score, qe_lat = qe.score(transcript, translation)
        qe_score = float(qe_score) if ok else 0.0
        print(f"  QE   [{qe_lat:.0f}ms]: score={qe_score:.4f} (threshold={QE_THRESHOLD})")

        final_translation = translation
        correction_applied = False

        if qe_score < QE_THRESHOLD:
            print(f"  ⚡ QE below threshold — generating {N_BEAMS} candidates...")

            # ── MT beam=N ─────────────────────────────────────
            t0 = time.time()
            ok2, beams = mt.translate_nbest(transcript, num_beams=N_BEAMS)
            beam_lat = (time.time() - t0) * 1000

            if ok2 and beams:
                print(f"  Beams [{beam_lat:.0f}ms]: {len(beams)} candidates generated")

                # QE score each beam, pick best
                best_text, best_score = beams[0], -1
                for i, beam in enumerate(beams):
                    ok3, score, _ = qe.score(transcript, beam)
                    score = float(score) if ok3 else 0.0
                    print(f"    [{i+1}] score={score:.4f}: \"{beam[:60]}\"")
                    if score > best_score:
                        best_score = score
                        best_text = beam

                final_translation = best_text
                correction_applied = True
                print(f"  ✓ Best candidate (score={best_score:.4f}): \"{final_translation}\"")
        else:
            print(f"  ✓ QE passed — no correction needed.")

        # ── TTS ──────────────────────────────────────────────
        print(f"\n  FINAL: \"{final_translation}\"")
        print(f"  Correction applied: {'Yes' if correction_applied else 'No'}")
        ok, _, tts_lat = tts.synthesize_and_play(final_translation, play=True)
        print(f"  TTS  [{tts_lat:.0f}ms]: played")
        print("=" * 60)

except KeyboardInterrupt:
    pass

finally:
    mic.stop_recording()
    mic.cleanup()
    asr.cleanup()
    mt.cleanup()
    qe.cleanup()
    tts.cleanup()
    print("\nSession ended.")
