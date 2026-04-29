"""
test_vosk.py - Minimal Vosk ASR test for Jetson Nano
Tests vosk-model-small-en-us-0.15 with a 16 kHz mono WAV file.
"""

import sys
import os
import json
import wave
from vosk import Model, KaldiRecognizer

def test_vosk_asr(model_path, audio_file):
    """
    Test Vosk ASR with minimal configuration.
    
    Args:
        model_path: Path to Vosk model directory (e.g., "vosk-model-small-en-us-0.15")
        audio_file: Path to 16 kHz mono WAV file (e.g., "test_audio.wav")
    
    Returns:
        tuple: (success, text_or_error)
    """
    
    # Step 1: Validate model path
    if not os.path.exists(model_path):
        return False, f"Model path does not exist: {model_path}"
    
    required_files = ["am/final.mdl", "graph/HCLG.fst", "graph/words.txt"]
    for rel_path in required_files:
        full_path = os.path.join(model_path, rel_path)
        if not os.path.exists(full_path):
            return False, f"Missing model file: {full_path}"
    
    # Step 2: Validate audio file
    if not os.path.exists(audio_file):
        return False, f"Audio file does not exist: {audio_file}"
    
    try:
        wf = wave.open(audio_file, "rb")
    except Exception as e:
        return False, f"Cannot open WAV file: {str(e)}"
    
    # Step 3: Validate WAV format
    if wf.getnchannels() != 1:
        wf.close()
        return False, f"Audio must be mono (1 channel). Found: {wf.getnchannels()} channels"
    
    if wf.getsampwidth() != 2:
        wf.close()
        return False, f"Audio must be 16-bit PCM (sampwidth=2). Found: {wf.getsampwidth()}"
    
    if wf.getcomptype() != "NONE":
        wf.close()
        return False, f"Audio must be uncompressed PCM. Found: {wf.getcomptype()}"
    
    sample_rate = wf.getframerate()
    if sample_rate != 16000:
        print(f"WARNING: Model expects 16000 Hz, audio is {sample_rate} Hz")
    
    # Step 4: Load Vosk model
    print(f"Loading Vosk model from {model_path}...")
    try:
        model = Model(model_path)
    except Exception as e:
        wf.close()
        return False, f"Failed to load Vosk model: {str(e)}"
    
    # Step 5: Initialize recognizer
    try:
        recognizer = KaldiRecognizer(model, sample_rate)
        recognizer.SetMaxAlternatives(0)  # Disable alternatives for speed
        recognizer.SetWords(True)  # Enable word-level timestamps if needed
    except Exception as e:
        wf.close()
        return False, f"Failed to initialize recognizer: {str(e)}"
    
    # Step 6: Process audio in chunks
    print(f"Processing audio file: {audio_file}")
    full_text = []
    
    try:
        while True:
            data = wf.readframes(4000)  # Read ~0.25s chunks at 16kHz
            if len(data) == 0:
                break
            
            if recognizer.AcceptWaveform(data):
                # Partial result when silence detected
                result = json.loads(recognizer.Result())
                if result.get("text", "").strip():
                    full_text.append(result["text"])
        
        # Get final result (remaining buffer)
        final_result = json.loads(recognizer.FinalResult())
        if final_result.get("text", "").strip():
            full_text.append(final_result["text"])
        
    except Exception as e:
        wf.close()
        return False, f"Error during recognition: {str(e)}"
    finally:
        wf.close()
    
    # Step 7: Return transcription
    transcription = " ".join(full_text).strip()
    
    if not transcription:
        return True, "[EMPTY - No speech detected or recognition failed]"
    
    return True, transcription


def main():
    """Main test function with clear output."""
    
    # Configuration
    MODEL_PATH = "vosk-model-small-en-us-0.15"
    AUDIO_FILE = "test_audio.wav"
    
    print("=" * 60)
    print("Vosk ASR Test for Jetson Nano")
    print("=" * 60)
    
    # Display environment info
    print(f"\nPython version: {sys.version}")
    
    try:
        import vosk
        print(f"Vosk version: {vosk._version_ if hasattr(vosk, '_version_') else 'unknown'}")
    except ImportError:
        print("ERROR: vosk module not found. Install with: pip3 install vosk")
        return 1
    
    print(f"\nModel path: {MODEL_PATH}")
    print(f"Audio file: {AUDIO_FILE}")
    print("-" * 60)
    
    # Run test
    success, result = test_vosk_asr(MODEL_PATH, AUDIO_FILE)
    
    if success:
        print("\n✓ SUCCESS")
        print(f"\nTranscription:\n{result}")
        return 0
    else:
        print("\n✗ FAILED")
        print(f"\nError:\n{result}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
