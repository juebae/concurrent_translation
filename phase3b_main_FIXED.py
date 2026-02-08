#!/usr/bin/env python3
"""
PHASE 3B - Real-time Microphone Speech Translation Pipeline
Integrates Phase 3A modules with live mic input on Jetson Nano
English → Translation → Text-to-Speech

Architecture:
  Microphone → ASR (Whisper) → MT (Opus-MT) → QE (mBERT) → TTS (espeak-ng)
"""

import sys, time, json, re, os
from pathlib import Path

# Thread/process optimization for Nano
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from phase3a_asr_module import WhisperASR
from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from phase3a_tts_module import EspeakNGTTS
from phase3b_audio_module import MicrophoneAudioCapture

class RealtimeTranslationPipeline:
    """
    Real-time end-to-end speech translation with live microphone input
    Designed for Jetson Nano constraints
    """
    
    def __init__(self, target_language="es", display_devices=False):
        """
        Initialize pipeline with all modules
        
        Args:
            target_language: Target language code (e.g., "es" for Spanish)
            display_devices: If True, list available audio devices
        """
        self.target_language = target_language
        self.mic = MicrophoneAudioCapture(sample_rate=16000)
        
        if display_devices:
            self._show_audio_devices()
        
        self.asr = WhisperASR(model_size="tiny",device= "cuda")
        self.mt = OpusMT()
        self.qe = QualityEstimation(model_type="mbert")
        self.tts = EspeakNGTTS(language=target_language)
        
        self.results = []
        self.session_dir = Path.home() / "disso" / "phase3b_sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _show_audio_devices(self):
        """Display available audio input devices"""
        print("\n" + "="*80)
        print("AVAILABLE AUDIO INPUT DEVICES")
        print("="*80 + "\n")
        
        devices = self.mic.get_device_info()
        if not devices:
            print("No input devices found!")
            return
        
        for dev in devices:
            print(f"[{dev['index']}] {dev['name']}")
            print(f"    Channels: {dev['channels']} | Sample Rate: {dev['rate']} Hz\n")
    
    def load_all_models(self):
        """Load all NLP/ML models"""
        print("="*80)
        print("PHASE 3B - REAL-TIME MICROPHONE SPEECH TRANSLATION")
        print("="*80 + "\n")
        
        print("[STEP 1] Loading ASR Module (Whisper tiny)...")
        success, msg = self.asr.load()
        print(f" {'✓' if success else '✗'} {msg}\n")
        if not success: return False
        
        print("[STEP 2] Loading MT Module (Opus-MT en→target)...")
        success, msg = self.mt.load()
        print(f" {'✓' if success else '✗'} {msg}\n")
        if not success: return False
        
        print("[STEP 3] Loading QE Module (mBERT)...")
        success, msg = self.qe.load()
        print(f" {'✓' if success else '✗'} {msg}\n")
        if not success: return False
        
        print("[STEP 4] Checking TTS Module (espeak-ng)...")
        success, msg = self.tts.load()
        print(f" {'✓' if success else '✗'} {msg}\n")
        
        print("[STEP5] Calibrating Noise Reduction...")
        success, msg = self.mic.calibrate_noise_profile(duration=2.0)
        print(f" {'✓' if success else '✗'} {msg}\n")
        return True
    
    def transcribe_audio(self, audio_array):
        """Transcribe English audio to text"""
        import tempfile
        import time
        
        # Create UNIQUE filename with timestamp (milliseconds)
        timestamp = int(time.time() * 1000)
        temp_file = f"/tmp/recording_{timestamp}.wav"
        
        self.mic.save_audio(audio_array, temp_file)
        
        try:
            success, text, latency = self.asr.transcribe(temp_file, language="en")
            return success, text, latency
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def translate_sentence(self, sentence):
        """Translate English sentence to target language"""
        success, translation, latency = self.mt.translate(sentence)
        return success, translation, latency
    
    def score_translation(self, source, target):
        """Quality estimation score for translation"""
        success, score, latency = self.qe.score(source, target)
        return success, score, latency
    
    def synthesize_audio(self, text):
        """Synthesize and play target language audio"""
        success, message, latency = self.tts.synthesize_and_play(text, play=True)
        return success, message, latency
    
    def run_interactive(self, num_recordings=None):
        """
        Run interactive session with continuous recording
        
        Args:
            num_recordings: Max number of recordings (None = infinite)
        """
        if not self.load_all_models():
            print("ERROR: Failed to load models\n")
            return
        
        print("[STEP 6] Initializing Microphone...\n")
        success, msg = self.mic.start_recording()
        print(f" {'✓' if success else '✗'} {msg}\n")
        
        if not success:
            print("ERROR: Failed to initialize microphone\n")
            return
        
        print("="*80)
        print("INTERACTIVE TRANSLATION SESSION")
        print("="*80 + "\n")
        print("Instructions:")
        print("  • Speak naturally when prompted")
        print("  • System detects speech automatically")
        print("  • Silence >2s ends recording\n")
        print("="*80 + "\n")
        
        recording_num = 0
        session_start = time.time()
        session_results = []
        
        try:
            while True:
                recording_num += 1
                
                if num_recordings and recording_num > num_recordings:
                    print(f"\nReached max recordings ({num_recordings})")
                    break
                
                print(f"\n[Recording #{recording_num}] Press Enter then speak...\n")
                try:
                    input()
                except KeyboardInterrupt:
                    print("\n\nSession interrupted by user")
                    break
                
                # Record until silence
                success, audio_array, duration = self.mic.record_until_silence(timeout_sec=10)
                
                if not success or len(audio_array) == 0:
                    print(" ✗ No speech detected, skipping\n")
                    continue
                
                print(f" ✓ Recorded {duration:.2f} seconds\n")
                
                # ASR: English transcription
                print("[Step A] Transcribing English...")
                asr_success, transcription, asr_latency = self.transcribe_audio(audio_array)
                
                if not asr_success or not transcription.strip():
                    print(f" ✗ Transcription failed\n")
                    continue
                
                print(f" ✓ Transcribed in {asr_latency:.2f}ms")
                print(f" Input: \"{transcription}\"\n")
                
                # Split into sentences and process
                sentences = [s.strip() for s in re.split(r'[.!?]+', transcription) if s.strip()]
                if not sentences:
                    sentences = [transcription]
                
                print(f"Processing {len(sentences)} sentence(s)...\n")
                print("="*80)
                
                pipeline_start = time.time()
                recording_results = []
                
                for idx, sentence in enumerate(sentences, start=1):
                    print(f"[{idx}] Input: {sentence}")
                    
                    # MT: Translation
                    mt_success, translation, mt_latency = self.translate_sentence(sentence)
                    if not mt_success:
                        print(f" ✗ Translation failed\n")
                        continue
                    
                    print(f" Output: {translation}")
                    print(f" MT: {mt_latency:.2f}ms", end="")
                    
                    # QE: Quality estimation
                    qe_success, qe_score, qe_latency = self.score_translation(sentence, translation)
                    qe_score = qe_score if qe_success else 0.0
                    print(f" | QE: {qe_latency:.2f}ms ({qe_score:.3f})", end="")
                    
                    # TTS: Synthesis and playback
                    tts_success, tts_msg, tts_latency = self.synthesize_audio(translation)
                    print(f" | TTS: {tts_latency:.2f}ms\n")
                    
                    recording_results.append({
                        "sentence_id": idx,
                        "source": sentence,
                        "target": translation,
                        "mt_time_ms": mt_latency,
                        "qe_score": qe_score,
                        "qe_time_ms": qe_latency,
                        "tts_time_ms": tts_latency,
                        "audio_played": tts_success,
                    })
                
                pipeline_time = (time.time() - pipeline_start) * 1000.0
                
                print("="*80)
                print(f"Recording #{recording_num} Complete")
                print(f" Pipeline time: {pipeline_time:.2f}ms ({len(recording_results)} sentences)")
                
                if recording_results:
                    avg_mt = sum(r["mt_time_ms"] for r in recording_results) / len(recording_results)
                    avg_qe = sum(r["qe_time_ms"] for r in recording_results) / len(recording_results)
                    avg_tts = sum(r["tts_time_ms"] for r in recording_results) / len(recording_results)
                    print(f" Average latencies - MT: {avg_mt:.2f}ms | QE: {avg_qe:.2f}ms | TTS: {avg_tts:.2f}ms")
                
                session_results.append({
                    "recording_id": recording_num,
                    "audio_duration_sec": duration,
                    "num_sentences": len(recording_results),
                    "pipeline_time_ms": pipeline_time,
                    "results": recording_results,
                })
                
                print()
        
        except KeyboardInterrupt:
            print("\n\nSession interrupted")
        
        finally:
            self.mic.stop_recording()
            session_time = time.time() - session_start
            
            self._print_session_summary(recording_num, session_time, session_results)
            self._save_session(recording_num, session_time, session_results)
            self.cleanup()
    
    def _print_session_summary(self, num_recordings, total_time, results):
        """Print session summary statistics"""
        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80 + "\n")
        
        print("Model Load Times:")
        print(f" ASR: {self.asr.get_load_time():.2f}s on {self.asr.get_device_name()}")
        print(f" MT: {self.mt.get_load_time():.2f}s")
        print(f" QE: {self.qe.get_load_time():.2f}s")
        total_load = (self.asr.get_load_time() + 
                     self.mt.get_load_time() + 
                     self.qe.get_load_time())
        print(f" Total: {total_load:.2f}s\n")
        
        print("Recordings Processed:")
        print(f" Total: {num_recordings}")
        print(f" Session Time: {total_time:.2f}s\n")
        
        if results:
            total_sentences = sum(r["num_sentences"] for r in results)
            total_pipeline = sum(r["pipeline_time_ms"] for r in results)
            avg_audio_dur = sum(r["audio_duration_sec"] for r in results) / len(results)
            
            print("Processing Statistics:")
            print(f" Total Sentences: {total_sentences}")
            print(f" Avg Audio Duration: {avg_audio_dur:.2f}s")
            print(f" Total Pipeline Time: {total_pipeline:.2f}ms\n")
        
        print("="*80)
        print("✓ SESSION COMPLETE")
        print("="*80 + "\n")
    
    def _save_session(self, num_recordings, total_time, results):
        """Save session results to JSON"""
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        session_file = self.session_dir / f"phase3b_session_{timestamp}.json"
        
        total_load = (self.asr.get_load_time() + 
                     self.mt.get_load_time() + 
                     self.qe.get_load_time())
        
        session_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "3b_realtime_mic",
            "architecture": "Mic → ASR → MT → QE → TTS",
            "target_language": self.target_language,
            "num_recordings": num_recordings,
            "total_session_time_sec": total_time,
            "model_loads_sec": {
                "asr": self.asr.get_load_time(),
                "mt": self.mt.get_load_time(),
                "qe": self.qe.get_load_time(),
                "total": total_load,
            },
            "recordings": results,
        }
        
        with open(str(session_file), "w") as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session saved to: {session_file}\n")
    
    def cleanup(self):
        """Cleanup all resources"""
        self.mic.cleanup()
        self.asr.cleanup()
        self.mt.cleanup()
        self.qe.cleanup()
        self.tts.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 3B Realtime Speech Translation")
    parser.add_argument("--target-lang", default="es", help="Target language (default: es)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--num-recordings", type=int, default=None, help="Max recordings (default: infinite)")
    
    args = parser.parse_args()
    
    pipeline = RealtimeTranslationPipeline(
        target_language=args.target_lang,
        display_devices=args.list_devices
    )
    
    if args.list_devices:
        sys.exit(0)
    
    pipeline.run_interactive(num_recordings=args.num_recordings)
