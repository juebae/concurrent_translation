"""
phase3a_asr_module.py - Whisper ASR Module
Interface-compatible replacement for Vosk version
"""

import os
import time
import torch
import whisper


class WhisperASR:

    def __init__(self, model_size="tiny", device=None):
        self.model_size = model_size
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_time = 0.0
        self._last_inference_time = 0.0

    def load(self):
        start_time = time.time()
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.model.eval()
            self._load_time = time.time() - start_time
            return True, f"Loaded Whisper in {self._load_time:.2f}s on {self.device}"
        except Exception as e:
            return False, f"Whisper load failed: {e}"

    def transcribe(self, audio_file, language="en"):
        start_time = time.time()
        try:
            if self.model is None:
                return False, "", 0.0
            if not os.path.exists(audio_file):
                return False, f"File not found: {audio_file}", 0.0
            with torch.no_grad():
                result = self.model.transcribe(audio_file, language=language)
            transcription = result["text"].strip()
            latency_ms = (time.time() - start_time) * 1000
            self._last_inference_time = latency_ms / 1000
            return True, transcription, latency_ms
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return False, str(e), latency_ms

    def get_load_time(self):
        return self._load_time

    def get_last_inference_time(self):
        return self._last_inference_time

    def cleanup(self):
        self.model = None
