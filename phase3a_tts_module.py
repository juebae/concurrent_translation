#!/usr/bin/env python3
import time
import subprocess

class EspeakNGTTS:
    def __init__(self, language="es"):
        self.language = language
        self.espeak_available = False
        self.aplay_available = False
        self.load_time = 0
        self.last_inference_time = 0
        self._check_availability()

    def _check_availability(self):
        try:
            subprocess.check_output(["which", "espeak-ng"], stderr=subprocess.STDOUT)
            self.espeak_available = True
        except:
            self.espeak_available = False
        try:
            subprocess.check_output(["which", "aplay"], stderr=subprocess.STDOUT)
            self.aplay_available = True
        except:
            self.aplay_available = False

    def load(self):
        if not self.espeak_available:
            return False, "espeak-ng not found"
        return True, "espeak-ng available"

    def synthesize_and_play(self, text, output_file="/tmp/tts_output.wav", play=True):
        if not self.espeak_available:
            return False, "unavailable", 0
        t0 = time.time()
        try:
            cmd_tts = ["espeak-ng", "-v", self.language, "-w", output_file, text]
            subprocess.call(cmd_tts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            played = False
            if play and self.aplay_available:
                try:
                    cmd_play = ["aplay", output_file]
                    subprocess.call(cmd_play, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    played = True
                except:
                    pass
            inference_time = (time.time() - t0) * 1000.0
            self.last_inference_time = inference_time
            msg = f"Synthesized {'and played' if played else 'and saved'}"
            return True, msg, inference_time
        except Exception as e:
            return False, str(e)[:100], 0

    def get_load_time(self):
        return self.load_time

    def get_last_inference_time(self):
        return self.last_inference_time

    def cleanup(self):
        pass
