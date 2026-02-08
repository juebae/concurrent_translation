#!/usr/bin/env python3
import time
import whisper
import functools
import torch


#whisper.torch.load = functools.partial(torch.load, weights_only = False)

class WhisperASR:
    def __init__(self, model_size="tiny"):
        self.model_size = model_size
        #self.device = device
        self.model = None
        self.load_time = 0
        self.last_inference_time = 0
        
        #self.device_name = device

    def load(self):
        t0 = time.time()
        try:
            self.model = whisper.load_model(self.model_size)
            #self.model.eval()
            self.load_time = time.time() - t0
            #device_name = str(next(self.model.parameters()).device)
            return True, f"Loaded {self.model_size} in {self.load_time:.2f}s on cpu"
        except Exception as e:
            return False, f"Failed: {str(e)[:100]}"

    def transcribe(self, audio_file, language="en"):
        if not self.model:
            return False, "", 0
        t0 = time.time()
        try:
            with torch.no_grad():
                result = self.model.transcribe(audio_file, language=language)
            text = result.get("text", "").strip()
            inference_time = (time.time() - t0) * 1000.0
            self.last_inference_time = inference_time
            return True, text, inference_time
        except Exception as e:
            return False, str(e)[:100], 0

    def get_load_time(self):
        return self.load_time
    
    def get_last_inference_time(self):
        return self.last_inference_time

    def cleanup(self):
        self.model = None
