#!/usr/bin/env python3
import time
import torch
from transformers import MarianTokenizer, MarianMTModel
from pathlib import Path

class OpusMT:
    def __init__(self, model_snapshot=None):
        if model_snapshot is None:
            cache_root = Path.home() / ".cache/huggingface/hub"
            model_snapshot = str(cache_root / "models--Helsinki-NLP--opus-mt-en-es/snapshots/5bc4493d463cf000c1f0b50f8d56886a392ed4ab")
        self.model_snapshot = model_snapshot
        self.device = "cuda"
        self.tokenizer = None
        self.model = None
        self.load_time = 0
        self.last_inference_time = 0

    def load(self):
        t0 = time.time()
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_snapshot)
            self.model = MarianMTModel.from_pretrained(self.model_snapshot)
            self.model.to(self.device)
            self.model.eval()
            self.load_time = time.time() - t0
            return True, f"Loaded Opus-MT in {self.load_time:.2f}s on {self.device}"
        except Exception as e:
            return False, f"Failed: {str(e)[:100]}"

    def translate(self, text):
        if not self.model or not self.tokenizer:
            return False, "", 0
        t0 = time.time()
        try:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding = True).to(self.device)
                
                outputs = self.model.generate(**inputs, max_length=128)
                translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            inference_time = (time.time() - t0) * 1000.0
            self.last_inference_time = inference_time
            return True, translation, inference_time
        except Exception as e:
            return False, str(e)[:100], 0

    def get_load_time(self):
        return self.load_time

    def get_last_inference_time(self):
        return self.last_inference_time

    def cleanup(self):
        self.model = None
        self.tokenizer = None
