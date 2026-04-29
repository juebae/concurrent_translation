#!/usr/bin/env python3
import time
import torch
from transformers import MarianTokenizer, MarianMTModel, LogitsProcessor, LogitsProcessorList
from pathlib import Path

SPANISH_STOPWORDS = {
    "el","la","los","las","un","una","de","en","y","a","que","es",
    "se","no","por","con","para","su","al","del","lo","le","les",
    "me","te","nos","yo","tu","si","más","pero","como","este",
    "esta","esto","son","fue","ser","estar","hay","ya","todo","bien",
    "durante","entre","sobre","bajo","ante","tras","según","hasta",
    "cuando","donde","porque","aunque","sino","pues","también","muy"
}

class AnchorLogitsProcessor(LogitsProcessor):
    def __init__(self, anchor_ids, boost=2.0):
        self.anchor_ids = anchor_ids
        self.boost = boost

    def __call__(self, input_ids, scores):
        for tid in self.anchor_ids:
            if tid < scores.shape[-1]:
                scores[:, tid] += self.boost
        return scores


class OpusMT:
    def __init__(self, model_snapshot=None):
        if model_snapshot is None:
            cache_root = Path.home() / ".cache/huggingface/hub"
            model_snapshot = str(cache_root / "models--Helsinki-NLP--opus-mt-en-es/snapshots/5bc4493d463cf000c1f0b50f8d56886a392ed4ab")
        self.model_snapshot = model_snapshot
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self.load_time = 0
        self.last_inference_time = 0

    def load(self):
        t0 = time.time()
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_snapshot)
            self.model = MarianMTModel.from_pretrained(self.model_snapshot)
            self.model.eval()
            self.load_time = time.time() - t0
            return True, f"Loaded Opus-MT in {self.load_time:.2f}s on {self.device}"
        except Exception as e:
            return False, f"Failed: {str(e)[:100]}"

    def translate(self, text, num_beams=1):
        if not self.model or not self.tokenizer:
            return False, "", 0
        t0 = time.time()
        try:
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
                outputs = self.model.generate(**inputs, max_length=128, num_beams=num_beams)
                translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            inference_time = (time.time() - t0) * 1000.0
            self.last_inference_time = inference_time
            return True, translation, inference_time
        except Exception as e:
            return False, str(e)[:100], 0

    def translate_nbest(self, text, num_beams=8):
        """Return list of N best translations."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=num_beams,
                    num_return_sequences=num_beams
                )
            translations = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            return True, translations
        except Exception as e:
            return False, [str(e)]

    def extract_anchors(self, translation):
        """Extract content word token IDs from a translation to use as CBS anchors."""
        words = translation.lower()
        for ch in ["¿", "¡", ".", ",", "!", "?", ";", ":"]:
            words = words.replace(ch, "")
        words = words.split()
        content = [w for w in words
                   if w not in SPANISH_STOPWORDS
                   and w.isalpha()
                   and len(w) > 4]
        ids = []
        for word in content:
            ids.extend(self.tokenizer.encode(word, add_special_tokens=False))
        return list(set(ids))

    def translate_nbest_constrained(self, text, anchor_ids, num_beams=5, boost=2.0):
        """N-best beam search with anchor token boosting (CBS)."""
        if not self.model or not self.tokenizer:
            return False, [], 0
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            processor = AnchorLogitsProcessor(anchor_ids, boost=boost)
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    logits_processor=LogitsProcessorList([processor])
                )
            latency = (time.time() - t0) * 1000.0
            candidates = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            return True, candidates, latency
        except Exception as e:
            return False, [], 0

    def get_load_time(self):
        return self.load_time

    def get_last_inference_time(self):
        return self.last_inference_time

    def cleanup(self):
        self.model = None
        self.tokenizer = None
