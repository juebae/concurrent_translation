#!/usr/bin/env python3
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

class QualityEstimation:
    def __init__(self, model_type="mbert", model_snapshot=None):
        self.model_type = model_type

        self.device = "cuda"
        if model_snapshot is None:
            if model_type == "mbert":
                model_snapshot = str(Path.home() / ".cache/huggingface/hub/models--bert-base-multilingual-cased/snapshots/mbert_pytorch_final")
            elif model_type == "tinybert":
                cache_root = Path.home() / ".cache/huggingface/hub"
                model_snapshot = str(cache_root / "models--huawei-noah--TinyBERT_General_6L_768D/snapshots/8b6152f3be8ab89055dea2d040cebb9591d97ef6")
            elif model_type == "comet-da":
                model_snapshot = "Unbabel/wmt21-comet-da"
        self.model_snapshot = model_snapshot
        self.tokenizer = None
        self.model = None
        self.load_time = 0
        self.last_inference_time = 0

    def load(self):
        t0 = time.time()
        try:
            if self.model_type == "mbert":
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_snapshot)
                self.model = AutoModel.from_pretrained(self.model_snapshot)
                self.model.to(self.device)
                self.model.eval()
            else:
                from transformers import AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_snapshot)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_snapshot)
                self.model.to(self.device)
                self.model.eval()
            self.load_time = time.time() - t0
            return True, f"Loaded {self.model_type} in {self.load_time:.2f}s on {self.device}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Failed: {str(e)[:100]}"

    def score(self, source_text, target_text):
        if not self.model or not self.tokenizer:
            return False, 0.0, 0
        t0 = time.time()
        try:
            if self.model_type == "mbert":
                src_input = self.tokenizer(source_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
                src_input = {k: v.to(self.device) for k, v in src_input.items()}
                with torch.no_grad():
                    src_output = self.model(**src_input)
                    src_emb = src_output.last_hidden_state.mean(dim=1).cpu().numpy()
                
                tgt_input = self.tokenizer(target_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
                tgt_input = {k: v.to(self.device) for k, v in src_input.items()}
                with torch.no_grad():
                    tgt_output = self.model(**tgt_input)
                    tgt_emb = tgt_output.last_hidden_state.mean(dim=1).cpu().numpy()
                
                sim = np.dot(src_emb.flatten(), tgt_emb.flatten()) / (np.linalg.norm(src_emb) * np.linalg.norm(tgt_emb) + 1e-8)
                score = max(0.0, min(1.0, (sim + 1) / 2))
                
            else:
                qe_input = f"{source_text} {target_text}"
                
                qe_tokens = self.tokenizer(qe_input, return_tensors="pt", max_length=512, truncation=True, padding=True)
                qe_tokens = {k: v.to(self.device) for k, v in src_input.items()}
                with torch.no_grad():    
                    qe_output = self.model(**qe_tokens)
                    if self.model_type == "tinybert":
                        score = torch.softmax(qe_output.logits[0], dim=0)[1].item()
                    else:
                        score = qe_output.logits[0].item()
                        score = max(0, min(1, score))
            
            inference_time = (time.time() - t0) * 1000.0
            self.last_inference_time = inference_time
            return True, score, inference_time
        except Exception as e:
            return False, 0.0, 0

    def get_load_time(self):
        return self.load_time

    def get_last_inference_time(self):
        return self.last_inference_time

    def cleanup(self):
        self.model = None
        self.tokenizer = None
