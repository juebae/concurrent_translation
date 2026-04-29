import sys
sys.path.insert(0, '/home/zubair/disso')
from transformers import MarianMTModel, MarianTokenizer, LogitsProcessor, LogitsProcessorList
import torch

MODEL_PATH = '/home/zubair/disso/models/opus_mt_original'
tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
model = MarianMTModel.from_pretrained(MODEL_PATH)
model.eval()

SPANISH_STOPWORDS = {
    "el","la","los","las","un","una","de","en","y","a","que","es",
    "se","no","por","con","para","su","al","del","lo","le","les",
    "me","te","nos","yo","tu","él","si","más","pero","como","este",
    "esta","esto","son","fue","ser","estar","hay","ya","todo","bien"
}

def extract_anchors(translation, tokenizer):
    words = translation.lower().replace("¿","").replace("¡","").split()
    content = [w.strip(".,!?;:") for w in words
               if w not in SPANISH_STOPWORDS
               and w.isalpha() and len(w) > 3]
    ids = []
    for word in content:
        ids.extend(tokenizer.encode(word, add_special_tokens=False))
    return list(set(ids))

class AnchorLogitsProcessor(LogitsProcessor):
    def __init__(self, anchor_ids, boost=2.0):
        self.anchor_ids = anchor_ids
        self.boost = boost
    def __call__(self, input_ids, scores):
        for tid in self.anchor_ids:
            if tid < scores.shape[-1]:
                scores[:, tid] += self.boost
        return scores

# Test sentence
src = "The hospital was completely overwhelmed with patients during the crisis."

# beam=1 first pass
inputs = tokenizer([src], return_tensors="pt", padding=True)
with torch.no_grad():
    out = model.generate(**inputs, num_beams=1, max_length=128)
beam1 = tokenizer.decode(out[0], skip_special_tokens=True)
print(f"Beam=1: {beam1}")

# extract anchors
anchors = extract_anchors(beam1, tokenizer)
print(f"Anchors ({len(anchors)} token IDs): {anchors}")
print(f"Anchor words: {[tokenizer.decode([a]) for a in anchors]}")

# CBS beam=5
processor = AnchorLogitsProcessor(anchors, boost=2.0)
with torch.no_grad():
    out5 = model.generate(
        **inputs,
        num_beams=5,
        num_return_sequences=5,
        logits_processor=LogitsProcessorList([processor]),
        max_length=128
    )
candidates = [tokenizer.decode(o, skip_special_tokens=True) for o in out5]
print(f"\nCBS candidates:")
for i, c in enumerate(candidates):
    print(f"  [{i+1}] {c}")
