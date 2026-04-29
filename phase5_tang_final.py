import torch, json, time, os
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu import corpus_bleu, corpus_chrf

MODEL_DIR = '/home/zubair/disso/models/tang_nano_v2'
EVAL_FILE = MODEL_DIR + '/flores_100.json'
OUT_DIR   = '/home/zubair/disso/results'
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...")
t0  = time.time()
tok = MarianTokenizer.from_pretrained(MODEL_DIR)
mdl = MarianMTModel.from_pretrained(MODEL_DIR)
mdl.eval()
load_time = time.time() - t0
print(f"Loaded in {load_time:.1f}s")

with open(EVAL_FILE) as f:
    data = json.load(f)
srcs = [d['en'] for d in data]
refs = [d['es'] for d in data]

hyps, lats = [], []
print(f"Translating {len(srcs)} sentences...")
for i, src in enumerate(srcs):
    enc = tok([src], return_tensors='pt', truncation=True, max_length=128)
    t = time.time()
    with torch.no_grad():
        out = mdl.generate(enc['input_ids'],
                           attention_mask=enc['attention_mask'],
                           num_beams=5, max_length=128)
    lat = (time.time() - t) * 1000
    hyp = tok.decode(out[0], skip_special_tokens=True)
    hyps.append(hyp)
    lats.append(lat)
    if i % 10 == 0:
        print(f"  [{i+1:>3}/100] {lat:>6.0f}ms | {hyp[:50]}")

bleu = corpus_bleu(hyps, [refs]).score
chrf = corpus_chrf(hyps, [refs]).score
avg  = sum(lats) / len(lats)
p95  = sorted(lats)[94]

print("")
print("=" * 58)
print("JETSON NANO RESULTS — Tang fine-tuned model")
print("=" * 58)
print(f"  BLEU           : {bleu:.4f}  (Colab ref: 16.91)")
print(f"  ChrF           : {chrf:.4f}  (Colab ref: 46.63)")
print(f"  Delta baseline : {bleu - 0.20:+.4f}")
print(f"  Avg latency    : {avg:.1f} ms")
print(f"  P95 latency    : {p95:.1f} ms")
print(f"  Load time      : {load_time:.1f} s")
print("=" * 58)

with open(OUT_DIR + '/tang_jetson_final.json', 'w') as f:
    json.dump({'bleu': bleu, 'chrf': chrf, 'delta': bleu - 0.20,
               'avg_lat_ms': avg, 'p95_lat_ms': p95,
               'load_time_s': load_time, 'hyps': hyps, 'refs': refs}, f, indent=2)
print(f"Saved to {OUT_DIR}/tang_jetson_final.json")
