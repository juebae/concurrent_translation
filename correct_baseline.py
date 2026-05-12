import torch, time, os, json
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu import corpus_bleu, corpus_chrf

MODEL_DIR='/home/zubair/disso/models/opus_mt_original'
FLORES= '/home/zubair/disso/models/opus_mt_original/flores_100.json'
PHASE4_RESULTS = '/home/zubair/disso/results/baseline_results.json'
OUT = '/home/zubair/disso/results/correct_baseline.json'
os.makedirs('/home/zubair/disso/results', exist_ok=True)

#Pull WER from Phase 4 results (already computed)
wer_score = None
if os.path.exists(PHASE4_RESULTS):
    with open(PHASE4_RESULTS) as f:
        phase4 = json.load(f)
    wer_score = phase4['summary']['wer']
    print(f"Loaded Phase 4 WER: {wer_score:.4f}")
else:
    print("Phase 4 results not found — WER will be N/A")

#MT-only BLEU/ChrF on clean FLORES text
print("\nLoading original opus-mt-en-es (no fine-tuning)...")
t0  = time.time()
tok = MarianTokenizer.from_pretrained(MODEL_DIR)
mdl = MarianMTModel.from_pretrained(MODEL_DIR)
mdl.eval()
load_time = time.time() - t0
print(f"Loaded in {load_time:.1f}s")

with open(FLORES) as f:
    data = json.load(f)
srcs = [d['en'] for d in data]
refs = [d['es'] for d in data]

print(f"\nFirst 3 FLORES pairs (sanity check):")
for i in range(3):
    print(f"  SRC: {srcs[i][:70]}")
    print(f"  REF: {refs[i][:70]}")
    print()

hyps, lats = [], []
print(f"Translating {len(srcs)} sentences (MT only, clean text)")
for i, src in enumerate(srcs):
    enc = tok([src], return_tensors='pt', truncation=True, max_length=128)
    t   = time.time()
    with torch.no_grad():
        out = mdl.generate(enc['input_ids'],
                           attention_mask=enc['attention_mask'],
                           num_beams=5, max_length=128)
    lat = (time.time()-t)*1000
    hyp = tok.decode(out[0], skip_special_tokens=True)
    hyps.append(hyp); lats.append(lat)
    if i < 3:
        print(f"  HYP: {hyp[:70]}")
    if i % 10 == 0 and i > 0:
        print(f"  [{i+1:>3}/100] {lat:>6.0f}ms")

bleu = corpus_bleu(hyps, [refs]).score
chrf = corpus_chrf(hyps, [refs]).score
avg  = sum(lats)/len(lats)
p95  = sorted(lats)[94]

print("\n" + "="*60)
print("CORRECT BASELINE on Jetson Nano")
print("="*60)
print(f"  WER  (ASR, Phase 4, Whisper Tiny) : {wer_score if wer_score else 'N/A'}")
print(f"  BLEU (MT only, clean text)        : {bleu:.4f}  (Colab: 25.36)")
print(f"  ChrF (MT only, clean text)        : {chrf:.4f}  (Colab: 54.94)")
print(f"  Avg latency                       : {avg:.1f} ms")
print(f"  P95 latency                       : {p95:.1f} ms")
print(f"  Load time                         : {load_time:.1f} s")
print("="*60)
print("NOTE: BLEU/ChrF = MT component only (clean FLORES-100 text)")
print("WER ASR component only (FLEURS speech input)")
print("These are separate evaluations, not a pipeline score")
print("="*60)

with open(OUT, 'w') as f:
    json.dump({
        'wer': wer_score,
        'bleu': bleu, 'chrf': chrf,
        'avg_lat_ms': avg, 'p95_lat_ms': p95,
        'load_time_s': load_time,
        'note': 'BLEU/ChrF on MT only (clean text). WER from Phase 4 ASR.',
        'hyps': hyps, 'refs': refs, 'srcs': srcs
    }, f, indent=2)
print(f"Saved → {OUT}")
