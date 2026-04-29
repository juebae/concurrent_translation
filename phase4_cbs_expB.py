"""
phase4_cbs_expB.py
Exp B: Strict anchor extraction (capitalised/digits only), boost=1.5
tau=0.90, N=5
"""
import sys, os, json, time, gc
sys.path.insert(0, '/home/zubair/disso')

from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from sacrebleu import corpus_bleu, corpus_chrf

FLORES_PATH  = "/home/zubair/disso/datasets/flores_test/samples.json"
RESULTS_PATH = "/home/zubair/disso/results/cbs_expB_results.json"
QE_THRESHOLD = 0.90
N_BEAMS      = 5

os.makedirs("/home/zubair/disso/results", exist_ok=True)

SPANISH_STOPWORDS = {
    "el","la","los","las","un","una","de","en","y","a","que","es",
    "se","no","por","con","para","su","al","del","lo","le","les",
    "me","te","nos","yo","tu","él","si","más","pero","como","este",
    "esta","esto","son","fue","ser","estar","hay","ya","todo","bien"
}

def extract_anchors_strict(text, max_anchors=3):
    tokens = text.split()
    preferred, fallback = [], []
    for tok in tokens:
        stripped = tok.strip(".,!?;:()\"'«»¿¡")
        low = stripped.lower()
        if len(low) <= 3 or low in SPANISH_STOPWORDS:
            continue
        if stripped[:1].isupper() or any(c.isdigit() for c in stripped):
            preferred.append(stripped)
        else:
            fallback.append(stripped)
    chosen = preferred if preferred else fallback[:2]
    return chosen[:max_anchors]

with open(FLORES_PATH) as f:
    flores = json.load(f)
samples = [{"id": s["id"], "src": s["source_en"], "ref": s["reference_es"]} for s in flores]
print(f"Loaded {len(samples)} samples.")

print("\n=== Beam-1 baseline ===")
mt = OpusMT()
ok, msg = mt.load(); print(msg)
records = []
for s in samples:
    ok1, trans, _ = mt.translate(s["src"])
    records.append({"id": s["id"], "src": s["src"], "ref": s["ref"],
                    "mt_b1": trans if ok1 else ""})
mt.cleanup(); del mt; gc.collect(); time.sleep(3)

print("\n=== QE scoring ===")
qe = QualityEstimation()
ok, msg = qe.load(); print(msg)
for r in records:
    if not r["mt_b1"]:
        r["qe_score"] = None; continue
    ok, score, _ = qe.score(r["src"], r["mt_b1"])
    r["qe_score"] = round(float(score), 4) if ok else None
qe.cleanup(); del qe; gc.collect(); time.sleep(3)

triggered_ids = {r["id"] for r in records
                 if r["qe_score"] is not None and r["qe_score"] < QE_THRESHOLD}
print(f"Triggered: {len(triggered_ids)}/{len(records)}")

print(f"\n=== Generating {N_BEAMS} beams ===")
mt2 = OpusMT()
ok, msg = mt2.load(); print(msg)
for r in records:
    if not r["mt_b1"]:
        r["beams"] = []; continue
    ok2, beams = mt2.translate_nbest(r["src"], num_beams=N_BEAMS)
    r["beams"] = beams if (ok2 and beams) else [r["mt_b1"]]
print("Beams done.")

print("\n=== Running Exp B (strict anchors, boost=1.5) ===")
gc.collect(); time.sleep(3)
qe2 = QualityEstimation()
ok, msg = qe2.load(); print(msg)

hyps_b1 = [r["mt_b1"] for r in records]
hyps_expB = []
for r in records:
    if r["id"] not in triggered_ids or not r["beams"]:
        hyps_expB.append(r["mt_b1"]); continue
    anchors = extract_anchors_strict(r["mt_b1"])
    if not anchors:
        hyps_expB.append(r["mt_b1"]); continue
    ok_cbs, cbs_beams, _ = mt2.translate_nbest_constrained(
        r["src"], anchors, num_beams=N_BEAMS, boost=1.5)
    if not ok_cbs or not cbs_beams:
        hyps_expB.append(r["mt_b1"]); continue
    best, best_s = r["mt_b1"], -1.0
    for cand in cbs_beams:
        ok_q, s, _ = qe2.score(r["src"], cand)
        s = float(s) if ok_q else 0.0
        if s > best_s:
            best_s = s; best = cand
    hyps_expB.append(best)

qe2.cleanup(); del qe2
mt2.cleanup(); del mt2
gc.collect()

refs = [r["ref"] for r in records]
def score(hyps):
    bleu = round(corpus_bleu(hyps, [refs]).score, 2)
    chrf = round(corpus_chrf(hyps, [refs]).score * 100, 2)
    return bleu, chrf

bleu_b1, chrf_b1 = score(hyps_b1)
bleu_b,  chrf_b  = score(hyps_expB)

print(f"\n{'Method':<25} {'BLEU':>8} {'ΔBLEU':>8} {'ChrF':>8} {'ΔChrF':>8}")
print("─" * 58)
print(f"{'Baseline':<25} {bleu_b1:>8.2f} {'—':>8} {chrf_b1:>8.2f} {'—':>8}")
print(f"{'CBS ExpB strict+1.5':<25} {bleu_b:>8.2f} {bleu_b-bleu_b1:>+8.2f} {chrf_b:>8.2f} {chrf_b-chrf_b1:>+8.2f}")

results = {
    "experiment": "B",
    "config": {"threshold": QE_THRESHOLD, "n_beams": N_BEAMS,
               "anchor": "strict_caps_digits", "boost": 1.5,
               "triggered": len(triggered_ids)},
    "baseline": {"bleu": bleu_b1, "chrf": chrf_b1},
    "expB":     {"bleu": bleu_b,  "chrf": chrf_b,
                 "bleu_delta": round(bleu_b-bleu_b1,2),
                 "chrf_delta": round(chrf_b-chrf_b1,2)},
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {RESULTS_PATH}")
