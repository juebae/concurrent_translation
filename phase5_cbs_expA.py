"""
phase4_cbs_expA.py
Exp A: Original anchor extraction, boost sweep [2.0, 1.5, 1.0]
tau=0.90, N=5
"""
import sys, os, json, time, gc
sys.path.insert(0, '/home/zubair/disso')

from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from sacrebleu import corpus_bleu, corpus_chrf

FLORES_PATH  = "/home/zubair/disso/datasets/flores_test/samples.json"
RESULTS_PATH = "/home/zubair/disso/results/cbs_expA_results.json"
QE_THRESHOLD = 0.90
N_BEAMS= 5

os.makedirs("/home/zubair/disso/results", exist_ok=True)

SPANISH_STOPWORDS = {
    "el","la","los","las","un","una","de","en","y","a","que","es",
    "se","no","por","con","para","su","al","del","lo","le","les",
    "me","te","nos","yo","tu","él","si","más","pero","como","este",
    "esta","esto","son","fue","ser","estar","hay","ya","todo","bien"
}

def extract_anchors_original(text):
    words = text.lower().replace("¿","").replace("¡","").split()
    return [w.strip(".,!?;:") for w in words
            if w not in SPANISH_STOPWORDS and w.isalpha() and len(w) > 3]

with open(FLORES_PATH) as f:
    flores = json.load(f)
samples = [{"id": s["id"], "src": s["source_en"], "ref": s["reference_es"]} for s in flores]
print(f"Loaded {len(samples)} samples.")

# Step 1: beam-1 baseline
print("\n=== Beam-1 baseline ===")
mt = OpusMT()
ok, msg = mt.load(); print(msg)
records = []
for s in samples:
    ok1, trans, _ = mt.translate(s["src"])
    records.append({"id": s["id"], "src": s["src"], "ref": s["ref"],
                    "mt_b1": trans if ok1 else ""})
mt.cleanup(); del mt; gc.collect(); time.sleep(3)

# Step 2: QE score
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

# Step 3: N beams
print(f"\n=== Generating {N_BEAMS} beams ===")
mt2 = OpusMT()
ok, msg = mt2.load(); print(msg)
for r in records:
    if not r["mt_b1"]:
        r["beams"] = []; continue
    ok2, beams = mt2.translate_nbest(r["src"], num_beams=N_BEAMS)
    r["beams"] = beams if (ok2 and beams) else [r["mt_b1"]]
print("Beams done.")

# Step 4: CBS with 3 boost values
print("\n=== Running Exp A (boost sweep) ===")
gc.collect(); time.sleep(3)
qe2 = QualityEstimation()
ok, msg = qe2.load(); print(msg)

def run_cbs_qe(records, anchor_fn, boost):
    hyps = []
    for r in records:
        if r["id"] not in triggered_ids or not r["beams"]:
            hyps.append(r["mt_b1"]); continue
        anchors = anchor_fn(r["mt_b1"])
        if not anchors:
            hyps.append(r["mt_b1"]); continue
        ok_cbs, cbs_beams, _ = mt2.translate_nbest_constrained(
            r["src"], anchors, num_beams=N_BEAMS, boost=boost)
        if not ok_cbs or not cbs_beams:
            hyps.append(r["mt_b1"]); continue
        best, best_s = r["mt_b1"], -1.0
        for cand in cbs_beams:
            ok_q, s, _ = qe2.score(r["src"], cand)
            s = float(s) if ok_q else 0.0
            if s > best_s:
                best_s = s; best = cand
        hyps.append(best)
    return hyps

hyps_b1 = [r["mt_b1"] for r in records]
hyps_2_0 = run_cbs_qe(records, extract_anchors_original, boost=2.0)
print("boost=2.0 done")
hyps_1_5 = run_cbs_qe(records, extract_anchors_original, boost=1.5)
print("boost=1.5 done")
hyps_1_0= run_cbs_qe(records, extract_anchors_original, boost=1.0)
print("boost=1.0 done")

qe2.cleanup(); del qe2
mt2.cleanup(); del mt2
gc.collect()

refs = [r["ref"] for r in records]
def score(hyps):
    bleu = round(corpus_bleu(hyps, [refs]).score, 2)
    chrf = round(corpus_chrf(hyps, [refs]).score * 100, 2)
    return bleu, chrf

bleu_b1, chrf_b1= score(hyps_b1)
bleu_20, chrf_20= score(hyps_2_0)
bleu_15, chrf_15 = score(hyps_1_5)
bleu_10, chrf_10 = score(hyps_1_0)

print(f"\n{'Method':<25} {'BLEU':>8} {'ΔBLEU':>8} {'ChrF':>8} {'ΔChrF':>8}")
print("─" * 58)
for name, bleu, chrf in [
    ("Baseline",bleu_b1, chrf_b1),
    ("CBS boost=2.0",bleu_20, chrf_20),
    ("CBS boost=1.5",bleu_15, chrf_15),
    ("CBS boost=1.0",     bleu_10, chrf_10),
]:
    print(f"{name:<25} {bleu:>8.2f} {bleu-bleu_b1:>+8.2f} {chrf:>8.2f} {chrf-chrf_b1:>+8.2f}")

results = {
    "experiment": "A", "config": {"threshold": QE_THRESHOLD, "n_beams": N_BEAMS,"triggered": len(triggered_ids)},
    "baseline":{"bleu": bleu_b1, "chrf": chrf_b1},
    "boost_2_0":{"bleu": bleu_20, "chrf": chrf_20,"bleu_delta": round(bleu_20-bleu_b1,2), "chrf_delta": round(chrf_20-chrf_b1,2)},
    "boost_1_5":{"bleu": bleu_15, "chrf": chrf_15,"bleu_delta": round(bleu_15-bleu_b1,2), "chrf_delta": round(chrf_15-chrf_b1,2)},
    "boost_1_0":{"bleu": bleu_10, "chrf": chrf_10,"bleu_delta": round(bleu_10-bleu_b1,2), "chrf_delta": round(chrf_10-chrf_b1,2)},
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {RESULTS_PATH}")
