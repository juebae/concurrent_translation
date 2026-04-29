"""
phase4_cbs_expC.py
Exp C: Hybrid — strict CBS candidates merged with unconstrained beams,
MBR selects over the combined pool.
tau=0.90, N=5
"""
import sys, os, json, time, gc
sys.path.insert(0, '/home/zubair/disso')

from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from sacrebleu import corpus_bleu, corpus_chrf

FLORES_PATH  = "/home/zubair/disso/datasets/flores_test/samples.json"
RESULTS_PATH = "/home/zubair/disso/results/cbs_expC_results.json"
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

def mbr_select(candidates):
    if len(candidates) == 1:
        return candidates[0]
    best_text, best_score = candidates[0], -1.0
    for i, hyp in enumerate(candidates):
        peers = [c for j, c in enumerate(candidates) if j != i]
        avg = sum(corpus_chrf([hyp], [[p]]).score for p in peers) / len(peers)
        if avg > best_score:
            best_score = avg; best_text = hyp
    return best_text

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

print("\n=== Running Exp C (hybrid CBS+MBR) ===")
# No QE needed for selection — MBR is model-free

hyps_b1   = [r["mt_b1"] for r in records]
hyps_mbr  = []   # plain MBR reference
hyps_expC = []   # hybrid

for r in records:
    cond = r["id"] in triggered_ids and r["beams"]

    # plain MBR
    hyps_mbr.append(mbr_select(r["beams"]) if cond else r["mt_b1"])

    # hybrid
    if not cond:
        hyps_expC.append(r["mt_b1"]); continue

    anchors = extract_anchors_strict(r["mt_b1"])
    if not anchors:
        hyps_expC.append(mbr_select(r["beams"])); continue

    ok_cbs, cbs_beams, _ = mt2.translate_nbest_constrained(
        r["src"], anchors, num_beams=N_BEAMS, boost=1.5)

    if ok_cbs and cbs_beams:
        combined = list(dict.fromkeys(r["beams"] + cbs_beams))
        hyps_expC.append(mbr_select(combined))
    else:
        hyps_expC.append(mbr_select(r["beams"]))

mt2.cleanup(); del mt2
gc.collect()

refs = [r["ref"] for r in records]
def score(hyps):
    bleu = round(corpus_bleu(hyps, [refs]).score, 2)
    chrf = round(corpus_chrf(hyps, [refs]).score * 100, 2)
    return bleu, chrf

bleu_b1, chrf_b1   = score(hyps_b1)
bleu_mbr, chrf_mbr = score(hyps_mbr)
bleu_c,   chrf_c   = score(hyps_expC)

print(f"\n{'Method':<30} {'BLEU':>8} {'ΔBLEU':>8} {'ChrF':>8} {'ΔChrF':>8}")
print("─" * 62)
print(f"{'Baseline':<30} {bleu_b1:>8.2f} {'—':>8} {chrf_b1:>8.2f} {'—':>8}")
print(f"{'MBR (plain, reference)':<30} {bleu_mbr:>8.2f} {bleu_mbr-bleu_b1:>+8.2f} {chrf_mbr:>8.2f} {chrf_mbr-chrf_b1:>+8.2f}")
print(f"{'CBS ExpC hybrid MBR+CBS':<30} {bleu_c:>8.2f} {bleu_c-bleu_b1:>+8.2f} {chrf_c:>8.2f} {chrf_c-chrf_b1:>+8.2f}")

results = {
    "experiment": "C",
    "config": {"threshold": QE_THRESHOLD, "n_beams": N_BEAMS,
               "anchor": "strict_caps_digits", "boost": 1.5,
               "selection": "mbr_over_combined_pool",
               "triggered": len(triggered_ids)},
    "baseline":       {"bleu": bleu_b1,  "chrf": chrf_b1},
    "mbr_reference":  {"bleu": bleu_mbr, "chrf": chrf_mbr,
                       "bleu_delta": round(bleu_mbr-bleu_b1,2),
                       "chrf_delta": round(chrf_mbr-chrf_b1,2)},
    "expC_hybrid":    {"bleu": bleu_c,   "chrf": chrf_c,
                       "bleu_delta": round(bleu_c-bleu_b1,2),
                       "chrf_delta": round(chrf_c-chrf_b1,2)},
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to: {RESULTS_PATH}")
