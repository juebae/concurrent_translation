"""
phase4_best_config.py
Re-run best config (tau=0.90, N=5) and save per-sample hypotheses
for B1, M1, M2, M3 (CBS) on FLORES, for COMET/GER scoring.
"""

import sys, os, json, time, gc
sys.path.insert(0, '/home/zubair/disso')

from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from sacrebleu import corpus_bleu, corpus_chrf

FLORES_PATH  = "/home/zubair/disso/datasets/flores_test/samples.json"
OUTPUT_PATH  = "/home/zubair/disso/results/best_config_hypotheses.json"

THRESHOLD = 0.90
N_BEAMS   = 5

os.makedirs("/home/zubair/disso/results", exist_ok=True)

# ── Load FLORES samples ──────────────────────────────────────
with open(FLORES_PATH) as f:
    flores = json.load(f)
samples = [{"id": s["id"], "src": s["source_en"], "ref": s["reference_es"]} for s in flores]
print(f"Loaded {len(samples)} FLORES samples.")

#  Step 1: Beam=1 baseline
print("\n=== Beam-1 baseline ===")
mt = OpusMT()
ok, msg = mt.load(); print(msg)
if not ok:
    raise SystemExit("Failed to load MT")

records = []
for s in samples:
    ok1, trans, _ = mt.translate(s["src"])
    records.append({
        "id":  s["id"],
        "src": s["src"],
        "ref": s["ref"],
        "mt_b1": trans if ok1 else "",
    })

mt.cleanup(); del mt
gc.collect(); time.sleep(3)
print("Baseline translations done.")

#  Step 2: QE score all baseline outputs
print("\n=== QE scoring ===")
qe = QualityEstimation()
ok, msg = qe.load(); print(msg)
if not ok:
    raise SystemExit("Failed to load QE")

for r in records:
    if not r["mt_b1"]:
        r["qe_score"] = None
        continue
    ok_qe, score, _ = qe.score(r["src"], r["mt_b1"])
    r["qe_score"] = round(float(score), 4) if ok_qe else None

qe.cleanup(); del qe
gc.collect(); time.sleep(3)

triggered_ids = {
    r["id"] for r in records
    if r["qe_score"] is not None and r["qe_score"] < THRESHOLD
}
print(f"Triggered (QE < {THRESHOLD}): {len(triggered_ids)}/{len(records)}")

#  Step 3: Generate N-beam candidates for all samples 
print(f"\n=== Generating {N_BEAMS} beams for all samples ===")
mt2 = OpusMT()
ok, msg = mt2.load(); print(msg)
if not ok:
    raise SystemExit("Failed to load MT for beams")

for r in records:
    if not r["mt_b1"]:
        r["beams"] = []
        continue
    ok2, beams = mt2.translate_nbest(r["src"], num_beams=N_BEAMS)
    r["beams"] = beams if (ok2 and beams) else [r["mt_b1"]]

print("Beam generation done.")

#  Helper: MBR selector (Method 2) 
def mbr_select(candidates):
    if len(candidates) == 1:
        return candidates[0]
    best_text, best_score = candidates[0], -1.0
    for i, hyp in enumerate(candidates):
        peers = [c for j, c in enumerate(candidates) if j != i]
        avg = sum(corpus_chrf([hyp], [[p]]).score for p in peers) / len(peers)
        if avg > best_score:
            best_score = avg
            best_text = hyp
    return best_text

#  Step 4: Compute M1, M2, M3 at best config
print("\n=== Computing M1 (QE rerank), M2 (MBR), M3 (CBS) ===")
gc.collect(); time.sleep(3)
qe2 = QualityEstimation()
ok, msg = qe2.load(); print(msg)
if not ok:
    raise SystemExit("Failed to load QE for reranking")

for r in records:
    cond = (r["id"] in triggered_ids and r["beams"])

    # Method 1: QE rerank over precomputed beams
    if cond:
        best_m1, best_score_m1 = r["mt_b1"], -1.0
        for beam in r["beams"]:
            ok_qe, score, _ = qe2.score(r["src"], beam)
            if ok_qe and float(score) > best_score_m1:
                best_score_m1 = float(score)
                best_m1 = beam
        r["mt_m1"] = best_m1
    else:
        r["mt_m1"] = r["mt_b1"]

    # Method 2: MBR over precomputed beams
    r["mt_m2"] = mbr_select(r["beams"]) if cond else r["mt_b1"]

    # Method 3: CBS (original version: mt2.extract_anchors + constrained beams + QE rerank)
    if cond:
        anchors = mt2.extract_anchors(r["mt_b1"])
        if anchors:
            ok_cbs, cbs_beams, _ = mt2.translate_nbest_constrained(
                r["src"], anchors, num_beams=N_BEAMS, boost=2.0
            )
            if ok_cbs and cbs_beams:
                best_m3, best_score_m3 = r["mt_b1"], -1.0
                for cand in cbs_beams:
                    ok_qe, s, _ = qe2.score(r["src"], cand)
                    s = float(s) if ok_qe else 0.0
                    if s > best_score_m3:
                        best_score_m3 = s
                        best_m3 = cand
                r["mt_m3"] = best_m3
            else:
                r["mt_m3"] = r["mt_b1"]
        else:
            r["mt_m3"] = r["mt_b1"]
    else:
        r["mt_m3"] = r["mt_b1"]

qe2.cleanup(); del qe2
mt2.cleanup(); del mt2
gc.collect()

#  Step 5: Sanity BLEU/ChrF check 
refs     = [r["ref"]   for r in records]
hyps_b1  = [r["mt_b1"] for r in records]
hyps_m1  = [r["mt_m1"] for r in records]
hyps_m2  = [r["mt_m2"] for r in records]
hyps_m3  = [r["mt_m3"] for r in records]

def score(hyps):
    bleu = round(corpus_bleu(hyps, [refs]).score, 2)
    chrf = round(corpus_chrf(hyps, [refs]).score * 100, 2)
    return bleu, chrf

bleu_b1, chrf_b1 = score(hyps_b1)
bleu_m1, chrf_m1 = score(hyps_m1)
bleu_m2, chrf_m2 = score(hyps_m2)
bleu_m3, chrf_m3 = score(hyps_m3)

print("\n Sanity check (FLORES, tau=0.90, N=5) ")
print(f"{'Method':<8} {'BLEU':>8} {'ΔBLEU':>8} {'ChrF':>8} {'ΔChrF':>8}")
print("─" * 50)
for name, bleu, chrf in [
    ("B1",  bleu_b1, chrf_b1),
    ("M1",  bleu_m1, chrf_m1),
    ("M2",  bleu_m2, chrf_m2),
    ("M3",  bleu_m3, chrf_m3),
]:
    print(f"{name:<8} {bleu:>8.2f} {bleu-bleu_b1:>+8.2f} {chrf:>8.2f} {chrf-chrf_b1:>+8.2f}")

#  Step 6: Save per-sample hypotheses 
out = []
for r in records:
    out.append({
        "id":     r["id"],
        "src":    r["src"],
        "ref":    r["ref"],
        "mt_b1":  r["mt_b1"],
        "mt_m1":  r["mt_m1"],
        "mt_m2":  r["mt_m2"],
        "mt_m3":  r["mt_m3"],
    })

with open(OUTPUT_PATH, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved {len(out)} samples to: {OUTPUT_PATH}")
print("Transfer this file to your laptop for COMET/GER scoring.")
