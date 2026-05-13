"""
phase4_grid_search.py - Grid search over QE threshold and N_BEAMS
with M1 (QE rerank), M2 (MBR), and M3 (CBS)
"""

import sys, os, json, time, gc
import numpy as np
sys.path.insert(0, '/home/zubair/disso')

from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from sacrebleu import corpus_bleu, corpus_chrf

FLORES_PATH  = "/home/zubair/disso/datasets/flores_test/samples.json"
RESULTS_PATH = "/home/zubair/disso/results/grid_search_results.json"
COLAB_PATH = "/home/zubair/disso/results/colab_inputs.json"


#Grid parameters
THRESHOLDS = [0.75, 0.80, 0.85, 0.90]
BEAM_SIZES = [3, 5, 10]

os.makedirs("/home/zubair/disso/results", exist_ok=True)

with open(FLORES_PATH) as f:
    flores = json.load(f)
samples = [{"id": s["id"], "src": s["source_en"], "ref": s["reference_es"]} for s in flores]
print(f"Loaded {len(samples)} FLORES samples.")

# ─ PHASE 1: MT beam=1 baseline (run once, reuse across grid) ─
print("\n=== Beam=1 baseline pass (shared across all grid runs) ===")
mt = OpusMT()
ok, msg = mt.load()
print(msg)

base_results = []
for s in samples:
    t0 = time.time()
    ok1, trans, _ = mt.translate(s["src"])
    lat = (time.time() - t0) * 1000
    base_results.append({
        "id": s["id"],
        "src": s["src"],
        "ref": s["ref"],
        "mt_text_b1": trans if ok1 else "",
        "mt_lat_b1_ms": round(lat, 1),
    })

mt.cleanup()
del mt
gc.collect()
time.sleep(3)
print("Baseline pass done.")

refs= [r["ref"] for r in base_results]
hyps_b1 = [r["mt_text_b1"] for r in base_results]
bleu_baseline = round(corpus_bleu(hyps_b1, [refs]).score, 2)
chrf_baseline = round(corpus_chrf(hyps_b1, [refs]).score * 100, 2)  # fix scale
print(f"Baseline BLEU: {bleu_baseline}  ChrF: {chrf_baseline}")

# ─ PHASE 2: QE score all samples once
print("\n=== QE scoring all samples (shared) ===")
gc.collect()
time.sleep(3)
qe = QualityEstimation()
ok, msg = qe.load()
print(msg)

for r in base_results:
    if not r["mt_text_b1"]:
        r["qe_score"] = None
        continue
    ok, score, _ = qe.score(r["src"], r["mt_text_b1"])
    r["qe_score"] = round(float(score), 4) if ok else None

qe.cleanup()
del qe
gc.collect()
time.sleep(3)
print("QE scoring done.")

grid_results = []

for N_BEAMS in BEAM_SIZES:

    # Generate N-beam candidates for ALL samples (reuse across thresholds)
    print(f"\n=== Generating {N_BEAMS} beams for all samples ===")
    gc.collect()
    time.sleep(3)
    mt2 = OpusMT()
    ok, msg = mt2.load()
    print(msg)

    for r in base_results:
        if not r["mt_text_b1"]:
            r[f"beams_{N_BEAMS}"] = []
            continue
        ok2, beams = mt2.translate_nbest(r["src"], num_beams=N_BEAMS)
        r[f"beams_{N_BEAMS}"] = beams if (ok2 and beams) else [r["mt_text_b1"]]

    print(f"Beam generation done for N={N_BEAMS}.")

    for THRESHOLD in THRESHOLDS:
        low_qe = [r for r in base_results
                  if r["qe_score"] is not None and r["qe_score"] < THRESHOLD]
        triggered = len(low_qe)

        # Shared QE for M1 and M3
        gc.collect()
        time.sleep(3)
        qe3 = QualityEstimation()
        ok, msg = qe3.load()
        print(f"Loaded QE for threshold={THRESHOLD}, N_BEAMS={N_BEAMS}")

        def mbr_select(candidates):
            if len(candidates) == 1:
                return candidates[0]
            best_text, best_score = candidates[0], -1
            for i, hyp in enumerate(candidates):
                peers = [c for j, c in enumerate(candidates) if j != i]
                avg = sum(corpus_chrf([hyp], [[p]]).score for p in peers) / len(peers)
                if avg > best_score:
                    best_score = avg
                    best_text = hyp
            return best_text

        hyps_m1 = []
        hyps_m2 = []
        hyps_m3 = []

        for r in base_results:
            cond = (
                r["qe_score"] is not None
                and r["qe_score"] < THRESHOLD
                and r[f"beams_{N_BEAMS}"]
            )

            # Method 1: QE rerank over precomputed beams
            if cond:
                best_text_m1, best_score_m1 = r["mt_text_b1"], -1.0
                for beam in r[f"beams_{N_BEAMS}"]:
                    ok3, score, _ = qe3.score(r["src"], beam)
                    if ok3 and float(score) > best_score_m1:
                        best_score_m1 = float(score)
                        best_text_m1 = beam
            else:
                best_text_m1 = r["mt_text_b1"]
            hyps_m1.append(best_text_m1)

            # Method 2: MBR over precomputed beams
            if cond:
                best_text_m2 = mbr_select(r[f"beams_{N_BEAMS}"])
            else:
                best_text_m2 = r["mt_text_b1"]
            hyps_m2.append(best_text_m2)

            # Method 3: CBS (constrained n-best + QE rerank)
            if cond:
                anchors = mt2.extract_anchors(r["mt_text_b1"])
                if anchors:
                    ok_cbs, cbs_beams, _ = mt2.translate_nbest_constrained(
                        r["src"], anchors, num_beams=N_BEAMS, boost=2.0
                    )
                    if ok_cbs and cbs_beams:
                        best_text_m3, best_score_m3 = r["mt_text_b1"], -1.0
                        for cand in cbs_beams:
                            ok4, s, _ = qe3.score(r["src"], cand)
                            s = float(s) if ok4 else 0.0
                            if s > best_score_m3:
                                best_score_m3 = s
                                best_text_m3 = cand
                    else:
                        best_text_m3 = r["mt_text_b1"]
                else:
                    best_text_m3 = r["mt_text_b1"]
            else:
                best_text_m3 = r["mt_text_b1"]
            hyps_m3.append(best_text_m3)

        qe3.cleanup()
        del qe3
        gc.collect()
        time.sleep(3)

        bleu_m1 = round(corpus_bleu(hyps_m1, [refs]).score, 2)
        bleu_m2 = round(corpus_bleu(hyps_m2, [refs]).score, 2)
        bleu_m3 = round(corpus_bleu(hyps_m3, [refs]).score, 2)
        chrf_m1 = round(corpus_chrf(hyps_m1, [refs]).score * 100, 2)
        chrf_m2 = round(corpus_chrf(hyps_m2, [refs]).score * 100, 2)
        chrf_m3 = round(corpus_chrf(hyps_m3, [refs]).score * 100, 2)

        # Save hypotheses for COMET on Colab later
        hyp_out = {
            "sources": [r["src"] for r in base_results],
            "references": refs,
            "baseline": hyps_b1,
            "m1": hyps_m1,
            "m2": hyps_m2,
            "m3": hyps_m3,
        }
        hyp_path = f"/home/zubair/disso/results/hypotheses_tau{THRESHOLD}_n{N_BEAMS}.json"
        with open(hyp_path, "w", encoding="utf-8") as f:
            json.dump(hyp_out, f, ensure_ascii=False, indent=2)
        print(f"Saved hypotheses to {hyp_path}")

        row = {
            "threshold": THRESHOLD,
            "n_beams": N_BEAMS,
            "triggered": triggered,
            "trigger_pct": round(triggered / len(base_results) * 100, 1),
            "bleu_baseline": bleu_baseline,
            "bleu_m1": bleu_m1,
            "bleu_m2": bleu_m2,
            "bleu_m3": bleu_m3,
            "bleu_delta_m1": round(bleu_m1 - bleu_baseline, 2),
            "bleu_delta_m2": round(bleu_m2 - bleu_baseline, 2),
            "bleu_delta_m3": round(bleu_m3 - bleu_baseline, 2),
            "chrf_baseline": chrf_baseline,
            "chrf_m1": chrf_m1,
            "chrf_m2": chrf_m2,
            "chrf_m3": chrf_m3,
            "chrf_delta_m1": round(chrf_m1 - chrf_baseline, 2),
            "chrf_delta_m2": round(chrf_m2 - chrf_baseline, 2),
            "chrf_delta_m3": round(chrf_m3 - chrf_baseline, 2),
        }
        grid_results.append(row)

        print(
            f"  thresh={THRESHOLD} beams={N_BEAMS} | triggered={triggered} "
            f"| BLEU M1={bleu_m1}({row['bleu_delta_m1']:+}) "
            f"M2={bleu_m2}({row['bleu_delta_m2']:+}) "
            f"M3={bleu_m3}({row['bleu_delta_m3']:+}) "
            f"| ChrF M1={chrf_m1}({row['chrf_delta_m1']:+}) "
            f"M2={chrf_m2}({row['chrf_delta_m2']:+}) "
            f"M3={chrf_m3}({row['chrf_delta_m3']:+})"
        )

    mt2.cleanup()
    del mt2
    gc.collect()
    time.sleep(3)


# -- SAVE 1: grid_search_results.json (BLEU/ChrF scores only) ----------
# Strip hyps before saving to keep this file small and readable
grid_results_clean = [
    {k: v for k, v in r.items() if k not in ("hyps_m1", "hyps_m2", "hyps_m3")}
    for r in grid_results
]
with open(RESULTS_PATH, "w") as f:
    json.dump(
        {
            "baseline_bleu": bleu_baseline,
            "baseline_chrf": chrf_baseline,
            "grid": grid_results_clean,
        },
        f,
        indent=2,
    )
print(f"Grid results saved to: {RESULTS_PATH}")

# -- SAVE 2: colab_inputs.json (raw translations for COMET) ------------
colab_data = {
    "sources":       [r["src"]         for r in base_results],
    "references":    [r["ref"]         for r in base_results],
    "baseline_hyps": [r["mt_text_b1"]  for r in base_results],
    "qe_scores":     [r["qe_score"]    for r in base_results],
    "all_beams_N5":  [r.get("beams_5", [r["mt_text_b1"]]) for r in base_results],
    "grid_hyps":     {},
}
for r in grid_results:
    tau = r["threshold"]
    n   = r["n_beams"]
    colab_data["grid_hyps"][f"M1_tau{tau}_N{n}"] = r["hyps_m1"]
    colab_data["grid_hyps"][f"M2_tau{tau}_N{n}"] = r["hyps_m2"]
    colab_data["grid_hyps"][f"M3_tau{tau}_N{n}"] = r["hyps_m3"]

with open(COLAB_PATH, "w") as f:
    json.dump(colab_data, f, indent=2)
print(f"Colab inputs saved to: {COLAB_PATH}")

print("\n=== GRID SEARCH COMPLETE ===")
print(
    f"{'Thresh':>8} {'Beams':>6} {'Trig%':>6} "
    f"{'BLEU-B':>8} {'BLEU-M1':>9} {'BLEU-M2':>9} {'BLEU-M3':>9} "
    f"{'ChrF-B':>8} {'ChrF-M1':>9} {'ChrF-M2':>9} {'ChrF-M3':>9}"
)
for r in grid_results:
    print(
        f"{r['threshold']:>8} {r['n_beams']:>6} {r['trigger_pct']:>5}% "
        f"{r['bleu_baseline']:>8} "
        f"{r['bleu_m1']:>8}({r['bleu_delta_m1']:+.2f}) "
        f"{r['bleu_m2']:>8}({r['bleu_delta_m2']:+.2f}) "
        f"{r['bleu_m3']:>8}({r['bleu_delta_m3']:+.2f}) "
        f"{r['chrf_baseline']:>8} "
        f"{r['chrf_m1']:>8}({r['chrf_delta_m1']:+.2f}) "
        f"{r['chrf_m2']:>8}({r['chrf_delta_m2']:+.2f}) "
        f"{r['chrf_m3']:>8}({r['chrf_delta_m3']:+.2f})"
    )
print(f"\nSaved to: {RESULTS_PATH}")
