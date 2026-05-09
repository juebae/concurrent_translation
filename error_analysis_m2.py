import sys, os, json, time, gc
sys.path.insert(0, os.path.expanduser('~/disso'))

from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from sacrebleu import corpus_bleu, corpus_chrf, sentence_chrf

FLORES_PATH  = os.path.expanduser("~/disso/datasets/flores_test/samples.json")
RESULTS_PATH = os.path.expanduser("~/disso/results/error_analysis_results.json")
SUMMARY_PATH = os.path.expanduser("~/disso/results/error_analysis_summary.txt")

QE_THRESHOLD = 0.90
N_BEAMS      = 10        # use N=10 — best BLEU config

os.makedirs(os.path.expanduser("~/disso/results"), exist_ok=True)

#Helpers
def mbr_select(candidates):
    """Select the MBR consensus candidate by peer-ChrF agreement."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else ""
    best_text, best_score = candidates[0], -1.0
    for i, hyp in enumerate(candidates):
        peers = [c for j, c in enumerate(candidates) if j != i]
        avg = sum(corpus_chrf([hyp], [[p]]).score for p in peers) / len(peers)
        if avg > best_score:
            best_score = avg
            best_text = hyp
    return best_text

def sent_chrf(hyp, ref):
    """Per-sentence ChrF score (0–100 scale)."""
    try:
        return round(sentence_chrf(hyp, [ref]).score, 4)
    except Exception:
        try:
            return round(corpus_chrf([hyp], [[ref]]).score * 100, 4)
        except Exception:
            return 0.0

def token_count(text):
    return len(text.strip().split())

def length_bucket(src_text):
    n = token_count(src_text)
    if n <= 10:
        return "short (≤10 tokens)"
    elif n <= 20:
        return "medium (11–20 tokens)"
    else:
        return "long (>20 tokens)"

#Load data 
print("=" * 65)
print(" M2 Error Analysis — qualitative examples for Section 5.5.3")
print("=" * 65)

with open(FLORES_PATH) as f:
    flores = json.load(f)
samples = [{"id": s["id"], "src": s["source_en"], "ref": s["reference_es"]}
           for s in flores]
print(f"\nLoaded {len(samples)} FLORES samples.")

#  Step 1: Beam-1 baseline
print("\n[1/4] Running beam-1 baseline translations...")
mt = OpusMT()
ok, msg = mt.load()
print(f"  {msg}")
if not ok:
    sys.exit("MT load failed")

records = []
for i, s in enumerate(samples):
    ok1, trans, lat = mt.translate(s["src"])
    records.append({
        "id":          s["id"],
        "src":         s["src"],
        "ref":         s["ref"],
        "mt_b1":       trans if ok1 else "",
        "mt_b1_lat_ms": round(float(lat), 1),
        "src_tokens":  token_count(s["src"]),
        "length_bucket": length_bucket(s["src"]),
    })
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(samples)} translated")

mt.cleanup()
del mt
gc.collect()
time.sleep(3)
print(f"  Baseline done.")

# Baseline corpus metrics
refs    = [r["ref"]   for r in records]
hyps_b1 = [r["mt_b1"] for r in records]
bleu_b1 = round(corpus_bleu(hyps_b1, [refs]).score, 2)
chrf_b1 = round(corpus_chrf(hyps_b1, [refs]).score * 100, 2)
print(f"  Baseline BLEU={bleu_b1}  ChrF={chrf_b1}")

# Per-sentence baseline ChrF
for r in records:
    r["chrf_b1"] = sent_chrf(r["mt_b1"], r["ref"])

#Step 2: QE score all baseline outputs
print(f"\n[2/4] QE scoring all {len(samples)} sentences (τ={QE_THRESHOLD})...")
qe = QualityEstimation()
ok, msg = qe.load()
print(f"  {msg}")
if not ok:
    sys.exit("QE load failed")

for i, r in enumerate(records):
    if not r["mt_b1"]:
        r["qe_score"] = None
        r["triggered"] = False
        continue
    ok_q, score, _ = qe.score(r["src"], r["mt_b1"])
    r["qe_score"] = round(float(score), 4) if ok_q else None
    r["triggered"] = (r["qe_score"] is not None and r["qe_score"] < QE_THRESHOLD)
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(samples)} scored")

qe.cleanup()
del qe
gc.collect()
time.sleep(3)

triggered = [r for r in records if r["triggered"]]
print(f"  Triggered: {len(triggered)}/{len(records)} ({100*len(triggered)/len(records):.1f}%)")

#Step 3: M2 MBR for triggered sentences
print(f"\n[3/4] Running M2 (MBR, N={N_BEAMS}) for {len(triggered)} triggered sentences...")
mt2 = OpusMT()
ok, msg = mt2.load()
print(f"  {msg}")
if not ok:
    sys.exit("MT2 load failed")

# Initialise final translation = baseline for all
for r in records:
    r["mt_m2"] = r["mt_b1"]
    r["mt_m2_changed"] = False

for i, r in enumerate(triggered):
    ok_nb, candidates = mt2.translate_nbest(r["src"], num_beams=N_BEAMS)
    if ok_nb and candidates:
        best = mbr_select(candidates)
        r["mt_m2"] = best
        r["mt_m2_changed"] = (best != r["mt_b1"])
        r["all_candidates"] = candidates   # save for inspection
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(triggered)} corrected")

mt2.cleanup()
del mt2
gc.collect()
print(f"  M2 correction done.")

#Step 4: Per-sentence ChrF delta 
print("\n[4/4] Computing per-sentence ChrF deltas...")

changed_count = 0
for r in records:
    r["chrf_m2"]    = sent_chrf(r["mt_m2"], r["ref"])
    r["chrf_delta"] = round(r["chrf_m2"] - r["chrf_b1"], 4)
    if r.get("mt_m2_changed"):
        changed_count += 1

hyps_m2  = [r["mt_m2"] for r in records]
bleu_m2  = round(corpus_bleu(hyps_m2, [refs]).score, 2)
chrf_m2c = round(corpus_chrf(hyps_m2, [refs]).score * 100, 2)

print(f"\n  Corpus results:")
print(f"    Baseline  BLEU={bleu_b1}  ChrF={chrf_b1}")
print(f"    M2        BLEU={bleu_m2}  ChrF={chrf_m2c}  "
      f"(ΔBLEU={round(bleu_m2-bleu_b1,2):+}  ΔChrF={round(chrf_m2c-chrf_b1,2):+})")
print(f"    Sentences with non-identical correction: {changed_count}/{len(triggered)}")

#Analysis
print("\n" + "=" * 65)
print(" RESULTS BY SENTENCE LENGTH")
print("=" * 65)
for bucket in ["short (≤10 tokens)", "medium (11–20 tokens)", "long (>20 tokens)"]:
    grp = [r for r in records if r["length_bucket"] == bucket and r["triggered"]]
    if not grp:
        continue
    avg_delta = round(sum(r["chrf_delta"] for r in grp) / len(grp), 4)
    improved  = sum(1 for r in grp if r["chrf_delta"] > 0)
    same      = sum(1 for r in grp if r["chrf_delta"] == 0)
    degraded  = sum(1 for r in grp if r["chrf_delta"] < 0)
    print(f"\n  {bucket}")
    print(f"    Triggered sentences : {len(grp)}")
    print(f"    Avg ΔChrF           : {avg_delta:+.4f}")
    print(f"    Improved / Same / Degraded : {improved} / {same} / {degraded}")

def print_examples(title, examples, n=5):
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)
    for i, r in enumerate(examples[:n], 1):
        print(f"\n  Example {i}  [id={r['id']}  bucket={r['length_bucket']}]")
        print(f"  SRC      : {r['src']}")
        print(f"  REF      : {r['ref']}")
        print(f"  BASELINE : {r['mt_b1']}")
        print(f"  M2       : {r['mt_m2']}")
        print(f"  QE score : {r['qe_score']}  ChrF_b1={r['chrf_b1']:.2f}  "
              f"ChrF_m2={r['chrf_m2']:.2f}  ΔChrF={r['chrf_delta']:+.4f}")
        if r.get("all_candidates"):
            print(f"  All {len(r['all_candidates'])} candidates:")
            for j, c in enumerate(r["all_candidates"], 1):
                marker = " ← SELECTED" if c == r["mt_m2"] else ""
                print(f"    [{j}] {c}{marker}")

# Best improved (triggered, changed, highest positive delta)
most_improved = sorted(
    [r for r in records if r["triggered"] and r.get("mt_m2_changed") and r["chrf_delta"] > 0],
    key=lambda x: x["chrf_delta"], reverse=True
)

# Identical  triggered but M2 selected same string as baseline
identical = [r for r in records if r["triggered"] and not r.get("mt_m2_changed")]

# Most degraded (triggered, changed, most negative delta)
most_degraded = sorted(
    [r for r in records if r["triggered"] and r.get("mt_m2_changed") and r["chrf_delta"] < 0],
    key=lambda x: x["chrf_delta"]
)

print_examples("TOP-5 MOST IMPROVED  (M2 clearly helped)", most_improved, n=5)
print_examples("TOP-5 IDENTICAL CORRECTIONS  (M2 no-op)", identical, n=5)
print_examples("TOP-5 MOST DEGRADED  (M2 hurt)", most_degraded, n=5)

# Strip all_candidates from non-example records to keep file manageable
save_records = []
example_ids = set(
    r["id"] for r in (most_improved[:5] + identical[:5] + most_degraded[:5])
)
for r in records:
    rec = {k: v for k, v in r.items() if k != "all_candidates"}
    if r["id"] in example_ids and "all_candidates" in r:
        rec["all_candidates"] = r["all_candidates"]
    save_records.append(rec)

output = {
    "config": {
        "qe_threshold": QE_THRESHOLD,
        "n_beams": N_BEAMS,
        "n_samples": len(records),
        "n_triggered": len(triggered),
        "n_changed": changed_count,
    },
    "corpus_metrics": {
        "baseline_bleu": bleu_b1, "baseline_chrf": chrf_b1,
        "m2_bleu": bleu_m2,       "m2_chrf": chrf_m2c,
        "delta_bleu": round(bleu_m2 - bleu_b1, 2),
        "delta_chrf": round(chrf_m2c - chrf_b1, 2),
    },
    "length_breakdown": {},
    "most_improved_ids":  [r["id"] for r in most_improved[:5]],
    "identical_ids":      [r["id"] for r in identical[:5]],
    "most_degraded_ids":  [r["id"] for r in most_degraded[:5]],
    "per_sentence":       save_records,
}

# Length breakdown
for bucket in ["short (≤10 tokens)", "medium (11–20 tokens)", "long (>20 tokens)"]:
    grp = [r for r in records if r["length_bucket"] == bucket and r["triggered"]]
    if grp:
        output["length_breakdown"][bucket] = {
            "n_triggered": len(grp),
            "avg_chrf_delta": round(sum(r["chrf_delta"] for r in grp) / len(grp), 4),
            "n_improved":  sum(1 for r in grp if r["chrf_delta"] > 0),
            "n_same":      sum(1 for r in grp if r["chrf_delta"] == 0),
            "n_degraded":  sum(1 for r in grp if r["chrf_delta"] < 0),
        }

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\n[SAVED] Full results → {RESULTS_PATH}")

#saving results
lines = []
lines.append("M2 Error Analysis Summary")
lines.append("=" * 65)
lines.append(f"Config: τ={QE_THRESHOLD}  N={N_BEAMS}  sentences={len(records)}")
lines.append(f"Triggered: {len(triggered)}  Changed: {changed_count}")
lines.append("")
lines.append(f"Baseline  BLEU={bleu_b1}  ChrF={chrf_b1}")
lines.append(f"M2        BLEU={bleu_m2}  ChrF={chrf_m2c}  "
             f"(ΔBLEU={round(bleu_m2-bleu_b1,2):+}  ΔChrF={round(chrf_m2c-chrf_b1,2):+})")
lines.append("")

for title, examples in [
    ("TOP-5 MOST IMPROVED", most_improved[:5]),
    ("TOP-5 IDENTICAL",     identical[:5]),
    ("TOP-5 MOST DEGRADED", most_degraded[:5]),
]:
    lines.append(title)
    lines.append("-" * 65)
    for i, r in enumerate(examples, 1):
        lines.append(f"[{i}] id={r['id']}  bucket={r['length_bucket']}")
        lines.append(f"  SRC      : {r['src']}")
        lines.append(f"  REF      : {r['ref']}")
        lines.append(f"  BASELINE : {r['mt_b1']}")
        lines.append(f"  M2       : {r['mt_m2']}")
        lines.append(f"  QE={r['qe_score']}  ΔChrF={r['chrf_delta']:+.4f}")
        lines.append("")

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"[SAVED] Plain-text summary → {SUMMARY_PATH}")

print("\n" + "=" * 65)
print(" DONE — transfer both output files to your laptop:")
print(f"   {RESULTS_PATH}")
print(f"   {SUMMARY_PATH}")
print("=" * 65)
