import sys, os, json, time, gc
sys.path.insert(0, '/home/zubair/disso')

from phase3a_mt_module import OpusMT
from phase3a_qe_module import QualityEstimation
from sacrebleu import corpus_bleu, corpus_chrf

FLORES_PATH  = "/home/zubair/disso/datasets/flores_test/samples.json"
RESULTS_PATH = "/home/zubair/disso/results/cbs_experiments_results.json"

QE_THRESHOLD = 0.90
N_BEAMS      = 5

os.makedirs("/home/zubair/disso/results", exist_ok=True)

# Anchor extractors

SPANISH_STOPWORDS = {
    "el","la","los","las","un","una","de","en","y","a","que","es",
    "se","no","por","con","para","su","al","del","lo","le","les",
    "me","te","nos","yo","tu","él","si","más","pero","como","este",
    "esta","esto","son","fue","ser","estar","hay","ya","todo","bien"
}

def extract_anchors_original(text):
    """Original: any content word > 3 chars, lowercased."""
    words = text.lower().replace("¿","").replace("¡","").split()
    return [w.strip(".,!?;:") for w in words
            if w not in SPANISH_STOPWORDS
            and w.isalpha() and len(w) > 3]

def extract_anchors_strict(text, max_anchors=3):
    """
    Strict: prefer capitalised words (names) and tokens with digits (numbers).
    Falls back to long content words if nothing preferred.
    Caps at max_anchors to avoid over-constraining.
    """
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

# MBR selector

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

# Phase 1: Baseline beam-1

with open(FLORES_PATH) as f:
    flores = json.load(f)
samples = [{"id": s["id"], "src": s["source_en"], "ref": s["reference_es"]} for s in flores]
print(f"Loaded {len(samples)} FLORES samples.")

print("\n=== Beam-1 baseline ===")
mt = OpusMT()
ok, msg = mt.load(); print(msg)

records = []
for s in samples:
    ok1, trans, _ = mt.translate(s["src"])
    records.append({
        "id":  s["id"],
        "src": s["src"],
        "ref": s["ref"],
        "mt_b1": trans if ok1 else "",
    })

mt.cleanup(); del mt; gc.collect(); time.sleep(3)
print("Baseline done.")

# Phase 2: QE score all

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
print(f"Triggered: {len(triggered_ids)}/{len(records)} ({100*len(triggered_ids)/len(records):.1f}%)")

# Phase 3: Generate N beams (reused across all experiments)

print(f"\n=== Generating {N_BEAMS} beams ===")
mt2 = OpusMT()
ok, msg = mt2.load(); print(msg)

for r in records:
    if not r["mt_b1"]:
        r["beams"] = []; continue
    ok2, beams = mt2.translate_nbest(r["src"], num_beams=N_BEAMS)
    r["beams"] = beams if (ok2 and beams) else [r["mt_b1"]]

print("Beam generation done.")

# Phase 4: Run all experiments

print("\n=== Running CBS experiments ===")
gc.collect(); time.sleep(3)
qe2 = QualityEstimation()
ok, msg = qe2.load(); print(msg)

def run_cbs(records, anchor_fn, boost, use_hybrid=False, label=""):
    """
    Run CBS for one configuration.
    anchor_fn  : function(mt_b1) -> list of anchor strings
    boost      : float, logit boost for constrained beams
    use_hybrid : if True, merge CBS+unconstrained beams into MBR pool
    """
    hyps = []
    for r in records:
        cond = r["id"] in triggered_ids and r["beams"]

        if not cond:
            hyps.append(r["mt_b1"]); continue

        anchors = anchor_fn(r["mt_b1"])
        if not anchors:
            hyps.append(r["mt_b1"]); continue

        ok_cbs, cbs_beams, _ = mt2.translate_nbest_constrained(
            r["src"], anchors, num_beams=N_BEAMS, boost=boost
        )

        if not ok_cbs or not cbs_beams:
            hyps.append(r["mt_b1"]); continue

        if use_hybrid:
            # Exp C: merge CBS + unconstrained beams, run MBR over combined pool
            combined = list(dict.fromkeys(r["beams"] + cbs_beams))  # deduplicate, preserve order
            hyps.append(mbr_select(combined))
        else:
            # Exp A/B: QE rerank CBS candidates
            best, best_s = r["mt_b1"], -1.0
            for cand in cbs_beams:
                ok_qe, s, _ = qe2.score(r["src"], cand)
                s = float(s) if ok_qe else 0.0
                if s > best_s:
                    best_s = s; best = cand
            hyps.append(best)

    return hyps

# Plain MBR (M2) as our reference winner
hyps_mbr = [mbr_select(r["beams"]) if r["id"] in triggered_ids and r["beams"]
            else r["mt_b1"] for r in records]

# Experiment A — original anchors, 3 boost values
hyps_expA_2_0 = run_cbs(records, extract_anchors_original, boost=2.0, label="ExpA boost=2.0")
hyps_expA_1_5 = run_cbs(records, extract_anchors_original, boost=1.5, label="ExpA boost=1.5")
hyps_expA_1_0 = run_cbs(records, extract_anchors_original, boost=1.0, label="ExpA boost=1.0")

# Experiment B — strict anchors, best boost from Exp A
hyps_expB     = run_cbs(records, extract_anchors_strict,   boost=1.5, label="ExpB strict+1.5")

# Experiment C — hybrid CBS+MBR pool
hyps_expC     = run_cbs(records, extract_anchors_strict,   boost=1.5,
                        use_hybrid=True, label="ExpC hybrid")

qe2.cleanup(); del qe2
mt2.cleanup(); del mt2
gc.collect()

# Metrics

refs    = [r["ref"]   for r in records]
hyps_b1 = [r["mt_b1"] for r in records]

def score(hyps, refs):
    bleu = round(corpus_bleu(hyps, [refs]).score, 2)
    chrf = round(corpus_chrf(hyps, [refs]).score * 100, 2)
    return bleu, chrf

bleu_b1, chrf_b1       = score(hyps_b1,       refs)
bleu_mbr, chrf_mbr     = score(hyps_mbr,       refs)
bleu_a20, chrf_a20     = score(hyps_expA_2_0,  refs)
bleu_a15, chrf_a15     = score(hyps_expA_1_5,  refs)
bleu_a10, chrf_a10     = score(hyps_expA_1_0,  refs)
bleu_b,   chrf_b       = score(hyps_expB,      refs)
bleu_c,   chrf_c       = score(hyps_expC,      refs)

# Results 

rows = [
    ("Baseline (B1)",            bleu_b1,  chrf_b1),
    ("MBR (M2, reference)",      bleu_mbr, chrf_mbr),
    ("CBS ExpA boost=2.0",       bleu_a20, chrf_a20),
    ("CBS ExpA boost=1.5",       bleu_a15, chrf_a15),
    ("CBS ExpA boost=1.0",       bleu_a10, chrf_a10),
    ("CBS ExpB strict+1.5",      bleu_b,   chrf_b),
    ("CBS ExpC hybrid MBR+CBS",  bleu_c,   chrf_c),
]

print(f"\n{'Method':<30} {'BLEU':>8} {'ΔBLEU':>8} {'ChrF':>8} {'ΔChrF':>8}")
print("─" * 66)
for name, bleu, chrf in rows:
    db = round(bleu - bleu_b1, 2)
    dc = round(chrf - chrf_b1, 2)
    marker = " ◀ BEST" if name.startswith("MBR") else ""
    print(f"{name:<30} {bleu:>8.2f} {db:>+8.2f} {chrf:>8.2f} {dc:>+8.2f}{marker}")
# Save

summary = {
    "config": {"threshold": QE_THRESHOLD, "n_beams": N_BEAMS,
               "triggered": len(triggered_ids),
               "trigger_pct": round(100*len(triggered_ids)/len(records), 1)},
    "results": {
        "baseline":       {"bleu": bleu_b1,  "chrf": chrf_b1},
        "mbr_reference":  {"bleu": bleu_mbr, "chrf": chrf_mbr,
                           "bleu_delta": round(bleu_mbr-bleu_b1,2),
                           "chrf_delta": round(chrf_mbr-chrf_b1,2)},
        "expA_boost2_0":  {"bleu": bleu_a20, "chrf": chrf_a20,
                           "bleu_delta": round(bleu_a20-bleu_b1,2),
                           "chrf_delta": round(chrf_a20-chrf_b1,2)},
        "expA_boost1_5":  {"bleu": bleu_a15, "chrf": chrf_a15,
                           "bleu_delta": round(bleu_a15-bleu_b1,2),
                           "chrf_delta": round(chrf_a15-chrf_b1,2)},
        "expA_boost1_0":  {"bleu": bleu_a10, "chrf": chrf_a10,
                           "bleu_delta": round(bleu_a10-bleu_b1,2),
                           "chrf_delta": round(chrf_a10-chrf_b1,2)},
        "expB_strict1_5": {"bleu": bleu_b,   "chrf": chrf_b,
                           "bleu_delta": round(bleu_b-bleu_b1,2),
                           "chrf_delta": round(chrf_b-chrf_b1,2)},
        "expC_hybrid":    {"bleu": bleu_c,   "chrf": chrf_c,
                           "bleu_delta": round(bleu_c-bleu_b1,2),
                           "chrf_delta": round(chrf_c-chrf_b1,2)},
    }
}

with open(RESULTS_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved to: {RESULTS_PATH}")
print("\nOnce you have these numbers, the best-performing config")
print("goes into the main grid code (replacing CBS current settings).")
