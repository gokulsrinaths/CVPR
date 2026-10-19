#!/usr/bin/env python
"""
Phase 1 + 2 (CPU only): MCQ reproduction audit and 28-error analysis.

Reads ONLY the frozen copies under results/reproduction_frozen plus the images and
the detector pickle; writes to analysis/. Deterministic: no sampling, no models.
Run with the medreason_env python (needs sklearn + cv2 to re-run the modality
detector with the exact functions used at inference).
"""
import collections
import csv
import json
import math
import os
import pickle
import re
import sys

import numpy as np
from PIL import Image

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
FZ = f"{ROOT}/results/reproduction_frozen"
AN = f"{ROOT}/analysis"
sys.path.insert(0, f"{ROOT}/code")
sys.path.insert(0, f"{ROOT}/scoring_suite/scoring")
from medreason_docker.systems.custom_system import _extract_features  # exact inference-time function
from medreason_scoring.normalize import tokens

os.makedirs(AN, exist_ok=True)

cases = json.load(open(f"{FZ}/data_identity/cases.json"))
cases = cases["cases"] if isinstance(cases, dict) else cases
gt = json.load(open(f"{FZ}/data_identity/ground_truth.json"))["ground_truth"]
res = json.load(open(f"{FZ}/predictions/results.json"))["answers"]
C = {c["case_id"]: c for c in cases}
G = {g["case_id"]: g for g in gt}
R = {r["case_id"]: r for r in res}
assert set(C) == set(G) == set(R), "case-id sets differ"
mcq = [cid for cid in C if C[cid]["task_type"] == "mcq"]
assert len(mcq) == 711


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)


# ---------------------------------------------------------------- Phase 1: recompute under every rule we know
strict = sum(R[c]["answer"].strip() == G[c]["answer"].strip() for c in mcq)
readme = sum(R[c]["answer"].strip().upper() == G[c]["answer"].strip().upper() for c in mcq)
labels_ok = sum(R[c]["answer"].strip() in {o["label"] for o in C[c]["options"]} for c in mcq)
pred_dist = collections.Counter(R[c]["answer"].strip() for c in mcq)
gt_dist = collections.Counter(G[c]["answer"].strip() for c in mcq)
print(f"strict scorer rule: {strict}/711 | README eval_holdout rule (case-folded): {readme}/711 | predictions that are valid option labels: {labels_ok}/711")
print("pred letters:", dict(sorted(pred_dist.items())), "| gold letters:", dict(sorted(gt_dist.items())))

# ---------------------------------------------------------------- modality detector re-run (exact inference functions)
det = pickle.load(open(f"{ROOT}/models/detector.pkl", "rb"))
labels = det["labels"]
std_i = labels.index("standard")
thr = det["threshold"]
print("detector labels:", labels, "threshold:", thr)


def detect(cid):
    p = C[cid]["image_path"]
    p = p if os.path.isabs(p) else os.path.join(f"{ROOT}/data/holdout886", p)
    rgb = np.array(Image.open(p).convert("RGB"))
    proba = det["clf"].predict_proba(_extract_features(rgb).reshape(1, -1))[0]
    ood = int(np.argmax([proba[i] if labels[i] != "standard" else -1 for i in range(len(labels))]))
    route = labels[ood] if (proba[ood] >= thr and proba[ood] > proba[std_i]) else "standard"
    return route, float(proba[ood]), float(proba[std_i]), labels[ood]


det_info = {cid: detect(cid) for cid in mcq}
agree = sum(det_info[c][0] == R[c]["metadata"]["modality"] for c in mcq)
print(f"detector re-run agrees with logged route on {agree}/711 MCQ cases")
routed = [c for c in mcq if R[c]["metadata"]["modality"] != "standard"]
print("MCQ cases routed to non-standard preprocessing:", len(routed),
      "| of which wrong:", sum(R[c]["answer"].strip() != G[c]["answer"].strip() for c in routed))
near = []
for c in mcq:
    route, po, ps, olab = det_info[c]
    margin = min(abs(po - thr), abs(po - ps))
    if margin < 0.05:
        near.append(c)
print("cases within 0.05 of the detector decision boundary:", len(near), "| wrong among them:",
      sum(R[c]["answer"].strip() != G[c]["answer"].strip() for c in near))

# ---------------------------------------------------------------- feature table for every MCQ case
VIS = re.compile(r"\b(image|images|scan|shown|shows|visible|figure|radiograph|x-ray|xray|ct|mri|ultrasound|"
                 r"microscop\w*|histolog\w*|biopsy|slide|stain\w*|photograph\w*|appear\w*|depict\w*|observed|seen|"
                 r"arrow\w*|asterisk|marked|lesion|opacit\w*|finding|findings)\b", re.I)
KNOW = re.compile(r"\b(most likely|first-line|mechanism|etiolog\w*|management|treatment|risk factor|complication|"
                  r"prognos\w*|diagnosis|which of the following)\b", re.I)


def jacc(a, b):
    a, b = set(tokens(a)), set(tokens(b))
    return len(a & b) / len(a | b) if (a | b) else 0.0


mod_counts = collections.Counter((G[c].get("metadata") or {}).get("modality") for c in mcq)
rows = []
for c in mcq:
    case, g, r = C[c], G[c], R[c]
    opts = {o["label"]: o["text"] for o in case["options"]}
    gl, pl = g["answer"].strip(), r["answer"].strip()
    gtxt, ptxt = opts.get(gl, ""), opts.get(pl, "")
    lens = {k: len(v) for k, v in opts.items()}
    rank = lambda k: 1 + sum(lens[j] > lens[k] for j in lens)
    pair_max = max((jacc(opts[a], opts[b]) for a in opts for b in opts if a < b), default=0.0)
    mod = (g.get("metadata") or {}).get("modality")
    route, po, ps, olab = det_info[c]
    rows.append({
        "case_id": c, "question": case["question"], "n_options": len(opts),
        "options": " || ".join(f"{k}: {v}" for k, v in opts.items()),
        "ground_truth": gl, "ground_truth_text": gtxt,
        "raw_prediction": "NOT_SAVED (only the parsed letter was written by process.py)",
        "parsed_prediction": pl, "parsed_prediction_text": ptxt,
        "correct": int(pl == gl),
        "modality_true": mod, "modality_multi_label": int(bool(mod) and "," in mod),
        "modality_count_in_holdout": mod_counts[mod],
        "question_type": (case.get("metadata") or {}).get("question_type"),
        "detector_route_logged": r["metadata"]["modality"], "detector_route_rerun": route,
        "detector_p_ood": round(po, 4), "detector_p_standard": round(ps, 4),
        "detector_boundary_margin": round(min(abs(po - thr), abs(po - ps)), 4),
        "jaccard_gt_vs_pred": round(jacc(gtxt, ptxt), 3), "max_pairwise_option_jaccard": round(pair_max, 3),
        "gt_option_len_rank": rank(gl) if gl in lens else None, "pred_option_len_rank": rank(pl) if pl in lens else None,
        "question_visual_cue": int(bool(VIS.search(case["question"]))),
        "question_knowledge_cue": int(bool(KNOW.search(case["question"]))),
        "trace_is_constant_stub": int(r["reasoning_trace"].strip() == "I examine the image and identify the relevant findings."),
    })

wrong = [x for x in rows if not x["correct"]]
assert len(wrong) == 711 - strict


def flags(x):
    f = []
    if x["jaccard_gt_vs_pred"] >= 0.5:
        f.append("near_duplicate_options(gt~pred)")
    if x["max_pairwise_option_jaccard"] >= 0.6:
        f.append("highly_overlapping_option_set")
    if x["detector_route_logged"] != "standard":
        f.append("ood_routed_preprocessing")
    if x["detector_boundary_margin"] < 0.05:
        f.append("near_detector_boundary")
    if x["parsed_prediction"] == "A":
        f.append("pred_A_could_be_parser_default")
    if x["modality_multi_label"]:
        f.append("multi_modality_label")
    if x["modality_count_in_holdout"] < 14:
        f.append("rare_modality(<2%)")
    if not x["question_visual_cue"]:
        f.append("no_visual_cue_in_question(knowledge_like)")
    return f


for x in rows:
    x["suspected_issue"] = ";".join(flags(x))
    x["notes"] = ""

# ---------------------------------------------------------------- write CSVs
disc_cols = ["case_id", "question", "ground_truth", "raw_prediction", "parsed_prediction", "correct",
             "suspected_issue", "notes", "options", "ground_truth_text", "parsed_prediction_text",
             "modality_true", "detector_route_logged", "detector_route_rerun", "detector_boundary_margin"]
with open(f"{AN}/mcq_discrepancy.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=disc_cols, extrasaction="ignore")
    w.writeheader()
    for x in wrong:
        w.writerow(x)

err_cols = list(rows[0].keys())
with open(f"{AN}/mcq_error_analysis.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=err_cols)
    w.writeheader()
    for x in wrong:
        w.writerow(x)
with open(f"{AN}/mcq_all711_features.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=err_cols)
    w.writeheader()
    for x in rows:
        w.writerow(x)

# ---------------------------------------------------------------- breakdowns (with Wilson CIs on error rate)
def breakdown(key):
    tot, err = collections.Counter(), collections.Counter()
    for x in rows:
        k = x[key] if x[key] not in (None, "") else "NA"
        tot[k] += 1
        err[k] += (1 - x["correct"])
    out = {}
    for k in sorted(tot, key=lambda z: -tot[z]):
        lo, hi = wilson(err[k], tot[k])
        out[str(k)] = {"n": tot[k], "errors": err[k], "error_rate": round(err[k] / tot[k], 4),
                       "wilson95": [round(lo, 4), round(hi, 4)]}
    return out


def primary_modality(m):
    return "multi:" + m if m and "," in m else m


for x in rows:
    x["modality_primary"] = primary_modality(x["modality_true"])
summary = {
    "counts": {"mcq_n": 711, "correct_strict": strict, "correct_readme_rule": readme, "errors": len(wrong),
               "valid_label_predictions": labels_ok},
    "detector": {"rerun_agrees_with_logged": agree, "routed_non_standard": len(routed),
                 "routed_wrong": sum(R[c]["answer"].strip() != G[c]["answer"].strip() for c in routed),
                 "within_0.05_of_boundary": len(near),
                 "near_boundary_wrong": sum(R[c]["answer"].strip() != G[c]["answer"].strip() for c in near)},
    "by_modality_true": breakdown("modality_primary"),
    "by_question_type": breakdown("question_type"),
    "by_question_visual_cue": breakdown("question_visual_cue"),
    "by_n_options": breakdown("n_options"),
    "by_detector_route": breakdown("detector_route_logged"),
    "pred_letter_distribution": dict(sorted(pred_dist.items())),
    "gold_letter_distribution": dict(sorted(gt_dist.items())),
    "error_flag_counts": dict(collections.Counter(fl for x in wrong for fl in flags(x))),
    "note": "Heuristic flags are deterministic text/metadata/detector features, NOT validated failure categories. "
            "MCQ reasoning traces are a single constant stub, so trace-based classification (reasoning instability, "
            "visual grounding) is impossible from the saved run; those categories need the logit-margin, sampling and "
            "retrieval experiments.",
}
json.dump(summary, open(f"{AN}/error_taxonomy_summary.json", "w"), indent=2)

print("\nerrors:", len(wrong))
for k in ("by_modality_true", "by_question_type", "by_question_visual_cue", "by_n_options", "by_detector_route"):
    print(f"\n{k}:")
    for kk, v in summary[k].items():
        print(f"  {kk:55s} n={v['n']:4d} err={v['errors']:3d} rate={v['error_rate']:.3f}  CI={v['wilson95']}")
print("\nflag counts among errors:", summary["error_flag_counts"])
print("\n==== 28 ERRORS (full text for manual inspection) ====")
for i, x in enumerate(wrong, 1):
    print(f"\n[{i}] {x['case_id']} | modality={x['modality_true']} | type={x['question_type']} | route={x['detector_route_logged']}(margin {x['detector_boundary_margin']}) | flags={x['suspected_issue']}")
    print("Q:", x["question"])
    print("OPTIONS:", x["options"])
    print(f"GT={x['ground_truth']} ({x['ground_truth_text'][:160]}) | PRED={x['parsed_prediction']} ({x['parsed_prediction_text'][:160]}) | jaccard={x['jaccard_gt_vs_pred']}")
