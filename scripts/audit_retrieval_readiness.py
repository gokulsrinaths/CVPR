#!/usr/bin/env python
"""
Retrieval-data readiness audit (read-only, CPU). Audits only what is ALREADY on the cluster for the held-out slice:
schema, ID structure, modality vocabulary, question-template structure, image files. No index, no embeddings, no model.
Writes analysis/retrieval_readiness_facts.json.
"""
import collections
import json
import os
import re
import statistics as st

from PIL import Image

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
FZ = f"{ROOT}/results/reproduction_frozen/data_identity"
H = f"{ROOT}/data/holdout886"
cases = json.load(open(f"{FZ}/cases.json"))
cases = cases["cases"] if isinstance(cases, dict) else cases
gt = json.load(open(f"{FZ}/ground_truth.json"))["ground_truth"]
G = {g["case_id"]: g for g in gt}
out = {}

# ---- schema
out["case_fields_union"] = sorted({k for c in cases for k in c})
out["case_metadata_fields_union"] = sorted({k for c in cases for k in (c.get("metadata") or {})})
out["gt_fields_union"] = sorted({k for g in gt for k in g})
out["gt_metadata_fields_union"] = sorted({k for g in gt for k in (g.get("metadata") or {})})
nonempty = {}
for t in ("mcq", "open"):
    sub = [g for g in gt if g["task_type"] == t]
    nonempty[t] = {"n": len(sub)}
    for f in ("answer", "acceptable_answers", "reference_trace", "evidence_keys", "forbidden_claims"):
        nonempty[t][f"{f}_nonempty"] = sum(bool(g.get(f)) for g in sub)
out["gt_nonempty_counts"] = nonempty
out["has_context_field"] = any("context" in c for c in cases)
out["has_any_source_or_patient_or_study_id_field"] = any(
    any(k in c or k in (c.get("metadata") or {}) for k in ("patient_id", "study_id", "figure_id", "source", "pmcid", "doi", "url", "article_id"))
    for c in cases)

# ---- case-id structure
ids = collections.defaultdict(list)
for c in cases:
    m = re.match(r"^([A-Za-z]+)_(\d+)$", c["case_id"])
    ids[m.group(1)].append(int(m.group(2)))
out["case_id_structure"] = {p: {"n": len(v), "min": min(v), "max": max(v), "span": max(v) - min(v) + 1} for p, v in ids.items()}
out["case_id_note"] = ("IDs are '<TYPE>_<integer>' dataset-level indices; they carry no patient, study or figure grouping.")

# ---- modality / question type vocab
mods = collections.Counter()
multi = 0
for g in gt:
    m = (g.get("metadata") or {}).get("modality")
    if m:
        parts = [p.strip() for p in m.split(",")]
        multi += len(parts) > 1
        for p in parts:
            mods[p] += 1
out["modality_vocab_base_counts_886"] = dict(mods.most_common())
out["multi_modality_cases_886"] = multi
out["question_type_values"] = dict(collections.Counter((c.get("metadata") or {}).get("question_type") for c in cases))

# ---- question template / option structure (matters for near-duplicate rules)
mcq = [c for c in cases if c["task_type"] == "mcq"]
q = collections.Counter(c["question"] for c in mcq)
out["mcq_question_text"] = {"n": len(mcq), "distinct_question_strings": len(q),
                            "questions_appearing_more_than_once": sum(1 for v in q.values() if v > 1),
                            "top5_repeated": [[k[:110], v] for k, v in q.most_common(5)]}
first6 = collections.Counter(" ".join(c["question"].split()[:6]).lower() for c in mcq)
out["mcq_question_first6words_top5"] = first6.most_common(5)
opt_texts = collections.Counter(o["text"].strip().lower() for c in mcq for o in c["options"])
out["mcq_option_texts"] = {"distinct": len(opt_texts), "appearing_in_2plus_cases": sum(1 for v in opt_texts.values() if v > 1),
                           "top5_repeated": [[k[:70], v] for k, v in opt_texts.most_common(5)]}

# ---- image files
sizes, dims, exts, missing = [], [], collections.Counter(), 0
for c in cases:
    p = c["image_path"]
    p = p if os.path.isabs(p) else os.path.join(H, p)
    if not os.path.exists(p):
        missing += 1
        continue
    exts[os.path.splitext(p)[1].lower()] += 1
    sizes.append(os.path.getsize(p))
for c in cases[::6]:
    p = c["image_path"]
    p = p if os.path.isabs(p) else os.path.join(H, p)
    with Image.open(p) as im:
        dims.append(im.size)
mean_mb = st.mean(sizes) / 2 ** 20
out["images"] = {"n": len(cases), "missing": missing, "extensions": dict(exts), "path_pattern_example": cases[0]["image_path"],
                 "total_mb": round(sum(sizes) / 2 ** 20, 1), "mean_mb": round(mean_mb, 3),
                 "median_mb": round(st.median(sizes) / 2 ** 20, 3),
                 "width_min_med_max": [min(d[0] for d in dims), int(st.median(d[0] for d in dims)), max(d[0] for d in dims)],
                 "height_min_med_max": [min(d[1] for d in dims), int(st.median(d[1] for d in dims)), max(d[1] for d in dims)],
                 "sampled_for_dims": len(dims),
                 "extrapolated_gb_for_16836_images": round(mean_mb * 16836 / 1024, 2)}
json.dump(out, open(f"{ROOT}/analysis/retrieval_readiness_facts.json", "w"), indent=2)
print(json.dumps(out, indent=1))
