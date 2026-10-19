#!/usr/bin/env python
"""
Readiness audit (metadata only; no images, no embeddings, no index). Streams the two small JSON members of the PUBLIC
Med-CMR zip (range requests; nothing saved to disk) and:
  1. verifies the 886 held-out cases are exact public records (question / options / answer / modality) and recovers the
     id mapping,
  2. summarises the pool (task, modality, organ system, subtype),
  3. measures caption-sharing (the only sibling signal available without patient/study ids):
     how many cases share a normalised `visual_description`, and how many held-out cases have a same-caption sibling.
No held-out label is used for anything except the equality check in (1). Writes analysis/medcmr_public_metadata_audit.json.
"""
import collections
import json
import re
import zipfile

from huggingface_hub import HfFileSystem

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
FZ = f"{ROOT}/results/reproduction_frozen/data_identity"
REPO = "datasets/aaassddaadf/Med-CMR/dataset.zip"

fs = HfFileSystem()
z = zipfile.ZipFile(fs.open(REPO, "rb", block_size=8 * 1024 * 1024))
mcq = json.loads(z.read("dataset/cmed.copy.mcq_cate.cleaned.json").decode("utf-8"))
opn = json.loads(z.read("dataset/cmed.copy.open_taged.cleaned.json").decode("utf-8"))
print("public records: mcq", len(mcq), "open", len(opn))


def num(rec):
    return int(re.search(r"-(\d+)\.png$", rec["image_path"]).group(1))


pub = {}
for r in mcq:
    pub[f"MCQ_{num(r)}"] = r
for r in opn:
    pub[f"OPEN_{num(r)}"] = r
out = {"public_records": {"mcq": len(mcq), "open": len(opn)}, "record_fields": {
    "mcq": sorted({k for r in mcq for k in r}), "open": sorted({k for r in opn for k in r})}}

# ---- 1. held-out cases are public records?
cases = json.load(open(f"{FZ}/cases.json"))
cases = cases["cases"] if isinstance(cases, dict) else cases
gt = {g["case_id"]: g for g in json.load(open(f"{FZ}/ground_truth.json"))["ground_truth"]}
norm = lambda s: re.sub(r"\s+", " ", str(s or "")).strip().lower()
strip_prefix = lambda s: re.sub(r"^[A-E]\)\s*", "", str(s or "").strip())
stat = collections.Counter()
examples = []
for c in cases:
    cid = c["case_id"]
    r = pub.get(cid)
    if r is None:
        stat["no_public_record"] += 1
        continue
    stat["matched_id"] += 1
    ok_q = norm(r["question"]) == norm(c["question"])
    ok_m = norm(r.get("modality")) == norm((gt[cid].get("metadata") or {}).get("modality"))
    stat["question_equal"] += ok_q
    stat["modality_equal"] += ok_m
    if c["task_type"] == "mcq":
        ok_a = r["answer"].strip() == gt[cid]["answer"].strip()
        ok_o = all(norm(strip_prefix(r[o["label"]])) == norm(o["text"]) for o in c["options"])
        stat["mcq_answer_equal"] += ok_a
        stat["mcq_options_equal"] += ok_o
        if not (ok_q and ok_a and ok_o) and len(examples) < 3:
            examples.append({"case": cid, "q": ok_q, "a": ok_a, "o": ok_o})
    else:
        ok_a = norm(r["answer"]) == norm(gt[cid]["answer"])
        stat["open_answer_equal"] += ok_a
out["holdout_vs_public"] = dict(stat)
out["holdout_vs_public_mismatch_examples"] = examples
print("holdout vs public:", dict(stat), examples)

# ---- 2. pool summary
for name, recs in (("mcq", mcq), ("open", opn)):
    out[f"{name}_task_counts"] = dict(collections.Counter(r.get("task") for r in recs).most_common())
    out[f"{name}_modality_counts_top"] = dict(collections.Counter(r.get("modality") for r in recs).most_common(14))
    out[f"{name}_organ_system_counts"] = dict(collections.Counter(r.get("organ system") for r in recs).most_common(14))
    out[f"{name}_subtype_counts"] = dict(collections.Counter(r.get("subtype") for r in recs).most_common(10))
    out[f"{name}_visual_description_present"] = sum(bool((r.get("visual_description") or "").strip()) for r in recs)

# ---- 3. caption sharing (sibling prevalence), exact on normalised text
cap_norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()
groups = collections.defaultdict(list)
for cid, r in pub.items():
    cn = cap_norm(r.get("visual_description"))
    if cn:
        groups[cn].append(cid)
sizes = collections.Counter(len(v) for v in groups.values())
test_ids = {c["case_id"] for c in cases}
with_sib_outside = with_sib_inside = 0
sib_sizes = []
for cid in test_ids:
    cn = cap_norm(pub[cid].get("visual_description")) if cid in pub else ""
    if not cn:
        continue
    others = [x for x in groups[cn] if x != cid]
    if any(x not in test_ids for x in others):
        with_sib_outside += 1
        sib_sizes.append(len([x for x in others if x not in test_ids]))
    if any(x in test_ids for x in others):
        with_sib_inside += 1
out["caption_sharing_all_public_cases"] = {
    "cases_with_caption": sum(len(v) for v in groups.values()), "distinct_normalised_captions": len(groups),
    "group_size_histogram": dict(sorted(sizes.items())), "max_group_size": max(sizes),
    "cases_in_a_group_of_2plus": sum(len(v) for v in groups.values() if len(v) > 1)}
out["heldout_exact_caption_siblings"] = {
    "heldout_cases": len(test_ids), "with_same_caption_case_outside_heldout": with_sib_outside,
    "with_same_caption_case_inside_heldout": with_sib_inside,
    "median_outside_siblings_when_present": sorted(sib_sizes)[len(sib_sizes) // 2] if sib_sizes else 0,
    "note": "upper bound on sibling prevalence: pool = all public Med-CMR cases, not only the authorised 16,836; "
            "exact normalised-caption match only (near-duplicates not measured)"}
json.dump(out, open(f"{ROOT}/analysis/medcmr_public_metadata_audit.json", "w"), indent=2)
print(json.dumps({k: out[k] for k in ("mcq_task_counts", "open_task_counts", "caption_sharing_all_public_cases", "heldout_exact_caption_siblings")}, indent=1))
