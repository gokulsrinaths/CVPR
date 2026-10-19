#!/usr/bin/env python
"""
Exploratory diagnostic (no tuning): is the frozen baseline more accurate on held-out MCQs that have an exact-caption
sibling in the public pool? Uses only frozen predictions + public caption text. Writes IDs-only list of sibling cases
to analysis/heldout_exact_caption_siblings.json (no labels). Exact normalised caption match only.
"""
import collections
import json
import math
import re
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfFileSystem

sys.path.insert(0, "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/scripts")
import mvqa_diag as md  # noqa: E402

ROOT = md.ROOT
RD = sorted(Path(f"{ROOT}/experiments/pilot").glob("mvqa_diag_holdout711_s0_21406682_*"))[0]
fs = HfFileSystem()
z = zipfile.ZipFile(fs.open("datasets/aaassddaadf/Med-CMR/dataset.zip", "rb", block_size=8 * 1024 * 1024))
recs = {}
for member, kind in (("dataset/cmed.copy.mcq_cate.cleaned.json", "MCQ"), ("dataset/cmed.copy.open_taged.cleaned.json", "OPEN")):
    for r in json.loads(z.read(member).decode("utf-8")):
        n = int(re.search(r"-(\d+)\.png$", r["image_path"]).group(1))
        recs[kind + "_" + str(n)] = r
cap = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()
groups = collections.defaultdict(list)
for cid, r in recs.items():
    if cap(r.get("visual_description")):
        groups[cap(r["visual_description"])].append(cid)

cases = json.load(open(f"{md.FROZEN}/data_identity/cases.json"))
cases = cases["cases"] if isinstance(cases, dict) else cases
test_ids = {c["case_id"] for c in cases}
sib = {cid for cid in test_ids if any(x not in test_ids for x in groups[cap(recs[cid]["visual_description"])] if x != cid)}
gt = {g["case_id"]: g["answer"].strip() for g in json.load(open(f"{md.FROZEN}/data_identity/ground_truth.json"))["ground_truth"] if g["task_type"] == "mcq"}
fr = {a["case_id"]: a["answer"].strip() for a in json.load(open(f"{md.FROZEN}/predictions/results.json"))["answers"] if a["task_type"] == "mcq"}
base = {r["case_id"]: r for r in md.read_jsonl(RD / "forward_arms.jsonl") if r["arm"] == "base"}
mcq = sorted(gt)
S = [c for c in mcq if c in sib]
N = [c for c in mcq if c not in sib]
acc = lambda L: sum(fr[c] == gt[c] for c in L)


def margin(c):
    p = sorted(md.softmax5(base[c]["direct"]["lp"]), reverse=True)
    return p[0] - p[1]


def fisher(a, b, c, d):
    n = a + b + c + d
    r1, c1 = a + b, a + c
    pr = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    obs = pr(a)
    return sum(pr(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if pr(x) <= obs + 1e-12)


sa, na = acc(S), acc(N)
res = {"heldout_cases_with_exact_caption_sibling_outside_heldout": len(sib),
       "mcq_with_sibling": len(S), "mcq_without_sibling": len(N),
       "baseline_correct_with_sibling": f"{sa}/{len(S)}", "baseline_correct_without_sibling": f"{na}/{len(N)}",
       "accuracy_with_sibling": round(sa / len(S), 4), "accuracy_without_sibling": round(na / len(N), 4),
       "fisher_exact_p_two_sided": round(fisher(sa, len(S) - sa, na, len(N) - na), 4),
       "median_prob_margin_with_sibling": round(sorted(margin(c) for c in S)[len(S) // 2], 5),
       "median_prob_margin_without_sibling": round(sorted(margin(c) for c in N)[len(N) // 2], 5),
       "caveat": "exploratory diagnostic on the frozen test set; exact-caption siblings only (a lower bound); the "
                 "pool is all public Med-CMR cases, so some siblings may not be in the authorised 16,836; "
                 "small n; not used to tune anything"}
json.dump(res, open(f"{ROOT}/analysis/sibling_accuracy_diagnostic.json", "w"), indent=2)
json.dump({"heldout_case_ids_with_exact_caption_sibling": sorted(sib), "n": len(sib),
           "note": "ids only; derived from public captions; no labels"},
          open(f"{ROOT}/analysis/heldout_exact_caption_siblings.json", "w"), indent=1)
print(json.dumps(res, indent=1))
