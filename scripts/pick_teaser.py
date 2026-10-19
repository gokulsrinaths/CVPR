#!/usr/bin/env python
"""Picks the teaser example: a random held-out Med-CMR MCQ (seeded) that the fine-tuned Qwen-3B (A9 reference) answers correctly and confidently with BOTH the true and the black image,
among questions whose text fits a figure (question <= 230 characters, every option <= 95 characters). Writes tmp/teaser_example.json."""
import json
import math
import random
import re

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LAB = re.compile(r"^\s*[A-E]\)\s*")
cases = {c["case_id"]: c for c in json.load(open(ROOT + "/data/holdout886/cases.json"))["cases"] if c["task_type"] == "mcq"}
gt = {g["case_id"]: g["answer"].strip() for g in json.load(open(ROOT + "/data/holdout886/ground_truth.json"))["ground_truth"]}
import glob
f = glob.glob(ROOT + "/experiments/pilot/ft_3b_s0_22569048_*/rows_ft.jsonl")[0]
P = {}
for ln in open(f):
    r = json.loads(ln)
    if r["group"] == "test711" and r["arm"] in ("base", "black"):
        lp = r["direct"]["lp"]
        m = max(lp)
        z = m + math.log(sum(math.exp(x - m) for x in lp))
        p = sorted([math.exp(x - z) for x in lp], reverse=True)
        P.setdefault(r["case_id"], {})[r["arm"]] = {"pred": "ABCDE"[max(range(5), key=lambda k: lp[k])], "top_prob": round(p[0], 4)}
cand = []
for cid, c in cases.items():
    if cid not in P or len(P[cid]) < 2:
        continue
    opts = [LAB.sub("", o["text"]).strip() for o in sorted(c["options"], key=lambda o: o["label"])]
    ok = all(P[cid][a]["pred"] == gt[cid] and P[cid][a]["top_prob"] >= 0.99 for a in ("base", "black"))
    if ok and len(c["question"]) <= 230 and max(len(o) for o in opts) <= 95 and min(range(5), key=lambda k: (len(opts[k]), k)) == "ABCDE".index(gt[cid]):
        cand.append((cid, opts))
random.Random(20261112).shuffle(cand)
cid, opts = cand[0]
c = cases[cid]
out = {"case_id": cid, "n_candidates": len(cand), "question": c["question"], "options": opts, "gold": gt[cid], "modality": c["metadata"].get("modality"), "pred": P[cid]}
json.dump(out, open(ROOT + "/tmp/teaser_example.json", "w"), indent=1)
print(json.dumps(out, indent=1))
