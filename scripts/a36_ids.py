#!/usr/bin/env python
"""Lists the case ids of the Med-CMR counter-cue questions (correct option not the tied-shortest) -> analysis/counter_cue_ids.json"""
import json
import re

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LAB = re.compile(r"^\s*[A-E]\)\s*")
D = json.load(open(ROOT + "/data/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
test = set(json.load(open(ROOT + "/data/train_pool/split_v1/test_ids.json")))
ids = []
for c in D:
    if c.get("question type") != "mcq":
        continue
    L = [len(LAB.sub("", c[l]).strip()) for l in "ABCDE"]
    if L["ABCDE".index(c["answer"].strip())] != min(L):
        ids.append(c["case_id"])
json.dump({"definition": "MCQ cases whose correct option is not the (tied) shortest of the five; character length after removing the letter label", "n": len(ids), "n_in_held_out_886": sum(i in test for i in ids), "case_ids": sorted(ids)}, open(ROOT + "/analysis/counter_cue_ids.json", "w"), indent=0)
print(len(ids))
