#!/usr/bin/env python
"""A37 input builder: the 175 open-ended cases of the 886-case held-out slice with the image replaced by (black) an all-black image of the same size, (wrong) the image of another open case of the same modality
(seeded derangement within modality), (true) the original image. Writes data/holdout886_open_{true,black,wrong}/{cases.json,images/} and data/holdout886_open_wrong/assignment.json."""
import json
import os
import random
import shutil

from PIL import Image

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
SRC = ROOT + "/data/holdout886"
cases = json.load(open(SRC + "/cases.json"))
op = [c for c in cases["cases"] if c["task_type"] == "open"]
assert len(op) == 175
by_mod = {}
for c in op:
    by_mod.setdefault(c["metadata"]["modality"], []).append(c["case_id"])
rng = random.Random(20261106)
assign = {}
for mod, ids in by_mod.items():
    if len(ids) == 1:
        pool = [i for m, v in by_mod.items() if m != mod for i in v]
        assign[ids[0]] = rng.choice(pool)
        continue
    for _ in range(1000):
        perm = ids[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(ids, perm)):
            break
    assign.update(dict(zip(ids, perm)))
img_of = {c["case_id"]: c["image_path"] for c in op}
for arm in ("true", "black", "wrong"):
    d = f"{ROOT}/data/holdout886_open_{arm}"
    os.makedirs(d + "/images", exist_ok=True)
    out = dict(cases)
    out["cases"] = op
    out["name"] = f"{cases['name']} open-ended only, image arm: {arm}"
    json.dump(out, open(d + "/cases.json", "w"))
    for c in op:
        dst = f"{d}/{c['image_path']}"
        src = f"{SRC}/{c['image_path']}"
        if arm == "true":
            shutil.copy(src, dst)
        elif arm == "black":
            Image.new("RGB", Image.open(src).size, (0, 0, 0)).save(dst)
        else:
            shutil.copy(f"{SRC}/{img_of[assign[c['case_id']]]}", dst)
json.dump({"seed": 20261106, "assignment": assign, "modality_counts": {m: len(v) for m, v in by_mod.items()}}, open(ROOT + "/data/holdout886_open_wrong/assignment.json", "w"), indent=1)
print({m: len(v) for m, v in by_mod.items()})
print("ok")
