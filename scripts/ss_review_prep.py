#!/usr/bin/env python
"""Build review montages for threshold selection (rule: docs/dev_split_thresholds_rule.md). CPU only.
10 seeded pairs per bin, excluding pairs that are byte-identical (E0). Writes data/train_pool/derived/review/*.png
and analysis/threshold_review_sample.json (pair ids, distances, captions) BEFORE any judgement is made."""
import collections
import json
import os
import random

from PIL import Image, ImageDraw

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOL = ROOT + "/data/train_pool"
D = POOL + "/derived"
os.makedirs(D + "/review", exist_ok=True)
T = json.load(open(POOL + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
P = json.load(open(D + "/pairs.json"))
man = json.load(open(POOL + "/IMAGE_MANIFEST.json"))["train"]
sha = [man[os.path.basename(c["image_path"])] for c in T]
imgp = lambda i: POOL + "/extracted/train/imgs/" + os.path.basename(T[i]["image_path"])
rng = random.Random(20261002)
E2 = [(i, j, dd, pp) for i, j, dd, pp in P["E2_pairs"] if sha[i] != sha[j]]
E3 = [(i, j, jc) for i, j, jc in P["E3_pairs"] if sha[i] != sha[j]]
cnt = collections.Counter(min(dd, pp) for _, _, dd, pp in E2)
print("E2 non-identical pairs by min(dH,pH):", dict(sorted(cnt.items())), flush=True)
cap = lambda i: T[i]["visual_description"][:95]
S = {"E2": {}, "E3": {}}
bins2 = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
bins3 = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def montage(pairs, name, with_caps, label):
    W = 190
    cols, rows = 2, 5
    cw, chh = 2 * W + 14, W + (40 if with_caps else 18)
    M = Image.new("RGB", (cols * cw, rows * chh), "white")
    dr = ImageDraw.Draw(M)
    for k, (i, j, info) in enumerate(pairs):
        x0, y0 = (k % cols) * cw, (k // cols) * chh
        for q, idx in enumerate((i, j)):
            im = Image.open(imgp(idx)).convert("RGB")
            im.thumbnail((W, W))
            M.paste(im, (x0 + q * (W + 6), y0 + 14))
        dr.text((x0 + 2, y0 + 1), f"#{k + 1} {label} {info}", fill="black")
        if with_caps:
            dr.text((x0 + 2, y0 + W + 15), "L: " + cap(i)[:60], fill="black")
            dr.text((x0 + 2, y0 + W + 27), "R: " + cap(j)[:60], fill="black")
    M.save(f"{D}/review/{name}.png")


for lo, hi in bins2:
    c = [(i, j, f"dH={dd} pH={pp}") for i, j, dd, pp in E2 if lo <= min(dd, pp) <= hi]
    pick = rng.sample(c, min(10, len(c)))
    S["E2"][f"{lo}-{hi}"] = {"n_pairs_in_bin": len(c), "sample": [[T[i]["case_id"], T[j]["case_id"], info] for i, j, info in pick]}
    montage(pick, f"E2_{lo}-{hi}", False, "img")
for lo, hi in bins3:
    c = [(i, j, f"J={jc}") for i, j, jc in E3 if lo <= jc < hi]
    pick = rng.sample(c, min(10, len(c)))
    S["E3"][f"{lo}-{hi}"] = {"n_pairs_in_bin": len(c), "sample": [[T[i]["case_id"], T[j]["case_id"], info] for i, j, info in pick]}
    montage(pick, f"E3_{lo}-{min(hi, 1.0)}", True, "cap")
json.dump(S, open(ROOT + "/analysis/threshold_review_sample.json", "w"), indent=1)
print({k: {b: v["n_pairs_in_bin"] for b, v in S[k].items()} for k in S}, flush=True)
print("DONE", flush=True)
