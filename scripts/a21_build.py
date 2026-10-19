#!/usr/bin/env python
"""A21: builds the group-disjoint A-OKVQA split used by the non-medical replication of the shortcut dose-response.
E1 = the official A-OKVQA validation questions (1,145; test labels are hidden). From the official training questions we remove every question whose image also occurs in E1, group the rest by image hash,
order the groups by sha256('A21|' + image hash) and fill VAL700 (about 700 unlabeled-use questions, used only for the text-saturation threshold) and TR (about 9,277 training questions, as in A11/A17) with whole groups.
Output (data/aokvqa/split_a21): records.json, ids.json, donors.json (E1 random mismatch donors), SHA256.json, images/. python a21_build.py"""
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
sys.path.insert(0, ROOT + "/data/aokvqa/pylibs")
import pyarrow.parquet as pq  # noqa: E402

RAW = ROOT + "/data/aokvqa/raw/data"
OUT = ROOT + "/data/aokvqa/split_a21"
N_VAL, N_TR = 700, 9277
os.makedirs(OUT + "/images", exist_ok=True)


def rows(split, with_image):
    cols = ["question_id", "question", "choices", "correct_choice_idx"] + (["image"] if with_image else [])
    for f in sorted(glob.glob(f"{RAW}/{split}-*.parquet")):
        for b in pq.ParquetFile(f).iter_batches(batch_size=100, columns=cols):
            for r in b.to_pylist():
                yield r


def ihash(r):
    return hashlib.sha256(r["image"]["bytes"]).hexdigest()


meta = {}
for split in ("train", "validation"):
    for r in rows(split, True):
        h = ihash(r)
        assert len(r["choices"]) == 4 and 0 <= r["correct_choice_idx"] < 4
        meta[(split, r["question_id"])] = {"qid": r["question_id"], "question": r["question"], "choices": r["choices"], "gold": "ABCD"[r["correct_choice_idx"]], "ihash": h}
val = [v for (s, _), v in meta.items() if s == "validation"]
trn = [v for (s, _), v in meta.items() if s == "train"]
val_h = {v["ihash"] for v in val}
pool = [v for v in trn if v["ihash"] not in val_h]
print("train", len(trn), "val", len(val), "train questions sharing an image with val (dropped):", len(trn) - len(pool))
groups = {}
for v in pool:
    groups.setdefault(v["ihash"], []).append(v)
order = sorted(groups, key=lambda h: hashlib.sha256(("A21|" + h).encode()).hexdigest())
sel = {"VAL700": [], "TR": []}
cur, cap = "VAL700", {"VAL700": N_VAL, "TR": N_TR}
for h in order:
    g = groups[h]
    if cur == "VAL700" and len(sel["VAL700"]) + len(g) > cap["VAL700"]:
        cur = "TR"
    if cur == "TR" and len(sel["TR"]) + len(g) > cap["TR"]:
        break
    sel[cur].extend(g)
sel["E1"] = val
need = {v["ihash"] for g in sel.values() for v in g}
written = {}
for split in ("train", "validation"):
    for r in rows(split, True):
        h = ihash(r)
        if h in need and h not in written:
            p = f"{OUT}/images/{h[:24]}.jpg"
            if not os.path.exists(p):
                open(p, "wb").write(r["image"]["bytes"])
            written[h] = p
assert set(written) == need, (len(written), len(need))
recs, ids = {}, {}
for g, lst in sel.items():
    pre = {"E1": "va", "VAL700": "vl", "TR": "tr"}[g]
    recs[g], ids[g] = {}, []
    for v in sorted(lst, key=lambda x: hashlib.sha256(("A21o|" + x["qid"]).encode()).hexdigest()):
        cid = f"{pre}_{v['qid']}"
        recs[g][cid] = {"id": cid, "question": v["question"], "choices": v["choices"], "gold": v["gold"], "image": written[v["ihash"]], "ihash": v["ihash"]}
        ids[g].append(cid)
hs = {g: {r["ihash"] for r in recs[g].values()} for g in recs}
assert not (hs["E1"] & hs["TR"]) and not (hs["E1"] & hs["VAL700"]) and not (hs["TR"] & hs["VAL700"]), "image leakage between splits"
rr = random.Random(20261021)
e1 = ids["E1"]
donors = {"E1": {"rand": {c: rr.choice([x for x in e1 if recs["E1"][x]["ihash"] != recs["E1"][c]["ihash"]]) for c in e1}}}
json.dump(recs, open(OUT + "/records.json", "w"))
json.dump(ids, open(OUT + "/ids.json", "w"))
json.dump(donors, open(OUT + "/donors.json", "w"))
json.dump({"records.json": hashlib.sha256(open(OUT + "/records.json", "rb").read()).hexdigest(), "ids.json": hashlib.sha256(open(OUT + "/ids.json", "rb").read()).hexdigest(),
           "donors.json": hashlib.sha256(open(OUT + "/donors.json", "rb").read()).hexdigest()}, open(OUT + "/SHA256.json", "w"), indent=1)
for g in ("E1", "VAL700", "TR"):
    n = len(ids[g])
    uni = sum((lambda L, gi: L[gi] == min(L) and L.count(min(L)) == 1)([len(c) for c in r["choices"]], "ABCD".index(r["gold"])) for r in recs[g].values())
    print(f"{g}: n={n} images={len(hs[g])} correct option is the unique shortest in {uni / n:.3f}; gold letter {dict(Counter(r['gold'] for r in recs[g].values()))}")
print("DONE", OUT)
