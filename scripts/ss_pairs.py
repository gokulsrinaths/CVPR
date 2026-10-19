#!/usr/bin/env python
"""Same-source graph, step 2 (CPU): candidate pairs at generous thresholds over the 17,722 official train cases.
E0 identical image sha256 | E1 identical normalised caption | E2 image near-duplicate (dHash/pHash Hamming, kept <= 14)
E3 caption token-set Jaccard >= 0.5 | E4 >= 3 identical normalised options AND same primary modality.
Thresholds are NOT chosen here. Writes data/train_pool/derived/pairs.json (+ summary printed)."""
import collections
import json
import re
import time

import numpy as np

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOL = ROOT + "/data/train_pool"
D = POOL + "/derived"
T = json.load(open(POOL + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]
ids = json.load(open(D + "/ids.json"))
assert ids == [c["case_id"] for c in T]
n = len(ids)
dh, ph = np.load(D + "/dhash.npy"), np.load(D + "/phash.npy")
man = json.load(open(POOL + "/IMAGE_MANIFEST.json"))["train"]
import os
sha = [man[os.path.basename(c["image_path"])] for c in T]
out = {"n": n}
t0 = time.time()

# ---- E0
g = collections.defaultdict(list)
for i, s in enumerate(sha):
    g[s].append(i)
out["E0_groups"] = [v for v in g.values() if len(v) > 1]
print("E0 groups:", len(out["E0_groups"]), "cases in them:", sum(len(v) for v in out["E0_groups"]), flush=True)

# ---- E1 / E3
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
caps = [norm(c["visual_description"]) for c in T]
g = collections.defaultdict(list)
for i, c in enumerate(caps):
    g[c].append(i)
out["E1_groups"] = [v for v in g.values() if len(v) > 1]
print("E1 groups:", len(out["E1_groups"]), "cases:", sum(len(v) for v in out["E1_groups"]), flush=True)
STOP = set("the a an of in on at to and or with without for by from is are was were be been this that these those as its it their there into over under between within which who whom whose may might can could shows show showing image images view demonstrates demonstrating".split())
toks = [frozenset(w for w in c.split() if w not in STOP and len(w) > 1) for c in caps]
df = collections.Counter(w for t in toks for w in t)
post = collections.defaultdict(list)
for i, t in enumerate(toks):
    for w in t:
        if df[w] <= 300:
            post[w].append(i)
e3 = []
for i, t in enumerate(toks):
    cnt = collections.Counter()
    for w in t:
        if df[w] <= 300:
            for j in post[w]:
                if j > i:
                    cnt[j] += 1
    for j, c in cnt.items():
        if c >= 2:
            u = len(t | toks[j])
            jac = len(t & toks[j]) / u if u else 0.0
            if jac >= 0.5:
                e3.append((i, j, round(jac, 3)))
out["E3_pairs"] = e3
print("E3 pairs (J>=0.5):", len(e3), "time", round(time.time() - t0), flush=True)

# ---- E2
LUT = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def ham(a, b):
    x = np.bitwise_xor(a, b)
    return LUT[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis=-1)


e2 = []
CH = 400
for s in range(0, n, CH):
    a_d, a_p = dh[s:s + CH, None], ph[s:s + CH, None]
    dd, pp = ham(a_d, dh[None, :]), ham(a_p, ph[None, :])
    m = np.minimum(dd, pp) <= 14
    ii, jj = np.nonzero(m)
    for i, j in zip(ii, jj):
        if s + i < j:
            e2.append((int(s + i), int(j), int(dd[i, j]), int(pp[i, j])))
out["E2_pairs"] = e2
print("E2 pairs (min(dH,pH)<=14):", len(e2), "time", round(time.time() - t0), flush=True)

# ---- E4
LAB = re.compile(r"^\s*[A-E]\)\s*")
opts = []
for c in T:
    if c["question type"] == "mcq":
        opts.append({norm(LAB.sub("", c[k])) for k in "ABCDE"})
    else:
        opts.append(set())
post = collections.defaultdict(list)
for i, s in enumerate(opts):
    for o in s:
        post[o].append(i)
e4 = []
for i, s in enumerate(opts):
    cnt = collections.Counter()
    for o in s:
        for j in post[o]:
            if j > i:
                cnt[j] += 1
    for j, c in cnt.items():
        if c >= 3 and T[i]["primary_modality"] == T[j]["primary_modality"]:
            e4.append((i, j, c))
out["E4_pairs"] = e4
print("E4 pairs (>=3 identical options, same modality):", len(e4), "time", round(time.time() - t0), flush=True)
json.dump(out, open(D + "/pairs.json", "w"))
print("DONE", flush=True)
