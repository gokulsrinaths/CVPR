#!/usr/bin/env python
"""A11 data builder: PMC-VQA splits (article-disjoint), image extraction, donor maps. Deterministic (seed 20261003). Never overwrites.
python pmc_build.py [--selftest]"""
import csv
import hashlib
import json
import os
import random
import sys
import zipfile

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
D = ROOT + "/data/pmcvqa"
OUT = D + "/split_a11"
SEED = 20261003
N_E2, N_VAL, N_TR = 3000, 700, 9277
LET = "ABCD"


def art(fp):
    return fp.split("_")[0]


def clean_choice(s, letter):
    s = s.strip()
    for pre in (letter + ":", letter + ")", letter + "."):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    return s.strip()


def rows_of(path, tag):
    out = []
    for i, r in enumerate(csv.DictReader(open(path, encoding="utf-8"))):
        ch = [clean_choice(r["Choice " + c], c) for c in LET]
        g = (r.get("Answer_label") or "").strip()
        if g not in LET or any(not c for c in ch) or len(set(ch)) < 4 or not r["Question"].strip():
            continue
        out.append({"id": f"{tag}{i}", "article": art(r["Figure_path"]), "fig": r["Figure_path"], "question": r["Question"].strip(), "choices": ch, "gold": g})
    return out


def build():
    if os.path.exists(OUT + "/ids.json"):
        sys.exit("refusing to overwrite " + OUT)
    os.makedirs(OUT + "/images", exist_ok=True)
    te = rows_of(D + "/test_clean.csv", "te")
    tr_all = rows_of(D + "/train.csv", "tr")
    ta = {r["article"] for r in te}
    tr_all = [r for r in tr_all if r["article"] not in ta]
    by = {}
    for r in tr_all:
        by.setdefault(r["article"], []).append(r)
    arts = sorted(by)
    random.Random(SEED).shuffle(arts)
    groups, cur, quota = {}, [], [("E2", N_E2), ("VAL700", N_VAL), ("TR", N_TR)]
    ai = 0
    for name, q in quota:
        rows = []
        while len(rows) < q and ai < len(arts):
            rows += by[arts[ai]]
            ai += 1
        groups[name] = rows[:q]
    groups["E1"] = te
    # image extraction (basename -> zip member)
    names = {}
    for z in ("images.zip", "images_2.zip"):
        zf = zipfile.ZipFile(D + "/" + z)
        for n in zf.namelist():
            if not n.endswith("/"):
                names.setdefault(n.split("/")[-1], (z, n))
    zfs = {z: zipfile.ZipFile(D + "/" + z) for z in ("images.zip", "images_2.zip")}
    for g, rows in groups.items():
        for r in rows:
            z, n = names[r["fig"]]
            p = OUT + "/images/" + r["fig"]
            if not os.path.exists(p):
                open(p, "wb").write(zfs[z].read(n))
            r["image"] = p
    ids = {g: [r["id"] for r in rows] for g, rows in groups.items()}
    json.dump({g: {r["id"]: r for r in rows} for g, rows in groups.items()}, open(OUT + "/records.json", "w"))
    # donor maps within E1 and E2 (random: fixed cyclic derangement; nn: PubMedCLIP top-1 excluding same article)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(ROOT + "/models/pubmedclip_vit_b32").eval()
    proc = CLIPProcessor.from_pretrained(ROOT + "/models/pubmedclip_vit_b32")
    donors = {}
    for g in ("E1", "E2"):
        rows = groups[g]
        embs = []
        for s in range(0, len(rows), 64):
            ims = [Image.open(r["image"]).convert("RGB") for r in rows[s:s + 64]]
            with torch.no_grad():
                e = model.get_image_features(**proc(images=ims, return_tensors="pt"))
                e = e.pooler_output if hasattr(e, "pooler_output") else e
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).numpy())
            print(g, s, flush=True)
        E = np.concatenate(embs)
        S = E @ E.T
        n = len(rows)
        a = [r["article"] for r in rows]
        nn, rand = {}, {}
        perm = list(range(n))
        random.Random(SEED + 1).shuffle(perm)
        for k in range(n):  # cyclic derangement over the shuffled order, skipping same-article donors
            i = perm[k]
            for step in range(1, n):
                j = perm[(k + step) % n]
                if a[j] != a[i]:
                    rand[rows[i]["id"]] = rows[j]["id"]
                    break
        for i in range(n):
            s = S[i].copy()
            for j in range(n):
                if a[j] == a[i]:
                    s[j] = -9
            nn[rows[i]["id"]] = rows[int(s.argmax())]["id"]
        donors[g] = {"nn": nn, "rand": rand, "nn_cos_mean": float(np.mean([S[i, [r["id"] for r in rows].index(nn[rows[i]["id"]])] for i in range(0, n, 10)]))}
    json.dump(donors, open(OUT + "/donors.json", "w"))
    json.dump(ids, open(OUT + "/ids.json", "w"))
    h = {f: hashlib.sha256(open(OUT + "/" + f, "rb").read()).hexdigest() for f in ("ids.json", "records.json", "donors.json")}
    json.dump(h, open(OUT + "/SHA256.json", "w"), indent=1)
    print({g: len(v) for g, v in ids.items()}, h)


def selftest():
    rec = json.load(open(OUT + "/records.json"))
    ids = json.load(open(OUT + "/ids.json"))
    don = json.load(open(OUT + "/donors.json"))
    A = {g: {rec[g][i]["article"] for i in v} for g, v in ids.items()}
    fails = []

    def check(n, ok):
        print(("ok   " if ok else "FAIL ") + n)
        if not ok:
            fails.append(n)
    check("sizes", [len(ids[g]) for g in ("E1", "E2", "VAL700", "TR")] == [1999, N_E2, N_VAL, N_TR])
    gs = ["E1", "E2", "VAL700", "TR"]
    check("article-disjoint across all groups", all(not (A[a] & A[b]) for i, a in enumerate(gs) for b in gs[i + 1:]))
    check("images exist", all(os.path.exists(rec[g][i]["image"]) for g in gs for i in ids[g][:50]))
    check("donors cover E1 and E2 and are in-set, other article", all(don[g][k][i] in rec[g] and rec[g][don[g][k][i]]["article"] != rec[g][i]["article"] for g in ("E1", "E2") for k in ("nn", "rand") for i in ids[g]))
    check("gold in ABCD", all(rec[g][i]["gold"] in LET for g in gs for i in ids[g]))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else build()
