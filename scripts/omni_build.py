#!/usr/bin/env python
"""A16 data builder: OmniMedVQA open-access splits (image-disjoint; patient-disjoint where a patient folder exists), image extraction, donor maps.
Same schema as data/pmcvqa/split_a11. Deterministic (seed 20261004). Never overwrites. python omni_build.py [--selftest]"""
import hashlib
import json
import os
import random
import re
import sys
import zipfile

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
D = ROOT + "/data/omnimed"
OUT = D + "/split_a16"
SEED = 20261004
CAP = 1200                      # questions per source dataset in the sampling frame
N_E1, N_E2, N_VAL, N_TR = 2000, 3000, 700, 9277
LET = "ABCD"


def group_of(r):
    p = r["image_path"]
    m = re.search(r"/(patient\d+)/", p)
    return r["dataset"] + "|" + (p[:p.index(m.group(0))] + m.group(0) if m else p)


def usable(r):
    ch = [r.get("option_" + k) for k in LET]
    if any(c in (None, "") for c in ch):
        return None
    ch = [c.strip() for c in ch]
    if len(set(c.lower() for c in ch)) < 4:
        return None
    gt = (r["gt_answer"] or "").strip()
    gi = [c.lower() for c in ch]
    if gt.lower() not in gi:
        return None
    return ch, LET[gi.index(gt.lower())]


def build():
    if os.path.exists(OUT + "/ids.json"):
        sys.exit("refusing to overwrite " + OUT)
    os.makedirs(OUT + "/images", exist_ok=True)
    z = zipfile.ZipFile(D + "/OmniMedVQA.zip")
    rows = []
    for n in z.namelist():
        if n.endswith(".json") and "Open-access" in n:
            rows += json.loads(z.read(n).decode("utf-8"))
    recs = []
    for r in rows:
        u = usable(r)
        if u is None or ("OmniMedVQA/" + r["image_path"]) not in z.NameToInfo:
            continue
        recs.append({"id": r["question_id"], "article": group_of(r), "fig": r["image_path"], "question": r["question"].strip(), "choices": u[0], "gold": u[1],
                     "dataset": r["dataset"], "qtype": r.get("question_type"), "modality": r.get("modality_type")})
    # sampling frame: groups shuffled, up to CAP questions per source dataset
    by_ds = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], {}).setdefault(r["article"], []).append(r)
    frame = []
    for ds, gs in sorted(by_ds.items()):
        keys = sorted(gs)
        random.Random(f"{SEED}|{ds}").shuffle(keys)
        n = 0
        for k in keys:
            if n >= CAP:
                break
            frame += gs[k]
            n += len(gs[k])
    garr = {}
    for r in frame:
        garr.setdefault(r["article"], []).append(r)
    keys = sorted(garr)
    random.Random(SEED).shuffle(keys)
    groups, ai = {}, 0
    for name, q in (("E1", N_E1), ("E2", N_E2), ("VAL700", N_VAL), ("TR", N_TR)):
        rows_g = []
        while len(rows_g) < q and ai < len(keys):
            rows_g += garr[keys[ai]]
            ai += 1
        groups[name] = rows_g[:q]
    for g, rs in groups.items():
        for r in rs:
            p = OUT + "/images/" + hashlib.sha1(r["fig"].encode()).hexdigest()[:16] + os.path.splitext(r["fig"])[1].lower()
            if not os.path.exists(p):
                open(p, "wb").write(z.read("OmniMedVQA/" + r["fig"]))
            r["image"] = p
    ids = {g: [r["id"] for r in rs] for g, rs in groups.items()}
    json.dump({g: {r["id"]: r for r in rs} for g, rs in groups.items()}, open(OUT + "/records.json", "w"))
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(ROOT + "/models/pubmedclip_vit_b32").eval()
    proc = CLIPProcessor.from_pretrained(ROOT + "/models/pubmedclip_vit_b32")
    donors = {}
    for g in ("E1", "E2"):
        rs = groups[g]
        embs = []
        for s in range(0, len(rs), 64):
            ims = [Image.open(r["image"]).convert("RGB") for r in rs[s:s + 64]]
            with torch.no_grad():
                e = model.get_image_features(**proc(images=ims, return_tensors="pt"))
                e = e.pooler_output if hasattr(e, "pooler_output") else e
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).numpy())
        E = np.concatenate(embs)
        S = E @ E.T
        n = len(rs)
        a = [r["article"] for r in rs]
        nn, rand = {}, {}
        perm = list(range(n))
        random.Random(SEED + 1).shuffle(perm)
        for k in range(n):
            i = perm[k]
            for step in range(1, n):
                j = perm[(k + step) % n]
                if a[j] != a[i]:
                    rand[rs[i]["id"]] = rs[j]["id"]
                    break
        for i in range(n):
            s = S[i].copy()
            for j in range(n):
                if a[j] == a[i]:
                    s[j] = -9
            nn[rs[i]["id"]] = rs[int(s.argmax())]["id"]
        donors[g] = {"nn": nn, "rand": rand}
        print(g, "donors done", flush=True)
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
    check("sizes", [len(ids[g]) for g in ("E1", "E2", "VAL700", "TR")] == [N_E1, N_E2, N_VAL, N_TR])
    gs = ["E1", "E2", "VAL700", "TR"]
    check("group-disjoint across all splits", all(not (A[a] & A[b]) for i, a in enumerate(gs) for b in gs[i + 1:]))
    check("images exist", all(os.path.exists(rec[g][i]["image"]) for g in gs for i in ids[g][:50]))
    check("donors in-set and from another group", all(don[g][k][i] in rec[g] and rec[g][don[g][k][i]]["article"] != rec[g][i]["article"] for g in ("E1", "E2") for k in ("nn", "rand") for i in ids[g]))
    check("gold in ABCD", all(rec[g][i]["gold"] in LET for g in gs for i in ids[g]))
    print("SELFTEST", "FAILED: " + ", ".join(fails) if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else build()
