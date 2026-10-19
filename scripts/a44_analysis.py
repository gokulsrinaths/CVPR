#!/usr/bin/env python
"""A44 analysis: exactly rules L1-L3 and the reading rule of docs/preregistration_v1_amendment_A44.md. python a44_analysis.py -> analysis/a44_summary.json"""
import glob
import json
import re

import numpy as np

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
LAB = re.compile(r"^\s*[A-E]\)\s*")
T = {c["case_id"]: c for c in json.load(open(ROOT + "/data/train_pool/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}


def shortest(cid):
    L = [len(LAB.sub("", T[cid][l]).strip()) for l in "ABCDE"]
    return min(range(5), key=lambda k: (L[k], k))


def curve(path, stage_groups=("test711",)):
    P = {}
    for ln in open(path):
        r = json.loads(ln)
        if r["group"] in stage_groups and r["arm"] in ("base", "black"):
            lp = r["direct"]["lp"]
            P.setdefault(r["case_id"], {})[r["arm"]] = (int(np.argmax(lp)), "ABCDE".index(r["gold"]))
    ids = [i for i in P if len(P[i]) == 2]
    acc = lambda a: 100 * float(np.mean([P[i][a][0] == P[i][a][1] for i in ids]))
    pk = lambda a: 100 * float(np.mean([P[i][a][0] == shortest(i) for i in ids]))
    return {"n": len(ids), "true": round(acc("base"), 2), "black": round(acc("black"), 2), "headroom": round(acc("base") - acc("black"), 2), "pick_shortest_true": round(pk("base"), 2), "pick_shortest_black": round(pk("black"), 2)}


out = {"seeds": {}}
zs = glob.glob(f"{ROOT}/experiments/pilot/ft_3b_s0_22569048_*/rows_zeroshot.jsonl")[0]
full = glob.glob(f"{ROOT}/experiments/pilot/ft_3b_s0_22569048_*/rows_ft.jsonl")[0]
Z, F = curve(zs), curve(full)
out["zero_shot_N0"], out["A9_full_N9277"] = Z, F
ok = {}
for seed in (20261003, 20261004):
    cur = {0: Z}
    for N in (100, 300, 1000, 3000):
        f = [x for x in glob.glob(f"{ROOT}/experiments/pilot/a44_3b_n{N}_seed{seed}_2*/rows_ft.jsonl") if "smoke" not in x]
        if f:
            cur[N] = curve(f[0])
    if seed == 20261003:
        cur[9277] = F
    out["seeds"][str(seed)] = {str(k): v for k, v in cur.items()}
    Ns = sorted(cur)
    if all(n in cur for n in (0, 100, 300, 1000, 3000)):
        l1 = bool(cur[1000]["black"] >= 80.0)
        l2 = bool(all(cur[b]["black"] >= cur[a]["black"] - 2.0 for a, b in zip(Ns[:-1], Ns[1:])))
        l3 = bool(all(abs(cur[n]["headroom"]) <= 2.0 for n in Ns if n >= 1000))
        ok[seed] = (l1, l2, l3)
        out["seeds"][str(seed)]["L1_L2_L3"] = {"L1": l1, "L2": l2, "L3": l3}
if len(ok) == 2:
    out["L1_both"] = all(v[0] for v in ok.values())
    out["L2_both"] = all(v[1] for v in ok.values())
    out["L3_both"] = all(v[2] for v in ok.values())
json.dump(out, open(ROOT + "/analysis/a44_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
