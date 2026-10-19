"""Exploratory: image benefit on FRESH3000 questions where the correct option is / is not the shortest (A9 models, existing rows)."""
import glob
import json
import math
import re
import sys

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
POOLD = ROOT + "/data/train_pool"
LAB = re.compile(r"^\s*[A-E]\)\s*")
T = {c["case_id"]: c for c in json.load(open(POOLD + "/official_json/medreason_train_selection.json", encoding="utf-8"))["cases"]}


def shortest(cid):
    L = [len(LAB.sub("", T[cid][l]).strip()) for l in "ABCDE"]
    return min(range(5), key=lambda k: (L[k], k))


def mc(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(b, c) + 1)) / 2 ** n)


for key, pat in (("A9 Qwen-3B", "ft_3b_s0_22569048_*"), ("A9 Qwen-7B", "ft_7b_s0_22569049_*")):
    d = glob.glob(f"{ROOT}/experiments/pilot/{pat}")[0]
    rows = [json.loads(x) for x in open(d + "/rows_ft.jsonl") if x.strip()]
    P = {}
    for r in rows:
        if r["group"] == "fresh" and r["arm"] in ("base", "black") and r["case_id"] in T:
            P.setdefault(r["case_id"], {})[r["arm"]] = max(range(5), key=lambda k: r["direct"]["lp"][k])
    gold = {c: "ABCDE".index(T[c]["answer"].strip()) for c in P}
    for name, sel in (("correct option NOT the shortest", [c for c in P if gold[c] != shortest(c)]), ("correct option IS the shortest", [c for c in P if gold[c] == shortest(c)])):
        n = len(sel)
        a = sum(P[c]["base"] == gold[c] for c in sel) / n
        b = sum(P[c]["black"] == gold[c] for c in sel) / n
        fx = sum(P[c]["base"] == gold[c] and P[c]["black"] != gold[c] for c in sel)
        br = sum(P[c]["base"] != gold[c] and P[c]["black"] == gold[c] for c in sel)
        print(f"{key} | {name}: n={n}  acc true image {100 * a:.1f}%  black {100 * b:.1f}%  effect {100 * (a - b):+.1f} pts  fixes {fx} breaks {br}  exact McNemar p={mc(fx, br):.3g}")
