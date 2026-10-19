"""Exploratory: for each intervention run, accuracy with the true / black image on FRESH3000 questions where the correct option is / is not the shortest."""
import glob
import json
import math
import re

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


RUNS = [("A9 Qwen-3B (reference)", "ft_3b_s0_22569048_*")]
for m in ("lenprior", "mu", "both"):
    for sd in (20261031, 20261032, 20261033):
        RUNS.append((f"A24 {m} seed {sd % 100}", f"a24_medcmr_{m}_3b_seed{sd}_2*"))
for sd in (20261011, 20261012, 20261013):
    RUNS.append((f"A20 tempered PoE seed {sd % 100}", f"a20_medcmr_poeT_3b_seed{sd}_2*"))
for name, pat in RUNS:
    g = [x for x in glob.glob(f"{ROOT}/experiments/pilot/{pat}") if "smoke" not in x]
    if not g:
        continue
    rows = [json.loads(x) for x in open(g[0] + "/rows_ft.jsonl") if x.strip()]
    P = {}
    for r in rows:
        if r["group"] == "fresh" and r["arm"] in ("base", "black") and r["case_id"] in T:
            P.setdefault(r["case_id"], {})[r["arm"]] = max(range(5), key=lambda k: r["direct"]["lp"][k])
    gold = {c: "ABCDE".index(T[c]["answer"].strip()) for c in P if len(P[c]) == 2}
    for lab, sel in (("cue-contradicting (n=%d)", [c for c in gold if gold[c] != shortest(c)]), ("cue-following", [c for c in gold if gold[c] == shortest(c)])):
        n = len(sel)
        a = sum(P[c]["base"] == gold[c] for c in sel) / n
        b = sum(P[c]["black"] == gold[c] for c in sel) / n
        fx = sum(P[c]["base"] == gold[c] and P[c]["black"] != gold[c] for c in sel)
        br = sum(P[c]["base"] != gold[c] and P[c]["black"] == gold[c] for c in sel)
        print(f"{name:28s} | {(lab % n) if '%d' in lab else lab + ' (n=%d)' % n:30s}: true {100 * a:5.1f}  black {100 * b:5.1f}  effect {100 * (a - b):+5.1f}  fixes {fx:3d} breaks {br:3d}  p={mc(fx, br):.3g}")
