#!/usr/bin/env python
"""EXPLORATORY (not pre-registered; uses already-seen TEST711 and DEV700 arms): is the image used mainly when the text cannot decide?
Reads existing true/black forward arms. CPU only. Prints tables; writes analysis/exploratory_conditional_reliance.json"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_summarize import read_jsonl, softmax, shortest_letter, gold_from_train, mcnemar_exact, LETTERS

ROOT = "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify"
TAU = json.load(open(glob.glob(ROOT + "/experiments/pilot/gateD_retrieval_s0_22468934_*/config.json")[0]))["tau_m"]
T = gold_from_train()
LAB = __import__("re").compile(r"^\s*[A-E]\)\s*")


def load():
    G = {}
    ga = {"base": {}, "black": {}}
    for pth in glob.glob(ROOT + "/experiments/pilot/mvqa_diag_holdout711_s0_21406682_*/forward_arms.jsonl"):
        for r in read_jsonl(pth):
            if r["arm"] in ga:
                ga[r["arm"]][r["case_id"]] = r["direct"]["lp"]
    G["TEST711"] = ga
    d = {"base": {}, "black": {}}
    for r in read_jsonl(glob.glob(ROOT + "/experiments/pilot/gateC_h2_dev_s0_22454795_*/forward_rows.jsonl")[0]):
        d[r["arm"]][r["case_id"]] = r["direct"]["lp"]
    G["DEV700"] = d
    return G


def margin(lp):
    p = sorted(softmax(lp), reverse=True)
    return p[0] - p[1]


def ratio(cid):
    c = T[cid]
    g = c["answer"].strip()
    o = {k: len(LAB.sub("", c[k]).strip()) for k in LETTERS}
    d = [v for k, v in o.items() if k != g]
    return o[g] / (sum(d) / len(d))


def effect(rows):
    f = sum(r["ct"] and not r["cb"] for r in rows)   # image fixes (true right, black wrong)
    b = sum((not r["ct"]) and r["cb"] for r in rows)  # image breaks (true wrong, black right)
    n = len(rows)
    return {"n": n, "acc_true": round(sum(r["ct"] for r in rows) / n, 4) if n else None, "acc_black": round(sum(r["cb"] for r in rows) / n, 4) if n else None,
            "image_fixes": f, "image_breaks": b, "net": f - b, "mcnemar_p": round(mcnemar_exact(f, b), 4) if (f + b) else None,
            "pred_changed": sum(r["chg"] for r in rows)}


G = load()
out = {"note": "EXPLORATORY; test/dev arms were seen before; confirmatory replication on fresh data would need its own amendment", "tau_m": TAU}
for g, arms in G.items():
    ids = sorted(arms["base"])
    rows = []
    for i in ids:
        gold = T[i]["answer"].strip()
        pt = LETTERS[max(range(5), key=lambda k: arms["base"][i][k])]
        pb = LETTERS[max(range(5), key=lambda k: arms["black"][i][k])]
        rows.append({"id": i, "ct": pt == gold, "cb": pb == gold, "chg": pt != pb, "mb": margin(arms["black"][i]), "mt": margin(arms["base"][i]),
                     "ratio": ratio(i), "short_wrong": shortest_letter(T[i]) != gold})
    res = {"all": effect(rows)}
    res["by_black_image_margin(text-only confidence)"] = {
        "<0.90": effect([r for r in rows if r["mb"] < 0.90]), "0.90-tau": effect([r for r in rows if 0.90 <= r["mb"] < TAU]), ">=tau": effect([r for r in rows if r["mb"] >= TAU])}
    res["by_length_cue_strength(gold/mean distractor chars)"] = {
        "<0.5": effect([r for r in rows if r["ratio"] < 0.5]), "0.5-0.7": effect([r for r in rows if 0.5 <= r["ratio"] < 0.7]),
        "0.7-0.9": effect([r for r in rows if 0.7 <= r["ratio"] < 0.9]), ">=0.9": effect([r for r in rows if r["ratio"] >= 0.9])}
    res["by_shortest_option_correct"] = {"shortest_is_gold": effect([r for r in rows if not r["short_wrong"]]), "shortest_is_wrong": effect([r for r in rows if r["short_wrong"]])}
    chg = [r for r in rows if r["chg"]]
    res["cases_where_black_image_changes_the_prediction"] = {"n": len(chg), "true_image_correct": sum(r["ct"] for r in chg), "black_image_correct": sum(r["cb"] for r in chg),
                                                              "share_of_all_true_errors_among_them": round(sum((not r["ct"]) for r in chg) / max(1, sum((not r["ct"]) for r in rows)), 4)}
    out[g] = res
json.dump(out, open(ROOT + "/analysis/exploratory_conditional_reliance.json", "w"), indent=1)
for g in ("TEST711", "DEV700"):
    print("=====", g)
    for k, v in out[g].items():
        print(k)
        if isinstance(v, dict) and "n" in v:
            print("  ", v)
        else:
            for kk, vv in v.items():
                print("  ", kk, vv)
